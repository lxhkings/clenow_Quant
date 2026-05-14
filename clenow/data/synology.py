"""Synology MariaDB DataProvider — adapts NAS database schema to DataProvider protocol.

Maps NAS schema (open/high/low/close) to system schema (raw_open/raw_high/raw_low/raw_close).
Uses constituent_changes table for PIT universe (ADDED/REMOVED tracking).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pymysql

from clenow.data.parquet_cache import ParquetCache
from clenow.data.provider import DataProvider
from clenow.errors import DataAccessError

logger = logging.getLogger(__name__)

# Default connection parameters
DEFAULT_DB_CONFIG = {
    "host": "192.168.8.9",
    "port": 3306,
    "user": "root",
    "password": "18620001807@Aa",
    "database": "stocks",
    "charset": "utf8mb4",
    "autocommit": False,
}


def _connect(config: dict[str, Any] | None = None, retries: int = 3, delay: float = 1.0) -> pymysql.Connection:
    """Open a MariaDB connection with retry logic."""
    cfg = config or DEFAULT_DB_CONFIG
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            conn = pymysql.connect(**cfg)
            conn.ping(reconnect=True)
            return conn
        except Exception as exc:
            last_exc = exc
            logger.warning("DB connection attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                import time
                time.sleep(delay)

    raise DataAccessError(f"Failed to connect to database after {retries} attempts: {last_exc}")


class SynologyDataProvider:
    """Data provider for Synology MariaDB with schema adaptation.

    Parameters
    ----------
    config : dict, optional
        Database connection config. Uses DEFAULT_DB_CONFIG if not provided.
    use_cache : bool
        If True, price data is cached as Parquet on the local filesystem.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        use_cache: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        self._config = config or DEFAULT_DB_CONFIG
        self._conn = _connect(self._config)
        self._use_cache = use_cache

        # PIT universe cache: built lazily on first get_universe call
        self._pit_index: dict[str, list[tuple[date, date | None]]] | None = None
        self._pit_index_id: str | None = None

        if use_cache:
            cache_root = cache_dir or (Path.home() / ".clenow" / "cache")
            self._cache = ParquetCache(
                cache_dir=cache_root,
                db_fetcher=self._query_prices_from_db,
            )
        else:
            self._cache = None

    # -- DataProvider interface -----------------------------------------------

    def _ensure_connection(self) -> None:
        """Ensure database connection is alive, reconnect if needed."""
        try:
            self._conn.ping(reconnect=True)
        except Exception:
            self._conn = _connect(self._config)

    def _read_sql(self, query: str, params: list | tuple | None = None) -> pd.DataFrame:
        """Execute query via cursor, return DataFrame. Avoids pandas DBAPI2 warning."""
        with self._conn.cursor() as cur:
            cur.execute(query, params or ())
            if cur.description is None:
                return pd.DataFrame()
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=columns)

    def load_prices(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Load price data with schema mapping.

        Returns DataFrame with MultiIndex (date, ticker) and columns:
            raw_open, raw_high, raw_low, raw_close, volume,
            adj_close, dividend, split_ratio
        """
        if not tickers:
            return self._empty_prices_frame()

        if self._cache is not None:
            return self._cache.load(tickers, start, end)

        self._ensure_connection()  # Only needed when hitting DB directly
        flat = self._query_prices_from_db(tickers, start, end)
        if flat.empty:
            return self._empty_prices_frame()
        return flat.set_index(["date", "ticker"]).sort_index()

    def _query_prices_from_db(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Pure MariaDB read. Returns flat DataFrame with columns:
        date, ticker, raw_open, raw_high, raw_low, raw_close,
        volume, adj_close, dividend, split_ratio. No index set.
        Empty result returns an empty DataFrame with the same schema (no MultiIndex).

        adj_close: If column exists and has data, use it. Otherwise fallback to raw_close.
        """
        self._ensure_connection()
        placeholders = ",".join(["%s"] * len(tickers))
        # Use COALESCE to fallback to close if adj_close is NULL or column missing
        query = f"""
            SELECT date, ticker, `open`, `high`, `low`, `close`, volume,
                   COALESCE(adj_close, `close`) AS adj_close
            FROM prices
            WHERE ticker IN ({placeholders})
              AND date >= %s
              AND date <= %s
            ORDER BY date, ticker
        """
        params = [*tickers, start.isoformat(), end.isoformat()]

        try:
            df = self._read_sql(query, params)
        except Exception as exc:
            # Fallback: adj_close column may not exist, try without it
            try:
                query_fallback = f"""
                    SELECT date, ticker, `open`, `high`, `low`, `close`, volume
                    FROM prices
                    WHERE ticker IN ({placeholders})
                      AND date >= %s
                      AND date <= %s
                    ORDER BY date, ticker
                """
                df = self._read_sql(query_fallback, params)
                if not df.empty:
                    df["adj_close"] = df["close"]
            except Exception as exc2:
                raise DataAccessError(f"Failed to load prices: {exc2}") from exc2

        if df.empty:
            return pd.DataFrame(columns=[
                "date", "ticker", "raw_open", "raw_high", "raw_low",
                "raw_close", "volume", "adj_close", "dividend", "split_ratio",
            ])

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.rename(columns={
            "open": "raw_open", "high": "raw_high",
            "low": "raw_low", "close": "raw_close",
        })
        for col in ("raw_open", "raw_high", "raw_low", "raw_close", "volume", "adj_close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        # Final fallback: if adj_close still missing or all NaN, use raw_close
        if "adj_close" not in df.columns or df["adj_close"].isna().all():
            df["adj_close"] = df["raw_close"]
        df["dividend"] = 0.0
        df["split_ratio"] = 1.0
        return df[[
            "date", "ticker", "raw_open", "raw_high", "raw_low",
            "raw_close", "volume", "adj_close", "dividend", "split_ratio",
        ]]

    def get_universe(self, as_of: date, index_id: str = "SP500") -> list[str]:
        """Return available stocks as of as_of.

        Strategy:
        1. Try PIT index (constituent_changes + index_constituents)
        2. Fallback: use all tickers with price data around as_of
        """
        self._ensure_connection()
        # First try PIT index; rebuild if index_id changed
        if self._pit_index is None or self._pit_index_id != index_id:
            self._build_pit_index(index_id)

        if self._pit_index:
            tickers: list[str] = []
            for ticker, ranges in self._pit_index.items():
                for added_date, removed_date in ranges:
                    if added_date <= as_of and (removed_date is None or removed_date > as_of):
                        tickers.append(ticker)
                        break

            if tickers:
                return sorted(tickers)

        # Fallback: get all tickers with price data around as_of
        query = """
            SELECT DISTINCT ticker
            FROM prices
            WHERE date >= %s AND date <= %s
            ORDER BY ticker
        """
        # Look for data within 30 days of as_of
        from datetime import timedelta
        params = [(as_of - timedelta(days=30)).isoformat(), as_of.isoformat()]

        try:
            df = self._read_sql(query, params)
            if not df.empty:
                tickers = df["ticker"].tolist()
                logger.info("Using %d tickers with price data around %s", len(tickers), as_of)
                return sorted(tickers)
        except Exception as exc:
            logger.warning("Failed to get tickers from prices: %s", exc)

        return []

    def get_index_prices(
        self, index_id: str, start: date, end: date
    ) -> pd.DataFrame:
        """Load index-level prices for regime detection."""
        self._ensure_connection()
        query = """
            SELECT date, close
            FROM index_prices
            WHERE index_id = %s
              AND date >= %s
              AND date <= %s
            ORDER BY date
        """
        params = [index_id, start.isoformat(), end.isoformat()]

        try:
            df = self._read_sql(query, params)
        except Exception as exc:
            raise DataAccessError(f"Failed to load index prices: {exc}") from exc

        if df.empty:
            return pd.DataFrame(columns=["date", "close"])

        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # -- PIT universe ---------------------------------------------------------

    def _build_pit_index(self, index_id: str = "SP500") -> None:
        """Build PIT index from constituent_changes and index_constituents.

        Strategy:
        1. Use constituent_changes for explicit ADDED/REMOVED tracking
        2. Use index_constituents snapshots to fill gaps:
           - Ticker appears in snapshots from date A to date B
           - If still in latest snapshot, removed_date = None
        """
        self._ensure_connection()
        self._pit_index: dict[str, list[tuple[date, date | None]]] = {}
        self._pit_index_id = index_id

        # Step 1: Load constituent_changes (explicit add/remove events)
        changes_by_ticker: dict[str, list[tuple[str, date]]] = {}
        try:
            query = """
                SELECT ticker, change_type, change_date
                FROM constituent_changes
                WHERE index_id = %s
                ORDER BY ticker, change_date
            """
            df = self._read_sql(query, [index_id])

            if not df.empty:
                for _, row in df.iterrows():
                    ticker = row["ticker"]
                    change_date = row["change_date"]
                    if isinstance(change_date, str):
                        change_date = date.fromisoformat(change_date)

                    if ticker not in changes_by_ticker:
                        changes_by_ticker[ticker] = []
                    changes_by_ticker[ticker].append((row["change_type"], change_date))

                logger.info("Loaded %d constituent changes", len(df))
        except Exception as exc:
            logger.warning("Failed to load constituent_changes: %s", exc)

        # Step 2: Load index_constituents snapshots
        snapshot_tickers: dict[str, list[date]] = {}
        try:
            query = """
                SELECT ticker, snapshot_date
                FROM index_constituents
                WHERE index_id = %s
                ORDER BY ticker, snapshot_date
            """
            df = self._read_sql(query, [index_id])

            if not df.empty:
                for _, row in df.iterrows():
                    ticker = row["ticker"]
                    snapshot_date = row["snapshot_date"]
                    if isinstance(snapshot_date, str):
                        snapshot_date = date.fromisoformat(snapshot_date)

                    if ticker not in snapshot_tickers:
                        snapshot_tickers[ticker] = []
                    snapshot_tickers[ticker].append(snapshot_date)

                logger.info("Loaded snapshot data for %d tickers", len(snapshot_tickers))
        except Exception as exc:
            logger.warning("Failed to load index_constituents: %s", exc)

        # Step 3: Build PIT index
        # For tickers with explicit changes, use those
        for ticker, changes in changes_by_ticker.items():
            ranges: list[tuple[date, date | None]] = []
            current_added: date | None = None

            for change_type, change_date in changes:
                if change_type == "ADDED":
                    current_added = change_date
                elif change_type == "REMOVED" and current_added:
                    ranges.append((current_added, change_date))
                    current_added = None

            # If last change was ADDED with no REMOVED, it's still active
            if current_added:
                # Check if this is recent-only data (no historical PIT)
                # If added_date is within 30 days of today and no historical changes,
                # treat as valid for all dates (fallback for sparse snapshot data)
                today = date.today()
                if (today - current_added).days <= 30 and len(changes) == 1:
                    ranges = [(date.min, None)]
                else:
                    ranges.append((current_added, None))

            if ranges:
                self._pit_index[ticker] = ranges

        # For tickers only in snapshots (no explicit changes), infer ranges
        for ticker, dates in snapshot_tickers.items():
            if ticker in self._pit_index:
                # Already have explicit change data, skip
                continue

            if not dates:
                continue

            dates_sorted = sorted(dates)
            last_date = dates_sorted[-1]

            # Check if ticker is still in the latest snapshot
            # If the latest snapshot is recent (within 30 days), assume still active
            today = date.today()
            if (today - last_date).days <= 30:
                # No historical PIT data — treat as valid for all dates
                self._pit_index[ticker] = [(date.min, None)]
            else:
                # Ticker was removed after last snapshot
                from datetime import timedelta
                self._pit_index[ticker] = [(date.min, last_date + timedelta(days=1))]

        logger.info("Built PIT index for %s: %d tickers total", index_id, len(self._pit_index))

    def get_stocks_meta(self, tickers: list[str]) -> pd.DataFrame:
        """Returns DataFrame indexed by ticker with at least 'name' column."""
        if not tickers:
            return pd.DataFrame(columns=["name"]).rename_axis("ticker")
        placeholders = ",".join(["%s"] * len(tickers))
        query = f"SELECT ticker, name FROM stocks WHERE ticker IN ({placeholders})"
        with self._conn.cursor() as cur:
            cur.execute(query, tuple(tickers))
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=["ticker", "name"]).set_index("ticker")

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _empty_prices_frame() -> pd.DataFrame:
        """Return an empty DataFrame with the expected schema."""
        cols = [
            "raw_open", "raw_high", "raw_low", "raw_close",
            "volume", "adj_close", "dividend", "split_ratio",
        ]
        idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return pd.DataFrame(columns=cols, index=idx)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and self._conn.open:
            self._conn.close()
