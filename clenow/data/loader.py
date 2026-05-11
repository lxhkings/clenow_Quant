"""SQL-backed DataProvider — loads prices and universe from a SQL database.

Supports both PostgreSQL and SQLite via a common interface.
Implements the DataProvider protocol defined in clenow.data.provider.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import date
from pathlib import Path
from typing import Union

import pandas as pd

from clenow.data.parquet_cache import ParquetCache
from clenow.errors import DataAccessError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _connect(connection_string: str, retries: int = 3, delay: float = 1.0) -> sqlite3.Connection:
    """Open an SQLite connection with retry logic.

    Returns an open sqlite3.Connection on success.
    Raises DataAccessError after *retries* consecutive failures.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            conn = sqlite3.connect(connection_string)
            # Quick sanity check — force a round-trip.
            conn.execute("SELECT 1")
            return conn
        except Exception as exc:
            last_exc = exc
            logger.warning("DB connection attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(delay)
    raise DataAccessError(
        f"Failed to connect to database after {retries} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Adjustment factor computation
# ---------------------------------------------------------------------------

def _compute_adjustment_factor(df: pd.DataFrame) -> pd.Series:
    """Compute a cumulative adjustment factor from dividend and split_ratio.

    The adjustment factor is computed per ticker, sorted chronologically,
    so that adj_close = raw_close * adjustment_factor.

    For a split of 2:1 (split_ratio=2.0), prices before the split are
    halved, so the factor accumulates as factor *= 1/split_ratio.
    For a dividend, the factor accumulates as factor *= (1 - dividend / raw_close).

    We build the factor backwards from the most recent row so that
    the latest price is unadjusted (factor=1) and older prices are scaled down.
    """
    # Sort by ticker then date ascending
    df = df.sort_values(["ticker", "date"])

    factors: list[pd.Series] = []
    for _ticker, group in df.groupby("ticker"):
        group = group.sort_values("date")
        # Walk backwards: most recent row gets factor 1.0
        factor = 1.0
        adj_factors = []
        for idx in reversed(group.index):
            split = group.loc[idx, "split_ratio"]
            div = group.loc[idx, "dividend"]
            close = group.loc[idx, "raw_close"]
            # Accumulate factor from future to past
            adj_factors.append(factor)
            # Move one step into the past — this row's corporate actions
            # affect all rows *before* it.
            if split != 1.0 and split != 0:
                factor /= split
            if div > 0 and close > 0:
                factor *= (1 - div / close)
        adj_factors.reverse()
        factors.append(pd.Series(adj_factors, index=group.index))

    return pd.concat(factors).sort_index()


# ---------------------------------------------------------------------------
# SQLDataProvider
# ---------------------------------------------------------------------------

class SQLDataProvider:
    """Data access backed by a SQL database (SQLite or PostgreSQL).

    Parameters
    ----------
    connection : str or sqlite3.Connection
        Either a connection string (file path for SQLite) or an already-open
        sqlite3.Connection.
    use_cache : bool
        If True, price data is cached as Parquet on the local filesystem.
    """

    def __init__(
        self,
        connection: Union[str, sqlite3.Connection],
        use_cache: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        if isinstance(connection, sqlite3.Connection):
            self._conn = connection
            self._owns_conn = False
        else:
            self._conn = _connect(connection)
            self._owns_conn = True

        self._use_cache = use_cache

        # PIT universe cache: populated lazily on first get_universe call.
        # _pit_index maps ticker -> (earliest_as_of, latest_removed_or_None)
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

    # -- public API ---------------------------------------------------------

    def load_prices(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Load price data for *tickers* between *start* and *end*.

        Returns DataFrame with MultiIndex (date, ticker) and columns:
            raw_open, raw_high, raw_low, raw_close, volume,
            adj_close, dividend, split_ratio
        """
        if not tickers:
            return self._empty_prices_frame()

        if self._cache is not None:
            return self._cache.load(tickers, start, end)

        flat = self._query_prices_from_db(tickers, start, end)
        if flat.empty:
            return self._empty_prices_frame()
        return flat.set_index(["date", "ticker"]).sort_index()

    def _query_prices_from_db(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Pure SQLite read. Returns flat DataFrame with columns:
        date, ticker, raw_open, raw_high, raw_low, raw_close,
        volume, adj_close, dividend, split_ratio. No index set.
        """
        placeholders = ",".join("?" for _ in tickers)
        query = f"""
            SELECT date, ticker, raw_open, raw_high, raw_low, raw_close,
                   volume, adj_close, dividend, split_ratio
            FROM prices
            WHERE ticker IN ({placeholders})
              AND date >= ?
              AND date <= ?
            ORDER BY date, ticker
        """
        params = [*tickers, start.isoformat(), end.isoformat()]

        try:
            df = pd.read_sql_query(query, self._conn, params=params)
        except Exception as exc:
            raise DataAccessError(f"Failed to load prices: {exc}") from exc

        if df.empty:
            return pd.DataFrame(columns=[
                "date", "ticker", "raw_open", "raw_high", "raw_low",
                "raw_close", "volume", "adj_close", "dividend", "split_ratio",
            ])

        # Parse dates
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # If adj_close is NULL / missing, compute from raw_close * adjustment factor
        has_adj = "adj_close" in df.columns and df["adj_close"].notna().any()
        if not has_adj:
            df["adj_close"] = df["raw_close"] * _compute_adjustment_factor(df)

        # Fill NaN dividends / split_ratios with defaults
        df["dividend"] = df["dividend"].fillna(0.0)
        df["split_ratio"] = df["split_ratio"].fillna(1.0)

        return df[[
            "date", "ticker", "raw_open", "raw_high", "raw_low",
            "raw_close", "volume", "adj_close", "dividend", "split_ratio",
        ]]

    def get_universe(self, as_of: date, index_id: str = "SP500") -> list[str]:
        """Return index constituents as of *as_of* (point-in-time).

        On the first call (or when *index_id* changes) this loads ALL rows
        from index_constituents for the given index and builds an in-memory
        index.  Subsequent calls with the same *index_id* are pure dictionary
        lookups — no DB hit.
        """
        if self._pit_index is None or self._pit_index_id != index_id:
            self._build_pit_index(index_id)

        assert self._pit_index is not None  # for type checker
        tickers: list[str] = []
        for ticker, ranges in self._pit_index.items():
            for as_of_date, removed_date in ranges:
                if as_of_date <= as_of and (removed_date is None or removed_date > as_of):
                    tickers.append(ticker)
                    break  # one match per ticker is enough
        return sorted(tickers)

    def get_index_prices(
        self, index_id: str, start: date, end: date
    ) -> pd.DataFrame:
        """Load index-level prices for regime detection."""
        query = """
            SELECT date, close
            FROM index_prices
            WHERE index_id = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        params = [index_id, start.isoformat(), end.isoformat()]

        try:
            df = pd.read_sql_query(query, self._conn, params=params)
        except Exception as exc:
            raise DataAccessError(f"Failed to load index prices: {exc}") from exc

        if df.empty:
            return pd.DataFrame(columns=["date", "close"])

        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def get_stocks_meta(self, tickers: list[str]) -> pd.DataFrame:
        """Returns DataFrame indexed by ticker with at least 'name' column."""
        if not tickers:
            return pd.DataFrame(columns=["name"]).rename_axis("ticker")
        placeholders = ",".join("?" for _ in tickers)
        query = f"SELECT ticker, name FROM stocks WHERE ticker IN ({placeholders})"
        rows = self._conn.execute(query, tickers).fetchall()
        return pd.DataFrame(rows, columns=["ticker", "name"]).set_index("ticker")

    # -- PIT cache ----------------------------------------------------------

    def _build_pit_index(self, index_id: str = "SP500") -> None:
        """Load all index_constituents rows and build an in-memory index."""
        query = """
            SELECT ticker, as_of_date, removed_date
            FROM index_constituents
            WHERE index_id = ?
        """
        params = [index_id]
        try:
            rows = pd.read_sql_query(query, self._conn, params=params)
        except Exception as exc:
            raise DataAccessError(f"Failed to load universe data: {exc}") from exc

        self._pit_index: dict[str, list[tuple[date, date | None]]] = {}
        self._pit_index_id = index_id
        for _, row in rows.iterrows():
            ticker = row["ticker"]
            as_of_date = date.fromisoformat(row["as_of_date"]) if isinstance(row["as_of_date"], str) else row["as_of_date"]
            removed_date = None
            if pd.notna(row["removed_date"]) and row["removed_date"] is not None:
                removed_date = date.fromisoformat(row["removed_date"]) if isinstance(row["removed_date"], str) else row["removed_date"]

            if ticker not in self._pit_index:
                self._pit_index[ticker] = []
            self._pit_index[ticker].append((as_of_date, removed_date))

    # -- helpers ------------------------------------------------------------

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
        """Close the database connection if we own it."""
        if self._owns_conn:
            self._conn.close()
