"""Per-ticker Parquet cache backed by DuckDB queries.

Layout:
    cache_dir/
        parquet/AAPL.parquet, MSFT.parquet, ...   # one file per ticker
        manifest.parquet                          # coverage index

Each ticker file accumulates the maximum span queried so far. The manifest
tracks (min_date, max_date, row_count, updated_at) per ticker so we can
detect coverage gaps without scanning every file.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

PRICE_COLUMNS = [
    "raw_open", "raw_high", "raw_low", "raw_close",
    "volume", "adj_close", "dividend", "split_ratio",
]
MANIFEST_COLUMNS = ["ticker", "min_date", "max_date", "row_count", "updated_at"]
PARQUET_SUBDIR = "parquet"
MANIFEST_FILENAME = "manifest.parquet"

DBFetcher = Callable[[list[str], date, date], pd.DataFrame]


class ParquetCache:
    """Per-ticker Parquet cache with DuckDB query backend.

    Parameters
    ----------
    cache_dir : Path
        Root directory; per-ticker files live in ``cache_dir/parquet/``,
        manifest at ``cache_dir/manifest.parquet``.
    db_fetcher : Callable
        Function ``(tickers, start, end) -> flat DataFrame`` with columns
        ``date, ticker`` plus PRICE_COLUMNS. Called only on cache miss.
    """

    def __init__(self, cache_dir: Path, db_fetcher: DBFetcher) -> None:
        self.cache_dir = Path(cache_dir)
        self.parquet_dir = self.cache_dir / PARQUET_SUBDIR
        self.manifest_path = self.cache_dir / MANIFEST_FILENAME
        self.db_fetcher = db_fetcher
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _read_manifest(self) -> pd.DataFrame:
        """Return manifest DataFrame, or an empty one with the right schema."""
        if not self.manifest_path.exists():
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        try:
            df = pd.read_parquet(self.manifest_path)
        except Exception as exc:
            logger.warning("Manifest unreadable (%s); treating as empty.", exc)
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        # Normalize date columns
        for col in ("min_date", "max_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        return df

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def _identify_gaps(
        self,
        manifest: pd.DataFrame,
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, list[tuple[date, date]]]:
        """Return ``{ticker: [(gap_start, gap_end), ...]}`` for missing slices.

        A ticker maps to:
            - one gap (start, end) if absent from manifest
            - one gap (start, min_date - 1) if manifest starts after `start`
            - one gap (max_date + 1, end) if manifest ends before `end`
            - both if request straddles both ends
            - omitted entirely if manifest fully covers [start, end]
        """
        from datetime import timedelta

        gaps: dict[str, list[tuple[date, date]]] = {}
        index = (
            manifest.set_index("ticker")
            if not manifest.empty
            else pd.DataFrame(columns=MANIFEST_COLUMNS).set_index("ticker")
        )
        for ticker in tickers:
            if ticker not in index.index:
                gaps[ticker] = [(start, end)]
                continue
            row = index.loc[ticker]
            min_d = row["min_date"]
            max_d = row["max_date"]
            ticker_gaps: list[tuple[date, date]] = []
            if start < min_d:
                ticker_gaps.append((start, min_d - timedelta(days=1)))
            if end > max_d:
                ticker_gaps.append((max_d + timedelta(days=1), end))
            if ticker_gaps:
                gaps[ticker] = ticker_gaps
        return gaps

    # ------------------------------------------------------------------
    # Per-ticker file write (atomic)
    # ------------------------------------------------------------------

    def _ticker_path(self, ticker: str) -> Path:
        return self.parquet_dir / f"{ticker}.parquet"

    def _write_ticker_atomic(self, ticker: str, new_rows: pd.DataFrame) -> None:
        """Merge new_rows into the per-ticker file. Atomic via temp+rename."""
        if new_rows.empty:
            return
        path = self._ticker_path(ticker)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows.copy()
        combined = (
            combined.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        tmp = path.with_suffix(".parquet.tmp")
        try:
            combined.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _update_manifest_entries(
        self,
        tickers: list[str],
        requested_range: tuple[date, date] | None = None,
    ) -> None:
        """Recompute manifest rows for `tickers` from their on-disk files.

        Parameters
        ----------
        requested_range : tuple[date, date] | None
            If provided, manifest min_date/max_date are expanded to cover this
            range. This prevents the fetcher from being re-called for ranges
            where the DB returned no rows (e.g., holidays/weekends).
        """
        if not tickers:
            return
        manifest = self._read_manifest()
        # Drop existing rows for tickers we're about to refresh
        manifest = manifest[~manifest["ticker"].isin(tickers)]

        new_rows = []
        now = datetime.now(UTC)
        for ticker in tickers:
            path = self._ticker_path(ticker)
            if not path.exists():
                logger.warning("Ticker %s parquet file missing; skipping manifest update", ticker)
                continue
            df = pd.read_parquet(path, columns=["date"])
            if df.empty:
                logger.warning("Ticker %s parquet file empty; skipping manifest update", ticker)
                continue
            min_d = df["date"].min()
            max_d = df["date"].max()
            if requested_range is not None:
                req_start, req_end = requested_range
                if req_start < min_d:
                    min_d = req_start
                if req_end > max_d:
                    max_d = req_end
            new_rows.append({
                "ticker": ticker,
                "min_date": min_d,
                "max_date": max_d,
                "row_count": int(len(df)),
                "updated_at": now,
            })
        if new_rows:
            additions = pd.DataFrame(new_rows, columns=MANIFEST_COLUMNS)
            manifest = pd.concat([manifest, additions], ignore_index=True)

        # Atomic write of manifest
        tmp = self.manifest_path.with_suffix(".parquet.tmp")
        try:
            manifest.to_parquet(tmp, index=False)
            os.replace(tmp, self.manifest_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Fetch + persist orchestration
    # ------------------------------------------------------------------

    def _fetch_and_persist(
        self, gaps: dict[str, list[tuple[date, date]]]
    ) -> None:
        """Group gaps by (start, end), call db_fetcher once per range,
        write per-ticker files, then refresh manifest entries.
        """
        if not gaps:
            return

        # Build {(start, end): [tickers]}
        batches: dict[tuple[date, date], list[str]] = {}
        for ticker, ranges in gaps.items():
            for rng in ranges:
                batches.setdefault(rng, []).append(ticker)

        touched: set[str] = set()
        # Track the union range across all gap batches so the manifest
        # records the full requested window (not just what the DB returned).
        overall_start = min(s for s, _e in batches)
        overall_end = max(e for _s, e in batches)
        for (start, end), tickers in batches.items():
            tickers_sorted = sorted(tickers)
            df = self.db_fetcher(tickers_sorted, start, end)
            if df is None or df.empty:
                logger.warning(
                    "db_fetcher returned no rows for %d tickers %s..%s",
                    len(tickers_sorted), start, end,
                )
                continue
            for ticker, slice_df in df.groupby("ticker", sort=False):
                self._write_ticker_atomic(str(ticker), slice_df)
                touched.add(str(ticker))

        if touched:
            self._update_manifest_entries(
                sorted(touched), requested_range=(overall_start, overall_end)
            )

    # ------------------------------------------------------------------
    # DuckDB read path
    # ------------------------------------------------------------------

    def _query_via_duckdb(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Single DuckDB SQL across per-ticker files. Returns flat DataFrame."""
        import duckdb

        empty = pd.DataFrame(columns=["date", "ticker", *PRICE_COLUMNS])
        files = sorted(self.parquet_dir.glob("*.parquet"))
        if not files:
            return empty

        glob = str(self.parquet_dir / "*.parquet")
        cols = ", ".join(PRICE_COLUMNS)
        sql = f"""
            SELECT date, ticker, {cols}
            FROM read_parquet(?)
            WHERE ticker IN (SELECT UNNEST(?))
              AND date BETWEEN ? AND ?
            ORDER BY date, ticker
        """
        con = duckdb.connect(":memory:")
        try:
            df = con.execute(sql, [glob, list(tickers), start, end]).df()
        finally:
            con.close()

        if df.empty:
            return empty
        # DuckDB returns date as np.datetime64; coerce back to python date for parity with DB path
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def load(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Return MultiIndex(date, ticker) DataFrame for [start, end]."""
        empty = pd.DataFrame(
            columns=PRICE_COLUMNS,
            index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
        )
        if not tickers:
            return empty

        manifest = self._read_manifest()
        gaps = self._identify_gaps(manifest, tickers, start, end)
        if gaps:
            self._fetch_and_persist(gaps)

        flat = self._query_via_duckdb(tickers, start, end)
        if flat.empty:
            return empty
        return flat.set_index(["date", "ticker"]).sort_index()