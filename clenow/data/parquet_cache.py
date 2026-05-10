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
from datetime import date
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