"""Data provider protocol — the interface between engine and data sources."""

import pandas as pd
from datetime import date
from typing import Protocol


class DataProvider(Protocol):
    """Abstraction over data access. Engine doesn't know if it's DB or broker."""

    def load_prices(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Load price data for given tickers and date range.

        Returns DataFrame with MultiIndex (date, ticker) and columns:
            raw_open, raw_high, raw_low, raw_close, volume,
            adj_close, dividend, split_ratio
        """
        ...

    def get_universe(self, as_of: date, index_id: str = "SP500") -> list[str]:
        """Return index constituents as of the given date (point-in-time).

        Args:
            as_of: Date for point-in-time lookup.
            index_id: Index identifier (e.g. "SP500", "CSI800", "HSI").
        """
        ...

    def get_index_prices(
        self, index_id: str, start: date, end: date
    ) -> pd.DataFrame:
        """Load index-level prices (e.g., SP500) for regime detection."""
        ...

    def get_stocks_meta(self, tickers: list[str]) -> pd.DataFrame:
        """Returns DataFrame indexed by ticker with at least 'name' column."""
        ...
