"""Shared helpers for working with price DataFrames."""
from __future__ import annotations

import pandas as pd


def get_ticker_series(prices: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Extract a single ticker's rows from a (date, ticker) MultiIndex frame.

    Returns None when the frame is empty, the ticker is missing, or the
    extracted slice is empty. The returned slice is sorted by index.
    """
    if prices.empty:
        return None
    try:
        if isinstance(prices.index, pd.MultiIndex):
            ticker_data = prices.xs(ticker, level=1, drop_level=True)
        else:
            ticker_data = prices
        if ticker_data.empty:
            return None
        return ticker_data.sort_index()
    except KeyError:
        return None