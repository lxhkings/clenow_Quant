"""Tests for clenow.data.utils module."""
from __future__ import annotations

import pandas as pd
import pytest

from clenow.data.utils import get_ticker_series


class TestGetTickerSeries:
    """Tests for get_ticker_series helper function."""

    def test_extracts_ticker_from_multiindex(self) -> None:
        """Extracts ticker data from MultiIndex (date, ticker) frame."""
        # Create a MultiIndex DataFrame with two tickers
        dates = pd.date_range("2024-01-01", periods=5)
        tickers = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product(
            [dates, tickers], names=["date", "ticker"]
        )
        # Data is laid out as: (d1, AAPL), (d1, MSFT), (d2, AAPL), (d2, MSFT), ...
        df = pd.DataFrame(
            {
                "raw_close": [100.0 + i for i in range(10)],
                "raw_high": [102.0 + i for i in range(10)],
                "raw_low": [98.0 + i for i in range(10)],
            },
            index=index,
        )

        # Extract AAPL data (indices 0, 2, 4, 6, 8 -> values 100, 102, 104, 106, 108)
        result = get_ticker_series(df, "AAPL")

        assert result is not None
        assert len(result) == 5
        # Verify ticker level was dropped
        assert not isinstance(result.index, pd.MultiIndex)
        # Verify values for AAPL (every other row)
        assert list(result["raw_close"]) == [100.0, 102.0, 104.0, 106.0, 108.0]

    def test_returns_none_when_empty(self) -> None:
        """Returns None for an empty DataFrame."""
        df = pd.DataFrame()
        result = get_ticker_series(df, "AAPL")
        assert result is None

    def test_returns_none_when_ticker_absent(self) -> None:
        """Returns None when ticker is not in the DataFrame."""
        dates = pd.date_range("2024-01-01", periods=3)
        tickers = ["AAPL", "MSFT"]
        index = pd.MultiIndex.from_product(
            [dates, tickers], names=["date", "ticker"]
        )
        df = pd.DataFrame(
            {"raw_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
            index=index,
        )

        result = get_ticker_series(df, "GOOGL")
        assert result is None

    def test_passthrough_for_single_index_frame(self) -> None:
        """Passes through non-MultiIndex frames (single ticker already extracted)."""
        df = pd.DataFrame(
            {"raw_close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )

        # For single-index frame, it should return the frame sorted
        result = get_ticker_series(df, "AAPL")

        assert result is not None
        assert len(result) == 3
        assert not isinstance(result.index, pd.MultiIndex)
        # Verify it's sorted
        assert result.index.is_monotonic_increasing

    def test_returns_sorted_data(self) -> None:
        """Returns data sorted by index."""
        # Create unsorted MultiIndex DataFrame
        dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        tickers = ["AAPL"]
        index = pd.MultiIndex.from_product(
            [dates, tickers], names=["date", "ticker"]
        )
        df = pd.DataFrame(
            {"raw_close": [103.0, 100.0, 101.0]},
            index=index,
        )

        result = get_ticker_series(df, "AAPL")

        assert result is not None
        assert result.index.is_monotonic_increasing
        assert list(result["raw_close"]) == [100.0, 101.0, 103.0]