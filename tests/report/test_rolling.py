"""Tests for compute_rolling_metrics — rolling Sharpe and max drawdown."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from clenow.report.rolling import compute_rolling_metrics


# ── Helpers ──────────────────────────────────────────────────────────


def _make_equity_curve(
    n_days: int = 504,
    start_value: float = 1_000_000.0,
    daily_return: float = 0.001,
    start_date: date = date(2022, 1, 3),
) -> pd.DataFrame:
    """Build an equity curve with constant daily returns."""
    values = [start_value * (1 + daily_return) ** i for i in range(n_days)]
    dates = pd.bdate_range(start_date, periods=n_days)
    return pd.DataFrame({
        "date": dates.date,
        "portfolio_value": values,
        "cash": [v * 0.1 for v in values],
    })


def _make_volatile_equity(
    n_days: int = 504,
    start_value: float = 1_000_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build an equity curve with random returns."""
    np.random.seed(seed)
    returns = np.random.normal(0.001, 0.015, n_days)
    values = [start_value]
    for r in returns:
        values.append(values[-1] * (1 + r))
    dates = pd.bdate_range(date(2022, 1, 3), periods=n_days + 1)
    return pd.DataFrame({
        "date": dates.date,
        "portfolio_value": values,
        "cash": [v * 0.1 for v in values],
    })


# ── Tests ────────────────────────────────────────────────────────────


class TestRollingSharpe:
    """Rolling Sharpe ratio calculation."""

    def test_positive_sharpe_for_positive_returns(self):
        """Consistently positive returns should yield positive rolling Sharpe."""
        eq = _make_equity_curve(n_days=300, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=252)
        # After the initial window, all rolling Sharpes should be positive
        non_nan_sharpes = result["rolling_sharpe"].dropna()
        assert len(non_nan_sharpes) > 0
        assert (non_nan_sharpes > 0).all()

    def test_rolling_sharpe_varies_with_volatile_returns(self):
        """Volatile returns should produce varying rolling Sharpe values."""
        eq = _make_volatile_equity(n_days=504)
        result = compute_rolling_metrics(eq, window=252)
        non_nan_sharpes = result["rolling_sharpe"].dropna()
        # Should have more than one unique value
        assert non_nan_sharpes.nunique() > 1

    def test_rolling_sharpe_infers_daily_rf(self):
        """Rolling Sharpe with zero risk-free rate should match standard formula."""
        eq = _make_equity_curve(n_days=300, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=252)
        non_nan_sharpes = result["rolling_sharpe"].dropna()
        # For constant positive returns, Sharpe should be very high
        assert non_nan_sharpes.iloc[0] > 1.0


class TestRollingMaxDrawdown:
    """Rolling max drawdown calculation."""

    def test_zero_dd_for_monotonic_increase(self):
        """Monotonically increasing equity should have zero rolling max DD."""
        eq = _make_equity_curve(n_days=300, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=252)
        non_nan_dds = result["rolling_max_dd"].dropna()
        assert len(non_nan_dds) > 0
        assert (non_nan_dds == 0.0).all()

    def test_negative_dd_for_volatile_equity(self):
        """Volatile equity should have some negative rolling max DD."""
        eq = _make_volatile_equity(n_days=504)
        result = compute_rolling_metrics(eq, window=252)
        non_nan_dds = result["rolling_max_dd"].dropna()
        # At least some windows should have a drawdown
        assert (non_nan_dds < 0).any()

    def test_dd_always_non_positive(self):
        """Rolling max DD should never be positive."""
        eq = _make_volatile_equity(n_days=504)
        result = compute_rolling_metrics(eq, window=252)
        non_nan_dds = result["rolling_max_dd"].dropna()
        assert (non_nan_dds <= 0).all()


class TestWindowHandling:
    """Window size and insufficient data handling."""

    def test_insufficient_data_produces_nan(self):
        """If equity has fewer days than window, result should be empty or all NaN."""
        # 100 days of equity, 252 window
        eq = _make_equity_curve(n_days=100, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=252)
        # Should be empty since we never have enough data
        assert len(result) == 0 or result["rolling_sharpe"].isna().all()

    def test_exactly_window_size(self):
        """Equity with exactly window+1 days should produce exactly one row."""
        eq = _make_equity_curve(n_days=253, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=252)
        non_nan = result["rolling_sharpe"].dropna()
        assert len(non_nan) == 1

    def test_more_data_than_window(self):
        """More data than window should produce multiple rolling values."""
        eq = _make_equity_curve(n_days=504, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=252)
        non_nan = result["rolling_sharpe"].dropna()
        assert len(non_nan) > 1

    def test_custom_window_size(self):
        """Custom window size should be respected."""
        eq = _make_equity_curve(n_days=100, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=50)
        non_nan = result["rolling_sharpe"].dropna()
        assert len(non_nan) > 0


class TestOutputFormat:
    """Verify output DataFrame structure."""

    def test_output_columns(self):
        """Output should have date, rolling_sharpe, rolling_max_dd."""
        eq = _make_equity_curve(n_days=300, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=252)
        assert list(result.columns) == ["date", "rolling_sharpe", "rolling_max_dd"]

    def test_dates_are_sorted(self):
        """Dates in output should be sorted ascending."""
        eq = _make_equity_curve(n_days=300, daily_return=0.001)
        result = compute_rolling_metrics(eq, window=252)
        if not result.empty:
            dates = result["date"].values
            for i in range(1, len(dates)):
                assert dates[i] >= dates[i - 1]


class TestEmptyInput:
    """Edge cases with empty or minimal input."""

    def test_empty_equity_curve(self):
        """Empty equity curve should return empty DataFrame."""
        eq = pd.DataFrame(columns=["date", "portfolio_value", "cash"])
        result = compute_rolling_metrics(eq, window=252)
        assert result.empty

    def test_single_row_equity(self):
        """Single-row equity curve should return empty DataFrame."""
        eq = pd.DataFrame({
            "date": [date(2023, 1, 3)],
            "portfolio_value": [1_000_000.0],
            "cash": [100_000.0],
        })
        result = compute_rolling_metrics(eq, window=252)
        assert result.empty
