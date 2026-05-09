"""Tests for build_equity_curve_with_benchmark — benchmark normalization."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from clenow.report.equity import build_equity_curve_with_benchmark


# ── Helpers ──────────────────────────────────────────────────────────


def _make_equity_curve(
    n_days: int = 252,
    start_value: float = 1_000_000.0,
    daily_return: float = 0.001,
    start_date: str = "2023-01-03",
) -> pd.DataFrame:
    """Build an equity curve DataFrame."""
    values = [start_value * (1 + daily_return) ** i for i in range(n_days)]
    dates = pd.bdate_range(start_date, periods=n_days)
    return pd.DataFrame({
        "date": dates.date,
        "portfolio_value": values,
        "cash": [v * 0.1 for v in values],
    })


def _make_benchmark(
    n_days: int = 252,
    start_price: float = 400.0,
    daily_return: float = 0.0005,
    start_date: str = "2023-01-03",
) -> pd.Series:
    """Build a benchmark close price Series."""
    dates = pd.bdate_range(start_date, periods=n_days)
    prices = [start_price * (1 + daily_return) ** i for i in range(n_days)]
    return pd.Series(prices, index=dates, name="SPY")


# ── Tests ────────────────────────────────────────────────────────────


class TestBenchmarkNormalization:
    """Benchmark should start at the same value as the portfolio."""

    def test_benchmark_starts_at_portfolio_value(self):
        """Normalized benchmark should start at the portfolio's initial value."""
        eq = _make_equity_curve(start_value=1_000_000.0)
        bench = _make_benchmark(start_price=400.0)

        result = build_equity_curve_with_benchmark(eq, bench)

        assert not result.empty
        assert "benchmark_value" in result.columns
        # First valid benchmark value should equal initial portfolio value
        first_valid = result["benchmark_value"].dropna().iloc[0]
        assert first_valid == pytest.approx(1_000_000.0)

    def test_benchmark_proportional_to_price(self):
        """Benchmark value should be proportional to its price changes."""
        eq = _make_equity_curve(n_days=10, daily_return=0.0)
        bench = _make_benchmark(n_days=10, start_price=100.0, daily_return=0.01)

        result = build_equity_curve_with_benchmark(eq, bench)

        # If benchmark went up 1% per day, normalized value should follow
        first_valid = result["benchmark_value"].dropna().iloc[0]
        last_valid = result["benchmark_value"].dropna().iloc[-1]
        # Last / first should be ~ (1.01)^9
        ratio = last_valid / first_valid
        expected_ratio = (1.01) ** 9
        assert ratio == pytest.approx(expected_ratio, rel=0.01)


class TestDifferentDateRanges:
    """Handle cases where benchmark and equity have different date ranges."""

    def test_benchmark_shorter_than_equity(self):
        """Benchmark with fewer dates should produce NaN for missing dates."""
        eq = _make_equity_curve(n_days=252)
        # Benchmark only covers the first 100 days
        bench = _make_benchmark(n_days=100)

        result = build_equity_curve_with_benchmark(eq, bench)

        # Should have same number of rows as equity curve
        assert len(result) == len(eq)
        # Some benchmark values should be NaN (days beyond benchmark range)
        assert result["benchmark_value"].isna().any()

    def test_benchmark_longer_than_equity(self):
        """Benchmark extending beyond equity should be trimmed."""
        eq = _make_equity_curve(n_days=100)
        bench = _make_benchmark(n_days=252)

        result = build_equity_curve_with_benchmark(eq, bench)

        # Result should have same number of rows as equity curve
        assert len(result) == len(eq)
        # All benchmark values should be valid (within date range)
        assert result["benchmark_value"].notna().all()


class TestEmptyInputs:
    """Edge cases with empty inputs."""

    def test_empty_equity_curve(self):
        """Empty equity curve should return empty DataFrame."""
        eq = pd.DataFrame(columns=["date", "portfolio_value", "cash"])
        bench = _make_benchmark()

        result = build_equity_curve_with_benchmark(eq, bench)

        assert result.empty
        assert list(result.columns) == ["date", "portfolio_value", "benchmark_value"]

    def test_empty_benchmark(self):
        """Empty benchmark should produce NaN benchmark values."""
        eq = _make_equity_curve(n_days=10)
        bench = pd.Series(dtype=float, name="SPY")

        result = build_equity_curve_with_benchmark(eq, bench)

        assert len(result) == 10
        assert result["benchmark_value"].isna().all()


class TestOutputColumns:
    """Verify output DataFrame structure."""

    def test_output_columns(self):
        """Output should have date, portfolio_value, benchmark_value."""
        eq = _make_equity_curve(n_days=10)
        bench = _make_benchmark(n_days=10)

        result = build_equity_curve_with_benchmark(eq, bench)

        assert list(result.columns) == ["date", "portfolio_value", "benchmark_value"]

    def test_portfolio_values_unchanged(self):
        """Portfolio values in output should match input exactly."""
        eq = _make_equity_curve(n_days=10)
        bench = _make_benchmark(n_days=10)

        result = build_equity_curve_with_benchmark(eq, bench)

        pd.testing.assert_series_equal(
            result["portfolio_value"].reset_index(drop=True),
            eq["portfolio_value"].reset_index(drop=True),
        )
