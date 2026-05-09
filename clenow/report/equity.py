"""Equity curve normalization with benchmark comparison.

Builds a combined equity curve showing portfolio performance alongside
a benchmark (e.g., SPY), both normalized to the same starting capital.
"""

from __future__ import annotations

import pandas as pd


def build_equity_curve_with_benchmark(
    equity_curve: pd.DataFrame,
    benchmark_close: pd.Series,
) -> pd.DataFrame:
    """Combine portfolio equity curve with a normalized benchmark.

    Args:
        equity_curve: DataFrame with columns: date, portfolio_value, cash
        benchmark_close: Series of benchmark close prices indexed by date
            (e.g., SPY daily closes). Can be a DatetimeIndex or date index.

    Returns:
        DataFrame with columns: date, portfolio_value, benchmark_value
        Both portfolio and benchmark are normalized so they start at the
        same capital (the portfolio's initial value).
    """
    if equity_curve.empty:
        return pd.DataFrame(columns=["date", "portfolio_value", "benchmark_value"])

    # Work with copies
    eq = equity_curve[["date", "portfolio_value"]].copy()

    # Normalize date column to consistent type
    eq["date"] = pd.to_datetime(eq["date"])

    # Normalize benchmark index to datetime
    bench = benchmark_close.copy()
    if not isinstance(bench.index, pd.DatetimeIndex):
        bench.index = pd.to_datetime(bench.index)

    # Align benchmark dates with equity curve date range
    eq_dates = eq["date"]
    start_date = eq_dates.min()
    end_date = eq_dates.max()

    # Filter benchmark to the equity curve's date range
    bench_filtered = bench[
        (bench.index >= start_date) & (bench.index <= end_date)
    ]

    if bench_filtered.empty:
        # No overlapping benchmark data — return without benchmark
        eq["benchmark_value"] = float("nan")
        eq["date"] = equity_curve["date"]
        return eq[["date", "portfolio_value", "benchmark_value"]]

    # Normalize benchmark to same starting capital as portfolio
    initial_portfolio_value = float(eq["portfolio_value"].iloc[0])
    initial_bench_price = float(bench_filtered.iloc[0])

    if initial_bench_price == 0:
        eq["benchmark_value"] = float("nan")
        eq["date"] = equity_curve["date"]
        return eq[["date", "portfolio_value", "benchmark_value"]]

    bench_normalized = bench_filtered / initial_bench_price * initial_portfolio_value

    # Create benchmark DataFrame for merge
    bench_df = pd.DataFrame({
        "date": bench_normalized.index,
        "benchmark_value": bench_normalized.values,
    })

    # Merge on date
    merged = eq.merge(bench_df, on="date", how="left")

    # Convert date back to original format
    # If original dates were date objects, convert back
    original_dates = equity_curve["date"]
    if len(merged) == len(original_dates):
        merged["date"] = original_dates.values
    else:
        # If lengths differ (shouldn't happen with left join on same dates),
        # keep the datetime version
        pass

    return merged[["date", "portfolio_value", "benchmark_value"]]
