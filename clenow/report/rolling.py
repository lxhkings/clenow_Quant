"""Rolling metrics computation — rolling Sharpe ratio and rolling max drawdown.

Provides a time-varying view of strategy performance, replacing walk-forward
validation (Clenow has no parameters to train).

Success criterion from design: regime-on months where rolling 1Y Sharpe > 0
should be >= 60%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_metrics(
    equity_curve: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Compute rolling 1Y Sharpe ratio and rolling max drawdown.

    Args:
        equity_curve: DataFrame with columns: date, portfolio_value, cash
        window: Rolling window size in trading days (default 252 = 1 year)

    Returns:
        DataFrame with columns: date, rolling_sharpe, rolling_max_dd
        Values are NaN where there is insufficient data for the window.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return pd.DataFrame(columns=["date", "rolling_sharpe", "rolling_max_dd"])

    values = equity_curve["portfolio_value"].values.astype(float)
    dates = equity_curve["date"].values
    daily_returns = np.diff(values) / values[:-1]

    # We need `window` returns to compute a single rolling value.
    # Returns array has len(values) - 1 elements, indexed from 0.
    # For a rolling window starting at return index i, we use returns[i:i+window].
    # The date for that window is the date of the last return's value, i.e. dates[i+window].

    n_returns = len(daily_returns)
    rolling_sharpes = []
    rolling_dds = []
    result_dates = []

    for i in range(n_returns):
        end_idx = i + 1  # at least 1 return
        start_idx = end_idx - window

        if start_idx < 0:
            # Not enough data yet
            continue

        window_returns = daily_returns[start_idx:end_idx]

        # Rolling Sharpe
        if np.std(window_returns) == 0:
            sharpe = 0.0
        else:
            sharpe = float(
                np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
            )

        # Rolling max drawdown within this window
        # Reconstruct equity values from returns for this window
        window_values = np.empty(window + 1)
        window_values[0] = 1.0  # Start with 1.0 for normalization
        for j in range(window):
            window_values[j + 1] = window_values[j] * (1 + window_returns[j])

        max_dd = _compute_window_drawdown(window_values)

        # Date corresponds to the last day of the window
        date_idx = i + 1  # dates index for the last return
        if date_idx < len(dates):
            result_dates.append(dates[date_idx])
            rolling_sharpes.append(sharpe)
            rolling_dds.append(max_dd)

    if not result_dates:
        return pd.DataFrame(columns=["date", "rolling_sharpe", "rolling_max_dd"])

    return pd.DataFrame({
        "date": result_dates,
        "rolling_sharpe": rolling_sharpes,
        "rolling_max_dd": rolling_dds,
    })


def _compute_window_drawdown(values: np.ndarray) -> float:
    """Compute max drawdown from a series of portfolio values.

    Returns the drawdown as a negative number (e.g., -0.15 for 15% DD).
    """
    peak = values[0]
    max_dd = 0.0

    for v in values[1:]:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return max_dd
