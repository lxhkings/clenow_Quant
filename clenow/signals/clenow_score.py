"""Clenow Smooth Momentum score calculation.

Core formula: score = (exp(slope * 252) - 1) * R^2

Where slope comes from a log-linear regression on adjusted close prices,
and R^2 measures the smoothness (fit quality) of that trend.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_clenow_score(
    adj_close: pd.Series,
    raw_close: pd.Series,
    score_window: int = 90,
    gap_threshold: float = 0.15,
) -> float:
    """Compute the Clenow Smooth Momentum score for a price series.

    Parameters
    ----------
    adj_close : pd.Series
        Adjusted close prices (reflects true compound returns for slope).
    raw_close : pd.Series
        Unadjusted close prices (used for gap / jump detection).
    score_window : int
        Number of trading days for the regression window (default 90).
    gap_threshold : float
        Maximum allowed single-day log return in raw_close before the
        window is zeroed out (default 0.15 = 15%).

    Returns
    -------
    float
        The Clenow score. Returns 0.0 when:
        - insufficient history (< score_window bars)
        - excessive NaN in adj_close (> 10% of window)
        - a gap/jump in raw_close exceeds gap_threshold
    """
    n = score_window

    # Short history: not enough data points
    if len(adj_close) < n:
        return 0.0

    # Take the last n values from each series
    window_adj = adj_close.iloc[-n:]
    window_raw = raw_close.iloc[-n:]

    # Missing data: if > 10% NaN in adj_close window, score = 0
    nan_count = window_adj.isna().sum()
    if nan_count > n * 0.10:
        return 0.0

    # Gap detection on raw_close: if any single-day log return exceeds
    # gap_threshold in absolute value, score = 0 for this window
    # We need at least 2 non-NaN raw_close values to compute returns
    raw_valid = window_raw.dropna()
    if len(raw_valid) >= 2:
        log_returns = np.log(raw_valid / raw_valid.shift(1)).dropna()
        if (log_returns.abs() > gap_threshold).any():
            return 0.0

    # Drop NaN from adj_close for regression
    valid_adj = window_adj.dropna()
    if len(valid_adj) < 2:
        return 0.0

    # Log-linear regression: regress log(adj_close) on time index
    log_prices = np.log(valid_adj.values)
    x = np.arange(len(log_prices), dtype=float)

    slope, intercept, r, p_value, std_err = linregress(x, log_prices)

    # Clenow score = (exp(slope * 252) - 1) * R^2
    r_squared = r ** 2
    annualized_return = np.exp(slope * 252) - 1
    score = annualized_return * r_squared

    return float(score)
