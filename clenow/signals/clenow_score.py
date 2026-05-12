"""Clenow Smooth Momentum score calculation.

Core formula: score = (exp(slope * annualization_days) - 1) * R^2

Where slope comes from a log-linear regression on adjusted close prices,
and R^2 measures the smoothness (fit quality) of that trend.
annualization_days defaults to 252 (US), use 244 for CN, 247 for HK.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_clenow_score(
    adj_close,
    raw_close,
    score_window: int = 90,
    annualization_days: int = 252,
    gap_threshold: float = 0.15,
) -> float:
    """Compute the Clenow Smooth Momentum score for a price series.

    Accepts both pd.Series and np.ndarray inputs.

    Parameters
    ----------
    adj_close : pd.Series or np.ndarray
        Adjusted close prices (reflects true compound returns for slope).
    raw_close : pd.Series or np.ndarray
        Unadjusted close prices (used for gap / jump detection).
    score_window : int
        Number of trading days for the regression window (default 90).
    annualization_days : int
        Number of trading days per year for annualization (default 252 for US).
        Use 244 for China, 247 for Hong Kong, etc.
    gap_threshold : float
        Maximum allowed single-day log return before the window is zeroed out
        (default 0.15 = 15%).

    Returns
    -------
    float
        The Clenow score. Returns 0.0 when:
        - insufficient history (< score_window bars)
        - excessive NaN in adj_close (> 10% of window)
        - a gap/jump in raw_close exceeds gap_threshold
    """
    n = score_window

    adj_np = np.asarray(adj_close, dtype=float)
    raw_np = np.asarray(raw_close, dtype=float)

    if len(adj_np) < n:
        return 0.0

    adj_w = adj_np[-n:]
    raw_w = raw_np[-n:]

    # Missing data: if > 10% NaN in adj_close window, score = 0
    nan_count = int(np.isnan(adj_w).sum())
    if nan_count > n * 0.10:
        return 0.0

    # Gap detection on raw_close
    raw_valid = raw_w[~np.isnan(raw_w)]
    if len(raw_valid) >= 2:
        log_returns = np.log(raw_valid[1:] / raw_valid[:-1])
        if np.any(np.abs(log_returns) > gap_threshold):
            return 0.0

    # Drop NaN from adj_close for regression
    valid_adj = adj_w[~np.isnan(adj_w)]
    if len(valid_adj) < 2:
        return 0.0

    log_prices = np.log(valid_adj)
    x = np.arange(len(log_prices), dtype=float)

    slope, _intercept, r, _p, _se = linregress(x, log_prices)

    r_squared = r ** 2
    annualized_return = np.exp(slope * annualization_days) - 1
    return float(annualized_return * r_squared)
