"""Clenow Smooth Momentum score calculation.

Core formula: score = (exp(slope * annualization_days) - 1) * R^2

Where slope comes from a log-linear regression on adjusted close prices,
and R^2 measures the smoothness (fit quality) of that trend.
annualization_days defaults to 252 (US), use 244 for CN, 247 for HK.
"""

from typing import NamedTuple

import numpy as np
from scipy.stats import linregress


class ClenowScoreComponents(NamedTuple):
    score: float
    slope: float
    r_squared: float
    annualized_return: float


_ZERO = ClenowScoreComponents(0.0, 0.0, 0.0, 0.0)


def compute_clenow_score_components(
    adj_close,
    raw_close,
    score_window: int = 90,
    annualization_days: int = 252,
    gap_threshold: float = 0.15,
) -> ClenowScoreComponents:
    """Compute Clenow score with intermediate components exposed.

    Args:
        adj_close: Adjusted close prices (array-like). Reflects true compound
            returns including dividends and splits. Used for slope calculation
            via log-linear regression.
        raw_close: Unadjusted close prices (array-like). Used for gap/jump
            detection. Corporate actions cause gaps in raw prices that should
            not penalize the score; this parameter helps filter such cases.
        score_window: Regression window length in trading days. Default 90.
        annualization_days: Days per year for annualization. Use 252 for US,
            244 for CN (A-shares), 247 for HK. Default 252.
        gap_threshold: Maximum allowed single-day gap in raw_close. If any
            log-returns exceed this threshold, score is zeroed. Default 0.15
            (15%).

    Returns:
        ClenowScoreComponents with fields:
            - score: (exp(slope * annualization_days) - 1) * R^2
            - slope: Daily log-price change from regression
            - r_squared: R^2 of the regression fit
            - annualized_return: exp(slope * annualization_days) - 1

    Returns all-zero ClenowScoreComponents when:
        - insufficient history (< score_window bars)
        - excessive NaN in adj_close (> 10% of window)
        - a gap/jump in raw_close exceeds gap_threshold
    """
    n = score_window

    adj_np = np.asarray(adj_close, dtype=float)
    raw_np = np.asarray(raw_close, dtype=float)

    if len(adj_np) < n:
        return _ZERO

    adj_w = adj_np[-n:]
    raw_w = raw_np[-n:]

    nan_count = int(np.isnan(adj_w).sum())
    if nan_count > n * 0.10:
        return _ZERO

    raw_valid = raw_w[~np.isnan(raw_w)]
    if len(raw_valid) >= 2:
        log_returns = np.log(raw_valid[1:] / raw_valid[:-1])
        if np.any(np.abs(log_returns) > gap_threshold):
            return _ZERO

    valid_adj = adj_w[~np.isnan(adj_w)]
    if len(valid_adj) < 2:
        return _ZERO

    log_prices = np.log(valid_adj)
    x = np.arange(len(log_prices), dtype=float)

    slope, _intercept, r, _p, _se = linregress(x, log_prices)

    r_squared = float(r ** 2)
    slope = float(slope)
    annualized_return = float(np.exp(slope * annualization_days) - 1)
    score = annualized_return * r_squared
    return ClenowScoreComponents(score, slope, r_squared, annualized_return)


def compute_clenow_score(
    adj_close,
    raw_close,
    score_window: int = 90,
    annualization_days: int = 252,
    gap_threshold: float = 0.15,
) -> float:
    """Compute the Clenow Smooth Momentum score for a price series.

    Thin wrapper over compute_clenow_score_components. See that function
    for parameter semantics. Returns the .score field as float.
    """
    return compute_clenow_score_components(
        adj_close,
        raw_close,
        score_window=score_window,
        annualization_days=annualization_days,
        gap_threshold=gap_threshold,
    ).score
