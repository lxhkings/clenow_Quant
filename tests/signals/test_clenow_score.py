"""Tests for clenow_score signal — all branches + hand-calculated fixtures."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

from clenow.signals.clenow_score import compute_clenow_score


# ---------------------------------------------------------------------------
# Helper: build a synthetic adj_close series with a known daily log-return
# ---------------------------------------------------------------------------

def _make_series(n: int, daily_log_ret: float, start_price: float = 100.0,
                 noise_std: float = 0.0, seed: int = 42) -> pd.Series:
    """Create a price series where log-returns are `daily_log_ret` +- noise."""
    rng = np.random.default_rng(seed)
    log_prices = np.empty(n)
    log_prices[0] = math.log(start_price)
    for i in range(1, n):
        log_prices[i] = log_prices[i - 1] + daily_log_ret + rng.normal(0, noise_std)
    prices = np.exp(log_prices)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(prices, index=dates, name="adj_close")


# ===========================================================================
# 1. Normal case: synthetic series with known slope
# ===========================================================================

class TestNormalCase:
    """Score on a noiseless uptrend should match the closed-form value."""

    def test_perfect_uptrend(self):
        n = 90
        daily_ret = 0.001  # 0.1% per day
        adj_close = _make_series(n, daily_ret, noise_std=0.0)
        raw_close = adj_close.copy()  # no gaps

        score = compute_clenow_score(adj_close, raw_close, score_window=90)

        # Hand calculation:
        # slope of log(price) vs time = daily_ret (perfect line)
        # annualized_return = exp(0.001 * 252) - 1
        # R^2 = 1.0 (perfect fit)
        expected_annualized = math.exp(daily_ret * 252) - 1
        expected_score = expected_annualized * 1.0  # R^2 = 1

        assert abs(score - expected_score) < 1e-6

    def test_perfect_downtrend(self):
        n = 90
        daily_ret = -0.001
        adj_close = _make_series(n, daily_ret, noise_std=0.0)
        raw_close = adj_close.copy()

        score = compute_clenow_score(adj_close, raw_close, score_window=90)

        expected_annualized = math.exp(daily_ret * 252) - 1
        expected_score = expected_annualized * 1.0
        assert abs(score - expected_score) < 1e-6


# ===========================================================================
# 2. Gap > 15%: raw_close has a 20% single-day jump -> score = 0
# ===========================================================================

class TestGapDetection:
    def test_large_gap_zeros_score(self):
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # Insert a 20% jump on day 45 (0-indexed)
        raw_close.iloc[45] *= 1.20  # 20% jump -> log(1.2) ≈ 0.182 > 0.15

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0

    def test_small_gap_does_not_zero(self):
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # Insert a 10% jump (log(1.1) ≈ 0.095 < 0.15)
        raw_close.iloc[45] *= 1.10

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score != 0.0  # Should still compute normally

    def test_gap_exactly_at_threshold(self):
        """A return exactly at the threshold (0.15) should NOT zero the score."""
        n = 90
        # Use a trending series so the score would be non-zero
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # Set raw_close[45] so that the log return from [44] to [45]
        # is exactly 0.15. The base log return is 0.001.
        # raw_close[45] = raw_close[44] * exp(0.15)
        raw_close.iloc[45] = raw_close.iloc[44] * math.exp(0.15)

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        # The condition is > gap_threshold (0.15), not >=
        # A log return of exactly 0.15 should NOT trigger zeroing
        assert score != 0.0

    def test_gap_just_above_threshold(self):
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # log return = 0.151, just above threshold
        raw_close.iloc[45] *= math.exp(0.151)

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0


# ===========================================================================
# 3. NaN > 10%: adj_close with 15% NaN in window -> score = 0
# ===========================================================================

class TestNaNHandling:
    def test_excessive_nan_zeros_score(self):
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # Set 14 values to NaN (14/90 ≈ 15.6% > 10%)
        nan_indices = list(range(10, 24))
        adj_close.iloc[nan_indices] = np.nan

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0

    def test_acceptable_nan_computes_score(self):
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # Set 9 values to NaN (9/90 = 10%, exactly at the boundary)
        # The condition is > 10%, so exactly 10% should still compute
        nan_indices = list(range(10, 19))
        adj_close.iloc[nan_indices] = np.nan

        # The raw_close for those days is still valid, so gap detection passes
        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score != 0.0  # Should still compute (exactly 10% is acceptable)

    def test_just_over_10_percent_nan(self):
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # 10 NaN values (10/90 ≈ 11.1% > 10%)
        nan_indices = list(range(10, 20))
        adj_close.iloc[nan_indices] = np.nan

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0


# ===========================================================================
# 4. History < window: only 50 data points when window is 90 -> score = 0
# ===========================================================================

class TestInsufficientHistory:
    def test_short_history_zeros_score(self):
        adj_close = _make_series(50, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0

    def test_exactly_window_length(self):
        """Exactly 90 data points with window=90 should compute."""
        adj_close = _make_series(90, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score != 0.0

    def test_one_more_than_window(self):
        adj_close = _make_series(91, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score != 0.0

    def test_empty_series(self):
        adj_close = pd.Series([], dtype=float)
        raw_close = pd.Series([], dtype=float)

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0


# ===========================================================================
# 5. Hand-calculated fixture: 5 fictional stocks
# ===========================================================================

class TestHandCalculatedFixtures:
    """Each stock's score is computed by hand using the exact formula.

    score = (exp(slope * 252) - 1) * R^2
    """

    def _build_stock(self, n: int, daily_log_ret: float, start_price: float,
                     noise_std: float = 0.0, seed: int = 42) -> tuple[pd.Series, pd.Series]:
        adj_close = _make_series(n, daily_log_ret, start_price, noise_std=noise_std, seed=seed)
        raw_close = adj_close.copy()
        return adj_close, raw_close

    def _hand_calc(self, adj_close: pd.Series, window: int = 90) -> float:
        """Replicate the score computation by hand to cross-check."""
        vals = adj_close.iloc[-window:].dropna().values
        log_p = np.log(vals)
        x = np.arange(len(log_p), dtype=float)
        slope, intercept, r, p_value, std_err = linregress(x, log_p)
        r_sq = r ** 2
        ann_ret = math.exp(slope * 252) - 1
        return ann_ret * r_sq

    def test_stock_a_steady_uptrend(self):
        """Stock A: steady uptrend (slope > 0, high R^2) -> large positive score."""
        adj_close, raw_close = self._build_stock(100, 0.002, start_price=100.0, noise_std=0.0)
        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        expected = self._hand_calc(adj_close, 90)

        assert score > 0.5, f"Stock A score should be large positive, got {score}"
        assert abs(score - expected) < 1e-6, f"Score {score} != expected {expected}"

    def test_stock_b_steady_downtrend(self):
        """Stock B: steady downtrend (slope < 0, high R^2) -> large negative score."""
        # Use daily log return of -0.003 -> annualized = exp(-0.756)-1 ≈ -0.53
        adj_close, raw_close = self._build_stock(100, -0.003, start_price=100.0, noise_std=0.0)
        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        expected = self._hand_calc(adj_close, 90)

        assert score < -0.3, f"Stock B score should be large negative, got {score}"
        assert abs(score - expected) < 1e-6, f"Score {score} != expected {expected}"

    def test_stock_c_flat(self):
        """Stock C: flat/random (slope ~ 0, low R^2) -> score near 0."""
        adj_close, raw_close = self._build_stock(100, 0.0, start_price=100.0, noise_std=0.03, seed=7)
        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        expected = self._hand_calc(adj_close, 90)

        assert abs(score) < 0.5, f"Stock C score should be near 0, got {score}"
        assert abs(score - expected) < 1e-6, f"Score {score} != expected {expected}"

    def test_stock_d_uptrend_with_noise(self):
        """Stock D: uptrend with noise (slope > 0, low R^2) -> score < Stock A.

        We construct D deterministically: same underlying slope as A,
        but with added sinusoidal noise that preserves the overall slope
        while reducing R^2.
        """
        n = 100
        # Stock A: clean uptrend
        adj_close_a = _make_series(n, 0.002, start_price=100.0, noise_std=0.0)
        raw_close_a = adj_close_a.copy()

        # Stock D: same base trend + sinusoidal noise that preserves slope
        # but greatly reduces R^2
        log_prices_a = np.log(adj_close_a.values)
        # Add oscillating noise with zero mean (doesn't change slope)
        noise = 0.1 * np.sin(np.linspace(0, 6 * np.pi, n))  # 3 full cycles
        log_prices_d = log_prices_a + noise
        adj_close_d = pd.Series(np.exp(log_prices_d), index=adj_close_a.index)
        raw_close_d = adj_close_d.copy()

        score_a = compute_clenow_score(adj_close_a, raw_close_a, score_window=90)
        score_d = compute_clenow_score(adj_close_d, raw_close_d, score_window=90)

        # Same slope direction but D has lower R^2 -> D's score magnitude < A's
        assert abs(score_d) < abs(score_a), \
            f"Noisy Stock D score {score_d} should be smaller than clean Stock A {score_a}"

    def test_stock_e_sharp_uptrend_then_reversal(self):
        """Stock E: sharp uptrend then reversal -> R^2 dampens the score."""
        n = 100
        rng = np.random.default_rng(55)
        # Build two-phase series: up for 50 days, down for 50 days
        log_prices = np.empty(n)
        log_prices[0] = math.log(100.0)
        for i in range(1, 50):
            log_prices[i] = log_prices[i - 1] + 0.003  # sharp up
        for i in range(50, n):
            log_prices[i] = log_prices[i - 1] - 0.003  # reversal down
        prices = np.exp(log_prices)
        dates = pd.bdate_range("2024-01-01", periods=n)
        adj_close = pd.Series(prices, index=dates)
        raw_close = adj_close.copy()

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        expected = self._hand_calc(adj_close, 90)

        # The reversal means the regression line is flatter and R^2 is lower
        # Score should be dampened compared to a pure uptrend
        assert abs(score) < 1.0, f"Reversal stock E score should be dampened, got {score}"
        assert abs(score - expected) < 1e-6, f"Score {score} != expected {expected}"


# ===========================================================================
# 6. AAPL split test: dual-data-source convention
# ===========================================================================

class TestAAPLSplitConvention:
    """When raw_close jumps on a split day but adj_close is smooth,
    gap detection should zero the score (using raw_close),
    while using only adj_close (no gap check) would give a normal score.

    This validates the dual-data-source convention.
    """

    def test_split_zeros_score_via_raw_close(self):
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # Simulate a 4:1 stock split on day 60:
        # raw_close drops 75% (log return ≈ -1.386 >> 0.15)
        # adj_close is continuous (no gap)
        raw_close.iloc[60:] = raw_close.iloc[60:] * 0.25

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0, "Split in raw_close should trigger gap detection -> score=0"

    def test_no_gap_in_adj_close_only(self):
        """If both series are smooth (no split), score is normal."""
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score != 0.0, "No gap -> normal score"

    def test_gap_in_raw_not_adj(self):
        """Explicit test: gap in raw_close only, adj_close is smooth."""
        n = 90
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # Put a big jump in raw_close but leave adj_close intact
        raw_close.iloc[45] *= 1.75  # 75% jump -> log(1.75) ≈ 0.56 >> 0.15

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0, "Gap in raw_close should zero the score despite smooth adj_close"


# ===========================================================================
# 7. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_constant_price(self):
        """Constant price -> slope=0, R^2 undefined -> scipy returns r=0."""
        n = 90
        prices = np.full(n, 100.0)
        dates = pd.bdate_range("2024-01-01", periods=n)
        adj_close = pd.Series(prices, index=dates)
        raw_close = adj_close.copy()

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        # With constant prices, log prices are also constant.
        # linregress slope = 0, r = nan or 0
        # exp(0) - 1 = 0, so score should be 0 regardless of R^2
        assert abs(score) < 1e-10 or score == 0.0

    def test_all_nan(self):
        n = 90
        dates = pd.bdate_range("2024-01-01", periods=n)
        adj_close = pd.Series([np.nan] * n, index=dates)
        raw_close = pd.Series([100.0] * n, index=dates)

        score = compute_clenow_score(adj_close, raw_close, score_window=90)
        assert score == 0.0

    def test_window_parameter_override(self):
        """Custom window size should be respected."""
        n = 60
        adj_close = _make_series(n, 0.001, noise_std=0.0)
        raw_close = adj_close.copy()

        # With window=50, 60 data points is enough
        score = compute_clenow_score(adj_close, raw_close, score_window=50)
        assert score != 0.0

        # With window=70, 60 data points is insufficient
        score = compute_clenow_score(adj_close, raw_close, score_window=70)
        assert score == 0.0


# ===========================================================================
# 8. Parameterize annualization_days
# ===========================================================================

class TestAnnualizationDays:
    def test_score_with_custom_annualization(self):
        """Score formula: (exp(slope * annual) - 1) * R^2. Custom annual changes magnitude."""
        # 90-day linear uptrend, 0.001 per day in log space
        dates = pd.date_range("2023-01-01", periods=90, freq="D")
        adj = pd.Series(np.exp(np.arange(90) * 0.001), index=dates)
        raw = adj.copy()

        score_252 = compute_clenow_score(adj_close=adj, raw_close=raw, annualization_days=252)
        score_244 = compute_clenow_score(adj_close=adj, raw_close=raw, annualization_days=244)

        # 252 produces higher annualized magnitude than 244
        assert score_252 > score_244 > 0

    def test_score_with_custom_window(self):
        """Custom score_window changes regression input length."""
        # Use a series with noise so different windows produce different slopes
        adj_close = _make_series(120, 0.002, noise_std=0.005, seed=42)
        raw_close = adj_close.copy()

        score_90 = compute_clenow_score(adj_close=adj_close, raw_close=raw_close, score_window=90)
        score_60 = compute_clenow_score(adj_close=adj_close, raw_close=raw_close, score_window=60)

        # Both positive (uptrend), values differ due to different windows
        assert score_90 > 0
        assert score_60 > 0
        assert score_90 != score_60

    def test_default_annualization_preserves_behavior(self):
        """Default annualization_days=252 should match existing hardcoded behavior."""
        n = 90
        daily_ret = 0.002
        adj_close = _make_series(n, daily_ret, noise_std=0.0)
        raw_close = adj_close.copy()

        score_default = compute_clenow_score(adj_close, raw_close)
        score_explicit = compute_clenow_score(adj_close, raw_close, annualization_days=252)

        assert abs(score_default - score_explicit) < 1e-10
