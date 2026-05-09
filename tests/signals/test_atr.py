"""Tests for ATR signal — Wilder RMA with hand-calculated fixtures."""

import math

import numpy as np
import pandas as pd
import pytest

from clenow.signals.atr import compute_atr


def _make_ohlc(n: int, base_price: float = 100.0, daily_range: float = 3.0,
               seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Create synthetic OHLC data with a known daily range."""
    rng = np.random.default_rng(seed)
    closes = np.empty(n)
    closes[0] = base_price
    for i in range(1, n):
        closes[i] = closes[i - 1] + rng.normal(0, 0.5)

    highs = closes + rng.uniform(0.5, daily_range, n)
    lows = closes - rng.uniform(0.5, daily_range, n)

    dates = pd.bdate_range("2024-01-01", periods=n)
    h = pd.Series(highs, index=dates)
    l = pd.Series(lows, index=dates)
    c = pd.Series(closes, index=dates)
    return h, l, c


class TestATRHandCalculation:
    """Compute TR and RMA by hand for a small dataset and verify."""

    def test_manual_tr_and_rma(self):
        """Use 6 bars with period=5 to compute ATR by hand."""
        # Hand-crafted data so we can compute everything manually
        dates = pd.bdate_range("2024-01-01", periods=6)
        high = pd.Series([105, 103, 108, 107, 106, 109], index=dates, dtype=float)
        low = pd.Series([98, 97, 100, 99, 100, 102], index=dates, dtype=float)
        close = pd.Series([102, 101, 106, 103, 104, 107], index=dates, dtype=float)

        # TR calculations:
        # Day 1: max(103-97, |103-102|, |97-102|) = max(6, 1, 5) = 6
        # Day 2: max(108-100, |108-101|, |100-101|) = max(8, 7, 1) = 8
        # Day 3: max(107-99, |107-106|, |99-106|) = max(8, 1, 7) = 8
        # Day 4: max(106-100, |106-103|, |100-103|) = max(6, 3, 3) = 6
        # Day 5: max(109-102, |109-104|, |102-104|) = max(7, 5, 2) = 7

        # TR array: [6, 8, 8, 6, 7]
        # period = 5
        # SMA of first 5 TRs = (6 + 8 + 8 + 6 + 7) / 5 = 35 / 5 = 7.0
        # Only 5 TR values -> RMA[4] = 7.0, no more values to iterate
        # ATR = 7.0

        result = compute_atr(high, low, close, period=5)
        assert abs(result - 7.0) < 1e-10

    def test_rma_iteration(self):
        """Test RMA iteration with more data than the period."""
        dates = pd.bdate_range("2024-01-01", periods=7)
        high = pd.Series([105, 103, 108, 107, 106, 109, 110], index=dates, dtype=float)
        low = pd.Series([98, 97, 100, 99, 100, 102, 101], index=dates, dtype=float)
        close = pd.Series([102, 101, 106, 103, 104, 107, 108], index=dates, dtype=float)

        # TR: [6, 8, 8, 6, 7, 3]
        # Day 0 -> 1: max(103-97, |103-102|, |97-102|) = 6
        # Day 1 -> 2: max(108-100, |108-101|, |100-101|) = 8
        # Day 2 -> 3: max(107-99, |107-106|, |99-106|) = 8
        # Day 3 -> 4: max(106-100, |106-103|, |100-103|) = 6
        # Day 4 -> 5: max(109-102, |109-104|, |102-104|) = 7
        # Day 5 -> 6: max(110-101, |110-107|, |101-107|) = max(9, 3, 6) = 9

        # period = 5
        # SMA of first 5: (6+8+8+6+7)/5 = 7.0
        # RMA[5] = (7.0 * 4 + 9) / 5 = (28 + 9) / 5 = 37/5 = 7.4
        # ATR = 7.4

        result = compute_atr(high, low, close, period=5)
        assert abs(result - 7.4) < 1e-10


class TestATRInsufficientData:
    def test_too_few_bars(self):
        """Need at least period + 1 bars."""
        dates = pd.bdate_range("2024-01-01", periods=5)
        high = pd.Series([105, 103, 108, 107, 106], index=dates, dtype=float)
        low = pd.Series([98, 97, 100, 99, 100], index=dates, dtype=float)
        close = pd.Series([102, 101, 106, 103, 104], index=dates, dtype=float)

        # period=20, only 5 bars -> 0.0
        result = compute_atr(high, low, close, period=20)
        assert result == 0.0

    def test_exactly_period_plus_one(self):
        """Exactly period + 1 bars should compute."""
        n = 21  # period=20 -> need 21 bars
        h, l, c = _make_ohlc(n, seed=10)
        result = compute_atr(h, l, c, period=20)
        assert result != 0.0

    def test_just_under_period_plus_one(self):
        """One less than period + 1 -> insufficient."""
        n = 20  # period=20 -> need 21 bars
        h, l, c = _make_ohlc(n, seed=10)
        result = compute_atr(h, l, c, period=20)
        assert result == 0.0


class TestATRDefensiveFloor:
    def test_atr_below_floor_returns_zero(self):
        """ATR < 0.01 should return 0.0 (illiquid/stopped stock)."""
        n = 25
        dates = pd.bdate_range("2024-01-01", periods=n)
        # Tiny price movements -> very low ATR
        close = pd.Series([1.0 + 0.0001 * i for i in range(n)], index=dates, dtype=float)
        high = close + 0.0001
        low = close - 0.0001

        result = compute_atr(high, low, close, period=20)
        # The ATR will be extremely small, well below 0.01
        assert result == 0.0

    def test_atr_above_floor_returns_value(self):
        """ATR >= 0.01 should return the computed value."""
        n = 25
        dates = pd.bdate_range("2024-01-01", periods=n)
        close = pd.Series([100.0 + i * 0.1 for i in range(n)], index=dates, dtype=float)
        high = close + 2.0
        low = close - 2.0

        result = compute_atr(high, low, close, period=20)
        assert result >= 0.01


class TestATRWithNaN:
    def test_nan_rows_are_dropped(self):
        """NaN rows should be dropped before calculation."""
        dates = pd.bdate_range("2024-01-01", periods=7)
        high = pd.Series([105, 103, np.nan, 107, 106, 109, 110], index=dates, dtype=float)
        low = pd.Series([98, 97, np.nan, 99, 100, 102, 101], index=dates, dtype=float)
        close = pd.Series([102, 101, np.nan, 103, 104, 107, 108], index=dates, dtype=float)

        # After dropping NaN, we have 6 bars -> 5 TR values
        # With period=5, this should compute
        result = compute_atr(high, low, close, period=5)
        assert result != 0.0  # Should still compute

    def test_too_many_nan_returns_zero(self):
        """If NaN drops us below period+1, return 0."""
        dates = pd.bdate_range("2024-01-01", periods=7)
        high = pd.Series([105, np.nan, np.nan, np.nan, 106, 109, 110], index=dates, dtype=float)
        low = pd.Series([98, np.nan, np.nan, np.nan, 100, 102, 101], index=dates, dtype=float)
        close = pd.Series([102, np.nan, np.nan, np.nan, 104, 107, 108], index=dates, dtype=float)

        # After dropping NaN, only 4 bars -> need 6 for period=5
        result = compute_atr(high, low, close, period=5)
        assert result == 0.0


class TestATRDefaultPeriod:
    def test_default_period_20(self):
        """Default period should be 20."""
        h, l, c = _make_ohlc(30, seed=10)
        result = compute_atr(h, l, c)
        assert result > 0.0  # Should compute with default period=20
