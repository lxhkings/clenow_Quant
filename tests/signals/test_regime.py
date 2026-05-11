"""Tests for regime signal — bull/bear detection via index SMA."""

import numpy as np
import pandas as pd
import pytest

from clenow.signals.regime import is_bear_regime, is_bull_regime


class TestBullRegime:
    """Index close above 200 SMA -> True (bull)."""

    def test_above_sma_is_bull(self):
        n = 200
        # Steadily rising index: last value is well above SMA
        dates = pd.bdate_range("2024-01-01", periods=n)
        index_close = pd.Series(np.linspace(3000, 4000, n), index=dates, dtype=float)

        assert is_bull_regime(index_close, sma_period=200) is True

    def test_rising_close_above_sma(self):
        """Even with noise, if last close > SMA, it's bull."""
        n = 200
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=n)
        # Upward trend with noise
        base = np.linspace(3000, 3500, n)
        index_close = pd.Series(base + rng.normal(0, 10, n), index=dates)

        result = is_bull_regime(index_close, sma_period=200)
        # Last value ~3500, SMA ~3250, should be bull
        assert result is True


class TestBearRegime:
    """Index close below 200 SMA -> False (bear)."""

    def test_below_sma_is_bear(self):
        n = 200
        # Steadily declining index: last value is well below SMA
        dates = pd.bdate_range("2024-01-01", periods=n)
        index_close = pd.Series(np.linspace(4000, 3000, n), index=dates, dtype=float)

        assert is_bull_regime(index_close, sma_period=200) is False

    def test_flat_then_drop(self):
        """Flat market then a drop -> bear."""
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        values = np.full(n, 3000.0, dtype=float)
        values[-10:] = 2800.0  # Drop at the end
        index_close = pd.Series(values, index=dates)

        # SMA ≈ (190 * 3000 + 10 * 2800) / 200 = 2990
        # Last close = 2800 < 2990 -> bear
        assert is_bull_regime(index_close, sma_period=200) is False


class TestInsufficientData:
    """Insufficient data for SMA -> default to True (bull)."""

    def test_short_series_defaults_bull(self):
        n = 100  # Less than 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        index_close = pd.Series(np.linspace(4000, 2000, n), index=dates, dtype=float)

        # Even though it's declining, insufficient data -> default bull
        assert is_bull_regime(index_close, sma_period=200) is True

    def test_empty_series_defaults_bull(self):
        index_close = pd.Series([], dtype=float)
        assert is_bull_regime(index_close, sma_period=200) is True

    def test_exactly_200_bars(self):
        """Exactly 200 bars should compute."""
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        index_close = pd.Series(np.linspace(3000, 4000, n), index=dates, dtype=float)

        assert is_bull_regime(index_close, sma_period=200) is True


class TestBoundaryCondition:
    def test_exactly_at_sma(self):
        """Last close == SMA -> not strictly above -> False."""
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        # All same value: close == SMA
        index_close = pd.Series(np.full(n, 3000.0), index=dates, dtype=float)

        # 3000 > 3000 is False
        assert is_bull_regime(index_close, sma_period=200) is False


class TestSP500HistoricalDate:
    """Test with data that mimics 2018-12-04 SP500 dropping below 200 SMA.

    The S&P 500 closed below its 200-day SMA on 2018-12-04 for the
    first time since mid-2016. We construct approximate data for this.
    """

    def test_sp500_2018_bear_signal(self):
        # Construct 200 days of approximate SP500 data
        # The index was around 2760 on 2018-12-04, having dropped from ~2940 peak
        # The 200-day SMA was around 2770
        n = 200
        dates = pd.bdate_range("2018-03-01", periods=n)
        rng = np.random.default_rng(88)

        # Build a series that rises to ~2940 then falls to ~2760
        rise = np.linspace(2600, 2940, 150)
        fall = np.linspace(2940, 2760, 50)
        base = np.concatenate([rise, fall])
        # Add tiny noise
        index_close = pd.Series(base + rng.normal(0, 5, n), index=dates)

        # The last value (~2760) should be below the 200-day SMA (~2800ish)
        result = is_bull_regime(index_close, sma_period=200)
        assert result is False


class TestWithNaN:
    def test_nan_values_are_dropped(self):
        """NaN values in the series should be dropped before computing SMA."""
        n = 210
        dates = pd.bdate_range("2024-01-01", periods=n)
        values = np.linspace(3000, 4000, n, dtype=float)
        # Set 10 values to NaN
        values[5:15] = np.nan
        index_close = pd.Series(values, index=dates)

        # After dropping NaN, we have 200 values, enough for SMA
        result = is_bull_regime(index_close, sma_period=200)
        # The series is still upward trending, so it should be bull
        assert result is True


class TestBearRegimeCustomWindow:
    """is_bear_regime with custom sma_window parameter."""

    def test_is_bear_regime_custom_window(self):
        """Verify sma_window parameter changes the lookback."""
        # Up-trend last 100 days, then down-trend last 150 days (total 250)
        dates = pd.date_range("2023-01-01", periods=250, freq="D")
        prices = pd.Series(
            list(range(100, 200)) + list(range(200, 50, -1)),  # 100 up, 150 down
            index=dates,
        )
        # 200-window: average ~125, current ~50 -> bear
        assert is_bear_regime(prices, sma_window=200) is True
        # 100-window: average ~100, current ~50 -> bear
        assert is_bear_regime(prices, sma_window=100) is True
        # 50-window: average ~75, current ~50 -> bear
        assert is_bear_regime(prices, sma_window=50) is True

    def test_bear_regime_above_sma(self):
        """Rising prices above SMA -> not bear."""
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)
        prices = pd.Series(np.linspace(3000, 4000, n), index=dates, dtype=float)
        assert is_bear_regime(prices, sma_window=200) is False

    def test_bear_regime_insufficient_data(self):
        """Insufficient data -> default to False (not bear)."""
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        prices = pd.Series(np.linspace(4000, 2000, n), index=dates, dtype=float)
        assert is_bear_regime(prices, sma_window=200) is False
