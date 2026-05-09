"""Tests for the portfolio selector module — regime, SMA, price, ADV filters."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from clenow.config import Config
from clenow.portfolio.selector import apply_filters
from clenow.types import Position


# ── Helpers ──────────────────────────────────────────────────────────
# ADV filter requires >= 20 rows, so all stock data must have >= 25 rows.
# Index data only needs enough for regime_sma + buffer.

N_STOCK = 25  # rows of stock data (enough for 20-day ADV window)


def _make_index_prices(dates: list, closes: list[float]) -> pd.DataFrame:
    """Build a DataFrame matching get_index_prices output."""
    return pd.DataFrame(
        {"raw_close": closes},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_stock_prices(
    dates: list, closes: list[float], volumes: list[float]
) -> pd.DataFrame:
    """Build a DataFrame matching load_prices output for a single ticker."""
    return pd.DataFrame(
        {"raw_close": closes, "volume": volumes},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _bull_index(n: int = 15) -> pd.DataFrame:
    """Index prices where SP500 > SMA (bull regime)."""
    dates = pd.date_range("2024-05-01", periods=n, freq="D")
    closes = [80.0] * (n // 2) + [100.0 + i for i in range(n - n // 2)]
    return _make_index_prices(dates.tolist(), closes)


def _bear_index(n: int = 15) -> pd.DataFrame:
    """Index prices where SP500 < SMA (bear regime)."""
    dates = pd.date_range("2024-05-01", periods=n, freq="D")
    closes = [100.0] * (n // 2) + [90.0 - i for i in range(n - n // 2)]
    return _make_index_prices(dates.tolist(), closes)


def _passing_stock(
    price: float = 50.0,
    volume: float = 1_000_000.0,
    n: int = N_STOCK,
) -> pd.DataFrame:
    """Stock that passes SMA, price, and ADV filters."""
    dates = pd.date_range("2024-05-01", periods=n, freq="D")
    # Rising prices so close > SMA
    closes = [price - 2.0] * (n // 2) + [price + i * 0.5 for i in range(n - n // 2)]
    volumes = [volume] * n
    return _make_stock_prices(dates.tolist(), closes, volumes)


# ── Tests ────────────────────────────────────────────────────────────


class TestRegimeFilter:
    """Bear regime: SP500 < 200-SMA → no new positions, but keep existing."""

    def test_bear_regime_blocks_new_entry(self):
        """In bear regime, new tickers are filtered out."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bear_index()
        provider.load_prices.return_value = _passing_stock()

        result = apply_filters(["AAPL"], provider, as_of, config, current_positions=None)
        assert result == []  # blocked by regime

    def test_bear_regime_keeps_existing_position(self):
        """In bear regime, existing positions in the ranked list are KEPT."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bear_index()
        provider.load_prices.return_value = _passing_stock()

        existing = {
            "AAPL": Position(
                ticker="AAPL", shares=100, entry_price=50.0,
                entry_date=date(2024, 1, 1), atr_at_entry=2.0,
            )
        }
        result = apply_filters(
            ["AAPL"], provider, as_of, config, current_positions=existing
        )
        assert result == ["AAPL"]  # existing position kept

    def test_bull_regime_allows_new_entry(self):
        """In bull regime (SP500 > SMA), new entries are allowed."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()
        provider.load_prices.return_value = _passing_stock()

        result = apply_filters(["AAPL"], provider, as_of, config, current_positions=None)
        assert result == ["AAPL"]


class TestSMAFilter:
    """Stock close < 100-day SMA → removed."""

    def test_below_sma_filtered_out(self):
        """Stock below its SMA is removed."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        # Declining stock: close < 5-day SMA
        n = N_STOCK
        dates = pd.date_range("2024-05-01", periods=n, freq="D")
        closes = [100.0] * (n // 2) + [80.0 - i * 2 for i in range(n - n // 2)]
        volumes = [1_000_000.0] * n
        provider.load_prices.return_value = _make_stock_prices(
            dates.tolist(), closes, volumes
        )

        result = apply_filters(["AAPL"], provider, as_of, config)
        assert result == []

    def test_above_sma_kept(self):
        """Stock above its SMA is kept."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()
        provider.load_prices.return_value = _passing_stock()

        result = apply_filters(["AAPL"], provider, as_of, config)
        assert result == ["AAPL"]


class TestPriceFilter:
    """raw_close < min_price → removed."""

    def test_penny_stock_filtered_out(self):
        """Stock priced below min_price is removed."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=5.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        # Flat prices below $5 so the latest close is < min_price
        n = N_STOCK
        dates = pd.date_range("2024-05-01", periods=n, freq="D")
        # Rising from 3 to 4 → latest close is 4.0, below $5
        closes = [3.0] * (n // 2) + [3.0 + i * 0.1 for i in range(n - n // 2)]
        volumes = [1_000_000.0] * n
        provider.load_prices.return_value = _make_stock_prices(
            dates.tolist(), closes, volumes
        )

        result = apply_filters(["AAPL"], provider, as_of, config)
        assert result == []

    def test_at_min_price_kept(self):
        """Stock priced exactly at min_price is kept."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=5.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()
        provider.load_prices.return_value = _passing_stock(price=5.0)

        result = apply_filters(["AAPL"], provider, as_of, config)
        assert result == ["AAPL"]


class TestADVFilter:
    """20-day ADV < min_adv_dollars → removed."""

    def test_low_adv_filtered_out(self):
        """Stock with ADV below threshold is removed."""
        as_of = date(2024, 6, 1)
        config = Config(
            regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=10_000_000
        )

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()
        # ADV = 50 * 100_000 = $5M → below $10M
        provider.load_prices.return_value = _passing_stock(
            price=50.0, volume=100_000.0
        )

        result = apply_filters(["AAPL"], provider, as_of, config)
        assert result == []

    def test_sufficient_adv_kept(self):
        """Stock with ADV above threshold is kept."""
        as_of = date(2024, 6, 1)
        config = Config(
            regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=10_000_000
        )

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()
        # ADV = 50 * 500_000 = $25M → above $10M
        provider.load_prices.return_value = _passing_stock(
            price=50.0, volume=500_000.0
        )

        result = apply_filters(["AAPL"], provider, as_of, config)
        assert result == ["AAPL"]


class TestMultipleFilters:
    """Multiple filters applied sequentially, preserving rank order."""

    def test_combined_filters_preserve_order(self):
        """Filters applied in order, rank order preserved for passing stocks."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=5.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()
        provider.load_prices.return_value = _passing_stock(price=50.0)

        result = apply_filters(["AAPL", "MSFT", "GOOG"], provider, as_of, config)
        assert result == ["AAPL", "MSFT", "GOOG"]

    def test_mixed_pass_fail_preserves_order(self):
        """Some stocks pass, some fail — output preserves rank order of passers."""
        as_of = date(2024, 6, 1)
        config = Config(
            regime_sma=5, stock_sma=5, min_price=5.0, min_adv_dollars=1.0
        )

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        n = N_STOCK
        dates = pd.date_range("2024-05-01", periods=n, freq="D")

        # AAPL passes (above SMA, above price, good ADV)
        aapl_closes = [50.0] * (n // 2) + [55.0 + i for i in range(n - n // 2)]
        aapl_volumes = [1_000_000.0] * n

        # MSFT fails SMA (declining)
        msft_closes = [100.0] * (n // 2) + [60.0 - i * 2 for i in range(n - n // 2)]
        msft_volumes = [1_000_000.0] * n

        # GOOG passes
        goog_closes = [30.0] * (n // 2) + [35.0 + i for i in range(n - n // 2)]
        goog_volumes = [1_000_000.0] * n

        def load_prices_side_effect(tickers, start, end):
            frames = {}
            for t in tickers:
                if t == "AAPL":
                    frames[t] = _make_stock_prices(dates.tolist(), aapl_closes, aapl_volumes)
                elif t == "MSFT":
                    frames[t] = _make_stock_prices(dates.tolist(), msft_closes, msft_volumes)
                elif t == "GOOG":
                    frames[t] = _make_stock_prices(dates.tolist(), goog_closes, goog_volumes)
                else:
                    frames[t] = _passing_stock()
            # Return concatenated with MultiIndex if multiple tickers
            if len(frames) == 1:
                return list(frames.values())[0]
            parts = []
            for t, df in frames.items():
                df_copy = df.copy()
                df_copy["ticker"] = t
                df_copy = df_copy.set_index("ticker", append=True)
                parts.append(df_copy)
            return pd.concat(parts)

        provider.load_prices.side_effect = load_prices_side_effect

        result = apply_filters(["AAPL", "MSFT", "GOOG"], provider, as_of, config)
        assert result == ["AAPL", "GOOG"]  # MSFT filtered out, order preserved


class TestEmptyInput:
    """Empty ranked_tickers → empty result."""

    def test_empty_input(self):
        as_of = date(2024, 6, 1)
        config = Config()
        provider = MagicMock()
        result = apply_filters([], provider, as_of, config)
        assert result == []
