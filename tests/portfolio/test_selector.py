"""Tests for the portfolio selector module — regime, SMA, price, ADV filters."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from clenow.config import Config
from clenow.markets.profiles import MarketProfile
from clenow.markets.costs import USCostModel
from clenow.portfolio.selector import apply_filters
from clenow.types import Position


# ── Helpers ──────────────────────────────────────────────────────────
# ADV filter requires >= 20 rows, so all stock data must have >= 25 rows.
# Index data only needs enough for regime_sma + buffer.

N_STOCK = 25  # rows of stock data (enough for 20-day ADV window)


def _test_profile(regime_sma_window: int = 5) -> MarketProfile:
    """Build a MarketProfile with small windows suitable for test data."""
    return MarketProfile(
        name="TEST",
        currency="USD",
        universe_index_id="SP500",
        regime_index_id="SP500",
        annualization_days=252,
        lot_size=1,
        min_price=1.0,
        min_adv_amount=1.0,
        trading_calendar_name="NYSE",
        cost_model=USCostModel(),
        default_starting_capital=100_000,
        regime_sma_window=regime_sma_window,
    )


def _make_index_prices(dates: list, closes: list[float]) -> pd.DataFrame:
    """Build a DataFrame matching get_index_prices output."""
    return pd.DataFrame(
        {"raw_close": closes},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_stock_all_prices(
    ticker: str, dates: list, closes: list[float], volumes: list[float]
) -> pd.DataFrame:
    """Build a MultiIndex DataFrame with (date, ticker) index for all_prices."""
    idx = pd.MultiIndex.from_tuples(
        [(d, ticker) for d in dates],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {
            "raw_close": closes,
            "raw_high": [c + 1.0 for c in closes],
            "raw_low": [c - 1.0 for c in closes],
            "volume": volumes,
        },
        index=idx,
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


def _passing_stock_all_prices(
    ticker: str = "AAPL",
    price: float = 50.0,
    volume: float = 1_000_000.0,
    n: int = N_STOCK,
) -> pd.DataFrame:
    """Stock that passes SMA, price, and ADV filters as MultiIndex all_prices."""
    dates = pd.date_range("2024-05-01", periods=n, freq="D")
    # Rising prices so close > SMA
    closes = [price - 2.0] * (n // 2) + [price + i * 0.5 for i in range(n - n // 2)]
    volumes = [volume] * n
    return _make_stock_all_prices(ticker, dates.tolist(), closes, volumes)


# ── Tests ────────────────────────────────────────────────────────────


class TestRegimeFilter:
    """Bear regime: index < SMA → no new positions, but keep existing."""

    def test_bear_regime_blocks_new_entry(self):
        """In bear regime, new tickers are filtered out."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)
        profile = _test_profile(regime_sma_window=5)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bear_index()

        all_prices = _passing_stock_all_prices("AAPL")

        result = apply_filters(
            ["AAPL"], all_prices, provider, as_of, config,
            current_positions=None, profile=profile,
        )
        assert result == []  # blocked by regime

    def test_bear_regime_keeps_existing_position(self):
        """In bear regime, existing positions in the ranked list are KEPT."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)
        profile = _test_profile(regime_sma_window=5)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bear_index()

        all_prices = _passing_stock_all_prices("AAPL")

        existing = {
            "AAPL": Position(
                ticker="AAPL",
                shares=100,
                entry_price=50.0,
                entry_date=date(2024, 1, 1),
                atr_at_entry=2.0,
            )
        }
        result = apply_filters(
            ["AAPL"], all_prices, provider, as_of, config,
            current_positions=existing, profile=profile,
        )
        assert result == ["AAPL"]  # existing position kept

    def test_bull_regime_allows_new_entry(self):
        """In bull regime (index > SMA), new entries are allowed."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)
        profile = _test_profile(regime_sma_window=5)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        all_prices = _passing_stock_all_prices("AAPL")

        result = apply_filters(
            ["AAPL"], all_prices, provider, as_of, config,
            current_positions=None, profile=profile,
        )
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
        all_prices = _make_stock_all_prices("AAPL", dates.tolist(), closes, volumes)

        result = apply_filters(["AAPL"], all_prices, provider, as_of, config)
        assert result == []

    def test_above_sma_kept(self):
        """Stock above its SMA is kept."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        all_prices = _passing_stock_all_prices("AAPL")

        result = apply_filters(["AAPL"], all_prices, provider, as_of, config)
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
        all_prices = _make_stock_all_prices("AAPL", dates.tolist(), closes, volumes)

        result = apply_filters(["AAPL"], all_prices, provider, as_of, config)
        assert result == []

    def test_at_min_price_kept(self):
        """Stock priced exactly at min_price is kept."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=5.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        # Explicit profile with min_price=5.0 and low ADV threshold
        profile = MarketProfile(
            name="TEST",
            currency="USD",
            universe_index_id="SP500",
            regime_index_id="SP500",
            annualization_days=252,
            lot_size=1,
            min_price=5.0,
            min_adv_amount=1.0,
            trading_calendar_name="NYSE",
            cost_model=USCostModel(),
            default_starting_capital=100_000,
            regime_sma_window=5,
        )

        all_prices = _passing_stock_all_prices("AAPL", price=5.0)

        result = apply_filters(
            ["AAPL"], all_prices, provider, as_of, config, profile=profile,
        )
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
        all_prices = _passing_stock_all_prices("AAPL", price=50.0, volume=100_000.0)

        result = apply_filters(["AAPL"], all_prices, provider, as_of, config)
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
        all_prices = _passing_stock_all_prices("AAPL", price=50.0, volume=500_000.0)

        result = apply_filters(["AAPL"], all_prices, provider, as_of, config)
        assert result == ["AAPL"]


class TestMultipleFilters:
    """Multiple filters applied sequentially, preserving rank order."""

    def test_combined_filters_preserve_order(self):
        """Filters applied in order, rank order preserved for passing stocks."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=5.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        # Build all_prices for multiple tickers
        n = N_STOCK
        dates = pd.date_range("2024-05-01", periods=n, freq="D")

        # AAPL passes
        aapl_data = _make_stock_all_prices(
            "AAPL", dates.tolist(),
            [50.0] * (n // 2) + [55.0 + i for i in range(n - n // 2)],
            [1_000_000.0] * n,
        )

        # MSFT passes
        msft_data = _make_stock_all_prices(
            "MSFT", dates.tolist(),
            [50.0] * (n // 2) + [55.0 + i for i in range(n - n // 2)],
            [1_000_000.0] * n,
        )

        # GOOG passes
        goog_data = _make_stock_all_prices(
            "GOOG", dates.tolist(),
            [50.0] * (n // 2) + [55.0 + i for i in range(n - n // 2)],
            [1_000_000.0] * n,
        )

        all_prices = pd.concat([aapl_data, msft_data, goog_data])

        result = apply_filters(
            ["AAPL", "MSFT", "GOOG"], all_prices, provider, as_of, config
        )
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
        aapl_data = _make_stock_all_prices(
            "AAPL", dates.tolist(),
            [50.0] * (n // 2) + [55.0 + i for i in range(n - n // 2)],
            [1_000_000.0] * n,
        )

        # MSFT fails SMA (declining)
        msft_data = _make_stock_all_prices(
            "MSFT", dates.tolist(),
            [100.0] * (n // 2) + [60.0 - i * 2 for i in range(n - n // 2)],
            [1_000_000.0] * n,
        )

        # GOOG passes
        goog_data = _make_stock_all_prices(
            "GOOG", dates.tolist(),
            [30.0] * (n // 2) + [35.0 + i for i in range(n - n // 2)],
            [1_000_000.0] * n,
        )

        all_prices = pd.concat([aapl_data, msft_data, goog_data])

        result = apply_filters(["AAPL", "MSFT", "GOOG"], all_prices, provider, as_of, config)
        assert result == ["AAPL", "GOOG"]  # MSFT filtered out, order preserved


class TestEmptyInput:
    """Empty ranked_tickers → empty result."""

    def test_empty_input(self):
        as_of = date(2024, 6, 1)
        config = Config()
        provider = MagicMock()
        all_prices = pd.DataFrame()
        result = apply_filters([], all_prices, provider, as_of, config)
        assert result == []


class TestProfileThresholds:
    """Verify that min_price / min_adv are sourced from MarketProfile, not Config."""

    def test_apply_filters_uses_profile_min_price(self):
        """When profile.min_price=5, a stock at $3 is filtered out."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        # Profile with min_price=5.0 (higher than config's 1.0)
        profile = MarketProfile(
            name="TEST",
            currency="USD",
            universe_index_id="SP500",
            regime_index_id="SP500",
            annualization_days=252,
            lot_size=1,
            min_price=5.0,
            min_adv_amount=1.0,
            trading_calendar_name="NYSE",
            cost_model=USCostModel(),
            default_starting_capital=100_000,
            regime_sma_window=5,
        )

        # Stock at $3 — would pass config.min_price=1.0 but fail profile.min_price=5.0
        n = N_STOCK
        dates = pd.date_range("2024-05-01", periods=n, freq="D")
        # Rising from 2 to 3.5 → latest close ~3.5, below profile min_price=5
        closes = [2.0] * (n // 2) + [2.0 + i * 0.2 for i in range(n - n // 2)]
        volumes = [1_000_000.0] * n
        all_prices = _make_stock_all_prices("LOW", dates.tolist(), closes, volumes)

        result = apply_filters(
            ["LOW"], all_prices, provider, as_of, config,
            current_positions=None, profile=profile,
        )
        assert "LOW" not in result

    def test_apply_filters_uses_profile_min_adv(self):
        """When profile.min_adv_amount=10M, a stock with $1M ADV is filtered out."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        # Profile with min_adv_amount=10M (higher than config's 1.0)
        profile = MarketProfile(
            name="TEST",
            currency="USD",
            universe_index_id="SP500",
            regime_index_id="SP500",
            annualization_days=252,
            lot_size=1,
            min_price=1.0,
            min_adv_amount=10_000_000,
            trading_calendar_name="NYSE",
            cost_model=USCostModel(),
            default_starting_capital=100_000,
            regime_sma_window=5,
        )

        # Stock with low volume → ADV = 50 * 10_000 = $500K, below $10M
        all_prices = _passing_stock_all_prices("THIN", price=50.0, volume=10_000.0)

        result = apply_filters(
            ["THIN"], all_prices, provider, as_of, config,
            current_positions=None, profile=profile,
        )
        assert "THIN" not in result

    def test_profile_overrides_config_thresholds(self):
        """Profile thresholds take precedence over config values."""
        as_of = date(2024, 6, 1)
        # Config allows penny stocks and thin liquidity
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = _bull_index()

        # Profile requires $5 min price and $10M min ADV
        profile = MarketProfile(
            name="TEST",
            currency="USD",
            universe_index_id="SP500",
            regime_index_id="SP500",
            annualization_days=252,
            lot_size=1,
            min_price=5.0,
            min_adv_amount=10_000_000,
            trading_calendar_name="NYSE",
            cost_model=USCostModel(),
            default_starting_capital=100_000,
            regime_sma_window=5,
        )

        # LOW: $3 stock with $3M ADV — passes config thresholds, fails profile
        # HIGH: $50 stock with $50M ADV — passes both
        n = N_STOCK
        dates = pd.date_range("2024-05-01", periods=n, freq="D")

        low_closes = [3.0] * (n // 2) + [3.0 + i * 0.1 for i in range(n - n // 2)]
        low_volumes = [1_000_000.0] * n
        low_data = _make_stock_all_prices("LOW", dates.tolist(), low_closes, low_volumes)

        high_closes = [50.0] * (n // 2) + [55.0 + i * 0.5 for i in range(n - n // 2)]
        high_volumes = [1_000_000.0] * n
        high_data = _make_stock_all_prices("HIGH", dates.tolist(), high_closes, high_volumes)

        all_prices = pd.concat([low_data, high_data])

        result = apply_filters(
            ["LOW", "HIGH"], all_prices, provider, as_of, config,
            current_positions=None, profile=profile,
        )
        assert "LOW" not in result
        assert "HIGH" in result


class TestApplyFiltersUsesPreloadedPrices:
    def test_filters_read_from_all_prices_not_provider(self):
        """apply_filters must NOT call data_provider.load_prices for stock data."""
        from clenow.config import Config
        from clenow.portfolio.selector import apply_filters

        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        provider = MagicMock()
        provider.get_index_prices.return_value = pd.DataFrame(
            {"raw_close": [100.0]},
            index=pd.DatetimeIndex([as_of], name="date"),  # bull regime
        )
        # Force any accidental call to surface as a test failure:
        provider.load_prices.side_effect = AssertionError(
            "apply_filters must not query prices when all_prices is supplied"
        )

        # Build all_prices with a ticker that passes all filters
        # Need >= 20 rows for ADV filter, and >= 5 for SMA
        n = 25
        dates = pd.date_range("2024-05-01", periods=n, freq="D")
        # Rising prices so close > SMA (passes SMA filter)
        closes = [10.0 - 1.0] * (n // 2) + [10.0 + i * 0.5 for i in range(n - n // 2)]
        volumes = [1_000_000.0] * n  # ADV = 10 * 1M = $10M, passes

        idx = pd.MultiIndex.from_tuples(
            [(d, "AAA") for d in dates.tolist()],
            names=["date", "ticker"],
        )
        all_prices = pd.DataFrame(
            {
                "raw_close": closes,
                "raw_high": [c + 1.0 for c in closes],
                "raw_low": [c - 1.0 for c in closes],
                "volume": volumes,
            },
            index=idx,
        )

        result = apply_filters(
            ranked_tickers=["AAA"],
            all_prices=all_prices,
            data_provider=provider,
            as_of=as_of,
            config=config,
            current_positions=None,
        )

        assert result == ["AAA"]
        provider.load_prices.assert_not_called()
        provider.get_index_prices.assert_called_once()


class TestBearRegimeCache:
    """bear_regime_cache parameter skips DB calls for regime detection."""

    def test_apply_filters_bear_regime_cache_skips_db(self):
        """When bear_regime_cache is provided, get_index_prices is not called."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        mock_provider = MagicMock()
        # If the cache is bypassed and DB is queried, this would be called
        mock_provider.get_index_prices.side_effect = AssertionError(
            "Should not query index prices when cache is provided"
        )

        # Build stock data that would pass all filters (but bear regime blocks entries)
        n = N_STOCK
        dates = pd.date_range("2024-05-01", periods=n, freq="D")
        closes = [50.0 - 2.0] * (n // 2) + [50.0 + i * 0.5 for i in range(n - n // 2)]
        volumes = [1_000_000.0] * n
        all_prices = _make_stock_all_prices("AAPL", dates.tolist(), closes, volumes)

        result = apply_filters(
            ranked_tickers=["AAPL"],
            all_prices=all_prices,
            data_provider=mock_provider,
            as_of=as_of,
            config=config,
            current_positions=None,
            bear_regime_cache={as_of: True},  # bear market via cache
        )

        # In bear regime with no existing positions → all filtered
        assert result == []
        # Verify get_index_prices was never called
        mock_provider.get_index_prices.assert_not_called()

    def test_apply_filters_bull_regime_cache_allows_entries(self):
        """When bear_regime_cache says bull (False), entries are allowed without DB call."""
        as_of = date(2024, 6, 1)
        config = Config(regime_sma=5, stock_sma=5, min_price=1.0, min_adv_dollars=1.0)

        mock_provider = MagicMock()
        # If the cache is bypassed and DB is queried, this would be called
        mock_provider.get_index_prices.side_effect = AssertionError(
            "Should not query index prices when cache is provided"
        )

        # Build stock data that passes all filters
        all_prices = _passing_stock_all_prices("AAPL")

        result = apply_filters(
            ranked_tickers=["AAPL"],
            all_prices=all_prices,
            data_provider=mock_provider,
            as_of=as_of,
            config=config,
            current_positions=None,
            bear_regime_cache={as_of: False},  # bull market via cache
        )

        # In bull regime, AAPL passes all filters
        assert "AAPL" in result
        # Verify get_index_prices was never called
        mock_provider.get_index_prices.assert_not_called()