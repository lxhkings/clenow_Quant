"""Tests for the backtest engine — compute_target_portfolio and run_backtest.

Uses mock DataProvider to isolate the engine from real data sources.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from clenow.backtest.engine import (
    BacktestResult,
    _slice_prices,
    compute_target_portfolio,
    run_backtest,
)
from clenow.config import Config
from clenow.types import Position, TargetPortfolio


# ── Helpers ──────────────────────────────────────────────────────────

def _make_stock_data(
    n_days: int = 300,
    start_date: date = date(2024, 1, 2),
    ticker: str = "AAPL",
    price: float = 100.0,
    trend: float = 0.0,  # daily price drift
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Build a MultiIndex (date, ticker) price DataFrame for one ticker.

    Generates enough days for 200-day SMA + buffer.
    """
    dates = pd.bdate_range(start_date, periods=n_days)
    closes = [price + i * trend for i in range(n_days)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    opens = [c + 0.5 for c in closes]
    adj_closes = closes.copy()  # No adjustments

    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d.date(),
            "ticker": ticker,
            "raw_open": opens[i],
            "raw_high": highs[i],
            "raw_low": lows[i],
            "raw_close": closes[i],
            "volume": volume,
            "adj_close": adj_closes[i],
            "dividend": 0.0,
            "split_ratio": 1.0,
        })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"])
    return df


def _make_index_data(
    n_days: int = 250,
    start_date: date = date(2024, 1, 2),
    bull: bool = True,
) -> pd.DataFrame:
    """Build index price DataFrame for regime detection.

    If bull=True, rising prices (above SMA).
    If bull=False, declining prices (below SMA).
    """
    dates = pd.date_range(start_date, periods=n_days, freq="B")
    if bull:
        closes = [4000 + i * 2 for i in range(n_days)]
    else:
        closes = [5000 - i * 2 for i in range(n_days)]

    return pd.DataFrame(
        {"raw_close": closes},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_mock_provider(
    universe: list[str] | None = None,
    stock_data: dict[str, pd.DataFrame] | None = None,
    index_data: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a mock DataProvider with configurable behavior."""
    provider = MagicMock()

    if universe is not None:
        provider.get_universe.return_value = universe
    else:
        provider.get_universe.return_value = ["AAPL", "MSFT", "GOOG"]

    def load_prices_side_effect(tickers, start, end):
        if not tickers:
            # Return empty DataFrame matching the expected schema
            idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
            return pd.DataFrame(
                columns=["raw_open", "raw_high", "raw_low", "raw_close",
                          "volume", "adj_close", "dividend", "split_ratio"],
                index=idx,
            )
        if stock_data is None:
            # Generate default data for each ticker
            frames = []
            for t in tickers:
                df = _make_stock_data(ticker=t)
                frames.append(df)
            if len(frames) == 1:
                return frames[0]
            return pd.concat(frames)
        else:
            frames = []
            for t in tickers:
                if t in stock_data:
                    frames.append(stock_data[t])
                else:
                    frames.append(_make_stock_data(ticker=t))
            if len(frames) == 1:
                return frames[0]
            return pd.concat(frames)

    provider.load_prices.side_effect = load_prices_side_effect

    if index_data is not None:
        provider.get_index_prices.return_value = index_data
    else:
        provider.get_index_prices.return_value = _make_index_data(bull=True)

    provider.get_stocks_meta.return_value = pd.DataFrame(
        columns=["name"], index=pd.Index([], name="ticker")
    )

    return provider


class TestComputeTargetPortfolioNormal:
    """Normal case: returns target positions."""

    def test_returns_target_portfolio(self):
        """compute_target_portfolio returns a TargetPortfolio with positions."""
        as_of = date(2024, 12, 31)
        config = Config(
            score_window=90,
            atr_period=20,
            regime_sma=200,
            stock_sma=100,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.20,
            risk_factor=0.001,
            max_position_pct=0.05,
        )

        # Build rising stock data so scores are positive
        stock_data = {}
        for ticker in ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]:
            stock_data[ticker] = _make_stock_data(
                n_days=300, ticker=ticker, price=100.0, trend=0.1, volume=1_000_000.0
            )

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],
            stock_data=stock_data,
        )

        result = compute_target_portfolio(
            as_of=as_of,
            current_positions={},
            current_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        assert isinstance(result, TargetPortfolio)
        assert result.as_of == as_of
        # Should have some positions (at least one of the top 20%)
        assert len(result.positions) >= 1

    def test_pure_function_deterministic(self):
        """Same inputs always produce the same output."""
        as_of = date(2024, 12, 31)
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
        )

        stock_data = {
            "AAPL": _make_stock_data(n_days=300, ticker="AAPL", trend=0.1),
            "MSFT": _make_stock_data(n_days=300, ticker="MSFT", trend=0.05),
        }

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT"],
            stock_data=stock_data,
        )

        result1 = compute_target_portfolio(
            as_of=as_of,
            current_positions={},
            current_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        result2 = compute_target_portfolio(
            as_of=as_of,
            current_positions={},
            current_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        assert result1.positions == result2.positions
        assert result1.as_of == result2.as_of


class TestRegimeOffNoNewPositions:
    """Bear regime: no NEW positions in target, but existing kept."""

    def test_bear_regime_no_new_positions(self):
        """In bear regime, new tickers are not added to the target."""
        as_of = date(2024, 12, 31)
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,  # Include more
        )

        # Declining stock — should pass SMA if it's existing but fail if new
        stock_data = {}
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            stock_data[ticker] = _make_stock_data(
                n_days=300, ticker=ticker, trend=0.1, volume=1_000_000.0
            )

        # Bear index
        bear_index = _make_index_data(n_days=250, bull=False)
        provider = _make_mock_provider(
            universe=["AAPL", "MSFT", "GOOG"],
            stock_data=stock_data,
            index_data=bear_index,
        )

        # No existing positions → everything should be blocked by regime
        result = compute_target_portfolio(
            as_of=as_of,
            current_positions={},
            current_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        assert isinstance(result, TargetPortfolio)
        # In bear regime with no existing positions, target should be empty
        assert len(result.positions) == 0

    def test_bear_regime_existing_positions_kept(self):
        """In bear regime, existing positions that pass SMA filter are kept."""
        as_of = date(2024, 12, 31)
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
        )

        # AAPL with rising prices (above its SMA)
        stock_data = {
            "AAPL": _make_stock_data(n_days=300, ticker="AAPL", trend=0.1),
        }

        bear_index = _make_index_data(n_days=250, bull=False)
        provider = _make_mock_provider(
            universe=["AAPL"],
            stock_data=stock_data,
            index_data=bear_index,
        )

        existing = {
            "AAPL": Position(
                ticker="AAPL", shares=100, entry_price=100.0,
                entry_date=date(2024, 6, 1), atr_at_entry=2.0,
            )
        }

        result = compute_target_portfolio(
            as_of=as_of,
            current_positions=existing,
            current_cash=500_000.0,
            config=config,
            data_provider=provider,
        )

        # AAPL should be in the target (existing position, passes SMA)
        assert "AAPL" in result.positions


class TestEmptyUniverse:
    """Empty universe returns empty TargetPortfolio."""

    def test_empty_universe(self):
        as_of = date(2024, 12, 31)
        config = Config()
        provider = _make_mock_provider(universe=[])

        result = compute_target_portfolio(
            as_of=as_of,
            current_positions={},
            current_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        assert result.positions == {}
        assert result.as_of == as_of


class TestDoubleExitRule:
    """Position breaking 100-day SMA is removed even if in top 20%."""

    def test_sma_break_forced_sell(self):
        """Existing position breaking 100-day SMA is removed from target,
        even if it's in the top 20% of the universe."""
        as_of = date(2024, 12, 31)
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
        )

        # Build stock data for AAPL that is DECLINING at the end
        # (below its 100-day SMA) but has a positive score from earlier rise
        n_days = 300
        dates = pd.bdate_range(date(2024, 1, 2), periods=n_days)
        # Rise for 200 days then decline for 100
        closes = []
        for i in range(n_days):
            if i < 200:
                closes.append(100 + i * 0.2)  # Rising
            else:
                closes.append(140 - (i - 200) * 0.5)  # Declining sharply

        aapl_data = pd.DataFrame({
            "raw_open": [c + 0.5 for c in closes],
            "raw_high": [c + 1.0 for c in closes],
            "raw_low": [c - 1.0 for c in closes],
            "raw_close": closes,
            "volume": 1_000_000.0,
            "adj_close": closes,
            "dividend": 0.0,
            "split_ratio": 1.0,
        }, index=pd.MultiIndex.from_arrays(
            [dates.date, ["AAPL"] * n_days],
            names=["date", "ticker"]
        ))

        stock_data = {"AAPL": aapl_data}
        bull_index = _make_index_data(n_days=250, bull=True)
        provider = _make_mock_provider(
            universe=["AAPL"],
            stock_data=stock_data,
            index_data=bull_index,
        )

        # AAPL is an existing position
        existing = {
            "AAPL": Position(
                ticker="AAPL", shares=100, entry_price=100.0,
                entry_date=date(2024, 6, 1), atr_at_entry=2.0,
            )
        }

        result = compute_target_portfolio(
            as_of=as_of,
            current_positions=existing,
            current_cash=500_000.0,
            config=config,
            data_provider=provider,
        )

        # AAPL breaks its 100-day SMA → forced sell → NOT in target
        assert "AAPL" not in result.positions

    def test_sma_break_overrides_top_rank(self):
        """Even if a stock is top-ranked, SMA break forces it out of target."""
        as_of = date(2024, 12, 31)
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
        )

        # MSFT: strong uptrend (in top 20%)
        # AAPL: recent decline (breaks 100-day SMA)
        n_days = 300
        dates = pd.bdate_range(date(2024, 1, 2), periods=n_days)

        msft_closes = [100 + i * 0.2 for i in range(n_days)]

        # AAPL: rise then decline
        aapl_closes = []
        for i in range(n_days):
            if i < 200:
                aapl_closes.append(100 + i * 0.2)
            else:
                aapl_closes.append(140 - (i - 200) * 0.5)

        msft_data = pd.DataFrame({
            "raw_open": [c + 0.5 for c in msft_closes],
            "raw_high": [c + 1.0 for c in msft_closes],
            "raw_low": [c - 1.0 for c in msft_closes],
            "raw_close": msft_closes,
            "volume": 1_000_000.0,
            "adj_close": msft_closes,
            "dividend": 0.0,
            "split_ratio": 1.0,
        }, index=pd.MultiIndex.from_arrays(
            [dates.date, ["MSFT"] * n_days],
            names=["date", "ticker"]
        ))

        aapl_data = pd.DataFrame({
            "raw_open": [c + 0.5 for c in aapl_closes],
            "raw_high": [c + 1.0 for c in aapl_closes],
            "raw_low": [c - 1.0 for c in aapl_closes],
            "raw_close": aapl_closes,
            "volume": 1_000_000.0,
            "adj_close": aapl_closes,
            "dividend": 0.0,
            "split_ratio": 1.0,
        }, index=pd.MultiIndex.from_arrays(
            [dates.date, ["AAPL"] * n_days],
            names=["date", "ticker"]
        ))

        stock_data = {"MSFT": msft_data, "AAPL": aapl_data}
        bull_index = _make_index_data(n_days=250, bull=True)
        provider = _make_mock_provider(
            universe=["AAPL", "MSFT"],
            stock_data=stock_data,
            index_data=bull_index,
        )

        # Both are existing positions
        existing = {
            "AAPL": Position(
                ticker="AAPL", shares=100, entry_price=100.0,
                entry_date=date(2024, 6, 1), atr_at_entry=2.0,
            ),
            "MSFT": Position(
                ticker="MSFT", shares=100, entry_price=100.0,
                entry_date=date(2024, 6, 1), atr_at_entry=2.0,
            ),
        }

        result = compute_target_portfolio(
            as_of=as_of,
            current_positions=existing,
            current_cash=500_000.0,
            config=config,
            data_provider=provider,
        )

        # AAPL breaks SMA → forced sell → NOT in target
        assert "AAPL" not in result.positions
        # MSFT stays in the target (above SMA, top 20%)
        assert "MSFT" in result.positions


class TestBacktestResult:
    """BacktestResult dataclass has expected fields."""

    def test_backtest_result_fields(self):
        result = BacktestResult(
            equity_curve=pd.DataFrame(columns=["date", "portfolio_value", "cash"]),
            trades=[],
            final_positions={},
            final_cash=1_000_000.0,
            config=Config(),
        )
        assert result.equity_curve is not None
        assert result.trades == []
        assert result.final_positions == {}
        assert result.final_cash == 1_000_000.0
        assert isinstance(result.config, Config)


class TestRunBacktestSmoke:
    """Smoke test for run_backtest — basic integration test."""

    def test_run_backtest_basic(self):
        """run_backtest completes and returns a BacktestResult."""
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
            rebalance_freq="weekly",
        )

        # Build comprehensive stock data for a 3-month period
        n_days = 300
        start_date = date(2024, 1, 2)
        dates = pd.bdate_range(start_date, periods=n_days)

        stock_data = {}
        for ticker in ["AAPL", "MSFT"]:
            closes = [100 + i * 0.1 for i in range(n_days)]
            stock_data[ticker] = pd.DataFrame({
                "raw_open": [c + 0.5 for c in closes],
                "raw_high": [c + 1.0 for c in closes],
                "raw_low": [c - 1.0 for c in closes],
                "raw_close": closes,
                "volume": 1_000_000.0,
                "adj_close": closes,
                "dividend": 0.0,
                "split_ratio": 1.0,
            }, index=pd.MultiIndex.from_arrays(
                [dates.date, [ticker] * n_days],
                names=["date", "ticker"]
            ))

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT"],
            stock_data=stock_data,
        )

        result = run_backtest(
            start=date(2024, 4, 1),
            end=date(2024, 6, 30),
            initial_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity_curve, pd.DataFrame)
        assert isinstance(result.trades, list)
        assert isinstance(result.final_positions, dict)
        assert isinstance(result.final_cash, float)


class TestProfileWiring:
    """Verify that MarketProfile is wired through compute_target_portfolio."""

    def test_compute_target_portfolio_uses_profile_exit_sma(self):
        """exit_sma_window from profile drives _check_sma_break threshold.

        With a short exit_sma_window=20, a stock that has been declining
        for 30 days should be forced-sold. With the default 100-day window
        the same stock would NOT be forced-sold (not enough data below SMA).
        """
        from clenow.markets.profiles import MarketProfile
        from clenow.markets.costs import USCostModel

        as_of = date(2024, 12, 31)

        # Custom profile with exit_sma_window=20 (short)
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
            score_window=90,
            regime_sma_window=200,
            exit_sma_window=20,  # SHORT window — easier to trigger SMA break
        )

        config = Config(
            score_window=90,
            atr_period=20,
            regime_sma=200,
            stock_sma=100,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
            risk_factor=0.001,
        )

        # Build AAPL data: rise for 200 days then decline sharply for 100
        # With 20-day SMA window, the recent decline will break the SMA.
        # With 100-day SMA window, the decline just barely starts to break.
        n_days = 300
        dates = pd.bdate_range(date(2024, 1, 2), periods=n_days)
        closes = []
        for i in range(n_days):
            if i < 200:
                closes.append(100 + i * 0.2)  # Rising to ~140
            else:
                closes.append(140 - (i - 200) * 1.0)  # Sharp decline

        aapl_data = pd.DataFrame({
            "raw_open": [c + 0.5 for c in closes],
            "raw_high": [c + 1.0 for c in closes],
            "raw_low": [c - 1.0 for c in closes],
            "raw_close": closes,
            "volume": 1_000_000.0,
            "adj_close": closes,
            "dividend": 0.0,
            "split_ratio": 1.0,
        }, index=pd.MultiIndex.from_arrays(
            [dates.date, ["AAPL"] * n_days],
            names=["date", "ticker"]
        ))

        # MSFT: strong uptrend (stays in target)
        msft_closes = [100 + i * 0.2 for i in range(n_days)]
        msft_data = pd.DataFrame({
            "raw_open": [c + 0.5 for c in msft_closes],
            "raw_high": [c + 1.0 for c in msft_closes],
            "raw_low": [c - 1.0 for c in msft_closes],
            "raw_close": msft_closes,
            "volume": 1_000_000.0,
            "adj_close": msft_closes,
            "dividend": 0.0,
            "split_ratio": 1.0,
        }, index=pd.MultiIndex.from_arrays(
            [dates.date, ["MSFT"] * n_days],
            names=["date", "ticker"]
        ))

        stock_data = {"AAPL": aapl_data, "MSFT": msft_data}
        bull_index = _make_index_data(n_days=250, bull=True)
        provider = _make_mock_provider(
            universe=["AAPL", "MSFT"],
            stock_data=stock_data,
            index_data=bull_index,
        )

        # AAPL is an existing position
        existing = {
            "AAPL": Position(
                ticker="AAPL", shares=100, entry_price=100.0,
                entry_date=date(2024, 6, 1), atr_at_entry=2.0,
            ),
            "MSFT": Position(
                ticker="MSFT", shares=100, entry_price=100.0,
                entry_date=date(2024, 6, 1), atr_at_entry=2.0,
            ),
        }

        # With exit_sma_window=20, AAPL's sharp decline breaks the 20-day SMA
        result = compute_target_portfolio(
            as_of=as_of,
            current_positions=existing,
            current_cash=500_000.0,
            config=config,
            data_provider=provider,
            profile=profile,
        )

        # AAPL should be forced-sold (breaks 20-day SMA with recent decline)
        assert "AAPL" not in result.positions
        # MSFT stays (strong uptrend)
        assert "MSFT" in result.positions

    def test_compute_target_portfolio_defaults_to_us_profile(self):
        """Without explicit profile, US profile is used (backward-compat)."""
        as_of = date(2024, 12, 31)
        config = Config(
            score_window=90,
            atr_period=20,
            regime_sma=200,
            stock_sma=100,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.20,
            risk_factor=0.001,
        )

        stock_data = {}
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            stock_data[ticker] = _make_stock_data(
                n_days=300, ticker=ticker, price=100.0, trend=0.1, volume=1_000_000.0
            )

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT", "GOOG"],
            stock_data=stock_data,
        )

        # No profile passed — should default to US
        result = compute_target_portfolio(
            as_of=as_of,
            current_positions={},
            current_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        assert isinstance(result, TargetPortfolio)
        assert len(result.positions) >= 1

    def test_run_backtest_accepts_profile(self):
        """run_backtest accepts profile parameter and passes it through."""
        from clenow.markets.profiles import MarketProfile
        from clenow.markets.costs import USCostModel

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
            score_window=90,
            regime_sma_window=200,
            exit_sma_window=100,
        )

        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
            rebalance_freq="weekly",
        )

        n_days = 300
        start_date = date(2024, 1, 2)
        dates = pd.bdate_range(start_date, periods=n_days)

        stock_data = {}
        for ticker in ["AAPL", "MSFT"]:
            closes = [100 + i * 0.1 for i in range(n_days)]
            stock_data[ticker] = pd.DataFrame({
                "raw_open": [c + 0.5 for c in closes],
                "raw_high": [c + 1.0 for c in closes],
                "raw_low": [c - 1.0 for c in closes],
                "raw_close": closes,
                "volume": 1_000_000.0,
                "adj_close": closes,
                "dividend": 0.0,
                "split_ratio": 1.0,
            }, index=pd.MultiIndex.from_arrays(
                [dates.date, [ticker] * n_days],
                names=["date", "ticker"]
            ))

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT"],
            stock_data=stock_data,
        )

        result = run_backtest(
            start=date(2024, 4, 1),
            end=date(2024, 6, 30),
            initial_cash=1_000_000.0,
            config=config,
            data_provider=provider,
            profile=profile,
        )

        assert isinstance(result, BacktestResult)


# ── _slice_prices tests ──────────────────────────────────────────────────────


def _make_prices(tickers: list[str], dates: list[date]) -> pd.DataFrame:
    """Build a (date, ticker) MultiIndex price DataFrame for testing."""
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": pd.Timestamp(d),
                "ticker": t,
                "raw_close": 100.0,
                "raw_open": 99.0,
                "raw_high": 101.0,
                "raw_low": 98.0,
                "adj_close": 100.0,
                "volume": 1_000_000.0,
            })
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df


def test_slice_prices_filters_date_range():
    """_slice_prices should filter to the specified date range."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    tickers = ["AAPL", "MSFT"]
    preloaded = _make_prices(tickers, dates)

    result = _slice_prices(preloaded, ["AAPL"], date(2024, 1, 1), date(2024, 1, 2))

    date_vals = result.index.get_level_values("date")
    ticker_vals = result.index.get_level_values("ticker")
    assert all(d <= pd.Timestamp(date(2024, 1, 2)) for d in date_vals)
    assert set(ticker_vals) == {"AAPL"}


def test_slice_prices_returns_empty_for_missing_ticker():
    """_slice_prices should return an empty DataFrame when ticker is not in preloaded."""
    dates = [date(2024, 1, 1)]
    preloaded = _make_prices(["AAPL"], dates)

    result = _slice_prices(preloaded, ["TSLA"], date(2024, 1, 1), date(2024, 1, 1))
    assert result.empty


def test_slice_prices_handles_none():
    """_slice_prices should return an empty DataFrame when preloaded is None."""
    result = _slice_prices(None, ["AAPL"], date(2024, 1, 1), date(2024, 1, 1))
    assert result.empty
