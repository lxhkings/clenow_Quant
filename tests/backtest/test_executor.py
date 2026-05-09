"""Tests for the simulated executor — fills, costs, rejections, time isolation."""

from datetime import date, datetime

import pandas as pd
import pytest

from clenow.backtest.executor import BrokerExecutor, SimulatedExecutor
from clenow.config import Config
from clenow.errors import OrderRejection
from clenow.types import Order, OrderType, Side


def _make_price_data(
    dates: list[date],
    tickers: list[str],
    opens: dict[str, list[float]] | None = None,
    closes: dict[str, list[float]] | None = None,
    highs: dict[str, list[float]] | None = None,
    lows: dict[str, list[float]] | None = None,
    volumes: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Build a MultiIndex (date, ticker) price DataFrame."""
    rows = []
    for i, d in enumerate(dates):
        for t in tickers:
            row = {
                "date": d,
                "ticker": t,
                "raw_open": opens[t][i] if opens else 100.0,
                "raw_close": closes[t][i] if closes else 100.0,
                "raw_high": highs[t][i] if highs else 101.0,
                "raw_low": lows[t][i] if lows else 99.0,
                "volume": volumes[t][i] if volumes else 1_000_000.0,
                "adj_close": closes[t][i] if closes else 100.0,
                "dividend": 0.0,
                "split_ratio": 1.0,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"])
    return df


class TestSimulatedExecutorFillAtOpen:
    """SimulatedExecutor fills at raw_open price."""

    def test_buy_filled_at_open(self):
        """A BUY order fills at raw_open on the execution date."""
        exec_date = date(2025, 1, 13)  # Monday
        price_data = _make_price_data(
            dates=[exec_date],
            tickers=["AAPL"],
            opens={"AAPL": [150.0]},
        )
        executor = SimulatedExecutor(price_data=price_data)

        order = Order(
            ticker="AAPL",
            shares=100,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            target_date=exec_date,
        )
        config = Config()
        fills = executor.submit([order], config)

        assert len(fills) == 1
        assert fills[0].fill_price == 150.0
        assert fills[0].shares == 100
        assert fills[0].side == Side.BUY

    def test_sell_filled_at_open(self):
        """A SELL order fills at raw_open on the execution date."""
        exec_date = date(2025, 1, 13)
        price_data = _make_price_data(
            dates=[exec_date],
            tickers=["MSFT"],
            opens={"MSFT": [200.0]},
        )
        executor = SimulatedExecutor(price_data=price_data)

        order = Order(
            ticker="MSFT",
            shares=50,
            side=Side.SELL,
            order_type=OrderType.MARKET,
            target_date=exec_date,
        )
        config = Config()
        fills = executor.submit([order], config)

        assert len(fills) == 1
        assert fills[0].fill_price == 200.0
        assert fills[0].side == Side.SELL


class TestCostModelApplied:
    """Cost model is applied correctly to fills."""

    def test_commission_and_slippage(self):
        """Commission and slippage are computed and attached to fills."""
        exec_date = date(2025, 1, 13)
        price_data = _make_price_data(
            dates=[exec_date],
            tickers=["AAPL"],
            opens={"AAPL": [100.0]},
        )
        adv_data = {"AAPL": 10_000_000.0}
        executor = SimulatedExecutor(price_data=price_data, adv_data=adv_data)

        order = Order(
            ticker="AAPL",
            shares=100,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            target_date=exec_date,
        )
        config = Config(
            commission_per_share=0.01,
            half_spread_bps=5.0,
            slippage_bps_per_pct_adv=5.0,
            slippage_bps_min=1.0,
            slippage_bps_max=30.0,
        )
        fills = executor.submit([order], config)

        assert fills[0].commission == pytest.approx(1.0)  # 0.01 * 100
        # order_notional = 100 * 100 = 10,000 → participation = 10,000 / 10,000,000 = 0.001
        # slippage = 5 * (0.001 / 0.01) = 0.5 → clipped to min 1.0
        assert fills[0].slippage_bps == 1.0


class TestOrderRejectionOnMissingPrice:
    """OrderRejection raised when no price on execution date."""

    def test_no_price_raises_rejection(self):
        """Missing price data → OrderRejection with reason='no_market'."""
        # Empty price data
        price_data = _make_price_data(
            dates=[date(2025, 1, 10)],
            tickers=["AAPL"],
            opens={"AAPL": [100.0]},
        )
        executor = SimulatedExecutor(price_data=price_data)

        # Try to fill on a date with no data
        order = Order(
            ticker="AAPL",
            shares=100,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            target_date=date(2025, 1, 13),  # No data for this date
        )
        config = Config()

        with pytest.raises(OrderRejection) as exc_info:
            executor.submit([order], config)

        assert exc_info.value.ticker == "AAPL"
        assert exc_info.value.reason == "no_market"

    def test_ticker_not_in_data_raises_rejection(self):
        """Ticker not present in price data → OrderRejection."""
        exec_date = date(2025, 1, 13)
        price_data = _make_price_data(
            dates=[exec_date],
            tickers=["AAPL"],
            opens={"AAPL": [100.0]},
        )
        executor = SimulatedExecutor(price_data=price_data)

        order = Order(
            ticker="MSFT",  # Not in price_data
            shares=100,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            target_date=exec_date,
        )
        config = Config()

        with pytest.raises(OrderRejection) as exc_info:
            executor.submit([order], config)

        assert exc_info.value.ticker == "MSFT"
        assert exc_info.value.reason == "no_market"


class TestTimeIsolation:
    """Signal date and execution date are always different bars."""

    def test_different_dates(self):
        """Signal_date (Friday) and execution_date (Monday) are never the same."""
        signal_date = date(2025, 1, 10)  # Friday
        exec_date = date(2025, 1, 13)  # Monday

        # This is a structural test: the rebalance module guarantees
        # signal_date < execution_date. The executor never reads data
        # from the signal_date bar.
        assert signal_date < exec_date
        assert signal_date.weekday() == 4  # Friday
        assert exec_date.weekday() == 0  # Monday

    def test_executor_only_reads_execution_date(self):
        """Executor fills at the execution_date price, not the signal_date price."""
        signal_date = date(2025, 1, 10)  # Friday
        exec_date = date(2025, 1, 13)  # Monday

        price_data = _make_price_data(
            dates=[signal_date, exec_date],
            tickers=["AAPL"],
            opens={"AAPL": [145.0, 150.0]},  # Different prices
        )
        executor = SimulatedExecutor(price_data=price_data)

        order = Order(
            ticker="AAPL",
            shares=100,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            target_date=exec_date,  # Execution on Monday
        )
        config = Config()
        fills = executor.submit([order], config)

        # Fill at Monday's open (150.0), not Friday's (145.0)
        assert fills[0].fill_price == 150.0


class TestBrokerExecutorStub:
    """BrokerExecutor raises NotImplementedError."""

    def test_submit_raises_not_implemented(self):
        executor = BrokerExecutor()
        config = Config()
        with pytest.raises(NotImplementedError):
            executor.submit([], config)
