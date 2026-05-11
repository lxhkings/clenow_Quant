import pandas as pd
import pytest
from datetime import date
from clenow.backtest.executor import SimulatedExecutor, OrderRejection
from clenow.types import Order, Side, OrderType


def _prices_locked_up(ticker="600519.SH"):
    """Day before close=100; today open=high=low=110 (10% limit up)."""
    dates = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {
            "raw_open": [100, 100, 110.0],
            "raw_high": [101, 101, 110.0],
            "raw_low": [99, 99, 110.0],
            "raw_close": [100, 100, 110.0],
            "volume": [1_000, 1_000, 100],
            "adj_close": [100, 100, 110.0],
        },
        index=pd.MultiIndex.from_product([dates, [ticker]], names=["date", "ticker"]),
    )
    return df


def test_executor_rejects_limit_up_buy():
    from clenow.markets import get_profile
    prices = _prices_locked_up("600519.SH")
    profile = get_profile("CN")
    executor = SimulatedExecutor(prices=prices, profile=profile)
    order = Order(
        ticker="600519.SH",
        side=Side.BUY,
        shares=100,
        target_date=date(2024, 1, 3),
        order_type=OrderType.MARKET_ON_OPEN,
    )
    fills, rejections = executor.execute([order])
    assert len(fills) == 0
    assert len(rejections) == 1
    assert "price_limit_locked" in rejections[0].reason


def test_executor_fills_normally_when_not_locked():
    """Open=105 (within 10% of prev close=100), high=106, low=104: not locked, fills normally."""
    from clenow.markets import get_profile
    ticker = "600519.SH"
    dates = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {
            "raw_open": [100, 100, 105.0],
            "raw_high": [101, 101, 106.0],
            "raw_low": [99, 99, 104.0],
            "raw_close": [100, 100, 105.5],
            "volume": [1_000, 1_000, 5_000],
            "adj_close": [100, 100, 105.0],
        },
        index=pd.MultiIndex.from_product([dates, [ticker]], names=["date", "ticker"]),
    )
    profile = get_profile("CN")
    executor = SimulatedExecutor(prices=df, profile=profile)
    order = Order(
        ticker=ticker, side=Side.BUY, shares=100,
        target_date=date(2024, 1, 3),
        order_type=OrderType.MARKET_ON_OPEN,
    )
    fills, _ = executor.execute([order])
    assert len(fills) == 1
    assert fills[0].fill_price == 105.0


def test_executor_us_profile_no_price_limit_check():
    """US profile has no price_limit_resolver — limit check disabled, fills normally."""
    from clenow.markets import get_profile
    ticker = "AAPL"
    dates = pd.date_range("2024-01-01", periods=2)
    df = pd.DataFrame(
        {
            "raw_open": [100, 110.0],  # 10% gap — would be CN-locked
            "raw_high": [101, 110.0],
            "raw_low": [99, 110.0],
            "raw_close": [100, 110.0],
            "volume": [1_000, 100],
            "adj_close": [100, 110.0],
        },
        index=pd.MultiIndex.from_product([dates, [ticker]], names=["date", "ticker"]),
    )
    profile = get_profile("US")
    executor = SimulatedExecutor(prices=df, profile=profile)
    order = Order(
        ticker=ticker, side=Side.BUY, shares=10,
        target_date=date(2024, 1, 2),
        order_type=OrderType.MARKET_ON_OPEN,
    )
    fills, _ = executor.execute([order])
    assert len(fills) == 1  # fills despite gap
