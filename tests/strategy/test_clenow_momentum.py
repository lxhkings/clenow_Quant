from datetime import date

import numpy as np
import pandas as pd
import pytest

from clenow.config import Config
from clenow.markets import get_profile
from clenow.strategy.clenow_momentum import ClenowMomentum
from clenow.types import Position


def _synth_ticker_data(n=100, drift=0.001, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    closes = 100 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range("2024-01-01", periods=n).date
    return pd.DataFrame({
        "raw_open": closes,
        "raw_high": closes * 1.01,
        "raw_low": closes * 0.99,
        "raw_close": closes,
        "adj_close": closes,
        "volume": np.full(n, 1_000_000),
    }, index=pd.Index(dates, name="date"))


def test_clenow_momentum_score_positive_trend():
    strat = ClenowMomentum()
    td = _synth_ticker_data(n=120, drift=0.002, seed=1)
    profile = get_profile("US")
    score = strat.score("AAPL", td, profile, Config())
    assert score > 0


def test_clenow_momentum_score_short_history_zero():
    strat = ClenowMomentum()
    td = _synth_ticker_data(n=10, drift=0.001, seed=1)  # < 90 bars
    profile = get_profile("US")
    score = strat.score("AAPL", td, profile, Config())
    assert score == 0.0


def test_clenow_momentum_rank_top_pct():
    strat = ClenowMomentum()
    scores = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.05, "E": -0.1}
    ranked = strat.rank(scores, Config(top_pct=0.40))  # 0.40 * 5 = 2
    assert ranked == ["A", "B"]


def test_clenow_momentum_exit_signal_below_sma_true():
    strat = ClenowMomentum()
    # Closes trending down: last < 100SMA
    closes = pd.Series([100 - i * 0.5 for i in range(150)])
    dates = pd.bdate_range("2024-01-01", periods=150).date
    df = pd.DataFrame({"raw_close": closes.values}, index=pd.Index(dates, name="date"))
    all_prices = pd.concat({"X": df}, names=["ticker"]).swaplevel().sort_index()
    profile = get_profile("US")
    result = strat.exit_signal("X", all_prices, date(2024, 7, 1), None, profile, Config())
    assert result is True


def test_clenow_momentum_size_basic_atr():
    strat = ClenowMomentum()
    filtered = ["X"]
    atrs = {"X": 2.0}
    prices = {"X": 100.0}
    profile = get_profile("US")
    config = Config(risk_factor=0.001, max_position_pct=0.05)
    target = strat.size(
        filtered_tickers=filtered, data_provider=None, as_of=date(2024, 7, 1),
        config=config, current_cash=100_000, current_positions={}, current_prices=prices,
        atrs=atrs, profile=profile,
    )
    # equity=100k, risk*equity/atr = 0.001*100000/2 = 50 shares; lot_size=1 (US)
    assert target == {"X": 50}