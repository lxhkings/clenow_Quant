import pytest
import pandas_market_calendars as mcal
from clenow.markets import get_profile, PROFILES
from clenow.markets.costs import USCostModel, CNCostModel, HKCostModel


def test_three_profiles_registered():
    assert set(PROFILES.keys()) == {"US", "CN", "HK"}


def test_get_profile_case_insensitive():
    assert get_profile("us") is get_profile("US")
    assert get_profile("Cn") is get_profile("CN")


def test_get_profile_unknown_raises():
    with pytest.raises(KeyError):
        get_profile("JP")


def test_us_profile_fields():
    p = get_profile("US")
    assert p.name == "US"
    assert p.currency == "USD"
    assert p.universe_index_id == "SP500"
    assert p.regime_index_id == "SP500"
    assert p.annualization_days == 252
    assert p.lot_size == 1
    assert p.min_price == 5.0
    assert p.min_adv_amount == 10_000_000
    assert p.trading_calendar_name == "NYSE"
    assert isinstance(p.cost_model, USCostModel)
    assert p.default_starting_capital == 100_000
    assert p.score_window == 90
    assert p.regime_sma_window == 200
    assert p.exit_sma_window == 100
    assert p.price_limit_pct is None  # US has no daily limit
    assert p.suspension_threshold_days == 0
    assert p.delisting_threshold_days == 10
    assert p.universe_exclusion_filters == ()


def test_cn_profile_fields():
    p = get_profile("CN")
    assert p.name == "CN"
    assert p.currency == "CNY"
    assert p.universe_index_id == "CSI800"
    assert p.regime_index_id == "CSI800"
    assert p.annualization_days == 244
    assert p.lot_size == 100
    assert p.min_price == 5.0
    assert p.min_adv_amount == 50_000_000
    assert p.trading_calendar_name == "XSHG"
    assert isinstance(p.cost_model, CNCostModel)
    assert p.default_starting_capital == 1_000_000
    assert p.price_limit_pct is None  # uses resolver
    assert callable(p.price_limit_resolver)
    assert p.suspension_threshold_days == 20
    assert p.delisting_threshold_days == 60
    assert "st" in p.universe_exclusion_filters


def test_hk_profile_fields():
    p = get_profile("HK")
    assert p.name == "HK"
    assert p.currency == "HKD"
    assert p.universe_index_id == "HSI"
    assert p.regime_index_id == "HSI"
    assert p.annualization_days == 247
    assert p.lot_size == 100
    assert p.min_price == 1.0
    assert p.min_adv_amount == 50_000_000
    assert p.trading_calendar_name == "XHKG"
    assert isinstance(p.cost_model, HKCostModel)
    assert p.default_starting_capital == 1_000_000
    assert p.price_limit_pct is None  # HK no daily limit
    assert p.suspension_threshold_days == 10
    assert p.delisting_threshold_days == 30


@pytest.mark.parametrize("market", ["US", "CN", "HK"])
def test_calendar_name_loads(market):
    """Calendar name must be loadable by pandas_market_calendars."""
    p = get_profile(market)
    cal = mcal.get_calendar(p.trading_calendar_name)
    assert cal is not None


def test_profile_is_frozen():
    """MarketProfile must be immutable."""
    p = get_profile("US")
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        p.lot_size = 999  # type: ignore
