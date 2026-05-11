import pytest
from clenow.config import Config
from clenow.markets import get_profile


def test_config_default_market_is_us():
    c = Config()
    assert c.market == "US"
    assert c.profile is get_profile("US")


def test_config_cn_market():
    c = Config(market="CN")
    assert c.profile.universe_index_id == "CSI800"
    assert c.profile.lot_size == 100


def test_config_market_case_insensitive():
    c = Config(market="cn")
    assert c.profile.name == "CN"


def test_config_unknown_market_raises_at_profile_access():
    c = Config(market="JP")
    with pytest.raises(KeyError):
        _ = c.profile
