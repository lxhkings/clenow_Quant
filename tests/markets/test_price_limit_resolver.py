from clenow.markets.price_limit import cn_price_limit_resolver


def test_sh_main_board_10pct():
    assert cn_price_limit_resolver("600519.SH", "贵州茅台") == 0.10
    assert cn_price_limit_resolver("601318.SH", "中国平安") == 0.10


def test_sh_star_market_20pct():
    assert cn_price_limit_resolver("688981.SH", "中芯国际") == 0.20


def test_sz_main_board_10pct():
    assert cn_price_limit_resolver("000001.SZ", "平安银行") == 0.10
    assert cn_price_limit_resolver("002594.SZ", "比亚迪") == 0.10


def test_sz_chinext_20pct():
    assert cn_price_limit_resolver("300750.SZ", "宁德时代") == 0.20


def test_st_5pct_overrides_board():
    """ST/*ST stocks limit 5% regardless of board."""
    assert cn_price_limit_resolver("600519.SH", "ST茅台") == 0.05
    assert cn_price_limit_resolver("000001.SZ", "*ST 平安") == 0.05
    assert cn_price_limit_resolver("300750.SZ", "ST宁德") == 0.05


def test_pt_stocks_5pct():
    """PT (particular transfer) treated as ST."""
    assert cn_price_limit_resolver("600001.SH", "PT 钢管") == 0.05


def test_bj_stocks_30pct():
    """BSE (北证) listings: 30% limit."""
    assert cn_price_limit_resolver("830799.BJ", "艾融软件") == 0.30


def test_unknown_format_returns_default_10pct():
    """Unrecognized ticker pattern: conservative 10% default."""
    assert cn_price_limit_resolver("ABCDEF.XX", "Unknown") == 0.10
