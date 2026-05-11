import pandas as pd
from clenow.portfolio.selector import _exclude_st


def test_exclude_st_removes_st_stocks():
    candidates = ["600519.SH", "000001.SZ", "300750.SZ"]
    stocks_meta = pd.DataFrame(
        {"ticker": candidates, "name": ["ST茅台", "平安银行", "宁德时代"]}
    ).set_index("ticker")
    out = _exclude_st(candidates, stocks_meta)
    assert "600519.SH" not in out
    assert "000001.SZ" in out
    assert "300750.SZ" in out


def test_exclude_st_handles_starST():
    candidates = ["600001.SH", "000002.SZ"]
    stocks_meta = pd.DataFrame(
        {"ticker": candidates, "name": ["*ST 钢管", "万科A"]}
    ).set_index("ticker")
    out = _exclude_st(candidates, stocks_meta)
    assert "600001.SH" not in out
    assert "000002.SZ" in out


def test_exclude_st_handles_pt_and_退():
    candidates = ["600001.SH", "000002.SZ", "300003.SZ"]
    stocks_meta = pd.DataFrame(
        {"ticker": candidates, "name": ["PT 钢管", "退市某股", "正常股"]}
    ).set_index("ticker")
    out = _exclude_st(candidates, stocks_meta)
    assert out == ["300003.SZ"]


def test_exclude_st_missing_meta_keeps_stock():
    """If a ticker has no metadata row, do not exclude (conservative)."""
    candidates = ["600519.SH"]
    stocks_meta = pd.DataFrame({"ticker": [], "name": []}).set_index("ticker")
    out = _exclude_st(candidates, stocks_meta)
    assert out == ["600519.SH"]
