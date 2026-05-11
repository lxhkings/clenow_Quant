from decimal import Decimal

from clenow.markets.costs import HKCostModel

C = Decimal("0.01")  # cent precision for quantize


def test_hk_buy_includes_stamp_duty():
    model = HKCostModel()  # defaults: 3.0/13.0/0.27/0.5/5.0
    commission, slippage = model.compute_side("buy", Decimal("100000"))
    # all fees bilateral: (3 + 13 + 0.27 + 0.5) bps = 16.77 bps -> 167.70 HKD
    assert commission.quantize(C) == Decimal("167.70")
    assert slippage.quantize(C) == Decimal("50.00")


def test_hk_sell_same_as_buy():
    """HK stamp duty applies both sides; buy/sell totals identical."""
    model = HKCostModel()
    buy_c, buy_s = model.compute_side("buy", Decimal("100000"))
    sell_c, sell_s = model.compute_side("sell", Decimal("100000"))
    assert buy_c == sell_c
    assert buy_s == sell_s
