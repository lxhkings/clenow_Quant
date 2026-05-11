from decimal import Decimal

from clenow.markets.costs import CNCostModel

C = Decimal("0.01")  # cent precision for quantize


def test_cn_buy_no_stamp_duty():
    model = CNCostModel()  # defaults: 2.5/5.0/0.2/5.0
    commission, slippage = model.compute_side("buy", Decimal("100000"))
    # buy: (2.5 + 0.2) bps = 2.7 bps -> 27 RMB
    assert commission.quantize(C) == Decimal("27.00")
    # 5 bps slippage -> 50 RMB
    assert slippage.quantize(C) == Decimal("50.00")


def test_cn_sell_includes_stamp_duty():
    model = CNCostModel()
    commission, slippage = model.compute_side("sell", Decimal("100000"))
    # sell: (2.5 + 0.2 + 5.0) bps = 7.7 bps -> 77 RMB
    assert commission.quantize(C) == Decimal("77.00")
    assert slippage.quantize(C) == Decimal("50.00")


def test_cn_custom_rates():
    model = CNCostModel(
        commission_bps=3.0,
        stamp_duty_bps_sell=10.0,
        transfer_fee_bps=0.0,
        slippage_bps=0.0,
    )
    commission, slippage = model.compute_side("sell", Decimal("1000000"))
    # (3 + 10) bps = 13 bps -> 1300 RMB
    assert commission.quantize(C) == Decimal("1300.00")
    assert slippage.quantize(C) == Decimal("0.00")
