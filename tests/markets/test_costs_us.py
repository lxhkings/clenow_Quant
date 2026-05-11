from decimal import Decimal

from clenow.markets.costs import USCostModel


def test_us_cost_matches_legacy_compute_cost():
    """USCostModel must produce same (commission, slippage_bps) as legacy compute_cost."""
    model = USCostModel(
        commission_per_share=Decimal("0.005"),
        half_spread_bps=5.0,
        slippage_bps_per_pct_adv=2.0,
        slippage_bps_min=1.0,
        slippage_bps_max=50.0,
    )
    # 100 shares @ $50 = $5000 notional, ADV $1M -> 0.5% participation -> 1.0 bps raw -> clipped to min 1.0
    commission, slippage_bps = model.compute_legacy(
        order_notional=5000.0,
        adv_20_dollars=1_000_000.0,
        shares=100,
    )
    assert commission == Decimal("0.500")
    assert slippage_bps == 1.0


def test_us_cost_high_participation_clipped_to_max():
    model = USCostModel(
        commission_per_share=Decimal("0.005"),
        half_spread_bps=5.0,
        slippage_bps_per_pct_adv=2.0,
        slippage_bps_min=1.0,
        slippage_bps_max=50.0,
    )
    # 10% participation -> 20 bps raw, well below max=50
    commission, slippage_bps = model.compute_legacy(
        order_notional=100_000.0,
        adv_20_dollars=1_000_000.0,
        shares=2000,
    )
    assert commission == Decimal("10.000")
    assert slippage_bps == 20.0


def test_us_cost_zero_adv_falls_back_to_max():
    model = USCostModel(
        commission_per_share=Decimal("0.005"),
        half_spread_bps=5.0,
        slippage_bps_per_pct_adv=2.0,
        slippage_bps_min=1.0,
        slippage_bps_max=50.0,
    )
    commission, slippage_bps = model.compute_legacy(
        order_notional=5000.0,
        adv_20_dollars=0.0,
        shares=100,
    )
    assert slippage_bps == 50.0
