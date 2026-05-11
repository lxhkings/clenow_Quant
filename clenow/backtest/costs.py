"""Legacy compute_cost — thin shim over USCostModel for backward compat.

New code should use Profile.cost_model.compute_side(...) directly.
"""

from __future__ import annotations

from decimal import Decimal

from clenow.config import Config
from clenow.markets.costs import USCostModel


def compute_cost(
    order_notional: float,
    adv_20_dollars: float,
    shares: int,
    config: Config,
) -> tuple[float, float]:
    """Per-side cost. Returns (commission_dollars, slippage_bps).

    Kept for backward compatibility with existing test suite. New code path
    uses Config.profile.cost_model.compute_side(...).
    """
    model = USCostModel(
        commission_per_share=Decimal(str(config.commission_per_share)),
        half_spread_bps=config.half_spread_bps,
        slippage_bps_per_pct_adv=config.slippage_bps_per_pct_adv,
        slippage_bps_min=config.slippage_bps_min,
        slippage_bps_max=config.slippage_bps_max,
    )
    commission, slippage_bps = model.compute_legacy(
        order_notional=order_notional,
        adv_20_dollars=adv_20_dollars,
        shares=shares,
    )
    return (float(commission), slippage_bps)
