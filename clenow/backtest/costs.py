"""Transaction cost model — commission and slippage.

Per-side cost model:
  half_spread_bps = config.half_spread_bps (default 5)
  slippage_bps = clip(
      slippage_bps_per_pct_adv * (order_notional / adv_20_dollars) / 0.01,
      slippage_bps_min,
      slippage_bps_max,
  )
  commission = commission_per_share * shares
"""

from __future__ import annotations

from clenow.config import Config


def compute_cost(
    order_notional: float,
    adv_20_dollars: float,
    shares: int,
    config: Config,
) -> tuple[float, float]:
    """Compute per-side transaction costs.

    Args:
        order_notional: Dollar value of the order (price * shares).
        adv_20_dollars: 20-day average dollar volume for the ticker.
        shares: Number of shares in the order.
        config: System configuration with cost parameters.

    Returns:
        Tuple of (commission_dollars, slippage_bps).
    """
    # Market-impact slippage: proportional to participation rate
    if adv_20_dollars > 0:
        participation_rate = order_notional / adv_20_dollars
        raw_slippage = config.slippage_bps_per_pct_adv * (participation_rate / 0.01)
    else:
        # No ADV data: use max slippage as a conservative fallback
        raw_slippage = config.slippage_bps_max

    slippage_bps = max(
        config.slippage_bps_min,
        min(config.slippage_bps_max, raw_slippage),
    )

    commission = config.commission_per_share * shares

    return (commission, slippage_bps)
