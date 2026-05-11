"""Configuration parameters for the Clenow system."""

from dataclasses import dataclass
from enum import Enum


class CashInterestPolicy(Enum):
    """Policy for interest earned on uninvested cash.

    ZERO: No interest on cash (conservative default).
    T_BILL: Use FRED DGS3MO rate (future implementation).
    """

    ZERO = "zero"
    T_BILL = "t_bill"


@dataclass(frozen=True)
class Config:
    # Score / Signal parameters
    score_window: int = 90
    atr_period: int = 20
    gap_threshold: float = 0.15

    # Regime / Entry filters
    regime_sma: int = 200
    stock_sma: int = 100  # 100-day SMA for exit filter
    min_price: float = 5.0  # Superseded by MarketProfile.min_price
    min_adv_dollars: float = 10_000_000  # Superseded by MarketProfile.min_adv_amount

    # Ranking
    top_pct: float = 0.20

    # Sizing / Risk
    risk_factor: float = 0.001
    max_position_pct: float = 0.05

    # Cost model
    half_spread_bps: float = 5.0
    slippage_bps_per_pct_adv: float = 5.0
    slippage_bps_min: float = 1.0
    slippage_bps_max: float = 30.0
    commission_per_share: float = 0.0

    # Rebalancing
    rebalance_freq: str = "weekly"  # "weekly" | "biweekly"
    rebalance_dow: int = 0  # 0 = Monday open

    # Cash interest
    cash_interest_policy: CashInterestPolicy = CashInterestPolicy.ZERO

    # Market selection
    market: str = "US"

    @property
    def profile(self):
        """Lazy market profile lookup. Avoids import cycles by deferring import."""
        from clenow.markets import get_profile
        return get_profile(self.market)
