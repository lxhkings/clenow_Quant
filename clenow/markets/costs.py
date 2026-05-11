"""Per-market transaction cost models.

CostModel Protocol unifies the per-side cost interface. Each market has its
own dataclass impl with market-appropriate fee structure:
  US: per-share commission + half-spread + ADV-scaled slippage.
  CN: bps commission + sell-side stamp duty + transfer fee + slippage.
  HK: bps commission + bilateral stamp + SFC levy + trading fee + slippage.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol


class CostModel(Protocol):
    """Stateless cost model. Returns per-side commission Decimal + slippage_bps float."""

    def compute_legacy(
        self,
        order_notional: float,
        adv_20_dollars: float,
        shares: int,
    ) -> tuple[Decimal, float]:
        """Returns (commission_in_native_currency, slippage_bps)."""
        ...


@dataclass(frozen=True)
class USCostModel:
    commission_per_share: Decimal = Decimal("0.005")
    half_spread_bps: float = 5.0
    slippage_bps_per_pct_adv: float = 2.0
    slippage_bps_min: float = 1.0
    slippage_bps_max: float = 50.0

    def compute_legacy(
        self,
        order_notional: float,
        adv_20_dollars: float,
        shares: int,
    ) -> tuple[Decimal, float]:
        if adv_20_dollars > 0:
            participation_rate = order_notional / adv_20_dollars
            raw_slippage = self.slippage_bps_per_pct_adv * (participation_rate / 0.01)
        else:
            raw_slippage = self.slippage_bps_max

        slippage_bps = max(
            self.slippage_bps_min,
            min(self.slippage_bps_max, raw_slippage),
        )
        commission = self.commission_per_share * Decimal(shares)
        return (commission, slippage_bps)


@dataclass(frozen=True)
class CNCostModel:
    """A-share fee structure (2026 rates):
    - commission_bps: broker fee, bilateral (~2.5 bps)
    - stamp_duty_bps_sell: sell-side stamp duty (5 bps after 2023 cut)
    - transfer_fee_bps: Shanghai transfer fee, bilateral (0.2 bps)
    - slippage_bps: fixed slippage (no ADV model in phase 1)
    """

    commission_bps: float = 2.5
    stamp_duty_bps_sell: float = 5.0
    transfer_fee_bps: float = 0.2
    slippage_bps: float = 5.0

    def compute_legacy(
        self,
        order_notional: float,
        adv_20_dollars: float,
        shares: int,
    ) -> tuple[Decimal, float]:
        return (Decimal("0"), self.slippage_bps)

    def compute_side(
        self,
        side: Literal["buy", "sell"],
        notional: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """Returns (commission_in_currency, slippage_in_currency)."""
        fee_bps = self.commission_bps + self.transfer_fee_bps
        if side == "sell":
            fee_bps += self.stamp_duty_bps_sell
        commission = notional * Decimal(str(fee_bps / 10000.0))
        slippage = notional * Decimal(str(self.slippage_bps / 10000.0))
        return (commission, slippage)


@dataclass(frozen=True)
class HKCostModel:
    """HK fee structure (2026 rates):
    - commission_bps: broker fee, bilateral
    - stamp_duty_bps: bilateral stamp duty (13 bps post-2023)
    - sfc_levy_bps: SFC levy, bilateral (0.27 bps)
    - trading_fee_bps: exchange trading fee, bilateral (0.5 bps)
    - slippage_bps: fixed slippage
    """

    commission_bps: float = 3.0
    stamp_duty_bps: float = 13.0
    sfc_levy_bps: float = 0.27
    trading_fee_bps: float = 0.5
    slippage_bps: float = 5.0

    def compute_legacy(
        self,
        order_notional: float,
        adv_20_dollars: float,
        shares: int,
    ) -> tuple[Decimal, float]:
        return (Decimal("0"), self.slippage_bps)

    def compute_side(
        self,
        side: Literal["buy", "sell"],
        notional: Decimal,
    ) -> tuple[Decimal, Decimal]:
        fee_bps = (
            self.commission_bps
            + self.stamp_duty_bps
            + self.sfc_levy_bps
            + self.trading_fee_bps
        )
        commission = notional * Decimal(str(fee_bps / 10000.0))
        slippage = notional * Decimal(str(self.slippage_bps / 10000.0))
        return (commission, slippage)
