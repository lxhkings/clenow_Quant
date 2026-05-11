"""MarketProfile: frozen registry of per-market constants.

Pure data + a price_limit_resolver Callable. No logic beyond field defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from clenow.markets.costs import CNCostModel, CostModel, HKCostModel, USCostModel
from clenow.markets.price_limit import cn_price_limit_resolver


@dataclass(frozen=True)
class MarketProfile:
    name: str
    currency: str
    universe_index_id: str
    regime_index_id: str
    annualization_days: int
    lot_size: int
    min_price: float
    min_adv_amount: float
    trading_calendar_name: str
    cost_model: CostModel
    default_starting_capital: float
    # Strategy windows (Clenow defaults; per-market override-able)
    score_window: int = 90
    regime_sma_window: int = 200
    exit_sma_window: int = 100
    # Market microstructure
    price_limit_pct: float | None = None  # If set, fixed limit; else use resolver
    price_limit_resolver: Callable[[str, str], float] | None = None  # (ticker, name) -> limit
    suspension_threshold_days: int = 0  # consec volume=0 <= this -> suspended (hold position)
    delisting_threshold_days: int = 10  # consec volume=0 > this -> force close
    universe_exclusion_filters: tuple[str, ...] = ()  # e.g. ("st",)


PROFILES: dict[str, MarketProfile] = {
    "US": MarketProfile(
        name="US",
        currency="USD",
        universe_index_id="SP500",
        regime_index_id="SP500",
        annualization_days=252,
        lot_size=1,
        min_price=5.0,
        min_adv_amount=10_000_000,
        trading_calendar_name="NYSE",
        cost_model=USCostModel(),
        default_starting_capital=100_000,
        suspension_threshold_days=0,
        delisting_threshold_days=10,
    ),
    "CN": MarketProfile(
        name="CN",
        currency="CNY",
        universe_index_id="CSI800",
        regime_index_id="CSI800",
        annualization_days=244,
        lot_size=100,
        min_price=5.0,
        min_adv_amount=50_000_000,
        trading_calendar_name="XSHG",
        cost_model=CNCostModel(),
        default_starting_capital=1_000_000,
        price_limit_resolver=cn_price_limit_resolver,
        suspension_threshold_days=20,
        delisting_threshold_days=60,
        universe_exclusion_filters=("st",),
    ),
    "HK": MarketProfile(
        name="HK",
        currency="HKD",
        universe_index_id="HSI",
        regime_index_id="HSI",
        annualization_days=247,
        lot_size=100,
        min_price=1.0,
        min_adv_amount=50_000_000,
        trading_calendar_name="XHKG",
        cost_model=HKCostModel(),
        default_starting_capital=1_000_000,
        suspension_threshold_days=10,
        delisting_threshold_days=30,
    ),
}


def get_profile(market: str) -> MarketProfile:
    """Case-insensitive profile lookup. Raises KeyError on unknown market."""
    key = market.upper()
    if key not in PROFILES:
        raise KeyError(f"Unknown market: {market!r}. Known: {sorted(PROFILES.keys())}")
    return PROFILES[key]
