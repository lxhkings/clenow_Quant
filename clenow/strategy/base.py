"""Strategy abstraction: 5 hooks the engine calls to delegate strategy-related logic.

Engine only keeps generic parts (universe/delisting/orders/corporate actions/nav).
Strategy decides score/rank/entry_filters/exit_signal/size.
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from clenow.config import Config
    from clenow.data.provider import DataProvider
    from clenow.markets.profiles import MarketProfile
    from clenow.types import Position


class Strategy(Protocol):
    """Strategy interface for compute_target_portfolio.

    Implementations must be stateless wrt rebalance dates -- any state
    (e.g. sector_mapping) must be passed via __init__ and frozen.
    """

    name: str

    def score(
        self,
        ticker: str,
        ticker_data: pd.DataFrame,
        profile: "MarketProfile",
        config: "Config",
    ) -> float: ...

    def rank(
        self,
        scores: dict[str, float],
        config: "Config",
    ) -> list[str]: ...

    def entry_filters(
        self,
        ranked: list[str],
        all_prices: pd.DataFrame,
        data_provider: "DataProvider",
        as_of: date,
        config: "Config",
        profile: "MarketProfile",
        current_positions: dict[str, "Position"],
    ) -> list[str]: ...

    def exit_signal(
        self,
        ticker: str,
        all_prices: pd.DataFrame,
        as_of: date,
        data_provider: "DataProvider",
        profile: "MarketProfile",
        config: "Config",
    ) -> bool: ...

    def size(
        self,
        filtered_tickers: list[str],
        data_provider: "DataProvider",
        as_of: date,
        config: "Config",
        current_cash: float,
        current_positions: dict[str, "Position"],
        current_prices: dict[str, float],
        atrs: dict[str, float],
        profile: "MarketProfile",
    ) -> dict[str, int]: ...


def make_strategy(name: str, **kwargs) -> Strategy:
    """Factory. Known names: 'clenow_momentum', 'sector_rotation'.

    sector_rotation requires kwargs['sector_mapping'].
    """
    if name == "clenow_momentum":
        from clenow.strategy.clenow_momentum import ClenowMomentum
        return ClenowMomentum()
    if name == "sector_rotation":
        from clenow.strategy.sector_rotation import SectorRotation
        sm = kwargs.get("sector_mapping")
        if not sm:
            raise ValueError("sector_rotation requires sector_mapping")
        return SectorRotation(sector_mapping=sm)
    raise ValueError(f"unknown strategy: {name!r}")