"""Strategy abstraction layer."""
from clenow.strategy.base import Strategy, make_strategy
from clenow.strategy.clenow_momentum import ClenowMomentum
from clenow.strategy.sector_rotation import SectorRotation

__all__ = ["Strategy", "ClenowMomentum", "SectorRotation", "make_strategy"]