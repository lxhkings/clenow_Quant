"""Signal calculation layer."""

from clenow.signals.atr import compute_atr
from clenow.signals.clenow_score import compute_clenow_score
from clenow.signals.regime import is_bull_regime

__all__ = [
    "compute_clenow_score",
    "compute_atr",
    "is_bull_regime",
]
