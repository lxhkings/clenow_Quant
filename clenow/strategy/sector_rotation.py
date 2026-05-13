"""1-stock-per-sector momentum strategy (Clenow score + sector rotation).

Overrides rank() only — score/entry_filters/exit_signal/size identical to
ClenowMomentum.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from clenow.strategy.clenow_momentum import ClenowMomentum

if TYPE_CHECKING:
    from clenow.config import Config

logger = logging.getLogger(__name__)


class SectorRotation(ClenowMomentum):
    """Sector rotation strategy: pick highest-score stock per sector.

    Inherits score/entry_filters/exit_signal/size from ClenowMomentum.
    Only overrides rank() to enforce one-stock-per-sector selection.
    """

    name = "sector_rotation"

    def __init__(self, sector_mapping: dict[str, str]) -> None:
        if not sector_mapping:
            raise ValueError("sector_mapping required")
        self.sector_mapping = sector_mapping

    def rank(self, scores: dict[str, float], config: "Config") -> list[str]:
        """Select highest-score stock from each sector.

        Args:
            scores: ticker -> Clenow score mapping.
            config: Config (unused, kept for interface consistency).

        Returns:
            List of tickers (one per sector) sorted by score descending.
        """
        # Group by sector
        by_sector: dict[str, list[tuple[str, float]]] = {}
        unmapped: list[str] = []
        for ticker, score in scores.items():
            if score <= 0:
                continue
            sector = self.sector_mapping.get(ticker)
            if sector:
                by_sector.setdefault(sector, []).append((ticker, score))
            else:
                unmapped.append(ticker)

        if unmapped:
            logger.warning(
                "%d tickers with positive score not in sector_mapping (excluded): %s...",
                len(unmapped),
                ", ".join(unmapped[:10]),
            )

        # Pick highest score per sector
        selected: list[tuple[str, float]] = []
        for sector, tickers in by_sector.items():
            sorted_tickers = sorted(tickers, key=lambda x: x[1], reverse=True)
            if sorted_tickers:
                selected.append(sorted_tickers[0])

        # Return sorted by score descending
        selected.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in selected]