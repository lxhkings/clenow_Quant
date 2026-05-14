"""Tests for SectorRotation strategy."""
import pytest

from clenow.config import Config
from clenow.strategy.sector_rotation import SectorRotation


def test_sector_rotation_picks_one_per_sector():
    sm = {"A": "Tech", "B": "Tech", "C": "Finance", "D": "Finance", "E": "Energy"}
    scores = {"A": 0.5, "B": 0.3, "C": 0.4, "D": 0.2, "E": 0.1}
    strat = SectorRotation(sector_mapping=sm)
    ranked = strat.rank(scores, Config())
    # Picks highest in each sector: A (Tech 0.5), C (Finance 0.4), E (Energy 0.1)
    assert ranked == ["A", "C", "E"]


def test_sector_rotation_drops_zero_or_negative():
    sm = {"A": "Tech", "B": "Finance"}
    scores = {"A": 0.5, "B": -0.1}
    strat = SectorRotation(sector_mapping=sm)
    ranked = strat.rank(scores, Config())
    assert ranked == ["A"]


def test_sector_rotation_missing_mapping_excludes():
    sm = {"A": "Tech"}  # B unmapped
    scores = {"A": 0.5, "B": 0.4}
    strat = SectorRotation(sector_mapping=sm)
    ranked = strat.rank(scores, Config())
    assert ranked == ["A"]