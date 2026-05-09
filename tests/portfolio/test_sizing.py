"""Tests for the portfolio sizing module — ATR allocation, truncation, cash exhaustion."""

from datetime import date

import pytest

from clenow.config import Config
from clenow.portfolio.sizing import compute_target_positions
from clenow.types import Position


class TestNormalATRSizing:
    """Normal case: floor((equity * risk_factor) / ATR) shares allocated."""

    def test_basic_sizing(self):
        """Simple ATR sizing with known values."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 1_000_000.0
        current_positions = {}

        # ATR = $2.00, price = $100
        # equity = 1_000_000
        # target_shares = floor(1_000_000 * 0.001 / 2) = floor(500) = 500
        # position_value = 500 * 100 = $50_000 (5% of equity → exactly at cap, no truncation needed)
        atrs = {"AAPL": 2.0}
        prices = {"AAPL": 100.0}

        result = compute_target_positions(
            ["AAPL"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert result["AAPL"] == 500

    def test_multiple_stocks_sequential(self):
        """Multiple stocks allocated in rank order, cash decremented."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 100_000.0
        current_positions = {}

        # AAPL: ATR=2.0, price=100 → floor(100_000*0.001/2) = floor(50) = 50, value=5000
        # MSFT: ATR=1.5, price=80  → floor(100_000*0.001/1.5) = floor(66) = 66, value=5280
        # But max_position_pct=0.05, so 5% of 100k = 5000.
        # AAPL value = 50*100 = 5000 → OK (exactly 5%)
        # MSFT value = 66*80 = 5280 > 5000 → truncated to floor(5000/80) = 62, value=4960
        atrs = {"AAPL": 2.0, "MSFT": 1.5}
        prices = {"AAPL": 100.0, "MSFT": 80.0}

        result = compute_target_positions(
            ["AAPL", "MSFT"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert result["AAPL"] == 50
        assert result["MSFT"] == 62


class TestFivePctTruncation:
    """5% single-stock cap truncates shares at entry."""

    def test_position_truncated_at_five_pct(self):
        """Large ATR allocation truncated to 5% of equity."""
        config = Config(risk_factor=0.01, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 1_000_000.0
        current_positions = {}

        # ATR = $1.0, price = $200
        # target_shares = floor(1_000_000 * 0.01 / 1) = 10000
        # position_value = 10000 * 200 = $2_000_000 > 5% of $1M ($50_000)
        # truncated: floor(50_000 / 200) = 250
        atrs = {"AAPL": 1.0}
        prices = {"AAPL": 200.0}

        result = compute_target_positions(
            ["AAPL"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert result["AAPL"] == 250


class TestCashExhaustion:
    """When available_cash < position_value, allocation stops (break)."""

    def test_cash_exhaustion_stops_allocation(self):
        """Later stocks not allocated when cash runs out."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 50_000.0
        current_positions = {}

        # equity = 50_000
        # AAPL: ATR=0.5, price=100 → floor(50_000*0.001/0.5) = 100, value=10000
        #   5% cap = 2500 → truncated to floor(2500/100) = 25, value=2500
        # MSFT: ATR=0.5, price=100 → same → value=2500
        # GOOG: ATR=0.5, price=100 → same → value=2500
        # After AAPL: 50_000 - 2500 = 47500
        # After MSFT: 47500 - 2500 = 45000
        # After GOOG: 45000 - 2500 = 42500
        # AMZN: ATR=0.5, price=100 → value=2500 → 40000
        # ... keep going until cash exhausted
        # With 50k cash and 2500 per position, we can fit 20 positions
        tickers = [f"T{i}" for i in range(25)]
        atrs = {t: 0.5 for t in tickers}
        prices = {t: 100.0 for t in tickers}

        result = compute_target_positions(
            tickers, None, as_of, config, current_cash, current_positions, prices, atrs
        )
        # Each position costs 2500, starting with 50_000 → 20 positions
        assert len(result) == 20
        # Verify they are in rank order
        assert list(result.keys()) == tickers[:20]

    def test_cash_too_low_for_any_position(self):
        """If even one position can't be afforded, result is empty."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 100.0  # very little cash
        current_positions = {}

        # equity = 100
        # AAPL: floor(100 * 0.001 / 2) = 0 → skip (target_shares <= 0)
        atrs = {"AAPL": 2.0}
        prices = {"AAPL": 100.0}

        result = compute_target_positions(
            ["AAPL"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert result == {}


class TestATRSkip:
    """ATR < $0.01 → stock skipped (illiquid/stopped)."""

    def test_zero_atr_skipped(self):
        """ATR of 0.0 → stock not allocated."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 1_000_000.0
        current_positions = {}

        atrs = {"AAPL": 0.0}
        prices = {"AAPL": 100.0}

        result = compute_target_positions(
            ["AAPL"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert "AAPL" not in result

    def test_tiny_atr_skipped(self):
        """ATR of $0.005 → stock not allocated (below $0.01 threshold)."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 1_000_000.0
        current_positions = {}

        atrs = {"AAPL": 0.005}
        prices = {"AAPL": 100.0}

        result = compute_target_positions(
            ["AAPL"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert "AAPL" not in result

    def test_missing_atr_skipped(self):
        """Ticker missing from atrs dict → treated as 0.0 and skipped."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 1_000_000.0
        current_positions = {}

        atrs = {}  # no ATR for AAPL
        prices = {"AAPL": 100.0}

        result = compute_target_positions(
            ["AAPL"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert "AAPL" not in result

    def test_atr_just_above_threshold(self):
        """ATR of exactly $0.01 → stock is allocated (>= 0.01)."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 1_000_000.0
        current_positions = {}

        # ATR = 0.01 → target_shares = floor(1_000_000 * 0.001 / 0.01) = floor(100_000) = 100_000
        # position_value = 100_000 * 100 = $10_000_000 → way over 5%
        # truncated: floor(50_000 / 100) = 500
        atrs = {"AAPL": 0.01}
        prices = {"AAPL": 100.0}

        result = compute_target_positions(
            ["AAPL"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert result["AAPL"] == 500


class TestSequentialAllocationOrder:
    """Verify allocation follows rank order and cash decrements correctly."""

    def test_allocation_decrements_available_cash(self):
        """First allocation reduces available cash for second."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 100_000.0
        current_positions = {}

        # Both stocks: ATR=1.0, price=100
        # equity=100_000, target=floor(100_000*0.001/1)=100, value=10_000
        # But 5% of 100_000 = 5_000, so truncated to floor(5_000/100) = 50, value=5_000
        # After AAPL: available = 100_000 - 5_000 = 95_000
        # After MSFT: available = 95_000 - 5_000 = 90_000
        atrs = {"AAPL": 1.0, "MSFT": 1.0}
        prices = {"AAPL": 100.0, "MSFT": 100.0}

        result = compute_target_positions(
            ["AAPL", "MSFT"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert result["AAPL"] == 50
        assert result["MSFT"] == 50

    def test_empty_tickers_returns_empty(self):
        """No tickers → empty result."""
        config = Config()
        result = compute_target_positions(
            [], None, date(2024, 6, 1), config, 100_000.0, {}, {}, {}
        )
        assert result == {}

    def test_with_existing_positions(self):
        """Equity includes existing position values."""
        config = Config(risk_factor=0.001, max_position_pct=0.05)
        as_of = date(2024, 6, 1)
        current_cash = 50_000.0
        current_positions = {
            "EXIST": Position(
                ticker="EXIST", shares=1000, entry_price=100.0,
                entry_date=date(2024, 1, 1), atr_at_entry=2.0,
            )
        }

        # equity = 50_000 + 1000*100 = 150_000
        # AAPL: ATR=3.0, price=100 → floor(150_000*0.001/3) = floor(50) = 50
        # position_value = 50*100 = 5000, 5% of 150k = 7500 → OK (no truncation)
        # available_cash = 50_000 >= 5000 → OK
        atrs = {"AAPL": 3.0}
        prices = {"AAPL": 100.0}

        result = compute_target_positions(
            ["AAPL"], None, as_of, config, current_cash, current_positions, prices, atrs
        )
        assert result["AAPL"] == 50
