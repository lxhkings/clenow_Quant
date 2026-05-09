"""Tests for the portfolio state module — PositionTracker."""

import json
from datetime import date, datetime

import pytest

from clenow.errors import InvariantError
from clenow.portfolio.state import PositionTracker
from clenow.types import Fill, Position, Side


class TestApplyFillsBuy:
    """Buy fills: create or increase positions, decrease cash."""

    def test_buy_creates_position(self):
        """A buy fill creates a new position and reduces cash."""
        tracker = PositionTracker(cash=100_000.0)
        fill = Fill(
            ticker="AAPL",
            shares=100,
            fill_price=150.0,
            timestamp=datetime(2024, 6, 1, 10, 0, 0),
            commission=1.0,
            slippage_bps=5.0,
            side=Side.BUY,
        )
        tracker.apply_fills([fill])

        pos = tracker.get_positions()["AAPL"]
        assert pos.shares == 100
        assert pos.entry_price == 150.0
        # cash = 100_000 - (150*100 + 1) = 84_999
        assert tracker.get_cash() == pytest.approx(84_999.0)

    def test_buy_increases_existing_position(self):
        """Additional buy increases shares with weighted average entry price."""
        tracker = PositionTracker(
            cash=100_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=100.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=2.0,
                )
            },
        )
        fill = Fill(
            ticker="AAPL",
            shares=100,
            fill_price=200.0,
            timestamp=datetime(2024, 6, 1, 10, 0, 0),
            commission=0.0,
            slippage_bps=0.0,
            side=Side.BUY,
        )
        tracker.apply_fills([fill])

        pos = tracker.get_positions()["AAPL"]
        assert pos.shares == 200
        # weighted average: (100*100 + 100*200) / 200 = 150.0
        assert pos.entry_price == pytest.approx(150.0)
        # cash = 100_000 - 200*100 = 80_000
        assert tracker.get_cash() == pytest.approx(80_000.0)


class TestApplyFillsSell:
    """Sell fills: reduce/remove positions, increase cash."""

    def test_sell_removes_position(self):
        """Full sell removes the position and increases cash."""
        tracker = PositionTracker(
            cash=0.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=2.0,
                )
            },
        )
        fill = Fill(
            ticker="AAPL",
            shares=100,
            fill_price=200.0,
            timestamp=datetime(2024, 6, 1, 10, 0, 0),
            commission=5.0,
            slippage_bps=0.0,
            side=Side.SELL,
        )
        tracker.apply_fills([fill])

        assert "AAPL" not in tracker.get_positions()
        # cash = 0 + 200*100 - 5 = 19_995
        assert tracker.get_cash() == pytest.approx(19_995.0)

    def test_partial_sell_reduces_shares(self):
        """Partial sell reduces shares but keeps position."""
        tracker = PositionTracker(
            cash=0.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=2.0,
                )
            },
        )
        fill = Fill(
            ticker="AAPL",
            shares=40,
            fill_price=200.0,
            timestamp=datetime(2024, 6, 1, 10, 0, 0),
            commission=2.0,
            slippage_bps=0.0,
            side=Side.SELL,
        )
        tracker.apply_fills([fill])

        pos = tracker.get_positions()["AAPL"]
        assert pos.shares == 60
        assert pos.entry_price == pytest.approx(150.0)  # unchanged
        # cash = 0 + 200*40 - 2 = 7_998
        assert tracker.get_cash() == pytest.approx(7_998.0)

    def test_sell_increases_cash(self):
        """Sell increases cash by (fill_price * shares - commission)."""
        tracker = PositionTracker(
            cash=1_000.0,
            positions={
                "MSFT": Position(
                    ticker="MSFT", shares=50, entry_price=300.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=5.0,
                )
            },
        )
        fill = Fill(
            ticker="MSFT",
            shares=50,
            fill_price=350.0,
            timestamp=datetime(2024, 6, 1, 10, 0, 0),
            commission=10.0,
            slippage_bps=0.0,
            side=Side.SELL,
        )
        tracker.apply_fills([fill])

        # cash = 1_000 + 350*50 - 10 = 18_490
        assert tracker.get_cash() == pytest.approx(18_490.0)

    def test_buy_decreases_cash(self):
        """Buy decreases cash by (fill_price * shares + commission)."""
        tracker = PositionTracker(cash=50_000.0)
        fill = Fill(
            ticker="GOOG",
            shares=50,
            fill_price=100.0,
            timestamp=datetime(2024, 6, 1, 10, 0, 0),
            commission=5.0,
            slippage_bps=0.0,
            side=Side.BUY,
        )
        tracker.apply_fills([fill])

        # cash = 50_000 - (100*50 + 5) = 44_995
        assert tracker.get_cash() == pytest.approx(44_995.0)


class TestApplySplit:
    """Stock split: shares *= ratio, entry_price /= ratio."""

    def test_four_to_one_split(self):
        """4:1 split: 100 shares @ $175.32 → 400 shares @ $43.83."""
        tracker = PositionTracker(
            cash=10_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=175.32,
                    entry_date=date(2024, 1, 1), atr_at_entry=4.0,
                )
            },
        )
        tracker.apply_split("AAPL", 4.0, date(2024, 6, 1))

        pos = tracker.get_positions()["AAPL"]
        assert pos.shares == 400
        assert pos.entry_price == pytest.approx(175.32 / 4.0)
        # 175.32 / 4 = 43.83
        assert pos.entry_price == pytest.approx(43.83, rel=1e-2)

    def test_reverse_split(self):
        """1:10 reverse split: 1000 shares @ $0.50 → 100 shares @ $5.00."""
        tracker = PositionTracker(
            cash=10_000.0,
            positions={
                "XYZ": Position(
                    ticker="XYZ", shares=1000, entry_price=0.50,
                    entry_date=date(2024, 1, 1), atr_at_entry=0.1,
                )
            },
        )
        tracker.apply_split("XYZ", 0.1, date(2024, 6, 1))

        pos = tracker.get_positions()["XYZ"]
        assert pos.shares == 100
        assert pos.entry_price == pytest.approx(5.0)

    def test_split_on_unknown_ticker_raises(self):
        """InvariantError raised for split on unknown ticker."""
        tracker = PositionTracker(cash=10_000.0)
        with pytest.raises(InvariantError) as exc_info:
            tracker.apply_split("UNKNOWN", 2.0, date(2024, 6, 1))
        assert exc_info.value.ticker == "UNKNOWN"
        assert exc_info.value.event_type == "split"


class TestApplyDividend:
    """Cash dividend: cash += shares * dividend_per_share."""

    def test_dividend_increases_cash(self):
        """$0.50 dividend on 100 shares → cash += $50."""
        tracker = PositionTracker(
            cash=1_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=3.0,
                )
            },
        )
        tracker.apply_dividend("AAPL", 0.50, date(2024, 6, 1))

        assert tracker.get_cash() == pytest.approx(1_050.0)
        pos = tracker.get_positions()["AAPL"]
        assert pos.shares == 100  # unchanged
        assert pos.entry_price == pytest.approx(150.0)  # unchanged

    def test_dividend_on_unknown_ticker_raises(self):
        """InvariantError raised for dividend on unknown ticker."""
        tracker = PositionTracker(cash=10_000.0)
        with pytest.raises(InvariantError) as exc_info:
            tracker.apply_dividend("UNKNOWN", 1.0, date(2024, 6, 1))
        assert exc_info.value.ticker == "UNKNOWN"
        assert exc_info.value.event_type == "dividend"


class TestApplyDelisting:
    """Delisting: force close, cash += last_price * shares, position removed."""

    def test_delisting_force_closes_position(self):
        """LEH delisted at $0.30 with 50 shares → cash += $15, position removed."""
        tracker = PositionTracker(
            cash=1_000.0,
            positions={
                "LEH": Position(
                    ticker="LEH", shares=50, entry_price=60.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=2.0,
                )
            },
        )
        tracker.apply_delisting("LEH", 0.30, date(2024, 6, 1))

        assert "LEH" not in tracker.get_positions()
        assert tracker.get_cash() == pytest.approx(1_015.0)

    def test_delisting_unknown_ticker_is_idempotent(self):
        """Delisting an unknown ticker is a no-op (idempotent)."""
        tracker = PositionTracker(cash=1_000.0)
        tracker.apply_delisting("UNKNOWN", 0.10, date(2024, 6, 1))
        assert tracker.get_cash() == pytest.approx(1_000.0)
        assert tracker.get_positions() == {}


class TestGetEquity:
    """get_equity: cash + sum(shares * prices[ticker])."""

    def test_equity_calculation(self):
        """Equity = cash + sum of position values at current prices."""
        tracker = PositionTracker(
            cash=10_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=3.0,
                ),
                "MSFT": Position(
                    ticker="MSFT", shares=50, entry_price=300.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=5.0,
                ),
            },
        )
        prices = {"AAPL": 200.0, "MSFT": 350.0}
        # equity = 10_000 + 100*200 + 50*350 = 10_000 + 20_000 + 17_500 = 47_500
        assert tracker.get_equity(prices) == pytest.approx(47_500.0)

    def test_equity_with_missing_prices(self):
        """Positions with missing prices are excluded from equity."""
        tracker = PositionTracker(
            cash=10_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=3.0,
                ),
            },
        )
        prices = {}  # no price for AAPL
        # equity = 10_000 + 0 = 10_000
        assert tracker.get_equity(prices) == pytest.approx(10_000.0)


class TestJsonRoundTrip:
    """to_json / from_json must produce byte-identical round-trip."""

    def test_simple_round_trip(self):
        """Simple tracker state survives serialization round-trip."""
        tracker = PositionTracker(
            cash=12345.67,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 15), atr_at_entry=3.5,
                ),
            },
        )
        json_str = tracker.to_json()
        restored = PositionTracker.from_json(json_str)

        assert restored.get_cash() == pytest.approx(12345.67)
        assert "AAPL" in restored.get_positions()
        pos = restored.get_positions()["AAPL"]
        assert pos.shares == 100
        assert pos.entry_price == pytest.approx(150.0)
        assert pos.entry_date == date(2024, 1, 15)
        assert pos.atr_at_entry == pytest.approx(3.5)

    def test_byte_identical_round_trip(self):
        """Serialize → deserialize → serialize produces identical output."""
        tracker = PositionTracker(
            cash=99999.99,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 15), atr_at_entry=3.5,
                ),
                "MSFT": Position(
                    ticker="MSFT", shares=200, entry_price=300.0,
                    entry_date=date(2024, 2, 20), atr_at_entry=5.0,
                ),
                "GOOG": Position(
                    ticker="GOOG", shares=50, entry_price=120.0,
                    entry_date=date(2024, 3, 10), atr_at_entry=2.5,
                ),
            },
        )
        json1 = tracker.to_json()
        restored = PositionTracker.from_json(json1)
        json2 = restored.to_json()
        assert json1 == json2

    def test_empty_positions_round_trip(self):
        """Tracker with no positions survives round-trip."""
        tracker = PositionTracker(cash=50_000.0)
        json_str = tracker.to_json()
        restored = PositionTracker.from_json(json_str)
        assert restored.get_cash() == pytest.approx(50_000.0)
        assert restored.get_positions() == {}
        assert json_str == restored.to_json()

    def test_complex_state_after_operations(self):
        """Complex state with multiple operations survives round-trip."""
        tracker = PositionTracker(
            cash=100_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=3.0,
                ),
            },
        )
        # Apply some operations
        tracker.apply_split("AAPL", 4.0, date(2024, 6, 1))
        tracker.apply_dividend("AAPL", 0.82, date(2024, 6, 15))

        json1 = tracker.to_json()
        restored = PositionTracker.from_json(json1)
        json2 = restored.to_json()
        assert json1 == json2
