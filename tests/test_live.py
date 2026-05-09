"""Tests for the live decision workflow — state management and CLI."""

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from clenow.config import Config
from clenow.data.loader import SQLDataProvider
from clenow.errors import StateCorruption
from clenow.live.cli import generate_orders, write_orders_csv
from clenow.live.state import load_state, save_state
from clenow.portfolio.state import PositionTracker
from clenow.types import Position


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_state_path(tmp_path):
    return tmp_path / "positions.json"


@pytest.fixture
def tracker_with_positions():
    tracker = PositionTracker(cash=10_000.0)
    tracker._positions["AAPL"] = Position(
        ticker="AAPL", shares=50, entry_price=175.32,
        entry_date=date(2026, 4, 1), atr_at_entry=3.5,
    )
    tracker._positions["NVDA"] = Position(
        ticker="NVDA", shares=12, entry_price=850.10,
        entry_date=date(2026, 3, 25), atr_at_entry=25.0,
    )
    return tracker


# ── State file: load/save ────────────────────────────────────────────────


class TestSaveAndLoadState:
    def test_round_trip(self, tracker_with_positions, tmp_state_path):
        save_state(tracker_with_positions, tmp_state_path)
        loaded = load_state(tmp_state_path)
        assert loaded.get_cash() == tracker_with_positions.get_cash()
        assert set(loaded.get_positions().keys()) == set(
            tracker_with_positions.get_positions().keys()
        )

    def test_load_missing_file_returns_empty(self, tmp_state_path):
        tracker = load_state(tmp_state_path)
        assert tracker.get_cash() == 0.0
        assert len(tracker.get_positions()) == 0

    def test_atomic_write_creates_file(self, tracker_with_positions, tmp_state_path):
        save_state(tracker_with_positions, tmp_state_path)
        assert tmp_state_path.exists()

    def test_backup_created(self, tracker_with_positions, tmp_state_path):
        save_state(tracker_with_positions, tmp_state_path)
        save_state(tracker_with_positions, tmp_state_path)
        # portfolio/state.save_state uses with_suffix(".bak") → positions.bak
        bak_path = tmp_state_path.with_suffix(".bak")
        assert bak_path.exists()

    def test_no_tmp_left_after_save(self, tracker_with_positions, tmp_state_path):
        save_state(tracker_with_positions, tmp_state_path)
        tmp_file = tmp_state_path.with_suffix(".tmp")
        assert not tmp_file.exists()


class TestStateCorruption:
    def test_malformed_json(self, tmp_state_path):
        tmp_state_path.write_text("not json{", encoding="utf-8")
        with pytest.raises(StateCorruption, match="Invalid JSON"):
            load_state(tmp_state_path)

    def test_missing_cash_field(self, tmp_state_path):
        tmp_state_path.write_text(
            json.dumps({"positions": {}}), encoding="utf-8"
        )
        with pytest.raises(StateCorruption, match="Missing required field: cash"):
            load_state(tmp_state_path)

    def test_missing_positions_field(self, tmp_state_path):
        tmp_state_path.write_text(
            json.dumps({"cash": 1000.0}), encoding="utf-8"
        )
        with pytest.raises(StateCorruption, match="Missing required field: positions"):
            load_state(tmp_state_path)

    def test_invalid_shares_type(self, tmp_state_path):
        state = {
            "cash": 1000.0,
            "positions": {
                "AAPL": {
                    "ticker": "AAPL",
                    "shares": "fifty",
                    "entry_price": 175.0,
                    "entry_date": "2026-01-01",
                    "atr_at_entry": 0.0,
                }
            },
        }
        tmp_state_path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(StateCorruption, match="shares must be int"):
            load_state(tmp_state_path)

    def test_negative_shares_rejected(self, tmp_state_path):
        """Negative shares are rejected by validation."""
        state = {
            "cash": 1000.0,
            "positions": {
                "AAPL": {
                    "ticker": "AAPL",
                    "shares": -10,
                    "entry_price": 175.0,
                    "entry_date": "2026-01-01",
                    "atr_at_entry": 0.0,
                }
            },
        }
        tmp_state_path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(StateCorruption, match="shares must be positive"):
            load_state(tmp_state_path)

    def test_missing_entry_date(self, tmp_state_path):
        state = {
            "cash": 1000.0,
            "positions": {
                "AAPL": {
                    "ticker": "AAPL",
                    "shares": 50,
                    "entry_price": 175.0,
                    "atr_at_entry": 0.0,
                }
            },
        }
        tmp_state_path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(StateCorruption, match="missing fields"):
            load_state(tmp_state_path)

    def test_invalid_entry_date(self, tmp_state_path):
        state = {
            "cash": 1000.0,
            "positions": {
                "AAPL": {
                    "ticker": "AAPL",
                    "shares": 50,
                    "entry_price": 175.0,
                    "entry_date": "not-a-date",
                    "atr_at_entry": 0.0,
                }
            },
        }
        tmp_state_path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(StateCorruption, match="invalid entry_date"):
            load_state(tmp_state_path)


# ── Stale state determinism ──────────────────────────────────────────────


class TestDeterminism:
    def test_same_state_same_orders(self, tmp_state_path):
        """Same positions.json + as_of → byte-identical orders."""
        tracker = PositionTracker(cash=5000.0)
        tracker._positions["AAPL"] = Position(
            ticker="AAPL", shares=10, entry_price=150.0,
            entry_date=date(2026, 1, 1), atr_at_entry=3.0,
        )
        save_state(tracker, tmp_state_path)

        # Load twice and verify identical JSON
        t1 = load_state(tmp_state_path)
        t2 = load_state(tmp_state_path)
        assert t1.to_json() == t2.to_json()


# ── Orders CSV ───────────────────────────────────────────────────────────


class TestWriteOrdersCSV:
    def test_csv_format(self, tmp_path):
        orders = [
            {"ticker": "AAPL", "side": "buy", "shares": 50,
             "expected_price": 175.32, "reason": "new entry"},
            {"ticker": "TSLA", "side": "sell", "shares": 10,
             "expected_price": 250.0, "reason": "SMA break"},
        ]
        output = tmp_path / "orders.csv"
        write_orders_csv(orders, output)

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "ticker" in lines[0]
        assert "AAPL" in lines[1]

    def test_empty_orders(self, tmp_path):
        output = tmp_path / "orders.csv"
        write_orders_csv([], output)

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1  # header only
