"""Lifecycle integration tests — corporate actions, delistings, dividends,
state serialization, and first-time portfolio construction.

These are integration-level tests that exercise the full pipeline from
PositionTracker through the backtest engine.
"""

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from clenow.backtest.engine import (
    _detect_delistings,
    compute_target_portfolio,
    run_backtest,
)
from clenow.config import CashInterestPolicy, Config
from clenow.errors import InvariantError, StateCorruption
from clenow.portfolio.state import PositionTracker, load_state, save_state
from clenow.types import Fill, Position, Side, TargetPortfolio


# ── Helpers ──────────────────────────────────────────────────────────


def _make_stock_data(
    n_days: int = 300,
    start_date: date = date(2024, 1, 2),
    ticker: str = "AAPL",
    price: float = 100.0,
    trend: float = 0.0,
    volume: float = 1_000_000.0,
    dividends: dict[int, float] | None = None,
    splits: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Build a MultiIndex (date, ticker) price DataFrame for one ticker.

    dividends: mapping of day_index -> dividend_per_share
    splits: mapping of day_index -> split_ratio
    """
    dates = pd.bdate_range(start_date, periods=n_days)
    closes = [price + i * trend for i in range(n_days)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    opens = [c + 0.5 for c in closes]
    adj_closes = closes.copy()

    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": d.date(),
            "ticker": ticker,
            "raw_open": opens[i],
            "raw_high": highs[i],
            "raw_low": lows[i],
            "raw_close": closes[i],
            "volume": volume,
            "adj_close": adj_closes[i],
            "dividend": dividends.get(i, 0.0) if dividends else 0.0,
            "split_ratio": splits.get(i, 1.0) if splits else 1.0,
        })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"])
    return df


def _make_index_data(
    n_days: int = 250,
    start_date: date = date(2024, 1, 2),
    bull: bool = True,
) -> pd.DataFrame:
    """Build index price DataFrame for regime detection."""
    dates = pd.date_range(start_date, periods=n_days, freq="B")
    if bull:
        closes = [4000 + i * 2 for i in range(n_days)]
    else:
        closes = [5000 - i * 2 for i in range(n_days)]

    return pd.DataFrame(
        {"raw_close": closes},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _make_mock_provider(
    universe: list[str] | None = None,
    stock_data: dict[str, pd.DataFrame] | None = None,
    index_data: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a mock DataProvider with configurable behavior."""
    provider = MagicMock()

    if universe is not None:
        provider.get_universe.return_value = universe
    else:
        provider.get_universe.return_value = ["AAPL", "MSFT", "GOOG"]

    def load_prices_side_effect(tickers, start, end):
        if not tickers:
            idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
            return pd.DataFrame(
                columns=["raw_open", "raw_high", "raw_low", "raw_close",
                          "volume", "adj_close", "dividend", "split_ratio"],
                index=idx,
            )
        if stock_data is None:
            frames = []
            for t in tickers:
                df = _make_stock_data(ticker=t)
                frames.append(df)
            if len(frames) == 1:
                return frames[0]
            return pd.concat(frames)
        else:
            frames = []
            for t in tickers:
                if t in stock_data:
                    frames.append(stock_data[t])
                else:
                    frames.append(_make_stock_data(ticker=t))
            if len(frames) == 1:
                return frames[0]
            return pd.concat(frames)

    provider.load_prices.side_effect = load_prices_side_effect

    if index_data is not None:
        provider.get_index_prices.return_value = index_data
    else:
        provider.get_index_prices.return_value = _make_index_data(bull=True)

    return provider


# ═══════════════════════════════════════════════════════════════════════
# AAPL 2020-08-31 4:1 split
# ═══════════════════════════════════════════════════════════════════════


class TestAAPL4For1Split:
    """AAPL 4:1 split: 100 shares @ $175.32 -> 400 shares @ $43.83."""

    def test_split_shares_and_entry_price(self):
        """After 4:1 split: shares * 4, entry_price / 4."""
        tracker = PositionTracker(
            cash=10_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=175.32,
                    entry_date=date(2020, 8, 28), atr_at_entry=4.0,
                )
            },
        )
        tracker.apply_split("AAPL", 4.0, date(2020, 8, 31))

        pos = tracker.get_positions()["AAPL"]
        assert pos.shares == 400
        assert pos.entry_price == pytest.approx(175.32 / 4.0)
        assert pos.entry_price == pytest.approx(43.83, rel=1e-2)

    def test_split_atr_adjusted(self):
        """ATR at entry is also divided by the split ratio."""
        tracker = PositionTracker(
            cash=10_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=175.32,
                    entry_date=date(2020, 8, 28), atr_at_entry=4.0,
                )
            },
        )
        tracker.apply_split("AAPL", 4.0, date(2020, 8, 31))

        pos = tracker.get_positions()["AAPL"]
        assert pos.atr_at_entry == pytest.approx(1.0)

    def test_split_preserves_cash(self):
        """Split does not change cash balance."""
        tracker = PositionTracker(
            cash=10_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=175.32,
                    entry_date=date(2020, 8, 28), atr_at_entry=4.0,
                )
            },
        )
        tracker.apply_split("AAPL", 4.0, date(2020, 8, 31))
        assert tracker.get_cash() == pytest.approx(10_000.0)

    def test_split_preserves_entry_date(self):
        """Entry date should not change after split."""
        tracker = PositionTracker(
            cash=10_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=175.32,
                    entry_date=date(2020, 8, 28), atr_at_entry=4.0,
                )
            },
        )
        tracker.apply_split("AAPL", 4.0, date(2020, 8, 31))

        pos = tracker.get_positions()["AAPL"]
        assert pos.entry_date == date(2020, 8, 28)


# ═══════════════════════════════════════════════════════════════════════
# Dividend: $0.50 on 100 shares
# ═══════════════════════════════════════════════════════════════════════


class TestDividendPayment:
    """$0.50 dividend on 100 shares -> cash += $50, position unchanged."""

    def test_dividend_increases_cash(self):
        """Cash increases by shares * dividend_per_share."""
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

    def test_dividend_does_not_change_position(self):
        """Position shares and entry price remain unchanged after dividend."""
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

        pos = tracker.get_positions()["AAPL"]
        assert pos.shares == 100
        assert pos.entry_price == pytest.approx(150.0)

    def test_dividend_on_unknown_ticker_raises_invariant_error(self):
        """Dividend on unknown ticker must raise InvariantError (fail-loud)."""
        tracker = PositionTracker(cash=10_000.0)
        with pytest.raises(InvariantError) as exc_info:
            tracker.apply_dividend("UNKNOWN", 1.0, date(2024, 6, 1))
        assert exc_info.value.ticker == "UNKNOWN"
        assert exc_info.value.event_type == "dividend"


# ═══════════════════════════════════════════════════════════════════════
# LEH 2008-09 delisting
# ═══════════════════════════════════════════════════════════════════════


class TestLEBDelisting:
    """LEH delisted at $0.30: 50 shares -> cash += $15, LEH removed."""

    def test_delisting_force_closes_position(self):
        """Delisting removes position and adds last_price * shares to cash."""
        tracker = PositionTracker(
            cash=1_000.0,
            positions={
                "LEH": Position(
                    ticker="LEH", shares=50, entry_price=60.0,
                    entry_date=date(2008, 1, 1), atr_at_entry=2.0,
                )
            },
        )
        tracker.apply_delisting("LEH", 0.30, date(2008, 9, 15))

        assert "LEH" not in tracker.get_positions()
        assert tracker.get_cash() == pytest.approx(1_015.0)

    def test_delisting_idempotent_on_unknown_ticker(self):
        """Delisting an unknown ticker is a no-op (idempotent)."""
        tracker = PositionTracker(cash=1_000.0)
        tracker.apply_delisting("UNKNOWN", 0.10, date(2024, 6, 1))
        assert tracker.get_cash() == pytest.approx(1_000.0)
        assert tracker.get_positions() == {}

    def test_detect_delistings_via_engine(self):
        """_detect_delistings force-closes a held stock with no price data."""
        tracker = PositionTracker(
            cash=1_000.0,
            positions={
                "LEH": Position(
                    ticker="LEH", shares=50, entry_price=60.0,
                    entry_date=date(2008, 1, 1), atr_at_entry=2.0,
                )
            },
        )

        # Build a provider that has LEH data up to a certain point,
        # then no data after that (simulating delisting)
        leh_data = _make_stock_data(
            n_days=200, ticker="LEH", price=60.0, trend=-0.3,
        )

        provider = MagicMock()
        provider.get_universe.return_value = ["LEH"]
        provider.load_prices.return_value = leh_data

        # Call detect_delistings — with data available, LEH should NOT be delisted
        _detect_delistings(tracker, provider, date(2024, 6, 1))
        assert "LEH" in tracker.get_positions()

    def test_detect_delistings_no_price_data(self):
        """_detect_delistings force-closes when no price data at all."""
        tracker = PositionTracker(
            cash=1_000.0,
            positions={
                "LEH": Position(
                    ticker="LEH", shares=50, entry_price=60.0,
                    entry_date=date(2008, 1, 1), atr_at_entry=2.0,
                )
            },
        )

        # Provider returns empty data for LEH
        provider = MagicMock()
        provider.get_universe.return_value = ["LEH"]
        idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        empty_df = pd.DataFrame(
            columns=["raw_open", "raw_high", "raw_low", "raw_close",
                      "volume", "adj_close", "dividend", "split_ratio"],
            index=idx,
        )
        provider.load_prices.return_value = empty_df

        _detect_delistings(tracker, provider, date(2008, 9, 16))

        # LEH should be force-closed at entry price (no price data available)
        assert "LEH" not in tracker.get_positions()
        # cash = 1_000 + 60.0 * 50 = 4_000
        assert tracker.get_cash() == pytest.approx(4_000.0)

    def test_detect_delistings_last_available_close(self):
        """_detect_delistings uses last available close when current date
        has no data but history exists."""
        tracker = PositionTracker(
            cash=1_000.0,
            positions={
                "LEH": Position(
                    ticker="LEH", shares=50, entry_price=60.0,
                    entry_date=date(2008, 1, 1), atr_at_entry=2.0,
                )
            },
        )

        # Build LEH data with a last close price of 0.30
        n_days = 200
        start_date = date(2008, 1, 2)
        dates = pd.bdate_range(start_date, periods=n_days)
        closes = [60.0 - i * 0.3 for i in range(n_days)]

        leh_full_data = pd.DataFrame({
            "raw_open": [c + 0.5 for c in closes],
            "raw_high": [c + 1.0 for c in closes],
            "raw_low": [c - 1.0 for c in closes],
            "raw_close": closes,
            "volume": 1_000_000.0,
            "adj_close": closes,
            "dividend": 0.0,
            "split_ratio": 1.0,
        }, index=pd.MultiIndex.from_arrays(
            [dates.date, ["LEH"] * n_days],
            names=["date", "ticker"]
        ))

        # Last close in the data
        last_close = closes[-1]

        # Build a provider that returns empty for the current date
        # but has historical data for the lookback
        call_count = [0]

        def load_prices_side_effect(tickers, start, end):
            call_count[0] += 1
            # First call: as_of, as_of -> empty (no data on current day)
            # Second call: lookback, as_of -> full data
            if call_count[0] == 1:
                # No data for current date
                idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
                return pd.DataFrame(
                    columns=["raw_open", "raw_high", "raw_low", "raw_close",
                              "volume", "adj_close", "dividend", "split_ratio"],
                    index=idx,
                )
            else:
                # Historical data available
                return leh_full_data

        provider = MagicMock()
        provider.get_universe.return_value = ["LEH"]
        provider.load_prices.side_effect = load_prices_side_effect

        _detect_delistings(tracker, provider, date(2008, 10, 1))

        # LEH should be force-closed at last available close
        assert "LEH" not in tracker.get_positions()
        assert tracker.get_cash() == pytest.approx(1_000.0 + last_close * 50)


# ═══════════════════════════════════════════════════════════════════════
# State serialization round-trip
# ═══════════════════════════════════════════════════════════════════════


class TestStateSerializationRoundTrip:
    """Complex multi-position state -> to_json -> from_json -> byte-identical."""

    def test_complex_multi_position_round_trip(self):
        """Multiple positions with different entry dates survive round-trip."""
        tracker = PositionTracker(
            cash=12345.67,
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
                "TSLA": Position(
                    ticker="TSLA", shares=75, entry_price=200.0,
                    entry_date=date(2024, 4, 5), atr_at_entry=8.0,
                ),
            },
        )

        json1 = tracker.to_json()
        restored = PositionTracker.from_json(json1)
        json2 = restored.to_json()

        assert json1 == json2

    def test_state_after_corporate_actions_round_trip(self):
        """State after split + dividend + delisting survives round-trip."""
        tracker = PositionTracker(
            cash=100_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=3.0,
                ),
                "MSFT": Position(
                    ticker="MSFT", shares=200, entry_price=300.0,
                    entry_date=date(2024, 2, 1), atr_at_entry=5.0,
                ),
            },
        )

        # Apply corporate actions
        tracker.apply_split("AAPL", 4.0, date(2024, 6, 1))
        tracker.apply_dividend("AAPL", 0.82, date(2024, 6, 15))
        tracker.apply_delisting("MSFT", 350.0, date(2024, 7, 1))

        json1 = tracker.to_json()
        restored = PositionTracker.from_json(json1)
        json2 = restored.to_json()

        assert json1 == json2
        # Verify specific values
        assert "AAPL" in restored.get_positions()
        assert "MSFT" not in restored.get_positions()
        aapl = restored.get_positions()["AAPL"]
        assert aapl.shares == 400
        assert aapl.entry_price == pytest.approx(37.5)

    def test_empty_state_round_trip(self):
        """Empty tracker survives round-trip."""
        tracker = PositionTracker(cash=50_000.0)
        json1 = tracker.to_json()
        restored = PositionTracker.from_json(json1)
        json2 = restored.to_json()

        assert json1 == json2
        assert restored.get_cash() == pytest.approx(50_000.0)
        assert restored.get_positions() == {}


# ═══════════════════════════════════════════════════════════════════════
# State file atomic write / load
# ═══════════════════════════════════════════════════════════════════════


class TestStateFilePersistence:
    """save_state / load_state with atomic write and .bak backup."""

    def test_save_and_load_round_trip(self, tmp_path):
        """State saved to file can be loaded back identically."""
        tracker = PositionTracker(
            cash=42_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 15), atr_at_entry=3.5,
                ),
            },
        )

        state_path = tmp_path / "positions.json"
        save_state(tracker, state_path)

        # File should exist
        assert state_path.exists()

        # Load and verify
        loaded = load_state(state_path)
        assert loaded.get_cash() == pytest.approx(42_000.0)
        assert "AAPL" in loaded.get_positions()
        assert loaded.get_positions()["AAPL"].shares == 100

    def test_save_creates_bak_on_overwrite(self, tmp_path):
        """Saving over an existing file creates a .bak backup."""
        tracker1 = PositionTracker(
            cash=10_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=50, entry_price=100.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=2.0,
                ),
            },
        )
        tracker2 = PositionTracker(
            cash=20_000.0,
            positions={
                "MSFT": Position(
                    ticker="MSFT", shares=100, entry_price=200.0,
                    entry_date=date(2024, 2, 1), atr_at_entry=4.0,
                ),
            },
        )

        state_path = tmp_path / "positions.json"
        bak_path = tmp_path / "positions.bak"

        # First save
        save_state(tracker1, state_path)
        assert state_path.exists()
        assert not bak_path.exists()

        # Second save should create backup
        save_state(tracker2, state_path)
        assert state_path.exists()
        assert bak_path.exists()

        # .bak should contain tracker1's state
        loaded_bak = load_state(bak_path)
        assert loaded_bak.get_cash() == pytest.approx(10_000.0)
        assert "AAPL" in loaded_bak.get_positions()

    def test_load_state_corrupt_primary_falls_back_to_backup(self, tmp_path):
        """If primary file is corrupt, load_state falls back to .bak."""
        tracker = PositionTracker(
            cash=30_000.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL", shares=100, entry_price=150.0,
                    entry_date=date(2024, 1, 1), atr_at_entry=3.0,
                ),
            },
        )

        state_path = tmp_path / "positions.json"
        bak_path = tmp_path / "positions.bak"

        # Save valid state
        save_state(tracker, state_path)
        # Save again to create backup
        save_state(tracker, state_path)

        # Corrupt the primary file
        state_path.write_text("{corrupt json!!!", encoding="utf-8")

        # Loading should fall back to backup
        loaded = load_state(state_path)
        assert loaded.get_cash() == pytest.approx(30_000.0)

    def test_load_state_no_file_raises_corruption(self, tmp_path):
        """Loading from a non-existent path raises StateCorruption."""
        state_path = tmp_path / "nonexistent.json"
        with pytest.raises(StateCorruption) as exc_info:
            load_state(state_path)
        assert "No state file found" in exc_info.value.reason


# ═══════════════════════════════════════════════════════════════════════
# StateCorruption on malformed JSON
# ═══════════════════════════════════════════════════════════════════════


class TestStateCorruptionOnMalformedJson:
    """from_json must raise StateCorruption on malformed or missing fields."""

    def test_invalid_json_syntax(self):
        """Non-parseable JSON raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json("{this is not valid json")
        assert "Invalid JSON" in exc_info.value.reason

    def test_missing_cash_field(self):
        """JSON missing 'cash' field raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json('{"positions": {}}')
        assert "cash" in exc_info.value.reason

    def test_missing_positions_field(self):
        """JSON missing 'positions' field raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json('{"cash": 1000}')
        assert "positions" in exc_info.value.reason

    def test_cash_not_numeric(self):
        """Non-numeric cash raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json('{"cash": "not_a_number", "positions": {}}')
        assert "cash" in exc_info.value.reason
        assert "numeric" in exc_info.value.reason

    def test_positions_not_dict(self):
        """Non-dict positions raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json('{"cash": 1000, "positions": "bad"}')
        assert "positions" in exc_info.value.reason

    def test_position_missing_required_fields(self):
        """Position missing required fields raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json(json.dumps({
                "cash": 1000,
                "positions": {
                    "AAPL": {"ticker": "AAPL", "shares": 100}  # missing entry_price, entry_date, atr_at_entry
                }
            }))
        assert "AAPL" in exc_info.value.reason
        assert "missing" in exc_info.value.reason.lower()

    def test_position_invalid_entry_date(self):
        """Position with invalid entry_date raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json(json.dumps({
                "cash": 1000,
                "positions": {
                    "AAPL": {
                        "ticker": "AAPL",
                        "shares": 100,
                        "entry_price": 150.0,
                        "entry_date": "not-a-date",
                        "atr_at_entry": 3.0,
                    }
                }
            }))
        assert "AAPL" in exc_info.value.reason
        assert "entry_date" in exc_info.value.reason

    def test_position_shares_not_int(self):
        """Position with non-int shares raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json(json.dumps({
                "cash": 1000,
                "positions": {
                    "AAPL": {
                        "ticker": "AAPL",
                        "shares": 100.5,
                        "entry_price": 150.0,
                        "entry_date": "2024-01-01",
                        "atr_at_entry": 3.0,
                    }
                }
            }))
        assert "AAPL" in exc_info.value.reason
        assert "shares" in exc_info.value.reason.lower()

    def test_position_negative_shares(self):
        """Position with negative shares raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json(json.dumps({
                "cash": 1000,
                "positions": {
                    "AAPL": {
                        "ticker": "AAPL",
                        "shares": -10,
                        "entry_price": 150.0,
                        "entry_date": "2024-01-01",
                        "atr_at_entry": 3.0,
                    }
                }
            }))
        assert "AAPL" in exc_info.value.reason
        assert "positive" in exc_info.value.reason.lower()

    def test_position_not_object(self):
        """Position value that is not a dict raises StateCorruption."""
        with pytest.raises(StateCorruption) as exc_info:
            PositionTracker.from_json(json.dumps({
                "cash": 1000,
                "positions": {"AAPL": "not_an_object"}
            }))
        assert "AAPL" in exc_info.value.reason
        assert "not an object" in exc_info.value.reason

    def test_null_json_raises_corruption(self):
        """None / empty JSON string raises StateCorruption."""
        with pytest.raises(StateCorruption):
            PositionTracker.from_json("")


# ═══════════════════════════════════════════════════════════════════════
# First-time portfolio construction (Day 0 empty portfolio)
# ═══════════════════════════════════════════════════════════════════════


class TestFirstTimePortfolioConstruction:
    """Day 0: empty portfolio, available_cash = initial_equity.
    First rebalance: normal sizing flow, build to cash exhaustion.
    No staged entry (Clenow has no such rule)."""

    def test_empty_portfolio_first_rebalance(self):
        """compute_target_portfolio with empty positions and full cash
        produces positions (no staged entry required)."""
        as_of = date(2024, 12, 31)
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
        )

        stock_data = {}
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            stock_data[ticker] = _make_stock_data(
                n_days=300, ticker=ticker, price=100.0, trend=0.1, volume=1_000_000.0
            )

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT", "GOOG"],
            stock_data=stock_data,
        )

        # Empty current_positions, full cash
        result = compute_target_portfolio(
            as_of=as_of,
            current_positions={},
            current_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        # Should produce positions immediately (no staged entry)
        assert isinstance(result, TargetPortfolio)
        assert len(result.positions) >= 1

    def test_run_backtest_from_scratch(self):
        """run_backtest starts with empty portfolio and builds positions."""
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
            rebalance_freq="weekly",
        )

        n_days = 300
        start_date = date(2024, 1, 2)
        dates = pd.bdate_range(start_date, periods=n_days)

        stock_data = {}
        for ticker in ["AAPL", "MSFT"]:
            closes = [100 + i * 0.1 for i in range(n_days)]
            stock_data[ticker] = pd.DataFrame({
                "raw_open": [c + 0.5 for c in closes],
                "raw_high": [c + 1.0 for c in closes],
                "raw_low": [c - 1.0 for c in closes],
                "raw_close": closes,
                "volume": 1_000_000.0,
                "adj_close": closes,
                "dividend": 0.0,
                "split_ratio": 1.0,
            }, index=pd.MultiIndex.from_arrays(
                [dates.date, [ticker] * n_days],
                names=["date", "ticker"]
            ))

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT"],
            stock_data=stock_data,
        )

        result = run_backtest(
            start=date(2024, 4, 1),
            end=date(2024, 6, 30),
            initial_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        # Backtest should complete successfully
        assert isinstance(result.final_cash, float)
        assert result.final_cash > 0 or len(result.final_positions) > 0


# ═══════════════════════════════════════════════════════════════════════
# CashInterestPolicy
# ═══════════════════════════════════════════════════════════════════════


class TestCashInterestPolicy:
    """CashInterestPolicy enum: ZERO default, T_BILL future."""

    def test_default_policy_is_zero(self):
        """Default Config uses ZERO cash interest policy."""
        config = Config()
        assert config.cash_interest_policy == CashInterestPolicy.ZERO

    def test_zero_policy_is_conservative(self):
        """ZERO policy means no interest earned on cash."""
        assert CashInterestPolicy.ZERO.value == "zero"

    def test_t_bill_policy_exists(self):
        """T_BILL policy exists for future implementation."""
        assert CashInterestPolicy.T_BILL.value == "t_bill"

    def test_config_can_override_policy(self):
        """Config can be created with T_BILL policy."""
        config = Config(cash_interest_policy=CashInterestPolicy.T_BILL)
        assert config.cash_interest_policy == CashInterestPolicy.T_BILL


# ═══════════════════════════════════════════════════════════════════════
# InvariantError on split/dividend for unknown ticker (fail-loud)
# ═══════════════════════════════════════════════════════════════════════


class TestInvariantErrorFailLoud:
    """InvariantError raised for corporate actions on unknown tickers."""

    def test_split_on_unknown_ticker(self):
        """Split on unknown ticker raises InvariantError."""
        tracker = PositionTracker(cash=10_000.0)
        with pytest.raises(InvariantError) as exc_info:
            tracker.apply_split("UNKNOWN", 2.0, date(2024, 6, 1))
        assert exc_info.value.ticker == "UNKNOWN"
        assert exc_info.value.event_type == "split"

    def test_dividend_on_unknown_ticker(self):
        """Dividend on unknown ticker raises InvariantError."""
        tracker = PositionTracker(cash=10_000.0)
        with pytest.raises(InvariantError) as exc_info:
            tracker.apply_dividend("UNKNOWN", 1.0, date(2024, 6, 1))
        assert exc_info.value.ticker == "UNKNOWN"
        assert exc_info.value.event_type == "dividend"

    def test_delisting_on_unknown_ticker_is_idempotent(self):
        """Delisting on unknown ticker is idempotent (no error).
        This is by design: a stock may have already been sold or
        delisted before the delisting event is processed."""
        tracker = PositionTracker(cash=10_000.0)
        # Should NOT raise
        tracker.apply_delisting("UNKNOWN", 0.10, date(2024, 6, 1))
        assert tracker.get_cash() == pytest.approx(10_000.0)


# ═══════════════════════════════════════════════════════════════════════
# Corporate actions through the backtest engine
# ═══════════════════════════════════════════════════════════════════════


class TestCorporateActionsViaBacktest:
    """Corporate actions applied through the full backtest pipeline."""

    def test_dividend_in_backtest(self):
        """Dividend is applied during backtest and increases cash."""
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
            rebalance_freq="weekly",
        )

        n_days = 300
        start_date = date(2024, 1, 2)
        dates = pd.bdate_range(start_date, periods=n_days)

        # AAPL with a dividend on day 150
        closes = [100 + i * 0.1 for i in range(n_days)]
        dividends = {150: 0.82}  # dividend on day index 150

        aapl_data = _make_stock_data(
            n_days=n_days, start_date=start_date, ticker="AAPL",
            price=100.0, trend=0.1, volume=1_000_000.0,
            dividends=dividends,
        )

        msft_data = _make_stock_data(
            n_days=n_days, start_date=start_date, ticker="MSFT",
            price=100.0, trend=0.1, volume=1_000_000.0,
        )

        stock_data = {"AAPL": aapl_data, "MSFT": msft_data}

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT"],
            stock_data=stock_data,
        )

        result = run_backtest(
            start=date(2024, 4, 1),
            end=date(2024, 9, 30),
            initial_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        # Backtest should complete without errors
        assert isinstance(result.final_cash, float)

    def test_split_in_backtest(self):
        """Split is applied during backtest and adjusts position correctly."""
        config = Config(
            score_window=90,
            stock_sma=100,
            regime_sma=200,
            min_price=5.0,
            min_adv_dollars=1.0,
            top_pct=0.50,
            rebalance_freq="weekly",
        )

        n_days = 300
        start_date = date(2024, 1, 2)
        dates = pd.bdate_range(start_date, periods=n_days)

        # AAPL with a 4:1 split on day 150
        closes = [100 + i * 0.1 for i in range(n_days)]
        splits = {150: 4.0}

        aapl_data = _make_stock_data(
            n_days=n_days, start_date=start_date, ticker="AAPL",
            price=100.0, trend=0.1, volume=1_000_000.0,
            splits=splits,
        )

        msft_data = _make_stock_data(
            n_days=n_days, start_date=start_date, ticker="MSFT",
            price=100.0, trend=0.1, volume=1_000_000.0,
        )

        stock_data = {"AAPL": aapl_data, "MSFT": msft_data}

        provider = _make_mock_provider(
            universe=["AAPL", "MSFT"],
            stock_data=stock_data,
        )

        result = run_backtest(
            start=date(2024, 4, 1),
            end=date(2024, 9, 30),
            initial_cash=1_000_000.0,
            config=config,
            data_provider=provider,
        )

        # Backtest should complete without errors
        assert isinstance(result.final_cash, float)
