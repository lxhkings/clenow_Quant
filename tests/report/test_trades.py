"""Tests for export_trade_log — CSV trade log output."""

import os
import tempfile
from datetime import date

import pandas as pd
import pytest

from clenow.report.trades import export_trade_log


# ── Helpers ──────────────────────────────────────────────────────────


def _make_trades() -> list[dict]:
    """Create a sample trade list."""
    return [
        {
            "entry_date": date(2023, 1, 10),
            "exit_date": date(2023, 1, 20),
            "ticker": "AAPL",
            "shares": 100,
            "entry_price": 150.0,
            "exit_price": 160.0,
            "pnl": 1000.0,
        },
        {
            "entry_date": date(2023, 2, 1),
            "exit_date": date(2023, 2, 15),
            "ticker": "MSFT",
            "shares": 50,
            "entry_price": 300.0,
            "exit_price": 280.0,
            "pnl": -1000.0,
        },
        {
            "entry_date": date(2023, 3, 5),
            "exit_date": date(2023, 3, 25),
            "ticker": "GOOG",
            "shares": 200,
            "entry_price": 100.0,
            "exit_price": 120.0,
            "pnl": 4000.0,
        },
    ]


# ── Tests ────────────────────────────────────────────────────────────


class TestCSVFormat:
    """Verify CSV output format and content."""

    def test_csv_has_correct_columns(self):
        """CSV should have the expected column headers."""
        trades = _make_trades()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_trade_log(trades, path)
            df = pd.read_csv(path)
            expected_cols = [
                "entry_date", "exit_date", "ticker", "side", "shares",
                "entry_price", "exit_price", "pnl", "pnl_pct", "holding_days",
            ]
            assert list(df.columns) == expected_cols
        finally:
            os.unlink(path)

    def test_csv_has_correct_row_count(self):
        """CSV should have one row per trade."""
        trades = _make_trades()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_trade_log(trades, path)
            df = pd.read_csv(path)
            assert len(df) == len(trades)
        finally:
            os.unlink(path)

    def test_pnl_pct_computed_correctly(self):
        """PnL percentage should be (exit - entry) / entry."""
        trades = _make_trades()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_trade_log(trades, path)
            df = pd.read_csv(path)

            # AAPL: (160 - 150) / 150 = 0.0667
            assert df.iloc[0]["pnl_pct"] == pytest.approx(0.0667, rel=0.01)
            # MSFT: (280 - 300) / 300 = -0.0667
            assert df.iloc[1]["pnl_pct"] == pytest.approx(-0.0667, rel=0.01)
        finally:
            os.unlink(path)

    def test_holding_days_computed_correctly(self):
        """Holding days should be the difference between exit and entry dates."""
        trades = _make_trades()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_trade_log(trades, path)
            df = pd.read_csv(path)

            # AAPL: Jan 10 to Jan 20 = 10 days
            assert df.iloc[0]["holding_days"] == 10
            # GOOG: Mar 5 to Mar 25 = 20 days
            assert df.iloc[2]["holding_days"] == 20
        finally:
            os.unlink(path)


class TestEmptyTrades:
    """Edge case: empty trade list."""

    def test_empty_trades_produces_empty_csv_with_headers(self):
        """Empty trade list should produce CSV with headers only."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_trade_log([], path)
            df = pd.read_csv(path)
            assert len(df) == 0
            expected_cols = [
                "entry_date", "exit_date", "ticker", "side", "shares",
                "entry_price", "exit_price", "pnl", "pnl_pct", "holding_days",
            ]
            assert list(df.columns) == expected_cols
        finally:
            os.unlink(path)


class TestTradeValues:
    """Verify specific trade values in CSV output."""

    def test_ticker_values(self):
        """Ticker column should match input."""
        trades = _make_trades()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_trade_log(trades, path)
            df = pd.read_csv(path)
            assert list(df["ticker"]) == ["AAPL", "MSFT", "GOOG"]
        finally:
            os.unlink(path)

    def test_pnl_values(self):
        """PnL column should match input."""
        trades = _make_trades()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_trade_log(trades, path)
            df = pd.read_csv(path)
            assert list(df["pnl"]) == [1000.0, -1000.0, 4000.0]
        finally:
            os.unlink(path)

    def test_shares_values(self):
        """Shares column should match input."""
        trades = _make_trades()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_trade_log(trades, path)
            df = pd.read_csv(path)
            assert list(df["shares"]) == [100, 50, 200]
        finally:
            os.unlink(path)
