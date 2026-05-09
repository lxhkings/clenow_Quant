"""Tests for generate_report — markdown report generation."""

import os
import tempfile
from datetime import date

import pandas as pd
import pytest

from clenow.backtest.engine import BacktestResult
from clenow.config import Config
from clenow.report.main import generate_report


# ── Helpers ──────────────────────────────────────────────────────────


def _make_equity_curve(
    n_days: int = 252,
    start_value: float = 1_000_000.0,
    daily_return: float = 0.001,
) -> pd.DataFrame:
    """Build an equity curve DataFrame."""
    values = [start_value * (1 + daily_return) ** i for i in range(n_days)]
    dates = pd.bdate_range(date(2023, 1, 3), periods=n_days)
    return pd.DataFrame({
        "date": dates.date,
        "portfolio_value": values,
        "cash": [v * 0.1 for v in values],
    })


def _make_trades() -> list[dict]:
    """Create sample trade dicts."""
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
    ]


def _make_backtest_result(
    equity_curve: pd.DataFrame | None = None,
    trades: list[dict] | None = None,
) -> BacktestResult:
    """Create a BacktestResult for testing."""
    if equity_curve is None:
        equity_curve = _make_equity_curve()
    if trades is None:
        trades = _make_trades()
    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        final_positions={},
        final_cash=500_000.0,
        config=Config(),
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestReportGeneration:
    """Verify report generation produces a file with expected content."""

    def test_report_file_created(self):
        """generate_report should create a report.md file."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            assert os.path.exists(report_path)
            assert report_path.endswith("report.md")

    def test_report_file_not_empty(self):
        """Report file should not be empty."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert len(content) > 0

    def test_trade_log_exported(self):
        """Report generation should also export trade log CSV."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_report(result, tmpdir)
            trade_log_path = os.path.join(tmpdir, "trades.csv")
            assert os.path.exists(trade_log_path)


class TestReportContent:
    """Verify report contains key sections."""

    def test_contains_header(self):
        """Report should contain the title header."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "Clenow Smooth Momentum" in content

    def test_contains_performance_summary(self):
        """Report should contain a performance summary section."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "Performance Summary" in content

    def test_contains_sharpe(self):
        """Report should mention Sharpe ratio."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "Sharpe" in content

    def test_contains_equity_curve_section(self):
        """Report should contain equity curve section."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "Equity Curve" in content

    def test_contains_trade_statistics(self):
        """Report should contain trade statistics section."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "Trade Statistics" in content

    def test_contains_configuration(self):
        """Report should contain configuration section."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "Configuration" in content

    def test_contains_cagr(self):
        """Report should mention CAGR."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "CAGR" in content

    def test_contains_max_drawdown(self):
        """Report should mention max drawdown."""
        result = _make_backtest_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "Max Drawdown" in content


class TestReportWithEmptyTrades:
    """Report with no trades."""

    def test_no_trades_report(self):
        """Report with no trades should still generate successfully."""
        result = _make_backtest_result(trades=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            with open(report_path) as f:
                content = f.read()
            assert "No closed trades" in content


class TestReportWithEmptyEquity:
    """Report with empty equity curve."""

    def test_empty_equity_report(self):
        """Report with empty equity curve should generate without error."""
        eq = pd.DataFrame(columns=["date", "portfolio_value", "cash"])
        result = _make_backtest_result(equity_curve=eq)
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = generate_report(result, tmpdir)
            assert os.path.exists(report_path)
            with open(report_path) as f:
                content = f.read()
            assert "No equity curve data" in content
