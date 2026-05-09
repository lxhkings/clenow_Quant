"""Run Clenow Smooth Momentum backtest."""

from datetime import date
from clenow.config import Config
from clenow.data.synology import SynologyDataProvider
from clenow.backtest.engine import run_backtest
from clenow.report.main import generate_report

dp = SynologyDataProvider()

result = run_backtest(
    start=date(2022, 1, 1),
    end=date(2026, 5, 8),
    initial_cash=100_000,
    config=Config(risk_factor=0.005),
    data_provider=dp,
)

report_path = generate_report(result, "./output")
print(f"Report: {report_path}")

dp.close()
