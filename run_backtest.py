"""Run Clenow Smooth Momentum backtest."""

import argparse
from datetime import date

from clenow.backtest.engine import run_backtest
from clenow.config import Config
from clenow.data.synology import SynologyDataProvider
from clenow.markets import get_profile
from clenow.report.main import generate_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Clenow Smooth Momentum backtest")
    parser.add_argument(
        "--market",
        default="us",
        choices=["us", "cn", "hk"],
        help="Market to backtest (default: us)",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(2010, 1, 1),
        help="Backtest start date (default: 2010-01-01)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date(2026, 5, 8),
        help="Backtest end date (default: 2026-05-08)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Starting capital (default: market profile default)",
    )
    parser.add_argument(
        "--risk-factor",
        type=float,
        default=0.005,
        help="Risk factor for position sizing (default: 0.005)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for report (default: ./output)",
    )

    args = parser.parse_args(argv)

    profile = get_profile(args.market)
    capital = args.capital if args.capital is not None else profile.default_starting_capital
    config = Config(market=args.market.upper(), risk_factor=args.risk_factor)

    dp = SynologyDataProvider()

    result = run_backtest(
        start=args.start,
        end=args.end,
        initial_cash=capital,
        config=config,
        data_provider=dp,
        profile=profile,
    )

    report_path = generate_report(result, args.output)
    print(f"Report: {report_path}")

    dp.close()


if __name__ == "__main__":
    main()
