"""Run Clenow Smooth Momentum backtest."""

import argparse
from datetime import date

from clenow.backtest.engine import run_backtest
from clenow.config import Config
from clenow.data.synology import SynologyDataProvider
from clenow.markets import get_profile
from clenow.report.main import generate_report
from clenow.strategy import make_strategy


def _get_sector_mapping(index_id: str) -> dict[str, str]:
    import pymysql
    from clenow.data.synology import DEFAULT_DB_CONFIG
    conn = pymysql.connect(**DEFAULT_DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT ticker, sector FROM index_constituents "
            "WHERE index_id = %s AND sector IS NOT NULL AND sector != '' "
            "AND snapshot_date = ("
            "  SELECT MAX(snapshot_date) FROM index_constituents "
            "  WHERE index_id = %s AND sector IS NOT NULL AND sector != ''"
            ")",
            (index_id, index_id),
        )
        return dict(cur.fetchall())
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Clenow Smooth Momentum backtest")
    parser.add_argument("--market", default="us", choices=["us", "cn", "hk"])
    parser.add_argument("--strategy", default="clenow_momentum",
                        choices=["clenow_momentum", "sector_rotation"])
    parser.add_argument("--rebalance", default="weekly",
                        choices=["daily", "weekly", "biweekly"])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 5, 8))
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--risk-factor", type=float, default=0.005)
    parser.add_argument("--max-position-pct", type=float, default=0.05)
    parser.add_argument("--top-pct", type=float, default=0.20)
    parser.add_argument("--output", type=str, default="./output")
    args = parser.parse_args(argv)

    profile = get_profile(args.market)
    capital = args.capital if args.capital is not None else profile.default_starting_capital
    config = Config(
        market=args.market.upper(),
        risk_factor=args.risk_factor,
        max_position_pct=args.max_position_pct,
        top_pct=args.top_pct,
        rebalance_freq=args.rebalance,
    )

    dp = SynologyDataProvider()

    strat_kwargs = {}
    if args.strategy == "sector_rotation":
        strat_kwargs["sector_mapping"] = _get_sector_mapping(profile.universe_index_id)
        if not strat_kwargs["sector_mapping"]:
            raise SystemExit(
                f"sector_rotation requires sector data in index_constituents for {profile.universe_index_id}; none found"
            )
    strategy = make_strategy(args.strategy, **strat_kwargs)

    result = run_backtest(
        start=args.start, end=args.end, initial_cash=capital,
        config=config, data_provider=dp, strategy=strategy, profile=profile,
    )

    report_path = generate_report(result, args.output)
    print(f"Report: {report_path}")
    dp.close()


if __name__ == "__main__":
    main()