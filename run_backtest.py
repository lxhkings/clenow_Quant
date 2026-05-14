"""Run Clenow Smooth Momentum backtest."""

import argparse
from datetime import date
from pathlib import Path

from clenow.backtest.engine import run_backtest
from clenow.config import Config
from clenow.data.sectors import load_sector_mapping
from clenow.data.synology import SynologyDataProvider
from clenow.markets import get_profile
from clenow.report.main import generate_report
from clenow.report.watchlist import build_watchlist, render_csv, render_markdown
from clenow.strategy import make_strategy


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
    parser.add_argument("--watchlist-only", action="store_true",
                        help="Generate daily watchlist Markdown instead of full backtest.")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None,
                        help="Date for --watchlist-only (defaults to --end).")
    parser.add_argument("--universe", default="index",
                        choices=["index", "all"],
                        help="Universe source: index (constituents) or all (full market, CN only)")
    args = parser.parse_args(argv)

    if args.watchlist_only:
        if args.strategy != "clenow_momentum":
            raise SystemExit(
                "--watchlist-only currently supports only --strategy clenow_momentum"
            )
        if args.universe == "all" and args.market != "cn":
            raise SystemExit("--universe all only supported for --market cn")
        as_of = args.as_of or args.end
        profile = get_profile(args.market)
        config = Config(
            market=args.market.upper(),
            risk_factor=args.risk_factor,
            rebalance_freq="weekly",
        )
        dp = SynologyDataProvider()
        try:
            # Universe selection
            if args.universe == "all":
                universe = dp.get_all_cn_tickers(as_of)
            else:
                universe = dp.get_universe(as_of, index_id=profile.universe_index_id)
            sector_map = load_sector_mapping(profile.universe_index_id)
            rows = build_watchlist(as_of, config, profile, dp, sector_map, universe=universe)
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.universe == "all":
                # CSV output for full market screening
                csv_content = render_csv(rows)
                out_path = out_dir / "watchlist.csv"
                out_path.write_text(csv_content, encoding="utf-8")
                print(f"Watchlist CSV: {out_path} ({len(rows)} rows from {len(universe)} tickers)")
            else:
                # Markdown output for index constituents
                md = render_markdown(rows, as_of, profile, config, total_universe=len(universe))
                out_path = out_dir / "watchlist.md"
                out_path.write_text(md, encoding="utf-8")
                print(f"Watchlist: {out_path}")
        finally:
            dp.close()
        return

    profile = get_profile(args.market)
    capital = args.capital if args.capital is not None else profile.default_starting_capital
    config = Config(market=args.market.upper(), risk_factor=args.risk_factor, rebalance_freq="weekly")

    dp = SynologyDataProvider()

    strat_kwargs = {}
    if args.strategy == "sector_rotation":
        strat_kwargs["sector_mapping"] = load_sector_mapping(profile.universe_index_id)
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
