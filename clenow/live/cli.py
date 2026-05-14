"""Daily Decision Workflow — single-point entry for live trading decisions.

Usage:
    python -m clenow.live --as-of 2026-05-08 --equity 100000 --output orders.csv

Internal flow:
  1. as_of = parsed date (Friday)
  2. Load price data for [as_of - 250 days, as_of] for SP500 PIT universe
  3. Run signals/portfolio flow (SAME compute_target_portfolio as backtest)
  4. Read current positions from ~/.clenow/positions.json
  5. Compare current vs target, generate diff orders
  6. Output orders.csv: ticker, side, shares, expected_price, reason
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from clenow.backtest.engine import compute_target_portfolio
from clenow.config import Config
from clenow.data.loader import SQLDataProvider
from clenow.live.state import load_state
from clenow.markets import MarketProfile, get_profile
from clenow.strategy import make_strategy
from clenow.types import Position


def generate_orders(
    as_of: date,
    equity: float,
    config: Config,
    data_provider: SQLDataProvider,
    state_path: Path | None = None,
    profile: MarketProfile | None = None,
    strategy=None,
) -> list[dict]:
    """Generate order list for the given date.

    Returns list of dicts with keys: ticker, side, shares, expected_price, reason.
    """
    if strategy is None:
        strategy = make_strategy("clenow_momentum")
    tracker = load_state(state_path)
    current_positions = tracker.get_positions()

    # If equity is specified, override cash (for fresh start or capital adjustment)
    if equity > 0:
        position_value = sum(
            pos.shares * pos.entry_price for pos in current_positions.values()
        )
        current_cash = max(0.0, equity - position_value)
    else:
        current_cash = tracker.get_cash()

    # Run the same compute_target_portfolio as backtest
    target = compute_target_portfolio(
        as_of=as_of,
        current_positions=current_positions,
        current_cash=current_cash,
        config=config,
        data_provider=data_provider,
        strategy=strategy,
        profile=profile,
    )

    # Load current prices for expected_price column
    all_tickers = set(current_positions.keys()) | set(target.positions.keys())
    current_prices = _load_current_prices(all_tickers, as_of, data_provider)

    # Generate diff orders
    orders: list[dict] = []

    # Sells: in current but not in target, or target shares < current shares
    for ticker, pos in current_positions.items():
        target_shares = target.positions.get(ticker, 0)
        if target_shares < pos.shares:
            sell_shares = pos.shares - target_shares
            reason = _sell_reason(
                ticker, target_shares == 0, as_of, data_provider, config, profile, strategy,
            )
            orders.append({
                "ticker": ticker,
                "side": "sell",
                "shares": sell_shares,
                "expected_price": current_prices.get(ticker, 0.0),
                "reason": reason,
            })

    # Buys: in target but not in current, or target shares > current shares
    for ticker, target_shares in target.positions.items():
        current_shares = (
            current_positions[ticker].shares if ticker in current_positions else 0
        )
        if target_shares > current_shares:
            buy_shares = target_shares - current_shares
            orders.append({
                "ticker": ticker,
                "side": "buy",
                "shares": buy_shares,
                "expected_price": current_prices.get(ticker, 0.0),
                "reason": (
                    "new entry" if ticker not in current_positions
                    else "increase position"
                ),
            })

    return orders


def _load_current_prices(
    tickers: set[str], as_of: date, data_provider: SQLDataProvider
) -> dict[str, float]:
    """Load the most recent close prices for a set of tickers."""
    if not tickers:
        return {}

    price_start = as_of - timedelta(days=5)
    prices_df = data_provider.load_prices(list(tickers), price_start, as_of)

    current_prices: dict[str, float] = {}
    if prices_df.empty:
        return current_prices

    for ticker in tickers:
        try:
            if isinstance(prices_df.index, pd.MultiIndex):
                td = prices_df.xs(ticker, level=1, drop_level=True)
            else:
                td = prices_df
            rc = td["raw_close"].dropna()
            if len(rc) > 0:
                current_prices[ticker] = float(rc.iloc[-1])
        except (KeyError, TypeError):
            pass

    return current_prices


def _sell_reason(
    ticker: str, full_exit: bool, as_of: date, data_provider,
    config: Config, profile: MarketProfile, strategy,
) -> str:
    """Determine reason for selling a position."""
    if not full_exit:
        return "reduce position"
    # Build minimal all_prices for strategy.exit_signal
    lookback = as_of - timedelta(days=450)
    all_prices = data_provider.load_prices([ticker], lookback, as_of)
    if strategy.exit_signal(ticker, all_prices, as_of, data_provider, profile, config):
        return "SMA break"
    return "fell out of top N"


def write_orders_csv(orders: list[dict], output_path: str | Path) -> None:
    """Write orders to CSV file."""
    fieldnames = ["ticker", "side", "shares", "expected_price", "reason"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(orders)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for daily decision workflow."""
    parser = argparse.ArgumentParser(
        description="Clenow Daily Decision Workflow — generate Monday order list"
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        required=True,
        help="Signal date (Friday close), YYYY-MM-DD format",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=0.0,
        help="Total equity override (for fresh start or capital adjustment)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="orders.csv",
        help="Output CSV path (default: orders.csv)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (SQLite file)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="State file path (default: ~/.clenow/positions.json)",
    )
    parser.add_argument(
        "--market",
        default="us",
        choices=["us", "cn", "hk"],
        help="Market (default: us)",
    )
    parser.add_argument(
        "--strategy",
        default="clenow_momentum",
        choices=["clenow_momentum", "sector_rotation"],
        help="Strategy to use (default: clenow_momentum)",
    )

    args = parser.parse_args(argv)

    profile = get_profile(args.market)
    config = Config(market=args.market.upper())

    if args.db:
        data_provider = SQLDataProvider(args.db)
    else:
        default_db = Path.home() / ".clenow" / "sp500.db"
        if default_db.exists():
            data_provider = SQLDataProvider(str(default_db))
        else:
            print("Error: No database specified. Use --db PATH", file=sys.stderr)
            print(f"  Looked for: {default_db}", file=sys.stderr)
            sys.exit(1)

    state_path = Path(args.state) if args.state else None

    # Resolve strategy
    strategy = make_strategy(args.strategy)

    print(f"Computing orders for {args.as_of} (market={args.market}, strategy={args.strategy}) ...")
    orders = generate_orders(
        as_of=args.as_of,
        equity=args.equity,
        config=config,
        data_provider=data_provider,
        state_path=state_path,
        profile=profile,
        strategy=strategy,
    )

    write_orders_csv(orders, args.output)

    buys = [o for o in orders if o["side"] == "buy"]
    sells = [o for o in orders if o["side"] == "sell"]
    print(f"Generated {len(orders)} orders: {len(buys)} buys, {len(sells)} sells")
    print(f"Output: {args.output}")

    data_provider.close()
