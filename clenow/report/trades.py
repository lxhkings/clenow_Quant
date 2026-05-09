"""Trade log export to CSV.

Exports the trade list from a backtest result to a CSV file with
standardized columns including PnL percentage and holding period.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def export_trade_log(trades: list[dict], output_path: str) -> None:
    """Export trades to a CSV file.

    Output columns: entry_date, exit_date, ticker, side, shares,
    entry_price, exit_price, pnl, pnl_pct, holding_days

    Args:
        trades: List of trade dicts with fields: entry_date, exit_date,
            ticker, shares, entry_price, exit_price, pnl
        output_path: Path to write the CSV file
    """
    if not trades:
        # Write empty CSV with headers
        empty_df = pd.DataFrame(columns=[
            "entry_date", "exit_date", "ticker", "side", "shares",
            "entry_price", "exit_price", "pnl", "pnl_pct", "holding_days",
        ])
        empty_df.to_csv(output_path, index=False)
        return

    rows = []
    for t in trades:
        entry_price = t.get("entry_price", 0.0)
        exit_price = t.get("exit_price", 0.0)
        pnl = t.get("pnl", 0.0)
        shares = t.get("shares", 0)

        # PnL percentage
        if entry_price != 0:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = 0.0

        # Holding days
        entry_date = t.get("entry_date")
        exit_date = t.get("exit_date")
        holding_days = _compute_holding_days(entry_date, exit_date)

        # Side: infer from pnl direction or default to "long"
        side = t.get("side", "long")

        rows.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "ticker": t.get("ticker", ""),
            "side": side,
            "shares": shares,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
        })

    df = pd.DataFrame(rows, columns=[
        "entry_date", "exit_date", "ticker", "side", "shares",
        "entry_price", "exit_price", "pnl", "pnl_pct", "holding_days",
    ])

    df.to_csv(output_path, index=False)


def _compute_holding_days(entry_date, exit_date) -> int:
    """Compute the number of calendar days between entry and exit."""
    if entry_date is None or exit_date is None:
        return 0

    entry = _to_date(entry_date)
    exit_d = _to_date(exit_date)

    if entry is None or exit_d is None:
        return 0

    return (exit_d - entry).days


def _to_date(val):
    """Convert a value to a date object."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, pd.Timestamp):
        return val.date()
    try:
        return pd.Timestamp(val).date()
    except Exception:
        return None
