#!/usr/bin/env python3
"""Database validation script — the project's trust anchor.

Outputs 6 facts about the database to verify data integrity before backtesting.
Each check produces a PASS / WARN / FAIL verdict.

Usage:
    python scripts/validate_db.py [--db PATH]

If the database is missing, reports what cannot be verified and exits 1.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default search paths for the SQLite database (first match wins)
_DEFAULT_DB_PATHS = [
    Path("data/clenow.db"),
    Path("data/db.sqlite"),
    Path("clenow.db"),
]

# Split-stock checks: (ticker, pre_date, split_date, expected_ratio)
_SPLIT_CHECKS = [
    ("AAPL", date(2020, 8, 28), date(2020, 8, 31), 4.0),   # 4:1 split
    ("NVDA", date(2024, 6, 7), date(2024, 6, 10), 10.0),   # 10:1 split
    ("TSLA", date(2022, 8, 24), date(2022, 8, 25), 3.0),   # 3:1 split
]

# Delisted / acquired stock checks: (ticker, expected_before_date)
_DELISTED_CHECKS = [
    ("LEH",  date(2008, 9, 15)),   # Lehman Brothers bankruptcy
    ("ENRN", date(2001, 12, 2)),   # Enron bankruptcy
    ("WB",   date(2008, 12, 31)),  # Wachovia acquired by Wells Fargo
]

# Large-cap tickers for liquidity check
_LARGE_CAPS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Small-cap tickers for liquidity check
_SMALL_CAPS = ["DISH", "IVZ", "VNO", "PBCT", "FRC"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Verdict:
    """Accumulates PASS / WARN / FAIL results."""
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []  # (check, status, detail)

    def pass_(self, check: str, detail: str = "") -> None:
        self.results.append((check, "PASS", detail))

    def warn(self, check: str, detail: str = "") -> None:
        self.results.append((check, "WARN", detail))

    def fail(self, check: str, detail: str = "") -> None:
        self.results.append((check, "FAIL", detail))

    def skip(self, check: str, detail: str = "") -> None:
        self.results.append((check, "SKIP", detail))

    @property
    def has_critical_failure(self) -> bool:
        return any(status == "FAIL" for _, status, _ in self.results)

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print("DATABASE VALIDATION REPORT")
        print("=" * 70)
        for check, status, detail in self.results:
            tag = f"[{status}]"
            line = f"  {tag:<8} {check}"
            if detail:
                line += f"\n           {detail}"
            print(line)
        print("=" * 70)
        pass_count = sum(1 for _, s, _ in self.results if s == "PASS")
        warn_count = sum(1 for _, s, _ in self.results if s == "WARN")
        fail_count = sum(1 for _, s, _ in self.results if s == "FAIL")
        skip_count = sum(1 for _, s, _ in self.results if s == "SKIP")
        total = len(self.results)
        print(f"  Totals: {pass_count} PASS, {warn_count} WARN, "
              f"{fail_count} FAIL, {skip_count} SKIP  (of {total} checks)")
        print("=" * 70 + "\n")


def _query(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run a query and return rows as list of dicts."""
    try:
        cur = conn.execute(sql, params)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]
    except sqlite3.OperationalError as exc:
        # Table or column missing — return empty so caller can handle gracefully
        return []


def _query_one(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    """Run a query and return the first row as a dict, or None."""
    rows = _query(conn, sql, params)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Check 1: PIT completeness
# ---------------------------------------------------------------------------

def check_pit_completeness(conn: sqlite3.Connection, v: Verdict) -> None:
    """How many distinct as_of_date values exist for SP500 constituents?"""
    row = _query_one(
        conn,
        "SELECT COUNT(DISTINCT as_of_date) AS n_dates FROM index_constituents WHERE index_id = 'SP500'",
    )
    if row is None:
        v.fail("PIT completeness", "index_constituents table not found or empty")
        return

    n = row["n_dates"]
    if n <= 2:
        v.fail(
            "PIT completeness",
            f"Only {n} distinct as_of_date value(s) — DB is NOT point-in-time. "
            "Severe survivorship bias. Must disclose prominently.",
        )
    elif n <= 12:
        v.warn(
            "PIT completeness",
            f"{n} distinct as_of_date values — sparse PIT coverage. "
            "Backtests may have mild survivorship bias.",
        )
    else:
        v.pass_("PIT completeness", f"{n} distinct as_of_date values — PIT basically usable")


# ---------------------------------------------------------------------------
# Check 2: Price adjustment status
# ---------------------------------------------------------------------------

def check_price_adjustment(conn: sqlite3.Connection, v: Verdict) -> None:
    """Check known split stocks: is adj_close smooth across the split?"""
    adjusted_count = 0
    unadjusted_count = 0
    details: list[str] = []

    for ticker, pre_date, split_date, expected_ratio in _SPLIT_CHECKS:
        rows = _query(
            conn,
            "SELECT date, raw_close, adj_close FROM prices "
            "WHERE ticker = ? AND date IN (?, ?) ORDER BY date",
            (ticker, pre_date.isoformat(), split_date.isoformat()),
        )
        if len(rows) < 2:
            details.append(f"{ticker}: insufficient data ({len(rows)} rows)")
            continue

        pre_row = next((r for r in rows if r["date"] == pre_date.isoformat()), None)
        split_row = next((r for r in rows if r["date"] == split_date.isoformat()), None)
        if pre_row is None or split_row is None:
            details.append(f"{ticker}: missing pre/split date row")
            continue

        # Prefer adj_close if available, fall back to raw_close
        pre_close = pre_row["adj_close"] if pre_row["adj_close"] else pre_row["raw_close"]
        split_close = split_row["adj_close"] if split_row["adj_close"] else split_row["raw_close"]

        if pre_close is None or split_close is None or pre_close == 0:
            details.append(f"{ticker}: null/zero close values")
            continue

        pct_change = abs(split_close - pre_close) / pre_close * 100

        if pct_change < 5:
            adjusted_count += 1
            details.append(f"{ticker}: smooth ({pct_change:.1f}% change) — adjusted")
        else:
            unadjusted_count += 1
            details.append(
                f"{ticker}: {pct_change:.1f}% jump — NOT adjusted "
                f"(expected ~{(expected_ratio - 1) * 100:.0f}% for {expected_ratio:.0f}:1 split)"
            )

    if not details:
        v.warn("Price adjustment", "No split-stock data found; cannot verify")
        return

    detail = "; ".join(details)
    if unadjusted_count > 0:
        v.warn("Price adjustment", f"{unadjusted_count} unadjusted, {adjusted_count} adjusted. {detail}")
    else:
        v.pass_("Price adjustment", f"All {adjusted_count} checks smooth. {detail}")


# ---------------------------------------------------------------------------
# Check 3: Delisted / acquired stock coverage
# ---------------------------------------------------------------------------

def check_delisted_coverage(conn: sqlite3.Connection, v: Verdict) -> None:
    """Check that delisted/acquired stocks have historical price data."""
    details: list[str] = []
    missing_count = 0

    for ticker, expected_before in _DELISTED_CHECKS:
        row = _query_one(
            conn,
            "SELECT COUNT(*) AS n_rows, MIN(date) AS min_date, MAX(date) AS max_date "
            "FROM prices WHERE ticker = ?",
            (ticker,),
        )
        if row is None or row["n_rows"] == 0:
            missing_count += 1
            details.append(f"{ticker}: completely missing — survivorship bias in price history")
        else:
            details.append(f"{ticker}: {row['n_rows']} rows, {row['min_date']} to {row['max_date']}")

    detail = "; ".join(details)
    if missing_count == len(_DELISTED_CHECKS):
        v.fail("Delisted stock coverage", f"All {missing_count} delisted stocks missing. {detail}")
    elif missing_count > 0:
        v.warn("Delisted stock coverage", f"{missing_count} of {len(_DELISTED_CHECKS)} missing. {detail}")
    else:
        v.pass_("Delisted stock coverage", f"All {len(_DELISTED_CHECKS)} found. {detail}")


# ---------------------------------------------------------------------------
# Check 4: SP500 index history range
# ---------------------------------------------------------------------------

def check_index_history(conn: sqlite3.Connection, v: Verdict) -> None:
    """Verify SP500 index price range and continuity."""
    row = _query_one(
        conn,
        "SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS n_rows "
        "FROM index_prices WHERE index_id = 'SP500'",
    )
    if row is None or row["n_rows"] == 0:
        v.fail("SP500 index history", "index_prices table not found or empty for SP500")
        return

    min_d = row["min_date"]
    max_d = row["max_date"]
    n_rows = row["n_rows"]

    # Check for gaps > 5 trading days
    gap_rows = _query(
        conn,
        "SELECT a.date AS d1, b.date AS d2, julianday(b.date) - julianday(a.date) AS gap_days "
        "FROM index_prices a JOIN index_prices b ON a.index_id = b.index_id "
        "AND b.date > a.date AND b.index_id = 'SP500' "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM index_prices c WHERE c.index_id = 'SP500' "
        "  AND c.date > a.date AND c.date < b.date"
        ") AND julianday(b.date) - julianday(a.date) > 7 "
        "ORDER BY gap_days DESC LIMIT 5",
    )

    if gap_rows:
        gap_detail = "; ".join(
            f"{r['d1']} -> {r['d2']} ({r['gap_days']:.0f} calendar days)" for r in gap_rows
        )
        v.warn(
            "SP500 index history",
            f"{min_d} to {max_d}, {n_rows} rows. Gaps found: {gap_detail}",
        )
    else:
        v.pass_("SP500 index history", f"{min_d} to {max_d}, {n_rows} rows. No gaps > 5 trading days")


# ---------------------------------------------------------------------------
# Check 5: Liquidity data availability
# ---------------------------------------------------------------------------

def check_liquidity(conn: sqlite3.Connection, v: Verdict) -> None:
    """Spot-check large-cap and small-cap stocks for volume data."""
    # Determine a cutoff date: roughly 60 trading days (~90 calendar days) before max date
    max_row = _query_one(
        conn,
        "SELECT MAX(date) AS max_date FROM prices",
    )
    if max_row is None or max_row["max_date"] is None:
        v.warn("Liquidity data", "No price data found at all")
        return

    max_date = date.fromisoformat(max_row["max_date"])
    cutoff = (max_date - timedelta(days=120)).isoformat()

    details: list[str] = []
    issues = 0

    for label, tickers in [("large-cap", _LARGE_CAPS), ("small-cap", _SMALL_CAPS)]:
        for ticker in tickers:
            row = _query_one(
                conn,
                "SELECT AVG(volume * raw_close) AS avg_dollar_vol, "
                "  AVG(volume) AS avg_vol, COUNT(*) AS n_days "
                "FROM prices WHERE ticker = ? AND date >= ?",
                (ticker, cutoff),
            )
            if row is None or row["n_days"] == 0:
                issues += 1
                details.append(f"{ticker} ({label}): no data")
            elif row["avg_vol"] is None or row["avg_vol"] <= 0:
                issues += 1
                details.append(f"{ticker} ({label}): null/zero volume")
            else:
                dollar_vol = row["avg_dollar_vol"] or 0
                details.append(f"{ticker} ({label}): avg ${dollar_vol:,.0f}/day")

    detail = "; ".join(details)
    if issues == 0:
        v.pass_("Liquidity data", detail)
    else:
        v.warn("Liquidity data", f"{issues} issues. {detail}")


# ---------------------------------------------------------------------------
# Check 6: Clenow book appendix note (manual)
# ---------------------------------------------------------------------------

def check_clenow_appendix(v: Verdict) -> None:
    """Print a reminder about the Clenow book appendix check."""
    v.warn(
        "Clenow book appendix",
        "MANUAL CHECK: verify if 'Stocks on the Move' (2015) has reproducible "
        "score examples. If yes, use as unit test gold standard. "
        "If no, use hand-calculated fixtures.",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_db_path(arg_path: str | None) -> Path | None:
    """Determine the database path from arg, env var, or defaults."""
    if arg_path:
        p = Path(arg_path)
        if p.exists():
            return p
        return None  # explicitly given but doesn't exist

    env_path = os.environ.get("CLENOW_DB")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    for p in _DEFAULT_DB_PATHS:
        if p.exists():
            return p

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Clenow backtesting database — 6 trust-anchor checks",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database (default: auto-detect or CLENOW_DB env var)",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args.db)
    v = Verdict()

    if db_path is None:
        print("=" * 70)
        print("DATABASE NOT FOUND")
        print("=" * 70)
        searched = [args.db] if args.db else []
        env_path = os.environ.get("CLENOW_DB")
        if env_path:
            searched.append(env_path)
        searched.extend(str(p) for p in _DEFAULT_DB_PATHS)

        print(f"  Searched: {', '.join(searched)}")
        print()
        print("  To set up the database:")
        print("    1. Place a SQLite DB at data/clenow.db (or another path)")
        print("    2. Or set the CLENOW_DB environment variable")
        print("    3. Or run: python scripts/validate_db.py --db /path/to/db")
        print()
        print("  Required tables: prices, index_constituents, index_prices")
        print("=" * 70)

        # Still run the manual check
        check_clenow_appendix(v)
        v.print_report()
        return 1

    print(f"Database: {db_path}")

    # Connect
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except Exception as exc:
        print(f"ERROR: Failed to open database: {exc}")
        return 1

    try:
        # Run all 6 checks
        check_pit_completeness(conn, v)
        check_price_adjustment(conn, v)
        check_delisted_coverage(conn, v)
        check_index_history(conn, v)
        check_liquidity(conn, v)
        check_clenow_appendix(v)
    finally:
        conn.close()

    v.print_report()
    return 1 if v.has_critical_failure else 0


if __name__ == "__main__":
    sys.exit(main())
