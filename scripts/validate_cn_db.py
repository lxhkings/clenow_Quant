#!/usr/bin/env python3
"""CN A-share database sanity check — read-only, 6 questions.

Outputs PASS / WARN / FAIL for each. Critical FAIL exits 1.
"""
from __future__ import annotations
import sys
from clenow.data.synology import _connect, DEFAULT_DB_CONFIG


def _query_one(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        cols = [d[0] for d in cur.description] if cur.description else []
    return dict(zip(cols, row)) if row else None


def _query_all(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
    return [dict(zip(cols, r)) for r in rows]


def main() -> int:
    conn = _connect(DEFAULT_DB_CONFIG)
    fails = 0

    # 1. A-share ticker count
    row = _query_one(conn, "SELECT COUNT(DISTINCT ticker) AS n FROM prices "
                            "WHERE ticker REGEXP '\\.(SH|SZ)$'")
    n = row["n"] if row else 0
    print(f"[{'PASS' if n > 1000 else 'FAIL'}] A-share tickers in prices: {n} (need > 1000)")
    if n <= 1000:
        fails += 1

    # 2. Earliest A-share data
    row = _query_one(conn, "SELECT MIN(date) AS d FROM prices WHERE ticker='600519.SH'")
    earliest = row["d"] if row else None
    print(f"[{'PASS' if earliest and str(earliest) <= '2010-01-01' else 'WARN'}] "
          f"600519.SH earliest date: {earliest}")

    # 3. CSI800 PIT
    row = _query_one(conn, "SELECT COUNT(*) AS n_rows, COUNT(DISTINCT snapshot_date) AS n_dates "
                            "FROM index_constituents WHERE index_id='CSI800'")
    n_rows = row["n_rows"] if row else 0
    n_dates = row["n_dates"] if row else 0
    print(f"[{'PASS' if n_dates >= 12 else 'FAIL'}] CSI800 index_constituents: "
          f"{n_rows} rows, {n_dates} snapshots (need >= 12)")
    if n_dates < 12:
        fails += 1

    # 4. CSI800 index prices
    row = _query_one(conn, "SELECT COUNT(*) AS n, MIN(date) AS d_min, MAX(date) AS d_max "
                            "FROM index_prices WHERE index_id='CSI800'")
    n = row["n"] if row else 0
    print(f"[{'PASS' if n > 1000 else 'FAIL'}] CSI800 index_prices: "
          f"{n} rows ({row['d_min'] if row else '?'} → {row['d_max'] if row else '?'})")
    if n <= 1000:
        fails += 1

    # 5. stocks.name column
    rows = _query_all(conn, "SHOW COLUMNS FROM stocks LIKE 'name'")
    has_name = len(rows) > 0
    print(f"[{'PASS' if has_name else 'FAIL'}] stocks.name column present")
    if not has_name:
        fails += 1

    # 6. Price data continuity check (hfq-adjusted data)
    # Note: CN prices.close is already hfq-adjusted, skip jump detection
    rows = _query_all(
        conn,
        "SELECT COUNT(*) AS n FROM prices "
        "WHERE ticker='600519.SH' AND date BETWEEN '2015-01-01' AND '2015-12-31'",
    )
    n_days = rows[0]["n"] if rows else 0
    print(f"[{'PASS' if n_days > 200 else 'WARN'}] 600519.SH 2015 coverage: {n_days} days")

    conn.close()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())