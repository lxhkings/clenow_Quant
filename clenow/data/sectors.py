"""Sector mapping from index_constituents (latest snapshot)."""
from __future__ import annotations

import pymysql

from clenow.data.synology import DEFAULT_DB_CONFIG


def load_sector_mapping(
    index_id: str,
    db_config: dict | None = None,
) -> dict[str, str]:
    """Load ticker -> sector mapping from latest snapshot of index_constituents.

    Args:
        index_id: Index identifier (e.g. "SP500", "CSI800", "HSI").
        db_config: Optional pymysql config; defaults to DEFAULT_DB_CONFIG.

    Returns:
        Dict mapping ticker to sector. Empty dict if no rows found.
    """
    cfg = db_config or DEFAULT_DB_CONFIG
    conn = pymysql.connect(**cfg)
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


def load_all_cn_sector_mapping(db_config: dict | None = None) -> dict[str, str]:
    """Load ticker -> sector mapping for all CN stocks from stocks.gics_sector.

    Args:
        db_config: Optional pymysql config; defaults to DEFAULT_DB_CONFIG.

    Returns:
        Dict mapping ticker to sector. Empty dict if no rows found.
    """
    cfg = db_config or DEFAULT_DB_CONFIG
    conn = pymysql.connect(**cfg)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT ticker, gics_sector FROM stocks "
            "WHERE (ticker LIKE '%.SH' OR ticker LIKE '%.SZ') "
            "AND gics_sector IS NOT NULL AND gics_sector != ''"
        )
        return dict(cur.fetchall())
    finally:
        conn.close()