"""A股 price limit resolver.

Rules (2026):
  ST/*ST/PT/退: 5% (overrides board)
  北证 .BJ:    30%
  科创板 688/689 .SH: 20%
  创业板 300/301 .SZ: 20%
  主板 600/601/603/605 .SH, 000/001/002/003 .SZ: 10%
  Unknown: 10% (conservative default)
"""

from __future__ import annotations

import re

_ST_PATTERN = re.compile(r"(\*?ST|PT|退)\s*", re.IGNORECASE)


def cn_price_limit_resolver(ticker: str, name: str) -> float:
    """Returns max single-day price move as fraction (e.g. 0.10 for 10%)."""
    if _ST_PATTERN.search(name or ""):
        return 0.05

    if ticker.endswith(".BJ"):
        return 0.30

    if ticker.endswith(".SH"):
        code = ticker.removesuffix(".SH")
        if code.startswith(("688", "689")):
            return 0.20
        return 0.10

    if ticker.endswith(".SZ"):
        code = ticker.removesuffix(".SZ")
        if code.startswith(("300", "301")):
            return 0.20
        return 0.10

    return 0.10
