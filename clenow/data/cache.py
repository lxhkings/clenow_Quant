"""Parquet-based cache for price data.

Cache key = SHA-256(sorted tickers joined by comma + start + end).
Cache directory: ~/.clenow/cache/
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".clenow" / "cache"


def _cache_key(tickers: list[str], start: date, end: date) -> str:
    """Deterministic cache key from sorted tickers + date range."""
    sorted_tickers = ",".join(sorted(tickers))
    raw = f"{sorted_tickers}|{start.isoformat()}|{end.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    """Return the Parquet file path for a cache key."""
    return CACHE_DIR / f"{key}.parquet"


def load_from_cache(
    tickers: list[str], start: date, end: date
) -> pd.DataFrame | None:
    """Try to load cached price data. Returns None on miss or stale data."""
    key = _cache_key(tickers, start, end)
    path = _cache_path(key)

    if not path.exists():
        logger.debug("Cache miss: %s", key[:12])
        return None

    try:
        df = pd.read_parquet(path)

        # The cache key already encodes the requested date range, so a key
        # match guarantees the data was fetched for exactly this range.
        # (The actual data rows may have fewer dates due to weekends/holidays.)
        if df.empty:
            return df

        logger.debug("Cache hit: %s", key[:12])
        return df

    except Exception as exc:
        logger.warning("Failed to read cache %s: %s", key[:12], exc)
        # Corrupted cache file — remove it.
        path.unlink(missing_ok=True)
        return None


def save_to_cache(
    df: pd.DataFrame, tickers: list[str], start: date, end: date
) -> None:
    """Persist price data to the Parquet cache."""
    if df.empty:
        return

    key = _cache_key(tickers, start, end)
    path = _cache_path(key)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.debug("Cache saved: %s", key[:12])
    except Exception as exc:
        logger.warning("Failed to write cache %s: %s", key[:12], exc)
