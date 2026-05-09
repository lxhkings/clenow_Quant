"""Sequential filters that preserve rank order — regime, SMA, price, ADV."""

from __future__ import annotations

from datetime import date

import pandas as pd

from clenow.config import Config
from clenow.data.utils import get_ticker_series
from clenow.types import Position


def _is_bear_regime(data_provider, as_of: date, config: Config) -> bool:
    """Return True if SP500 close < its 200-day SMA (bear regime)."""
    lookback = config.regime_sma + 10  # small buffer for missing data
    start = date(as_of.year - 1, as_of.month, as_of.day)  # ~1yr lookback
    idx = data_provider.get_index_prices("SP500", start, as_of)
    if idx.empty:
        # No data → assume bull (don't block on missing data)
        return False
    # Handle different column names: raw_close, close, or first numeric column
    if "raw_close" in idx.columns:
        close_col = "raw_close"
    elif "close" in idx.columns:
        close_col = "close"
    else:
        # Find first numeric column
        numeric_cols = idx.select_dtypes(include=["number"]).columns
        close_col = numeric_cols[0] if len(numeric_cols) > 0 else idx.columns[0]
    closes = idx[close_col].sort_index()
    if len(closes) < config.regime_sma:
        return False
    sma = closes.iloc[-config.regime_sma :].mean()
    current = closes.iloc[-1]
    return current < sma


def _stock_fails_sma(ticker_data: pd.DataFrame | None, config: Config) -> bool:
    """Return True if stock close < its 100-day SMA (fails filter)."""
    if ticker_data is None:
        return True  # no data → filter out
    closes = ticker_data["raw_close"].dropna()
    if len(closes) < config.stock_sma:
        return True
    sma = closes.iloc[-config.stock_sma :].mean()
    current = closes.iloc[-1]
    return current < sma


def _stock_fails_price(ticker_data: pd.DataFrame | None, config: Config) -> bool:
    """Return True if raw_close < min_price threshold."""
    if ticker_data is None:
        return True
    closes = ticker_data["raw_close"].dropna()
    if closes.empty:
        return True
    return closes.iloc[-1] < config.min_price


def _stock_fails_adv(ticker_data: pd.DataFrame | None, config: Config) -> bool:
    """Return True if 20-day ADV < min_adv_dollars."""
    if ticker_data is None or len(ticker_data) < 20:
        return True
    recent = ticker_data.iloc[-20:]
    dollar_volume = recent["volume"] * recent["raw_close"]
    return dollar_volume.mean() < config.min_adv_dollars


def apply_filters(
    ranked_tickers: list[str],
    all_prices: pd.DataFrame,
    data_provider,
    as_of: date,
    config: Config,
    current_positions: dict[str, Position] | None = None,
) -> list[str]:
    """Apply sequential filters preserving rank order.

    Filters applied in order:
      1. Regime: if SP500 < 200-SMA → block new entries (keep existing positions)
      2. Stock SMA: close < 100-day SMA → remove
      3. Price: raw_close < min_price → remove
      4. ADV: 20-day ADV < min_adv_dollars → remove

    Args:
        ranked_tickers: tickers in rank order (highest first).
        all_prices: pre-loaded DataFrame with (date, ticker) MultiIndex containing
            raw_close, raw_high, raw_low, volume columns for all universe tickers.
        data_provider: DataProvider implementation (only used for regime filter).
        as_of: the date for point-in-time filtering.
        config: system configuration.
        current_positions: existing positions (used for regime filter).

    Returns:
        Filtered list of tickers in original rank order.
    """
    if not ranked_tickers:
        return []

    existing = set(current_positions.keys()) if current_positions else set()
    bear = _is_bear_regime(data_provider, as_of, config)

    result: list[str] = []
    for ticker in ranked_tickers:
        # Regime filter: in bear regime, only keep existing positions
        if bear and ticker not in existing:
            continue

        # Extract ticker data from pre-loaded all_prices
        ticker_data = get_ticker_series(all_prices, ticker)

        # Stock SMA filter
        if _stock_fails_sma(ticker_data, config):
            continue

        # Price filter
        if _stock_fails_price(ticker_data, config):
            continue

        # ADV filter
        if _stock_fails_adv(ticker_data, config):
            continue

        result.append(ticker)

    return result