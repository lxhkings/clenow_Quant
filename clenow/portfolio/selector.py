"""Sequential filters that preserve rank order — regime, SMA, price, ADV."""

from __future__ import annotations

from datetime import date

import pandas as pd

from clenow.config import Config
from clenow.types import Position


def _is_bear_regime(data_provider, as_of: date, config: Config) -> bool:
    """Return True if SP500 close < its 200-day SMA (bear regime)."""
    lookback = config.regime_sma + 10  # small buffer for missing data
    start = date(as_of.year - 1, as_of.month, as_of.day)  # ~1yr lookback
    idx = data_provider.get_index_prices("SP500", start, as_of)
    if idx.empty:
        # No data → assume bull (don't block on missing data)
        return False
    close_col = "raw_close" if "raw_close" in idx.columns else idx.columns[0]
    closes = idx[close_col].sort_index()
    if len(closes) < config.regime_sma:
        return False
    sma = closes.iloc[-config.regime_sma :].mean()
    current = closes.iloc[-1]
    return current < sma


def _stock_fails_sma(
    ticker: str, data_provider, as_of: date, config: Config
) -> bool:
    """Return True if stock close < its 100-day SMA (fails filter)."""
    lookback = config.stock_sma + 10
    start = date(as_of.year - 1, as_of.month, as_of.day)
    prices = data_provider.load_prices([ticker], start, as_of)
    if prices.empty:
        return True  # no data → filter out
    # Get the ticker's data — handle both MultiIndex and single-index
    if isinstance(prices.index, pd.MultiIndex):
        ticker_data = prices.xs(ticker, level=1, drop_level=True)
    else:
        ticker_data = prices
    close_col = "raw_close"
    closes = ticker_data[close_col].sort_index()
    if len(closes) < config.stock_sma:
        return True
    sma = closes.iloc[-config.stock_sma :].mean()
    current = closes.iloc[-1]
    return current < sma


def _stock_fails_price(
    ticker: str, data_provider, as_of: date, config: Config
) -> bool:
    """Return True if raw_close < min_price threshold."""
    start = date(as_of.year - 1, as_of.month, as_of.day)
    prices = data_provider.load_prices([ticker], start, as_of)
    if prices.empty:
        return True
    if isinstance(prices.index, pd.MultiIndex):
        ticker_data = prices.xs(ticker, level=1, drop_level=True)
    else:
        ticker_data = prices
    closes = ticker_data["raw_close"].sort_index()
    if closes.empty:
        return True
    return closes.iloc[-1] < config.min_price


def _stock_fails_adv(
    ticker: str, data_provider, as_of: date, config: Config
) -> bool:
    """Return True if 20-day ADV < min_adv_dollars."""
    start = date(as_of.year - 1, as_of.month, as_of.day)
    prices = data_provider.load_prices([ticker], start, as_of)
    if prices.empty:
        return True
    if isinstance(prices.index, pd.MultiIndex):
        ticker_data = prices.xs(ticker, level=1, drop_level=True)
    else:
        ticker_data = prices
    ticker_data = ticker_data.sort_index()
    if len(ticker_data) < 20:
        return True
    recent = ticker_data.iloc[-20:]
    dollar_volume = recent["volume"] * recent["raw_close"]
    adv = dollar_volume.mean()
    return adv < config.min_adv_dollars


def apply_filters(
    ranked_tickers: list[str],
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
        data_provider: DataProvider implementation.
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

        # Stock SMA filter
        if _stock_fails_sma(ticker, data_provider, as_of, config):
            continue

        # Price filter
        if _stock_fails_price(ticker, data_provider, as_of, config):
            continue

        # ADV filter
        if _stock_fails_adv(ticker, data_provider, as_of, config):
            continue

        result.append(ticker)

    return result
