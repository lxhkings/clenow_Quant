"""Sequential filters that preserve rank order — regime, SMA, price, ADV."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from clenow.config import Config
from clenow.data.utils import get_ticker_series
from clenow.signals.regime import is_bear_regime
from clenow.types import Position

if TYPE_CHECKING:
    from clenow.markets.profiles import MarketProfile


def _is_bear_regime(
    data_provider,
    as_of: date,
    regime_index_id: str,
    sma_window: int,
) -> bool:
    """Bear regime: index < N-day SMA. Profile drives index_id and window."""
    start = as_of - timedelta(days=sma_window * 2)
    idx = data_provider.get_index_prices(regime_index_id, start, as_of)
    if idx.empty:
        return False
    # Handle different column names: raw_close, close, or first numeric column
    if "raw_close" in idx.columns:
        close_col = "raw_close"
    elif "close" in idx.columns:
        close_col = "close"
    else:
        numeric_cols = idx.select_dtypes(include=["number"]).columns
        close_col = numeric_cols[0] if len(numeric_cols) > 0 else idx.columns[0]
    closes = idx[close_col].sort_index()
    return is_bear_regime(closes, sma_window=sma_window)


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


def _stock_fails_price(ticker_data: pd.DataFrame | None, min_price: float) -> bool:
    """Return True if raw_close < min_price threshold."""
    if ticker_data is None:
        return True
    closes = ticker_data["raw_close"].dropna()
    if closes.empty:
        return True
    return closes.iloc[-1] < min_price


def _stock_fails_adv(ticker_data: pd.DataFrame | None, min_adv: float) -> bool:
    """Return True if 20-day ADV < min_adv."""
    if ticker_data is None or len(ticker_data) < 20:
        return True
    recent = ticker_data.iloc[-20:]
    dollar_volume = recent["volume"] * recent["raw_close"]
    return dollar_volume.mean() < min_adv


_ST_NAME_PATTERN = re.compile(r"(\*?ST|PT|退)", re.IGNORECASE)


def _exclude_st(candidates: list[str], stocks_meta: pd.DataFrame) -> list[str]:
    """Drop tickers whose name in stocks_meta matches ST/*ST/PT/退. Unknown tickers kept."""
    if stocks_meta is None or stocks_meta.empty:
        return list(candidates)
    kept = []
    for ticker in candidates:
        if ticker not in stocks_meta.index:
            kept.append(ticker)  # missing meta → conservative keep
            continue
        name = stocks_meta.loc[ticker, "name"]
        if not isinstance(name, str) or not _ST_NAME_PATTERN.search(name):
            kept.append(ticker)
    return kept


def apply_filters(
    ranked_tickers: list[str],
    all_prices: pd.DataFrame,
    data_provider,
    as_of: date,
    config: Config,
    current_positions: dict[str, Position] | None = None,
    profile: MarketProfile | None = None,
    bear_regime_cache: dict[date, bool] | None = None,
) -> list[str]:
    """Apply sequential filters preserving rank order.

    Filters applied in order:
      1. Regime: if index < regime_sma-SMA → block new entries (keep existing positions)
      2. Stock SMA: close < stock_sma-day SMA → remove
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
        profile: MarketProfile driving regime_index_id and regime_sma_window.
            Defaults to US profile when None.
        bear_regime_cache: optional dict mapping date → is_bear (bool). When provided,
            regime check uses cache lookup instead of querying data_provider.
            Defaults to None (queries data_provider).

    Returns:
        Filtered list of tickers in original rank order.
    """
    if not ranked_tickers:
        return []

    if profile is None:
        from clenow.markets import get_profile
        profile = get_profile(config.market if hasattr(config, "market") else "US")

    existing = set(current_positions.keys()) if current_positions else set()

    if bear_regime_cache is not None:
        bear = bear_regime_cache.get(as_of, False)
    else:
        bear = _is_bear_regime(
            data_provider,
            as_of,
            profile.regime_index_id,
            profile.regime_sma_window,
        )

    # ST exclusion filter (CN markets only)
    if "st" in profile.universe_exclusion_filters:
        stocks_meta = data_provider.get_stocks_meta(ranked_tickers)
        ranked_tickers = _exclude_st(ranked_tickers, stocks_meta)

    min_price = profile.min_price
    min_adv = profile.min_adv_amount

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
        if _stock_fails_price(ticker_data, min_price):
            continue

        # ADV filter
        if _stock_fails_adv(ticker_data, min_adv):
            continue

        result.append(ticker)

    return result