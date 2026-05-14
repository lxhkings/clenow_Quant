"""Clenow Smooth Momentum strategy implementation.

Wraps existing signals/portfolio functions to fit the Strategy Protocol.
Logic unchanged from pre-refactor engine.py.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from clenow.data.utils import get_ticker_series
from clenow.portfolio.ranker import rank_by_score
from clenow.portfolio.selector import apply_filters
from clenow.portfolio.sizing import compute_target_positions
from clenow.signals.clenow_score import compute_clenow_score

if TYPE_CHECKING:
    from clenow.config import Config
    from clenow.data.provider import DataProvider
    from clenow.markets.profiles import MarketProfile
    from clenow.types import Position

_LOOKBACK_CALENDAR = timedelta(days=450)


class ClenowMomentum:
    """Clenow Smooth Momentum strategy implementation.

    Wraps existing signals/portfolio functions to fit the Strategy Protocol.
    All logic delegated to existing implementations - no semantic changes.
    """

    name = "clenow_momentum"

    def score(self, ticker, ticker_data, profile, config):
        """Compute Clenow momentum score for a single ticker.

        Args:
            ticker: Stock ticker symbol (unused, for interface consistency).
            ticker_data: DataFrame with adj_close and raw_close columns.
            profile: MarketProfile with score_window and annualization_days.
            config: Config with gap_threshold.

        Returns:
            Float score value (0.0 for insufficient data or gaps).
        """
        if ticker_data is None or ticker_data.empty:
            return 0.0
        adj = ticker_data["adj_close"].dropna() if "adj_close" in ticker_data.columns else pd.Series(dtype=float)
        raw = ticker_data["raw_close"].dropna() if "raw_close" in ticker_data.columns else pd.Series(dtype=float)
        if len(adj) == 0 or len(raw) == 0:
            return 0.0
        return compute_clenow_score(
            adj_close=adj, raw_close=raw,
            score_window=profile.score_window,
            annualization_days=profile.annualization_days,
            gap_threshold=config.gap_threshold,
        )

    def rank(self, scores, config):
        """Rank tickers by score, return top percentage.

        Args:
            scores: Dict mapping ticker -> score.
            config: Config with top_pct field.

        Returns:
            List of tickers in descending score order, truncated to top_pct.
        """
        return rank_by_score(scores, config.top_pct)

    def entry_filters(self, ranked, all_prices, data_provider, as_of, config, profile, current_positions):
        """Apply sequential entry filters preserving rank order.

        Args:
            ranked: List of tickers in rank order.
            all_prices: DataFrame with (date, ticker) MultiIndex.
            data_provider: DataProvider for regime filter.
            as_of: Date for point-in-time filtering.
            config: Config with filter parameters.
            profile: MarketProfile with market-specific filter settings.
            current_positions: Dict of current positions for regime filter.

        Returns:
            Filtered list of tickers in original rank order.
        """
        return apply_filters(
            ranked, all_prices, data_provider, as_of, config,
            current_positions=current_positions, profile=profile,
        )

    def exit_signal(self, ticker, all_prices, as_of, data_provider, profile, config):
        """Check if position should be exited (100-day SMA break).

        Clenow double-exit rule: if close < 100-day SMA, force exit
        regardless of whether stock is in top 20%.

        Args:
            ticker: Stock ticker to check.
            all_prices: DataFrame with (date, ticker) MultiIndex, or None.
            as_of: Date for point-in-time check.
            data_provider: DataProvider for loading prices if not in all_prices.
            profile: MarketProfile with exit_sma_window.
            config: Config (unused, for interface consistency).

        Returns:
            True if should exit (close < SMA), False otherwise.
        """
        sma_window = profile.exit_sma_window
        ticker_data = get_ticker_series(all_prices, ticker) if all_prices is not None and not all_prices.empty else None
        if ticker_data is None:
            if data_provider is None:
                return True
            start = as_of - _LOOKBACK_CALENDAR
            prices = data_provider.load_prices([ticker], start, as_of)
            ticker_data = get_ticker_series(prices, ticker)
            if ticker_data is None:
                return True
        closes = ticker_data["raw_close"].dropna()
        if len(closes) < sma_window:
            return True
        sma = closes.iloc[-sma_window:].mean()
        return bool(closes.iloc[-1] < sma)

    def size(self, filtered_tickers, data_provider, as_of, config,
             current_cash, current_positions, current_prices, atrs, profile):
        """Compute target share counts via sequential ATR allocation.

        Args:
            filtered_tickers: Tickers in rank order (already filtered).
            data_provider: DataProvider (unused - ATRs passed separately).
            as_of: Rebalance date.
            config: Config with risk_factor and max_position_pct.
            current_cash: Available cash before rebalance.
            current_positions: Dict of current positions.
            current_prices: Dict mapping ticker -> current price.
            atrs: Dict mapping ticker -> ATR in dollars/share.
            profile: MarketProfile with lot_size.

        Returns:
            Dict mapping ticker -> target share count.
        """
        return compute_target_positions(
            filtered_tickers=filtered_tickers,
            data_provider=data_provider,
            as_of=as_of,
            config=config,
            current_cash=current_cash,
            current_positions=current_positions,
            current_prices=current_prices,
            atrs=atrs,
            profile=profile,
        )