"""Backtest engine — compute_target_portfolio and run_backtest.

compute_target_portfolio is the CORE FUNCTION: pure, deterministic, single
code path shared by both backtest and live CLI.

Steps:
  1. Get PIT universe
  2. Compute Clenow score and ATR for each ticker
  3. Rank by score (top 20%)
  4. Apply filters (regime, SMA, price, ADV)
  5. Compute current prices
  6. Compute target positions via sequential ATR sizing
  7. Return TargetPortfolio

Double exit rule (CRITICAL):
  - Every rebalance: FIRST check existing positions for 100-day SMA break
  - THEN use the new top 20% list to decide which to keep
  - Even if a stock is in the top 20%, if it breaks its 100-day SMA, sell it
  - Regime filter (SP500 < 200 SMA): no NEW positions, existing positions
    only sold on their own exit triggers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from clenow.backtest.executor import SimulatedExecutor
from clenow.backtest.rebalance import get_rebalance_dates
from clenow.config import Config
from clenow.data.provider import DataProvider
from clenow.data.utils import get_ticker_series as _get_ticker_series
from clenow.errors import OrderRejection
from clenow.portfolio.ranker import rank_by_score
from clenow.portfolio.selector import apply_filters
from clenow.portfolio.sizing import compute_target_positions
from clenow.portfolio.state import PositionTracker
from clenow.signals.atr import compute_atr
from clenow.signals.clenow_score import compute_clenow_score
from clenow.types import Order, OrderType, Position, Side, TargetPortfolio

logger = logging.getLogger(__name__)

# Lookback for price data: 300 trading days to cover 200-day SMA + buffer
_PRICE_LOOKBACK_DAYS = 300
# Approximate calendar days for 300 trading days (~420 calendar days)
_PRICE_LOOKBACK_CALENDAR = timedelta(days=450)


def _slice_prices(
    preloaded: pd.DataFrame | None,
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Extract date+ticker slice from a preloaded (date, ticker) MultiIndex DataFrame.

    Args:
        preloaded: DataFrame with MultiIndex (date, ticker) or None.
        tickers: List of tickers to include in the slice.
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        DataFrame with matching rows, or empty DataFrame if preloaded is None/empty.
    """
    _COLS = ["raw_open", "raw_high", "raw_low", "raw_close", "volume", "adj_close", "dividend", "split_ratio"]
    if preloaded is None or preloaded.empty:
        return pd.DataFrame(columns=_COLS)
    date_idx = preloaded.index.get_level_values("date")
    ticker_idx = preloaded.index.get_level_values("ticker")
    # Convert date_idx to dates if it contains Timestamps (handles both formats)
    if date_idx.dtype == "object" and len(date_idx) > 0:
        # Index has Python date objects — convert start/end to dates for comparison
        mask = (
            (date_idx >= start)
            & (date_idx <= end)
            & (ticker_idx.isin(set(tickers)))
        )
    else:
        # Index has Timestamps — use Timestamp comparison
        mask = (
            (date_idx >= pd.Timestamp(start))
            & (date_idx <= pd.Timestamp(end))
            & (ticker_idx.isin(set(tickers)))
        )
    sliced = preloaded.loc[mask]
    return sliced if not sliced.empty else preloaded.iloc[0:0]


def _precompute_bear_regime(
    index_prices: pd.Series,
    sma_window: int,
) -> dict[date, bool]:
    """Pre-compute bear regime flag for every date in index_prices.

    Returns {date: True_if_bear} only for dates where enough history
    exists to compute the SMA (at least sma_window prior data points).
    """
    if index_prices.empty:
        return {}

    sma = index_prices.rolling(sma_window).mean()
    result: dict[date, bool] = {}
    for ts, price in index_prices.items():
        sma_val = sma.get(ts)
        if sma_val is None or pd.isna(sma_val):
            continue
        d = ts.date() if isinstance(ts, pd.Timestamp) else ts
        result[d] = bool(price < sma_val)
    return result


def _select_one_per_sector(
    scores: dict[str, float],
    sector_mapping: dict[str, str],
) -> list[str]:
    """Select highest-score stock from each sector.

    Args:
        scores: ticker -> Clenow score mapping.
        sector_mapping: ticker -> sector mapping.

    Returns:
        List of tickers (one per sector) sorted by score descending.
    """
    # Group by sector
    by_sector: dict[str, list[tuple[str, float]]] = {}
    unmapped = []
    for ticker, score in scores.items():
        if score <= 0:
            continue
        sector = sector_mapping.get(ticker)
        if sector:
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append((ticker, score))
        else:
            unmapped.append(ticker)

    if unmapped:
        logger.warning(
            "%d tickers with positive score not in sector_mapping (excluded): %s...",
            len(unmapped),
            ", ".join(unmapped[:10]),
        )

    # Pick highest score per sector
    selected: list[tuple[str, float]] = []
    for sector, tickers in by_sector.items():
        sorted_tickers = sorted(tickers, key=lambda x: x[1], reverse=True)
        if sorted_tickers:
            selected.append(sorted_tickers[0])

    # Return sorted by score descending
    selected.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in selected]


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    equity_curve: pd.DataFrame  # columns: date, portfolio_value, cash
    trades: list[dict]  # entry_date, exit_date, ticker, shares, entry_price, exit_price, pnl
    final_positions: dict[str, Position]
    final_cash: float
    config: Config


def _check_sma_break(
    ticker: str,
    all_prices: pd.DataFrame,
    as_of: date,
    data_provider: DataProvider,
    config: Config,
    sma_window: int = 100,
) -> bool:
    """Return True when the stock closes below its `sma_window`-day SMA.

    Reads from the preloaded `all_prices` frame first; falls back to a
    per-ticker query only when the position is no longer in the universe
    (which `compute_target_portfolio` no longer covers in `all_prices`).
    """
    ticker_data = _get_ticker_series(all_prices, ticker)
    if ticker_data is None:
        # Held position not in the current universe (e.g. removed from PIT) —
        # one-off query as a fallback. This is rare and bounded.
        start = as_of - _PRICE_LOOKBACK_CALENDAR
        prices = data_provider.load_prices([ticker], start, as_of)
        ticker_data = _get_ticker_series(prices, ticker)
        if ticker_data is None:
            return True
    closes = ticker_data["raw_close"].dropna()
    if len(closes) < sma_window:
        return True
    sma = closes.iloc[-sma_window:].mean()
    return closes.iloc[-1] < sma


def compute_target_portfolio(
    as_of: date,
    current_positions: dict[str, Position],
    current_cash: float,
    config: Config,
    data_provider: DataProvider,
    sector_mapping: dict[str, str] | None = None,
    select_one_per_sector: bool = False,
    profile: "MarketProfile | None" = None,
    preloaded_prices: pd.DataFrame | None = None,
    bear_regime_cache: dict[date, bool] | None = None,
) -> TargetPortfolio:
    """Compute the target portfolio for a given date.

    Pure function: given the same inputs, ALWAYS returns the same TargetPortfolio.
    No I/O outside data_provider. No clock. No broker.

    Steps (in order):
      1. Get PIT universe
      2. Compute Clenow score and ATR for each ticker
      3. Rank by score (top top_pct, or one per sector if select_one_per_sector)
      4. Apply filters
      5. Apply double exit rule (existing positions breaking 100-day SMA)
      6. Compute current prices
      7. Compute target positions via sequential ATR sizing
      8. Return TargetPortfolio
    """
    # Resolve profile (backward-compat: falls back to US)
    if profile is None:
        from clenow.markets import get_profile
        profile = get_profile("US")

    # Step 1: Get point-in-time universe
    universe = data_provider.get_universe(as_of, index_id=profile.universe_index_id)

    if not universe:
        return TargetPortfolio(positions={}, as_of=as_of)

    # Step 2: Compute score and ATR for each ticker
    price_start = as_of - _PRICE_LOOKBACK_CALENDAR
    if preloaded_prices is not None:
        all_prices = _slice_prices(preloaded_prices, universe, price_start, as_of)
    else:
        all_prices = data_provider.load_prices(universe, price_start, as_of)

    scores: dict[str, float] = {}
    atrs: dict[str, float] = {}
    current_prices: dict[str, float] = {}

    for ticker in universe:
        ticker_data = _get_ticker_series(all_prices, ticker)

        if ticker_data is None or ticker_data.empty:
            scores[ticker] = 0.0
            atrs[ticker] = 0.0
            continue

        # Compute Clenow score (adj_close + raw_close)
        adj_close = ticker_data["adj_close"].dropna() if "adj_close" in ticker_data.columns else pd.Series(dtype=float)
        raw_close = ticker_data["raw_close"].dropna() if "raw_close" in ticker_data.columns else pd.Series(dtype=float)

        if len(adj_close) > 0 and len(raw_close) > 0:
            scores[ticker] = compute_clenow_score(
                adj_close=adj_close,
                raw_close=raw_close,
                score_window=profile.score_window,
                annualization_days=profile.annualization_days,
                gap_threshold=config.gap_threshold,
            )
        else:
            scores[ticker] = 0.0

        # Compute ATR (raw high, low, close)
        if all(col in ticker_data.columns for col in ["raw_high", "raw_low", "raw_close"]):
            high = ticker_data["raw_high"].dropna()
            low = ticker_data["raw_low"].dropna()
            close = ticker_data["raw_close"].dropna()
            if len(high) > 0 and len(low) > 0 and len(close) > 0:
                atrs[ticker] = compute_atr(
                    high=high, low=low, close=close, period=config.atr_period
                )
            else:
                atrs[ticker] = 0.0
        else:
            atrs[ticker] = 0.0

        # Track current price (last raw_close)
        if "raw_close" in ticker_data.columns:
            rc = ticker_data["raw_close"].dropna()
            if len(rc) > 0:
                current_prices[ticker] = float(rc.iloc[-1])

    # Step 3: Rank by score
    if select_one_per_sector:
        if not sector_mapping:
            raise ValueError(
                "select_one_per_sector=True requires a non-empty sector_mapping"
            )
        # Select one stock per sector (highest score in each sector)
        ranked = _select_one_per_sector(scores, sector_mapping)
    else:
        # Traditional: top pct of universe
        ranked = rank_by_score(scores, config.top_pct)

    # Step 4: Apply sequential filters (regime, SMA, price, ADV)
    # Pass pre-loaded all_prices to eliminate N+1 queries
    filtered = apply_filters(ranked, all_prices, data_provider, as_of, config, current_positions, profile=profile, bear_regime_cache=bear_regime_cache)

    # Step 4b: Exclude suspended tickers from new entries
    suspended = get_suspended_tickers(filtered, all_prices, as_of, profile)
    if suspended:
        filtered = [t for t in filtered if t not in suspended]

    # Step 5: Double exit rule — CRITICAL
    # FIRST check all existing positions for 100-day SMA break
    # Remove broken positions from the target, even if they're in the top 20%
    forced_sells: set[str] = set()
    for ticker in current_positions:
        if _check_sma_break(ticker, all_prices, as_of, data_provider, config, sma_window=profile.exit_sma_window):
            forced_sells.add(ticker)

    # Remove forced-sell tickers from the filtered list
    filtered = [t for t in filtered if t not in forced_sells]

    # Step 6: current_prices already computed in step 2

    # Step 7: Compute target positions via sequential ATR sizing
    target_shares = compute_target_positions(
        filtered_tickers=filtered,
        data_provider=data_provider,
        as_of=as_of,
        config=config,
        current_cash=current_cash,
        current_positions=current_positions,
        current_prices=current_prices,
        atrs=atrs,
        profile=profile,
    )

    return TargetPortfolio(positions=target_shares, as_of=as_of)


def _compute_diff(
    current_positions: dict[str, Position],
    target: TargetPortfolio,
    execution_date: date,
) -> list[Order]:
    """Compute orders needed to move from current positions to target.

    For each ticker in target:
      - If not currently held → BUY order
      - If held but target shares > current → BUY additional shares
      - If held but target shares < current → SELL excess shares

    For each ticker in current but NOT in target:
      - SELL all shares (full exit)
    """
    orders: list[Order] = []

    # Sells first (free up cash)
    for ticker, pos in current_positions.items():
        target_shares = target.positions.get(ticker, 0)
        if target_shares < pos.shares:
            sell_shares = pos.shares - target_shares
            orders.append(
                Order(
                    ticker=ticker,
                    shares=sell_shares,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    target_date=execution_date,
                )
            )

    # Buys
    for ticker, target_shares in target.positions.items():
        current_shares = current_positions[ticker].shares if ticker in current_positions else 0
        if target_shares > current_shares:
            buy_shares = target_shares - current_shares
            orders.append(
                Order(
                    ticker=ticker,
                    shares=buy_shares,
                    side=Side.BUY,
                    order_type=OrderType.MARKET,
                    target_date=execution_date,
                )
            )

    return orders


def _get_daily_prices(
    data_provider: DataProvider,
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Load daily close prices for equity curve tracking."""
    if not tickers:
        return pd.DataFrame()
    return data_provider.load_prices(tickers, start, end)


def run_backtest(
    start: date,
    end: date,
    initial_cash: float,
    config: Config,
    data_provider: DataProvider,
    sector_mapping: dict[str, str] | None = None,
    select_one_per_sector: bool = False,
    profile: "MarketProfile | None" = None,
) -> BacktestResult:
    """Run a full backtest and return results.

    Thin wrapper that:
      1. Gets rebalance dates
      2. For each (signal_date, execution_date) pair:
         a. On signal_date: compute target portfolio
         b. Compute diff: compare current positions vs target
         c. On execution_date: submit orders to SimulatedExecutor
         d. Apply fills to PositionTracker
         e. Apply any corporate actions between signal and execution dates
         f. Record equity inline (no duplicate loop)
      3. Return BacktestResult

    Args:
        sector_mapping: ticker -> sector mapping for sector-based selection.
        select_one_per_sector: if True, select 1 stock per sector instead of top_pct.
        profile: MarketProfile driving all market-specific parameters.
            Defaults to US profile when None.
    """
    # Resolve profile (backward-compat: falls back to US)
    if profile is None:
        from clenow.markets import get_profile
        profile = get_profile("US")

    # Use profile capital if initial_cash not explicitly provided (0 or negative)
    if initial_cash <= 0:
        initial_cash = profile.default_starting_capital

    # Step 1: Get rebalance schedule
    rebalance_pairs = get_rebalance_dates(start, end, config, calendar_name=profile.trading_calendar_name)

    # Initialize position tracker
    tracker = PositionTracker(cash=initial_cash)

    # Trade log for closed positions
    closed_trades: list[dict] = []
    # Track entry info for open positions
    entry_info: dict[str, dict] = {}  # ticker -> {entry_date, entry_price, shares}

    # Build ADV data (will be populated as we go)
    adv_data: dict[str, float] = {}

    # Initialize equity records with start-of-backtest entry
    equity_records: list[dict] = [{
        "date": start,
        "portfolio_value": initial_cash,
        "cash": initial_cash,
    }]

    # Helper to get empty prices frame
    def _empty_prices_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["raw_open", "raw_high", "raw_low", "raw_close", "volume", "adj_close", "dividend", "split_ratio"]
        )

    total = len(rebalance_pairs)

    # ── One-time preload ───────────────────────────────────────────────────
    # Collect full universe superset: get_universe() hits in-memory PIT dict
    # after first build, so 4111 calls cost microseconds, not DB queries.
    logger.info("Collecting universe superset for preload...")
    all_universe_tickers: set[str] = set()
    for _sd, _ in rebalance_pairs:
        all_universe_tickers.update(
            data_provider.get_universe(_sd, index_id=profile.universe_index_id)
        )
    global_price_start = start - _PRICE_LOOKBACK_CALENDAR - timedelta(days=60)
    logger.info(
        "Preloading prices for %d tickers from %s to %s...",
        len(all_universe_tickers), global_price_start, end,
    )
    preloaded_prices = data_provider.load_prices(
        sorted(all_universe_tickers), global_price_start, end
    )
    logger.info("Preload complete: %d rows", len(preloaded_prices))

    # Precompute regime filter: one DB query → {date: bool} dict.
    _sma_window = profile.regime_sma_window
    _raw_index = data_provider.get_index_prices(
        profile.regime_index_id,
        start - timedelta(days=_sma_window * 2 + 60),
        end,
    )
    if not _raw_index.empty:
        _close_col = (
            "raw_close" if "raw_close" in _raw_index.columns
            else "close" if "close" in _raw_index.columns
            else _raw_index.select_dtypes("number").columns[0]
        )
        bear_regime_cache: dict[date, bool] = _precompute_bear_regime(
            _raw_index[_close_col].sort_index(), _sma_window
        )
    else:
        bear_regime_cache = {}
    logger.info("Regime cache: %d dates", len(bear_regime_cache))
    # ── End preload ────────────────────────────────────────────────────────

    # Process each rebalance pair
    for i, (signal_date, execution_date) in enumerate(rebalance_pairs, 1):
        n_pos = len(tracker.get_positions())
        print(f"[{i}/{total}] signal={signal_date} exec={execution_date} pos={n_pos}", flush=True)
        # Step 0: Detect and handle delisted positions before computing target
        _detect_delistings(
            tracker, data_provider, signal_date,
            profile=profile, preloaded_prices=preloaded_prices,
        )

        # Step 2a: Compute target portfolio on signal_date
        current_positions = tracker.get_positions()
        current_cash = tracker.get_cash()

        target = compute_target_portfolio(
            as_of=signal_date,
            current_positions=current_positions,
            current_cash=current_cash,
            config=config,
            data_provider=data_provider,
            sector_mapping=sector_mapping,
            select_one_per_sector=select_one_per_sector,
            profile=profile,
            preloaded_prices=preloaded_prices,
            bear_regime_cache=bear_regime_cache,
        )

        # Step 2b: Compute diff (orders needed)
        orders = _compute_diff(current_positions, target, execution_date)

        # Load prices for valuation: order tickers AND current position tickers
        position_tickers = list(current_positions.keys())
        order_tickers = list({o.ticker for o in orders}) if orders else []
        valuation_tickers = list(set(order_tickers) | set(position_tickers))

        if valuation_tickers:
            # 15 days covers CN Spring Festival / Golden Week; 5 was too short
            exec_lookback = execution_date - timedelta(days=15)
            exec_prices = _slice_prices(
                preloaded_prices, valuation_tickers, exec_lookback, execution_date
            )
        else:
            exec_prices = _empty_prices_frame()

        if orders:
            # Step 2c: Batched ADV lookup — served from preloaded in-memory data
            adv_lookback_start = execution_date - timedelta(days=30)
            adv_hist = _slice_prices(
                preloaded_prices, order_tickers, adv_lookback_start, execution_date
            )

            for ticker in order_tickers:
                td = _get_ticker_series(adv_hist, ticker)
                if td is None or len(td) < 20:
                    continue
                recent = td.iloc[-20:]
                dv = recent["volume"] * recent["raw_close"]
                adv_data[ticker] = float(dv.mean())

            # Build stocks_meta for price-limit resolver (CN/HK need name)
            stocks_meta_map: dict[str, str] = {}
            if profile.price_limit_resolver is not None and order_tickers:
                try:
                    meta_df = data_provider.get_stocks_meta(order_tickers)
                    if meta_df is not None and not meta_df.empty and "name" in meta_df.columns:
                        stocks_meta_map = meta_df["name"].to_dict()
                except AttributeError:
                    pass  # Provider doesn't implement get_stocks_meta
                except Exception as e:
                    logger.warning("Failed to load stocks_meta for price-limit check: %s", e)

            executor = SimulatedExecutor(
                price_data=exec_prices,
                adv_data=adv_data,
                profile=profile,
                stocks_meta=stocks_meta_map,
            )

            # Submit orders, handling rejections
            sells = [o for o in orders if o.side == Side.SELL]
            buys = [o for o in orders if o.side == Side.BUY]

            # Process sells first (free up cash)
            for order in sells:
                try:
                    fills = executor.submit([order], config)
                    tracker.apply_fills(fills)
                    # Record closed trade if fully sold
                    if order.ticker in current_positions and order.ticker not in target.positions:
                        pos = current_positions[order.ticker]
                        fill = fills[0]
                        if order.ticker in entry_info:
                            info = entry_info.pop(order.ticker)
                            pnl = (fill.fill_price - info["entry_price"]) * info["shares"] - fill.commission
                            closed_trades.append({
                                "entry_date": info["entry_date"],
                                "exit_date": execution_date,
                                "ticker": order.ticker,
                                "shares": info["shares"],
                                "entry_price": info["entry_price"],
                                "exit_price": fill.fill_price,
                                "pnl": pnl,
                            })
                except OrderRejection:
                    logger.warning(
                        "Order rejected: SELL %s on %s (no_market)",
                        order.ticker, execution_date,
                    )

            # Process buys
            for order in buys:
                try:
                    fills = executor.submit([order], config)
                    tracker.apply_fills(fills)
                    # Track entry info for new positions
                    fill = fills[0]
                    if order.ticker not in current_positions:
                        entry_info[order.ticker] = {
                            "entry_date": execution_date,
                            "entry_price": fill.fill_price,
                            "shares": fill.shares,
                        }
                    else:
                        # Adding to existing position — update entry info
                        if order.ticker in entry_info:
                            entry_info[order.ticker]["shares"] += fill.shares
                        else:
                            entry_info[order.ticker] = {
                                "entry_date": current_positions[order.ticker].entry_date,
                                "entry_price": current_positions[order.ticker].entry_price,
                                "shares": current_positions[order.ticker].shares + fill.shares,
                            }
                except OrderRejection:
                    logger.warning(
                        "Order rejected: BUY %s on %s (no_market)",
                        order.ticker, execution_date,
                    )

        # Step 2e: Apply corporate actions between signal and execution dates
        # (split and dividend handling — uses data from price data)
        _apply_corporate_actions(
            tracker, data_provider, signal_date, execution_date,
            preloaded_prices=preloaded_prices,
        )

        # Step 2f: Record equity inline (was second loop)
        positions_now = tracker.get_positions()
        if positions_now:
            day_prices: dict[str, float] = {}
            if not exec_prices.empty and isinstance(exec_prices.index, pd.MultiIndex):
                try:
                    day_data = exec_prices.xs(execution_date, level=0)
                    for t in day_data.index:
                        if "raw_close" in day_data.columns:
                            day_prices[t] = float(day_data.loc[t, "raw_close"])
                except (KeyError, TypeError):
                    pass
            equity = tracker.get_equity(day_prices)
        else:
            equity = tracker.get_cash()

        equity_records.append({
            "date": execution_date,
            "portfolio_value": equity,
            "cash": tracker.get_cash(),
        })

    equity_df = pd.DataFrame(equity_records)
    if equity_df.empty:
        equity_df = pd.DataFrame(
            columns=["date", "portfolio_value", "cash"]
        )

    return BacktestResult(
        equity_curve=equity_df,
        trades=closed_trades,
        final_positions=tracker.get_positions(),
        final_cash=tracker.get_cash(),
        config=config,
    )


def _apply_corporate_actions(
    tracker: PositionTracker,
    data_provider: DataProvider,
    signal_date: date,
    execution_date: date,
    preloaded_prices: pd.DataFrame | None = None,
) -> None:
    """Apply corporate actions (splits, dividends) between two dates.

    Uses price data to detect splits (split_ratio != 1.0) and dividends
    (dividend > 0). Only applies to currently-held positions.
    """
    positions = tracker.get_positions()
    if not positions:
        return

    tickers = list(positions.keys())
    if preloaded_prices is not None:
        prices = _slice_prices(preloaded_prices, tickers, signal_date, execution_date)
    else:
        prices = data_provider.load_prices(tickers, signal_date, execution_date)

    for ticker in tickers:
        ticker_data = _get_ticker_series(prices, ticker)
        if ticker_data is None or ticker_data.empty:
            continue

        for row_date, row in ticker_data.iterrows():
            # Handle date type from index
            if isinstance(row_date, pd.Timestamp):
                row_date = row_date.date()
            elif not isinstance(row_date, date):
                continue

            # Skip the signal_date itself (already accounted for)
            if row_date <= signal_date:
                continue

            # Apply split
            if "split_ratio" in row.index:
                ratio = row["split_ratio"]
                if pd.notna(ratio) and ratio != 1.0 and ratio != 0:
                    try:
                        tracker.apply_split(ticker, float(ratio), row_date)
                    except Exception:
                        pass  # InvariantError if not in positions — ignore

            # Apply dividend
            if "dividend" in row.index:
                div = row["dividend"]
                if pd.notna(div) and div > 0:
                    try:
                        tracker.apply_dividend(ticker, float(div), row_date)
                    except Exception:
                        pass  # InvariantError if not in positions — ignore


def _classify_inactive(
    volume_history: pd.Series,
    profile: "MarketProfile",
) -> Literal["active", "suspended", "delisted"]:
    """Classify a ticker's recent state from trailing volume zeros.

    Trailing zero run length L:
      L == 0:                                    active
      L <= profile.delisting_threshold_days:     suspended (hold position, skip rebalance)
      L > profile.delisting_threshold_days:      delisted (force close)
    """
    if len(volume_history) == 0:
        return "delisted"

    trailing_zeros = 0
    for v in reversed(volume_history.tolist()):
        if v == 0:
            trailing_zeros += 1
        else:
            break

    if trailing_zeros == 0:
        return "active"
    if trailing_zeros > profile.delisting_threshold_days:
        return "delisted"
    return "suspended"


def get_suspended_tickers(
    candidates: list[str],
    all_prices: pd.DataFrame,
    as_of: date,
    profile: "MarketProfile",
) -> set[str]:
    """Return set of currently-suspended tickers from candidate list."""
    suspended: set[str] = set()
    for ticker in candidates:
        try:
            vol = all_prices.xs(ticker, level="ticker")["volume"].loc[:as_of]
        except KeyError:
            continue
        state = _classify_inactive(vol.tail(profile.delisting_threshold_days + 5), profile)
        if state == "suspended":
            suspended.add(ticker)
    return suspended


def _detect_delistings(
    tracker: PositionTracker,
    data_provider: DataProvider,
    as_of: date,
    profile: "MarketProfile | None" = None,
    preloaded_prices: pd.DataFrame | None = None,
) -> None:
    """Detect and force-close delisted positions.

    A stock is considered delisted if:
    1. It has no raw_close data on the as_of date, OR
    2. It has consecutive zero-volume days exceeding profile.delisting_threshold_days

    Suspended stocks (zero-volume ≤ threshold) are kept (not force-closed).

    This should be called before computing the target portfolio at each
    rebalance to ensure we don't try to hold delisted stocks.
    """
    if profile is None:
        from clenow.markets import get_profile
        profile = get_profile("US")

    positions = tracker.get_positions()
    if not positions:
        return

    tickers = list(positions.keys())

    # Load price data with 30-day lookback for consistent delisting detection
    # regardless of whether preloaded prices are available.
    _delisting_start = as_of - timedelta(days=30)
    if preloaded_prices is not None:
        prices = _slice_prices(preloaded_prices, tickers, _delisting_start, as_of)
    else:
        prices = data_provider.load_prices(tickers, _delisting_start, as_of)

    for ticker in tickers:
        ticker_data = _get_ticker_series(prices, ticker)

        # If no price data at all for as_of, stock may be delisted
        if ticker_data is None or ticker_data.empty:
            # Look back to find the last available close price
            lookback_start = as_of - timedelta(days=30)
            if preloaded_prices is not None:
                hist_prices = _slice_prices(preloaded_prices, [ticker], lookback_start, as_of)
            else:
                hist_prices = data_provider.load_prices([ticker], lookback_start, as_of)
            hist_data = _get_ticker_series(hist_prices, ticker)

            if hist_data is None or hist_data.empty:
                # No price data at all — force close at entry price (best effort)
                pos = positions[ticker]
                tracker.apply_delisting(ticker, pos.entry_price, as_of)
                logger.warning(
                    "Delisted %s on %s: no price data, force-closing at entry price %.2f",
                    ticker, as_of, pos.entry_price,
                )
                continue

            # Get the last available close price
            closes = hist_data["raw_close"].dropna()
            if closes.empty:
                pos = positions[ticker]
                tracker.apply_delisting(ticker, pos.entry_price, as_of)
                logger.warning(
                    "Delisted %s on %s: no close data, force-closing at entry price %.2f",
                    ticker, as_of, pos.entry_price,
                )
            else:
                last_close = float(closes.iloc[-1])
                tracker.apply_delisting(ticker, last_close, as_of)
                logger.info(
                    "Delisted %s on %s: force-closing at last close %.2f",
                    ticker, as_of, last_close,
                )
        else:
            # Price data exists — check volume-based suspension/delisting
            vol = ticker_data["volume"].loc[:as_of]
            state = _classify_inactive(vol.tail(profile.delisting_threshold_days + 5), profile)
            if state == "delisted":
                last_close = float(ticker_data["raw_close"].dropna().iloc[-1])
                tracker.apply_delisting(ticker, last_close, as_of)
                logger.info(
                    "Delisted %s on %s: zero-volume run > %d days, force-closing at %.2f",
                    ticker, as_of, profile.delisting_threshold_days, last_close,
                )
            # state == "suspended": no action (hold position)
