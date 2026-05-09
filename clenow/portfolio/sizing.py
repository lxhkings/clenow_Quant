"""Sequential ATR-based position sizing."""

from __future__ import annotations

from datetime import date
from math import floor

from clenow.config import Config
from clenow.types import Position


def compute_target_positions(
    filtered_tickers: list[str],
    data_provider,
    as_of: date,
    config: Config,
    current_cash: float,
    current_positions: dict[str, Position],
    current_prices: dict[str, float],
    atrs: dict[str, float],
) -> dict[str, int]:
    """Compute target share counts via sequential ATR allocation.

    Algorithm (from Clenow design):
      equity = current_cash + sum(position.value for position in current_positions.values())
      available_cash = current_cash
      for ticker in filtered_ranked_list:
        atr_per_share = atrs[ticker]
        if atr_per_share < 0.01: continue
        target_shares = floor((equity * risk_factor) / atr_per_share)
        position_value = target_shares * current_prices[ticker]
        if position_value > max_position_pct * equity:
          target_shares = floor(max_position_pct * equity / current_prices[ticker])
          position_value = target_shares * current_prices[ticker]
        if available_cash < position_value:
          break
        result[ticker] = target_shares
        available_cash -= position_value

    Args:
        filtered_tickers: tickers in rank order (already filtered).
        data_provider: DataProvider (unused — ATRs passed separately for testability).
        as_of: the rebalance date.
        config: system configuration.
        current_cash: available cash before rebalance.
        current_positions: current open positions.
        current_prices: mapping ticker -> current price.
        atrs: mapping ticker -> ATR in dollars/share.

    Returns:
        Mapping of ticker -> target share count.
    """
    if not filtered_tickers:
        return {}

    equity = current_cash + sum(
        pos.shares * pos.entry_price for pos in current_positions.values()
    )
    available_cash = current_cash
    risk_factor = config.risk_factor
    max_position_pct = config.max_position_pct

    result: dict[str, int] = {}

    for ticker in filtered_tickers:
        atr_per_share = atrs.get(ticker, 0.0)
        if atr_per_share < 0.01:
            continue  # defensive: illiquid/stopped stock

        price = current_prices[ticker]
        target_shares = floor((equity * risk_factor) / atr_per_share)
        if target_shares <= 0:
            continue

        position_value = target_shares * price

        # 5% single-stock cap (truncation at entry only)
        if position_value > max_position_pct * equity:
            target_shares = floor(max_position_pct * equity / price)
            if target_shares <= 0:
                continue
            position_value = target_shares * price

        # Cash exhaustion check
        if available_cash < position_value:
            break

        result[ticker] = target_shares
        available_cash -= position_value

    return result
