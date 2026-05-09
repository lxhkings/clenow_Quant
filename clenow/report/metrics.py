"""Core performance metrics for the Clenow Smooth Momentum system.

Computes Sharpe, Sortino, Calmar, max drawdown, CAGR, turnover, win rate,
and average holding period from an equity curve and trade log.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: list[dict],
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute performance metrics from a backtest result.

    Args:
        equity_curve: DataFrame with columns: date, portfolio_value, cash
        trades: List of trade dicts with fields: entry_date, exit_date,
            ticker, shares, entry_price, exit_price, pnl
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Dict with keys: sharpe, sortino, calmar, max_drawdown,
        max_drawdown_start, max_drawdown_end, cagr, turnover_rate,
        win_rate, avg_holding_period
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return _empty_metrics()

    # Daily returns
    values = equity_curve["portfolio_value"].values.astype(float)
    daily_returns = np.diff(values) / values[:-1]
    daily_rf = risk_free_rate / 252.0
    excess_returns = daily_returns - daily_rf

    # Sharpe ratio (annualized)
    if np.std(daily_returns) == 0:
        sharpe = 0.0
    else:
        sharpe = float(np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252))

    # Sortino ratio
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0 or np.std(negative_returns) == 0:
        sortino = float("inf") if np.mean(excess_returns) > 0 else 0.0
    else:
        sortino = float(
            np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252)
        )

    # Max drawdown with start/end dates
    max_dd, dd_start, dd_end = _compute_max_drawdown(equity_curve)

    # CAGR
    initial_value = float(values[0])
    final_value = float(values[-1])
    trading_days = len(values) - 1
    if trading_days <= 0 or initial_value <= 0:
        cagr = 0.0
    else:
        cagr = float((final_value / initial_value) ** (252.0 / trading_days) - 1)

    # Calmar ratio
    if max_dd == 0:
        calmar = float("inf")
    else:
        annualized_return = cagr
        calmar = annualized_return / abs(max_dd)

    # Turnover rate
    turnover_rate = _compute_turnover(equity_curve, trades)

    # Win rate
    win_rate = _compute_win_rate(trades)

    # Average holding period
    avg_holding_period = _compute_avg_holding_period(trades)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_start": dd_start,
        "max_drawdown_end": dd_end,
        "cagr": cagr,
        "turnover_rate": turnover_rate,
        "win_rate": win_rate,
        "avg_holding_period": avg_holding_period,
    }


def _empty_metrics() -> dict:
    """Return default metrics for empty/insufficient data."""
    return {
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": float("inf"),
        "max_drawdown": 0.0,
        "max_drawdown_start": None,
        "max_drawdown_end": None,
        "cagr": 0.0,
        "turnover_rate": 0.0,
        "win_rate": 0.0,
        "avg_holding_period": 0.0,
    }


def _compute_max_drawdown(
    equity_curve: pd.DataFrame,
) -> tuple[float, date | None, date | None]:
    """Compute max drawdown with start/end dates.

    Returns:
        Tuple of (max_drawdown_pct, peak_date, trough_date)
        max_drawdown_pct is negative (e.g., -0.15 for 15% drawdown)
    """
    values = equity_curve["portfolio_value"].values.astype(float)
    dates = equity_curve["date"].values

    peak = values[0]
    peak_idx = 0
    max_dd = 0.0
    dd_start = None
    dd_end = None

    for i in range(1, len(values)):
        if values[i] > peak:
            peak = values[i]
            peak_idx = i

        dd = (values[i] - peak) / peak
        if dd < max_dd:
            max_dd = dd
            dd_start = _to_date(dates[peak_idx])
            dd_end = _to_date(dates[i])

    return max_dd, dd_start, dd_end


def _to_date(val) -> date | None:
    """Convert a value to a date object, handling various input types."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, pd.Timestamp):
        return val.date()
    # numpy datetime64
    try:
        return pd.Timestamp(val).date()
    except Exception:
        return None


def _compute_turnover(equity_curve: pd.DataFrame, trades: list[dict]) -> float:
    """Compute average annual turnover rate.

    Turnover per year = sum(abs(trade_value)) / (2 * avg_equity)
    Averaged across all years.
    """
    if not trades:
        return 0.0

    values = equity_curve["portfolio_value"].values.astype(float)
    dates = equity_curve["date"].values
    avg_equity = float(np.mean(values))

    if avg_equity <= 0:
        return 0.0

    # Group trade values by year
    yearly_turnover: dict[int, float] = {}
    yearly_counts: dict[int, int] = {}

    for trade in trades:
        trade_value = abs(trade.get("shares", 0) * trade.get("entry_price", 0))
        exit_date = trade.get("exit_date")
        if exit_date is None:
            continue

        exit_d = _to_date(exit_date)
        if exit_d is None:
            continue

        year = exit_d.year
        yearly_turnover[year] = yearly_turnover.get(year, 0.0) + trade_value
        yearly_counts[year] = yearly_counts.get(year, 0) + 1

    if not yearly_turnover:
        return 0.0

    # Average turnover across years
    turnover_rates = [
        yearly_turnover[yr] / (2 * avg_equity)
        for yr in yearly_turnover
    ]

    return float(np.mean(turnover_rates))


def _compute_win_rate(trades: list[dict]) -> float:
    """Compute percentage of profitable closed trades."""
    closed_trades = [t for t in trades if "pnl" in t]
    if not closed_trades:
        return 0.0

    winners = sum(1 for t in closed_trades if t["pnl"] > 0)
    return winners / len(closed_trades)


def _compute_avg_holding_period(trades: list[dict]) -> float:
    """Compute mean days held across all closed trades."""
    closed_trades = [t for t in trades if "entry_date" in t and "exit_date" in t]
    if not closed_trades:
        return 0.0

    holding_days = []
    for t in closed_trades:
        entry = _to_date(t["entry_date"])
        exit_d = _to_date(t["exit_date"])
        if entry is not None and exit_d is not None:
            holding_days.append((exit_d - entry).days)

    if not holding_days:
        return 0.0

    return float(np.mean(holding_days))
