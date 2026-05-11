"""Rebalance schedule — signal_date (Friday close) and execution_date (Monday open).

Uses pandas_market_calendars for trading-day validation.
Signal date = last trading day of the week (usually Friday).
Execution date = first trading day of the following week (usually Monday).
If Monday is a holiday, execution falls to Tuesday.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pandas_market_calendars as mcal

from clenow.config import Config


def get_rebalance_dates(
    start: date,
    end: date,
    config: Config,
    calendar_name: str = "NYSE",
) -> list[tuple[date, date]]:
    """Return (signal_date, execution_date) pairs for each rebalance cycle.

    Args:
        start: Backtest start date (inclusive).
        end: Backtest end date (inclusive).
        config: System configuration (rebalance_freq: "weekly" or "biweekly").
        calendar_name: Trading calendar to use (default "NYSE").
            Other options: "XSHG" (Shanghai), "XHKG" (Hong Kong).

    Returns:
        List of (signal_date, execution_date) tuples, chronologically sorted.
        signal_date is always < execution_date (no look-ahead).
    """
    calendar = mcal.get_calendar(calendar_name)
    # Get trading sessions covering the full range with buffer
    schedule = calendar.valid_days(
        start_date=start - timedelta(days=7),
        end_date=end + timedelta(days=7),
    )
    trading_days = [d.date() for d in schedule]

    if not trading_days:
        return []

    # Build a set for fast lookup
    trading_set = set(trading_days)

    # Group trading days by ISO week number + year
    weeks: dict[tuple[int, int], list[date]] = {}
    for td in trading_days:
        iso = td.isocalendar()
        key = (iso[0], iso[1])  # (year, week)
        if key not in weeks:
            weeks[key] = []
        weeks[key].append(td)

    # Sort weeks chronologically
    sorted_weeks = sorted(weeks.keys())

    # Build signal_date / execution_date pairs
    pairs: list[tuple[date, date]] = []
    for i in range(len(sorted_weeks) - 1):
        current_week = sorted_weeks[i]
        next_week = sorted_weeks[i + 1]

        # Signal date: last trading day of the current week
        signal_date = weeks[current_week][-1]
        # Execution date: first trading day of the next week
        execution_date = weeks[next_week][0]

        # Only include pairs within the [start, end] range
        if signal_date < start:
            continue
        if execution_date > end:
            continue
        # Safety: signal_date must be strictly before execution_date
        if signal_date >= execution_date:
            continue

        pairs.append((signal_date, execution_date))

    # Apply frequency filter
    if config.rebalance_freq == "biweekly" and len(pairs) > 1:
        # Keep every other pair (1st, 3rd, 5th, ...)
        pairs = pairs[::2]

    return pairs
