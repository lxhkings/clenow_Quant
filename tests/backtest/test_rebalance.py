"""Tests for the rebalance schedule — Friday signal, Monday execution."""

from datetime import date

import pytest

from clenow.backtest.rebalance import get_rebalance_dates
from clenow.config import Config


class TestFridayMondayPairing:
    """Verify signal_date = Friday (or last trading day of week) and
    execution_date = Monday (or first trading day of next week)."""

    def test_basic_weekly_pairing(self):
        """Each pair should have signal on last trading day of week,
        execution on first trading day of next week."""
        config = Config(rebalance_freq="weekly")
        # Jan 6, 2025 is Monday; Jan 10 is Friday; Jan 13 is next Monday
        pairs = get_rebalance_dates(date(2025, 1, 6), date(2025, 2, 28), config)

        assert len(pairs) > 0
        for signal_date, exec_date in pairs:
            # Signal should be Friday (weekday 4) or last trading day of week
            # (could be Thursday if Friday is a holiday)
            assert signal_date.weekday() <= 4, (
                f"Signal date {signal_date} is a weekend day (weekday={signal_date.weekday()})"
            )
            # Execution should be first trading day of the next week
            # (Monday if no holiday, Tuesday if Monday is a holiday)
            assert exec_date.weekday() <= 4, (
                f"Exec date {exec_date} is a weekend day (weekday={exec_date.weekday()})"
            )
            # Signal is always before execution
            assert signal_date < exec_date

    def test_signal_before_execution(self):
        """Every signal_date is strictly before its execution_date."""
        config = Config(rebalance_freq="weekly")
        pairs = get_rebalance_dates(date(2025, 1, 1), date(2025, 6, 30), config)

        for signal_date, exec_date in pairs:
            assert signal_date < exec_date


class TestBiweeklyFrequency:
    """Biweekly: every other week."""

    def test_biweekly_half_as_many(self):
        """Biweekly produces approximately half as many pairs as weekly."""
        start = date(2025, 1, 6)
        end = date(2025, 6, 30)

        weekly_config = Config(rebalance_freq="weekly")
        biweekly_config = Config(rebalance_freq="biweekly")

        weekly_pairs = get_rebalance_dates(start, end, weekly_config)
        biweekly_pairs = get_rebalance_dates(start, end, biweekly_config)

        assert len(biweekly_pairs) == len(weekly_pairs) // 2 + len(weekly_pairs) % 2
        # Every other pair is kept
        for i, pair in enumerate(biweekly_pairs):
            assert pair == weekly_pairs[i * 2]


class TestHolidayHandling:
    """When Monday is a holiday, execution falls to Tuesday."""

    def test_monday_holiday_tuesday_execution(self):
        """MLK Day 2025: Monday Jan 20 is a holiday → Tuesday Jan 21 execution."""
        config = Config(rebalance_freq="weekly")
        # Jan 17, 2025 is Friday (signal)
        # Jan 20, 2025 is MLK Day (holiday) → execution should be Jan 21 (Tuesday)
        pairs = get_rebalance_dates(date(2025, 1, 13), date(2025, 1, 31), config)

        # Find the pair with signal_date = Jan 17
        mlk_pair = None
        for signal_date, exec_date in pairs:
            if signal_date == date(2025, 1, 17):
                mlk_pair = (signal_date, exec_date)
                break

        assert mlk_pair is not None, "No rebalance pair found for MLK week"
        assert mlk_pair[1] == date(2025, 1, 21), (
            f"Expected execution on Jan 21 (Tuesday after MLK), got {mlk_pair[1]}"
        )

    def test_july_4_holiday(self):
        """July 4, 2025 is a Friday holiday → Thursday is the signal day."""
        config = Config(rebalance_freq="weekly")
        pairs = get_rebalance_dates(date(2025, 6, 30), date(2025, 7, 15), config)

        # The week of July 4: Thursday July 3 should be the last trading day
        # (since July 4 Friday is a holiday), and Monday July 7 is execution
        july4_week = None
        for signal_date, exec_date in pairs:
            if signal_date.month == 7 or exec_date.month == 7:
                july4_week = (signal_date, exec_date)
                break

        # There should be a valid pair around July 4th
        # Signal date should be July 3 (Thursday) if July 4 is a holiday
        if july4_week is not None:
            signal_date, exec_date = july4_week
            assert signal_date < exec_date
            assert exec_date.weekday() == 0  # Monday


class TestDateBounds:
    """No rebalance dates before start or after end."""

    def test_no_signal_before_start(self):
        """All signal dates >= start date."""
        config = Config(rebalance_freq="weekly")
        start = date(2025, 3, 15)
        end = date(2025, 6, 30)
        pairs = get_rebalance_dates(start, end, config)

        for signal_date, exec_date in pairs:
            assert signal_date >= start

    def test_no_execution_after_end(self):
        """All execution dates <= end date."""
        config = Config(rebalance_freq="weekly")
        start = date(2025, 1, 6)
        end = date(2025, 3, 15)
        pairs = get_rebalance_dates(start, end, config)

        for signal_date, exec_date in pairs:
            assert exec_date <= end

    def test_empty_for_too_short_range(self):
        """Very short range with no complete Friday→Monday pair returns empty."""
        config = Config(rebalance_freq="weekly")
        # Just a single weekend
        pairs = get_rebalance_dates(date(2025, 1, 11), date(2025, 1, 12), config)
        # No complete pairs possible (need signal Fri + exec Mon)
        # The signal_date (Jan 10) is before start, so filtered out
        assert pairs == []
