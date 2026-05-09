"""Tests for the transaction cost model — 5 boundary cases."""

import pytest

from clenow.backtest.costs import compute_cost
from clenow.config import Config


class TestTypicalOrder:
    """Typical order = 0.5% ADV → slippage ~2.5 bps, total ~7.5 bps."""

    def test_typical_order(self):
        config = Config(
            half_spread_bps=5.0,
            slippage_bps_per_pct_adv=5.0,
            slippage_bps_min=1.0,
            slippage_bps_max=30.0,
            commission_per_share=0.0,
        )
        # order = 0.5% of ADV
        adv = 10_000_000.0
        order_notional = 50_000.0  # 0.5% of 10M
        shares = 500

        commission, slippage_bps = compute_cost(order_notional, adv, shares, config)

        # participation = 0.5% → slippage = 5 * (0.005 / 0.01) = 5 * 0.5 = 2.5 bps
        assert commission == 0.0
        assert slippage_bps == pytest.approx(2.5, abs=0.01)


class TestMinClip:
    """Tiny order → slippage clipped to 1 bp minimum."""

    def test_min_clip(self):
        config = Config(
            half_spread_bps=5.0,
            slippage_bps_per_pct_adv=5.0,
            slippage_bps_min=1.0,
            slippage_bps_max=30.0,
            commission_per_share=0.0,
        )
        # Very small order: 0.001% of ADV
        adv = 10_000_000.0
        order_notional = 100.0  # 0.001% of 10M
        shares = 1

        commission, slippage_bps = compute_cost(order_notional, adv, shares, config)

        # participation = 0.00001 → raw_slippage = 5 * (0.00001/0.01) = 0.005
        # clipped to min 1.0
        assert commission == 0.0
        assert slippage_bps == 1.0


class TestMaxClip:
    """Huge order → slippage clipped to 30 bps maximum."""

    def test_max_clip(self):
        config = Config(
            half_spread_bps=5.0,
            slippage_bps_per_pct_adv=5.0,
            slippage_bps_min=1.0,
            slippage_bps_max=30.0,
            commission_per_share=0.0,
        )
        adv = 10_000_000.0
        order_notional = 20_000_000.0  # 200% of ADV
        shares = 200_000

        commission, slippage_bps = compute_cost(order_notional, adv, shares, config)

        # participation = 2.0 → raw_slippage = 5 * (2.0/0.01) = 1000
        # clipped to max 30.0
        assert commission == 0.0
        assert slippage_bps == 30.0


class TestOrderEqualsADV:
    """Order = 100% of ADV → slippage = 5 * (1/0.01) = 500 bps → clipped to 30."""

    def test_order_equals_adv(self):
        config = Config(
            half_spread_bps=5.0,
            slippage_bps_per_pct_adv=5.0,
            slippage_bps_min=1.0,
            slippage_bps_max=30.0,
            commission_per_share=0.0,
        )
        adv = 10_000_000.0
        order_notional = 10_000_000.0  # 100% of ADV
        shares = 100_000

        commission, slippage_bps = compute_cost(order_notional, adv, shares, config)

        # participation = 1.0 → raw_slippage = 5 * (1.0/0.01) = 500
        # clipped to max 30.0
        assert commission == 0.0
        assert slippage_bps == 30.0


class TestOrderGreaterThanADV:
    """Order > ADV → slippage clipped to 30 bps."""

    def test_order_greater_than_adv(self):
        config = Config(
            half_spread_bps=5.0,
            slippage_bps_per_pct_adv=5.0,
            slippage_bps_min=1.0,
            slippage_bps_max=30.0,
            commission_per_share=0.0,
        )
        adv = 10_000_000.0
        order_notional = 50_000_000.0  # 500% of ADV
        shares = 500_000

        commission, slippage_bps = compute_cost(order_notional, adv, shares, config)

        # participation = 5.0 → raw_slippage = 5 * (5.0/0.01) = 2500
        # clipped to max 30.0
        assert commission == 0.0
        assert slippage_bps == 30.0


class TestWithCommission:
    """Non-zero commission_per_share is computed correctly."""

    def test_commission(self):
        config = Config(
            half_spread_bps=5.0,
            slippage_bps_per_pct_adv=5.0,
            slippage_bps_min=1.0,
            slippage_bps_max=30.0,
            commission_per_share=0.01,
        )
        adv = 10_000_000.0
        order_notional = 50_000.0
        shares = 500

        commission, slippage_bps = compute_cost(order_notional, adv, shares, config)

        assert commission == pytest.approx(5.0)  # 0.01 * 500
        assert slippage_bps == pytest.approx(2.5, abs=0.01)


class TestZeroADV:
    """Zero ADV → fallback to max slippage (conservative)."""

    def test_zero_adv(self):
        config = Config(
            half_spread_bps=5.0,
            slippage_bps_per_pct_adv=5.0,
            slippage_bps_min=1.0,
            slippage_bps_max=30.0,
            commission_per_share=0.0,
        )
        adv = 0.0
        order_notional = 50_000.0
        shares = 500

        commission, slippage_bps = compute_cost(order_notional, adv, shares, config)

        assert commission == 0.0
        assert slippage_bps == 30.0  # fallback to max
