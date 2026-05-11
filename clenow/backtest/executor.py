"""Order execution — simulated and broker executors.

SimulatedExecutor: fills at raw_open on execution_date, applies cost model.
BrokerExecutor: Phase 2 stub.

CRITICAL: Time isolation — signal_date (Friday close) and execution_date
(Monday open) are always different bars. No look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import pandas as pd

from clenow.backtest.costs import compute_cost
from clenow.config import Config
from clenow.errors import OrderRejection as _OrderRejectionException
from clenow.types import Fill, Order, Side


@dataclass
class OrderRejection:
    """Data-class for rejected orders (returned, not raised)."""
    order: Order
    reason: str


class Executor(Protocol):
    """Protocol for order execution."""

    def submit(self, orders: list[Order], config: Config) -> list[Fill]:
        """Submit orders and return fills.

        Orders that cannot be filled raise OrderRejection within
        the implementation — the caller catches and records them.
        """
        ...


class SimulatedExecutor:
    """Simulated executor that fills at raw_open prices.

    Parameters
    ----------
    price_data : pd.DataFrame
        Full price data with MultiIndex (date, ticker) and columns including
        raw_open, raw_high, raw_low, raw_close, volume, adj_close.
    adv_data : dict[str, float]
        Mapping of ticker -> 20-day ADV in dollars as of the execution date.
    profile : MarketProfile | None
        Market profile for price-limit detection. None disables limit checks.
    stocks_meta : dict[str, str] | None
        Mapping of ticker -> stock name, used by price_limit_resolver.
    """

    def __init__(
        self,
        price_data: pd.DataFrame | None = None,
        adv_data: dict[str, float] | None = None,
        profile: object | None = None,
        stocks_meta: dict[str, str] | None = None,
        *,
        prices: pd.DataFrame | None = None,
    ) -> None:
        # Accept either price_data or prices (test convenience)
        resolved = prices if prices is not None else price_data
        if resolved is None:
            raise TypeError("SimulatedExecutor requires price_data or prices")
        self._prices = resolved
        self._adv_data = adv_data or {}
        self.profile = profile
        self._stocks_meta = stocks_meta or {}

    def submit(self, orders: list[Order], config: Config) -> list[Fill]:
        """Fill orders at raw_open price on execution_date.

        For each order:
        1. Check price-limit lock; if locked, raise OrderRejection.
        2. Look up raw_open on the order's target_date.
        3. Apply cost model via compute_cost.
        4. If price is missing, raise OrderRejection (caller handles).

        Returns list of successful Fills. OrderRejection is raised per-order;
        the engine event loop catches it and records rejected orders.
        """
        fills: list[Fill] = []

        for order in orders:
            # Price-limit lock check
            if self._is_price_limit_locked(order.ticker, order.target_date):
                raise _OrderRejectionException(
                    order.ticker,
                    reason=f"price_limit_locked: {order.target_date}",
                )

            fill_price = self._get_open_price(order.ticker, order.target_date)

            if fill_price is None:
                raise _OrderRejectionException(order.ticker, reason="no_market")

            # Compute ADV for cost model
            adv = self._adv_data.get(order.ticker, 0.0)
            order_notional = fill_price * order.shares

            commission, slippage_bps = compute_cost(
                order_notional=order_notional,
                adv_20_dollars=adv,
                shares=order.shares,
                config=config,
            )

            fill = Fill(
                ticker=order.ticker,
                shares=order.shares,
                fill_price=fill_price,
                timestamp=datetime.combine(order.target_date, datetime.min.time()),
                commission=commission,
                slippage_bps=slippage_bps,
                side=order.side,
            )
            fills.append(fill)

        return fills

    def execute(self, orders: list[Order]) -> tuple[list[Fill], list[OrderRejection]]:
        """Execute orders with price-limit lock detection.

        For each order:
        1. Check if target_date is price-limit locked; if so, reject.
        2. Look up raw_open price; fill at that price (no cost model).

        Returns (fills, rejections) tuple.
        """
        fills: list[Fill] = []
        rejections: list[OrderRejection] = []

        for order in orders:
            # Price-limit lock check
            if self._is_price_limit_locked(order.ticker, order.target_date):
                rejections.append(OrderRejection(
                    order=order,
                    reason=f"price_limit_locked: {order.target_date}",
                ))
                continue

            fill_price = self._get_open_price(order.ticker, order.target_date)

            if fill_price is None:
                rejections.append(OrderRejection(
                    order=order,
                    reason=f"no_market: {order.target_date}",
                ))
                continue

            fill = Fill(
                ticker=order.ticker,
                shares=order.shares,
                fill_price=fill_price,
                timestamp=datetime.combine(order.target_date, datetime.min.time()),
                commission=0.0,
                slippage_bps=0.0,
                side=order.side,
            )
            fills.append(fill)

        return fills, rejections

    def _is_price_limit_locked(self, ticker: str, target_date: date) -> bool:
        """True if target_date is price-limit locked for ticker per profile rules."""
        if self.profile is None:
            return False
        if self.profile.price_limit_pct is None and self.profile.price_limit_resolver is None:
            return False

        # Look up today's bar — handle both date and Timestamp index keys
        today = None
        for dt in (target_date, pd.Timestamp(target_date)):
            key = (dt, ticker)
            if key in self._prices.index:
                today = self._prices.loc[key]
                break
        if today is None:
            return False

        if not (today["raw_open"] == today["raw_high"] == today["raw_low"]):
            return False  # not locked: range > 0

        # Find prev trading day close
        ticker_prices = self._prices.xs(ticker, level="ticker")
        prev_dates = ticker_prices.index[ticker_prices.index < pd.Timestamp(target_date)]
        if len(prev_dates) == 0:
            return False
        prev_close = ticker_prices.loc[prev_dates[-1], "raw_close"]

        move = abs(today["raw_open"] / prev_close - 1.0)

        if self.profile.price_limit_pct is not None:
            limit = self.profile.price_limit_pct
        else:
            name = self._stocks_meta.get(ticker, "")
            limit = self.profile.price_limit_resolver(ticker, name)

        return move >= limit - 1e-6

    def _get_open_price(self, ticker: str, target_date: date) -> float | None:
        """Look up raw_open for a ticker on a given date.

        Returns None if no price data exists for that date/ticker combination.
        Handles both date and pd.Timestamp index types.
        """
        if self._prices.empty:
            return None

        try:
            if isinstance(self._prices.index, pd.MultiIndex):
                # Try exact key match first (date, ticker)
                key = (target_date, ticker)
                if key in self._prices.index:
                    return float(self._prices.loc[key, "raw_open"])

                # Try with pd.Timestamp for the date level
                ts_key = (pd.Timestamp(target_date), ticker)
                if ts_key in self._prices.index:
                    return float(self._prices.loc[ts_key, "raw_open"])

                # Fallback: xs lookup with both date and Timestamp
                for dt in (target_date, pd.Timestamp(target_date)):
                    try:
                        day_data = self._prices.xs(dt, level=0)
                        if ticker in day_data.index:
                            return float(day_data.loc[ticker, "raw_open"])
                    except (KeyError, TypeError):
                        pass
            else:
                # Single-index (date only) — assumes single ticker
                if target_date in self._prices.index:
                    return float(self._prices.loc[target_date, "raw_open"])
                ts = pd.Timestamp(target_date)
                if ts in self._prices.index:
                    return float(self._prices.loc[ts, "raw_open"])
        except (KeyError, TypeError):
            pass

        return None


class BrokerExecutor:
    """Phase 2 stub — live broker execution.

    Will implement the Executor protocol when connected to a real broker.
    """

    def submit(self, orders: list[Order], config: Config) -> list[Fill]:
        raise NotImplementedError("BrokerExecutor is not yet implemented")
