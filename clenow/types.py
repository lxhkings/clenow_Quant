"""Core data types for the Clenow backtesting system."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"


@dataclass(frozen=True)
class Order:
    ticker: str
    shares: int
    side: Side
    order_type: OrderType
    target_date: date


@dataclass(frozen=True)
class Fill:
    ticker: str
    shares: int
    fill_price: float
    timestamp: datetime
    commission: float
    slippage_bps: float
    side: Side


@dataclass
class Position:
    ticker: str
    shares: int
    entry_price: float
    entry_date: date
    atr_at_entry: float


@dataclass(frozen=True)
class TargetPortfolio:
    positions: dict[str, int]
    as_of: date
