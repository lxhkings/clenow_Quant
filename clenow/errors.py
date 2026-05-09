"""Custom exceptions for the Clenow system.

Fail-loud policy: silent failure is the enemy of live trading.
Every high-risk path must raise, not silently continue.
"""


class DataAccessError(Exception):
    """DB connection or query failure after retries."""


class OrderRejection(Exception):
    """Order cannot be filled (e.g., missing price on execution date)."""

    def __init__(self, ticker: str, reason: str) -> None:
        self.ticker = ticker
        self.reason = reason
        super().__init__(f"Order rejected for {ticker}: {reason}")


class InvariantError(Exception):
    """An invariant was violated (e.g., corporate action for unknown ticker)."""

    def __init__(self, ticker: str, event_type: str) -> None:
        self.ticker = ticker
        self.event_type = event_type
        super().__init__(f"Invariant violation: {event_type} for unknown ticker {ticker}")


class StateCorruption(Exception):
    """State file is malformed or has missing critical fields."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"State corruption at {path}: {reason}")
