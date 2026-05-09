"""Mutable position tracker — fills, splits, dividends, delistings, serialization.

Also provides atomic file persistence: save_state / load_state with .bak backup
and StateCorruption detection on malformed files.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from clenow.errors import InvariantError, StateCorruption
from clenow.types import Fill, Position, Side


class PositionTracker:
    """Track cash and positions across fills, corporate actions, and lifecycle events.

    Mutable, KISS — personal project doesn't need audit-grade immutability.
    """

    def __init__(
        self,
        cash: float = 0.0,
        positions: dict[str, Position] | None = None,
    ) -> None:
        self._cash = cash
        self._positions: dict[str, Position] = dict(positions) if positions else {}

    # ── Fill processing ──────────────────────────────────────────────

    def apply_fills(self, fills: list[Fill]) -> None:
        """Process a list of fills, updating positions and cash.

        BUY:  add/increase position, reduce cash by (fill_price * shares + commission).
        SELL: reduce/remove position, increase cash by (fill_price * shares - commission).
        For additional buys, entry_price is updated as a weighted average.
        """
        for fill in fills:
            if fill.side == Side.BUY:
                self._apply_buy(fill)
            elif fill.side == Side.SELL:
                self._apply_sell(fill)

    def _apply_buy(self, fill: Fill) -> None:
        cost = fill.fill_price * fill.shares + fill.commission
        self._cash -= cost

        if fill.ticker in self._positions:
            existing = self._positions[fill.ticker]
            total_shares = existing.shares + fill.shares
            # Weighted average entry price
            new_entry = (
                (existing.shares * existing.entry_price + fill.shares * fill.fill_price)
                / total_shares
            )
            self._positions[fill.ticker] = Position(
                ticker=fill.ticker,
                shares=total_shares,
                entry_price=new_entry,
                entry_date=existing.entry_date,
                atr_at_entry=existing.atr_at_entry,
            )
        else:
            self._positions[fill.ticker] = Position(
                ticker=fill.ticker,
                shares=fill.shares,
                entry_price=fill.fill_price,
                entry_date=fill.timestamp.date(),
                atr_at_entry=0.0,
            )

    def _apply_sell(self, fill: Fill) -> None:
        proceeds = fill.fill_price * fill.shares - fill.commission
        self._cash += proceeds

        if fill.ticker in self._positions:
            existing = self._positions[fill.ticker]
            remaining = existing.shares - fill.shares
            if remaining <= 0:
                del self._positions[fill.ticker]
            else:
                self._positions[fill.ticker] = Position(
                    ticker=fill.ticker,
                    shares=remaining,
                    entry_price=existing.entry_price,
                    entry_date=existing.entry_date,
                    atr_at_entry=existing.atr_at_entry,
                )

    # ── Corporate actions ────────────────────────────────────────────

    def apply_split(self, ticker: str, ratio: float, date_: date) -> None:
        """Apply a stock split: shares *= ratio, entry_price /= ratio.

        Raises InvariantError if ticker is not in positions.
        """
        if ticker not in self._positions:
            raise InvariantError(ticker, "split")

        pos = self._positions[ticker]
        self._positions[ticker] = Position(
            ticker=ticker,
            shares=round(pos.shares * ratio),
            entry_price=pos.entry_price / ratio,
            entry_date=pos.entry_date,
            atr_at_entry=pos.atr_at_entry / ratio,
        )

    def apply_dividend(self, ticker: str, dividend_per_share: float, date_: date) -> None:
        """Apply a cash dividend: cash += shares * dividend_per_share.

        Raises InvariantError if ticker is not in positions.
        """
        if ticker not in self._positions:
            raise InvariantError(ticker, "dividend")

        pos = self._positions[ticker]
        self._cash += pos.shares * dividend_per_share

    def apply_delisting(self, ticker: str, last_price: float, date_: date) -> None:
        """Force-close a delisted position: cash += last_price * shares, remove position."""
        if ticker not in self._positions:
            return  # already gone — idempotent

        pos = self._positions[ticker]
        self._cash += last_price * pos.shares
        del self._positions[ticker]

    # ── Accessors ────────────────────────────────────────────────────

    def get_positions(self) -> dict[str, Position]:
        """Return a copy of the current positions dict."""
        return dict(self._positions)

    def get_cash(self) -> float:
        """Return current cash balance."""
        return self._cash

    def get_equity(self, prices: dict[str, float]) -> float:
        """Return total equity = cash + sum(shares * prices[ticker])."""
        position_value = sum(
            pos.shares * prices[pos.ticker]
            for pos in self._positions.values()
            if pos.ticker in prices
        )
        return self._cash + position_value

    # ── Serialization ────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize tracker state to JSON string."""
        state = {
            "cash": self._cash,
            "positions": {
                ticker: {
                    "ticker": pos.ticker,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "entry_date": pos.entry_date.isoformat(),
                    "atr_at_entry": pos.atr_at_entry,
                }
                for ticker, pos in self._positions.items()
            },
        }
        return json.dumps(state, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PositionTracker:
        """Deserialize a PositionTracker from JSON string.

        Raises StateCorruption if the JSON is malformed or missing required fields.
        """
        try:
            state = json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as exc:
            raise StateCorruption("<json_string>", f"Invalid JSON: {exc}") from exc

        _validate_state_dict(state, "<json_string>")

        positions = {}
        for ticker, pos_data in state["positions"].items():
            positions[ticker] = Position(
                ticker=pos_data["ticker"],
                shares=pos_data["shares"],
                entry_price=pos_data["entry_price"],
                entry_date=date.fromisoformat(pos_data["entry_date"]),
                atr_at_entry=pos_data["atr_at_entry"],
            )
        return cls(cash=state["cash"], positions=positions)


# ── State file persistence (atomic write with .bak backup) ────────────

_REQUIRED_POSITION_FIELDS = {"ticker", "shares", "entry_price", "entry_date", "atr_at_entry"}


def _validate_state_dict(state: dict, path: str) -> None:
    """Validate a deserialized state dict has required structure.

    Raises StateCorruption if missing required fields or has invalid types.
    """
    if not isinstance(state, dict):
        raise StateCorruption(path, "State is not a JSON object")

    if "cash" not in state:
        raise StateCorruption(path, "Missing required field: cash")

    if not isinstance(state["cash"], (int, float)):
        raise StateCorruption(path, f"Field 'cash' must be numeric, got {type(state['cash']).__name__}")

    if "positions" not in state:
        raise StateCorruption(path, "Missing required field: positions")

    if not isinstance(state["positions"], dict):
        raise StateCorruption(path, f"Field 'positions' must be an object, got {type(state['positions']).__name__}")

    for ticker, pos_data in state["positions"].items():
        if not isinstance(pos_data, dict):
            raise StateCorruption(path, f"Position '{ticker}' is not an object")

        missing = _REQUIRED_POSITION_FIELDS - set(pos_data.keys())
        if missing:
            raise StateCorruption(path, f"Position '{ticker}' missing fields: {sorted(missing)}")

        if not isinstance(pos_data["shares"], int):
            raise StateCorruption(path, f"Position '{ticker}' shares must be int, got {type(pos_data['shares']).__name__}")

        if not isinstance(pos_data["entry_price"], (int, float)):
            raise StateCorruption(path, f"Position '{ticker}' entry_price must be numeric")

        try:
            date.fromisoformat(pos_data["entry_date"])
        except (ValueError, TypeError) as exc:
            raise StateCorruption(path, f"Position '{ticker}' has invalid entry_date: {pos_data['entry_date']}") from exc

        if not isinstance(pos_data["atr_at_entry"], (int, float)):
            raise StateCorruption(path, f"Position '{ticker}' atr_at_entry must be numeric")


def save_state(tracker: PositionTracker, path: str | Path) -> None:
    """Persist tracker state to a file with atomic write and .bak backup.

    Writes to a .tmp file first, then renames (atomic on POSIX).
    Keeps a .bak backup of the previous version if it exists.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    bak_path = path.with_suffix(path.suffix + ".bak")

    json_str = tracker.to_json()

    # Write to temp file
    tmp_path.write_text(json_str, encoding="utf-8")

    # Create backup of existing file
    if path.exists():
        # On POSIX, rename is atomic
        os.replace(str(path), str(bak_path))

    # Atomic rename temp -> final
    os.replace(str(tmp_path), str(path))


def load_state(path: str | Path) -> PositionTracker:
    """Load a PositionTracker from a state file.

    Validates the file contents and raises StateCorruption on malformed data.
    Falls back to .bak if the primary file is missing or corrupt.
    """
    path = Path(path)
    bak_path = path.with_suffix(path.suffix + ".bak")

    # Try primary file first
    for candidate, label in [(path, str(path)), (bak_path, str(bak_path))]:
        if candidate.exists():
            try:
                json_str = candidate.read_text(encoding="utf-8")
                return PositionTracker.from_json(json_str)
            except StateCorruption:
                if candidate == path and bak_path.exists():
                    # Primary is corrupt, try backup
                    continue
                raise

    raise StateCorruption(str(path), "No state file found (neither primary nor backup)")
