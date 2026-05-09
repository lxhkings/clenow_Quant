"""State file management — atomic read/write for ~/.clenow/positions.json.

Fail-loud: StateCorruption on malformed data. Never fallback to empty state.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from clenow.errors import StateCorruption
from clenow.portfolio.state import PositionTracker
from clenow.types import Position

DEFAULT_STATE_DIR = Path.home() / ".clenow"
DEFAULT_STATE_PATH = DEFAULT_STATE_DIR / "positions.json"


def load_state(path: Path | str | None = None) -> PositionTracker:
    """Load PositionTracker from state file.

    Raises StateCorruption if the file exists but is malformed.
    Returns a fresh (empty) tracker if the file does not exist.
    """
    path = Path(path) if path else DEFAULT_STATE_PATH

    if not path.exists():
        return PositionTracker()

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StateCorruption(str(path), f"Cannot read state file: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StateCorruption(str(path), f"Invalid JSON: {exc}") from exc

    _validate_state(data, path)

    positions: dict[str, Position] = {}
    for ticker, pos_data in data["positions"].items():
        positions[ticker] = Position(
            ticker=ticker,
            shares=pos_data["shares"],
            entry_price=pos_data["entry_price"],
            entry_date=date.fromisoformat(pos_data["entry_date"]),
            atr_at_entry=pos_data.get("atr_at_entry", 0.0),
        )

    return PositionTracker(cash=data["cash"], positions=positions)


def save_state(tracker: PositionTracker, path: Path | str | None = None) -> None:
    """Save PositionTracker to state file with atomic write and backup.

    Atomic write: write to .tmp then rename.
    Backup: keep .bak of previous version.
    """
    path = Path(path) if path else DEFAULT_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "as_of": date.today().isoformat(),
        "cash": tracker.get_cash(),
        "positions": {
            ticker: {
                "ticker": pos.ticker,
                "shares": pos.shares,
                "entry_price": pos.entry_price,
                "entry_date": pos.entry_date.isoformat(),
                "atr_at_entry": pos.atr_at_entry,
            }
            for ticker, pos in tracker.get_positions().items()
        },
    }

    json_str = json.dumps(state, sort_keys=True, indent=2) + "\n"

    # Backup existing state
    if path.exists():
        bak_path = path.with_suffix(".bak")
        try:
            os.replace(str(path), str(bak_path))
        except OSError:
            pass  # best-effort backup

    # Atomic write: tmp → rename
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json_str, encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    except OSError as exc:
        # Clean up tmp if rename fails
        tmp_path.unlink(missing_ok=True)
        raise StateCorruption(str(path), f"Failed to write state: {exc}") from exc


def _validate_state(data: dict, path: Path) -> None:
    """Validate state dict has required fields and correct types.

    Raises StateCorruption on any issue.
    """
    if not isinstance(data, dict):
        raise StateCorruption(str(path), "State is not a JSON object")

    if "cash" not in data:
        raise StateCorruption(str(path), "Missing required field: cash")
    if not isinstance(data["cash"], (int, float)):
        raise StateCorruption(str(path), "Field 'cash' must be a number")

    if "positions" not in data:
        raise StateCorruption(str(path), "Missing required field: positions")
    if not isinstance(data["positions"], dict):
        raise StateCorruption(str(path), "Field 'positions' must be an object")

    for ticker, pos_data in data["positions"].items():
        if not isinstance(pos_data, dict):
            raise StateCorruption(
                str(path), f"Position for {ticker} must be an object"
            )
        for field in ("shares", "entry_price", "entry_date"):
            if field not in pos_data:
                raise StateCorruption(
                    str(path), f"Position {ticker} missing required field: {field}"
                )
        if not isinstance(pos_data["shares"], int) or pos_data["shares"] <= 0:
            raise StateCorruption(
                str(path), f"Position {ticker} shares must be a positive integer"
            )
        if not isinstance(pos_data["entry_price"], (int, float)):
            raise StateCorruption(
                str(path), f"Position {ticker} entry_price must be a number"
            )
        try:
            date.fromisoformat(pos_data["entry_date"])
        except (ValueError, TypeError):
            raise StateCorruption(
                str(path), f"Position {ticker} entry_date is not a valid date"
            )
