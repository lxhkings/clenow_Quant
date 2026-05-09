"""State file management — atomic read/write for ~/.clenow/positions.json.

Fail-loud: StateCorruption on malformed data. Never fallback to empty state.
Delegates core persistence to clenow.portfolio.state.save_state / load_state,
adding live-specific behavior (default path, first-run empty tracker).
"""

from __future__ import annotations

from pathlib import Path

from clenow.errors import StateCorruption
from clenow.portfolio.state import PositionTracker
from clenow.portfolio.state import load_state as _load_state
from clenow.portfolio.state import save_state as _save_state

DEFAULT_STATE_DIR = Path.home() / ".clenow"
DEFAULT_STATE_PATH = DEFAULT_STATE_DIR / "positions.json"


def load_state(path: Path | str | None = None) -> PositionTracker:
    """Load PositionTracker from state file.

    Raises StateCorruption if the file exists but is malformed.
    Returns a fresh (empty) tracker if the file does not exist (first run).
    Falls back to .bak if primary file is corrupt.
    """
    path = Path(path) if path else DEFAULT_STATE_PATH

    if not path.exists():
        # First run — no state file yet. This is fine.
        bak_path = path.with_suffix(path.suffix + ".bak")
        if bak_path.exists():
            return _load_state(bak_path)
        return PositionTracker()

    return _load_state(path)


def save_state(tracker: PositionTracker, path: Path | str | None = None) -> None:
    """Save PositionTracker to state file with atomic write and backup.

    Delegates to portfolio.state.save_state for atomic write and .bak backup.
    Ensures the state directory exists before writing.
    """
    path = Path(path) if path else DEFAULT_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    _save_state(tracker, path)
