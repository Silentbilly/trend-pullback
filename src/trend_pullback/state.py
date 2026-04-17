"""
Live trading state persistence for Trend Pullback Pro.

Saves and loads the current trade state to a JSON file so the bot
can survive restarts without losing track of open positions and orders.

State contains:
  - Whether a position is currently open
  - Entry price, side, size
  - TP order ID and SL order ID
  - Signal bar datetime (for anti-duplicate)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LiveState:
    """Mutable live trading state."""

    # Position
    in_position: bool = False
    position_side: str = "none"      # "long" or "short"
    position_size: float = 0.0
    entry_price: float = 0.0

    # Open exit orders
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None

    # Anti-duplicate: track last signal bar datetime as ISO string
    last_signal_bar: Optional[str] = None


class StateManager:
    """Load and persist LiveState to a JSON file.

    Args:
        path: Path to the state JSON file.
               Created automatically if it does not exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._state = self._load()

    @property
    def state(self) -> LiveState:
        return self._state

    def save(self) -> None:
        """Persist current state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self._state), f, indent=2, default=str)
        logger.debug("State saved → %s", self._path)

    def reset(self) -> None:
        """Reset to clean (no position) state and persist."""
        self._state = LiveState()
        self.save()
        logger.info("State reset")

    # ------------------------------------------------------------------
    # Convenience mutators — each calls save() automatically
    # ------------------------------------------------------------------

    def on_entry(
        self,
        side: str,
        size: float,
        entry_price: float,
        tp_order_id: str,
        sl_order_id: str,
        signal_bar: str,
    ) -> None:
        """Record a new open position."""
        self._state.in_position = True
        self._state.position_side = side
        self._state.position_size = size
        self._state.entry_price = entry_price
        self._state.tp_order_id = tp_order_id
        self._state.sl_order_id = sl_order_id
        self._state.last_signal_bar = signal_bar
        self.save()
        logger.info(
            "State: ENTERED %s  size=%.6f  entry=%.5f",
            side, size, entry_price,
        )

    def on_exit(self, exit_type: str = "unknown") -> None:
        """Record position closed (TP or SL hit)."""
        logger.info(
            "State: EXITED (%s)  was %s  entry=%.5f",
            exit_type, self._state.position_side, self._state.entry_price,
        )
        self._state.in_position = False
        self._state.position_side = "none"
        self._state.position_size = 0.0
        self._state.entry_price = 0.0
        self._state.tp_order_id = None
        self._state.sl_order_id = None
        self.save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> LiveState:
        if not self._path.exists():
            logger.info("No state file found at %s — starting fresh", self._path)
            return LiveState()
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            state = LiveState(**data)
            logger.info(
                "State loaded from %s — in_position=%s  side=%s",
                self._path, state.in_position, state.position_side,
            )
            return state
        except Exception as exc:
            logger.warning("Failed to load state (%s) — starting fresh", exc)
            return LiveState()
