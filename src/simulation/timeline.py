"""Timeline builder: constructs per-user chronological timeline of sensing windows and EMA events.

Aligns sensing data with EMA timestamps to create an ordered sequence of events
that the simulator can replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class TimelineEvent:
    """A single event in the user timeline."""

    user_id: str
    timestamp: datetime
    event_type: str  # "ema_window" or "sensing_batch"
    window: str | None = None  # "morning", "afternoon", "evening"
    data: dict | None = None


class TimelineBuilder:
    """Builds per-user timelines from raw EMA and sensing data."""

    def __init__(self, ema_data: pd.DataFrame, sensing_data: dict[str, pd.DataFrame]) -> None:
        self.ema_data = ema_data
        self.sensing_data = sensing_data

    def build(self, user_id: str) -> list[TimelineEvent]:
        """Build chronological timeline for a single user."""
        raise NotImplementedError

    def build_all(self) -> dict[str, list[TimelineEvent]]:
        """Build timelines for all users. Returns {user_id: [events]}."""
        raise NotImplementedError

    def _identify_ema_windows(self, user_id: str) -> list[TimelineEvent]:
        """Extract EMA window events for a user."""
        raise NotImplementedError

    def _gather_sensing_before(
        self, user_id: str, timestamp: datetime, lookback_hours: int = 4
    ) -> dict:
        """Gather all sensing data in the lookback window before a timestamp."""
        raise NotImplementedError
