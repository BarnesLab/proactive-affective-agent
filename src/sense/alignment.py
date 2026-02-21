"""Align sensing windows with EMA timestamps.

Ensures that for each EMA window (morning/afternoon/evening), we select
the correct slice of sensing data that precedes it.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class SensingAligner:
    """Aligns sensing data to EMA trigger windows."""

    def __init__(self, lookback_hours: int = 4) -> None:
        self.lookback_hours = lookback_hours

    def align(
        self,
        sensing_df: pd.DataFrame,
        ema_timestamp: datetime,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Select sensing data in the lookback window before an EMA timestamp.

        Args:
            sensing_df: Full sensing DataFrame for one sensor.
            ema_timestamp: Timestamp of the EMA trigger.
            timestamp_col: Name of the timestamp column.

        Returns:
            Filtered DataFrame containing only rows within the lookback window.
        """
        raise NotImplementedError

    def align_all_sensors(
        self,
        sensing_data: dict[str, pd.DataFrame],
        ema_timestamp: datetime,
    ) -> dict[str, pd.DataFrame]:
        """Align all sensor DataFrames to an EMA timestamp."""
        raise NotImplementedError
