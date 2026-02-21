"""Data loader: loads all data types (EMA, sensing, baseline, processed).

Provides unified access to all data sources with consistent formatting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Loads and caches all data types from the data directory."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"

    def load_daily_ema(self) -> pd.DataFrame:
        """Load daily EMA survey data."""
        raise NotImplementedError

    def load_weekly_ema(self) -> pd.DataFrame:
        """Load weekly EMA survey data."""
        raise NotImplementedError

    def load_sensing(self, sensor_name: str) -> pd.DataFrame:
        """Load raw sensing data for a specific sensor."""
        raise NotImplementedError

    def load_all_sensing(self) -> dict[str, pd.DataFrame]:
        """Load all 8 sensing data files. Returns {sensor_name: DataFrame}."""
        raise NotImplementedError

    def load_baseline(self) -> pd.DataFrame:
        """Load baseline trait questionnaire data."""
        raise NotImplementedError

    def load_split(self, group: int, split: str = "train") -> pd.DataFrame:
        """Load a train/test split file.

        Args:
            group: Fold number (1-5).
            split: "train" or "test".
        """
        raise NotImplementedError

    def load_memory_documents(self) -> dict[str, str]:
        """Load all 754 pre-generated memory documents. Returns {filename: content}."""
        raise NotImplementedError

    def get_user_ids(self) -> list[str]:
        """Get list of all participant IDs from the EMA data."""
        raise NotImplementedError
