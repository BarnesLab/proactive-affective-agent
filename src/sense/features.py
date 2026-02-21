"""Feature extraction per sensor type.

Extracts interpretable features from raw sensing streams:
- Accelerometer → activity level, movement variance
- GPS → location entropy, distance traveled, time at home
- Sleep → duration, quality, timing
- Screen → on-time, unlock count
- App usage → total minutes, social app usage
- etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class FeatureExtractor:
    """Extracts features from raw sensing data for a given time window."""

    def __init__(self, sensor_config: dict) -> None:
        self.sensor_config = sensor_config

    def extract(self, sensor_name: str, raw_data: pd.DataFrame, window_minutes: int = 240) -> dict:
        """Extract features from a single sensor's raw data within a time window.

        Args:
            sensor_name: Name of the sensor (e.g., "accelerometer", "gps").
            raw_data: Raw sensor DataFrame filtered to the time window.
            window_minutes: Duration of the aggregation window.

        Returns:
            Dict of feature_name → value.
        """
        raise NotImplementedError

    def extract_all(self, sensing_data: dict[str, pd.DataFrame], window_minutes: int = 240) -> dict:
        """Extract features from all available sensors.

        Args:
            sensing_data: {sensor_name: DataFrame} for the current time window.
            window_minutes: Duration of the aggregation window.

        Returns:
            Flat dict of all features across sensors.
        """
        raise NotImplementedError

    def _extract_accelerometer(self, data: pd.DataFrame) -> dict:
        raise NotImplementedError

    def _extract_gps(self, data: pd.DataFrame) -> dict:
        raise NotImplementedError

    def _extract_sleep(self, data: pd.DataFrame) -> dict:
        raise NotImplementedError

    def _extract_screen(self, data: pd.DataFrame) -> dict:
        raise NotImplementedError

    def _extract_app_usage(self, data: pd.DataFrame) -> dict:
        raise NotImplementedError

    def _extract_motion(self, data: pd.DataFrame) -> dict:
        raise NotImplementedError

    def _extract_key_input(self, data: pd.DataFrame) -> dict:
        raise NotImplementedError
