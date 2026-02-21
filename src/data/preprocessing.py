"""State variable derivation and receptivity labeling.

Derives emotional state variables from raw EMA responses and
computes the Receptivity = Desire ∧ Availability label.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class Preprocessor:
    """Derives state variables and receptivity labels from raw EMA data."""

    def derive_receptivity(self, ema_df: pd.DataFrame) -> pd.DataFrame:
        """Add receptivity column: Receptivity = Desire AND Availability.

        Uses ER_desire_State and INT_availability columns.
        """
        raise NotImplementedError

    def derive_emotional_states(self, ema_df: pd.DataFrame) -> pd.DataFrame:
        """Derive emotional state variables (valence, arousal, stress, loneliness)."""
        raise NotImplementedError

    def preprocess_ema(self, ema_df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing pipeline for EMA data."""
        raise NotImplementedError

    def preprocess_sensing(self, sensing_df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
        """Preprocess a single sensor's raw data (timestamp parsing, cleaning, etc.)."""
        raise NotImplementedError
