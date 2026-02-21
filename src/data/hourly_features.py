"""Hourly feature extraction from raw minute-level sensing data.

PLACEHOLDER: waiting for colleague to provide raw minute-level data and
hourly feature extraction code. Once available:
1. HourlyFeatureLoader loads raw data and extracts hourly features
2. Features are aligned to EMA timestamps with configurable lookback
3. Output formats serve both LLM prompts (natural language) and ML (numeric matrix)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HourlySensingWindow:
    """Sensing features aggregated into an hourly window."""

    hour_start: str  # e.g., "2024-03-15T14:00"
    hour_end: str

    # Activity
    steps: int | None = None
    stationary_min: float | None = None
    walking_min: float | None = None

    # Heart rate (if available)
    hr_mean: float | None = None
    hr_min: float | None = None
    hr_max: float | None = None

    # Screen
    screen_on_min: float | None = None
    screen_sessions: int | None = None

    # Location
    at_home: bool | None = None
    distance_km: float | None = None


@dataclass
class SensingContext:
    """Multi-hour sensing context aligned to an EMA entry."""

    study_id: int
    ema_timestamp: str
    lookback_hours: int = 24
    windows: list[HourlySensingWindow] = field(default_factory=list)

    def to_text(self) -> str:
        """Format as natural language for LLM prompts."""
        if not self.windows:
            return "No hourly sensing data available."

        lines = [f"Sensing data for the past {self.lookback_hours} hours:"]
        for w in self.windows:
            parts = [f"  [{w.hour_start} - {w.hour_end}]"]
            if w.steps is not None:
                parts.append(f"steps={w.steps}")
            if w.walking_min is not None:
                parts.append(f"walking={w.walking_min:.0f}min")
            if w.screen_on_min is not None:
                parts.append(f"screen={w.screen_on_min:.0f}min")
            if w.at_home is not None:
                parts.append(f"{'at home' if w.at_home else 'away'}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def to_feature_vector(self) -> dict[str, float]:
        """Flatten to numeric feature dict for ML baselines."""
        # Will produce features like: hour_0_steps, hour_0_walking_min, ...
        features = {}
        for i, w in enumerate(self.windows):
            prefix = f"h{i}"
            if w.steps is not None:
                features[f"{prefix}_steps"] = float(w.steps)
            if w.stationary_min is not None:
                features[f"{prefix}_stationary_min"] = w.stationary_min
            if w.walking_min is not None:
                features[f"{prefix}_walking_min"] = w.walking_min
            if w.screen_on_min is not None:
                features[f"{prefix}_screen_on_min"] = w.screen_on_min
            if w.at_home is not None:
                features[f"{prefix}_at_home"] = float(w.at_home)
        return features


class HourlyFeatureLoader:
    """Load raw minute-level data and extract hourly features.

    PLACEHOLDER: implement when raw data becomes available.
    """

    def __init__(self, raw_data_dir: str | None = None) -> None:
        self.raw_data_dir = raw_data_dir

    def get_features_for_ema(
        self,
        study_id: int,
        timestamp: str,
        lookback_hours: int = 24,
    ) -> SensingContext:
        """Get hourly features aligned to an EMA timestamp.

        Args:
            study_id: User's Study_ID.
            timestamp: EMA timestamp (ISO format).
            lookback_hours: Hours of history to include.

        Returns:
            SensingContext with hourly windows.
        """
        raise NotImplementedError(
            "Waiting for raw minute-level data from colleague. "
            "Use daily aggregate features for now."
        )

    def format_hourly_sensing_text(
        self,
        study_id: int,
        timestamp: str,
        lookback_hours: int = 24,
    ) -> str:
        """Get formatted hourly sensing text for LLM prompts."""
        ctx = self.get_features_for_ema(study_id, timestamp, lookback_hours)
        return ctx.to_text()

    def get_feature_matrix(
        self,
        study_id: int,
        timestamp: str,
        lookback_hours: int = 24,
    ) -> dict[str, float]:
        """Get numeric feature dict for ML baselines."""
        ctx = self.get_features_for_ema(study_id, timestamp, lookback_hours)
        return ctx.to_feature_vector()
