"""Dataclass definitions for structured data objects.

Defines typed schemas for EMA responses, sensing data, predictions,
and evaluation results matching the actual CSV column structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class EMAResponse:
    """A single EMA survey response row from the processed split CSV."""

    study_id: int
    timestamp_local: datetime
    date_local: date
    group: int

    # Continuous targets
    PANAS_Pos: float | None = None
    PANAS_Neg: float | None = None
    ER_desire: float | None = None

    # Raw emotional items (1-8 scale)
    happy: float | None = None
    cheerful: float | None = None
    pleased: float | None = None
    sad: float | None = None
    afraid: float | None = None
    miserable: float | None = None
    worried: float | None = None
    grateful: float | None = None
    lonely: float | None = None

    # Context
    interactions_quality: float | None = None
    pain: float | None = None
    forecasting: float | None = None
    emotion_driver: str | None = None
    INT_availability: str | None = None  # "yes" / "no"

    # Binary state flags (individual level)
    Individual_level_PA_State: bool | None = None
    Individual_level_NA_State: bool | None = None
    Individual_level_happy_State: bool | None = None
    Individual_level_sad_State: bool | None = None
    Individual_level_afraid_State: bool | None = None
    Individual_level_miserable_State: bool | None = None
    Individual_level_worried_State: bool | None = None
    Individual_level_cheerful_State: bool | None = None
    Individual_level_pleased_State: bool | None = None
    Individual_level_grateful_State: bool | None = None
    Individual_level_lonely_State: bool | None = None
    Individual_level_interactions_quality_State: bool | None = None
    Individual_level_pain_State: bool | None = None
    Individual_level_forecasting_State: bool | None = None
    Individual_level_ER_desire_State: bool | None = None

    def get_ground_truth(self) -> dict[str, Any]:
        """Extract all prediction targets as a dict."""
        return {
            "PANAS_Pos": self.PANAS_Pos,
            "PANAS_Neg": self.PANAS_Neg,
            "ER_desire": self.ER_desire,
            "INT_availability": self.INT_availability,
            "Individual_level_PA_State": self.Individual_level_PA_State,
            "Individual_level_NA_State": self.Individual_level_NA_State,
            "Individual_level_happy_State": self.Individual_level_happy_State,
            "Individual_level_sad_State": self.Individual_level_sad_State,
            "Individual_level_afraid_State": self.Individual_level_afraid_State,
            "Individual_level_miserable_State": self.Individual_level_miserable_State,
            "Individual_level_worried_State": self.Individual_level_worried_State,
            "Individual_level_cheerful_State": self.Individual_level_cheerful_State,
            "Individual_level_pleased_State": self.Individual_level_pleased_State,
            "Individual_level_grateful_State": self.Individual_level_grateful_State,
            "Individual_level_lonely_State": self.Individual_level_lonely_State,
            "Individual_level_interactions_quality_State": self.Individual_level_interactions_quality_State,
            "Individual_level_pain_State": self.Individual_level_pain_State,
            "Individual_level_forecasting_State": self.Individual_level_forecasting_State,
            "Individual_level_ER_desire_State": self.Individual_level_ER_desire_State,
        }


@dataclass
class SensingDay:
    """Aggregated sensing data for one user on one day (daily granularity)."""

    id_participant: str
    date: date

    # Accelerometer-based sleep
    accel_sleep_duration_min: float | None = None
    accel_count: int | None = None

    # Sleep sensor
    sleep_duration_min: float | None = None

    # Android sleep
    android_sleep_min: float | None = None
    android_sleep_status: str | None = None

    # GPS / mobility
    gps_captures: int | None = None
    travel_events: int | None = None
    travel_km: float | None = None
    travel_minutes: float | None = None
    home_minutes: float | None = None
    max_distance_from_home_km: float | None = None
    location_variance: float | None = None

    # Screen
    screen_sessions: int | None = None
    screen_minutes: float | None = None
    screen_max_session_min: float | None = None
    screen_mean_session_min: float | None = None

    # Motion / activity
    stationary_min: float | None = None
    walking_min: float | None = None
    automotive_min: float | None = None
    running_min: float | None = None
    cycling_min: float | None = None

    # Key input
    chars_typed: int | None = None
    words_typed: int | None = None
    negative_words: int | None = None
    positive_words: int | None = None
    prop_negative: float | None = None
    prop_positive: float | None = None

    # App usage (aggregated)
    total_app_seconds: float | None = None
    top_apps: list[tuple[str, float]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return non-None fields as a flat dict for prompt formatting."""
        result = {}
        for k, v in self.__dict__.items():
            if v is not None and k not in ("id_participant", "date", "top_apps"):
                result[k] = v
            elif k == "top_apps" and v:
                result["top_apps"] = v
        return result


@dataclass
class PredictionOutput:
    """Structured output from the LLM prediction."""

    # Continuous predictions
    PANAS_Pos: float | None = None
    PANAS_Neg: float | None = None
    ER_desire: float | None = None

    # Binary state predictions
    Individual_level_PA_State: bool | None = None
    Individual_level_NA_State: bool | None = None
    Individual_level_happy_State: bool | None = None
    Individual_level_sad_State: bool | None = None
    Individual_level_afraid_State: bool | None = None
    Individual_level_miserable_State: bool | None = None
    Individual_level_worried_State: bool | None = None
    Individual_level_cheerful_State: bool | None = None
    Individual_level_pleased_State: bool | None = None
    Individual_level_grateful_State: bool | None = None
    Individual_level_lonely_State: bool | None = None
    Individual_level_interactions_quality_State: bool | None = None
    Individual_level_pain_State: bool | None = None
    Individual_level_forecasting_State: bool | None = None
    Individual_level_ER_desire_State: bool | None = None

    # Availability
    INT_availability: str | None = None

    # Metadata
    reasoning: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dict for evaluation."""
        return {k: v for k, v in self.__dict__.items() if k not in ("reasoning", "confidence")}


@dataclass
class UserProfile:
    """Baseline demographic and trait profile for a user."""

    study_id: int
    age: int | None = None
    gender: str | None = None
    cancer_diagnosis: str | None = None
    cancer_stage: str | None = None
    cancer_years: float | None = None
    depression_phq8: float | None = None
    anxiety_gad7: float | None = None
    trait_positive_affect: float | None = None
    trait_negative_affect: float | None = None
    extraversion: float | None = None
    neuroticism_stability: float | None = None
    social_support: float | None = None
    self_efficacy: float | None = None

    def to_text(self) -> str:
        """Format profile as natural language for prompts."""
        parts = [f"User {self.study_id}:"]
        if self.age:
            parts.append(f"Age {self.age}")
        if self.gender:
            parts.append(f"Gender: {self.gender}")
        if self.cancer_diagnosis:
            parts.append(f"Cancer: {self.cancer_diagnosis}")
        if self.cancer_stage:
            parts.append(f"Stage: {self.cancer_stage}")
        if self.depression_phq8 is not None:
            parts.append(f"Depression (PHQ-8): {self.depression_phq8}")
        if self.anxiety_gad7 is not None:
            parts.append(f"Anxiety (GAD-7): {self.anxiety_gad7}")
        if self.trait_positive_affect is not None:
            parts.append(f"Trait PA: {self.trait_positive_affect}")
        if self.trait_negative_affect is not None:
            parts.append(f"Trait NA: {self.trait_negative_affect}")
        if self.extraversion is not None:
            parts.append(f"Extraversion: {self.extraversion}")
        if self.social_support is not None:
            parts.append(f"Social support: {self.social_support}")
        return " | ".join(parts)
