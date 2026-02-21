"""Dataclass definitions for structured data objects.

Defines typed schemas for EMA responses, sensing features,
predictions, and evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EMAResponse:
    """A single EMA survey response."""

    user_id: str
    timestamp: datetime
    window: str  # "morning", "afternoon", "evening"
    valence: float | None = None
    arousal: float | None = None
    stress: float | None = None
    loneliness: float | None = None
    er_desire: bool | None = None
    availability: bool | None = None
    receptivity: bool | None = None


@dataclass
class SensingSnapshot:
    """Aggregated sensing features for a time window."""

    user_id: str
    timestamp: datetime
    window_minutes: int
    features: dict = field(default_factory=dict)
    missing_sensors: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """Agent's prediction for an EMA window."""

    user_id: str
    timestamp: datetime
    window: str
    emotional_states: dict = field(default_factory=dict)
    receptivity: bool | None = None
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class EvaluationResult:
    """Result of comparing a prediction to ground truth."""

    user_id: str
    timestamp: datetime
    prediction: Prediction | None = None
    ground_truth: EMAResponse | None = None
    accuracy: dict = field(default_factory=dict)
    notes: str = ""
