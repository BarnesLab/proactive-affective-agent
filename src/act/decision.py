"""Receptivity prediction and intervention decision logic.

Takes emotional state predictions + confidence scores → decides whether
to intervene, what type of intervention, and with what content.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    """An intervention decision."""

    should_intervene: bool
    intervention_type: str | None = None
    content: str | None = None
    confidence: float = 0.0
    reasoning: str = ""


class DecisionMaker:
    """Decides whether and how to intervene based on predictions."""

    def __init__(
        self,
        receptivity_threshold: float = 0.6,
        intervention_threshold: float = 0.7,
    ) -> None:
        self.receptivity_threshold = receptivity_threshold
        self.intervention_threshold = intervention_threshold

    def decide(self, predictions: dict, user_context: dict | None = None) -> Decision:
        """Make an intervention decision based on predictions.

        Args:
            predictions: Dict with emotional states, receptivity, and confidence scores.
            user_context: Additional context (time of day, recent interactions, etc.)

        Returns:
            Decision object.
        """
        raise NotImplementedError

    def compute_receptivity(self, desire: bool, availability: bool) -> bool:
        """Compute receptivity = Desire AND Availability."""
        return desire and availability
