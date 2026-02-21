"""Adapt per-user confidence and receptivity thresholds over time.

If the agent is consistently over- or under-confident, adjusts thresholds
to improve calibration.
"""

from __future__ import annotations


class ThresholdTuner:
    """Adapts per-user prediction thresholds based on accumulated evaluations."""

    def __init__(
        self,
        initial_receptivity_threshold: float = 0.6,
        initial_intervention_threshold: float = 0.7,
        learning_rate: float = 0.1,
    ) -> None:
        self.receptivity_threshold = initial_receptivity_threshold
        self.intervention_threshold = initial_intervention_threshold
        self.learning_rate = learning_rate
        self.history: list[dict] = []

    def update(self, evaluation: dict) -> None:
        """Update thresholds based on a new evaluation result.

        Args:
            evaluation: Self-evaluation dict with accuracy and calibration info.
        """
        raise NotImplementedError

    def get_thresholds(self) -> dict:
        """Return current thresholds."""
        return {
            "receptivity_threshold": self.receptivity_threshold,
            "intervention_threshold": self.intervention_threshold,
        }
