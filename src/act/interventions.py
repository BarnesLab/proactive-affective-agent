"""Intervention content templates.

Defines types of interventions and their content generation.
"""

from __future__ import annotations

from enum import Enum


class InterventionType(Enum):
    """Types of interventions the agent can deliver."""

    BREATHING = "breathing"
    MINDFULNESS = "mindfulness"
    COGNITIVE_REAPPRAISAL = "cognitive_reappraisal"
    SOCIAL_SUPPORT = "social_support"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    CHECK_IN = "check_in"


class InterventionGenerator:
    """Generates intervention content based on type and user context."""

    def generate(
        self,
        intervention_type: InterventionType,
        emotional_state: dict,
        user_profile: dict | None = None,
    ) -> str:
        """Generate personalized intervention content.

        Args:
            intervention_type: Type of intervention to generate.
            emotional_state: Current predicted emotional state.
            user_profile: User baseline profile for personalization.

        Returns:
            Intervention content text.
        """
        raise NotImplementedError

    def select_type(self, emotional_state: dict) -> InterventionType:
        """Select the most appropriate intervention type given emotional state."""
        raise NotImplementedError
