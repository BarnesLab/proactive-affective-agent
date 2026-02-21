"""V1 Structured Workflow: fixed Sense → Retrieve → Reason → Decide pipeline.

Every prediction follows the same 4 steps in order. Reproducible and debuggable.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.personal_agent import PersonalAgent


class StructuredWorkflow:
    """Fixed agentic workflow: sense → retrieve_memory → reason → decide."""

    def __init__(self, agent: PersonalAgent) -> None:
        self.agent = agent

    def run(self, sensing_data: dict, context: dict | None = None) -> dict:
        """Execute the full 4-step pipeline.

        Returns:
            Dict with predictions, reasoning trace, and decision.
        """
        sensed = self.sense(sensing_data)
        memory = self.retrieve_memory(sensed, context)
        reasoning = self.reason(sensed, memory, context)
        decision = self.decide(reasoning)

        return {
            "sensed": sensed,
            "memory": memory,
            "reasoning": reasoning,
            "decision": decision,
        }

    def sense(self, sensing_data: dict) -> dict:
        """Step 1: Summarize available sensing features for the current time window."""
        raise NotImplementedError

    def retrieve_memory(self, sensed: dict, context: dict | None = None) -> dict:
        """Step 2: Query user's personal memory for relevant past patterns."""
        raise NotImplementedError

    def reason(self, sensed: dict, memory: dict, context: dict | None = None) -> dict:
        """Step 3: LLM interprets sensing + memory → predicts emotional state and receptivity."""
        raise NotImplementedError

    def decide(self, reasoning: dict) -> dict:
        """Step 4: Based on predictions and confidence, decide whether to intervene."""
        raise NotImplementedError
