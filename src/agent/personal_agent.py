"""PersonalAgent: orchestrates V1 (structured) or V2 (autonomous) workflows.

Each participant gets their own PersonalAgent instance with:
- Persistent memory (personal patterns, prediction history, learned thresholds)
- Self-evaluation loop (compares predictions to EMA ground truth)
- Access to shared knowledge (population norms, peer cases)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentState:
    """Persistent state for a PersonalAgent."""

    user_id: str
    prediction_log: list[dict] = field(default_factory=list)
    receptivity_history: list[dict] = field(default_factory=list)
    learned_params: dict[str, Any] = field(default_factory=dict)
    memory_path: Path | None = None


class PersonalAgent:
    """Per-user agent that predicts emotional state and receptivity from sensing data."""

    def __init__(self, user_id: str, version: str = "v1", state_dir: Path | None = None) -> None:
        self.user_id = user_id
        self.version = version  # "v1" or "v2"
        self.state = AgentState(user_id=user_id)
        self.state_dir = state_dir

        self._workflow = self._init_workflow()

    def _init_workflow(self):
        """Initialize the appropriate workflow (V1 or V2)."""
        if self.version == "v1":
            from src.agent.structured import StructuredWorkflow
            return StructuredWorkflow(self)
        elif self.version == "v2":
            from src.agent.autonomous import AutonomousWorkflow
            return AutonomousWorkflow(self)
        else:
            raise ValueError(f"Unknown agent version: {self.version}")

    def predict(self, sensing_data: dict, context: dict | None = None) -> dict:
        """Make predictions for current EMA window.

        Args:
            sensing_data: Extracted sensing features for the lookback window.
            context: Additional context (time of day, day of week, etc.)

        Returns:
            Dict with predicted emotional states and receptivity.
        """
        return self._workflow.run(sensing_data, context)

    def receive_ground_truth(self, ema_response: dict) -> dict:
        """Receive EMA ground truth and trigger self-evaluation.

        Args:
            ema_response: Actual EMA response with emotional state and receptivity labels.

        Returns:
            Self-evaluation results (accuracy, confidence calibration, etc.)
        """
        raise NotImplementedError

    def save_state(self) -> None:
        """Persist agent state to disk."""
        raise NotImplementedError

    def load_state(self) -> None:
        """Load agent state from disk."""
        raise NotImplementedError
