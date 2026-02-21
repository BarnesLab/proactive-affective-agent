"""DataSimulator: replays historical data chronologically per user.

For each user, iterates through study days and EMA windows (morning, afternoon, evening).
At each window:
  1. Feeds available sensing data to the agent (only past data)
  2. Agent makes predictions
  3. Delivers EMA ground truth for self-evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from src.agent.personal_agent import PersonalAgent


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    user_ids: list[str]
    agent_version: str  # "v1" or "v2"
    sensing_lookback_hours: int = 4


class DataSimulator:
    """Replays historical data chronologically, driving agent predictions and evaluations."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.agents: dict[str, PersonalAgent] = {}

    def setup_agents(self) -> None:
        """Initialize a PersonalAgent for each user."""
        raise NotImplementedError

    def run(self) -> dict:
        """Run full simulation across all users and days. Returns evaluation results."""
        raise NotImplementedError

    def run_user(self, user_id: str) -> dict:
        """Run simulation for a single user. Returns per-user results."""
        raise NotImplementedError

    def _process_ema_window(
        self, user_id: str, day: str, window: str
    ) -> dict:
        """Process one EMA window: gather sensing → agent predicts → deliver ground truth."""
        raise NotImplementedError
