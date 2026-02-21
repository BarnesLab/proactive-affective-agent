"""V2 Autonomous Workflow: LLM-driven tool-use agent (ReAct-style).

The LLM decides which tools to call, in what order, and when to stop.
More flexible than V1 but less predictable.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.personal_agent import PersonalAgent


class AutonomousWorkflow:
    """ReAct-style agent: LLM autonomously decides tool use and reasoning steps."""

    TOOLS = [
        "query_sensing",
        "read_memory",
        "check_peers",
        "retrieve_rag",
        "predict_affect",
        "check_history",
        "intervene",
    ]

    def __init__(self, agent: PersonalAgent, max_steps: int = 10) -> None:
        self.agent = agent
        self.max_steps = max_steps

    def run(self, sensing_data: dict, context: dict | None = None) -> dict:
        """Run the autonomous agent loop.

        The LLM iteratively:
        1. Observes current state
        2. Thinks about what to do
        3. Calls a tool
        4. Incorporates the result
        Until it decides to produce a final answer.

        Returns:
            Dict with predictions, tool call trace, and decision.
        """
        raise NotImplementedError

    def _build_system_prompt(self, context: dict | None = None) -> str:
        """Build the system prompt describing available tools and objectives."""
        raise NotImplementedError

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """Execute a single tool call and return the result."""
        raise NotImplementedError

    # --- Tool implementations ---

    def query_sensing(self, sensor: str, window: str) -> dict:
        """Retrieve specific sensing data for a sensor and time window."""
        raise NotImplementedError

    def read_memory(self, query: str) -> str:
        """Search user's personal memory for relevant entries."""
        raise NotImplementedError

    def check_peers(self, pattern: str) -> list[dict]:
        """Find similar patterns in other users' histories."""
        raise NotImplementedError

    def retrieve_rag(self, query: str) -> list[str]:
        """Semantic search over pre-computed memory documents."""
        raise NotImplementedError

    def predict_affect(self, evidence: dict) -> dict:
        """Make an emotional state prediction given accumulated evidence."""
        raise NotImplementedError

    def check_history(self) -> dict:
        """Review past prediction accuracy for this user."""
        raise NotImplementedError

    def intervene(self, intervention_type: str, content: str) -> dict:
        """Deliver an intervention to the user."""
        raise NotImplementedError
