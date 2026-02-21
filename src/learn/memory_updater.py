"""Update user memory with new observations after self-evaluation.

Integrates new prediction outcomes, pattern discoveries, and
calibration insights into the user's persistent memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.remember.memory import UserMemory


class MemoryUpdater:
    """Updates user memory based on self-evaluation results."""

    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client

    def update(self, memory: UserMemory, evaluation: dict, sensing_context: dict) -> None:
        """Update user memory with insights from the latest self-evaluation.

        Args:
            memory: The user's memory object.
            evaluation: Self-evaluation results.
            sensing_context: Sensing data context for the evaluated prediction.
        """
        raise NotImplementedError

    def update_prediction_log(self, memory: UserMemory, entry: dict) -> None:
        """Add a new entry to the prediction log section of memory."""
        raise NotImplementedError

    def update_patterns(self, memory: UserMemory, new_pattern: str) -> None:
        """Add or update a behavioral pattern in memory."""
        raise NotImplementedError
