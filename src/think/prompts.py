"""All prompt templates for agent reasoning.

Centralized prompt management. Each prompt is a function that takes
structured inputs and returns a formatted prompt string.
"""

from __future__ import annotations


# --- V1 Structured Workflow Prompts ---

def sensing_summary_prompt(features: dict, context: dict | None = None) -> str:
    """Prompt to summarize sensing features into a natural language description."""
    raise NotImplementedError


def reasoning_prompt(
    sensing_summary: str,
    memory_context: str,
    user_profile: str | None = None,
) -> str:
    """Prompt for the main reasoning step: predict emotional state and receptivity.

    Takes sensing summary + memory context → outputs structured predictions.
    """
    raise NotImplementedError


def decision_prompt(predictions: dict, confidence: dict) -> str:
    """Prompt for the intervention decision step."""
    raise NotImplementedError


def self_eval_prompt(prediction: dict, ground_truth: dict) -> str:
    """Prompt for self-evaluation after receiving EMA ground truth."""
    raise NotImplementedError


# --- V2 Autonomous Agent Prompts ---

def autonomous_system_prompt(tools: list[str], user_context: str) -> str:
    """System prompt for the autonomous agent with tool descriptions."""
    raise NotImplementedError


# --- Memory Update Prompts ---

def memory_update_prompt(current_memory: str, new_observation: str) -> str:
    """Prompt to update user memory with new observations."""
    raise NotImplementedError
