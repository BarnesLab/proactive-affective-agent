"""Structured output parsing from LLM responses.

Parses LLM text output into structured dicts for predictions,
decisions, and self-evaluations.
"""

from __future__ import annotations

import json
from typing import Any


class OutputParser:
    """Parses LLM responses into structured data."""

    @staticmethod
    def parse_prediction(response: str) -> dict:
        """Parse a prediction response into structured emotional state + receptivity.

        Expected fields: valence, arousal, stress, loneliness, receptivity, confidence.
        """
        raise NotImplementedError

    @staticmethod
    def parse_decision(response: str) -> dict:
        """Parse an intervention decision response.

        Expected fields: should_intervene, intervention_type, reasoning.
        """
        raise NotImplementedError

    @staticmethod
    def parse_self_eval(response: str) -> dict:
        """Parse a self-evaluation response.

        Expected fields: accuracy, calibration_notes, memory_updates.
        """
        raise NotImplementedError

    @staticmethod
    def parse_json_block(text: str) -> dict:
        """Extract and parse a JSON block from LLM output (handles ```json ... ``` wrapping)."""
        raise NotImplementedError
