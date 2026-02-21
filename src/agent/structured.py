"""V1 Structured Workflow: sensing data → memory → single LLM call → prediction.

For the pilot: simplified from 4-step to a single LLM call with all context
pre-assembled. The LLM does step-by-step reasoning within one call.
"""

from __future__ import annotations

import logging
from typing import Any

from src.think.llm_client import ClaudeCodeClient
from src.think.prompts import (
    build_trait_summary,
    format_sensing_summary,
    v1_prompt,
    v1_system_prompt,
)

logger = logging.getLogger(__name__)


class StructuredWorkflow:
    """V1: Fixed pipeline — assemble context, single LLM call, parse output."""

    def __init__(
        self,
        llm_client: ClaudeCodeClient,
    ) -> None:
        self.llm = llm_client

    def run(
        self,
        sensing_day,
        memory_doc: str,
        profile,
        date_str: str = "",
    ) -> dict[str, Any]:
        """Execute V1 prediction pipeline.

        Args:
            sensing_day: SensingDay dataclass (or None).
            memory_doc: Pre-generated memory document text.
            profile: UserProfile dataclass.
            date_str: Date string for context.

        Returns:
            Dict with predictions + metadata.
        """
        # Step 1: Format sensing data
        sensing_summary = format_sensing_summary(sensing_day)

        # Step 2: Build prompt with all context
        trait_text = build_trait_summary(profile)
        prompt = v1_prompt(
            sensing_summary=sensing_summary,
            memory_doc=memory_doc,
            trait_profile=trait_text,
            date_str=date_str,
        )
        system = v1_system_prompt()

        # Step 3: Single LLM call
        logger.debug("V1: Calling LLM with sensing + memory context")
        result = self.llm.generate_structured(
            prompt=prompt,
            system_prompt=system,
        )

        # Add trace info
        result["_version"] = "v1"
        result["_prompt_length"] = len(prompt) + len(system)
        result["_sensing_summary"] = sensing_summary[:500]

        return result
