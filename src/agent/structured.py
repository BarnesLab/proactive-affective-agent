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
        raw_response = self.llm.generate(prompt=prompt, system_prompt=system)
        usage = getattr(self.llm, "last_usage", {})
        from src.think.parser import parse_prediction
        result = parse_prediction(raw_response)

        # Comprehensive trace
        result["_version"] = "v1"
        result["_model"] = self.llm.model
        result["_prompt_length"] = len(prompt) + len(system)
        result["_sensing_summary"] = sensing_summary
        result["_full_prompt"] = prompt
        result["_system_prompt"] = system
        result["_full_response"] = raw_response
        result["_memory_excerpt"] = memory_doc[:500] if memory_doc else ""
        result["_trait_summary"] = trait_text
        result["_input_tokens"] = usage.get("input_tokens", 0)
        result["_output_tokens"] = usage.get("output_tokens", 0)
        result["_total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        result["_cost_usd"] = usage.get("cost_usd", 0)
        result["_llm_calls"] = 1

        logger.info(
            f"V1: tokens={usage.get('input_tokens', '?')}in+"
            f"{usage.get('output_tokens', '?')}out, "
            f"confidence={result.get('confidence', '?')}"
        )

        return result
