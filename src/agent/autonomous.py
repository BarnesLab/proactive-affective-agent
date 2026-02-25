"""V2 Autonomous Workflow: Single-call LLM agent with full context.

Unlike V1's fixed step-by-step pipeline, V2 gives the LLM all available context
upfront and lets it autonomously decide how to reason about the data.

Single LLM call per EMA entry (same as V1/CALLM).
"""

from __future__ import annotations

import logging
from typing import Any

from src.think.llm_client import ClaudeCodeClient
from src.think.parser import parse_prediction
from src.think.prompts import (
    build_trait_summary,
    format_sensing_summary,
    v2_prompt,
    v2_system_prompt,
)

logger = logging.getLogger(__name__)


class AutonomousWorkflow:
    """V2: Autonomous agent — single LLM call with full context."""

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
        sensing_dfs: dict | None = None,
        study_id: int | None = None,
    ) -> dict[str, Any]:
        """Execute V2 autonomous prediction workflow.

        Single LLM call with all available context.
        """
        trait_text = build_trait_summary(profile)
        system = v2_system_prompt(trait_text)
        sensing_summary = format_sensing_summary(sensing_day)

        # Build full context prompt
        prompt = v2_prompt(
            sensing_summary=sensing_summary,
            memory_doc=memory_doc,
            date_str=date_str,
        )

        logger.debug("V2: Single-call autonomous prediction")
        response = self.llm.generate(prompt=prompt, system_prompt=system)

        result = parse_prediction(response)
        result["_version"] = "v2"
        result["_prompt_length"] = len(prompt) + len(system)
        result["_llm_calls"] = 1
        result["_full_prompt"] = prompt
        result["_full_response"] = response
        result["_sensing_summary"] = sensing_summary
        result["_memory_excerpt"] = memory_doc[:500] if memory_doc else ""
        result["_trait_summary"] = trait_text
        result["_system_prompt"] = system
        return result
