"""V4 Autonomous Full Workflow: diary + sensing + multimodal RAG → prediction.

Like V2, gives the LLM all available context upfront and lets it autonomously
decide how to reason. Unlike V2, includes diary text and multimodal RAG examples
(diary + sensing from similar cases).

Single LLM call per EMA entry.
"""

from __future__ import annotations

import logging
from typing import Any

from src.remember.retriever import MultiModalRetriever
from src.think.llm_client import ClaudeCodeClient
from src.think.parser import parse_prediction
from src.think.prompts import (
    build_trait_summary,
    format_sensing_summary,
    v4_prompt,
    v4_system_prompt,
)

logger = logging.getLogger(__name__)


class AutonomousFullWorkflow:
    """V4: Autonomous agent with diary + sensing + multimodal RAG."""

    def __init__(
        self,
        llm_client: ClaudeCodeClient,
        retriever: MultiModalRetriever | None = None,
        study_id: int | None = None,
    ) -> None:
        self.llm = llm_client
        self.retriever = retriever
        self.study_id = study_id

    def run(
        self,
        ema_row=None,
        sensing_day=None,
        memory_doc: str = "",
        profile=None,
        date_str: str = "",
    ) -> dict[str, Any]:
        """Execute V4 autonomous prediction workflow.

        Single LLM call with all available data modalities.

        Args:
            ema_row: pandas Series of the EMA entry (for emotion_driver).
            sensing_day: SensingDay dataclass (or None).
            memory_doc: Pre-generated memory document text.
            profile: UserProfile dataclass.
            date_str: Date string for context.

        Returns:
            Dict with predictions + metadata.
        """
        # Extract diary text
        emotion_driver = ""
        if ema_row is not None:
            emotion_driver = str(ema_row.get("emotion_driver", ""))
            if emotion_driver == "nan" or not emotion_driver.strip():
                emotion_driver = ""

        # Format sensing data
        sensing_summary = format_sensing_summary(sensing_day)

        # Retrieve similar cases with sensing data
        rag_examples = "No similar cases available."
        rag_raw = []
        if self.retriever and emotion_driver:
            rag_raw = self.retriever.search_with_sensing(
                emotion_driver, top_k=10, exclude_study_id=self.study_id
            )
            rag_examples = self.retriever.format_examples_with_sensing(
                rag_raw, max_examples=8
            )

        # Build prompt
        trait_text = build_trait_summary(profile)
        system = v4_system_prompt(trait_text)
        prompt = v4_prompt(
            emotion_driver=emotion_driver or "(No diary entry provided)",
            sensing_summary=sensing_summary,
            rag_examples=rag_examples,
            memory_doc=memory_doc,
            date_str=date_str,
        )

        # Single LLM call
        logger.debug("V4: Autonomous prediction with diary + sensing + RAG")
        response = self.llm.generate(prompt=prompt, system_prompt=system)
        result = parse_prediction(response)

        # Comprehensive trace
        result["_version"] = "v4"
        result["_prompt_length"] = len(prompt) + len(system)
        result["_llm_calls"] = 1
        result["_emotion_driver"] = emotion_driver
        result["_has_diary"] = bool(emotion_driver.strip())
        result["_diary_length"] = len(emotion_driver) if emotion_driver.strip() else 0
        result["_sensing_summary"] = sensing_summary
        result["_full_prompt"] = prompt
        result["_system_prompt"] = system
        result["_full_response"] = response
        result["_rag_top5"] = [
            {
                "text": r.get("text", "")[:200],
                "similarity": r.get("similarity", 0),
                "PANAS_Pos": r.get("PANAS_Pos"),
                "PANAS_Neg": r.get("PANAS_Neg"),
                "has_sensing": bool(r.get("sensing_summary")),
            }
            for r in rag_raw[:5]
        ]
        result["_memory_excerpt"] = memory_doc[:500] if memory_doc else ""
        result["_trait_summary"] = trait_text

        return result
