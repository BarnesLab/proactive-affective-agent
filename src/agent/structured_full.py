"""V3 Structured Full Workflow: diary + sensing + multimodal RAG → prediction.

Combines all three data modalities (diary text, passive sensing, RAG examples
with both diary and sensing) using a fixed 5-step reasoning pipeline.
Single LLM call per EMA entry.
"""

from __future__ import annotations

import logging
from typing import Any

from src.remember.retriever import MultiModalRetriever
from src.think.llm_client import ClaudeCodeClient
from src.think.prompts import (
    build_trait_summary,
    format_sensing_summary,
    v3_prompt,
    v3_system_prompt,
)

logger = logging.getLogger(__name__)


class StructuredFullWorkflow:
    """V3: Structured pipeline with diary + sensing + multimodal RAG."""

    def __init__(
        self,
        llm_client: ClaudeCodeClient,
        retriever: MultiModalRetriever | None = None,
    ) -> None:
        self.llm = llm_client
        self.retriever = retriever

    def run(
        self,
        ema_row=None,
        sensing_day=None,
        memory_doc: str = "",
        profile=None,
        date_str: str = "",
    ) -> dict[str, Any]:
        """Execute V3 prediction pipeline.

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
            rag_raw = self.retriever.search_with_sensing(emotion_driver, top_k=10)
            rag_examples = self.retriever.format_examples_with_sensing(
                rag_raw, max_examples=8
            )

        # Build prompt
        trait_text = build_trait_summary(profile)
        system = v3_system_prompt()
        prompt = v3_prompt(
            emotion_driver=emotion_driver or "(No diary entry provided)",
            sensing_summary=sensing_summary,
            rag_examples=rag_examples,
            memory_doc=memory_doc,
            trait_profile=trait_text,
            date_str=date_str,
        )

        # Single LLM call
        logger.debug("V3: Calling LLM with diary + sensing + RAG context")
        raw_response = self.llm.generate(prompt=prompt, system_prompt=system)
        from src.think.parser import parse_prediction
        result = parse_prediction(raw_response)

        # Comprehensive trace
        result["_version"] = "v3"
        result["_prompt_length"] = len(prompt) + len(system)
        result["_emotion_driver"] = emotion_driver
        result["_sensing_summary"] = sensing_summary
        result["_full_prompt"] = prompt
        result["_system_prompt"] = system
        result["_full_response"] = raw_response
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
