"""PersonalAgent: orchestrates CALLM, V1, or V2 workflows for a single user.

Each user gets their own PersonalAgent. The agent delegates to the appropriate
workflow based on the version parameter.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.autonomous import AutonomousWorkflow
from src.agent.structured import StructuredWorkflow
from src.data.schema import UserProfile
from src.remember.retriever import TFIDFRetriever
from src.think.llm_client import ClaudeCodeClient
from src.think.prompts import build_trait_summary, callm_prompt, format_sensing_summary

logger = logging.getLogger(__name__)


class PersonalAgent:
    """Per-user agent that predicts emotional states using CALLM, V1, or V2."""

    def __init__(
        self,
        study_id: int,
        version: str,
        llm_client: ClaudeCodeClient,
        profile: UserProfile | None = None,
        memory_doc: str = "",
        retriever: TFIDFRetriever | None = None,
    ) -> None:
        self.study_id = study_id
        self.version = version  # "callm", "v1", "v2"
        self.llm = llm_client
        self.profile = profile or UserProfile(study_id=study_id)
        self.memory_doc = memory_doc
        self.retriever = retriever  # Only used by CALLM

        # Initialize workflow
        if version == "v1":
            self._v1 = StructuredWorkflow(llm_client)
        elif version == "v2":
            self._v2 = AutonomousWorkflow(llm_client)

    def predict(
        self,
        ema_row=None,
        sensing_day=None,
        date_str: str = "",
        sensing_dfs: dict | None = None,
    ) -> dict[str, Any]:
        """Make predictions for a single EMA entry.

        Args:
            ema_row: pandas Series of the EMA entry (needed for CALLM's emotion_driver).
            sensing_day: SensingDay dataclass (needed for V1/V2).
            date_str: Date string for context.
            sensing_dfs: Full sensing DataFrames (for V2 deeper queries).

        Returns:
            Dict with predictions, metadata, and trace info.
        """
        if self.version == "callm":
            return self._run_callm(ema_row, date_str)
        elif self.version == "v1":
            return self._run_v1(sensing_day, date_str)
        elif self.version == "v2":
            return self._run_v2(sensing_day, date_str, sensing_dfs)
        else:
            raise ValueError(f"Unknown version: {self.version}")

    def _run_callm(self, ema_row, date_str: str) -> dict[str, Any]:
        """CALLM baseline: diary text + TF-IDF RAG + memory → single LLM call."""
        emotion_driver = ""
        if ema_row is not None:
            emotion_driver = str(ema_row.get("emotion_driver", ""))
            if emotion_driver == "nan" or not emotion_driver.strip():
                emotion_driver = ""

        # Retrieve similar cases
        rag_examples = "No similar cases available."
        if self.retriever and emotion_driver:
            results = self.retriever.search(emotion_driver, top_k=20)
            rag_examples = self.retriever.format_examples(results, max_examples=10)

        trait_text = build_trait_summary(self.profile)
        prompt = callm_prompt(
            emotion_driver=emotion_driver or "(No diary entry provided)",
            rag_examples=rag_examples,
            memory_doc=self.memory_doc,
            trait_profile=trait_text,
            date_str=date_str,
        )

        logger.debug(f"CALLM: Calling LLM for user {self.study_id}")
        result = self.llm.generate_structured(prompt=prompt)
        result["_version"] = "callm"
        result["_prompt_length"] = len(prompt)
        result["_emotion_driver"] = emotion_driver[:200]
        return result

    def _run_v1(self, sensing_day, date_str: str) -> dict[str, Any]:
        """V1 Structured: sensing data → single LLM call."""
        return self._v1.run(
            sensing_day=sensing_day,
            memory_doc=self.memory_doc,
            profile=self.profile,
            date_str=date_str,
        )

    def _run_v2(self, sensing_day, date_str: str, sensing_dfs: dict | None) -> dict[str, Any]:
        """V2 Autonomous: multi-turn ReAct agent."""
        return self._v2.run(
            sensing_day=sensing_day,
            memory_doc=self.memory_doc,
            profile=self.profile,
            date_str=date_str,
            sensing_dfs=sensing_dfs,
            study_id=self.study_id,
        )
