"""PersonalAgent: orchestrates CALLM, V1, V2, V3, V4 workflows for a single user.

Each user gets their own PersonalAgent. The agent delegates to the appropriate
workflow based on the version parameter.

2x2 Research Design:
                    Structured (fixed pipeline)    Agentic (autonomous tool-use)
  Sensing-only      V1                             V2
  Multimodal        V3                             V4 <- key contribution

  CALLM: diary + TF-IDF RAG (diary only) -- CHI 2025 baseline
  ML baselines: RF, XGBoost, LogReg, Ridge on sensor features
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.structured import StructuredWorkflow
from src.agent.structured_full import StructuredFullWorkflow
from src.data.schema import UserProfile
from src.remember.retriever import MultiModalRetriever, TFIDFRetriever
from src.think.llm_client import ClaudeCodeClient
from src.think.prompts import build_trait_summary, callm_prompt, format_sensing_summary

logger = logging.getLogger(__name__)


class PersonalAgent:
    """Per-user agent that predicts emotional states using CALLM/V1/V2/V3/V4."""

    def __init__(
        self,
        study_id: int,
        version: str,
        llm_client: ClaudeCodeClient,
        profile: UserProfile | None = None,
        memory_doc: str = "",
        retriever: TFIDFRetriever | None = None,
        query_engine=None,
        agentic_model: str = "claude-sonnet-4-6",
        agentic_soft_limit: int = 8,
        agentic_hard_limit: int = 20,
    ) -> None:
        self.study_id = study_id
        self.version = version  # "callm", "v1", "v2", "v3", "v4"
        self.llm = llm_client
        self.profile = profile or UserProfile(study_id=study_id)
        self.memory_doc = memory_doc
        self.retriever = retriever  # CALLM: TFIDFRetriever, V3: MultiModalRetriever

        # Initialize workflow
        if version == "v1":
            self._v1 = StructuredWorkflow(llm_client)
        elif version == "v2":
            # V2: Sensing-only + Agentic (tool-use, no diary)
            if query_engine is None:
                raise ValueError(
                    "V2 requires a SensingQueryEngine. "
                    "Pass query_engine= when constructing PersonalAgent with version='v2'."
                )
            from src.agent.agentic_sensing_only import AgenticSensingOnlyAgent
            self._v2 = AgenticSensingOnlyAgent(
                study_id=study_id,
                profile=self.profile,
                memory_doc=memory_doc,
                query_engine=query_engine,
                model=agentic_model,
                soft_limit=agentic_soft_limit,
                hard_limit=agentic_hard_limit,
            )
        elif version == "v3":
            mm_retriever = retriever if isinstance(retriever, MultiModalRetriever) else None
            self._v3 = StructuredFullWorkflow(
                llm_client, retriever=mm_retriever, study_id=study_id
            )
        elif version == "v4":
            # V4: Multimodal + Agentic (tool-use + diary) -- key contribution
            if query_engine is None:
                raise ValueError(
                    "V4 requires a SensingQueryEngine. "
                    "Pass query_engine= when constructing PersonalAgent with version='v4'."
                )
            from src.agent.agentic_sensing import AgenticSensingAgent
            self._v4 = AgenticSensingAgent(
                study_id=study_id,
                profile=self.profile,
                memory_doc=memory_doc,
                query_engine=query_engine,
                model=agentic_model,
                soft_limit=agentic_soft_limit,
                hard_limit=agentic_hard_limit,
            )

    def predict(
        self,
        ema_row=None,
        sensing_day=None,
        date_str: str = "",
        sensing_dfs: dict | None = None,
        session_memory: str | None = None,
    ) -> dict[str, Any]:
        """Make predictions for a single EMA entry.

        Args:
            ema_row: pandas Series of the EMA entry (needed for CALLM/V3/V4 diary).
            sensing_day: SensingDay dataclass (needed for V1/V3 structured pipelines).
            date_str: Date string for context.
            sensing_dfs: Not used (kept for backward compatibility).
            session_memory: Accumulated per-user session memory for V2/V4 agentic agents.

        Returns:
            Dict with predictions, metadata, and trace info.
        """
        if self.version == "callm":
            return self._run_callm(ema_row, date_str)
        elif self.version == "v1":
            return self._run_v1(sensing_day, date_str)
        elif self.version == "v2":
            return self._run_v2(ema_row, session_memory=session_memory)
        elif self.version == "v3":
            return self._run_v3(ema_row, sensing_day, date_str)
        elif self.version == "v4":
            return self._run_v4(ema_row, session_memory=session_memory)
        else:
            raise ValueError(f"Unknown version: {self.version}")

    def _run_callm(self, ema_row, date_str: str) -> dict[str, Any]:
        """CALLM baseline: diary text + TF-IDF RAG + memory -> single LLM call."""
        emotion_driver = ""
        if ema_row is not None:
            emotion_driver = str(ema_row.get("emotion_driver", ""))
            if emotion_driver == "nan" or not emotion_driver.strip():
                emotion_driver = ""

        # Retrieve similar cases
        rag_examples = "No similar cases available."
        rag_raw = []
        if self.retriever and emotion_driver:
            rag_raw = self.retriever.search(
                emotion_driver, top_k=20, exclude_study_id=self.study_id
            )
            rag_examples = self.retriever.format_examples(rag_raw, max_examples=10)

        trait_text = build_trait_summary(self.profile)
        prompt = callm_prompt(
            emotion_driver=emotion_driver or "(No diary entry provided)",
            rag_examples=rag_examples,
            memory_doc=self.memory_doc,
            trait_profile=trait_text,
            date_str=date_str,
        )

        logger.debug(f"CALLM: Calling LLM for user {self.study_id}")
        raw_response = self.llm.generate(prompt=prompt)
        usage = getattr(self.llm, "last_usage", {})
        from src.think.parser import parse_prediction
        result = parse_prediction(raw_response)

        # Comprehensive trace
        result["_version"] = "callm"
        result["_model"] = self.llm.model
        result["_prompt_length"] = len(prompt)
        result["_emotion_driver"] = emotion_driver
        result["_has_diary"] = bool(emotion_driver.strip())
        result["_diary_length"] = len(emotion_driver) if emotion_driver.strip() else 0
        result["_full_prompt"] = prompt
        result["_full_response"] = raw_response
        result["_rag_top5"] = [
            {"text": r.get("text", "")[:200], "similarity": r.get("similarity", 0),
             "PANAS_Pos": r.get("PANAS_Pos"), "PANAS_Neg": r.get("PANAS_Neg")}
            for r in rag_raw[:5]
        ]
        result["_memory_excerpt"] = self.memory_doc[:500] if self.memory_doc else ""
        result["_trait_summary"] = trait_text
        result["_input_tokens"] = usage.get("input_tokens", 0)
        result["_output_tokens"] = usage.get("output_tokens", 0)
        result["_total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        result["_cost_usd"] = usage.get("cost_usd", 0)
        result["_llm_calls"] = 1

        logger.info(
            f"CALLM: tokens={usage.get('input_tokens', '?')}in+"
            f"{usage.get('output_tokens', '?')}out, "
            f"confidence={result.get('confidence', '?')}"
        )

        return result

    def _run_v1(self, sensing_day, date_str: str) -> dict[str, Any]:
        """V1 Structured: sensing data -> fixed pipeline -> single LLM call."""
        return self._v1.run(
            sensing_day=sensing_day,
            memory_doc=self.memory_doc,
            profile=self.profile,
            date_str=date_str,
        )

    def _run_v2(self, ema_row, session_memory: str | None = None) -> dict[str, Any]:
        """V2 Agentic Sensing-Only: autonomous tool-use loop, NO diary text."""
        return self._v2.predict(ema_row=ema_row, session_memory=session_memory)

    def _run_v3(self, ema_row, sensing_day, date_str: str) -> dict[str, Any]:
        """V3 Structured Full: diary + sensing + multimodal RAG -> structured pipeline."""
        return self._v3.run(
            ema_row=ema_row,
            sensing_day=sensing_day,
            memory_doc=self.memory_doc,
            profile=self.profile,
            date_str=date_str,
        )

    def _run_v4(self, ema_row, session_memory: str | None = None) -> dict[str, Any]:
        """V4 Agentic Multimodal: autonomous tool-use loop + diary text."""
        diary_text = None
        if ema_row is not None:
            diary_text = str(ema_row.get("emotion_driver", ""))
            if diary_text.lower() == "nan" or not diary_text.strip():
                diary_text = None
        return self._v4.predict(ema_row=ema_row, diary_text=diary_text, session_memory=session_memory)
