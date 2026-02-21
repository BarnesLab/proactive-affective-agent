"""V2 Autonomous Workflow: ReAct-style multi-turn LLM agent.

Round 1: LLM sees initial sensing overview, decides what to examine deeper.
Round 2: LLM receives requested context, reasons further or predicts.
Round 3: (if needed) Force final prediction.

Typically 2-3 LLM calls per EMA entry.
"""

from __future__ import annotations

import logging
from typing import Any

from src.think.llm_client import ClaudeCodeClient
from src.think.parser import parse_json_block, parse_prediction
from src.think.prompts import (
    build_trait_summary,
    format_sensing_summary,
    v2_followup_prompt,
    v2_initial_prompt,
    v2_system_prompt,
)

logger = logging.getLogger(__name__)


class AutonomousWorkflow:
    """V2: ReAct-style agent with 2-3 LLM calls per prediction."""

    MAX_ROUNDS = 3

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

        Args:
            sensing_day: SensingDay dataclass (or None).
            memory_doc: Pre-generated memory document text.
            profile: UserProfile dataclass.
            date_str: Date string for context.
            sensing_dfs: Full sensing DataFrames (for deeper queries).
            study_id: User's Study_ID (for deeper queries).

        Returns:
            Dict with predictions + trace of multi-turn reasoning.
        """
        trait_text = build_trait_summary(profile)
        system = v2_system_prompt(trait_text)
        sensing_summary = format_sensing_summary(sensing_day)

        trace = []
        llm_calls = 0

        # Round 1: Initial overview
        prompt_r1 = v2_initial_prompt(
            sensing_summary=sensing_summary,
            memory_doc=memory_doc,
            date_str=date_str,
        )

        logger.debug("V2 Round 1: Initial overview")
        response_r1 = self.llm.generate(prompt=prompt_r1, system_prompt=system)
        llm_calls += 1
        trace.append({"round": 1, "prompt_len": len(prompt_r1), "prompt": prompt_r1, "response": response_r1})

        # Check if Round 1 already has a JSON prediction
        result = self._try_parse_prediction(response_r1)
        if result:
            result["_version"] = "v2"
            result["_llm_calls"] = llm_calls
            result["_trace"] = trace
            result["_sensing_summary"] = sensing_summary
            result["_memory_excerpt"] = memory_doc[:500] if memory_doc else ""
            result["_trait_summary"] = trait_text
            result["_system_prompt"] = system
            return result

        # Round 2+: Handle requests
        for round_num in range(2, self.MAX_ROUNDS + 1):
            request = self._extract_request(response_r1 if round_num == 2 else prev_response)

            if request:
                additional_context = self._gather_deeper_context(
                    request=request,
                    sensing_day=sensing_day,
                    memory_doc=memory_doc,
                    sensing_dfs=sensing_dfs,
                    study_id=study_id,
                )
            else:
                additional_context = "No specific request detected. Please provide your final prediction."

            prompt_rn = v2_followup_prompt(
                additional_context=additional_context,
                round_num=round_num,
            )

            # Build full conversation for context
            full_prompt = f"{prompt_r1}\n\n---\nYour previous response:\n{response_r1[:1000]}\n\n---\n{prompt_rn}"

            logger.debug(f"V2 Round {round_num}: Follow-up")
            prev_response = self.llm.generate(prompt=full_prompt, system_prompt=system)
            llm_calls += 1
            trace.append({"round": round_num, "request": request, "prompt": full_prompt, "response": prev_response})

            result = self._try_parse_prediction(prev_response)
            if result:
                result["_version"] = "v2"
                result["_llm_calls"] = llm_calls
                result["_trace"] = trace
                result["_sensing_summary"] = sensing_summary
                result["_memory_excerpt"] = memory_doc[:500] if memory_doc else ""
                result["_trait_summary"] = trait_text
                result["_system_prompt"] = system
                return result

        # If we got here, force parse the last response
        logger.warning("V2: Max rounds reached without clean JSON. Force-parsing last response.")
        result = parse_prediction(prev_response)
        result["_version"] = "v2"
        result["_llm_calls"] = llm_calls
        result["_trace"] = trace
        result["_forced_parse"] = True
        result["_sensing_summary"] = sensing_summary
        result["_memory_excerpt"] = memory_doc[:500] if memory_doc else ""
        result["_trait_summary"] = trait_text
        result["_system_prompt"] = system
        return result

    def _try_parse_prediction(self, response: str) -> dict[str, Any] | None:
        """Try to parse a prediction from response. Returns None if it's a request."""
        if not response:
            return None

        # If response contains REQUEST:, it's asking for more info
        if "REQUEST:" in response.upper():
            return None

        # Try to parse JSON
        json_data = parse_json_block(response)
        if json_data and "PANAS_Pos" in json_data:
            return parse_prediction(response)

        return None

    def _extract_request(self, response: str) -> str | None:
        """Extract the REQUEST: content from the LLM's response."""
        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("REQUEST:"):
                return line[len("REQUEST:"):].strip()
        return None

    def _gather_deeper_context(
        self,
        request: str,
        sensing_day,
        memory_doc: str,
        sensing_dfs: dict | None = None,
        study_id: int | None = None,
    ) -> str:
        """Map the LLM's natural language request to actual data lookups.

        This interprets what the LLM asks for and provides relevant data.
        """
        request_lower = request.lower()
        parts = []

        # Sleep-related requests
        if any(kw in request_lower for kw in ["sleep", "rest", "bedtime"]):
            if sensing_day:
                d = sensing_day.to_summary_dict()
                sleep_info = {k: v for k, v in d.items() if "sleep" in k.lower()}
                if sleep_info:
                    parts.append(f"Detailed sleep data: {sleep_info}")
                else:
                    parts.append("No sleep data available for this day.")

        # Mobility requests
        if any(kw in request_lower for kw in ["mobility", "gps", "travel", "location", "movement"]):
            if sensing_day:
                d = sensing_day.to_summary_dict()
                gps_info = {k: v for k, v in d.items()
                           if any(kw in k for kw in ["travel", "home", "distance", "location", "gps"])}
                if gps_info:
                    parts.append(f"Detailed mobility data: {gps_info}")

        # Activity requests
        if any(kw in request_lower for kw in ["activity", "exercise", "motion", "walk", "stationary"]):
            if sensing_day:
                d = sensing_day.to_summary_dict()
                motion_info = {k: v for k, v in d.items()
                              if any(kw in k for kw in ["walking", "running", "cycling", "stationary", "automotive"])}
                if motion_info:
                    parts.append(f"Detailed activity data: {motion_info}")

        # Social/communication requests
        if any(kw in request_lower for kw in ["social", "communication", "typing", "app", "phone"]):
            if sensing_day:
                d = sensing_day.to_summary_dict()
                social_info = {k: v for k, v in d.items()
                              if any(kw in k for kw in ["word", "char", "typed", "screen", "app", "positive", "negative"])}
                if social_info:
                    parts.append(f"Social/communication signals: {social_info}")

        # Memory/history requests
        if any(kw in request_lower for kw in ["history", "pattern", "typical", "baseline", "average", "usual"]):
            if memory_doc:
                # Provide more of the memory doc
                parts.append(f"Extended user history:\n{memory_doc[:4000]}")

        # Stress/emotional pattern requests
        if any(kw in request_lower for kw in ["stress", "emotion", "mood", "affect", "coping"]):
            if memory_doc:
                parts.append(f"Emotional patterns from memory:\n{memory_doc[:3000]}")

        if not parts:
            # Generic fallback: provide full sensing + extended memory
            parts.append(f"Full sensing data: {sensing_day.to_summary_dict() if sensing_day else 'None'}")
            if memory_doc:
                parts.append(f"User history:\n{memory_doc[:2000]}")

        return "\n\n".join(parts)
