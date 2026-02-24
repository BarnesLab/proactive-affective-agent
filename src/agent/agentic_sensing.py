"""V4: Autonomous Sensing Agent.

Unlike V1-V4 (which receive pre-formatted sensing summaries), V4 has tool access
to the raw processed sensing data. It autonomously decides what to investigate
and builds its reasoning from first principles — like a detective with access
to behavioral telemetry.

Key insight: The "missing puzzle piece" is the current emotional state.
All past data is visible. V4 finds the patterns that predict it.

Uses the Anthropic Python SDK directly (not claude CLI) for proper tool use support.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic
import pandas as pd

from src.data.schema import UserProfile
from src.sense.query_tools import SENSING_TOOLS, SensingQueryEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert behavioral data scientist specializing in affective computing and just-in-time adaptive interventions (JITAI) for cancer survivorship.

Your task: predict the emotional state of a cancer survivor at the moment they are about to complete an EMA survey, using their passive smartphone sensing data.

You have access to tools that let you query their behavioral data (motion, location, screen usage, keyboard activity, etc.). Use these tools like a detective — investigate the signals that matter, compare to their personal baseline, look for behavioral anomalies.

The data you can see: everything BEFORE the EMA timestamp. You cannot see the EMA response itself — that is what you are predicting.

Investigation strategy:
1. Start with get_daily_summary to orient yourself on today and recent days
2. Use query_sensing to zoom into specific modalities that seem informative (hourly aggregates)
3. Use query_raw_events to drill into raw event streams when you need fine-grained detail:
   - screen: exact lock/unlock times (infer wake-up time, phone checking frequency)
   - app: which specific apps were used, for how long (social media? messaging?)
   - motion: exact activity transition timestamps (when did she leave home?)
   - keyboard: what was typed, in which app (communication content)
   - music: which songs/artists played (mood signal from genre)
4. Use compare_to_baseline to identify anomalies (z-scores reveal what is unusual)
5. Use get_receptivity_history to understand this person's typical patterns and past availability
6. Use find_similar_days to reason analogically from past behavioral-emotional pairings
7. Synthesize all evidence into a coherent prediction

Be a rigorous analyst. Only claim signals you actually see in the data. If data is missing, say so explicitly and adjust your confidence downward. Do not hallucinate patterns.

Your final prediction must be in valid JSON format enclosed in ```json ... ``` fences."""

# ---------------------------------------------------------------------------
# Prediction request template
# ---------------------------------------------------------------------------

PREDICTION_REQUEST = """Based on your investigation, provide your final prediction in the following JSON format enclosed in ```json ... ``` fences:

```json
{
  "PANAS_Pos": <number 0-30, predicted positive affect>,
  "PANAS_Neg": <number 0-30, predicted negative affect>,
  "ER_desire": <number 0-10, predicted emotion regulation desire>,
  "Individual_level_PA_State": <true|false, positive affect unusually high for this person?>,
  "Individual_level_NA_State": <true|false, negative affect unusually high for this person?>,
  "Individual_level_happy_State": <true|false>,
  "Individual_level_sad_State": <true|false>,
  "Individual_level_afraid_State": <true|false>,
  "Individual_level_miserable_State": <true|false>,
  "Individual_level_worried_State": <true|false>,
  "Individual_level_cheerful_State": <true|false>,
  "Individual_level_pleased_State": <true|false>,
  "Individual_level_grateful_State": <true|false>,
  "Individual_level_lonely_State": <true|false>,
  "Individual_level_interactions_quality_State": <true|false>,
  "Individual_level_pain_State": <true|false>,
  "Individual_level_forecasting_State": <true|false>,
  "Individual_level_ER_desire_State": <true|false>,
  "INT_availability": <"yes" or "no", is user available for intervention?>,
  "reasoning": "<concise behavioral analysis supporting your prediction>",
  "confidence": <number 0-1, your overall confidence>
}
```"""


# ---------------------------------------------------------------------------
# AgenticSensingAgent
# ---------------------------------------------------------------------------

class AgenticSensingAgent:
    """V4 autonomous sensing agent.

    Uses the Anthropic SDK tool-use loop to iteratively query sensing data
    and build evidence before making a final emotional state prediction.
    Unlike V1-V4, V4 actively chooses what to investigate rather than
    receiving pre-formatted summaries.
    """

    def __init__(
        self,
        study_id: int,
        profile: UserProfile,
        memory_doc: str | None,
        query_engine: SensingQueryEngine,
        model: str = "claude-sonnet-4-6",
        max_tool_calls: int = 8,
    ) -> None:
        """Initialize the agentic sensing agent.

        Args:
            study_id: Participant Study_ID (int).
            profile: UserProfile dataclass with demographic and trait data.
            memory_doc: Pre-generated longitudinal memory document text.
            query_engine: SensingQueryEngine providing tool implementations.
            model: Anthropic model ID to use.
            max_tool_calls: Maximum number of tool call iterations before forcing prediction.
        """
        self.study_id = study_id
        self.profile = profile
        self.memory_doc = memory_doc
        self.engine = query_engine
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.pid = str(study_id).zfill(3)

    def predict(
        self,
        ema_row: pd.Series,
        diary_text: str | None,
        session_memory: str | None = None,
    ) -> dict[str, Any]:
        """Run autonomous investigation and produce an emotional state prediction.

        Drives the agentic tool-use loop:
          1. Send initial context (+ accumulated session memory) to the model
          2. Process tool calls until end_turn or max_tool_calls reached
          3. If no prediction extracted yet, explicitly request one
          4. Parse and return the prediction dict

        Args:
            ema_row: A single row from the EMA DataFrame (current entry to predict).
            diary_text: Free-text diary entry for this EMA slot, or None.
            session_memory: Accumulated per-user memory from prior EMA entries (receptivity
                + optional PA/NA in oracle mode). Never contains current EMA labels.

        Returns:
            Dict with all prediction targets plus _reasoning, _n_tool_calls, _version.
        """
        ema_timestamp = str(ema_row.get("timestamp_local", ""))
        ema_date = str(ema_row.get("date_local", ""))
        ema_slot = self._get_ema_slot(ema_row)

        initial_context = self._build_initial_context(
            ema_timestamp, ema_date, ema_slot, diary_text, session_memory
        )
        messages: list[dict] = [{"role": "user", "content": initial_context}]

        tool_call_count = 0
        full_reasoning: list[str] = []
        final_response = None
        total_input_tokens = 0
        total_output_tokens = 0

        logger.info(f"[V4] User {self.study_id} | {ema_date} {ema_slot} | starting agentic loop")

        # ------------------------------------------------------------------
        # Agentic tool-use loop
        # ------------------------------------------------------------------
        while tool_call_count < self.max_tool_calls:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=SENSING_TOOLS,
                    messages=messages,
                )
            except Exception as exc:
                logger.error(f"[V4] Anthropic API error: {exc}")
                break

            final_response = response
            if hasattr(response, "usage") and response.usage:
                total_input_tokens += response.usage.input_tokens or 0
                total_output_tokens += response.usage.output_tokens or 0

            if response.stop_reason == "end_turn":
                logger.debug(f"[V4] Agent ended turn after {tool_call_count} tool calls")
                break

            if response.stop_reason == "tool_use":
                # Build tool_results list and record reasoning
                tool_results: list[dict] = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input if isinstance(block.input, dict) else {}

                        logger.debug(f"[V4] Tool call: {tool_name}({tool_input})")

                        result_text = self._execute_tool(
                            tool_name, tool_input, ema_timestamp, ema_date
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                        full_reasoning.append(
                            f"[Tool #{tool_call_count + 1}: {tool_name}]\n"
                            f"Input: {json.dumps(tool_input)}\n"
                            f"Result:\n{result_text}"
                        )
                        tool_call_count += 1

                        if tool_call_count >= self.max_tool_calls:
                            logger.info(f"[V4] Max tool calls ({self.max_tool_calls}) reached")
                            break

                # Extend conversation: assistant turn + tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason (max_tokens, etc.)
                logger.warning(f"[V4] Unexpected stop_reason: {response.stop_reason}")
                break

        # ------------------------------------------------------------------
        # Extract final text from last response
        # ------------------------------------------------------------------
        last_text = self._extract_text(final_response)

        # If the agent didn't produce a prediction yet, explicitly request one
        if not self._has_prediction(last_text):
            logger.debug("[V4] No prediction in final response — requesting explicit prediction")
            messages.append({
                "role": "assistant",
                "content": final_response.content if final_response else [],
            })
            messages.append({"role": "user", "content": PREDICTION_REQUEST})

            try:
                pred_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
                if hasattr(pred_response, "usage") and pred_response.usage:
                    total_input_tokens += pred_response.usage.input_tokens or 0
                    total_output_tokens += pred_response.usage.output_tokens or 0
                last_text = self._extract_text(pred_response)
            except Exception as exc:
                logger.error(f"[V4] Error requesting prediction: {exc}")

        # ------------------------------------------------------------------
        # Parse prediction
        # ------------------------------------------------------------------
        prediction = self._parse_prediction(last_text)
        prediction["_reasoning"] = "\n\n".join(full_reasoning)
        prediction["_n_tool_calls"] = tool_call_count
        prediction["_version"] = "v4"
        prediction["_model"] = self.model
        prediction["_final_response"] = last_text
        prediction["_input_tokens"] = total_input_tokens
        prediction["_output_tokens"] = total_output_tokens
        prediction["_total_tokens"] = total_input_tokens + total_output_tokens

        logger.info(
            f"[V4] User {self.study_id} done: {tool_call_count} tool calls, "
            f"tokens={total_input_tokens}in+{total_output_tokens}out, "
            f"confidence={prediction.get('confidence', '?')}"
        )
        return prediction

    # ------------------------------------------------------------------
    # Private: agentic loop helpers
    # ------------------------------------------------------------------

    def _build_initial_context(
        self,
        ema_timestamp: str,
        ema_date: str,
        ema_slot: str,
        diary_text: str | None,
        session_memory: str | None = None,
    ) -> str:
        """Build the initial investigation context message for the agent."""
        diary_section = (
            f'Diary entry: "{diary_text}"'
            if diary_text and diary_text.strip() and diary_text.lower() != "nan"
            else "No diary entry for this EMA."
        )

        memory_excerpt = ""
        if self.memory_doc:
            memory_excerpt = f"""## Baseline Personal History (pre-study memory)
{self.memory_doc[:1200]}
"""

        session_section = ""
        if session_memory and session_memory.strip():
            trimmed = session_memory[-2000:] if len(session_memory) > 2000 else session_memory
            session_section = f"""## Accumulated Session Memory (your prior observations of this person)
{trimmed}
"""

        # Check which Parquet modality files exist for this participant
        available_modalities = [
            mod for mod in self.engine.MODALITIES
            if self.engine._parquet_path(self.study_id, mod).exists()
        ]
        modalities_str = ", ".join(available_modalities) if available_modalities else "none loaded"

        return f"""You are investigating participant {self.pid}'s behavioral data to predict their emotional state.

## Current Situation
Timestamp: {ema_timestamp} ({ema_slot} EMA)
Date: {ema_date}
{diary_section}

## User Profile
{self.profile.to_text()}

{memory_excerpt}{session_section}## Available Sensing Modalities
{modalities_str}

## Your Task
Investigate the sensing data to understand this person's behavioral state leading up to this EMA.
Use the available tools to build your evidence. Use the session memory above to calibrate against
this person's known receptivity and behavioral patterns. Then predict their emotional state.

Start by calling get_daily_summary for {ema_date} to orient yourself."""

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        ema_timestamp: str,
        ema_date: str,
    ) -> str:
        """Dispatch a tool call to SensingQueryEngine.call_tool().

        The Parquet-backed SensingQueryEngine.call_tool() handles all routing
        internally using the SENSING_TOOLS schema signatures. This keeps the
        dispatch logic in one place.

        Args:
            tool_name: The tool name from the Anthropic tool_use block.
            tool_input: The input dict from the tool_use block.
            ema_timestamp: EMA timestamp (injected into call_tool for cutoff enforcement).
            ema_date: EMA date string (used for fallback defaults in tool_input).

        Returns:
            String result to return to the model as a tool_result.
        """
        # Inject date default for tools that need it
        if tool_name == "get_daily_summary" and "date" not in tool_input:
            tool_input = {**tool_input, "date": ema_date}

        return self.engine.call_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            study_id=self.study_id,
            ema_timestamp=ema_timestamp,
        )

    def _extract_text(self, response: anthropic.types.Message | None) -> str:
        """Extract all TextBlock content from an Anthropic message."""
        if response is None:
            return ""
        parts = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _has_prediction(self, text: str) -> bool:
        """Check whether the text contains a JSON prediction block."""
        if not text:
            return False
        # Look for PANAS_Pos in JSON-like content
        return bool(re.search(r'"PANAS_Pos"\s*:', text))

    def _parse_prediction(self, text: str) -> dict[str, Any]:
        """Parse the final prediction JSON from the agent's response text.

        Tries multiple extraction strategies:
          1. ```json ... ``` fenced block
          2. First { ... } object in the text
          3. Fallback to a neutral default prediction

        Returns:
            Parsed prediction dict with all expected fields.
        """
        from src.think.parser import parse_prediction as _parse
        result = _parse(text)

        if result.get("_parse_error"):
            logger.warning(f"[V4] Failed to parse prediction from response: {text[:300]}")
            # Return a neutral fallback so we don't crash the pipeline
            result = self._fallback_prediction()
            result["_parse_error"] = True
            result["_raw_response"] = text[:500]

        return result

    def _fallback_prediction(self) -> dict[str, Any]:
        """Return a neutral prediction used when parsing fails entirely."""
        from src.utils.mappings import BINARY_STATE_TARGETS
        pred: dict[str, Any] = {
            "PANAS_Pos": 15.0,
            "PANAS_Neg": 8.0,
            "ER_desire": 3.0,
            "INT_availability": "yes",
            "reasoning": "[V4 fallback: prediction parsing failed]",
            "confidence": 0.1,
        }
        for target in BINARY_STATE_TARGETS:
            pred[target] = False
        return pred

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    def _get_ema_slot(self, ema_row: pd.Series) -> str:
        """Determine the EMA slot (morning / afternoon / evening) from timestamp."""
        try:
            ts = pd.to_datetime(ema_row.get("timestamp_local", ""))
            hour = ts.hour
            if hour < 12:
                return "morning"
            elif hour < 17:
                return "afternoon"
            else:
                return "evening"
        except Exception:
            return "unknown"
