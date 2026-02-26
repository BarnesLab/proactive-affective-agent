"""Unified agentic agent using claude --print subprocess (Max subscription, FREE).

Replaces the SDK-based V2 (agentic_sensing_only.py) and V4 (agentic_sensing.py) agents
which required ANTHROPIC_API_KEY and billed against the paid API.

Uses claude --print + MCP server to run the agentic tool-use loop. All inference
is routed through the user's Claude Max subscription (no API cost).

Supports two modes:
  - "multimodal" (V4): diary text + sensing tools
  - "sensing_only" (V2): sensing tools only, no diary
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import UserProfile

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# JSON prediction schema (shared by both modes)
# ---------------------------------------------------------------------------

PREDICTION_SCHEMA = """\
```json
{
  "PANAS_Pos": <number 0-30>,
  "PANAS_Neg": <number 0-30>,
  "ER_desire": <number 0-10>,
  "Individual_level_PA_State": <true|false>,
  "Individual_level_NA_State": <true|false>,
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
  "INT_availability": <"yes" or "no">,
  "reasoning": "<concise behavioral analysis>",
  "confidence": <number 0-1>
}
```"""

# ---------------------------------------------------------------------------
# System prompts — V4 (multimodal) and V2 (sensing-only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_MULTIMODAL = f"""You are an expert behavioral data scientist specializing in affective computing and just-in-time adaptive interventions (JITAI) for cancer survivorship.

Your task: predict the emotional state of a cancer survivor at the moment they are about to complete an EMA survey, using BOTH their diary text AND passive smartphone sensing data.

IMPORTANT — Diary-First Analysis:
The participant's diary entry is your most direct window into their emotional state. Prior research shows diary content is highly predictive — it reveals what the person is experiencing, thinking, and feeling in their own words. Your investigation should be GUIDED by the diary:
- First, carefully read the diary text. Identify emotional themes, concerns, stressors, positive events, social context, and coping language.
- Then form hypotheses about their likely emotional state based on the diary content.
- Use sensing tools to VALIDATE and CALIBRATE those hypotheses — look for behavioral evidence that confirms or contradicts what the diary suggests.
- Pay special attention to cross-modal consistency: does their behavior (sleep, activity, screen use) align with what they wrote? Discrepancies are informative.

You have access to MCP tools that let you query their behavioral data. Use these tools strategically — investigate the signals that MATTER given what the diary tells you, rather than exploring everything exhaustively.

The data you can see: everything BEFORE the EMA timestamp. You cannot see the EMA response itself — that is what you are predicting.

Investigation strategy:
1. Analyze the diary entry first — extract emotional signals, themes, and form initial hypotheses
2. Call get_daily_summary to see overall behavioral patterns for today and recent days
3. Use query_sensing to zoom into modalities most relevant to the diary content (hourly aggregates)
4. Use compare_to_baseline to check if today's behavior is unusual for this person (z-scores)
5. Use find_similar_days to find past days with similar behavioral patterns and their emotional outcomes
6. Synthesize diary + sensing evidence into a coherent, calibrated prediction

Be efficient: focus your tool calls on the most informative signals given the diary context. You have a limited tool call budget — make each call count.

Be a rigorous analyst. Only claim signals you actually see in the data. If data is missing, say so explicitly and adjust your confidence downward. Do not hallucinate patterns.

Your final prediction MUST be in valid JSON enclosed in ```json ... ``` fences:

{PREDICTION_SCHEMA}"""


SYSTEM_PROMPT_SENSING_ONLY = f"""You are an expert behavioral data scientist specializing in affective computing and just-in-time adaptive interventions (JITAI) for cancer survivorship.

Your task: predict the emotional state of a cancer survivor at the moment they are about to complete an EMA survey, using ONLY their passive smartphone sensing data. You do NOT have access to any diary text or self-report — your prediction must be based entirely on behavioral signals.

You have access to MCP tools that let you query their behavioral data (motion, location, screen usage, keyboard activity, etc.). Use these tools like a detective — investigate the signals that matter, compare to their personal baseline, look for behavioral anomalies.

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

Your final prediction MUST be in valid JSON enclosed in ```json ... ``` fences:

{PREDICTION_SCHEMA}"""


# ---------------------------------------------------------------------------
# AgenticCCAgent — unified V2/V4 agent via claude --print
# ---------------------------------------------------------------------------

class AgenticCCAgent:
    """Agentic agent using claude --print subprocess (Max subscription, no API cost).

    Launches the sensing MCP server as a child process of claude --print,
    which handles the full tool-use agentic loop natively.

    Supports two modes:
      - "multimodal" (V4): diary text + sensing tools
      - "sensing_only" (V2): sensing tools only, no diary
    """

    def __init__(
        self,
        study_id: int,
        profile: UserProfile,
        memory_doc: str | None,
        processed_dir: Path,
        model: str = "sonnet",
        max_turns: int = 16,
        mode: str = "multimodal",
    ) -> None:
        """Initialize the agentic CC agent.

        Args:
            study_id: Participant Study_ID.
            profile: UserProfile dataclass with demographic and trait data.
            memory_doc: Pre-generated longitudinal memory document text.
            processed_dir: Path to data/processed/ directory (for MCP server).
            model: Claude model alias (e.g. "sonnet", "haiku").
            max_turns: Maximum agentic turns for claude --print.
            mode: "multimodal" (V4, diary+sensing) or "sensing_only" (V2, no diary).
        """
        if mode not in ("multimodal", "sensing_only"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'multimodal' or 'sensing_only'.")
        self.study_id = study_id
        self.profile = profile
        self.memory_doc = memory_doc
        self.processed_dir = Path(processed_dir)
        self.model = model
        self.max_turns = max_turns
        self.mode = mode
        self.pid = str(study_id).zfill(3)
        self._version = "v4" if mode == "multimodal" else "v2"

        # Detect Python binary with mcp + pandas
        self.python_bin = self._find_python()

    def predict(
        self,
        ema_row: pd.Series,
        diary_text: str | None = None,
        session_memory: str | None = None,
    ) -> dict[str, Any]:
        """Run agentic investigation via claude --print and return prediction.

        Args:
            ema_row: A single EMA row (timestamp_local, date_local, etc.).
            diary_text: Free-text diary entry. Ignored in sensing_only mode.
            session_memory: Accumulated per-user memory from prior entries.

        Returns:
            Dict with all prediction targets + trace metadata.
        """
        ema_timestamp = str(ema_row.get("timestamp_local", ""))
        ema_date = str(ema_row.get("date_local", ""))
        ema_slot = self._get_ema_slot(ema_row)

        # In sensing_only mode, drop diary text
        effective_diary = diary_text if self.mode == "multimodal" else None

        prompt = self._build_prompt(ema_timestamp, ema_date, ema_slot, effective_diary, session_memory)
        system_prompt = SYSTEM_PROMPT_MULTIMODAL if self.mode == "multimodal" else SYSTEM_PROMPT_SENSING_ONLY

        tag = f"[{self._version.upper()}]"
        logger.info(f"{tag} User {self.study_id} | {ema_date} {ema_slot} | launching claude --print ({self.mode})")

        output = self._run_claude(ema_timestamp, ema_date, prompt, system_prompt)

        prediction = self._parse_prediction(output)
        prediction["_reasoning"] = output
        prediction["_raw_output"] = output
        prediction["_raw_output_length"] = len(output)
        prediction["_version"] = self._version
        prediction["_model"] = self.model
        prediction["_full_response"] = output
        prediction["_system_prompt"] = system_prompt
        prediction["_full_prompt"] = prompt
        prediction["_has_diary"] = bool(effective_diary and effective_diary.strip() and effective_diary.lower() != "nan")
        prediction["_diary_length"] = len(effective_diary) if prediction["_has_diary"] else 0
        prediction["_emotion_driver"] = effective_diary or ""

        # Best-effort structured tool call extraction from stdout
        parsed_tool_calls = self._parse_tool_calls_from_output(output)
        prediction["_tool_calls"] = parsed_tool_calls
        prediction["_n_tool_calls"] = len(parsed_tool_calls)
        prediction["_n_rounds"] = len(parsed_tool_calls)  # approximate: 1 tool ≈ 1 round in CC mode
        prediction["_conversation_length"] = 0  # not available in CC mode
        prediction["_input_tokens"] = 0  # not available from claude --print
        prediction["_output_tokens"] = 0
        prediction["_total_tokens"] = 0
        prediction["_cost_usd"] = 0.0  # free via Max subscription
        prediction["_llm_calls"] = 1  # one subprocess call

        logger.info(
            f"{tag} User {self.study_id} done | "
            f"{len(parsed_tool_calls)} tools | "
            f"confidence={prediction.get('confidence', '?')}"
        )
        return prediction

    # ------------------------------------------------------------------
    # Private: core subprocess call
    # ------------------------------------------------------------------

    def _run_claude(self, ema_timestamp: str, ema_date: str, prompt: str, system_prompt: str) -> str:
        """Write temp MCP config and call claude --print, returning stdout text."""
        mcp_server_path = PROJECT_ROOT / "src" / "sense" / "mcp_server.py"

        mcp_config = {
            "mcpServers": {
                "sensing": {
                    "command": self.python_bin,
                    "args": [
                        str(mcp_server_path),
                        "--study-id", str(self.study_id),
                        "--ema-timestamp", ema_timestamp,
                        "--ema-date", ema_date,
                        "--data-dir", str(self.processed_dir),
                    ],
                    "env": {
                        "PYTHONPATH": str(PROJECT_ROOT),
                    },
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="sensing_mcp_", delete=False
        ) as f:
            json.dump(mcp_config, f)
            mcp_config_path = f.name

        tag = f"[{self._version.upper()}]"
        try:
            cmd = [
                "claude",
                "--print",
                "--model", self.model,
                "--max-turns", str(self.max_turns),
                "--mcp-config", mcp_config_path,
                "--append-system-prompt", system_prompt,
                "--no-session-persistence",
                prompt,
                "--disallowed-tools", "Bash,Edit,Write,Read,Glob,Grep,Task",
            ]

            # Strip nested-session guards so claude --print can be launched as subprocess
            _blocked = {"CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_SESSION_ID"}
            env = {k: v for k, v in os.environ.items() if k not in _blocked}
            env["PYTHONPATH"] = str(PROJECT_ROOT)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                cwd=str(PROJECT_ROOT),
            )

            if result.returncode != 0:
                logger.warning(f"{tag} claude --print exited {result.returncode}: {result.stderr[:300]}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            logger.error(f"{tag} claude --print timed out after 600s")
            return ""
        except Exception as exc:
            logger.error(f"{tag} subprocess error: {exc}")
            return ""
        finally:
            Path(mcp_config_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Private: prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ema_timestamp: str,
        ema_date: str,
        ema_slot: str,
        diary_text: str | None,
        session_memory: str | None = None,
    ) -> str:
        """Build the user prompt for claude --print."""
        # Diary section (only for multimodal mode)
        if self.mode == "multimodal" and diary_text and diary_text.strip() and diary_text.lower() != "nan":
            diary_section = f"""## Diary Entry (PRIMARY emotional signal — analyze this FIRST)
"{diary_text}" """
        elif self.mode == "multimodal":
            diary_section = "No diary entry for this EMA."
        else:
            diary_section = "(Sensing-only mode — no diary text available. Rely entirely on passive behavioral signals.)"

        memory_excerpt = ""
        if self.memory_doc:
            memory_excerpt = f"\n## Baseline Personal History (pre-study memory)\n{self.memory_doc[:3000]}\n"

        session_section = ""
        if session_memory and session_memory.strip():
            trimmed = session_memory[-6000:] if len(session_memory) > 6000 else session_memory
            session_section = f"\n## Accumulated Session Memory (your prior observations of this person)\n{trimmed}\n"

        task_instruction = (
            "Investigate the sensing data using the available MCP tools to understand this person's behavioral state.\n"
            "Use the session memory above to calibrate against this person's known receptivity and behavioral patterns.\n"
            "Then predict their emotional state in the required JSON format."
        )
        if self.mode == "multimodal":
            task_instruction = (
                "1. FIRST: Analyze the diary entry above. What emotional themes, stressors, or positive signals does it reveal?\n"
                "2. THEN: Use sensing MCP tools to validate and calibrate your hypotheses.\n"
                "3. FINALLY: Synthesize diary + sensing evidence into a prediction in the required JSON format."
            )

        return f"""You are investigating participant {self.pid}'s behavioral data to predict their emotional state.

## Current Situation
Timestamp: {ema_timestamp} ({ema_slot} EMA)
Date: {ema_date}
{diary_section}

## User Profile
{self.profile.to_text()}
{memory_excerpt}{session_section}
## Your Task
{task_instruction}

Start by calling get_daily_summary for {ema_date}."""

    # ------------------------------------------------------------------
    # Private: parsing helpers
    # ------------------------------------------------------------------

    def _parse_tool_calls_from_output(self, output: str) -> list[dict[str, Any]]:
        """Best-effort extraction of tool calls from claude --print output."""
        tool_calls: list[dict[str, Any]] = []

        pattern = re.compile(
            r'(?:'
            r'\[Tool(?:\s*#\d+)?:\s*(\w+)\]'
            r'|(?:Using|Calling|Called)\s+(?:tool\s+)?[`"]?(\w+)[`"]?'
            r'|Tool(?:\s+call)?:\s*[`"]?(\w+)[`"]?'
            r')',
            re.IGNORECASE,
        )

        known_tools = {
            "get_daily_summary", "query_sensing", "query_raw_events",
            "compare_to_baseline", "get_receptivity_history", "find_similar_days",
        }

        matches = list(pattern.finditer(output))
        for idx, match in enumerate(matches):
            tool_name = match.group(1) or match.group(2) or match.group(3)
            if not tool_name:
                continue
            if tool_name.lower() not in {t.lower() for t in known_tools}:
                continue

            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else min(start + 2000, len(output))
            context = output[start:end]

            input_dict: dict[str, Any] = {}
            json_match = re.search(r'\{[^{}]*\}', context)
            if json_match:
                try:
                    candidate = json.loads(json_match.group())
                    if isinstance(candidate, dict) and len(candidate) <= 10:
                        input_dict = candidate
                except (json.JSONDecodeError, ValueError):
                    pass

            tool_calls.append({
                "index": len(tool_calls) + 1,
                "tool_name": tool_name,
                "input": input_dict,
                "result_length": len(context),
                "result_preview": context[:500],
            })

        return tool_calls

    def _parse_prediction(self, text: str) -> dict[str, Any]:
        from src.think.parser import parse_prediction as _parse
        result = _parse(text)

        if result.get("_parse_error"):
            tag = f"[{self._version.upper()}]"
            logger.warning(f"{tag} Failed to parse prediction: {text[:300]}")
            result = self._fallback_prediction()
            result["_parse_error"] = True
            result["_raw_response"] = text[:500]

        return result

    def _fallback_prediction(self) -> dict[str, Any]:
        from src.utils.mappings import BINARY_STATE_TARGETS
        pred: dict[str, Any] = {
            "PANAS_Pos": 15.0,
            "PANAS_Neg": 8.0,
            "ER_desire": 3.0,
            "INT_availability": "yes",
            "reasoning": f"[{self._version} fallback: prediction parsing failed]",
            "confidence": 0.1,
        }
        for target in BINARY_STATE_TARGETS:
            pred[target] = False
        return pred

    def _get_ema_slot(self, ema_row: pd.Series) -> str:
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

    def _find_python(self) -> str:
        """Find a Python binary that has mcp + pandas installed."""
        candidates = [
            "/opt/homebrew/bin/python3.13",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return "python3"
