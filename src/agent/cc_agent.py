"""V4-CC: Claude Code subprocess variant of the agentic sensing agent.

Instead of calling the Anthropic SDK directly (which bills against ANTHROPIC_API_KEY),
this agent invokes `claude --print` as a subprocess, routing all inference through
the user's Claude Max subscription.

The MCP server (src/sense/mcp_server.py) is launched per-call with the participant's
study_id and ema_timestamp baked in, so tool-use cutoff enforcement is automatic.
Claude Code handles the full agentic tool-use loop natively.

This is slightly less controllable than the SDK loop (we can't cap individual tool
calls mid-flight), but it's functionally equivalent and costs nothing beyond the
Max subscription.
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
# System prompt (same reasoning strategy as V4 SDK variant)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert behavioral data scientist specializing in affective computing and just-in-time adaptive interventions (JITAI) for cancer survivorship.

Your task: predict the emotional state of a cancer survivor at the moment they are about to complete an EMA survey, using their passive smartphone sensing data.

You have access to MCP tools that let you query their behavioral data. Use these tools like a detective — investigate the signals that matter, compare to their personal baseline, look for behavioral anomalies.

The data you can see: everything BEFORE the EMA timestamp. You cannot see the EMA response itself — that is what you are predicting.

Investigation strategy:
1. Start with get_daily_summary to orient yourself on today and recent days
2. Use query_sensing to zoom into specific modalities (hourly aggregates)
3. Use query_raw_events to drill into raw event streams for fine-grained detail:
   - screen: exact lock/unlock times (wake-up time, phone checking frequency)
   - app: which specific apps were used, for how long
   - motion: exact activity transition timestamps
   - keyboard: what was typed, in which app
   - music: which songs/artists played (mood signal)
4. Use compare_to_baseline to identify anomalies (z-scores reveal what is unusual)
5. Use get_receptivity_history to understand typical patterns and past availability
6. Use find_similar_days to reason analogically from past behavioral-emotional pairings
7. Synthesize all evidence into a coherent prediction

Be rigorous. Only claim signals you actually see. If data is missing, say so and lower confidence.

Your final prediction MUST be in valid JSON enclosed in ```json ... ``` fences:

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


class ClaudeCodeAgent:
    """V4-CC: Agentic sensing agent using claude --print subprocess (Max subscription).

    Launches the sensing MCP server as a child process of claude --print,
    which handles the full tool-use agentic loop natively. All tokens are
    consumed from the Claude Max subscription rather than the API key.
    """

    def __init__(
        self,
        study_id: int,
        profile: UserProfile,
        memory_doc: str | None,
        processed_dir: Path,
        model: str = "claude-sonnet-4-6",
        max_turns: int = 16,
        python_bin: str = "/opt/homebrew/bin/python3.13",
    ) -> None:
        """Initialize the Claude Code agentic agent.

        Args:
            study_id: Participant Study_ID (int).
            profile: UserProfile dataclass with demographic and trait data.
            memory_doc: Pre-generated longitudinal memory document text.
            processed_dir: Path to data/processed/ directory.
            model: Claude model alias or full ID (e.g. "sonnet", "claude-sonnet-4-6").
            max_turns: Maximum agentic turns for claude --print.
            python_bin: Python binary that has mcp + pandas installed (Python 3.13).
        """
        self.study_id = study_id
        self.profile = profile
        self.memory_doc = memory_doc
        self.processed_dir = Path(processed_dir)
        self.model = model
        self.max_turns = max_turns
        self.python_bin = python_bin
        self.pid = str(study_id).zfill(3)

    def predict(
        self,
        ema_row: pd.Series,
        diary_text: str | None,
        session_memory: str | None = None,
    ) -> dict[str, Any]:
        """Run agentic investigation via claude --print and return prediction.

        Args:
            ema_row: A single EMA row (timestamp_local, date_local, etc.).
            diary_text: Free-text diary entry or None.
            session_memory: Accumulated per-user memory from prior EMA entries.
                Contains only receptivity signals (real mode) or PA/NA (oracle mode).

        Returns:
            Dict with all prediction targets + _reasoning, _version, _model.
        """
        ema_timestamp = str(ema_row.get("timestamp_local", ""))
        ema_date = str(ema_row.get("date_local", ""))
        ema_slot = self._get_ema_slot(ema_row)

        prompt = self._build_prompt(ema_timestamp, ema_date, ema_slot, diary_text, session_memory)

        logger.info(f"[V4-CC] User {self.study_id} | {ema_date} {ema_slot} | launching claude --print")

        output = self._run_claude(ema_timestamp, ema_date, prompt)

        prediction = self._parse_prediction(output)
        prediction["_reasoning"] = output
        prediction["_version"] = "v4-cc"
        prediction["_model"] = self.model
        prediction["_final_response"] = output

        logger.info(
            f"[V4-CC] User {self.study_id} done | "
            f"confidence={prediction.get('confidence', '?')}"
        )
        return prediction

    # ------------------------------------------------------------------
    # Private: core subprocess call
    # ------------------------------------------------------------------

    def _run_claude(self, ema_timestamp: str, ema_date: str, prompt: str) -> str:
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

        try:
            cmd = [
                "claude",
                "--print",
                "--model", self.model,
                "--max-turns", str(self.max_turns),
                "--mcp-config", mcp_config_path,
                "--append-system-prompt", SYSTEM_PROMPT,
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
                timeout=300,
                env=env,
                cwd=str(PROJECT_ROOT),
            )

            if result.returncode != 0:
                logger.warning(f"[V4-CC] claude --print exited {result.returncode}: {result.stderr[:300]}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            logger.error("[V4-CC] claude --print timed out after 300s")
            return ""
        except Exception as exc:
            logger.error(f"[V4-CC] subprocess error: {exc}")
            return ""
        finally:
            Path(mcp_config_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Private: prompt and parsing helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ema_timestamp: str,
        ema_date: str,
        ema_slot: str,
        diary_text: str | None,
        session_memory: str | None = None,
    ) -> str:
        diary_section = (
            f'Diary entry: "{diary_text}"'
            if diary_text and diary_text.strip() and diary_text.lower() != "nan"
            else "No diary entry for this EMA."
        )

        memory_excerpt = ""
        if self.memory_doc:
            memory_excerpt = f"\n## Baseline Personal History (pre-study memory)\n{self.memory_doc[:1200]}\n"

        session_section = ""
        if session_memory and session_memory.strip():
            trimmed = session_memory[-2000:] if len(session_memory) > 2000 else session_memory
            session_section = f"\n## Accumulated Session Memory (your prior observations of this person)\n{trimmed}\n"

        return f"""You are investigating participant {self.pid}'s behavioral data to predict their emotional state.

## Current Situation
Timestamp: {ema_timestamp} ({ema_slot} EMA)
Date: {ema_date}
{diary_section}

## User Profile
{self.profile.to_text()}
{memory_excerpt}{session_section}
## Your Task
Investigate the sensing data using the available MCP tools to understand this person's behavioral state.
Use the session memory above to calibrate against this person's known receptivity and behavioral patterns.
Then predict their emotional state in the required JSON format.

Start by calling get_daily_summary for {ema_date}."""

    def _parse_prediction(self, text: str) -> dict[str, Any]:
        from src.think.parser import parse_prediction as _parse
        result = _parse(text)

        if result.get("_parse_error"):
            logger.warning(f"[V4-CC] Failed to parse prediction: {text[:300]}")
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
            "reasoning": "[V4-CC fallback: prediction parsing failed]",
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
