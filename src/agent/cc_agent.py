"""Unified agentic agent using claude --print subprocess (Max subscription, FREE).

Implements V2 (sensing-only), V4 (multimodal), V5 (filtered sensing), and
V6 (filtered multimodal) agentic agents.
Uses claude --print + MCP server to run the agentic tool-use loop. All inference
is routed through the user's Claude Max subscription (no API cost).

Supports four modes:
  - "sensing_only" (V2): sensing tools only, no diary
  - "multimodal" (V4): diary text + sensing tools
  - "filtered_sensing" (V5): filtered narrative + sensing tools, no diary
  - "filtered_multimodal" (V6): filtered narrative + diary + sensing tools
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
from src.utils.rate_limit import (
    RateLimitError,
    RateLimitType,
    classify_error,
    send_telegram,
)

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
2. Call get_behavioral_timeline to reconstruct the day chronologically and infer within-day behavioral/affective shifts
3. Call get_daily_summary to see overall behavioral patterns for today and recent days
4. Use query_sensing to zoom into modalities most relevant to the diary content (hourly aggregates)
5. Use compare_to_baseline to check if today's behavior is unusual for this person (z-scores)
6. Use find_similar_days to find past days with similar behavioral patterns and their emotional outcomes
7. Use find_peer_cases (search_mode="text") to find OTHER participants with similar diary entries — their actual EMA outcomes serve as calibration anchors for your prediction
8. Synthesize diary + sensing + peer evidence into a coherent, calibrated prediction

Be efficient: focus your tool calls on the most informative signals given the diary context. You have a limited tool call budget — make each call count.

Be a rigorous analyst. Only claim signals you actually see in the data. If data is missing, say so explicitly and adjust your confidence downward. Do not hallucinate patterns.

Your final prediction MUST be in valid JSON enclosed in ```json ... ``` fences:

{PREDICTION_SCHEMA}"""


SYSTEM_PROMPT_SENSING_ONLY = f"""You are an expert behavioral data scientist specializing in affective computing and just-in-time adaptive interventions (JITAI) for cancer survivorship.

Your task: predict the emotional state of a cancer survivor at the moment they are about to complete an EMA survey, using ONLY their passive smartphone sensing data. You do NOT have access to any diary text or self-report — your prediction must be based entirely on behavioral signals.

You have access to MCP tools that let you query their behavioral data (motion, location, screen usage, keyboard activity, etc.). Use these tools like a detective — investigate the signals that matter, compare to their personal baseline, look for behavioral anomalies.

The data you can see: everything BEFORE the EMA timestamp. You cannot see the EMA response itself — that is what you are predicting.

Investigation strategy:
1. Start with get_behavioral_timeline to reconstruct the day chronologically before the EMA
2. Use get_daily_summary to orient yourself on today and recent days
3. Use query_sensing to zoom into specific modalities that seem informative (hourly aggregates)
4. Use query_raw_events to drill into raw event streams when you need fine-grained detail:
   - screen: exact lock/unlock times (infer wake-up time, phone checking frequency)
   - app: which specific apps were used, for how long (social media? messaging?)
   - motion: exact activity transition timestamps (when did she leave home?)
   - keyboard: what was typed, in which app (communication content)
   - music: which songs/artists played (mood signal from genre)
5. Use compare_to_baseline to identify anomalies (z-scores reveal what is unusual)
6. Use get_receptivity_history to understand this person's typical patterns and past availability
7. Use find_similar_days to reason analogically from past behavioral-emotional pairings
8. Use find_peer_cases (search_mode="sensing") to find OTHER participants with similar behavioral patterns — their actual EMA outcomes serve as calibration anchors
9. Synthesize all evidence into a coherent prediction

Be a rigorous analyst. Only claim signals you actually see in the data. If data is missing, say so explicitly and adjust your confidence downward. Do not hallucinate patterns.

Your final prediction MUST be in valid JSON enclosed in ```json ... ``` fences:

{PREDICTION_SCHEMA}"""


SYSTEM_PROMPT_FILTERED_SENSING = f"""You are an expert behavioral data scientist specializing in affective computing and just-in-time adaptive interventions (JITAI) for cancer survivorship.

Your task: predict the emotional state of a cancer survivor at the moment they are about to complete an EMA survey, using their pre-computed daily behavioral narrative AND passive smartphone sensing tools. You do NOT have access to any diary text or self-report — your prediction must be based entirely on behavioral signals.

You will receive a structured daily behavioral narrative that summarizes the participant's day: motion patterns, screen usage, app categories, keyboard activity, and environmental context. This narrative is your PRIMARY input — it gives you a comprehensive overview of the day.

You also have access to MCP tools that let you query the raw behavioral data for deeper analysis. Use these tools SELECTIVELY — the narrative already covers the big picture, so only drill into raw data when you need:
- Exact timestamps (e.g., when did they wake up, when was their last screen session)
- Hourly patterns within the day (e.g., morning vs evening activity)
- Specific app usage details beyond category-level summaries
- Historical comparison via baselines and similar days

Investigation strategy:
1. Read the behavioral narrative carefully. Extract key signals: activity level, social engagement, sleep/wake patterns, screen habits, environmental context.
2. Call get_behavioral_timeline to infer how their behavioral and affective cues shifted across the day.
3. Form hypotheses about their emotional state based on these behavioral patterns.
4. Use compare_to_baseline to check if today's behavior deviates from their personal norm — anomalies are the strongest signals.
5. If needed, use query_sensing or query_raw_events to drill into specific modalities for finer temporal resolution.
6. Use find_similar_days to find past days with similar behavioral patterns and their emotional outcomes.
7. Use find_peer_cases (search_mode="sensing") to find OTHER participants with similar behavioral fingerprints — their actual EMA outcomes serve as calibration anchors.
8. Synthesize all evidence into a coherent, calibrated prediction.

Be a rigorous analyst. Only claim signals you actually see in the data. If data is missing, say so explicitly and adjust your confidence downward. Do not hallucinate patterns.

Your final prediction MUST be in valid JSON enclosed in ```json ... ``` fences:

{PREDICTION_SCHEMA}"""


SYSTEM_PROMPT_FILTERED_MULTIMODAL = f"""You are an expert behavioral data scientist specializing in affective computing and just-in-time adaptive interventions (JITAI) for cancer survivorship.

Your task: predict the emotional state of a cancer survivor at the moment they are about to complete an EMA survey, using BOTH their diary text AND their pre-computed daily behavioral narrative AND passive smartphone sensing tools.

IMPORTANT — Diary-First Analysis:
The participant's diary entry is your most direct window into their emotional state. Prior research shows diary content is highly predictive — it reveals what the person is experiencing, thinking, and feeling in their own words.

You will also receive a structured daily behavioral narrative that summarizes the participant's day: motion patterns, screen usage, app categories, keyboard activity, and environmental context. Use this to VALIDATE and CALIBRATE your diary-based hypotheses.

You also have access to MCP tools that let you query the raw behavioral data for deeper analysis. Use these tools SELECTIVELY — between the diary and the narrative, you already have rich context. Only drill into raw data when you need specific drill-down.

Investigation strategy:
1. FIRST: Analyze the diary entry. Identify emotional themes, concerns, stressors, positive events, social context, and coping language. Form initial hypotheses.
2. THEN: Read the behavioral narrative. Does the behavioral evidence align with the diary? Look for cross-modal consistency or discrepancies.
3. Call get_behavioral_timeline to inspect how behavioral and affective cues evolved across the day.
4. Use compare_to_baseline to check if today's behavior deviates from their personal norm — anomalies help calibrate the diary's emotional signal.
5. If needed, use query_sensing or query_raw_events to drill into specific modalities where the diary and narrative seem inconsistent.
6. Use find_similar_days to find past days with similar patterns and outcomes.
7. Use find_peer_cases (search_mode="text") to find OTHER participants with similar diary entries — their actual EMA outcomes serve as calibration anchors.
8. Synthesize diary + narrative + sensing + peer evidence into a coherent, well-calibrated prediction.

Be efficient: the diary and narrative together give you ~80% of the signal. Use MCP tools for the remaining 20% — targeted validation, not exhaustive exploration.

Be a rigorous analyst. Only claim signals you actually see in the data. If data is missing, say so explicitly and adjust your confidence downward. Do not hallucinate patterns.

Your final prediction MUST be in valid JSON enclosed in ```json ... ``` fences:

{PREDICTION_SCHEMA}"""


# ---------------------------------------------------------------------------
# AgenticCCAgent — unified V2/V4/V5/V6 agent via claude --print
# ---------------------------------------------------------------------------

class AgenticCCAgent:
    """Agentic agent using claude --print subprocess (Max subscription, no API cost).

    Launches the sensing MCP server as a child process of claude --print,
    which handles the full tool-use agentic loop natively.

    Supports four modes:
      - "sensing_only" (V2): sensing tools only, no diary
      - "multimodal" (V4): diary text + sensing tools
      - "filtered_sensing" (V5): filtered narrative + sensing tools, no diary
      - "filtered_multimodal" (V6): filtered narrative + diary + sensing tools
    """

    _MODE_TO_VERSION = {
        "sensing_only": "v2",
        "multimodal": "v4",
        "filtered_sensing": "v5",
        "filtered_multimodal": "v6",
    }

    _MODE_TO_PROMPT = {
        "sensing_only": "SYSTEM_PROMPT_SENSING_ONLY",
        "multimodal": "SYSTEM_PROMPT_MULTIMODAL",
        "filtered_sensing": "SYSTEM_PROMPT_FILTERED_SENSING",
        "filtered_multimodal": "SYSTEM_PROMPT_FILTERED_MULTIMODAL",
    }

    def __init__(
        self,
        study_id: int,
        profile: UserProfile,
        memory_doc: str | None,
        processed_dir: Path,
        model: str = "sonnet",
        max_turns: int = 16,
        mode: str = "multimodal",
        filtered_data_dir: Path | None = None,
        peer_db_path: str | None = None,
    ) -> None:
        """Initialize the agentic CC agent.

        Args:
            study_id: Participant Study_ID.
            profile: UserProfile dataclass with demographic and trait data.
            memory_doc: Pre-generated longitudinal memory document text.
            processed_dir: Path to data/processed/ directory (for MCP server).
            model: Claude model alias (e.g. "sonnet", "haiku").
            max_turns: Maximum agentic turns for claude --print.
            mode: Agent mode — "multimodal", "sensing_only", "filtered_sensing", "filtered_multimodal".
            filtered_data_dir: Path to data/processed/filtered/ directory (for V5/V6).
            peer_db_path: Path to peer database parquet for cross-user search.
        """
        if mode not in self._MODE_TO_VERSION:
            raise ValueError(f"Invalid mode: {mode!r}. Must be one of {list(self._MODE_TO_VERSION)}.")
        self.study_id = study_id
        self.profile = profile
        self.memory_doc = memory_doc
        self.processed_dir = Path(processed_dir)
        self.model = model
        self.max_turns = max_turns
        self.mode = mode
        self.pid = str(study_id).zfill(3)
        self._version = self._MODE_TO_VERSION[mode]

        # Load filtered narrative data for V5/V6
        self._filtered_df: pd.DataFrame | None = None
        if mode in ("filtered_sensing", "filtered_multimodal"):
            if filtered_data_dir is None:
                raise ValueError(
                    f"{self._version.upper()} requires filtered_data_dir "
                    f"(path to data/processed/filtered/). "
                    f"Pass filtered_data_dir= when constructing AgenticCCAgent."
                )
            parquet_path = Path(filtered_data_dir) / f"{self.pid}_daily_filtered.parquet"
            if not parquet_path.exists():
                logger.warning(f"Filtered data not found: {parquet_path}")
            else:
                self._filtered_df = pd.read_parquet(parquet_path)
                logger.info(f"Loaded filtered data for {self.pid}: {len(self._filtered_df)} days")

        # Peer database path for cross-user search
        self.peer_db_path = peer_db_path

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

        # Only include diary text in multimodal modes (V4, V6)
        if self.mode in ("multimodal", "filtered_multimodal"):
            effective_diary = diary_text
        else:
            effective_diary = None

        prompt = self._build_prompt(ema_timestamp, ema_date, ema_slot, effective_diary, session_memory)
        system_prompt = {
            "sensing_only": SYSTEM_PROMPT_SENSING_ONLY,
            "multimodal": SYSTEM_PROMPT_MULTIMODAL,
            "filtered_sensing": SYSTEM_PROMPT_FILTERED_SENSING,
            "filtered_multimodal": SYSTEM_PROMPT_FILTERED_MULTIMODAL,
        }[self.mode]

        tag = f"[{self._version.upper()}]"
        logger.info(f"{tag} User {self.study_id} | {ema_date} {ema_slot} | launching claude --print ({self.mode})")

        # Retry loop: if parsing fails, re-run claude (never produce fallback)
        import time as _time
        _MAX_PARSE_RETRIES = 5
        for _parse_attempt in range(1, _MAX_PARSE_RETRIES + 1):
            output = self._run_claude(ema_timestamp, ema_date, prompt, system_prompt)
            prediction = self._parse_prediction(output)
            if not prediction.get("_parse_error"):
                break
            if _parse_attempt < _MAX_PARSE_RETRIES:
                logger.warning(
                    f"{tag} Parse failed (attempt {_parse_attempt}/{_MAX_PARSE_RETRIES}), "
                    f"re-running claude in 30s. Output: {output[:200]}"
                )
                _time.sleep(30)
            else:
                logger.error(
                    f"{tag} Parse failed after {_MAX_PARSE_RETRIES} attempts. "
                    f"Last output: {output[:500]}"
                )
                # Still don't fallback — raise so the entry can be retried later
                raise RuntimeError(
                    f"Unparseable response after {_MAX_PARSE_RETRIES} attempts for "
                    f"user {self.study_id} {ema_date} {ema_slot}"
                )

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

        # Tool call extraction: use num_turns from JSON wrapper (most reliable),
        # supplemented by regex extraction from text for tool names
        num_turns = getattr(self, "_last_num_turns", 1)
        parsed_tool_calls = self._parse_tool_calls_from_output(output)
        # num_turns > 1 means tool-use occurred; n_tool_calls = turns - 1
        # (first turn = initial prompt, subsequent turns = tool call + response)
        n_tool_calls_from_turns = max(0, num_turns - 1)
        prediction["_tool_calls"] = parsed_tool_calls
        prediction["_n_tool_calls"] = n_tool_calls_from_turns or len(parsed_tool_calls)
        prediction["_n_rounds"] = n_tool_calls_from_turns
        prediction["_conversation_length"] = num_turns
        usage = getattr(self, "_last_usage", {})
        prediction["_input_tokens"] = usage.get("input_tokens", 0)
        prediction["_output_tokens"] = usage.get("output_tokens", 0)
        prediction["_cost_usd"] = usage.get("cost_usd", 0)
        prediction["_total_tokens"] = 0
        prediction["_cost_usd"] = 0.0  # free via Max subscription
        prediction["_llm_calls"] = num_turns

        logger.info(
            f"{tag} User {self.study_id} done | "
            f"{len(parsed_tool_calls)} tools | "
            f"confidence={prediction.get('confidence', '?')}"
        )
        return prediction

    # ------------------------------------------------------------------
    # Private: core subprocess call
    # ------------------------------------------------------------------

    # Retry constants
    _TRANSIENT_RETRIES = 3
    _TRANSIENT_BACKOFF = [2, 4, 8]  # seconds
    _PATIENT_WAIT = 300  # 5 minutes — slow retry after fast retries exhausted
    _HOURLY_WAIT = 1800  # 30 minutes
    _HOURLY_MAX_RETRIES = 12  # up to 6 hours total
    _TIMEOUT_RETRIES = 3
    _TIMEOUT_BACKOFF = [10, 20, 40]

    def _run_claude(self, ema_timestamp: str, ema_date: str, prompt: str, system_prompt: str) -> str:
        """Write temp MCP config and call claude --print with retry logic.

        Retry strategy:
        - Transient errors: 3 retries, exponential backoff (2s, 4s, 8s)
        - Hourly rate limit: wait 30 min, retry up to 12 times (6h total)
        - Weekly limit: send Telegram alert, raise RateLimitError
        - Timeout: 3 retries with backoff

        Returns the LLM text output. Token usage is stored in self._last_usage.
        Raises RateLimitError on weekly limit (caller should stop experiment).
        """
        self._last_usage: dict = {}
        mcp_server_path = PROJECT_ROOT / "src" / "sense" / "mcp_server.py"

        mcp_args = [
            str(mcp_server_path),
            "--study-id", str(self.study_id),
            "--ema-timestamp", ema_timestamp,
            "--ema-date", ema_date,
            "--data-dir", str(self.processed_dir),
        ]
        if self.peer_db_path:
            mcp_args.extend(["--peer-db-path", str(self.peer_db_path)])

        mcp_config = {
            "mcpServers": {
                "sensing": {
                    "command": self.python_bin,
                    "args": mcp_args,
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
        # Pass prompt via stdin to avoid shell escaping issues with special
        # characters in diary text, behavioral narratives, etc.  Positional
        # prompt arg was silently corrupted for many entries, causing Claude
        # to fall back to single-call mode without MCP tool access.
        cmd = [
            "claude",
            "--print",
            "--output-format", "json",
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--mcp-config", mcp_config_path,
            "--append-system-prompt", system_prompt,
            "--no-session-persistence",
            "--disallowed-tools", "Bash,Edit,Write,Read,Glob,Grep,Task",
        ]

        # Strip nested-session guards so claude --print can be launched as subprocess
        _blocked = {"CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_SESSION_ID"}
        env = {k: v for k, v in os.environ.items() if k not in _blocked}
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        import time

        transient_attempts = 0
        hourly_attempts = 0
        timeout_attempts = 0
        patient_attempts = 0
        _notified_hourly = False
        _notified_patient = False

        try:
            while True:
                try:
                    result = subprocess.run(
                        cmd,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        env=env,
                        cwd=str(PROJECT_ROOT),
                    )

                    if result.returncode == 0:
                        output = self._unwrap_json_output(result.stdout.strip(), tag)
                        if output:
                            return output
                        # Empty output on success — treat as transient
                        logger.warning(f"{tag} Empty output on success, treating as transient")

                    # Non-zero exit or empty output — classify the error
                    stderr_text = result.stderr if result.returncode != 0 else ""
                    limit_type = classify_error(stderr_text, result.returncode, result.stdout)
                    stderr_preview = stderr_text[:300] if stderr_text else "(empty)"

                    if limit_type == RateLimitType.WEEKLY:
                        weekly_wait = 14400  # 4 hours — may need to wait overnight
                        msg = (
                            f"[proactive-affective-agent] Weekly rate limit hit\n"
                            f"Version: {self._version.upper()}, User: {self.study_id}\n"
                            f"Waiting {weekly_wait // 60}min and retrying (user may switch accounts)."
                        )
                        send_telegram(msg, dedup_key="weekly_agentic", dedup_ttl=7200)
                        logger.warning(f"{tag} Weekly rate limit. Waiting {weekly_wait}s...")
                        time.sleep(weekly_wait)
                        continue

                    if limit_type == RateLimitType.HOURLY:
                        hourly_attempts += 1
                        self._log_rate_limit_event("hourly")
                        if not _notified_hourly:
                            send_telegram(
                                f"[proactive-affective-agent] Rate limit hit — waiting 30min\n"
                                f"Version: {self._version.upper()}, User: {self.study_id}\n"
                                f"Will keep retrying (no fallback)."
                            )
                            _notified_hourly = True
                        logger.warning(
                            f"{tag} Hourly rate limit (attempt {hourly_attempts}). "
                            f"Waiting {self._HOURLY_WAIT}s..."
                        )
                        time.sleep(self._HOURLY_WAIT)
                        continue

                    # Transient error — fast retries first, then patient retry
                    transient_attempts += 1
                    if transient_attempts <= self._TRANSIENT_RETRIES:
                        wait = self._TRANSIENT_BACKOFF[min(transient_attempts - 1, len(self._TRANSIENT_BACKOFF) - 1)]
                        logger.warning(f"{tag} Transient error (attempt {transient_attempts}), retrying in {wait}s: {stderr_preview}")
                        time.sleep(wait)
                        continue

                    # Fast retries exhausted — switch to patient retry (likely rate limit)
                    patient_attempts += 1
                    self._log_rate_limit_event("patient_retry")
                    if not _notified_patient:
                        send_telegram(
                            f"[proactive-affective-agent] Rate limit (patient mode)\n"
                            f"Version: {self._version.upper()}, User: {self.study_id}\n"
                            f"Waiting {self._PATIENT_WAIT}s between retries. No fallback."
                        )
                        _notified_patient = True
                    logger.warning(
                        f"{tag} Patient retry #{patient_attempts}, waiting {self._PATIENT_WAIT}s: {stderr_preview}"
                    )
                    time.sleep(self._PATIENT_WAIT)
                    # Reset transient counter for next round of fast retries
                    transient_attempts = 0
                    continue

                except subprocess.TimeoutExpired:
                    timeout_attempts += 1
                    if timeout_attempts <= self._TIMEOUT_RETRIES:
                        wait = self._TIMEOUT_BACKOFF[min(timeout_attempts - 1, len(self._TIMEOUT_BACKOFF) - 1)]
                        logger.warning(f"{tag} Timeout (attempt {timeout_attempts}), retrying in {wait}s")
                        time.sleep(wait)
                        continue
                    # Timeout retries exhausted — patient retry
                    patient_attempts += 1
                    self._log_rate_limit_event("timeout_patient")
                    logger.warning(f"{tag} Timeout → patient retry #{patient_attempts}, waiting {self._PATIENT_WAIT}s")
                    time.sleep(self._PATIENT_WAIT)
                    timeout_attempts = 0
                    continue

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # Unexpected error — still don't fallback, raise to caller
            logger.error(f"{tag} Unexpected subprocess error: {exc}")
            raise
        finally:
            Path(mcp_config_path).unlink(missing_ok=True)

    def _unwrap_json_output(self, raw: str, tag: str = "") -> str:
        """Unwrap --output-format json response, extract token usage and turn count."""
        if not raw:
            return ""
        try:
            wrapper = json.loads(raw)
            if isinstance(wrapper, dict):
                usage = wrapper.get("usage", {})
                if isinstance(usage, dict):
                    self._last_usage = {
                        "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                        "cost_usd": wrapper.get("total_cost_usd", 0),
                    }
                    logger.info(f"{tag} tokens: {self._last_usage.get('input_tokens', 0)}in + {self._last_usage.get('output_tokens', 0)}out")
                # Extract num_turns from JSON wrapper (indicates tool-use rounds)
                num_turns = wrapper.get("num_turns", 1)
                self._last_num_turns = num_turns
                if num_turns > 1:
                    logger.info(f"{tag} Multi-turn: {num_turns} turns (tool-use confirmed)")
                if "result" in wrapper:
                    return str(wrapper["result"])
        except json.JSONDecodeError:
            pass
        return raw

    # ------------------------------------------------------------------
    # Private: prompt building
    # ------------------------------------------------------------------

    def _get_filtered_narrative(self, ema_date: str) -> str | None:
        """Look up the pre-computed filtered narrative for a given date."""
        if self._filtered_df is None:
            return None
        # Compare as strings to avoid date/datetime type mismatches
        match = self._filtered_df[self._filtered_df["date_local"].astype(str) == str(ema_date)]
        if match.empty:
            return None
        narrative = match.iloc[0].get("narrative", "")
        if pd.isna(narrative) or not str(narrative).strip():
            return None
        return str(narrative)

    def _build_prompt(
        self,
        ema_timestamp: str,
        ema_date: str,
        ema_slot: str,
        diary_text: str | None,
        session_memory: str | None = None,
    ) -> str:
        """Build the user prompt for claude --print."""
        # Diary section (only for multimodal modes: V4, V6)
        has_diary = self.mode in ("multimodal", "filtered_multimodal")
        if has_diary and diary_text and diary_text.strip() and diary_text.lower() != "nan":
            diary_section = f"""## Diary Entry (PRIMARY emotional signal — analyze this FIRST)
"{diary_text}" """
        elif has_diary:
            diary_section = "No diary entry for this EMA."
        else:
            diary_section = "(Sensing-only mode — no diary text available. Rely entirely on passive behavioral signals.)"

        # Filtered narrative section (V5/V6)
        narrative_section = ""
        if self.mode in ("filtered_sensing", "filtered_multimodal"):
            narrative = self._get_filtered_narrative(ema_date)
            if narrative:
                narrative_section = f"\n## Daily Behavioral Narrative (pre-computed summary for {ema_date})\n{narrative}\n"
            else:
                narrative_section = f"\n## Daily Behavioral Narrative\nNo filtered narrative available for {ema_date}. Rely on MCP tools to query raw data.\n"

        memory_excerpt = ""
        if self.memory_doc:
            memory_excerpt = f"\n## Baseline Personal History (pre-study memory)\n{self.memory_doc[:3000]}\n"

        session_section = ""
        if session_memory and session_memory.strip():
            trimmed = session_memory[-6000:] if len(session_memory) > 6000 else session_memory
            session_section = f"\n## Accumulated Session Memory (your prior observations of this person)\n{trimmed}\n"

        # Task instructions vary by mode
        if self.mode == "multimodal":
            task_instruction = (
                "1. FIRST: Analyze the diary entry above. What emotional themes, stressors, or positive signals does it reveal?\n"
                "2. THEN: Call get_behavioral_timeline to reconstruct the day and infer within-day shifts.\n"
                "3. Use sensing MCP tools to validate and calibrate your hypotheses.\n"
                "4. FINALLY: Synthesize diary + sensing evidence into a prediction in the required JSON format."
            )
        elif self.mode == "filtered_sensing":
            task_instruction = (
                "1. FIRST: Analyze the daily behavioral narrative above. Extract key signals about activity, social engagement, screen habits, and environmental context.\n"
                "2. THEN: Call get_behavioral_timeline to inspect within-day state changes.\n"
                "3. Use compare_to_baseline to check if today's behavior is unusual for this person.\n"
                "4. OPTIONALLY: Use query_sensing or query_raw_events if you need finer temporal detail (hourly patterns, exact timestamps).\n"
                "5. FINALLY: Synthesize narrative + tool evidence into a prediction in the required JSON format."
            )
        elif self.mode == "filtered_multimodal":
            task_instruction = (
                "1. FIRST: Analyze the diary entry above. What emotional themes, stressors, or positive signals does it reveal?\n"
                "2. THEN: Read the daily behavioral narrative. Does the behavior align with the diary? Look for cross-modal consistency.\n"
                "3. Call get_behavioral_timeline to inspect within-day state changes.\n"
                "4. Use compare_to_baseline to check if today's behavior is unusual for this person.\n"
                "5. OPTIONALLY: Drill into raw data with MCP tools where diary and narrative seem inconsistent.\n"
                "6. FINALLY: Synthesize diary + narrative + sensing evidence into a prediction in the required JSON format."
            )
        else:  # sensing_only
            task_instruction = (
                "Start by calling get_behavioral_timeline to reconstruct the day before the EMA.\n"
                "Investigate the sensing data using the available MCP tools to understand this person's behavioral state.\n"
                "Use the session memory above to calibrate against this person's known receptivity and behavioral patterns.\n"
                "Then predict their emotional state in the required JSON format."
            )

        # For V5/V6, start with compare_to_baseline instead of get_daily_summary
        # since the narrative already provides the daily overview
        if self.mode in ("filtered_sensing", "filtered_multimodal"):
            start_hint = (
                f"Start by calling get_behavioral_timeline for {ema_date}, then "
                f"compare_to_baseline for key features, then use find_similar_days "
                f"to look for analogous past days."
            )
        else:
            start_hint = f"Start by calling get_behavioral_timeline for {ema_date}, then get_daily_summary."

        return f"""You are investigating participant {self.pid}'s behavioral data to predict their emotional state.

## Current Situation
Timestamp: {ema_timestamp} ({ema_slot} EMA)
Date: {ema_date}
{diary_section}
{narrative_section}
## User Profile
{self.profile.to_text()}
{memory_excerpt}{session_section}
## Your Task
{task_instruction}

{start_hint}"""

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
            "get_daily_summary", "get_behavioral_timeline", "query_sensing", "query_raw_events",
            "compare_to_baseline", "get_receptivity_history", "find_similar_days",
            "find_peer_cases",
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

    def _log_rate_limit_event(self, event_type: str) -> None:
        """Log a rate limit event to a shared file for the queue runner."""
        import datetime
        rate_limit_log = PROJECT_ROOT / "outputs" / "pilot_v2" / ".rate_limit_events.jsonl"
        try:
            rate_limit_log.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "version": self._version,
                "user_id": self.study_id,
                "event_type": event_type,
            }
            with open(rate_limit_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Best-effort logging

    def _parse_prediction(self, text: str) -> dict[str, Any]:
        """Parse prediction. Returns dict with _parse_error=True if parsing fails.

        NOTE: No fallback here — caller (predict()) handles retries.
        """
        from src.think.parser import parse_prediction as _parse
        result = _parse(text)

        if result.get("_parse_error"):
            tag = f"[{self._version.upper()}]"
            logger.warning(f"{tag} Failed to parse prediction: {text[:300]}")
            result["_raw_response"] = text[:500]

        return result

    def _fallback_prediction(self) -> dict[str, Any]:
        """Legacy fallback — kept for old scripts. NOT used by predict() anymore."""
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
