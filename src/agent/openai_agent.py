"""Agentic OpenAI agent with function-calling tools (V2/V4/V5/V6 GPT variants)."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.agent.cc_agent import (
    SYSTEM_PROMPT_FILTERED_MULTIMODAL,
    SYSTEM_PROMPT_FILTERED_SENSING,
    SYSTEM_PROMPT_MULTIMODAL,
    SYSTEM_PROMPT_SENSING_ONLY,
)
from src.data.schema import UserProfile
from src.sense.query_tools import SensingQueryEngine
from src.utils.mappings import BINARY_STATE_TARGETS

logger = logging.getLogger(__name__)
USAGE_LIMIT_RE = re.compile(r"try again at (\d{1,2}:\d{2}\s*[AP]M)", re.IGNORECASE)


OPENAI_SENSING_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_sensing",
            "description": "Query passive sensing data for a modality and pre-EMA time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "enum": ["accelerometer", "gps", "motion", "screen", "keyboard", "music", "light"],
                    },
                    "hours_before_ema": {"type": "integer", "minimum": 1, "maximum": 48},
                    "hours_duration": {"type": "integer", "default": 1, "maximum": 24},
                    "granularity": {"type": "string", "enum": ["hourly", "daily"], "default": "hourly"},
                },
                "required": ["modality", "hours_before_ema"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_summary",
            "description": "Get daily behavioral summary across modalities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "lookback_days": {"type": "integer", "default": 0},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_behavioral_timeline",
            "description": "Reconstruct the day chronologically with coarse behavioral-state labels and affective cues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "segment_hours": {"type": "integer", "default": 3, "minimum": 1, "maximum": 6},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_to_baseline",
            "description": "Compare a feature value to participant's personal baseline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {"type": "string"},
                    "feature": {"type": "string"},
                    "current_value": {"type": "number"},
                },
                "required": ["modality", "feature", "current_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_receptivity_history",
            "description": "Retrieve historical receptivity patterns before EMA timestamp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_days": {"type": "integer", "default": 14},
                    "include_emotion_driver": {"type": "boolean", "default": False},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_days",
            "description": "Find behaviorally similar past days for this participant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "default": 5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_raw_events",
            "description": "Query raw event streams (screen/app/motion/keyboard/music).",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {"type": "string", "enum": ["screen", "app", "motion", "keyboard", "music"]},
                    "hours_before_ema": {"type": "integer"},
                    "hours_duration": {"type": "integer", "default": 4},
                    "max_events": {"type": "integer", "default": 30},
                },
                "required": ["modality", "hours_before_ema"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_peer_cases",
            "description": "Search other participants for similar text or sensing cases with outcomes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_mode": {"type": "string", "enum": ["text", "sensing"], "default": "text"},
                    "query_text": {"type": "string", "default": ""},
                    "top_k": {"type": "integer", "default": 5},
                },
            },
        },
    },
]


class AgenticOpenAIAgent:
    """OpenAI function-calling agent for V2/V4/V5/V6."""

    _MODE_TO_VERSION = {
        "sensing_only": "v2",
        "multimodal": "v4",
        "filtered_sensing": "v5",
        "filtered_multimodal": "v6",
    }

    _SYSTEM_BY_MODE = {
        "sensing_only": SYSTEM_PROMPT_SENSING_ONLY,
        "multimodal": SYSTEM_PROMPT_MULTIMODAL,
        "filtered_sensing": SYSTEM_PROMPT_FILTERED_SENSING,
        "filtered_multimodal": SYSTEM_PROMPT_FILTERED_MULTIMODAL,
    }

    _SENSING_FEATURE_COLS = [
        "screen_total_min", "screen_n_sessions",
        "motion_stationary_min", "motion_walking_min", "motion_automotive_min",
        "motion_tracked_hours",
        "app_total_min", "app_social_min", "app_comm_min", "app_entertainment_min",
        "keyboard_n_sessions", "keyboard_total_chars",
        "light_mean_lux_raw",
    ]

    def __init__(
        self,
        study_id: int,
        profile: UserProfile,
        memory_doc: str | None,
        processed_dir: Path,
        ema_df: pd.DataFrame,
        model: str = "gpt-5.1-codex-mini",
        max_turns: int = 16,
        mode: str = "multimodal",
        filtered_data_dir: Path | None = None,
        peer_db_path: str | None = None,
        dry_run: bool = False,
    ) -> None:
        if mode not in self._MODE_TO_VERSION:
            raise ValueError(f"Invalid mode: {mode}")
        self.study_id = study_id
        self.profile = profile
        self.memory_doc = memory_doc or ""
        self.processed_dir = Path(processed_dir)
        self.ema_df = ema_df if ema_df is not None else pd.DataFrame()
        self.model = model
        self.max_turns = max_turns
        self.mode = mode
        self.pid = str(study_id).zfill(3)
        self._version = self._MODE_TO_VERSION[mode]
        self.dry_run = dry_run

        self.query_engine = SensingQueryEngine(processed_dir=self.processed_dir, ema_df=self.ema_df)
        self.client = None
        self._last_usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

        self._filtered_df: pd.DataFrame | None = None
        filtered_root = filtered_data_dir or (self.processed_dir.parent / "filtered")
        filtered_path = Path(filtered_root) / f"{self.pid}_daily_filtered.parquet"
        if filtered_path.exists():
            self._filtered_df = pd.read_parquet(filtered_path)

        self._peer_db: pd.DataFrame | None = None
        self._peer_tfidf_vectorizer: TfidfVectorizer | None = None
        self._peer_tfidf_matrix = None
        self._peer_sensing_matrix = None
        self._peer_sensing_features: list[str] = []
        if peer_db_path and Path(peer_db_path).exists():
            self._load_peer_db(peer_db_path)

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                return None
        return None

    def _codex_exec(self, prompt: str) -> str:
        def _usage_limit_sleep_seconds(error_text: str) -> float | None:
            text = error_text.lower()
            if "usage limit" not in text and "purchase more credits" not in text:
                return None

            match = USAGE_LIMIT_RE.search(error_text)
            if not match:
                return 15 * 60

            try:
                target_time = datetime.strptime(match.group(1).upper(), "%I:%M %p").time()
            except ValueError:
                return 15 * 60

            now = datetime.now()
            target = now.replace(
                hour=target_time.hour,
                minute=target_time.minute,
                second=0,
                microsecond=0,
            )
            if target <= now:
                target += timedelta(days=1)
            return max((target - now).total_seconds() + 60, 60)

        def _run_once(p: str) -> str:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
                out_path = tmp.name
            cmd = [
                "codex", "exec", p,
                "--model", self.model,
                "--sandbox", "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--output-last-message", out_path,
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "codex exec failed")
                return Path(out_path).read_text(encoding="utf-8").strip()
            finally:
                Path(out_path).unlink(missing_ok=True)

        def _policy_safe_prompt(p: str) -> str:
            # Keep task semantics but remove raw diary content likely to trigger policy filters.
            p = re.sub(
                r'(##\s+Diary Entry[^\n]*\n)(.*?)(\n##\s+)',
                r"\1[Diary content redacted for safety retry]\3",
                p,
                flags=re.DOTALL,
            )
            p = re.sub(
                r'(##\s+Current Diary Entry[^\n]*\n)(.*?)(\n##\s+)',
                r"\1[Diary content redacted for safety retry]\3",
                p,
                flags=re.DOTALL,
            )
            return p

        active_prompt = prompt
        redacted_once = False
        while True:
            try:
                return _run_once(active_prompt)
            except Exception as exc:
                err_text = str(exc)
                msg = err_text.lower()
                if "invalid prompt" in msg or "usage policy" in msg:
                    if redacted_once:
                        raise
                    logger.warning("Codex prompt blocked by policy; retrying with redacted diary section")
                    active_prompt = _policy_safe_prompt(active_prompt)
                    redacted_once = True
                    continue

                limit_wait = _usage_limit_sleep_seconds(err_text)
                if limit_wait is not None:
                    logger.warning(
                        "Codex usage limit hit inside agentic loop. Sleeping %.1f minutes before retry.",
                        limit_wait / 60.0,
                    )
                    time.sleep(limit_wait)
                    continue
                raise

    def predict(
        self,
        ema_row: pd.Series,
        diary_text: str | None = None,
        session_memory: str | None = None,
    ) -> dict[str, Any]:
        ema_timestamp = str(ema_row.get("timestamp_local", ""))
        ema_date = str(ema_row.get("date_local", ""))
        ema_slot = self._get_ema_slot(ema_row)

        if self.mode in ("multimodal", "filtered_multimodal"):
            effective_diary = diary_text
        else:
            effective_diary = None

        prompt = self._build_prompt(
            ema_timestamp=ema_timestamp,
            ema_date=ema_date,
            ema_slot=ema_slot,
            diary_text=effective_diary,
            session_memory=session_memory,
        )
        system_prompt = self._SYSTEM_BY_MODE[self.mode]

        if self.dry_run:
            pred = self._fallback_prediction()
            pred["_version"] = self._version
            pred["_model"] = self.model
            pred["_full_prompt"] = prompt
            pred["_system_prompt"] = system_prompt
            pred["_full_response"] = json.dumps(pred)
            pred["_tool_calls"] = []
            pred["_n_tool_calls"] = 0
            pred["_n_rounds"] = 1
            pred["_conversation_length"] = 2
            pred["_input_tokens"] = 0
            pred["_output_tokens"] = 0
            pred["_total_tokens"] = 0
            pred["_cost_usd"] = 0.0
            pred["_llm_calls"] = 1
            pred["_has_diary"] = bool(effective_diary and effective_diary.strip() and effective_diary.lower() != "nan")
            pred["_diary_length"] = len(effective_diary) if pred["_has_diary"] else 0
            pred["_emotion_driver"] = effective_diary or ""
            return pred

        trace_tool_calls: list[dict[str, Any]] = []
        final_text = ""
        n_rounds = 0
        tool_unavailable_retries = 0
        total_input_tokens = 0
        total_output_tokens = 0
        llm_calls = 0
        transcript: list[dict[str, Any]] = []
        tools_spec = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            }
            for t in OPENAI_SENSING_TOOLS
        ]

        for _ in range(self.max_turns):
            n_rounds += 1
            step_prompt = (
                f"{system_prompt}\n\n"
                f"{prompt}\n\n"
                "You may call tools iteratively. Respond ONLY with JSON:\n"
                "{\n"
                '  "tool_calls": [{"name":"<tool_name>","arguments":{...}}],\n'
                '  "final_prediction": <object or null>,\n'
                '  "message": "<optional short rationale>"\n'
                "}\n"
                "Rules:\n"
                "- If you need data, return non-empty tool_calls and final_prediction=null.\n"
                "- If ready to answer, return tool_calls=[] and final_prediction as the prediction object.\n"
                "- Do not include markdown fences.\n\n"
                "Important:\n"
                "- The tools listed below are available through this JSON tool_calls protocol.\n"
                "- Do not say that tools are unavailable, disabled, or need to be enabled.\n"
                "- If you need data, request it with tool_calls instead of refusing.\n\n"
                f"Available tools:\n{json.dumps(tools_spec, ensure_ascii=False)}\n\n"
                f"Conversation so far:\n{json.dumps(transcript, ensure_ascii=False)}\n"
            )
            response_text = self._codex_exec(step_prompt)
            llm_calls += 1
            response_text_lower = response_text.lower()
            if (
                "tools unavailable" in response_text_lower
                or "tool is unavailable" in response_text_lower
                or "system isn’t letting me issue requests" in response_text_lower
                or "system isn't letting me issue requests" in response_text_lower
                or "enabling the tool calls" in response_text_lower
            ):
                tool_unavailable_retries += 1
                if tool_unavailable_retries <= 2:
                    transcript.append(
                        {
                            "role": "system",
                            "message": (
                                "Reminder: tools are available in this task via the JSON tool_calls envelope. "
                                "Do not refuse due to tool availability. Either emit tool_calls or return final_prediction."
                            ),
                        }
                    )
                    continue
            envelope = self._extract_json_object(response_text)
            if not envelope:
                final_text = response_text
                break

            raw_tool_calls = envelope.get("tool_calls", []) or []
            model_msg = str(envelope.get("message", "") or "")
            transcript.append({"role": "assistant", "message": model_msg, "tool_calls": raw_tool_calls})

            if not raw_tool_calls:
                fp = envelope.get("final_prediction")
                if isinstance(fp, dict):
                    final_text = json.dumps(fp)
                else:
                    final_text = response_text
                break

            for call in raw_tool_calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name", "") or "")
                args = call.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}

                tool_result = self._dispatch_tool(
                    tool_name=name,
                    tool_input=args,
                    ema_timestamp=ema_timestamp,
                    ema_date=ema_date,
                    diary_text=effective_diary or "",
                )

                trace_tool_calls.append(
                    {
                        "index": len(trace_tool_calls) + 1,
                        "tool_name": name,
                        "input": args,
                        "result_length": len(tool_result),
                        "result_preview": tool_result[:500],
                    }
                )
                transcript.append({"role": "tool", "name": name, "arguments": args, "result": tool_result[:4000]})

        if not final_text:
            final_step_prompt = (
                f"{system_prompt}\n\n"
                f"{prompt}\n\n"
                "You have reached the end of the tool-use budget.\n"
                "Do not request any more tools.\n"
                "Using only the evidence already present in the conversation so far, "
                "return ONLY the final prediction object as raw JSON with no markdown fences.\n\n"
                f"Conversation so far:\n{json.dumps(transcript, ensure_ascii=False)}\n"
            )
            final_text = self._codex_exec(final_step_prompt)
            llm_calls += 1

        from src.think.parser import parse_prediction

        parsed = parse_prediction(final_text)
        if parsed.get("_parse_error"):
            raise RuntimeError(
                f"{self._version} prediction parse failed: {final_text[:500]}"
            )

        self._last_usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost_usd": 0.0,
        }

        parsed["_version"] = self._version
        parsed["_model"] = self.model
        parsed["_full_response"] = final_text
        parsed["_system_prompt"] = system_prompt
        parsed["_full_prompt"] = prompt
        parsed["_has_diary"] = bool(effective_diary and effective_diary.strip() and effective_diary.lower() != "nan")
        parsed["_diary_length"] = len(effective_diary) if parsed["_has_diary"] else 0
        parsed["_emotion_driver"] = effective_diary or ""
        parsed["_tool_calls"] = trace_tool_calls
        parsed["_n_tool_calls"] = len(trace_tool_calls)
        parsed["_n_rounds"] = n_rounds
        parsed["_conversation_length"] = len(transcript)
        parsed["_input_tokens"] = total_input_tokens
        parsed["_output_tokens"] = total_output_tokens
        parsed["_total_tokens"] = total_input_tokens + total_output_tokens
        parsed["_cost_usd"] = 0.0
        parsed["_llm_calls"] = llm_calls
        return parsed

    def _dispatch_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        ema_timestamp: str,
        ema_date: str,
        diary_text: str,
    ) -> str:
        if tool_name == "find_peer_cases":
            search_mode = str(tool_input.get("search_mode", "text"))
            query_text = str(tool_input.get("query_text", "") or "")
            if search_mode == "text" and not query_text:
                query_text = diary_text
            top_k = int(tool_input.get("top_k", 5) or 5)
            return self._find_peer_cases(search_mode=search_mode, query_text=query_text, top_k=top_k, ema_date=ema_date)

        return self.query_engine.call_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            study_id=self.study_id,
            ema_timestamp=ema_timestamp,
        )

    def _find_peer_cases(self, search_mode: str, query_text: str, top_k: int, ema_date: str) -> str:
        if self._peer_db is None:
            return "Peer database not available."
        top_k = min(max(top_k, 1), 10)
        if search_mode == "text":
            return self._peer_search_text(query_text=query_text, top_k=top_k)
        if search_mode == "sensing":
            return self._peer_search_sensing(top_k=top_k, ema_date=ema_date)
        return f"Unknown search_mode '{search_mode}'. Use 'text' or 'sensing'."

    def _load_peer_db(self, path: str) -> None:
        db = pd.read_parquet(path)
        if "Study_ID" in db.columns:
            db = db[db["Study_ID"] != self.study_id].reset_index(drop=True)
        if db.empty:
            return
        self._peer_db = db

        if "emotion_driver" in db.columns:
            texts = db["emotion_driver"].fillna("").astype(str).tolist()
            if any(t.strip() for t in texts):
                vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=1)
                try:
                    self._peer_tfidf_matrix = vec.fit_transform(texts)
                    self._peer_tfidf_vectorizer = vec
                except ValueError:
                    self._peer_tfidf_matrix = None
                    self._peer_tfidf_vectorizer = None

        available = [c for c in self._SENSING_FEATURE_COLS if c in db.columns]
        if available:
            self._peer_sensing_features = available
            data = db[available].fillna(0).values.astype(np.float64)
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            stds[stds == 0] = 1.0
            self._peer_sensing_matrix = (data - means) / stds

    def _peer_search_text(self, query_text: str, top_k: int) -> str:
        if self._peer_tfidf_matrix is None or self._peer_tfidf_vectorizer is None:
            return "Text-based peer search not available."
        if not query_text.strip():
            return "No query text provided for text-based peer search."
        query_vec = self._peer_tfidf_vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self._peer_tfidf_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        lines = [f"Found {min(top_k, len(top_idx))} similar cases from peers (text):\n"]
        for rank, idx in enumerate(top_idx, 1):
            if sims[idx] <= 0:
                break
            row = self._peer_db.iloc[idx]
            lines.append(f"--- Peer Case {rank} (similarity: {sims[idx]:.3f}) ---")
            diary = str(row.get("emotion_driver", "") or "").strip()
            if diary and diary.lower() != "nan":
                lines.append(f"  Diary: {diary[:300]}")
            self._append_outcome_lines(row, lines)
            lines.append("")
        return "\n".join(lines) if len(lines) > 1 else "No similar text cases found."

    def _peer_search_sensing(self, top_k: int, ema_date: str) -> str:
        if self._peer_sensing_matrix is None or not self._peer_sensing_features:
            return "Sensing-based peer search not available."
        current = self._get_current_sensing_vector(ema_date)
        if current is None:
            return "Cannot build current sensing fingerprint for this EMA date."
        available_cols = [c for c in self._SENSING_FEATURE_COLS if c in self._peer_db.columns]
        peer_raw = self._peer_db[available_cols].fillna(0).values.astype(np.float64)
        means = peer_raw.mean(axis=0)
        stds = peer_raw.std(axis=0)
        stds[stds == 0] = 1.0
        norm_current = ((current - means) / stds).reshape(1, -1)
        sims = cosine_similarity(norm_current, self._peer_sensing_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        lines = [f"Found {min(top_k, len(top_idx))} similar cases from peers (sensing):\n"]
        for rank, idx in enumerate(top_idx, 1):
            if sims[idx] <= 0:
                break
            row = self._peer_db.iloc[idx]
            lines.append(f"--- Peer Case {rank} (similarity: {sims[idx]:.3f}) ---")
            sensing_parts = []
            for feat in self._peer_sensing_features:
                val = row.get(feat)
                if val is not None and pd.notna(val) and val != 0:
                    sensing_parts.append(f"{feat}={val:.1f}")
            if sensing_parts:
                lines.append(f"  Behavior: {'; '.join(sensing_parts)}")
            self._append_outcome_lines(row, lines)
            lines.append("")
        return "\n".join(lines) if len(lines) > 1 else "No similar sensing cases found."

    def _get_current_sensing_vector(self, ema_date: str) -> np.ndarray | None:
        if self._filtered_df is None:
            return None
        match = self._filtered_df[self._filtered_df["date_local"].astype(str) == str(ema_date)]
        if match.empty:
            return None
        row = match.iloc[0]
        available = [c for c in self._SENSING_FEATURE_COLS if c in self._peer_db.columns]
        vector = []
        for col in available:
            val = row.get(col, 0)
            vector.append(float(val) if pd.notna(val) else 0.0)
        return np.array(vector, dtype=np.float64)

    @staticmethod
    def _append_outcome_lines(row: pd.Series, lines: list[str]) -> None:
        outcome_parts = []
        for col, label in [("PANAS_Pos", "PA"), ("PANAS_Neg", "NA"), ("ER_desire", "ER")]:
            val = row.get(col)
            if val is not None and pd.notna(val):
                outcome_parts.append(f"{label}={float(val):.1f}")
        avail = row.get("INT_availability")
        if avail is not None and pd.notna(avail):
            outcome_parts.append(f"Avail={avail}")
        if outcome_parts:
            lines.append(f"  Outcomes: {', '.join(outcome_parts)}")
        state_parts = []
        for col in [
            "Individual_level_PA_State",
            "Individual_level_NA_State",
            "Individual_level_ER_desire_State",
        ]:
            val = row.get(col)
            if val is not None and pd.notna(val):
                label = col.replace("Individual_level_", "").replace("_State", "")
                state_parts.append(f"{label}={'high' if bool(val) else 'typical'}")
        if state_parts:
            lines.append(f"  States: {', '.join(state_parts)}")

    def _get_filtered_narrative(self, ema_date: str) -> str | None:
        if self._filtered_df is None:
            return None
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
        has_diary = self.mode in ("multimodal", "filtered_multimodal")
        if has_diary and diary_text and diary_text.strip() and diary_text.lower() != "nan":
            diary_section = f"""## Diary Entry (PRIMARY emotional signal — analyze this FIRST)
"{diary_text}" """
        elif has_diary:
            diary_section = "No diary entry for this EMA."
        else:
            diary_section = "(Sensing-only mode — no diary text available. Rely entirely on passive behavioral signals.)"

        narrative_section = ""
        if self.mode in ("filtered_sensing", "filtered_multimodal"):
            narrative = self._get_filtered_narrative(ema_date)
            if narrative:
                narrative_section = f"\n## Daily Behavioral Narrative (pre-computed summary for {ema_date})\n{narrative}\n"
            else:
                narrative_section = f"\n## Daily Behavioral Narrative\nNo filtered narrative available for {ema_date}. Rely on tools to query raw data.\n"

        memory_excerpt = f"\n## Baseline Personal History (pre-study memory)\n{self.memory_doc[:3000]}\n" if self.memory_doc else ""
        session_section = ""
        if session_memory and session_memory.strip():
            trimmed = session_memory[-6000:] if len(session_memory) > 6000 else session_memory
            session_section = f"\n## Accumulated Session Memory (your prior observations of this person)\n{trimmed}\n"

        if self.mode == "multimodal":
            task_instruction = (
                "1. FIRST: Analyze the diary entry.\n"
                "2. THEN: Call get_behavioral_timeline to inspect within-day shifts.\n"
                "3. Use tools to validate/calibrate hypotheses.\n"
                "4. FINALLY: Output prediction in required JSON format."
            )
        elif self.mode == "filtered_sensing":
            task_instruction = (
                "1. FIRST: Analyze the behavioral narrative.\n"
                "2. THEN: Call get_behavioral_timeline to inspect within-day shifts.\n"
                "3. Use compare_to_baseline and find_similar_days.\n"
                "4. OPTIONALLY: query_sensing/query_raw_events for drill-down.\n"
                "5. FINALLY: Output prediction in required JSON format."
            )
        elif self.mode == "filtered_multimodal":
            task_instruction = (
                "1. FIRST: Analyze diary.\n"
                "2. THEN: Analyze narrative and compare consistency.\n"
                "3. Call get_behavioral_timeline to inspect within-day shifts.\n"
                "4. Use tools for calibration where needed.\n"
                "5. FINALLY: Output prediction in required JSON format."
            )
        else:
            task_instruction = (
                "Start with get_behavioral_timeline, then investigate sensing data with tools and predict emotional state in required JSON format."
            )

        start_hint = (
            f"Start by calling get_behavioral_timeline for {ema_date}, then compare_to_baseline for key features, then find_similar_days."
            if self.mode in ("filtered_sensing", "filtered_multimodal")
            else f"Start by calling get_behavioral_timeline for {ema_date}, then get_daily_summary."
        )

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

    def _fallback_prediction(self) -> dict[str, Any]:
        """Dry-run only placeholder prediction.

        Non-dry execution must never silently fall back. Parse or generation
        failures are raised so the entry is retried later instead of being
        checkpointed as a fake valid prediction.
        """
        pred: dict[str, Any] = {
            "PANAS_Pos": 15.0,
            "PANAS_Neg": 8.0,
            "ER_desire": 3.0,
            "INT_availability": "yes",
            "reasoning": f"[{self._version} fallback prediction]",
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
            if hour < 17:
                return "afternoon"
            return "evening"
        except Exception:
            return "unknown"
