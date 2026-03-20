"""LLM client wrapping the Claude Code CLI for the pilot study.

Uses `claude -p` with Max subscription — no API cost. Falls back gracefully
for dry-run mode (returns empty response without calling Claude).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from src.utils.rate_limit import (
    RateLimitError,
    RateLimitType,
    classify_error,
    send_telegram,
)

logger = logging.getLogger(__name__)

# Path to the JSON schema for structured output
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prediction_schema.json"


class ClaudeCodeClient:
    """Wraps the Claude Code CLI (`claude -p`) for LLM calls."""

    # Retry constants (mirrored from AgenticCCAgent._run_claude)
    _TRANSIENT_RETRIES = 3
    _TRANSIENT_BACKOFF = [2, 4, 8]  # seconds
    _PATIENT_WAIT = 300  # 5 minutes — slow retry after fast retries exhausted
    _HOURLY_WAIT = 1800  # 30 minutes
    _HOURLY_MAX_RETRIES = 12  # up to 6 hours total
    _TIMEOUT_RETRIES = 3
    _TIMEOUT_BACKOFF = [10, 20, 40]

    def __init__(
        self,
        model: str = "sonnet",
        timeout: int = 300,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        delay_between_calls: float = 2.0,
        dry_run: bool = False,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.delay_between_calls = delay_between_calls
        self.dry_run = dry_run
        self._call_count = 0
        self.last_usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_schema: dict | None = None,
        schema_path: Path | None = None,
    ) -> str:
        """Generate a completion using Claude Code CLI with robust retry logic.

        Retry strategy (same as AgenticCCAgent._run_claude):
        - Transient errors: 3 fast retries with exponential backoff (2s, 4s, 8s)
        - Patient retry: after fast retries exhausted, wait 5 min and retry
        - Hourly rate limit: wait 30 min, retry up to 12 times (6h total)
        - Weekly rate limit: send Telegram alert, raise RateLimitError
        - Timeout: 3 retries with backoff (10s, 20s, 40s), then patient retry

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            json_schema: Optional JSON schema dict for structured output.
            schema_path: Optional path to JSON schema file (alternative to json_schema).

        Returns:
            Raw text response from Claude.

        Raises:
            RateLimitError: On weekly rate limit (caller should stop experiment).
            RuntimeError: On unexpected unrecoverable errors.
            subprocess.TimeoutExpired: Should not propagate (handled internally).

        Side effect:
            Sets self.last_usage with token counts from the most recent call.
        """
        if self.dry_run:
            self.last_usage = {"input_tokens": 0, "output_tokens": 0}
            return self._dry_run_response(prompt)

        cmd = self._build_command(system_prompt, json_schema, schema_path)

        # Remove all CLAUDE* env vars to avoid nested session detection
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith("CLAUDE")}

        transient_attempts = 0
        hourly_attempts = 0
        timeout_attempts = 0
        patient_attempts = 0
        _notified_hourly = False
        _notified_patient = False

        while True:
            try:
                # Rate limiting between calls
                if self._call_count > 0:
                    time.sleep(self.delay_between_calls)

                logger.debug(
                    f"LLM call #{self._call_count + 1} "
                    f"(transient={transient_attempts}, hourly={hourly_attempts}, "
                    f"timeout={timeout_attempts}, patient={patient_attempts})"
                )

                result = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                )

                self._call_count += 1

                if result.returncode == 0:
                    raw = result.stdout.strip()
                    output = self._unwrap_cli_response(raw)
                    if output:
                        return output
                    # Empty output on success — treat as transient
                    logger.warning("Empty output on success, treating as transient")

                # Non-zero exit or empty output — classify the error
                stderr = result.stderr.strip() if result.returncode != 0 else ""
                limit_type = classify_error(stderr, result.returncode, result.stdout)
                stderr_preview = stderr[:300] if stderr else "(empty)"

                if limit_type == RateLimitType.WEEKLY:
                    weekly_wait = 3600  # 1 hour
                    send_telegram(
                        f"[proactive-affective-agent] Weekly rate limit hit (structured agent)\n"
                        f"Waiting {weekly_wait // 60}min and retrying (user may switch accounts).",
                        dedup_key="weekly_structured",
                        dedup_ttl=7200,
                    )
                    logger.warning(
                        f"Weekly rate limit. Waiting {weekly_wait}s..."
                    )
                    import time as _time
                    _time.sleep(weekly_wait)
                    continue

                if limit_type == RateLimitType.HOURLY:
                    hourly_attempts += 1
                    if not _notified_hourly:
                        send_telegram(
                            f"[proactive-affective-agent] Rate limit hit (structured agent) — waiting 30min\n"
                            f"Will keep retrying (no fallback)."
                        )
                        _notified_hourly = True
                    logger.warning(
                        f"Hourly rate limit (attempt {hourly_attempts}/{self._HOURLY_MAX_RETRIES}). "
                        f"Waiting {self._HOURLY_WAIT}s..."
                    )
                    time.sleep(self._HOURLY_WAIT)
                    continue

                # Transient error — fast retries first, then patient retry
                transient_attempts += 1
                if transient_attempts <= self._TRANSIENT_RETRIES:
                    wait = self._TRANSIENT_BACKOFF[min(transient_attempts - 1, len(self._TRANSIENT_BACKOFF) - 1)]
                    logger.warning(
                        f"Transient error (attempt {transient_attempts}/{self._TRANSIENT_RETRIES}), "
                        f"retrying in {wait}s: {stderr_preview}"
                    )
                    time.sleep(wait)
                    continue

                # Fast retries exhausted — switch to patient retry
                patient_attempts += 1
                if not _notified_patient:
                    send_telegram(
                        f"[proactive-affective-agent] Rate limit (patient mode, structured agent)\n"
                        f"Waiting {self._PATIENT_WAIT}s between retries. No fallback."
                    )
                    _notified_patient = True
                logger.warning(
                    f"Patient retry #{patient_attempts}, waiting {self._PATIENT_WAIT}s: {stderr_preview}"
                )
                time.sleep(self._PATIENT_WAIT)
                # Reset transient counter for next round of fast retries
                transient_attempts = 0
                continue

            except subprocess.TimeoutExpired:
                timeout_attempts += 1
                if timeout_attempts <= self._TIMEOUT_RETRIES:
                    wait = self._TIMEOUT_BACKOFF[min(timeout_attempts - 1, len(self._TIMEOUT_BACKOFF) - 1)]
                    logger.warning(f"Claude CLI timed out (attempt {timeout_attempts}/{self._TIMEOUT_RETRIES}), retrying in {wait}s")
                    time.sleep(wait)
                    continue
                # Timeout retries exhausted — patient retry
                patient_attempts += 1
                if not _notified_patient:
                    send_telegram(
                        f"[proactive-affective-agent] Timeout → patient mode (structured agent)\n"
                        f"Waiting {self._PATIENT_WAIT}s between retries. No fallback."
                    )
                    _notified_patient = True
                logger.warning(f"Timeout → patient retry #{patient_attempts}, waiting {self._PATIENT_WAIT}s")
                time.sleep(self._PATIENT_WAIT)
                timeout_attempts = 0
                continue

            except RateLimitError:
                raise
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Unexpected error — don't silently return empty, raise to caller
                logger.error(f"Unexpected error in generate(): {exc}")
                raise

    # Parse-retry constants (mirrors AgenticCCAgent)
    _MAX_PARSE_RETRIES = 5
    _PARSE_RETRY_WAIT = 30  # seconds

    def generate_and_parse(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, dict]:
        """Generate + parse with retry. Returns (raw_response, parsed_dict).

        Retries up to _MAX_PARSE_RETRIES times if LLM output fails to parse.
        Raises RuntimeError after all retries exhausted.
        """
        from src.think.parser import parse_prediction

        for attempt in range(1, self._MAX_PARSE_RETRIES + 1):
            raw = self.generate(prompt=prompt, system_prompt=system_prompt)
            result = parse_prediction(raw)
            if not result.get("_parse_error"):
                return raw, result

            if attempt < self._MAX_PARSE_RETRIES:
                logger.warning(
                    f"Parse failed (attempt {attempt}/{self._MAX_PARSE_RETRIES}), "
                    f"re-calling claude in {self._PARSE_RETRY_WAIT}s. "
                    f"Output: {raw[:200]}"
                )
                time.sleep(self._PARSE_RETRY_WAIT)
            else:
                logger.error(
                    f"Parse failed after {self._MAX_PARSE_RETRIES} attempts. "
                    f"Last output: {raw[:500]}"
                )
                raise RuntimeError(
                    f"Unparseable response after {self._MAX_PARSE_RETRIES} attempts"
                )

        return raw, result  # unreachable, satisfies type checker

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict:
        """Generate a structured JSON response using the prediction schema.

        Retries up to _MAX_PARSE_RETRIES times if the LLM output cannot be
        parsed into valid JSON (mirrors AgenticCCAgent.predict() behaviour).

        Returns:
            Parsed JSON dict.

        Raises:
            RuntimeError: If parsing fails after all retries.
        """
        if self.dry_run:
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                schema_path=SCHEMA_PATH if SCHEMA_PATH.exists() else None,
            )
            return json.loads(response) if response.strip().startswith("{") else {}

        from src.think.parser import parse_prediction

        for attempt in range(1, self._MAX_PARSE_RETRIES + 1):
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                schema_path=SCHEMA_PATH if SCHEMA_PATH.exists() else None,
            )
            result = parse_prediction(response)
            if not result.get("_parse_error"):
                return result

            if attempt < self._MAX_PARSE_RETRIES:
                logger.warning(
                    f"Parse failed (attempt {attempt}/{self._MAX_PARSE_RETRIES}), "
                    f"re-calling claude in {self._PARSE_RETRY_WAIT}s. "
                    f"Output: {response[:200]}"
                )
                time.sleep(self._PARSE_RETRY_WAIT)
            else:
                logger.error(
                    f"Parse failed after {self._MAX_PARSE_RETRIES} attempts. "
                    f"Last output: {response[:500]}"
                )
                raise RuntimeError(
                    f"Unparseable response after {self._MAX_PARSE_RETRIES} attempts"
                )

        # Should never reach here, but satisfy type checker
        return result

    def _build_command(
        self,
        system_prompt: str | None = None,
        json_schema: dict | None = None,
        schema_path: Path | None = None,
    ) -> list[str]:
        """Build the `claude` CLI command.

        The prompt is NOT included here — it is passed via stdin to
        subprocess.run(input=prompt) to avoid shell escaping issues with
        diary text containing quotes, newlines, and special characters.
        """
        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--model", self.model,
            "--no-session-persistence",
            "--tools", "",
        ]

        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        return cmd

    def _dry_run_response(self, prompt: str) -> str:
        """Return a placeholder response for dry-run mode."""
        logger.info(f"[DRY RUN] Prompt length: {len(prompt)} chars")
        self._call_count += 1
        return json.dumps({
            "PANAS_Pos": 15.0,
            "PANAS_Neg": 8.0,
            "ER_desire": 3.0,
            "Individual_level_PA_State": False,
            "Individual_level_NA_State": False,
            "Individual_level_happy_State": False,
            "Individual_level_sad_State": False,
            "Individual_level_afraid_State": False,
            "Individual_level_miserable_State": False,
            "Individual_level_worried_State": False,
            "Individual_level_cheerful_State": False,
            "Individual_level_pleased_State": False,
            "Individual_level_grateful_State": False,
            "Individual_level_lonely_State": False,
            "Individual_level_interactions_quality_State": False,
            "Individual_level_pain_State": False,
            "Individual_level_forecasting_State": False,
            "Individual_level_ER_desire_State": False,
            "INT_availability": "yes",
            "reasoning": "[DRY RUN] Placeholder prediction",
            "confidence": 0.5,
        })

    def _unwrap_cli_response(self, raw: str) -> str:
        """Unwrap Claude CLI JSON output format.

        The CLI with --output-format json returns:
        {"type":"result","result":"<actual LLM output>","usage":{...},...}
        We extract the "result" field and store token usage in self.last_usage.
        """
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}
        if not raw:
            return ""
        try:
            wrapper = json.loads(raw)
            if isinstance(wrapper, dict):
                # Extract token usage from CLI wrapper
                usage = wrapper.get("usage", {})
                if isinstance(usage, dict):
                    self.last_usage = {
                        "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                        "cost_usd": wrapper.get("total_cost_usd", 0),
                    }
                if "result" in wrapper:
                    return str(wrapper["result"])
        except json.JSONDecodeError:
            pass
        return raw

    @property
    def call_count(self) -> int:
        return self._call_count
