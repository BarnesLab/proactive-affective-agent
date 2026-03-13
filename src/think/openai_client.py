"""Codex CLI client for GPT structured pilot agents (CALLM/V1/V3).

Uses `codex exec` so calls consume Codex/ChatGPT limits, not OpenAI API keys.
"""

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

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prediction_schema.json"
USAGE_LIMIT_RE = re.compile(r"try again at (\d{1,2}:\d{2}\s*[AP]M)", re.IGNORECASE)


class OpenAIClient:
    """Wrapper around Codex CLI (`codex exec`) with retry + dry-run support."""

    def __init__(
        self,
        model: str = "gpt-4o",
        timeout: int = 300,
        max_retries: int = 50,
        backoff_base: float = 2.0,
        delay_between_calls: float = 1.0,
        temperature: float = 0.0,
        dry_run: bool = False,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.delay_between_calls = delay_between_calls
        self.temperature = temperature
        self.dry_run = dry_run
        self._call_count = 0
        self.last_usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    @staticmethod
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

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_schema: dict | None = None,
        schema_path=None,
    ) -> str:
        """Generate a response string from Codex CLI."""
        if self.dry_run:
            self.last_usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            return self._dry_run_response(prompt)

        merged_prompt = prompt
        if system_prompt:
            merged_prompt = f"{system_prompt}\n\n{prompt}"

        attempt = 1
        while True:
            if self._call_count > 0:
                time.sleep(self.delay_between_calls)
            try:
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
                    out_path = tmp.name
                cmd = [
                    "codex", "exec", merged_prompt,
                    "--model", self.model,
                    "--sandbox", "read-only",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--output-last-message", out_path,
                ]
                if schema_path or (json_schema and isinstance(json_schema, dict)):
                    schema_file = str(schema_path or SCHEMA_PATH)
                    if Path(schema_file).exists():
                        cmd.extend(["--output-schema", schema_file])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
                self._call_count += 1
                if result.returncode != 0:
                    err = result.stderr.strip() or result.stdout.strip() or "codex exec failed"
                    err_l = err.lower()
                    if "invalid prompt" in err_l or "usage policy" in err_l:
                        safe_prompt = re.sub(
                            r'(##\s+Current Diary Entry[^\n]*\n)(.*?)(\n##\s+)',
                            r"\1[Diary content redacted for safety retry]\3",
                            merged_prompt,
                            flags=re.DOTALL,
                        )
                        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp2:
                            out_path2 = tmp2.name
                        cmd2 = cmd.copy()
                        cmd2[2] = safe_prompt
                        # keep same output path slot
                        cmd2[-1] = out_path2
                        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=self.timeout)
                        self._call_count += 1
                        if result2.returncode == 0:
                            self.last_usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                            try:
                                return Path(out_path2).read_text(encoding="utf-8").strip()
                            finally:
                                Path(out_path2).unlink(missing_ok=True)
                        raise RuntimeError(result2.stderr.strip() or result2.stdout.strip() or err)
                    raise RuntimeError(err)

                self.last_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
                try:
                    text = Path(out_path).read_text(encoding="utf-8").strip()
                finally:
                    Path(out_path).unlink(missing_ok=True)
                return text
            except Exception as exc:
                err_raw = str(exc)
                limit_wait = self._usage_limit_sleep_seconds(err_raw)
                if limit_wait is not None:
                    logger.warning(
                        "Codex usage limit hit. Sleeping %.1f minutes before retry: %s",
                        limit_wait / 60.0,
                        err_raw[:240],
                    )
                    time.sleep(limit_wait)
                    continue

                err_text = err_raw.lower()
                is_retryable = any(
                    key in err_text
                    for key in (
                        "rate limit",
                        "429",
                        "timeout",
                        "temporarily",
                        "overloaded",
                        "connection",
                        "try again",
                    )
                )
                if attempt < self.max_retries and is_retryable:
                    wait = self.backoff_base ** attempt
                    logger.warning(
                        "OpenAI call failed (attempt %s/%s), retrying in %.1fs: %s",
                        attempt,
                        self.max_retries,
                        wait,
                        err_raw[:240],
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise

    def _dry_run_response(self, prompt: str) -> str:
        logger.info("[DRY RUN] OpenAI prompt length: %s chars", len(prompt))
        self._call_count += 1
        return json.dumps(
            {
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
            }
        )

    @property
    def call_count(self) -> int:
        return self._call_count
