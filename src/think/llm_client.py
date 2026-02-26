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

logger = logging.getLogger(__name__)

# Path to the JSON schema for structured output
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prediction_schema.json"


class ClaudeCodeClient:
    """Wraps the Claude Code CLI (`claude -p`) for LLM calls."""

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
        """Generate a completion using Claude Code CLI.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            json_schema: Optional JSON schema dict for structured output.
            schema_path: Optional path to JSON schema file (alternative to json_schema).

        Returns:
            Raw text response from Claude.

        Side effect:
            Sets self.last_usage with token counts from the most recent call.
        """
        if self.dry_run:
            self.last_usage = {"input_tokens": 0, "output_tokens": 0}
            return self._dry_run_response(prompt)

        cmd = self._build_command(prompt, system_prompt, json_schema, schema_path)

        for attempt in range(1, self.max_retries + 1):
            try:
                # Rate limiting
                if self._call_count > 0:
                    time.sleep(self.delay_between_calls)

                logger.debug(f"LLM call #{self._call_count + 1} (attempt {attempt})")

                # Remove all CLAUDE* env vars to avoid nested session detection
                env = {k: v for k, v in os.environ.items()
                       if not k.upper().startswith("CLAUDE")}

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                )

                self._call_count += 1

                if result.returncode != 0:
                    stderr = result.stderr.strip()
                    logger.warning(f"Claude CLI returned code {result.returncode}: {stderr}")
                    if attempt < self.max_retries:
                        wait = self.backoff_base ** attempt
                        logger.info(f"Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"Claude CLI failed after {self.max_retries} attempts: {stderr}")

                raw = result.stdout.strip()
                # Claude CLI --output-format json wraps in {"result": "...", ...}
                return self._unwrap_cli_response(raw)

            except subprocess.TimeoutExpired:
                logger.warning(f"Claude CLI timed out after {self.timeout}s (attempt {attempt})")
                if attempt < self.max_retries:
                    wait = self.backoff_base ** attempt
                    time.sleep(wait)
                    continue
                raise

        return ""

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict:
        """Generate a structured JSON response using the prediction schema.

        Returns:
            Parsed JSON dict.
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            schema_path=SCHEMA_PATH if SCHEMA_PATH.exists() else None,
        )

        if self.dry_run:
            return json.loads(response) if response.strip().startswith("{") else {}

        # Parse JSON from response
        from src.think.parser import parse_prediction
        return parse_prediction(response)

    def _build_command(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_schema: dict | None = None,
        schema_path: Path | None = None,
    ) -> list[str]:
        """Build the `claude` CLI command."""
        cmd = [
            "claude", "-p", prompt,
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
