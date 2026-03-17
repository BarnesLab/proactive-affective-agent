"""Rate-limit error classification and Telegram notification helpers.

Used by cc_agent.py (agentic agents) and llm_client.py (structured agents)
to detect Claude Max rate limits and respond appropriately:
- Transient errors: short backoff retries
- Hourly rate limit (5-hour rolling window): wait 30 minutes, retry
- Weekly limit: alert user, stop experiment gracefully
"""

from __future__ import annotations

import json
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

# Telegram Boo bot for user notifications
_BOT_TOKEN = "7740709485:AAF35LkeavJ5-F4C6hcG5PC_7RdC9AeI8lI"
_CHAT_ID = "7542082932"


class RateLimitType:
    NONE = "none"
    HOURLY = "hourly"
    WEEKLY = "weekly"
    TRANSIENT = "transient"


class RateLimitError(Exception):
    """Raised when a non-recoverable rate limit is hit (e.g. weekly cap)."""

    def __init__(self, message: str, limit_type: str = RateLimitType.WEEKLY):
        super().__init__(message)
        self.limit_type = limit_type


def classify_error(stderr: str, returncode: int, stdout: str = "") -> str:
    """Classify a claude CLI error into a rate limit category.

    Args:
        stderr: Standard error output from the subprocess.
        returncode: Process exit code (0 = success).
        stdout: Standard output (checked for limit messages in result field
                when returncode==0 but output indicates an error).

    Returns:
        One of RateLimitType constants.
    """
    # Check stdout/result for limit messages even on returncode==0.
    # Claude CLI can return exit 0 with is_error=true and a limit message
    # in the "result" field (e.g. "You've hit your limit · resets Mar 22").
    all_text = ((stderr or "") + " " + (stdout or "")).lower()

    # Weekly / hard cap patterns
    weekly_patterns = [
        r"weekly\s*(usage\s*)?limit",
        r"weekly\s*cap",
        r"quota\s*exceeded",
        r"billing.*limit",
        r"max\s*usage.*reached",
        r"you'?ve\s+hit\s+your\s+limit",
        r"hit\s+your\s+limit.*resets",
    ]
    for pat in weekly_patterns:
        if re.search(pat, all_text):
            return RateLimitType.WEEKLY

    # If returncode==0 and no limit detected, it's not an error
    if returncode == 0:
        return RateLimitType.NONE

    text = stderr.lower() if stderr else ""

    # Hourly / rolling-window rate limit patterns
    hourly_patterns = [
        r"rate\s*limit",
        r"too\s*many\s*requests",
        r"429",
        r"throttl",
        r"overloaded",
        r"capacity",
        r"try\s*again\s*later",
        r"resource.*exhausted",
        r"concurrent.*limit",
    ]
    for pat in hourly_patterns:
        if re.search(pat, text):
            return RateLimitType.HOURLY

    # Non-zero exit but no known rate-limit pattern → transient error
    return RateLimitType.TRANSIENT


def send_telegram(text: str) -> None:
    """Send a notification message via Telegram Boo bot.

    Best-effort: logs but does not raise on failure.
    """
    try:
        payload = json.dumps({"chat_id": _CHAT_ID, "text": text})
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage",
                "-H", "Content-Type: application/json",
                "-d", payload,
            ],
            capture_output=True,
            timeout=15,
        )
        logger.info(f"Telegram notification sent: {text[:80]}")
    except Exception as exc:
        logger.warning(f"Failed to send Telegram notification: {exc}")
