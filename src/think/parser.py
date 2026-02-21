"""Structured output parsing from LLM responses.

Parses LLM text/JSON output into structured prediction dicts.
Handles various response formats (raw JSON, ```json blocks, mixed text+JSON).
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS


def parse_prediction(response: str) -> dict[str, Any]:
    """Parse a prediction response into structured output.

    Extracts JSON from the response, validates ranges, and fills defaults.

    Args:
        response: Raw LLM response text (may contain JSON block).

    Returns:
        Dict with all prediction targets + reasoning + confidence.
    """
    raw = parse_json_block(response)
    if not raw:
        return {"_parse_error": True, "_raw_response": response[:500]}

    result = {}

    # If claude CLI returns {"result": "...", "cost_usd": ...} wrapper, unwrap it
    if "result" in raw and isinstance(raw["result"], str):
        inner = parse_json_block(raw["result"])
        if inner:
            raw = inner

    # Continuous targets: clamp to valid ranges
    for target, (lo, hi) in CONTINUOUS_TARGETS.items():
        val = raw.get(target)
        if val is not None:
            try:
                val = float(val)
                result[target] = max(lo, min(hi, val))
            except (ValueError, TypeError):
                result[target] = None
        else:
            result[target] = None

    # Binary state targets
    for target in BINARY_STATE_TARGETS:
        val = raw.get(target)
        if val is not None:
            result[target] = _to_bool(val)
        else:
            result[target] = None

    # Availability
    avail = raw.get("INT_availability")
    if avail is not None:
        if isinstance(avail, bool):
            result["INT_availability"] = "yes" if avail else "no"
        else:
            result["INT_availability"] = str(avail).lower().strip()
    else:
        result["INT_availability"] = None

    # Metadata
    result["reasoning"] = str(raw.get("reasoning", ""))
    try:
        result["confidence"] = float(raw.get("confidence", 0.0))
    except (ValueError, TypeError):
        result["confidence"] = 0.0

    return result


def parse_json_block(text: str) -> dict[str, Any] | None:
    """Extract and parse a JSON block from LLM output.

    Handles:
    - Raw JSON: {"key": "value"}
    - Code block: ```json\\n{...}\\n```
    - Mixed text with embedded JSON
    - Claude CLI output format: {"result": "...", ...}
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    # Find the first { and try to parse from there
    brace_start = text.find("{")
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def _to_bool(val: Any) -> bool | None:
    """Convert various representations to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        lower = val.lower().strip()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
    return None
