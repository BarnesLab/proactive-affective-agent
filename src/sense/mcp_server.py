#!/usr/bin/env python3
"""MCP server exposing SensingQueryEngine tools for Claude Code subprocess calls.

Designed to be launched by claude --print --mcp-config ... as a stdio MCP server.
The study_id and ema_timestamp are passed as CLI arguments so the server
enforces temporal cutoff automatically — no risk of the agent seeing future data.

Usage (internal, via run_agentic_pilot_cc.py):
    python3.13 src/sense/mcp_server.py \\
        --study-id 71 \\
        --ema-timestamp "2023-05-15 14:30:00" \\
        --data-dir data/processed

The server is instantiated fresh per (participant, EMA) call and exits when
the parent claude --print process exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Parse CLI args before building server (args bake participant context in)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensing MCP server for agentic evaluation")
    p.add_argument("--study-id", type=int, required=True, help="Participant Study_ID (int)")
    p.add_argument("--ema-timestamp", type=str, required=True, help="EMA timestamp (YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--ema-date", type=str, default="", help="EMA date (YYYY-MM-DD), inferred if absent")
    p.add_argument("--data-dir", type=str, default="", help="Path to data/processed/ directory")
    return p.parse_args()


_args = _parse_args()

_study_id: int = _args.study_id
_ema_timestamp: str = _args.ema_timestamp
_ema_date: str = _args.ema_date or _ema_timestamp[:10]
_data_dir: Path = Path(_args.data_dir) if _args.data_dir else PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Lazy-init SensingQueryEngine (imports pandas/numpy — avoid at module level)
# ---------------------------------------------------------------------------

from src.sense.query_tools import SensingQueryEngine  # noqa: E402

_engine = SensingQueryEngine(processed_dir=_data_dir)

# ---------------------------------------------------------------------------
# Build MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "sensing",
    instructions=(
        "You have access to passive smartphone sensing data for a cancer survivor. "
        "Use these tools to investigate behavioral patterns before making an emotional "
        "state prediction. Data is automatically restricted to before the EMA timestamp."
    ),
)


@mcp.tool()
def query_sensing(
    modality: str,
    hours_before_ema: int,
    hours_duration: int = 1,
    granularity: str = "hourly",
) -> str:
    """Query passive sensing data for a specific modality and time window.

    Args:
        modality: Sensor type — accelerometer, gps, motion, screen, keyboard, music, light
        hours_before_ema: How many hours before the EMA to look back (1-48)
        hours_duration: Duration of window in hours (default 1, max 24)
        granularity: hourly or daily
    """
    return _engine.call_tool(
        tool_name="query_sensing",
        tool_input={
            "modality": modality,
            "hours_before_ema": hours_before_ema,
            "hours_duration": hours_duration,
            "granularity": granularity,
        },
        study_id=_study_id,
        ema_timestamp=_ema_timestamp,
    )


@mcp.tool()
def get_daily_summary(date: str = "", lookback_days: int = 0) -> str:
    """Get a natural language summary of a full day's behavioral patterns.

    Args:
        date: Date in YYYY-MM-DD format (defaults to EMA date)
        lookback_days: Also return summaries for N prior days (0-7, default 0)
    """
    return _engine.call_tool(
        tool_name="get_daily_summary",
        tool_input={
            "date": date or _ema_date,
            "lookback_days": lookback_days,
        },
        study_id=_study_id,
        ema_timestamp=_ema_timestamp,
    )


@mcp.tool()
def compare_to_baseline(modality: str, feature: str, current_value: float) -> str:
    """Compare a current sensor reading to this person's personal historical baseline.

    Args:
        modality: Sensor modality name
        feature: Feature name (e.g., screen_on_min, gps_distance_km, motion_walking_min)
        current_value: The value to compare against the personal baseline
    """
    return _engine.call_tool(
        tool_name="compare_to_baseline",
        tool_input={
            "modality": modality,
            "feature": feature,
            "current_value": current_value,
        },
        study_id=_study_id,
        ema_timestamp=_ema_timestamp,
    )


@mcp.tool()
def get_receptivity_history(lookback_days: int = 14) -> str:
    """Retrieve this person's past intervention receptivity and mood patterns.

    Args:
        lookback_days: How many days of history to retrieve (default 14, max 30)
    """
    return _engine.call_tool(
        tool_name="get_receptivity_history",
        tool_input={"lookback_days": lookback_days},
        study_id=_study_id,
        ema_timestamp=_ema_timestamp,
    )


@mcp.tool()
def find_similar_days(top_k: int = 3) -> str:
    """Find behaviorally similar past days to reason about likely emotional state.

    Args:
        top_k: Number of similar days to return (default 3, max 10)
    """
    return _engine.call_tool(
        tool_name="find_similar_days",
        tool_input={"top_k": top_k},
        study_id=_study_id,
        ema_timestamp=_ema_timestamp,
    )


@mcp.tool()
def query_raw_events(
    modality: str,
    hours_before_ema: int,
    hours_duration: int = 4,
    max_events: int = 30,
) -> str:
    """Query the raw event stream for fine-grained behavioral detail.

    Use this when hourly aggregates aren't enough. Returns individual events:
    - screen: exact lock/unlock times (infer wake-up time, phone-checking frequency)
    - app: which specific apps were used and for how long
    - motion: exact activity transition timestamps
    - keyboard: what was typed, in which app
    - music: which songs/artists played

    Args:
        modality: screen, app, motion, keyboard, or music
        hours_before_ema: How many hours before the EMA to look back (1-48)
        hours_duration: Duration of window in hours (default 4)
        max_events: Maximum number of events to return (default 30, max 100)
    """
    return _engine.call_tool(
        tool_name="query_raw_events",
        tool_input={
            "modality": modality,
            "hours_before_ema": hours_before_ema,
            "hours_duration": hours_duration,
            "max_events": max_events,
        },
        study_id=_study_id,
        ema_timestamp=_ema_timestamp,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
