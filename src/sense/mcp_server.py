#!/usr/bin/env python3
"""MCP server exposing SensingQueryEngine tools for Claude Code subprocess calls.

Designed to be launched by claude --print --mcp-config ... as a stdio MCP server.
The study_id and ema_timestamp are passed as CLI arguments so the server
enforces temporal cutoff automatically — no risk of the agent seeing future data.

Usage (internal, via run_agentic_pilot_cc.py):
    python3.13 src/sense/mcp_server.py \\
        --study-id 71 \\
        --ema-timestamp "2023-05-15 14:30:00" \\
        --data-dir data/processed \\
        --peer-db-path /tmp/peer_db.parquet

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
    p.add_argument("--peer-db-path", type=str, default="", help="Path to peer database parquet (training EMA + sensing)")
    return p.parse_args()


_args = _parse_args()

_study_id: int = _args.study_id
_ema_timestamp: str = _args.ema_timestamp
_ema_date: str = _args.ema_date or _ema_timestamp[:10]
_data_dir: Path = Path(_args.data_dir) if _args.data_dir else PROJECT_ROOT / "data" / "processed" / "hourly"
_peer_db_path: str = _args.peer_db_path

# ---------------------------------------------------------------------------
# Lazy-init SensingQueryEngine (imports pandas/numpy — avoid at module level)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.sense.query_tools import SensingQueryEngine  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402

# Load EMA data needed by SensingQueryEngine for baseline/history queries
_loader = DataLoader()
try:
    _ema_df = _loader.load_all_ema()
except Exception:
    _ema_df = pd.DataFrame()

_engine = SensingQueryEngine(processed_dir=_data_dir, ema_df=_ema_df)

# ---------------------------------------------------------------------------
# Peer database for cross-user search (loaded once at startup)
# ---------------------------------------------------------------------------

_peer_db: pd.DataFrame | None = None
_peer_tfidf_matrix = None
_peer_tfidf_vectorizer = None
_peer_sensing_matrix = None
_peer_sensing_features: list[str] = []

# All sensing features used for fingerprint-based peer search
_SENSING_FEATURE_COLS = [
    "screen_total_min", "screen_n_sessions",
    "motion_stationary_min", "motion_walking_min", "motion_automotive_min",
    "motion_tracked_hours",
    "app_total_min", "app_social_min", "app_comm_min", "app_entertainment_min",
    "keyboard_n_sessions", "keyboard_total_chars",
    "light_mean_lux_raw",
]

# Target columns to include in peer search results
_OUTCOME_COLS = [
    "PANAS_Pos", "PANAS_Neg", "ER_desire", "INT_availability",
    "Individual_level_PA_State", "Individual_level_NA_State",
    "Individual_level_ER_desire_State",
]


def _load_peer_db() -> None:
    """Load peer database and build search indexes."""
    global _peer_db, _peer_tfidf_matrix, _peer_tfidf_vectorizer
    global _peer_sensing_matrix, _peer_sensing_features

    if not _peer_db_path or not Path(_peer_db_path).exists():
        return

    _peer_db = pd.read_parquet(_peer_db_path)

    # Exclude current participant's data to prevent leakage
    if "Study_ID" in _peer_db.columns:
        _peer_db = _peer_db[_peer_db["Study_ID"] != _study_id].reset_index(drop=True)

    if _peer_db.empty:
        _peer_db = None
        return

    # Build TF-IDF index on diary text (for text-based search)
    if "emotion_driver" in _peer_db.columns:
        texts = _peer_db["emotion_driver"].fillna("").tolist()
        has_text = any(t.strip() for t in texts)
        if has_text:
            from sklearn.feature_extraction.text import TfidfVectorizer
            _peer_tfidf_vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english",
                ngram_range=(1, 2), min_df=1,
            )
            try:
                _peer_tfidf_matrix = _peer_tfidf_vectorizer.fit_transform(texts)
            except ValueError:
                _peer_tfidf_vectorizer = None
                _peer_tfidf_matrix = None

    # Build sensing fingerprint matrix (for sensing-based search)
    available_cols = [c for c in _SENSING_FEATURE_COLS if c in _peer_db.columns]
    if available_cols:
        _peer_sensing_features = available_cols
        sensing_data = _peer_db[available_cols].fillna(0).values.astype(np.float64)
        # Normalize per-feature (z-score) for cosine similarity
        means = sensing_data.mean(axis=0)
        stds = sensing_data.std(axis=0)
        stds[stds == 0] = 1.0
        _peer_sensing_matrix = (sensing_data - means) / stds


_load_peer_db()

# ---------------------------------------------------------------------------
# Build MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "sensing",
    instructions=(
        "You have access to passive smartphone sensing data for a cancer survivor. "
        "Use these tools to investigate behavioral patterns before making an emotional "
        "state prediction. Data is automatically restricted to before the EMA timestamp. "
        "You can also search for similar cases from OTHER participants using find_peer_cases."
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
        tool_input={"n_days": lookback_days},
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
        tool_input={"n": top_k},
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


@mcp.tool()
def find_peer_cases(
    search_mode: str = "text",
    query_text: str = "",
    top_k: int = 5,
) -> str:
    """Search OTHER participants' past EMA entries to find similar cases WITH ground truth outcomes.

    This lets you see how other cancer survivors with similar experiences/behaviors actually felt,
    providing calibration anchors for your prediction. Results include their actual PANAS scores,
    ER desire, and binary emotional states.

    Two search modes:
    - "text": Search by diary text similarity (TF-IDF). Use when diary text is available.
    - "sensing": Search by behavioral fingerprint similarity (daily sensing features).
      Compares the current participant's daily patterns against other participants' daily patterns.
      Use when no diary is available or to find behaviorally similar people.

    Args:
        search_mode: "text" (search by diary similarity) or "sensing" (search by behavioral fingerprint)
        query_text: Diary text to search for (required for text mode, ignored for sensing mode)
        top_k: Number of similar cases to return (default 5, max 10)
    """
    if _peer_db is None:
        return "Peer database not available."

    top_k = min(max(top_k, 1), 10)

    if search_mode == "text":
        return _peer_search_text(query_text, top_k)
    elif search_mode == "sensing":
        return _peer_search_sensing(top_k)
    else:
        return f"Unknown search_mode '{search_mode}'. Use 'text' or 'sensing'."


def _peer_search_text(query_text: str, top_k: int) -> str:
    """Search peers by diary text similarity (TF-IDF)."""
    if _peer_tfidf_matrix is None or _peer_tfidf_vectorizer is None:
        return "Text-based peer search not available (no diary data in training set)."

    if not query_text or not query_text.strip():
        return "No query text provided. For text-based search, provide query_text parameter."

    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = _peer_tfidf_vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, _peer_tfidf_matrix).flatten()

    top_indices = np.argsort(sims)[::-1][:top_k]

    lines = [f"Found {min(top_k, len(top_indices))} similar cases from other participants (by diary text):\n"]
    for rank, idx in enumerate(top_indices, 1):
        if sims[idx] <= 0:
            break
        row = _peer_db.iloc[idx]
        lines.append(f"--- Peer Case {rank} (similarity: {sims[idx]:.3f}) ---")
        diary = row.get("emotion_driver", "")
        if diary and str(diary).strip() and str(diary) != "nan":
            lines.append(f"  Diary: {str(diary)[:300]}")
        _append_outcome_lines(row, lines)
        lines.append("")

    return "\n".join(lines) if len(lines) > 1 else "No similar text cases found."


def _peer_search_sensing(top_k: int) -> str:
    """Search peers by behavioral fingerprint similarity (daily sensing features)."""
    if _peer_sensing_matrix is None or not _peer_sensing_features:
        return "Sensing-based peer search not available (no sensing features in peer database)."

    # Build the current participant's sensing vector for the EMA date
    current_vector = _get_current_sensing_vector()
    if current_vector is None:
        return "Cannot build current participant's sensing fingerprint for today. No filtered data available."

    from sklearn.metrics.pairwise import cosine_similarity
    # Normalize current vector using same stats as peer matrix
    available_cols = [c for c in _SENSING_FEATURE_COLS if c in _peer_db.columns]
    peer_raw = _peer_db[available_cols].fillna(0).values.astype(np.float64)
    means = peer_raw.mean(axis=0)
    stds = peer_raw.std(axis=0)
    stds[stds == 0] = 1.0
    norm_current = ((current_vector - means) / stds).reshape(1, -1)

    sims = cosine_similarity(norm_current, _peer_sensing_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]

    lines = [f"Found {min(top_k, len(top_indices))} similar cases from other participants (by behavioral patterns):\n"]
    for rank, idx in enumerate(top_indices, 1):
        if sims[idx] <= 0:
            break
        row = _peer_db.iloc[idx]
        lines.append(f"--- Peer Case {rank} (similarity: {sims[idx]:.3f}) ---")
        # Show sensing features for context
        sensing_parts = []
        for feat in _peer_sensing_features:
            val = row.get(feat)
            if val is not None and pd.notna(val) and val != 0:
                sensing_parts.append(f"{feat}={val:.1f}")
        if sensing_parts:
            lines.append(f"  Behavior: {'; '.join(sensing_parts)}")
        _append_outcome_lines(row, lines)
        lines.append("")

    return "\n".join(lines) if len(lines) > 1 else "No similar sensing cases found."


def _get_current_sensing_vector() -> np.ndarray | None:
    """Build a sensing feature vector for the current participant + EMA date."""
    # Try loading from filtered data
    pid = str(_study_id).zfill(3)
    filtered_dir = PROJECT_ROOT / "data" / "processed" / "filtered"
    parquet_path = filtered_dir / f"{pid}_daily_filtered.parquet"

    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)
    match = df[df["date_local"].astype(str) == str(_ema_date)]
    if match.empty:
        return None

    row = match.iloc[0]
    available_cols = [c for c in _SENSING_FEATURE_COLS if c in _peer_db.columns]
    vector = []
    for col in available_cols:
        val = row.get(col, 0)
        vector.append(float(val) if pd.notna(val) else 0.0)

    return np.array(vector, dtype=np.float64)


def _append_outcome_lines(row: pd.Series, lines: list[str]) -> None:
    """Append ground truth outcome lines to the output."""
    outcome_parts = []
    pa = row.get("PANAS_Pos")
    if pa is not None and pd.notna(pa):
        outcome_parts.append(f"PA={pa:.1f}")
    na = row.get("PANAS_Neg")
    if na is not None and pd.notna(na):
        outcome_parts.append(f"NA={na:.1f}")
    er = row.get("ER_desire")
    if er is not None and pd.notna(er):
        outcome_parts.append(f"ER={er:.1f}")
    avail = row.get("INT_availability")
    if avail is not None and pd.notna(avail):
        outcome_parts.append(f"Avail={avail}")
    if outcome_parts:
        lines.append(f"  Outcomes: {', '.join(outcome_parts)}")

    # Binary states
    state_parts = []
    for col in ["Individual_level_PA_State", "Individual_level_NA_State",
                 "Individual_level_ER_desire_State"]:
        val = row.get(col)
        if val is not None and pd.notna(val):
            label = col.replace("Individual_level_", "").replace("_State", "")
            state_parts.append(f"{label}={'high' if val else 'typical'}")
    if state_parts:
        lines.append(f"  States: {', '.join(state_parts)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
