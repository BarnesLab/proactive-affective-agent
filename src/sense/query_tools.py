"""Sensing query engine and Anthropic tool definitions for V5 agentic agent.

Provides structured access to raw sensing data so the V5 agent can
autonomously query behavioral signals via tool calls. Each tool returns
human-readable summaries suitable for LLM reasoning.

Tool inventory:
  - get_daily_summary      : Full day-level sensing aggregates for one date
  - query_sensing          : Targeted sensor query for a date range
  - compare_to_baseline    : Compare a metric to the user's historical baseline
  - get_ema_history        : Retrieve past EMA responses for context
  - find_similar_days      : Find historically similar behavioral days
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.mappings import SENSING_COLUMNS, study_id_to_participant_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic tool schemas (passed to client.messages.create as `tools=`)
# ---------------------------------------------------------------------------

SENSING_TOOLS: list[dict] = [
    {
        "name": "get_daily_summary",
        "description": (
            "Get a comprehensive daily summary of all sensing modalities for a specific date. "
            "Use this first to orient yourself — it returns sleep, GPS mobility, screen usage, "
            "motion/activity, and typing patterns for the entire day. "
            "Only data before the EMA timestamp is included."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (e.g. '2022-09-15')",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": (
                        "If > 0, also return summaries for the N prior days for trend context. "
                        "Default 0 (today only). Max 7."
                    ),
                    "default": 0,
                },
            },
            "required": ["date"],
        },
    },
    {
        "name": "query_sensing",
        "description": (
            "Query a specific sensing modality for a date range. "
            "Use this to zoom in on a specific signal type when you want more detail. "
            "Supported modalities: accelerometer, sleep, android_sleep, gps, screen, "
            "motion, key_input, app_usage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "modality": {
                    "type": "string",
                    "description": (
                        "Sensing modality to query. One of: "
                        "accelerometer, sleep, android_sleep, gps, screen, "
                        "motion, key_input, app_usage"
                    ),
                    "enum": [
                        "accelerometer", "sleep", "android_sleep", "gps",
                        "screen", "motion", "key_input", "app_usage",
                    ],
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date YYYY-MM-DD (inclusive)",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date YYYY-MM-DD (inclusive). Defaults to start_date.",
                },
            },
            "required": ["modality", "start_date"],
        },
    },
    {
        "name": "compare_to_baseline",
        "description": (
            "Compare a sensing metric on a specific date to this user's historical baseline "
            "(mean ± SD computed from their past data). Returns the raw value, baseline stats, "
            "and a z-score indicating how unusual the current value is. "
            "Use this to identify behavioral anomalies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "modality": {
                    "type": "string",
                    "description": "Sensing modality (same options as query_sensing)",
                    "enum": [
                        "accelerometer", "sleep", "android_sleep", "gps",
                        "screen", "motion", "key_input", "app_usage",
                    ],
                },
                "metric": {
                    "type": "string",
                    "description": (
                        "Specific metric column to compare. Examples: "
                        "'val_sleep_duration_min', 'amt_sleep_day_min', "
                        "'amt_travel_day_km', 'amt_home_day_minutes', "
                        "'amt_screenon_day_minutes', 'amt_walking_day_min', "
                        "'n_word_day_allapps', 'prop_word_neg_day_allapps'"
                    ),
                },
                "date": {
                    "type": "string",
                    "description": "Date to compare YYYY-MM-DD",
                },
                "baseline_days": {
                    "type": "integer",
                    "description": (
                        "Number of past days to use for baseline computation. Default 30. "
                        "Only days strictly before `date` are used."
                    ),
                    "default": 30,
                },
            },
            "required": ["modality", "metric", "date"],
        },
    },
    {
        "name": "get_ema_history",
        "description": (
            "Retrieve past EMA survey responses for this user, up to a cutoff date. "
            "Use this to understand their typical emotional patterns, identify recurring states, "
            "and find days with similar behaviors. Returns PANAS scores and state flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "before_date": {
                    "type": "string",
                    "description": "Only return EMA entries strictly before this date YYYY-MM-DD",
                },
                "n_entries": {
                    "type": "integer",
                    "description": "Number of most recent entries to return. Default 10, max 30.",
                    "default": 10,
                },
                "include_emotion_driver": {
                    "type": "boolean",
                    "description": "Include the diary text (emotion_driver) field. Default false.",
                    "default": False,
                },
            },
            "required": ["before_date"],
        },
    },
    {
        "name": "find_similar_days",
        "description": (
            "Find past days with behavioral profiles similar to the current day. "
            "Compares key sensing features (sleep, mobility, screen time, typing) to find "
            "days with matching patterns. Returns those days with their EMA outcomes — "
            "useful for analogical reasoning about the current emotional state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reference_date": {
                    "type": "string",
                    "description": "The target date to find similar days for YYYY-MM-DD",
                },
                "n_similar": {
                    "type": "integer",
                    "description": "Number of similar days to return. Default 5, max 10.",
                    "default": 5,
                },
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Sensing features to use for similarity. Defaults to: "
                        "sleep, travel, screen, walking, typing_volume. "
                        "Options: sleep, travel, screen, walking, typing_volume, typing_sentiment, home_time"
                    ),
                },
            },
            "required": ["reference_date"],
        },
    },
]


# ---------------------------------------------------------------------------
# SensingQueryEngine
# ---------------------------------------------------------------------------

class SensingQueryEngine:
    """Query engine for V5 agentic sensing tools.

    Wraps the raw sensing DataFrames and EMA history to provide
    structured, LLM-friendly query responses.
    """

    def __init__(
        self,
        sensing_dfs: dict[str, pd.DataFrame],
        ema_df: pd.DataFrame,
    ) -> None:
        """Initialize with pre-loaded sensing and EMA data.

        Args:
            sensing_dfs: {sensor_name: DataFrame} from DataLoader.load_all_sensing().
            ema_df: Full EMA DataFrame from DataLoader.load_all_ema().
        """
        self.sensing_dfs = sensing_dfs
        self.ema_df = ema_df

    # ------------------------------------------------------------------
    # Public tool methods (called by AgenticSensingAgent._execute_tool)
    # ------------------------------------------------------------------

    def get_daily_summary(
        self,
        study_id: int,
        date: str,
        lookback_days: int = 0,
        cutoff_timestamp: str | None = None,
    ) -> str:
        """Return a human-readable daily sensing summary.

        Args:
            study_id: Study participant ID.
            date: Date string YYYY-MM-DD.
            lookback_days: Also include N prior days (trend context).
            cutoff_timestamp: EMA timestamp — data after this is excluded.

        Returns:
            Formatted text summary.
        """
        pid = study_id_to_participant_id(study_id)
        target = _parse_date(date)
        if target is None:
            return f"Error: invalid date '{date}'. Use YYYY-MM-DD format."

        lines = []

        # Generate summaries for target + lookback days
        dates_to_summarize = []
        for i in range(lookback_days, -1, -1):
            d = target - timedelta(days=i)
            dates_to_summarize.append(d)

        for d in dates_to_summarize:
            label = "TODAY" if d == target else f"{(target - d).days}d ago ({d})"
            section = [f"\n--- Daily Summary: {d} ({label}) ---"]
            has_any = False

            # Sleep signals
            sleep_parts = self._get_sleep_for_date(pid, d)
            if sleep_parts:
                section.append("Sleep: " + " | ".join(sleep_parts))
                has_any = True

            # GPS / mobility
            gps_parts = self._get_gps_for_date(pid, d)
            if gps_parts:
                section.append("Mobility/GPS: " + " | ".join(gps_parts))
                has_any = True

            # Screen
            screen_parts = self._get_screen_for_date(pid, d)
            if screen_parts:
                section.append("Screen: " + " | ".join(screen_parts))
                has_any = True

            # Motion/activity
            motion_parts = self._get_motion_for_date(pid, d)
            if motion_parts:
                section.append("Motion: " + " | ".join(motion_parts))
                has_any = True

            # Key input / typing
            typing_parts = self._get_typing_for_date(pid, d)
            if typing_parts:
                section.append("Typing: " + " | ".join(typing_parts))
                has_any = True

            # App usage
            app_parts = self._get_apps_for_date(pid, d)
            if app_parts:
                section.append("App usage: " + " | ".join(app_parts))
                has_any = True

            if not has_any:
                section.append("No sensing data found for this date.")

            lines.extend(section)

        return "\n".join(lines) if lines else "No sensing data found."

    def query_sensing(
        self,
        study_id: int,
        modality: str,
        start_date: str,
        end_date: str | None = None,
        cutoff_timestamp: str | None = None,
    ) -> str:
        """Query a specific sensing modality for a date range.

        Args:
            study_id: Study participant ID.
            modality: Sensor name (must be in SENSING_COLUMNS).
            start_date: Start date YYYY-MM-DD.
            end_date: End date YYYY-MM-DD (defaults to start_date).
            cutoff_timestamp: EMA timestamp — data after this is excluded.

        Returns:
            Formatted text summary of the queried data.
        """
        if modality not in SENSING_COLUMNS:
            available = list(SENSING_COLUMNS.keys())
            return f"Error: unknown modality '{modality}'. Available: {available}"

        pid = study_id_to_participant_id(study_id)
        start = _parse_date(start_date)
        end = _parse_date(end_date) if end_date else start

        if start is None or end is None:
            return f"Error: invalid date(s). Use YYYY-MM-DD."
        if end < start:
            return f"Error: end_date {end} is before start_date {start}."

        if modality not in self.sensing_dfs:
            return f"No {modality} data loaded."

        df = self.sensing_dfs[modality]
        info = SENSING_COLUMNS[modality]
        id_col = info["id_col"]
        date_col = info["date_col"]
        features = info["features"]

        mask = (df[id_col] == pid) & (df[date_col] >= start) & (df[date_col] <= end)
        rows = df[mask].sort_values(date_col)

        if rows.empty:
            return f"No {modality} data for participant {pid} between {start} and {end}."

        lines = [f"{modality} data for {pid} ({start} to {end}):"]

        if modality == "app_usage":
            # Group by date and show top apps
            for d, grp in rows.groupby(date_col):
                total_sec = grp["amt_foreground_day_sec"].sum()
                top = grp.nlargest(5, "amt_foreground_day_sec")
                top_str = ", ".join(
                    f"{r['id_app']} ({r['amt_foreground_day_sec']:.0f}s)"
                    for _, r in top.iterrows()
                )
                lines.append(f"  {d}: total={total_sec / 60:.0f}min | top: {top_str}")
        else:
            for _, row in rows.iterrows():
                row_parts = [str(row.get(date_col, "?"))]
                for feat in features:
                    val = row.get(feat)
                    if val is not None and not _is_nan(val):
                        row_parts.append(f"{feat}={_fmt(val)}")
                lines.append("  " + " | ".join(row_parts))

        return "\n".join(lines)

    def compare_to_baseline(
        self,
        study_id: int,
        modality: str,
        metric: str,
        date: str,
        baseline_days: int = 30,
        cutoff_timestamp: str | None = None,
    ) -> str:
        """Compare a metric on a given date to the user's historical baseline.

        Args:
            study_id: Study participant ID.
            modality: Sensor name.
            metric: Column name within the modality.
            date: Target date YYYY-MM-DD.
            baseline_days: Days of history to use.
            cutoff_timestamp: EMA timestamp cutoff.

        Returns:
            Formatted text with current value, baseline stats, z-score, and interpretation.
        """
        if modality not in SENSING_COLUMNS:
            return f"Error: unknown modality '{modality}'."

        pid = study_id_to_participant_id(study_id)
        target = _parse_date(date)
        if target is None:
            return f"Error: invalid date '{date}'."

        if modality not in self.sensing_dfs:
            return f"No {modality} data loaded."

        df = self.sensing_dfs[modality]
        info = SENSING_COLUMNS[modality]
        id_col = info["id_col"]
        date_col = info["date_col"]

        if metric not in df.columns:
            available = [c for c in df.columns if c not in (id_col, date_col)]
            return f"Error: metric '{metric}' not in {modality}. Available: {available}"

        # Current value
        today_rows = df[(df[id_col] == pid) & (df[date_col] == target)]
        if today_rows.empty:
            return f"No {modality} data for {pid} on {target}."

        current_val = today_rows.iloc[0].get(metric)
        if _is_nan(current_val):
            return f"Metric '{metric}' is null on {target}."

        current_val = float(current_val)

        # Historical baseline (past N days strictly before target)
        baseline_start = target - timedelta(days=baseline_days)
        hist = df[
            (df[id_col] == pid)
            & (df[date_col] >= baseline_start)
            & (df[date_col] < target)
        ][metric].dropna()

        if len(hist) < 3:
            return (
                f"{modality}.{metric} on {target}: {current_val:.2f}\n"
                f"Insufficient history ({len(hist)} days) to compute baseline."
            )

        hist_vals = hist.astype(float)
        mean = float(hist_vals.mean())
        sd = float(hist_vals.std())
        z = (current_val - mean) / sd if sd > 0 else 0.0

        # Qualitative interpretation
        if abs(z) < 0.5:
            interpretation = "within normal range"
        elif abs(z) < 1.0:
            direction = "above" if z > 0 else "below"
            interpretation = f"slightly {direction} average"
        elif abs(z) < 2.0:
            direction = "higher" if z > 0 else "lower"
            interpretation = f"notably {direction} than usual"
        else:
            direction = "much higher" if z > 0 else "much lower"
            interpretation = f"ANOMALOUS — {direction} than usual (z={z:.1f})"

        percentile = float(np.mean(hist_vals < current_val) * 100)

        return (
            f"{modality}.{metric} on {target}:\n"
            f"  Current: {current_val:.2f}\n"
            f"  Baseline ({len(hist_vals)}d): mean={mean:.2f}, SD={sd:.2f}\n"
            f"  Z-score: {z:+.2f} | Percentile: {percentile:.0f}th\n"
            f"  Assessment: {interpretation}"
        )

    def get_ema_history(
        self,
        study_id: int,
        before_date: str,
        n_entries: int = 10,
        include_emotion_driver: bool = False,
        cutoff_timestamp: str | None = None,
    ) -> str:
        """Retrieve past EMA entries for the user.

        Args:
            study_id: Study participant ID.
            before_date: Only return entries before this date YYYY-MM-DD.
            n_entries: Max number of most-recent entries to return.
            include_emotion_driver: Include diary text.
            cutoff_timestamp: EMA timestamp cutoff.

        Returns:
            Formatted text of past EMA responses.
        """
        n_entries = min(n_entries, 30)
        cutoff = _parse_date(before_date)
        if cutoff is None:
            return f"Error: invalid date '{before_date}'."

        user_ema = self.ema_df[
            (self.ema_df["Study_ID"] == study_id)
            & (pd.to_datetime(self.ema_df["date_local"]).dt.date < cutoff)
        ].sort_values("timestamp_local", ascending=False).head(n_entries)

        if user_ema.empty:
            return f"No past EMA entries found for user {study_id} before {before_date}."

        lines = [f"Past {len(user_ema)} EMA entries for user {study_id} (most recent first):"]

        for _, row in user_ema.iterrows():
            ts = str(row.get("timestamp_local", "?"))[:16]
            panas_pos = _fmt_maybe(row.get("PANAS_Pos"))
            panas_neg = _fmt_maybe(row.get("PANAS_Neg"))
            er = _fmt_maybe(row.get("ER_desire"))
            avail = row.get("INT_availability", "?")

            entry = f"  {ts} | PANAS_Pos={panas_pos} PANAS_Neg={panas_neg} ER_desire={er} avail={avail}"

            # Add notable binary states (only True ones)
            state_cols = [c for c in row.index if c.startswith("Individual_level_") and c.endswith("_State")]
            true_states = [
                c.replace("Individual_level_", "").replace("_State", "")
                for c in state_cols
                if str(row.get(c, "")).lower() in ("true", "1", "yes")
            ]
            if true_states:
                entry += f" | elevated: {', '.join(true_states)}"

            if include_emotion_driver:
                driver = str(row.get("emotion_driver", ""))
                if driver and driver != "nan":
                    entry += f"\n    Diary: \"{driver[:200]}\""

            lines.append(entry)

        return "\n".join(lines)

    def find_similar_days(
        self,
        study_id: int,
        reference_date: str,
        n_similar: int = 5,
        features: list[str] | None = None,
        cutoff_timestamp: str | None = None,
    ) -> str:
        """Find past days with behavioral profiles similar to the reference date.

        Args:
            study_id: Study participant ID.
            reference_date: Target date YYYY-MM-DD.
            n_similar: Number of similar days to return.
            features: Which feature groups to use for similarity.
            cutoff_timestamp: EMA timestamp cutoff.

        Returns:
            Formatted text of similar days and their EMA outcomes.
        """
        n_similar = min(n_similar, 10)
        ref = _parse_date(reference_date)
        if ref is None:
            return f"Error: invalid date '{reference_date}'."

        pid = study_id_to_participant_id(study_id)

        # Build daily feature vectors
        daily_features = self._build_daily_feature_vectors(pid, before_date=ref)

        if ref not in daily_features:
            return (
                f"Insufficient sensing data on {reference_date} to compute similarity. "
                f"Try get_daily_summary first."
            )

        ref_vec = daily_features[ref]
        feature_keys = features or ["sleep", "travel", "screen", "walking", "typing_volume"]

        # Compute distances
        distances = []
        for d, vec in daily_features.items():
            if d >= ref:
                continue
            dist = _feature_distance(ref_vec, vec, feature_keys)
            if dist is not None:
                distances.append((d, dist))

        if not distances:
            return f"No comparable past days found before {reference_date}."

        distances.sort(key=lambda x: x[1])
        top_days = distances[:n_similar]

        # Look up EMA outcomes for those days
        user_ema = self.ema_df[self.ema_df["Study_ID"] == study_id].copy()
        if "date_local" in user_ema.columns:
            user_ema["_date"] = pd.to_datetime(user_ema["date_local"]).dt.date
        else:
            user_ema["_date"] = pd.to_datetime(user_ema["timestamp_local"]).dt.date

        lines = [
            f"Top {len(top_days)} days similar to {reference_date} (behavioral pattern match):",
            f"Similarity based on: {', '.join(feature_keys)}",
            "",
        ]

        for d, dist in top_days:
            lines.append(f"  Date: {d} (similarity distance: {dist:.3f})")

            # Reference-day feature values
            vec = daily_features[d]
            feat_parts = []
            for k in feature_keys:
                v = vec.get(k)
                if v is not None:
                    feat_parts.append(f"{k}={v:.1f}")
            if feat_parts:
                lines.append(f"    Features: {' | '.join(feat_parts)}")

            # EMA outcomes on that day
            day_ema = user_ema[user_ema["_date"] == d]
            if day_ema.empty:
                lines.append("    EMA: no survey data for this day")
            else:
                for _, row in day_ema.iterrows():
                    ts = str(row.get("timestamp_local", "?"))[:16]
                    pos = _fmt_maybe(row.get("PANAS_Pos"))
                    neg = _fmt_maybe(row.get("PANAS_Neg"))
                    er = _fmt_maybe(row.get("ER_desire"))
                    avail = row.get("INT_availability", "?")
                    lines.append(f"    EMA @ {ts}: PANAS_Pos={pos} PANAS_Neg={neg} ER_desire={er} avail={avail}")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sleep_for_date(self, pid: str, d: date) -> list[str]:
        """Extract sleep signals for a participant on a date."""
        parts = []

        if "accelerometer" in self.sensing_dfs:
            df = self.sensing_dfs["accelerometer"]
            info = SENSING_COLUMNS["accelerometer"]
            rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
            if not rows.empty:
                r = rows.iloc[0]
                v = r.get("val_sleep_duration_min")
                if not _is_nan(v):
                    parts.append(f"accel_sleep={float(v):.0f}min")

        if "sleep" in self.sensing_dfs:
            df = self.sensing_dfs["sleep"]
            info = SENSING_COLUMNS["sleep"]
            rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
            if not rows.empty:
                v = rows.iloc[0].get("amt_sleep_day_min")
                if not _is_nan(v):
                    parts.append(f"passive_sleep={float(v):.0f}min")

        if "android_sleep" in self.sensing_dfs:
            df = self.sensing_dfs["android_sleep"]
            info = SENSING_COLUMNS["android_sleep"]
            rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
            if not rows.empty:
                r = rows.iloc[0]
                v = r.get("amt_sleep_min")
                status = r.get("cat_status", "")
                if not _is_nan(v):
                    parts.append(f"android_sleep={float(v):.0f}min({status})")

        return parts

    def _get_gps_for_date(self, pid: str, d: date) -> list[str]:
        """Extract GPS/mobility signals for a participant on a date."""
        parts = []
        if "gps" not in self.sensing_dfs:
            return parts

        df = self.sensing_dfs["gps"]
        info = SENSING_COLUMNS["gps"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
        if rows.empty:
            return parts

        r = rows.iloc[0]
        for col, label in [
            ("amt_travel_day_km", "travel_km"),
            ("amt_travel_day_minutes", "travel_min"),
            ("amt_home_day_minutes", "home_min"),
            ("amt_distancefromhome_day_max_km", "max_dist_km"),
            ("n_travelevent_day", "travel_events"),
        ]:
            v = r.get(col)
            if not _is_nan(v):
                parts.append(f"{label}={_fmt(v)}")

        return parts

    def _get_screen_for_date(self, pid: str, d: date) -> list[str]:
        """Extract screen usage signals for a participant on a date."""
        parts = []
        if "screen" not in self.sensing_dfs:
            return parts

        df = self.sensing_dfs["screen"]
        info = SENSING_COLUMNS["screen"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
        if rows.empty:
            return parts

        r = rows.iloc[0]
        for col, label in [
            ("amt_screenon_day_minutes", "screen_min"),
            ("n_session_screenon_day", "sessions"),
            ("amt_screenon_session_day_mean_minutes", "mean_session_min"),
        ]:
            v = r.get(col)
            if not _is_nan(v):
                parts.append(f"{label}={_fmt(v)}")

        return parts

    def _get_motion_for_date(self, pid: str, d: date) -> list[str]:
        """Extract motion/activity signals for a participant on a date."""
        parts = []
        if "motion" not in self.sensing_dfs:
            return parts

        df = self.sensing_dfs["motion"]
        info = SENSING_COLUMNS["motion"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
        if rows.empty:
            return parts

        r = rows.iloc[0]
        for col, label in [
            ("amt_stationary_day_min", "stationary_min"),
            ("amt_walking_day_min", "walking_min"),
            ("amt_automotive_day_min", "driving_min"),
            ("amt_running_day_min", "running_min"),
        ]:
            v = r.get(col)
            if not _is_nan(v) and float(v) > 0:
                parts.append(f"{label}={_fmt(v)}")

        return parts

    def _get_typing_for_date(self, pid: str, d: date) -> list[str]:
        """Extract key input/typing signals for a participant on a date."""
        parts = []
        if "key_input" not in self.sensing_dfs:
            return parts

        df = self.sensing_dfs["key_input"]
        info = SENSING_COLUMNS["key_input"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
        if rows.empty:
            return parts

        r = rows.iloc[0]
        for col, label in [
            ("n_word_day_allapps", "words"),
            ("prop_word_pos_day_allapps", "pos_ratio"),
            ("prop_word_neg_day_allapps", "neg_ratio"),
        ]:
            v = r.get(col)
            if not _is_nan(v):
                if "ratio" in label:
                    parts.append(f"{label}={float(v):.1%}")
                else:
                    parts.append(f"{label}={_fmt(v)}")

        return parts

    def _get_apps_for_date(self, pid: str, d: date) -> list[str]:
        """Extract app usage signals for a participant on a date."""
        parts = []
        if "app_usage" not in self.sensing_dfs:
            return parts

        df = self.sensing_dfs["app_usage"]
        info = SENSING_COLUMNS["app_usage"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
        if rows.empty:
            return parts

        total_sec = rows["amt_foreground_day_sec"].sum()
        parts.append(f"total={total_sec / 60:.0f}min")

        top = rows.nlargest(3, "amt_foreground_day_sec")
        top_str = ", ".join(
            f"{r['id_app']}({r['amt_foreground_day_sec'] / 60:.0f}m)"
            for _, r in top.iterrows()
            if not _is_nan(r.get("amt_foreground_day_sec"))
        )
        if top_str:
            parts.append(f"top: {top_str}")

        return parts

    def _build_daily_feature_vectors(
        self, pid: str, before_date: date
    ) -> dict[date, dict[str, float]]:
        """Build normalized feature vectors for each day in the sensing data.

        Used by find_similar_days for cosine/Euclidean similarity.

        Returns:
            {date: {feature_name: float_value}}
        """
        all_dates: set[date] = set()
        # Collect all dates from all modalities
        for modality, df in self.sensing_dfs.items():
            info = SENSING_COLUMNS[modality]
            id_col = info["id_col"]
            date_col = info["date_col"]
            user_rows = df[df[id_col] == pid]
            for d in user_rows[date_col].dropna().unique():
                if isinstance(d, date) and d <= before_date:
                    all_dates.add(d)

        vectors: dict[date, dict[str, float]] = {}

        for d in sorted(all_dates):
            vec: dict[str, float] = {}

            # sleep: accelerometer or passive
            sleep_parts = self._get_sleep_for_date(pid, d)
            for part in sleep_parts:
                if "accel_sleep=" in part:
                    vec["sleep"] = float(part.split("=")[1].replace("min", ""))
                    break
                elif "passive_sleep=" in part:
                    vec["sleep"] = float(part.split("=")[1].replace("min", ""))
                    break

            # travel
            gps_parts = self._get_gps_for_date(pid, d)
            for part in gps_parts:
                if "travel_km=" in part:
                    vec["travel"] = float(part.split("=")[1])
                elif "home_min=" in part:
                    vec["home_time"] = float(part.split("=")[1])

            # screen
            screen_parts = self._get_screen_for_date(pid, d)
            for part in screen_parts:
                if "screen_min=" in part:
                    vec["screen"] = float(part.split("=")[1])

            # walking
            motion_parts = self._get_motion_for_date(pid, d)
            for part in motion_parts:
                if "walking_min=" in part:
                    vec["walking"] = float(part.split("=")[1])

            # typing
            typing_parts = self._get_typing_for_date(pid, d)
            for part in typing_parts:
                if "words=" in part:
                    vec["typing_volume"] = float(part.split("=")[1])
                elif "neg_ratio=" in part:
                    raw = part.split("=")[1].replace("%", "")
                    try:
                        vec["typing_sentiment"] = float(raw) / 100.0
                    except ValueError:
                        pass

            if vec:
                vectors[d] = vec

        return vectors


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str | None) -> date | None:
    """Parse a YYYY-MM-DD string to a date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_nan(val: Any) -> bool:
    """Check if a value is None or NaN."""
    if val is None:
        return True
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


def _fmt(val: Any) -> str:
    """Format a numeric value for display."""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return f"{f:.2f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_maybe(val: Any) -> str:
    """Format a nullable numeric value."""
    if val is None or _is_nan(val):
        return "N/A"
    return _fmt(val)


def _feature_distance(
    ref: dict[str, float],
    other: dict[str, float],
    feature_keys: list[str],
) -> float | None:
    """Compute normalized Euclidean distance between two feature vectors.

    Returns None if there are no common features to compare.
    """
    common = [k for k in feature_keys if k in ref and k in other]
    if not common:
        return None

    total = 0.0
    for k in common:
        diff = ref[k] - other[k]
        # Normalize by ref value to make scale-independent, with a floor
        scale = max(abs(ref[k]), 1.0)
        total += (diff / scale) ** 2

    return (total / len(common)) ** 0.5
