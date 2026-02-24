"""Sensing query tools for autonomous agent investigation.

The agentic sensing agent uses these tools like a detective:
- It starts with minimal context (user profile + diary + current timestamp)
- It decides WHAT to investigate by calling these tools
- It builds up evidence autonomously before making a prediction

Two data backends are supported:
  1. Parquet-backed (SensingQueryEngine): reads hourly Parquet files from
     data/processed/hourly/{modality}/{pid}_{modality}_hourly.parquet.
     Primary backend for V5 agentic agent.
  2. CSV-backed (SensingQueryEngineLegacy): reads from pre-loaded daily CSV
     DataFrames. Preserved for backward compatibility with V1–V4 pipelines.

Tool inventory (SENSING_TOOLS list, Anthropic SDK format):
  - query_sensing          : targeted hourly sensor query
  - get_daily_summary      : full day-level overview across all modalities
  - compare_to_baseline    : compare a metric to personal historical baseline
  - get_ema_history        : retrieve past EMA responses for context
  - find_similar_days      : find behaviorally similar past days via cosine similarity
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.mappings import SENSING_COLUMNS, study_id_to_participant_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic SDK tool schemas (passed to client.messages.create as `tools=`)
# ---------------------------------------------------------------------------

SENSING_TOOLS: list[dict] = [
    {
        "name": "query_sensing",
        "description": (
            "Query passive sensing data for a specific modality and time window. "
            "Use this to investigate behavioral signals before making a prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "modality": {
                    "type": "string",
                    "enum": [
                        "accelerometer", "gps", "motion",
                        "screen", "keyboard", "music", "light",
                    ],
                    "description": "Which sensor modality to query",
                },
                "hours_before_ema": {
                    "type": "integer",
                    "description": (
                        "How many hours before the EMA timestamp to look back (1-48)"
                    ),
                    "minimum": 1,
                    "maximum": 48,
                },
                "hours_duration": {
                    "type": "integer",
                    "description": "Duration of window in hours (default 1, max 24)",
                    "default": 1,
                    "maximum": 24,
                },
                "granularity": {
                    "type": "string",
                    "enum": ["hourly", "daily"],
                    "default": "hourly",
                },
            },
            "required": ["modality", "hours_before_ema"],
        },
    },
    {
        "name": "get_daily_summary",
        "description": (
            "Get a natural language summary of a full day's behavioral patterns "
            "across all sensors. Good starting point for an overview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format",
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
        "name": "compare_to_baseline",
        "description": (
            "Compare a current sensor reading to this person's personal historical "
            "baseline. Use this to detect anomalies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "modality": {"type": "string"},
                "feature": {
                    "type": "string",
                    "description": (
                        "Feature name (e.g., screen_on_min, gps_distance_km, "
                        "motion_walking_min)"
                    ),
                },
                "current_value": {"type": "number"},
            },
            "required": ["modality", "feature", "current_value"],
        },
    },
    {
        "name": "get_ema_history",
        "description": (
            "Retrieve this person's past EMA responses and diary entries. "
            "Use this to understand their emotional patterns and personal baselines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_days": {
                    "type": "integer",
                    "description": "Number of past days to retrieve (default 14)",
                    "default": 14,
                },
                "include_emotion_driver": {
                    "type": "boolean",
                    "description": "Include the diary text (emotion_driver). Default false.",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "find_similar_days",
        "description": (
            "Find past days with similar behavioral patterns and show what this "
            "person's mood was on those days. Useful for analog-based prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of similar days to return (default 5)",
                    "default": 5,
                }
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    f = _safe_float(val)
    return None if f is None else int(f)


def _is_nan(val: Any) -> bool:
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def _fmt(val: Any) -> str:
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else f"{f:.2f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_maybe(val: Any) -> str:
    if val is None or _is_nan(val):
        return "N/A"
    return _fmt(val)


def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _normalize_ts(ts: str | datetime) -> datetime:
    """Parse timestamp string or return datetime; strips timezone."""
    if isinstance(ts, str):
        result = pd.to_datetime(ts)
    else:
        result = pd.Timestamp(ts)
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.to_pydatetime()


def _normalize_hour_start(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a tz-naive datetime column named 'hour_start'."""
    if "hour_start" not in df.columns:
        for alt in ("timestamp", "ts", "datetime", "hour"):
            if alt in df.columns:
                df = df.rename(columns={alt: "hour_start"})
                break
    if "hour_start" in df.columns:
        df["hour_start"] = pd.to_datetime(df["hour_start"], errors="coerce")
        if hasattr(df["hour_start"].dtype, "tz") and df["hour_start"].dt.tz is not None:
            df["hour_start"] = df["hour_start"].dt.tz_localize(None)
    return df


def _time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 24:
        return "evening"
    return "night"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _feature_distance(
    ref: dict[str, float],
    other: dict[str, float],
    feature_keys: list[str],
) -> float | None:
    """Normalized Euclidean distance over shared feature keys."""
    common = [k for k in feature_keys if k in ref and k in other]
    if not common:
        return None
    total = sum(((ref[k] - other[k]) / max(abs(ref[k]), 1.0)) ** 2 for k in common)
    return (total / len(common)) ** 0.5


# ---------------------------------------------------------------------------
# Hourly row formatters  (used by Parquet-backed engine)
# ---------------------------------------------------------------------------

def _format_motion_row(row: pd.Series) -> str:
    activities: list[tuple[float, str]] = []
    seen: set[str] = set()
    for col, label in [
        ("motion_walking_min", "walking"), ("walking_min", "walking"),
        ("motion_stationary_min", "stationary"), ("stationary_min", "stationary"),
        ("motion_automotive_min", "automotive"), ("automotive_min", "automotive"),
        ("motion_running_min", "running"), ("running_min", "running"),
    ]:
        if label in seen:
            continue
        if col in row.index and pd.notna(row[col]) and float(row[col]) > 0:
            activities.append((float(row[col]), label))
            seen.add(label)
    activities.sort(reverse=True)
    parts = [f"{lbl} {mins:.0f}min" for mins, lbl in activities]
    return ", ".join(parts) if parts else "no activity data"


def _format_gps_row(row: pd.Series) -> str:
    parts: list[str] = []
    for col, label in [("gps_distance_km", "traveled"), ("distance_km", "traveled")]:
        if col in row.index and pd.notna(row[col]) and float(row[col]) > 0:
            parts.append(f"traveled {float(row[col]):.2f} km")
            break
    for col in ("gps_at_home_min", "at_home_min"):
        if col in row.index and pd.notna(row[col]):
            parts.append(f"home {float(row[col]):.0f}min")
            break
    for col in ("gps_n_captures", "n_captures"):
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{float(row[col]):.0f} GPS fixes")
            break
    return ", ".join(parts) if parts else "GPS data present"


def _format_screen_row(row: pd.Series) -> str:
    parts: list[str] = []
    if "screen_on_min" in row.index and pd.notna(row["screen_on_min"]):
        parts.append(f"screen on {float(row['screen_on_min']):.0f}min")
    for col in ("screen_n_sessions", "n_sessions"):
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{float(row[col]):.0f} sessions")
            break
    return ", ".join(parts) if parts else "screen data present"


def _format_keyboard_row(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("key_chars_typed", "chars_typed", "n_char_day_allapps"):
        if col in row.index and pd.notna(row[col]) and float(row[col]) > 0:
            parts.append(f"{float(row[col]):.0f} chars typed")
            break
    for col in ("key_prop_neg", "prop_word_neg_day_allapps"):
        if col in row.index and pd.notna(row[col]):
            v = float(row[col])
            if v > 0:
                parts.append(f"{v:.0%} negative sentiment")
            break
    for col in ("key_prop_pos", "prop_word_pos_day_allapps"):
        if col in row.index and pd.notna(row[col]):
            v = float(row[col])
            if v > 0:
                parts.append(f"{v:.0%} positive sentiment")
            break
    return ", ".join(parts) if parts else "no typing data"


def _format_accelerometer_row(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("accel_activity_counts", "activity_counts"):
        if col in row.index and pd.notna(row[col]):
            parts.append(f"activity counts {float(row[col]):.1f}")
            break
    for col in ("accel_coverage_pct", "coverage_pct"):
        if col in row.index and pd.notna(row[col]):
            parts.append(f"coverage {float(row[col]):.0f}%")
            break
    return ", ".join(parts) if parts else "actigraphy data present"


def _format_music_row(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("mus_is_listening", "is_listening"):
        if col in row.index and pd.notna(row[col]):
            parts.append("listening to music" if row[col] else "not listening")
            break
    for col in ("mus_n_tracks", "n_tracks"):
        if col in row.index and pd.notna(row[col]) and float(row[col]) > 0:
            parts.append(f"{float(row[col]):.0f} tracks")
            break
    return ", ".join(parts) if parts else "no music data"


def _format_light_row(row: pd.Series) -> str:
    for col in ("light_mean_lux", "mean_lux", "lux"):
        if col in row.index and pd.notna(row[col]):
            return f"ambient light {float(row[col]):.0f} lux"
    return "no light data"


_HOURLY_FORMATTERS = {
    "accelerometer": _format_accelerometer_row,
    "gps": _format_gps_row,
    "motion": _format_motion_row,
    "screen": _format_screen_row,
    "keyboard": _format_keyboard_row,
    "music": _format_music_row,
    "light": _format_light_row,
}


# ---------------------------------------------------------------------------
# SensingQueryEngine — Parquet-backed, primary engine for V5 agentic agent
# ---------------------------------------------------------------------------

class SensingQueryEngine:
    """Query engine backed by hourly Parquet files.

    Data path convention:
        {processed_dir}/{modality}/{pid}_{modality}_hourly.parquet

    Each tool method returns a plain-text string suitable for LLM consumption.

    Args:
        processed_dir: Path to data/processed/hourly/.
        ema_df: Full EMA DataFrame (all users, all entries). Required for
                baseline computation, EMA history, and similar-day search.
    """

    MODALITIES = [
        "accelerometer", "gps", "motion",
        "screen", "keyboard", "music", "light",
    ]

    def __init__(
        self,
        processed_dir: str | Path,
        ema_df: pd.DataFrame,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self._parquet_cache: dict[str, pd.DataFrame] = {}
        self._baseline_cache: dict[str, dict] = {}

        # Optional supporting Parquet files
        self._platform_df = self._try_load_parquet(
            self.processed_dir / "participant_platform.parquet"
        )
        self._home_df = self._try_load_parquet(
            self.processed_dir / "home_locations.parquet"
        )

        self._ema_df = ema_df.copy() if ema_df is not None and not ema_df.empty else pd.DataFrame()
        if not self._ema_df.empty and "timestamp_local" in self._ema_df.columns:
            self._ema_df["timestamp_local"] = pd.to_datetime(
                self._ema_df["timestamp_local"], errors="coerce"
            )
            if self._ema_df["timestamp_local"].dt.tz is not None:
                self._ema_df["timestamp_local"] = (
                    self._ema_df["timestamp_local"].dt.tz_localize(None)
                )

    # ------------------------------------------------------------------
    # Internal data loading
    # ------------------------------------------------------------------

    def _try_load_parquet(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not load %s: %s", path, exc)
            return pd.DataFrame()

    def _parquet_path(self, study_id: int, modality: str) -> Path:
        pid = str(study_id).zfill(3)
        return self.processed_dir / modality / f"{pid}_{modality}_hourly.parquet"

    def _load_modality_df(self, study_id: int, modality: str) -> pd.DataFrame:
        cache_key = f"{study_id}_{modality}"
        if cache_key in self._parquet_cache:
            return self._parquet_cache[cache_key]

        path = self._parquet_path(study_id, modality)
        if not path.exists():
            self._parquet_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
            df = _normalize_hour_start(df)
            self._parquet_cache[cache_key] = df
            return df
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            self._parquet_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

    def _get_window(
        self,
        study_id: int,
        modality: str,
        window_start: datetime,
        window_end: datetime,
    ) -> pd.DataFrame:
        df = self._load_modality_df(study_id, modality)
        if df.empty or "hour_start" not in df.columns:
            return pd.DataFrame()
        mask = (df["hour_start"] >= window_start) & (df["hour_start"] < window_end)
        return df[mask].sort_values("hour_start")

    def _ema_for_user(self, study_id: int) -> pd.DataFrame:
        if self._ema_df.empty:
            return pd.DataFrame()
        return self._ema_df[self._ema_df["Study_ID"] == study_id].copy()

    def _col_sum(self, df: pd.DataFrame, cols: list[str]) -> float | None:
        for col in cols:
            if col in df.columns:
                val = df[col].sum(skipna=True)
                return None if pd.isna(val) else float(val)
        return None

    def _col_mean(self, df: pd.DataFrame, cols: list[str]) -> float | None:
        for col in cols:
            if col in df.columns:
                val = df[col].mean(skipna=True)
                return None if pd.isna(val) else float(val)
        return None

    # ------------------------------------------------------------------
    # Public tool methods
    # ------------------------------------------------------------------

    def query_sensing(
        self,
        study_id: int,
        modality: str,
        hours_before_ema: int,
        ema_timestamp: str | datetime,
        hours_duration: int = 1,
        granularity: str = "hourly",
    ) -> str:
        """Query a specific modality for a time window before an EMA timestamp.

        Args:
            study_id: Participant's Study_ID.
            modality: Sensor modality name.
            hours_before_ema: End of look-back window in hours before EMA.
            ema_timestamp: EMA trigger time.
            hours_duration: Width of window in hours (default 1).
            granularity: "hourly" or "daily".

        Returns:
            Natural language string describing the sensor data window.
        """
        ema_dt = _normalize_ts(ema_timestamp)
        window_end = ema_dt - timedelta(hours=hours_before_ema - hours_duration)
        window_start = ema_dt - timedelta(hours=hours_before_ema)

        header = f"[{modality.title()}: {hours_before_ema}h before EMA]"
        df = self._get_window(study_id, modality, window_start, window_end)

        if df.empty:
            start_str = window_start.strftime("%H:%M")
            end_str = window_end.strftime("%H:%M")
            return (
                f"{header}\n"
                f"No {modality} data for this window "
                f"(device appears to have been off during {start_str}-{end_str})"
            )

        formatter = _HOURLY_FORMATTERS.get(modality)

        if granularity == "daily":
            agg = self._aggregate_rows(df, modality)
            body = formatter(agg) if formatter else "data present"
            date_str = window_start.strftime("%Y-%m-%d")
            return f"{header}\n{date_str}: {body}"

        # Hourly granularity — one line per hour
        lines = [header]
        n_missing = 0
        total_hours = max(hours_duration, 1)

        current = window_start
        while current < window_end:
            hour_end = current + timedelta(hours=1)
            match = df[df["hour_start"] == current]
            start_str = current.strftime("%H:%M")
            end_str = hour_end.strftime("%H:%M")

            if match.empty:
                lines.append(f"{start_str}-{end_str}: no data")
                n_missing += 1
            else:
                detail = formatter(match.iloc[0]) if formatter else "data present"
                lines.append(f"{start_str}-{end_str}: {detail}")

            current = hour_end

        covered = total_hours - n_missing
        if n_missing == 0:
            lines.append(f"Note: Good coverage ({covered}/{total_hours} hours tracked)")
        elif n_missing < total_hours:
            pct = covered / total_hours * 100
            lines.append(f"Note: Partial coverage ({pct:.0f}% of hours tracked)")
        else:
            lines.append(
                f"Note: Device appears to have been off for this window "
                f"({window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')})"
            )

        return "\n".join(lines)

    def _aggregate_rows(self, df: pd.DataFrame, modality: str) -> pd.Series:
        """Aggregate multiple hourly rows into one representative Series."""
        sum_cols = {
            "accelerometer": ["accel_activity_counts", "activity_counts"],
            "gps": ["gps_distance_km", "distance_km", "gps_at_home_min",
                    "at_home_min", "gps_n_captures", "n_captures"],
            "motion": ["motion_stationary_min", "stationary_min",
                       "motion_walking_min", "walking_min",
                       "motion_automotive_min", "automotive_min",
                       "motion_running_min", "running_min"],
            "screen": ["screen_on_min", "screen_n_sessions", "n_sessions", "app_total_min"],
            "keyboard": ["key_chars_typed", "chars_typed", "n_char_day_allapps",
                         "key_words_typed", "words_typed"],
            "music": ["mus_n_tracks", "n_tracks"],
        }
        mean_cols = {
            "accelerometer": ["accel_mean_mag", "accel_mean_magnitude",
                              "accel_coverage_pct", "coverage_pct"],
            "keyboard": ["key_prop_neg", "prop_word_neg_day_allapps",
                         "key_prop_pos", "prop_word_pos_day_allapps"],
            "light": ["light_mean_lux", "mean_lux", "lux"],
        }
        sum_set = set(sum_cols.get(modality, []))
        mean_set = set(mean_cols.get(modality, []))

        result: dict[str, Any] = {}
        for col in df.select_dtypes(include="number").columns:
            if col in sum_set:
                result[col] = df[col].sum(skipna=True)
            else:
                result[col] = df[col].mean(skipna=True)
        for col in df.select_dtypes(include="bool").columns:
            result[col] = df[col].any()

        return pd.Series(result)

    def get_daily_summary(self, study_id: int, date_str: str, lookback_days: int = 0) -> str:
        """Return a natural language summary of a full day's sensing across modalities.

        Args:
            study_id: Participant's Study_ID.
            date_str: Date in "YYYY-MM-DD" format.
            lookback_days: Also include N prior days for trend context (max 7).

        Returns:
            Multi-paragraph natural language string.
        """
        target = _parse_date(date_str)
        if target is None:
            return f"[Daily Summary: {date_str}]\nInvalid date format. Use YYYY-MM-DD."

        lookback_days = min(lookback_days, 7)
        sections: list[str] = []

        for offset in range(lookback_days, -1, -1):
            d = target - timedelta(days=offset)
            label = "TODAY" if d == target else f"{offset}d ago ({d})"
            sections.append(self._summarize_one_day(study_id, d, label))

        return "\n\n".join(sections)

    def _summarize_one_day(self, study_id: int, d: date, label: str) -> str:
        day_start = datetime(d.year, d.month, d.day)
        day_end = day_start + timedelta(days=1)
        header = f"[Daily Summary: {d} ({label})]"
        lines = [header]

        # Sleep (inferred from accelerometer 00:00-08:00)
        sleep_window_end = day_start + timedelta(hours=8)
        accel_df = self._get_window(study_id, "accelerometer", day_start, sleep_window_end)
        lines.append(f"Sleep: {self._infer_sleep(accel_df)}")

        # Time-of-day breakdowns
        periods = [
            ("Morning (6am-12pm)", day_start + timedelta(hours=6), day_start + timedelta(hours=12)),
            ("Afternoon (12pm-6pm)", day_start + timedelta(hours=12), day_start + timedelta(hours=18)),
            ("Evening (6pm-11pm)", day_start + timedelta(hours=18), day_start + timedelta(hours=23)),
        ]
        for period_name, p_start, p_end in periods:
            parts: list[str] = []
            motion_df = self._get_window(study_id, "motion", p_start, p_end)
            if not motion_df.empty:
                s = self._summarize_motion(motion_df)
                if s:
                    parts.append(s)
            gps_df = self._get_window(study_id, "gps", p_start, p_end)
            if not gps_df.empty:
                s = self._summarize_gps(gps_df)
                if s:
                    parts.append(s)
            screen_df = self._get_window(study_id, "screen", p_start, p_end)
            if not screen_df.empty:
                s = self._summarize_screen(screen_df)
                if s:
                    parts.append(s)
            key_df = self._get_window(study_id, "keyboard", p_start, p_end)
            if not key_df.empty:
                s = self._summarize_keyboard(key_df)
                if s:
                    parts.append(s)
            lines.append(f"{period_name}: {'; '.join(parts) if parts else 'no data'}")

        # Overall day summary
        overall_parts: list[str] = []
        all_motion = self._get_window(study_id, "motion", day_start, day_end)
        if not all_motion.empty:
            walk = self._col_sum(all_motion, ["motion_walking_min", "walking_min"])
            stat = self._col_sum(all_motion, ["motion_stationary_min", "stationary_min"])
            if walk is not None:
                label_mob = "active day" if walk > 60 else "low-mobility day"
                overall_parts.append(f"{label_mob} ({walk:.0f}min walking)")
            elif stat is not None and stat > 600:
                overall_parts.append("low-mobility day (mostly stationary)")

        all_screen = self._get_window(study_id, "screen", day_start, day_end)
        if not all_screen.empty:
            total_screen = self._col_sum(all_screen, ["screen_on_min"])
            if total_screen is not None:
                screen_label = "elevated" if total_screen > 300 else "moderate"
                overall_parts.append(f"{screen_label} screen activity ({total_screen:.0f}min)")

        all_key = self._get_window(study_id, "keyboard", day_start, day_end)
        if not all_key.empty:
            neg_prop = self._col_mean(all_key, ["key_prop_neg", "prop_word_neg_day_allapps"])
            if neg_prop is not None and neg_prop > 0.15:
                overall_parts.append(f"elevated negative typing sentiment ({neg_prop:.0%})")

        if overall_parts:
            lines.append(f"Overall: {'; '.join(overall_parts)}")

        return "\n".join(lines)

    def _infer_sleep(self, accel_df: pd.DataFrame) -> str:
        if accel_df.empty:
            return "no accelerometer data to infer sleep"
        for col in ("accel_activity_counts", "activity_counts", "accel_mean_mag", "mean_magnitude"):
            if col in accel_df.columns:
                valid = accel_df[col].dropna()
                if not valid.empty:
                    threshold = valid.quantile(0.25)
                    sleep_hours = (valid <= threshold).sum()
                    return f"~{sleep_hours:.1f}h based on accel (low activity hours)"
        return "sleep duration unknown"

    def _summarize_motion(self, df: pd.DataFrame) -> str:
        parts: list[str] = []
        stat = self._col_sum(df, ["motion_stationary_min", "stationary_min"])
        walk = self._col_sum(df, ["motion_walking_min", "walking_min"])
        auto = self._col_sum(df, ["motion_automotive_min", "automotive_min"])
        total = (stat or 0) + (walk or 0) + (auto or 0)
        if total > 0 and stat is not None:
            parts.append(f"mostly stationary ({stat / total:.0%})")
        if walk is not None and walk > 0:
            parts.append(f"walking {walk:.0f}min")
        if auto is not None and auto > 0:
            parts.append(f"driving {auto:.0f}min")
        return ", ".join(parts)

    def _summarize_gps(self, df: pd.DataFrame) -> str:
        dist = self._col_sum(df, ["gps_distance_km", "distance_km"])
        if dist is not None and dist > 0:
            return f"traveled {dist:.1f}km"
        return ""

    def _summarize_screen(self, df: pd.DataFrame) -> str:
        total = self._col_sum(df, ["screen_on_min"])
        if total is None:
            return ""
        return f"screen on {total:.1f}h" if total >= 60 else f"screen on {total:.0f}min"

    def _summarize_keyboard(self, df: pd.DataFrame) -> str:
        chars = self._col_sum(df, ["key_chars_typed", "chars_typed", "n_char_day_allapps"])
        if chars is None or chars == 0:
            return ""
        neg = self._col_mean(df, ["key_prop_neg", "prop_word_neg_day_allapps"])
        text = f"{chars:.0f} chars typed"
        if neg is not None and neg > 0.15:
            text += f" ({neg:.0%} negative)"
        return text

    def compare_to_baseline(
        self,
        study_id: int,
        modality: str,
        feature: str,
        current_value: float,
        ema_timestamp: str | datetime,
    ) -> str:
        """Compare a current feature value to this participant's personal baseline.

        Baseline is stratified by time-of-day (morning/afternoon/evening/night).

        Args:
            study_id: Participant's Study_ID.
            modality: Sensor modality.
            feature: Column/feature name (e.g. "screen_on_min").
            current_value: The observed value to compare.
            ema_timestamp: The EMA timestamp (determines time-of-day).

        Returns:
            Natural language comparison string.
        """
        ema_dt = _normalize_ts(ema_timestamp)
        tod = _time_of_day(ema_dt.hour)
        header = f"[Baseline Comparison: {feature}]"

        baseline_key = f"{study_id}_{modality}_{feature}_{tod}"
        if baseline_key not in self._baseline_cache:
            self._baseline_cache[baseline_key] = self._compute_baseline(
                study_id, modality, feature, tod
            )
        baseline = self._baseline_cache[baseline_key]

        if not baseline or baseline.get("mean") is None:
            return (
                f"{header}\n"
                f"Current hour: {current_value:.1f}\n"
                f"Baseline: not enough historical data to compute baseline"
            )

        mean = baseline["mean"]
        std = max(baseline.get("std") or 0.0, 1e-6)
        n = baseline["n"]
        week_avg = baseline.get("week_avg")
        z_score = (current_value - mean) / std

        # Qualitative interpretation
        z_abs = abs(z_score)
        if z_abs < 0.5:
            interpretation = "within typical range"
        elif z_abs < 1.0:
            interpretation = "slightly elevated" if z_score > 0 else "slightly below typical"
        elif z_abs < 2.0:
            interpretation = "elevated" if z_score > 0 else "below typical"
        else:
            direction = "very elevated" if z_score > 0 else "very low"
            pct = self._z_to_pct(z_score)
            interpretation = f"{direction} — {pct}"

        lines = [
            header,
            f"Current hour: {current_value:.1f}",
            f"Your typical {tod}: {mean:.1f} (SD={std:.1f}, n={n} past hours)",
            f"Z-score: {z_score:+.2f} ({interpretation})",
        ]
        if week_avg is not None:
            lines.append(f"Past week average: {week_avg:.1f}")

        return "\n".join(lines)

    def _compute_baseline(
        self,
        study_id: int,
        modality: str,
        feature: str,
        tod: str,
    ) -> dict[str, Any]:
        df = self._load_modality_df(study_id, modality)
        if df.empty or "hour_start" not in df.columns:
            return {}

        # Find the column (accept feature as-is or stripped of prefix)
        col = None
        candidates = [feature]
        for prefix in ("key_", "gps_", "motion_", "screen_", "accel_", "mus_", "light_"):
            stripped = feature.replace(prefix, "", 1)
            if stripped != feature:
                candidates.append(stripped)
        for c in candidates:
            if c in df.columns:
                col = c
                break
        if col is None:
            return {}

        tod_ranges = {
            "morning": (5, 12), "afternoon": (12, 18),
            "evening": (18, 24), "night": (0, 5),
        }
        lo, hi = tod_ranges.get(tod, (0, 24))
        tod_mask = df["hour_start"].dt.hour.between(lo, hi - 1)
        tod_series = df[tod_mask][col].dropna()
        if tod_series.empty:
            return {}

        most_recent = df["hour_start"].max()
        week_ago = most_recent - timedelta(days=7)
        week_df = df[df["hour_start"] >= week_ago][col].dropna()
        week_avg = float(week_df.mean()) if not week_df.empty else None

        return {
            "mean": float(tod_series.mean()),
            "std": float(tod_series.std()) if len(tod_series) > 1 else 0.0,
            "n": int(len(tod_series)),
            "week_avg": week_avg,
        }

    @staticmethod
    def _z_to_pct(z: float) -> str:
        if z > 2.5:
            return "top 1% of your hours"
        if z > 2.0:
            return "top 3% of your hours"
        if z > 1.5:
            return "top 7% of your hours"
        if z > 1.0:
            return "top 16% of your hours"
        if z < -2.0:
            return "bottom 3% of your hours"
        if z < -1.5:
            return "bottom 7% of your hours"
        return ""

    def get_ema_history(
        self,
        study_id: int,
        n_days: int,
        before_timestamp: str | datetime,
        include_emotion_driver: bool = False,
    ) -> str:
        """Retrieve formatted EMA history for a participant.

        Args:
            study_id: Participant's Study_ID.
            n_days: Number of past days to include (default 14).
            before_timestamp: Only entries strictly before this timestamp.
            include_emotion_driver: Include diary text if True.

        Returns:
            Natural language EMA history with pattern summary.
        """
        header = f"[EMA History: last {n_days} days]"
        user_ema = self._ema_for_user(study_id)
        if user_ema.empty:
            return f"{header}\nNo EMA history available."

        cutoff = _normalize_ts(before_timestamp)
        user_ema = user_ema[user_ema["timestamp_local"] < cutoff].copy()
        if user_ema.empty:
            return f"{header}\nNo EMA entries before {before_timestamp}."

        earliest = cutoff - timedelta(days=n_days)
        user_ema = user_ema[user_ema["timestamp_local"] >= earliest]
        if user_ema.empty:
            return f"{header}\nNo EMA entries in the last {n_days} days."

        user_ema = user_ema.sort_values("timestamp_local", ascending=False)

        lines = [header]
        pa_by_tod: dict[str, list[float]] = {"morning": [], "afternoon": [], "evening": []}
        pa_weekday: list[float] = []
        pa_weekend: list[float] = []

        for _, row in user_ema.iterrows():
            ts = row["timestamp_local"]
            date_str = ts.strftime("%Y-%m-%d")
            tod = _time_of_day(ts.hour)
            tod_abbr = tod[:3].capitalize()

            pa = _safe_float(row.get("PANAS_Pos"))
            na = _safe_float(row.get("PANAS_Neg"))
            er = _safe_float(row.get("ER_desire"))
            avail = row.get("INT_availability", "?")

            score_parts = []
            if pa is not None:
                score_parts.append(f"PA={pa:.0f}")
            if na is not None:
                score_parts.append(f"NA={na:.0f}")
            if er is not None:
                score_parts.append(f"ER={er:.0f}")
            scores = ", ".join(score_parts) if score_parts else "no scores"

            # Notable elevated states
            state_cols = [c for c in row.index if c.startswith("Individual_level_") and c.endswith("_State")]
            true_states = [
                c.replace("Individual_level_", "").replace("_State", "")
                for c in state_cols
                if str(row.get(c, "")).lower() in ("true", "1", "yes")
            ]
            state_str = f" | elevated: {', '.join(true_states)}" if true_states else ""
            avail_str = f" | avail={avail}" if avail not in (None, "?") else ""

            entry = f"{date_str} {tod_abbr}: {scores}{avail_str}{state_str}"

            if include_emotion_driver:
                driver = str(row.get("emotion_driver", "") or "").strip()
                if driver and driver != "nan":
                    entry += f' | "{driver[:80]}"'

            lines.append(entry)

            if pa is not None:
                if tod in pa_by_tod:
                    pa_by_tod[tod].append(pa)
                dow = ts.weekday()
                (pa_weekday if dow < 5 else pa_weekend).append(pa)

        # Pattern summary
        patterns: list[str] = []
        morn = pa_by_tod["morning"]
        eve = pa_by_tod["evening"]
        if morn and eve:
            diff = float(np.mean(eve)) - float(np.mean(morn))
            if abs(diff) > 1.5:
                direction = "higher" if diff > 0 else "lower"
                patterns.append(
                    f"Evenings tend to be {direction} PA than mornings (avg {abs(diff):.1f} points)"
                )
        if pa_weekday and pa_weekend:
            diff = float(np.mean(pa_weekend)) - float(np.mean(pa_weekday))
            if abs(diff) > 1.5:
                direction = "higher" if diff > 0 else "lower"
                patterns.append(
                    f"Weekend PA averages {abs(diff):.1f} points {direction} than weekdays"
                )
        if patterns:
            lines.append("Pattern: " + "; ".join(patterns))

        return "\n".join(lines)

    def find_similar_days(
        self,
        study_id: int,
        reference_date: str,
        n: int = 5,
    ) -> str:
        """Find past days with similar behavioral fingerprints via cosine similarity.

        Feature vector per day:
            [accel_activity_counts, screen_on_min, gps_distance_km,
             motion_walking_min, key_chars_typed, motion_stationary_min]

        Only days with EMA data are included.

        Args:
            study_id: Participant's Study_ID.
            reference_date: Reference date as "YYYY-MM-DD".
            n: Number of similar days to return.

        Returns:
            Natural language ranking with mood outcomes.
        """
        header = f"[Similar Past Days: n={n} most behaviorally similar]"
        try:
            ref_date = pd.to_datetime(reference_date).date()
        except Exception:
            return f"{header}\nInvalid date format."

        ref_vector, labels = self._build_daily_fingerprint(study_id, ref_date)
        if ref_vector is None:
            return (
                f"{header}\n"
                f"Not enough sensing data on {reference_date} to compute fingerprint."
            )

        user_ema = self._ema_for_user(study_id)
        if user_ema.empty:
            return f"{header}\nNo EMA history available to compare against."

        user_ema["date_local"] = user_ema["timestamp_local"].dt.date
        past_days = [
            d for d in user_ema["date_local"].unique() if d != ref_date and d < ref_date
        ]

        similarities: list[tuple[float, Any, dict[str, Any]]] = []
        for past_date in past_days:
            fp, _ = self._build_daily_fingerprint(study_id, past_date)
            if fp is None:
                continue
            sim = _cosine_similarity(ref_vector, fp)
            day_ema = user_ema[user_ema["date_local"] == past_date].sort_values("timestamp_local")
            ema_info: dict[str, Any] = {}
            if not day_ema.empty:
                last = day_ema.iloc[-1]
                ema_info["pa"] = _safe_float(last.get("PANAS_Pos"))
                ema_info["na"] = _safe_float(last.get("PANAS_Neg"))
                ema_info["diary"] = str(last.get("emotion_driver", "") or "").strip()
            similarities.append((sim, past_date, ema_info))

        if not similarities:
            return (
                f"{header}\n"
                f"No comparable past days found (need EMA entries on other days)."
            )

        similarities.sort(key=lambda x: x[0], reverse=True)
        top = similarities[:n]

        lines = [header]
        for rank, (sim, past_date, ema) in enumerate(top, 1):
            mood_parts = []
            if ema.get("pa") is not None:
                mood_parts.append(f"PA={ema['pa']:.0f}")
            if ema.get("na") is not None:
                mood_parts.append(f"NA={ema['na']:.0f}")
            mood_str = ", ".join(mood_parts) if mood_parts else "no scores"
            diary = ema.get("diary", "")
            diary_str = f' | "{diary[:80]}"' if diary else ""
            lines.append(
                f"{rank}. {past_date} (similarity: {sim:.2f})\n"
                f"   That day's mood: {mood_str}{diary_str}"
            )

        # Pattern note
        low_pa = sum(1 for _, _, e in top if e.get("pa") is not None and e["pa"] < 12)
        if low_pa >= n // 2 + 1:
            lines.append(
                "Common pattern: Days with this behavioral profile tend to correlate "
                "with lower positive affect for this person"
            )
        high_na = sum(1 for _, _, e in top if e.get("na") is not None and e["na"] > 8)
        if high_na >= n // 2 + 1:
            lines.append(
                "Common pattern: Days with this behavioral profile tend to correlate "
                "with elevated negative affect for this person"
            )

        return "\n".join(lines)

    def _build_daily_fingerprint(
        self, study_id: int, target_date: Any
    ) -> tuple[np.ndarray | None, list[str]]:
        """Build normalized daily feature vector for cosine similarity."""
        if not hasattr(target_date, "year"):
            target_date = pd.to_datetime(str(target_date)).date()
        day_start = datetime(target_date.year, target_date.month, target_date.day)
        day_end = day_start + timedelta(days=1)
        labels = [
            "accel_activity_counts", "screen_on_min", "gps_distance_km",
            "motion_walking_min", "key_chars_typed", "motion_stationary_min",
        ]
        accel_df = self._get_window(study_id, "accelerometer", day_start, day_end)
        screen_df = self._get_window(study_id, "screen", day_start, day_end)
        gps_df = self._get_window(study_id, "gps", day_start, day_end)
        motion_df = self._get_window(study_id, "motion", day_start, day_end)
        key_df = self._get_window(study_id, "keyboard", day_start, day_end)

        values = [
            self._col_sum(accel_df, ["accel_activity_counts", "activity_counts"]) or 0.0,
            self._col_sum(screen_df, ["screen_on_min"]) or 0.0,
            self._col_sum(gps_df, ["gps_distance_km", "distance_km"]) or 0.0,
            self._col_sum(motion_df, ["motion_walking_min", "walking_min"]) or 0.0,
            self._col_sum(key_df, ["key_chars_typed", "chars_typed", "n_char_day_allapps"]) or 0.0,
            self._col_sum(motion_df, ["motion_stationary_min", "stationary_min"]) or 0.0,
        ]
        vec = np.array(values, dtype=float)
        if (vec != 0).sum() < 2:
            return None, labels
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec, labels

    def call_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        study_id: int,
        ema_timestamp: str | datetime,
    ) -> str:
        """Dispatch a tool call from the agentic loop.

        Args:
            tool_name: One of the SENSING_TOOLS names.
            tool_input: Input dict from the LLM tool-use block.
            study_id: Current participant's Study_ID (injected by agent runner).
            ema_timestamp: Current EMA timestamp (injected by agent runner).

        Returns:
            String result to feed back as the tool_result block.
        """
        try:
            if tool_name == "query_sensing":
                return self.query_sensing(
                    study_id=study_id,
                    modality=tool_input["modality"],
                    hours_before_ema=tool_input["hours_before_ema"],
                    ema_timestamp=ema_timestamp,
                    hours_duration=tool_input.get("hours_duration", 1),
                    granularity=tool_input.get("granularity", "hourly"),
                )
            if tool_name == "get_daily_summary":
                return self.get_daily_summary(
                    study_id=study_id,
                    date_str=tool_input["date"],
                    lookback_days=tool_input.get("lookback_days", 0),
                )
            if tool_name == "compare_to_baseline":
                return self.compare_to_baseline(
                    study_id=study_id,
                    modality=tool_input["modality"],
                    feature=tool_input["feature"],
                    current_value=float(tool_input["current_value"]),
                    ema_timestamp=ema_timestamp,
                )
            if tool_name == "get_ema_history":
                return self.get_ema_history(
                    study_id=study_id,
                    n_days=tool_input.get("n_days", 14),
                    before_timestamp=ema_timestamp,
                    include_emotion_driver=tool_input.get("include_emotion_driver", False),
                )
            if tool_name == "find_similar_days":
                ema_dt = _normalize_ts(ema_timestamp)
                ref_date = ema_dt.strftime("%Y-%m-%d")
                return self.find_similar_days(
                    study_id=study_id,
                    reference_date=ref_date,
                    n=tool_input.get("n", 5),
                )
            return f"Unknown tool: {tool_name}"

        except Exception as exc:
            logger.error("Tool call '%s' failed: %s", tool_name, exc, exc_info=True)
            return f"Tool '{tool_name}' encountered an error: {exc}"


# ---------------------------------------------------------------------------
# SensingQueryEngineLegacy — CSV/daily-granularity backend (V1–V4 compatible)
# ---------------------------------------------------------------------------

class SensingQueryEngineLegacy:
    """Query engine backed by pre-loaded daily CSV DataFrames.

    Preserved for backward compatibility with V1–V4 pipelines that work at
    daily granularity. Wraps the original SensingQueryEngine logic that reads
    from SENSING_COLUMNS-keyed DataFrames.

    Args:
        sensing_dfs: {sensor_name: DataFrame} from DataLoader.load_all_sensing().
        ema_df: Full EMA DataFrame from DataLoader.load_all_ema().
    """

    def __init__(
        self,
        sensing_dfs: dict[str, pd.DataFrame],
        ema_df: pd.DataFrame,
    ) -> None:
        self.sensing_dfs = sensing_dfs
        self.ema_df = ema_df

    # ------------------------------------------------------------------
    # Public tool methods (matching SENSING_TOOLS signatures)
    # ------------------------------------------------------------------

    def get_daily_summary(
        self,
        study_id: int,
        date: str,
        lookback_days: int = 0,
        cutoff_timestamp: str | None = None,
    ) -> str:
        """Return a human-readable daily sensing summary (daily-granularity CSV backend)."""
        pid = study_id_to_participant_id(study_id)
        target = _parse_date(date)
        if target is None:
            return f"Error: invalid date '{date}'. Use YYYY-MM-DD format."

        lines: list[str] = []
        dates_to_summarize = [
            target - timedelta(days=i) for i in range(lookback_days, -1, -1)
        ]

        for d in dates_to_summarize:
            label = "TODAY" if d == target else f"{(target - d).days}d ago ({d})"
            section = [f"\n--- Daily Summary: {d} ({label}) ---"]
            has_any = False

            sleep_parts = self._get_sleep_for_date(pid, d)
            if sleep_parts:
                section.append("Sleep: " + " | ".join(sleep_parts))
                has_any = True

            gps_parts = self._get_gps_for_date(pid, d)
            if gps_parts:
                section.append("Mobility/GPS: " + " | ".join(gps_parts))
                has_any = True

            screen_parts = self._get_screen_for_date(pid, d)
            if screen_parts:
                section.append("Screen: " + " | ".join(screen_parts))
                has_any = True

            motion_parts = self._get_motion_for_date(pid, d)
            if motion_parts:
                section.append("Motion: " + " | ".join(motion_parts))
                has_any = True

            typing_parts = self._get_typing_for_date(pid, d)
            if typing_parts:
                section.append("Typing: " + " | ".join(typing_parts))
                has_any = True

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
        """Query a specific sensing modality for a date range (daily CSV backend)."""
        if modality not in SENSING_COLUMNS:
            return f"Error: unknown modality '{modality}'. Available: {list(SENSING_COLUMNS.keys())}"

        pid = study_id_to_participant_id(study_id)
        start = _parse_date(start_date)
        end = _parse_date(end_date) if end_date else start

        if start is None or end is None:
            return "Error: invalid date(s). Use YYYY-MM-DD."
        if end < start:
            return f"Error: end_date {end} is before start_date {start}."
        if modality not in self.sensing_dfs:
            return f"No {modality} data loaded."

        df = self.sensing_dfs[modality]
        info = SENSING_COLUMNS[modality]
        id_col, date_col, features = info["id_col"], info["date_col"], info["features"]

        mask = (df[id_col] == pid) & (df[date_col] >= start) & (df[date_col] <= end)
        rows = df[mask].sort_values(date_col)

        if rows.empty:
            return f"No {modality} data for participant {pid} between {start} and {end}."

        lines = [f"{modality} data for {pid} ({start} to {end}):"]
        if modality == "app_usage":
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
        """Compare a metric on a given date to the user's historical baseline."""
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
        id_col, date_col = info["id_col"], info["date_col"]

        if metric not in df.columns:
            available = [c for c in df.columns if c not in (id_col, date_col)]
            return f"Error: metric '{metric}' not in {modality}. Available: {available}"

        today_rows = df[(df[id_col] == pid) & (df[date_col] == target)]
        if today_rows.empty:
            return f"No {modality} data for {pid} on {target}."

        current_val = today_rows.iloc[0].get(metric)
        if _is_nan(current_val):
            return f"Metric '{metric}' is null on {target}."
        current_val = float(current_val)

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

        if abs(z) < 0.5:
            interpretation = "within normal range"
        elif abs(z) < 1.0:
            interpretation = f"slightly {'above' if z > 0 else 'below'} average"
        elif abs(z) < 2.0:
            interpretation = f"notably {'higher' if z > 0 else 'lower'} than usual"
        else:
            interpretation = f"ANOMALOUS — {'much higher' if z > 0 else 'much lower'} than usual (z={z:.1f})"

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
        """Retrieve past EMA entries for the user (daily CSV backend)."""
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
            entry = (
                f"  {ts} | PANAS_Pos={panas_pos} PANAS_Neg={panas_neg} "
                f"ER_desire={er} avail={avail}"
            )
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
                    entry += f'\n    Diary: "{driver[:200]}"'
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
        """Find past days with behavioral profiles similar to reference_date (daily CSV backend)."""
        n_similar = min(n_similar, 10)
        ref = _parse_date(reference_date)
        if ref is None:
            return f"Error: invalid date '{reference_date}'."

        pid = study_id_to_participant_id(study_id)
        daily_features = self._build_daily_feature_vectors(pid, before_date=ref)

        if ref not in daily_features:
            return (
                f"Insufficient sensing data on {reference_date} to compute similarity. "
                f"Try get_daily_summary first."
            )

        ref_vec = daily_features[ref]
        feature_keys = features or ["sleep", "travel", "screen", "walking", "typing_volume"]

        distances: list[tuple[date, float]] = []
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
            vec = daily_features[d]
            feat_parts = [f"{k}={vec[k]:.1f}" for k in feature_keys if k in vec]
            if feat_parts:
                lines.append(f"    Features: {' | '.join(feat_parts)}")
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
                    lines.append(
                        f"    EMA @ {ts}: PANAS_Pos={pos} PANAS_Neg={neg} "
                        f"ER_desire={er} avail={avail}"
                    )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers (CSV/daily granularity)
    # ------------------------------------------------------------------

    def _get_sleep_for_date(self, pid: str, d: date) -> list[str]:
        parts: list[str] = []
        for sensor, col, label in [
            ("accelerometer", "val_sleep_duration_min", "accel_sleep"),
            ("sleep", "amt_sleep_day_min", "passive_sleep"),
        ]:
            if sensor in self.sensing_dfs:
                df = self.sensing_dfs[sensor]
                info = SENSING_COLUMNS[sensor]
                rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
                if not rows.empty:
                    v = rows.iloc[0].get(col)
                    if not _is_nan(v):
                        parts.append(f"{label}={float(v):.0f}min")

        if "android_sleep" in self.sensing_dfs:
            df = self.sensing_dfs["android_sleep"]
            info = SENSING_COLUMNS["android_sleep"]
            rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == d)]
            if not rows.empty:
                r = rows.iloc[0]
                v = r.get("amt_sleep_min")
                if not _is_nan(v):
                    status = r.get("cat_status", "")
                    parts.append(f"android_sleep={float(v):.0f}min({status})")
        return parts

    def _get_gps_for_date(self, pid: str, d: date) -> list[str]:
        parts: list[str] = []
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
        parts: list[str] = []
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
        parts: list[str] = []
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
        parts: list[str] = []
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
                parts.append(
                    f"{label}={float(v):.1%}" if "ratio" in label else f"{label}={_fmt(v)}"
                )
        return parts

    def _get_apps_for_date(self, pid: str, d: date) -> list[str]:
        parts: list[str] = []
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
        all_dates: set[date] = set()
        for modality, df in self.sensing_dfs.items():
            info = SENSING_COLUMNS[modality]
            id_col, date_col = info["id_col"], info["date_col"]
            user_rows = df[df[id_col] == pid]
            for d in user_rows[date_col].dropna().unique():
                if isinstance(d, date) and d <= before_date:
                    all_dates.add(d)

        vectors: dict[date, dict[str, float]] = {}
        for d in sorted(all_dates):
            vec: dict[str, float] = {}

            for part in self._get_sleep_for_date(pid, d):
                if "accel_sleep=" in part or "passive_sleep=" in part:
                    try:
                        vec["sleep"] = float(part.split("=")[1].replace("min", ""))
                    except ValueError:
                        pass
                    break

            for part in self._get_gps_for_date(pid, d):
                if "travel_km=" in part:
                    try:
                        vec["travel"] = float(part.split("=")[1])
                    except ValueError:
                        pass
                elif "home_min=" in part:
                    try:
                        vec["home_time"] = float(part.split("=")[1])
                    except ValueError:
                        pass

            for part in self._get_screen_for_date(pid, d):
                if "screen_min=" in part:
                    try:
                        vec["screen"] = float(part.split("=")[1])
                    except ValueError:
                        pass

            for part in self._get_motion_for_date(pid, d):
                if "walking_min=" in part:
                    try:
                        vec["walking"] = float(part.split("=")[1])
                    except ValueError:
                        pass

            for part in self._get_typing_for_date(pid, d):
                if "words=" in part:
                    try:
                        vec["typing_volume"] = float(part.split("=")[1])
                    except ValueError:
                        pass
                elif "neg_ratio=" in part:
                    raw = part.split("=")[1].replace("%", "")
                    try:
                        vec["typing_sentiment"] = float(raw) / 100.0
                    except ValueError:
                        pass

            if vec:
                vectors[d] = vec

        return vectors
