"""Hourly feature extraction from Parquet-backed processed sensing data.

Loads pre-processed hourly Parquet files and provides aligned feature windows
for EMA entries. Supports both LLM prompt formatting and ML numeric feature
vectors.

Data path convention:
    data/processed/hourly/{modality}/{pid}_{modality}_hourly.parquet

Each Parquet file is expected to have a column 'hour_start' (datetime or str)
that indexes each hourly row for a single participant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from modality name to Parquet subdirectory name
MODALITY_DIRS: dict[str, str] = {
    "accelerometer": "accelerometer",
    "gps": "gps",
    "motion": "motion",
    "screen": "screen",
    "keyboard": "keyboard",
    "music": "music",
    "light": "light",
}

# Time-of-day labels used for baseline stratification
TOD_LABELS = {
    "morning": range(5, 12),    # 05:00–11:59
    "afternoon": range(12, 18), # 12:00–17:59
    "evening": range(18, 24),   # 18:00–23:59
    "night": range(0, 5),       # 00:00–04:59
}


def _hour_of_day_label(hour: int) -> str:
    """Return the time-of-day label for a given hour (0-23)."""
    for label, rng in TOD_LABELS.items():
        if hour in rng:
            return label
    return "night"


@dataclass
class HourlySensingWindow:
    """Sensing features aggregated into a single hourly window."""

    hour_start: str  # ISO datetime string, e.g. "2024-03-15T14:00"
    hour_end: str

    # Accelerometer / actigraphy
    accel_mean_mag: float | None = None
    accel_std_mag: float | None = None
    accel_activity_counts: float | None = None
    accel_coverage_pct: float | None = None

    # GPS / mobility
    gps_n_captures: int | None = None
    gps_distance_km: float | None = None
    gps_at_home_min: float | None = None
    gps_speed_max_kmh: float | None = None
    gps_location_entropy: float | None = None
    gps_dist_from_home_max_km: float | None = None

    # Motion / activity classification
    motion_stationary_min: float | None = None
    motion_walking_min: float | None = None
    motion_automotive_min: float | None = None
    motion_running_min: float | None = None
    motion_active_min: float | None = None
    motion_coverage_pct: float | None = None

    # Screen / App
    screen_on_min: float | None = None
    screen_n_sessions: int | None = None
    app_total_min: float | None = None
    app_social_min: float | None = None
    app_comm_min: float | None = None

    # Keyboard / typing
    key_n_sessions: int | None = None
    key_chars_typed: int | None = None
    key_words_typed: int | None = None
    key_prop_neg: float | None = None
    key_prop_pos: float | None = None

    # Music
    mus_is_listening: bool | None = None
    mus_n_tracks: int | None = None

    # Light
    light_mean_lux: float | None = None

    # Missingness summary
    n_structural_missing: int = 0   # modality not collected for this participant
    n_device_missing: int = 0       # device was off / no data recorded
    n_participant_missing: int = 0  # participant opted out

    def to_text(self) -> str:
        """Format window as a human-readable natural language sentence."""
        parts: list[str] = []

        # Motion
        motion_pieces: list[str] = []
        if self.motion_walking_min is not None and self.motion_walking_min > 0:
            motion_pieces.append(f"walking {self.motion_walking_min:.0f}min")
        if self.motion_stationary_min is not None and self.motion_stationary_min > 0:
            motion_pieces.append(f"stationary {self.motion_stationary_min:.0f}min")
        if self.motion_automotive_min is not None and self.motion_automotive_min > 0:
            motion_pieces.append(f"in vehicle {self.motion_automotive_min:.0f}min")
        if self.motion_running_min is not None and self.motion_running_min > 0:
            motion_pieces.append(f"running {self.motion_running_min:.0f}min")
        if motion_pieces:
            parts.append(", ".join(motion_pieces))

        # GPS
        if self.gps_distance_km is not None and self.gps_distance_km > 0:
            parts.append(f"traveled {self.gps_distance_km:.1f}km")
        if self.gps_at_home_min is not None:
            parts.append(f"home {self.gps_at_home_min:.0f}min")

        # Accelerometer
        if self.accel_activity_counts is not None:
            parts.append(f"activity counts {self.accel_activity_counts:.0f}")

        # Screen
        if self.screen_on_min is not None:
            parts.append(f"screen on {self.screen_on_min:.0f}min")
            if self.screen_n_sessions is not None:
                parts[-1] += f" ({self.screen_n_sessions} sessions)"

        # Keyboard
        if self.key_chars_typed is not None and self.key_chars_typed > 0:
            parts.append(f"{self.key_chars_typed} chars typed")
            if self.key_prop_neg is not None and self.key_prop_neg > 0.15:
                parts[-1] += f" ({self.key_prop_neg:.0%} neg)"

        # Music
        if self.mus_is_listening:
            extra = f" ({self.mus_n_tracks} tracks)" if self.mus_n_tracks else ""
            parts.append(f"listening to music{extra}")

        # Light
        if self.light_mean_lux is not None:
            parts.append(f"ambient light {self.light_mean_lux:.0f} lux")

        start = self.hour_start[11:16] if len(self.hour_start) >= 16 else self.hour_start
        end = self.hour_end[11:16] if len(self.hour_end) >= 16 else self.hour_end
        body = "; ".join(parts) if parts else "no data"
        return f"{start}-{end}: {body}"

    def to_feature_dict(self, prefix: str = "") -> dict[str, float]:
        """Serialize to a flat numeric dict suitable for ML feature matrices.

        Args:
            prefix: Optional prefix (e.g. "h0_") to namespace the keys.

        Returns:
            Dict mapping feature_name → float value (None fields omitted).
        """
        result: dict[str, float] = {}
        for attr, val in self.__dict__.items():
            if attr in ("hour_start", "hour_end"):
                continue
            if val is None:
                continue
            if isinstance(val, bool):
                result[f"{prefix}{attr}"] = float(val)
            elif isinstance(val, (int, float)):
                result[f"{prefix}{attr}"] = float(val)
        return result


@dataclass
class SensingContext:
    """Multi-hour sensing context aligned to an EMA entry."""

    study_id: int
    ema_timestamp: str
    lookback_hours: int = 24
    windows: list[HourlySensingWindow] = field(default_factory=list)

    def to_text(self) -> str:
        """Format as natural language for LLM prompts.

        Returns a multi-line string with one row per hour, plus a coverage note.
        """
        if not self.windows:
            return "No hourly sensing data available."

        lines = [f"Sensing data — {self.lookback_hours}h before EMA:"]
        for w in self.windows:
            lines.append(f"  {w.to_text()}")

        # Coverage note
        total = len(self.windows)
        missing = sum(1 for w in self.windows if w.n_device_missing > 0)
        if missing == 0:
            lines.append(f"Note: Good coverage ({total} hours tracked)")
        elif missing < total:
            pct = (total - missing) / total * 100
            lines.append(f"Note: Partial coverage ({pct:.0f}% of hours tracked)")
        else:
            lines.append("Note: Device appears to have been off for this window")

        return "\n".join(lines)

    def to_feature_vector(self) -> dict[str, float]:
        """Flatten all hourly windows to a numeric feature dict for ML baselines.

        Produces features like h0_screen_on_min, h1_motion_walking_min, etc.
        """
        features: dict[str, float] = {}
        for i, w in enumerate(self.windows):
            features.update(w.to_feature_dict(prefix=f"h{i}_"))
        return features


# ---------------------------------------------------------------------------
# Column name mappings: Parquet column → HourlySensingWindow attribute
# These are the columns we expect inside each modality's Parquet file.
# ---------------------------------------------------------------------------

_ACCEL_MAP: dict[str, str] = {
    "accel_mean_magnitude": "accel_mean_mag",
    "accel_std_magnitude": "accel_std_mag",
    "accel_activity_counts": "accel_activity_counts",
    "accel_coverage_pct": "accel_coverage_pct",
    # Common alternatives
    "mean_magnitude": "accel_mean_mag",
    "std_magnitude": "accel_std_mag",
    "activity_counts": "accel_activity_counts",
    "coverage_pct": "accel_coverage_pct",
}

_GPS_MAP: dict[str, str] = {
    "n_captures": "gps_n_captures",
    "distance_km": "gps_distance_km",
    "at_home_min": "gps_at_home_min",
    "speed_max_kmh": "gps_speed_max_kmh",
    "location_entropy": "gps_location_entropy",
    "dist_from_home_max_km": "gps_dist_from_home_max_km",
    "gps_n_captures": "gps_n_captures",
    "gps_distance_km": "gps_distance_km",
    "gps_at_home_min": "gps_at_home_min",
    "gps_speed_max_kmh": "gps_speed_max_kmh",
    "gps_location_entropy": "gps_location_entropy",
    "gps_dist_from_home_max_km": "gps_dist_from_home_max_km",
}

_MOTION_MAP: dict[str, str] = {
    "stationary_min": "motion_stationary_min",
    "walking_min": "motion_walking_min",
    "automotive_min": "motion_automotive_min",
    "running_min": "motion_running_min",
    "active_min": "motion_active_min",
    "coverage_pct": "motion_coverage_pct",
    "motion_stationary_min": "motion_stationary_min",
    "motion_walking_min": "motion_walking_min",
    "motion_automotive_min": "motion_automotive_min",
    "motion_running_min": "motion_running_min",
    "motion_active_min": "motion_active_min",
    "motion_coverage_pct": "motion_coverage_pct",
}

_SCREEN_MAP: dict[str, str] = {
    "screen_on_min": "screen_on_min",
    "n_sessions": "screen_n_sessions",
    "screen_n_sessions": "screen_n_sessions",
    "app_total_min": "app_total_min",
    "app_social_min": "app_social_min",
    "app_comm_min": "app_comm_min",
}

_KEYBOARD_MAP: dict[str, str] = {
    "n_sessions": "key_n_sessions",
    "chars_typed": "key_chars_typed",
    "words_typed": "key_words_typed",
    "prop_neg": "key_prop_neg",
    "prop_pos": "key_prop_pos",
    "key_n_sessions": "key_n_sessions",
    "key_chars_typed": "key_chars_typed",
    "key_words_typed": "key_words_typed",
    "key_prop_neg": "key_prop_neg",
    "key_prop_pos": "key_prop_pos",
}

_MUSIC_MAP: dict[str, str] = {
    "is_listening": "mus_is_listening",
    "n_tracks": "mus_n_tracks",
    "mus_is_listening": "mus_is_listening",
    "mus_n_tracks": "mus_n_tracks",
}

_LIGHT_MAP: dict[str, str] = {
    "mean_lux": "light_mean_lux",
    "light_mean_lux": "light_mean_lux",
}

_MODALITY_COL_MAPS: dict[str, dict[str, str]] = {
    "accelerometer": _ACCEL_MAP,
    "gps": _GPS_MAP,
    "motion": _MOTION_MAP,
    "screen": _SCREEN_MAP,
    "keyboard": _KEYBOARD_MAP,
    "music": _MUSIC_MAP,
    "light": _LIGHT_MAP,
}


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None for NaN / missing."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    """Convert a value to int, returning None for NaN / missing."""
    f = _safe_float(val)
    return None if f is None else int(f)


def _safe_bool(val: Any) -> bool | None:
    """Convert a value to bool, returning None for NaN / missing."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return bool(val)


def _apply_row_to_window(
    row: pd.Series,
    window: HourlySensingWindow,
    col_map: dict[str, str],
) -> None:
    """Set attributes on window from a Parquet row using a column map."""
    bool_attrs = {"mus_is_listening"}
    int_attrs = {"gps_n_captures", "screen_n_sessions", "key_n_sessions",
                 "key_chars_typed", "key_words_typed", "mus_n_tracks"}

    for parquet_col, attr in col_map.items():
        if parquet_col not in row.index:
            continue
        val = row[parquet_col]
        if attr in bool_attrs:
            setattr(window, attr, _safe_bool(val))
        elif attr in int_attrs:
            setattr(window, attr, _safe_int(val))
        else:
            setattr(window, attr, _safe_float(val))


def _normalize_hour_start(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the 'hour_start' column is a tz-naive datetime column."""
    if "hour_start" not in df.columns:
        # Try common alternatives
        for alt in ("timestamp", "ts", "datetime", "hour"):
            if alt in df.columns:
                df = df.rename(columns={alt: "hour_start"})
                break

    if "hour_start" in df.columns:
        df["hour_start"] = pd.to_datetime(df["hour_start"], utc=False, errors="coerce")
        # Strip timezone if present
        if hasattr(df["hour_start"].dtype, "tz") and df["hour_start"].dt.tz is not None:
            df["hour_start"] = df["hour_start"].dt.tz_localize(None)

    return df


class HourlyFeatureLoader:
    """Load Parquet-backed hourly sensing features and align them to EMA timestamps.

    Data path convention:
        {processed_dir}/{modality}/{pid}_{modality}_hourly.parquet

    Usage:
        loader = HourlyFeatureLoader(processed_dir="data/processed/hourly")
        ctx = loader.get_features_for_ema(study_id=101, timestamp="2024-03-15T14:30:00")
        print(ctx.to_text())
    """

    MODALITIES = list(MODALITY_DIRS.keys())

    def __init__(self, processed_dir: str | Path | None = None) -> None:
        """Initialize the loader.

        Args:
            processed_dir: Path to data/processed/hourly/. Defaults to
                "data/processed/hourly" relative to the working directory.
        """
        self.processed_dir = Path(processed_dir or "data/processed/hourly")

        # In-memory caches: (study_id, modality) → DataFrame
        self._parquet_cache: dict[tuple[int, str], pd.DataFrame] = {}

        # participant_platform.parquet (optional, for platform metadata)
        self._platform_df: pd.DataFrame | None = None
        self._load_platform()

    def _load_platform(self) -> None:
        """Load participant_platform.parquet if it exists."""
        path = self.processed_dir / "participant_platform.parquet"
        if path.exists():
            try:
                self._platform_df = pd.read_parquet(path)
                logger.debug("Loaded participant_platform from %s", path)
            except Exception as exc:
                logger.warning("Could not load participant_platform.parquet: %s", exc)

    def _get_parquet_path(self, study_id: int, modality: str) -> Path:
        """Return expected Parquet path for a participant + modality."""
        pid = str(study_id).zfill(3)
        subdir = MODALITY_DIRS.get(modality, modality)
        return self.processed_dir / subdir / f"{pid}_{modality}_hourly.parquet"

    def _load_modality(self, study_id: int, modality: str) -> pd.DataFrame | None:
        """Load (and cache) a modality Parquet for one participant.

        Returns None if the file does not exist or fails to load.
        """
        cache_key = (study_id, modality)
        if cache_key in self._parquet_cache:
            return self._parquet_cache[cache_key]

        path = self._get_parquet_path(study_id, modality)
        if not path.exists():
            logger.debug("Parquet not found: %s", path)
            self._parquet_cache[cache_key] = pd.DataFrame()
            return None

        try:
            df = pd.read_parquet(path)
            df = _normalize_hour_start(df)
            self._parquet_cache[cache_key] = df
            logger.debug("Loaded %s rows from %s", len(df), path)
            return df
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            self._parquet_cache[cache_key] = pd.DataFrame()
            return None

    def _slice_window(
        self,
        df: pd.DataFrame,
        window_start: datetime,
        window_end: datetime,
    ) -> pd.DataFrame:
        """Slice a Parquet DataFrame to rows whose hour_start falls in [start, end)."""
        if df.empty or "hour_start" not in df.columns:
            return df
        mask = (df["hour_start"] >= window_start) & (df["hour_start"] < window_end)
        return df[mask]

    def _build_window(
        self,
        hour_start: datetime,
        modality_rows: dict[str, pd.Series | None],
    ) -> HourlySensingWindow:
        """Build a HourlySensingWindow for a single hour from per-modality rows."""
        hour_end = hour_start + timedelta(hours=1)
        window = HourlySensingWindow(
            hour_start=hour_start.strftime("%Y-%m-%dT%H:%M"),
            hour_end=hour_end.strftime("%Y-%m-%dT%H:%M"),
        )

        n_missing = 0
        for modality, row in modality_rows.items():
            if row is None:
                n_missing += 1
                window.n_device_missing += 1
                continue
            col_map = _MODALITY_COL_MAPS.get(modality, {})
            _apply_row_to_window(row, window, col_map)

        return window

    def get_features_for_ema(
        self,
        study_id: int,
        timestamp: str | datetime,
        lookback_hours: int = 24,
    ) -> SensingContext:
        """Get hourly sensing features aligned to an EMA timestamp.

        Loads all available modalities from Parquet files and assembles a
        SensingContext with one HourlySensingWindow per hour in the lookback
        window.

        Args:
            study_id: User's Study_ID (int).
            timestamp: EMA timestamp (ISO string or datetime). The lookback
                window is [timestamp - lookback_hours, timestamp).
            lookback_hours: Number of hours of history to retrieve.

        Returns:
            SensingContext dataclass containing one window per hour.
            Empty windows are included to indicate missingness.
        """
        if isinstance(timestamp, str):
            ts = pd.to_datetime(timestamp)
        else:
            ts = pd.Timestamp(timestamp)

        # Strip timezone for consistent comparison
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)

        ema_dt = ts.to_pydatetime()
        window_end = ema_dt.replace(minute=0, second=0, microsecond=0)
        window_start = window_end - timedelta(hours=lookback_hours)

        ctx = SensingContext(
            study_id=study_id,
            ema_timestamp=str(timestamp),
            lookback_hours=lookback_hours,
        )

        # Pre-load all modality DataFrames and slice to the time window
        modality_dfs: dict[str, pd.DataFrame] = {}
        for modality in self.MODALITIES:
            df = self._load_modality(study_id, modality)
            if df is not None and not df.empty:
                sliced = self._slice_window(df, window_start, window_end)
                modality_dfs[modality] = sliced
            else:
                modality_dfs[modality] = pd.DataFrame()

        # Build one window per hour
        current = window_start
        while current < window_end:
            hour_rows: dict[str, pd.Series | None] = {}
            for modality, df in modality_dfs.items():
                if df.empty or "hour_start" not in df.columns:
                    hour_rows[modality] = None
                else:
                    match = df[df["hour_start"] == current]
                    hour_rows[modality] = match.iloc[0] if not match.empty else None

            window_obj = self._build_window(current, hour_rows)
            ctx.windows.append(window_obj)
            current += timedelta(hours=1)

        return ctx

    def format_hourly_sensing_text(
        self,
        study_id: int,
        timestamp: str | datetime,
        lookback_hours: int = 24,
    ) -> str:
        """Get formatted hourly sensing text for LLM prompts.

        Args:
            study_id: User's Study_ID.
            timestamp: EMA timestamp.
            lookback_hours: Hours of history to include.

        Returns:
            Natural language multi-line string.
        """
        ctx = self.get_features_for_ema(study_id, timestamp, lookback_hours)
        return ctx.to_text()

    def get_feature_matrix(
        self,
        study_id: int,
        timestamp: str | datetime,
        lookback_hours: int = 24,
    ) -> dict[str, float]:
        """Get numeric feature dict for ML baselines.

        Args:
            study_id: User's Study_ID.
            timestamp: EMA timestamp.
            lookback_hours: Hours of history to include.

        Returns:
            Flat dict of feature_name → float value.
        """
        ctx = self.get_features_for_ema(study_id, timestamp, lookback_hours)
        return ctx.to_feature_vector()
