"""Feature extraction per sensor type.

Extracts interpretable features from pre-processed hourly sensing DataFrames.
Each extractor method receives a DataFrame that has already been filtered to the
relevant time window; it returns a flat dict of feature_name → value.

Sensors supported:
    accelerometer   → activity level, movement magnitude, coverage
    gps             → mobility distance, location entropy, time at home
    motion          → activity classification (stationary / walking / driving …)
    screen          → on-time minutes, session count, app-category minutes
    keyboard        → characters typed, word count, positive/negative sentiment
    music           → listening flag, track count
    light           → ambient light level
    sleep           → duration inferred from accelerometer low-movement periods
    app_usage       → total foreground time, social / communication app minutes
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    """Return float or None; silently absorbs NaN and conversion errors."""
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


def _safe_sum(series: pd.Series) -> float | None:
    """Sum a Series, returning None if the Series is empty or all-NaN."""
    if series.empty:
        return None
    total = series.sum(skipna=True)
    return None if pd.isna(total) else float(total)


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    mean = series.mean(skipna=True)
    return None if pd.isna(mean) else float(mean)


def _safe_std(series: pd.Series) -> float | None:
    if series.empty:
        return None
    std = series.std(skipna=True)
    return None if pd.isna(std) else float(std)


def _coverage_pct(series: pd.Series) -> float | None:
    """Fraction of non-NaN values in a Series, as a percentage 0-100."""
    if series.empty:
        return None
    pct = series.notna().mean() * 100.0
    return float(pct)


def _location_entropy(locations: pd.Series) -> float | None:
    """Shannon entropy (bits) of a categorical location series."""
    if locations.empty or locations.dropna().empty:
        return None
    counts = locations.dropna().value_counts(normalize=True)
    entropy = -sum(p * math.log2(p) for p in counts if p > 0)
    return float(entropy)


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extracts features from pre-loaded hourly sensing DataFrames.

    The extractor accepts DataFrames that have already been filtered to the
    desired time window (e.g., a 60-minute hourly slice). It does NOT load
    raw CSVs — that responsibility belongs to HourlyFeatureLoader.

    Usage:
        extractor = FeatureExtractor(sensor_config={})
        feats = extractor.extract("motion", hourly_df, window_minutes=60)
    """

    def __init__(self, sensor_config: dict | None = None) -> None:
        self.sensor_config = sensor_config or {}

        self._dispatch: dict[str, Any] = {
            "accelerometer": self._extract_accelerometer,
            "gps": self._extract_gps,
            "sleep": self._extract_sleep,
            "screen": self._extract_screen,
            "app_usage": self._extract_app_usage,
            "motion": self._extract_motion,
            "keyboard": self._extract_key_input,
            "key_input": self._extract_key_input,
            "music": self._extract_music,
            "light": self._extract_light,
        }

    def extract(
        self,
        sensor_name: str,
        hourly_df: pd.DataFrame,
        window_minutes: int = 60,
    ) -> dict[str, Any]:
        """Extract features from a single sensor's hourly DataFrame.

        Args:
            sensor_name: Sensor identifier (e.g. "motion", "screen").
            hourly_df: DataFrame pre-filtered to the target time window.
                       May be empty — returns empty dict in that case.
            window_minutes: Duration of the aggregation window in minutes.
                           Used to compute coverage percentages.

        Returns:
            Dict of feature_name → value. Missing values are None.
        """
        if hourly_df is None or hourly_df.empty:
            return {}

        handler = self._dispatch.get(sensor_name)
        if handler is None:
            logger.warning("No feature extractor registered for sensor '%s'", sensor_name)
            return {}

        try:
            return handler(hourly_df)
        except Exception as exc:
            logger.error("Feature extraction failed for '%s': %s", sensor_name, exc)
            return {}

    def extract_all(
        self,
        sensing_data: dict[str, pd.DataFrame],
        window_minutes: int = 60,
    ) -> dict[str, Any]:
        """Extract features from all available sensors and merge into one dict.

        Args:
            sensing_data: {sensor_name: DataFrame} filtered to the time window.
            window_minutes: Duration of the aggregation window in minutes.

        Returns:
            Flat dict of all features across sensors (keys prefixed by sensor).
        """
        all_features: dict[str, Any] = {}
        for sensor_name, df in sensing_data.items():
            feats = self.extract(sensor_name, df, window_minutes)
            all_features.update(feats)
        return all_features

    # ------------------------------------------------------------------
    # Per-sensor extractors
    # ------------------------------------------------------------------

    def _extract_accelerometer(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract accelerometer / actigraphy features.

        Expected columns (any subset may be present):
            accel_x, accel_y, accel_z       (raw axes, g)
            magnitude                        (pre-computed vector magnitude)
            accel_mean_magnitude             (hourly pre-aggregated)
            accel_std_magnitude
            accel_activity_counts
            accel_coverage_pct
        """
        feats: dict[str, Any] = {}

        # If data is already hourly-aggregated, prefer those columns
        for col_agg in ("accel_mean_magnitude", "mean_magnitude"):
            if col_agg in data.columns:
                feats["accel_mean_mag"] = _safe_mean(data[col_agg])
                break

        for col_agg in ("accel_std_magnitude", "std_magnitude"):
            if col_agg in data.columns:
                feats["accel_std_mag"] = _safe_mean(data[col_agg])
                break

        for col_agg in ("accel_activity_counts", "activity_counts"):
            if col_agg in data.columns:
                feats["accel_activity_counts"] = _safe_sum(data[col_agg])
                break

        for col_agg in ("accel_coverage_pct", "coverage_pct"):
            if col_agg in data.columns:
                feats["accel_coverage_pct"] = _safe_mean(data[col_agg])
                break

        # Compute from raw axes if pre-aggregated columns are absent
        if "accel_mean_mag" not in feats:
            axes_present = [c for c in ("accel_x", "accel_y", "accel_z") if c in data.columns]
            if axes_present:
                mag = (data[axes_present] ** 2).sum(axis=1) ** 0.5
                feats["accel_mean_mag"] = _safe_mean(mag)
                feats["accel_std_mag"] = _safe_std(mag)
                # Simple activity counts: samples above 0.1 g deviation from 1 g
                feats["accel_activity_counts"] = float((abs(mag - 1.0) > 0.1).sum())
            elif "magnitude" in data.columns:
                feats["accel_mean_mag"] = _safe_mean(data["magnitude"])
                feats["accel_std_mag"] = _safe_std(data["magnitude"])

        if "accel_coverage_pct" not in feats:
            ref_col = next((c for c in ("magnitude", "accel_x") if c in data.columns), None)
            if ref_col:
                feats["accel_coverage_pct"] = _coverage_pct(data[ref_col])

        return feats

    def _extract_gps(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract GPS / mobility features.

        Expected columns (any subset):
            latitude, longitude             (raw GPS fixes)
            distance_km                     (pre-computed segment distance)
            gps_distance_km                 (hourly aggregate)
            at_home_min, gps_at_home_min
            speed_max_kmh, gps_speed_max_kmh
            location_entropy, gps_location_entropy
            dist_from_home_max_km, gps_dist_from_home_max_km
            n_captures, gps_n_captures
            location_label                  (cluster label for entropy calc)
        """
        feats: dict[str, Any] = {}

        # n_captures
        for col in ("gps_n_captures", "n_captures"):
            if col in data.columns:
                feats["gps_n_captures"] = _safe_sum(data[col])
                break
        if "gps_n_captures" not in feats:
            feats["gps_n_captures"] = len(data) if not data.empty else None

        # distance
        for col in ("gps_distance_km", "distance_km"):
            if col in data.columns:
                feats["gps_distance_km"] = _safe_sum(data[col])
                break

        # at_home minutes
        for col in ("gps_at_home_min", "at_home_min"):
            if col in data.columns:
                feats["gps_at_home_min"] = _safe_sum(data[col])
                break

        # max speed
        for col in ("gps_speed_max_kmh", "speed_max_kmh", "speed_kmh"):
            if col in data.columns:
                feats["gps_speed_max_kmh"] = _safe_float(data[col].max())
                break

        # location entropy
        for col in ("gps_location_entropy", "location_entropy"):
            if col in data.columns:
                feats["gps_location_entropy"] = _safe_mean(data[col])
                break
        if "gps_location_entropy" not in feats and "location_label" in data.columns:
            feats["gps_location_entropy"] = _location_entropy(data["location_label"])

        # max distance from home
        for col in ("gps_dist_from_home_max_km", "dist_from_home_max_km", "dist_from_home_km"):
            if col in data.columns:
                feats["gps_dist_from_home_max_km"] = _safe_float(data[col].max())
                break

        return feats

    def _extract_sleep(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract sleep features from accelerometer-derived or dedicated sleep sensor.

        Expected columns (any subset):
            sleep_duration_min, amt_sleep_day_min  (from sleep sensor)
            val_sleep_duration_min                  (from accel-based sleep)
            sleep_onset, sleep_offset               (timestamps)
            is_sleep                                (binary flag per sample)
        """
        feats: dict[str, Any] = {}

        for col in ("sleep_duration_min", "amt_sleep_day_min", "val_sleep_duration_min"):
            if col in data.columns:
                feats["sleep_duration_min"] = _safe_sum(data[col])
                break

        # Derive duration from binary flag
        if "sleep_duration_min" not in feats and "is_sleep" in data.columns:
            sleep_count = data["is_sleep"].sum(skipna=True)
            # Assume 1-minute resolution
            feats["sleep_duration_min"] = _safe_float(sleep_count)

        # Sleep quality proxy: movement during sleep (lower = better)
        if "accel_mean_mag" in data.columns:
            feats["sleep_movement_mean"] = _safe_mean(data["accel_mean_mag"])

        return feats

    def _extract_screen(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract screen usage features.

        Expected columns (any subset):
            screen_on_min                   (minutes screen was on)
            n_sessions, screen_n_sessions   (number of unlock episodes)
            app_total_min                   (total foreground app time)
            app_social_min                  (social app minutes)
            app_comm_min                    (communication app minutes)
        """
        feats: dict[str, Any] = {}

        for col in ("screen_on_min",):
            if col in data.columns:
                feats["screen_on_min"] = _safe_sum(data[col])
                break

        for col in ("screen_n_sessions", "n_sessions"):
            if col in data.columns:
                feats["screen_n_sessions"] = _safe_sum(data[col])
                break

        for col in ("app_total_min",):
            if col in data.columns:
                feats["app_total_min"] = _safe_sum(data[col])
                break

        for col in ("app_social_min",):
            if col in data.columns:
                feats["app_social_min"] = _safe_sum(data[col])
                break

        for col in ("app_comm_min",):
            if col in data.columns:
                feats["app_comm_min"] = _safe_sum(data[col])
                break

        return feats

    def _extract_app_usage(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract application usage features.

        Expected columns (any subset):
            id_app                      (app identifier)
            amt_foreground_day_sec      (seconds in foreground)
            category                    (app category label)
        """
        feats: dict[str, Any] = {}

        if "amt_foreground_day_sec" in data.columns:
            total_sec = _safe_sum(data["amt_foreground_day_sec"])
            feats["app_total_min"] = (total_sec / 60.0) if total_sec is not None else None

            # Category breakdown
            if "category" in data.columns:
                for cat_key, cat_label in [
                    ("app_social_min", "social"),
                    ("app_comm_min", "communication"),
                ]:
                    cat_data = data[data["category"].str.lower().str.contains(
                        cat_label, na=False
                    )]
                    cat_sec = _safe_sum(cat_data["amt_foreground_day_sec"])
                    feats[cat_key] = (cat_sec / 60.0) if cat_sec is not None else None

        return feats

    def _extract_motion(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract motion / activity classification features.

        Expected columns (any subset):
            motion_stationary_min, stationary_min
            motion_walking_min, walking_min
            motion_automotive_min, automotive_min
            motion_running_min, running_min
            motion_active_min, active_min
            motion_coverage_pct, coverage_pct
        """
        feats: dict[str, Any] = {}

        col_pairs = [
            ("motion_stationary_min", "stationary_min", "motion_stationary_min"),
            ("motion_walking_min", "walking_min", "motion_walking_min"),
            ("motion_automotive_min", "automotive_min", "motion_automotive_min"),
            ("motion_running_min", "running_min", "motion_running_min"),
            ("motion_active_min", "active_min", "motion_active_min"),
            ("motion_coverage_pct", "coverage_pct", "motion_coverage_pct"),
        ]

        for primary, secondary, feat_key in col_pairs:
            for col in (primary, secondary):
                if col in data.columns:
                    if "coverage" in feat_key:
                        feats[feat_key] = _safe_mean(data[col])
                    else:
                        feats[feat_key] = _safe_sum(data[col])
                    break

        # Derive active_min from walking + running if not present
        if "motion_active_min" not in feats:
            walking = feats.get("motion_walking_min")
            running = feats.get("motion_running_min")
            if walking is not None or running is not None:
                feats["motion_active_min"] = (walking or 0.0) + (running or 0.0)

        return feats

    def _extract_key_input(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract keyboard / typing features.

        Expected columns (any subset):
            key_chars_typed, n_char_day_allapps
            key_words_typed, n_word_day_allapps
            key_n_sessions, n_sessions
            key_prop_neg, prop_word_neg_day_allapps
            key_prop_pos, prop_word_pos_day_allapps
        """
        feats: dict[str, Any] = {}

        for col in ("key_chars_typed", "n_char_day_allapps", "chars_typed"):
            if col in data.columns:
                feats["key_chars_typed"] = _safe_sum(data[col])
                break

        for col in ("key_words_typed", "n_word_day_allapps", "words_typed"):
            if col in data.columns:
                feats["key_words_typed"] = _safe_sum(data[col])
                break

        for col in ("key_n_sessions", "n_sessions"):
            if col in data.columns:
                feats["key_n_sessions"] = _safe_sum(data[col])
                break

        for col in ("key_prop_neg", "prop_word_neg_day_allapps", "prop_neg"):
            if col in data.columns:
                feats["key_prop_neg"] = _safe_mean(data[col])
                break

        for col in ("key_prop_pos", "prop_word_pos_day_allapps", "prop_pos"):
            if col in data.columns:
                feats["key_prop_pos"] = _safe_mean(data[col])
                break

        return feats

    def _extract_music(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract music listening features.

        Expected columns (any subset):
            is_listening, mus_is_listening
            n_tracks, mus_n_tracks
        """
        feats: dict[str, Any] = {}

        for col in ("mus_is_listening", "is_listening"):
            if col in data.columns:
                any_listening = data[col].any()
                feats["mus_is_listening"] = bool(any_listening) if pd.notna(any_listening) else None
                break

        for col in ("mus_n_tracks", "n_tracks"):
            if col in data.columns:
                feats["mus_n_tracks"] = _safe_sum(data[col])
                break

        return feats

    def _extract_light(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract ambient light sensor features.

        Expected columns (any subset):
            mean_lux, light_mean_lux, lux
        """
        feats: dict[str, Any] = {}

        for col in ("light_mean_lux", "mean_lux", "lux"):
            if col in data.columns:
                # If already hourly-aggregated means, average them; otherwise mean raw
                feats["light_mean_lux"] = _safe_mean(data[col])
                break

        return feats
