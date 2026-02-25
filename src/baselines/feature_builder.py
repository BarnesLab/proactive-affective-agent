"""Feature extraction: convert sensing data to fixed-length feature vectors for ML.

Provides daily aggregate features (current) and hourly features (placeholder).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessing import align_sensing_to_ema
from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

logger = logging.getLogger(__name__)

# Feature names for the daily aggregate sensing vector
DAILY_FEATURE_NAMES = [
    "accel_sleep_duration_min",
    "sleep_duration_min",
    "android_sleep_min",
    "travel_km",
    "travel_minutes",
    "home_minutes",
    "max_distance_from_home_km",
    "location_variance",
    "screen_sessions",
    "screen_minutes",
    "screen_max_session_min",
    "stationary_min",
    "walking_min",
    "automotive_min",
    "running_min",
    "cycling_min",
    "words_typed",
    "prop_positive",
    "prop_negative",
    "total_app_seconds",
]


def build_daily_features(
    ema_df: pd.DataFrame,
    sensing_dfs: dict[str, pd.DataFrame],
    user_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a feature matrix from daily aggregate sensing aligned to EMA entries.

    Args:
        ema_df: EMA DataFrame with Study_ID, date_local, and target columns.
        sensing_dfs: Pre-loaded {sensor_name: DataFrame}.
        user_ids: If provided, only build features for these users.

    Returns:
        Tuple of (X, y_continuous, y_binary):
        - X: DataFrame of shape (n_samples, n_features) with DAILY_FEATURE_NAMES columns.
        - y_continuous: DataFrame with CONTINUOUS_TARGETS columns.
        - y_binary: DataFrame with BINARY_STATE_TARGETS + INT_availability columns.
    """
    if user_ids is not None:
        ema_df = ema_df[ema_df["Study_ID"].isin(user_ids)]

    rows_X = []
    rows_y_cont = []
    rows_y_bin = []
    meta_rows = []

    for _, ema_row in ema_df.iterrows():
        sid = int(ema_row["Study_ID"])
        sensing_day = align_sensing_to_ema(ema_row, sensing_dfs, sid)

        # Extract feature vector
        feature_vec = _sensing_day_to_vector(sensing_day)
        rows_X.append(feature_vec)

        # Extract targets
        cont = {}
        for target in CONTINUOUS_TARGETS:
            val = ema_row.get(target)
            cont[target] = float(val) if pd.notna(val) else np.nan
        rows_y_cont.append(cont)

        bin_targets = {}
        for target in BINARY_STATE_TARGETS:
            val = ema_row.get(target)
            if pd.notna(val):
                if isinstance(val, bool):
                    bin_targets[target] = int(val)
                elif isinstance(val, str):
                    bin_targets[target] = 1 if val.lower().strip() in ("true", "1", "yes") else 0
                else:
                    bin_targets[target] = int(bool(val))
            else:
                bin_targets[target] = np.nan

        # Availability
        avail = ema_row.get("INT_availability")
        bin_targets["INT_availability"] = (
            1 if str(avail).lower().strip() == "yes" else 0
        ) if pd.notna(avail) else np.nan
        rows_y_bin.append(bin_targets)

        meta_rows.append({"Study_ID": sid, "date_local": ema_row.get("date_local")})

    X = pd.DataFrame(rows_X, columns=DAILY_FEATURE_NAMES)
    y_cont = pd.DataFrame(rows_y_cont)
    y_bin = pd.DataFrame(rows_y_bin)

    # Add metadata index
    meta = pd.DataFrame(meta_rows)
    X.index = meta.index
    y_cont.index = meta.index
    y_bin.index = meta.index

    # Store metadata as attributes
    X.attrs["meta"] = meta

    logger.info(
        f"Built daily features: {X.shape[0]} samples, {X.shape[1]} features, "
        f"missing rate={X.isna().mean().mean():.1%}"
    )
    return X, y_cont, y_bin


def _sensing_day_to_vector(sensing_day) -> dict[str, float]:
    """Convert a SensingDay to a flat feature dict matching DAILY_FEATURE_NAMES."""
    if sensing_day is None:
        return {name: np.nan for name in DAILY_FEATURE_NAMES}

    data = sensing_day.to_summary_dict()
    return {name: data.get(name, np.nan) for name in DAILY_FEATURE_NAMES}


def fit_imputer(
    X_train: pd.DataFrame, strategy: str = "median"
) -> dict[str, float]:
    """Fit an imputer on training data, returning per-column fill values.

    Args:
        X_train: Training feature DataFrame (used to compute statistics).
        strategy: "median" or "mean".

    Returns:
        Dict mapping column name to its fill value (median or mean from X_train).
    """
    if strategy == "median":
        fill_values = X_train.median()
    elif strategy == "mean":
        fill_values = X_train.mean()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return fill_values.to_dict()


def apply_imputer(X: pd.DataFrame, imputer: dict[str, float]) -> pd.DataFrame:
    """Apply a fitted imputer to a DataFrame.

    Args:
        X: Feature DataFrame with possible NaN values.
        imputer: Dict of column -> fill value (from :func:`fit_imputer`).

    Returns:
        Imputed DataFrame with NaN values filled using the provided statistics.
    """
    return X.fillna(imputer)


def impute_features(X: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing values in the feature matrix.

    .. deprecated::
        This function computes fill statistics from *X* itself, which causes
        data leakage when applied to a test set.  Use :func:`fit_imputer` +
        :func:`apply_imputer` instead.

    Args:
        X: Feature DataFrame with possible NaN values.
        strategy: "median" or "mean".

    Returns:
        Imputed DataFrame (NaN columns filled with column median/mean).
    """
    warnings.warn(
        "impute_features() computes statistics from the input itself, which "
        "causes train/test leakage. Use fit_imputer() + apply_imputer() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if strategy == "median":
        return X.fillna(X.median())
    elif strategy == "mean":
        return X.fillna(X.mean())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def build_parquet_features(
    ema_df: pd.DataFrame,
    processed_dir: str | Path,
    user_ids: list[int] | None = None,
    lookback_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ML feature matrix from processed hourly Parquet files.

    Uses HourlyFeatureLoader to load pre-processed Parquet data for each EMA
    entry, producing a flat hourly feature vector (h0_screen_on_min, etc.).
    Requires Phase 1 Parquet outputs in processed_dir/{modality}/.

    Args:
        ema_df: EMA DataFrame with Study_ID, timestamp_local, and target columns.
        processed_dir: Path to data/processed/hourly/ directory.
        user_ids: If provided, only build features for these users.
        lookback_hours: Hours of sensing history before each EMA (default 24).

    Returns:
        Tuple of (X, y_continuous, y_binary) — same format as build_daily_features.
    """
    from src.data.hourly_features import HourlyFeatureLoader
    from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

    loader = HourlyFeatureLoader(processed_dir=processed_dir)

    if user_ids is not None:
        ema_df = ema_df[ema_df["Study_ID"].isin(user_ids)]

    rows_X = []
    rows_y_cont = []
    rows_y_bin = []
    meta_rows = []

    for _, ema_row in ema_df.iterrows():
        sid = int(ema_row["Study_ID"])
        ts = ema_row.get("timestamp_local") or ema_row.get("Timestamp_start")

        # Feature vector from Parquet
        try:
            feat = loader.get_feature_matrix(sid, ts, lookback_hours=lookback_hours)
        except Exception:
            feat = {}
        rows_X.append(feat)

        # Continuous targets
        cont = {t: ema_row.get(t, np.nan) for t in CONTINUOUS_TARGETS}
        rows_y_cont.append(cont)

        # Binary targets
        bin_targets: dict[str, float] = {}
        for target in BINARY_STATE_TARGETS:
            val = ema_row.get(target)
            if pd.notna(val):
                if isinstance(val, bool):
                    bin_targets[target] = int(val)
                elif isinstance(val, str):
                    bin_targets[target] = 1 if val.lower().strip() in ("true", "1", "yes") else 0
                else:
                    bin_targets[target] = int(bool(val))
            else:
                bin_targets[target] = np.nan
        avail = ema_row.get("INT_availability")
        bin_targets["INT_availability"] = (
            1 if str(avail).lower().strip() == "yes" else 0
        ) if pd.notna(avail) else np.nan
        rows_y_bin.append(bin_targets)

        meta_rows.append({"Study_ID": sid, "timestamp_local": ts})

    X = pd.DataFrame(rows_X).fillna(np.nan)
    y_cont = pd.DataFrame(rows_y_cont)
    y_bin = pd.DataFrame(rows_y_bin)

    logger.info(
        f"Built Parquet features: {X.shape[0]} samples, {X.shape[1]} features, "
        f"missing rate={X.isna().mean().mean():.1%}"
    )
    return X, y_cont, y_bin
