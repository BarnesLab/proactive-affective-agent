"""Feature extraction: convert sensing data to fixed-length feature vectors for ML.

Provides daily aggregate features (current) and hourly features (placeholder).
"""

from __future__ import annotations

import logging
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


def impute_features(X: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing values in the feature matrix.

    Args:
        X: Feature DataFrame with possible NaN values.
        strategy: "median" or "mean".

    Returns:
        Imputed DataFrame (NaN columns filled with column median/mean).
    """
    if strategy == "median":
        return X.fillna(X.median())
    elif strategy == "mean":
        return X.fillna(X.mean())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def build_hourly_features(
    ema_df: pd.DataFrame,
    sensing_dfs: dict[str, pd.DataFrame],
    user_ids: list[int] | None = None,
    lookback_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Placeholder: build hourly features from raw minute-level data.

    Will be implemented when colleague provides raw data + extraction code.

    Args:
        ema_df: EMA DataFrame.
        sensing_dfs: Raw minute-level sensing DataFrames.
        user_ids: Filter to these users.
        lookback_hours: Hours of history before each EMA.

    Returns:
        Same format as build_daily_features.
    """
    raise NotImplementedError(
        "Hourly features require raw minute-level data from colleague. "
        "Use build_daily_features() for now."
    )
