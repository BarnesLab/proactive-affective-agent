"""Data preprocessing: sensing alignment, trait extraction, pilot data preparation.

Joins daily sensing aggregates to EMA entries by date, extracts user trait profiles,
and selects/prepares data for the 5-user pilot.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.data.schema import SensingDay, UserProfile
from src.utils.mappings import (
    BASELINE_DEMOGRAPHICS,
    BASELINE_SCALES,
    SENSING_COLUMNS,
    study_id_to_participant_id,
)


def align_sensing_to_ema(
    ema_row: pd.Series,
    sensing_dfs: dict[str, pd.DataFrame],
    study_id: int,
) -> SensingDay | None:
    """Join sensing data to a single EMA entry by date.

    Args:
        ema_row: A single row from the EMA split DataFrame.
        sensing_dfs: Pre-loaded {sensor_name: DataFrame} from loader.
        study_id: The user's Study_ID.

    Returns:
        SensingDay dataclass or None if no sensing data for that date.
    """
    target_date = ema_row["date_local"]
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()

    pid = study_id_to_participant_id(study_id)
    day = SensingDay(id_participant=pid, date=target_date)
    has_data = False

    # Accelerometer
    if "accelerometer" in sensing_dfs:
        df = sensing_dfs["accelerometer"]
        info = SENSING_COLUMNS["accelerometer"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)]
        if not rows.empty:
            r = rows.iloc[0]
            day.accel_sleep_duration_min = _safe_float(r, "val_sleep_duration_min")
            day.accel_count = _safe_int(r, "n_acc")
            has_data = True

    # Sleep
    if "sleep" in sensing_dfs:
        df = sensing_dfs["sleep"]
        info = SENSING_COLUMNS["sleep"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)]
        if not rows.empty:
            day.sleep_duration_min = _safe_float(rows.iloc[0], "amt_sleep_day_min")
            has_data = True

    # Android sleep
    if "android_sleep" in sensing_dfs:
        df = sensing_dfs["android_sleep"]
        info = SENSING_COLUMNS["android_sleep"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)]
        if not rows.empty:
            r = rows.iloc[0]
            day.android_sleep_min = _safe_float(r, "amt_sleep_min")
            day.android_sleep_status = str(r.get("cat_status", ""))
            has_data = True

    # GPS
    if "gps" in sensing_dfs:
        df = sensing_dfs["gps"]
        info = SENSING_COLUMNS["gps"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)]
        if not rows.empty:
            r = rows.iloc[0]
            day.gps_captures = _safe_int(r, "n_capture_day")
            day.travel_events = _safe_int(r, "n_travelevent_day")
            day.travel_km = _safe_float(r, "amt_travel_day_km")
            day.travel_minutes = _safe_float(r, "amt_travel_day_minutes")
            day.home_minutes = _safe_float(r, "amt_home_day_minutes")
            day.max_distance_from_home_km = _safe_float(r, "amt_distancefromhome_day_max_km")
            day.location_variance = _safe_float(r, "amt_location_day_variance")
            has_data = True

    # Screen
    if "screen" in sensing_dfs:
        df = sensing_dfs["screen"]
        info = SENSING_COLUMNS["screen"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)]
        if not rows.empty:
            r = rows.iloc[0]
            day.screen_sessions = _safe_int(r, "n_session_screenon_day")
            day.screen_minutes = _safe_float(r, "amt_screenon_day_minutes")
            day.screen_max_session_min = _safe_float(r, "amt_screenon_session_day_max_minutes")
            day.screen_mean_session_min = _safe_float(r, "amt_screenon_session_day_mean_minutes")
            has_data = True

    # Motion
    if "motion" in sensing_dfs:
        df = sensing_dfs["motion"]
        info = SENSING_COLUMNS["motion"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)]
        if not rows.empty:
            r = rows.iloc[0]
            day.stationary_min = _safe_float(r, "amt_stationary_day_min")
            day.walking_min = _safe_float(r, "amt_walking_day_min")
            day.automotive_min = _safe_float(r, "amt_automotive_day_min")
            day.running_min = _safe_float(r, "amt_running_day_min")
            day.cycling_min = _safe_float(r, "amt_cycling_day_min")
            has_data = True

    # Key input
    if "key_input" in sensing_dfs:
        df = sensing_dfs["key_input"]
        info = SENSING_COLUMNS["key_input"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)]
        if not rows.empty:
            r = rows.iloc[0]
            day.chars_typed = _safe_int(r, "n_char_day_allapps")
            day.words_typed = _safe_int(r, "n_word_day_allapps")
            day.negative_words = _safe_int(r, "n_word_neg_day_allapps")
            day.positive_words = _safe_int(r, "n_word_pos_day_allapps")
            day.prop_negative = _safe_float(r, "prop_word_neg_day_allapps")
            day.prop_positive = _safe_float(r, "prop_word_pos_day_allapps")
            has_data = True

    # App usage (aggregate total seconds across all apps for the day)
    if "app_usage" in sensing_dfs:
        df = sensing_dfs["app_usage"]
        info = SENSING_COLUMNS["app_usage"]
        rows = df[(df[info["id_col"]] == pid) & (df[info["date_col"]] == target_date)]
        if not rows.empty:
            day.total_app_seconds = rows["amt_foreground_day_sec"].sum()
            # Top 5 apps by usage
            top = rows.nlargest(5, "amt_foreground_day_sec")
            day.top_apps = [
                (str(r["id_app"]), float(r["amt_foreground_day_sec"]))
                for _, r in top.iterrows()
            ]
            has_data = True

    return day if has_data else None


def get_user_trait_profile(baseline_df: pd.DataFrame, study_id: int) -> UserProfile:
    """Extract a structured trait profile for a user from baseline data.

    Args:
        baseline_df: Baseline DataFrame indexed by Study_ID.
        study_id: The user's Study_ID.

    Returns:
        UserProfile dataclass.
    """
    profile = UserProfile(study_id=study_id)

    if study_id not in baseline_df.index:
        return profile

    row = baseline_df.loc[study_id]

    profile.age = _safe_int_val(row.get("age_demo"))
    gender_code = row.get("gender")
    if gender_code == 1:
        profile.gender = "Female"
    elif gender_code == 2:
        profile.gender = "Male"
    elif gender_code == 3:
        profile.gender = "Non-binary"
    else:
        profile.gender = str(gender_code) if pd.notna(gender_code) else None

    profile.cancer_diagnosis = str(row.get("cancerdx", "")) if pd.notna(row.get("cancerdx")) else None
    profile.cancer_stage = str(row.get("cancer_stage", "")) if pd.notna(row.get("cancer_stage")) else None
    profile.cancer_years = _safe_float_val(row.get("cancer_years"))
    profile.depression_phq8 = _safe_float_val(row.get("PHQ8_TOTAL"))
    profile.anxiety_gad7 = _safe_float_val(row.get("GAD7_TOTAL"))
    profile.trait_positive_affect = _safe_float_val(row.get("PANAS_POS"))
    profile.trait_negative_affect = _safe_float_val(row.get("PANAS_NEG"))
    profile.extraversion = _safe_float_val(row.get("TIPI_Extraversion"))
    profile.neuroticism_stability = _safe_float_val(row.get("TIPI_Stability"))
    profile.social_support = _safe_float_val(row.get("MSPSS_TOTAL"))
    profile.self_efficacy = _safe_float_val(row.get("GSE_TOTAL"))

    return profile


def prepare_pilot_data(
    loader,
    pilot_user_ids: list[int],
    ema_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Prepare structured pilot data for selected users.

    Args:
        loader: DataLoader instance.
        pilot_user_ids: Specific user IDs to use.
        ema_df: Pre-loaded EMA DataFrame (all users). If None, loads via load_all_ema().

    Returns:
        List of dicts, one per user, with keys:
        - study_id, profile, memory, ema_entries (list of Series), sensing_days (list of SensingDay)
    """
    if ema_df is None:
        ema_df = loader.load_all_ema()

    sensing_dfs = loader.load_all_sensing()

    try:
        baseline_df = loader.load_baseline()
    except FileNotFoundError:
        baseline_df = pd.DataFrame()

    users_data = []
    for sid in pilot_user_ids:
        user_ema = ema_df[ema_df["Study_ID"] == sid].sort_values("timestamp_local")
        if user_ema.empty:
            continue

        profile = get_user_trait_profile(baseline_df, sid) if not baseline_df.empty else UserProfile(study_id=sid)
        memory = loader.load_memory_for_user(sid)

        ema_entries = []
        sensing_days = []
        for _, row in user_ema.iterrows():
            ema_entries.append(row)
            sd = align_sensing_to_ema(row, sensing_dfs, sid)
            sensing_days.append(sd)

        users_data.append({
            "study_id": sid,
            "profile": profile,
            "memory": memory or "",
            "ema_entries": ema_entries,
            "sensing_days": sensing_days,
        })

    return users_data


# --- Helpers ---

def _safe_float(row, col: str) -> float | None:
    val = row.get(col)
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(row, col: str) -> int | None:
    val = row.get(col)
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float_val(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int_val(val) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
