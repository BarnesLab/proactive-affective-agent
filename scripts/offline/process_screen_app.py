"""
Phase 1 Offline Processing — Screen Time & App Usage
=====================================================
Handles ScreenOnTime (iOS) and APPUSAGE (Android) into a unified hourly
output.

Input directories
-----------------
  iOS   : data/bucs-data/ScreenOnTime/
  Android: data/bucs-data/APPUSAGE/

Output
------
  data/processed/hourly/screen/{pid}_screen_hourly.parquet

Per-hour columns
----------------
  hour_utc, hour_local, participant_id,
  screen_on_min, screen_n_sessions, screen_mean_session_min, screen_max_session_min,
  app_total_min, app_social_min, app_comm_min, app_entertainment_min,
  app_game_min, app_n_apps,
  structural_missing_app  (True for iOS: no app-level data)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "bucs-data"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "screen"
PLATFORM_FILE = PROJECT_ROOT / "data" / "processed" / "hourly" / "participant_platform.parquet"

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    WITHDRAW_LIST,
    parse_tz_minutes,
    epoch_to_hour_utc,
    pid_from_int,
    load_platform_map,
    tz_mode_per_hour,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SESSION_HOURS = 3          # cap individual screen sessions at 3 h (artifact filter)
MAX_APPUSAGE_HOURS = 10        # drop app usage windows > 10 h (artifact)

# ---------------------------------------------------------------------------
# App category mapping
# ---------------------------------------------------------------------------
APP_CATEGORIES: dict[str, str] = {
    # Social
    'com.facebook.katana': 'SOCIAL',
    'com.instagram.android': 'SOCIAL',
    'com.twitter.android': 'SOCIAL',
    'com.snapchat.android': 'SOCIAL',
    'com.tiktok.': 'SOCIAL',
    'com.reddit': 'SOCIAL',
    # Communication
    'com.whatsapp': 'COMMUNICATION',
    'com.google.android.apps.messaging': 'COMMUNICATION',
    'com.facebook.orca': 'COMMUNICATION',
    'org.telegram': 'COMMUNICATION',
    'com.skype': 'COMMUNICATION',
    'com.discord': 'COMMUNICATION',
    # Entertainment
    'com.spotify.music': 'ENTERTAINMENT',
    'com.netflix': 'ENTERTAINMENT',
    'com.youtube': 'ENTERTAINMENT',
    'com.google.android.youtube': 'ENTERTAINMENT',
    # Game
    'com.game': 'GAME',
    # Productivity
    'com.google.android.apps.docs': 'PRODUCTIVITY',
    'com.microsoft': 'PRODUCTIVITY',
    # Health
    'com.health': 'HEALTH',
}


def categorize_app(app_id) -> str:
    """Map an app package ID to a high-level category string."""
    if not app_id or str(app_id).strip().lower() in ('nan', 'android', ''):
        return 'SYSTEM'
    app_lower = str(app_id).lower()
    for prefix, cat in APP_CATEGORIES.items():
        if prefix in app_lower:
            return cat
    return 'OTHER'


# ---------------------------------------------------------------------------
# iOS — ScreenOnTime
# ---------------------------------------------------------------------------

def load_ios_screen_files(pid_str: str) -> pd.DataFrame:
    """Load all ScreenOnTime data CSV files (not header files) for a PID."""
    sot_dir = DATA_ROOT / "ScreenOnTime"
    # Primary pattern: ScreenOnTime_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
    files = sorted(sot_dir.glob(f"ScreenOnTime_data_{pid_str}_*.csv"))
    # Alternate pattern: {PID}_ScreenOnTime_*
    files += sorted(sot_dir.glob(f"{pid_str}_ScreenOnTime_*.csv"))
    # Drop header files
    files = [f for f in files if 'header' not in f.name.lower()]

    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Header-only files have no epoch_capture column
            if 'epoch_capture' not in df.columns:
                continue
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"[screen/ios] PID {pid_str}: failed to read {f.name} — {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ios_build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build (start_ms, end_ms) screen-on sessions from ScreenOnTime events.

    Sessions are consecutive (unlocked_epoch, next_locked_epoch) pairs.
    Sessions longer than MAX_SESSION_HOURS are dropped as artifacts.
    """
    df = df.copy()
    df['epoch_capture'] = pd.to_numeric(df['epoch_capture'], errors='coerce')
    df = df.dropna(subset=['epoch_capture', 'cat_state'])
    df = df.sort_values('epoch_capture').reset_index(drop=True)

    max_session_ms = MAX_SESSION_HOURS * 3_600_000

    sessions = []
    pending_unlock: float | None = None

    for _, row in df.iterrows():
        state = str(row['cat_state']).lower().strip()
        epoch = float(row['epoch_capture'])
        tz = row.get('timezone', '+00:00')

        if state == 'unlocked':
            pending_unlock = epoch
        elif state == 'locked' and pending_unlock is not None:
            duration_ms = epoch - pending_unlock
            if 0 < duration_ms <= max_session_ms:
                sessions.append({
                    'start_ms': pending_unlock,
                    'end_ms': epoch,
                    'timezone': tz,
                })
            pending_unlock = None

    if not sessions:
        return pd.DataFrame(columns=['start_ms', 'end_ms', 'timezone'])
    return pd.DataFrame(sessions)


def ios_aggregate_hour(sessions: pd.DataFrame, hour_start_ms: float) -> dict:
    """Aggregate screen sessions for one hour into feature dict."""
    hour_end_ms = hour_start_ms + 3_600_000
    df = sessions.copy()
    df = df[(df['end_ms'] > hour_start_ms) & (df['start_ms'] < hour_end_ms)]
    if df.empty:
        return _ios_empty_hour()

    df = df.copy()
    df['clip_start'] = df['start_ms'].clip(lower=hour_start_ms)
    df['clip_end'] = df['end_ms'].clip(upper=hour_end_ms)
    df['duration_s'] = (df['clip_end'] - df['clip_start']) / 1000.0
    df = df[df['duration_s'] > 0]

    if df.empty:
        return _ios_empty_hour()

    total_s = df['duration_s'].sum()
    n_sessions = len(df)
    mean_min = (total_s / n_sessions) / 60.0 if n_sessions > 0 else np.nan
    max_min = df['duration_s'].max() / 60.0

    return {
        'screen_on_min': total_s / 60.0,
        'screen_n_sessions': n_sessions,
        'screen_mean_session_min': mean_min,
        'screen_max_session_min': max_min,
        # App columns are structurally missing for iOS
        'app_total_min': np.nan,
        'app_social_min': np.nan,
        'app_comm_min': np.nan,
        'app_entertainment_min': np.nan,
        'app_game_min': np.nan,
        'app_n_apps': np.nan,
        'structural_missing_app': True,
    }


def _ios_empty_hour() -> dict:
    return {
        'screen_on_min': 0.0,
        'screen_n_sessions': 0,
        'screen_mean_session_min': np.nan,
        'screen_max_session_min': np.nan,
        'app_total_min': np.nan,
        'app_social_min': np.nan,
        'app_comm_min': np.nan,
        'app_entertainment_min': np.nan,
        'app_game_min': np.nan,
        'app_n_apps': np.nan,
        'structural_missing_app': True,
    }


# ---------------------------------------------------------------------------
# Android — APPUSAGE
# ---------------------------------------------------------------------------

def load_android_appusage_files(pid_str: str) -> pd.DataFrame:
    """Load all APPUSAGE data CSV files for a PID."""
    appusage_dir = DATA_ROOT / "APPUSAGE"
    # Primary: {PID}_APPUSAGE_usageLog_{date}.csv
    files = sorted(appusage_dir.glob(f"{pid_str}_APPUSAGE_*.csv"))
    # Alternate: APPUSAGE_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
    files += sorted(appusage_dir.glob(f"APPUSAGE_data_{pid_str}_*.csv"))
    files = [f for f in files if 'header' not in f.name.lower()]

    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'epoch_usagewindow_start' not in df.columns:
                continue
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"[screen/android] PID {pid_str}: failed to read {f.name} — {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def android_aggregate_hour(df: pd.DataFrame, hour_start_ms: float) -> dict:
    """Aggregate APPUSAGE windows for one hour into feature dict."""
    hour_end_ms = hour_start_ms + 3_600_000
    sub = df[(df['epoch_usagewindow_end'] > hour_start_ms) &
             (df['epoch_usagewindow_start'] < hour_end_ms)].copy()

    if sub.empty:
        return _android_empty_hour()

    sub['clip_start'] = sub['epoch_usagewindow_start'].clip(lower=hour_start_ms)
    sub['clip_end'] = sub['epoch_usagewindow_end'].clip(upper=hour_end_ms)
    sub['duration_s'] = (sub['clip_end'] - sub['clip_start']) / 1000.0
    sub = sub[sub['duration_s'] > 0]

    if sub.empty:
        return _android_empty_hour()

    total_s = sub['duration_s'].sum()
    n_windows = len(sub)
    n_apps = sub['id_app'].nunique() if 'id_app' in sub.columns else np.nan

    # Category breakdowns
    if 'app_cat' in sub.columns:
        cat_min = sub.groupby('app_cat')['duration_s'].sum() / 60.0
    else:
        cat_min = pd.Series(dtype='float64')

    def _cat(name: str) -> float:
        return float(cat_min.get(name, 0.0))

    return {
        'screen_on_min': total_s / 60.0,          # proxy for screen time
        'screen_n_sessions': n_windows,
        'screen_mean_session_min': (total_s / n_windows) / 60.0,
        'screen_max_session_min': sub['duration_s'].max() / 60.0,
        'app_total_min': total_s / 60.0,
        'app_social_min': _cat('SOCIAL'),
        'app_comm_min': _cat('COMMUNICATION'),
        'app_entertainment_min': _cat('ENTERTAINMENT'),
        'app_game_min': _cat('GAME'),
        'app_n_apps': float(n_apps),
        'structural_missing_app': False,
    }


def _android_empty_hour() -> dict:
    return {
        'screen_on_min': 0.0,
        'screen_n_sessions': 0,
        'screen_mean_session_min': np.nan,
        'screen_max_session_min': np.nan,
        'app_total_min': 0.0,
        'app_social_min': 0.0,
        'app_comm_min': 0.0,
        'app_entertainment_min': 0.0,
        'app_game_min': 0.0,
        'app_n_apps': 0.0,
        'structural_missing_app': False,
    }


# ---------------------------------------------------------------------------
# Per-participant pipeline
# ---------------------------------------------------------------------------

def process_participant_ios(pid_str: str) -> pd.DataFrame | None:
    """Full iOS ScreenOnTime pipeline for one participant."""
    raw = load_ios_screen_files(pid_str)
    if raw.empty:
        return None

    raw['epoch_capture'] = pd.to_numeric(raw['epoch_capture'], errors='coerce')
    raw = raw.dropna(subset=['epoch_capture'])

    sessions = ios_build_sessions(raw)
    if sessions.empty:
        return None

    # Hour range from raw events
    raw['hour_utc'] = epoch_to_hour_utc(raw['epoch_capture'])
    tz_map = tz_mode_per_hour(raw, 'hour_utc', 'timezone')
    first_hour = raw['hour_utc'].min()
    last_hour = raw['hour_utc'].max()
    hour_range = pd.date_range(start=first_hour, end=last_hour, freq='h', tz='UTC')

    rows = []
    for hour_utc in hour_range:
        hour_start_ms = float(hour_utc.value // 1_000_000)
        feats = ios_aggregate_hour(sessions, hour_start_ms)
        tz_str = tz_map.get(hour_utc, '+00:00')
        offset_min = parse_tz_minutes(tz_str)
        hour_local = hour_utc.tz_localize(None) + pd.Timedelta(minutes=offset_min)
        rows.append({'hour_utc': hour_utc, 'hour_local': hour_local,
                     'participant_id': pid_str, **feats})

    return pd.DataFrame(rows) if rows else None


def process_participant_android(pid_str: str) -> pd.DataFrame | None:
    """Full Android APPUSAGE pipeline for one participant."""
    raw = load_android_appusage_files(pid_str)
    if raw.empty:
        return None

    raw['epoch_usagewindow_start'] = pd.to_numeric(
        raw['epoch_usagewindow_start'], errors='coerce')
    raw['epoch_usagewindow_end'] = pd.to_numeric(
        raw['epoch_usagewindow_end'], errors='coerce')
    raw = raw.dropna(subset=['epoch_usagewindow_start', 'epoch_usagewindow_end'])

    # Filter artifacts: duration > MAX_APPUSAGE_HOURS
    max_ms = MAX_APPUSAGE_HOURS * 3_600_000
    raw = raw[(raw['epoch_usagewindow_end'] - raw['epoch_usagewindow_start']) <= max_ms]

    if raw.empty:
        return None

    # Assign app categories
    if 'id_app' in raw.columns:
        raw['app_cat'] = raw['id_app'].apply(categorize_app)

    raw['hour_utc'] = epoch_to_hour_utc(raw['epoch_usagewindow_start'])
    tz_map = tz_mode_per_hour(raw, 'hour_utc', 'timezone')
    first_hour = raw['hour_utc'].min()
    last_hour = raw['hour_utc'].max()
    hour_range = pd.date_range(start=first_hour, end=last_hour, freq='h', tz='UTC')

    rows = []
    for hour_utc in hour_range:
        hour_start_ms = float(hour_utc.value // 1_000_000)
        feats = android_aggregate_hour(raw, hour_start_ms)
        tz_str = tz_map.get(hour_utc, '+00:00')
        offset_min = parse_tz_minutes(tz_str)
        hour_local = hour_utc.tz_localize(None) + pd.Timedelta(minutes=offset_min)
        rows.append({'hour_utc': hour_utc, 'hour_local': hour_local,
                     'participant_id': pid_str, **feats})

    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        platform_map = load_platform_map(PLATFORM_FILE)
    except FileNotFoundError as exc:
        print(f"[screen] ERROR: {exc}")
        return

    # Discover PIDs
    ios_dir = DATA_ROOT / "ScreenOnTime"
    android_dir = DATA_ROOT / "APPUSAGE"

    ios_pids = {
        f.name.split('_')[2]
        for f in ios_dir.glob("ScreenOnTime_data_*.csv")
        if len(f.name.split('_')) > 2
    }
    android_pids_primary = {
        f.name.split('_')[0]
        for f in android_dir.glob("*_APPUSAGE_*.csv")
        if not f.name.startswith('APPUSAGE') and 'header' not in f.name.lower()
    }
    android_pids_alt = {
        f.name.split('_')[2]
        for f in android_dir.glob("APPUSAGE_data_*.csv")
        if len(f.name.split('_')) > 2
    }
    android_pids = android_pids_primary | android_pids_alt

    all_pids = sorted(ios_pids | android_pids)
    print(f"[screen] Found {len(all_pids)} participants "
          f"(iOS screen={len(ios_pids)}, Android appusage={len(android_pids)})")
    print(f"[screen] Withdraw list: {sorted(WITHDRAW_LIST)}")

    processed = skipped_withdraw = failed = 0

    for pid_str in tqdm(all_pids, desc="Processing screen/app", unit="participant"):
        try:
            pid_int = int(pid_str)
        except ValueError:
            continue

        if pid_int in WITHDRAW_LIST:
            skipped_withdraw += 1
            continue

        pid_str_norm = pid_from_int(pid_int)
        platform = platform_map.get(pid_str_norm, 'unknown')
        out_path = OUT_DIR / f"{pid_str_norm}_screen_hourly.parquet"

        try:
            if platform == 'ios':
                result = process_participant_ios(pid_str_norm)
            else:
                result = process_participant_android(pid_str_norm)
        except Exception as exc:
            warnings.warn(f"[screen] PID {pid_str_norm}: unhandled error — {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

        if result is None or result.empty:
            warnings.warn(f"[screen] PID {pid_str_norm}: produced no rows, skipping output.")
            failed += 1
            continue

        col_order = [
            'hour_utc', 'hour_local', 'participant_id',
            'screen_on_min', 'screen_n_sessions',
            'screen_mean_session_min', 'screen_max_session_min',
            'app_total_min', 'app_social_min', 'app_comm_min',
            'app_entertainment_min', 'app_game_min', 'app_n_apps',
            'structural_missing_app',
        ]
        # Keep only columns that exist
        col_order = [c for c in col_order if c in result.columns]
        result[col_order].to_parquet(out_path, index=False, compression='snappy')
        processed += 1

        n_hours = len(result)
        total_screen = result['screen_on_min'].sum()
        tqdm.write(
            f"  PID {pid_str_norm} [{platform}]: {n_hours} hours | "
            f"total screen {total_screen:.0f} min | "
            f"-> {out_path.name}"
        )

    print(
        f"\n[screen] Done. "
        f"Processed: {processed} | "
        f"Withdrawn: {skipped_withdraw} | "
        f"Failed/empty: {failed}"
    )


if __name__ == "__main__":
    main()
