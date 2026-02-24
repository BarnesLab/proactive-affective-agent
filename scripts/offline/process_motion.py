"""
Phase 1 Offline Processing — Motion / Physical Activity
========================================================
Handles both iOS (MotionActivity) and Android (ActivityTransition / MOTION
pre-harmonized) into a unified hourly output.

Input directories
-----------------
  iOS   : data/bucs-data/MotionActivity/
  Android: data/bucs-data/ActivityTransition/
  Pre-harmonized (Android): data/bucs-data/MOTION/

Output
------
  data/processed/hourly/motion/{pid}_motion_hourly.parquet

Per-hour columns
----------------
  hour_utc, hour_local, participant_id,
  motion_stationary_min, motion_walking_min, motion_automotive_min,
  motion_running_min, motion_cycling_min, motion_active_min,
  motion_coverage_pct, device_missing
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
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "motion"
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
    to_local_time,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE_MISSING_THRESHOLD = 0.05   # coverage < 5 % → device off

# iOS confidence priority when categories tie
IOS_TIE_PRIORITY = ['automotive', 'walking', 'running', 'cycling', 'stationary', 'unknown']

# Activity categories we track
ACTIVITY_CATS = ['stationary', 'walking', 'running', 'automotive', 'cycling']


# ---------------------------------------------------------------------------
# iOS — MotionActivity processing
# ---------------------------------------------------------------------------

def load_ios_files(pid_str: str) -> pd.DataFrame:
    """Load all MotionActivity CSV files for a PID, return concatenated DataFrame."""
    pattern = f"MotionActivity_data_{pid_str}_*.csv"
    files = sorted((DATA_ROOT / "MotionActivity").glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            warnings.warn(f"[motion/ios] PID {pid_str}: failed to read {f.name} — {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ios_to_cat_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert iOS CoreMotion boolean flags to a cat_activity string column.

    Disambiguation rule:
      - If is_stationary==1 AND is_automotive==1 → set is_stationary=0
      - Whichever is_X==1 with priority: automotive > walking > running >
        cycling > stationary > unknown
      - All zeros → 'unclassified'
    Returns a copy with added columns: cat_activity, epoch (ms).
    """
    df = df.copy()

    # Rename epoch column
    df = df.rename(columns={'epoch_motioncapture': 'epoch'})

    # Coerce boolean flags to int
    for col in ['is_stationary', 'is_walking', 'is_running', 'is_automotive',
                'is_cycling', 'is_unknown']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # CoreMotion disambiguation
    mask_both = (df['is_stationary'] == 1) & (df['is_automotive'] == 1)
    df.loc[mask_both, 'is_stationary'] = 0

    # Map flags to category string
    flag_to_cat = [
        ('is_automotive', 'automotive'),
        ('is_walking', 'walking'),
        ('is_running', 'running'),
        ('is_cycling', 'cycling'),
        ('is_stationary', 'stationary'),
        ('is_unknown', 'unknown'),
    ]

    def _assign_cat(row):
        for col, name in flag_to_cat:
            if row.get(col, 0) == 1:
                return name
        return 'unclassified'

    df['cat_activity'] = df.apply(_assign_cat, axis=1)
    return df[['epoch', 'timezone', 'cat_activity']].dropna(subset=['epoch'])


# ---------------------------------------------------------------------------
# Android — ActivityTransition processing
# ---------------------------------------------------------------------------

def load_android_files(pid_str: str) -> pd.DataFrame:
    """Load all ActivityTransition CSV files for a PID."""
    pattern = f"ActivityTransition_data_{pid_str}_*.csv"
    files = sorted((DATA_ROOT / "ActivityTransition").glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            warnings.warn(f"[motion/android] PID {pid_str}: failed to read {f.name} — {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def android_transitions_to_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ActivityTransition start/end events to (epoch, cat_activity) rows
    suitable for interval reconstruction.
    """
    df = df.copy()
    df = df.rename(columns={'epoch_motioncapture': 'epoch'})
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df = df.dropna(subset=['epoch', 'cat_activity']).sort_values('epoch').reset_index(drop=True)
    # Keep only start events (is_end == 0) as activity start timestamps;
    # the interval ends when the next start event arrives.
    starts = df[df['is_end'] == 0][['epoch', 'timezone', 'cat_activity']].copy()
    return starts


# ---------------------------------------------------------------------------
# Pre-harmonized MOTION files (Android preferred)
# ---------------------------------------------------------------------------

def load_motion_harmonized(pid_str: str) -> pd.DataFrame:
    """
    Load pre-harmonized MOTION files for a PID.
    Returns DataFrame with columns: epoch (ms), timezone, cat_activity.
    """
    pattern = f"{pid_str}_MOTION_*.csv"
    files = sorted((DATA_ROOT / "MOTION").glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            warnings.warn(f"[motion/harmonized] PID {pid_str}: failed to read {f.name} — {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Column mapping: harmonized files use 'epoch' already
    if 'epoch' not in df.columns and 'epoch_motioncapture' in df.columns:
        df = df.rename(columns={'epoch_motioncapture': 'epoch'})
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df = df.dropna(subset=['epoch', 'cat_activity'])
    return df[['epoch', 'timezone', 'cat_activity']]


# ---------------------------------------------------------------------------
# Interval reconstruction and hourly aggregation
# ---------------------------------------------------------------------------

def events_to_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """
    Convert sorted event rows (epoch ms, cat_activity) to intervals
    (start_ms, end_ms, cat_activity).

    Each event defines the start of an activity; the next event's epoch
    defines the end.  The last event has no following event, so we assign
    a 60-second duration as a fallback.
    """
    events = events.sort_values('epoch').reset_index(drop=True)
    starts = events['epoch'].values
    cats = events['cat_activity'].values

    # End of each interval = start of the next event
    ends = np.empty(len(starts), dtype='float64')
    ends[:-1] = starts[1:]
    ends[-1] = starts[-1] + 60_000  # fallback: 60 s for the last event

    # Drop zero or negative durations
    mask = ends > starts
    return pd.DataFrame({
        'start_ms': starts[mask],
        'end_ms': ends[mask],
        'cat_activity': cats[mask],
    })


def clip_intervals_to_hour(intervals: pd.DataFrame,
                           hour_start_ms: float,
                           hour_end_ms: float) -> pd.DataFrame:
    """
    Clip a set of (start_ms, end_ms) intervals to [hour_start_ms, hour_end_ms).
    Returns a filtered DataFrame with clipped start/end.
    """
    df = intervals.copy()
    df = df[(df['end_ms'] > hour_start_ms) & (df['start_ms'] < hour_end_ms)]
    df = df.copy()
    df['start_ms'] = df['start_ms'].clip(lower=hour_start_ms)
    df['end_ms'] = df['end_ms'].clip(upper=hour_end_ms)
    df['duration_s'] = (df['end_ms'] - df['start_ms']) / 1000.0
    return df[df['duration_s'] > 0]


def aggregate_hour(intervals: pd.DataFrame, hour_start_ms: float) -> dict:
    """
    Aggregate clipped intervals for one hour into feature dict.
    """
    clipped = clip_intervals_to_hour(intervals, hour_start_ms, hour_start_ms + 3_600_000)
    if clipped.empty:
        return _empty_hour_features()

    per_cat = clipped.groupby('cat_activity')['duration_s'].sum()

    def _cat_min(name: str) -> float:
        return per_cat.get(name, 0.0) / 60.0

    # Exclude 'unclassified' from coverage
    classified_cats = [c for c in ACTIVITY_CATS]
    classified_s = clipped[clipped['cat_activity'].isin(classified_cats)]['duration_s'].sum()
    coverage = min(classified_s / 3600.0, 1.0)

    stat_min = _cat_min('stationary')
    walk_min = _cat_min('walking')
    run_min = _cat_min('running')
    auto_min = _cat_min('automotive')
    cycl_min = _cat_min('cycling')

    return {
        'motion_stationary_min': stat_min,
        'motion_walking_min': walk_min,
        'motion_automotive_min': auto_min,
        'motion_running_min': run_min,
        'motion_cycling_min': cycl_min,
        'motion_active_min': walk_min + run_min + cycl_min,
        'motion_coverage_pct': coverage,
        'device_missing': coverage < DEVICE_MISSING_THRESHOLD,
    }


def _empty_hour_features() -> dict:
    return {
        'motion_stationary_min': np.nan,
        'motion_walking_min': np.nan,
        'motion_automotive_min': np.nan,
        'motion_running_min': np.nan,
        'motion_cycling_min': np.nan,
        'motion_active_min': np.nan,
        'motion_coverage_pct': 0.0,
        'device_missing': True,
    }


# ---------------------------------------------------------------------------
# Per-participant pipeline
# ---------------------------------------------------------------------------

def process_participant(pid_int: int, pid_str: str, platform: str) -> pd.DataFrame | None:
    """
    Full motion processing pipeline for one participant.
    Returns hourly DataFrame or None if no usable data.
    """
    # ------------------------------------------------------------------
    # Load raw events based on platform
    # ------------------------------------------------------------------
    events = pd.DataFrame()

    if platform == 'ios':
        raw = load_ios_files(pid_str)
        if not raw.empty:
            events = ios_to_cat_activity(raw)
    else:
        # Android: prefer pre-harmonized MOTION; fall back to ActivityTransition
        harmonized = load_motion_harmonized(pid_str)
        if not harmonized.empty:
            events = harmonized[['epoch', 'timezone', 'cat_activity']].copy()
        else:
            raw = load_android_files(pid_str)
            if not raw.empty:
                events = android_transitions_to_events(raw)

    if events.empty:
        warnings.warn(f"[motion] PID {pid_str}: no data found for platform={platform}")
        return None

    # ------------------------------------------------------------------
    # Ensure numeric epoch
    # ------------------------------------------------------------------
    events['epoch'] = pd.to_numeric(events['epoch'], errors='coerce')
    events = events.dropna(subset=['epoch']).sort_values('epoch').reset_index(drop=True)

    if events.empty:
        return None

    # ------------------------------------------------------------------
    # Build intervals
    # ------------------------------------------------------------------
    intervals = events_to_intervals(events)

    if intervals.empty:
        return None

    # ------------------------------------------------------------------
    # Per-hour timezone: most common tz string from events in that hour
    # ------------------------------------------------------------------
    events['hour_utc'] = epoch_to_hour_utc(events['epoch'])
    tz_map = tz_mode_per_hour(events, 'hour_utc', 'timezone')

    # ------------------------------------------------------------------
    # Build hour range from data span
    # ------------------------------------------------------------------
    first_hour = events['hour_utc'].min()
    last_hour = events['hour_utc'].max()
    hour_range = pd.date_range(start=first_hour, end=last_hour, freq='h', tz='UTC')

    # ------------------------------------------------------------------
    # Aggregate per hour
    # ------------------------------------------------------------------
    rows = []
    for hour_utc in hour_range:
        hour_start_ms = hour_utc.value // 1_000_000
        hour_end_ms = hour_start_ms + 3_600_000

        feats = aggregate_hour(intervals, float(hour_start_ms))
        tz_str = tz_map.get(hour_utc, '+00:00')
        offset_min = parse_tz_minutes(tz_str)
        hour_local = hour_utc.tz_localize(None) + pd.Timedelta(minutes=offset_min)

        row = {
            'hour_utc': hour_utc,
            'hour_local': hour_local,
            'participant_id': pid_str,
            **feats,
        }
        rows.append(row)

    if not rows:
        return None

    result = pd.DataFrame(rows)
    col_order = [
        'hour_utc', 'hour_local', 'participant_id',
        'motion_stationary_min', 'motion_walking_min', 'motion_automotive_min',
        'motion_running_min', 'motion_cycling_min', 'motion_active_min',
        'motion_coverage_pct', 'device_missing',
    ]
    return result[col_order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load platform map
    try:
        platform_map = load_platform_map(PLATFORM_FILE)
    except FileNotFoundError as exc:
        print(f"[motion] ERROR: {exc}")
        return

    # Discover all PIDs from both MotionActivity (iOS) and ActivityTransition (Android)
    ios_pids = {
        f.name.split('_')[2]
        for f in (DATA_ROOT / "MotionActivity").glob("MotionActivity_data_*.csv")
        if len(f.name.split('_')) > 2
    }
    android_pids = {
        f.name.split('_')[2]
        for f in (DATA_ROOT / "ActivityTransition").glob("ActivityTransition_data_*.csv")
        if len(f.name.split('_')) > 2
    }
    # Also include PIDs from MOTION harmonized files
    motion_pids = {
        f.name.split('_')[0]
        for f in (DATA_ROOT / "MOTION").glob("*_MOTION_*.csv")
    }

    all_pids = sorted(ios_pids | android_pids | motion_pids)
    print(f"[motion] Found {len(all_pids)} participants total "
          f"(iOS={len(ios_pids)}, Android={len(android_pids)}, "
          f"MOTION={len(motion_pids)})")
    print(f"[motion] Withdraw list: {sorted(WITHDRAW_LIST)}")

    processed = skipped_withdraw = failed = 0

    for pid_str in tqdm(all_pids, desc="Processing motion", unit="participant"):
        try:
            pid_int = int(pid_str)
        except ValueError:
            warnings.warn(f"[motion] Cannot parse PID from '{pid_str}', skipping.")
            continue

        if pid_int in WITHDRAW_LIST:
            skipped_withdraw += 1
            continue

        pid_str_norm = pid_from_int(pid_int)
        platform = platform_map.get(pid_str_norm, 'unknown')

        out_path = OUT_DIR / f"{pid_str_norm}_motion_hourly.parquet"

        try:
            result = process_participant(pid_int, pid_str_norm, platform)
        except Exception as exc:
            warnings.warn(f"[motion] PID {pid_str_norm}: unhandled error — {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

        if result is None or result.empty:
            warnings.warn(f"[motion] PID {pid_str_norm}: produced no rows, skipping output.")
            failed += 1
            continue

        result.to_parquet(out_path, index=False, compression='snappy')
        processed += 1

        n_hours = len(result)
        n_missing = result['device_missing'].sum()
        cov_mean = result['motion_coverage_pct'].mean()
        tqdm.write(
            f"  PID {pid_str_norm} [{platform}]: {n_hours} hours | "
            f"mean coverage {cov_mean:.1%} | "
            f"device_missing {n_missing} hrs | "
            f"-> {out_path.name}"
        )

    print(
        f"\n[motion] Done. "
        f"Processed: {processed} | "
        f"Withdrawn: {skipped_withdraw} | "
        f"Failed/empty: {failed}"
    )


if __name__ == "__main__":
    main()
