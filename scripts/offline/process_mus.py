"""
Phase 1 Offline Processing — Music Streaming (MUS)
===================================================
Simple hourly aggregation of music track events.

Input
-----
  data/bucs-data/MUS/   (pattern: {PID}_MUS_complete_{date}.csv)

Output
------
  data/processed/hourly/mus/{pid}_mus_hourly.parquet

Per-hour columns
----------------
  hour_utc, hour_local, participant_id,
  mus_n_tracks, mus_n_ads, mus_is_listening
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
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "mus"
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
# File loading
# ---------------------------------------------------------------------------

def load_mus_files(pid_str: str) -> pd.DataFrame:
    """Load all MUS complete CSV files for a PID."""
    mus_dir = DATA_ROOT / "MUS"
    # Primary: {PID}_MUS_complete_{date}.csv
    files = sorted(mus_dir.glob(f"{pid_str}_MUS_complete*.csv"))
    # Also try harmonized suffix
    files += sorted(mus_dir.glob(f"{pid_str}_MUS_harmonized*.csv"))
    files = [f for f in files if 'header' not in f.name.lower()]

    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Require epoch column
            if 'epoch' not in df.columns:
                continue
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"[mus] PID {pid_str}: failed to read {f.name} — {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-participant pipeline
# ---------------------------------------------------------------------------

def process_participant(pid_int: int, pid_str: str) -> pd.DataFrame | None:
    """Aggregate MUS events into hourly feature rows."""
    raw = load_mus_files(pid_str)
    if raw.empty:
        return None

    # Coerce epoch
    raw['epoch'] = pd.to_numeric(raw['epoch'], errors='coerce')
    raw = raw.dropna(subset=['epoch'])

    if raw.empty:
        return None

    # Coerce boolean columns
    for col in ['is_advertisement', 'is_missing']:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0).astype(int)
        else:
            raw[col] = 0

    # Drop missing-data sentinel rows (no actual track event)
    raw = raw[raw['is_missing'] == 0]
    if raw.empty:
        return None

    # Assign UTC hour
    raw['hour_utc'] = epoch_to_hour_utc(raw['epoch'])
    tz_map = tz_mode_per_hour(raw, 'hour_utc', 'timezone')

    first_hour = raw['hour_utc'].min()
    last_hour = raw['hour_utc'].max()
    hour_range = pd.date_range(start=first_hour, end=last_hour, freq='h', tz='UTC')

    rows = []
    for hour_utc in hour_range:
        sub = raw[raw['hour_utc'] == hour_utc]

        if sub.empty:
            n_tracks = 0
            n_ads = 0
        else:
            is_ad = sub['is_advertisement'].astype(bool)
            n_ads = int(is_ad.sum())
            n_tracks = int((~is_ad).sum())

        mus_is_listening = (n_tracks + n_ads) > 0

        tz_str = tz_map.get(hour_utc, '+00:00')
        offset_min = parse_tz_minutes(tz_str)
        hour_local = hour_utc.tz_localize(None) + pd.Timedelta(minutes=offset_min)

        rows.append({
            'hour_utc': hour_utc,
            'hour_local': hour_local,
            'participant_id': pid_str,
            'mus_n_tracks': n_tracks,
            'mus_n_ads': n_ads,
            'mus_is_listening': mus_is_listening,
        })

    if not rows:
        return None

    result = pd.DataFrame(rows)
    col_order = [
        'hour_utc', 'hour_local', 'participant_id',
        'mus_n_tracks', 'mus_n_ads', 'mus_is_listening',
    ]
    return result[col_order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        platform_map = load_platform_map(PLATFORM_FILE)
    except FileNotFoundError as exc:
        print(f"[mus] ERROR: {exc}")
        return

    mus_dir = DATA_ROOT / "MUS"
    # Discover PIDs from MUS_complete files
    all_pids = sorted({
        f.name.split('_')[0]
        for f in mus_dir.glob("*_MUS_*.csv")
        if 'header' not in f.name.lower()
        and not f.name.startswith('MUS')
    })

    print(f"[mus] Found {len(all_pids)} participants in {mus_dir}")
    print(f"[mus] Withdraw list: {sorted(WITHDRAW_LIST)}")

    processed = skipped_withdraw = failed = 0

    for pid_str in tqdm(all_pids, desc="Processing music", unit="participant"):
        try:
            pid_int = int(pid_str)
        except ValueError:
            continue

        if pid_int in WITHDRAW_LIST:
            skipped_withdraw += 1
            continue

        pid_str_norm = pid_from_int(pid_int)
        out_path = OUT_DIR / f"{pid_str_norm}_mus_hourly.parquet"

        try:
            result = process_participant(pid_int, pid_str_norm)
        except Exception as exc:
            warnings.warn(f"[mus] PID {pid_str_norm}: unhandled error — {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

        if result is None or result.empty:
            warnings.warn(f"[mus] PID {pid_str_norm}: produced no rows, skipping output.")
            failed += 1
            continue

        result.to_parquet(out_path, index=False, compression='snappy')
        processed += 1

        n_hours = len(result)
        listening_hours = result['mus_is_listening'].sum()
        total_tracks = result['mus_n_tracks'].sum()
        tqdm.write(
            f"  PID {pid_str_norm}: {n_hours} hours | "
            f"listening_hours={listening_hours} | "
            f"total_tracks={total_tracks} | "
            f"-> {out_path.name}"
        )

    print(
        f"\n[mus] Done. "
        f"Processed: {processed} | "
        f"Withdrawn: {skipped_withdraw} | "
        f"Failed/empty: {failed}"
    )


if __name__ == "__main__":
    main()
