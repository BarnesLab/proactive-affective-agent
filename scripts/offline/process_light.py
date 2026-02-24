"""
Phase 1 Offline Processing — Ambient Light Sensor (LIGHT)
==========================================================
Android-only modality.  iOS participants receive structural_missing = True.

Input
-----
  data/bucs-data/LIGHT/   (pattern: LIGHT_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv)

Output
------
  data/processed/hourly/light/{pid}_light_hourly.parquet

Per-hour columns
----------------
  hour_utc, hour_local, participant_id,
  light_n_captures, light_mean_lux, light_max_lux, light_min_lux,
  structural_missing
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
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "light"
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

def load_light_files(pid_str: str) -> pd.DataFrame:
    """Load all LIGHT CSV data files for a PID."""
    light_dir = DATA_ROOT / "LIGHT"
    # Primary: LIGHT_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
    files = sorted(light_dir.glob(f"LIGHT_data_{pid_str}_*.csv"))
    files = [f for f in files if 'header' not in f.name.lower()]

    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'epoch_lightcapture' not in df.columns:
                continue
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"[light] PID {pid_str}: failed to read {f.name} — {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-participant pipeline
# ---------------------------------------------------------------------------

def process_participant_android(pid_int: int, pid_str: str) -> pd.DataFrame | None:
    """Aggregate LIGHT captures into hourly features for an Android participant."""
    raw = load_light_files(pid_str)
    if raw.empty:
        return None

    raw['epoch_lightcapture'] = pd.to_numeric(raw['epoch_lightcapture'], errors='coerce')
    raw['val_lightlevel'] = pd.to_numeric(raw['val_lightlevel'], errors='coerce')
    raw = raw.dropna(subset=['epoch_lightcapture', 'val_lightlevel'])

    if raw.empty:
        return None

    raw['hour_utc'] = epoch_to_hour_utc(raw['epoch_lightcapture'])
    tz_map = tz_mode_per_hour(raw, 'hour_utc', 'timezone')

    hourly = raw.groupby('hour_utc')['val_lightlevel'].agg(
        light_n_captures='count',
        light_mean_lux='mean',
        light_max_lux='max',
        light_min_lux='min',
    ).reset_index()

    hourly['structural_missing'] = False
    hourly = hourly.join(tz_map.rename('_tz'), on='hour_utc')

    def _local(row):
        tz_str = row['_tz'] if pd.notna(row['_tz']) else '+00:00'
        offset_min = parse_tz_minutes(tz_str)
        return row['hour_utc'].tz_localize(None) + pd.Timedelta(minutes=offset_min)

    hourly['hour_local'] = hourly.apply(_local, axis=1)
    hourly['participant_id'] = pid_str
    hourly = hourly.drop(columns=['_tz'])

    col_order = [
        'hour_utc', 'hour_local', 'participant_id',
        'light_n_captures', 'light_mean_lux', 'light_max_lux', 'light_min_lux',
        'structural_missing',
    ]
    return hourly[col_order]


def make_ios_structural_missing_frame(pid_str: str, platform_map: dict) -> pd.DataFrame:
    """
    For iOS participants, return a single-row DataFrame indicating that light
    data is structurally missing (sensor not available on iOS).

    We don't know the date range without cross-referencing another modality, so
    we return an empty indicator frame — the pipeline caller handles this.
    """
    # Structural missing for iOS is indicated by returning an empty frame;
    # the caller writes nothing if result is None (iOS skip).
    # Alternatively, downstream feature assembly will gap-fill with NaN and
    # mark structural_missing = True when light parquet is absent for iOS.
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        platform_map = load_platform_map(PLATFORM_FILE)
    except FileNotFoundError as exc:
        print(f"[light] ERROR: {exc}")
        return

    light_dir = DATA_ROOT / "LIGHT"
    # Discover Android PIDs from filenames
    android_pids = sorted({
        f.name.split('_')[2]
        for f in light_dir.glob("LIGHT_data_*.csv")
        if len(f.name.split('_')) > 2 and 'header' not in f.name.lower()
    })

    # Also iterate over all known participants to mark iOS structural missing
    all_known_pids = sorted(set(platform_map.keys()))

    print(f"[light] Found {len(android_pids)} Android participants with light data")
    print(f"[light] Total known participants: {len(all_known_pids)}")
    print(f"[light] Withdraw list: {sorted(WITHDRAW_LIST)}")

    processed = skipped_withdraw = failed = skipped_ios = 0

    # Process Android participants
    for pid_str in tqdm(android_pids, desc="Processing light (Android)", unit="participant"):
        try:
            pid_int = int(pid_str)
        except ValueError:
            continue

        if pid_int in WITHDRAW_LIST:
            skipped_withdraw += 1
            continue

        pid_str_norm = pid_from_int(pid_int)
        platform = platform_map.get(pid_str_norm, 'unknown')

        if platform == 'ios':
            # LIGHT_data files with UUID device IDs shouldn't exist but guard anyway
            skipped_ios += 1
            continue

        out_path = OUT_DIR / f"{pid_str_norm}_light_hourly.parquet"

        try:
            result = process_participant_android(pid_int, pid_str_norm)
        except Exception as exc:
            warnings.warn(f"[light] PID {pid_str_norm}: unhandled error — {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

        if result is None or result.empty:
            warnings.warn(f"[light] PID {pid_str_norm}: produced no rows, skipping output.")
            failed += 1
            continue

        result.to_parquet(out_path, index=False, compression='snappy')
        processed += 1

        n_hours = len(result)
        mean_lux = result['light_mean_lux'].mean()
        tqdm.write(
            f"  PID {pid_str_norm} [android]: {n_hours} hours | "
            f"mean_lux={mean_lux:.1f} | "
            f"-> {out_path.name}"
        )

    # Report iOS participants as structurally missing (no output file written;
    # downstream feature assembly uses absence of the parquet as the signal)
    ios_count = sum(
        1 for pid, plat in platform_map.items()
        if plat == 'ios' and int(pid) not in WITHDRAW_LIST
    )
    print(
        f"\n[light] Done. "
        f"Processed (Android): {processed} | "
        f"iOS participants (structural_missing, no file written): {ios_count} | "
        f"Withdrawn: {skipped_withdraw} | "
        f"Failed/empty: {failed}"
    )
    print(
        "[light] NOTE: iOS participants have no ambient light sensor. "
        "Downstream scripts should treat missing light parquet for iOS as "
        "structural_missing=True."
    )


if __name__ == "__main__":
    main()
