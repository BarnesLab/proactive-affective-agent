"""
Phase 1 Offline Processing — Accelerometer
==========================================
Input:  data/bucs-data/Accelerometer/{PID}_ACCEL_complete_agg_1sec_{timestamp}.csv
Output: data/processed/hourly/accel/{PID}_accel_hourly.parquet

Epoch units: SECONDS (not ms, unlike other sensors).
One row per participant-hour; coverage-based device_missing flag.
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "bucs-data" / "Accelerometer"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "accel"
PLATFORM_FILE = PROJECT_ROOT / "data" / "processed" / "hourly" / "participant_platform.parquet"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WITHDRAW_LIST = {4, 20, 70, 94, 214, 253, 283, 494, 153}
CHUNK_SIZE_ROWS = 5_000_000  # ~500 MB safety threshold for chunked reading
MAG_CLIP_MAX = 5.0
DEVICE_MISSING_THRESHOLD = 0.1  # coverage < 10 % → device off


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

def parse_tz_minutes(tz_str: str) -> int:
    """Parse timezone offset string like '-04:00' to total minutes."""
    sign = 1 if str(tz_str).strip()[0] != '-' else -1
    parts = str(tz_str).strip().lstrip('+-').split(':')
    return sign * (int(parts[0]) * 60 + int(parts[1]))


def to_local_time(epoch_ms_series: pd.Series, timezone_series: pd.Series) -> pd.Series:
    """Convert epoch milliseconds + timezone offset string to local naive datetime."""
    utc = pd.to_datetime(epoch_ms_series / 1000, unit='s', utc=True)
    offset_minutes = timezone_series.map(parse_tz_minutes)
    local = utc + pd.to_timedelta(offset_minutes, unit='m')
    return local.dt.tz_localize(None)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def find_accel_files(pid_str: str) -> list[Path]:
    """Return all accelerometer CSV files for a given zero-padded PID string."""
    pattern = str(RAW_DIR / f"{pid_str}_ACCEL_complete_agg_1sec_*.csv")
    return sorted(Path(p) for p in glob.glob(pattern))


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def load_accel_csv(path: Path) -> pd.DataFrame:
    """Load a single accelerometer CSV, chunked if the file is large (>500 MB)."""
    dtype = {
        'epoch_1sec': 'int64',
        'timezone': 'str',
        'x_1sec': 'float32',
        'y_1sec': 'float32',
        'z_1sec': 'float32',
        'id_participant': 'str',
    }
    if _file_size_mb(path) > 500:
        chunks = []
        for chunk in pd.read_csv(path, dtype=dtype, chunksize=CHUNK_SIZE_ROWS):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    return pd.read_csv(path, dtype=dtype)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def compute_mag_diff(df: pd.DataFrame) -> pd.Series:
    """
    Compute |mag[t] - mag[t-1]| only for consecutive-second pairs.
    Non-consecutive gaps get 0.
    """
    epoch = df['epoch_1sec'].values
    mag = df['mag'].values

    mag_diff = np.zeros(len(df), dtype='float32')
    # gap between successive rows in seconds
    gaps = np.diff(epoch, prepend=epoch[0])  # prepend keeps length == len(df)
    consecutive_mask = gaps == 1
    mag_diff[consecutive_mask] = np.abs(
        np.diff(mag, prepend=mag[0])
    )[consecutive_mask]
    return pd.Series(mag_diff, index=df.index, dtype='float32')


def process_participant(pid: int, pid_str: str) -> pd.DataFrame | None:
    """
    Load, process, and aggregate accelerometer data for one participant.
    Returns a DataFrame of hourly features, or None if no data found.
    """
    files = find_accel_files(pid_str)
    if not files:
        warnings.warn(f"[accel] PID {pid_str}: no files found, skipping.")
        return None

    # ------------------------------------------------------------------
    # Load & concatenate all export files
    # ------------------------------------------------------------------
    dfs = []
    for f in files:
        try:
            dfs.append(load_accel_csv(f))
        except Exception as exc:
            warnings.warn(f"[accel] PID {pid_str}: failed to read {f.name} — {exc}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------
    required_cols = {'epoch_1sec', 'timezone', 'x_1sec', 'y_1sec', 'z_1sec'}
    if not required_cols.issubset(df.columns):
        warnings.warn(f"[accel] PID {pid_str}: missing expected columns, skipping.")
        return None

    df = df.dropna(subset=['epoch_1sec', 'x_1sec', 'y_1sec', 'z_1sec'])
    df = df.sort_values('epoch_1sec').drop_duplicates(subset=['epoch_1sec']).reset_index(drop=True)

    if df.empty:
        warnings.warn(f"[accel] PID {pid_str}: empty after dedup, skipping.")
        return None

    # ------------------------------------------------------------------
    # Magnitude + clipping
    # ------------------------------------------------------------------
    df['mag'] = np.sqrt(
        df['x_1sec'].astype('float64') ** 2
        + df['y_1sec'].astype('float64') ** 2
        + df['z_1sec'].astype('float64') ** 2
    ).clip(0, MAG_CLIP_MAX).astype('float32')

    # ------------------------------------------------------------------
    # Consecutive-second mag difference
    # ------------------------------------------------------------------
    df['mag_diff'] = compute_mag_diff(df)

    # ------------------------------------------------------------------
    # UTC hour bucket
    # ------------------------------------------------------------------
    df['hour_utc'] = pd.to_datetime(df['epoch_1sec'], unit='s', utc=True).dt.floor('h')

    # ------------------------------------------------------------------
    # Per-hour timezone: most common value for the hour
    # ------------------------------------------------------------------
    tz_per_hour = (
        df.groupby('hour_utc')['timezone']
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )

    # ------------------------------------------------------------------
    # Aggregate per hour
    # ------------------------------------------------------------------
    hourly = df.groupby('hour_utc').agg(
        accel_mean_mag=('mag', 'mean'),
        accel_std_mag=('mag', 'std'),
        accel_activity_counts=('mag_diff', 'sum'),
        accel_n_samples=('mag', 'count'),
    ).reset_index()

    hourly['accel_coverage_pct'] = hourly['accel_n_samples'] / 3600.0
    hourly['device_missing'] = hourly['accel_coverage_pct'] < DEVICE_MISSING_THRESHOLD

    # ------------------------------------------------------------------
    # Local time
    # ------------------------------------------------------------------
    hourly = hourly.join(tz_per_hour.rename('_tz'), on='hour_utc')

    # hour_utc is tz-aware; convert to ms for the to_local_time helper
    epoch_ms = hourly['hour_utc'].astype('int64') // 1_000_000
    hourly['hour_local'] = to_local_time(epoch_ms, hourly['_tz'])
    hourly = hourly.drop(columns=['_tz'])

    # ------------------------------------------------------------------
    # Participant ID
    # ------------------------------------------------------------------
    hourly['participant_id'] = pid_str

    # ------------------------------------------------------------------
    # Final column order
    # ------------------------------------------------------------------
    col_order = [
        'hour_utc', 'hour_local', 'participant_id',
        'accel_mean_mag', 'accel_std_mag', 'accel_activity_counts',
        'accel_n_samples', 'accel_coverage_pct', 'device_missing',
    ]
    return hourly[col_order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all PIDs from filenames present in the raw directory
    all_files = sorted(RAW_DIR.glob("*_ACCEL_complete_agg_1sec_*.csv"))
    pid_strs = sorted({f.name.split('_')[0] for f in all_files})

    print(f"[accel] Found {len(pid_strs)} participants in {RAW_DIR}")
    print(f"[accel] Withdraw list: {sorted(WITHDRAW_LIST)}")

    skipped_withdraw = 0
    processed = 0
    failed = 0

    for pid_str in tqdm(pid_strs, desc="Processing accelerometer", unit="participant"):
        try:
            pid_int = int(pid_str)
        except ValueError:
            warnings.warn(f"[accel] Cannot parse PID from '{pid_str}', skipping.")
            continue

        if pid_int in WITHDRAW_LIST:
            skipped_withdraw += 1
            continue

        out_path = OUT_DIR / f"{pid_str}_accel_hourly.parquet"

        try:
            result = process_participant(pid_int, pid_str)
        except Exception as exc:
            warnings.warn(f"[accel] PID {pid_str}: unhandled error — {exc}")
            failed += 1
            continue

        if result is None or result.empty:
            warnings.warn(f"[accel] PID {pid_str}: produced no rows, skipping output.")
            failed += 1
            continue

        result.to_parquet(out_path, index=False, compression='snappy')
        processed += 1

        n_hours = len(result)
        n_device_off = result['device_missing'].sum()
        cov_mean = result['accel_coverage_pct'].mean()
        tqdm.write(
            f"  PID {pid_str}: {n_hours} hours | "
            f"mean coverage {cov_mean:.1%} | "
            f"device_missing {n_device_off} hrs | "
            f"→ {out_path.name}"
        )

    print(
        f"\n[accel] Done. "
        f"Processed: {processed} | "
        f"Withdrawn: {skipped_withdraw} | "
        f"Failed/empty: {failed}"
    )


if __name__ == "__main__":
    main()
