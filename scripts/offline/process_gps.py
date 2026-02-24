"""
Phase 1 Offline Processing — GPS
=================================
Input:  data/bucs-data/GPS/GPS_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
Also:   data/processed/hourly/participant_platform.parquet
        data/processed/hourly/home_locations.parquet

Output: data/processed/hourly/gps/{PID}_gps_hourly.parquet

Epoch units: MILLISECONDS (epoch_gpscapture).
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "bucs-data" / "GPS"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "gps"
PLATFORM_FILE = PROJECT_ROOT / "data" / "processed" / "hourly" / "participant_platform.parquet"
HOME_LOC_FILE = PROJECT_ROOT / "data" / "processed" / "hourly" / "home_locations.parquet"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WITHDRAW_LIST = {4, 20, 70, 94, 214, 253, 283, 494, 153}

MAX_HORIZONTAL_ACCURACY = 500.0   # metres
MAX_SPEED_KMH = 180.0             # artifact removal
MAX_GAP_SEC = 900.0               # 15 minutes — max gap for consecutive pair
AT_HOME_RADIUS_KM = 0.150         # 150 m
GRID_RES_DEG = 0.001              # ~111 m per cell


# ---------------------------------------------------------------------------
# Timezone helpers  (shared across processing scripts)
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
# Haversine (vectorized)
# ---------------------------------------------------------------------------

def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Vectorized haversine distance in kilometres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (
        np.radians(lat1),
        np.radians(lon1),
        np.radians(lat2),
        np.radians(lon2),
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Location entropy
# ---------------------------------------------------------------------------

def location_entropy(lat_arr: np.ndarray, lon_arr: np.ndarray) -> float:
    """
    Shannon entropy (bits) of discretized 0.001-degree grid cells.
    Returns 0.0 if fewer than 2 fixes.
    """
    if len(lat_arr) < 2:
        return 0.0
    cells = list(zip(
        np.floor(lat_arr / GRID_RES_DEG).astype(int),
        np.floor(lon_arr / GRID_RES_DEG).astype(int),
    ))
    unique, counts = np.unique(cells, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def find_gps_files(pid_str: str) -> list[Path]:
    """Return all GPS CSV files for a given zero-padded PID string."""
    pattern = str(RAW_DIR / f"GPS_data_{pid_str}_*.csv")
    return sorted(Path(p) for p in glob.glob(pattern))


def load_gps_csv(path: Path) -> pd.DataFrame:
    dtype = {
        'id_file': 'float32',
        'epoch_gpscapture': 'float64',
        'val_lat': 'float64',
        'val_lon': 'float64',
        'val_horizontal_accuracy': 'float64',
        'timezone': 'str',
    }
    try:
        return pd.read_csv(path, dtype=dtype)
    except Exception as exc:
        warnings.warn(f"[gps] Could not read {path.name}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Reference data loaders
# ---------------------------------------------------------------------------

def load_platform_map() -> dict[str, str]:
    """
    Returns {pid_str: 'android'|'ios'}.
    Tries to load participant_platform.parquet; falls back to empty dict.
    """
    if not PLATFORM_FILE.exists():
        warnings.warn(f"[gps] Platform file not found: {PLATFORM_FILE}. Will default to 'android' for all.")
        return {}
    pf = pd.read_parquet(PLATFORM_FILE)
    # Expect columns: participant_id (str or int), platform (str)
    pf['participant_id'] = pf['participant_id'].astype(str).str.zfill(3)
    if 'platform' not in pf.columns:
        warnings.warn("[gps] Platform file missing 'platform' column.")
        return {}
    return dict(zip(pf['participant_id'], pf['platform'].str.lower()))


def load_home_locations() -> dict[str, tuple[float, float]]:
    """
    Returns {pid_str: (home_lat, home_lon)}.
    """
    if not HOME_LOC_FILE.exists():
        warnings.warn(f"[gps] Home locations file not found: {HOME_LOC_FILE}. Home features will be NaN.")
        return {}
    hl = pd.read_parquet(HOME_LOC_FILE)
    hl['participant_id'] = hl['participant_id'].astype(str).str.zfill(3)
    result = {}
    for _, row in hl.iterrows():
        try:
            result[row['participant_id']] = (float(row['home_lat']), float(row['home_lon']))
        except (KeyError, ValueError):
            pass
    return result


# ---------------------------------------------------------------------------
# Per-participant processing
# ---------------------------------------------------------------------------

def compute_consecutive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a sorted GPS DataFrame (one row per fix), compute for each row
    (relative to the previous fix):
      - time_gap_sec: time since previous fix
      - dist_km: haversine distance from previous fix
      - speed_kmh: dist_km / time_gap_hours

    Only pairs with time_gap_sec <= MAX_GAP_SEC are considered.
    Returns the same df with new columns appended (NaN for the first row or
    for pairs exceeding the gap threshold).
    """
    epoch_sec = df['epoch_gpscapture'].values / 1000.0
    lat = df['val_lat'].values
    lon = df['val_lon'].values

    n = len(df)
    dist_km = np.full(n, np.nan)
    speed_kmh = np.full(n, np.nan)
    time_gap_sec = np.full(n, np.nan)

    if n < 2:
        df = df.copy()
        df['dist_km'] = dist_km
        df['speed_kmh'] = speed_kmh
        df['time_gap_sec'] = time_gap_sec
        return df

    gaps = np.diff(epoch_sec)           # shape (n-1,)
    valid_gap = gaps <= MAX_GAP_SEC

    d = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])  # shape (n-1,)
    time_hrs = gaps / 3600.0
    spd = np.where(time_hrs > 0, d / time_hrs, np.nan)

    # Write into positions [1..n]
    dist_km[1:] = np.where(valid_gap, d, np.nan)
    speed_kmh[1:] = np.where(valid_gap, spd, np.nan)
    time_gap_sec[1:] = gaps

    df = df.copy()
    df['dist_km'] = dist_km
    df['speed_kmh'] = speed_kmh
    df['time_gap_sec'] = time_gap_sec
    return df


def process_participant(
    pid_int: int,
    pid_str: str,
    platform: str,
    home_coords: tuple[float, float] | None,
) -> pd.DataFrame | None:
    """
    Load, filter, and aggregate GPS data for one participant.
    Returns hourly DataFrame or None.
    """
    files = find_gps_files(pid_str)
    if not files:
        warnings.warn(f"[gps] PID {pid_str}: no files found, skipping.")
        return None

    # ------------------------------------------------------------------
    # Load & concatenate
    # ------------------------------------------------------------------
    dfs = []
    for f in files:
        chunk = load_gps_csv(f)
        if not chunk.empty:
            dfs.append(chunk)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    required_cols = {'epoch_gpscapture', 'val_lat', 'val_lon', 'val_horizontal_accuracy', 'timezone'}
    if not required_cols.issubset(df.columns):
        warnings.warn(f"[gps] PID {pid_str}: missing expected columns, skipping.")
        return None

    # ------------------------------------------------------------------
    # Sort and dedup
    # ------------------------------------------------------------------
    df = df.dropna(subset=['epoch_gpscapture', 'val_lat', 'val_lon'])
    df = (
        df.sort_values('epoch_gpscapture')
        .drop_duplicates(subset=['epoch_gpscapture', 'val_lat', 'val_lon'])
        .reset_index(drop=True)
    )

    if df.empty:
        warnings.warn(f"[gps] PID {pid_str}: empty after dedup.")
        return None

    # ------------------------------------------------------------------
    # Filter bad fixes
    # ------------------------------------------------------------------
    bad_accuracy = df['val_horizontal_accuracy'] > MAX_HORIZONTAL_ACCURACY
    bad_coords = (df['val_lat'] == 0.0) & (df['val_lon'] == 0.0)
    df = df[~(bad_accuracy | bad_coords)].reset_index(drop=True)

    if df.empty:
        warnings.warn(f"[gps] PID {pid_str}: no valid fixes after quality filter.")
        return None

    # ------------------------------------------------------------------
    # UTC hour bucket
    # ------------------------------------------------------------------
    df['hour_utc'] = pd.to_datetime(
        df['epoch_gpscapture'] / 1000, unit='s', utc=True
    ).dt.floor('h')

    # ------------------------------------------------------------------
    # Consecutive pair metrics
    # ------------------------------------------------------------------
    df = compute_consecutive_pairs(df)

    # Drop speed artifacts AFTER computing (so we don't lose downstream pair info)
    speed_artifact = df['speed_kmh'] > MAX_SPEED_KMH
    df.loc[speed_artifact, ['dist_km', 'speed_kmh']] = np.nan

    # ------------------------------------------------------------------
    # Distance from home
    # ------------------------------------------------------------------
    if home_coords is not None:
        home_lat, home_lon = home_coords
        df['dist_from_home_km'] = haversine_km(
            df['val_lat'].values, df['val_lon'].values,
            home_lat, home_lon
        )
        df['at_home'] = df['dist_from_home_km'] <= AT_HOME_RADIUS_KM
    else:
        df['dist_from_home_km'] = np.nan
        df['at_home'] = False

    # ------------------------------------------------------------------
    # Aggregate per hour
    # ------------------------------------------------------------------
    def _agg_hour(grp: pd.DataFrame) -> pd.Series:
        n_caps = len(grp)
        dist_km_sum = grp['dist_km'].sum(min_count=1)   # NaN if no valid pairs
        speed_max = grp['speed_kmh'].max()

        # at-home: fraction × 60 → proxy minutes
        at_home_min = float(grp['at_home'].sum()) / n_caps * 60.0 if n_caps > 0 else np.nan

        # location entropy
        entropy = location_entropy(grp['val_lat'].values, grp['val_lon'].values)

        dist_home_max = grp['dist_from_home_km'].max()

        return pd.Series({
            'gps_n_captures': n_caps,
            'gps_distance_km': dist_km_sum,
            'gps_speed_max_kmh': speed_max,
            'gps_at_home_min': at_home_min,
            'gps_location_entropy': entropy,
            'gps_dist_from_home_max_km': dist_home_max,
        })

    hourly = df.groupby('hour_utc').apply(_agg_hour).reset_index()

    # ------------------------------------------------------------------
    # Missingness flags
    # ------------------------------------------------------------------
    hourly['structural_missing'] = False  # GPS exists on both platforms

    no_data = hourly['gps_n_captures'] == 0
    if platform == 'android':
        hourly['device_missing'] = no_data
        hourly['participant_missing'] = False
    else:
        # iOS: no data may be legitimately indoor — treat as participant_missing
        hourly['device_missing'] = False
        hourly['participant_missing'] = no_data

    # ------------------------------------------------------------------
    # Local time
    # ------------------------------------------------------------------
    tz_per_hour = (
        df.groupby('hour_utc')['timezone']
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )
    hourly = hourly.join(tz_per_hour.rename('_tz'), on='hour_utc')

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
        'gps_n_captures', 'gps_distance_km', 'gps_speed_max_kmh',
        'gps_at_home_min', 'gps_location_entropy', 'gps_dist_from_home_max_km',
        'device_missing', 'participant_missing',
    ]
    return hourly[col_order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load reference tables
    platform_map = load_platform_map()
    home_map = load_home_locations()

    # Discover all PIDs from filenames
    all_files = sorted(RAW_DIR.glob("GPS_data_*_*.csv"))
    pid_strs = sorted({f.name.split('_')[2] for f in all_files})

    print(f"[gps] Found {len(pid_strs)} participants in {RAW_DIR}")
    print(f"[gps] Withdraw list: {sorted(WITHDRAW_LIST)}")
    print(f"[gps] Platform info for {len(platform_map)} participants loaded.")
    print(f"[gps] Home locations for {len(home_map)} participants loaded.")

    skipped_withdraw = 0
    processed = 0
    failed = 0

    for pid_str in tqdm(pid_strs, desc="Processing GPS", unit="participant"):
        try:
            pid_int = int(pid_str)
        except ValueError:
            warnings.warn(f"[gps] Cannot parse PID from '{pid_str}', skipping.")
            continue

        if pid_int in WITHDRAW_LIST:
            skipped_withdraw += 1
            continue

        platform = platform_map.get(pid_str, 'android')   # default android if unknown
        home_coords = home_map.get(pid_str)                # None if unavailable

        out_path = OUT_DIR / f"{pid_str}_gps_hourly.parquet"

        try:
            result = process_participant(pid_int, pid_str, platform, home_coords)
        except Exception as exc:
            warnings.warn(f"[gps] PID {pid_str}: unhandled error — {exc}")
            failed += 1
            continue

        if result is None or result.empty:
            warnings.warn(f"[gps] PID {pid_str}: produced no rows, skipping output.")
            failed += 1
            continue

        result.to_parquet(out_path, index=False, compression='snappy')
        processed += 1

        n_hours = len(result)
        n_dev_miss = int(result['device_missing'].sum())
        n_part_miss = int(result['participant_missing'].sum())
        median_caps = result['gps_n_captures'].median()
        home_str = f"home={home_coords}" if home_coords else "no home"
        tqdm.write(
            f"  PID {pid_str} [{platform}|{home_str}]: "
            f"{n_hours} hours | "
            f"median {median_caps:.0f} fixes/hr | "
            f"device_miss={n_dev_miss} part_miss={n_part_miss} | "
            f"→ {out_path.name}"
        )

    print(
        f"\n[gps] Done. "
        f"Processed: {processed} | "
        f"Withdrawn: {skipped_withdraw} | "
        f"Failed/empty: {failed}"
    )


if __name__ == "__main__":
    main()
