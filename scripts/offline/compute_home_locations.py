"""
Phase 0 Script 2: Compute Home Locations

For each participant, loads all nighttime GPS data and runs DBSCAN clustering
to estimate their home location (centroid of the largest cluster).

Input : data/bucs-data/GPS/
        data/processed/hourly/participant_platform.parquet
Output: data/processed/hourly/home_locations.parquet
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
GPS_DIR = PROJECT_ROOT / "data" / "bucs-data" / "GPS"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "hourly"
ROSTER_PATH = PROCESSED_DIR / "participant_platform.parquet"
OUTPUT_PATH = PROCESSED_DIR / "home_locations.parquet"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# DBSCAN parameters
DBSCAN_EPS = 0.00135      # ~150 m in degrees latitude/longitude
DBSCAN_MIN_SAMPLES = 4

# Nighttime window (local time hours, inclusive of start, exclusive of end)
NIGHT_START_H = 22        # 10 pm
NIGHT_END_H = 6           # 6 am  (next day)

# GPS quality thresholds
MAX_HORIZONTAL_ACCURACY_M = 500.0

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Return distance in metres between (lat1, lon1) and (lat2, lon2)."""
    R = 6_371_000.0  # Earth radius in metres
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def parse_tz_offset_minutes(tz_str: str) -> int:
    """
    Parse a timezone offset string like '-04:00' or '+05:30' into total minutes.
    Returns integer minutes offset (positive = east of UTC).
    """
    tz_str = tz_str.strip()
    sign = 1 if tz_str[0] == "+" else -1
    body = tz_str.lstrip("+-")
    parts = body.split(":")
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 else 0
    return sign * (hours * 60 + minutes)


def epoch_ms_to_local(
    epoch_ms_series: pd.Series,
    timezone_series: pd.Series,
) -> pd.Series:
    """
    Convert epoch-milliseconds to timezone-aware local datetime (tz-naive output).

    Each row may have a different UTC offset stored in the 'timezone' column
    (e.g. '-04:00').  We compute the offset per unique timezone value to avoid
    a Python-level loop over every row.
    """
    epoch_sec = epoch_ms_series / 1000.0
    utc_dt = pd.to_datetime(epoch_sec, unit="s", utc=True)

    # Build a minutes-offset series using the unique timezone strings
    unique_tzs = timezone_series.dropna().unique()
    tz_offset_map: dict[str, int] = {}
    for tz_str in unique_tzs:
        try:
            tz_offset_map[tz_str] = parse_tz_offset_minutes(str(tz_str))
        except (ValueError, IndexError):
            tz_offset_map[tz_str] = 0  # fall back to UTC

    minutes_offset = timezone_series.map(tz_offset_map).fillna(0).astype(int)
    local_dt = utc_dt + pd.to_timedelta(minutes_offset, unit="m")
    # Strip timezone info so downstream hour comparisons are straightforward
    return local_dt.dt.tz_localize(None)


def is_nighttime(local_dt: pd.Series) -> pd.Series:
    """
    Return a boolean Series marking nighttime hours (22:00–05:59 local time).
    That is, hour >= NIGHT_START_H OR hour < NIGHT_END_H.
    """
    h = local_dt.dt.hour
    return (h >= NIGHT_START_H) | (h < NIGHT_END_H)


def load_participant_gps(pid: str) -> Optional[pd.DataFrame]:
    """
    Load and concatenate all GPS CSV files for a given participant.
    Returns None if no files are found.
    """
    pattern = f"GPS_data_{pid}_*.csv"
    files = sorted(GPS_DIR.glob(pattern))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(
                f,
                usecols=["epoch_gpscapture", "val_lat", "val_lon",
                          "val_horizontal_accuracy", "timezone"],
                dtype={
                    "epoch_gpscapture": "float64",
                    "val_lat": "float64",
                    "val_lon": "float64",
                    "val_horizontal_accuracy": "float64",
                    "timezone": "object",
                },
                low_memory=False,
            )
            dfs.append(df)
        except Exception as exc:
            print(f"\n  [WARN] Could not read {f.name}: {exc}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def filter_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality and nighttime filters to a raw GPS DataFrame."""
    # Drop rows with missing coordinates or epoch
    df = df.dropna(subset=["epoch_gpscapture", "val_lat", "val_lon"])

    # Drop zero-coordinates (invalid fix)
    zero_mask = (df["val_lat"] == 0.0) & (df["val_lon"] == 0.0)
    df = df[~zero_mask]

    # Drop poor accuracy
    df = df[df["val_horizontal_accuracy"] <= MAX_HORIZONTAL_ACCURACY_M]

    if df.empty:
        return df

    # Convert epoch to local time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = df.copy()
        df["local_dt"] = epoch_ms_to_local(df["epoch_gpscapture"], df["timezone"])

    # Keep only nighttime points
    df = df[is_nighttime(df["local_dt"])]

    return df


def compute_home_cluster(df: pd.DataFrame) -> dict:
    """
    Run DBSCAN on filtered GPS points and return a result dict with keys:
    home_lat, home_lon, home_radius_m, n_gps_points_used, n_clusters_found.
    """
    coords = df[["val_lat", "val_lon"]].values
    n_points = len(coords)

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, algorithm="ball_tree",
                metric="haversine").fit(np.radians(coords))
    labels = db.labels_

    # Labels == -1 are noise
    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        return {
            "home_lat": float("nan"),
            "home_lon": float("nan"),
            "home_radius_m": float("nan"),
            "n_gps_points_used": n_points,
            "n_clusters_found": 0,
        }

    unique_clusters = np.unique(valid_labels)
    n_clusters = len(unique_clusters)

    # Find largest cluster
    cluster_sizes = {lbl: np.sum(labels == lbl) for lbl in unique_clusters}
    home_label = max(cluster_sizes, key=cluster_sizes.__getitem__)
    home_mask = labels == home_label

    home_points = coords[home_mask]
    home_lat = float(np.mean(home_points[:, 0]))
    home_lon = float(np.mean(home_points[:, 1]))

    # Compute mean distance from centroid to cluster points
    distances = haversine_m(
        home_points[:, 0], home_points[:, 1],
        home_lat, home_lon,
    )
    home_radius_m = float(np.mean(distances))

    return {
        "home_lat": home_lat,
        "home_lon": home_lon,
        "home_radius_m": home_radius_m,
        "n_gps_points_used": n_points,
        "n_clusters_found": n_clusters,
    }


def load_participant_ids() -> list[str]:
    """
    Load participant IDs from the roster parquet, or fall back to scanning
    the GPS directory directly if the roster does not exist yet.
    """
    if ROSTER_PATH.exists():
        roster = pd.read_parquet(ROSTER_PATH, columns=["participant_id", "has_gps"])
        gps_pids = roster.loc[roster["has_gps"], "participant_id"].tolist()
        print(f"Loaded roster: {len(gps_pids)} participants with GPS data.")
        return sorted(gps_pids)
    else:
        print(
            f"[WARN] Roster not found at {ROSTER_PATH}. "
            "Scanning GPS directory directly."
        )
        pids: set[str] = set()
        pid_re = __import__("re").compile(r"^GPS_data_(\d{3})_")
        for f in GPS_DIR.glob("GPS_data_*.csv"):
            m = pid_re.match(f.name)
            if m:
                pids.add(m.group(1))
        return sorted(pids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 0 — Script 2: Compute Home Locations")
    print(f"GPS dir   : {GPS_DIR}")
    print(f"Output    : {OUTPUT_PATH}")
    print("=" * 60)

    if not GPS_DIR.exists():
        print(f"[ERROR] GPS directory not found: {GPS_DIR}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    participant_ids = load_participant_ids()
    print(f"\nProcessing {len(participant_ids)} participants...\n")

    results = []
    no_files = 0
    no_night = 0
    no_cluster = 0

    for pid in tqdm(participant_ids, desc="Home location", unit="participant"):
        raw_df = load_participant_gps(pid)
        if raw_df is None:
            no_files += 1
            continue

        filtered = filter_gps(raw_df)
        if filtered.empty:
            no_night += 1
            result = {
                "participant_id": pid,
                "home_lat": float("nan"),
                "home_lon": float("nan"),
                "home_radius_m": float("nan"),
                "n_gps_points_used": 0,
                "n_clusters_found": 0,
            }
        else:
            cluster_result = compute_home_cluster(filtered)
            if cluster_result["n_clusters_found"] == 0:
                no_cluster += 1
            result = {"participant_id": pid, **cluster_result}

        results.append(result)

    if not results:
        print("\n[ERROR] No results produced. Check GPS data path.")
        return

    out_df = pd.DataFrame(results)

    # Cast types explicitly
    out_df = out_df.astype(
        {
            "participant_id": str,
            "home_lat": float,
            "home_lon": float,
            "home_radius_m": float,
            "n_gps_points_used": int,
            "n_clusters_found": int,
        }
    )

    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved home locations to: {OUTPUT_PATH}")

    # Summary
    n_total = len(results)
    n_success = out_df["n_clusters_found"].gt(0).sum()
    print("\n--- Summary ---")
    print(f"  Participants processed   : {n_total}")
    print(f"  Home found (>=1 cluster) : {n_success}")
    print(f"  No GPS files             : {no_files}")
    print(f"  No nighttime points      : {no_night}")
    print(f"  No clusters found        : {no_cluster}")
    if n_success > 0:
        valid = out_df[out_df["n_clusters_found"] > 0]
        print(f"\n  Median home_radius_m     : {valid['home_radius_m'].median():.1f} m")
        print(f"  Median n_gps_points_used : {valid['n_gps_points_used'].median():.0f}")
        print(f"  Median n_clusters_found  : {valid['n_clusters_found'].median():.0f}")

    print("\nPhase 0 Script 2 complete.")


if __name__ == "__main__":
    main()
