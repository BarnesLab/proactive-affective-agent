"""
Phase 0 Script 1: Build Participant Roster

Scans data/bucs-data/ filenames across all 11 sensing modalities to construct
a participant roster with platform detection, enrollment dates, and available
modalities.

Output: data/processed/hourly/participant_platform.parquet
"""

import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "bucs-data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WITHDRAWN_PIDS = {4, 20, 70, 94, 214, 253, 283, 494, 153}

# Regex patterns for each modality.
# Groups: (pid, device_id_or_None, date_str)
# device_id group is None for modalities that don't carry it in the filename.
MODALITY_PATTERNS: dict[str, list[re.Pattern]] = {
    "gps": [
        # GPS_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
        re.compile(
            r"^GPS_data_(\d{3})_([A-Fa-f0-9\-]+)_(\d{4}-\d{2}-\d{2})_\S+\.csv$"
        ),
    ],
    "motion_activity": [
        # MotionActivity_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
        re.compile(
            r"^MotionActivity_data_(\d{3})_([A-Fa-f0-9\-]+)_(\d{4}-\d{2}-\d{2})_\S+\.csv$"
        ),
    ],
    "activity_transition": [
        # ActivityTransition_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
        re.compile(
            r"^ActivityTransition_data_(\d{3})_([A-Fa-f0-9\-]+)_(\d{4}-\d{2}-\d{2})_\S+\.csv$"
        ),
    ],
    "sleep": [
        # AndroidSleep_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
        re.compile(
            r"^AndroidSleep_data_(\d{3})_([A-Fa-f0-9\-]+)_(\d{4}-\d{2}-\d{2})_\S+\.csv$"
        ),
    ],
    "screen": [
        # ScreenOnTime_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
        re.compile(
            r"^ScreenOnTime_data_(\d{3})_([A-Fa-f0-9\-]+)_(\d{4}-\d{2}-\d{2})_\S+\.csv$"
        ),
        # ScreenOnTime_{PID}_* (alternate)
        re.compile(
            r"^ScreenOnTime_(\d{3})_.*_(\d{4}-\d{2}-\d{2}).*\.csv$"
        ),
    ],
    "accel": [
        # {PID}_ACCEL_complete_agg_1sec_{date}.csv
        re.compile(
            r"^(\d{3})_ACCEL_complete_agg_1sec_(\d{4}-\d{2}-\d{2}).*\.csv$"
        ),
    ],
    "appusage": [
        # {PID}_APPUSAGE_usageLog_{date}.csv
        re.compile(
            r"^(\d{3})_APPUSAGE_usageLog_(\d{4}-\d{2}-\d{2}).*\.csv$"
        ),
    ],
    "keyinput": [
        # FleksyKeyInput_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
        re.compile(
            r"^FleksyKeyInput_data_(\d{3})_([A-Fa-f0-9\-]+)_(\d{4}-\d{2}-\d{2})_\S+\.csv$"
        ),
        # {PID}_FleksyKeyInput_* (alternate)
        re.compile(
            r"^(\d{3})_FleksyKeyInput_.*_(\d{4}-\d{2}-\d{2}).*\.csv$"
        ),
    ],
    "mus": [
        # {PID}_MUS_complete_{date}.csv
        re.compile(
            r"^(\d{3})_MUS_complete_(\d{4}-\d{2}-\d{2}).*\.csv$"
        ),
    ],
    "motion": [
        # {PID}_MOTION_harmonized_{date}.csv  (actual files use 'harmonized')
        re.compile(
            r"^(\d{3})_MOTION_(?:complete|harmonized)_(\d{4}-\d{2}-\d{2}).*\.csv$"
        ),
        # MOTION_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv (alternate)
        re.compile(
            r"^MOTION_data_(\d{3})_([A-Fa-f0-9\-]+)_(\d{4}-\d{2}-\d{2})_\S+\.csv$"
        ),
    ],
    "light": [
        # LIGHT_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
        re.compile(
            r"^LIGHT_data_(\d{3})_([A-Fa-f0-9\-]+)_(\d{4}-\d{2}-\d{2})_\S+\.csv$"
        ),
    ],
}

# Map modality key -> directory name in bucs-data
MODALITY_DIRS: dict[str, str] = {
    "gps": "GPS",
    "motion_activity": "MotionActivity",
    "activity_transition": "ActivityTransition",
    "sleep": "AndroidSleep",
    "screen": "ScreenOnTime",
    "accel": "Accelerometer",
    "appusage": "APPUSAGE",
    "keyinput": "FleksyKeyInput",
    "mus": "MUS",
    "motion": "MOTION",
    "light": "LIGHT",
}

# UUID pattern (iOS device ID)
IOS_UUID_RE = re.compile(
    r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$",
    re.IGNORECASE,
)
# Hex lowercase pattern (Android device ID)
ANDROID_HEX_RE = re.compile(r"^[0-9a-f]{16}$")


def detect_platform(device_id: Optional[str]) -> str:
    """Return 'ios', 'android', or 'unknown' from a device ID string."""
    if device_id is None:
        return "unknown"
    if IOS_UUID_RE.match(device_id):
        return "ios"
    if ANDROID_HEX_RE.match(device_id):
        return "android"
    return "unknown"


def parse_date(date_str: str) -> Optional[date]:
    """Parse a YYYY-MM-DD string into a date object, or None on failure."""
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        return None


def try_match(filename: str, patterns: list) -> tuple:
    """
    Try each pattern in order.  Returns (pid, device_id, date_str) where
    device_id may be None for patterns without that group.
    """
    for pat in patterns:
        m = pat.match(filename)
        if m:
            groups = m.groups()
            if len(groups) == 3:
                # pid, device_id, date
                return groups[0], groups[1], groups[2]
            elif len(groups) == 2:
                # pid, date  (no device_id)
                return groups[0], None, groups[1]
    return None, None, None


def scan_modality(
    modality_key: str,
    participant_data: dict,
) -> int:
    """
    Scan one modality directory and update participant_data in-place.
    Returns number of files matched.
    """
    dir_name = MODALITY_DIRS[modality_key]
    modality_dir = DATA_ROOT / dir_name
    if not modality_dir.exists():
        print(f"  [WARN] Directory not found, skipping: {modality_dir}")
        return 0

    patterns = MODALITY_PATTERNS[modality_key]
    matched = 0

    for csv_file in modality_dir.glob("*.csv"):
        fname = csv_file.name
        pid_str, device_id, date_str = try_match(fname, patterns)

        if pid_str is None:
            continue  # filename didn't match any known pattern

        pid_int = int(pid_str)
        if pid_int in WITHDRAWN_PIDS:
            continue

        # Normalise PID to zero-padded 3-char string
        pid = f"{pid_int:03d}"

        # Platform detection
        platform = detect_platform(device_id)

        # Date
        file_date = parse_date(date_str) if date_str else None

        # Update participant record
        rec = participant_data[pid]
        rec["modalities"].add(modality_key)

        if platform != "unknown":
            rec["platforms"].add(platform)

        if file_date is not None:
            if rec["first_date"] is None or file_date < rec["first_date"]:
                rec["first_date"] = file_date
            if rec["last_date"] is None or file_date > rec["last_date"]:
                rec["last_date"] = file_date

        matched += 1

    return matched


def build_roster() -> pd.DataFrame:
    """Main routine: scan all modalities and build the participant DataFrame."""
    print("=" * 60)
    print("Phase 0 — Script 1: Build Participant Roster")
    print(f"Data root : {DATA_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    # participant_data[pid] = {modalities, platforms, first_date, last_date}
    participant_data: dict[str, dict] = defaultdict(
        lambda: {
            "modalities": set(),
            "platforms": set(),
            "first_date": None,
            "last_date": None,
        }
    )

    print("\nScanning modality directories:")
    for modality_key in MODALITY_PATTERNS:
        n = scan_modality(modality_key, participant_data)
        print(f"  {MODALITY_DIRS[modality_key]:<22} {n:>5} files matched")

    if not participant_data:
        print("\n[ERROR] No participant data found. Check DATA_ROOT path.")
        sys.exit(1)

    print(f"\nBuilding DataFrame for {len(participant_data)} participants...")

    rows = []
    for pid, rec in sorted(participant_data.items()):
        mods = rec["modalities"]
        platforms = rec["platforms"]

        # Resolve platform: if only one unique non-unknown platform, use it
        if len(platforms) == 1:
            platform = next(iter(platforms))
        elif len(platforms) > 1:
            # Conflict — pick the most specific one; log a warning
            platform = sorted(platforms)[0]
            print(
                f"  [WARN] PID {pid} has multiple platforms detected {platforms}; "
                f"using '{platform}'"
            )
        else:
            platform = "unknown"

        first_date = rec["first_date"]
        last_date = rec["last_date"]
        if first_date and last_date:
            n_study_days = (last_date - first_date).days + 1
        else:
            n_study_days = 0

        rows.append(
            {
                "participant_id": pid,
                "platform": platform,
                "has_accel": "accel" in mods,
                "has_gps": "gps" in mods,
                "has_motion": "motion" in mods,
                "has_screen": "screen" in mods,
                "has_appusage": "appusage" in mods,
                "has_keyinput": "keyinput" in mods,
                "has_mus": "mus" in mods,
                "has_light": "light" in mods,
                "has_sleep": "sleep" in mods,
                "first_date": first_date,
                "last_date": last_date,
                "n_study_days": n_study_days,
            }
        )

    df = pd.DataFrame(rows)
    return df


def save_and_summarize(df: pd.DataFrame) -> None:
    """Save the roster to parquet and print a summary table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "participant_platform.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved roster to: {out_path}")

    print("\n--- Summary: Participants by Platform ---")
    platform_counts = df["platform"].value_counts()
    for platform, count in platform_counts.items():
        print(f"  {platform:<12} {count:>4} participants")
    print(f"  {'TOTAL':<12} {len(df):>4} participants")

    print("\n--- Modality Coverage ---")
    bool_cols = [c for c in df.columns if c.startswith("has_")]
    for col in bool_cols:
        label = col.replace("has_", "")
        n = df[col].sum()
        pct = 100 * n / len(df)
        print(f"  {label:<15} {n:>4} / {len(df)}  ({pct:5.1f}%)")

    print("\n--- Study Duration ---")
    print(f"  Median study days: {df['n_study_days'].median():.0f}")
    print(f"  Min study days   : {df['n_study_days'].min()}")
    print(f"  Max study days   : {df['n_study_days'].max()}")

    print("\n--- Per-Platform Modality Breakdown ---")
    for platform in sorted(df["platform"].unique()):
        sub = df[df["platform"] == platform]
        print(f"\n  Platform: {platform} (n={len(sub)})")
        for col in bool_cols:
            label = col.replace("has_", "")
            n = sub[col].sum()
            pct = 100 * n / len(sub)
            print(f"    {label:<15} {n:>3} / {len(sub)}  ({pct:5.1f}%)")


def main() -> None:
    df = build_roster()
    save_and_summarize(df)
    print("\nPhase 0 Script 1 complete.")


if __name__ == "__main__":
    main()
