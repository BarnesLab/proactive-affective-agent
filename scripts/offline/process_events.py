"""
Phase 1 Offline Processing — Raw Event Streams
================================================
Converts raw sensor CSV files into per-participant event-level Parquets
so the AI agent can query actual events (not just hourly aggregates).

Output: data/processed/events/{modality}/{pid}_{modality}_events.parquet

Modalities produced
-------------------
  screen_events   : lock/unlock events with local timestamps (iOS ScreenOnTime)
  app_events      : per-app-session events with duration (Android APPUSAGE)
  motion_events   : activity-type transitions with timestamps (MOTION harmonized)
  keyboard_events : per-typing-session events with word count (FleksyKeyInput)
  music_events    : per-track play events with title/artist (MUS)

Usage
-----
  python scripts/offline/process_events.py                  # all modalities
  python scripts/offline/process_events.py --modality screen motion
  python scripts/offline/process_events.py --dry-run        # count only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "bucs-data"
OUT_BASE = PROJECT_ROOT / "data" / "processed" / "events"

sys.path.insert(0, str(Path(__file__).parent))
from utils import WITHDRAW_LIST, to_local_time, pid_from_int, load_platform_map

PLATFORM_FILE = PROJECT_ROOT / "data" / "processed" / "hourly" / "participant_platform.parquet"

# App category lookup (package prefix → category label)
APP_CATEGORIES: dict[str, str] = {
    "com.facebook.katana": "SOCIAL",
    "com.instagram.android": "SOCIAL",
    "com.twitter.android": "SOCIAL",
    "com.snapchat.android": "SOCIAL",
    "com.tiktok": "SOCIAL",
    "com.reddit": "SOCIAL",
    "com.whatsapp": "COMMUNICATION",
    "com.google.android.apps.messaging": "COMMUNICATION",
    "com.facebook.orca": "COMMUNICATION",
    "org.telegram": "COMMUNICATION",
    "com.skype": "COMMUNICATION",
    "com.discord": "COMMUNICATION",
    "com.spotify.music": "MUSIC",
    "com.apple.Music": "MUSIC",
    "com.pandora": "MUSIC",
    "com.youtube": "ENTERTAINMENT",
    "com.netflix": "ENTERTAINMENT",
    "com.google.android.youtube": "ENTERTAINMENT",
    "com.amazon.avod": "ENTERTAINMENT",
    "com.google.android.gm": "PRODUCTIVITY",
    "com.microsoft.office": "PRODUCTIVITY",
    "com.google.android.apps.docs": "PRODUCTIVITY",
    "com.android.chrome": "BROWSER",
    "org.mozilla.firefox": "BROWSER",
    "com.apple.mobilesafari": "BROWSER",
}

def _app_category(pkg: str) -> str:
    for prefix, cat in APP_CATEGORIES.items():
        if str(pkg).startswith(prefix):
            return cat
    return "OTHER"


# ---------------------------------------------------------------------------
# Screen events (iOS ScreenOnTime)
# ---------------------------------------------------------------------------

def _pid_from_screen_filename(path: Path) -> str | None:
    """Extract PID from ScreenOnTime_data_012_<uuid>_... filename."""
    m = re.match(r"ScreenOnTime_data_(\d+)_", path.name)
    return pid_from_int(m.group(1)) if m else None


def process_screen_events(dry_run: bool = False) -> None:
    src = DATA_ROOT / "ScreenOnTime"
    out_dir = OUT_BASE / "screen_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.csv"))
    print(f"[screen_events] {len(files)} raw files")
    if dry_run:
        return

    # Group files by PID
    by_pid: dict[str, list[Path]] = {}
    for f in files:
        pid = _pid_from_screen_filename(f)
        if pid is None:
            continue
        if int(pid) in WITHDRAW_LIST:
            continue
        by_pid.setdefault(pid, []).append(f)

    for pid, pid_files in tqdm(by_pid.items(), desc="screen_events"):
        chunks = []
        for f in pid_files:
            try:
                df = pd.read_csv(f, usecols=["epoch_capture", "cat_state", "timezone"])
                df = df.dropna(subset=["epoch_capture", "cat_state"])
                chunks.append(df)
            except Exception:
                continue
        if not chunks:
            continue

        df = pd.concat(chunks, ignore_index=True)
        df["timestamp_local"] = to_local_time(df["epoch_capture"], df["timezone"])
        df = df.sort_values("timestamp_local").drop_duplicates("timestamp_local")
        df = df.rename(columns={"cat_state": "event"})
        df = df[["timestamp_local", "event"]].reset_index(drop=True)

        out = out_dir / f"{pid}_screen_events.parquet"
        df.to_parquet(out, index=False)

    print(f"[screen_events] Saved {len(by_pid)} participant files → {out_dir}")


# ---------------------------------------------------------------------------
# App usage events (Android APPUSAGE)
# ---------------------------------------------------------------------------

def process_app_events(dry_run: bool = False) -> None:
    src = DATA_ROOT / "APPUSAGE"
    out_dir = OUT_BASE / "app_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.csv"))
    print(f"[app_events] {len(files)} raw files")
    if dry_run:
        return

    # All APPUSAGE files have id_participant column; read all then group
    chunks = []
    for f in tqdm(files, desc="app_events reading"):
        try:
            df = pd.read_csv(f, usecols=[
                "id_app", "epoch_usagewindow_start", "epoch_usagewindow_end",
                "timezone", "n_foreground_ms", "id_participant"
            ])
            chunks.append(df)
        except Exception:
            continue

    if not chunks:
        print("[app_events] No data found")
        return

    all_df = pd.concat(chunks, ignore_index=True)
    all_df = all_df.dropna(subset=["epoch_usagewindow_start", "id_participant"])
    all_df["id_participant"] = all_df["id_participant"].apply(
        lambda x: pid_from_int(str(x).strip().lstrip("0") or "0")
    )

    for pid, grp in tqdm(all_df.groupby("id_participant"), desc="app_events saving"):
        if int(pid) in WITHDRAW_LIST:
            continue
        grp = grp.copy()
        grp["timestamp_local_start"] = to_local_time(
            grp["epoch_usagewindow_start"], grp["timezone"]
        )
        grp["timestamp_local_end"] = to_local_time(
            grp["epoch_usagewindow_end"], grp["timezone"]
        )
        grp["duration_min"] = (grp["n_foreground_ms"] / 60000).clip(0, 600).round(2)
        grp["app_category"] = grp["id_app"].apply(_app_category)
        grp = grp.rename(columns={"id_app": "app_package"})
        out_df = grp[[
            "timestamp_local_start", "timestamp_local_end",
            "app_package", "app_category", "duration_min"
        ]].sort_values("timestamp_local_start").reset_index(drop=True)

        out = out_dir / f"{pid}_app_events.parquet"
        out_df.to_parquet(out, index=False)

    print(f"[app_events] Done → {out_dir}")


# ---------------------------------------------------------------------------
# Motion events (MOTION harmonized, both platforms)
# ---------------------------------------------------------------------------

def process_motion_events(dry_run: bool = False) -> None:
    src = DATA_ROOT / "MOTION"
    out_dir = OUT_BASE / "motion_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.csv"))
    print(f"[motion_events] {len(files)} raw files")
    if dry_run:
        return

    chunks = []
    for f in tqdm(files, desc="motion_events reading"):
        try:
            df = pd.read_csv(f, usecols=["epoch", "timezone", "cat_activity", "id_participant"])
            chunks.append(df)
        except Exception:
            continue

    if not chunks:
        print("[motion_events] No data")
        return

    all_df = pd.concat(chunks, ignore_index=True)
    all_df = all_df.dropna(subset=["epoch", "cat_activity", "id_participant"])
    all_df["id_participant"] = all_df["id_participant"].apply(
        lambda x: pid_from_int(str(x).strip().lstrip("0") or "0")
    )

    for pid, grp in tqdm(all_df.groupby("id_participant"), desc="motion_events saving"):
        if int(pid) in WITHDRAW_LIST:
            continue
        grp = grp.copy()
        grp["timestamp_local"] = to_local_time(grp["epoch"], grp["timezone"])
        out_df = grp[["timestamp_local", "cat_activity"]].rename(
            columns={"cat_activity": "activity"}
        ).sort_values("timestamp_local").drop_duplicates("timestamp_local").reset_index(drop=True)

        out = out_dir / f"{pid}_motion_events.parquet"
        out_df.to_parquet(out, index=False)

    print(f"[motion_events] Done → {out_dir}")


# ---------------------------------------------------------------------------
# Keyboard events (FleksyKeyInput)
# ---------------------------------------------------------------------------

def _pid_from_keyinput_filename(path: Path) -> str | None:
    m = re.match(r"FleksyKeyInput_data_(\d+)_", path.name)
    return pid_from_int(m.group(1)) if m else None


def process_keyboard_events(dry_run: bool = False) -> None:
    src = DATA_ROOT / "FleksyKeyInput"
    out_dir = OUT_BASE / "keyboard_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.csv"))
    print(f"[keyboard_events] {len(files)} raw files")
    if dry_run:
        return

    by_pid: dict[str, list[Path]] = {}
    for f in files:
        pid = _pid_from_keyinput_filename(f)
        if pid is None:
            continue
        if int(pid) in WITHDRAW_LIST:
            continue
        by_pid.setdefault(pid, []).append(f)

    for pid, pid_files in tqdm(by_pid.items(), desc="keyboard_events"):
        chunks = []
        for f in pid_files:
            try:
                df = pd.read_csv(f, usecols=[
                    "epoch_session_start", "epoch_session_end",
                    "id_app", "cat_language", "strinput_text", "timezone"
                ])
                chunks.append(df)
            except Exception:
                continue
        if not chunks:
            continue

        df = pd.concat(chunks, ignore_index=True)
        df = df.dropna(subset=["epoch_session_start", "strinput_text"])
        df["timestamp_local_start"] = to_local_time(df["epoch_session_start"], df["timezone"])
        df["timestamp_local_end"] = to_local_time(df["epoch_session_end"], df["timezone"])
        df["words_typed"] = df["strinput_text"].astype(str).str.split().str.len().fillna(0).astype(int)
        df["text"] = df["strinput_text"].astype(str).str.strip()
        df = df.rename(columns={"id_app": "app_package", "cat_language": "language"})
        out_df = df[[
            "timestamp_local_start", "timestamp_local_end",
            "app_package", "language", "words_typed", "text"
        ]].sort_values("timestamp_local_start").reset_index(drop=True)

        out = out_dir / f"{pid}_keyboard_events.parquet"
        out_df.to_parquet(out, index=False)

    print(f"[keyboard_events] Done → {out_dir}")


# ---------------------------------------------------------------------------
# Music events (MUS)
# ---------------------------------------------------------------------------

def process_music_events(dry_run: bool = False) -> None:
    src = DATA_ROOT / "MUS"
    out_dir = OUT_BASE / "music_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.csv"))
    print(f"[music_events] {len(files)} raw files")
    if dry_run:
        return

    chunks = []
    for f in tqdm(files, desc="music_events reading"):
        try:
            df = pd.read_csv(f, usecols=[
                "epoch", "id_app", "id_title", "id_artist",
                "timezone", "is_advertisement", "id_participant"
            ])
            chunks.append(df)
        except Exception:
            continue

    if not chunks:
        print("[music_events] No data")
        return

    all_df = pd.concat(chunks, ignore_index=True)
    all_df = all_df.dropna(subset=["epoch", "id_participant"])
    all_df["id_participant"] = all_df["id_participant"].apply(
        lambda x: pid_from_int(str(x).strip().lstrip("0") or "0")
    )

    for pid, grp in tqdm(all_df.groupby("id_participant"), desc="music_events saving"):
        if int(pid) in WITHDRAW_LIST:
            continue
        grp = grp.copy()
        grp["timestamp_local"] = to_local_time(grp["epoch"], grp["timezone"])
        grp = grp.rename(columns={
            "id_app": "app", "id_title": "title",
            "id_artist": "artist", "is_advertisement": "is_ad"
        })
        out_df = grp[[
            "timestamp_local", "app", "title", "artist", "is_ad"
        ]].sort_values("timestamp_local").reset_index(drop=True)

        out = out_dir / f"{pid}_music_events.parquet"
        out_df.to_parquet(out, index=False)

    print(f"[music_events] Done → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODALITY_FUNCS = {
    "screen":   process_screen_events,
    "app":      process_app_events,
    "motion":   process_motion_events,
    "keyboard": process_keyboard_events,
    "music":    process_music_events,
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level Parquets from raw BUCS sensor CSVs")
    parser.add_argument(
        "--modality", nargs="+",
        choices=list(MODALITY_FUNCS.keys()),
        default=list(MODALITY_FUNCS.keys()),
        help="Which modalities to process (default: all)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Count files only, no output")
    args = parser.parse_args()

    for mod in args.modality:
        print(f"\n{'='*50}")
        MODALITY_FUNCS[mod](dry_run=args.dry_run)

    print("\nDone. Event Parquets at:", OUT_BASE)


if __name__ == "__main__":
    main()
