"""Build filtered daily behavioral summaries from hourly sensing data.

Reads raw hourly parquets, applies noise filtering per docs/data-filtering-plan.md,
and outputs per-user daily parquets with pre-generated narrative summaries.

Output: data/processed/filtered/{pid}_daily_filtered.parquet

Usage:
    PYTHONPATH=. python3 scripts/build_filtered_data.py
    PYTHONPATH=. python3 scripts/build_filtered_data.py --users 71,119  # specific users
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HOURLY_DIR = Path("data/processed/hourly")
OUTPUT_DIR = Path("data/processed/filtered")

# ── Platform detection ──────────────────────────────────────────────


def detect_platform(pid: str) -> str:
    """Detect platform from data availability. Android users have app + light data."""
    screen_f = HOURLY_DIR / "screen" / f"{pid}_screen_hourly.parquet"
    light_f = HOURLY_DIR / "light" / f"{pid}_light_hourly.parquet"

    has_app = False
    if screen_f.exists():
        df = pd.read_parquet(screen_f, columns=["app_total_min"])
        has_app = df["app_total_min"].notna().any()

    has_light = light_f.exists()
    return "android" if (has_app or has_light) else "ios"


# ── Load hourly data ────────────────────────────────────────────────


def load_hourly(pid: str, modality: str) -> pd.DataFrame | None:
    """Load hourly parquet for a user+modality. Returns None if missing.

    Also deduplicates *ghost rows* — rows where ``hour_local == hour_utc``
    (i.e. timezone offset was erroneously set to 0).  When duplicate
    ``hour_local`` values exist, the ghost row is dropped and the correctly
    offset row is kept.
    """
    f = HOURLY_DIR / modality / f"{pid}_{modality}_hourly.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)

    # ── Deduplicate ghost rows ──────────────────────────────────────
    if "hour_local" in df.columns and "hour_utc" in df.columns:
        utc_naive = df["hour_utc"].dt.tz_localize(None)
        df["_is_ghost"] = df["hour_local"] == utc_naive
        # Sort so non-ghost (real) rows come first, then drop dups
        df = df.sort_values("_is_ghost").drop_duplicates(
            subset=["hour_local"], keep="first"
        )
        df = df.drop(columns=["_is_ghost"]).reset_index(drop=True)

    # Ensure date column
    if "date_local" not in df.columns:
        for col in df.columns:
            if "date" in col.lower():
                df["date_local"] = pd.to_datetime(df[col]).dt.date
                break
    if "date_local" not in df.columns and "hour_local" in df.columns:
        df["date_local"] = pd.to_datetime(df["hour_local"]).dt.date
    return df


# ── Per-modality daily aggregation ──────────────────────────────────


def agg_motion_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate motion hourly → daily with filtering."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Drop low-coverage hours (coverage_pct is 0-1 fraction)
    if "motion_coverage_pct" in df.columns:
        df = df[df["motion_coverage_pct"] >= 0.50].copy()

    keep_cols = ["motion_stationary_min", "motion_walking_min", "motion_automotive_min"]
    available = [c for c in keep_cols if c in df.columns]
    if not available:
        return pd.DataFrame()

    agg = df.groupby("date_local")[available].sum().reset_index()

    # Walking episodes: count hours with walking > 1 min
    if "motion_walking_min" in df.columns:
        episodes = df[df["motion_walking_min"] > 1].groupby("date_local").size().reset_index(name="motion_n_walking_episodes")
        agg = agg.merge(episodes, on="date_local", how="left")
        agg["motion_n_walking_episodes"] = agg["motion_n_walking_episodes"].fillna(0).astype(int)

    # Cap motion total to 1440 min/day (24h), scale proportionally
    MAX_MOTION_MIN = 1440
    motion_sum_cols = [c for c in ["motion_stationary_min", "motion_walking_min", "motion_automotive_min"] if c in agg.columns]
    if motion_sum_cols:
        agg["_motion_total"] = agg[motion_sum_cols].sum(axis=1)
        over = agg["_motion_total"] > MAX_MOTION_MIN
        if over.any():
            scale = MAX_MOTION_MIN / agg.loc[over, "_motion_total"]
            for col in motion_sum_cols:
                agg.loc[over, col] = (agg.loc[over, col] * scale).round(1)
        agg.drop(columns=["_motion_total"], inplace=True)

    # Primary activity
    activity_cols = [c for c in ["motion_stationary_min", "motion_walking_min", "motion_automotive_min"] if c in agg.columns]
    if activity_cols:
        agg["motion_primary_activity"] = agg[activity_cols].idxmax(axis=1).str.replace("motion_", "").str.replace("_min", "")

    # Tracked hours (cap at 24)
    tracked = df.groupby("date_local").size().reset_index(name="motion_tracked_hours")
    tracked["motion_tracked_hours"] = tracked["motion_tracked_hours"].clip(upper=24)
    agg = agg.merge(tracked, on="date_local", how="left")

    return agg


def agg_screen_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate screen hourly → daily."""
    if df is None or df.empty:
        return pd.DataFrame()

    agg_dict = {}
    if "screen_on_min" in df.columns:
        agg_dict["screen_on_min"] = "sum"
    if "screen_n_sessions" in df.columns:
        agg_dict["screen_n_sessions"] = "sum"
    if not agg_dict:
        return pd.DataFrame()

    agg = df.groupby("date_local").agg(agg_dict).reset_index()
    agg.rename(columns={"screen_on_min": "screen_total_min", "screen_n_sessions": "screen_n_sessions"}, inplace=True)

    # Cap screen time to 1440 min/day (24h)
    if "screen_total_min" in agg.columns:
        agg["screen_total_min"] = agg["screen_total_min"].clip(upper=1440)

    return agg


def agg_app_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate app usage hourly → daily (Android only).

    App times are capped at 1440 min/day (24h) since parallel foreground
    counting causes raw sums to exceed physical time.  Sub-categories are
    scaled proportionally when the total is capped.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    app_cols_sum = ["app_total_min", "app_social_min", "app_comm_min", "app_entertainment_min"]
    available_sum = [c for c in app_cols_sum if c in df.columns and df[c].notna().any()]
    if not available_sum:
        return pd.DataFrame()

    agg = df.groupby("date_local")[available_sum].sum().reset_index()

    # Cap app_total_min to 1440 and scale sub-categories proportionally
    MAX_APP_MIN = 1440
    if "app_total_min" in agg.columns:
        over = agg["app_total_min"] > MAX_APP_MIN
        if over.any():
            scale = MAX_APP_MIN / agg.loc[over, "app_total_min"]
            agg.loc[over, "app_total_min"] = MAX_APP_MIN
            for sub in ["app_social_min", "app_comm_min", "app_entertainment_min"]:
                if sub in agg.columns:
                    agg.loc[over, sub] = (agg.loc[over, sub] * scale).round(1)

    # Mean apps per active hour
    if "app_n_apps" in df.columns:
        apps_mean = df[df["app_n_apps"] > 0].groupby("date_local")["app_n_apps"].mean().reset_index()
        apps_mean.rename(columns={"app_n_apps": "app_n_apps_mean"}, inplace=True)
        agg = agg.merge(apps_mean, on="date_local", how="left")

    return agg


def agg_keyboard_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate keyboard hourly → daily. Only keep hours with actual typing."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter to hours with actual sessions
    if "key_n_sessions" in df.columns:
        df = df[df["key_n_sessions"] > 0].copy()

    if df.empty:
        return pd.DataFrame()

    agg_dict = {}
    if "key_n_sessions" in df.columns:
        agg_dict["key_n_sessions"] = "sum"
    if "key_chars_typed" in df.columns:
        agg_dict["key_chars_typed"] = "sum"
    if not agg_dict:
        return pd.DataFrame()

    agg = df.groupby("date_local").agg(agg_dict).reset_index()
    agg.rename(columns={"key_n_sessions": "keyboard_n_sessions", "key_chars_typed": "keyboard_total_chars"}, inplace=True)
    return agg


def agg_light_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate light hourly → daily categorical."""
    if df is None or df.empty:
        return pd.DataFrame()

    if "light_mean_lux" not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=["light_mean_lux"])
    if valid.empty:
        return pd.DataFrame()

    daily_lux = valid.groupby("date_local")["light_mean_lux"].mean().reset_index()

    def categorize(lux):
        if lux < 10:
            return "dark"
        elif lux < 500:
            return "indoor"
        else:
            return "outdoor"

    daily_lux["light_category"] = daily_lux["light_mean_lux"].apply(categorize)
    daily_lux["light_mean_lux_raw"] = daily_lux["light_mean_lux"]
    daily_lux.drop(columns=["light_mean_lux"], inplace=True)

    return daily_lux


def agg_music_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate music hourly → daily. Only include if actually listening."""
    if df is None or df.empty:
        return pd.DataFrame()

    if "mus_is_listening" not in df.columns:
        return pd.DataFrame()

    listening = df[df["mus_is_listening"] == True]  # noqa: E712
    if listening.empty:
        return pd.DataFrame()

    agg_dict = {"mus_is_listening": "any"}
    if "mus_n_tracks" in df.columns:
        agg_dict["mus_n_tracks"] = "sum"

    agg = listening.groupby("date_local").agg(agg_dict).reset_index()
    agg.rename(columns={"mus_is_listening": "music_listening", "mus_n_tracks": "music_n_tracks"}, inplace=True)
    return agg


# ── Narrative generation ────────────────────────────────────────────


def _safe_int(val, default: int = 0) -> int:
    """Safely convert a value to int, handling NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return int(val)


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float, handling NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def _fmt_duration(minutes: float) -> str:
    """Format minutes as human-readable duration."""
    if pd.isna(minutes) or minutes <= 0:
        return "0m"
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h > 0 and m > 0:
        return f"{h}h {m}m"
    elif h > 0:
        return f"{h}h"
    else:
        return f"{m}m"


def generate_narrative(row: pd.Series, platform: str) -> str:
    """Generate a human-readable behavioral narrative for one day."""
    parts = []

    # Motion
    stat = _safe_float(row.get("motion_stationary_min"))
    walk = _safe_float(row.get("motion_walking_min"))
    auto = _safe_float(row.get("motion_automotive_min"))
    episodes = _safe_int(row.get("motion_n_walking_episodes"))
    tracked = _safe_int(row.get("motion_tracked_hours"))

    if tracked > 0:
        motion_parts = []
        primary = row.get("motion_primary_activity", "stationary")
        if stat > 0:
            motion_parts.append(f"stationary {_fmt_duration(stat)}")
        if walk > 0:
            ep_str = f" in {episodes} episode{'s' if episodes != 1 else ''}" if episodes > 0 else ""
            motion_parts.append(f"walked {_fmt_duration(walk)}{ep_str}")
        if auto > 0:
            motion_parts.append(f"drove {_fmt_duration(auto)}")
        if motion_parts:
            parts.append(f"[Motion] {', '.join(motion_parts)} ({tracked}h tracked)")

    # Screen
    scr_min = _safe_float(row.get("screen_total_min"))
    scr_n = _safe_int(row.get("screen_n_sessions"))
    if scr_n > 0:
        parts.append(f"[Screen] {scr_n} opens, {_fmt_duration(scr_min)} total")

    # Apps (Android only)
    if platform == "android":
        app_total = _safe_float(row.get("app_total_min"))
        if app_total > 0:
            app_parts = [f"{_fmt_duration(app_total)} total"]
            social = _safe_float(row.get("app_social_min"))
            comm = _safe_float(row.get("app_comm_min"))
            ent = _safe_float(row.get("app_entertainment_min"))
            sub = []
            if social > 0:
                sub.append(f"social {_fmt_duration(social)}")
            if comm > 0:
                sub.append(f"comm {_fmt_duration(comm)}")
            if ent > 0:
                sub.append(f"entertainment {_fmt_duration(ent)}")
            if sub:
                app_parts.append(f"({', '.join(sub)})")
            n_apps = row.get("app_n_apps_mean", 0)
            if n_apps and not pd.isna(n_apps) and n_apps > 0:
                app_parts.append(f"avg {n_apps:.0f} apps/hr")
            parts.append(f"[Apps] {' '.join(app_parts)}")

    # Keyboard
    kb_sessions = _safe_int(row.get("keyboard_n_sessions"))
    kb_chars = _safe_int(row.get("keyboard_total_chars"))
    if kb_sessions > 0:
        parts.append(f"[Keyboard] {kb_chars} chars in {kb_sessions} sessions")

    # Light (Android only)
    if platform == "android":
        light_cat = row.get("light_category")
        if light_cat and not pd.isna(light_cat):
            parts.append(f"[Environment] {light_cat}")

    # Music
    music = row.get("music_listening", False)
    if music and not pd.isna(music) and music:
        n_tracks = _safe_int(row.get("music_n_tracks"))
        parts.append(f"[Music] listening, {n_tracks} tracks")

    if not parts:
        # Fallback for days where device was present but no activity detected
        return "[Minimal activity day] Device was active but no significant behavioral signals were recorded. This may indicate the phone was idle, sensors were off, or the participant was not carrying the device."

    return "\n".join(parts)


# ── Main pipeline ───────────────────────────────────────────────────


def build_filtered_for_user(pid: str) -> pd.DataFrame | None:
    """Build filtered daily data for a single user."""
    platform = detect_platform(pid)

    # Load all modalities
    motion_h = load_hourly(pid, "motion")
    screen_h = load_hourly(pid, "screen")
    keyboard_h = load_hourly(pid, "keyinput")
    light_h = load_hourly(pid, "light") if platform == "android" else None
    music_h = load_hourly(pid, "mus")

    # Aggregate to daily
    motion_d = agg_motion_daily(motion_h)
    screen_d = agg_screen_daily(screen_h)
    keyboard_d = agg_keyboard_daily(keyboard_h)
    light_d = agg_light_daily(light_h)
    music_d = agg_music_daily(music_h)

    # App data from screen parquet (Android only)
    app_d = agg_app_daily(screen_h) if platform == "android" else pd.DataFrame()

    # Collect all dates
    all_dates = set()
    for df in [motion_d, screen_d, keyboard_d, light_d, music_d, app_d]:
        if df is not None and not df.empty and "date_local" in df.columns:
            all_dates.update(df["date_local"].tolist())

    if not all_dates:
        return None

    # Build daily dataframe
    daily = pd.DataFrame({"date_local": sorted(all_dates)})
    daily["platform"] = platform

    # Modalities available for this user
    available_mods = ["motion", "screen"]
    if not keyboard_d.empty:
        available_mods.append("keyboard")
    if platform == "android":
        if not app_d.empty:
            available_mods.append("app")
        if not light_d.empty:
            available_mods.append("light")
    if not music_d.empty:
        available_mods.append("music")
    daily["modalities_available"] = [available_mods] * len(daily)

    # Merge all modalities
    for df in [motion_d, screen_d, app_d, keyboard_d, light_d, music_d]:
        if df is not None and not df.empty and "date_local" in df.columns:
            daily = daily.merge(df, on="date_local", how="left")

    # Data quality flags (computed BEFORE capping for transparency)
    daily["data_quality_flags"] = ""

    def _flag_row(row):
        flags = []
        # These checks use the already-capped values, so flag based on
        # tracked hours exceeding normal expectations
        scr = _safe_float(row.get("screen_n_sessions"))
        if scr > 5000:
            flags.append("high_screen_sessions")
        return ",".join(flags) if flags else ""

    daily["data_quality_flags"] = daily.apply(_flag_row, axis=1)

    # Generate narratives
    daily["narrative"] = daily.apply(lambda row: generate_narrative(row, platform), axis=1)

    return daily


def main():
    parser = argparse.ArgumentParser(description="Build filtered daily behavioral data")
    parser.add_argument("--users", type=str, default=None, help="Comma-separated user IDs (default: all)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all users from ALL modality directories (not just screen)
    if args.users:
        user_ids = [int(u.strip()) for u in args.users.split(",")]
    else:
        all_pids = set()
        for modality in ["screen", "motion", "keyinput", "light", "mus"]:
            mod_dir = HOURLY_DIR / modality
            if mod_dir.exists():
                for f in mod_dir.glob("*_hourly.parquet"):
                    try:
                        pid = int(f.stem.split("_")[0])
                        all_pids.add(pid)
                    except ValueError:
                        pass
        user_ids = sorted(all_pids)

    logger.info(f"Processing {len(user_ids)} users → {output_dir}")

    stats = {"android": 0, "ios": 0, "total_days": 0, "skipped": 0}

    for uid in user_ids:
        pid = str(uid).zfill(3)
        daily = build_filtered_for_user(pid)

        if daily is None or daily.empty:
            stats["skipped"] += 1
            continue

        platform = daily["platform"].iloc[0]
        stats[platform] += 1
        stats["total_days"] += len(daily)

        out_path = output_dir / f"{pid}_daily_filtered.parquet"
        # Convert list columns to string for parquet compatibility
        daily["modalities_available"] = daily["modalities_available"].apply(lambda x: ",".join(x) if isinstance(x, list) else x)
        daily.to_parquet(out_path, index=False)

    logger.info(f"Done. Android={stats['android']}, iOS={stats['ios']}, "
                f"skipped={stats['skipped']}, total_days={stats['total_days']}")

    # Print sample narratives for pilot users
    pilot = [71, 119, 164, 310, 458]
    for uid in pilot:
        pid = str(uid).zfill(3)
        f = output_dir / f"{pid}_daily_filtered.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            sample = df.iloc[len(df) // 2]  # middle of the study
            logger.info(f"\n--- User {uid} ({sample['platform']}) sample day {sample['date_local']} ---")
            logger.info(f"\n{sample['narrative']}")


if __name__ == "__main__":
    main()
