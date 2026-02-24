"""
Phase 1 Offline Processing — Keyboard Input (FleksyKeyInput)
============================================================
Aggregates typing sessions into hourly features including word counts and
LIWC-inspired sentiment proxies.  Raw text is NOT stored in the output.

Input
-----
  data/bucs-data/FleksyKeyInput/

Output
------
  data/processed/hourly/keyinput/{pid}_keyinput_hourly.parquet

Per-hour columns
----------------
  hour_utc, hour_local, participant_id,
  key_n_sessions, key_chars_typed, key_words_typed, key_session_min,
  key_prop_neg, key_prop_pos,
  device_missing, participant_missing
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
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "keyinput"
PLATFORM_FILE = PROJECT_ROOT / "data" / "processed" / "hourly" / "participant_platform.parquet"
MOTION_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "motion"
SCREEN_DIR = PROJECT_ROOT / "data" / "processed" / "hourly" / "screen"

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
MAX_SESSION_MIN = 30          # cap sessions at 30 min; longer are artifacts

# ---------------------------------------------------------------------------
# LIWC-inspired word lists (hardcoded)
# ---------------------------------------------------------------------------
POSITIVE_WORDS: set[str] = {
    'good', 'great', 'happy', 'love', 'wonderful', 'excellent', 'amazing', 'joy',
    'excited', 'thankful', 'grateful', 'glad', 'fantastic', 'beautiful', 'enjoyed',
    'enjoy', 'fun', 'nice', 'awesome', 'blessed', 'positive', 'smile', 'laughed',
    'better', 'best', 'perfect', 'well', 'pleased', 'cheerful', 'hope', 'hopeful',
}

NEGATIVE_WORDS: set[str] = {
    'bad', 'sad', 'hurt', 'pain', 'tired', 'anxious', 'worried', 'stress',
    'stressed', 'depressed', 'lonely', 'upset', 'angry', 'frustrated', 'afraid',
    'scared', 'difficult', 'hard', 'sick', 'terrible', 'awful', 'horrible',
    'worse', 'worst', 'miserable', 'unhappy', 'crying', 'cried', 'fear', 'hate',
}


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    import re
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = re.sub(r"[^\w\s']", ' ', text.lower())
    return cleaned.split()


def count_sentiment(words: list[str]) -> tuple[int, int]:
    """Return (n_positive, n_negative) word counts."""
    n_pos = sum(1 for w in words if w in POSITIVE_WORDS)
    n_neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return n_pos, n_neg


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_keyinput_files(pid_str: str) -> pd.DataFrame:
    """Load all FleksyKeyInput CSV data files for a PID."""
    ki_dir = DATA_ROOT / "FleksyKeyInput"
    # Primary: FleksyKeyInput_data_{PID}_{DEVICE_ID}_{date}_{seq}.csv
    files = sorted(ki_dir.glob(f"FleksyKeyInput_data_{pid_str}_*.csv"))
    # Alternate: {PID}_FleksyKeyInput_*
    files += sorted(ki_dir.glob(f"{pid_str}_FleksyKeyInput_*.csv"))
    files = [f for f in files if 'header' not in f.name.lower()]

    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'epoch_session_start' not in df.columns:
                continue
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"[keyinput] PID {pid_str}: failed to read {f.name} — {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-session filtering and annotation
# ---------------------------------------------------------------------------

def preprocess_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter raw session rows:
      - Drop password fields (cat_textfield == 'Pwd')
      - Compute duration, cap at MAX_SESSION_MIN
      - Drop NaN epochs
    """
    df = df.copy()

    # Drop password sessions
    if 'cat_textfield' in df.columns:
        df = df[df['cat_textfield'].str.strip().str.lower() != 'pwd']

    df['epoch_session_start'] = pd.to_numeric(df['epoch_session_start'], errors='coerce')
    df['epoch_session_end'] = pd.to_numeric(df['epoch_session_end'], errors='coerce')
    df = df.dropna(subset=['epoch_session_start'])

    # Duration in minutes (fallback to 0 if end is missing/earlier)
    df['duration_ms'] = (
        df['epoch_session_end'] - df['epoch_session_start']
    ).clip(lower=0)

    max_ms = MAX_SESSION_MIN * 60_000
    df['duration_ms'] = df['duration_ms'].clip(upper=max_ms)

    # Assign hour bucket by session start
    df['hour_utc'] = epoch_to_hour_utc(df['epoch_session_start'])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-participant pipeline
# ---------------------------------------------------------------------------

def _load_motion_device_missing(pid_str: str) -> pd.Series | None:
    """Load device_missing from motion hourly parquet, indexed by hour_utc."""
    path = MOTION_DIR / f"{pid_str}_motion_hourly.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=['hour_utc', 'device_missing'])
    return df.set_index('hour_utc')['device_missing']


def _load_screen_on(pid_str: str) -> pd.Series | None:
    """Load screen_on_min from screen hourly parquet, indexed by hour_utc."""
    path = SCREEN_DIR / f"{pid_str}_screen_hourly.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=['hour_utc', 'screen_on_min'])
    return df.set_index('hour_utc')['screen_on_min']


def process_participant(pid_int: int, pid_str: str) -> pd.DataFrame | None:
    """Full keyinput pipeline for one participant."""
    raw = load_keyinput_files(pid_str)
    if raw.empty:
        return None

    sessions = preprocess_sessions(raw)
    if sessions.empty:
        return None

    # ------------------------------------------------------------------
    # Timezone mapping per hour
    # ------------------------------------------------------------------
    tz_map = tz_mode_per_hour(sessions, 'hour_utc', 'timezone')

    # ------------------------------------------------------------------
    # Load cross-modal context (for missing flags)
    # ------------------------------------------------------------------
    motion_missing = _load_motion_device_missing(pid_str)
    screen_on = _load_screen_on(pid_str)

    # ------------------------------------------------------------------
    # Aggregate per hour
    # ------------------------------------------------------------------
    first_hour = sessions['hour_utc'].min()
    last_hour = sessions['hour_utc'].max()
    hour_range = pd.date_range(start=first_hour, end=last_hour, freq='h', tz='UTC')

    rows = []
    for hour_utc in hour_range:
        sub = sessions[sessions['hour_utc'] == hour_utc]

        # Default features
        if sub.empty:
            n_sess = 0
            chars_typed = 0
            words_typed = 0
            session_min = 0.0
            n_pos = 0
            n_neg = 0
        else:
            n_sess = len(sub)
            session_min = sub['duration_ms'].sum() / 60_000.0

            # Text aggregation — do NOT store raw text in output
            all_words: list[str] = []
            total_chars = 0
            for text_val in sub['strinput_text']:
                if not isinstance(text_val, str):
                    continue
                total_chars += len(text_val)
                all_words.extend(tokenize(text_val))

            chars_typed = total_chars
            words_typed = len(all_words)
            n_pos, n_neg = count_sentiment(all_words)

        # Sentiment proportions
        key_prop_pos = float(n_pos / words_typed) if words_typed > 0 else np.nan
        key_prop_neg = float(n_neg / words_typed) if words_typed > 0 else np.nan

        # Missing flags
        dev_miss = False
        if motion_missing is not None and hour_utc in motion_missing.index:
            dev_miss = bool(motion_missing[hour_utc])
        # device_missing: hour has no keyinput AND motion says device was used
        device_missing = (n_sess == 0) and (not dev_miss)  # device used but no typing

        # participant_missing: screen was on but no keyinput (reading / passive use)
        participant_missing = False
        if n_sess == 0 and screen_on is not None and hour_utc in screen_on.index:
            participant_missing = float(screen_on[hour_utc]) > 0

        # Local time
        tz_str = tz_map.get(hour_utc, '+00:00')
        offset_min = parse_tz_minutes(tz_str)
        hour_local = hour_utc.tz_localize(None) + pd.Timedelta(minutes=offset_min)

        rows.append({
            'hour_utc': hour_utc,
            'hour_local': hour_local,
            'participant_id': pid_str,
            'key_n_sessions': n_sess,
            'key_chars_typed': chars_typed,
            'key_words_typed': words_typed,
            'key_session_min': session_min,
            'key_prop_pos': key_prop_pos,
            'key_prop_neg': key_prop_neg,
            'device_missing': device_missing,
            'participant_missing': participant_missing,
        })

    if not rows:
        return None

    result = pd.DataFrame(rows)
    col_order = [
        'hour_utc', 'hour_local', 'participant_id',
        'key_n_sessions', 'key_chars_typed', 'key_words_typed', 'key_session_min',
        'key_prop_pos', 'key_prop_neg',
        'device_missing', 'participant_missing',
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
        print(f"[keyinput] ERROR: {exc}")
        return

    ki_dir = DATA_ROOT / "FleksyKeyInput"
    # Gather PIDs from both filename patterns
    pids_primary = {
        f.name.split('_')[2]
        for f in ki_dir.glob("FleksyKeyInput_data_*.csv")
        if len(f.name.split('_')) > 2 and 'header' not in f.name.lower()
    }
    pids_alt = {
        f.name.split('_')[0]
        for f in ki_dir.glob("*_FleksyKeyInput_*.csv")
        if 'header' not in f.name.lower()
    }
    all_pids = sorted(pids_primary | pids_alt)

    print(f"[keyinput] Found {len(all_pids)} participants in {ki_dir}")
    print(f"[keyinput] Withdraw list: {sorted(WITHDRAW_LIST)}")

    processed = skipped_withdraw = failed = 0

    for pid_str in tqdm(all_pids, desc="Processing keyinput", unit="participant"):
        try:
            pid_int = int(pid_str)
        except ValueError:
            continue

        if pid_int in WITHDRAW_LIST:
            skipped_withdraw += 1
            continue

        pid_str_norm = pid_from_int(pid_int)
        out_path = OUT_DIR / f"{pid_str_norm}_keyinput_hourly.parquet"

        try:
            result = process_participant(pid_int, pid_str_norm)
        except Exception as exc:
            warnings.warn(f"[keyinput] PID {pid_str_norm}: unhandled error — {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

        if result is None or result.empty:
            warnings.warn(f"[keyinput] PID {pid_str_norm}: produced no rows, skipping output.")
            failed += 1
            continue

        result.to_parquet(out_path, index=False, compression='snappy')
        processed += 1

        n_hours = len(result)
        n_active = (result['key_n_sessions'] > 0).sum()
        total_words = result['key_words_typed'].sum()
        tqdm.write(
            f"  PID {pid_str_norm}: {n_hours} hours | "
            f"active_hours={n_active} | "
            f"total_words={total_words:,} | "
            f"-> {out_path.name}"
        )

    print(
        f"\n[keyinput] Done. "
        f"Processed: {processed} | "
        f"Withdrawn: {skipped_withdraw} | "
        f"Failed/empty: {failed}"
    )


if __name__ == "__main__":
    main()
