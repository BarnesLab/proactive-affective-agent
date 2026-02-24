"""
Shared utilities for Phase 1 offline batch processing scripts.

Import from here in process_motion.py, process_screen_app.py, etc.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Withdrawn participants — exclude from all processing
# ---------------------------------------------------------------------------
WITHDRAW_LIST: set[int] = {4, 20, 70, 94, 214, 253, 283, 494, 153}


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

def parse_tz_minutes(tz_str) -> int:
    """
    Parse a timezone offset string such as '-04:00' or '+05:30'
    into total signed minutes from UTC.
    """
    s = str(tz_str).strip()
    sign = 1 if s[0] != '-' else -1
    parts = s.lstrip('+-').split(':')
    return sign * (int(parts[0]) * 60 + int(parts[1]))


def to_local_time(epoch_ms_series: pd.Series, timezone_series: pd.Series) -> pd.Series:
    """
    Convert epoch milliseconds + per-row timezone offset string to a
    timezone-naive local datetime Series.
    """
    utc = pd.to_datetime(epoch_ms_series / 1000, unit='s', utc=True)
    offset_minutes = timezone_series.map(parse_tz_minutes)
    local = utc + pd.to_timedelta(offset_minutes, unit='m')
    return local.dt.tz_localize(None)


def epoch_to_hour_utc(epoch_ms_series: pd.Series) -> pd.Series:
    """
    Convert epoch milliseconds to timezone-aware UTC datetime floored to the
    nearest hour.
    """
    return pd.to_datetime(epoch_ms_series / 1000, unit='s', utc=True).dt.floor('h')


def get_hour_local(hour_utc: pd.Timestamp, tz_str_mode: str) -> pd.Timestamp:
    """
    Convert a UTC hour Timestamp to local time using the most common timezone
    string observed during that hour.
    """
    offset_min = parse_tz_minutes(tz_str_mode)
    return hour_utc + pd.Timedelta(minutes=offset_min)


def pid_from_int(n) -> str:
    """Return a zero-padded 3-character participant ID string."""
    return str(int(n)).zfill(3)


# ---------------------------------------------------------------------------
# Participant platform lookup
# ---------------------------------------------------------------------------

def load_platform_map(platform_file) -> dict[str, str]:
    """
    Load participant_platform.parquet and return a dict mapping
    zero-padded PID string -> platform string ('ios' | 'android' | 'unknown').
    """
    import pandas as pd
    from pathlib import Path

    path = Path(platform_file)
    if not path.exists():
        raise FileNotFoundError(
            f"participant_platform.parquet not found at {path}. "
            "Run scripts/offline/build_participant_roster.py first."
        )
    df = pd.read_parquet(path)
    return dict(zip(df['participant_id'].astype(str), df['platform'].astype(str)))


# ---------------------------------------------------------------------------
# Common hour-level timezone aggregation
# ---------------------------------------------------------------------------

def tz_mode_per_hour(df: pd.DataFrame, hour_col: str, tz_col: str) -> pd.Series:
    """
    Given a DataFrame with an hour column and a timezone string column,
    return a Series indexed by hour with the most common timezone string.
    """
    return (
        df.groupby(hour_col)[tz_col]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )
