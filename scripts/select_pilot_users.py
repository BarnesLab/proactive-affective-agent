#!/usr/bin/env python3
"""Select the best pilot users based on data coverage.

Combines all 5 test splits (no group dependency) and ranks users by:
  - EMA entry count (more is better)
  - Sensing data coverage (overlap with EMA dates)
  - Memory document existence

Usage:
    python scripts/select_pilot_users.py
    python scripts/select_pilot_users.py --n-users 10
    python scripts/select_pilot_users.py --min-ema 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import DataLoader
from src.utils.mappings import SENSING_COLUMNS, study_id_to_participant_id


def main():
    parser = argparse.ArgumentParser(description="Select pilot users with best data coverage")
    parser.add_argument("--n-users", type=int, default=5, help="Number of users to select")
    parser.add_argument("--min-ema", type=int, default=50, help="Minimum EMA entries")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    loader = DataLoader(data_dir=project_root / args.data_dir)

    print("Loading all EMA test data (combining 5 splits)...")
    all_ema = loader.load_all_ema()
    print(f"Total: {len(all_ema)} EMA entries, {all_ema['Study_ID'].nunique()} users")

    # Count EMA entries per user
    user_counts = all_ema.groupby("Study_ID").size().reset_index(name="ema_count")
    user_counts = user_counts.sort_values("ema_count", ascending=False)
    print(f"Users with >= {args.min_ema} EMA entries: {(user_counts['ema_count'] >= args.min_ema).sum()}")

    # Filter by minimum EMA
    candidates = user_counts[user_counts["ema_count"] >= args.min_ema].copy()
    if candidates.empty:
        print(f"No users with >= {args.min_ema} EMA entries. Lowering threshold...")
        candidates = user_counts.head(30).copy()

    # Check memory document availability
    print("\nChecking memory documents...")
    candidates["has_memory"] = candidates["Study_ID"].apply(
        lambda sid: loader.load_memory_for_user(sid) is not None
    )
    print(f"Users with memory docs: {candidates['has_memory'].sum()}/{len(candidates)}")

    # Check sensing coverage
    print("\nLoading sensing data...")
    sensing_dfs = loader.load_all_sensing()

    sensing_scores = []
    for _, row in candidates.iterrows():
        sid = row["Study_ID"]
        pid = study_id_to_participant_id(sid)

        # Get date range from EMA
        user_ema = all_ema[all_ema["Study_ID"] == sid]
        ema_dates = set(user_ema["date_local"].unique())

        # Count how many sensing sources have data for this user's dates
        sensor_coverage = {}
        for sensor_name, df in sensing_dfs.items():
            info = SENSING_COLUMNS[sensor_name]
            user_sensing = df[df[info["id_col"]] == pid]
            if not user_sensing.empty:
                sensing_dates = set(user_sensing[info["date_col"]].unique())
                overlap = len(ema_dates & sensing_dates)
                sensor_coverage[sensor_name] = overlap / max(len(ema_dates), 1)
            else:
                sensor_coverage[sensor_name] = 0.0

        avg_coverage = sum(sensor_coverage.values()) / max(len(sensor_coverage), 1)
        n_sensors_with_data = sum(1 for v in sensor_coverage.values() if v > 0)
        sensing_scores.append({
            "Study_ID": sid,
            "avg_sensing_coverage": avg_coverage,
            "n_sensors": n_sensors_with_data,
        })

    sensing_df = pd.DataFrame(sensing_scores)
    candidates = candidates.merge(sensing_df, on="Study_ID")

    # Composite score
    max_ema = candidates["ema_count"].max()
    candidates["score"] = (
        candidates["ema_count"] / max_ema * 0.3
        + candidates["avg_sensing_coverage"] * 0.4
        + candidates["n_sensors"] / 8.0 * 0.2
        + candidates["has_memory"].astype(float) * 0.1
    )

    candidates = candidates.sort_values("score", ascending=False)
    selected = candidates.head(args.n_users)

    print(f"\n{'='*80}")
    print(f"SELECTED {args.n_users} PILOT USERS")
    print(f"{'='*80}")
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        print(f"\n{i}. Study_ID={row['Study_ID']}")
        print(f"   EMA entries: {row['ema_count']}")
        print(f"   Memory doc: {'Yes' if row['has_memory'] else 'No'}")
        print(f"   Sensing coverage: {row['avg_sensing_coverage']:.1%} ({row['n_sensors']}/8 sensors)")
        print(f"   Composite score: {row['score']:.3f}")

    selected_ids = selected["Study_ID"].tolist()
    print(f"\nPilot user IDs: {selected_ids}")
    print(f"\nUsage: python scripts/run_pilot.py --users {','.join(str(x) for x in selected_ids)}")


if __name__ == "__main__":
    main()
