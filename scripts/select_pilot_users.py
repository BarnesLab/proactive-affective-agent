#!/usr/bin/env python3
"""Select the best 5 pilot users based on data coverage.

Criteria:
  - At least 50 EMA entries in the test set
  - Good sensing data coverage (many dates with sensing)
  - Memory document exists

Usage:
    python scripts/select_pilot_users.py --group 1
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import DataLoader
from src.utils.mappings import SENSING_COLUMNS, study_id_to_participant_id


def main():
    parser = argparse.ArgumentParser(description="Select pilot users with best data coverage")
    parser.add_argument("--group", type=int, default=1, help="CV group number (1-5)")
    parser.add_argument("--n-users", type=int, default=5, help="Number of users to select")
    parser.add_argument("--min-ema", type=int, default=50, help="Minimum EMA entries")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    loader = DataLoader(data_dir=project_root / args.data_dir)

    print(f"Loading group {args.group} test data...")
    test_df = loader.load_split(args.group, "test")

    # Count EMA entries per user
    user_counts = test_df.groupby("Study_ID").size().reset_index(name="ema_count")
    user_counts = user_counts.sort_values("ema_count", ascending=False)
    print(f"\nTotal users in group {args.group} test: {len(user_counts)}")
    print(f"Users with >= {args.min_ema} EMA entries: {(user_counts['ema_count'] >= args.min_ema).sum()}")

    # Filter by minimum EMA
    candidates = user_counts[user_counts["ema_count"] >= args.min_ema].copy()
    if candidates.empty:
        print(f"No users with >= {args.min_ema} EMA entries. Lowering threshold...")
        candidates = user_counts.head(20).copy()

    # Check memory document availability
    print("\nChecking memory documents...")
    candidates["has_memory"] = candidates["Study_ID"].apply(
        lambda sid: loader.load_memory_for_user(sid) is not None
    )
    print(f"Users with memory docs: {candidates['has_memory'].sum()}/{len(candidates)}")

    # Check sensing coverage
    print("\nLoading sensing data (this may take a moment)...")
    sensing_dfs = loader.load_all_sensing()

    sensing_scores = []
    for _, row in candidates.iterrows():
        sid = row["Study_ID"]
        pid = study_id_to_participant_id(sid)

        # Get date range from EMA
        user_ema = test_df[test_df["Study_ID"] == sid]
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
            **{f"cov_{k}": v for k, v in sensor_coverage.items()},
        })

    sensing_df = pd.DataFrame(sensing_scores)
    candidates = candidates.merge(sensing_df, on="Study_ID")

    # Composite score: EMA count (normalized) + sensing coverage + memory bonus
    max_ema = candidates["ema_count"].max()
    candidates["score"] = (
        candidates["ema_count"] / max_ema * 0.3
        + candidates["avg_sensing_coverage"] * 0.4
        + candidates["n_sensors"] / 8.0 * 0.2
        + candidates["has_memory"].astype(float) * 0.1
    )

    # Sort by composite score
    candidates = candidates.sort_values("score", ascending=False)

    # Select top N
    selected = candidates.head(args.n_users)

    print(f"\n{'='*80}")
    print(f"SELECTED {args.n_users} PILOT USERS (Group {args.group})")
    print(f"{'='*80}")
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        print(f"\n{i}. Study_ID={row['Study_ID']}")
        print(f"   EMA entries: {row['ema_count']}")
        print(f"   Memory doc: {'Yes' if row['has_memory'] else 'No'}")
        print(f"   Sensing coverage: {row['avg_sensing_coverage']:.1%} ({row['n_sensors']}/8 sensors)")
        print(f"   Composite score: {row['score']:.3f}")

    # Output as list for use in other scripts
    selected_ids = selected["Study_ID"].tolist()
    print(f"\nPilot user IDs: {selected_ids}")
    print(f"\nUsage: python scripts/run_pilot.py --users {','.join(str(x) for x in selected_ids)}")


if __name__ == "__main__":
    main()
