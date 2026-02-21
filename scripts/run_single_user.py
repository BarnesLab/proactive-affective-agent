"""Run simulation for a single user (for debugging and development).

Usage:
    python scripts/run_single_user.py --user_id BUCS_001 --version v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run single-user simulation for debugging")
    parser.add_argument("--user_id", type=str, required=True, help="Participant ID")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1", help="Agent version")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    args = parser.parse_args()

    print(f"Running single-user simulation: user={args.user_id}, version={args.version}")

    # TODO: Load data for this user
    # TODO: Create PersonalAgent
    # TODO: Run simulation day by day
    # TODO: Print results

    raise NotImplementedError("Single-user simulation not yet implemented")


if __name__ == "__main__":
    main()
