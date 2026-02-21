"""Run full simulation across all users and days.

Usage:
    python scripts/run_simulation.py --version v1 --folds 1,2,3,4,5
    python scripts/run_simulation.py --version v2 --folds 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run full Proactive Affective Agent simulation")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1", help="Agent version")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5", help="Comma-separated fold numbers")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs"), help="Output directory")
    args = parser.parse_args()

    folds = [int(f) for f in args.folds.split(",")]
    print(f"Running simulation: version={args.version}, folds={folds}")
    print(f"Output: {args.output_dir}")

    # TODO: Initialize DataLoader, Simulator, Evaluator
    # TODO: Run simulation for each fold
    # TODO: Generate reports

    raise NotImplementedError("Full simulation pipeline not yet implemented")


if __name__ == "__main__":
    main()
