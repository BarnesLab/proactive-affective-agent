#!/usr/bin/env python3
"""Main entry point for the pilot study.

Usage:
    # Full pilot (all 5 versions, sequential)
    python scripts/run_pilot.py --version all --users 2,34,56,78,90

    # Single version (for parallel execution)
    python scripts/run_pilot.py --version callm --users 2,34,56,78,90
    python scripts/run_pilot.py --version v3 --users 2,34,56,78,90
    python scripts/run_pilot.py --version v4 --users 2,34,56,78,90

    # Dry run (no LLM calls, test pipeline)
    python scripts/run_pilot.py --version all --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.simulation.simulator import PilotSimulator


def main():
    parser = argparse.ArgumentParser(
        description="Run the pilot study: CALLM vs V1 vs V2 vs V3 vs V4"
    )
    parser.add_argument(
        "--version", type=str, default="all",
        help="Which version to run: callm, v1, v2, v3, v4, or all (default: all)"
    )
    parser.add_argument(
        "--users", type=str, default=None,
        help="Comma-separated Study_IDs to use (default: auto-select 5)"
    )
    parser.add_argument(
        "--n-users", type=int, default=5,
        help="Number of users to auto-select (default: 5, ignored if --users set)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run pipeline without LLM calls (for testing)"
    )
    parser.add_argument(
        "--model", type=str, default="sonnet",
        help="Claude model to use (default: sonnet)"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between LLM calls (default: 2.0)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/pilot/)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory (default: data/)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse versions
    if args.version == "all":
        versions = ["callm", "v1", "v2", "v3", "v4"]
    else:
        versions = [v.strip() for v in args.version.split(",")]

    # Parse user IDs
    pilot_user_ids = None
    if args.users:
        pilot_user_ids = [int(x.strip()) for x in args.users.split(",")]

    # Setup paths
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "pilot"

    # Initialize
    loader = DataLoader(data_dir=data_dir)
    simulator = PilotSimulator(
        loader=loader,
        output_dir=output_dir,
        pilot_user_ids=pilot_user_ids,
        dry_run=args.dry_run,
        model=args.model,
        delay=args.delay,
    )

    # Setup (load data)
    logging.info("Setting up pilot study...")
    logging.info(f"  Versions: {versions}")
    logging.info(f"  Users: {pilot_user_ids or 'auto-select'}")
    logging.info(f"  Dry run: {args.dry_run}")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Output: {output_dir}")

    simulator.setup()

    # Show selected users
    if simulator._users_data:
        selected = [u["study_id"] for u in simulator._users_data]
        total_entries = sum(len(u["ema_entries"]) for u in simulator._users_data)
        logging.info(f"  Selected users: {selected}")
        logging.info(f"  Total EMA entries: {total_entries}")

    # Run
    results = simulator.run_all(versions=versions)

    # Print summary
    print(f"\n{'='*60}")
    print("PILOT STUDY SUMMARY")
    print(f"{'='*60}")
    for version, result in results.items():
        print(f"\n{version.upper()}:")
        print(f"  LLM calls: {result['total_llm_calls']}")
        metrics = result.get("metrics", {})
        agg = metrics.get("aggregate", {})
        if agg.get("mean_mae") is not None:
            print(f"  Mean MAE: {agg['mean_mae']:.3f}")
        if agg.get("mean_balanced_accuracy") is not None:
            print(f"  Mean Balanced Accuracy: {agg['mean_balanced_accuracy']:.3f}")
        if agg.get("mean_f1") is not None:
            print(f"  Mean F1: {agg['mean_f1']:.3f}")
        if agg.get("personal_threshold_mean_ba") is not None:
            print(f"  Personal Threshold BA: {agg['personal_threshold_mean_ba']:.3f}")
        if agg.get("personal_threshold_mean_f1") is not None:
            print(f"  Personal Threshold F1: {agg['personal_threshold_mean_f1']:.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
