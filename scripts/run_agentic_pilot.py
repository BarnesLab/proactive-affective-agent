#!/usr/bin/env python3
"""Run V5 agentic sensing agent evaluation.

V5 is an autonomous sensing agent that uses tool calls to investigate raw
behavioral data before making emotional state predictions. Unlike V1-V4,
which receive pre-formatted sensing summaries in a single prompt, V5
actively queries the data like a behavioral data scientist.

Usage:
    python scripts/run_agentic_pilot.py --users 71,164
    python scripts/run_agentic_pilot.py --users all --model claude-haiku-4-5-20251001
    python scripts/run_agentic_pilot.py --users 71 --dry-run --max-tool-calls 3
    python scripts/run_agentic_pilot.py --users 71,164 --model claude-opus-4-6 --output-dir outputs/v5_run1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.loader import DataLoader
from src.data.preprocessing import get_user_trait_profile
from src.evaluation.metrics import compute_all
from src.sense.query_tools import SensingQueryEngine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "agentic_pilot"
DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_MAX_TOOL_CALLS = 8
DELAY_BETWEEN_USERS = 3.0   # seconds
DELAY_BETWEEN_EMAS = 1.0    # seconds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("V5 Agentic Sensing Agent — Pilot Evaluation")
    logger.info(f"  Model:          {args.model}")
    logger.info(f"  Max tool calls: {args.max_tool_calls}")
    logger.info(f"  Dry run:        {args.dry_run}")
    logger.info(f"  Users:          {args.users}")
    logger.info(f"  Output dir:     {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    loader = DataLoader(data_dir=data_dir)

    logger.info("Loading EMA data...")
    try:
        ema_df = loader.load_all_ema()
    except FileNotFoundError as exc:
        logger.error(f"Could not load EMA data: {exc}")
        sys.exit(1)

    logger.info(f"Loaded {len(ema_df)} EMA entries for {ema_df['Study_ID'].nunique()} users")

    logger.info("Loading sensing data...")
    sensing_dfs = loader.load_all_sensing()
    logger.info(f"Loaded sensing modalities: {list(sensing_dfs.keys())}")

    try:
        baseline_df = loader.load_baseline()
    except FileNotFoundError:
        baseline_df = pd.DataFrame()
        logger.warning("Baseline trait data not found — using empty profiles")

    # ------------------------------------------------------------------
    # Build query engine (shared across users)
    # ------------------------------------------------------------------
    query_engine = SensingQueryEngine(
        sensing_dfs=sensing_dfs,
        ema_df=ema_df,
    )

    # ------------------------------------------------------------------
    # Select users
    # ------------------------------------------------------------------
    all_user_ids = sorted(ema_df["Study_ID"].unique().tolist())

    if args.users.lower() == "all":
        pilot_user_ids = all_user_ids
    else:
        try:
            pilot_user_ids = [int(x.strip()) for x in args.users.split(",")]
        except ValueError:
            logger.error(f"Invalid --users argument: '{args.users}'")
            sys.exit(1)

    # Verify selected users exist
    missing = [uid for uid in pilot_user_ids if uid not in all_user_ids]
    if missing:
        logger.warning(f"Users not found in EMA data: {missing}")
        pilot_user_ids = [uid for uid in pilot_user_ids if uid not in missing]

    if not pilot_user_ids:
        logger.error("No valid users to process.")
        sys.exit(1)

    logger.info(f"Processing {len(pilot_user_ids)} users: {pilot_user_ids}")

    # ------------------------------------------------------------------
    # Run per-user evaluation
    # ------------------------------------------------------------------
    all_predictions: list[dict] = []
    all_ground_truths: list[dict] = []
    all_metadata: list[dict] = []

    run_start = datetime.now()

    for user_idx, study_id in enumerate(pilot_user_ids):
        logger.info(f"\n[{user_idx + 1}/{len(pilot_user_ids)}] Processing user {study_id}")

        user_predictions, user_gts, user_meta = _run_user(
            study_id=study_id,
            ema_df=ema_df,
            baseline_df=baseline_df,
            loader=loader,
            query_engine=query_engine,
            model=args.model,
            max_tool_calls=args.max_tool_calls,
            dry_run=args.dry_run,
            dry_run_limit=2,
            checkpoints_dir=checkpoints_dir,
            logger=logger,
        )

        all_predictions.extend(user_predictions)
        all_ground_truths.extend(user_gts)
        all_metadata.extend(user_meta)

        if user_idx < len(pilot_user_ids) - 1:
            time.sleep(DELAY_BETWEEN_USERS)

    # ------------------------------------------------------------------
    # Compute and print metrics
    # ------------------------------------------------------------------
    logger.info("\nComputing evaluation metrics...")
    if all_predictions and all_ground_truths:
        metrics = compute_all(all_predictions, all_ground_truths, metadata=all_metadata)
    else:
        metrics = {}
        logger.warning("No predictions to evaluate.")

    _print_summary(metrics, len(all_predictions), run_start, args.model)

    # ------------------------------------------------------------------
    # Save final results
    # ------------------------------------------------------------------
    results = {
        "run_config": {
            "model": args.model,
            "max_tool_calls": args.max_tool_calls,
            "dry_run": args.dry_run,
            "users": pilot_user_ids,
            "n_users": len(pilot_user_ids),
            "n_ema_entries": len(all_predictions),
            "run_start": run_start.isoformat(),
            "run_end": datetime.now().isoformat(),
        },
        "metrics": metrics,
        "predictions": all_predictions,
        "ground_truths": all_ground_truths,
        "metadata": all_metadata,
    }

    timestamp = run_start.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"v5_results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_path}")
    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Per-user processing
# ---------------------------------------------------------------------------

def _run_user(
    study_id: int,
    ema_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    loader: DataLoader,
    query_engine: SensingQueryEngine,
    model: str,
    max_tool_calls: int,
    dry_run: bool,
    dry_run_limit: int,
    checkpoints_dir: Path,
    logger: logging.Logger,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Process all EMA entries for a single user.

    Args:
        study_id: Study_ID of the user.
        ema_df: Full EMA DataFrame.
        baseline_df: Baseline trait DataFrame.
        loader: DataLoader instance.
        query_engine: SensingQueryEngine instance.
        model: Anthropic model ID.
        max_tool_calls: Max tool calls per prediction.
        dry_run: If True, only process dry_run_limit EMA entries.
        dry_run_limit: Number of entries to process in dry run.
        checkpoints_dir: Directory for per-EMA checkpoints.
        logger: Logger instance.

    Returns:
        Tuple of (predictions, ground_truths, metadata) lists.
    """
    from src.agent.agentic_sensing import AgenticSensingAgent
    from src.data.schema import UserProfile

    # Load user-specific data
    profile = (
        get_user_trait_profile(baseline_df, study_id)
        if not baseline_df.empty
        else UserProfile(study_id=study_id)
    )
    memory_doc = loader.load_memory_for_user(study_id)

    # Get this user's EMA entries in chronological order
    user_ema = ema_df[ema_df["Study_ID"] == study_id].sort_values("timestamp_local")

    if user_ema.empty:
        logger.warning(f"User {study_id}: no EMA entries found")
        return [], [], []

    entries = list(user_ema.iterrows())
    if dry_run:
        entries = entries[:dry_run_limit]
        logger.info(f"  [DRY RUN] Limiting to {dry_run_limit} EMA entries")

    logger.info(f"  Profile: {profile.to_text()}")
    logger.info(f"  Memory doc: {'loaded' if memory_doc else 'not found'}")
    logger.info(f"  EMA entries to process: {len(entries)}")

    # Checkpoint file for this user
    checkpoint_path = checkpoints_dir / f"user_{study_id}_v5.jsonl"

    # Load existing checkpoint to skip already-processed entries
    processed_timestamps: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    processed_timestamps.add(rec.get("timestamp_local", ""))
                except json.JSONDecodeError:
                    pass
        logger.info(f"  Checkpoint: {len(processed_timestamps)} already processed")

    predictions: list[dict] = []
    ground_truths: list[dict] = []
    metadata: list[dict] = []

    if dry_run:
        # Dry run: skip actual Anthropic API calls
        for idx, (_, ema_row) in enumerate(entries):
            ts = str(ema_row.get("timestamp_local", ""))
            pred = _dry_run_prediction()
            gt = _extract_ground_truth(ema_row)
            meta = {"study_id": study_id, "timestamp_local": ts}
            predictions.append(pred)
            ground_truths.append(gt)
            metadata.append(meta)
            logger.info(f"    [DRY RUN] Entry {idx + 1}: {ts}")
        return predictions, ground_truths, metadata

    # Create agent (one per user; stateless across EMA calls)
    agent = AgenticSensingAgent(
        study_id=study_id,
        profile=profile,
        memory_doc=memory_doc,
        query_engine=query_engine,
        model=model,
        max_tool_calls=max_tool_calls,
    )

    for idx, (_, ema_row) in enumerate(entries):
        ts = str(ema_row.get("timestamp_local", ""))
        ema_date = str(ema_row.get("date_local", ""))

        # Skip if already checkpointed
        if ts in processed_timestamps:
            logger.debug(f"    Skipping already-processed entry: {ts}")
            continue

        logger.info(f"    Entry {idx + 1}/{len(entries)}: {ts}")

        diary_text = str(ema_row.get("emotion_driver", ""))
        if diary_text.lower() == "nan" or not diary_text.strip():
            diary_text = None

        # Run prediction
        t0 = time.time()
        try:
            pred = agent.predict(ema_row=ema_row, diary_text=diary_text)
        except Exception as exc:
            logger.error(f"    Prediction error for {ts}: {exc}")
            pred = agent._fallback_prediction()
            pred["_error"] = str(exc)
            pred["_version"] = "v5"
            pred["_n_tool_calls"] = 0

        elapsed = time.time() - t0
        logger.info(
            f"    Done in {elapsed:.1f}s | tool_calls={pred.get('_n_tool_calls', '?')} "
            f"| confidence={pred.get('confidence', '?'):.2f}"
        )

        gt = _extract_ground_truth(ema_row)
        meta = {
            "study_id": study_id,
            "timestamp_local": ts,
            "date_local": ema_date,
            "n_tool_calls": pred.get("_n_tool_calls", 0),
        }

        predictions.append(pred)
        ground_truths.append(gt)
        metadata.append(meta)

        # Save checkpoint
        checkpoint_record = {
            "study_id": study_id,
            "timestamp_local": ts,
            "date_local": ema_date,
            "prediction": pred,
            "ground_truth": gt,
        }
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps(checkpoint_record, default=str) + "\n")

        if idx < len(entries) - 1:
            time.sleep(DELAY_BETWEEN_EMAS)

    logger.info(f"  User {study_id}: {len(predictions)} predictions completed")
    return predictions, ground_truths, metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ground_truth(ema_row: pd.Series) -> dict[str, Any]:
    """Extract all prediction targets as a ground-truth dict from an EMA row."""
    from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

    gt: dict[str, Any] = {}

    for target in CONTINUOUS_TARGETS:
        val = ema_row.get(target)
        try:
            gt[target] = float(val) if val is not None and str(val) != "nan" else None
        except (ValueError, TypeError):
            gt[target] = None

    for target in BINARY_STATE_TARGETS:
        val = ema_row.get(target)
        if val is None or str(val).lower() == "nan":
            gt[target] = None
        elif str(val).lower() in ("true", "1", "yes"):
            gt[target] = True
        elif str(val).lower() in ("false", "0", "no"):
            gt[target] = False
        else:
            gt[target] = None

    avail = ema_row.get("INT_availability")
    gt["INT_availability"] = str(avail).lower().strip() if avail is not None else None

    return gt


def _dry_run_prediction() -> dict[str, Any]:
    """Return a placeholder prediction for dry-run testing."""
    from src.utils.mappings import BINARY_STATE_TARGETS
    pred: dict[str, Any] = {
        "PANAS_Pos": 15.0,
        "PANAS_Neg": 8.0,
        "ER_desire": 3.0,
        "INT_availability": "yes",
        "reasoning": "[DRY RUN] Placeholder V5 prediction",
        "confidence": 0.5,
        "_version": "v5",
        "_n_tool_calls": 0,
        "_dry_run": True,
    }
    for target in BINARY_STATE_TARGETS:
        pred[target] = False
    return pred


def _print_summary(
    metrics: dict,
    n_predictions: int,
    run_start: datetime,
    model: str,
) -> None:
    """Print evaluation summary to stdout."""
    elapsed = (datetime.now() - run_start).total_seconds()
    print(f"\n{'='*65}")
    print("V5 AGENTIC SENSING AGENT — EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"Model:           {model}")
    print(f"Predictions:     {n_predictions}")
    print(f"Elapsed:         {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    agg = metrics.get("aggregate", {})

    print(f"\nAggregate Metrics:")
    if agg.get("mean_mae") is not None:
        print(f"  Mean MAE (continuous):          {agg['mean_mae']:.4f}")
    if agg.get("mean_balanced_accuracy") is not None:
        print(f"  Mean Balanced Accuracy (binary): {agg['mean_balanced_accuracy']:.4f}")
    if agg.get("mean_f1") is not None:
        print(f"  Mean F1 (binary):               {agg['mean_f1']:.4f}")
    if agg.get("personal_threshold_mean_ba") is not None:
        print(f"  Personal Threshold BA:          {agg['personal_threshold_mean_ba']:.4f}")
    if agg.get("personal_threshold_mean_f1") is not None:
        print(f"  Personal Threshold F1:          {agg['personal_threshold_mean_f1']:.4f}")

    print(f"\nContinuous Targets:")
    for target, vals in metrics.get("continuous", {}).items():
        print(f"  {target}: MAE={vals['mae']:.3f} (n={vals['n']})")

    print(f"\nBinary Targets (sorted by balanced accuracy):")
    binary = sorted(
        metrics.get("binary", {}).items(),
        key=lambda x: x[1].get("balanced_accuracy", 0),
        reverse=True,
    )
    for target, vals in binary:
        ba = vals.get("balanced_accuracy", 0)
        f1 = vals.get("f1", 0)
        n = vals.get("n", 0)
        print(f"  {target}: BA={ba:.3f} F1={f1:.3f} (n={n})")

    avail = metrics.get("availability", {})
    if avail:
        print(
            f"\nAvailability: BA={avail.get('balanced_accuracy', 0):.3f} "
            f"F1={avail.get('f1', 0):.3f} (n={avail.get('n', 0)})"
        )

    print(f"{'='*65}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 agentic sensing agent evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/run_agentic_pilot.py --users 71,164
  python scripts/run_agentic_pilot.py --users all --model claude-haiku-4-5-20251001
  python scripts/run_agentic_pilot.py --users 71 --dry-run
  python scripts/run_agentic_pilot.py --users 71,164 --max-tool-calls 5 --output-dir outputs/v5_short
""",
    )
    parser.add_argument(
        "--users",
        type=str,
        default="71,164",
        help=(
            "Comma-separated Study_IDs (e.g. '71,164') or 'all' for all users. "
            "Default: '71,164'"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Anthropic model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=DEFAULT_MAX_TOOL_CALLS,
        help=f"Maximum tool calls per EMA entry (default: {DEFAULT_MAX_TOOL_CALLS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only process the first 2 EMA entries per user without LLM calls (pipeline test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: data/ relative to project root)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    main()
