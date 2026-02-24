#!/usr/bin/env python3
"""Run V4 agentic sensing agent evaluation.

V4 is an autonomous sensing agent that uses tool calls to investigate raw
behavioral data before making emotional state predictions. Unlike V1-V4,
which receive pre-formatted sensing summaries in a single prompt, V4
actively queries the data like a behavioral data scientist.

Two backends are available:
  --backend api  (default) — Anthropic SDK, bills against ANTHROPIC_API_KEY
  --backend cc             — claude --print subprocess, bills against Claude Max subscription

Usage:
    python scripts/run_agentic_pilot.py --users 71,164
    python scripts/run_agentic_pilot.py --users 71,164 --backend cc
    python scripts/run_agentic_pilot.py --users all --model sonnet --backend cc
    python scripts/run_agentic_pilot.py --users 71 --dry-run --max-tool-calls 3
    python scripts/run_agentic_pilot.py --users 71,164 --model claude-sonnet-4-6 --output-dir outputs/v4_run1
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
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_BACKEND = "cc"  # "cc" = Claude Max subscription via subprocess, "api" = Anthropic SDK
CC_PYTHON_BIN = "/opt/homebrew/bin/python3.13"  # Python with mcp + pandas installed

# Feedback modes control what the agent accumulates in its per-user session memory.
# "real":   diary + receptivity only (ER_desire + INT_availability).
#           Simulates real-world deployment — no full EMA battery available as feedback.
# "oracle": diary + receptivity + raw PA/NA scores.
#           Research upper bound — agent calibrates against full affect battery.
DEFAULT_FEEDBACK_MODE = "real"
DELAY_BETWEEN_USERS = 3.0   # seconds
DELAY_BETWEEN_EMAS = 1.0    # seconds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("V4 Agentic Sensing Agent — Pilot Evaluation")
    logger.info(f"  Backend:        {args.backend} ({'Claude Code Max subscription' if args.backend == 'cc' else 'Anthropic API key'})")
    logger.info(f"  Model:          {args.model}")
    logger.info(f"  Feedback mode:  {args.feedback_mode} ({'receptivity-only (real-world)' if args.feedback_mode == 'real' else 'full PA/NA oracle (upper bound)'})")
    logger.info(f"  Max tool calls: {args.max_tool_calls}")
    logger.info(f"  Dry run:        {args.dry_run}")
    logger.info(f"  Users:          {args.users}")
    logger.info(f"  Output dir:     {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    memory_dir = output_dir / "memory"
    memory_dir.mkdir(exist_ok=True)

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

    try:
        baseline_df = loader.load_baseline()
    except FileNotFoundError:
        baseline_df = pd.DataFrame()
        logger.warning("Baseline trait data not found — using empty profiles")

    # ------------------------------------------------------------------
    # Build query engine (Parquet-backed, shared across users)
    # ------------------------------------------------------------------
    processed_dir = (Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data") / "processed" / "hourly"
    if not processed_dir.exists():
        logger.error(f"Processed hourly data directory not found: {processed_dir}")
        logger.error("Run scripts/offline/run_phase1.sh first to generate Parquet files.")
        sys.exit(1)

    query_engine = SensingQueryEngine(
        processed_dir=processed_dir,
        ema_df=ema_df,
    )
    logger.info(f"Parquet query engine loaded from: {processed_dir}")

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
            memory_dir=memory_dir,
            logger=logger,
            backend=args.backend,
            processed_dir=processed_dir.parent,
            feedback_mode=args.feedback_mode,
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

    _print_summary(metrics, len(all_predictions), run_start, args.model, all_metadata)

    # ------------------------------------------------------------------
    # Save final results
    # ------------------------------------------------------------------
    total_tokens = sum(m.get("total_tokens", 0) for m in all_metadata)
    total_tool_calls = sum(m.get("n_tool_calls", 0) for m in all_metadata)
    results = {
        "run_config": {
            "backend": args.backend,
            "feedback_mode": args.feedback_mode,
            "model": args.model,
            "max_tool_calls": args.max_tool_calls,
            "dry_run": args.dry_run,
            "diary_only": True,
            "users": pilot_user_ids,
            "n_users": len(pilot_user_ids),
            "n_ema_entries": len(all_predictions),
            "run_start": run_start.isoformat(),
            "run_end": datetime.now().isoformat(),
        },
        "token_stats": {
            "total_tokens": total_tokens,
            "input_tokens": sum(m.get("input_tokens", 0) for m in all_metadata),
            "output_tokens": sum(m.get("output_tokens", 0) for m in all_metadata),
            "avg_tokens_per_ema": total_tokens / len(all_predictions) if all_predictions else 0,
            "total_tool_calls": total_tool_calls,
            "avg_tool_calls_per_ema": total_tool_calls / len(all_predictions) if all_predictions else 0,
        },
        "metrics": metrics,
        "predictions": all_predictions,
        "ground_truths": all_ground_truths,
        "metadata": all_metadata,
    }

    timestamp = run_start.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"v4_results_{timestamp}.json"
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
    memory_dir: Path | None = None,
    backend: str = "cc",
    processed_dir: Path | None = None,
    feedback_mode: str = "real",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Process all EMA entries for one user in chronological order.

    The agent accumulates a per-user session memory across EMA entries.
    After each prediction, a record is appended to the user's memory file
    (outputs/agentic_pilot/memory/user_{pid}.md). The next EMA prediction
    includes the growing memory, so the agent genuinely learns this person
    over the study period — not just cold-start each time.

    Feedback modes control what enters the memory:
      "real":   diary + receptivity only (ER_desire + INT_availability).
                Simulates real-world deployment with no full EMA battery.
      "oracle": diary + receptivity + raw PA/NA scores.
                Research upper bound showing ceiling with full affect feedback.

    Args:
        study_id: Study_ID of the user.
        ema_df: Full EMA DataFrame.
        baseline_df: Baseline trait DataFrame.
        loader: DataLoader instance.
        query_engine: SensingQueryEngine instance (used only by api backend).
        model: Model ID / alias.
        max_tool_calls: Max tool calls per prediction.
        dry_run: If True, only process dry_run_limit EMA entries.
        dry_run_limit: Number of entries to process in dry run.
        checkpoints_dir: Directory for per-EMA checkpoints.
        logger: Logger instance.
        memory_dir: Directory to write per-user growing memory docs.
        backend: "cc" (Claude Code subprocess) or "api" (Anthropic SDK).
        processed_dir: Path to data/processed/ (required for cc backend).
        feedback_mode: "real" or "oracle" (controls session memory content).

    Returns:
        Tuple of (predictions, ground_truths, metadata) lists.
    """
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

    total_ema = len(user_ema)

    # Diary-present filter: for apple-to-apple comparison with CALLM (diary-based baseline),
    # only evaluate on EMA entries where the participant also wrote a diary entry.
    # This ensures the evaluation set is identical for all methods.
    diary_mask = (
        user_ema["emotion_driver"].notna()
        & (user_ema["emotion_driver"].astype(str).str.strip() != "")
        & (user_ema["emotion_driver"].astype(str).str.lower() != "nan")
    )
    user_ema_with_diary = user_ema[diary_mask]

    if user_ema_with_diary.empty:
        logger.warning(
            f"User {study_id}: no diary-present EMA entries found "
            f"(diary_only=True). Falling back to all {total_ema} entries."
        )
        eval_ema = user_ema
    else:
        eval_ema = user_ema_with_diary
        logger.info(
            f"  Diary filter: {len(eval_ema)}/{total_ema} EMA entries have diary text"
        )

    entries = list(eval_ema.iterrows())
    if dry_run:
        entries = entries[:dry_run_limit]
        logger.info(f"  [DRY RUN] Limiting to {dry_run_limit} EMA entries")

    logger.info(f"  Profile: {profile.to_text()}")
    logger.info(f"  Memory doc: {'loaded' if memory_doc else 'not found'}")
    logger.info(f"  EMA entries to process: {len(entries)}")

    # Checkpoint file for this user
    checkpoint_path = checkpoints_dir / f"user_{study_id}_v4.jsonl"

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
        # Dry run: skip actual LLM calls
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

    # Per-user session memory: grows as we process each EMA entry chronologically.
    # The agent sees the full accumulated memory at the start of each prediction,
    # enabling genuine longitudinal learning about this person.
    pid_str = str(study_id).zfill(3)
    session_memory_path = (memory_dir or checkpoints_dir) / f"user_{pid_str}_session.md"
    session_memory = _load_session_memory(session_memory_path)
    if session_memory:
        logger.info(f"  Session memory: loaded {len(session_memory.splitlines())} lines")
    else:
        _init_session_memory(session_memory_path, study_id, feedback_mode)
        session_memory = _load_session_memory(session_memory_path)

    # Create agent (one per user; stateless across EMA calls — statefulness
    # is managed externally via session_memory passed at each predict() call)
    if backend == "cc":
        from src.agent.cc_agent import ClaudeCodeAgent
        agent = ClaudeCodeAgent(
            study_id=study_id,
            profile=profile,
            memory_doc=memory_doc,
            processed_dir=processed_dir or PROJECT_ROOT / "data" / "processed",
            model=model,
            max_turns=max_tool_calls + 4,  # extra turns for reasoning + final prediction
            python_bin=CC_PYTHON_BIN,
        )
    else:
        from src.agent.agentic_sensing import AgenticSensingAgent
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

        # Run prediction (passes accumulated session memory as longitudinal context)
        t0 = time.time()
        try:
            pred = agent.predict(
                ema_row=ema_row,
                diary_text=diary_text,
                session_memory=session_memory,
            )
        except Exception as exc:
            logger.error(f"    Prediction error for {ts}: {exc}")
            pred = agent._fallback_prediction()
            pred["_error"] = str(exc)
            pred["_version"] = "v4"
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
            "input_tokens": pred.get("_input_tokens", 0),
            "output_tokens": pred.get("_output_tokens", 0),
            "total_tokens": pred.get("_total_tokens", 0),
        }

        predictions.append(pred)
        ground_truths.append(gt)
        metadata.append(meta)

        # Update session memory with receptivity outcome (+ PA/NA if oracle mode).
        # This grows the agent's longitudinal user model for the next EMA entry.
        session_memory = _update_session_memory(
            path=session_memory_path,
            ts=ts,
            ema_row=ema_row,
            pred=pred,
            feedback_mode=feedback_mode,
        )

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

def _load_session_memory(path: Path) -> str:
    """Load the current session memory document for a user, or return empty string."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _init_session_memory(path: Path, study_id: int, feedback_mode: str) -> None:
    """Initialize a fresh session memory file for a user."""
    mode_desc = (
        "Feedback: diary text + receptivity (ER_desire + INT_availability) only."
        if feedback_mode == "real"
        else "Feedback: diary text + receptivity + raw PA/NA scores (oracle/upper-bound)."
    )
    path.write_text(
        f"# Session Memory — Participant {str(study_id).zfill(3)}\n\n"
        f"Mode: {feedback_mode}\n"
        f"{mode_desc}\n\n"
        f"## EMA History (accumulated chronologically)\n\n",
        encoding="utf-8",
    )


def _update_session_memory(
    path: Path,
    ts: str,
    ema_row: pd.Series,
    pred: dict[str, Any],
    feedback_mode: str,
) -> str:
    """Append the outcome of the current EMA prediction to the session memory file.

    Only feeds back what a deployed JITAI system would observe:
    - 'real' mode:   diary text + receptivity (ER_desire + INT_availability)
    - 'oracle' mode: diary text + receptivity + raw PA/NA scores

    NEVER writes Individual_level_* state labels, which are prediction targets.

    Returns:
        Updated session memory text.
    """
    er = ema_row.get("ER_desire")
    avail = str(ema_row.get("INT_availability", "") or "").lower().strip()
    try:
        er_val = float(er) if er is not None and str(er) != "nan" else None
    except (ValueError, TypeError):
        er_val = None

    er_high = (er_val is not None) and (er_val >= 5)
    is_avail = avail in ("yes", "1", "true")
    receptive = er_high and is_avail

    er_str = f"{er_val:.0f}" if er_val is not None else "?"
    rec_str = "receptive=YES" if receptive else "receptive=no"

    diary = str(ema_row.get("emotion_driver", "") or "").strip()
    diary_str = f'  Diary: "{diary[:120]}"\n' if diary and diary.lower() != "nan" else ""

    pred_avail = pred.get("INT_availability", "?")
    pred_confidence = pred.get("confidence", "?")

    entry = (
        f"### {ts}\n"
        f"  ER_desire={er_str}, INT_availability={avail} → {rec_str}\n"
        f"{diary_str}"
        f"  Agent predicted: INT_availability={pred_avail}, confidence={pred_confidence}\n"
    )

    if feedback_mode == "oracle":
        pa = ema_row.get("PANAS_Pos")
        na = ema_row.get("PANAS_Neg")
        try:
            pa_val = float(pa) if pa is not None and str(pa) != "nan" else None
            na_val = float(na) if na is not None and str(na) != "nan" else None
        except (ValueError, TypeError):
            pa_val = na_val = None
        pa_str = f"PA={pa_val:.0f}" if pa_val is not None else "PA=?"
        na_str = f"NA={na_val:.0f}" if na_val is not None else "NA=?"
        entry += f"  [oracle] Actual: {pa_str}, {na_str}\n"

    entry += "\n"

    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)

    return _load_session_memory(path)


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
        "reasoning": "[DRY RUN] Placeholder V4 prediction",
        "confidence": 0.5,
        "_version": "v4",
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
    all_metadata: list[dict] | None = None,
) -> None:
    """Print evaluation summary to stdout."""
    elapsed = (datetime.now() - run_start).total_seconds()
    print(f"\n{'='*65}")
    print("V4 AGENTIC SENSING AGENT — EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"Model:           {model}")
    print(f"Predictions:     {n_predictions}")
    print(f"Elapsed:         {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    if all_metadata:
        total_tool_calls = sum(m.get("n_tool_calls", 0) for m in all_metadata)
        total_tokens = sum(m.get("total_tokens", 0) for m in all_metadata)
        input_tokens = sum(m.get("input_tokens", 0) for m in all_metadata)
        output_tokens = sum(m.get("output_tokens", 0) for m in all_metadata)
        avg_tool_calls = total_tool_calls / n_predictions if n_predictions else 0
        avg_tokens = total_tokens / n_predictions if n_predictions else 0
        print(f"\nToken Usage:")
        print(f"  Total tokens:    {total_tokens:,}  ({input_tokens:,} in / {output_tokens:,} out)")
        print(f"  Avg/EMA entry:   {avg_tokens:.0f} tokens")
        print(f"  Avg tool calls:  {avg_tool_calls:.1f}")

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
        description="Run V4 agentic sensing agent evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/run_agentic_pilot.py --users 71,164
  python scripts/run_agentic_pilot.py --users all --model claude-haiku-4-5-20251001
  python scripts/run_agentic_pilot.py --users 71 --dry-run
  python scripts/run_agentic_pilot.py --users 71,164 --max-tool-calls 5 --output-dir outputs/v4_short
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
        "--backend",
        type=str,
        choices=["cc", "api"],
        default=DEFAULT_BACKEND,
        help=(
            "Inference backend: 'cc' = claude --print subprocess (Claude Max subscription, recommended), "
            "'api' = Anthropic SDK (bills against ANTHROPIC_API_KEY). Default: cc"
        ),
    )
    parser.add_argument(
        "--feedback-mode",
        type=str,
        choices=["real", "oracle"],
        default=DEFAULT_FEEDBACK_MODE,
        help=(
            "Session memory feedback mode: "
            "'real' = diary + receptivity only (ER_desire + INT_availability), simulates real JITAI deployment; "
            "'oracle' = adds raw PA/NA scores, shows upper bound with full affect battery. Default: real"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model ID or alias (default: {DEFAULT_MODEL}). For cc backend, use alias like 'sonnet'.",
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
