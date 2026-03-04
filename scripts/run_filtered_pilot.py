#!/usr/bin/env python3
"""Run V5/V6 agentic agents with filtered behavioral narratives.

V5 = filtered narrative + MCP tools (sensing-only, no diary)
V6 = filtered narrative + diary + MCP tools (multimodal)

Both use claude --print subprocess (Claude Max subscription, FREE) with
pre-computed daily behavioral narratives from data/processed/filtered/.

Supports parallel execution: each (version, user) pair runs as an independent
subprocess via concurrent.futures, maximizing throughput within rate limits.

Usage:
    # Single version, single user
    python scripts/run_filtered_pilot.py --version v6 --users 71

    # Both versions, all 5 pilot users, parallel (10 concurrent workers)
    python scripts/run_filtered_pilot.py --version v5,v6 --users 71,119,164,310,458 --workers 5

    # Dry run (no LLM calls)
    python scripts/run_filtered_pilot.py --version v5,v6 --users 71 --dry-run

    # Limit entries per user (smoke test)
    python scripts/run_filtered_pilot.py --version v6 --users 71 --dry-run-limit 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.loader import DataLoader
from src.data.preprocessing import get_user_trait_profile
from src.evaluation.metrics import compute_all


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "filtered_pilot"
DEFAULT_MODEL = "sonnet"
DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_WORKERS = 1  # sequential is safer; avoids multi-process rate limit amplification

DELAY_BETWEEN_EMAS = 1.0  # seconds between EMA entries within a user


# ---------------------------------------------------------------------------
# Per-user worker (runs in subprocess for parallelism)
# ---------------------------------------------------------------------------

def _run_single_user(
    version: str,
    study_id: int,
    data_dir: str,
    processed_dir: str,
    filtered_data_dir: str,
    output_dir: str,
    model: str,
    max_tool_calls: int,
    dry_run: bool,
    dry_run_limit: int | None,
) -> dict[str, Any]:
    """Process all EMA entries for one (version, user) pair.

    Designed to run in a separate process via ProcessPoolExecutor.
    Returns a dict with predictions, ground_truths, metadata.
    """
    # Re-import inside subprocess
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{version.upper()}|{study_id}] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(f"{version}_{study_id}")

    from src.data.loader import DataLoader
    from src.data.preprocessing import get_user_trait_profile
    from src.data.schema import UserProfile
    from src.agent.cc_agent import AgenticCCAgent
    from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS
    from src.utils.rate_limit import RateLimitError

    loader = DataLoader(data_dir=Path(data_dir))

    # Load EMA data
    ema_df = loader.load_all_ema()
    ema_df = _apply_midpoint_thresholds(ema_df)

    try:
        baseline_df = loader.load_baseline()
    except FileNotFoundError:
        baseline_df = pd.DataFrame()

    profile = (
        get_user_trait_profile(baseline_df, study_id)
        if not baseline_df.empty
        else UserProfile(study_id=study_id)
    )
    memory_doc = loader.load_memory_for_user(study_id)

    # Get this user's EMA entries (diary-present only, for fair comparison)
    user_ema = ema_df[ema_df["Study_ID"] == study_id].sort_values("timestamp_local")
    if user_ema.empty:
        log.warning(f"No EMA entries found")
        return {"version": version, "study_id": study_id, "predictions": [], "ground_truths": [], "metadata": []}

    diary_mask = (
        user_ema["emotion_driver"].notna()
        & (user_ema["emotion_driver"].astype(str).str.strip() != "")
        & (user_ema["emotion_driver"].astype(str).str.lower() != "nan")
    )
    eval_ema = user_ema[diary_mask] if diary_mask.any() else user_ema
    log.info(f"Diary filter: {len(eval_ema)}/{len(user_ema)} entries")

    entries = list(eval_ema.iterrows())
    if dry_run_limit is not None:
        entries = entries[:dry_run_limit]
        log.info(f"Limiting to {dry_run_limit} entries")

    # Checkpoint setup
    out_dir = Path(output_dir)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = out_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / f"user_{study_id}_{version}.jsonl"

    # Load existing checkpoint
    processed_timestamps: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    processed_timestamps.add(rec.get("timestamp_local", ""))
                except json.JSONDecodeError:
                    pass
        log.info(f"Checkpoint: {len(processed_timestamps)} already processed")

    # Session memory
    pid_str = str(study_id).zfill(3)
    session_memory_path = memory_dir / f"user_{pid_str}_{version}_session.md"
    if session_memory_path.exists():
        session_memory = session_memory_path.read_text(encoding="utf-8")
        log.info(f"Session memory: {len(session_memory.splitlines())} lines")
    else:
        session_memory_path.write_text(
            f"# Session Memory — Participant {pid_str} ({version.upper()})\n\n"
            "Feedback: diary text + receptivity (ER_desire raw + INT_availability).\n\n"
            "## EMA History (accumulated chronologically)\n\n",
            encoding="utf-8",
        )
        session_memory = session_memory_path.read_text(encoding="utf-8")

    # Create agent
    mode = "filtered_sensing" if version == "v5" else "filtered_multimodal"
    agent = AgenticCCAgent(
        study_id=study_id,
        profile=profile,
        memory_doc=memory_doc,
        processed_dir=Path(processed_dir),
        model=model,
        max_turns=max_tool_calls + 4,
        mode=mode,
        filtered_data_dir=Path(filtered_data_dir),
    )

    predictions: list[dict] = []
    ground_truths: list[dict] = []
    metadata: list[dict] = []

    if dry_run:
        for idx, (_, ema_row) in enumerate(entries):
            ts = str(ema_row.get("timestamp_local", ""))
            pred = _dry_run_prediction(version)
            gt = _extract_ground_truth(ema_row)
            meta = {"study_id": study_id, "timestamp_local": ts, "version": version}
            predictions.append(pred)
            ground_truths.append(gt)
            metadata.append(meta)
            log.info(f"[DRY RUN] Entry {idx + 1}: {ts}")
        return {"version": version, "study_id": study_id, "predictions": predictions, "ground_truths": ground_truths, "metadata": metadata}

    for idx, (_, ema_row) in enumerate(entries):
        ts = str(ema_row.get("timestamp_local", ""))
        ema_date = str(ema_row.get("date_local", ""))

        if ts in processed_timestamps:
            continue

        diary_text = str(ema_row.get("emotion_driver", ""))
        if diary_text.lower() == "nan" or not diary_text.strip():
            diary_text = None

        log.info(f"Entry {idx + 1}/{len(entries)}: {ts}")

        t0 = time.time()
        try:
            pred = agent.predict(
                ema_row=ema_row,
                diary_text=diary_text if version == "v6" else None,
                session_memory=session_memory,
            )
        except RateLimitError as exc:
            log.error(f"Rate limit hit — stopping user {study_id}: {exc}")
            break
        except Exception as exc:
            log.error(f"Prediction error: {exc}")
            pred = agent._fallback_prediction()
            pred["_error"] = str(exc)
            pred["_version"] = version
            pred["_n_tool_calls"] = 0

        elapsed = time.time() - t0
        log.info(f"Done in {elapsed:.1f}s | tools={pred.get('_n_tool_calls', '?')} | conf={pred.get('confidence', '?')}")

        # Skip checkpointing obvious fallback predictions (empty output + low confidence)
        is_fallback = (
            pred.get("confidence") == 0.1
            and "fallback" in str(pred.get("reasoning", "")).lower()
        )
        if is_fallback:
            log.warning(f"Skipping fallback prediction for {ts} (not checkpointed)")
            continue

        gt = _extract_ground_truth(ema_row)
        meta = {
            "study_id": study_id,
            "timestamp_local": ts,
            "date_local": ema_date,
            "version": version,
            "n_tool_calls": pred.get("_n_tool_calls", 0),
            "elapsed_seconds": round(elapsed, 1),
        }

        predictions.append(pred)
        ground_truths.append(gt)
        metadata.append(meta)

        # Update session memory
        session_memory = _update_session_memory(
            path=session_memory_path,
            ts=ts,
            ema_row=ema_row,
            pred=pred,
        )

        # Save checkpoint
        checkpoint_record = {
            "study_id": study_id,
            "timestamp_local": ts,
            "date_local": ema_date,
            "version": version,
            "elapsed_seconds": round(elapsed, 1),
            "n_tool_calls": pred.get("_n_tool_calls", 0),
            "prediction": pred,
            "ground_truth": gt,
        }
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps(checkpoint_record, default=str) + "\n")

        if idx < len(entries) - 1:
            time.sleep(DELAY_BETWEEN_EMAS)

    log.info(f"Completed: {len(predictions)} predictions")
    return {
        "version": version,
        "study_id": study_id,
        "predictions": predictions,
        "ground_truths": ground_truths,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_midpoint_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Override Individual_level_ER_desire_State using scale midpoint (>= 5)."""
    if "ER_desire" not in df.columns:
        return df
    df = df.copy()
    er = pd.to_numeric(df["ER_desire"], errors="coerce")
    df["Individual_level_ER_desire_State"] = (er >= 5).where(er.notna(), other=None)
    return df


def _extract_ground_truth(ema_row: pd.Series) -> dict[str, Any]:
    """Extract all prediction targets as a ground-truth dict."""
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


def _update_session_memory(
    path: Path, ts: str, ema_row: pd.Series, pred: dict[str, Any]
) -> str:
    """Append receptivity feedback to session memory. No data leakage."""
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
        "\n"
    )

    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)

    return path.read_text(encoding="utf-8")


def _dry_run_prediction(version: str) -> dict[str, Any]:
    """Return a placeholder prediction for dry-run testing."""
    from src.utils.mappings import BINARY_STATE_TARGETS
    pred: dict[str, Any] = {
        "PANAS_Pos": 15.0,
        "PANAS_Neg": 8.0,
        "ER_desire": 3.0,
        "INT_availability": "yes",
        "reasoning": f"[DRY RUN] Placeholder {version.upper()} prediction",
        "confidence": 0.5,
        "_version": version,
        "_n_tool_calls": 0,
        "_dry_run": True,
    }
    for target in BINARY_STATE_TARGETS:
        pred[target] = False
    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    versions = [v.strip() for v in args.version.split(",")]
    for v in versions:
        if v not in ("v5", "v6"):
            logger.error(f"Invalid version: {v}. Must be v5 or v6.")
            sys.exit(1)

    if args.users.lower() == "all":
        pilot_user_ids = [71, 119, 164, 310, 458]
    else:
        pilot_user_ids = [int(x.strip()) for x in args.users.split(",")]

    logger.info(f"V5/V6 Filtered Agentic Pilot")
    logger.info(f"  Versions:  {versions}")
    logger.info(f"  Users:     {pilot_user_ids}")
    logger.info(f"  Model:     {args.model}")
    logger.info(f"  Workers:   {args.workers}")
    logger.info(f"  Dry run:   {args.dry_run}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    processed_dir = data_dir / "processed" / "hourly"
    filtered_data_dir = data_dir / "processed" / "filtered"

    if not processed_dir.exists():
        logger.error(f"Processed hourly dir not found: {processed_dir}")
        sys.exit(1)
    if not filtered_data_dir.exists():
        logger.error(f"Filtered data dir not found: {filtered_data_dir}")
        sys.exit(1)

    # Validate users exist
    loader = DataLoader(data_dir=data_dir)
    ema_df = loader.load_all_ema()
    all_user_ids = set(ema_df["Study_ID"].unique())
    missing = [u for u in pilot_user_ids if u not in all_user_ids]
    if missing:
        logger.warning(f"Users not found in EMA data: {missing}")
        pilot_user_ids = [u for u in pilot_user_ids if u not in missing]

    # Build (version, user) job list
    jobs = [(v, uid) for v in versions for uid in pilot_user_ids]
    logger.info(f"Total jobs: {len(jobs)} ({len(versions)} versions × {len(pilot_user_ids)} users)")

    run_start = datetime.now()
    all_results: list[dict] = []

    dry_run_limit = args.dry_run_limit if args.dry_run_limit else None

    if args.workers <= 1 or len(jobs) == 1:
        # Sequential execution
        for version, uid in jobs:
            result = _run_single_user(
                version=version,
                study_id=uid,
                data_dir=str(data_dir),
                processed_dir=str(processed_dir),
                filtered_data_dir=str(filtered_data_dir),
                output_dir=str(output_dir),
                model=args.model,
                max_tool_calls=args.max_tool_calls,
                dry_run=args.dry_run,
                dry_run_limit=dry_run_limit,
            )
            all_results.append(result)
    else:
        # Parallel execution
        logger.info(f"Launching {min(args.workers, len(jobs))} parallel workers...")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            futures = {}
            for version, uid in jobs:
                future = executor.submit(
                    _run_single_user,
                    version=version,
                    study_id=uid,
                    data_dir=str(data_dir),
                    processed_dir=str(processed_dir),
                    filtered_data_dir=str(filtered_data_dir),
                    output_dir=str(output_dir),
                    model=args.model,
                    max_tool_calls=args.max_tool_calls,
                    dry_run=args.dry_run,
                    dry_run_limit=dry_run_limit,
                )
                futures[future] = (version, uid)

            for future in as_completed(futures):
                version, uid = futures[future]
                try:
                    result = future.result()
                    n = len(result.get("predictions", []))
                    logger.info(f"[DONE] {version.upper()} user {uid}: {n} predictions")
                    all_results.append(result)
                except Exception as exc:
                    logger.error(f"[FAIL] {version.upper()} user {uid}: {exc}")

    # Aggregate results per version
    for version in versions:
        version_results = [r for r in all_results if r["version"] == version]
        all_preds = []
        all_gts = []
        all_meta = []
        for r in version_results:
            all_preds.extend(r.get("predictions", []))
            all_gts.extend(r.get("ground_truths", []))
            all_meta.extend(r.get("metadata", []))

        if all_preds:
            metrics = compute_all(all_preds, all_gts, metadata=all_meta)
        else:
            metrics = {}

        _print_version_summary(version, metrics, len(all_preds), all_meta)

        # Save per-version results
        timestamp = run_start.strftime("%Y%m%d_%H%M%S")
        results_path = output_dir / f"{version}_results_{timestamp}.json"
        results_path.write_text(json.dumps({
            "run_config": {
                "version": version,
                "model": args.model,
                "users": pilot_user_ids,
                "n_entries": len(all_preds),
                "workers": args.workers,
                "dry_run": args.dry_run,
                "run_start": run_start.isoformat(),
                "run_end": datetime.now().isoformat(),
            },
            "metrics": metrics,
            "predictions": all_preds,
            "ground_truths": all_gts,
            "metadata": all_meta,
        }, indent=2, default=str))
        logger.info(f"Results saved: {results_path}")

    # Also save combined checkpoint files compatible with evaluate_pilot.py
    _save_combined_checkpoints(all_results, output_dir)

    elapsed = (datetime.now() - run_start).total_seconds()
    logger.info(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)")


def _save_combined_checkpoints(all_results: list[dict], output_dir: Path) -> None:
    """Save checkpoints in the format evaluate_pilot.py expects:
    {version}_user{uid}_checkpoint.json with predictions/ground_truths arrays.
    """
    # Also write to the standard pilot checkpoint dir for evaluate_pilot.py
    pilot_cp_dir = PROJECT_ROOT / "outputs" / "pilot" / "checkpoints"
    pilot_cp_dir.mkdir(parents=True, exist_ok=True)

    for result in all_results:
        version = result["version"]
        uid = result["study_id"]
        preds = result.get("predictions", [])
        gts = result.get("ground_truths", [])
        meta = result.get("metadata", [])

        if not preds:
            continue

        # Clean predictions: keep reasoning/confidence, remove internal _ keys
        clean_preds = []
        for p in preds:
            cp = {k: v for k, v in p.items() if not k.startswith("_")}
            # Preserve reasoning from _ key if not already present
            if "reasoning" not in cp and "_reasoning" in p:
                cp["reasoning"] = p["_reasoning"]
            clean_preds.append(cp)

        checkpoint = {
            "version": version,
            "n_entries": len(clean_preds),
            "current_user": uid,
            "current_entry": len(clean_preds) - 1,
            "predictions": clean_preds,
            "ground_truths": gts,
            "metadata": meta,
        }

        path = pilot_cp_dir / f"{version}_user{uid}_checkpoint.json"
        path.write_text(json.dumps(checkpoint, default=str))


def _print_version_summary(
    version: str, metrics: dict, n_predictions: int, all_metadata: list[dict]
) -> None:
    """Print evaluation summary for a single version."""
    print(f"\n{'='*65}")
    print(f"{version.upper()} FILTERED AGENTIC AGENT — SUMMARY")
    print(f"{'='*65}")
    print(f"Predictions: {n_predictions}")

    if all_metadata:
        total_tool_calls = sum(m.get("n_tool_calls", 0) for m in all_metadata)
        avg_tool_calls = total_tool_calls / n_predictions if n_predictions else 0
        total_elapsed = sum(m.get("elapsed_seconds", 0) for m in all_metadata)
        print(f"Total elapsed: {total_elapsed:.0f}s")
        print(f"Avg tool calls/entry: {avg_tool_calls:.1f}")

    agg = metrics.get("aggregate", {})
    if agg.get("mean_mae") is not None:
        print(f"Mean MAE:  {agg['mean_mae']:.4f}")
    if agg.get("mean_balanced_accuracy") is not None:
        print(f"Mean BA:   {agg['mean_balanced_accuracy']:.4f}")
    if agg.get("mean_f1") is not None:
        print(f"Mean F1:   {agg['mean_f1']:.4f}")
    print(f"{'='*65}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5/V6 filtered agentic agent evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/run_filtered_pilot.py --version v6 --users 71
  python scripts/run_filtered_pilot.py --version v5,v6 --users 71,119,164,310,458 --workers 5
  python scripts/run_filtered_pilot.py --version v5,v6 --users 71 --dry-run
""",
    )
    parser.add_argument("--version", type=str, default="v5,v6",
                        help="Comma-separated versions to run (v5, v6, or v5,v6). Default: v5,v6")
    parser.add_argument("--users", type=str, default="71,119,164,310,458",
                        help="Comma-separated Study_IDs or 'all'. Default: 71,119,164,310,458")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model alias (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS,
                        help=f"Max tool calls per EMA entry (default: {DEFAULT_MAX_TOOL_CALLS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Max parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, use placeholder predictions")
    parser.add_argument("--dry-run-limit", type=int, default=None,
                        help="Limit entries per user (for smoke testing with real LLM)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: data/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging")
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
