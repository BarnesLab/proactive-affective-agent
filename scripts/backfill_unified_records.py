#!/usr/bin/env python3
"""Backfill unified JSONL records from existing checkpoint + trace files.

V1/V3/CALLM traces were generated before unified record saving was added.
This script reconstructs the unified records by merging:
  - Checkpoint files (predictions, ground truths, metadata)
  - Trace files (full prompts, responses, system prompts, etc.)

Usage:
    python scripts/backfill_unified_records.py [--versions callm,v1,v3] [--output-dir outputs/pilot]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _ema_slot(timestamp_str: str) -> str:
    """Classify an EMA timestamp into morning/afternoon/evening."""
    try:
        hour = int(timestamp_str.split(" ")[1].split(":")[0])
        if hour < 12:
            return "morning"
        if hour < 17:
            return "afternoon"
        return "evening"
    except Exception:
        return "unknown"


def _build_ema_timestamp_lookup() -> dict[int, list[str]]:
    """Build user_id -> sorted list of timestamps from EMA test splits."""
    splits_dir = Path("data/processed/splits")
    all_dfs = []
    for g in range(1, 6):
        fp = splits_dir / f"group_{g}_test.csv"
        if fp.exists():
            all_dfs.append(pd.read_csv(fp))
    if not all_dfs:
        return {}
    combined = pd.concat(all_dfs, ignore_index=True)
    lookup: dict[int, list[str]] = defaultdict(list)
    for _, row in combined.iterrows():
        uid = int(row["Study_ID"])
        lookup[uid].append(str(row["timestamp_local"]))
    for uid in lookup:
        lookup[uid].sort()
    return dict(lookup)


def backfill_version(version: str, output_dir: Path, ema_lookup: dict[int, list[str]] | None = None) -> int:
    """Backfill unified records for a single version.

    Returns number of records written.
    """
    checkpoint_dir = output_dir / "checkpoints"
    trace_dir = output_dir / "traces"

    # Find all user checkpoints for this version
    checkpoint_files = sorted(checkpoint_dir.glob(f"{version}_user*_checkpoint.json"))
    if not checkpoint_files:
        logger.warning(f"  No checkpoints found for {version}")
        return 0

    total = 0

    for cp_path in checkpoint_files:
        with open(cp_path) as f:
            cp = json.load(f)

        sid = cp["current_user"]
        predictions = cp["predictions"]
        ground_truths = cp["ground_truths"]
        metadata = cp["metadata"]

        # Output JSONL path
        jsonl_path = output_dir / f"{version}_user{sid}_records.jsonl"
        if jsonl_path.exists():
            existing_lines = sum(1 for _ in open(jsonl_path))
            if existing_lines >= len(predictions):
                logger.info(f"  User {sid}: already has {existing_lines} records, skipping")
                continue
            else:
                logger.info(f"  User {sid}: has {existing_lines}/{len(predictions)} records, regenerating")

        records = []
        for i, (pred, gt, meta) in enumerate(zip(predictions, ground_truths, metadata)):
            # Load corresponding trace file
            trace_path = trace_dir / f"{version}_user{sid}_entry{i}.json"
            trace = {}
            if trace_path.exists():
                with open(trace_path) as f:
                    trace = json.load(f)

            # Reconstruct timestamp from EMA data or metadata
            date_str = meta.get("date", "")
            timestamp_str = ""

            # Look up timestamp from EMA data
            if ema_lookup and sid in ema_lookup and i < len(ema_lookup[sid]):
                timestamp_str = ema_lookup[sid][i]

            # Try to get timestamp from the full_prompt (it contains date info)
            full_prompt = trace.get("_full_prompt", "")
            full_response = trace.get("_full_response", trace.get("_final_response", ""))

            # Determine diary presence
            emotion_driver = trace.get("_emotion_driver", "")
            has_diary = trace.get("_has_diary", bool(emotion_driver and emotion_driver.strip()))
            diary_length = trace.get("_diary_length", len(emotion_driver) if has_diary else 0)

            record = {
                # Identity
                "study_id": sid,
                "entry_idx": i,
                "date_local": date_str,
                "timestamp_local": timestamp_str,
                "ema_slot": _ema_slot(timestamp_str) if timestamp_str else "unknown",
                "version": version,
                "model": trace.get("_model", "sonnet"),
                # Data availability
                "modalities_available": [],
                "modalities_missing": [],
                "has_diary": has_diary,
                "diary_length": diary_length if has_diary else None,
                "emotion_driver": emotion_driver,
                # Prediction & ground truth
                "prediction": pred,
                "ground_truth": gt,
                "reasoning": pred.get("reasoning", trace.get("_reasoning", "")),
                "confidence": pred.get("confidence", 0.0),
                # Context given to LLM
                "prompt_length": trace.get("_prompt_length"),
                "full_prompt": full_prompt,
                "system_prompt": trace.get("_system_prompt", ""),
                "sensing_summary": trace.get("_sensing_summary", ""),
                "rag_cases": trace.get("_rag_top5", []),
                "memory_excerpt": trace.get("_memory_excerpt", ""),
                "trait_summary": trace.get("_trait_summary", ""),
                # LLM response
                "full_response": full_response,
                # Agentic-specific (null for structured versions)
                "n_tool_calls": trace.get("_n_tool_calls"),
                "n_rounds": trace.get("_n_rounds"),
                "tool_calls": trace.get("_tool_calls"),
                "conversation_length": trace.get("_conversation_length"),
                # Performance
                "elapsed_seconds": None,  # not available in old traces
                "llm_calls": trace.get("_llm_calls", 1),
                # Token usage (not available in old traces for CLI-based versions)
                "input_tokens": trace.get("_input_tokens", 0),
                "output_tokens": trace.get("_output_tokens", 0),
                "total_tokens": trace.get("_total_tokens", 0),
                "cost_usd": trace.get("_cost_usd", 0),
                # Backfill metadata
                "_backfilled": True,
            }
            records.append(record)

        # Write JSONL
        with open(jsonl_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")

        logger.info(f"  User {sid}: wrote {len(records)} records -> {jsonl_path.name}")
        total += len(records)

    return total


def main():
    parser = argparse.ArgumentParser(description="Backfill unified JSONL records from existing traces")
    parser.add_argument("--versions", default="callm,v1,v3", help="Comma-separated versions to backfill")
    parser.add_argument("--output-dir", default="outputs/pilot", help="Pilot output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    versions = [v.strip() for v in args.versions.split(",")]

    logger.info(f"Backfilling unified records for: {', '.join(versions)}")
    logger.info(f"Output dir: {output_dir}")

    logger.info("Building EMA timestamp lookup...")
    ema_lookup = _build_ema_timestamp_lookup()
    logger.info(f"  Found timestamps for {len(ema_lookup)} users")

    grand_total = 0
    for version in versions:
        logger.info(f"\n--- {version.upper()} ---")
        count = backfill_version(version, output_dir, ema_lookup)
        grand_total += count

    logger.info(f"\nDone. Total records backfilled: {grand_total}")


if __name__ == "__main__":
    main()
