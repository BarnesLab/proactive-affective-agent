#!/usr/bin/env python3
"""Integration tests for V1-V4 + CALLM agent pipelines.

Tests each version end-to-end with real LLM calls on a small number of EMA entries,
saving detailed logs to test_logs/ for inspection.

Usage:
    # Test all versions (V1/V3/CALLM via claude CLI, V2/V4 via Anthropic SDK)
    python scripts/integration_test.py

    # Test specific versions
    python scripts/integration_test.py --versions v1,v2

    # Dry-run mode (no LLM calls)
    python scripts/integration_test.py --dry-run

    # Use specific user
    python scripts/integration_test.py --user 275

Cost note:
    V1/V3/CALLM use `claude -p` (Max subscription, no API cost).
    V2/V4 use the Anthropic Python SDK (requires ANTHROPIC_API_KEY, incurs API cost).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if present
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

import numpy as np
import pandas as pd

from src.data.schema import SensingDay, UserProfile
from src.sense.query_tools import SensingQueryEngine
from src.think.llm_client import ClaudeCodeClient
from src.think.parser import parse_prediction
from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEST_LOG_DIR = PROJECT_ROOT / "test_logs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "hourly"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

# Default test user with 4 sensing modalities + diary entries
DEFAULT_USER = 275

# Number of EMA entries to test per version (evenly sampled across date range)
N_TEST_ENTRIES = 10

# Expected prediction fields
EXPECTED_CONTINUOUS = list(CONTINUOUS_TARGETS.keys())
EXPECTED_BINARY = list(BINARY_STATE_TARGETS)
EXPECTED_FIELDS = EXPECTED_CONTINUOUS + EXPECTED_BINARY + ["INT_availability", "reasoning", "confidence"]


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"integration_test_{timestamp}.log"

    logger = logging.getLogger("integration_test")
    logger.setLevel(logging.DEBUG)

    # File handler — detailed
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(fh)

    # Console handler — summary
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(ch)

    # Also route library logs to file
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)

    logger.info(f"Log file: {log_file}")
    return logger


def load_test_data(user_id: int, n_entries: int = N_TEST_ENTRIES):
    """Load EMA test data and pick entries with diary text + diverse coverage.

    Selection strategy:
    - Sort by timestamp, then evenly sample across the full date range
    - This ensures coverage of early/mid/late study periods AND different time slots
    - When n_entries >= total available, return all entries
    """
    test_csv = SPLITS_DIR / "group_1_test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df = pd.read_csv(test_csv)
    user_df = df[df["Study_ID"] == user_id].copy()
    if user_df.empty:
        raise ValueError(f"No test data for user {user_id}")

    # Filter to entries with diary text
    diary_mask = (
        user_df["emotion_driver"].notna()
        & (user_df["emotion_driver"].astype(str) != "nan")
        & (user_df["emotion_driver"].str.strip() != "")
    )
    diary_df = user_df[diary_mask]

    if diary_df.empty:
        diary_df = user_df

    # Sort by timestamp for temporal ordering
    diary_df = diary_df.copy()
    diary_df["timestamp_local"] = pd.to_datetime(diary_df["timestamp_local"])
    diary_df = diary_df.sort_values("timestamp_local").reset_index(drop=True)

    # Return all if n_entries >= available
    if n_entries >= len(diary_df):
        return df, [diary_df.iloc[i] for i in range(len(diary_df))]

    # Evenly sample across the sorted entries (covers full date range + slot diversity)
    indices = np.linspace(0, len(diary_df) - 1, n_entries, dtype=int)
    # Ensure unique indices
    indices = sorted(set(indices))
    selected = [diary_df.iloc[i] for i in indices]

    return df, selected[:n_entries]


def build_sensing_day(user_id: int, date_str: str) -> SensingDay | None:
    """Try to build a SensingDay from Parquet data for V1/V3."""
    pid = f"{user_id:03d}"
    day = SensingDay(id_participant=pid, date=pd.to_datetime(date_str).date())

    found_any = False
    for mod in ["screen", "motion", "keyinput", "mus", "light"]:
        pq_path = PROCESSED_DIR / mod / f"{pid}_{mod}_hourly.parquet"
        if not pq_path.exists():
            continue
        try:
            df = pd.read_parquet(pq_path)
            # Filter to the date
            if "hour_local" in df.columns:
                df["hour_local"] = pd.to_datetime(df["hour_local"])
                day_df = df[df["hour_local"].dt.date == pd.to_datetime(date_str).date()]
            elif "date_local" in df.columns:
                day_df = df[df["date_local"] == date_str]
            else:
                continue

            if day_df.empty:
                continue

            found_any = True

            # Aggregate hourly data into daily summaries
            if mod == "screen":
                if "screen_on_min" in day_df.columns:
                    day.screen_minutes = float(day_df["screen_on_min"].sum())
                if "n_sessions" in day_df.columns:
                    day.screen_sessions = int(day_df["n_sessions"].sum())
            elif mod == "motion":
                for col, attr in [
                    ("stationary_min", "stationary_min"),
                    ("walking_min", "walking_min"),
                    ("automotive_min", "automotive_min"),
                    ("running_min", "running_min"),
                    ("cycling_min", "cycling_min"),
                ]:
                    if col in day_df.columns:
                        setattr(day, attr, float(day_df[col].sum()))
            elif mod == "keyinput":
                if "words_typed" in day_df.columns:
                    day.words_typed = int(day_df["words_typed"].sum())
                if "prop_positive" in day_df.columns:
                    day.prop_positive = float(day_df["prop_positive"].mean())
                if "prop_negative" in day_df.columns:
                    day.prop_negative = float(day_df["prop_negative"].mean())
            elif mod == "mus":
                if "mus_is_listening" in day_df.columns or "is_listening" in day_df.columns:
                    col = "mus_is_listening" if "mus_is_listening" in day_df.columns else "is_listening"
                    # Any hour of listening counts
        except Exception:
            continue

    return day if found_any else None


def validate_prediction(pred: dict, version: str, logger: logging.Logger) -> dict:
    """Validate a prediction dict and return a diagnostic report."""
    report = {
        "version": version,
        "valid": True,
        "issues": [],
        "field_coverage": {},
    }

    # Check required fields
    for field in EXPECTED_FIELDS:
        val = pred.get(field)
        report["field_coverage"][field] = val is not None
        if val is None and field not in ("reasoning",):
            report["issues"].append(f"Missing field: {field}")

    # Validate continuous ranges
    for target, (lo, hi) in CONTINUOUS_TARGETS.items():
        val = pred.get(target)
        if val is not None:
            try:
                v = float(val)
                if v < lo or v > hi:
                    report["issues"].append(f"{target}={v} out of range [{lo}, {hi}]")
            except (ValueError, TypeError):
                report["issues"].append(f"{target}={val} is not numeric")

    # Validate binary fields
    for target in BINARY_STATE_TARGETS:
        val = pred.get(target)
        if val is not None and not isinstance(val, bool) and val not in (True, False, 0, 1):
            report["issues"].append(f"{target}={val} is not boolean-like")

    # Validate INT_availability
    avail = pred.get("INT_availability")
    if avail is not None and avail not in ("yes", "no"):
        report["issues"].append(f"INT_availability='{avail}' not in ('yes', 'no')")

    # Validate confidence
    conf = pred.get("confidence")
    if conf is not None:
        try:
            c = float(conf)
            if c < 0 or c > 1:
                report["issues"].append(f"confidence={c} out of [0, 1]")
        except (ValueError, TypeError):
            report["issues"].append(f"confidence={conf} is not numeric")

    # Check for parse errors
    if pred.get("_parse_error"):
        report["issues"].append("Parse error occurred")
        report["valid"] = False

    if report["issues"]:
        report["valid"] = False
        for issue in report["issues"]:
            logger.warning(f"  [{version}] {issue}")
    else:
        logger.info(f"  [{version}] All fields valid")

    return report


# ---------------------------------------------------------------------------
# Per-version test runners
# ---------------------------------------------------------------------------

def test_v1(ema_row, sensing_day, llm_client, profile, logger, dry_run=False):
    """Test V1: Structured sensing-only pipeline."""
    logger.info("--- Testing V1 (Structured Sensing-Only) ---")
    from src.agent.structured import StructuredWorkflow

    wf = StructuredWorkflow(llm_client)
    date_str = str(ema_row.get("date_local", ""))

    t0 = time.time()
    result = wf.run(
        sensing_day=sensing_day,
        memory_doc="(Integration test — no memory document.)",
        profile=profile,
        date_str=date_str,
    )
    elapsed = time.time() - t0

    logger.info(f"  V1 completed in {elapsed:.1f}s")
    logger.debug(f"  V1 prompt length: {result.get('_prompt_length', '?')}")
    logger.debug(f"  V1 sensing summary: {result.get('_sensing_summary', '')[:200]}")
    logger.debug(f"  V1 full response: {result.get('_full_response', '')[:500]}")

    result["_elapsed_s"] = elapsed
    return result


def test_v3(ema_row, sensing_day, llm_client, profile, train_df, logger, dry_run=False):
    """Test V3: Structured multimodal (diary + sensing + RAG)."""
    logger.info("--- Testing V3 (Structured Multimodal + RAG) ---")
    from src.agent.structured_full import StructuredFullWorkflow
    from src.remember.retriever import MultiModalRetriever

    # Build retriever on training data
    retriever = MultiModalRetriever()
    retriever.fit(train_df, text_column="emotion_driver")
    logger.info(f"  V3 retriever fitted on {len(train_df)} training entries")

    wf = StructuredFullWorkflow(
        llm_client, retriever=retriever, study_id=int(ema_row.get("Study_ID", 0))
    )
    date_str = str(ema_row.get("date_local", ""))

    t0 = time.time()
    result = wf.run(
        ema_row=ema_row,
        sensing_day=sensing_day,
        memory_doc="(Integration test — no memory document.)",
        profile=profile,
        date_str=date_str,
    )
    elapsed = time.time() - t0

    logger.info(f"  V3 completed in {elapsed:.1f}s")
    logger.info(f"  V3 has_diary={result.get('_has_diary')}, diary_length={result.get('_diary_length')}")
    logger.debug(f"  V3 full response: {result.get('_full_response', '')[:500]}")

    result["_elapsed_s"] = elapsed
    return result


def test_callm(ema_row, llm_client, profile, train_df, logger, dry_run=False):
    """Test CALLM: Diary + TF-IDF RAG baseline."""
    logger.info("--- Testing CALLM (Diary + TF-IDF RAG) ---")
    from src.remember.retriever import TFIDFRetriever
    from src.think.prompts import build_trait_summary, callm_prompt

    # Build retriever
    retriever = TFIDFRetriever()
    retriever.fit(train_df, text_column="emotion_driver")

    study_id = int(ema_row.get("Study_ID", 0))
    emotion_driver = str(ema_row.get("emotion_driver", ""))
    date_str = str(ema_row.get("date_local", ""))

    # Search for similar cases
    rag_raw = retriever.search(emotion_driver, top_k=20, exclude_study_id=study_id)
    rag_examples = retriever.format_examples(rag_raw, max_examples=10)

    trait_text = build_trait_summary(profile)
    prompt = callm_prompt(
        emotion_driver=emotion_driver or "(No diary entry)",
        rag_examples=rag_examples,
        memory_doc="(Integration test — no memory document.)",
        trait_profile=trait_text,
        date_str=date_str,
    )

    logger.info(f"  CALLM prompt length: {len(prompt)}")
    logger.info(f"  CALLM diary: '{emotion_driver[:100]}'")
    logger.info(f"  CALLM RAG results: {len(rag_raw)}")

    t0 = time.time()
    raw_response = llm_client.generate(prompt=prompt)
    elapsed = time.time() - t0
    result = parse_prediction(raw_response)

    result["_version"] = "callm"
    result["_elapsed_s"] = elapsed
    result["_prompt_length"] = len(prompt)
    result["_emotion_driver"] = emotion_driver
    result["_full_response"] = raw_response

    logger.info(f"  CALLM completed in {elapsed:.1f}s")
    logger.debug(f"  CALLM full response: {raw_response[:500]}")

    return result


def test_v2(ema_row, query_engine, profile, logger, model="claude-sonnet-4-6", dry_run=False):
    """Test V2: Agentic sensing-only (no diary, tool-use loop)."""
    logger.info("--- Testing V2 (Agentic Sensing-Only) ---")
    from src.agent.agentic_sensing_only import AgenticSensingOnlyAgent

    agent = AgenticSensingOnlyAgent(
        study_id=int(ema_row.get("Study_ID", 0)),
        profile=profile,
        memory_doc="(Integration test — no memory document.)",
        query_engine=query_engine,
        model=model,
        max_tool_calls=4,  # Limit tool calls for testing
    )

    t0 = time.time()
    result = agent.predict(ema_row=ema_row)
    elapsed = time.time() - t0

    logger.info(f"  V2 completed in {elapsed:.1f}s")
    logger.info(f"  V2 tool calls: {result.get('_n_tool_calls', '?')}")
    logger.info(f"  V2 tokens: {result.get('_input_tokens', '?')}in + {result.get('_output_tokens', '?')}out")
    logger.info(f"  V2 model: {result.get('_model', '?')}")

    # Log tool call details
    for tc in result.get("_tool_calls", []):
        logger.debug(f"  V2 tool #{tc['index']}: {tc['tool_name']}({tc['input']}) -> {tc['result_preview'][:200]}")

    logger.debug(f"  V2 final response: {result.get('_final_response', '')[:500]}")

    result["_elapsed_s"] = elapsed
    return result


def test_v4(ema_row, query_engine, profile, logger, model="claude-sonnet-4-6", dry_run=False):
    """Test V4: Agentic multimodal (diary + sensing, tool-use loop)."""
    logger.info("--- Testing V4 (Agentic Multimodal) ---")
    from src.agent.agentic_sensing import AgenticSensingAgent

    diary_text = str(ema_row.get("emotion_driver", ""))
    if diary_text.lower() == "nan" or not diary_text.strip():
        diary_text = None

    agent = AgenticSensingAgent(
        study_id=int(ema_row.get("Study_ID", 0)),
        profile=profile,
        memory_doc="(Integration test — no memory document.)",
        query_engine=query_engine,
        model=model,
        max_tool_calls=4,  # Limit tool calls for testing
    )

    t0 = time.time()
    result = agent.predict(ema_row=ema_row, diary_text=diary_text)
    elapsed = time.time() - t0

    logger.info(f"  V4 completed in {elapsed:.1f}s")
    logger.info(f"  V4 tool calls: {result.get('_n_tool_calls', '?')}")
    logger.info(f"  V4 tokens: {result.get('_input_tokens', '?')}in + {result.get('_output_tokens', '?')}out")
    logger.info(f"  V4 diary present: {diary_text is not None}")

    for tc in result.get("_tool_calls", []):
        logger.debug(f"  V4 tool #{tc['index']}: {tc['tool_name']}({tc['input']}) -> {tc['result_preview'][:200]}")

    logger.debug(f"  V4 final response: {result.get('_final_response', '')[:500]}")

    result["_elapsed_s"] = elapsed
    return result


# ---------------------------------------------------------------------------
# Main test orchestrator
# ---------------------------------------------------------------------------

def run_integration_tests(
    versions: list[str],
    user_id: int = DEFAULT_USER,
    n_entries: int = N_TEST_ENTRIES,
    dry_run: bool = False,
    agentic_model: str = "claude-sonnet-4-6",
):
    """Run integration tests for specified versions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = TEST_LOG_DIR / f"run_{timestamp}"
    logger = setup_logging(run_dir)

    logger.info("=" * 70)
    logger.info(f"Integration Test Run: {timestamp}")
    logger.info(f"Versions: {versions}")
    logger.info(f"User: {user_id}")
    logger.info(f"Entries per version: {n_entries}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Agentic model: {agentic_model}")
    logger.info("=" * 70)

    # Check prerequisites
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    agentic_versions = {"v2", "v4"}
    cli_versions = {"v1", "v3", "callm"}

    if not has_api_key and agentic_versions & set(versions):
        logger.warning(
            "ANTHROPIC_API_KEY not set. V2/V4 tests will be SKIPPED. "
            "Set the key to test agentic agents."
        )

    # Load data
    logger.info("Loading test data...")
    full_df, test_entries = load_test_data(user_id, n_entries)
    logger.info(f"Selected {len(test_entries)} test entries for user {user_id}")

    for i, entry in enumerate(test_entries):
        logger.info(
            f"  Entry {i+1}: {entry.get('timestamp_local')} | "
            f"diary: '{str(entry.get('emotion_driver', ''))[:60]}'"
        )

    # Load training data for CALLM/V3 retrievers
    train_csv = SPLITS_DIR / "group_1_train.csv"
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        # Filter to entries with diary text for retriever
        train_diary_mask = (
            train_df["emotion_driver"].notna()
            & (train_df["emotion_driver"].astype(str) != "nan")
            & (train_df["emotion_driver"].str.strip() != "")
        )
        train_diary_df = train_df[train_diary_mask]
        logger.info(f"Training data: {len(train_df)} total, {len(train_diary_df)} with diary")
    else:
        train_df = pd.DataFrame()
        train_diary_df = pd.DataFrame()
        logger.warning(f"Training CSV not found: {train_csv}")

    # Build shared resources
    profile = UserProfile(study_id=user_id)

    # LLM client for V1/V3/CALLM
    llm_client = ClaudeCodeClient(
        model="sonnet",
        timeout=120,
        max_retries=2,
        delay_between_calls=1.0,
        dry_run=dry_run,
    )

    # Query engine for V2/V4
    query_engine = None
    if agentic_versions & set(versions) and has_api_key:
        ema_df = full_df  # Full EMA data for baseline/history lookups
        query_engine = SensingQueryEngine(
            processed_dir=str(PROCESSED_DIR),
            ema_df=ema_df,
        )
        pid = f"{user_id:03d}"
        available_mods = [
            mod for mod in query_engine.MODALITIES
            if query_engine._parquet_path(user_id, mod).exists()
        ]
        logger.info(f"Query engine initialized. User {user_id} modalities: {available_mods}")

    # Run tests
    all_results = {}
    summary = {"passed": [], "failed": [], "skipped": []}

    for entry_idx, ema_row in enumerate(test_entries):
        date_str = str(ema_row.get("date_local", ""))
        ts_str = str(ema_row.get("timestamp_local", ""))
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"EMA Entry {entry_idx + 1}/{len(test_entries)}: {ts_str}")
        logger.info("=" * 50)

        # Ground truth for comparison
        ground_truth = {}
        for target in EXPECTED_CONTINUOUS:
            val = ema_row.get(target)
            ground_truth[target] = float(val) if pd.notna(val) else None
        for target in EXPECTED_BINARY:
            val = ema_row.get(target)
            ground_truth[target] = bool(val) if pd.notna(val) else None
        ground_truth["INT_availability"] = str(ema_row.get("INT_availability", "")).lower().strip()

        logger.info(f"Ground truth: PANAS_Pos={ground_truth['PANAS_Pos']}, "
                     f"PANAS_Neg={ground_truth['PANAS_Neg']}, "
                     f"ER_desire={ground_truth['ER_desire']}, "
                     f"availability={ground_truth['INT_availability']}")

        # Build SensingDay for V1/V3
        sensing_day = build_sensing_day(user_id, date_str)
        if sensing_day:
            logger.info(f"SensingDay built: {sensing_day.available_modalities()}")
        else:
            logger.warning("SensingDay is None (no daily sensing data found)")

        entry_key = f"entry_{entry_idx}"
        all_results[entry_key] = {
            "timestamp": ts_str,
            "ground_truth": ground_truth,
            "diary": str(ema_row.get("emotion_driver", "")),
            "predictions": {},
            "validation": {},
        }

        for version in versions:
            test_key = f"{version}_entry{entry_idx}"
            try:
                if version == "v1":
                    result = test_v1(ema_row, sensing_day, llm_client, profile, logger, dry_run)
                elif version == "v3":
                    result = test_v3(ema_row, sensing_day, llm_client, profile, train_diary_df, logger, dry_run)
                elif version == "callm":
                    result = test_callm(ema_row, llm_client, profile, train_diary_df, logger, dry_run)
                elif version == "v2":
                    if not has_api_key:
                        logger.warning(f"  SKIP V2 (no ANTHROPIC_API_KEY)")
                        summary["skipped"].append(test_key)
                        continue
                    result = test_v2(ema_row, query_engine, profile, logger, agentic_model, dry_run)
                elif version == "v4":
                    if not has_api_key:
                        logger.warning(f"  SKIP V4 (no ANTHROPIC_API_KEY)")
                        summary["skipped"].append(test_key)
                        continue
                    result = test_v4(ema_row, query_engine, profile, logger, agentic_model, dry_run)
                else:
                    logger.warning(f"  Unknown version: {version}")
                    continue

                # Validate
                validation = validate_prediction(result, version, logger)

                # Log comparison to ground truth
                for target in EXPECTED_CONTINUOUS:
                    pred_val = result.get(target)
                    true_val = ground_truth.get(target)
                    if pred_val is not None and true_val is not None:
                        error = abs(float(pred_val) - float(true_val))
                        logger.info(f"  {version} {target}: pred={pred_val:.1f}, true={true_val:.1f}, error={error:.1f}")

                # Store results
                # Remove non-serializable items
                serializable = {}
                for k, v in result.items():
                    try:
                        json.dumps(v)
                        serializable[k] = v
                    except (TypeError, ValueError):
                        serializable[k] = str(v)

                all_results[entry_key]["predictions"][version] = serializable
                all_results[entry_key]["validation"][version] = validation

                if validation["valid"]:
                    summary["passed"].append(test_key)
                    logger.info(f"  [{version}] PASSED")
                else:
                    summary["failed"].append(test_key)
                    logger.warning(f"  [{version}] FAILED: {validation['issues']}")

            except Exception as exc:
                logger.error(f"  [{version}] EXCEPTION: {exc}")
                logger.error(traceback.format_exc())
                summary["failed"].append(test_key)
                all_results[entry_key]["predictions"][version] = {
                    "_error": str(exc),
                    "_traceback": traceback.format_exc(),
                }

    # Save results
    results_file = run_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {results_file}")

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Passed:  {len(summary['passed'])} — {summary['passed']}")
    logger.info(f"  Failed:  {len(summary['failed'])} — {summary['failed']}")
    logger.info(f"  Skipped: {len(summary['skipped'])} — {summary['skipped']}")
    logger.info("=" * 70)

    # Save summary
    summary_file = run_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Integration tests for V1-V4 + CALLM")
    parser.add_argument(
        "--versions", type=str, default="v1,v3,callm,v2,v4",
        help="Comma-separated versions to test (default: v1,v3,callm,v2,v4)",
    )
    parser.add_argument("--user", type=int, default=DEFAULT_USER, help="Test user Study_ID")
    parser.add_argument("--n-entries", type=int, default=N_TEST_ENTRIES, help="EMA entries per version")
    parser.add_argument("--dry-run", action="store_true", help="No LLM calls (placeholder responses)")
    parser.add_argument(
        "--agentic-model", type=str, default="claude-sonnet-4-6",
        help="Model for V2/V4 agentic agents",
    )
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    summary = run_integration_tests(
        versions=versions,
        user_id=args.user,
        n_entries=args.n_entries,
        dry_run=args.dry_run,
        agentic_model=args.agentic_model,
    )

    # Exit code: 1 if any failures
    sys.exit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
