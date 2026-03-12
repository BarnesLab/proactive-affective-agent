#!/usr/bin/env python3
"""Backfill missing/empty predictions in Haiku pilot checkpoints.

Handles two cases:
  1. Checkpoint has fewer entries than total (append mode)
  2. Checkpoint has empty {} predictions at specific indices (patch mode)

Spawns one process per (version, user) gap for maximum parallelism.

Usage:
    PYTHONPATH=. python3 scripts/backfill_missing.py
    PYTHONPATH=. python3 scripts/backfill_missing.py --version v4 --user 71
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pilot_haiku"
USERS = [71, 458, 310, 164, 119]
USER_TOTALS = {71: 93, 458: 82, 310: 81, 164: 87, 119: 84}
MODEL = "haiku"


def load_checkpoint(version: str, uid: int) -> Optional[dict]:
    cp = OUTPUT_DIR / "checkpoints" / f"{version}_user{uid}_checkpoint.json"
    if cp.exists():
        return json.loads(cp.read_text())
    return None


def save_checkpoint(version: str, uid: int, data: dict):
    cp = OUTPUT_DIR / "checkpoints" / f"{version}_user{uid}_checkpoint.json"
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w") as f:
        json.dump(data, f, default=str)


def is_empty_pred(pred):
    """Check if a prediction is empty/invalid."""
    if not pred:
        return True
    if isinstance(pred, dict) and len(pred) == 0:
        return True
    return False


def find_empty_indices(cp):
    """Find indices of empty predictions in a checkpoint."""
    if not cp:
        return []
    return [i for i, p in enumerate(cp.get("predictions", [])) if is_empty_pred(p)]


def find_gaps():
    """Find all (version, user) pairs with missing or empty entries."""
    gaps = []
    for v in ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]:
        for uid in USERS:
            total = USER_TOTALS[uid]
            cp = load_checkpoint(v, uid)
            if not cp:
                gaps.append((v, uid, 0, total, "missing"))
                continue
            n_preds = len(cp.get("predictions", []))
            empty_indices = find_empty_indices(cp)
            if n_preds < total:
                gaps.append((v, uid, n_preds, total, "incomplete"))
            elif empty_indices:
                gaps.append((v, uid, n_preds - len(empty_indices), total, f"{len(empty_indices)} empty"))
    return gaps


def run_single(version: str, uid: int):
    """Backfill a single (version, user) pair — both append and patch modes."""
    from src.data.loader import DataLoader
    from src.data.preprocessing import prepare_pilot_data
    from src.agent.personal_agent import PersonalAgent
    from src.remember.retriever import MultiModalRetriever, TFIDFRetriever
    from src.think.llm_client import ClaudeCodeClient
    from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

    loader = DataLoader(data_dir=PROJECT_ROOT / "data")
    all_ema = loader.load_all_ema()
    sensing_dfs = loader.load_all_sensing()
    train_df = loader.load_all_train()

    retriever = TFIDFRetriever()
    retriever.fit(train_df)
    mm_retriever = MultiModalRetriever()
    mm_retriever.fit(train_df, sensing_dfs=sensing_dfs)

    processed_dir = loader.data_dir / "processed" / "hourly"
    filtered_dir = loader.data_dir / "processed" / "filtered"
    peer_db_path = str(OUTPUT_DIR / "peer_database.parquet") if (OUTPUT_DIR / "peer_database.parquet").exists() else None

    users_data = prepare_pilot_data(loader, pilot_user_ids=[uid], ema_df=all_ema)
    if not users_data:
        logger.error(f"No data for user {uid}")
        return
    ud = users_data[0]
    total = len(ud["ema_entries"])

    cp = load_checkpoint(version, uid)

    # Determine what needs to be done
    empty_indices = find_empty_indices(cp) if cp else []
    n_existing = len(cp["predictions"]) if cp else 0
    append_start = n_existing if n_existing < total else total

    if not empty_indices and append_start >= total:
        logger.info(f"{version.upper()} user {uid}: already complete, no empty predictions")
        return

    if cp is None:
        cp = {
            "version": version, "n_entries": 0,
            "current_user": uid, "current_entry": 0,
            "predictions": [], "ground_truths": [], "metadata": [],
        }

    # Setup agent
    if version == "callm":
        ret = retriever
    elif version == "v3":
        ret = mm_retriever
    else:
        ret = None

    llm = ClaudeCodeClient(model=MODEL, dry_run=False, delay_between_calls=0.5)
    agent = PersonalAgent(
        study_id=uid, version=version, llm_client=llm,
        profile=ud["profile"], memory_doc=ud["memory"], retriever=ret,
        processed_dir=processed_dir if version in ("v2", "v4", "v5", "v6") else None,
        filtered_data_dir=filtered_dir if version in ("v5", "v6") else None,
        agentic_model=MODEL, peer_db_path=peer_db_path,
    )

    def build_gt(ema_row):
        gt = {}
        for target in CONTINUOUS_TARGETS:
            val = ema_row.get(target)
            if pd.notna(val):
                gt[target] = float(val)
        for target in BINARY_STATE_TARGETS:
            val = ema_row.get(target)
            if pd.notna(val):
                gt[target] = bool(val)
        return gt

    def run_prediction(i):
        ema_row = ud["ema_entries"][i]
        sensing_day = ud["sensing_days"][i]
        date_str = str(ema_row.get("date_local", ""))
        logger.info(f"  Entry {i+1}/{total} ({date_str})")
        try:
            pred = agent.predict(ema_row=ema_row, sensing_day=sensing_day, date_str=date_str)
        except Exception as e:
            logger.error(f"  Error: {e}")
            pred = {"_error": str(e)}
        clean_pred = {k: v for k, v in pred.items() if not k.startswith("_")}
        gt = build_gt(ema_row)
        meta = {
            "study_id": uid, "entry_index": i,
            "date_str": date_str,
            "timestamp": str(ema_row.get("timestamp_local", "")),
        }
        conf = pred.get("confidence", "?")
        logger.info(f"    OK (confidence={conf})")
        return clean_pred, gt, meta

    # Phase 1: Patch empty predictions in-place
    if empty_indices:
        logger.info(f"{version.upper()} user {uid}: patching {len(empty_indices)} empty predictions at indices {empty_indices[:10]}{'...' if len(empty_indices) > 10 else ''}")
        for idx in empty_indices:
            if idx >= total:
                continue
            clean_pred, gt, meta = run_prediction(idx)
            cp["predictions"][idx] = clean_pred
            cp["ground_truths"][idx] = gt
            cp["metadata"][idx] = meta
            cp["n_entries"] = len(cp["predictions"])
            save_checkpoint(version, uid, cp)
            time.sleep(0.3)
        logger.info(f"{version.upper()} user {uid}: patching done")

    # Phase 2: Append missing entries
    if append_start < total:
        logger.info(f"{version.upper()} user {uid}: appending entries {append_start+1}..{total}")
        for i in range(append_start, total):
            clean_pred, gt, meta = run_prediction(i)
            cp["predictions"].append(clean_pred)
            cp["ground_truths"].append(gt)
            cp["metadata"].append(meta)
            cp["n_entries"] = len(cp["predictions"])
            cp["current_user"] = uid
            cp["current_entry"] = i
            save_checkpoint(version, uid, cp)
            time.sleep(0.3)
        logger.info(f"{version.upper()} user {uid}: append done")

    logger.info(f"{version.upper()} user {uid}: all done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--user", type=int, default=None)
    args = parser.parse_args()

    if args.version and args.user:
        # Single worker mode
        run_single(args.version, args.user)
        return

    # Orchestrator mode: find gaps and spawn parallel workers
    gaps = find_gaps()
    if not gaps:
        logger.info("Nothing to backfill!")
        return

    logger.info(f"Found {len(gaps)} gaps:")
    for v, uid, done, total, reason in gaps:
        logger.info(f"  {v.upper()} user {uid}: {done}/{total} ({reason})")

    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for v, uid, done, total, reason in gaps:
        cmd = [
            sys.executable, __file__,
            "--version", v, "--user", str(uid),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        log_file = open(log_dir / f"backfill_{v}_u{uid}.log", "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        procs.append((v, uid, p, done, total))
        logger.info(f"  Started {v.upper()} user {uid} (PID {p.pid})")
        time.sleep(1)

    logger.info(f"\n{len(procs)} backfill workers launched. Monitoring...\n")

    # Monitor
    while True:
        time.sleep(30)
        still_running = [(v, uid, p, d, t) for v, uid, p, d, t in procs if p.poll() is None]
        finished = [(v, uid, p, d, t) for v, uid, p, d, t in procs if p.poll() is not None]
        for v, uid, p, d, t in finished:
            if p.returncode == 0:
                logger.info(f"  {v.upper()} user {uid}: finished OK")
            else:
                logger.warning(f"  {v.upper()} user {uid}: exited code {p.returncode}")
        procs = still_running
        if not procs:
            break
        logger.info(f"  {len(procs)} workers still running...")

    logger.info("All backfill workers done!")


if __name__ == "__main__":
    main()
