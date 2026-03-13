#!/usr/bin/env python3
"""Queue runner for GPT-5.1-codex-mini pilot — mirrors Claude's queue_runner.py.

- 10 parallel workers (one per user effectively)
- Extremely patient: waits hours for rate limits, only stops after 24h with zero progress
- Checkpoint-based: safe to re-run if interrupted
- Auto-detects incomplete tasks from checkpoints
- Logs metadata to JSON for reproducibility

Usage:
    python scripts/queue_runner_gpt.py                     # Run all incomplete tasks
    python scripts/queue_runner_gpt.py --dry-run           # Show tasks without running
    python scripts/queue_runner_gpt.py --workers 5         # Use 5 workers
    python scripts/queue_runner_gpt.py --model gpt-5.4     # Override model
"""

import argparse
import glob
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = "gpt-5.1-codex-mini"

# Match Claude pilot_v2 users exactly
USERS = [43, 71, 258, 275, 338, 362, 399, 403, 437, 513]
ALL_VERSIONS = ["gpt-callm", "gpt-v1", "gpt-v2", "gpt-v3", "gpt-v4", "gpt-v5", "gpt-v6"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("queue_runner_gpt")

_USER_ENTRY_COUNTS: dict[int, int] = {}


def get_output_dir() -> Path:
    """Get or create a stable output directory for GPT-5.1-codex-mini runs."""
    base = PROJECT_ROOT / "outputs" / "pilot_gpt51mini"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_user_total_entries(uid: int) -> int:
    """Get total EMA entries for a user from the actual data (cached)."""
    if not _USER_ENTRY_COUNTS:
        data_dir = PROJECT_ROOT / "data" / "processed" / "splits"
        dfs = []
        for f in sorted(glob.glob(str(data_dir / "group_*_test.csv"))):
            dfs.append(pd.read_csv(f))
        if dfs:
            df = pd.concat(dfs)
            for u in df["Study_ID"].unique():
                _USER_ENTRY_COUNTS[int(u)] = int(len(df[df["Study_ID"] == u]))
    return _USER_ENTRY_COUNTS.get(uid, 0)


def scan_tasks(output_dir: Path) -> list[tuple[str, int, int, str]]:
    """Scan checkpoints and determine which (version, user) tasks need work.

    Returns: list of (version, user_id, remaining_entries, reason)
    """
    checkpoint_dir = output_dir / "checkpoints"
    tasks = []
    for ver in ALL_VERSIONS:
        for uid in USERS:
            total = get_user_total_entries(uid)
            if total == 0:
                logger.warning(f"Cannot determine total entries for user {uid}, skipping")
                continue

            cp_path = checkpoint_dir / f"{ver}_user{uid}_checkpoint.json"
            if not cp_path.exists():
                tasks.append((ver, uid, total, "no_checkpoint"))
                continue

            try:
                with open(cp_path) as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {cp_path}: {e}")
                tasks.append((ver, uid, total, "corrupt_checkpoint"))
                continue

            n_completed = data.get("n_entries", 0)
            remaining = total - n_completed
            if remaining > 0:
                tasks.append((ver, uid, remaining, f"incomplete ({n_completed}/{total})"))
            # else: fully complete, skip

    # Sort: most remaining work first (maximize parallelism benefit)
    tasks.sort(key=lambda t: -t[2])
    return tasks


def run_task(version: str, user_id: int, output_dir: Path, model: str) -> int:
    """Run a single (version, user) task via run_pilot.py subprocess."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{version}_user{user_id}.log"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pilot.py"),
        "--version", version,
        "--users", str(user_id),
        "--model", model,
        "--delay", "1.0",
        "--output-dir", str(output_dir),
        "--verbose",
    ]

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    # Remove nested session guards
    for key in ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_SESSION_ID"):
        env.pop(key, None)

    with open(log_path, "a") as log_f:
        log_f.write(f"\n{'=' * 60}\n")
        log_f.write(f"Started: {datetime.now().isoformat()}\n")
        log_f.write(f"Model: {model}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"{'=' * 60}\n\n")
        log_f.flush()

        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

    return result.returncode


def send_telegram(msg: str) -> None:
    """Send Telegram notification via Boo bot."""
    try:
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                "https://api.telegram.org/bot7740709485:AAF35LkeavJ5-F4C6hcG5PC_7RdC9AeI8lI/sendMessage",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"chat_id": 7542082932, "text": msg}),
            ],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def save_run_metadata(output_dir: Path, model: str, n_workers: int, tasks: list) -> None:
    """Save run metadata to JSON for reproducibility."""
    meta = {
        "run_started": datetime.now().isoformat(),
        "model": model,
        "n_workers": n_workers,
        "users": USERS,
        "versions": ALL_VERSIONS,
        "n_tasks": len(tasks),
        "tasks": [
            {"version": v, "user_id": u, "remaining": r, "reason": reason}
            for v, u, r, reason in tasks
        ],
        "claude_pilot_v2_users": USERS,
        "note": "GPT-5.1-codex-mini mirror of Claude pilot_v2 experiment",
    }
    meta_path = output_dir / "run_metadata.json"
    # Append to existing metadata if file exists
    existing = []
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing = content
                else:
                    existing = [content]
        except Exception:
            pass
    existing.append(meta)
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)


def worker_loop(
    worker_id: int,
    task_queue: queue.Queue,
    stats: dict,
    lock: threading.Lock,
    output_dir: Path,
    model: str,
):
    """Worker thread: pull tasks from queue and run them."""
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 20  # Give up on a worker after 20 consecutive failures

    while True:
        try:
            ver, uid = task_queue.get(timeout=60)
        except queue.Empty:
            with lock:
                if stats["all_queued"] and task_queue.empty():
                    break
            continue

        logger.info(f"[W{worker_id}] Starting {ver} user{uid}")
        with lock:
            stats["active"][worker_id] = f"{ver}_u{uid}"
            stats["last_activity"] = datetime.now()

        t0 = time.monotonic()
        try:
            returncode = run_task(ver, uid, output_dir, model)
            elapsed = time.monotonic() - t0
            elapsed_str = f"{elapsed / 60:.1f}min" if elapsed > 60 else f"{elapsed:.0f}s"

            if returncode == 0:
                logger.info(f"[W{worker_id}] DONE {ver} user{uid} ({elapsed_str})")
                consecutive_failures = 0
                with lock:
                    stats["completed"].append((ver, uid, elapsed_str))
                    stats["last_activity"] = datetime.now()
            else:
                logger.warning(f"[W{worker_id}] FAIL {ver} user{uid} exit={returncode} ({elapsed_str})")
                consecutive_failures += 1
                with lock:
                    stats["failed"].append((ver, uid, returncode))
                    # Re-queue failed tasks (may be transient rate limit)
                    if returncode != 2 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
                        # Wait before re-queuing to avoid tight loops
                        wait_time = min(60 * consecutive_failures, 600)
                        logger.info(
                            f"[W{worker_id}] Re-queued {ver} user{uid}, "
                            f"waiting {wait_time}s before next attempt "
                            f"(consecutive failures: {consecutive_failures})"
                        )
                        time.sleep(wait_time)
                        task_queue.put((ver, uid))
                    elif consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error(
                            f"[W{worker_id}] {MAX_CONSECUTIVE_FAILURES} consecutive failures, "
                            f"skipping {ver} user{uid}"
                        )
                        consecutive_failures = 0
        except Exception as e:
            logger.error(f"[W{worker_id}] Error running {ver} user{uid}: {e}")
            with lock:
                stats["failed"].append((ver, uid, -1))
        finally:
            with lock:
                stats["active"][worker_id] = None
            task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="GPT-5.1-codex-mini queue runner")
    parser.add_argument("--dry-run", action="store_true", help="Show tasks without running")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    output_dir = get_output_dir()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    (output_dir / "memory").mkdir(parents=True, exist_ok=True)

    # Save users/versions for dashboard compatibility
    (output_dir / "users.txt").write_text("\n".join(str(u) for u in USERS) + "\n")
    (output_dir / "versions.txt").write_text("\n".join(ALL_VERSIONS) + "\n")

    # Scan tasks
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info("Scanning tasks...")
    tasks = scan_tasks(output_dir)
    if not tasks:
        logger.info("No tasks to run — all complete!")
        send_telegram("[proactive-agent] GPT queue runner: all tasks already complete!")
        return

    logger.info(f"Found {len(tasks)} tasks:")
    total_remaining = 0
    for ver, uid, remaining, reason in tasks:
        logger.info(f"  {ver} user{uid}: {remaining} entries ({reason})")
        total_remaining += remaining

    logger.info(f"Total entries to process: {total_remaining}")

    if args.dry_run:
        logger.info("Dry run — not starting workers.")
        return

    # Save metadata
    save_run_metadata(output_dir, args.model, args.workers, tasks)

    # Start workers
    n_workers = min(args.workers, len(tasks))
    logger.info(f"\nStarting {n_workers} workers...")

    send_telegram(
        f"[proactive-agent] GPT queue runner started\n"
        f"Model: {args.model}\n"
        f"{len(tasks)} tasks, {total_remaining} entries\n"
        f"{n_workers} parallel workers\n"
        f"Users: {USERS}"
    )

    task_queue: queue.Queue = queue.Queue()
    for ver, uid, _, _ in tasks:
        task_queue.put((ver, uid))

    stats = {
        "completed": [],
        "failed": [],
        "active": {i: None for i in range(n_workers)},
        "all_queued": True,
        "start_time": datetime.now(),
        "last_activity": datetime.now(),
    }
    lock = threading.Lock()

    threads = []
    for i in range(n_workers):
        t = threading.Thread(
            target=worker_loop,
            args=(i, task_queue, stats, lock, output_dir, args.model),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(2)  # Stagger starts to avoid thundering herd

    # Monitor progress
    try:
        last_progress_report = time.monotonic()
        last_telegram_report = time.monotonic()
        while True:
            alive = [t for t in threads if t.is_alive()]
            if not alive:
                break

            now = time.monotonic()

            # Progress report every 5 minutes to log
            if now - last_progress_report > 300:
                with lock:
                    active_list = [v for v in stats["active"].values() if v]
                    n_done = len(stats["completed"])
                    n_failed = len(stats["failed"])
                    last_act = stats["last_activity"]
                    hours_idle = (datetime.now() - last_act).total_seconds() / 3600

                logger.info(
                    f"Progress: {n_done}/{len(tasks)} done, {n_failed} fails, "
                    f"{len(active_list)} active: {active_list}, "
                    f"queue: {task_queue.qsize()} remaining, "
                    f"idle: {hours_idle:.1f}h"
                )
                last_progress_report = now

                # Check 24h no-progress condition
                if hours_idle > 24:
                    msg = (
                        f"[proactive-agent] GPT queue runner: 24h no progress!\n"
                        f"Completed: {n_done}/{len(tasks)}\n"
                        f"Stopping."
                    )
                    logger.error(msg)
                    send_telegram(msg)
                    break

            # Telegram report every 30 minutes
            if now - last_telegram_report > 1800:
                with lock:
                    n_done = len(stats["completed"])
                    n_failed = len(stats["failed"])
                    active_list = [v for v in stats["active"].values() if v]
                elapsed_total = datetime.now() - stats["start_time"]
                send_telegram(
                    f"[proactive-agent] GPT progress update\n"
                    f"Done: {n_done}/{len(tasks)} | Failed: {n_failed}\n"
                    f"Active: {active_list}\n"
                    f"Queue: {task_queue.qsize()} remaining\n"
                    f"Elapsed: {elapsed_total}"
                )
                last_telegram_report = now

            time.sleep(15)

    except KeyboardInterrupt:
        logger.info("\nInterrupted. Waiting for active tasks to finish...")
        send_telegram("[proactive-agent] GPT queue runner interrupted by user")

    # Final report
    with lock:
        n_done = len(stats["completed"])
        n_failed = len(stats["failed"])
        completed_list = stats["completed"]
    elapsed = datetime.now() - stats["start_time"]

    # Save final results
    results = {
        "run_finished": datetime.now().isoformat(),
        "elapsed": str(elapsed),
        "completed": n_done,
        "failed": n_failed,
        "completed_tasks": [(v, u, t) for v, u, t in completed_list],
    }
    results_path = output_dir / "run_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = (
        f"[proactive-agent] GPT queue runner finished\n"
        f"Model: {args.model}\n"
        f"Completed: {n_done}/{len(tasks)} tasks\n"
        f"Failed: {n_failed}\n"
        f"Elapsed: {elapsed}"
    )
    logger.info(f"\n{'=' * 60}")
    logger.info(summary)
    logger.info(f"{'=' * 60}")
    send_telegram(summary)


if __name__ == "__main__":
    main()
