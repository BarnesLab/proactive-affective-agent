#!/usr/bin/env python3
"""Dynamic queue runner for pilot_v2 all versions.

- 5 parallel workers
- No fallbacks (cc_agent waits on rate limits indefinitely)
- Auto-detects incomplete/fallback tasks from checkpoints
- Runs until all tasks complete
- Supports all 7 versions (CALLM, v1-v6) and configurable user lists

Usage:
    python scripts/queue_runner.py                 # Run all incomplete tasks
    python scripts/queue_runner.py --clean         # Clean checkpoints first, then run
    python scripts/queue_runner.py --dry-run       # Show tasks without running
    python scripts/queue_runner.py --workers 3     # Use 3 workers instead of 5
"""

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PILOT_DIR = PROJECT_ROOT / "outputs" / "pilot_v2"
CHECKPOINT_DIR = PILOT_DIR / "checkpoints"
RATE_LIMIT_LOG = PILOT_DIR / ".rate_limit_events.jsonl"
LOG_DIR = PILOT_DIR / "queue_logs"

ALL_VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]
AGENTIC_VERSIONS = {"v2", "v4", "v5", "v6"}  # set for fast lookup
# All 50 users (matches evaluate_pilot.py PRIMARY_USERS + remaining)
USERS = [24, 25, 40, 41, 43, 60, 61, 71, 75, 82, 83, 86, 89, 95, 98, 99,
         103, 119, 140, 164, 169, 187, 189, 211, 232, 242, 257, 258, 260,
         275, 299, 310, 320, 335, 338, 351, 361, 362, 363, 392, 399, 403,
         437, 455, 458, 464, 499, 503, 505, 513]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("queue_runner")


_USER_ENTRY_COUNTS = {}  # cached


def get_user_total_entries(uid: int) -> int:
    """Get total EMA entries for a user from the actual data (cached)."""
    if not _USER_ENTRY_COUNTS:
        import glob
        import pandas as pd

        data_dir = PROJECT_ROOT / "data" / "processed" / "splits"
        dfs = []
        for f in sorted(glob.glob(str(data_dir / "group_*_test.csv"))):
            dfs.append(pd.read_csv(f))

        if dfs:
            df = pd.concat(dfs)
            for u in df["Study_ID"].unique():
                _USER_ENTRY_COUNTS[int(u)] = int(len(df[df["Study_ID"] == u]))

    return _USER_ENTRY_COUNTS.get(uid, 0)


def scan_tasks() -> list[tuple[str, int, int, str]]:
    """Scan checkpoints and determine which (version, user) tasks need work.

    Returns: list of (version, user_id, remaining_entries, reason)
    """
    tasks = []
    for ver in ALL_VERSIONS:
        for uid in USERS:
            total = get_user_total_entries(uid)
            if total == 0:
                logger.warning(f"Cannot determine total entries for user {uid}, skipping")
                continue

            cp_path = CHECKPOINT_DIR / f"{ver}_user{uid}_checkpoint.json"
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
                tasks.append((ver, uid, remaining, "incomplete"))
            elif ver in AGENTIC_VERSIONS:
                # Fully complete agentic version — check for fallbacks
                preds = data.get("predictions", [])
                n_fallback = sum(
                    1 for p in preds
                    if "fallback" in str(p.get("reasoning", "")).lower()
                )
                if n_fallback > 0:
                    tasks.append((ver, uid, n_fallback, "has_fallbacks"))

    # Sort: most remaining work first (maximize parallelism benefit)
    tasks.sort(key=lambda t: -t[2])
    return tasks


def run_task(version: str, user_id: int) -> int:
    """Run a single (version, user) task via run_pilot.py subprocess."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{version}_user{user_id}.log"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pilot.py"),
        "--version", version,
        "--users", str(user_id),
        "--output-dir", str(PILOT_DIR),
        "--model", "sonnet",
    ]

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    # Remove nested session guards
    for key in ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_CODE_SESSION_ID"):
        env.pop(key, None)

    with open(log_path, "a") as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Started: {datetime.now().isoformat()}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"{'='*60}\n\n")
        log_f.flush()

        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

    return result.returncode


def last_rate_limit_time():
    """Get the timestamp of the most recent rate limit event."""
    if not RATE_LIMIT_LOG.exists():
        return None

    last = None
    try:
        for line in RATE_LIMIT_LOG.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"])
                if last is None or ts > last:
                    last = ts
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    except Exception:
        pass
    return last


def send_telegram(msg: str) -> None:
    """Send Telegram notification via Boo bot."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "7542082932")
    if not bot_token:
        return
    try:
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"chat_id": int(chat_id), "text": msg}),
            ],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def worker_loop(worker_id: int, task_queue: queue.Queue, stats: dict, lock: threading.Lock):
    """Worker thread: pull tasks from queue and run them."""
    while True:
        try:
            ver, uid = task_queue.get(timeout=30)
        except queue.Empty:
            # Check if we should stop
            with lock:
                if stats["all_queued"] and task_queue.empty():
                    break
            continue

        logger.info(f"[W{worker_id}] Starting {ver} user{uid}")
        with lock:
            stats["active"][worker_id] = f"{ver}_u{uid}"

        t0 = time.monotonic()
        try:
            returncode = run_task(ver, uid)
            elapsed = time.monotonic() - t0
            elapsed_str = f"{elapsed/60:.1f}min" if elapsed > 60 else f"{elapsed:.0f}s"

            if returncode == 0:
                logger.info(f"[W{worker_id}] ✅ {ver} user{uid} done ({elapsed_str})")
                with lock:
                    stats["completed"].append((ver, uid))
            else:
                logger.warning(f"[W{worker_id}] ❌ {ver} user{uid} exit={returncode} ({elapsed_str})")
                with lock:
                    stats["failed"].append((ver, uid, returncode))
                    # Re-queue failed tasks (may be transient)
                    if returncode != 2:  # Don't requeue keyboard interrupt
                        task_queue.put((ver, uid))
                        logger.info(f"[W{worker_id}] Re-queued {ver} user{uid}")
        except Exception as e:
            logger.error(f"[W{worker_id}] Error running {ver} user{uid}: {e}")
            with lock:
                stats["failed"].append((ver, uid, -1))
        finally:
            with lock:
                stats["active"][worker_id] = None
            task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Queue runner for pilot_v2 agentic versions")
    parser.add_argument("--clean", action="store_true", help="Clean checkpoints before running")
    parser.add_argument("--dry-run", action="store_true", help="Show tasks without running")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    args = parser.parse_args()

    # Step 1: Optionally clean checkpoints
    if args.clean:
        logger.info("Step 1: Cleaning fallback entries from checkpoints...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "clean_checkpoints.py")],
            cwd=str(PROJECT_ROOT),
        )
        print()

    # Step 2: Scan tasks
    logger.info("Scanning tasks...")
    tasks = scan_tasks()
    if not tasks:
        logger.info("No tasks to run — all complete!")
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

    # Step 3: Start workers
    n_workers = min(args.workers, len(tasks))
    logger.info(f"\nStarting {n_workers} workers...")

    # Send Telegram notification
    send_telegram(
        f"[proactive-agent] Queue runner started\n"
        f"{len(tasks)} tasks, {total_remaining} entries to process\n"
        f"{n_workers} parallel workers, no fallback mode"
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
    }
    lock = threading.Lock()

    # Start worker threads
    threads = []
    for i in range(n_workers):
        t = threading.Thread(target=worker_loop, args=(i, task_queue, stats, lock), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1)  # Stagger worker starts

    # Step 4: Monitor progress
    try:
        last_progress_report = time.monotonic()
        while True:
            # Check if all tasks are done
            alive = [t for t in threads if t.is_alive()]
            if not alive:
                break

            # Progress report every 10 minutes
            now = time.monotonic()
            if now - last_progress_report > 600:
                with lock:
                    active_list = [v for v in stats["active"].values() if v]
                    n_done = len(stats["completed"])
                    n_failed = len(stats["failed"])
                logger.info(
                    f"Progress: {n_done}/{len(tasks)} done, {n_failed} failed, "
                    f"{len(active_list)} active: {active_list}, "
                    f"queue: {task_queue.qsize()} remaining"
                )
                last_progress_report = now

            # Check 24h no rate limit condition
            last_rl = last_rate_limit_time()
            if last_rl and (datetime.now() - last_rl) > timedelta(hours=24):
                logger.info(f"24h with no rate limits since {last_rl}. Continuing normally.")

            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("\nInterrupted. Waiting for active tasks to finish...")
        send_telegram("[proactive-agent] Queue runner interrupted by user")
        # Workers will finish their current task then stop
        # (daemon threads will be killed when main thread exits)

    # Final report
    with lock:
        n_done = len(stats["completed"])
        n_failed = len(stats["failed"])
    elapsed = datetime.now() - stats["start_time"]

    summary = (
        f"[proactive-agent] Queue runner finished\n"
        f"Completed: {n_done}/{len(tasks)} tasks\n"
        f"Failed: {n_failed}\n"
        f"Elapsed: {elapsed}"
    )
    logger.info(f"\n{'='*60}")
    logger.info(summary)
    logger.info(f"{'='*60}")
    send_telegram(summary)


if __name__ == "__main__":
    main()
