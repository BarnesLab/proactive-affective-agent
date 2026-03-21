#!/usr/bin/env python3
"""Sonnet watcher v2: pure Python, maintains TARGET parallel run_pilot processes.

Replaces the bash sonnet_watcher.sh which kept dying due to heavy inline Python calls.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TARGET = 10
OUTDIR = PROJECT_ROOT / "outputs" / "pilot_v2"
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
LOG_FILE = OUTDIR / "logs" / "watcher.log"
PULSE_FILE = Path.home() / ".openclaw" / "pulse.json"
PAUSE_FILE = PROJECT_ROOT / ".pause"
RATE_LIMIT_FLAG = OUTDIR / ".rate_limited"

VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]
TARGET_USERS = [
    399, 258, 43, 71, 211, 505, 513, 363, 275, 437, 362, 86, 24, 164, 169,
    119, 99, 61, 458, 403, 503, 41, 310, 338, 25, 40, 89, 232, 242, 299,
    455, 187, 499, 320, 257, 361, 95, 103, 75, 83, 464, 335, 392, 351, 60,
    82, 260, 189, 140, 98,
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("watcher")


def is_peak_hours() -> bool:
    now = datetime.now()
    # Weekday (Mon=0..Fri=4) AND 8AM-2PM — weekends run full speed
    return now.weekday() < 5 and 8 <= now.hour < 14


def get_running() -> set[str]:
    """Get set of 'version_userUID' keys currently running."""
    running = set()
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in r.stdout.split("\n"):
        if "run_pilot" not in line or "sonnet" not in line or "grep" in line:
            continue
        if "python" not in line.lower():
            continue
        mv = re.search(r"--version (\S+)", line)
        mu = re.search(r"--users (\d+)", line)
        if mv and mu:
            running.add(f"{mv.group(1)}_user{mu.group(1)}")
    return running


_done_cache: set[str] = set()
_done_cache_time: float = 0


def get_done() -> set[str]:
    """Get done set, cached for 5 minutes to avoid reading 350 files every 30s."""
    global _done_cache, _done_cache_time
    if time.time() - _done_cache_time < 300:
        return _done_cache
    from scripts.quality_check import get_done_set
    _done_cache = get_done_set(VERSIONS)
    _done_cache_time = time.time()
    log.info(f"Refreshed done set: {len(_done_cache)}/350 complete")
    return _done_cache


def pick_tasks(needed: int, done: set[str], running: set[str]) -> list[tuple[str, int]]:
    """User-first scheduling: pick tasks for users closest to 7/7 completion."""
    user_completion = {}
    for u in TARGET_USERS:
        completed = sum(1 for v in VERSIONS if f"{v}_user{u}" in done)
        in_progress = sum(1 for v in VERSIONS if f"{v}_user{u}" in running)
        remaining = 7 - completed - in_progress
        if remaining > 0:
            user_completion[u] = (completed, remaining)

    sorted_users = sorted(user_completion.keys(), key=lambda u: -user_completion[u][0])

    tasks = []
    launched_users = set()
    for u in sorted_users:
        if len(tasks) >= needed:
            break
        for v in VERSIONS:
            if len(tasks) >= needed:
                break
            key = f"{v}_user{u}"
            if key not in done and key not in running and u not in launched_users:
                tasks.append((v, u))
                launched_users.add(u)
    return tasks


def launch_task(version: str, uid: int) -> None:
    log_dir = OUTDIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{version}_user{uid}.log"
    cmd = (
        f"cd {PROJECT_ROOT} && nohup {VENV_PYTHON} scripts/run_pilot.py "
        f"--version {version} --users {uid} --model sonnet --delay 1.0 "
        f"--output-dir {OUTDIR} --verbose > {log_file} 2>&1 &"
    )
    os.system(cmd)
    log.info(f"Launched: {version} user{uid}")


def update_pulse() -> None:
    try:
        p = {}
        if PULSE_FILE.exists():
            p = json.loads(PULSE_FILE.read_text())
        p["sonnet-watcher"] = int(time.time())
        PULSE_FILE.write_text(json.dumps(p, indent=2))
    except Exception:
        pass


def main():
    log.info(f"Watcher v2 started (target={TARGET})")

    while True:
        try:
            # Manual pause
            if PAUSE_FILE.exists():
                time.sleep(60)
                continue

            # Rate limit flag
            if RATE_LIMIT_FLAG.exists():
                try:
                    expiry = RATE_LIMIT_FLAG.read_text().strip()
                    from datetime import datetime as dt
                    exp_dt = dt.fromisoformat(expiry)
                    if datetime.now() < exp_dt:
                        time.sleep(1800)
                        continue
                except Exception:
                    pass
                RATE_LIMIT_FLAG.unlink(missing_ok=True)
                log.info("Rate limit expired, removing flag")

            # Peak hours: no new launches, let running finish
            if is_peak_hours():
                running = get_running()
                if running:
                    log.info(f"Peak hours, {len(running)} tasks running (no new launches)")
                update_pulse()
                time.sleep(300)
                continue

            # Normal operation
            running = get_running()
            current = len(running)

            if current >= TARGET:
                # At or over target, just wait
                pass
            elif current < TARGET:
                needed = TARGET - current
                done = get_done()
                tasks = pick_tasks(needed, done, running)

                if tasks:
                    log.info(f"Processes: {current}/{TARGET}, launching {len(tasks)}")
                    for v, u in tasks:
                        launch_task(v, u)
                elif current == 0:
                    log.info("No more tasks to launch — all done or all running")

            update_pulse()
            time.sleep(30)

        except KeyboardInterrupt:
            log.info("Watcher stopped (Ctrl+C)")
            break
        except Exception as e:
            log.error(f"Watcher error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
