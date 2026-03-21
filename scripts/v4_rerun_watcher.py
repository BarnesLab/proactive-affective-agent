#!/usr/bin/env python3
"""V4 rerun watcher: runs V4 for all 50 users with fixed MCP.
Independent from the main sonnet_watcher_v2.py.
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

TARGET = 5  # conservative to avoid rate limit fights with main watcher
OUTDIR = PROJECT_ROOT / "outputs" / "pilot_v3_v4_rerun_mar_21"
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
LOG_FILE = OUTDIR / "logs" / "watcher.log"

VERSION = "v4"
TARGET_USERS = [
    24, 25, 40, 41, 43, 60, 61, 71, 75, 82, 83, 86, 89, 95, 98, 99,
    103, 119, 140, 164, 169, 187, 189, 211, 232, 242, 257, 258, 260,
    275, 299, 310, 320, 335, 338, 351, 361, 362, 363, 392, 399, 403,
    437, 455, 458, 464, 499, 503, 505, 513,
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
log = logging.getLogger("v4_rerun")


def is_peak_hours() -> bool:
    now = datetime.now()
    return now.weekday() < 5 and 8 <= now.hour < 14


def get_running() -> set[str]:
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    running = set()
    for line in r.stdout.split("\n"):
        if "run_pilot" not in line or "pilot_v3_v4_rerun" not in line:
            continue
        if "grep" in line:
            continue
        mu = re.search(r"--users (\d+)", line)
        if mu:
            running.add(int(mu.group(1)))
    return running


def get_done() -> set[int]:
    """Check which users have complete V4 checkpoints."""
    import pandas as pd
    expected = {}
    splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    for csv in sorted(splits_dir.glob("group_*_test.csv")):
        df = pd.read_csv(csv)
        for uid, count in df["Study_ID"].value_counts().items():
            expected[int(uid)] = int(count)

    done = set()
    cp_dir = OUTDIR / "checkpoints"
    for f in cp_dir.glob(f"{VERSION}_user*_checkpoint.json"):
        m = re.search(r"user(\d+)", f.name)
        if not m:
            continue
        uid = int(m.group(1))
        try:
            with open(f) as fh:
                d = json.load(fh)
            n = d.get("n_entries", len(d.get("predictions", [])))
            if uid in expected and n >= expected[uid] * 0.95:
                done.add(uid)
        except Exception:
            pass
    return done


def launch_task(uid: int) -> None:
    log_dir = OUTDIR / "logs"
    log_file = log_dir / f"{VERSION}_user{uid}.log"
    cmd = [
        VENV_PYTHON, "scripts/run_pilot.py",
        "--version", VERSION, "--users", str(uid),
        "--model", "sonnet", "--delay", "1.0",
        "--output-dir", str(OUTDIR), "--verbose",
    ]
    # Clean env: strip ALL CLAUDE* vars to prevent MCP interference
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    clean_env["PYTHONPATH"] = str(PROJECT_ROOT)
    with open(log_file, "a") as lf:
        subprocess.Popen(
            cmd, stdout=lf, stderr=lf, env=clean_env, cwd=str(PROJECT_ROOT),
            start_new_session=True,  # fully detach from parent
        )
    log.info(f"Launched: {VERSION} user{uid}")


def main():
    log.info(f"V4 rerun watcher started (target={TARGET}, {len(TARGET_USERS)} users)")

    while True:
        try:
            if is_peak_hours():
                time.sleep(300)
                continue

            running = get_running()
            done = get_done()
            remaining = [u for u in TARGET_USERS if u not in done and u not in running]

            current = len(running)
            if current < TARGET and remaining:
                needed = min(TARGET - current, len(remaining))
                log.info(f"Done: {len(done)}/50, running: {current}, launching {needed}")
                for uid in remaining[:needed]:
                    launch_task(uid)
            elif not remaining and current == 0:
                log.info(f"ALL DONE! {len(done)}/50 users complete.")
                # Notify
                import subprocess as sp
                bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                chat_id = os.environ.get("TELEGRAM_CHAT_ID", "7542082932")
                if bot_token:
                    sp.run([
                        "curl", "-s", "-X", "POST",
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps({"chat_id": int(chat_id),
                            "text": f"[PAA] V4 rerun COMPLETE! {len(done)}/50 users done with fixed MCP."})
                    ], capture_output=True)
                break

            time.sleep(30)

        except KeyboardInterrupt:
            log.info("Stopped (Ctrl+C)")
            break
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
