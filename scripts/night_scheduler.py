#!/usr/bin/env python3
"""Automated night scheduler for pilot experiments.

Monitors system activity, waits for idle time (after midnight, no active
Claude Code sessions), then launches all pilot versions in parallel to
maximize Claude Max token usage. Handles rate limits by pausing and retrying.

Usage:
    # Start the scheduler (runs until all work is done or manually stopped)
    python scripts/night_scheduler.py

    # Custom settings
    python scripts/night_scheduler.py --start-hour 23 --stop-hour 8 --output-dir outputs/pilot_v2

    # One-shot: skip idle detection, just run everything now
    python scripts/night_scheduler.py --now
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rate_limit import send_telegram

# ── Config ──────────────────────────────────────────────────────────────

VERSIONS = ["v1", "callm", "v3", "v2", "v4"]
USERS = "399,258,43,403,338"
MODEL = "sonnet"
DELAY = 1.5
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pilot_v2"
USAGE_STATS_PATH = Path.home() / ".claude" / "hooks" / "usage-stats.json"

# Scheduling
DEFAULT_START_HOUR = 0   # midnight
DEFAULT_STOP_HOUR = 8    # 8am — yield to user
IDLE_CHECK_INTERVAL = 300  # 5 min between idle checks
RATE_LIMIT_WAIT = 1800     # 30 min wait on rate limit
PROGRESS_INTERVAL = 600    # 10 min between progress checks

logger = logging.getLogger("night_scheduler")


# ── Helpers ─────────────────────────────────────────────────────────────

def is_within_window(start_hour: int, stop_hour: int) -> bool:
    """Check if current time is within the allowed run window."""
    now = datetime.now().hour
    if start_hour < stop_hour:
        return start_hour <= now < stop_hour
    else:
        # Wraps midnight, e.g., 23..8
        return now >= start_hour or now < stop_hour


def get_active_claude_sessions() -> int:
    """Count active Claude Code sessions consuming tokens."""
    try:
        data = json.loads(USAGE_STATS_PATH.read_text())
        rate = data.get("rate_per_hour", 0)
        # If rate > 5/hour, someone is actively using Claude Code
        return rate
    except Exception:
        return 0


def is_user_idle(start_hour: int, stop_hour: int) -> bool:
    """Determine if the user is idle (in run window + low token activity)."""
    if not is_within_window(start_hour, stop_hour):
        return False
    rate = get_active_claude_sessions()
    # If rate is high (>10/hour), user is likely active
    # But during scheduled window, we're more lenient — only bail if very active
    if rate > 40:
        logger.info(f"High token rate ({rate}/hr), user may be active")
        return False
    return True


def get_checkpoint_progress(version: str, users: list[int]) -> dict:
    """Read checkpoint progress for a version across all users."""
    progress = {}
    for uid in users:
        cp_path = OUTPUT_DIR / "checkpoints" / f"{version}_user{uid}_checkpoint.json"
        if cp_path.exists():
            try:
                data = json.loads(cp_path.read_text())
                n_done = len(data.get("predictions", []))
                n_total = data.get("n_entries", 0)
                progress[uid] = {"done": n_done, "total": n_total}
            except Exception:
                progress[uid] = {"done": 0, "total": 0}
        else:
            progress[uid] = {"done": 0, "total": 0}
    return progress


def is_version_complete(version: str, users: list[int]) -> bool:
    """Check if all users are fully completed for this version."""
    progress = get_checkpoint_progress(version, users)
    for uid, p in progress.items():
        if p["total"] == 0:
            # No checkpoint yet — not started or unknown total
            return False
        if p["done"] < p["total"]:
            return False
    return True


def format_progress_report(users: list[int]) -> str:
    """Build a compact progress summary for all versions."""
    lines = ["Pilot V2 Progress:"]
    all_done = True
    for v in VERSIONS:
        progress = get_checkpoint_progress(v, users)
        # Count users with actual checkpoints (total > 0)
        started_users = {uid: p for uid, p in progress.items() if p["total"] > 0}
        not_started = len(users) - len(started_users)
        total_done = sum(p["done"] for p in progress.values())
        total_all = sum(p["total"] for p in started_users.values())

        if not started_users:
            status = "not started"
            all_done = False
        elif not_started > 0:
            # Some users haven't started yet
            status = f"{total_done}/{total_all} ({len(started_users)}/{len(users)} users)"
            all_done = False
        elif total_done >= total_all:
            status = "COMPLETE"
        else:
            status = f"{total_done}/{total_all} ({100*total_done/total_all:.0f}%)"
            all_done = False
        lines.append(f"  {v.upper()}: {status}")
    return "\n".join(lines), all_done


# ── Process Management ──────────────────────────────────────────────────

class PilotProcess:
    """Manages a single pilot run_pilot.py subprocess."""

    def __init__(self, version: str, users: str, model: str, delay: float, output_dir: Path):
        self.version = version
        self.users = users
        self.model = model
        self.delay = delay
        self.output_dir = output_dir
        self.process: subprocess.Popen | None = None
        self.log_path = output_dir / "logs" / f"night_{version}.log"
        self.start_time: datetime | None = None
        self.rate_limit_count = 0

    def start(self) -> None:
        """Launch the pilot subprocess."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self.log_path, "a")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_pilot.py"),
            "--version", self.version,
            "--users", self.users,
            "--model", self.model,
            "--delay", str(self.delay),
            "--output-dir", str(self.output_dir),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        self.start_time = datetime.now()
        logger.info(f"Started {self.version.upper()} (PID {self.process.pid})")

    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self) -> None:
        """Gracefully stop the subprocess."""
        if self.process and self.is_running():
            logger.info(f"Stopping {self.version.upper()} (PID {self.process.pid})")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)

    def check_rate_limit(self) -> bool:
        """Check if the log shows rate limit errors."""
        if not self.log_path.exists():
            return False
        try:
            # Read last 5KB of log
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 5000))
                tail = f.read().decode("utf-8", errors="replace")
            rate_patterns = [
                r"rate.?limit", r"too.?many.?requests", r"429",
                r"throttl", r"overloaded", r"capacity",
                r"weekly.*limit", r"quota.*exceeded",
            ]
            for pat in rate_patterns:
                if re.search(pat, tail, re.IGNORECASE):
                    return True
        except Exception:
            pass
        return False

    @property
    def returncode(self) -> int | None:
        if self.process is None:
            return None
        return self.process.returncode


class NightScheduler:
    """Orchestrates parallel pilot runs during idle hours."""

    def __init__(
        self,
        start_hour: int = DEFAULT_START_HOUR,
        stop_hour: int = DEFAULT_STOP_HOUR,
        force_now: bool = False,
    ):
        self.start_hour = start_hour
        self.stop_hour = stop_hour
        self.force_now = force_now
        self.users = [int(x) for x in USERS.split(",")]
        self.processes: list[PilotProcess] = []
        self._shutdown = False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown = True
        self._stop_all()

    def _stop_all(self):
        for p in self.processes:
            p.stop()

    def run(self):
        """Main scheduler loop."""
        logger.info("Night scheduler started")
        logger.info(f"  Run window: {self.start_hour}:00 - {self.stop_hour}:00")
        logger.info(f"  Versions: {VERSIONS}")
        logger.info(f"  Users: {self.users}")
        logger.info(f"  Output: {OUTPUT_DIR}")

        # Check what's already done
        report, all_done = format_progress_report(self.users)
        logger.info(report)
        if all_done:
            msg = "[proactive-agent] All pilot V2 runs already complete!"
            logger.info(msg)
            send_telegram(msg)
            return

        if self.force_now:
            logger.info("Force mode: skipping idle check, starting immediately")
            send_telegram(f"[proactive-agent] Night scheduler starting NOW (force mode)\n{report}")
            self._run_experiment_loop()
            return

        # Wait for idle window
        send_telegram(f"[proactive-agent] Night scheduler armed, waiting for idle window ({self.start_hour}:00-{self.stop_hour}:00)\n{report}")
        self._wait_for_idle()

        if self._shutdown:
            return

        send_telegram(f"[proactive-agent] Idle detected, starting pilot runs\n{report}")
        self._run_experiment_loop()

    def _wait_for_idle(self):
        """Block until we detect idle conditions."""
        while not self._shutdown:
            if is_user_idle(self.start_hour, self.stop_hour):
                logger.info("Idle conditions met")
                return
            now = datetime.now()
            if not is_within_window(self.start_hour, self.stop_hour):
                # Calculate time until next window opens
                target_hour = self.start_hour
                target = now.replace(hour=target_hour, minute=0, second=0)
                if target <= now:
                    target += timedelta(days=1)
                wait_sec = (target - now).total_seconds()
                logger.info(f"Outside run window. Sleeping {wait_sec/3600:.1f}h until {target_hour}:00")
                # Sleep in chunks so we can respond to signals
                end_time = time.time() + wait_sec
                while time.time() < end_time and not self._shutdown:
                    time.sleep(min(60, end_time - time.time()))
            else:
                logger.info("In window but not idle, checking again in 5 min")
                time.sleep(IDLE_CHECK_INTERVAL)

    def _run_experiment_loop(self):
        """Launch all incomplete versions in parallel and monitor them."""
        # Determine which versions still need work
        pending = [v for v in VERSIONS if not is_version_complete(v, self.users)]
        if not pending:
            logger.info("All versions complete!")
            return

        logger.info(f"Launching {len(pending)} versions: {pending}")

        # Launch all pending versions
        for v in pending:
            p = PilotProcess(v, USERS, MODEL, DELAY, OUTPUT_DIR)
            p.start()
            self.processes.append(p)
            time.sleep(2)  # Small stagger to avoid thundering herd

        # Monitor loop
        last_progress_time = time.time()
        last_report = ""
        consecutive_rate_limits = 0

        while not self._shutdown:
            time.sleep(30)  # Check every 30 seconds

            # Check if we should yield (outside run window and not force mode)
            if not self.force_now and not is_within_window(self.start_hour, self.stop_hour):
                logger.info("Run window ended, pausing all processes")
                self._stop_all()
                send_telegram("[proactive-agent] Run window ended, pausing experiments. Will resume tonight.")
                self._wait_for_idle()
                if self._shutdown:
                    break
                # Re-launch after coming back
                self.processes.clear()
                pending = [v for v in VERSIONS if not is_version_complete(v, self.users)]
                if not pending:
                    break
                send_telegram(f"[proactive-agent] Resuming {len(pending)} versions")
                for v in pending:
                    p = PilotProcess(v, USERS, MODEL, DELAY, OUTPUT_DIR)
                    p.start()
                    self.processes.append(p)
                    time.sleep(2)
                continue

            # Check for rate limits
            any_rate_limited = False
            for p in self.processes:
                if p.is_running() and p.check_rate_limit():
                    any_rate_limited = True
                    p.rate_limit_count += 1

            if any_rate_limited:
                consecutive_rate_limits += 1
                if consecutive_rate_limits >= 3:
                    # Persistent rate limit — pause all and wait
                    logger.warning("Persistent rate limit detected, pausing all for 30 min")
                    self._stop_all()
                    send_telegram("[proactive-agent] Rate limit hit, pausing 30 min then retrying")
                    wait_end = time.time() + RATE_LIMIT_WAIT
                    while time.time() < wait_end and not self._shutdown:
                        time.sleep(30)
                    if self._shutdown:
                        break
                    # Re-launch
                    consecutive_rate_limits = 0
                    self.processes.clear()
                    pending = [v for v in VERSIONS if not is_version_complete(v, self.users)]
                    if not pending:
                        break
                    logger.info(f"Resuming after rate limit pause: {pending}")
                    for v in pending:
                        p = PilotProcess(v, USERS, MODEL, DELAY, OUTPUT_DIR)
                        p.start()
                        self.processes.append(p)
                        time.sleep(2)
                    continue
            else:
                consecutive_rate_limits = 0

            # Check for completed/crashed processes
            running = [p for p in self.processes if p.is_running()]
            finished = [p for p in self.processes if not p.is_running() and p.process is not None]

            for p in finished:
                if p.returncode == 0:
                    logger.info(f"{p.version.upper()} completed successfully")
                else:
                    logger.warning(f"{p.version.upper()} exited with code {p.returncode}")
                    # Check if it was a rate limit crash — will be retried
                    if p.check_rate_limit():
                        logger.info(f"{p.version.upper()} crashed from rate limit, will retry")

            # Remove finished processes
            self.processes = [p for p in self.processes if p.is_running()]

            # Re-launch crashed versions that aren't complete
            for p in finished:
                if p.returncode != 0 and not is_version_complete(p.version, self.users):
                    # Wait a bit before restarting crashed process
                    time.sleep(10)
                    new_p = PilotProcess(p.version, USERS, MODEL, DELAY, OUTPUT_DIR)
                    new_p.rate_limit_count = p.rate_limit_count
                    new_p.start()
                    self.processes.append(new_p)

            # Periodic progress report
            if time.time() - last_progress_time > PROGRESS_INTERVAL:
                report, all_done = format_progress_report(self.users)
                if report != last_report:
                    logger.info(report)
                    last_report = report
                last_progress_time = time.time()

                if all_done:
                    logger.info("All versions complete!")
                    break

            # All processes exited and nothing to relaunch
            if not self.processes:
                report, all_done = format_progress_report(self.users)
                if all_done:
                    break
                else:
                    # Some versions incomplete but no processes running — try relaunching
                    pending = [v for v in VERSIONS if not is_version_complete(v, self.users)]
                    if pending:
                        logger.info(f"No processes running but {len(pending)} versions incomplete, relaunching")
                        for v in pending:
                            p = PilotProcess(v, USERS, MODEL, DELAY, OUTPUT_DIR)
                            p.start()
                            self.processes.append(p)
                            time.sleep(2)
                    else:
                        break

        # Final summary
        self._stop_all()
        report, all_done = format_progress_report(self.users)
        status = "COMPLETE" if all_done else "PAUSED"
        msg = f"[proactive-agent] Night scheduler {status}\n{report}"
        logger.info(msg)
        send_telegram(msg)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Night scheduler for pilot experiments")
    parser.add_argument("--start-hour", type=int, default=DEFAULT_START_HOUR,
                        help=f"Start of run window (default: {DEFAULT_START_HOUR})")
    parser.add_argument("--stop-hour", type=int, default=DEFAULT_STOP_HOUR,
                        help=f"End of run window (default: {DEFAULT_STOP_HOUR})")
    parser.add_argument("--now", action="store_true",
                        help="Skip idle detection, run immediately")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "logs" / "night_scheduler.log"),
        ],
    )

    # Ensure log dir exists
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    scheduler = NightScheduler(
        start_hour=args.start_hour,
        stop_hour=args.stop_hour,
        force_now=args.now,
    )
    scheduler.run()


if __name__ == "__main__":
    main()
