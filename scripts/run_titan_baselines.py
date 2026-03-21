#!/usr/bin/env python3
"""Resource-aware orchestrator: runs ML/DL baselines on Titan server via SSH.

Runs on the local iMac, dispatches jobs to Titan (zhiyuan@172.29.39.82)
with resource guardrails (CPU load, GPU memory) and Telegram notifications.

Usage:
    PYTHONPATH=. python3 scripts/run_titan_baselines.py
    PYTHONPATH=. python3 scripts/run_titan_baselines.py --steps ml,lstm  # subset
    PYTHONPATH=. python3 scripts/run_titan_baselines.py --dry-run        # show commands only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TITAN_HOST = "zhiyuan@172.29.39.82"
TITAN_PROJECT = "/home/zhiyuan/proactive-affective-agent"
CONDA_ACTIVATE = "source ~/anaconda3/etc/profile.d/conda.sh && conda activate efficient-ser"

# Resource limits
MAX_CPU_LOAD = 20          # 50% of 40 cores
MAX_GPU_MEM_MB = 30_000    # 30GB on A6000 (user limit)
GPU_DEVICE = "1"           # A6000

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7542082932")

LOCAL_PROJECT = Path(__file__).resolve().parent.parent


def send_telegram(msg: str) -> None:
    """Send a Telegram notification via Boo bot."""
    try:
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"chat_id": int(TELEGRAM_CHAT_ID), "text": msg}),
            ],
            capture_output=True, timeout=15,
        )
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


def ssh_run(cmd: str, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command on Titan via SSH."""
    full_cmd = ["ssh", "-o", "ConnectTimeout=10", TITAN_HOST, cmd]
    logger.info(f"SSH: {cmd[:120]}...")
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        logger.error(f"SSH command failed (rc={result.returncode}): {result.stderr[:500]}")
    return result


class TitanOrchestrator:
    """Resource-aware job orchestrator for Titan server."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def check_resources(self) -> dict:
        """Check CPU load and GPU memory usage on Titan."""
        info = {"cpu_load": 99.0, "gpu_used_mb": 99999}

        # CPU load (1-min average)
        result = ssh_run("uptime", check=False)
        if result.returncode == 0:
            try:
                load_str = result.stdout.strip().split("load average:")[1].split(",")[0].strip()
                info["cpu_load"] = float(load_str)
            except (IndexError, ValueError):
                pass

        # GPU memory (device 1 = A6000)
        result = ssh_run(
            f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {GPU_DEVICE}",
            check=False,
        )
        if result.returncode == 0:
            try:
                info["gpu_used_mb"] = int(result.stdout.strip())
            except ValueError:
                pass

        logger.info(f"Titan resources: CPU load={info['cpu_load']:.1f}, GPU{GPU_DEVICE} used={info['gpu_used_mb']}MB")
        return info

    def is_safe(self, job_type: str) -> bool:
        """Check if it's safe to launch a job of the given type."""
        info = self.check_resources()
        if job_type == "cpu":
            return info["cpu_load"] < MAX_CPU_LOAD
        elif job_type == "gpu":
            return info["gpu_used_mb"] < MAX_GPU_MEM_MB and info["cpu_load"] < MAX_CPU_LOAD
        return False

    def wait_if_busy(self, job_type: str, max_wait_min: int = 60) -> bool:
        """Poll resources every 5 minutes until safe, up to max_wait_min."""
        waited = 0
        while not self.is_safe(job_type):
            if waited >= max_wait_min:
                logger.error(f"Resources busy after {max_wait_min}min — aborting")
                return False
            logger.info(f"Resources busy, waiting 5 min... ({waited}/{max_wait_min}min)")
            time.sleep(300)
            waited += 5
        return True

    def run_job(self, cmd: str, job_type: str = "cpu", timeout: int = 7200) -> subprocess.CompletedProcess:
        """Run a job on Titan with resource guardrails."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would run: {cmd[:200]}")
            return subprocess.CompletedProcess([], 0, stdout="[dry run]", stderr="")

        if not self.wait_if_busy(job_type):
            send_telegram(f"[baselines] Resource wait timeout for {job_type} job, skipping")
            return subprocess.CompletedProcess([], 1, stdout="", stderr="resource timeout")

        # Build environment prefix
        env_parts = [CONDA_ACTIVATE]
        if job_type == "gpu":
            env_parts.append(f"export CUDA_VISIBLE_DEVICES={GPU_DEVICE}")
        env_parts.extend([
            "export OMP_NUM_THREADS=8",
            "export MKL_NUM_THREADS=8",
            f"export PYTHONPATH={TITAN_PROJECT}",
        ])
        env_prefix = " && ".join(env_parts)

        full_cmd = f"{env_prefix} && cd {TITAN_PROJECT} && nice -n 19 {cmd}"
        return ssh_run(full_cmd, timeout=timeout, check=False)

    def rsync_to_titan(self) -> bool:
        """Sync latest code to Titan."""
        logger.info("Syncing code to Titan...")
        if self.dry_run:
            logger.info("[DRY RUN] Would rsync to Titan")
            return True

        result = subprocess.run(
            [
                "rsync", "-az", "--delete",
                "--exclude", "__pycache__",
                "--exclude", ".git",
                "--exclude", "outputs",
                "--exclude", "data",
                "--exclude", ".claude",
                "--exclude", "*.pyc",
                f"{LOCAL_PROJECT}/src/",
                f"{TITAN_HOST}:{TITAN_PROJECT}/src/",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"rsync src failed: {result.stderr[:300]}")
            return False

        # Also sync scripts
        result = subprocess.run(
            [
                "rsync", "-az",
                "--exclude", "__pycache__",
                f"{LOCAL_PROJECT}/scripts/",
                f"{TITAN_HOST}:{TITAN_PROJECT}/scripts/",
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.error(f"rsync scripts failed: {result.stderr[:300]}")
            return False

        logger.info("Code synced successfully")
        return True

    def rsync_results_back(self) -> bool:
        """Pull results from Titan to local."""
        logger.info("Pulling results from Titan...")
        if self.dry_run:
            logger.info("[DRY RUN] Would rsync results back")
            return True

        local_outputs = LOCAL_PROJECT / "outputs"
        local_outputs.mkdir(exist_ok=True)

        for subdir in ["ml_baselines", "advanced_baselines"]:
            result = subprocess.run(
                [
                    "rsync", "-az",
                    f"{TITAN_HOST}:{TITAN_PROJECT}/outputs/{subdir}/",
                    f"{local_outputs}/{subdir}/",
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(f"rsync {subdir} back failed: {result.stderr[:200]}")
        return True

    # ------------------------------------------------------------------
    # Job steps
    # ------------------------------------------------------------------

    def run_ml_baselines(self) -> bool:
        """Run ML baselines (RF, XGB, SVM, Ridge, LogReg) — all 5 folds, CPU."""
        logger.info("=== Running ML baselines (CPU) ===")
        cmd = (
            "python3 scripts/run_ml_baselines.py "
            "--features parquet "
            "--models rf,xgboost,logistic,ridge,svm "
            "--n-jobs 8"
        )
        result = self.run_job(cmd, job_type="cpu", timeout=7200)
        if result.returncode != 0:
            logger.error(f"ML baselines failed: {result.stderr[:300]}")
            return False
        logger.info("ML baselines completed")
        return True

    def run_mlp_verify(self) -> bool:
        """Verify/re-run MLP baseline — GPU."""
        logger.info("=== Verifying MLP baseline (GPU) ===")
        cmd = (
            "python3 scripts/run_dl_baselines.py "
            "--pipelines dl"
        )
        result = self.run_job(cmd, job_type="gpu", timeout=5400)
        if result.returncode != 0:
            logger.error(f"MLP verify failed: {result.stderr[:300]}")
            return False
        logger.info("MLP verify completed")
        return True

    def run_lstm(self) -> bool:
        """Run LSTM baseline — GPU."""
        logger.info("=== Running LSTM baseline (GPU) ===")
        cmd = (
            "python3 scripts/run_dl_baselines.py "
            "--pipelines lstm"
        )
        result = self.run_job(cmd, job_type="gpu", timeout=7200)
        if result.returncode != 0:
            logger.error(f"LSTM failed: {result.stderr[:300]}")
            return False
        logger.info("LSTM completed")
        return True

    def run_combined(self) -> bool:
        """Run combined baseline — CPU+GPU."""
        logger.info("=== Running combined baseline ===")
        cmd = (
            "python3 scripts/run_dl_baselines.py "
            "--pipelines combined"
        )
        result = self.run_job(cmd, job_type="gpu", timeout=5400)
        if result.returncode != 0:
            logger.error(f"Combined failed: {result.stderr[:300]}")
            return False
        logger.info("Combined completed")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate ML/DL baselines on Titan server")
    parser.add_argument(
        "--steps", type=str, default="sync,ml,mlp,lstm,combined,pull",
        help="Comma-separated steps: sync, ml, mlp, lstm, combined, pull"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    steps = [s.strip().lower() for s in args.steps.split(",")]
    orch = TitanOrchestrator(dry_run=args.dry_run)

    send_telegram("[baselines] Starting ML/DL baseline pipeline on Titan")

    results = {}
    failed = False

    try:
        if "sync" in steps:
            if not orch.rsync_to_titan():
                send_telegram("[baselines] Code sync failed — aborting")
                sys.exit(1)

        if "ml" in steps:
            ok = orch.run_ml_baselines()
            results["ml"] = ok
            if not ok:
                failed = True

        if "mlp" in steps:
            ok = orch.run_mlp_verify()
            results["mlp"] = ok
            if not ok:
                failed = True

        if "lstm" in steps:
            ok = orch.run_lstm()
            results["lstm"] = ok
            if not ok:
                failed = True

        if "combined" in steps:
            ok = orch.run_combined()
            results["combined"] = ok
            if not ok:
                failed = True

        if "pull" in steps:
            orch.rsync_results_back()

    except Exception as e:
        send_telegram(f"[baselines] Orchestrator crashed: {e}")
        raise

    # Summary
    summary_parts = []
    for step, ok in results.items():
        status = "OK" if ok else "FAILED"
        summary_parts.append(f"{step}: {status}")
    summary = ", ".join(summary_parts)

    if failed:
        send_telegram(f"[baselines] Completed with errors: {summary}")
    else:
        send_telegram(f"[baselines] All baselines completed successfully: {summary}")

    logger.info(f"Done: {summary}")


if __name__ == "__main__":
    main()
