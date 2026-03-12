#!/usr/bin/env python3
"""Scan, truncate, and rerun bad Codex-mini tasks with a fixed worker limit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scan_bad_codexmini import scan_output_dir


MODEL = "gpt-5.1-codex-mini"
DELAY = "0.8"
DEFAULT_OUTPUT_DIRS = [
    PROJECT_ROOT / "outputs" / "pilot_codexmini",
    PROJECT_ROOT / "outputs" / "pilot_codexmini_v2",
]


def parse_task_identity(record_path: Path) -> tuple[Path, str, str, int]:
    output_dir = record_path.parents[1]
    group = record_path.parent.name
    stem = record_path.stem.replace("_records", "")
    version, user_part = stem.split("_user", 1)
    return output_dir, group, version, int(user_part)


def truncate_one(record_path: Path, first_bad_entry_idx: int, backup_root: Path) -> dict:
    output_dir, group, version, uid = parse_task_identity(record_path)
    group_dir = output_dir / group
    cp_path = group_dir / "checkpoints" / f"{version}_user{uid}_checkpoint.json"
    mem_path = group_dir / "memory" / f"{version}_user_{uid:03d}_session.md"

    backup_dir = backup_root / output_dir.name / group
    (backup_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (backup_dir / "records").mkdir(parents=True, exist_ok=True)
    (backup_dir / "memory").mkdir(parents=True, exist_ok=True)

    shutil.copy2(record_path, backup_dir / "records" / record_path.name)
    if cp_path.exists():
        shutil.copy2(cp_path, backup_dir / "checkpoints" / cp_path.name)
    if mem_path.exists():
        shutil.copy2(mem_path, backup_dir / "memory" / mem_path.name)

    if cp_path.exists():
        data = json.loads(cp_path.read_text(encoding="utf-8"))
        preds = list(data.get("predictions", []))[:first_bad_entry_idx]
        gts = list(data.get("ground_truths", []))[:first_bad_entry_idx]
        meta = list(data.get("metadata", []))[:first_bad_entry_idx]
        data["predictions"] = preds
        data["ground_truths"] = gts
        data["metadata"] = meta
        data["n_entries"] = len(preds)
        data["current_user"] = uid
        data["current_entry"] = first_bad_entry_idx - 1
        cp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    kept_lines = []
    with record_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if int(obj.get("entry_idx", 10**9)) < first_bad_entry_idx:
                kept_lines.append(json.dumps(obj, ensure_ascii=False))
    record_path.write_text(("\n".join(kept_lines) + ("\n" if kept_lines else "")), encoding="utf-8")

    if mem_path.exists():
        text = mem_path.read_text(encoding="utf-8")
        parts = text.split("\n- **")
        header = parts[0]
        kept_entries = parts[1:first_bad_entry_idx + 1]
        rebuilt = header.rstrip() + "\n"
        if kept_entries:
            rebuilt = rebuilt + "\n- **" + "\n- **".join(kept_entries)
        mem_path.write_text(rebuilt, encoding="utf-8")

    return {
        "output_dir": output_dir,
        "group": group,
        "version": version,
        "uid": uid,
        "first_bad_entry_idx": first_bad_entry_idx,
    }


def collect_tasks(output_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    truncations = []
    grouped: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = PROJECT_ROOT / "outputs" / "_rerun_backups" / f"final_cleanup_{stamp}"

    for output_dir in output_dirs:
        for row in scan_output_dir(output_dir):
            first_bad = row["first_bad_entry_idx"]
            if first_bad is None:
                continue
            rec_path = Path(row["path"])
            truncation = truncate_one(rec_path, int(first_bad), backup_root)
            truncations.append(truncation)
            key = (truncation["output_dir"].name, truncation["group"], truncation["version"])
            grouped[key].add(truncation["uid"])

    tasks = []
    for (output_name, group, version), users in sorted(grouped.items()):
        tasks.append(
            {
                "output_name": output_name,
                "group": group,
                "version": version,
                "users": sorted(users),
                "output_dir": str(PROJECT_ROOT / "outputs" / output_name / group),
            }
        )
    return truncations, tasks


def run_tasks(tasks: list[dict], workers: int, log_root: Path) -> int:
    if not tasks:
        return 0
    log_root.mkdir(parents=True, exist_ok=True)
    commands_path = log_root / "commands.txt"
    lines = []
    for task in tasks:
        users = ",".join(str(u) for u in task["users"])
        lines.append(
            " ".join(
                [
                    "uv", "run", "python", "scripts/run_pilot.py",
                    "--version", task["version"],
                    "--users", users,
                    "--model", MODEL,
                    "--delay", DELAY,
                    "--output-dir", task["output_dir"],
                ]
            )
        )
    commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    queue_log = log_root / "queue.log"
    queue_log.write_text(f"[queue] start {datetime.now().isoformat()}\n", encoding="utf-8")
    cmd = f'cat "{commands_path}" | xargs -I{{}} -P {workers} bash -lc "{{}}"'
    with queue_log.open("a", encoding="utf-8") as log_f:
        return subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            stdout=log_f,
            stderr=subprocess.STDOUT,
        ).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dirs", nargs="+", default=[str(p) for p in DEFAULT_OUTPUT_DIRS])
    args = parser.parse_args()

    output_dirs = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in args.output_dirs]

    if args.dry_run:
        for output_dir in output_dirs:
            rows = scan_output_dir(output_dir)
            print(output_dir)
            for row in rows:
                print(row["path"], row["bad_count"], row["first_bad_entry_idx"], row["reason_counts"])
        return

    truncations, tasks = collect_tasks(output_dirs)
    print(f"truncated={len(truncations)}")
    for item in truncations:
        print(
            f"truncate {item['output_dir'].name}/{item['group']} {item['version']} "
            f"user{item['uid']} from entry {item['first_bad_entry_idx']}"
        )

    print(f"tasks={len(tasks)}")
    for task in tasks:
        print(f"run {task['version']} {task['output_name']}/{task['group']} users={task['users']}")

    log_root = PROJECT_ROOT / "outputs" / "_rerun_logs" / f"final_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rc = run_tasks(tasks, args.workers, log_root)
    print(f"log_root={log_root}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
