#!/usr/bin/env python3
"""Scan Codex-mini pilot outputs for invalid records that require rerun."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = {
    "PANAS_Pos",
    "PANAS_Neg",
    "ER_desire",
    "INT_availability",
    "confidence",
    "reasoning",
}
TOOL_REFUSAL_MARKERS = (
    "tools unavailable",
    "tool is unavailable",
    "need to be enabled",
    "enabling the tool calls",
    "system isn’t letting me issue requests",
    "system isn't letting me issue requests",
)


def classify_prediction(pred: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(pred, dict):
        return ["non_dict_prediction"]
    if not pred:
        return ["empty_prediction"]
    missing = sorted(REQUIRED_KEYS - set(pred.keys()))
    if missing:
        reasons.append("missing_required_keys")
    reasoning = str(pred.get("reasoning", "") or "").lower()
    if "fallback" in reasoning:
        reasons.append("fallback_reasoning")
    return reasons


def classify_record(obj: dict[str, Any]) -> list[str]:
    reasons = classify_prediction(obj.get("prediction"))
    full_response = str(obj.get("full_response", "") or "").lower()
    reasoning = str(obj.get("reasoning", "") or "").lower()
    if any(marker in full_response for marker in TOOL_REFUSAL_MARKERS) or any(
        marker in reasoning for marker in TOOL_REFUSAL_MARKERS
    ):
        reasons.append("tool_refusal")
    if obj.get("prediction") and not full_response and not reasoning:
        reasons.append("empty_trace")
    return sorted(set(reasons))


def scan_output_dir(base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec_path in sorted(base.glob("group*/gpt-v*_user*_records.jsonl")):
        bad_entries: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        with rec_path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_entries.append(
                        {
                            "entry_idx": None,
                            "lineno": lineno,
                            "reasons": ["jsonl_decode_error"],
                        }
                    )
                    counts["jsonl_decode_error"] += 1
                    continue
                reasons = classify_record(obj)
                if reasons:
                    bad_entries.append(
                        {
                            "entry_idx": obj.get("entry_idx"),
                            "lineno": lineno,
                            "reasons": reasons,
                        }
                    )
                    counts.update(reasons)
        if bad_entries:
            rows.append(
                {
                    "path": str(rec_path),
                    "bad_count": len(bad_entries),
                    "first_bad_entry_idx": bad_entries[0]["entry_idx"],
                    "reason_counts": dict(sorted(counts.items())),
                    "entries": bad_entries,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=["outputs/pilot_codexmini", "outputs/pilot_codexmini_v2"],
        help="Output directories to scan",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {}
    total_bad_files = 0
    total_bad_entries = 0
    aggregate_counts: Counter[str] = Counter()

    for raw_base in args.output_dirs:
        base = PROJECT_ROOT / raw_base
        rows = scan_output_dir(base)
        report[str(base)] = rows
        total_bad_files += len(rows)
        for row in rows:
            total_bad_entries += row["bad_count"]
            aggregate_counts.update(row["reason_counts"])

    if args.json:
        print(
            json.dumps(
                {
                    "total_bad_files": total_bad_files,
                    "total_bad_entries": total_bad_entries,
                    "aggregate_reason_counts": dict(sorted(aggregate_counts.items())),
                    "outputs": report,
                },
                indent=2,
            )
        )
        return

    print(f"bad_files={total_bad_files}")
    print(f"bad_entries={total_bad_entries}")
    print(f"aggregate_reason_counts={dict(sorted(aggregate_counts.items()))}")
    for base, rows in report.items():
        print(f"\n[{base}]")
        if not rows:
            print("clean")
            continue
        for row in rows:
            print(
                f"{row['path']} bad={row['bad_count']} first_bad={row['first_bad_entry_idx']} "
                f"reasons={row['reason_counts']}"
            )


if __name__ == "__main__":
    main()
