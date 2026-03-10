#!/usr/bin/env python3
"""Clean fallback entries from pilot_v2 checkpoints.

Scans agentic version checkpoints (v2/v4/v5/v6) for fallback predictions.
Truncates each checkpoint to the last good entry before the first fallback.
Backs up originals to checkpoints/_backup_pre_clean/.
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = PROJECT_ROOT / "outputs" / "pilot_v2"
CHECKPOINT_DIR = PILOT_DIR / "checkpoints"
BACKUP_DIR = CHECKPOINT_DIR / "_backup_pre_clean"

AGENTIC_VERSIONS = ["v2", "v4", "v5", "v6"]
USERS = [43, 258, 338, 399, 403]


def is_fallback(pred: dict) -> bool:
    """Check if a prediction is a fallback."""
    reasoning = str(pred.get("reasoning", ""))
    return "fallback" in reasoning.lower()


def clean_checkpoint(cp_path: Path) -> dict:
    """Clean a single checkpoint. Returns stats."""
    with open(cp_path) as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    gts = data.get("ground_truths", [])
    meta = data.get("metadata", [])

    total = len(preds)
    if total == 0:
        return {"total": 0, "fallbacks": 0, "kept": 0, "action": "empty"}

    # Count fallbacks
    n_fallback = sum(1 for p in preds if is_fallback(p))
    if n_fallback == 0:
        return {"total": total, "fallbacks": 0, "kept": total, "action": "clean"}

    # Find the first fallback entry index
    first_fallback_idx = next(i for i, p in enumerate(preds) if is_fallback(p))

    # Truncate to before the first fallback
    kept = first_fallback_idx
    data["predictions"] = preds[:kept]
    data["ground_truths"] = gts[:kept]
    data["metadata"] = meta[:kept]
    data["n_entries"] = kept
    data["current_entry"] = kept - 1 if kept > 0 else -1

    # Backup original
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / cp_path.name
    if not backup_path.exists():
        shutil.copy2(cp_path, backup_path)

    # Save cleaned checkpoint
    with open(cp_path, "w") as f:
        json.dump(data, f, default=str)

    return {
        "total": total,
        "fallbacks": n_fallback,
        "kept": kept,
        "first_fallback": first_fallback_idx,
        "action": "truncated",
    }


def main():
    if not CHECKPOINT_DIR.exists():
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        sys.exit(1)

    print(f"Scanning checkpoints in {CHECKPOINT_DIR}")
    print(f"Agentic versions: {AGENTIC_VERSIONS}")
    print(f"Users: {USERS}")
    print()

    total_cleaned = 0
    total_entries_removed = 0

    for ver in AGENTIC_VERSIONS:
        for uid in USERS:
            cp_path = CHECKPOINT_DIR / f"{ver}_user{uid}_checkpoint.json"
            if not cp_path.exists():
                print(f"  {ver} user{uid}: NO CHECKPOINT")
                continue

            stats = clean_checkpoint(cp_path)
            action = stats["action"]
            if action == "clean":
                print(f"  {ver} user{uid}: {stats['total']} entries, no fallbacks ✅")
            elif action == "empty":
                print(f"  {ver} user{uid}: empty checkpoint")
            elif action == "truncated":
                removed = stats["total"] - stats["kept"]
                total_cleaned += 1
                total_entries_removed += removed
                print(
                    f"  {ver} user{uid}: {stats['total']} → {stats['kept']} entries "
                    f"(removed {removed}, first fallback at #{stats['first_fallback']}) ⚠️"
                )

    print(f"\nSummary: cleaned {total_cleaned} checkpoints, removed {total_entries_removed} entries total")
    if total_cleaned > 0:
        print(f"Backups saved to: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
