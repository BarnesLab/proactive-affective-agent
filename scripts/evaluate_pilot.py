"""Evaluate pilot study results from checkpoint files.

Computes MAE, BA, F1 for all versions (CALLM, V1-V6) across 18 primary
users and targets. Outputs a formatted comparison table and saves results
to outputs/pilot/evaluation.json.

Updated 2026-03-17: expanded to 18 clean complete users with V2/V4/V5/V6.
Added --users CLI override and corrupted-checkpoint filtering.

Usage:
    PYTHONPATH=. python3 scripts/evaluate_pilot.py
    PYTHONPATH=. python3 scripts/evaluate_pilot.py --output outputs/pilot/evaluation.json
    PYTHONPATH=. python3 scripts/evaluate_pilot.py --users 24 43 71
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error

from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Primary evaluation set: 18 users with clean complete V2/V4/V5/V6
# Updated 2026-03-17
PRIMARY_USERS = [24, 43, 71, 119, 164, 232, 242, 258, 275, 310, 338, 362, 399, 403, 437, 458, 505, 513]
PILOT_USERS = PRIMARY_USERS  # backward compat
VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]

# Multiple checkpoint directories (newer takes precedence)
DEFAULT_CHECKPOINT_DIRS = [
    "outputs/pilot_v2/checkpoints",
    "outputs/pilot/checkpoints",
]

# Targets to show prominently in the summary table
KEY_CONTINUOUS = ["PANAS_Pos", "PANAS_Neg", "ER_desire"]
KEY_BINARY = [
    "Individual_level_happy_State",
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_sad_State",
    "Individual_level_worried_State",
    "INT_availability",
]

# AR baseline values (from outputs/ar_baseline/ar_results.json)
AR_BASELINES = {
    "PANAS_Pos": {"mae": 2.758, "ba": 0.658},
    "PANAS_Neg": {"mae": 2.140, "ba": 0.658},
    "ER_desire": {"mae": 1.012, "ba": 0.658},
    "mean_ba": 0.658,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_checkpoint(version: str, uid: int, checkpoint_dirs: list[Path]) -> Path | None:
    """Find checkpoint file across multiple dirs, trying case variants for CALLM."""
    names = [f"{version}_user{uid}_checkpoint.json"]
    if version.lower() == "callm":
        names.append(f"CALLM_user{uid}_checkpoint.json")
    for cp_dir in checkpoint_dirs:
        for name in names:
            f = cp_dir / name
            if f.exists():
                return f
    return None


def _get_expected_entries() -> dict[int, int]:
    """Build a mapping of user_id -> expected EMA entry count from CSV test splits.

    Scans all group_*_test.csv files and counts rows per Study_ID.
    Each user appears in exactly one test fold, so the count is unique.
    """
    splits_dir = Path("data/processed/splits")
    expected: dict[int, int] = {}
    if not splits_dir.exists():
        return expected
    for csv_file in sorted(splits_dir.glob("group_*_test.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        if "Study_ID" not in df.columns:
            continue
        for uid, count in df["Study_ID"].value_counts().items():
            expected[int(uid)] = int(count)
    return expected


def load_checkpoints(version: str, checkpoint_dirs: list[Path]) -> tuple[list, list]:
    """Load all predictions and ground truths for a version across pilot users.

    Filters out corrupted checkpoints where the number of predictions
    exceeds 1.5x the expected entry count (looked up from CSV test splits).
    """
    expected_entries = _get_expected_entries()
    all_preds, all_gts = [], []
    for uid in PILOT_USERS:
        f = find_checkpoint(version, uid, checkpoint_dirs)
        if f is None:
            continue
        data = json.loads(f.read_text())
        preds = data.get("predictions", [])
        gts = data.get("ground_truths", [])

        # Filter out corrupted checkpoints (predictions > 1.5x expected)
        if uid in expected_entries:
            expected = expected_entries[uid]
            if len(preds) > 1.5 * expected:
                print(f"    WARNING: skipping corrupted checkpoint {f.name} "
                      f"({len(preds)} preds vs {expected} expected)")
                continue

        all_preds.extend(preds)
        all_gts.extend(gts)
    return all_preds, all_gts


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _to_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.lower().strip()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def compute_continuous_metrics(
    preds: list[dict], gts: list[dict], target: str
) -> dict | None:
    """Compute MAE for a continuous target."""
    y, yhat = [], []
    for p, g in zip(preds, gts):
        gv = _to_float(g.get(target))
        pv = _to_float(p.get(target))
        if gv is not None and pv is not None:
            y.append(gv)
            yhat.append(pv)
    if len(y) < 5:
        return None
    y, yhat = np.array(y), np.array(yhat)
    return {
        "mae": float(mean_absolute_error(y, yhat)),
        "pred_mean": float(np.mean(yhat)),
        "pred_std": float(np.std(yhat)),
        "gt_mean": float(np.mean(y)),
        "gt_std": float(np.std(y)),
        "n": len(y),
    }


def compute_binary_metrics(
    preds: list[dict], gts: list[dict], target: str
) -> dict | None:
    """Compute BA and F1 for a binary target."""
    y, yhat = [], []
    for p, g in zip(preds, gts):
        gv = _to_bool(g.get(target))
        pv = _to_bool(p.get(target))
        if gv is not None and pv is not None:
            y.append(int(gv))
            yhat.append(int(pv))
    if len(y) < 5:
        return None
    return {
        "ba": float(balanced_accuracy_score(y, yhat)),
        # macro F1: average of F1 for both classes (positive and negative)
        "f1": float(f1_score(y, yhat, average='macro', zero_division=0)),
        "n": len(y),
        "pos_rate_gt": float(np.mean(y)),
        "pos_rate_pred": float(np.mean(yhat)),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_version(
    version: str, checkpoint_dirs: list[Path]
) -> dict:
    """Compute all metrics for a single version."""
    preds, gts = load_checkpoints(version, checkpoint_dirs)
    if not preds:
        return {"n_entries": 0, "available": False}

    results: dict = {"n_entries": len(preds), "available": True}

    # Continuous targets
    results["continuous"] = {}
    for target in CONTINUOUS_TARGETS:
        m = compute_continuous_metrics(preds, gts, target)
        if m:
            results["continuous"][target] = m

    # Binary targets
    results["binary"] = {}
    for target in BINARY_STATE_TARGETS + ["INT_availability"]:
        m = compute_binary_metrics(preds, gts, target)
        if m:
            results["binary"][target] = m

    # Aggregate metrics
    all_maes = [v["mae"] for v in results["continuous"].values()]
    all_bas = [v["ba"] for v in results["binary"].values()]
    results["aggregate"] = {
        "mean_mae": float(np.mean(all_maes)) if all_maes else None,
        "mean_ba": float(np.mean(all_bas)) if all_bas else None,
        "mean_f1": float(np.mean([v["f1"] for v in results["binary"].values()])) if results["binary"] else None,
    }

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary_table(all_results: dict[str, dict]) -> None:
    """Print a formatted comparison table."""
    available = [v for v in VERSIONS if all_results.get(v, {}).get("available")]
    if not available:
        print("No results available.")
        return

    # Header
    col_w = 10
    header = f"{'Target':<38}"
    for v in available:
        n = all_results[v]["n_entries"]
        header += f"{v.upper():>{col_w}}"
    print("\n" + "=" * (38 + col_w * len(available)))
    print(f"PILOT EVALUATION — {', '.join(available)} (n per version shown)")
    print("=" * (38 + col_w * len(available)))

    # Continuous targets
    print(f"\n{'── MAE (lower is better) ──':}")
    print(f"\n  AR baseline: PANAS_Pos={AR_BASELINES['PANAS_Pos']['mae']:.3f}  (autocorrelation ceiling)\n")
    for target in KEY_CONTINUOUS:
        row = f"  {target:<36}"
        for v in available:
            m = all_results[v].get("continuous", {}).get(target)
            row += f"{m['mae']:>{col_w}.3f}" if m else f"{'—':>{col_w}}"
        # Show GT stats
        ref = next((all_results[v]["continuous"].get(target) for v in available if all_results[v]["continuous"].get(target)), None)
        if ref:
            row += f"  [GT: μ={ref['gt_mean']:.1f}±{ref['gt_std']:.1f}]"
        print(row)

    # Binary targets
    print(f"\n{'── Balanced Accuracy (higher is better) ──':}")
    print(f"\n  AR baseline: BA={AR_BASELINES['mean_ba']:.3f}  (autocorrelation ceiling)\n")
    for target in KEY_BINARY:
        row = f"  {target:<36}"
        for v in available:
            m = all_results[v].get("binary", {}).get(target)
            row += f"{m['ba']:>{col_w}.3f}" if m else f"{'—':>{col_w}}"
        print(row)

    # Aggregates
    print(f"\n{'── Aggregate Metrics ──':}")
    row = f"  {'Mean MAE':<36}"
    for v in available:
        agg = all_results[v].get("aggregate", {})
        val = agg.get("mean_mae")
        row += f"{val:>{col_w}.3f}" if val else f"{'—':>{col_w}}"
    print(row)

    row = f"  {'Mean BA':<36}"
    for v in available:
        agg = all_results[v].get("aggregate", {})
        val = agg.get("mean_ba")
        row += f"{val:>{col_w}.3f}" if val else f"{'—':>{col_w}}"
    print(row)

    row = f"  {'Mean F1':<36}"
    for v in available:
        agg = all_results[v].get("aggregate", {})
        val = agg.get("mean_f1")
        row += f"{val:>{col_w}.3f}" if val else f"{'—':>{col_w}}"
    print(row)

    # Prediction calibration
    print(f"\n{'── Prediction Calibration (PANAS_Pos) ──':}")
    for v in available:
        m = all_results[v].get("continuous", {}).get("PANAS_Pos")
        if m:
            print(f"  {v.upper():<8}: pred μ={m['pred_mean']:.1f}±{m['pred_std']:.1f}  GT μ={m['gt_mean']:.1f}±{m['gt_std']:.1f}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate pilot study results")
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        default=None,
        help="Checkpoint directories (searched in order, first match wins)",
    )
    parser.add_argument(
        "--output",
        default="outputs/pilot_v2/evaluation.json",
        help="Where to save evaluation results JSON",
    )
    parser.add_argument(
        "--users", nargs="+", type=int, default=None,
        help="Override user list",
    )
    args = parser.parse_args()

    if args.checkpoint_dirs:
        cp_dirs = [Path(d) for d in args.checkpoint_dirs]
    else:
        cp_dirs = [Path(d) for d in DEFAULT_CHECKPOINT_DIRS]
    output_path = Path(args.output)

    print(f"Evaluating from: {[str(d) for d in cp_dirs]}")
    print(f"Users: {PILOT_USERS} ({len(PILOT_USERS)} participants)")

    all_results: dict[str, dict] = {}
    for version in VERSIONS:
        print(f"  {version}...", end=" ", flush=True)
        r = evaluate_version(version, cp_dirs)
        all_results[version] = r
        if r.get("available"):
            n = r["n_entries"]
            mean_ba = r["aggregate"].get("mean_ba") or 0
            print(f"OK ({n} entries, mean BA={mean_ba:.3f})")
        else:
            print("not available")

    print_summary_table(all_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
