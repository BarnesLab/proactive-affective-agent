#!/usr/bin/env python3
"""Merge per-fold baseline results into final aggregated output.

After running 5 parallel fold processes, use this to combine results.

Usage:
    # Merge ML results from outputs/ml_baselines_v2/fold_{1-5}/
    python scripts/merge_baseline_results.py --output outputs/ml_baselines_v2 --type ml

    # Merge DL/MLP results from outputs/advanced_baselines/dl/fold_{1-5}/
    python scripts/merge_baseline_results.py --output outputs/advanced_baselines --type dl --pipeline dl

    # Merge transformer results
    python scripts/merge_baseline_results.py --output outputs/advanced_baselines --type dl --pipeline transformer
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_fold_dicts(fold_dicts: list[dict]) -> dict:
    """Merge list of per-fold raw result dicts into one combined dict."""
    merged: dict = {}
    for fold_data in fold_dicts:
        for model_name, targets in fold_data.items():
            merged.setdefault(model_name, {})
            for target, results in targets.items():
                merged[model_name].setdefault(target, [])
                if isinstance(results, list):
                    merged[model_name][target].extend(results)
                else:
                    merged[model_name][target].append(results)
    return merged


def aggregate_folds(all_results: dict) -> dict:
    """Compute mean/std metrics across folds."""
    aggregated: dict = {}
    for model_name, targets in all_results.items():
        aggregated[model_name] = {}
        all_mae: list[float] = []
        all_ba: list[float] = []
        all_f1: list[float] = []

        for target, fold_results in targets.items():
            if not fold_results:
                continue
            if "mae" in fold_results[0]:
                maes = [r["mae"] for r in fold_results]
                aggregated[model_name][target] = {
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes)),
                    "n_folds": len(fold_results),
                }
                all_mae.append(float(np.mean(maes)))
            else:
                bas = [r["balanced_accuracy"] for r in fold_results]
                f1s = [r["f1"] for r in fold_results]
                aggregated[model_name][target] = {
                    "ba_mean": float(np.mean(bas)),
                    "ba_std": float(np.std(bas)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "n_folds": len(fold_results),
                }
                all_ba.append(float(np.mean(bas)))
                all_f1.append(float(np.mean(f1s)))

        aggregated[model_name]["_aggregate"] = {
            "mean_mae": float(np.mean(all_mae)) if all_mae else None,
            "mean_ba": float(np.mean(all_ba)) if all_ba else None,
            "mean_f1": float(np.mean(all_f1)) if all_f1 else None,
        }
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-fold baseline results")
    parser.add_argument("--output", required=True, help="Base output directory")
    parser.add_argument(
        "--type", choices=["ml", "dl"], default="ml",
        help="ml = ml_baseline_folds.json; dl = dl_baseline_folds.json"
    )
    parser.add_argument(
        "--pipeline", default=None,
        help="For DL: pipeline subdirectory name (dl, text, transformer, combined). "
             "If not set, merges directly from output/fold_N/"
    )
    args = parser.parse_args()

    base_dir = Path(args.output)
    filename = "ml_baseline_folds.json" if args.type == "ml" else "dl_baseline_folds.json"

    # Determine fold directories
    if args.pipeline:
        fold_dirs = [base_dir / args.pipeline / f"fold_{i}" for i in range(1, 6)]
        out_dir = base_dir / args.pipeline
    else:
        fold_dirs = [base_dir / f"fold_{i}" for i in range(1, 6)]
        out_dir = base_dir

    fold_dicts = []
    for fold_dir in fold_dirs:
        raw_file = fold_dir / filename
        if not raw_file.exists():
            logger.warning(f"Missing: {raw_file} — skipping fold")
            continue
        with open(raw_file) as f:
            fold_dicts.append(json.load(f))
        logger.info(f"Loaded {raw_file}")

    if not fold_dicts:
        logger.error("No fold results found! Check --output and --pipeline paths.")
        sys.exit(1)

    logger.info(f"Merging {len(fold_dicts)} folds...")
    merged = merge_fold_dicts(fold_dicts)
    aggregated = aggregate_folds(merged)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "ml_baseline" if args.type == "ml" else "dl_baseline"

    with open(out_dir / f"{prefix}_folds.json", "w") as f:
        json.dump(merged, f, indent=2, default=str)
    with open(out_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    logger.info(f"Saved merged results to {out_dir}")

    print(f"\n{'='*60}")
    print(f"MERGED RESULTS ({len(fold_dicts)}-fold CV)")
    print(f"{'='*60}")
    for model_name, targets in aggregated.items():
        agg = targets.get("_aggregate", {})
        ba = agg.get("mean_ba")
        f1 = agg.get("mean_f1")
        mae = agg.get("mean_mae")
        ba_s = f"{ba:.3f}" if isinstance(ba, float) else "  —  "
        f1_s = f"{f1:.3f}" if isinstance(f1, float) else "  —  "
        mae_s = f"{mae:.3f}" if isinstance(mae, float) else "  —  "
        print(f"  {model_name:<20} BA={ba_s}  F1={f1_s}  MAE={mae_s}")


if __name__ == "__main__":
    main()
