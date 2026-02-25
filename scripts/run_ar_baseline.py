#!/usr/bin/env python3
"""Affect Autocorrelation (AR) baseline for BUCS pilot study.

This script computes the trivial upper bound based purely on temporal
autocorrelation of self-reported affect — no sensing data required.

Two variants:
  - last_value:  predict current value = most recent prior observation (AR(1) naive)
  - rolling_mean: predict current value = mean of last N prior observations (AR(N))

Key framing: if our sensing-based agent (V4) approaches or exceeds the AR
baseline, it demonstrates that passive behavioral signals carry predictive
information *beyond* affect autocorrelation. If V4 falls short, it quantifies
the gap between behavioral sensing and direct affect measurement.

ER_desire binary: >= 5 (scale midpoint, consistent with run_agentic_pilot.py).

Evaluation methodology:
  - Per-fold evaluation (5-fold CV, same splits as all other baselines)
  - For each fold, AR predictions are computed per-user within that fold's
    test data using shift(1) (last_value) or rolling mean (rolling_mean)
  - The first EMA entry per user per fold has no prior history and is skipped
    (NaN from shift(1)). This means the AR baseline evaluates on slightly
    fewer samples per fold than baselines that predict every test row.
  - Metrics are computed per-fold, then aggregated as mean +/- std (ddof=1)

Usage:
    python scripts/run_ar_baseline.py
    python scripts/run_ar_baseline.py --window 3 --output-dir outputs/ar_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ar_baseline"

CONTINUOUS_TARGETS = ["PANAS_Pos", "PANAS_Neg", "ER_desire"]

BINARY_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_happy_State",
    "Individual_level_sad_State",
    "Individual_level_afraid_State",
    "Individual_level_miserable_State",
    "Individual_level_worried_State",
    "Individual_level_cheerful_State",
    "Individual_level_pleased_State",
    "Individual_level_grateful_State",
    "Individual_level_lonely_State",
    "Individual_level_interactions_quality_State",
    "Individual_level_pain_State",
    "Individual_level_forecasting_State",
    "Individual_level_ER_desire_State",
]


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve data directory
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    splits_dir = data_dir / "processed" / "splits"

    # Verify splits exist
    missing = [g for g in range(1, 6) if not (splits_dir / f"group_{g}_test.csv").exists()]
    if missing:
        logger.error(f"Missing test split CSVs for groups {missing} in {splits_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Define variants
    # ------------------------------------------------------------------
    variants = {
        "last_value": _predict_last_value,
        f"rolling_mean_w{args.window}": lambda g: _predict_rolling_mean(g, args.window),
    }

    # ------------------------------------------------------------------
    # Per-fold evaluation for each variant
    # ------------------------------------------------------------------
    # raw_results: {variant: {target: [fold_dicts]}}
    raw_results: dict[str, dict[str, list[dict]]] = {}
    aggregated_results: dict[str, Any] = {}

    for variant_name, pred_fn in variants.items():
        logger.info(f"Running variant: {variant_name}")
        raw_results[variant_name] = {}

        for fold in range(1, 6):
            logger.info(f"  Fold {fold}/5")

            # Load only this fold's test data
            fold_df = pd.read_csv(splits_dir / f"group_{fold}_test.csv")
            fold_df["timestamp_local"] = pd.to_datetime(fold_df["timestamp_local"])
            fold_df = fold_df.sort_values(["Study_ID", "timestamp_local"]).reset_index(drop=True)

            # Apply midpoint threshold for ER_desire binary
            er = pd.to_numeric(fold_df["ER_desire"], errors="coerce")
            fold_df["Individual_level_ER_desire_State"] = (er >= 5).where(er.notna(), other=None)

            # Compute AR predictions and evaluate within this fold
            all_preds, all_gts = _run_variant(fold_df, pred_fn)
            fold_metrics = _compute_fold_metrics(all_preds, all_gts)

            # Store per-fold results
            for col, metrics in fold_metrics["continuous"].items():
                raw_results[variant_name].setdefault(col, []).append({
                    "fold": fold,
                    "mae": metrics["mae"],
                    "n_test": metrics["n"],
                    "mean_true": metrics["mean_true"],
                    "mean_pred": metrics["mean_pred"],
                })

            for col, metrics in fold_metrics["binary"].items():
                raw_results[variant_name].setdefault(col, []).append({
                    "fold": fold,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "n_test": metrics["n"],
                    "prevalence": metrics["prevalence"],
                })

        # Aggregate across folds for this variant
        aggregated_results[variant_name] = _aggregate_folds(raw_results[variant_name])
        _print_variant_summary(variant_name, aggregated_results[variant_name])

    # ------------------------------------------------------------------
    # Save results (two files, matching other baselines' conventions)
    # ------------------------------------------------------------------
    # 1. Raw per-fold results
    raw_path = output_dir / "ar_baseline_folds.json"
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=2, default=str)
    logger.info(f"Per-fold results saved to: {raw_path}")

    # 2. Aggregated metrics (mean +/- std across folds)
    agg_path = output_dir / "ar_baseline_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated_results, f, indent=2, default=str)
    logger.info(f"Aggregated metrics saved to: {agg_path}")

    # 3. Also save the legacy ar_results.json for backward compatibility,
    #    now containing the corrected per-fold aggregated numbers
    legacy_path = output_dir / "ar_results.json"
    with open(legacy_path, "w") as f:
        json.dump(aggregated_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Prediction variants
# ---------------------------------------------------------------------------

def _predict_last_value(group: pd.DataFrame) -> pd.DataFrame:
    """AR(1) naive: predict using the immediately prior observation."""
    result = group.copy()
    for col in CONTINUOUS_TARGETS + BINARY_TARGETS:
        if col in result.columns:
            result[f"pred_{col}"] = result[col].shift(1)
    return result


def _predict_rolling_mean(group: pd.DataFrame, window: int) -> pd.DataFrame:
    """Predict using rolling mean of prior N observations (continuous targets only).
    Binary targets fall back to last_value (majority of recent window).
    """
    result = group.copy()
    for col in CONTINUOUS_TARGETS:
        if col in result.columns:
            numeric = pd.to_numeric(result[col], errors="coerce")
            result[f"pred_{col}"] = (
                numeric.shift(1).rolling(window=window, min_periods=1).mean()
            )
    for col in BINARY_TARGETS:
        if col in result.columns:
            result[f"pred_{col}"] = result[col].shift(1)
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_variant(
    df: pd.DataFrame,
    pred_fn,
) -> tuple[list[dict], list[dict]]:
    """Apply prediction function per user and collect (pred, gt) pairs.

    The first EMA entry per user has no prior history — it's excluded
    from evaluation (pred = NaN from shift(1)).
    """
    all_preds: list[dict] = []
    all_gts: list[dict] = []

    for _, group in df.groupby("Study_ID"):
        group = group.sort_values("timestamp_local").reset_index(drop=True)
        group = pred_fn(group)

        # Skip first row per user (no prior history)
        for _, row in group.iloc[1:].iterrows():
            pred: dict[str, Any] = {}
            gt: dict[str, Any] = {}

            for col in CONTINUOUS_TARGETS:
                pred_col = f"pred_{col}"
                pv = row.get(pred_col)
                tv = row.get(col)
                try:
                    pred[col] = float(pv) if pv is not None and not _is_nan(pv) else None
                    gt[col] = float(tv) if tv is not None and not _is_nan(tv) else None
                except (ValueError, TypeError):
                    pred[col] = None
                    gt[col] = None

            for col in BINARY_TARGETS:
                pred_col = f"pred_{col}"
                pv = row.get(pred_col)
                tv = row.get(col)
                pred[col] = _to_bool(pv)
                gt[col] = _to_bool(tv)

            all_preds.append(pred)
            all_gts.append(gt)

    return all_preds, all_gts


def _compute_fold_metrics(
    preds: list[dict],
    gts: list[dict],
) -> dict[str, Any]:
    """Compute metrics for a single fold's predictions."""
    results: dict[str, Any] = {"continuous": {}, "binary": {}}

    for col in CONTINUOUS_TARGETS:
        y_true = [g[col] for g, p in zip(gts, preds) if g[col] is not None and p[col] is not None]
        y_pred = [p[col] for g, p in zip(gts, preds) if g[col] is not None and p[col] is not None]
        if len(y_true) >= 2:
            results["continuous"][col] = {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "n": len(y_true),
                "mean_true": float(np.mean(y_true)),
                "mean_pred": float(np.mean(y_pred)),
            }

    for col in BINARY_TARGETS:
        y_true = [g[col] for g, p in zip(gts, preds) if g[col] is not None and p[col] is not None]
        y_pred = [p[col] for g, p in zip(gts, preds) if g[col] is not None and p[col] is not None]
        if len(y_true) >= 2 and len(set(y_true)) > 1:
            results["binary"][col] = {
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                # binary F1: measures detection of the positive (elevated) state
                "f1": float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
                "n": len(y_true),
                "prevalence": float(np.mean(y_true)),
            }

    return results


def _aggregate_folds(fold_results: dict[str, list[dict]]) -> dict[str, Any]:
    """Average metrics across 5 folds for each target, matching ML/DL baseline format.

    Output structure matches other baselines:
        {target: {metric_mean, metric_std, n_folds}, ..., _aggregate: {mean_mae, mean_ba, mean_f1}}
    """
    aggregated: dict[str, Any] = {}
    all_mae_means: list[float] = []
    all_ba_means: list[float] = []
    all_f1_means: list[float] = []

    for target, fold_dicts in fold_results.items():
        if not fold_dicts:
            continue

        if "mae" in fold_dicts[0]:
            # Continuous target
            maes = [r["mae"] for r in fold_dicts]
            aggregated[target] = {
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes, ddof=1)),
                "n_folds": len(fold_dicts),
                "per_fold": fold_dicts,
            }
            all_mae_means.append(aggregated[target]["mae_mean"])
        else:
            # Binary target
            bas = [r["balanced_accuracy"] for r in fold_dicts]
            f1s = [r["f1"] for r in fold_dicts]
            aggregated[target] = {
                "ba_mean": float(np.mean(bas)),
                "ba_std": float(np.std(bas, ddof=1)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=1)),
                "n_folds": len(fold_dicts),
                "per_fold": fold_dicts,
            }
            all_ba_means.append(aggregated[target]["ba_mean"])
            all_f1_means.append(aggregated[target]["f1_mean"])

    aggregated["_aggregate"] = {
        "mean_mae": float(np.mean(all_mae_means)) if all_mae_means else None,
        "mean_ba": float(np.mean(all_ba_means)) if all_ba_means else None,
        "mean_f1": float(np.mean(all_f1_means)) if all_f1_means else None,
    }

    return aggregated


def _print_variant_summary(name: str, aggregated: dict) -> None:
    """Print a human-readable summary of aggregated results for one variant."""
    agg = aggregated.get("_aggregate", {})
    print(f"\n{'='*60}")
    print(f"AR Baseline: {name}")
    print(f"{'='*60}")
    if agg.get("mean_mae") is not None:
        print(f"  Mean MAE (continuous):           {agg['mean_mae']:.4f}")
    else:
        print("  Mean MAE: N/A")
    if agg.get("mean_ba") is not None:
        print(f"  Mean Balanced Accuracy (binary): {agg['mean_ba']:.4f}")
    else:
        print("  Mean BA: N/A")
    if agg.get("mean_f1") is not None:
        print(f"  Mean F1 (binary):                {agg['mean_f1']:.4f}")
    else:
        print("  Mean F1: N/A")

    # Continuous targets detail
    print("\nContinuous targets:")
    for col in CONTINUOUS_TARGETS:
        if col in aggregated and "mae_mean" in aggregated[col]:
            v = aggregated[col]
            print(f"  {col}: MAE={v['mae_mean']:.3f} +/- {v['mae_std']:.3f} ({v['n_folds']} folds)")

    # Binary targets detail (top 5 by BA)
    binary_items = [
        (col, aggregated[col])
        for col in BINARY_TARGETS
        if col in aggregated and "ba_mean" in aggregated[col]
    ]
    binary_sorted = sorted(binary_items, key=lambda x: x[1]["ba_mean"], reverse=True)
    print(f"\nBinary targets (top 5 by BA, {len(binary_sorted)} total):")
    for col, v in binary_sorted[:5]:
        print(f"  {col}: BA={v['ba_mean']:.3f}+/-{v['ba_std']:.3f}  F1={v['f1_mean']:.3f}+/-{v['f1_std']:.3f}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_nan(v) -> bool:
    try:
        return bool(np.isnan(float(v)))
    except (ValueError, TypeError):
        return False


def _to_bool(val) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).lower().strip()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Affect autocorrelation (AR) baseline for BUCS pilot study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/run_ar_baseline.py
  python scripts/run_ar_baseline.py --window 3
""",
    )
    parser.add_argument("--window", type=int, default=3,
                        help="Rolling window size for rolling_mean variant (default: 3)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: data/ relative to project root)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
