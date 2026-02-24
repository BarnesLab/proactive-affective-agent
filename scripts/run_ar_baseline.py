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
    # Load all EMA data
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    splits_dir = data_dir / "processed" / "splits"

    dfs = []
    for group in range(1, 6):
        p = splits_dir / f"group_{group}_test.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        logger.error(f"No split CSVs found in {splits_dir}")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    df = df.sort_values(["Study_ID", "timestamp_local"]).reset_index(drop=True)

    # Apply midpoint threshold for ER_desire binary (consistent with V4 pipeline)
    er = pd.to_numeric(df["ER_desire"], errors="coerce")
    df["Individual_level_ER_desire_State"] = (er >= 5).where(er.notna(), other=None)

    logger.info(f"Loaded {len(df)} EMA entries for {df['Study_ID'].nunique()} users")

    # ------------------------------------------------------------------
    # Compute AR predictions (participant-level, chronological)
    # ------------------------------------------------------------------
    variants = {
        "last_value": _predict_last_value,
        f"rolling_mean_w{args.window}": lambda g: _predict_rolling_mean(g, args.window),
    }

    results = {}
    for variant_name, pred_fn in variants.items():
        logger.info(f"Running variant: {variant_name}")
        all_preds, all_gts = _run_variant(df, pred_fn)
        metrics = _compute_metrics(all_preds, all_gts)
        results[variant_name] = {
            "n_predictions": len(all_preds),
            "metrics": metrics,
        }
        _print_variant_summary(variant_name, metrics, len(all_preds))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_path = output_dir / "ar_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_path}")
    print(f"\nResults saved to: {output_path}")


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

        # Skip first row (no prior history)
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


def _compute_metrics(
    preds: list[dict],
    gts: list[dict],
) -> dict[str, Any]:
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
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "n": len(y_true),
                "prevalence": float(np.mean(y_true)),
            }

    # Aggregate
    all_mae = [v["mae"] for v in results["continuous"].values()]
    all_ba = [v["balanced_accuracy"] for v in results["binary"].values()]
    all_f1 = [v["f1"] for v in results["binary"].values()]
    results["aggregate"] = {
        "mean_mae": float(np.mean(all_mae)) if all_mae else None,
        "mean_balanced_accuracy": float(np.mean(all_ba)) if all_ba else None,
        "mean_f1": float(np.mean(all_f1)) if all_f1 else None,
    }

    return results


def _print_variant_summary(name: str, metrics: dict, n: int) -> None:
    agg = metrics.get("aggregate", {})
    print(f"\n{'='*60}")
    print(f"AR Baseline: {name}  (n={n} predictions)")
    print(f"{'='*60}")
    print(f"  Mean MAE (continuous):           {agg.get('mean_mae', 'N/A'):.4f}" if agg.get('mean_mae') else "  Mean MAE: N/A")
    print(f"  Mean Balanced Accuracy (binary): {agg.get('mean_balanced_accuracy', 'N/A'):.4f}" if agg.get('mean_balanced_accuracy') else "  Mean BA: N/A")
    print(f"  Mean F1 (binary):                {agg.get('mean_f1', 'N/A'):.4f}" if agg.get('mean_f1') else "  Mean F1: N/A")
    print("\nContinuous:")
    for col, v in metrics.get("continuous", {}).items():
        print(f"  {col}: MAE={v['mae']:.3f} (n={v['n']})")
    print("\nBinary (top 5 by BA):")
    binary_sorted = sorted(
        metrics.get("binary", {}).items(),
        key=lambda x: x[1].get("balanced_accuracy", 0),
        reverse=True,
    )
    for col, v in binary_sorted[:5]:
        print(f"  {col}: BA={v['balanced_accuracy']:.3f} F1={v['f1']:.3f} (n={v['n']})")


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
