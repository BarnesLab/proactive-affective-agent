"""Evaluation metrics: balanced accuracy, MAE, F1 for the pilot study.

Computes metrics per-target and aggregated, comparing predictions to ground truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error

from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS


def compute_all(
    predictions: list[dict],
    ground_truths: list[dict],
) -> dict[str, Any]:
    """Compute all metrics comparing predictions to ground truths.

    Args:
        predictions: List of prediction dicts (from LLM output).
        ground_truths: List of ground truth dicts (from EMA data).

    Returns:
        Nested dict with per-target and aggregate metrics.
    """
    results = {
        "continuous": {},
        "binary": {},
        "availability": {},
        "aggregate": {},
    }

    # Continuous targets: MAE
    for target in CONTINUOUS_TARGETS:
        y_true, y_pred = _extract_pairs(ground_truths, predictions, target, is_numeric=True)
        if len(y_true) >= 2:
            results["continuous"][target] = {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "n": len(y_true),
                "mean_true": float(np.mean(y_true)),
                "mean_pred": float(np.mean(y_pred)),
            }

    # Binary state targets: balanced accuracy + F1
    for target in BINARY_STATE_TARGETS:
        y_true, y_pred = _extract_pairs(ground_truths, predictions, target, is_bool=True)
        if len(y_true) >= 2 and len(set(y_true)) > 1:
            results["binary"][target] = {
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "n": len(y_true),
                "prevalence": float(np.mean(y_true)),
            }

    # Availability: treated as binary (yes=True, no=False)
    avail_true, avail_pred = _extract_availability_pairs(ground_truths, predictions)
    if len(avail_true) >= 2 and len(set(avail_true)) > 1:
        results["availability"] = {
            "balanced_accuracy": float(balanced_accuracy_score(avail_true, avail_pred)),
            "f1": float(f1_score(avail_true, avail_pred, zero_division=0)),
            "n": len(avail_true),
        }

    # Aggregate metrics
    all_mae = [v["mae"] for v in results["continuous"].values() if "mae" in v]
    all_ba = [v["balanced_accuracy"] for v in results["binary"].values() if "balanced_accuracy" in v]
    all_f1 = [v["f1"] for v in results["binary"].values() if "f1" in v]

    results["aggregate"] = {
        "mean_mae": float(np.mean(all_mae)) if all_mae else None,
        "mean_balanced_accuracy": float(np.mean(all_ba)) if all_ba else None,
        "mean_f1": float(np.mean(all_f1)) if all_f1 else None,
        "n_continuous_evaluated": len(all_mae),
        "n_binary_evaluated": len(all_ba),
    }

    return results


def _extract_pairs(
    truths: list[dict],
    preds: list[dict],
    key: str,
    is_numeric: bool = False,
    is_bool: bool = False,
) -> tuple[list, list]:
    """Extract matched (truth, pred) pairs for a given key, skipping None values."""
    y_true, y_pred = [], []
    for t, p in zip(truths, preds):
        tv = t.get(key)
        pv = p.get(key)
        if tv is None or pv is None:
            continue
        if is_numeric:
            try:
                y_true.append(float(tv))
                y_pred.append(float(pv))
            except (ValueError, TypeError):
                continue
        elif is_bool:
            tb = _to_bool(tv)
            pb = _to_bool(pv)
            if tb is not None and pb is not None:
                y_true.append(tb)
                y_pred.append(pb)
    return y_true, y_pred


def _extract_availability_pairs(
    truths: list[dict],
    preds: list[dict],
) -> tuple[list[bool], list[bool]]:
    """Extract matched availability pairs (yes→True, no→False)."""
    y_true, y_pred = [], []
    for t, p in zip(truths, preds):
        tv = t.get("INT_availability")
        pv = p.get("INT_availability")
        if tv is None or pv is None:
            continue
        tb = str(tv).lower().strip() == "yes"
        pb = str(pv).lower().strip() == "yes"
        y_true.append(tb)
        y_pred.append(pb)
    return y_true, y_pred


def _to_bool(val) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        lower = val.lower().strip()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
    return None
