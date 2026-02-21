"""Evaluation metrics: balanced accuracy, MAE, F1 for the pilot study.

Computes metrics per-target and aggregated, comparing predictions to ground truth.
Includes CHI paper personal threshold evaluation (per-user mean ± SD binary classification).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error

from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

# Maps continuous prediction fields to the raw EMA items used for personal threshold.
# CHI paper: Individual_level_X_State is True when X > user_mean + user_sd (high)
# or X < user_mean - user_sd (low, for negative items).
# For aggregate PANAS scores, we threshold the predicted PANAS value.
PERSONAL_THRESHOLD_MAP = {
    "PANAS_Pos": {
        "state_col": "Individual_level_PA_State",
        "direction": "high",  # above threshold = True
    },
    "PANAS_Neg": {
        "state_col": "Individual_level_NA_State",
        "direction": "high",
    },
    "ER_desire": {
        "state_col": "Individual_level_ER_desire_State",
        "direction": "high",
    },
}


def compute_all(
    predictions: list[dict],
    ground_truths: list[dict],
    metadata: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute all metrics comparing predictions to ground truths.

    Args:
        predictions: List of prediction dicts (from LLM output).
        ground_truths: List of ground truth dicts (from EMA data).
        metadata: Optional list of metadata dicts with study_id for per-user grouping.

    Returns:
        Nested dict with per-target and aggregate metrics.
    """
    results = {
        "continuous": {},
        "binary": {},
        "availability": {},
        "personal_threshold": {},
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

    # CHI paper personal threshold evaluation
    if metadata:
        results["personal_threshold"] = compute_personal_threshold(
            predictions, ground_truths, metadata
        )

    # Aggregate metrics
    all_mae = [v["mae"] for v in results["continuous"].values() if "mae" in v]
    all_ba = [v["balanced_accuracy"] for v in results["binary"].values() if "balanced_accuracy" in v]
    all_f1 = [v["f1"] for v in results["binary"].values() if "f1" in v]

    pt = results.get("personal_threshold", {})
    pt_ba_vals = [v["balanced_accuracy"] for v in pt.values() if isinstance(v, dict) and "balanced_accuracy" in v]
    pt_f1_vals = [v["f1"] for v in pt.values() if isinstance(v, dict) and "f1" in v]

    results["aggregate"] = {
        "mean_mae": float(np.mean(all_mae)) if all_mae else None,
        "mean_balanced_accuracy": float(np.mean(all_ba)) if all_ba else None,
        "mean_f1": float(np.mean(all_f1)) if all_f1 else None,
        "n_continuous_evaluated": len(all_mae),
        "n_binary_evaluated": len(all_ba),
        "personal_threshold_mean_ba": float(np.mean(pt_ba_vals)) if pt_ba_vals else None,
        "personal_threshold_mean_f1": float(np.mean(pt_f1_vals)) if pt_f1_vals else None,
    }

    return results


def compute_personal_threshold(
    predictions: list[dict],
    ground_truths: list[dict],
    metadata: list[dict],
) -> dict[str, Any]:
    """CHI paper personal threshold evaluation.

    For each continuous target (PANAS_Pos, PANAS_Neg, ER_desire):
    1. Group predictions by user (study_id from metadata)
    2. For each user, compute mean and SD of their PREDICTED values
    3. Classify: predicted > mean + SD → True (high state)
    4. Compare against ground truth binary state column

    Returns per-target results with balanced accuracy and F1.
    """
    results = {}

    for cont_target, mapping in PERSONAL_THRESHOLD_MAP.items():
        state_col = mapping["state_col"]
        direction = mapping["direction"]

        # Group by user
        user_preds: dict[int, list] = {}
        user_gts: dict[int, list] = {}
        user_gt_states: dict[int, list] = {}

        for pred, gt, meta in zip(predictions, ground_truths, metadata):
            sid = meta.get("study_id")
            pred_val = pred.get(cont_target)
            gt_state = gt.get(state_col)

            if sid is None or pred_val is None or gt_state is None:
                continue

            try:
                pv = float(pred_val)
            except (ValueError, TypeError):
                continue

            gt_bool = _to_bool(gt_state)
            if gt_bool is None:
                continue

            user_preds.setdefault(sid, []).append(pv)
            user_gt_states.setdefault(sid, []).append(gt_bool)

        # Now compute per-user thresholds and aggregate
        all_pt_true = []
        all_pt_pred = []

        for sid in user_preds:
            pvals = np.array(user_preds[sid])
            gt_states = user_gt_states[sid]

            if len(pvals) < 3:  # need enough data for meaningful threshold
                continue

            user_mean = np.mean(pvals)
            user_sd = np.std(pvals)

            if user_sd < 1e-6:  # no variance
                continue

            # Apply threshold: above mean+SD → True
            if direction == "high":
                pt_pred = [bool(v > user_mean + user_sd) for v in pvals]
            else:
                pt_pred = [bool(v < user_mean - user_sd) for v in pvals]

            all_pt_true.extend(gt_states)
            all_pt_pred.extend(pt_pred)

        if len(all_pt_true) >= 2 and len(set(all_pt_true)) > 1:
            results[cont_target] = {
                "balanced_accuracy": float(balanced_accuracy_score(all_pt_true, all_pt_pred)),
                "f1": float(f1_score(all_pt_true, all_pt_pred, zero_division=0)),
                "n": len(all_pt_true),
                "n_users": len(user_preds),
                "state_col": state_col,
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
