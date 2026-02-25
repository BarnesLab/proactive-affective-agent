"""Text-based baseline pipeline: TF-IDF and BoW on diary text (emotion_driver).

Evaluates text-only features for predicting affective states.
Only uses rows where the diary text (emotion_driver) is present.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _coerce_binary_standalone(arr) -> np.ndarray:
    """Convert raw target column to float array with NaN for missing (module-level)."""
    out = np.full(len(arr), np.nan, dtype=float)
    for i, v in enumerate(arr):
        if pd.isna(v):
            continue
        if isinstance(v, bool):
            out[i] = float(v)
        elif isinstance(v, str):
            low = v.lower().strip()
            if low in ("true", "1", "yes"):
                out[i] = 1.0
            elif low in ("false", "0", "no"):
                out[i] = 0.0
        else:
            try:
                out[i] = float(v)
            except (ValueError, TypeError):
                pass
    return out


def _fit_evaluate_one_text(
    model_name: str,
    target: str,
    task_type: str,
    X_train,
    X_test,
    y_tr_raw: np.ndarray,
    y_te_raw: np.ndarray,
    fold: int,
) -> dict[str, Any] | None:
    """Fit one (model_name, target) combination and return metrics.

    Module-level function for joblib pickling. Accepts sparse or dense X.
    Returns None on skip/error.
    """
    try:
        if task_type == "regression":
            y_tr = pd.to_numeric(pd.Series(y_tr_raw), errors="coerce").values
            y_te = pd.to_numeric(pd.Series(y_te_raw), errors="coerce").values
            mask_tr = ~np.isnan(y_tr)
            mask_te = ~np.isnan(y_te)
            if mask_tr.sum() < 10 or mask_te.sum() < 5:
                return None

            reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            reg.fit(X_train[mask_tr], y_tr[mask_tr])
            preds = reg.predict(X_test[mask_te])
            mae = float(mean_absolute_error(y_te[mask_te], preds))

            return {
                "model_name": model_name,
                "target": target,
                "fold": fold,
                "mae": mae,
                "n_train": int(mask_tr.sum()),
                "n_test": int(mask_te.sum()),
            }
        else:  # classification
            y_tr = _coerce_binary_standalone(y_tr_raw)
            y_te = _coerce_binary_standalone(y_te_raw)
            mask_tr = ~np.isnan(y_tr)
            mask_te = ~np.isnan(y_te)
            if mask_tr.sum() < 10 or mask_te.sum() < 5:
                return None

            y_tr_int = y_tr[mask_tr].astype(int)
            y_te_int = y_te[mask_te].astype(int)
            if len(set(y_tr_int)) < 2:
                return None

            clf = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],
                cv=3,
                class_weight="balanced",
                max_iter=1000,
                scoring="balanced_accuracy",
                random_state=42,
            )
            clf.fit(X_train[mask_tr], y_tr_int)
            preds = clf.predict(X_test[mask_te])

            ba = float(balanced_accuracy_score(y_te_int, preds))
            f1 = float(f1_score(y_te_int, preds, average='binary', zero_division=0))

            return {
                "model_name": model_name,
                "target": target,
                "fold": fold,
                "balanced_accuracy": ba,
                "f1": f1,
                "n_train": int(mask_tr.sum()),
                "n_test": int(mask_te.sum()),
            }
    except Exception as e:
        logger.warning(f"  {model_name}/{target} fold {fold}: {e}")
        return None


class TextBaselinePipeline:
    """Run TF-IDF and BoW text baselines on diary entries (emotion_driver).

    Only evaluates on rows where emotion_driver is non-empty.
    Supports the same 5-fold CV structure as MLBaselinePipeline.
    """

    # Text representations
    TEXT_MODELS = ["tfidf", "bow"]

    def __init__(
        self,
        splits_dir: Path,
        output_dir: Path,
        processed_hourly_dir: Path | None = None,
        model_names: list[str] | None = None,
    ) -> None:
        """
        Args:
            splits_dir: Directory with group_{1-5}_{train,test}.csv files.
            output_dir: Where to save results.
            processed_hourly_dir: Unused for text baselines (kept for interface parity).
            model_names: Which text models to run (default: all).
        """
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_hourly_dir = processed_hourly_dir
        self.model_names = model_names if model_names is not None else self.TEXT_MODELS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_folds(self, folds: list | None = None) -> dict[str, Any]:
        """Run 5-fold CV for all text models and targets.

        After vectorizing text for each model, all target fit+evaluate jobs
        are parallelized via joblib within each (fold, model_name) pair.

        Args:
            folds: List of fold indices to run (e.g. [1]). If None, runs all 5 folds.

        Returns:
            Nested dict: {model_name: {target: {metric: value, ...}, "_aggregate": {...}}}
        """
        from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

        all_results: dict[str, Any] = {}

        for fold in (folds if folds is not None else range(1, 6)):
            logger.info(f"=== Text Baseline Fold {fold}/5 ===")

            train_df = pd.read_csv(self.splits_dir / f"group_{fold}_train.csv")
            test_df = pd.read_csv(self.splits_dir / f"group_{fold}_test.csv")

            # Filter to diary-present rows only
            train_df = train_df[
                train_df["emotion_driver"].notna() & (train_df["emotion_driver"].str.strip() != "")
            ].copy()
            test_df = test_df[
                test_df["emotion_driver"].notna() & (test_df["emotion_driver"].str.strip() != "")
            ].copy()

            if len(train_df) < 10 or len(test_df) < 5:
                logger.warning(f"  Fold {fold}: not enough diary entries — skipping")
                continue

            train_texts = train_df["emotion_driver"].fillna("").tolist()
            test_texts = test_df["emotion_driver"].fillna("").tolist()

            binary_targets = list(BINARY_STATE_TARGETS) + ["INT_availability"]

            # Build vectorizer for each text model, then parallelize targets
            for model_name in self.model_names:
                vectorizer = self._make_vectorizer(model_name)
                try:
                    X_train = vectorizer.fit_transform(train_texts)
                    X_test = vectorizer.transform(test_texts)
                except Exception as e:
                    logger.warning(f"  {model_name} vectorizer fold {fold}: {e}")
                    continue

                # Build all target tasks
                tasks = []
                for target in CONTINUOUS_TARGETS:
                    y_tr_raw = pd.to_numeric(train_df[target], errors="coerce").values
                    y_te_raw = pd.to_numeric(test_df[target], errors="coerce").values
                    tasks.append((model_name, target, "regression", y_tr_raw, y_te_raw))

                for target in binary_targets:
                    y_tr_raw = train_df[target].values
                    y_te_raw = test_df[target].values
                    tasks.append((model_name, target, "classification", y_tr_raw, y_te_raw))

                logger.info(f"  Fold {fold}/{model_name}: dispatching {len(tasks)} parallel target jobs")

                results = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(_fit_evaluate_one_text)(
                        mn, target, task_type, X_train, X_test,
                        y_tr_raw, y_te_raw, fold,
                    )
                    for mn, target, task_type, y_tr_raw, y_te_raw in tasks
                )

                # Collect results into all_results dict
                for res in results:
                    if res is None:
                        continue
                    mn = res.pop("model_name")
                    target = res.pop("target")
                    all_results.setdefault(mn, {}).setdefault(target, []).append(res)

        aggregated = self._aggregate_folds(all_results)
        self._save_results(all_results, aggregated)
        return aggregated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_vectorizer(self, model_name: str):
        if model_name == "tfidf":
            return TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )
        elif model_name == "bow":
            return CountVectorizer(
                max_features=300,
                ngram_range=(1, 1),
                min_df=2,
                binary=False,
            )
        else:
            raise ValueError(f"Unknown text model: {model_name}")

    @staticmethod
    def _coerce_binary(arr) -> np.ndarray:
        """Convert raw target column to float array with NaN for missing."""
        return _coerce_binary_standalone(arr)

    def _aggregate_folds(self, all_results: dict) -> dict[str, Any]:
        """Average metrics across folds for each model x target."""
        aggregated: dict[str, Any] = {}

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
                        "mae_std": float(np.std(maes, ddof=1)),
                        "n_folds": len(fold_results),
                    }
                    all_mae.append(float(np.mean(maes)))
                else:
                    bas = [r["balanced_accuracy"] for r in fold_results]
                    f1s = [r["f1"] for r in fold_results]
                    aggregated[model_name][target] = {
                        "ba_mean": float(np.mean(bas)),
                        "ba_std": float(np.std(bas, ddof=1)),
                        "f1_mean": float(np.mean(f1s)),
                        "f1_std": float(np.std(f1s, ddof=1)),
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

    def _save_results(self, raw: dict, aggregated: dict) -> None:
        """Persist raw fold results and aggregated metrics to disk."""
        with open(self.output_dir / "text_baseline_folds.json", "w") as f:
            json.dump(raw, f, indent=2, default=str)

        with open(self.output_dir / "text_baseline_metrics.json", "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        lines = ["# Text Baseline Results (5-fold CV, diary-present rows only)\n"]
        for model_name, targets in aggregated.items():
            agg = targets.get("_aggregate", {})
            lines.append(f"## {model_name.upper()}")
            if agg.get("mean_mae") is not None:
                lines.append(f"  Mean MAE (regression): {agg['mean_mae']:.3f}")
            if agg.get("mean_ba") is not None:
                lines.append(f"  Mean Balanced Accuracy: {agg['mean_ba']:.3f}")
            if agg.get("mean_f1") is not None:
                lines.append(f"  Mean F1: {agg['mean_f1']:.3f}")
            lines.append("")

        (self.output_dir / "text_baseline_summary.md").write_text("\n".join(lines))
        logger.info(f"Text baseline results saved to {self.output_dir}")
