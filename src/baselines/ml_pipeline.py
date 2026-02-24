"""ML baseline pipeline: RF, XGBoost, LogisticRegression on sensor features.

Runs all models × all targets × 5-fold CV using the same splits as LLM versions.
No LLM calls needed — can run anytime without worrying about API limits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("xgboost not installed — XGBoost baselines will be skipped")


class MLBaseline:
    """A single ML model for one target variable."""

    def __init__(self, model_name: str, task: str = "regression") -> None:
        """
        Args:
            model_name: "rf", "xgboost", "logistic", or "ridge".
            task: "regression" or "classification".
        """
        self.model_name = model_name
        self.task = task
        self.model = self._create_model(model_name, task)
        self.scaler = StandardScaler()

    def _create_model(self, name: str, task: str):
        if name == "rf":
            if task == "regression":
                return RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
            else:
                return RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42,
                    n_jobs=-1, class_weight="balanced",
                )
        elif name == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError("xgboost not installed")
            if task == "regression":
                return XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, verbosity=0,
                )
            else:
                return XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, verbosity=0, eval_metric="logloss",
                )
        elif name == "logistic":
            if task != "classification":
                raise ValueError("LogisticRegression only for classification")
            return LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
        elif name == "ridge":
            if task != "regression":
                raise ValueError("Ridge only for regression")
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model: {name}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)


class MLBaselinePipeline:
    """Run all models × all targets × 5-fold CV using existing splits."""

    REGRESSION_MODELS = ["rf", "xgboost", "ridge"]
    CLASSIFICATION_MODELS = ["rf", "xgboost", "logistic"]

    def __init__(
        self,
        splits_dir: Path,
        sensing_dfs: dict[str, pd.DataFrame] | None,
        feature_builder,
        output_dir: Path,
        model_names: list[str] | None = None,
        processed_hourly_dir: Path | None = None,
    ) -> None:
        """
        Args:
            splits_dir: Directory with group_{1-5}_{train,test}.csv files.
            sensing_dfs: Pre-loaded legacy sensing DataFrames (None if using Parquet).
            feature_builder: Module with build_daily_features/build_parquet_features/impute_features.
            output_dir: Where to save results.
            model_names: Which models to run (default: all available).
            processed_hourly_dir: Path to data/processed/hourly/ for Parquet mode.
        """
        self.splits_dir = splits_dir
        self.sensing_dfs = sensing_dfs
        self.fb = feature_builder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_hourly_dir = processed_hourly_dir

        if model_names is None:
            model_names = ["rf", "xgboost", "logistic", "ridge"]
        if not HAS_XGBOOST:
            model_names = [m for m in model_names if m != "xgboost"]
        self.model_names = model_names

    def run_all_folds(self) -> dict[str, Any]:
        """Run 5-fold cross-validation for all models and targets.

        Returns:
            Nested dict: {model_name: {target: {metric: value, ...}}}
        """
        from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

        all_results = {}

        for fold in range(1, 6):
            logger.info(f"=== Fold {fold}/5 ===")

            # Load split data
            train_df = pd.read_csv(self.splits_dir / f"group_{fold}_train.csv")
            test_df = pd.read_csv(self.splits_dir / f"group_{fold}_test.csv")
            train_df["date_local"] = pd.to_datetime(train_df["date_local"]).dt.date
            test_df["date_local"] = pd.to_datetime(test_df["date_local"]).dt.date

            # Build features
            if self.processed_hourly_dir is not None:
                X_train, y_cont_train, y_bin_train = self.fb.build_parquet_features(
                    train_df, self.processed_hourly_dir
                )
                X_test, y_cont_test, y_bin_test = self.fb.build_parquet_features(
                    test_df, self.processed_hourly_dir
                )
            else:
                X_train, y_cont_train, y_bin_train = self.fb.build_daily_features(
                    train_df, self.sensing_dfs
                )
                X_test, y_cont_test, y_bin_test = self.fb.build_daily_features(
                    test_df, self.sensing_dfs
                )

            # Impute missing values
            X_train = self.fb.impute_features(X_train)
            X_test = self.fb.impute_features(X_test)

            X_train_np = X_train.values
            X_test_np = X_test.values

            # Run continuous targets (regression)
            for target in CONTINUOUS_TARGETS:
                y_tr = y_cont_train[target].values
                y_te = y_cont_test[target].values

                # Drop rows with NaN targets
                mask_tr = ~np.isnan(y_tr)
                mask_te = ~np.isnan(y_te)
                if mask_tr.sum() < 10 or mask_te.sum() < 5:
                    continue

                reg_models = [m for m in self.model_names if m in ("rf", "xgboost", "ridge")]
                for model_name in reg_models:
                    try:
                        model = MLBaseline(model_name, task="regression")
                        model.fit(X_train_np[mask_tr], y_tr[mask_tr])
                        preds = model.predict(X_test_np[mask_te])

                        mae = float(mean_absolute_error(y_te[mask_te], preds))
                        key = f"{model_name}"
                        all_results.setdefault(key, {}).setdefault(target, []).append({
                            "fold": fold,
                            "mae": mae,
                            "n_train": int(mask_tr.sum()),
                            "n_test": int(mask_te.sum()),
                        })
                    except Exception as e:
                        logger.warning(f"  {model_name}/{target} fold {fold}: {e}")

            # Run binary targets (classification)
            binary_targets = BINARY_STATE_TARGETS + ["INT_availability"]
            for target in binary_targets:
                y_tr = y_bin_train[target].values
                y_te = y_bin_test[target].values

                mask_tr = ~np.isnan(y_tr)
                mask_te = ~np.isnan(y_te)
                if mask_tr.sum() < 10 or mask_te.sum() < 5:
                    continue

                y_tr_int = y_tr[mask_tr].astype(int)
                y_te_int = y_te[mask_te].astype(int)

                # Need at least 2 classes in train
                if len(set(y_tr_int)) < 2:
                    continue

                clf_models = [m for m in self.model_names if m in ("rf", "xgboost", "logistic")]
                for model_name in clf_models:
                    try:
                        model = MLBaseline(model_name, task="classification")
                        model.fit(X_train_np[mask_tr], y_tr_int)
                        preds = model.predict(X_test_np[mask_te])

                        ba = float(balanced_accuracy_score(y_te_int, preds))
                        f1 = float(f1_score(y_te_int, preds, zero_division=0))

                        key = f"{model_name}"
                        all_results.setdefault(key, {}).setdefault(target, []).append({
                            "fold": fold,
                            "balanced_accuracy": ba,
                            "f1": f1,
                            "n_train": int(mask_tr.sum()),
                            "n_test": int(mask_te.sum()),
                            "prevalence_train": float(y_tr_int.mean()),
                            "prevalence_test": float(y_te_int.mean()),
                        })
                    except Exception as e:
                        logger.warning(f"  {model_name}/{target} fold {fold}: {e}")

        # Aggregate across folds
        aggregated = self._aggregate_folds(all_results)

        # Save
        self._save_results(all_results, aggregated)

        return aggregated

    def _aggregate_folds(self, all_results: dict) -> dict[str, Any]:
        """Average metrics across 5 folds for each model × target."""
        aggregated = {}

        for model_name, targets in all_results.items():
            aggregated[model_name] = {}
            all_mae = []
            all_ba = []
            all_f1 = []

            for target, fold_results in targets.items():
                if "mae" in fold_results[0]:
                    maes = [r["mae"] for r in fold_results]
                    avg = {
                        "mae_mean": float(np.mean(maes)),
                        "mae_std": float(np.std(maes)),
                        "n_folds": len(fold_results),
                    }
                    aggregated[model_name][target] = avg
                    all_mae.append(avg["mae_mean"])
                else:
                    bas = [r["balanced_accuracy"] for r in fold_results]
                    f1s = [r["f1"] for r in fold_results]
                    avg = {
                        "ba_mean": float(np.mean(bas)),
                        "ba_std": float(np.std(bas)),
                        "f1_mean": float(np.mean(f1s)),
                        "f1_std": float(np.std(f1s)),
                        "n_folds": len(fold_results),
                    }
                    aggregated[model_name][target] = avg
                    all_ba.append(avg["ba_mean"])
                    all_f1.append(avg["f1_mean"])

            aggregated[model_name]["_aggregate"] = {
                "mean_mae": float(np.mean(all_mae)) if all_mae else None,
                "mean_ba": float(np.mean(all_ba)) if all_ba else None,
                "mean_f1": float(np.mean(all_f1)) if all_f1 else None,
            }

        return aggregated

    def _save_results(self, raw: dict, aggregated: dict) -> None:
        """Save raw fold results and aggregated metrics."""
        raw_path = self.output_dir / "ml_baseline_folds.json"
        with open(raw_path, "w") as f:
            json.dump(raw, f, indent=2, default=str)

        agg_path = self.output_dir / "ml_baseline_metrics.json"
        with open(agg_path, "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        # Also generate a readable summary
        lines = ["# ML Baseline Results (5-fold CV)\n"]
        for model_name, targets in aggregated.items():
            agg = targets.get("_aggregate", {})
            lines.append(f"## {model_name.upper()}")
            if agg.get("mean_mae") is not None:
                lines.append(f"  Mean MAE: {agg['mean_mae']:.3f}")
            if agg.get("mean_ba") is not None:
                lines.append(f"  Mean BA: {agg['mean_ba']:.3f}")
            if agg.get("mean_f1") is not None:
                lines.append(f"  Mean F1: {agg['mean_f1']:.3f}")
            lines.append("")

        summary_path = self.output_dir / "ml_baseline_summary.md"
        summary_path.write_text("\n".join(lines))

        logger.info(f"ML baseline results saved to {self.output_dir}")
