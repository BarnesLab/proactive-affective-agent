"""Combined (late fusion) baseline pipeline: Parquet hourly features + diary embeddings.

Concatenates:
  - Hourly Parquet feature vector (numeric sensor features)
  - Sentence-transformer diary embedding (384-dim, zeros for diary-absent rows)

Then fits RF + Ridge (regression) or RF + LogisticRegression (binary).
All rows are evaluated (diary-absent rows get zero embeddings).
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
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "sentence-transformers not installed — diary embeddings will be zeros. "
        "Install with: pip install sentence-transformers"
    )

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def _embed_texts(
    texts: list[str | None],
    embedder: Any,
) -> np.ndarray:
    """Embed a list of texts, using zeros for None/empty entries.

    Args:
        texts: List of diary texts (may contain None or "").
        embedder: SentenceTransformer instance.

    Returns:
        Array of shape (n, EMBED_DIM).
    """
    result = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    if embedder is None:
        return result

    # Collect non-empty texts with their original indices
    valid_idx = [i for i, t in enumerate(texts) if t and str(t).strip()]
    if not valid_idx:
        return result

    valid_texts = [str(texts[i]).strip() for i in valid_idx]
    embeddings = embedder.encode(
        valid_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    for out_i, orig_i in enumerate(valid_idx):
        result[orig_i] = embeddings[out_i]

    return result


class CombinedBaselinePipeline:
    """Late fusion: Parquet sensor features + sentence-transformer diary embeddings.

    Diary-absent rows receive zero vectors for the embedding component.
    Supports RF + Ridge for regression, RF + LogisticRegression for binary.
    """

    REGRESSION_MODELS = ["rf", "ridge"]
    CLASSIFICATION_MODELS = ["rf", "logistic"]

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
            processed_hourly_dir: Path to data/processed/hourly/ for Parquet features.
            model_names: Which models to run (default: ["rf", "ridge", "logistic"]).
        """
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_hourly_dir = Path(processed_hourly_dir) if processed_hourly_dir else None
        self.model_names = model_names if model_names is not None else ["rf", "ridge", "logistic"]

        # Load sentence-transformer once (or set to None gracefully)
        if HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"Loading sentence-transformer model: {EMBED_MODEL}")
            self._embedder: Any = SentenceTransformer(EMBED_MODEL)
        else:
            logger.warning("No sentence-transformers — diary component will be zero vectors")
            self._embedder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_folds(self) -> dict[str, Any]:
        """Run 5-fold CV for combined sensor + text features.

        Returns:
            Nested dict: {model_name: {target: {metric: ...}, "_aggregate": {...}}}
        """
        from src.baselines import feature_builder
        from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

        all_results: dict[str, Any] = {}

        for fold in range(1, 6):
            logger.info(f"=== Combined Baseline Fold {fold}/5 ===")

            train_df = pd.read_csv(self.splits_dir / f"group_{fold}_train.csv")
            test_df = pd.read_csv(self.splits_dir / f"group_{fold}_test.csv")
            train_df["date_local"] = pd.to_datetime(train_df["date_local"]).dt.date
            test_df["date_local"] = pd.to_datetime(test_df["date_local"]).dt.date

            # --- 1. Sensor features (Parquet hourly) ---
            if self.processed_hourly_dir is not None and self.processed_hourly_dir.exists():
                X_sens_tr_df, y_cont_tr, y_bin_tr = feature_builder.build_parquet_features(
                    train_df, self.processed_hourly_dir
                )
                X_sens_te_df, y_cont_te, y_bin_te = feature_builder.build_parquet_features(
                    test_df, self.processed_hourly_dir
                )
            else:
                logger.warning(f"  Fold {fold}: processed_hourly_dir unavailable — zeros")
                X_sens_tr_df = pd.DataFrame(np.zeros((len(train_df), 1)), columns=["dummy"])
                X_sens_te_df = pd.DataFrame(np.zeros((len(test_df), 1)), columns=["dummy"])
                y_cont_tr = pd.DataFrame({t: train_df.get(t, pd.Series(dtype=float)) for t in CONTINUOUS_TARGETS})
                y_cont_te = pd.DataFrame({t: test_df.get(t, pd.Series(dtype=float)) for t in CONTINUOUS_TARGETS})
                y_bin_tr = pd.DataFrame({t: train_df.get(t, pd.Series(dtype=float)) for t in BINARY_STATE_TARGETS})
                y_bin_te = pd.DataFrame({t: test_df.get(t, pd.Series(dtype=float)) for t in BINARY_STATE_TARGETS})

            X_sens_tr_df = feature_builder.impute_features(X_sens_tr_df)
            X_sens_te_df = feature_builder.impute_features(X_sens_te_df)

            # Align columns
            all_sens_cols = sorted(set(X_sens_tr_df.columns) | set(X_sens_te_df.columns))
            for col in all_sens_cols:
                if col not in X_sens_tr_df.columns:
                    X_sens_tr_df[col] = 0.0
                if col not in X_sens_te_df.columns:
                    X_sens_te_df[col] = 0.0
            X_sens_tr_df = X_sens_tr_df[all_sens_cols]
            X_sens_te_df = X_sens_te_df[all_sens_cols]

            # Scale sensor features
            sens_scaler = StandardScaler()
            X_sens_tr = sens_scaler.fit_transform(X_sens_tr_df.values.astype(np.float32))
            X_sens_te = sens_scaler.transform(X_sens_te_df.values.astype(np.float32))

            # --- 2. Diary embeddings (zeros for absent) ---
            train_texts = train_df["emotion_driver"].tolist() if "emotion_driver" in train_df.columns else [None] * len(train_df)
            test_texts = test_df["emotion_driver"].tolist() if "emotion_driver" in test_df.columns else [None] * len(test_df)

            X_emb_tr = _embed_texts(train_texts, self._embedder)
            X_emb_te = _embed_texts(test_texts, self._embedder)

            # Scale embeddings independently
            emb_scaler = StandardScaler()
            X_emb_tr = emb_scaler.fit_transform(X_emb_tr)
            X_emb_te = emb_scaler.transform(X_emb_te)

            # --- 3. Concatenate ---
            X_train = np.concatenate([X_sens_tr, X_emb_tr], axis=1)
            X_test = np.concatenate([X_sens_te, X_emb_te], axis=1)

            # --- 4. Train and evaluate ---
            reg_models = [m for m in self.model_names if m in self.REGRESSION_MODELS]
            clf_models = [m for m in self.model_names if m in self.CLASSIFICATION_MODELS]

            # Continuous regression
            for target in CONTINUOUS_TARGETS:
                y_tr = pd.to_numeric(y_cont_tr[target], errors="coerce").values if target in y_cont_tr.columns else np.full(len(train_df), np.nan)
                y_te = pd.to_numeric(y_cont_te[target], errors="coerce").values if target in y_cont_te.columns else np.full(len(test_df), np.nan)

                mask_tr = ~np.isnan(y_tr)
                mask_te = ~np.isnan(y_te)
                if mask_tr.sum() < 10 or mask_te.sum() < 5:
                    continue

                for model_name in reg_models:
                    try:
                        model = self._make_model(model_name, task="regression")
                        model.fit(X_train[mask_tr], y_tr[mask_tr])
                        preds = model.predict(X_test[mask_te])
                        mae = float(mean_absolute_error(y_te[mask_te], preds))

                        all_results.setdefault(model_name, {}).setdefault(target, []).append({
                            "fold": fold,
                            "mae": mae,
                            "n_train": int(mask_tr.sum()),
                            "n_test": int(mask_te.sum()),
                        })
                    except Exception as e:
                        logger.warning(f"  {model_name}/{target} fold {fold}: {e}")

            # Binary classification
            binary_targets = list(BINARY_STATE_TARGETS) + ["INT_availability"]
            for target in binary_targets:
                src_tr = y_bin_tr if target in (y_bin_tr.columns if hasattr(y_bin_tr, "columns") else []) else train_df
                src_te = y_bin_te if target in (y_bin_te.columns if hasattr(y_bin_te, "columns") else []) else test_df

                y_tr_raw = src_tr[target].values if target in src_tr.columns else train_df.get(target, pd.Series(dtype=float)).values
                y_te_raw = src_te[target].values if target in src_te.columns else test_df.get(target, pd.Series(dtype=float)).values

                y_tr = self._coerce_binary(y_tr_raw)
                y_te = self._coerce_binary(y_te_raw)

                mask_tr = ~np.isnan(y_tr)
                mask_te = ~np.isnan(y_te)
                if mask_tr.sum() < 10 or mask_te.sum() < 5:
                    continue

                y_tr_int = y_tr[mask_tr].astype(int)
                y_te_int = y_te[mask_te].astype(int)
                if len(set(y_tr_int)) < 2:
                    continue

                for model_name in clf_models:
                    try:
                        clf = self._make_model(model_name, task="classification")
                        clf.fit(X_train[mask_tr], y_tr_int)
                        preds = clf.predict(X_test[mask_te])

                        ba = float(balanced_accuracy_score(y_te_int, preds))
                        f1 = float(f1_score(y_te_int, preds, zero_division=0))

                        all_results.setdefault(model_name, {}).setdefault(target, []).append({
                            "fold": fold,
                            "balanced_accuracy": ba,
                            "f1": f1,
                            "n_train": int(mask_tr.sum()),
                            "n_test": int(mask_te.sum()),
                        })
                    except Exception as e:
                        logger.warning(f"  {model_name}/{target} fold {fold}: {e}")

        aggregated = self._aggregate_folds(all_results)
        self._save_results(all_results, aggregated)
        return aggregated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_model(name: str, task: str):
        """Instantiate the requested sklearn model."""
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
        elif name == "ridge":
            if task != "regression":
                raise ValueError("Ridge only for regression")
            return Ridge(alpha=1.0)
        elif name == "logistic":
            if task != "classification":
                raise ValueError("LogisticRegression only for classification")
            return LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
        else:
            raise ValueError(f"Unknown model: {name}")

    @staticmethod
    def _coerce_binary(arr) -> np.ndarray:
        """Convert raw column to float array with NaN for missing."""
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

    def _save_results(self, raw: dict, aggregated: dict) -> None:
        """Persist raw fold results and aggregated metrics to disk."""
        with open(self.output_dir / "combined_baseline_folds.json", "w") as f:
            json.dump(raw, f, indent=2, default=str)

        with open(self.output_dir / "combined_baseline_metrics.json", "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        lines = [
            "# Combined Baseline Results (5-fold CV)\n",
            "Features: Parquet hourly sensor features + sentence-transformer diary embeddings (zeros for absent)\n",
        ]
        for model_name, targets in aggregated.items():
            agg = targets.get("_aggregate", {})
            lines.append(f"## {model_name.upper()}")
            if agg.get("mean_mae") is not None:
                lines.append(f"  Mean MAE: {agg['mean_mae']:.3f}")
            if agg.get("mean_ba") is not None:
                lines.append(f"  Mean Balanced Accuracy: {agg['mean_ba']:.3f}")
            if agg.get("mean_f1") is not None:
                lines.append(f"  Mean F1: {agg['mean_f1']:.3f}")
            lines.append("")

        (self.output_dir / "combined_baseline_summary.md").write_text("\n".join(lines))
        logger.info(f"Combined baseline results saved to {self.output_dir}")
