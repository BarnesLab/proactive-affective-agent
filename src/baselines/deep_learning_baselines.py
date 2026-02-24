"""Deep learning baseline pipeline: MLP on hourly Parquet features.

Requires PyTorch. If torch is not installed the pipeline raises ImportError
gracefully when instantiated so the caller can decide whether to skip.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)  # prevent SIGSEGV in OpenMP/KMP on macOS Apple Silicon
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed — DLBaselinePipeline will not be functional")


# ---------------------------------------------------------------------------
# MLP architecture
# ---------------------------------------------------------------------------

def _build_mlp(input_dim: int) -> "nn.Module":
    """Return a 3-layer MLP: input → 256 → 64 → 1."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def _train_mlp(
    model: "nn.Module",
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> "nn.Module":
    """Train model in-place and return it."""
    device = torch.device("cpu")
    model = model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model


def _predict_mlp(
    model: "nn.Module",
    X_test: np.ndarray,
    task: str = "regression",
) -> np.ndarray:
    """Run inference and return numpy predictions."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_t).squeeze(1).cpu().numpy()

    if task == "regression":
        return logits
    else:
        # Binary: sigmoid + threshold at 0.5
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DLBaselinePipeline:
    """MLP on hourly Parquet features, 5-fold CV matching MLBaselinePipeline."""

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
            model_names: Currently only ["mlp"] is supported (kept for interface parity).
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for DLBaselinePipeline. "
                "Install it with: pip install torch"
            )

        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_hourly_dir = Path(processed_hourly_dir) if processed_hourly_dir else None
        self.model_names = model_names if model_names is not None else ["mlp"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_folds(self) -> dict[str, Any]:
        """Run 5-fold CV for MLP on Parquet hourly features.

        Returns:
            Nested dict: {model_name: {target: {metric: ...}, "_aggregate": {...}}}
        """
        from src.baselines import feature_builder
        from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

        all_results: dict[str, Any] = {}

        for fold in range(1, 6):
            logger.info(f"=== DL Baseline Fold {fold}/5 ===")

            train_df = pd.read_csv(self.splits_dir / f"group_{fold}_train.csv")
            test_df = pd.read_csv(self.splits_dir / f"group_{fold}_test.csv")
            train_df["date_local"] = pd.to_datetime(train_df["date_local"]).dt.date
            test_df["date_local"] = pd.to_datetime(test_df["date_local"]).dt.date

            # Build Parquet features (or fall back to all-NaN if dir missing)
            if self.processed_hourly_dir is not None and self.processed_hourly_dir.exists():
                X_train_df, y_cont_tr, y_bin_tr = feature_builder.build_parquet_features(
                    train_df, self.processed_hourly_dir
                )
                X_test_df, y_cont_te, y_bin_te = feature_builder.build_parquet_features(
                    test_df, self.processed_hourly_dir
                )
            else:
                logger.warning(f"  Fold {fold}: processed_hourly_dir not available — zeros")
                X_train_df = pd.DataFrame(np.zeros((len(train_df), 1)), columns=["dummy"])
                X_test_df = pd.DataFrame(np.zeros((len(test_df), 1)), columns=["dummy"])
                y_cont_tr = pd.DataFrame(
                    {t: train_df.get(t, pd.Series(dtype=float)) for t in CONTINUOUS_TARGETS}
                )
                y_cont_te = pd.DataFrame(
                    {t: test_df.get(t, pd.Series(dtype=float)) for t in CONTINUOUS_TARGETS}
                )
                y_bin_tr = pd.DataFrame(
                    {t: train_df.get(t, pd.Series(dtype=float)) for t in BINARY_STATE_TARGETS}
                )
                y_bin_te = pd.DataFrame(
                    {t: test_df.get(t, pd.Series(dtype=float)) for t in BINARY_STATE_TARGETS}
                )

            # Impute and scale
            X_train_df = feature_builder.impute_features(X_train_df)
            X_test_df = feature_builder.impute_features(X_test_df)

            # Align columns
            all_cols = sorted(set(X_train_df.columns) | set(X_test_df.columns))
            for col in all_cols:
                if col not in X_train_df.columns:
                    X_train_df[col] = 0.0
                if col not in X_test_df.columns:
                    X_test_df[col] = 0.0
            X_train_df = X_train_df[all_cols]
            X_test_df = X_test_df[all_cols]

            scaler = StandardScaler()
            X_train_np = scaler.fit_transform(X_train_df.values.astype(np.float32))
            X_test_np = scaler.transform(X_test_df.values.astype(np.float32))
            input_dim = X_train_np.shape[1]

            for model_name in self.model_names:
                if model_name != "mlp":
                    continue

                # Continuous regression
                for target in CONTINUOUS_TARGETS:
                    y_tr = pd.to_numeric(y_cont_tr[target], errors="coerce").values if target in y_cont_tr.columns else np.full(len(train_df), np.nan)
                    y_te = pd.to_numeric(y_cont_te[target], errors="coerce").values if target in y_cont_te.columns else np.full(len(test_df), np.nan)

                    mask_tr = ~np.isnan(y_tr)
                    mask_te = ~np.isnan(y_te)
                    if mask_tr.sum() < 10 or mask_te.sum() < 5:
                        continue

                    try:
                        net = _build_mlp(input_dim)
                        net = _train_mlp(
                            net, X_train_np[mask_tr], y_tr[mask_tr].astype(np.float32),
                            task="regression",
                        )
                        preds = _predict_mlp(net, X_test_np[mask_te], task="regression")
                        mae = float(mean_absolute_error(y_te[mask_te], preds))

                        all_results.setdefault(model_name, {}).setdefault(target, []).append({
                            "fold": fold,
                            "mae": mae,
                            "n_train": int(mask_tr.sum()),
                            "n_test": int(mask_te.sum()),
                        })
                    except Exception as e:
                        logger.warning(f"  mlp/{target} fold {fold}: {e}")

                # Binary classification
                binary_targets = list(BINARY_STATE_TARGETS) + ["INT_availability"]
                for target in binary_targets:
                    src_df_tr = y_bin_tr if target in (y_bin_tr.columns if hasattr(y_bin_tr, "columns") else []) else train_df
                    src_df_te = y_bin_te if target in (y_bin_te.columns if hasattr(y_bin_te, "columns") else []) else test_df

                    y_tr_raw = src_df_tr[target].values if target in src_df_tr.columns else train_df.get(target, pd.Series(dtype=float)).values
                    y_te_raw = src_df_te[target].values if target in src_df_te.columns else test_df.get(target, pd.Series(dtype=float)).values

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

                    try:
                        net = _build_mlp(input_dim)
                        net = _train_mlp(
                            net, X_train_np[mask_tr], y_tr_int.astype(np.float32),
                            task="classification",
                        )
                        preds = _predict_mlp(net, X_test_np[mask_te], task="classification")

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
                        logger.warning(f"  mlp/{target} fold {fold}: {e}")

        aggregated = self._aggregate_folds(all_results)
        self._save_results(all_results, aggregated)
        return aggregated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        with open(self.output_dir / "dl_baseline_folds.json", "w") as f:
            json.dump(raw, f, indent=2, default=str)

        with open(self.output_dir / "dl_baseline_metrics.json", "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        lines = ["# Deep Learning Baseline Results (5-fold CV)\n"]
        lines.append("Architecture: Linear(→256) → ReLU → Dropout(0.3) → Linear(→64) → ReLU → Linear(→1)\n")
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

        (self.output_dir / "dl_baseline_summary.md").write_text("\n".join(lines))
        logger.info(f"DL baseline results saved to {self.output_dir}")
