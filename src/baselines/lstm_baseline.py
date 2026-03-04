"""LSTM baseline pipeline: temporal model on hourly Parquet features.

Input shape: (batch, 24, F) — 24 hourly windows x F sensor features.
Architecture: LSTM -> Linear head, with hyperparameter search over
hidden_size, num_layers, and dropout.

Follows the same interface as DLBaselinePipeline for consistency.
Requires PyTorch.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
)

logger = logging.getLogger(__name__)

try:
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed — LSTMBaselinePipeline will not be functional")


# ---------------------------------------------------------------------------
# LSTM architecture & configs
# ---------------------------------------------------------------------------

LSTM_CONFIGS: list[dict[str, Any]] = [
    {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "lr": 1e-3},
    {"hidden_size": 64, "num_layers": 2, "dropout": 0.3, "lr": 1e-3},
    {"hidden_size": 128, "num_layers": 1, "dropout": 0.3, "lr": 5e-4},
    {"hidden_size": 128, "num_layers": 2, "dropout": 0.4, "lr": 5e-4},
]


class LSTMModel(nn.Module):
    """LSTM with a linear head for regression or classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden)
        out = h_n[-1]  # last layer hidden state: (batch, hidden)
        out = self.dropout(out)
        return self.fc(out)  # (batch, 1)


def _train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 5,
    val_fraction: float = 0.15,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[nn.Module, float]:
    """Train an LSTM model and return (model, best_val_loss)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train.shape[2]  # (n, seq_len, features)
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)

    # Validation split
    n = len(X_train)
    indices = np.random.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32).unsqueeze(1).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if task == "regression":
        criterion = nn.MSELoss()
    else:
        n_pos = float(y_train[train_idx].sum())
        n_neg = float(len(train_idx) - n_pos)
        pw = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def _search_best_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
    configs: list[dict[str, Any]] | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> nn.Module:
    """Train one LSTM per config in parallel and return the best model."""
    if configs is None:
        configs = LSTM_CONFIGS

    best_model: nn.Module | None = None
    best_loss = float("inf")
    best_cfg: dict[str, Any] = {}

    def _train_one(cfg: dict[str, Any]) -> tuple[nn.Module, float, dict[str, Any]]:
        model, val_loss = _train_lstm(
            X_train, y_train, task=task,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            lr=cfg.get("lr", 1e-3),
            seed=seed, device=device,
        )
        return model, val_loss, cfg

    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        futures = {pool.submit(_train_one, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            model, val_loss, cfg = future.result()
            logger.debug(
                "  LSTM config h=%d layers=%d dr=%.2f lr=%.4f -> val_loss=%.4f",
                cfg["hidden_size"], cfg["num_layers"], cfg["dropout"],
                cfg.get("lr", 1e-3), val_loss,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
                best_cfg = cfg

    logger.info(
        "  Best LSTM: h=%d layers=%d dr=%.2f lr=%.4f (val_loss=%.4f)",
        best_cfg.get("hidden_size", 0), best_cfg.get("num_layers", 0),
        best_cfg.get("dropout", 0), best_cfg.get("lr", 0), best_loss,
    )
    assert best_model is not None
    return best_model


def _predict_lstm(
    model: nn.Module,
    X_test: np.ndarray,
    task: str = "regression",
    device: torch.device | None = None,
) -> np.ndarray:
    """Run LSTM inference and return numpy predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_t).squeeze(1).cpu().numpy()

    if task == "regression":
        return logits
    else:
        logits_clipped = np.clip(logits, -500.0, 500.0)
        probs = 1.0 / (1.0 + np.exp(-logits_clipped))
        return (probs >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Standalone target function for multiprocessing
# ---------------------------------------------------------------------------

def _train_and_evaluate_lstm_target(
    target: str,
    task_type: str,
    X_train_3d: np.ndarray,
    X_test_3d: np.ndarray,
    y_tr_vals: np.ndarray,
    y_te_vals: np.ndarray,
    mask_tr: np.ndarray,
    mask_te: np.ndarray,
    target_index: int,
    device_str: str,
) -> dict[str, Any] | None:
    """Train LSTM for a single target and return result dict."""
    import os as _worker_os
    _worker_os.environ["OMP_NUM_THREADS"] = "1"
    _worker_os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    device = torch.device(device_str)
    seed = 42 + target_index

    try:
        if task_type == "regression":
            net = _search_best_lstm(
                X_train_3d[mask_tr],
                y_tr_vals[mask_tr].astype(np.float32),
                task="regression", seed=seed, device=device,
            )
            preds = _predict_lstm(net, X_test_3d[mask_te], task="regression", device=device)
            mae = float(mean_absolute_error(y_te_vals[mask_te], preds))
            return {
                "target": target, "task_type": task_type,
                "fold_result": {"mae": mae, "n_train": int(mask_tr.sum()), "n_test": int(mask_te.sum())},
            }
        else:
            y_tr_int = y_tr_vals[mask_tr].astype(int)
            y_te_int = y_te_vals[mask_te].astype(int)
            net = _search_best_lstm(
                X_train_3d[mask_tr],
                y_tr_int.astype(np.float32),
                task="classification", seed=seed, device=device,
            )
            preds = _predict_lstm(net, X_test_3d[mask_te], task="classification", device=device)
            ba = float(balanced_accuracy_score(y_te_int, preds))
            f1_val = float(f1_score(y_te_int, preds, zero_division=0))
            return {
                "target": target, "task_type": task_type,
                "fold_result": {"balanced_accuracy": ba, "f1": f1_val,
                                "n_train": int(mask_tr.sum()), "n_test": int(mask_te.sum())},
            }
    except Exception as e:
        logger.warning(f"  lstm/{target}: {e}")
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LSTMBaselinePipeline:
    """LSTM on hourly Parquet features (3D temporal), 5-fold CV."""

    def __init__(
        self,
        splits_dir: Path,
        output_dir: Path,
        processed_hourly_dir: Path | None = None,
        n_workers: int | None = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LSTMBaselinePipeline.")

        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_hourly_dir = Path(processed_hourly_dir) if processed_hourly_dir else None
        self.n_workers = n_workers

    def run_all_folds(self, folds: list | None = None) -> dict[str, Any]:
        """Run CV for LSTM on hourly 3D features."""
        from src.baselines import feature_builder
        from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

        all_results: dict[str, Any] = {}
        devices = ["cpu"]
        if HAS_TORCH and torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

        for fold in (folds if folds is not None else range(1, 6)):
            logger.info(f"=== LSTM Baseline Fold {fold}/5 ===")

            train_df = pd.read_csv(self.splits_dir / f"group_{fold}_train.csv")
            test_df = pd.read_csv(self.splits_dir / f"group_{fold}_test.csv")
            train_df["date_local"] = pd.to_datetime(train_df["date_local"]).dt.date
            test_df["date_local"] = pd.to_datetime(test_df["date_local"]).dt.date

            if self.processed_hourly_dir is not None and self.processed_hourly_dir.exists():
                X_train_3d, y_cont_tr, y_bin_tr = feature_builder.build_parquet_features_3d(
                    train_df, self.processed_hourly_dir
                )
                X_test_3d, y_cont_te, y_bin_te = feature_builder.build_parquet_features_3d(
                    test_df, self.processed_hourly_dir
                )
            else:
                logger.warning(f"  Fold {fold}: processed_hourly_dir not available — zeros")
                n_feat = 28
                X_train_3d = np.zeros((len(train_df), 24, n_feat), dtype=np.float32)
                X_test_3d = np.zeros((len(test_df), 24, n_feat), dtype=np.float32)
                y_cont_tr = pd.DataFrame({t: train_df.get(t, pd.Series(dtype=float)) for t in CONTINUOUS_TARGETS})
                y_cont_te = pd.DataFrame({t: test_df.get(t, pd.Series(dtype=float)) for t in CONTINUOUS_TARGETS})
                y_bin_tr = pd.DataFrame({t: train_df.get(t, pd.Series(dtype=float)) for t in BINARY_STATE_TARGETS})
                y_bin_te = pd.DataFrame({t: test_df.get(t, pd.Series(dtype=float)) for t in BINARY_STATE_TARGETS})

            # Impute NaN with 0 for 3D data and normalize per-feature
            X_train_3d = np.nan_to_num(X_train_3d, nan=0.0).astype(np.float32)
            X_test_3d = np.nan_to_num(X_test_3d, nan=0.0).astype(np.float32)

            # Normalize: compute mean/std across samples and hours for each feature
            n_samples_tr, n_hours, n_feat = X_train_3d.shape
            flat_tr = X_train_3d.reshape(-1, n_feat)
            feat_mean = flat_tr.mean(axis=0)
            feat_std = flat_tr.std(axis=0)
            feat_std[feat_std < 1e-8] = 1.0  # avoid divide by zero

            X_train_3d = (X_train_3d - feat_mean) / feat_std
            X_test_3d = (X_test_3d - feat_mean) / feat_std

            # Collect target tasks
            target_tasks: list[dict[str, Any]] = []

            for target in CONTINUOUS_TARGETS:
                y_tr = pd.to_numeric(y_cont_tr[target], errors="coerce").values if target in y_cont_tr.columns else np.full(len(train_df), np.nan)
                y_te = pd.to_numeric(y_cont_te[target], errors="coerce").values if target in y_cont_te.columns else np.full(len(test_df), np.nan)
                mask_tr = ~np.isnan(y_tr)
                mask_te = ~np.isnan(y_te)
                if mask_tr.sum() < 10 or mask_te.sum() < 5:
                    continue
                target_tasks.append({
                    "target": target, "task_type": "regression",
                    "y_tr_vals": y_tr, "y_te_vals": y_te,
                    "mask_tr": mask_tr, "mask_te": mask_te,
                })

            binary_targets = list(BINARY_STATE_TARGETS) + ["INT_availability"]
            for target in binary_targets:
                src_tr = y_bin_tr if target in y_bin_tr.columns else train_df
                src_te = y_bin_te if target in y_bin_te.columns else test_df
                y_tr_raw = src_tr[target].values if target in src_tr.columns else np.full(len(train_df), np.nan)
                y_te_raw = src_te[target].values if target in src_te.columns else np.full(len(test_df), np.nan)
                y_tr = self._coerce_binary(y_tr_raw)
                y_te = self._coerce_binary(y_te_raw)
                mask_tr = ~np.isnan(y_tr)
                mask_te = ~np.isnan(y_te)
                if mask_tr.sum() < 10 or mask_te.sum() < 5:
                    continue
                y_tr_int = y_tr[mask_tr].astype(int)
                if len(set(y_tr_int)) < 2:
                    continue
                target_tasks.append({
                    "target": target, "task_type": "classification",
                    "y_tr_vals": y_tr, "y_te_vals": y_te,
                    "mask_tr": mask_tr, "mask_te": mask_te,
                })

            if not target_tasks:
                continue

            max_workers = self.n_workers
            if max_workers is None:
                cpu_count = mp.cpu_count() or 4
                max_workers = min(len(target_tasks), cpu_count, 20)
            max_workers = max(1, max_workers)

            logger.info(f"  Fold {fold}: Training {len(target_tasks)} targets, workers={max_workers}")

            if max_workers == 1:
                results_list = []
                for idx, tt in enumerate(target_tasks):
                    result = _train_and_evaluate_lstm_target(
                        target=tt["target"], task_type=tt["task_type"],
                        X_train_3d=X_train_3d, X_test_3d=X_test_3d,
                        y_tr_vals=tt["y_tr_vals"], y_te_vals=tt["y_te_vals"],
                        mask_tr=tt["mask_tr"], mask_te=tt["mask_te"],
                        target_index=idx, device_str=devices[idx % len(devices)],
                    )
                    results_list.append(result)
            else:
                from joblib import Parallel, delayed
                results_list = Parallel(n_jobs=max_workers, backend="loky", verbose=0)(
                    delayed(_train_and_evaluate_lstm_target)(
                        target=tt["target"], task_type=tt["task_type"],
                        X_train_3d=X_train_3d, X_test_3d=X_test_3d,
                        y_tr_vals=tt["y_tr_vals"], y_te_vals=tt["y_te_vals"],
                        mask_tr=tt["mask_tr"], mask_te=tt["mask_te"],
                        target_index=idx, device_str=devices[idx % len(devices)],
                    )
                    for idx, tt in enumerate(target_tasks)
                )

            for result in results_list:
                if result is None:
                    continue
                target = result["target"]
                fold_result = result["fold_result"]
                fold_result["fold"] = fold
                all_results.setdefault("lstm", {}).setdefault(target, []).append(fold_result)

        aggregated = self._aggregate_folds(all_results)
        self._save_results(all_results, aggregated)
        return aggregated

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
        aggregated: dict[str, Any] = {}
        for model_name, targets in all_results.items():
            aggregated[model_name] = {}
            all_mae, all_ba, all_f1 = [], [], []
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
        with open(self.output_dir / "lstm_baseline_folds.json", "w") as f:
            json.dump(raw, f, indent=2, default=str)
        with open(self.output_dir / "lstm_baseline_metrics.json", "w") as f:
            json.dump(aggregated, f, indent=2, default=str)

        lines = ["# LSTM Baseline Results (5-fold CV)\n"]
        lines.append("Architecture: LSTM -> Linear, hyperparameter search over 4 configs")
        lines.append("  Configs: " + ", ".join(
            f"h={c['hidden_size']} layers={c['num_layers']} dr={c['dropout']} lr={c['lr']}"
            for c in LSTM_CONFIGS
        ))
        lines.append("")
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
        (self.output_dir / "lstm_baseline_summary.md").write_text("\n".join(lines))
        logger.info(f"LSTM baseline results saved to {self.output_dir}")
