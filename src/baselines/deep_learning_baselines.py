"""Deep learning baseline pipeline: MLP on hourly Parquet features.

Optimized for multi-GPU servers (e.g. 2x RTX A5000/A6000 + 40 CPU cores):
- Config search parallelized via ThreadPoolExecutor (MLP is tiny, fits many on one GPU)
- Target-level parallelism via joblib ProcessPoolExecutor across CPU cores
- Round-robin GPU assignment when multiple GPUs are available

Requires PyTorch. If torch is not installed the pipeline raises ImportError
gracefully when instantiated so the caller can decide whether to skip.
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
from sklearn.impute import SimpleImputer
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
# GPU device helpers
# ---------------------------------------------------------------------------

def _get_available_devices() -> list[str]:
    """Return list of available torch device strings, e.g. ['cuda:0', 'cuda:1'] or ['cpu']."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _pick_device(index: int = 0) -> "torch.device":
    """Round-robin device selection based on an integer index."""
    devices = _get_available_devices()
    return torch.device(devices[index % len(devices)])


# ---------------------------------------------------------------------------
# MLP architecture & hyperparameter configs
# ---------------------------------------------------------------------------

# Lightweight grid of configs to search over per target.
# Each config is small enough that 4 x 30 epochs is fast.
MLP_CONFIGS: list[dict[str, Any]] = [
    {"hidden_dims": [128, 32], "dropout": 0.2, "lr": 1e-3},
    {"hidden_dims": [256, 64], "dropout": 0.3, "lr": 1e-3},
    {"hidden_dims": [256, 128], "dropout": 0.3, "lr": 5e-4},
    {"hidden_dims": [512, 128], "dropout": 0.4, "lr": 5e-4},
]


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
) -> "nn.Module":
    """Build a variable-depth MLP: input -> [hidden -> ReLU -> Dropout]* -> 1.

    Args:
        input_dim: Number of input features.
        hidden_dims: Sizes of hidden layers (default ``[256, 64]``).
        dropout: Dropout probability applied after each hidden layer.
    """
    if hidden_dims is None:
        hidden_dims = [256, 64]

    layers: list[nn.Module] = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)])
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
    patience: int = 5,
    val_fraction: float = 0.15,
    seed: int = 42,
    device: "torch.device | None" = None,
) -> tuple["nn.Module", float]:
    """Build, train, and return (model, best_val_loss).

    Holds out *val_fraction* of the training data for validation monitoring.
    Training stops when validation loss has not improved for *patience* epochs,
    and the best-performing weights are restored before returning.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        task: ``"regression"`` or ``"classification"``.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizer.
        hidden_dims: Hidden layer sizes.
        dropout: Dropout probability.
        patience: Early-stopping patience.
        val_fraction: Fraction held out for validation.
        seed: Random seed for reproducibility.
        device: Torch device to train on. Auto-detected if None.

    Returns:
        A tuple of (trained_model, best_validation_loss).
    """
    # Prevent thread oversubscription in worker processes
    torch.set_num_threads(1)

    input_dim = X_train.shape[1]
    model = _build_mlp(input_dim, hidden_dims=hidden_dims, dropout=dropout)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Validation split (shuffled, deterministic via seed above) ---
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
        # Compute pos_weight from training labels to handle class imbalance
        n_pos = float(y_train[train_idx].sum())
        n_neg = float(len(train_idx) - n_pos)
        pw = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _epoch in range(epochs):
        # --- Training ---
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validation ---
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
                logger.debug(f"Early stopping at epoch {_epoch + 1} (patience={patience})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def _search_best_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
    configs: list[dict[str, Any]] | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 5,
    val_fraction: float = 0.15,
    seed: int = 42,
    device: "torch.device | None" = None,
) -> "nn.Module":
    """Train one MLP per config **in parallel** and return the best model.

    All configs are trained concurrently using a ThreadPoolExecutor. Threads
    are efficient here because PyTorch releases the GIL during GPU/BLAS ops,
    and the MLP is tiny enough that multiple configs fit in GPU memory at once.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        task: ``"regression"`` or ``"classification"``.
        configs: List of dicts with keys ``hidden_dims``, ``dropout``, ``lr``.
            Defaults to :data:`MLP_CONFIGS`.
        epochs: Maximum training epochs per config.
        batch_size: Mini-batch size.
        patience: Early-stopping patience (epochs without val improvement).
        val_fraction: Fraction of training data held out for validation.
        seed: Base random seed (each config gets the same seed for fair comparison).
        device: Torch device. Auto-detected if None.

    Returns:
        The best-performing trained model.
    """
    if configs is None:
        configs = MLP_CONFIGS

    best_model: nn.Module | None = None
    best_loss = float("inf")
    best_cfg: dict[str, Any] = {}

    def _train_one_config(cfg: dict[str, Any]) -> tuple["nn.Module", float, dict[str, Any]]:
        """Train a single config and return (model, val_loss, cfg)."""
        model, val_loss = _train_mlp(
            X_train,
            y_train,
            task=task,
            epochs=epochs,
            batch_size=batch_size,
            lr=cfg.get("lr", 1e-3),
            hidden_dims=cfg.get("hidden_dims"),
            dropout=cfg.get("dropout", 0.3),
            patience=patience,
            val_fraction=val_fraction,
            seed=seed,
            device=device,
        )
        return model, val_loss, cfg

    # Train all configs in parallel using threads (MLP is tiny, GIL released
    # during CUDA/BLAS ops, so threads give real concurrency here).
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        futures = {
            pool.submit(_train_one_config, cfg): cfg
            for cfg in configs
        }
        for future in as_completed(futures):
            model, val_loss, cfg = future.result()
            logger.debug(
                "  Config hidden_dims=%s dropout=%.2f lr=%.4f -> val_loss=%.4f",
                cfg.get("hidden_dims"), cfg.get("dropout", 0.3), cfg.get("lr", 1e-3), val_loss,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
                best_cfg = cfg

    logger.info(
        "  Best MLP config: hidden_dims=%s, dropout=%.2f, lr=%.4f (val_loss=%.4f)",
        best_cfg.get("hidden_dims"),
        best_cfg.get("dropout", 0.3),
        best_cfg.get("lr", 1e-3),
        best_loss,
    )

    assert best_model is not None  # at least one config must have been evaluated
    return best_model


def _predict_mlp(
    model: "nn.Module",
    X_test: np.ndarray,
    task: str = "regression",
    device: "torch.device | None" = None,
) -> np.ndarray:
    """Run inference and return numpy predictions."""
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
        # Binary: numerically stable sigmoid + threshold at 0.5
        logits_clipped = np.clip(logits, -500.0, 500.0)
        probs = 1.0 / (1.0 + np.exp(-logits_clipped))
        return (probs >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Standalone target training function (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _train_and_evaluate_target(
    target: str,
    task_type: str,
    X_train_np: np.ndarray,
    X_test_np: np.ndarray,
    y_tr_vals: np.ndarray,
    y_te_vals: np.ndarray,
    mask_tr: np.ndarray,
    mask_te: np.ndarray,
    target_index: int,
    device_str: str,
) -> dict[str, Any] | None:
    """Train MLP for a single target and return the result dict.

    This is a top-level function (not a method) so it can be pickled for
    multiprocessing. Each worker process gets its own GIL and torch threads.

    Args:
        target: Target column name.
        task_type: ``"regression"`` or ``"classification"``.
        X_train_np: Scaled training features (all rows, pre-imputed).
        X_test_np: Scaled test features (all rows, pre-imputed).
        y_tr_vals: Training target values (with NaN for missing).
        y_te_vals: Test target values (with NaN for missing).
        mask_tr: Boolean mask for valid training rows.
        mask_te: Boolean mask for valid test rows.
        target_index: Index for deterministic seed = 42 + target_index.
        device_str: Torch device string, e.g. "cuda:0" or "cpu".

    Returns:
        Dict with metrics, or None if the target was skipped.
    """
    # Prevent thread oversubscription inside worker process
    import os as _worker_os
    _worker_os.environ["OMP_NUM_THREADS"] = "1"
    _worker_os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    device = torch.device(device_str)
    seed = 42 + target_index

    try:
        if task_type == "regression":
            net = _search_best_mlp(
                X_train_np[mask_tr],
                y_tr_vals[mask_tr].astype(np.float32),
                task="regression",
                seed=seed,
                device=device,
            )
            preds = _predict_mlp(net, X_test_np[mask_te], task="regression", device=device)
            mae = float(mean_absolute_error(y_te_vals[mask_te], preds))
            return {
                "target": target,
                "task_type": task_type,
                "fold_result": {
                    "mae": mae,
                    "n_train": int(mask_tr.sum()),
                    "n_test": int(mask_te.sum()),
                },
            }
        else:
            y_tr_int = y_tr_vals[mask_tr].astype(int)
            y_te_int = y_te_vals[mask_te].astype(int)
            net = _search_best_mlp(
                X_train_np[mask_tr],
                y_tr_int.astype(np.float32),
                task="classification",
                seed=seed,
                device=device,
            )
            preds = _predict_mlp(net, X_test_np[mask_te], task="classification", device=device)

            ba = float(balanced_accuracy_score(y_te_int, preds))
            f1_val = float(f1_score(y_te_int, preds, zero_division=0))

            return {
                "target": target,
                "task_type": task_type,
                "fold_result": {
                    "balanced_accuracy": ba,
                    "f1": f1_val,
                    "n_train": int(mask_tr.sum()),
                    "n_test": int(mask_te.sum()),
                },
            }
    except Exception as e:
        logger.warning(f"  mlp/{target}: {e}")
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DLBaselinePipeline:
    """MLP on hourly Parquet features, 5-fold CV matching MLBaselinePipeline.

    Parallelism strategy (optimized for 2-GPU + 40-core servers):
    - **Config search** (inner loop): ThreadPoolExecutor trains 4 MLP configs
      simultaneously on the same GPU. The MLP is tiny (<1 MB) so all fit easily.
    - **Target training** (outer loop): joblib or ProcessPoolExecutor distributes
      independent targets across CPU cores. Each worker gets a GPU via round-robin.
    """

    def __init__(
        self,
        splits_dir: Path,
        output_dir: Path,
        processed_hourly_dir: Path | None = None,
        model_names: list[str] | None = None,
        n_workers: int | None = None,
    ) -> None:
        """
        Args:
            splits_dir: Directory with group_{1-5}_{train,test}.csv files.
            output_dir: Where to save results.
            processed_hourly_dir: Path to data/processed/hourly/ for Parquet features.
            model_names: Currently only ["mlp"] is supported (kept for interface parity).
            n_workers: Number of parallel workers for target-level parallelism.
                Defaults to min(num_targets, cpu_count) capped at 20 to avoid
                memory pressure. Set to 1 to disable parallelism.
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
        self.n_workers = n_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_folds(self, folds: list | None = None) -> dict[str, Any]:
        """Run CV for MLP on Parquet hourly features.

        Targets within each fold are trained in parallel using process-based
        parallelism. GPU devices are distributed round-robin across workers.

        Args:
            folds: List of fold indices to run (e.g. [1], [3, 4]). If None, runs all 5 folds.

        Returns:
            Nested dict: {model_name: {target: {metric: ...}, "_aggregate": {...}}}
        """
        from src.baselines import feature_builder
        from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

        all_results: dict[str, Any] = {}
        devices = _get_available_devices()

        for fold in (folds if folds is not None else range(1, 6)):
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

            # Impute (fit on train only to avoid data leakage)
            imputer = SimpleImputer(strategy="median")
            X_train_df = pd.DataFrame(
                imputer.fit_transform(X_train_df),
                columns=X_train_df.columns,
                index=X_train_df.index,
            )
            X_test_df = pd.DataFrame(
                imputer.transform(X_test_df),
                columns=X_test_df.columns,
                index=X_test_df.index,
            )

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

            for model_name in self.model_names:
                if model_name != "mlp":
                    continue

                # ---- Collect all target tasks for parallel execution ----
                target_tasks: list[dict[str, Any]] = []

                # Continuous regression targets
                for target in CONTINUOUS_TARGETS:
                    y_tr = pd.to_numeric(y_cont_tr[target], errors="coerce").values if target in y_cont_tr.columns else np.full(len(train_df), np.nan)
                    y_te = pd.to_numeric(y_cont_te[target], errors="coerce").values if target in y_cont_te.columns else np.full(len(test_df), np.nan)

                    mask_tr = ~np.isnan(y_tr)
                    mask_te = ~np.isnan(y_te)
                    if mask_tr.sum() < 10 or mask_te.sum() < 5:
                        continue

                    target_tasks.append({
                        "target": target,
                        "task_type": "regression",
                        "y_tr_vals": y_tr,
                        "y_te_vals": y_te,
                        "mask_tr": mask_tr,
                        "mask_te": mask_te,
                    })

                # Binary classification targets
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
                    if len(set(y_tr_int)) < 2:
                        continue

                    target_tasks.append({
                        "target": target,
                        "task_type": "classification",
                        "y_tr_vals": y_tr,
                        "y_te_vals": y_te,
                        "mask_tr": mask_tr,
                        "mask_te": mask_te,
                    })

                # ---- Execute all targets in parallel ----
                n_targets = len(target_tasks)
                if n_targets == 0:
                    continue

                # Default workers: min(num_targets, cpu_count, 20)
                max_workers = self.n_workers
                if max_workers is None:
                    cpu_count = mp.cpu_count() or 4
                    max_workers = min(n_targets, cpu_count, 20)
                max_workers = max(1, max_workers)

                logger.info(
                    f"  Fold {fold}: Training {n_targets} targets with "
                    f"{max_workers} parallel workers on devices={devices}"
                )

                if max_workers == 1:
                    # Sequential fallback (useful for debugging)
                    results_list = []
                    for idx, tt in enumerate(target_tasks):
                        device_str = devices[idx % len(devices)]
                        result = _train_and_evaluate_target(
                            target=tt["target"],
                            task_type=tt["task_type"],
                            X_train_np=X_train_np,
                            X_test_np=X_test_np,
                            y_tr_vals=tt["y_tr_vals"],
                            y_te_vals=tt["y_te_vals"],
                            mask_tr=tt["mask_tr"],
                            mask_te=tt["mask_te"],
                            target_index=idx,
                            device_str=device_str,
                        )
                        results_list.append(result)
                else:
                    # Parallel execution via joblib (loky backend handles pickling
                    # and process reuse better than raw ProcessPoolExecutor).
                    try:
                        from joblib import Parallel, delayed

                        results_list = Parallel(
                            n_jobs=max_workers,
                            backend="loky",
                            verbose=0,
                        )(
                            delayed(_train_and_evaluate_target)(
                                target=tt["target"],
                                task_type=tt["task_type"],
                                X_train_np=X_train_np,
                                X_test_np=X_test_np,
                                y_tr_vals=tt["y_tr_vals"],
                                y_te_vals=tt["y_te_vals"],
                                mask_tr=tt["mask_tr"],
                                mask_te=tt["mask_te"],
                                target_index=idx,
                                device_str=devices[idx % len(devices)],
                            )
                            for idx, tt in enumerate(target_tasks)
                        )
                    except ImportError:
                        # Fallback: use ProcessPoolExecutor if joblib not available
                        from concurrent.futures import ProcessPoolExecutor

                        logger.info("  joblib not available, falling back to ProcessPoolExecutor")
                        with ProcessPoolExecutor(max_workers=max_workers) as pool:
                            futures = []
                            for idx, tt in enumerate(target_tasks):
                                device_str = devices[idx % len(devices)]
                                fut = pool.submit(
                                    _train_and_evaluate_target,
                                    target=tt["target"],
                                    task_type=tt["task_type"],
                                    X_train_np=X_train_np,
                                    X_test_np=X_test_np,
                                    y_tr_vals=tt["y_tr_vals"],
                                    y_te_vals=tt["y_te_vals"],
                                    mask_tr=tt["mask_tr"],
                                    mask_te=tt["mask_te"],
                                    target_index=idx,
                                    device_str=device_str,
                                )
                                futures.append(fut)
                            results_list = [f.result() for f in futures]

                # ---- Collect results ----
                for result in results_list:
                    if result is None:
                        continue
                    target = result["target"]
                    fold_result = result["fold_result"]
                    fold_result["fold"] = fold
                    all_results.setdefault(model_name, {}).setdefault(target, []).append(fold_result)

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
        lines.append("Architecture: MLP with hyperparameter search over 4 configs")
        lines.append("  Configs: " + ", ".join(
            f"{c['hidden_dims']} dr={c['dropout']} lr={c['lr']}" for c in MLP_CONFIGS
        ))
        lines.append("  Best config selected per target via validation loss\n")
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
