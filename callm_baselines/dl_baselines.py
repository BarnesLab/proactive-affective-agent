"""Deep learning baselines on sensing features (MLP).

Uses the same 20 daily aggregate features as ML baselines but with
a PyTorch MLP (2-3 hidden layers, BatchNorm, Dropout).
5-fold participant-grouped CV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

# 4 main binary targets
MAIN_BINARY_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_ER_desire_State",
    "INT_availability",
]


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SensingDataset(Dataset):
    """Sensing features + multi-target dataset."""

    def __init__(self, X: np.ndarray, targets: dict[str, np.ndarray]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.target_names = list(targets.keys())
        self.labels = {}
        self.masks = {}

        for t, y in targets.items():
            labels = np.zeros(len(y), dtype=np.int64)
            masks = np.zeros(len(y), dtype=np.float32)
            for i in range(len(y)):
                if not np.isnan(y[i]):
                    labels[i] = int(y[i])
                    masks[i] = 1.0
            self.labels[t] = torch.tensor(labels, dtype=torch.long)
            self.masks[t] = torch.tensor(masks, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "labels": {t: self.labels[t][idx] for t in self.target_names},
            "masks": {t: self.masks[t][idx] for t in self.target_names},
        }


class MultiHeadMLP(nn.Module):
    """MLP with shared hidden layers and per-target classification heads."""

    def __init__(
        self,
        input_dim: int,
        target_names: list[str],
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.target_names = target_names

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({
            t: nn.Linear(prev_dim, 2) for t in target_names
        })

    def forward(self, features, labels=None, masks=None):
        h = self.backbone(features)
        logits = {t: head(h) for t, head in self.heads.items()}

        loss = None
        if labels is not None and masks is not None:
            criterion = nn.CrossEntropyLoss(reduction="none")
            losses = []
            for t in self.target_names:
                if t in labels and t in masks:
                    t_loss = criterion(logits[t], labels[t])
                    t_loss = (t_loss * masks[t]).sum() / masks[t].sum().clamp(min=1e-9)
                    losses.append(t_loss)
            if losses:
                loss = torch.stack(losses).mean()

        return {"loss": loss, "logits": logits}


class MLPBaseline:
    """MLP baseline on sensing features."""

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        batch_size: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
    ):
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.device = _get_device()

        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X: np.ndarray, targets: dict[str, np.ndarray]) -> None:
        X_scaled = self.scaler.fit_transform(X)

        # Replace NaN with 0 after scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        dataset = SensingDataset(X_scaled, targets)
        loader = TorchDataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        input_dim = X_scaled.shape[1]
        self.model = MultiHeadMLP(
            input_dim=input_dim,
            target_names=list(targets.keys()),
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                optimizer.zero_grad()

                features = batch["features"].to(self.device)
                labels = {t: batch["labels"][t].to(self.device) for t in targets}
                masks = {t: batch["masks"][t].to(self.device) for t in targets}

                outputs = self.model(features, labels=labels, masks=masks)
                loss = outputs["loss"]

                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("  Early stopping at epoch %d", epoch + 1)
                    break

            if (epoch + 1) % 10 == 0:
                logger.info("  Epoch %d/%d, loss=%.4f", epoch + 1, self.epochs, avg_loss)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        all_preds: dict[str, list] = {t: [] for t in self.model.target_names}

        for start in range(0, len(X_tensor), self.batch_size):
            end = min(start + self.batch_size, len(X_tensor))
            batch_x = X_tensor[start:end]
            outputs = self.model(batch_x)

            for t in self.model.target_names:
                preds = outputs["logits"][t].argmax(dim=-1).cpu().numpy()
                all_preds[t].append(preds)

        return {t: np.concatenate(v) for t, v in all_preds.items()}


def run_dl_baselines(
    splits_dir,
    sensing_dfs: dict[str, pd.DataFrame],
    feature_builder,
    output_dir,
    epochs: int = 50,
) -> dict[str, Any]:
    """Run deep learning baselines on sensing features across 5-fold CV.

    Args:
        splits_dir: Path to group_{1-5}_{train,test}.csv.
        sensing_dfs: Pre-loaded sensing DataFrames.
        feature_builder: Module with build_daily_features/impute_features.
        output_dir: Where to save results.
        epochs: Max training epochs.

    Returns:
        Aggregated results dict.
    """
    import json

    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, list]] = {}

    for fold in range(1, 6):
        logger.info("=== DL Baselines Fold %d/5 ===", fold)

        train_df = pd.read_csv(splits_dir / f"group_{fold}_train.csv")
        test_df = pd.read_csv(splits_dir / f"group_{fold}_test.csv")
        train_df["date_local"] = pd.to_datetime(train_df["date_local"]).dt.date
        test_df["date_local"] = pd.to_datetime(test_df["date_local"]).dt.date

        # Build features
        X_train, _, y_bin_train = feature_builder.build_daily_features(
            train_df, sensing_dfs
        )
        X_test, _, y_bin_test = feature_builder.build_daily_features(
            test_df, sensing_dfs
        )

        X_train = feature_builder.impute_features(X_train)
        X_test = feature_builder.impute_features(X_test)

        X_train_np = X_train.values
        X_test_np = X_test.values

        # Prepare targets
        train_targets = {}
        test_targets = {}
        for t in MAIN_BINARY_TARGETS:
            if t in y_bin_train.columns:
                train_targets[t] = y_bin_train[t].values.astype(float)
                test_targets[t] = y_bin_test[t].values.astype(float)

        # MLP baseline
        try:
            model = MLPBaseline(epochs=epochs)
            model.fit(X_train_np, train_targets)
            preds = model.predict(X_test_np)

            for target_name in MAIN_BINARY_TARGETS:
                if target_name not in preds or target_name not in test_targets:
                    continue
                y_true = test_targets[target_name]
                y_pred = preds[target_name]

                mask = ~np.isnan(y_true)
                if mask.sum() < 5:
                    continue

                ba = float(balanced_accuracy_score(
                    y_true[mask].astype(int), y_pred[mask]
                ))
                f1 = float(f1_score(
                    y_true[mask].astype(int), y_pred[mask], zero_division=0
                ))

                all_results.setdefault("mlp", {}).setdefault(
                    target_name, []
                ).append({
                    "fold": fold,
                    "balanced_accuracy": ba,
                    "f1": f1,
                    "n_train": len(X_train_np),
                    "n_test": int(mask.sum()),
                })

        except Exception as e:
            logger.error("  MLP fold %d failed: %s", fold, e)

    # Aggregate
    aggregated = {}
    for model_name, targets in all_results.items():
        aggregated[model_name] = {}
        all_ba = []
        for target, fold_results in targets.items():
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

        aggregated[model_name]["_aggregate"] = {
            "mean_ba": float(np.mean(all_ba)) if all_ba else None,
        }

    # Save
    with open(output_dir / "dl_baseline_folds.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(output_dir / "dl_baseline_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    lines = ["# DL Baseline Results (5-fold CV)\n"]
    lines.append("| Model | PosAff BA | NegAff BA | RegDesire BA | IntAvail BA | Mean BA |")
    lines.append("|-------|-----------|-----------|--------------|-------------|---------|")

    for model_name, targets in aggregated.items():
        cols = [model_name]
        for t in MAIN_BINARY_TARGETS:
            if t in targets:
                cols.append(
                    "%.3f +/- %.3f" % (targets[t]["ba_mean"], targets[t]["ba_std"])
                )
            else:
                cols.append("N/A")
        agg = targets.get("_aggregate", {})
        cols.append("%.3f" % agg.get("mean_ba", 0))
        lines.append("| " + " | ".join(cols) + " |")

    lines.append("")
    (output_dir / "dl_baseline_summary.md").write_text("\n".join(lines))
    logger.info("DL baseline results saved to %s", output_dir)

    return aggregated
