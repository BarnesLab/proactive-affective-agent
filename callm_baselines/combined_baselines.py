"""Combined (text + sensing) baselines: late fusion and deep fusion.

Late fusion: TF-IDF/transformer embeddings + sensing features -> concat -> LogReg/RF.
Deep fusion: text encoder + sensing MLP -> concat -> classifier (PyTorch).
5-fold participant-grouped CV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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


def _parse_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract the 4 main binary targets from a DataFrame."""
    targets = {}
    for t in MAIN_BINARY_TARGETS:
        if t == "INT_availability":
            vals = df[t].map({"Yes": 1, "yes": 1, "No": 0, "no": 0})
            targets[t] = vals.values.astype(float)
        else:
            targets[t] = pd.to_numeric(df[t], errors="coerce").values
    return targets


# ---------------------------------------------------------------------------
# Late Fusion (sklearn-based)
# ---------------------------------------------------------------------------


class LateFusionBaseline:
    """Concatenate text features + sensing features, then classify with LogReg or RF.

    Text features can be TF-IDF or pre-extracted transformer embeddings.
    """

    def __init__(
        self,
        text_method: str = "tfidf",
        classifier: str = "logistic",
        max_features: int = 3000,
    ):
        self.text_method = text_method
        self.classifier_name = classifier
        self.max_features = max_features

        self.vectorizer = None
        self.scaler = StandardScaler()
        self.classifiers: dict[str, Any] = {}

    def _build_text_features(self, texts: list[str], fit: bool = False) -> np.ndarray:
        if self.text_method == "tfidf":
            if fit:
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words="english",
                    min_df=2,
                    sublinear_tf=True,
                )
                return self.vectorizer.fit_transform(texts).toarray()
            return self.vectorizer.transform(texts).toarray()
        else:
            raise ValueError(
                "For transformer embeddings, pass pre-extracted features "
                "via fit_with_embeddings()"
            )

    def _make_classifier(self):
        if self.classifier_name == "logistic":
            return LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
        elif self.classifier_name == "rf":
            return RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
                n_jobs=-1, class_weight="balanced",
            )
        else:
            raise ValueError("Unknown classifier: %s" % self.classifier_name)

    def fit(
        self,
        texts: list[str],
        sensing_features: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> None:
        """Fit with TF-IDF text features + sensing features."""
        text_feats = self._build_text_features(texts, fit=True)
        sensing_scaled = self.scaler.fit_transform(
            np.nan_to_num(sensing_features, nan=0.0)
        )

        X = np.concatenate([text_feats, sensing_scaled], axis=1)

        for target_name, y in targets.items():
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                continue
            y_valid = y[mask].astype(int)
            if len(set(y_valid)) < 2:
                continue

            clf = self._make_classifier()
            clf.fit(X[mask], y_valid)
            self.classifiers[target_name] = clf

    def predict(
        self,
        texts: list[str],
        sensing_features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        text_feats = self._build_text_features(texts, fit=False)
        sensing_scaled = self.scaler.transform(
            np.nan_to_num(sensing_features, nan=0.0)
        )
        X = np.concatenate([text_feats, sensing_scaled], axis=1)

        return {
            target: clf.predict(X)
            for target, clf in self.classifiers.items()
        }

    def fit_with_embeddings(
        self,
        text_embeddings: np.ndarray,
        sensing_features: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> None:
        """Fit with pre-extracted transformer embeddings + sensing features."""
        sensing_scaled = self.scaler.fit_transform(
            np.nan_to_num(sensing_features, nan=0.0)
        )
        X = np.concatenate([text_embeddings, sensing_scaled], axis=1)

        for target_name, y in targets.items():
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                continue
            y_valid = y[mask].astype(int)
            if len(set(y_valid)) < 2:
                continue

            clf = self._make_classifier()
            clf.fit(X[mask], y_valid)
            self.classifiers[target_name] = clf

    def predict_with_embeddings(
        self,
        text_embeddings: np.ndarray,
        sensing_features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        sensing_scaled = self.scaler.transform(
            np.nan_to_num(sensing_features, nan=0.0)
        )
        X = np.concatenate([text_embeddings, sensing_scaled], axis=1)

        return {
            target: clf.predict(X)
            for target, clf in self.classifiers.items()
        }


# ---------------------------------------------------------------------------
# Deep Fusion (PyTorch-based) -- only defined when torch is available
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CombinedDataset(Dataset):
    """Dataset combining text token IDs and sensing features."""

    def __init__(
        self,
        texts: list[str],
        sensing: np.ndarray,
        targets: dict[str, np.ndarray],
        tokenizer,
        max_length: int = 64,
    ):
        self.texts = texts
        self.sensing = torch.tensor(sensing, dtype=torch.float32)
        self.target_names = list(targets.keys())
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = {}
        self.masks = {}
        for t, y in targets.items():
            labels_arr = np.zeros(len(y), dtype=np.int64)
            masks_arr = np.zeros(len(y), dtype=np.float32)
            for i in range(len(y)):
                if not np.isnan(y[i]):
                    labels_arr[i] = int(y[i])
                    masks_arr[i] = 1.0
            self.labels[t] = torch.tensor(labels_arr, dtype=torch.long)
            self.masks[t] = torch.tensor(masks_arr, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["sensing"] = self.sensing[idx]
        item["labels"] = {t: self.labels[t][idx] for t in self.target_names}
        item["masks"] = {t: self.masks[t][idx] for t in self.target_names}
        return item


class DeepFusionModel(nn.Module):
    """Text encoder + sensing MLP -> concat -> multi-head classifier."""

    def __init__(
        self,
        text_model_name: str,
        sensing_dim: int,
        target_names: list[str],
        sensing_hidden: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
        freeze_text: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        self.target_names = target_names
        self.freeze_text = freeze_text

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.sensing_mlp = nn.Sequential(
            nn.Linear(sensing_dim, sensing_hidden),
            nn.BatchNorm1d(sensing_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fusion_dim = text_dim + sensing_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict({
            t: nn.Linear(fusion_hidden, 2) for t in target_names
        })

    def forward(self, sensing, labels=None, masks=None, **text_kwargs):
        # Text encoding
        with torch.set_grad_enabled(not self.freeze_text):
            text_out = self.text_encoder(**text_kwargs)

        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            text_repr = text_out.pooler_output
        else:
            attn = text_kwargs.get("attention_mask", None)
            if attn is not None:
                mask_exp = attn.unsqueeze(-1).float()
                text_repr = (
                    (text_out.last_hidden_state * mask_exp).sum(1)
                    / mask_exp.sum(1).clamp(min=1e-9)
                )
            else:
                text_repr = text_out.last_hidden_state.mean(dim=1)

        # Sensing encoding
        sensing_repr = self.sensing_mlp(sensing)

        # Fusion
        fused = torch.cat([text_repr, sensing_repr], dim=1)
        fused = self.fusion(fused)

        logits = {t: head(fused) for t, head in self.heads.items()}

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


class DeepFusionBaseline:
    """Deep multimodal fusion: text encoder + sensing MLP -> classifier."""

    def __init__(
        self,
        text_model: str = "bert-base-uncased",
        freeze_text: bool = True,
        max_length: int = 64,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 1e-3,
    ):
        self.text_model = text_model
        self.freeze_text = freeze_text
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = _get_device()

        self.scaler = StandardScaler()
        self.tokenizer = None
        self.model = None

    def fit(
        self,
        texts: list[str],
        sensing_features: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> None:
        from transformers import AutoTokenizer

        sensing_scaled = self.scaler.fit_transform(
            np.nan_to_num(sensing_features, nan=0.0)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model)
        self.model = DeepFusionModel(
            text_model_name=self.text_model,
            sensing_dim=sensing_scaled.shape[1],
            target_names=list(targets.keys()),
            freeze_text=self.freeze_text,
        ).to(self.device)

        dataset = CombinedDataset(
            texts, sensing_scaled, targets, self.tokenizer, self.max_length
        )
        loader = TorchDataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=0.01,
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                optimizer.zero_grad()

                text_kwargs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                if "token_type_ids" in batch:
                    text_kwargs["token_type_ids"] = batch["token_type_ids"].to(
                        self.device
                    )

                sensing = batch["sensing"].to(self.device)
                labels = {t: batch["labels"][t].to(self.device) for t in targets}
                masks = {t: batch["masks"][t].to(self.device) for t in targets}

                outputs = self.model(
                    sensing=sensing, labels=labels, masks=masks, **text_kwargs
                )
                loss = outputs["loss"]

                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "  Deep Fusion Epoch %d/%d, loss=%.4f",
                    epoch + 1, self.epochs, avg_loss,
                )

    @torch.no_grad()
    def predict(
        self,
        texts: list[str],
        sensing_features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        self.model.eval()
        sensing_scaled = self.scaler.transform(
            np.nan_to_num(sensing_features, nan=0.0)
        )

        dummy_targets = {t: np.zeros(len(texts)) for t in self.model.target_names}
        dataset = CombinedDataset(
            texts, sensing_scaled, dummy_targets, self.tokenizer, self.max_length
        )
        loader = TorchDataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        all_preds: dict[str, list] = {t: [] for t in self.model.target_names}

        for batch in loader:
            text_kwargs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            if "token_type_ids" in batch:
                text_kwargs["token_type_ids"] = batch["token_type_ids"].to(
                    self.device
                )

            sensing = batch["sensing"].to(self.device)
            outputs = self.model(sensing=sensing, **text_kwargs)

            for t in self.model.target_names:
                preds = outputs["logits"][t].argmax(dim=-1).cpu().numpy()
                all_preds[t].append(preds)

        return {t: np.concatenate(v) for t, v in all_preds.items()}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_combined_baselines(
    splits_dir,
    sensing_dfs: dict[str, pd.DataFrame],
    feature_builder,
    output_dir,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Run combined (text + sensing) baselines across 5-fold CV.

    Args:
        splits_dir: Path to group_{1-5}_{train,test}.csv.
        sensing_dfs: Pre-loaded sensing DataFrames.
        feature_builder: Module with build_daily_features/impute_features.
        output_dir: Where to save results.
        methods: Which methods to run. Default: all available.

    Returns:
        Aggregated results dict.
    """
    import json

    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["tfidf+logistic", "tfidf+rf", "deep_fusion"]

    all_results: dict[str, dict[str, list]] = {}

    for fold in range(1, 6):
        logger.info("=== Combined Baselines Fold %d/5 ===", fold)

        train_df = pd.read_csv(splits_dir / f"group_{fold}_train.csv")
        test_df = pd.read_csv(splits_dir / f"group_{fold}_test.csv")

        # Filter non-empty emotion_driver
        train_df = train_df[
            train_df["emotion_driver"].notna()
            & (train_df["emotion_driver"].str.strip() != "")
        ].reset_index(drop=True)
        test_df = test_df[
            test_df["emotion_driver"].notna()
            & (test_df["emotion_driver"].str.strip() != "")
        ].reset_index(drop=True)

        train_texts = train_df["emotion_driver"].tolist()
        test_texts = test_df["emotion_driver"].tolist()

        train_targets = _parse_targets(train_df)
        test_targets = _parse_targets(test_df)

        # Build sensing features
        train_df_sensing = train_df.copy()
        test_df_sensing = test_df.copy()
        train_df_sensing["date_local"] = pd.to_datetime(
            train_df_sensing["date_local"]
        ).dt.date
        test_df_sensing["date_local"] = pd.to_datetime(
            test_df_sensing["date_local"]
        ).dt.date

        X_train_s, _, _ = feature_builder.build_daily_features(
            train_df_sensing, sensing_dfs
        )
        X_test_s, _, _ = feature_builder.build_daily_features(
            test_df_sensing, sensing_dfs
        )
        X_train_s = feature_builder.impute_features(X_train_s).values
        X_test_s = feature_builder.impute_features(X_test_s).values

        for method in methods:
            logger.info("  Running %s...", method)

            try:
                if method == "tfidf+logistic":
                    model = LateFusionBaseline(
                        text_method="tfidf", classifier="logistic"
                    )
                    model.fit(train_texts, X_train_s, train_targets)
                    preds = model.predict(test_texts, X_test_s)

                elif method == "tfidf+rf":
                    model = LateFusionBaseline(
                        text_method="tfidf", classifier="rf"
                    )
                    model.fit(train_texts, X_train_s, train_targets)
                    preds = model.predict(test_texts, X_test_s)

                elif method == "deep_fusion":
                    model = DeepFusionBaseline(
                        text_model="bert-base-uncased",
                        freeze_text=True,
                        epochs=10,
                    )
                    model.fit(train_texts, X_train_s, train_targets)
                    preds = model.predict(test_texts, X_test_s)

                else:
                    logger.warning("Unknown combined method: %s", method)
                    continue

                # Score
                for target_name in MAIN_BINARY_TARGETS:
                    if target_name not in preds:
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

                    all_results.setdefault(method, {}).setdefault(
                        target_name, []
                    ).append({
                        "fold": fold,
                        "balanced_accuracy": ba,
                        "f1": f1,
                        "n_train": len(train_texts),
                        "n_test": int(mask.sum()),
                    })

            except Exception as e:
                logger.error("  %s fold %d failed: %s", method, fold, e)

        # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Aggregate
    aggregated = {}
    for method_name, targets in all_results.items():
        aggregated[method_name] = {}
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
            aggregated[method_name][target] = avg
            all_ba.append(avg["ba_mean"])

        aggregated[method_name]["_aggregate"] = {
            "mean_ba": float(np.mean(all_ba)) if all_ba else None,
        }

    # Save
    with open(output_dir / "combined_baseline_folds.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(output_dir / "combined_baseline_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    lines = ["# Combined Baseline Results (5-fold CV)\n"]
    lines.append(
        "| Method | PosAff BA | NegAff BA | RegDesire BA | IntAvail BA | Mean BA |"
    )
    lines.append(
        "|--------|-----------|-----------|--------------|-------------|---------|"
    )

    for method_name, targets in aggregated.items():
        cols = [method_name]
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
    (output_dir / "combined_baseline_summary.md").write_text("\n".join(lines))
    logger.info("Combined baseline results saved to %s", output_dir)

    return aggregated
