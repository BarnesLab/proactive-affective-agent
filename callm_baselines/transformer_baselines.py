"""Fine-tuned Transformer baselines for text classification.

Supports BERT, SentenceBERT, XLNet, RoBERTa, and emotion-specialized models.
Multi-head architecture: shared backbone -> one classification head per target.
Uses HuggingFace Transformers with 5-fold participant-grouped CV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

# 4 main binary targets
MAIN_BINARY_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_ER_desire_State",
    "INT_availability",
]

# Available pretrained models
TRANSFORMER_MODELS = {
    "bert": "bert-base-uncased",
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "xlnet": "xlnet-base-cased",
    "roberta": "roberta-base",
    "emobert": "monologg/bert-base-cased-goemotions-original",
    "distilbert-emotion": "bhadresh-savani/distilbert-base-uncased-emotion",
    "deberta": "microsoft/deberta-base",
    "roberta-emotion": "SamLowe/roberta-base-go_emotions",
}


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextDataset(Dataset):
    """Simple text + multi-label dataset for PyTorch."""

    def __init__(
        self,
        texts: list[str],
        targets: dict[str, np.ndarray],
        tokenizer,
        max_length: int = 64,
    ):
        self.texts = texts
        self.target_names = list(targets.keys())
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        labels = {}
        masks = {}
        for t in self.target_names:
            val = self.targets[t][idx]
            if np.isnan(val):
                labels[t] = torch.tensor(0, dtype=torch.long)
                masks[t] = torch.tensor(0, dtype=torch.float)
            else:
                labels[t] = torch.tensor(int(val), dtype=torch.long)
                masks[t] = torch.tensor(1, dtype=torch.float)

        item["labels"] = labels
        item["masks"] = masks
        return item


class MultiHeadClassifier(nn.Module):
    """Shared transformer backbone with one binary head per target."""

    def __init__(self, model_name_or_path: str, target_names: list[str], dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.target_names = target_names

        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            t: nn.Linear(hidden_size, 2) for t in target_names
        })

    def forward(self, **kwargs):
        labels = kwargs.pop("labels", None)
        masks = kwargs.pop("masks", None)

        outputs = self.encoder(**kwargs)

        # Use CLS token or mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pooling over non-padding tokens
            token_embeddings = outputs.last_hidden_state
            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                pooled = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            else:
                pooled = token_embeddings.mean(dim=1)

        pooled = self.dropout(pooled)

        logits = {t: head(pooled) for t, head in self.heads.items()}

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


class TransformerBaseline:
    """Fine-tuned transformer for multi-target text classification."""

    def __init__(
        self,
        model_key: str = "bert",
        max_length: int = 64,
        batch_size: int = 32,
        epochs: int = 4,
        lr: float = 2e-5,
        freeze_encoder: bool = False,
    ):
        self.model_key = model_key
        self.model_name = TRANSFORMER_MODELS.get(model_key, model_key)
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.freeze_encoder = freeze_encoder
        self.device = _get_device()

        self.tokenizer = None
        self.model = None

    def fit(self, texts: list[str], targets: dict[str, np.ndarray]) -> None:
        logger.info("Loading tokenizer and model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = MultiHeadClassifier(
            self.model_name, list(targets.keys())
        ).to(self.device)

        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        dataset = TextDataset(texts, targets, self.tokenizer, self.max_length)
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

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = {t: batch["labels"][t].to(self.device) for t in targets}
                masks = {t: batch["masks"][t].to(self.device) for t in targets}

                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "masks": masks,
                }
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**kwargs)
                loss = outputs["loss"]

                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            logger.info("  Epoch %d/%d, loss=%.4f", epoch + 1, self.epochs, avg_loss)

    @torch.no_grad()
    def predict(self, texts: list[str]) -> dict[str, np.ndarray]:
        self.model.eval()

        # Build a dummy targets dict for dataset (values are unused)
        dummy_targets = {
            t: np.zeros(len(texts)) for t in self.model.target_names
        }
        dataset = TextDataset(texts, dummy_targets, self.tokenizer, self.max_length)
        loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        all_preds: dict[str, list] = {t: [] for t in self.model.target_names}

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

            outputs = self.model(**kwargs)
            for t in self.model.target_names:
                preds = outputs["logits"][t].argmax(dim=-1).cpu().numpy()
                all_preds[t].append(preds)

        return {t: np.concatenate(v) for t, v in all_preds.items()}

    @torch.no_grad()
    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Extract frozen embeddings (for combined baselines)."""
        self.model.eval()

        dummy_targets = {
            t: np.zeros(len(texts)) for t in self.model.target_names
        }
        dataset = TextDataset(texts, dummy_targets, self.tokenizer, self.max_length)
        loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        all_embeds = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

            outputs = self.model.encoder(**kwargs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                pooled = (outputs.last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

            all_embeds.append(pooled.cpu().numpy())

        return np.concatenate(all_embeds, axis=0)


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


def run_transformer_baselines(
    splits_dir,
    output_dir,
    model_keys: list[str] | None = None,
    epochs: int = 4,
    batch_size: int = 32,
    max_length: int = 64,
) -> dict[str, Any]:
    """Run transformer baselines across 5-fold CV.

    Args:
        splits_dir: Path to group_{1-5}_{train,test}.csv.
        output_dir: Where to save results.
        model_keys: Which models to run (keys in TRANSFORMER_MODELS).
        epochs: Training epochs per fold.
        batch_size: Batch size for training/inference.
        max_length: Max token length for tokenizer.

    Returns:
        Aggregated results dict.
    """
    import json

    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_keys is None:
        model_keys = list(TRANSFORMER_MODELS.keys())

    all_results: dict[str, dict[str, list]] = {}

    for model_key in model_keys:
        logger.info("")
        logger.info("=" * 50)
        logger.info("Transformer: %s (%s)", model_key, TRANSFORMER_MODELS.get(model_key, model_key))
        logger.info("=" * 50)

        for fold in range(1, 6):
            logger.info("  Fold %d/5", fold)

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

            try:
                model = TransformerBaseline(
                    model_key=model_key,
                    epochs=epochs,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                model.fit(train_texts, train_targets)
                preds = model.predict(test_texts)

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

                    all_results.setdefault(model_key, {}).setdefault(
                        target_name, []
                    ).append({
                        "fold": fold,
                        "balanced_accuracy": ba,
                        "f1": f1,
                        "n_train": len(train_texts),
                        "n_test": int(mask.sum()),
                    })

            except Exception as e:
                logger.error("  %s fold %d failed: %s", model_key, fold, e)
                continue

            # Free memory between folds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del model
            import gc
            gc.collect()

    # Aggregate
    aggregated = _aggregate_transformer_results(all_results)

    # Save
    _save_transformer_results(all_results, aggregated, output_dir)

    return aggregated


def _aggregate_transformer_results(all_results: dict) -> dict[str, Any]:
    """Average metrics across 5 folds."""
    aggregated = {}
    for model_key, targets in all_results.items():
        aggregated[model_key] = {}
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
            aggregated[model_key][target] = avg
            all_ba.append(avg["ba_mean"])

        aggregated[model_key]["_aggregate"] = {
            "mean_ba": float(np.mean(all_ba)) if all_ba else None,
        }
    return aggregated


def _save_transformer_results(raw: dict, aggregated: dict, output_dir) -> None:
    """Save transformer baseline results."""
    import json

    output_dir = Path(output_dir)

    with open(output_dir / "transformer_baseline_folds.json", "w") as f:
        json.dump(raw, f, indent=2, default=str)

    with open(output_dir / "transformer_baseline_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    lines = ["# Transformer Baseline Results (5-fold CV)\n"]
    lines.append("| Model | PosAff BA | NegAff BA | RegDesire BA | IntAvail BA | Mean BA |")
    lines.append("|-------|-----------|-----------|--------------|-------------|---------|")

    for model_key, targets in aggregated.items():
        cols = [model_key]
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
    (output_dir / "transformer_baseline_summary.md").write_text("\n".join(lines))
    logger.info("Transformer baseline results saved to %s", output_dir)
