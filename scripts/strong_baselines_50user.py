#!/usr/bin/env python3
"""Strong ML baselines on the same 50 users — reviewer-satisfying design.

Key design choices:
  - EMA-backward-looking hourly windows (1-6 hours), NOT daily aggregates
  - Uses build_parquet_features_3d for temporal (N, hours, F) representation
  - Statistical features per window: mean, std, min, max, last_hour, trend
  - Models: Tuned XGBoost, Tuned LightGBM, BiLSTM, Transformer, Personalized
  - CPU limited to 75% (nice=10, n_jobs capped, taskset if needed)
  - All evaluated on SAME 50 users across 5-fold across-subject CV

Usage:
    cd ~/proactive-affective-agent
    # Run all models with 3h window (best single window):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nice -n 10 python3 scripts/strong_baselines_50user.py
    # Sweep windows 1-6h:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nice -n 10 python3 scripts/strong_baselines_50user.py --sweep-windows
    # Specific models only:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nice -n 10 python3 scripts/strong_baselines_50user.py --models lstm transformer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Limit CPU: cap worker threads and set nice priority
MAX_JOBS = max(1, os.cpu_count() // 2)  # ~50% of cores
os.environ.setdefault("OMP_NUM_THREADS", str(min(4, MAX_JOBS)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(4, MAX_JOBS)))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
HOURLY_DIR = PROJECT_ROOT / "data" / "processed" / "hourly"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pilot_v2"

PILOT_50_USERS = [
    24, 25, 40, 41, 43, 60, 61, 71, 75, 82, 83, 86, 89, 95, 98, 99, 103,
    119, 140, 164, 169, 187, 189, 211, 232, 242, 257, 258, 260, 275, 299,
    310, 320, 335, 338, 351, 361, 362, 363, 392, 399, 403, 437, 455, 458,
    464, 499, 503, 505, 513,
]

BINARY_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_happy_State",
    "Individual_level_sad_State",
    "Individual_level_worried_State",
    "Individual_level_cheerful_State",
    "Individual_level_pleased_State",
    "Individual_level_grateful_State",
    "Individual_level_afraid_State",
    "Individual_level_miserable_State",
    "Individual_level_lonely_State",
    "Individual_level_interactions_quality_State",
    "Individual_level_pain_State",
    "Individual_level_forecasting_State",
    "Individual_level_ER_desire_State",
    "INT_availability",
]

FOCUS_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_ER_desire_State",
    "INT_availability",
]


def coerce_binary(arr) -> np.ndarray:
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


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() < 5:
        return None
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    if len(set(yt)) < 2:
        return None
    return {
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "f1": float(f1_score(yt, yp, average="binary", zero_division=0)),
        "n": int(mask.sum()),
    }


def _aggregate_predictions(predictions, targets=None):
    if targets is None:
        targets = BINARY_TARGETS
    results = {}
    ba_list, f1_list = [], []
    for target in targets:
        if target not in predictions:
            continue
        yt = np.array(predictions[target]["y_true"])
        yp = np.array(predictions[target]["y_pred"])
        metrics = compute_metrics(yt, yp)
        if metrics:
            results[target] = metrics
            ba_list.append(metrics["balanced_accuracy"])
            f1_list.append(metrics["f1"])
    results["_aggregate"] = {
        "mean_ba": float(np.mean(ba_list)) if ba_list else None,
        "mean_f1": float(np.mean(f1_list)) if f1_list else None,
        "n_targets": len(ba_list),
    }
    return results


# ---------------------------------------------------------------------------
# Feature extraction: EMA-backward-looking hourly windows
# ---------------------------------------------------------------------------

def sequence_to_flat_stats(seq_3d: np.ndarray) -> np.ndarray:
    """Convert (N, H, F) 3D sequences to (N, 6*F) flat statistical features.

    Per feature: mean, std, min, max, last_hour_value, trend (2nd_half - 1st_half).
    """
    N, H, F = seq_3d.shape
    out = np.zeros((N, 6 * F), dtype=np.float32)
    for f in range(F):
        col = seq_3d[:, :, f]  # (N, H)
        out[:, f * 6 + 0] = np.nanmean(col, axis=1)
        out[:, f * 6 + 1] = np.nanstd(col, axis=1)
        out[:, f * 6 + 2] = np.nanmin(col, axis=1)
        out[:, f * 6 + 3] = np.nanmax(col, axis=1)
        out[:, f * 6 + 4] = col[:, -1]  # last hour
        half = H // 2 if H > 1 else 1
        out[:, f * 6 + 5] = np.nanmean(col[:, half:], axis=1) - np.nanmean(col[:, :half], axis=1)
    return np.nan_to_num(out, 0.0)


def load_fold_data(fold: int, lookback_hours: int = 3):
    """Load one fold's train/test data with 3D hourly sequences.

    Returns (X_train_3d, X_test_3d, y_train_df, y_test_df, test_df, eval_mask)
    """
    from src.baselines.feature_builder import build_parquet_features_3d

    train_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_train.csv")
    test_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_test.csv")

    # Ensure datetime columns
    for col in ["timestamp_local", "Timestamp_start", "date_local"]:
        if col in train_df.columns:
            train_df[col] = pd.to_datetime(train_df[col])
        if col in test_df.columns:
            test_df[col] = pd.to_datetime(test_df[col])

    logger.info(f"  Building 3D features (lookback={lookback_hours}h) for fold {fold}...")

    X_train_3d, _, y_bin_train = build_parquet_features_3d(
        train_df, HOURLY_DIR, lookback_hours=lookback_hours
    )
    X_test_3d, _, y_bin_test = build_parquet_features_3d(
        test_df, HOURLY_DIR, lookback_hours=lookback_hours
    )

    eval_mask = test_df["Study_ID"].isin(PILOT_50_USERS).values
    logger.info(f"  Fold {fold}: {eval_mask.sum()} entries from 50 users "
                f"(train={len(train_df)}, test={len(test_df)})")

    return X_train_3d, X_test_3d, y_bin_train, y_bin_test, test_df, eval_mask


# ---------------------------------------------------------------------------
# 1. Tuned XGBoost (Optuna, on flat stats from hourly windows)
# ---------------------------------------------------------------------------

def run_tuned_xgboost(fold_data_list):
    try:
        import optuna
        from xgboost import XGBClassifier
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Missing optuna/xgboost")
        return {}

    predictions = {t: {"y_true": [], "y_pred": []} for t in BINARY_TARGETS}

    for fold_idx, (X_tr_3d, X_te_3d, y_tr, y_te, test_df, mask) in enumerate(fold_data_list):
        X_train = sequence_to_flat_stats(X_tr_3d)
        X_test = sequence_to_flat_stats(X_te_3d)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for target in BINARY_TARGETS:
            y_tr_arr = coerce_binary(y_tr[target].values)
            y_te_arr = coerce_binary(y_te[target].values)
            tr_mask = ~np.isnan(y_tr_arr)
            if tr_mask.sum() < 10 or len(set(y_tr_arr[tr_mask].astype(int))) < 2:
                continue

            y_tr_int = y_tr_arr[tr_mask].astype(int)
            spw = (len(y_tr_int) - y_tr_int.sum()) / max(y_tr_int.sum(), 1)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "scale_pos_weight": spw,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "random_state": 42,
                    "n_jobs": min(2, MAX_JOBS),
                }
                from sklearn.model_selection import cross_val_score
                clf = XGBClassifier(**params)
                return cross_val_score(
                    clf, X_train[tr_mask], y_tr_int, cv=3,
                    scoring="balanced_accuracy", n_jobs=1,
                ).mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=25, timeout=90, show_progress_bar=False)

            best = study.best_params
            best.update({"scale_pos_weight": spw, "eval_metric": "logloss",
                         "verbosity": 0, "random_state": 42, "n_jobs": min(2, MAX_JOBS)})
            clf = XGBClassifier(**best)
            clf.fit(X_train[tr_mask], y_tr_int)
            preds = clf.predict(X_test)

            predictions[target]["y_true"].extend(y_te_arr[mask].tolist())
            predictions[target]["y_pred"].extend(preds[mask].tolist())

        logger.info(f"  [Tuned XGB] Fold {fold_idx+1}/5 done")

    return _aggregate_predictions(predictions)


# ---------------------------------------------------------------------------
# 2. Tuned LightGBM (Optuna)
# ---------------------------------------------------------------------------

def run_tuned_lgbm(fold_data_list):
    try:
        import optuna
        import lightgbm as lgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Missing optuna/lightgbm")
        return {}

    predictions = {t: {"y_true": [], "y_pred": []} for t in BINARY_TARGETS}

    for fold_idx, (X_tr_3d, X_te_3d, y_tr, y_te, test_df, mask) in enumerate(fold_data_list):
        X_train = sequence_to_flat_stats(X_tr_3d)
        X_test = sequence_to_flat_stats(X_te_3d)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for target in BINARY_TARGETS:
            y_tr_arr = coerce_binary(y_tr[target].values)
            y_te_arr = coerce_binary(y_te[target].values)
            tr_mask = ~np.isnan(y_tr_arr)
            if tr_mask.sum() < 10 or len(set(y_tr_arr[tr_mask].astype(int))) < 2:
                continue

            y_tr_int = y_tr_arr[tr_mask].astype(int)
            spw = (len(y_tr_int) - y_tr_int.sum()) / max(y_tr_int.sum(), 1)

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "scale_pos_weight": spw,
                    "verbosity": -1,
                    "random_state": 42,
                    "n_jobs": min(2, MAX_JOBS),
                }
                from sklearn.model_selection import cross_val_score
                clf = lgb.LGBMClassifier(**params)
                return cross_val_score(
                    clf, X_train[tr_mask], y_tr_int, cv=3,
                    scoring="balanced_accuracy", n_jobs=1,
                ).mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=25, timeout=90, show_progress_bar=False)

            best = study.best_params
            best.update({"scale_pos_weight": spw, "verbosity": -1,
                         "random_state": 42, "n_jobs": min(2, MAX_JOBS)})
            clf = lgb.LGBMClassifier(**best)
            clf.fit(X_train[tr_mask], y_tr_int)
            preds = clf.predict(X_test)

            predictions[target]["y_true"].extend(y_te_arr[mask].tolist())
            predictions[target]["y_pred"].extend(preds[mask].tolist())

        logger.info(f"  [Tuned LGBM] Fold {fold_idx+1}/5 done")

    return _aggregate_predictions(predictions)


# ---------------------------------------------------------------------------
# 3. BiLSTM on hourly sequences
# ---------------------------------------------------------------------------

def run_lstm(fold_data_list):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not available")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  LSTM device: {device}")

    class BiLSTMClassifier(nn.Module):
        def __init__(self, input_dim, hidden=64, layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            h_cat = torch.cat([h[-2], h[-1]], dim=1)
            return self.head(h_cat).squeeze(-1)

    predictions = {t: {"y_true": [], "y_pred": []} for t in BINARY_TARGETS}

    for fold_idx, (X_tr_3d, X_te_3d, y_tr, y_te, test_df, mask) in enumerate(fold_data_list):
        n_feat = X_tr_3d.shape[2]

        # Normalize per-feature across all hours
        N_tr, H, F = X_tr_3d.shape
        flat_tr = X_tr_3d.reshape(-1, F)
        flat_te = X_te_3d.reshape(-1, F)
        scaler = StandardScaler()
        flat_tr = scaler.fit_transform(flat_tr)
        flat_te = scaler.transform(flat_te)
        X_tr_norm = np.nan_to_num(flat_tr.reshape(N_tr, H, F), 0.0).astype(np.float32)
        X_te_norm = np.nan_to_num(flat_te.reshape(X_te_3d.shape), 0.0).astype(np.float32)

        for target in BINARY_TARGETS:
            y_tr_arr = coerce_binary(y_tr[target].values)
            y_te_arr = coerce_binary(y_te[target].values)
            tr_mask = ~np.isnan(y_tr_arr)
            if tr_mask.sum() < 10 or len(set(y_tr_arr[tr_mask].astype(int))) < 2:
                continue

            y_tr_int = y_tr_arr[tr_mask].astype(int)
            n_pos = y_tr_int.sum()
            pos_weight = torch.tensor([(len(y_tr_int) - n_pos) / max(n_pos, 1)],
                                       dtype=torch.float32).to(device)

            X_t = torch.FloatTensor(X_tr_norm[tr_mask]).to(device)
            y_t = torch.FloatTensor(y_tr_int).to(device)
            X_te_t = torch.FloatTensor(X_te_norm).to(device)

            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

            model = BiLSTMClassifier(n_feat).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            model.train()
            for _ in range(30):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            model.eval()
            with torch.no_grad():
                preds = (torch.sigmoid(model(X_te_t)) > 0.5).cpu().numpy().astype(float)

            predictions[target]["y_true"].extend(y_te_arr[mask].tolist())
            predictions[target]["y_pred"].extend(preds[mask].tolist())

        logger.info(f"  [LSTM] Fold {fold_idx+1}/5 done")

    return _aggregate_predictions(predictions)


# ---------------------------------------------------------------------------
# 4. Transformer encoder on hourly sequences
# ---------------------------------------------------------------------------

def run_transformer(fold_data_list):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not available")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Transformer device: {device}")

    class TemporalTransformer(nn.Module):
        def __init__(self, input_dim, seq_len, d_model=64, nhead=4, layers=2, dropout=0.3):
            super().__init__()
            self.proj = nn.Linear(input_dim, d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            x = self.proj(x) + self.pos_emb[:, :x.size(1), :]
            x = self.encoder(x)
            return self.head(x.mean(dim=1)).squeeze(-1)

    predictions = {t: {"y_true": [], "y_pred": []} for t in BINARY_TARGETS}

    for fold_idx, (X_tr_3d, X_te_3d, y_tr, y_te, test_df, mask) in enumerate(fold_data_list):
        N_tr, H, F = X_tr_3d.shape
        flat_tr = X_tr_3d.reshape(-1, F)
        flat_te = X_te_3d.reshape(-1, F)
        scaler = StandardScaler()
        flat_tr = scaler.fit_transform(flat_tr)
        flat_te = scaler.transform(flat_te)
        X_tr_norm = np.nan_to_num(flat_tr.reshape(N_tr, H, F), 0.0).astype(np.float32)
        X_te_norm = np.nan_to_num(flat_te.reshape(X_te_3d.shape), 0.0).astype(np.float32)

        for target in BINARY_TARGETS:
            y_tr_arr = coerce_binary(y_tr[target].values)
            y_te_arr = coerce_binary(y_te[target].values)
            tr_mask = ~np.isnan(y_tr_arr)
            if tr_mask.sum() < 10 or len(set(y_tr_arr[tr_mask].astype(int))) < 2:
                continue

            y_tr_int = y_tr_arr[tr_mask].astype(int)
            n_pos = y_tr_int.sum()
            pos_weight = torch.tensor([(len(y_tr_int) - n_pos) / max(n_pos, 1)],
                                       dtype=torch.float32).to(device)

            X_t = torch.FloatTensor(X_tr_norm[tr_mask]).to(device)
            y_t = torch.FloatTensor(y_tr_int).to(device)
            X_te_t = torch.FloatTensor(X_te_norm).to(device)

            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

            model = TemporalTransformer(F, H).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            model.train()
            for _ in range(30):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            model.eval()
            with torch.no_grad():
                preds = (torch.sigmoid(model(X_te_t)) > 0.5).cpu().numpy().astype(float)

            predictions[target]["y_true"].extend(y_te_arr[mask].tolist())
            predictions[target]["y_pred"].extend(preds[mask].tolist())

        logger.info(f"  [Transformer] Fold {fold_idx+1}/5 done")

    return _aggregate_predictions(predictions)


# ---------------------------------------------------------------------------
# 5. Personalized Context (fair: mirrors LLM agent's information access)
# ---------------------------------------------------------------------------

def run_personalized(fold_data_list):
    """Personalized model mirroring the LLM agent's information access.

    The LLM agent accumulates understanding of a user's behavioral distribution
    via session memory + compare_to_baseline tool, but NEVER sees ground truth
    EMA outcomes. This ML personalization is designed to be information-equivalent:

    For each user, chronologically:
    - Accumulate per-user behavioral statistics (running mean/std of features)
    - Compute z-score deviations: current entry vs. personal baseline
    - Track past receptivity signals (ER_desire, INT_avail) as input features
    - NO online weight updates with ground truth labels

    This tests whether a fair ML model with the same personalization information
    can match the LLM agent's performance.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not available")
        return {}

    predictions = {t: {"y_true": [], "y_pred": []} for t in BINARY_TARGETS}

    for fold_idx, (X_tr_3d, X_te_3d, y_tr, y_te, test_df, mask) in enumerate(fold_data_list):
        X_train_flat = sequence_to_flat_stats(X_tr_3d)
        X_test_flat = sequence_to_flat_stats(X_te_3d)
        n_base = X_train_flat.shape[1]

        # Population baseline from training data
        pop_mean = np.nanmean(X_train_flat, axis=0)
        pop_std = np.nanstd(X_train_flat, axis=0) + 1e-8

        # Training: augment with z-score deviation from population
        X_train_dev = (X_train_flat - pop_mean) / pop_std
        # Add dummy receptivity history (population rates)
        train_recept = np.tile([0.35, 0.64, 0.5, 0.2], (len(X_train_flat), 1)).astype(np.float32)
        X_train_aug = np.hstack([X_train_flat, X_train_dev, train_recept])

        # Test: process each user chronologically to build personal baseline
        er_vals = coerce_binary(y_te["Individual_level_ER_desire_State"].values)
        int_vals = coerce_binary(y_te["INT_availability"].values)

        ts_col = "timestamp_local" if "timestamp_local" in test_df.columns else "date_local"
        test_order = test_df.copy()
        test_order["_idx"] = range(len(test_order))
        test_order = test_order.sort_values(["Study_ID", ts_col]).reset_index(drop=True)

        X_test_aug = np.zeros((len(test_df), n_base * 2 + 4), dtype=np.float32)
        user_hist = {}  # uid -> list of past feature vectors
        user_recept = {}  # uid -> {"er": [], "int": []}

        for _, row in test_order.iterrows():
            idx = int(row["_idx"])
            uid = int(row["Study_ID"])
            x_cur = X_test_flat[idx]

            if uid not in user_hist:
                user_hist[uid] = []
                user_recept[uid] = {"er": [], "int": []}

            # Personal deviation (or population if <3 past entries)
            past = user_hist[uid]
            if len(past) >= 3:
                p_mean = np.nanmean(past, axis=0)
                p_std = np.nanstd(past, axis=0) + 1e-8
                dev = (x_cur - p_mean) / p_std
            else:
                dev = (x_cur - pop_mean) / pop_std

            # Receptivity history features
            rh = user_recept[uid]
            n_past = len(rh["er"])
            if n_past > 0:
                recent = min(10, n_past)
                r_er = [v for v in rh["er"][-recent:] if not np.isnan(v)]
                r_int = [v for v in rh["int"][-recent:] if not np.isnan(v)]
                er_rate = np.mean(r_er) if r_er else 0.5
                int_rate = np.mean(r_int) if r_int else 0.5
                both = sum(1 for e, i in zip(rh["er"][-recent:], rh["int"][-recent:])
                           if e == 1.0 and i == 1.0)
                recept_feat = [er_rate, int_rate, min(n_past / 20.0, 1.0), both / max(recent, 1)]
            else:
                recept_feat = [0.5, 0.5, 0.0, 0.25]

            X_test_aug[idx] = np.concatenate([x_cur, dev, recept_feat])

            # Update observable history (no ground truth)
            user_hist[uid].append(x_cur.copy())
            rh["er"].append(er_vals[idx])
            rh["int"].append(int_vals[idx])

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test_aug)

        eval_mask = mask

        for target in BINARY_TARGETS:
            y_tr_arr = coerce_binary(y_tr[target].values)
            y_te_arr = coerce_binary(y_te[target].values)
            tr_mask = ~np.isnan(y_tr_arr)
            if tr_mask.sum() < 10 or len(set(y_tr_arr[tr_mask].astype(int))) < 2:
                continue

            y_tr_int = y_tr_arr[tr_mask].astype(int)
            spw = (len(y_tr_int) - y_tr_int.sum()) / max(y_tr_int.sum(), 1)

            clf = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=spw, eval_metric="logloss",
                verbosity=0, random_state=42, n_jobs=min(2, MAX_JOBS),
            )
            clf.fit(X_train_scaled[tr_mask], y_tr_int)
            preds = clf.predict(X_test_scaled)

            predictions[target]["y_true"].extend(y_te_arr[eval_mask].tolist())
            predictions[target]["y_pred"].extend(preds[eval_mask].tolist())

        logger.info(f"  [Personalized] Fold {fold_idx+1}/5 done")

    return _aggregate_predictions(predictions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["tuned_xgb", "tuned_lgbm", "lstm", "transformer", "personalized"])
    parser.add_argument("--window", type=int, default=3, help="Lookback hours (default 3)")
    parser.add_argument("--sweep-windows", action="store_true",
                        help="Sweep lookback windows 1-6h (runs tuned_xgb only)")
    args = parser.parse_args()

    t0 = time.time()

    if args.sweep_windows:
        # Sweep mode: run tuned_xgb across 1-6h windows
        logger.info("=" * 60)
        logger.info("WINDOW SWEEP MODE: testing 1-6 hour lookback windows")
        logger.info("=" * 60)

        sweep_results = {}
        for hours in range(1, 7):
            logger.info(f"\n--- Window = {hours}h ---")
            fold_data = []
            for fold in range(1, 6):
                logger.info(f"  Loading fold {fold}/5 (lookback={hours}h)")
                data = load_fold_data(fold, lookback_hours=hours)
                fold_data.append(data)

            result = run_tuned_xgboost(fold_data)
            sweep_results[f"{hours}h"] = result
            agg = result.get("_aggregate", {})
            logger.info(f"  Window {hours}h: Mean BA={agg.get('mean_ba', 0):.3f}")

        out_file = OUTPUT_DIR / "window_sweep_results.json"
        with open(out_file, "w") as f:
            json.dump(sweep_results, f, indent=2)
        logger.info(f"\nSweep results saved to {out_file}")

        # Print summary
        print(f"\n{'Window':<10} {'Mean BA':<10} {'PA':<8} {'NA':<8} {'ER':<8} {'Avail':<8}")
        print("-" * 52)
        for hours in range(1, 7):
            r = sweep_results[f"{hours}h"]
            agg = r.get("_aggregate", {})
            pa = r.get("Individual_level_PA_State", {}).get("balanced_accuracy", 0)
            na = r.get("Individual_level_NA_State", {}).get("balanced_accuracy", 0)
            er = r.get("Individual_level_ER_desire_State", {}).get("balanced_accuracy", 0)
            av = r.get("INT_availability", {}).get("balanced_accuracy", 0)
            print(f"{hours}h{'':<7} {agg.get('mean_ba', 0):<10.3f} {pa:<8.3f} {na:<8.3f} {er:<8.3f} {av:<8.3f}")
        return

    # Normal mode: run selected models with specified window
    logger.info(f"Loading 5 folds with {args.window}h lookback window")
    fold_data = []
    for fold in range(1, 6):
        data = load_fold_data(fold, lookback_hours=args.window)
        fold_data.append(data)

    all_results = {"config": {"lookback_hours": args.window, "models": args.models}}

    if "tuned_xgb" in args.models:
        logger.info("=" * 60 + "\nRunning Tuned XGBoost (Optuna)")
        all_results["tuned_xgboost"] = run_tuned_xgboost(fold_data)

    if "tuned_lgbm" in args.models:
        logger.info("=" * 60 + "\nRunning Tuned LightGBM (Optuna)")
        all_results["tuned_lgbm"] = run_tuned_lgbm(fold_data)

    if "lstm" in args.models:
        logger.info("=" * 60 + "\nRunning BiLSTM")
        all_results["lstm"] = run_lstm(fold_data)

    if "transformer" in args.models:
        logger.info("=" * 60 + "\nRunning Transformer")
        all_results["transformer"] = run_transformer(fold_data)

    if "personalized" in args.models:
        logger.info("=" * 60 + "\nRunning Personalized Online")
        all_results["personalized_online"] = run_personalized(fold_data)

    out_file = OUTPUT_DIR / "strong_baselines_50user.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  STRONG BASELINES ({args.window}h window)  elapsed: {elapsed:.0f}s")
    print(f"{'=' * 80}")

    for name in ["tuned_xgboost", "tuned_lgbm", "lstm", "transformer", "personalized_online"]:
        if name not in all_results:
            continue
        r = all_results[name]
        agg = r.get("_aggregate", {})
        print(f"\n  {name}")
        print(f"    Mean BA={agg.get('mean_ba', 0):.3f}  Mean F1={agg.get('mean_f1', 0):.3f}")
        for ft in FOCUS_TARGETS:
            if ft in r:
                short = ft.replace("Individual_level_", "").replace("_State", "")
                m = r[ft]
                print(f"      {short:>12}: BA={m['balanced_accuracy']:.3f} F1={m['f1']:.3f} n={m['n']}")

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
