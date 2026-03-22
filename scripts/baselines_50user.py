#!/usr/bin/env python3
"""Evaluate ML baselines on the SAME 50 users that LLM agents were evaluated on.

Re-runs the 5-fold across-subject CV baselines but computes metrics ONLY on
the 50-user evaluation subset.  Each of the 50 users appears in exactly one
test fold, so their predictions are still held-out (no data leakage).

Baselines covered:
  Sensing-only:   RF, XGBoost, Logistic  (Parquet hourly features)
  Text-only:      TF-IDF+LR, BoW+LR     (diary text via TF-IDF / BoW)
  Transformer:    MiniLM+LR              (sentence-transformer embeddings)
  Combined:       Combined RF, Combined Logistic  (Parquet + diary embeddings)

Usage:
    cd /Users/zwang/projects/proactive-affective-agent
    PYTHONPATH=. python3 scripts/baselines_50user.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
HOURLY_DIR = PROJECT_ROOT / "data" / "processed" / "hourly"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "pilot_v2" / "checkpoints"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "pilot_v2" / "baselines_50user.json"

# Binary targets that the paper reports
MAIN_BINARY_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_happy_State",
    "Individual_level_sad_State",
    "Individual_level_worried_State",
    "Individual_level_ER_desire_State",
    "INT_availability",
]


PILOT_50_USERS = [
    24, 25, 40, 41, 43, 60, 61, 71, 75, 82, 83, 86, 89, 95, 98, 99, 103,
    119, 140, 164, 169, 187, 189, 211, 232, 242, 257, 258, 260, 275, 299,
    310, 320, 335, 338, 351, 361, 362, 363, 392, 399, 403, 437, 455, 458,
    464, 499, 503, 505, 513,
]


def get_50_user_ids() -> list[int]:
    """Return the 50 evaluation user IDs (hardcoded for portability)."""
    return PILOT_50_USERS


def coerce_binary(arr) -> np.ndarray:
    """Convert raw target column to float array with NaN for missing."""
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict | None:
    """Compute BA and F1 for binary classification, filtering NaN targets."""
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


# ---------------------------------------------------------------------------
# Sensing-only baselines (RF, XGBoost, Logistic) on Parquet features
# ---------------------------------------------------------------------------

def run_sensing_baselines(
    user_50: list[int],
) -> dict[str, dict]:
    """Run RF, XGBoost, Logistic on Parquet hourly sensing features.

    Returns {model_name: {target: {ba, f1, n}}} aggregated across 5 folds.
    """
    from src.baselines.feature_builder import build_parquet_features, fit_imputer, apply_imputer

    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False

    models_to_run = ["rf", "logistic"]
    if has_xgb:
        models_to_run.append("xgboost")

    # Collect per-entry predictions across folds
    # {model: {target: {"y_true": [], "y_pred": []}}}
    predictions = {m: {t: {"y_true": [], "y_pred": []} for t in MAIN_BINARY_TARGETS} for m in models_to_run}

    for fold in range(1, 6):
        logger.info(f"[Sensing] Fold {fold}/5")
        train_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_train.csv")
        test_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_test.csv")
        train_df["date_local"] = pd.to_datetime(train_df["date_local"]).dt.date
        test_df["date_local"] = pd.to_datetime(test_df["date_local"]).dt.date

        # Which of the 50 users are in this fold's test set?
        test_user_mask = test_df["Study_ID"].isin(user_50)
        n_50_in_fold = test_user_mask.sum()
        if n_50_in_fold == 0:
            logger.info(f"  No 50-user entries in fold {fold} test set, skipping")
            continue
        logger.info(f"  {n_50_in_fold} entries from 50-user subset in fold {fold} test")

        # Build features for ALL train and test (same as full run)
        X_train, _, y_bin_train = build_parquet_features(train_df, HOURLY_DIR)
        X_test, _, y_bin_test = build_parquet_features(test_df, HOURLY_DIR)

        # Impute (fit on train only)
        imputer = fit_imputer(X_train)
        X_train = apply_imputer(X_train, imputer)
        X_test = apply_imputer(X_test, imputer)

        X_train_np = X_train.values
        X_test_np = X_test.values

        # Get the subset mask for evaluation (only 50 users)
        eval_mask = test_user_mask.values

        for target in MAIN_BINARY_TARGETS:
            y_tr = coerce_binary(y_bin_train[target].values)
            y_te = coerce_binary(y_bin_test[target].values)

            mask_tr = ~np.isnan(y_tr)
            if mask_tr.sum() < 10:
                continue
            y_tr_int = y_tr[mask_tr].astype(int)
            if len(set(y_tr_int)) < 2:
                continue

            # Compute scale_pos_weight for XGBoost
            n_pos = int(y_tr_int.sum())
            n_neg = len(y_tr_int) - n_pos
            spw = n_neg / max(n_pos, 1)

            for model_name in models_to_run:
                try:
                    if model_name == "rf":
                        clf = RandomForestClassifier(
                            n_estimators=200, max_depth=None, random_state=42,
                            n_jobs=-1, class_weight="balanced",
                        )
                    elif model_name == "xgboost":
                        clf = XGBClassifier(
                            n_estimators=200, max_depth=6, learning_rate=0.1,
                            random_state=42, verbosity=0, eval_metric="logloss",
                            scale_pos_weight=spw,
                        )
                    elif model_name == "logistic":
                        clf = LogisticRegressionCV(
                            Cs=[0.1, 1.0, 10.0], cv=3,
                            class_weight="balanced", max_iter=1000,
                            scoring="balanced_accuracy", random_state=42,
                        )
                    else:
                        continue

                    # Scale features for logistic
                    if model_name == "logistic":
                        scaler = StandardScaler()
                        X_tr_scaled = scaler.fit_transform(X_train_np[mask_tr])
                        X_te_all = scaler.transform(X_test_np)
                    else:
                        X_tr_scaled = X_train_np[mask_tr]
                        X_te_all = X_test_np

                    clf.fit(X_tr_scaled, y_tr_int)
                    preds_all = clf.predict(X_te_all)

                    # Only store predictions for the 50-user subset
                    y_te_50 = y_te[eval_mask]
                    preds_50 = preds_all[eval_mask]

                    predictions[model_name][target]["y_true"].extend(y_te_50.tolist())
                    predictions[model_name][target]["y_pred"].extend(preds_50.tolist())

                except Exception as e:
                    logger.warning(f"  {model_name}/{target} fold {fold}: {e}")

    # Aggregate
    results = {}
    for model_name in models_to_run:
        results[model_name] = {}
        ba_list, f1_list = [], []
        for target in MAIN_BINARY_TARGETS:
            yt = np.array(predictions[model_name][target]["y_true"])
            yp = np.array(predictions[model_name][target]["y_pred"])
            metrics = compute_metrics(yt, yp)
            if metrics:
                results[model_name][target] = metrics
                ba_list.append(metrics["balanced_accuracy"])
                f1_list.append(metrics["f1"])
        results[model_name]["_aggregate"] = {
            "mean_ba": float(np.mean(ba_list)) if ba_list else None,
            "mean_f1": float(np.mean(f1_list)) if f1_list else None,
            "n_targets": len(ba_list),
        }
    return results


# ---------------------------------------------------------------------------
# Text-only baselines (TF-IDF+LR, BoW+LR) on diary text
# ---------------------------------------------------------------------------

def run_text_baselines(
    user_50: list[int],
) -> dict[str, dict]:
    """Run TF-IDF and BoW baselines on diary text (emotion_driver).

    Returns {model_name: {target: {ba, f1, n}}} aggregated across folds.
    """
    text_models = ["tfidf", "bow"]
    predictions = {m: {t: {"y_true": [], "y_pred": []} for t in MAIN_BINARY_TARGETS} for m in text_models}

    for fold in range(1, 6):
        logger.info(f"[Text] Fold {fold}/5")
        train_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_train.csv")
        test_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_test.csv")

        # Filter to diary-present rows
        train_mask = train_df["emotion_driver"].notna() & (train_df["emotion_driver"].str.strip() != "")
        test_mask = test_df["emotion_driver"].notna() & (test_df["emotion_driver"].str.strip() != "")
        train_df = train_df[train_mask].reset_index(drop=True)
        test_df_diary = test_df[test_mask].reset_index(drop=True)

        if len(train_df) < 10 or len(test_df_diary) < 5:
            continue

        # 50-user evaluation mask (within diary-present test rows)
        eval_mask = test_df_diary["Study_ID"].isin(user_50).values
        n_50 = eval_mask.sum()
        if n_50 == 0:
            continue
        logger.info(f"  {n_50} diary entries from 50-user subset in fold {fold} test")

        train_texts = train_df["emotion_driver"].fillna("").tolist()
        test_texts = test_df_diary["emotion_driver"].fillna("").tolist()

        for model_name in text_models:
            if model_name == "tfidf":
                vectorizer = TfidfVectorizer(
                    max_features=500, ngram_range=(1, 2),
                    min_df=2, sublinear_tf=True,
                )
            else:
                vectorizer = CountVectorizer(
                    max_features=300, ngram_range=(1, 1),
                    min_df=2, binary=False,
                )

            try:
                X_train = vectorizer.fit_transform(train_texts)
                X_test = vectorizer.transform(test_texts)
            except Exception as e:
                logger.warning(f"  {model_name} vectorizer fold {fold}: {e}")
                continue

            for target in MAIN_BINARY_TARGETS:
                y_tr = coerce_binary(train_df[target].values)
                y_te = coerce_binary(test_df_diary[target].values)

                mask_tr = ~np.isnan(y_tr)
                if mask_tr.sum() < 10:
                    continue
                y_tr_int = y_tr[mask_tr].astype(int)
                if len(set(y_tr_int)) < 2:
                    continue

                try:
                    clf = LogisticRegressionCV(
                        Cs=[0.01, 0.1, 1.0, 10.0], cv=3,
                        class_weight="balanced", max_iter=1000,
                        scoring="balanced_accuracy", random_state=42,
                    )
                    clf.fit(X_train[mask_tr], y_tr_int)
                    preds_all = clf.predict(X_test)

                    y_te_50 = y_te[eval_mask]
                    preds_50 = preds_all[eval_mask]

                    predictions[model_name][target]["y_true"].extend(y_te_50.tolist())
                    predictions[model_name][target]["y_pred"].extend(preds_50.tolist())
                except Exception as e:
                    logger.warning(f"  {model_name}/{target} fold {fold}: {e}")

    # Aggregate
    results = {}
    for model_name in text_models:
        results[model_name] = {}
        ba_list, f1_list = [], []
        for target in MAIN_BINARY_TARGETS:
            yt = np.array(predictions[model_name][target]["y_true"])
            yp = np.array(predictions[model_name][target]["y_pred"])
            metrics = compute_metrics(yt, yp)
            if metrics:
                results[model_name][target] = metrics
                ba_list.append(metrics["balanced_accuracy"])
                f1_list.append(metrics["f1"])
        results[model_name]["_aggregate"] = {
            "mean_ba": float(np.mean(ba_list)) if ba_list else None,
            "mean_f1": float(np.mean(f1_list)) if f1_list else None,
            "n_targets": len(ba_list),
        }
    return results


# ---------------------------------------------------------------------------
# Transformer baseline (MiniLM+LR) on diary embeddings
# ---------------------------------------------------------------------------

def run_transformer_baseline(
    user_50: list[int],
) -> dict[str, dict]:
    """Run MiniLM sentence-transformer + LogReg on diary text.

    Returns {model_name: {target: {ba, f1, n}}} aggregated across folds.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model: all-MiniLM-L6-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    predictions = {"minilm": {t: {"y_true": [], "y_pred": []} for t in MAIN_BINARY_TARGETS}}

    for fold in range(1, 6):
        logger.info(f"[MiniLM] Fold {fold}/5")
        train_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_train.csv")
        test_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_test.csv")

        # Filter diary-present rows
        train_mask = train_df["emotion_driver"].notna() & (train_df["emotion_driver"].str.strip() != "")
        test_mask = test_df["emotion_driver"].notna() & (test_df["emotion_driver"].str.strip() != "")
        train_df = train_df[train_mask].reset_index(drop=True)
        test_df_diary = test_df[test_mask].reset_index(drop=True)

        if len(train_df) < 10 or len(test_df_diary) < 5:
            continue

        eval_mask = test_df_diary["Study_ID"].isin(user_50).values
        n_50 = eval_mask.sum()
        if n_50 == 0:
            continue
        logger.info(f"  {n_50} diary entries from 50-user subset in fold {fold} test")

        # Embed texts
        logger.info(f"  Embedding {len(train_df)} train + {len(test_df_diary)} test texts")
        X_train = embedder.encode(
            train_df["emotion_driver"].tolist(),
            batch_size=64, show_progress_bar=False, convert_to_numpy=True,
        ).astype(np.float32)
        X_test = embedder.encode(
            test_df_diary["emotion_driver"].tolist(),
            batch_size=64, show_progress_bar=False, convert_to_numpy=True,
        ).astype(np.float32)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for target in MAIN_BINARY_TARGETS:
            y_tr = coerce_binary(train_df[target].values)
            y_te = coerce_binary(test_df_diary[target].values)

            mask_tr = ~np.isnan(y_tr)
            if mask_tr.sum() < 10:
                continue
            y_tr_int = y_tr[mask_tr].astype(int)
            if len(set(y_tr_int)) < 2:
                continue

            try:
                clf = LogisticRegressionCV(
                    Cs=[0.01, 0.1, 1.0, 10.0], cv=3,
                    class_weight="balanced", max_iter=1000,
                    scoring="balanced_accuracy", random_state=42,
                )
                clf.fit(X_train[mask_tr], y_tr_int)
                preds_all = clf.predict(X_test)

                y_te_50 = y_te[eval_mask]
                preds_50 = preds_all[eval_mask]

                predictions["minilm"][target]["y_true"].extend(y_te_50.tolist())
                predictions["minilm"][target]["y_pred"].extend(preds_50.tolist())
            except Exception as e:
                logger.warning(f"  minilm/{target} fold {fold}: {e}")

    # Aggregate
    results = {}
    for model_name in ["minilm"]:
        results[model_name] = {}
        ba_list, f1_list = [], []
        for target in MAIN_BINARY_TARGETS:
            yt = np.array(predictions[model_name][target]["y_true"])
            yp = np.array(predictions[model_name][target]["y_pred"])
            metrics = compute_metrics(yt, yp)
            if metrics:
                results[model_name][target] = metrics
                ba_list.append(metrics["balanced_accuracy"])
                f1_list.append(metrics["f1"])
        results[model_name]["_aggregate"] = {
            "mean_ba": float(np.mean(ba_list)) if ba_list else None,
            "mean_f1": float(np.mean(f1_list)) if f1_list else None,
            "n_targets": len(ba_list),
        }
    return results


# ---------------------------------------------------------------------------
# Combined baselines (Parquet + diary embeddings, late fusion)
# ---------------------------------------------------------------------------

def run_combined_baselines(
    user_50: list[int],
) -> dict[str, dict]:
    """Run Combined RF and Combined Logistic (Parquet + MiniLM embeddings).

    Uses ALL rows (diary-absent rows get zero embeddings).
    Returns {model_name: {target: {ba, f1, n}}}.
    """
    from src.baselines.feature_builder import build_parquet_features, fit_imputer, apply_imputer
    from sentence_transformers import SentenceTransformer

    EMBED_DIM = 384
    logger.info("Loading sentence-transformer model for combined baselines")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    combined_models = ["combined_rf", "combined_logistic"]
    predictions = {m: {t: {"y_true": [], "y_pred": []} for t in MAIN_BINARY_TARGETS} for m in combined_models}

    for fold in range(1, 6):
        logger.info(f"[Combined] Fold {fold}/5")
        train_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_train.csv")
        test_df = pd.read_csv(SPLITS_DIR / f"group_{fold}_test.csv")
        train_df["date_local"] = pd.to_datetime(train_df["date_local"]).dt.date
        test_df["date_local"] = pd.to_datetime(test_df["date_local"]).dt.date

        eval_mask = test_df["Study_ID"].isin(user_50).values
        n_50 = eval_mask.sum()
        if n_50 == 0:
            continue
        logger.info(f"  {n_50} entries from 50-user subset in fold {fold} test")

        # Build Parquet sensing features
        X_train_pq, _, y_bin_train = build_parquet_features(train_df, HOURLY_DIR)
        X_test_pq, _, y_bin_test = build_parquet_features(test_df, HOURLY_DIR)

        # Impute sensing features
        imputer = fit_imputer(X_train_pq)
        X_train_pq = apply_imputer(X_train_pq, imputer)
        X_test_pq = apply_imputer(X_test_pq, imputer)

        # Scale sensing features
        sensing_scaler = StandardScaler()
        X_train_sensing = sensing_scaler.fit_transform(X_train_pq.values)
        X_test_sensing = sensing_scaler.transform(X_test_pq.values)

        # Embed diary text (zeros for absent entries)
        def embed_texts(df):
            texts = df["emotion_driver"].fillna("").tolist()
            result = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
            valid_idx = [i for i, t in enumerate(texts) if t and str(t).strip()]
            if valid_idx:
                valid_texts = [str(texts[i]).strip() for i in valid_idx]
                embs = embedder.encode(
                    valid_texts, batch_size=64,
                    show_progress_bar=False, convert_to_numpy=True,
                ).astype(np.float32)
                for out_i, orig_i in enumerate(valid_idx):
                    result[orig_i] = embs[out_i]
            return result

        logger.info(f"  Embedding diary texts for combined features")
        X_train_emb = embed_texts(train_df)
        X_test_emb = embed_texts(test_df)

        # Scale embeddings
        emb_scaler = StandardScaler()
        X_train_emb = emb_scaler.fit_transform(X_train_emb)
        X_test_emb = emb_scaler.transform(X_test_emb)

        # Concatenate sensing + embeddings
        X_train_combined = np.concatenate([X_train_sensing, X_train_emb], axis=1)
        X_test_combined = np.concatenate([X_test_sensing, X_test_emb], axis=1)

        for target in MAIN_BINARY_TARGETS:
            y_tr = coerce_binary(y_bin_train[target].values)
            y_te = coerce_binary(y_bin_test[target].values)

            mask_tr = ~np.isnan(y_tr)
            if mask_tr.sum() < 10:
                continue
            y_tr_int = y_tr[mask_tr].astype(int)
            if len(set(y_tr_int)) < 2:
                continue

            for model_name in combined_models:
                try:
                    if model_name == "combined_rf":
                        clf = RandomForestClassifier(
                            n_estimators=100, max_depth=None, random_state=42,
                            n_jobs=-1, class_weight="balanced",
                        )
                    else:
                        clf = LogisticRegressionCV(
                            Cs=[0.01, 0.1, 1.0, 10.0], cv=3,
                            class_weight="balanced", max_iter=1000,
                            scoring="balanced_accuracy", random_state=42,
                        )

                    clf.fit(X_train_combined[mask_tr], y_tr_int)
                    preds_all = clf.predict(X_test_combined)

                    y_te_50 = y_te[eval_mask]
                    preds_50 = preds_all[eval_mask]

                    predictions[model_name][target]["y_true"].extend(y_te_50.tolist())
                    predictions[model_name][target]["y_pred"].extend(preds_50.tolist())
                except Exception as e:
                    logger.warning(f"  {model_name}/{target} fold {fold}: {e}")

    # Aggregate
    results = {}
    for model_name in combined_models:
        results[model_name] = {}
        ba_list, f1_list = [], []
        for target in MAIN_BINARY_TARGETS:
            yt = np.array(predictions[model_name][target]["y_true"])
            yp = np.array(predictions[model_name][target]["y_pred"])
            metrics = compute_metrics(yt, yp)
            if metrics:
                results[model_name][target] = metrics
                ba_list.append(metrics["balanced_accuracy"])
                f1_list.append(metrics["f1"])
        results[model_name]["_aggregate"] = {
            "mean_ba": float(np.mean(ba_list)) if ba_list else None,
            "mean_f1": float(np.mean(f1_list)) if f1_list else None,
            "n_targets": len(ba_list),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    # Get the 50 evaluation users
    user_50 = get_50_user_ids()
    logger.info(f"50-user evaluation set: {len(user_50)} users: {user_50}")

    all_results = {}

    # 1. Sensing-only baselines
    logger.info("=" * 60)
    logger.info("Running SENSING baselines (RF, XGBoost, Logistic)")
    logger.info("=" * 60)
    sensing_results = run_sensing_baselines(user_50)
    all_results.update(sensing_results)

    # 2. Text-only baselines
    logger.info("=" * 60)
    logger.info("Running TEXT baselines (TF-IDF, BoW)")
    logger.info("=" * 60)
    text_results = run_text_baselines(user_50)
    all_results.update(text_results)

    # 3. Transformer baseline
    logger.info("=" * 60)
    logger.info("Running TRANSFORMER baseline (MiniLM)")
    logger.info("=" * 60)
    transformer_results = run_transformer_baseline(user_50)
    all_results.update(transformer_results)

    # 4. Combined baselines
    logger.info("=" * 60)
    logger.info("Running COMBINED baselines (RF, Logistic)")
    logger.info("=" * 60)
    combined_results = run_combined_baselines(user_50)
    all_results.update(combined_results)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # Print summary table
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  ML BASELINES ON 50-USER EVALUATION SET  (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 80}")

    # Header
    short_targets = {
        "Individual_level_PA_State": "PA",
        "Individual_level_NA_State": "NA",
        "Individual_level_happy_State": "happy",
        "Individual_level_sad_State": "sad",
        "Individual_level_worried_State": "worried",
        "Individual_level_ER_desire_State": "ER_desire",
        "INT_availability": "Avail",
    }
    header = f"{'Model':<22}" + "".join(f"  {short_targets[t]:>8}" for t in MAIN_BINARY_TARGETS) + f"  {'Mean BA':>8}  {'Mean F1':>8}"
    print(header)
    print("-" * len(header))

    # Paper name mapping
    display_names = {
        "rf": "RF (Sensing)",
        "xgboost": "XGBoost (Sensing)",
        "logistic": "Logistic (Sensing)",
        "tfidf": "TF-IDF + LR",
        "bow": "BoW + LR",
        "minilm": "MiniLM + LR",
        "combined_rf": "Combined RF",
        "combined_logistic": "Combined Logistic",
    }

    for model_name in ["rf", "xgboost", "logistic", "tfidf", "bow", "minilm", "combined_rf", "combined_logistic"]:
        if model_name not in all_results:
            continue
        row = f"{display_names.get(model_name, model_name):<22}"
        for target in MAIN_BINARY_TARGETS:
            if target in all_results[model_name] and target != "_aggregate":
                ba = all_results[model_name][target]["balanced_accuracy"]
                row += f"  {ba:>8.3f}"
            else:
                row += f"  {'N/A':>8}"
        agg = all_results[model_name].get("_aggregate", {})
        mean_ba = agg.get("mean_ba")
        mean_f1 = agg.get("mean_f1")
        row += f"  {mean_ba:>8.3f}" if mean_ba else f"  {'N/A':>8}"
        row += f"  {mean_f1:>8.3f}" if mean_f1 else f"  {'N/A':>8}"
        print(row)

    # Also print F1 table
    print(f"\n{'F1 per target:'}")
    print(f"{'Model':<22}" + "".join(f"  {short_targets[t]:>8}" for t in MAIN_BINARY_TARGETS))
    print("-" * 90)
    for model_name in ["rf", "xgboost", "logistic", "tfidf", "bow", "minilm", "combined_rf", "combined_logistic"]:
        if model_name not in all_results:
            continue
        row = f"{display_names.get(model_name, model_name):<22}"
        for target in MAIN_BINARY_TARGETS:
            if target in all_results[model_name]:
                f1 = all_results[model_name][target]["f1"]
                row += f"  {f1:>8.3f}"
            else:
                row += f"  {'N/A':>8}"
        print(row)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
