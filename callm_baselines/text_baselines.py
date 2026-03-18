"""Text-only baselines: naive, BoW, TF-IDF, LIWC + LogisticRegression.

Matches CHI paper Table 1 text-only baselines. Uses emotion_driver column
from EMA data with 5-fold participant-grouped CV.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

logger = logging.getLogger(__name__)

# 4 main binary targets for evaluation
MAIN_BINARY_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_ER_desire_State",
    "INT_availability",
]

# LIWC-like lexicon categories (simplified)
LIWC_LEXICON = {
    "posemo": [
        "happy", "joy", "love", "good", "great", "wonderful", "blessed",
        "grateful", "thankful", "amazing", "beautiful", "enjoy", "fun",
        "excited", "glad", "pleased", "cheerful", "pleasant", "hope",
        "awesome", "fantastic", "smile", "laugh", "comfort", "peaceful",
        "calm", "relax", "content", "delight", "satisfy",
    ],
    "negemo": [
        "sad", "angry", "upset", "worried", "anxious", "fear", "pain",
        "hurt", "stress", "tired", "exhausted", "frustrated", "annoyed",
        "depressed", "lonely", "miserable", "awful", "terrible", "horrible",
        "sick", "suffering", "cry", "grief", "sorrow", "nervous",
        "overwhelm", "disappoint", "struggle", "difficult", "hard",
    ],
    "health": [
        "cancer", "treatment", "chemo", "radiation", "doctor", "hospital",
        "pain", "symptom", "medication", "therapy", "health", "medical",
        "surgery", "appointment", "nausea", "fatigue", "scan", "diagnosis",
        "recovery", "survivor",
    ],
    "social": [
        "family", "friend", "husband", "wife", "partner", "child", "kids",
        "son", "daughter", "mother", "father", "sister", "brother",
        "people", "together", "talk", "visit", "call", "support",
        "relationship", "neighbor", "church", "group",
    ],
    "cogproc": [
        "think", "know", "feel", "believe", "understand", "remember",
        "realize", "wonder", "consider", "decide", "expect", "hope",
        "wish", "plan", "try", "learn", "thought", "mind",
    ],
    "work": [
        "work", "job", "office", "meeting", "project", "task", "busy",
        "deadline", "colleague", "boss", "retire", "career",
    ],
    "leisure": [
        "walk", "exercise", "garden", "cook", "read", "watch", "movie",
        "music", "game", "play", "travel", "shop", "hobby", "art",
        "craft", "yoga", "swim", "hike", "bike", "dance",
    ],
}


def _extract_liwc_features(texts: list[str]) -> np.ndarray:
    """Extract LIWC-like lexicon features from texts.

    Features per text:
    - Word count
    - Per-category: count and proportion
    - Total positive / negative ratio
    """
    n_cats = len(LIWC_LEXICON)
    # Features: word_count + per-cat count + per-cat proportion + pos/neg ratio
    n_features = 1 + n_cats * 2 + 1
    features = np.zeros((len(texts), n_features), dtype=np.float32)

    cat_names = list(LIWC_LEXICON.keys())

    for i, text in enumerate(texts):
        words = str(text).lower().split()
        wc = max(len(words), 1)
        features[i, 0] = wc

        for j, cat in enumerate(cat_names):
            lexicon = LIWC_LEXICON[cat]
            count = sum(1 for w in words if any(w.startswith(lex) for lex in lexicon))
            features[i, 1 + j] = count
            features[i, 1 + n_cats + j] = count / wc

        # Positive/negative ratio
        pos_count = features[i, 1 + cat_names.index("posemo")]
        neg_count = features[i, 1 + cat_names.index("negemo")]
        features[i, -1] = (pos_count + 1) / (neg_count + 1)

    return features


def get_liwc_feature_names() -> list[str]:
    """Return feature names for LIWC features."""
    cat_names = list(LIWC_LEXICON.keys())
    names = ["word_count"]
    names += [f"{cat}_count" for cat in cat_names]
    names += [f"{cat}_prop" for cat in cat_names]
    names += ["pos_neg_ratio"]
    return names


class MajorityBaseline:
    """Predict the majority class from training set."""

    def __init__(self):
        self.majority_class: dict[str, int] = {}

    def fit(self, texts: list[str], targets: dict[str, np.ndarray]) -> None:
        for target_name, y in targets.items():
            mask = ~np.isnan(y)
            if mask.sum() > 0:
                vals = y[mask].astype(int)
                self.majority_class[target_name] = int(np.argmax(np.bincount(vals)))

    def predict(self, texts: list[str]) -> dict[str, np.ndarray]:
        n = len(texts)
        return {
            target: np.full(n, cls, dtype=int)
            for target, cls in self.majority_class.items()
        }


class TemporalBaseline:
    """Predict based on hour-of-day majority class."""

    def __init__(self):
        self.hour_majority: dict[str, dict[int, int]] = {}
        self.fallback: dict[str, int] = {}

    def fit(
        self,
        texts: list[str],
        targets: dict[str, np.ndarray],
        hours: np.ndarray,
    ) -> None:
        for target_name, y in targets.items():
            mask = ~np.isnan(y)
            self.hour_majority[target_name] = {}
            y_valid = y[mask].astype(int)
            h_valid = hours[mask]

            # Overall fallback
            if len(y_valid) > 0:
                self.fallback[target_name] = int(np.argmax(np.bincount(y_valid)))
            else:
                self.fallback[target_name] = 0

            # Per-hour majority
            for hour in range(24):
                h_mask = h_valid == hour
                if h_mask.sum() > 0:
                    vals = y_valid[h_mask]
                    self.hour_majority[target_name][hour] = int(
                        np.argmax(np.bincount(vals))
                    )

    def predict(self, texts: list[str], hours: np.ndarray) -> dict[str, np.ndarray]:
        n = len(texts)
        results = {}
        for target_name in self.hour_majority:
            preds = np.zeros(n, dtype=int)
            for i in range(n):
                h = int(hours[i]) if not np.isnan(hours[i]) else -1
                preds[i] = self.hour_majority[target_name].get(
                    h, self.fallback[target_name]
                )
            results[target_name] = preds
        return results


class TextMLBaseline:
    """Text feature extraction + LogisticRegression baseline.

    Supports BoW, TF-IDF, and LIWC feature extraction methods.
    Multi-head: trains a separate LogReg per target.
    """

    def __init__(self, method: str = "tfidf", max_features: int = 5000):
        """
        Args:
            method: "bow", "tfidf", or "liwc".
            max_features: Max vocabulary size for BoW/TF-IDF.
        """
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self.classifiers: dict[str, LogisticRegression] = {}

    def _build_features(self, texts: list[str], fit: bool = False) -> np.ndarray:
        if self.method == "bow":
            if fit:
                self.vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    stop_words="english",
                    min_df=2,
                )
                return self.vectorizer.fit_transform(texts).toarray()
            return self.vectorizer.transform(texts).toarray()

        elif self.method == "tfidf":
            if fit:
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words="english",
                    min_df=2,
                    sublinear_tf=True,
                )
                return self.vectorizer.fit_transform(texts).toarray()
            return self.vectorizer.transform(texts).toarray()

        elif self.method == "liwc":
            return _extract_liwc_features(texts)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, texts: list[str], targets: dict[str, np.ndarray]) -> None:
        X = self._build_features(texts, fit=True)

        for target_name, y in targets.items():
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                logger.warning(f"Skipping {target_name}: only {mask.sum()} valid samples")
                continue
            y_valid = y[mask].astype(int)
            if len(set(y_valid)) < 2:
                logger.warning(f"Skipping {target_name}: only one class")
                continue

            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                C=1.0,
            )
            clf.fit(X[mask], y_valid)
            self.classifiers[target_name] = clf

    def predict(self, texts: list[str]) -> dict[str, np.ndarray]:
        X = self._build_features(texts, fit=False)
        return {
            target: clf.predict(X)
            for target, clf in self.classifiers.items()
        }


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


def _extract_hours(df: pd.DataFrame) -> np.ndarray:
    """Extract hour-of-day from timestamp_local."""
    ts = pd.to_datetime(df["timestamp_local"], errors="coerce")
    return ts.dt.hour.values.astype(float)


def run_text_baselines(
    splits_dir,
    output_dir,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Run all text-only baselines across 5-fold CV.

    Args:
        splits_dir: Path to directory with group_{1-5}_{train,test}.csv.
        output_dir: Where to save results.
        methods: Which methods to run. Default: all.

    Returns:
        Aggregated results dict.
    """
    from pathlib import Path
    import json

    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["majority", "temporal", "bow", "tfidf", "liwc"]

    all_results: dict[str, dict[str, list]] = {}

    for fold in range(1, 6):
        logger.info(f"=== Text Baselines Fold {fold}/5 ===")

        train_df = pd.read_csv(splits_dir / f"group_{fold}_train.csv")
        test_df = pd.read_csv(splits_dir / f"group_{fold}_test.csv")

        # Filter to rows with non-empty emotion_driver
        train_mask = train_df["emotion_driver"].notna() & (
            train_df["emotion_driver"].str.strip() != ""
        )
        test_mask = test_df["emotion_driver"].notna() & (
            test_df["emotion_driver"].str.strip() != ""
        )
        train_df = train_df[train_mask].reset_index(drop=True)
        test_df = test_df[test_mask].reset_index(drop=True)

        train_texts = train_df["emotion_driver"].fillna("").tolist()
        test_texts = test_df["emotion_driver"].fillna("").tolist()

        train_targets = _parse_targets(train_df)
        test_targets = _parse_targets(test_df)

        for method in methods:
            logger.info(f"  Running {method}...")

            if method == "majority":
                model = MajorityBaseline()
                model.fit(train_texts, train_targets)
                preds = model.predict(test_texts)

            elif method == "temporal":
                model = TemporalBaseline()
                train_hours = _extract_hours(train_df)
                test_hours = _extract_hours(test_df)
                model.fit(train_texts, train_targets, train_hours)
                preds = model.predict(test_texts, test_hours)

            elif method in ("bow", "tfidf", "liwc"):
                model = TextMLBaseline(method=method)
                model.fit(train_texts, train_targets)
                preds = model.predict(test_texts)

            else:
                logger.warning(f"Unknown method: {method}")
                continue

            # Evaluate
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

                key = method
                all_results.setdefault(key, {}).setdefault(target_name, []).append({
                    "fold": fold,
                    "balanced_accuracy": ba,
                    "f1": f1,
                    "n_train": len(train_texts),
                    "n_test": int(mask.sum()),
                })

    # Aggregate
    aggregated = _aggregate_text_results(all_results)

    # Save
    _save_text_results(all_results, aggregated, output_dir)

    return aggregated


def _aggregate_text_results(all_results: dict) -> dict[str, Any]:
    """Average metrics across 5 folds."""
    aggregated = {}
    for method, targets in all_results.items():
        aggregated[method] = {}
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
            aggregated[method][target] = avg
            all_ba.append(avg["ba_mean"])

        aggregated[method]["_aggregate"] = {
            "mean_ba": float(np.mean(all_ba)) if all_ba else None,
        }
    return aggregated


def _save_text_results(raw: dict, aggregated: dict, output_dir) -> None:
    """Save text baseline results to JSON and markdown."""
    import json
    from pathlib import Path

    output_dir = Path(output_dir)

    with open(output_dir / "text_baseline_folds.json", "w") as f:
        json.dump(raw, f, indent=2, default=str)

    with open(output_dir / "text_baseline_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    # Readable summary
    lines = ["# Text Baseline Results (5-fold CV)\n"]
    lines.append("| Method | PosAff BA | NegAff BA | RegDesire BA | IntAvail BA | Mean BA |")
    lines.append("|--------|-----------|-----------|--------------|-------------|---------|")

    target_keys = MAIN_BINARY_TARGETS

    for method, targets in aggregated.items():
        cols = [method]
        for t in target_keys:
            if t in targets:
                cols.append(f"{targets[t]['ba_mean']:.3f} +/- {targets[t]['ba_std']:.3f}")
            else:
                cols.append("N/A")
        agg = targets.get("_aggregate", {})
        cols.append(f"{agg.get('mean_ba', 0):.3f}")
        lines.append("| " + " | ".join(cols) + " |")

    lines.append("")
    summary_path = output_dir / "text_baseline_summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info(f"Text baseline results saved to {output_dir}")
