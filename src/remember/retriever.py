"""RAG retrieval using TF-IDF similarity over EMA text data.

For the pilot study, we use TF-IDF instead of pre-computed OpenAI embeddings
(which can't be used with Claude CLI). The TFIDFRetriever fits on the training
set's emotion_driver texts and retrieves similar cases for CALLM prompts.

MultiModalRetriever extends TFIDFRetriever: searches diary text via TF-IDF
but returns results with attached sensing data (for V3/V4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.mappings import SENSING_COLUMNS, study_id_to_participant_id


class TFIDFRetriever:
    """TF-IDF based retriever for similar EMA cases (used by CALLM baseline)."""

    def __init__(self, max_features: int = 5000) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        self._matrix = None
        self._train_df: pd.DataFrame | None = None
        self._fitted = False

    def fit(self, train_df: pd.DataFrame, text_column: str = "emotion_driver") -> None:
        """Build TF-IDF matrix from training set texts.

        Args:
            train_df: Training DataFrame with text column + target columns.
            text_column: Column name containing the text to vectorize.
        """
        # Filter to rows with non-empty text
        mask = train_df[text_column].notna() & (train_df[text_column].str.strip() != "")
        self._train_df = train_df[mask].reset_index(drop=True)

        texts = self._train_df[text_column].fillna("").tolist()
        try:
            self._matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            # Fallback for small corpora: relax min_df and stop words
            from sklearn.feature_extraction.text import TfidfVectorizer
            fallback = TfidfVectorizer(max_features=self.vectorizer.max_features, min_df=1)
            self._matrix = fallback.fit_transform(texts)
            self.vectorizer = fallback
        self._fitted = True

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Find the top-k most similar training cases to the query text.

        Args:
            query: The emotion_driver text from the current EMA entry.
            top_k: Number of similar cases to return.

        Returns:
            List of dicts with keys: text, similarity, Study_ID, _row_idx,
            date_local, and target columns.
        """
        if not self._fitted or self._train_df is None:
            return []

        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if sims[idx] <= 0:
                break
            row = self._train_df.iloc[idx]
            results.append({
                "text": row.get("emotion_driver", ""),
                "similarity": float(sims[idx]),
                "Study_ID": int(row.get("Study_ID", 0)),
                "_row_idx": int(idx),
                "date_local": row.get("date_local"),
                "PANAS_Pos": row.get("PANAS_Pos"),
                "PANAS_Neg": row.get("PANAS_Neg"),
                "ER_desire": row.get("ER_desire"),
                "INT_availability": row.get("INT_availability"),
                "Individual_level_PA_State": row.get("Individual_level_PA_State"),
                "Individual_level_NA_State": row.get("Individual_level_NA_State"),
                "Individual_level_ER_desire_State": row.get("Individual_level_ER_desire_State"),
            })

        return results

    def format_examples(self, results: list[dict], max_examples: int = 10) -> str:
        """Format retrieval results as text for prompt injection.

        Args:
            results: Output from search().
            max_examples: Max examples to include in formatted text.

        Returns:
            Formatted string of similar cases with their outcomes.
        """
        if not results:
            return "No similar cases found."

        lines = []
        for i, r in enumerate(results[:max_examples], 1):
            lines.append(f"Case {i} (similarity: {r['similarity']:.2f}):")
            lines.append(f"  Diary: {r['text']}")
            outcomes = []
            if r.get("PANAS_Pos") is not None:
                outcomes.append(f"PA={r['PANAS_Pos']:.0f}")
            if r.get("PANAS_Neg") is not None:
                outcomes.append(f"NA={r['PANAS_Neg']:.0f}")
            if r.get("ER_desire") is not None:
                outcomes.append(f"ER_desire={r['ER_desire']:.0f}")
            if r.get("INT_availability") is not None:
                outcomes.append(f"Avail={r['INT_availability']}")
            if outcomes:
                lines.append(f"  Outcomes: {', '.join(outcomes)}")
            lines.append("")

        return "\n".join(lines)


class MultiModalRetriever(TFIDFRetriever):
    """TF-IDF retriever that also attaches sensing data to retrieved cases.

    Used by V3/V4 which need diary + sensing + RAG. Searches diary text via
    TF-IDF (same as CALLM), but returns results enriched with the matched
    participant's sensing data for that date.
    """

    def __init__(self, max_features: int = 5000) -> None:
        super().__init__(max_features=max_features)
        self._sensing_dfs: dict[str, pd.DataFrame] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        text_column: str = "emotion_driver",
        sensing_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Build TF-IDF matrix and store sensing data reference.

        Args:
            train_df: Training DataFrame with text + target columns.
            text_column: Column containing text to vectorize.
            sensing_dfs: Pre-loaded {sensor_name: DataFrame} for sensing lookup.
        """
        super().fit(train_df, text_column)
        if sensing_dfs is not None:
            self._sensing_dfs = sensing_dfs

    def search_with_sensing(
        self, query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search diary text and attach sensing data for each matched case.

        Args:
            query: Diary text to search for similar cases.
            top_k: Number of results.

        Returns:
            List of dicts with diary text, outcomes, and sensing summary.
        """
        results = self.search(query, top_k=top_k)

        for r in results:
            sensing_summary = self._get_sensing_for_result(r)
            r["sensing_summary"] = sensing_summary

        return results

    def _get_sensing_for_result(self, result: dict) -> str:
        """Look up sensing data for a retrieved case by Study_ID + date."""
        if not self._sensing_dfs:
            return ""

        study_id = result.get("Study_ID", 0)
        date_local = result.get("date_local")
        if not study_id or date_local is None:
            return ""

        pid = study_id_to_participant_id(study_id)
        parts = []

        for sensor_name, df in self._sensing_dfs.items():
            info = SENSING_COLUMNS.get(sensor_name)
            if info is None:
                continue
            id_col = info["id_col"]
            date_col = info["date_col"]
            mask = (df[id_col] == pid) & (df[date_col] == date_local)
            matched = df[mask]
            if matched.empty:
                continue

            row = matched.iloc[0]
            for feat in info["features"]:
                val = row.get(feat)
                if val is not None and pd.notna(val):
                    parts.append(f"{feat}={val}")

        return "; ".join(parts) if parts else ""

    def format_examples_with_sensing(
        self, results: list[dict], max_examples: int = 8
    ) -> str:
        """Format retrieval results with both diary text and sensing data.

        Args:
            results: Output from search_with_sensing().
            max_examples: Max examples to include.

        Returns:
            Formatted string with diary + sensing + outcomes per case.
        """
        if not results:
            return "No similar cases found."

        lines = []
        for i, r in enumerate(results[:max_examples], 1):
            lines.append(f"Case {i} (similarity: {r['similarity']:.2f}):")
            lines.append(f"  Diary: {r['text']}")

            sensing = r.get("sensing_summary", "")
            if sensing:
                lines.append(f"  Sensing: {sensing}")

            outcomes = []
            if r.get("PANAS_Pos") is not None:
                outcomes.append(f"PA={r['PANAS_Pos']:.0f}")
            if r.get("PANAS_Neg") is not None:
                outcomes.append(f"NA={r['PANAS_Neg']:.0f}")
            if r.get("ER_desire") is not None:
                outcomes.append(f"ER_desire={r['ER_desire']:.0f}")
            if r.get("INT_availability") is not None:
                outcomes.append(f"Avail={r['INT_availability']}")
            if outcomes:
                lines.append(f"  Outcomes: {', '.join(outcomes)}")
            lines.append("")

        return "\n".join(lines)


# Keep RAGRetriever stub for backward compatibility
class RAGRetriever:
    """FAISS-based retriever (not used in pilot — OpenAI embeddings incompatible with Claude CLI)."""

    def __init__(self, embeddings_dir: Path, documents_dir: Path) -> None:
        self.embeddings_dir = embeddings_dir
        self.documents_dir = documents_dir

    def load(self) -> None:
        raise NotImplementedError("Use TFIDFRetriever for pilot study")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        raise NotImplementedError("Use TFIDFRetriever for pilot study")
