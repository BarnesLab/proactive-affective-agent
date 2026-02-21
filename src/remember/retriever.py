"""RAG retrieval using TF-IDF similarity over EMA text data.

For the pilot study, we use TF-IDF instead of pre-computed OpenAI embeddings
(which can't be used with Claude CLI). The TFIDFRetriever fits on the training
set's emotion_driver texts and retrieves similar cases for CALLM prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        self._matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Find the top-k most similar training cases to the query text.

        Args:
            query: The emotion_driver text from the current EMA entry.
            top_k: Number of similar cases to return.

        Returns:
            List of dicts with keys: text, similarity, Study_ID, and target columns.
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
