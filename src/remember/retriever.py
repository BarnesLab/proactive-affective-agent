"""RAG retriever: FAISS similarity search over pre-computed memory document embeddings.

Uses 754 pre-generated memory documents with FAISS vectors for
semantic similarity search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RAGRetriever:
    """Retrieval-Augmented Generation using FAISS similarity search."""

    def __init__(self, embeddings_dir: Path, documents_dir: Path) -> None:
        self.embeddings_dir = embeddings_dir
        self.documents_dir = documents_dir
        self._index = None
        self._documents: list[str] = []

    def load(self) -> None:
        """Load FAISS index and memory documents."""
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for documents most similar to the query.

        Args:
            query: Natural language query.
            top_k: Number of results to return.

        Returns:
            List of {document, score, metadata} dicts.
        """
        raise NotImplementedError

    def _embed_query(self, query: str) -> Any:
        """Embed a query string into the same vector space as the documents."""
        raise NotImplementedError
