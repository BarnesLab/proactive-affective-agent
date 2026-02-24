"""Tests for src/remember/retriever.py — TFIDFRetriever and MultiModalRetriever."""

from __future__ import annotations

import pytest
import pandas as pd

from src.remember.retriever import TFIDFRetriever


# ---------------------------------------------------------------------------
# TFIDFRetriever
# ---------------------------------------------------------------------------

class TestTFIDFRetriever:

    def test_fit_does_not_crash(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)  # Should not raise
        assert r._fitted

    def test_search_returns_list(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("Feeling great and energetic", top_k=3)
        assert isinstance(results, list)

    def test_search_returns_correct_max_results(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("Feeling tired today", top_k=2)
        assert len(results) <= 2

    def test_search_returns_similarity_scores(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("Feeling great", top_k=3)
        for item in results:
            assert "similarity" in item
            assert 0.0 <= item["similarity"] <= 1.0

    def test_search_results_contain_panas_targets(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("Very tired and anxious today", top_k=3)
        for item in results:
            assert "PANAS_Pos" in item or "text" in item

    def test_search_empty_query_returns_empty(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("", top_k=5)
        assert results == []

    def test_search_whitespace_query_returns_empty(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("   ", top_k=5)
        assert results == []

    def test_search_before_fit_returns_empty(self):
        r = TFIDFRetriever()
        results = r.search("anything", top_k=5)
        assert results == []

    def test_top_result_relevant(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("Feeling great today, lots of energy", top_k=3)
        if results:
            # Top result should have high similarity (same text as training)
            assert results[0]["similarity"] > 0.5

    def test_results_sorted_by_similarity_descending(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("Tired and sad", top_k=5)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True), "Results not sorted by descending similarity"

    def test_format_examples_returns_string(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        results = r.search("tired today", top_k=3)
        formatted = r.format_examples(results, max_examples=2)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_examples_empty_list(self, small_train_df):
        r = TFIDFRetriever()
        r.fit(small_train_df)
        formatted = r.format_examples([], max_examples=5)
        assert isinstance(formatted, str)

    def test_fit_filters_empty_diary_rows(self):
        """Rows with empty emotion_driver should be excluded from the TF-IDF matrix."""
        df = pd.DataFrame([
            {"emotion_driver": "Great day today", "Study_ID": 1, "PANAS_Pos": 20.0, "PANAS_Neg": 2.0},
            {"emotion_driver": "", "Study_ID": 2, "PANAS_Pos": 10.0, "PANAS_Neg": 5.0},
            {"emotion_driver": None, "Study_ID": 3, "PANAS_Pos": 8.0, "PANAS_Neg": 12.0},
            {"emotion_driver": "Bad news today", "Study_ID": 4, "PANAS_Pos": 5.0, "PANAS_Neg": 18.0},
        ])
        r = TFIDFRetriever()
        r.fit(df)
        # Only 2 valid rows should be in the matrix
        assert r._matrix.shape[0] == 2


# ---------------------------------------------------------------------------
# MultiModalRetriever (if available)
# ---------------------------------------------------------------------------

class TestMultiModalRetriever:

    def test_import_succeeds(self):
        from src.remember.retriever import MultiModalRetriever
        assert MultiModalRetriever is not None

    def test_fit_and_search_with_sensing(self, small_train_df):
        from src.remember.retriever import MultiModalRetriever
        r = MultiModalRetriever()
        r.fit(small_train_df, sensing_dfs={})  # No sensing DFs — should not crash
        results = r.search_with_sensing("Feeling great", top_k=3)
        assert isinstance(results, list)

    def test_format_examples_with_sensing_returns_string(self, small_train_df):
        from src.remember.retriever import MultiModalRetriever
        r = MultiModalRetriever()
        r.fit(small_train_df, sensing_dfs={})
        results = r.search_with_sensing("tired today", top_k=2)
        formatted = r.format_examples_with_sensing(results, max_examples=2)
        assert isinstance(formatted, str)
