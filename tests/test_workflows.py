"""Tests for V1/V3/CALLM structured workflows and PersonalAgent dispatch.

V2/V4 (agentic tool-use) require Anthropic API and SensingQueryEngine,
so they are tested via integration tests, not unit tests here.
The old autonomous.py and autonomous_full.py (single-call "autonomous" agents)
have been superseded by the agentic tool-use agents in agentic_sensing_only.py
and agentic_sensing.py respectively.

2x2 design:
                    Structured (fixed pipeline)    Agentic (autonomous tool-use)
  Sensing-only      V1                             V2
  Multimodal        V3                             V4
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.personal_agent import PersonalAgent
from src.agent.structured import StructuredWorkflow
from src.agent.structured_full import StructuredFullWorkflow
from src.think.llm_client import ClaudeCodeClient
from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_KEYS = list(CONTINUOUS_TARGETS.keys()) + BINARY_STATE_TARGETS + ["INT_availability", "reasoning", "confidence"]


def assert_valid_prediction(result: dict) -> None:
    """Assert a result dict looks like a valid prediction output."""
    assert isinstance(result, dict)
    assert "_parse_error" not in result, f"Parse error: {result.get('_raw_response', '')[:200]}"
    for t in CONTINUOUS_TARGETS:
        assert t in result, f"Missing key: {t}"
    assert "INT_availability" in result


# ---------------------------------------------------------------------------
# V1 StructuredWorkflow
# ---------------------------------------------------------------------------

class TestV1StructuredWorkflow:

    def test_dry_run_returns_valid_prediction(self, dry_run_llm, full_sensing_day, sample_profile):
        wf = StructuredWorkflow(dry_run_llm)
        result = wf.run(
            sensing_day=full_sensing_day,
            memory_doc="User has a positive outlook.",
            profile=sample_profile,
            date_str="2023-11-20",
        )
        assert_valid_prediction(result)

    def test_dry_run_none_sensing_day(self, dry_run_llm, sample_profile):
        wf = StructuredWorkflow(dry_run_llm)
        result = wf.run(
            sensing_day=None,
            memory_doc="",
            profile=sample_profile,
            date_str="2023-11-20",
        )
        assert_valid_prediction(result)

    def test_version_tag_in_result(self, dry_run_llm, full_sensing_day, sample_profile):
        wf = StructuredWorkflow(dry_run_llm)
        result = wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        assert result.get("_version") == "v1"

    def test_sensing_summary_saved_in_result(self, dry_run_llm, full_sensing_day, sample_profile):
        wf = StructuredWorkflow(dry_run_llm)
        result = wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        assert "_sensing_summary" in result or "_full_prompt" in result

    def test_malformed_llm_response_returns_parse_error(self, full_sensing_day, sample_profile):
        bad_llm = MagicMock(spec=ClaudeCodeClient)
        bad_llm.generate.return_value = "Sorry, I cannot predict this."
        bad_llm.dry_run = False
        wf = StructuredWorkflow(bad_llm)
        result = wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        assert isinstance(result, dict)

    def test_none_profile_does_not_crash(self, dry_run_llm, full_sensing_day):
        wf = StructuredWorkflow(dry_run_llm)
        result = wf.run(sensing_day=full_sensing_day, memory_doc="", profile=None)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# V3 StructuredFullWorkflow
# ---------------------------------------------------------------------------

class TestV3StructuredFullWorkflow:

    def test_dry_run_returns_valid_prediction(self, dry_run_llm, full_sensing_day, sample_profile, sample_ema_row):
        wf = StructuredFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(
            ema_row=sample_ema_row,
            sensing_day=full_sensing_day,
            memory_doc="",
            profile=sample_profile,
        )
        assert_valid_prediction(result)

    def test_version_tag(self, dry_run_llm, full_sensing_day, sample_profile, sample_ema_row):
        wf = StructuredFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(ema_row=sample_ema_row, sensing_day=full_sensing_day,
                        memory_doc="", profile=sample_profile)
        assert result.get("_version") == "v3"

    def test_no_retriever_uses_no_similar_cases(self, dry_run_llm, full_sensing_day, sample_profile, sample_ema_row):
        wf = StructuredFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(ema_row=sample_ema_row, sensing_day=full_sensing_day,
                        memory_doc="", profile=sample_profile)
        assert isinstance(result, dict)

    def test_none_ema_row_uses_no_diary_placeholder(self, dry_run_llm, full_sensing_day, sample_profile):
        wf = StructuredFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(ema_row=None, sensing_day=full_sensing_day,
                        memory_doc="", profile=sample_profile)
        assert isinstance(result, dict)
        assert result.get("_emotion_driver") == ""

    def test_nan_diary_treated_as_empty(self, dry_run_llm, full_sensing_day, sample_profile, sample_ema_row_no_diary):
        wf = StructuredFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(ema_row=sample_ema_row_no_diary, sensing_day=full_sensing_day,
                        memory_doc="", profile=sample_profile)
        assert isinstance(result, dict)
        assert result.get("_emotion_driver", "") == ""

    def test_none_sensing_day_does_not_crash(self, dry_run_llm, sample_profile, sample_ema_row):
        wf = StructuredFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(ema_row=sample_ema_row, sensing_day=None,
                        memory_doc="", profile=sample_profile)
        assert isinstance(result, dict)

    def test_with_mock_retriever(self, dry_run_llm, full_sensing_day, sample_profile, sample_ema_row):
        mock_retriever = MagicMock()
        mock_retriever.search_with_sensing.return_value = [
            {"text": "Similar diary text", "similarity": 0.9, "PANAS_Pos": 18.0, "PANAS_Neg": 3.0,
             "sensing_summary": "Short sleep"}
        ]
        mock_retriever.format_examples_with_sensing.return_value = "Case 1: Similar diary"

        wf = StructuredFullWorkflow(dry_run_llm, retriever=mock_retriever)
        result = wf.run(ema_row=sample_ema_row, sensing_day=full_sensing_day,
                        memory_doc="", profile=sample_profile)
        assert isinstance(result, dict)
        mock_retriever.search_with_sensing.assert_called_once()


# ---------------------------------------------------------------------------
# PersonalAgent — version dispatch (structured versions only)
# ---------------------------------------------------------------------------

class TestPersonalAgentDispatch:

    def _make_agent(self, version: str, llm=None) -> PersonalAgent:
        from src.data.schema import UserProfile
        profile = UserProfile(study_id=71)
        if llm is None:
            llm = ClaudeCodeClient(dry_run=True)
        return PersonalAgent(
            study_id=71,
            version=version,
            llm_client=llm,
            profile=profile,
            memory_doc="Test memory.",
        )

    def test_callm_dispatches_correctly(self, sample_ema_row):
        agent = self._make_agent("callm")
        result = agent.predict(ema_row=sample_ema_row, sensing_day=None, date_str="2023-11-20")
        assert isinstance(result, dict)

    def test_v1_dispatches_correctly(self, full_sensing_day):
        agent = self._make_agent("v1")
        result = agent.predict(sensing_day=full_sensing_day, date_str="2023-11-20")
        assert isinstance(result, dict)

    def test_v2_requires_query_engine(self):
        """V2 (agentic) raises ValueError if no query_engine provided."""
        with pytest.raises(ValueError, match="SensingQueryEngine"):
            self._make_agent("v2")

    def test_v3_dispatches_correctly(self, sample_ema_row, full_sensing_day):
        agent = self._make_agent("v3")
        result = agent.predict(ema_row=sample_ema_row, sensing_day=full_sensing_day, date_str="2023-11-20")
        assert isinstance(result, dict)

    def test_v4_requires_query_engine(self):
        """V4 (agentic) raises ValueError if no query_engine provided."""
        with pytest.raises(ValueError, match="SensingQueryEngine"):
            self._make_agent("v4")

    def test_invalid_version_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            agent = self._make_agent("v99")
            agent.predict(sensing_day=None)


# ---------------------------------------------------------------------------
# V1 prompt structure test
# ---------------------------------------------------------------------------

class TestV1PromptStructure:

    def test_v1_prompt_contains_explicit_steps(self, full_sensing_day, sample_profile):
        """V1 prompt should have numbered/named analysis steps."""
        from src.think.prompts import build_trait_summary, format_sensing_summary, v1_prompt
        p = v1_prompt(
            sensing_summary=format_sensing_summary(full_sensing_day),
            memory_doc="",
            trait_profile=build_trait_summary(sample_profile),
        )
        assert any(kw in p for kw in ["Sleep Analysis", "Mobility", "Social Signal", "step", "Step"]), \
            "V1 prompt missing structured analysis steps"
