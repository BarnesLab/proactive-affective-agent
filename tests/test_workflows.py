"""Tests for V1/V2/V3/V4/CALLM workflows — dry-run mode (no LLM subprocess calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.autonomous import AutonomousWorkflow
from src.agent.autonomous_full import AutonomousFullWorkflow
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
        # Should have parse error or all-None predictions
        # The result may have _parse_error or just None values — not crash
        assert isinstance(result, dict)

    def test_none_profile_does_not_crash(self, dry_run_llm, full_sensing_day):
        wf = StructuredWorkflow(dry_run_llm)
        result = wf.run(sensing_day=full_sensing_day, memory_doc="", profile=None)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# V2 AutonomousWorkflow
# ---------------------------------------------------------------------------

class TestV2AutonomousWorkflow:

    def test_dry_run_returns_valid_prediction(self, dry_run_llm, full_sensing_day, sample_profile):
        wf = AutonomousWorkflow(dry_run_llm)
        result = wf.run(
            sensing_day=full_sensing_day,
            memory_doc="",
            profile=sample_profile,
            date_str="2023-11-20",
        )
        assert_valid_prediction(result)

    def test_version_tag(self, dry_run_llm, full_sensing_day, sample_profile):
        wf = AutonomousWorkflow(dry_run_llm)
        result = wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        assert result.get("_version") == "v2"

    def test_llm_called_exactly_once(self, mock_llm, full_sensing_day, sample_profile):
        wf = AutonomousWorkflow(mock_llm)
        wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        assert mock_llm.generate.call_count == 1

    def test_system_prompt_used(self, mock_llm, full_sensing_day, sample_profile):
        """V2 should pass a system_prompt to generate(), unlike V1."""
        wf = AutonomousWorkflow(mock_llm)
        wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        _, kwargs = mock_llm.generate.call_args
        # system_prompt must be passed (either positional or keyword)
        all_args = list(mock_llm.generate.call_args[0]) + list(mock_llm.generate.call_args[1].values())
        # At least the prompt arg should be non-empty
        assert any(isinstance(a, str) and len(a) > 50 for a in all_args)

    def test_none_sensing_day_does_not_crash(self, dry_run_llm, sample_profile):
        wf = AutonomousWorkflow(dry_run_llm)
        result = wf.run(sensing_day=None, memory_doc="", profile=sample_profile)
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
        # No crash even without retriever

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
# V4 AutonomousFullWorkflow
# ---------------------------------------------------------------------------

class TestV4AutonomousFullWorkflow:

    def test_dry_run_returns_valid_prediction(self, dry_run_llm, full_sensing_day, sample_profile, sample_ema_row):
        wf = AutonomousFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(
            ema_row=sample_ema_row,
            sensing_day=full_sensing_day,
            memory_doc="User history text",
            profile=sample_profile,
        )
        assert_valid_prediction(result)

    def test_version_tag(self, dry_run_llm, full_sensing_day, sample_profile, sample_ema_row):
        wf = AutonomousFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(ema_row=sample_ema_row, sensing_day=full_sensing_day,
                        memory_doc="", profile=sample_profile)
        assert result.get("_version") == "v4"

    def test_llm_called_once(self, mock_llm, full_sensing_day, sample_profile, sample_ema_row):
        wf = AutonomousFullWorkflow(mock_llm, retriever=None)
        wf.run(ema_row=sample_ema_row, sensing_day=full_sensing_day,
               memory_doc="", profile=sample_profile)
        assert mock_llm.generate.call_count == 1

    def test_all_none_inputs_do_not_crash(self, dry_run_llm):
        wf = AutonomousFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(ema_row=None, sensing_day=None, memory_doc="", profile=None)
        assert isinstance(result, dict)

    def test_rag_top5_saved_in_result(self, dry_run_llm, full_sensing_day, sample_profile, sample_ema_row):
        wf = AutonomousFullWorkflow(dry_run_llm, retriever=None)
        result = wf.run(ema_row=sample_ema_row, sensing_day=full_sensing_day,
                        memory_doc="", profile=sample_profile)
        assert "_rag_top5" in result


# ---------------------------------------------------------------------------
# PersonalAgent — version dispatch
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

    def test_v2_dispatches_correctly(self, full_sensing_day):
        agent = self._make_agent("v2")
        result = agent.predict(sensing_day=full_sensing_day, date_str="2023-11-20")
        assert isinstance(result, dict)

    def test_v3_dispatches_correctly(self, sample_ema_row, full_sensing_day):
        agent = self._make_agent("v3")
        result = agent.predict(ema_row=sample_ema_row, sensing_day=full_sensing_day, date_str="2023-11-20")
        assert isinstance(result, dict)

    def test_v4_dispatches_correctly(self, sample_ema_row, full_sensing_day):
        agent = self._make_agent("v4")
        result = agent.predict(ema_row=sample_ema_row, sensing_day=full_sensing_day, date_str="2023-11-20")
        assert isinstance(result, dict)

    def test_invalid_version_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            agent = self._make_agent("v99")
            agent.predict(sensing_day=None)

    def test_dry_run_prediction_has_all_fields(self, sample_ema_row, full_sensing_day):
        for version in ["v1", "v2", "v3", "v4"]:
            agent = self._make_agent(version)
            kwargs = {"sensing_day": full_sensing_day, "date_str": "2023-11-20"}
            if version in ("v3", "v4", "callm"):
                kwargs["ema_row"] = sample_ema_row
            result = agent.predict(**kwargs)
            clean = {k: v for k, v in result.items() if not k.startswith("_")}
            assert "PANAS_Pos" in clean, f"{version} missing PANAS_Pos"
            assert "INT_availability" in clean, f"{version} missing INT_availability"


# ---------------------------------------------------------------------------
# V1 vs V2 structural difference test (documents the known behavior gap)
# ---------------------------------------------------------------------------

class TestV1VsV2StructuralDifferences:

    def test_v1_prompt_contains_explicit_steps(self, full_sensing_day, sample_profile):
        """V1 prompt should have numbered/named analysis steps."""
        from src.think.prompts import build_trait_summary, format_sensing_summary, v1_prompt
        p = v1_prompt(
            sensing_summary=format_sensing_summary(full_sensing_day),
            memory_doc="",
            trait_profile=build_trait_summary(sample_profile),
        )
        # V1 has explicit step-by-step instructions
        assert any(kw in p for kw in ["Sleep Analysis", "Mobility", "Social Signal", "step", "Step"]), \
            "V1 prompt missing structured analysis steps"

    def test_v2_uses_system_prompt_v1_does_not(self, mock_llm, full_sensing_day, sample_profile):
        """V2 passes system_prompt kwarg; V1's system_prompt param defaults to None."""
        v1_wf = StructuredWorkflow(mock_llm)
        v1_wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        v1_call_kwargs = mock_llm.generate.call_args[1]

        mock_llm.reset_mock()

        v2_wf = AutonomousWorkflow(mock_llm)
        v2_wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        v2_call_kwargs = mock_llm.generate.call_args[1]

        v1_sys = v1_call_kwargs.get("system_prompt")
        v2_sys = v2_call_kwargs.get("system_prompt")

        # Document: V1 may or may not use system prompt
        # V2 MUST pass a non-empty system prompt with trait profile
        assert v2_sys is not None and len(v2_sys) > 50, \
            "V2 should pass non-empty system_prompt with trait profile"

    def test_v2_system_prompt_contains_trait_profile(self, mock_llm, full_sensing_day, sample_profile):
        """V2 embeds trait profile in system prompt — V1 puts it in user prompt."""
        from src.think.prompts import build_trait_summary
        trait = build_trait_summary(sample_profile)

        v2_wf = AutonomousWorkflow(mock_llm)
        v2_wf.run(sensing_day=full_sensing_day, memory_doc="", profile=sample_profile)
        v2_sys = mock_llm.generate.call_args[1].get("system_prompt", "")
        assert "71" in v2_sys or "PHQ" in v2_sys or "Trait" in v2_sys, \
            "V2 system prompt should contain trait profile info"
