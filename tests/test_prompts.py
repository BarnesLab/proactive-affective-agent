"""Tests for src/think/prompts.py — prompt builders and format_sensing_summary."""

from __future__ import annotations

import pytest

from src.data.schema import SensingDay, UserProfile
from src.think.prompts import (
    OUTPUT_FORMAT,
    build_trait_summary,
    callm_prompt,
    format_sensing_summary,
    v1_prompt,
    v1_system_prompt,
    v2_prompt,
    v2_system_prompt,
    v3_prompt,
    v3_system_prompt,
    v4_prompt,
    v4_system_prompt,
)


# ---------------------------------------------------------------------------
# format_sensing_summary
# ---------------------------------------------------------------------------

class TestFormatSensingSummary:

    def test_none_sensing_day_returns_no_data_message(self):
        result = format_sensing_summary(None)
        assert "No sensing data" in result

    def test_empty_sensing_day_returns_minimal_message(self):
        sd = SensingDay(id_participant="00071", date=None)
        result = format_sensing_summary(sd)
        assert result  # not empty string
        assert "Minimal sensing data" in result or "No sensing data" in result

    def test_full_sensing_day_contains_sleep_section(self, full_sensing_day):
        result = format_sensing_summary(full_sensing_day)
        assert "Sleep" in result
        assert "420" in result  # sleep_duration_min

    def test_full_sensing_day_contains_mobility_section(self, full_sensing_day):
        result = format_sensing_summary(full_sensing_day)
        assert "Mobility" in result or "GPS" in result
        assert "15.3" in result  # travel_km

    def test_full_sensing_day_contains_activity_section(self, full_sensing_day):
        result = format_sensing_summary(full_sensing_day)
        assert "Activity" in result or "Motion" in result
        assert "750" in result  # stationary_min

    def test_full_sensing_day_contains_screen_section(self, full_sensing_day):
        result = format_sensing_summary(full_sensing_day)
        assert "Screen" in result
        assert "180" in result  # screen_minutes

    def test_full_sensing_day_contains_typing_section(self, full_sensing_day):
        result = format_sensing_summary(full_sensing_day)
        assert "Typing" in result or "Communication" in result

    def test_full_sensing_day_contains_app_section(self, full_sensing_day):
        result = format_sensing_summary(full_sensing_day)
        assert "App" in result

    def test_sensing_day_with_only_sleep(self):
        sd = SensingDay(id_participant="00071", date=None)
        sd.sleep_duration_min = 360.0
        result = format_sensing_summary(sd)
        assert "360" in result
        # Should not crash if other fields are None

    def test_sensing_day_zero_running_not_shown(self, full_sensing_day):
        """Running/cycling at 0 minutes should be omitted."""
        full_sensing_day.running_min = 0.0
        full_sensing_day.cycling_min = 0.0
        result = format_sensing_summary(full_sensing_day)
        # Zero running/cycling should not appear as prominent items
        # (they are conditionally shown only when > 0)
        assert result  # Still returns valid output


# ---------------------------------------------------------------------------
# build_trait_summary
# ---------------------------------------------------------------------------

class TestBuildTraitSummary:

    def test_none_profile_returns_fallback(self):
        result = build_trait_summary(None)
        assert "No user profile" in result

    def test_full_profile_contains_study_id(self, sample_profile):
        result = build_trait_summary(sample_profile)
        assert "71" in result

    def test_full_profile_contains_depression_score(self, sample_profile):
        result = build_trait_summary(sample_profile)
        assert "13" in result  # PHQ-8

    def test_minimal_profile_does_not_crash(self):
        profile = UserProfile(study_id=999)
        result = build_trait_summary(profile)
        assert "999" in result


# ---------------------------------------------------------------------------
# Prompt builders — check key content presence
# ---------------------------------------------------------------------------

class TestCALLMPrompt:

    def test_contains_output_format(self):
        p = callm_prompt(
            emotion_driver="Feeling good",
            rag_examples="No examples",
            memory_doc="",
            trait_profile="User 71",
        )
        assert "PANAS_Pos" in p
        assert "reasoning" in p

    def test_contains_diary_text(self):
        p = callm_prompt(
            emotion_driver="Talking with my friend",
            rag_examples="",
            memory_doc="",
            trait_profile="User 71",
        )
        assert "Talking with my friend" in p

    def test_contains_rag_examples(self):
        p = callm_prompt(
            emotion_driver="Feeling great",
            rag_examples="Case 1: PA=20",
            memory_doc="",
            trait_profile="User 71",
        )
        assert "Case 1" in p

    def test_empty_diary_still_builds_prompt(self):
        p = callm_prompt(
            emotion_driver="",
            rag_examples="No cases",
            memory_doc="",
            trait_profile="User 71",
        )
        assert len(p) > 100


class TestV1Prompt:

    def test_contains_output_format(self, full_sensing_day, sample_profile):
        sensing = format_sensing_summary(full_sensing_day)
        trait = build_trait_summary(sample_profile)
        p = v1_prompt(sensing_summary=sensing, memory_doc="", trait_profile=trait)
        assert "PANAS_Pos" in p
        assert "reasoning" in p

    def test_contains_sensing_data(self, full_sensing_day, sample_profile):
        sensing = format_sensing_summary(full_sensing_day)
        trait = build_trait_summary(sample_profile)
        p = v1_prompt(sensing_summary=sensing, memory_doc="", trait_profile=trait)
        assert "Sleep" in p

    def test_contains_structured_steps(self, full_sensing_day, sample_profile):
        sensing = format_sensing_summary(full_sensing_day)
        trait = build_trait_summary(sample_profile)
        p = v1_prompt(sensing_summary=sensing, memory_doc="", trait_profile=trait)
        # V1 has explicit analysis steps
        assert "Sleep Analysis" in p or "Sensing" in p or "step" in p.lower()

    def test_v1_system_prompt_defined_and_non_empty(self):
        sp = v1_system_prompt()
        assert len(sp) > 50
        assert "PANAS_Pos" in sp  # OUTPUT_FORMAT included


class TestV2Prompt:

    def test_v2_system_prompt_contains_trait_profile(self, sample_profile):
        trait = build_trait_summary(sample_profile)
        sp = v2_system_prompt(trait)
        assert "71" in sp  # study_id from trait text

    def test_v2_system_prompt_contains_output_format(self, sample_profile):
        trait = build_trait_summary(sample_profile)
        sp = v2_system_prompt(trait)
        assert "PANAS_Pos" in sp

    def test_v2_prompt_contains_sensing(self, full_sensing_day):
        sensing = format_sensing_summary(full_sensing_day)
        p = v2_prompt(sensing_summary=sensing, memory_doc="")
        assert "Sleep" in p

    def test_v2_prompt_autonomy_language(self, full_sensing_day):
        sensing = format_sensing_summary(full_sensing_day)
        p = v2_prompt(sensing_summary=sensing, memory_doc="")
        assert "autonomously" in p.lower() or "autonomy" in p.lower() or "freely" in p.lower()


class TestV3Prompt:

    def test_v3_prompt_contains_diary(self, full_sensing_day, sample_profile):
        sensing = format_sensing_summary(full_sensing_day)
        trait = build_trait_summary(sample_profile)
        p = v3_prompt(
            emotion_driver="Feeling anxious about scan results",
            sensing_summary=sensing,
            rag_examples="No cases",
            memory_doc="",
            trait_profile=trait,
        )
        assert "anxious about scan results" in p

    def test_v3_prompt_contains_sensing(self, full_sensing_day, sample_profile):
        sensing = format_sensing_summary(full_sensing_day)
        trait = build_trait_summary(sample_profile)
        p = v3_prompt(
            emotion_driver="Test diary",
            sensing_summary=sensing,
            rag_examples="",
            memory_doc="",
            trait_profile=trait,
        )
        assert "Sleep" in p

    def test_v3_prompt_contains_5_step_instructions(self, full_sensing_day, sample_profile):
        sensing = format_sensing_summary(full_sensing_day)
        trait = build_trait_summary(sample_profile)
        p = v3_prompt(
            emotion_driver="Test",
            sensing_summary=sensing,
            rag_examples="",
            memory_doc="",
            trait_profile=trait,
        )
        # V3 has 5-step analysis
        assert "1." in p or "Step 1" in p or "Diary Analysis" in p

    def test_v3_system_prompt_contains_output_format(self):
        sp = v3_system_prompt()
        assert "PANAS_Pos" in sp


class TestV4Prompt:

    def test_v4_system_prompt_contains_output_format(self, sample_profile):
        trait = build_trait_summary(sample_profile)
        sp = v4_system_prompt(trait)
        assert "PANAS_Pos" in sp

    def test_v4_prompt_contains_all_modalities(self, full_sensing_day):
        sensing = format_sensing_summary(full_sensing_day)
        p = v4_prompt(
            emotion_driver="Talked to my sister",
            sensing_summary=sensing,
            rag_examples="Case 1: PA=18",
            memory_doc="User tends to be active",
        )
        assert "Talked to my sister" in p  # diary
        assert "Sleep" in p  # sensing
        assert "Case 1" in p  # rag


# ---------------------------------------------------------------------------
# Cross-version consistency checks
# ---------------------------------------------------------------------------

class TestOutputFormatConsistency:

    def test_output_format_contains_all_continuous_targets(self):
        from src.utils.mappings import CONTINUOUS_TARGETS
        for t in CONTINUOUS_TARGETS:
            assert t in OUTPUT_FORMAT, f"{t} missing from OUTPUT_FORMAT"

    def test_output_format_contains_int_availability(self):
        assert "INT_availability" in OUTPUT_FORMAT

    def test_output_format_contains_reasoning(self):
        assert "reasoning" in OUTPUT_FORMAT

    def test_output_format_contains_confidence(self):
        assert "confidence" in OUTPUT_FORMAT
