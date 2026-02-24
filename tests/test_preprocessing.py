"""Tests for src/data/preprocessing.py — sensing alignment and profile extraction."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    align_sensing_to_ema,
    get_user_trait_profile,
    prepare_pilot_data,
)
from src.data.schema import SensingDay, UserProfile
from src.utils.mappings import study_id_to_participant_id


# ---------------------------------------------------------------------------
# study_id_to_participant_id
# ---------------------------------------------------------------------------

class TestStudyIdToParticipantId:

    def test_single_digit_zero_padded(self):
        result = study_id_to_participant_id(1)
        assert result == "00001" or result.endswith("001")  # at least some zero-padding

    def test_two_digit_zero_padded(self):
        result = study_id_to_participant_id(71)
        assert "71" in result
        assert result.startswith("0")

    def test_three_digit_zero_padded(self):
        result = study_id_to_participant_id(310)
        assert "310" in result

    def test_returns_string(self):
        assert isinstance(study_id_to_participant_id(100), str)


# ---------------------------------------------------------------------------
# align_sensing_to_ema — using mock DataFrames
# ---------------------------------------------------------------------------

def _make_sensing_dfs(pid: str, target_date: str) -> dict:
    """Build minimal mock sensing DataFrames for one user, one date."""
    td = pd.to_datetime(target_date).date()
    screen = pd.DataFrame([{
        "id_participant": pid,
        "dt_feature": td,
        "n_session_screenon_day": 30,
        "amt_screenon_day_minutes": 120.0,
        "amt_screenon_session_day_max_minutes": 45.0,
        "amt_screenon_session_day_mean_minutes": 4.0,
    }])
    motion = pd.DataFrame([{
        "id_participant": pid,
        "dt_feature": td,
        "amt_stationary_day_min": 600.0,
        "amt_walking_day_min": 60.0,
        "amt_automotive_day_min": 20.0,
        "amt_running_day_min": 0.0,
        "amt_cycling_day_min": 0.0,
        "amt_keyboard_day_min": 10.0,
        "amt_unclassified_day_min": 0.0,
        "amt_unknown_day_min": 0.0,
    }])
    return {"screen": screen, "motion": motion}


class TestAlignSensingToEma:

    def test_basic_alignment_returns_sensing_day(self):
        pid = study_id_to_participant_id(71)
        sensing_dfs = _make_sensing_dfs(pid, "2023-11-20")
        ema_row = pd.Series({"Study_ID": 71, "date_local": "2023-11-20", "timestamp_local": "2023-11-20 18:00:00"})
        result = align_sensing_to_ema(ema_row, sensing_dfs, 71)
        assert result is not None
        assert isinstance(result, SensingDay)

    def test_screen_data_populated(self):
        pid = study_id_to_participant_id(71)
        sensing_dfs = _make_sensing_dfs(pid, "2023-11-20")
        ema_row = pd.Series({"Study_ID": 71, "date_local": "2023-11-20"})
        result = align_sensing_to_ema(ema_row, sensing_dfs, 71)
        assert result is not None
        assert result.screen_minutes == 120.0
        assert result.screen_sessions == 30

    def test_motion_data_populated(self):
        pid = study_id_to_participant_id(71)
        sensing_dfs = _make_sensing_dfs(pid, "2023-11-20")
        ema_row = pd.Series({"Study_ID": 71, "date_local": "2023-11-20"})
        result = align_sensing_to_ema(ema_row, sensing_dfs, 71)
        assert result is not None
        assert result.stationary_min == 600.0
        assert result.walking_min == 60.0

    def test_no_matching_date_returns_none(self):
        pid = study_id_to_participant_id(71)
        sensing_dfs = _make_sensing_dfs(pid, "2023-11-20")
        # Query for a different date
        ema_row = pd.Series({"Study_ID": 71, "date_local": "2023-12-01"})
        result = align_sensing_to_ema(ema_row, sensing_dfs, 71)
        assert result is None

    def test_no_matching_user_returns_none(self):
        pid = study_id_to_participant_id(71)
        sensing_dfs = _make_sensing_dfs(pid, "2023-11-20")
        # Query for a different user
        ema_row = pd.Series({"Study_ID": 999, "date_local": "2023-11-20"})
        result = align_sensing_to_ema(ema_row, sensing_dfs, 999)
        assert result is None

    def test_empty_sensing_dfs_returns_none(self):
        ema_row = pd.Series({"Study_ID": 71, "date_local": "2023-11-20"})
        result = align_sensing_to_ema(ema_row, {}, 71)
        assert result is None

    def test_date_as_date_object_works(self):
        """date_local can be a date object, not just a string."""
        pid = study_id_to_participant_id(71)
        sensing_dfs = _make_sensing_dfs(pid, "2023-11-20")
        ema_row = pd.Series({"Study_ID": 71, "date_local": date(2023, 11, 20)})
        result = align_sensing_to_ema(ema_row, sensing_dfs, 71)
        assert result is not None

    def test_sensing_day_has_correct_participant_id(self):
        pid = study_id_to_participant_id(71)
        sensing_dfs = _make_sensing_dfs(pid, "2023-11-20")
        ema_row = pd.Series({"Study_ID": 71, "date_local": "2023-11-20"})
        result = align_sensing_to_ema(ema_row, sensing_dfs, 71)
        assert result is not None
        assert "71" in result.id_participant


# ---------------------------------------------------------------------------
# get_user_trait_profile
# ---------------------------------------------------------------------------

class TestGetUserTraitProfile:

    def _make_baseline_df(self, study_id: int = 71) -> pd.DataFrame:
        return pd.DataFrame([{
            "Study_ID": study_id,
            "age_demo": 58,
            "gender": 1,
            "cancerdx": "Breast",
            "cancer_stage": "II",
            "cancer_years": 3.5,
            "PHQ8_TOTAL": 13.0,
            "GAD7_TOTAL": 8.0,
            "PANAS_POS": 22.0,
            "PANAS_NEG": 15.0,
            "TIPI_Extraversion": 4.0,
            "TIPI_Stability": 3.0,
            "MSPSS_TOTAL": 60.0,
            "GSE_TOTAL": 28.0,
        }]).set_index("Study_ID")

    def test_returns_user_profile(self):
        df = self._make_baseline_df(71)
        profile = get_user_trait_profile(df, 71)
        assert isinstance(profile, UserProfile)

    def test_study_id_set(self):
        df = self._make_baseline_df(71)
        profile = get_user_trait_profile(df, 71)
        assert profile.study_id == 71

    def test_depression_score_loaded(self):
        df = self._make_baseline_df(71)
        profile = get_user_trait_profile(df, 71)
        assert profile.depression_phq8 == 13.0

    def test_gender_female_mapped(self):
        df = self._make_baseline_df(71)
        profile = get_user_trait_profile(df, 71)
        assert profile.gender == "Female"

    def test_trait_pa_loaded(self):
        df = self._make_baseline_df(71)
        profile = get_user_trait_profile(df, 71)
        assert profile.trait_positive_affect == 22.0

    def test_missing_user_returns_default_profile(self):
        df = self._make_baseline_df(71)
        profile = get_user_trait_profile(df, 999)
        assert isinstance(profile, UserProfile)
        assert profile.study_id == 999
        assert profile.depression_phq8 is None


# ---------------------------------------------------------------------------
# SensingDay.to_summary_dict
# ---------------------------------------------------------------------------

class TestSensingDayToSummaryDict:

    def test_none_fields_excluded(self):
        sd = SensingDay(id_participant="00071", date=date(2023, 1, 1))
        sd.sleep_duration_min = 420.0
        d = sd.to_summary_dict()
        assert "sleep_duration_min" in d
        assert "stationary_min" not in d  # None field excluded

    def test_all_none_returns_empty_dict(self):
        sd = SensingDay(id_participant="00071", date=date(2023, 1, 1))
        d = sd.to_summary_dict()
        assert len(d) == 0

    def test_id_participant_excluded_from_summary(self, full_sensing_day):
        d = full_sensing_day.to_summary_dict()
        assert "id_participant" not in d
        assert "date" not in d

    def test_top_apps_included_when_present(self, full_sensing_day):
        d = full_sensing_day.to_summary_dict()
        assert "top_apps" in d
        assert len(d["top_apps"]) == 2
