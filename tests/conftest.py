"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.schema import SensingDay, UserProfile
from src.think.llm_client import ClaudeCodeClient
from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Canonical valid prediction JSON (all fields present, in-range)
# ---------------------------------------------------------------------------

VALID_PRED_DICT = {
    "PANAS_Pos": 15.0,
    "PANAS_Neg": 5.0,
    "ER_desire": 3.0,
    "Individual_level_PA_State": False,
    "Individual_level_NA_State": False,
    "Individual_level_happy_State": True,
    "Individual_level_sad_State": False,
    "Individual_level_afraid_State": False,
    "Individual_level_miserable_State": False,
    "Individual_level_worried_State": False,
    "Individual_level_cheerful_State": True,
    "Individual_level_pleased_State": True,
    "Individual_level_grateful_State": False,
    "Individual_level_lonely_State": False,
    "Individual_level_interactions_quality_State": True,
    "Individual_level_pain_State": False,
    "Individual_level_forecasting_State": False,
    "Individual_level_ER_desire_State": False,
    "INT_availability": "yes",
    "reasoning": "Test reasoning text",
    "confidence": 0.75,
}


@pytest.fixture
def valid_pred_dict():
    return VALID_PRED_DICT.copy()


@pytest.fixture
def valid_pred_json():
    return json.dumps(VALID_PRED_DICT)


@pytest.fixture
def full_sensing_day():
    """SensingDay with all major fields populated."""
    sd = SensingDay(id_participant="00071", date=date(2023, 11, 20))
    sd.sleep_duration_min = 420.0
    sd.accel_sleep_duration_min = 400.0
    sd.travel_km = 15.3
    sd.travel_minutes = 25.0
    sd.home_minutes = 800.0
    sd.max_distance_from_home_km = 8.5
    sd.location_variance = 0.0023
    sd.stationary_min = 750.0
    sd.walking_min = 45.0
    sd.automotive_min = 30.0
    sd.running_min = 0.0
    sd.screen_minutes = 180.0
    sd.screen_sessions = 42
    sd.words_typed = 350
    sd.prop_positive = 0.12
    sd.prop_negative = 0.03
    sd.total_app_seconds = 9000.0
    sd.top_apps = [("com.instagram.android", 2400.0), ("com.facebook", 900.0)]
    return sd


@pytest.fixture
def empty_sensing_day():
    """SensingDay with no data fields set."""
    return SensingDay(id_participant="00071", date=date(2023, 11, 20))


@pytest.fixture
def sample_profile():
    return UserProfile(
        study_id=71,
        age=58,
        gender="Female",
        cancer_diagnosis="Breast",
        depression_phq8=13.0,
        anxiety_gad7=8.0,
        trait_positive_affect=22.0,
        trait_negative_affect=15.0,
    )


@pytest.fixture
def dry_run_llm():
    """ClaudeCodeClient in dry-run mode — no subprocess calls."""
    return ClaudeCodeClient(model="sonnet", dry_run=True)


@pytest.fixture
def mock_llm(valid_pred_json):
    """Mock LLM that returns a valid prediction JSON string."""
    client = MagicMock(spec=ClaudeCodeClient)
    client.generate.return_value = valid_pred_json
    client.dry_run = False
    client.call_count = 0
    return client


@pytest.fixture
def sample_ema_row():
    """Minimal pandas Series mimicking one EMA CSV row."""
    return pd.Series({
        "Study_ID": 71,
        "timestamp_local": "2023-11-20 18:00:00",
        "date_local": "2023-11-20",
        "emotion_driver": "Talking with my best friend today.",
        "PANAS_Pos": 18.0,
        "PANAS_Neg": 2.0,
        "ER_desire": 1.0,
        "Individual_level_PA_State": False,
        "Individual_level_NA_State": False,
        "Individual_level_happy_State": True,
        "Individual_level_sad_State": False,
        "Individual_level_afraid_State": False,
        "Individual_level_miserable_State": False,
        "Individual_level_worried_State": False,
        "Individual_level_cheerful_State": True,
        "Individual_level_pleased_State": True,
        "Individual_level_grateful_State": False,
        "Individual_level_lonely_State": False,
        "Individual_level_interactions_quality_State": True,
        "Individual_level_pain_State": False,
        "Individual_level_forecasting_State": False,
        "Individual_level_ER_desire_State": False,
        "INT_availability": "yes",
    })


@pytest.fixture
def sample_ema_row_no_diary():
    """EMA row with no emotion_driver text."""
    return pd.Series({
        "Study_ID": 71,
        "timestamp_local": "2023-11-21 18:00:00",
        "date_local": "2023-11-21",
        "emotion_driver": float("nan"),
        "PANAS_Pos": 12.0,
        "PANAS_Neg": 4.0,
        "ER_desire": 2.0,
        "INT_availability": "no",
    })


@pytest.fixture
def small_train_df():
    """Minimal training DataFrame for retriever fitting."""
    rows = [
        {"Study_ID": 100, "date_local": "2023-01-01", "emotion_driver": "Feeling great today, lots of energy",
         "PANAS_Pos": 25.0, "PANAS_Neg": 2.0, "ER_desire": 0.0, "INT_availability": "no"},
        {"Study_ID": 100, "date_local": "2023-01-02", "emotion_driver": "Very tired, bad sleep, anxious",
         "PANAS_Pos": 8.0, "PANAS_Neg": 18.0, "ER_desire": 7.0, "INT_availability": "yes"},
        {"Study_ID": 101, "date_local": "2023-01-01", "emotion_driver": "Went for a walk, feeling calm",
         "PANAS_Pos": 18.0, "PANAS_Neg": 5.0, "ER_desire": 1.0, "INT_availability": "yes"},
        {"Study_ID": 101, "date_local": "2023-01-02", "emotion_driver": "Family dinner, grateful",
         "PANAS_Pos": 22.0, "PANAS_Neg": 1.0, "ER_desire": 0.0, "INT_availability": "no"},
        {"Study_ID": 102, "date_local": "2023-01-01", "emotion_driver": "Sad news from the doctor today",
         "PANAS_Pos": 5.0, "PANAS_Neg": 22.0, "ER_desire": 9.0, "INT_availability": "yes"},
        {"Study_ID": 102, "date_local": "2023-01-02", "emotion_driver": "Feeling hopeful about treatment",
         "PANAS_Pos": 20.0, "PANAS_Neg": 8.0, "ER_desire": 3.0, "INT_availability": "yes"},
    ]
    # Add binary state columns with False defaults
    for r in rows:
        for t in BINARY_STATE_TARGETS:
            r[t] = False
    return pd.DataFrame(rows)
