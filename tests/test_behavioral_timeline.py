"""Tests for behavioral timeline reconstruction and agent wiring."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.agent.cc_agent import (
    SYSTEM_PROMPT_FILTERED_MULTIMODAL,
    SYSTEM_PROMPT_FILTERED_SENSING,
    SYSTEM_PROMPT_MULTIMODAL,
    SYSTEM_PROMPT_SENSING_ONLY,
)
from src.sense.query_tools import SensingQueryEngine


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty and "hour_start" in df.columns:
        df["hour_start"] = pd.to_datetime(df["hour_start"])
    return df


class TestBehavioralTimeline:
    def test_timeline_reconstructs_day_and_infers_segment_states(self):
        engine = SensingQueryEngine(processed_dir=".", ema_df=pd.DataFrame())

        modality_frames = {
            "motion": _df([
                {
                    "hour_start": datetime(2023, 11, 20, 9, 0),
                    "motion_walking_min": 30.0,
                    "motion_stationary_min": 30.0,
                    "motion_automotive_min": 15.0,
                },
                {
                    "hour_start": datetime(2023, 11, 20, 12, 0),
                    "motion_walking_min": 0.0,
                    "motion_stationary_min": 180.0,
                    "motion_automotive_min": 0.0,
                },
            ]),
            "gps": _df([
                {
                    "hour_start": datetime(2023, 11, 20, 9, 0),
                    "gps_distance_km": 3.5,
                    "gps_at_home_min": 30.0,
                },
                {
                    "hour_start": datetime(2023, 11, 20, 12, 0),
                    "gps_distance_km": 0.0,
                    "gps_at_home_min": 180.0,
                },
            ]),
            "screen": _df([
                {
                    "hour_start": datetime(2023, 11, 20, 9, 0),
                    "screen_on_min": 55.0,
                    "screen_n_sessions": 8,
                },
                {
                    "hour_start": datetime(2023, 11, 20, 12, 0),
                    "screen_on_min": 5.0,
                    "screen_n_sessions": 1,
                },
            ]),
            "keyboard": _df([
                {
                    "hour_start": datetime(2023, 11, 20, 9, 0),
                    "key_chars_typed": 320,
                    "key_words_typed": 60,
                    "key_prop_pos": 0.22,
                    "key_prop_neg": 0.03,
                },
                {
                    "hour_start": datetime(2023, 11, 20, 12, 0),
                    "key_chars_typed": 10,
                    "key_words_typed": 2,
                    "key_prop_pos": 0.01,
                    "key_prop_neg": 0.02,
                },
            ]),
            "light": _df([
                {
                    "hour_start": datetime(2023, 11, 20, 12, 0),
                    "light_mean_lux": 20.0,
                },
            ]),
        }

        engine._load_modality_df = lambda study_id, modality: modality_frames.get(modality, pd.DataFrame())

        result = engine.get_behavioral_timeline(
            study_id=71,
            date_str="2023-11-20",
            ema_timestamp="2023-11-20 18:00:00",
            segment_hours=3,
        )

        assert "[Behavioral Timeline: 2023-11-20]" in result
        assert "Coverage window: 00:00-18:00" in result
        assert "09:00-12:00" in result
        assert "12:00-15:00" in result
        assert "physically active" in result
        assert "communication-heavy" in result
        assert "possible engaged / activated state" in result
        assert "possible low-energy / withdrawn state" in result

    def test_call_tool_dispatches_behavioral_timeline(self):
        engine = SensingQueryEngine(processed_dir=".", ema_df=pd.DataFrame())
        engine.get_behavioral_timeline = lambda **kwargs: "timeline ok"

        result = engine.call_tool(
            tool_name="get_behavioral_timeline",
            tool_input={"date": "2023-11-20", "segment_hours": 2},
            study_id=71,
            ema_timestamp="2023-11-20 18:00:00",
        )

        assert result == "timeline ok"


class TestAgenticTimelineWiring:
    def test_all_agentic_system_prompts_reference_behavioral_timeline(self):
        prompts = [
            SYSTEM_PROMPT_SENSING_ONLY,
            SYSTEM_PROMPT_MULTIMODAL,
            SYSTEM_PROMPT_FILTERED_SENSING,
            SYSTEM_PROMPT_FILTERED_MULTIMODAL,
        ]
        assert all("get_behavioral_timeline" in prompt for prompt in prompts)

