"""Equivalence tests for GPT backend implementations.

These tests verify interface/behavioral parity between Claude and GPT paths:
- Same version orchestration shape
- Same tool inventory coverage for agentic runs
- Same prediction schema keys in dry-run execution
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.agent.openai_agent import OPENAI_SENSING_TOOLS
from src.agent.personal_agent import PersonalAgent
from src.simulation.simulator import PilotSimulator
from src.think.openai_client import OpenAIClient
from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS


def _make_min_ema(n_entries: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_entries):
        rows.append(
            {
                "Study_ID": 71,
                "timestamp_local": f"2023-11-2{i} 18:00:00",
                "date_local": f"2023-11-2{i}",
                "emotion_driver": f"Test diary {i}",
                "PANAS_Pos": 15.0 + i,
                "PANAS_Neg": 5.0,
                "ER_desire": 2.0,
                "INT_availability": "yes",
                **{t: False for t in BINARY_STATE_TARGETS},
            }
        )
    return pd.DataFrame(rows)


def _make_min_train() -> pd.DataFrame:
    rows = [
        {
            "Study_ID": 100,
            "emotion_driver": "Feeling good",
            "PANAS_Pos": 20.0,
            "PANAS_Neg": 3.0,
            "ER_desire": 1.0,
            "INT_availability": "yes",
            **{t: False for t in BINARY_STATE_TARGETS},
        },
        {
            "Study_ID": 101,
            "emotion_driver": "Feeling stressed",
            "PANAS_Pos": 8.0,
            "PANAS_Neg": 18.0,
            "ER_desire": 7.0,
            "INT_availability": "no",
            **{t: False for t in BINARY_STATE_TARGETS},
        },
    ]
    return pd.DataFrame(rows)


class TestOpenAIClientDryRun:
    def test_dry_run_prediction_schema_matches_structured_targets(self):
        client = OpenAIClient(dry_run=True)
        raw = client.generate("test prompt")
        from src.think.parser import parse_prediction

        pred = parse_prediction(raw)
        for t in CONTINUOUS_TARGETS:
            assert t in pred
        for t in BINARY_STATE_TARGETS:
            assert t in pred
        assert "INT_availability" in pred
        assert "reasoning" in pred
        assert "confidence" in pred


class TestAgenticToolEquivalence:
    def test_openai_tool_inventory_matches_core_sensing_tools(self):
        tool_names = {t["function"]["name"] for t in OPENAI_SENSING_TOOLS}
        expected = {
            "query_sensing",
            "get_daily_summary",
            "get_behavioral_timeline",
            "compare_to_baseline",
            "get_receptivity_history",
            "find_similar_days",
            "query_raw_events",
            "find_peer_cases",
        }
        assert expected.issubset(tool_names)


class TestGPTAgentDispatch:
    def test_personal_agent_gpt_v2_dry_run_returns_full_prediction(self, sample_profile, sample_ema_row, tmp_path):
        llm = OpenAIClient(dry_run=True)
        agent = PersonalAgent(
            study_id=71,
            version="gpt-v2",
            llm_client=llm,
            profile=sample_profile,
            memory_doc="memory",
            processed_dir=tmp_path,
            ema_df=pd.DataFrame([sample_ema_row]),
        )
        pred = agent.predict(ema_row=sample_ema_row, date_str="2023-11-20")
        for t in CONTINUOUS_TARGETS:
            assert t in pred
        for t in BINARY_STATE_TARGETS:
            assert t in pred
        assert pred.get("_version") == "v2"
        assert "_tool_calls" in pred

    def test_personal_agent_gpt_v4_dry_run_multimodal_tags_diary(self, sample_profile, sample_ema_row, tmp_path):
        llm = OpenAIClient(dry_run=True)
        agent = PersonalAgent(
            study_id=71,
            version="gpt-v4",
            llm_client=llm,
            profile=sample_profile,
            memory_doc="memory",
            processed_dir=tmp_path,
            ema_df=pd.DataFrame([sample_ema_row]),
        )
        pred = agent.predict(ema_row=sample_ema_row, date_str="2023-11-20")
        assert pred.get("_version") == "v4"
        assert pred.get("_has_diary") is True

    def test_personal_agent_gpt_v5_dry_run_filtered_sensing(self, sample_profile, sample_ema_row, tmp_path):
        llm = OpenAIClient(dry_run=True)
        (tmp_path / "filtered").mkdir(parents=True, exist_ok=True)
        agent = PersonalAgent(
            study_id=71,
            version="gpt-v5",
            llm_client=llm,
            profile=sample_profile,
            memory_doc="memory",
            processed_dir=tmp_path,
            filtered_data_dir=tmp_path / "filtered",
            ema_df=pd.DataFrame([sample_ema_row]),
        )
        pred = agent.predict(ema_row=sample_ema_row, date_str="2023-11-20")
        assert pred.get("_version") == "v5"
        assert pred.get("_has_diary") is False

    def test_personal_agent_gpt_v6_dry_run_filtered_multimodal(self, sample_profile, sample_ema_row, tmp_path):
        llm = OpenAIClient(dry_run=True)
        (tmp_path / "filtered").mkdir(parents=True, exist_ok=True)
        agent = PersonalAgent(
            study_id=71,
            version="gpt-v6",
            llm_client=llm,
            profile=sample_profile,
            memory_doc="memory",
            processed_dir=tmp_path,
            filtered_data_dir=tmp_path / "filtered",
            ema_df=pd.DataFrame([sample_ema_row]),
        )
        pred = agent.predict(ema_row=sample_ema_row, date_str="2023-11-20")
        assert pred.get("_version") == "v6"
        assert pred.get("_has_diary") is True


class TestSimulatorGPTVersions:
    def _make_sim(self, tmp_path: Path) -> PilotSimulator:
        (tmp_path / "processed" / "hourly").mkdir(parents=True, exist_ok=True)
        (tmp_path / "processed" / "filtered").mkdir(parents=True, exist_ok=True)
        loader = MagicMock()
        loader.load_all_ema.return_value = _make_min_ema(2)
        loader.load_all_sensing.return_value = {}
        loader.load_all_train.return_value = _make_min_train()
        loader.load_baseline.return_value = pd.DataFrame()
        loader.load_memory_for_user.return_value = "memory"
        loader.data_dir = tmp_path
        return PilotSimulator(
            loader=loader,
            output_dir=tmp_path / "out",
            pilot_user_ids=[71],
            dry_run=True,
            model="gpt-5.3-codex-spark",
            agentic_model="gpt-5.3-codex-spark",
        )

    def test_run_version_gpt_v1_completes(self, tmp_path):
        sim = self._make_sim(tmp_path)
        sim.setup()
        result = sim.run_version("gpt-v1")
        assert len(result["predictions"]) == 2

    def test_run_version_gpt_v2_completes(self, tmp_path):
        sim = self._make_sim(tmp_path)
        sim.setup()
        result = sim.run_version("gpt-v2")
        assert len(result["predictions"]) == 2

    def test_run_version_gpt_v5_completes(self, tmp_path):
        sim = self._make_sim(tmp_path)
        sim.setup()
        result = sim.run_version("gpt-v5")
        assert len(result["predictions"]) == 2

    def test_run_version_gpt_v6_completes(self, tmp_path):
        sim = self._make_sim(tmp_path)
        sim.setup()
        result = sim.run_version("gpt-v6")
        assert len(result["predictions"]) == 2
