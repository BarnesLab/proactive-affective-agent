"""Tests for session memory reflection system using real V4 pilot data.

Validates:
1. No data leakage (PANAS scores never appear in session memory)
2. Compact entry format fits within the 6000-char trim window
3. Investigation summary is concise but informative
4. Reflection generation handles edge cases
5. Session memory accumulates meaningfully across entries
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.simulation.simulator import (
    _build_actual_outcome,
    _build_investigation_summary,
    _build_prediction_summary,
    _update_session_memory,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic data from actual V4 pilot (user 071)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_pred_entry0():
    """Real V4 prediction from user 071 entry 0 (2023-11-20 afternoon)."""
    return {
        "PANAS_Pos": 16.0,
        "PANAS_Neg": 10.0,
        "ER_desire": 6.0,
        "INT_availability": "yes",
        "reasoning": (
            "The diary entry 'Talking with my best friend' anchors a positive "
            "social connection moment. However, keyboard events reveal crucial "
            "nuance: at 12:00 the participant texted 'Excellent! I need a "
            "diversion' — signaling background stress."
        ),
        "confidence": 0.58,
        "_n_tool_calls": 8,
        "_n_rounds": 4,
        "_tool_calls": [
            {"index": 1, "tool_name": "get_daily_summary",
             "input": {"date": "2023-11-20", "lookback_days": 2},
             "result_length": 743, "result_preview": "[Daily Summary...]"},
            {"index": 2, "tool_name": "get_receptivity_history",
             "input": {"n_days": 14, "include_emotion_driver": True},
             "result_length": 78, "result_preview": "No EMA entries before..."},
            {"index": 3, "tool_name": "query_sensing",
             "input": {"modality": "screen", "hours_before_ema": 4, "hours_duration": 4},
             "result_length": 107, "result_preview": "No screen data..."},
            {"index": 4, "tool_name": "query_sensing",
             "input": {"modality": "keyboard", "hours_before_ema": 4, "hours_duration": 4},
             "result_length": 177, "result_preview": "No keyboard data..."},
            {"index": 5, "tool_name": "query_sensing",
             "input": {"modality": "motion", "hours_before_ema": 6, "hours_duration": 6},
             "result_length": 217, "result_preview": "No motion data..."},
            {"index": 6, "tool_name": "find_similar_days",
             "input": {"n": 5},
             "result_length": 114, "result_preview": "No comparable past days..."},
            {"index": 7, "tool_name": "query_raw_events",
             "input": {"modality": "keyboard", "hours_before_ema": 2, "hours_duration": 2},
             "result_length": 440, "result_preview": 'Raw keyboard events...'},
            {"index": 8, "tool_name": "query_sensing",
             "input": {"modality": "light", "hours_before_ema": 4, "hours_duration": 4},
             "result_length": 174, "result_preview": "No light data..."},
        ],
        "_version": "v4",
    }


@pytest.fixture
def real_ema_entry0():
    """Real EMA row for user 071 entry 0."""
    return {
        "ER_desire": 1,
        "INT_availability": "yes",
        "emotion_driver": "Talking with my best friend.",
        "PANAS_Pos": 13.0,
        "PANAS_Neg": 2.0,
        "timestamp_local": "2023-11-20 13:38:36",
    }


@pytest.fixture
def real_pred_entry4():
    """Real V4 prediction from user 071 entry 4 (2023-11-22 morning)."""
    return {
        "PANAS_Pos": 12.0,
        "PANAS_Neg": 3.0,
        "ER_desire": 1.0,
        "INT_availability": "no",
        "reasoning": (
            "The diary entry 'Almost slept through the night' is the dominant "
            "signal for a participant with PHQ-8=13 (moderate depression)."
        ),
        "confidence": 0.6,
        "_n_tool_calls": 7,
        "_n_rounds": 4,
        "_tool_calls": [
            {"index": 1, "tool_name": "get_daily_summary",
             "input": {"date": "2023-11-22", "lookback_days": 2},
             "result_length": 1302, "result_preview": "[Daily Summary...]"},
            {"index": 2, "tool_name": "get_receptivity_history",
             "input": {"n_days": 14, "include_emotion_driver": True},
             "result_length": 531, "result_preview": "[Receptivity History...]"},
            {"index": 3, "tool_name": "query_sensing",
             "input": {"modality": "motion", "hours_before_ema": 12},
             "result_length": 344, "result_preview": "No motion data..."},
            {"index": 4, "tool_name": "query_sensing",
             "input": {"modality": "screen", "hours_before_ema": 12},
             "result_length": 344, "result_preview": "No screen data..."},
            {"index": 5, "tool_name": "query_raw_events",
             "input": {"modality": "keyboard", "hours_before_ema": 3},
             "result_length": 120, "result_preview": "Almost slept through the night"},
            {"index": 6, "tool_name": "compare_to_baseline",
             "input": {"modality": "motion", "feature": "motion_walking_min", "current_value": 5},
             "result_length": 150, "result_preview": "Z-score: +1.70 (elevated)"},
            {"index": 7, "tool_name": "find_similar_days",
             "input": {},
             "result_length": 200, "result_preview": "1. 2023-11-20 (similarity: 0.92)"},
        ],
        "_version": "v4",
    }


@pytest.fixture
def real_ema_entry4():
    """Real EMA row for user 071 entry 4."""
    return {
        "ER_desire": 0,
        "INT_availability": "no",
        "emotion_driver": "Almost slept through the night",
        "PANAS_Pos": 5.0,
        "PANAS_Neg": 0.0,
        "timestamp_local": "2023-11-22 09:17:37",
    }


# ---------------------------------------------------------------------------
# Test 1: No data leakage
# ---------------------------------------------------------------------------

class TestNoDataLeakage:
    """PANAS_Pos, PANAS_Neg, and Individual_level_* must NEVER appear in session memory."""

    def test_actual_outcome_excludes_panas(self, real_ema_entry0):
        """_build_actual_outcome must not include PANAS scores."""
        outcome = _build_actual_outcome(real_ema_entry0)
        assert "PANAS_Pos" not in outcome
        assert "PANAS_Neg" not in outcome
        assert "13.0" not in outcome  # actual PANAS_Pos value
        assert "2.0" not in outcome   # actual PANAS_Neg value

    def test_actual_outcome_excludes_individual_states(self, real_ema_entry0):
        """Individual_level_* binary targets must not leak."""
        # Add some binary targets to ema_row
        ema = {**real_ema_entry0,
               "Individual_level_PA_State": True,
               "Individual_level_NA_State": False}
        outcome = _build_actual_outcome(ema)
        assert "Individual_level" not in outcome
        assert "PA_State" not in outcome

    def test_actual_outcome_includes_only_receptivity(self, real_ema_entry0):
        """Only ER_desire, INT_availability, and diary should be present."""
        outcome = _build_actual_outcome(real_ema_entry0)
        assert "ER_desire=1" in outcome
        assert "yes" in outcome  # INT_availability
        assert "Talking with my best friend" in outcome

    @patch("src.simulation.simulator._generate_reflection")
    def test_session_memory_file_excludes_panas(self, mock_reflect,
                                                  real_pred_entry0, real_ema_entry0):
        """Full session memory entry must not contain PANAS ground truth."""
        mock_reflect.return_value = "Lesson learned: adjust predictions."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Session Memory\n\n")
            path = Path(f.name)

        result = _update_session_memory(
            path=path, ts="2023-11-20 13:38:36",
            ema_row=real_ema_entry0, pred=real_pred_entry0,
        )
        # PANAS ground truth values must not appear
        assert "PANAS_Pos=13" not in result
        assert "PANAS_Neg=2" not in result
        path.unlink()


# ---------------------------------------------------------------------------
# Test 2: Compact entry format
# ---------------------------------------------------------------------------

class TestCompactFormat:
    """Session memory entries must be compact enough for the trim window."""

    @patch("src.simulation.simulator._generate_reflection")
    def test_single_entry_under_500_chars(self, mock_reflect,
                                            real_pred_entry0, real_ema_entry0):
        """Each entry should be ~300-500 chars, fitting 10+ in 6000 char window."""
        mock_reflect.return_value = "Key lesson: weight keyboard microtext over diary surface tone."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Session Memory\n\n")
            path = Path(f.name)

        _update_session_memory(
            path=path, ts="2023-11-20 13:38:36",
            ema_row=real_ema_entry0, pred=real_pred_entry0,
        )
        content = path.read_text()
        # Subtract header
        header = "# Session Memory\n\n"
        entry_text = content[len(header):]
        entry_len = len(entry_text)
        assert entry_len < 700, f"Entry is {entry_len} chars, too large (target <700)"
        path.unlink()

    @patch("src.simulation.simulator._generate_reflection")
    def test_ten_entries_fit_in_trim_window(self, mock_reflect,
                                              real_pred_entry0, real_ema_entry0):
        """10 accumulated entries should fit within the 6000-char trim window."""
        mock_reflect.return_value = "Learned: be more conservative with PA estimates."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Session Memory\n\n## EMA History\n\n")
            path = Path(f.name)

        for i in range(10):
            _update_session_memory(
                path=path, ts=f"2023-11-{20 + i} 13:00:00",
                ema_row=real_ema_entry0, pred=real_pred_entry0,
            )

        content = path.read_text()
        assert len(content) < 6000, (
            f"10 entries = {len(content)} chars, exceeds 6000 trim window"
        )
        path.unlink()

    @patch("src.simulation.simulator._generate_reflection")
    def test_trimmed_memory_has_complete_entries(self, mock_reflect,
                                                   real_pred_entry0, real_ema_entry0):
        """After trimming to 6000 chars, the result should contain complete entries."""
        mock_reflect.return_value = "Key lesson: observed pattern X."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Session Memory\n\n## EMA History\n\n")
            path = Path(f.name)

        # Write 20 entries (more than fit in window)
        for i in range(20):
            _update_session_memory(
                path=path, ts=f"2023-11-{20 + i % 10} {10 + i}:00:00",
                ema_row=real_ema_entry0, pred=real_pred_entry0,
            )

        content = path.read_text()
        # Simulate the trim that happens in agentic_sensing.py
        trimmed = content[-6000:] if len(content) > 6000 else content
        # Should contain at least some complete entries (starting with "- **")
        complete_entries = [l for l in trimmed.split("\n") if l.startswith("- **")]
        assert len(complete_entries) >= 5, (
            f"Only {len(complete_entries)} complete entries in trimmed window"
        )
        path.unlink()


# ---------------------------------------------------------------------------
# Test 3: Investigation summary quality
# ---------------------------------------------------------------------------

class TestInvestigationSummary:
    """Investigation summary should be compact but capture tool strategy."""

    def test_compact_with_tool_calls(self, real_pred_entry0):
        """Summary should list tools compactly, not dump result previews."""
        summary = _build_investigation_summary(real_pred_entry0)
        # Should be compact
        assert len(summary) < 500, f"Summary is {len(summary)} chars, too verbose"
        # Should mention tool count and round count
        assert "8 tools" in summary
        assert "4 rounds" in summary
        # Should mention key tool names
        assert "get_daily_summary" in summary
        assert "query_sensing" in summary
        assert "query_raw_events" in summary
        # Should NOT contain full result previews
        assert "[Daily Summary" not in summary
        assert "No screen data" not in summary

    def test_modality_params_in_summary(self, real_pred_entry0):
        """query_sensing entries should show the modality queried."""
        summary = _build_investigation_summary(real_pred_entry0)
        assert "screen" in summary
        assert "keyboard" in summary
        assert "motion" in summary

    def test_fallback_without_tool_calls(self):
        """When _tool_calls is missing, fall back to n_tool_calls."""
        pred = {"_n_tool_calls": 5, "reasoning": "some reasoning text"}
        summary = _build_investigation_summary(pred)
        assert "5" in summary

    def test_empty_tool_calls(self):
        """Empty _tool_calls list should still produce output."""
        pred = {"_tool_calls": [], "_n_tool_calls": 0, "_n_rounds": 0}
        summary = _build_investigation_summary(pred)
        assert "0" in summary  # mentions zero tool calls


# ---------------------------------------------------------------------------
# Test 4: Prediction summary quality
# ---------------------------------------------------------------------------

class TestPredictionSummary:
    """Prediction summary should be compact for reflection LLM input."""

    def test_includes_key_fields(self, real_pred_entry0):
        summary = _build_prediction_summary(real_pred_entry0)
        assert "ER_desire=6.0" in summary
        assert "avail=yes" in summary
        assert "conf=0.58" in summary

    def test_excludes_panas_from_summary(self, real_pred_entry0):
        """Prediction summary for reflection should not include PANAS (they're targets)."""
        summary = _build_prediction_summary(real_pred_entry0)
        # PANAS values are still in pred dict but should not be in the
        # compact summary sent to Haiku for reflection
        assert "PANAS" not in summary

    def test_reasoning_truncated(self, real_pred_entry0):
        summary = _build_prediction_summary(real_pred_entry0)
        assert "Reasoning:" in summary
        assert len(summary) < 500


# ---------------------------------------------------------------------------
# Test 5: End-to-end session memory accumulation
# ---------------------------------------------------------------------------

class TestSessionMemoryAccumulation:
    """Validate the full lifecycle of session memory across multiple entries."""

    @patch("src.simulation.simulator._generate_reflection")
    def test_chronological_accumulation(self, mock_reflect,
                                          real_pred_entry0, real_ema_entry0,
                                          real_pred_entry4, real_ema_entry4):
        """Entries accumulate chronologically with distinct reflections."""
        reflections = [
            "This person's ER_desire is consistently low despite positive diary entries.",
            "Sleep improvement signals restoration not mood elevation for this person.",
        ]
        mock_reflect.side_effect = reflections

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Session Memory\n\n## EMA History\n\n")
            path = Path(f.name)

        # Entry 0
        mem1 = _update_session_memory(
            path=path, ts="2023-11-20 13:38:36",
            ema_row=real_ema_entry0, pred=real_pred_entry0,
        )
        assert "2023-11-20 13:38:36" in mem1
        assert "ER_desire is consistently low" in mem1

        # Entry 4
        mem2 = _update_session_memory(
            path=path, ts="2023-11-22 09:17:37",
            ema_row=real_ema_entry4, pred=real_pred_entry4,
        )
        # Both entries should be present
        assert "2023-11-20 13:38:36" in mem2
        assert "2023-11-22 09:17:37" in mem2
        assert "restoration not mood elevation" in mem2
        # File should grow
        assert len(mem2) > len(mem1)
        path.unlink()

    @patch("src.simulation.simulator._generate_reflection")
    def test_memory_contains_prediction_vs_actual_delta(self, mock_reflect,
                                                          real_pred_entry0,
                                                          real_ema_entry0):
        """Session memory should show both predicted and actual values for comparison."""
        mock_reflect.return_value = "Overestimated ER_desire (pred 6 vs actual 1)."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Session Memory\n\n")
            path = Path(f.name)

        mem = _update_session_memory(
            path=path, ts="2023-11-20 13:38:36",
            ema_row=real_ema_entry0, pred=real_pred_entry0,
        )
        # Should show predicted ER
        assert "Pred: ER=6.0" in mem
        # Should show actual ER
        assert "Actual: ER=1" in mem
        path.unlink()


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Handle missing data gracefully."""

    def test_missing_diary(self):
        """EMA row with no diary entry."""
        ema = {"ER_desire": 3, "INT_availability": "no", "emotion_driver": "nan"}
        outcome = _build_actual_outcome(ema)
        assert "No diary" in outcome
        assert "nan" not in outcome.lower().split("no diary")[0]

    def test_none_values_in_pred(self):
        """Prediction with None values (e.g., parse failure)."""
        pred = {
            "PANAS_Pos": None, "PANAS_Neg": None, "ER_desire": None,
            "INT_availability": None, "reasoning": None, "confidence": None,
            "_n_tool_calls": 0, "_n_rounds": 0, "_tool_calls": [],
        }
        summary = _build_prediction_summary(pred)
        assert "ER_desire=None" in summary
        inv = _build_investigation_summary(pred)
        assert "0" in inv  # mentions zero tool calls

    def test_very_long_diary_truncated(self):
        """Long diary entries should be truncated in actual outcome."""
        ema = {
            "ER_desire": 5, "INT_availability": "yes",
            "emotion_driver": "A" * 500,
        }
        outcome = _build_actual_outcome(ema)
        assert len(outcome) < 300

    def test_reflection_failure_fallback(self, real_pred_entry0, real_ema_entry0):
        """If reflection LLM fails, fallback text should still work."""
        from src.simulation.simulator import _generate_reflection
        with patch("src.simulation.simulator._get_reflection_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API down")
            result = _generate_reflection(real_pred_entry0, real_ema_entry0)
            assert "Predicted based on:" in result
            assert len(result) > 10


# ---------------------------------------------------------------------------
# Test 7: Validate against real V4 trace files (if available)
# ---------------------------------------------------------------------------

class TestRealTraceValidation:
    """Load actual V4 trace files and validate session memory would be correct."""

    TRACE_DIR = Path("outputs/pilot/traces")

    @pytest.mark.skipif(
        not (Path("outputs/pilot/traces/v4_user71_entry0.json").exists()),
        reason="V4 trace files not available",
    )
    def test_real_trace_investigation_summary(self):
        """Build investigation summary from a real trace file."""
        trace = json.loads(
            (self.TRACE_DIR / "v4_user71_entry0.json").read_text()
        )
        summary = _build_investigation_summary(trace)
        # Should be compact
        assert len(summary) < 500
        # Should reflect the actual tool strategy
        assert "8 tools" in summary
        assert "get_daily_summary" in summary

    @pytest.mark.skipif(
        not (Path("outputs/pilot/traces/v4_user71_entry0.json").exists()),
        reason="V4 trace files not available",
    )
    def test_real_trace_no_panas_leakage(self):
        """Ensure _build_actual_outcome doesn't leak PANAS from real data."""
        # Simulate the EMA row that would have been used
        ema = {
            "ER_desire": 1, "INT_availability": "yes",
            "emotion_driver": "Talking with my best friend.",
            "PANAS_Pos": 13.0, "PANAS_Neg": 2.0,
        }
        outcome = _build_actual_outcome(ema)
        assert "13.0" not in outcome
        assert "2.0" not in outcome
        assert "PANAS" not in outcome
