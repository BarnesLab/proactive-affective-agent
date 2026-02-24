"""Tests for src/think/parser.py — parse_prediction and parse_json_block."""

from __future__ import annotations

import json

import pytest

from src.think.parser import parse_json_block, parse_prediction
from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS


# ---------------------------------------------------------------------------
# parse_json_block
# ---------------------------------------------------------------------------

class TestParseJsonBlock:

    def test_raw_json_object(self):
        text = '{"key": "value", "num": 42}'
        result = parse_json_block(text)
        assert result == {"key": "value", "num": 42}

    def test_json_code_block(self):
        text = '```json\n{"PANAS_Pos": 15}\n```'
        result = parse_json_block(text)
        assert result == {"PANAS_Pos": 15}

    def test_json_code_block_no_lang(self):
        text = '```\n{"PANAS_Pos": 15}\n```'
        result = parse_json_block(text)
        assert result == {"PANAS_Pos": 15}

    def test_mixed_text_then_json(self):
        text = 'Here is my reasoning.\n\n{"PANAS_Pos": 12, "PANAS_Neg": 5}'
        result = parse_json_block(text)
        assert result is not None
        assert result["PANAS_Pos"] == 12

    def test_claude_cli_wrapper_format(self):
        inner = json.dumps({"PANAS_Pos": 18, "PANAS_Neg": 3})
        outer = json.dumps({"type": "result", "result": inner, "cost_usd": 0.01})
        result = parse_json_block(outer)
        # Should return the outer wrapper dict (unwrapping happens in parse_prediction)
        assert result is not None
        assert "result" in result

    def test_empty_string_returns_none(self):
        assert parse_json_block("") is None

    def test_whitespace_only_returns_none(self):
        assert parse_json_block("   \n\t  ") is None

    def test_no_json_returns_none(self):
        assert parse_json_block("Just plain text here.") is None

    def test_nested_json_object(self):
        text = '{"outer": {"inner": 42}, "flat": true}'
        result = parse_json_block(text)
        assert result["outer"] == {"inner": 42}
        assert result["flat"] is True

    def test_json_with_unicode(self):
        text = '{"text": "caf\\u00e9 visit", "num": 1}'
        result = parse_json_block(text)
        assert result is not None

    def test_multiline_json_code_block(self):
        text = (
            "Some preamble\n"
            "```json\n"
            "{\n"
            '  "PANAS_Pos": 20,\n'
            '  "reasoning": "feels good"\n'
            "}\n"
            "```\n"
            "Some postamble"
        )
        result = parse_json_block(text)
        assert result["PANAS_Pos"] == 20


# ---------------------------------------------------------------------------
# parse_prediction — full pipeline
# ---------------------------------------------------------------------------

class TestParsePrediction:

    def test_valid_complete_json(self, valid_pred_json):
        result = parse_prediction(valid_pred_json)
        assert result["PANAS_Pos"] == 15.0
        assert result["PANAS_Neg"] == 5.0
        assert result["ER_desire"] == 3.0
        assert result["INT_availability"] == "yes"
        assert result["reasoning"] == "Test reasoning text"
        assert result["confidence"] == 0.75
        assert "_parse_error" not in result

    def test_all_continuous_targets_present(self, valid_pred_json):
        result = parse_prediction(valid_pred_json)
        for t in CONTINUOUS_TARGETS:
            assert t in result, f"Missing continuous target: {t}"

    def test_all_binary_targets_present(self, valid_pred_json):
        result = parse_prediction(valid_pred_json)
        for t in BINARY_STATE_TARGETS:
            assert t in result, f"Missing binary target: {t}"

    def test_range_clamping_panas_pos_above_max(self):
        d = {"PANAS_Pos": 999.0, "PANAS_Neg": 5.0, "ER_desire": 3.0,
             "INT_availability": "yes", "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS})
        result = parse_prediction(json.dumps(d))
        assert result["PANAS_Pos"] == 30.0

    def test_range_clamping_panas_neg_below_zero(self):
        d = {"PANAS_Pos": 15.0, "PANAS_Neg": -5.0, "ER_desire": 3.0,
             "INT_availability": "yes", "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS})
        result = parse_prediction(json.dumps(d))
        assert result["PANAS_Neg"] == 0.0

    def test_range_clamping_er_desire_above_max(self):
        d = {"PANAS_Pos": 15.0, "PANAS_Neg": 5.0, "ER_desire": 50.0,
             "INT_availability": "yes", "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS})
        result = parse_prediction(json.dumps(d))
        assert result["ER_desire"] == 10.0

    def test_missing_fields_become_none(self):
        result = parse_prediction('{"PANAS_Pos": 15.0, "reasoning": "ok"}')
        assert result["PANAS_Neg"] is None
        assert result["ER_desire"] is None
        assert result["INT_availability"] is None
        for t in BINARY_STATE_TARGETS:
            assert result[t] is None

    def test_boolean_variants_string_true(self):
        d = {"PANAS_Pos": 15.0, "PANAS_Neg": 5.0, "ER_desire": 3.0,
             "Individual_level_PA_State": "true", "INT_availability": "yes",
             "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS if t != "Individual_level_PA_State"})
        result = parse_prediction(json.dumps(d))
        assert result["Individual_level_PA_State"] is True

    def test_boolean_variants_string_false(self):
        d = {"PANAS_Pos": 15.0, "PANAS_Neg": 5.0, "ER_desire": 3.0,
             "Individual_level_PA_State": "false", "INT_availability": "yes",
             "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS if t != "Individual_level_PA_State"})
        result = parse_prediction(json.dumps(d))
        assert result["Individual_level_PA_State"] is False

    def test_boolean_variants_integer_1(self):
        d = {"PANAS_Pos": 15.0, "PANAS_Neg": 5.0, "ER_desire": 3.0,
             "Individual_level_happy_State": 1, "INT_availability": "yes",
             "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS if t != "Individual_level_happy_State"})
        result = parse_prediction(json.dumps(d))
        assert result["Individual_level_happy_State"] is True

    def test_boolean_variants_integer_0(self):
        d = {"PANAS_Pos": 15.0, "PANAS_Neg": 5.0, "ER_desire": 3.0,
             "Individual_level_happy_State": 0, "INT_availability": "yes",
             "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS if t != "Individual_level_happy_State"})
        result = parse_prediction(json.dumps(d))
        assert result["Individual_level_happy_State"] is False

    def test_int_availability_bool_true_converted_to_yes(self):
        d = {"PANAS_Pos": 15.0, "PANAS_Neg": 5.0, "ER_desire": 3.0,
             "INT_availability": True, "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS})
        result = parse_prediction(json.dumps(d))
        assert result["INT_availability"] == "yes"

    def test_int_availability_bool_false_converted_to_no(self):
        d = {"PANAS_Pos": 15.0, "PANAS_Neg": 5.0, "ER_desire": 3.0,
             "INT_availability": False, "reasoning": "", "confidence": 0.5}
        d.update({t: False for t in BINARY_STATE_TARGETS})
        result = parse_prediction(json.dumps(d))
        assert result["INT_availability"] == "no"

    def test_empty_response_returns_parse_error(self):
        result = parse_prediction("")
        assert result.get("_parse_error") is True

    def test_plain_text_no_json_returns_parse_error(self):
        result = parse_prediction("I cannot predict this emotional state.")
        assert result.get("_parse_error") is True

    def test_claude_cli_wrapper_unwrapped(self):
        inner = json.dumps({
            "PANAS_Pos": 20.0, "PANAS_Neg": 4.0, "ER_desire": 2.0,
            "INT_availability": "yes", "reasoning": "test", "confidence": 0.8,
            **{t: False for t in BINARY_STATE_TARGETS},
        })
        wrapper = json.dumps({"result": inner, "cost_usd": 0.01})
        result = parse_prediction(wrapper)
        assert result["PANAS_Pos"] == 20.0
        assert "_parse_error" not in result

    def test_json_in_code_block_parsed(self):
        inner = {
            "PANAS_Pos": 12.0, "PANAS_Neg": 8.0, "ER_desire": 4.0,
            "INT_availability": "no", "reasoning": "test", "confidence": 0.6,
            **{t: False for t in BINARY_STATE_TARGETS},
        }
        text = f'```json\n{json.dumps(inner)}\n```'
        result = parse_prediction(text)
        assert result["PANAS_Pos"] == 12.0

    def test_confidence_defaults_to_zero_when_missing(self):
        result = parse_prediction('{"PANAS_Pos": 15.0}')
        assert result["confidence"] == 0.0

    def test_confidence_invalid_value_defaults_to_zero(self):
        result = parse_prediction('{"PANAS_Pos": 15.0, "confidence": "high"}')
        assert result["confidence"] == 0.0

    def test_reasoning_defaults_to_empty_string(self):
        result = parse_prediction('{"PANAS_Pos": 15.0}')
        assert result["reasoning"] == ""

    def test_string_float_values_parsed(self):
        """Some LLMs return numbers as strings."""
        d = {"PANAS_Pos": "18", "PANAS_Neg": "3", "ER_desire": "2",
             "INT_availability": "yes", "reasoning": "", "confidence": "0.7"}
        d.update({t: False for t in BINARY_STATE_TARGETS})
        result = parse_prediction(json.dumps(d))
        assert result["PANAS_Pos"] == 18.0
        assert result["confidence"] == 0.7
