"""Tests for src/think/llm_client.py — ClaudeCodeClient (dry-run focus)."""

from __future__ import annotations

import json

import pytest

from src.think.llm_client import ClaudeCodeClient
from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS


class TestClaudeCodeClientDryRun:

    def test_dry_run_returns_json_string(self):
        client = ClaudeCodeClient(dry_run=True)
        response = client.generate("some prompt")
        assert isinstance(response, str)
        data = json.loads(response)
        assert isinstance(data, dict)

    def test_dry_run_response_has_all_continuous_targets(self):
        client = ClaudeCodeClient(dry_run=True)
        response = client.generate("some prompt")
        data = json.loads(response)
        for t in CONTINUOUS_TARGETS:
            assert t in data, f"Missing {t} in dry-run response"

    def test_dry_run_response_has_all_binary_targets(self):
        client = ClaudeCodeClient(dry_run=True)
        response = client.generate("some prompt")
        data = json.loads(response)
        for t in BINARY_STATE_TARGETS:
            assert t in data, f"Missing {t} in dry-run response"

    def test_dry_run_response_has_int_availability(self):
        client = ClaudeCodeClient(dry_run=True)
        response = client.generate("some prompt")
        data = json.loads(response)
        assert "INT_availability" in data

    def test_dry_run_response_has_reasoning_and_confidence(self):
        client = ClaudeCodeClient(dry_run=True)
        response = client.generate("some prompt")
        data = json.loads(response)
        assert "reasoning" in data
        assert "confidence" in data

    def test_dry_run_increments_call_count(self):
        client = ClaudeCodeClient(dry_run=True)
        assert client.call_count == 0
        client.generate("prompt 1")
        assert client.call_count == 1
        client.generate("prompt 2")
        assert client.call_count == 2

    def test_dry_run_ignores_system_prompt(self):
        client = ClaudeCodeClient(dry_run=True)
        response1 = client.generate("prompt")
        response2 = client.generate("prompt", system_prompt="Different system context")
        # Both should return same placeholder
        data1 = json.loads(response1)
        data2 = json.loads(response2)
        assert data1["PANAS_Pos"] == data2["PANAS_Pos"]

    def test_panas_pos_in_valid_range(self):
        client = ClaudeCodeClient(dry_run=True)
        response = client.generate("some prompt")
        data = json.loads(response)
        assert 0 <= data["PANAS_Pos"] <= 30

    def test_panas_neg_in_valid_range(self):
        client = ClaudeCodeClient(dry_run=True)
        response = client.generate("some prompt")
        data = json.loads(response)
        assert 0 <= data["PANAS_Neg"] <= 30

    def test_er_desire_in_valid_range(self):
        client = ClaudeCodeClient(dry_run=True)
        response = client.generate("some prompt")
        data = json.loads(response)
        assert 0 <= data["ER_desire"] <= 10


class TestUnwrapCliResponse:

    def test_unwrap_result_field(self):
        client = ClaudeCodeClient(dry_run=True)
        inner = '{"PANAS_Pos": 15}'
        wrapped = json.dumps({"type": "result", "result": inner, "cost_usd": 0.01})
        result = client._unwrap_cli_response(wrapped)
        assert result == inner

    def test_no_result_field_returns_raw(self):
        client = ClaudeCodeClient(dry_run=True)
        raw = '{"PANAS_Pos": 15}'
        result = client._unwrap_cli_response(raw)
        assert result == raw

    def test_empty_string_returns_empty(self):
        client = ClaudeCodeClient(dry_run=True)
        result = client._unwrap_cli_response("")
        assert result == ""

    def test_non_json_string_returned_as_is(self):
        client = ClaudeCodeClient(dry_run=True)
        raw = "plain text response"
        result = client._unwrap_cli_response(raw)
        assert result == raw
