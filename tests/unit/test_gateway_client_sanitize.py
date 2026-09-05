"""Tests for tool_call_id sanitization in gateway client."""

from __future__ import annotations

from norax.gateway_client import _ensure_tool_call_id


class TestEnsureToolCallId:
    def test_missing_id_gets_assigned(self):
        msg = {"role": "tool", "content": "result", "name": "get_weather"}
        result = _ensure_tool_call_id(msg, index=3)
        assert result["tool_call_id"] == "call_sanitized_0003"
        assert result["content"] == "result"

    def test_empty_string_id_replaced(self):
        msg = {"role": "tool", "tool_call_id": "", "content": "result"}
        result = _ensure_tool_call_id(msg, index=1)
        assert result["tool_call_id"] == "call_sanitized_0001"

    def test_literal_tool_id_replaced(self):
        msg = {"role": "tool", "tool_call_id": "tool", "content": "result"}
        result = _ensure_tool_call_id(msg, index=5)
        assert result["tool_call_id"] == "call_sanitized_0005"

    def test_two_tool_messages_get_distinct_ids(self):
        msg1 = {"role": "tool", "tool_call_id": "tool", "content": "r1"}
        msg2 = {"role": "tool", "tool_call_id": "tool", "content": "r2"}
        r1 = _ensure_tool_call_id(msg1, index=1)
        r2 = _ensure_tool_call_id(msg2, index=2)
        assert r1["tool_call_id"] != r2["tool_call_id"]
        assert r1["tool_call_id"] == "call_sanitized_0001"
        assert r2["tool_call_id"] == "call_sanitized_0002"

    def test_present_nonempty_id_untouched(self):
        msg = {"role": "tool", "tool_call_id": "call_abc123", "content": "result"}
        result = _ensure_tool_call_id(msg, index=1)
        assert result["tool_call_id"] == "call_abc123"

    def test_non_tool_message_untouched(self):
        msg = {"role": "user", "content": "hello"}
        result = _ensure_tool_call_id(msg, index=1)
        assert result == msg
        assert "tool_call_id" not in result

    def test_whitespace_only_id_replaced(self):
        msg = {"role": "tool", "tool_call_id": "   ", "content": "result"}
        result = _ensure_tool_call_id(msg, index=7)
        assert result["tool_call_id"] == "call_sanitized_0007"

    def test_original_msg_not_mutated(self):
        msg = {"role": "tool", "tool_call_id": "", "content": "result"}
        result = _ensure_tool_call_id(msg, index=1)
        assert msg["tool_call_id"] == ""
        assert result["tool_call_id"] == "call_sanitized_0001"
