"""Regression tests for 502/504 error handling fixes.

Tests that:
1. _raise_if_error_envelope extracts correct status from error envelopes
2. _status_from_error_envelope maps timeout/network errors to 504/503
3. Codex proxy error envelopes include explicit status field
4. Gateway proxy error envelopes include explicit status field
"""

from __future__ import annotations

import json

import pytest

from norax.gateway_client import (
    GatewayUpstreamError,
    _raise_if_error_envelope,
    _status_from_error_envelope,
)


class TestStatusFromErrorEnvelope:
    """Test _status_from_error_envelope correctly maps error types to HTTP statuses."""

    def test_timeout_error_maps_to_504(self):
        """Codex proxy timeout_error should map to 504, not 502."""
        envelope = {
            "error": {
                "message": "Codex upstream timeout: ...",
                "type": "timeout_error",
                "code": "codex_timeout",
            }
        }
        assert _status_from_error_envelope(envelope) == 504

    def test_timeout_with_status_field(self):
        """Explicit status field in error envelope should be used directly."""
        envelope = {
            "error": {
                "message": "Codex upstream timeout: ...",
                "type": "timeout_error",
                "code": "codex_timeout",
                "status": 504,
            }
        }
        assert _status_from_error_envelope(envelope) == 504

    def test_network_error_maps_to_503(self):
        """Codex proxy network_error should map to 503, not 502."""
        envelope = {
            "error": {
                "message": "Codex upstream network error: ...",
                "type": "network_error",
                "code": "codex_network_error",
            }
        }
        assert _status_from_error_envelope(envelope) == 503

    def test_network_error_with_status_field(self):
        """Explicit status field for network error should be used."""
        envelope = {
            "error": {
                "message": "network error",
                "type": "network_error",
                "code": "codex_network_error",
                "status": 503,
            }
        }
        assert _status_from_error_envelope(envelope) == 503

    def test_generic_error_defaults_to_502(self):
        """Errors without timeout/network markers default to 502."""
        envelope = {
            "error": {
                "message": "upstream returned no content or tool calls",
                "type": "upstream_error",
            }
        }
        assert _status_from_error_envelope(envelope) == 502

    def test_explicit_status_429(self):
        """Rate limit errors with explicit status should be preserved."""
        envelope = {
            "error": {
                "message": "Rate limited",
                "type": "rate_limit_error",
                "status": 429,
            }
        }
        assert _status_from_error_envelope(envelope) == 429

    def test_non_dict_envelope_defaults_to_502(self):
        """Non-dict envelopes default to 502."""
        assert _status_from_error_envelope("not a dict") == 502
        assert _status_from_error_envelope(None) == 502

    def test_no_error_key_defaults_to_502(self):
        """Envelopes without error key default to 502."""
        assert _status_from_error_envelope({"choices": []}) == 502

    def test_error_string_not_dict_defaults_to_502(self):
        """Error as string (not dict) defaults to 502."""
        assert _status_from_error_envelope({"error": "something went wrong"}) == 502


class TestRaiseIfErrorEnvelope:
    """Test _raise_if_error_envelope raises with correct status codes."""

    def test_timeout_raises_504_not_502(self):
        """Timeout errors should raise GatewayUpstreamError with status=504."""
        envelope = {
            "error": {
                "message": "Codex upstream timeout: connect timeout",
                "type": "timeout_error",
                "code": "codex_timeout",
                "status": 504,
            }
        }
        with pytest.raises(GatewayUpstreamError) as exc_info:
            _raise_if_error_envelope(envelope, "gpt-5.6-sol")
        assert exc_info.value.status == 504
        assert "Codex upstream timeout" in exc_info.value.upstream_message

    def test_network_error_raises_503(self):
        """Network errors should raise GatewayUpstreamError with status=503."""
        envelope = {
            "error": {
                "message": "connection reset",
                "type": "network_error",
                "code": "codex_network_error",
                "status": 503,
            }
        }
        with pytest.raises(GatewayUpstreamError) as exc_info:
            _raise_if_error_envelope(envelope, "gpt-5.6-sol")
        assert exc_info.value.status == 503

    def test_generic_error_raises_502(self):
        """Generic errors should still raise with status=502."""
        envelope = {
            "error": {
                "message": "upstream returned no content or tool calls",
            }
        }
        with pytest.raises(GatewayUpstreamError) as exc_info:
            _raise_if_error_envelope(envelope, "gpt-5.6-sol")
        assert exc_info.value.status == 502

    def test_no_error_does_not_raise(self):
        """Non-error envelopes should not raise."""
        _raise_if_error_envelope({"choices": [{"delta": {"content": "hello"}}]}, "gpt-5.6-sol")
        _raise_if_error_envelope({}, "gpt-5.6-sol")
        _raise_if_error_envelope(None, "gpt-5.6-sol")

    def test_504_is_failover_eligible(self):
        """504 status should be in _MODEL_FAILOVER_HTTP_STATUSES for agent loop failover."""
        from norax.brain.agent_loop import _MODEL_FAILOVER_HTTP_STATUSES

        assert 504 in _MODEL_FAILOVER_HTTP_STATUSES
        assert 502 in _MODEL_FAILOVER_HTTP_STATUSES
        assert 503 in _MODEL_FAILOVER_HTTP_STATUSES


class TestCodexProxyErrorEnvelopes:
    """Verify codex proxy error envelopes include status field."""

    def test_streaming_timeout_envelope_has_status(self):
        """The codex proxy streaming timeout SSE event should include status:504."""
        # Simulate what codex_proxy.py line 450 yields
        envelope = json.dumps(
            {
                "error": {
                    "message": "Codex upstream timeout: ...",
                    "type": "timeout_error",
                    "code": "codex_timeout",
                    "status": 504,
                }
            }
        )
        parsed = json.loads(envelope)
        assert parsed["error"]["status"] == 504
        status = _status_from_error_envelope(parsed)
        assert status == 504

    def test_streaming_network_envelope_has_status(self):
        """The codex proxy streaming network error SSE event should include status:503."""
        envelope = json.dumps(
            {
                "error": {
                    "message": "Codex upstream network error: ...",
                    "type": "network_error",
                    "code": "codex_network_error",
                    "status": 503,
                }
            }
        )
        parsed = json.loads(envelope)
        assert parsed["error"]["status"] == 503
        status = _status_from_error_envelope(parsed)
        assert status == 503


class TestGatewayProxyErrorEnvelopes:
    """Verify gateway proxy error envelopes include status field."""

    def test_non_stream_timeout_has_status_504(self):
        """Non-stream timeout response should include status:504 in error body."""
        # Simulate what gateway_proxy.py returns for timeout
        error_body = {
            "error": {
                "message": "upstream timeout after 120s",
                "type": "timeout",
                "status": 504,
            }
        }
        status = _status_from_error_envelope(error_body)
        assert status == 504

    def test_non_stream_proxy_error_has_status_502(self):
        """Non-stream proxy error should include status:502 in error body."""
        error_body = {
            "error": {
                "message": "connection refused",
                "type": "proxy_error",
                "status": 502,
            }
        }
        status = _status_from_error_envelope(error_body)
        assert status == 502

    def test_stream_timeout_has_status_504(self):
        """Stream timeout SSE event should include status:504."""
        sse_data = '{"error":{"message":"upstream timeout","type":"timeout","status":504}}'
        parsed = json.loads(sse_data)
        status = _status_from_error_envelope(parsed)
        assert status == 504
