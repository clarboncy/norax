"""Tests for user-visible failure replies on turn death."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from norax.gateway_client import GatewayUpstreamError, SpendGuardTripped
from norax.runtime.core import _turn_failure_message


class TestFailureReply:
    """Verify that turn failures produce user-visible messages."""

    @pytest.mark.asyncio
    async def test_gateway_error_produces_failure_reply(self):
        """When the agent loop raises GatewayUpstreamError, the user gets a
        failure message containing 'couldn't complete' instead of silence."""
        from norax.brain.agent_loop import run_agent_loop

        err = GatewayUpstreamError(502, "upstream returned no content", "bad-model")

        gw = MagicMock()
        gw.chat = AsyncMock(side_effect=[err, err, err, err])
        gw.base_url = "http://localhost:11434/v1"

        event_log = MagicMock()
        event_log.append = AsyncMock()

        with pytest.raises(GatewayUpstreamError):
            await run_agent_loop(
                gateway=gw,
                model="bad-model",
                system_prompt="test",
                user_prompt="hello",
                allowed_tools=[],
                sender_tier="owner",
                event_log=event_log,
                on_delta=None,
                timeout_seconds=10,
            )

    def test_failure_message_contains_couldnt_complete(self):
        """The synthesized failure message text includes 'couldn't complete'."""
        msg = _turn_failure_message(
            GatewayUpstreamError(502, "upstream returned no content", "bad-model")
        )
        assert "couldn't complete" in msg
        assert "502" in msg
        assert "bad-model" in msg

    def test_402_is_a_balance_alert_without_fallback(self):
        msg = _turn_failure_message(
            GatewayUpstreamError(
                402,
                "requested 16384 tokens but can only afford 14171",
                "openrouter/moonshotai/kimi-k3",
            )
        )

        assert "balance alert" in msg
        assert "No fallback model was used" in msg
        assert "selected model has not changed" in msg
        assert "openrouter/moonshotai/kimi-k3" in msg
        assert "can only afford 14171" in msg

    def test_internal_error_message_contains_couldnt_complete(self):
        """Internal errors also produce a user-visible failure message."""
        msg = _turn_failure_message(RuntimeError("boom"))
        assert "couldn't complete" in msg
        assert "RuntimeError" in msg

    def test_spend_guard_message_is_distinct(self):
        """Spend guard trips produce a distinct message (not 'couldn't complete')."""
        msg = _turn_failure_message(SpendGuardTripped(window="60 seconds", count=10, limit=5))
        assert "Spend guard" in msg
        assert "couldn't complete" not in msg
