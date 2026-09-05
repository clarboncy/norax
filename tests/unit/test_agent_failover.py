"""Tests for 429 usage_limit_reached failover in agent loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from norax.brain.agent_loop import (
    AGENT_MAX_OUTPUT_TOKENS,
    _is_model_failover_error,
    _is_usage_limit,
)
from norax.gateway_client import GatewayResponse, GatewayUpstreamError, StreamEvent


class TestIsUsageLimit:
    def test_429_with_usage_limit(self):
        e = GatewayUpstreamError(429, "usage_limit_reached: plan exhausted", "gpt-5.6-sol")
        assert _is_usage_limit(e) is True

    def test_502_with_usage_limit(self):
        e = GatewayUpstreamError(502, "upstream 429 usage_limit_reached", "gpt-5.6-sol")
        assert _is_usage_limit(e) is True

    def test_429_without_usage_limit(self):
        e = GatewayUpstreamError(429, "rate limited", "gpt-5.6-sol")
        assert _is_usage_limit(e) is False

    def test_500_without_usage_limit(self):
        e = GatewayUpstreamError(500, "internal error", "gpt-5.6-sol")
        assert _is_usage_limit(e) is False

    def test_non_gateway_error(self):
        e = RuntimeError("something")
        assert _is_usage_limit(e) is False

    def test_empty_upstream_message(self):
        e = GatewayUpstreamError(429, "", "model")
        assert _is_usage_limit(e) is False


def test_agent_reserves_full_16k_output_budget():
    assert AGENT_MAX_OUTPUT_TOKENS == 16384


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connect failed"),
        httpx.ReadTimeout("read timed out"),
        httpx.RemoteProtocolError("truncated stream"),
    ],
)
def test_httpx_transport_errors_are_failover_eligible(error):
    assert _is_model_failover_error(error)


@pytest.mark.parametrize("status", [401, 404, 408, 425, 429, 500, 502, 503, 504])
def test_provider_or_model_availability_errors_are_failover_eligible(status):
    assert _is_model_failover_error(GatewayUpstreamError(status, "upstream failure", "m"))


@pytest.mark.parametrize("status", [400, 402, 403, 409, 413, 422])
def test_request_or_policy_errors_are_not_failover_eligible(status):
    assert not _is_model_failover_error(GatewayUpstreamError(status, "bad request", "m"))


def _make_resp(content: str = "OK", model: str = "test") -> GatewayResponse:
    return GatewayResponse(
        request_id="test",
        model=model,
        content=content,
        tool_calls=[],
        usage={"input_tokens": 1, "output_tokens": 1},
        raw={},
    )


class TestFailoverNonStream:
    """Test the non-streaming failover path via _chat_maybe_stream."""

    @pytest.mark.asyncio
    async def test_usage_limit_skips_retries_then_failover(self):
        """429 usage_limit on primary -> skip retries, failover succeeds."""
        from norax.brain.agent_loop import run_agent_loop

        primary_err = GatewayUpstreamError(429, "usage_limit_reached", "primary-model")
        failover_resp = _make_resp(content="failover OK", model="failover-model")

        gw = MagicMock()
        # First call: primary raises usage_limit. Second call: failover succeeds.
        gw.chat = AsyncMock(side_effect=[primary_err, failover_resp])
        gw.base_url = "http://localhost:11434/v1"

        event_log = MagicMock()
        event_log.append = AsyncMock()

        # Build a minimal _chat_maybe_stream by calling run_agent_loop with
        # on_delta=None (non-streaming path) and no tools (single round).
        resp, _trace, _rounds, _ = await run_agent_loop(
            gateway=gw,
            model="primary-model",
            system_prompt="test",
            user_prompt="hello",
            allowed_tools=[],
            sender_tier="owner",
            event_log=event_log,
            on_delta=None,
            timeout_seconds=10,
            failover_models=["failover-model"],
        )
        assert "failover" in resp.content
        # Primary called once (usage_limit breaks retry), then failover once
        assert gw.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_generic_429_skips_retries_then_failover(self):
        """A plain provider rate limit must fail over just like a usage limit."""
        from norax.brain.agent_loop import run_agent_loop

        primary_err = GatewayUpstreamError(429, "rate limited", "primary-model")
        failover_resp = _make_resp(content="failover OK", model="failover-model")

        gw = MagicMock()
        gw.chat = AsyncMock(side_effect=[primary_err, failover_resp])
        gw.base_url = "http://localhost:11434/v1"

        event_log = MagicMock()
        event_log.append = AsyncMock()

        resp, _trace, _rounds, _ = await run_agent_loop(
            gateway=gw,
            model="primary-model",
            system_prompt="test",
            user_prompt="hello",
            allowed_tools=[],
            sender_tier="owner",
            event_log=event_log,
            on_delta=None,
            timeout_seconds=10,
            failover_models=["failover-model"],
        )

        assert resp.content == "failover OK"
        assert [call.args[0].model for call in gw.chat.await_args_list] == [
            "primary-model",
            "failover-model",
        ]

    @pytest.mark.asyncio
    async def test_token_limited_answer_fails_over_instead_of_presenting_partial(self):
        from norax.brain.agent_loop import run_agent_loop

        partial = _make_resp(content="cut off mid-sentence", model="primary-model")
        partial.metadata["finish_reason"] = "length"
        complete = _make_resp(content="Complete answer.", model="failover-model")
        gateway = MagicMock()
        gateway.chat = AsyncMock(side_effect=[partial, complete])
        gateway.base_url = "http://localhost:11434/v1"
        event_log = MagicMock()
        event_log.append = AsyncMock()

        response, _trace, _rounds, _state = await run_agent_loop(
            gateway=gateway,
            model="primary-model",
            system_prompt="test",
            user_prompt="give a complete answer",
            allowed_tools=[],
            sender_tier="owner",
            event_log=event_log,
            failover_models=["failover-model"],
        )

        assert response.content == "Complete answer."
        assert [call.args[0].model for call in gateway.chat.await_args_list] == [
            "primary-model",
            "failover-model",
        ]

    @pytest.mark.asyncio
    async def test_weak_to_cloud_failover_rebuilds_prompt_and_tools(self):
        """A cloud fallback must never inherit the local Gemma harness."""
        from norax.brain.agent_loop import run_agent_loop

        primary_err = GatewayUpstreamError(429, "rate limited", "gemma4:12b")
        failover_resp = _make_resp(content="Cloud baseline OK", model="kimi-k2.6:cloud")
        gw = MagicMock()
        gw.chat = AsyncMock(side_effect=[primary_err, failover_resp])
        gw.provider_kind_for_model = lambda _model: "ollama"
        gw.base_url = "http://localhost:11434/v1"
        event_log = MagicMock()
        event_log.append = AsyncMock()

        resp, _trace, _rounds, _ = await run_agent_loop(
            gateway=gw,
            model="gemma4:12b",
            system_prompt="BASE",
            user_prompt="hello",
            allowed_tools=["read", "list_dir", "exec", "write", "edit", "web_search"],
            sender_tier="owner",
            event_log=event_log,
            on_delta=None,
            timeout_seconds=10,
            failover_models=["kimi-k2.6:cloud"],
        )

        primary_req, cloud_req = [call.args[0] for call in gw.chat.await_args_list]
        assert resp.content == "Cloud baseline OK"
        assert "STRONG_SMALL_MODEL_SCAFFOLD" in primary_req.messages[0]["content"]
        assert "GOLDEN_EXAMPLES" in primary_req.messages[0]["content"]
        assert cloud_req.messages[0]["content"].startswith("BASE\n\nCURRENT_TURN_PRIORITY")
        assert "CURRENT_USER_REQUEST:\nhello" in cloud_req.messages[0]["content"]
        assert "STRONG_SMALL_MODEL_SCAFFOLD" not in cloud_req.messages[0]["content"]
        assert len(cloud_req.tools or []) == 6
        assert not (cloud_req.metadata or {}).get("ollama_use_grammar")

    @pytest.mark.asyncio
    async def test_credit_limited_402_never_switches_models(self):
        """A balance error must remain visible instead of changing the model."""
        from norax.brain.agent_loop import run_agent_loop

        primary_err = GatewayUpstreamError(
            402,
            "requested 16384 tokens but can only afford 14171",
            "primary-model",
        )

        gw = MagicMock()
        gw.chat = AsyncMock(side_effect=primary_err)
        gw.base_url = "http://localhost:11434/v1"
        event_log = MagicMock()
        event_log.append = AsyncMock()

        with pytest.raises(GatewayUpstreamError) as caught:
            await run_agent_loop(
                gateway=gw,
                model="primary-model",
                system_prompt="test",
                user_prompt="finish the task",
                allowed_tools=[],
                sender_tier="owner",
                event_log=event_log,
                on_delta=None,
                timeout_seconds=10,
                failover_models=["failover-model"],
            )

        assert caught.value.status == 402
        assert [call.args[0].model for call in gw.chat.await_args_list] == ["primary-model"]

    @pytest.mark.asyncio
    async def test_successful_failover_is_sticky_for_later_rounds(self):
        """A long turn must not probe a rate-limited primary every round."""
        from norax.brain.agent_loop import run_agent_loop

        primary_err = GatewayUpstreamError(429, "rate limited", "primary-model")
        continuing_resp = _make_resp(
            content="I need to inspect the implementation next.",
            model="failover-model",
        )
        final_resp = _make_resp(content="Inspection complete.", model="failover-model")

        gw = MagicMock()
        gw.chat = AsyncMock(side_effect=[primary_err, continuing_resp, final_resp])
        gw.base_url = "http://localhost:11434/v1"

        event_log = MagicMock()
        event_log.append = AsyncMock()

        resp, _trace, rounds, _ = await run_agent_loop(
            gateway=gw,
            model="primary-model",
            system_prompt="test",
            user_prompt="inspect this",
            allowed_tools=[],
            sender_tier="owner",
            event_log=event_log,
            on_delta=None,
            timeout_seconds=10,
            max_rounds=3,
            failover_models=["failover-model"],
        )

        assert resp.content == "Inspection complete."
        assert rounds == 2
        assert [call.args[0].model for call in gw.chat.await_args_list] == [
            "primary-model",
            "failover-model",
            "failover-model",
        ]

    @pytest.mark.asyncio
    async def test_prompted_midtask_narration_continues_with_autonomy_disabled(self):
        """Background autonomy off must not truncate the user's active task."""
        from norax.brain.agent_loop import run_agent_loop

        continuing_resp = _make_resp(content="I need to inspect the implementation next.")
        final_resp = _make_resp(content="Inspection complete.")

        gw = MagicMock()
        gw.chat = AsyncMock(side_effect=[continuing_resp, final_resp])
        gw.base_url = "http://localhost:11434/v1"

        event_log = MagicMock()
        event_log.append = AsyncMock()

        resp, _trace, rounds, _ = await run_agent_loop(
            gateway=gw,
            model="primary-model",
            system_prompt="test",
            user_prompt="inspect this",
            allowed_tools=[],
            sender_tier="owner",
            event_log=event_log,
            on_delta=None,
            timeout_seconds=10,
            max_rounds=3,
        )

        assert resp.content == "Inspection complete."
        assert rounds == 2

    @pytest.mark.asyncio
    async def test_failover_exhausted_raises(self):
        """All failover models fail -> exception propagates."""
        from norax.brain.agent_loop import run_agent_loop

        primary_err = GatewayUpstreamError(429, "usage_limit_reached", "primary-model")
        fo_err = GatewayUpstreamError(429, "usage_limit_reached", "failover-model")

        gw = MagicMock()
        gw.chat = AsyncMock(side_effect=[primary_err, fo_err, fo_err, fo_err])
        gw.base_url = "http://localhost:11434/v1"

        event_log = MagicMock()
        event_log.append = AsyncMock()

        with pytest.raises(GatewayUpstreamError):
            await run_agent_loop(
                gateway=gw,
                model="primary-model",
                system_prompt="test",
                user_prompt="hello",
                allowed_tools=[],
                sender_tier="owner",
                event_log=event_log,
                on_delta=None,
                timeout_seconds=10,
                failover_models=["failover-model"],
            )

    @pytest.mark.asyncio
    async def test_empty_failover_list_propagates(self):
        """No failover models -> primary error propagates after retries."""
        from norax.brain.agent_loop import run_agent_loop

        primary_err = GatewayUpstreamError(429, "usage_limit_reached", "primary-model")

        gw = MagicMock()
        gw.chat = AsyncMock(side_effect=[primary_err, primary_err, primary_err, primary_err])
        gw.base_url = "http://localhost:11434/v1"

        event_log = MagicMock()
        event_log.append = AsyncMock()

        with pytest.raises(GatewayUpstreamError):
            await run_agent_loop(
                gateway=gw,
                model="primary-model",
                system_prompt="test",
                user_prompt="hello",
                allowed_tools=[],
                sender_tier="owner",
                event_log=event_log,
                on_delta=None,
                timeout_seconds=10,
                failover_models=[],
            )


@pytest.mark.asyncio
async def test_partial_stream_failure_is_replaced_by_complete_fallback_response():
    """Partial streamed text must never be returned as a completed response."""
    from norax.brain.agent_loop import run_agent_loop

    async def failing_stream(_req):
        yield StreamEvent(kind="delta", text="partial response that must be replaced")
        raise GatewayUpstreamError(503, "provider overloaded", "primary-model")

    gw = MagicMock()
    gw.chat_stream = failing_stream
    gw.chat = AsyncMock(
        return_value=_make_resp(content="complete fallback response", model="failover-model")
    )
    gw.base_url = "http://localhost:11434/v1"
    event_log = MagicMock()
    event_log.append = AsyncMock()
    deltas: list[str] = []

    async def on_delta(text: str) -> None:
        deltas.append(text)

    resp, _trace, _rounds, _ = await run_agent_loop(
        gateway=gw,
        model="primary-model",
        system_prompt="test",
        user_prompt="finish the task",
        allowed_tools=[],
        sender_tier="owner",
        event_log=event_log,
        on_delta=on_delta,
        timeout_seconds=10,
        failover_models=["failover-model"],
    )

    assert deltas == ["partial response that must be replaced"]
    assert resp.content == "complete fallback response"
    assert gw.chat.await_args.args[0].model == "failover-model"
