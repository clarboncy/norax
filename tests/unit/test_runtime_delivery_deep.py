"""Failure-aware delivery and mirroring contracts for every outbound path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from norax.runtime.delivery import DeliveryMixin, _delivery_error


class _Family:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.labels_seen: list[dict[str, str]] = []
        self.increments: list[int | None] = []

    def labels(self, **labels: str) -> _Family:
        if self.fail:
            raise RuntimeError("metric failed")
        self.labels_seen.append(labels)
        return self

    def inc(self, amount: int | None = None) -> None:
        if self.fail:
            raise RuntimeError("metric failed")
        self.increments.append(amount)


class _Metrics:
    def __init__(self) -> None:
        self.outbound_sends = _Family()
        self.gateway_tokens_in = _Family()
        self.gateway_tokens_out = _Family()
        self.gateway_requests = _Family()


class _Events:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.rows: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []

    async def append(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        if self.fail:
            raise OSError("event append failed")
        self.rows.append((kind, payload, attrs))


class _Outbound:
    def __init__(self, result: object = None, *, available: bool = True) -> None:
        self.result = {"ok": True} if result is None else result
        self.available = available
        self.sent: list[tuple[str, str, str, str | None]] = []

    def has(self, _source: str) -> bool:
        return self.available

    async def send(
        self,
        source: str,
        target: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> object:
        self.sent.append((source, target, text, reply_to))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _Bridge:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[str] = []

    async def broadcast_outbound(self, *, text: str) -> None:
        if self.fail:
            raise RuntimeError("bridge failed")
        self.messages.append(text)


class _Stream:
    def __init__(
        self,
        *,
        finalize_result: object = None,
        finalize_error: BaseException | None = None,
        delete_error: BaseException | None = None,
    ) -> None:
        self.finalize_result = {"ok": True} if finalize_result is None else finalize_result
        self.finalize_error = finalize_error
        self.delete_error = delete_error
        self.finalized: list[str] = []
        self.deleted = False

    async def finalize(self, text: str) -> object:
        self.finalized.append(text)
        if self.finalize_error is not None:
            raise self.finalize_error
        return self.finalize_result

    async def delete(self) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted = True


def _runtime() -> DeliveryMixin:
    runtime = DeliveryMixin()
    runtime.events = _Events()
    runtime.metrics = _Metrics()
    runtime.outbound = _Outbound()
    runtime.agent_os_bridge = None
    runtime.default_model = "default/model"
    return runtime


def _env(
    *,
    source: str = "discord",
    channel_id: str | None = "channel",
    thread_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        source=source,
        raw={"channel_id": channel_id} if source == "discord" else {},
        thread_binding=(SimpleNamespace(thread_id=thread_id) if thread_id is not None else None),
        message_id="message",
    )


def _response(
    content: str,
    *,
    model: str = "model",
    usage: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        model=model,
        usage={} if usage is None else usage,
        request_id="request",
    )


async def _deliver(
    runtime: DeliveryMixin,
    *,
    content: str,
    stream: object | None,
    target: str | None = "channel",
    model: str = "model",
    usage: object = None,
    decision: str = "emit_reply",
) -> dict[str, Any]:
    return await runtime._deliver_turn_response(
        env=_env(),
        ctx=SimpleNamespace(decision=decision),
        resp=_response(content, model=model, usage=usage),
        trace=[{"name": "tool"}],
        rounds=2,
        streaming_msg=stream,
        stream_delta_buf=["one", "two"],
        target_channel=target,
    )


def test_delivery_errors_are_secret_safe_and_bounded() -> None:
    secret = "sk-" + ("a" * 32)
    value = _delivery_error(RuntimeError(f"credential={secret} " + ("x" * 1_000)))
    assert secret not in value
    assert "<REDACTED:openai_key>" in value
    assert len(value) == 500


@pytest.mark.asyncio
async def test_delivery_event_and_metric_failures_never_rewrite_truth() -> None:
    runtime = _runtime()
    await runtime._append_delivery_event("send", {"ok": True})
    runtime._record_outbound_metric(source="discord", delivered=True)
    assert runtime.events.rows[0][1] == {"ok": True}
    assert runtime.metrics.outbound_sends.labels_seen == [{"channel": "discord", "ok": "true"}]

    runtime.events = _Events(fail=True)
    runtime.metrics.outbound_sends = _Family(fail=True)
    await runtime._append_delivery_event("send", {"ok": True})
    runtime._record_outbound_metric(source="discord", delivered=False)


@pytest.mark.asyncio
async def test_command_reply_requires_text_route_target_and_literal_success() -> None:
    runtime = _runtime()
    assert await runtime._send_command_reply(_env(), "  ") is False

    runtime.outbound = _Outbound(available=False)
    assert await runtime._send_command_reply(_env(), "reply") is False
    runtime.outbound = _Outbound()
    assert await runtime._send_command_reply(_env(channel_id=None), "reply") is False

    runtime.agent_os_bridge = _Bridge()
    assert await runtime._send_command_reply(_env(), " reply ", reply_to="parent") is True
    assert runtime.agent_os_bridge.messages == ["reply"]

    runtime.outbound = _Outbound({"ok": 1})
    assert await runtime._send_command_reply(_env(), "reply") is False
    assert runtime.agent_os_bridge.messages == ["reply"]

    runtime.outbound = _Outbound(RuntimeError("adapter down"))
    assert (
        await runtime._send_command_reply(
            _env(source="a2a", channel_id=None, thread_id="thread"), "reply"
        )
        is False
    )
    send_event = runtime.events.rows[-1][1]
    assert send_event["ok"] is False
    assert send_event["result"]["error"].startswith("RuntimeError:")


@pytest.mark.asyncio
async def test_discord_mirror_has_strict_eligibility_and_contains_bridge_failure() -> None:
    runtime = _runtime()
    runtime.agent_os_bridge = _Bridge()
    await runtime._mirror_discord_reply(_env(source="a2a"), "reply")
    runtime.agent_os_bridge = None
    await runtime._mirror_discord_reply(_env(), "reply")
    runtime.agent_os_bridge = _Bridge()
    await runtime._mirror_discord_reply(_env(), "  ")
    await runtime._mirror_discord_reply(_env(), " reply ")
    assert runtime.agent_os_bridge.messages == ["reply"]

    runtime.agent_os_bridge = _Bridge(fail=True)
    await runtime._mirror_discord_reply(_env(), "reply")


@pytest.mark.asyncio
async def test_stream_sentinels_are_suppressed_only_when_cleanup_succeeds() -> None:
    runtime = _runtime()
    stream = _Stream()
    result = await _deliver(runtime, content="NO_REPLY", stream=stream)
    assert result == {"ok": True, "state": "intentionally_suppressed", "attempted": False}
    assert stream.deleted is True

    stream = _Stream(delete_error=RuntimeError("delete failed"))
    result = await _deliver(runtime, content="HEARTBEAT_OK", stream=stream)
    assert result == {"ok": False, "state": "suppression_cleanup_failed", "attempted": True}


@pytest.mark.asyncio
async def test_stream_finalization_handles_empty_gemma_normal_and_invalid_results() -> None:
    runtime = _runtime()
    runtime.agent_os_bridge = _Bridge()

    empty = _Stream()
    result = await _deliver(runtime, content="", stream=empty)
    assert result["state"] == "delivered"
    assert empty.finalized[0].startswith("⚠️ Empty upstream response")

    gemma = _Stream(finalize_result={"ok": False})
    result = await _deliver(runtime, content=".", stream=gemma, model="gemma3")
    assert result["state"] == "delivery_failed"
    assert gemma.finalized[0].startswith("⚠️ Suspicious single-dot")

    normal = _Stream(finalize_result="not-a-receipt")
    result = await _deliver(runtime, content="[[reply_to_current]] hello", stream=normal)
    assert result["state"] == "delivery_failed"
    assert result["result"]["error"] == "invalid_stream_finalize_result"

    success = _Stream()
    result = await _deliver(runtime, content="delivered text", stream=success)
    assert result["state"] == "delivered"
    assert runtime.agent_os_bridge.messages == ["delivered text"]


@pytest.mark.asyncio
async def test_stream_finalize_failure_has_explicit_fallback_outcomes() -> None:
    runtime = _runtime()
    failure = RuntimeError("finalize failed")

    runtime.outbound = _Outbound({"ok": True})
    result = await _deliver(runtime, content="answer", stream=_Stream(finalize_error=failure))
    assert result["state"] == "delivered"

    runtime.outbound = _Outbound("invalid")
    result = await _deliver(runtime, content="answer", stream=_Stream(finalize_error=failure))
    assert result["state"] == "delivery_failed"
    assert result["result"]["error"] == "invalid_outbound_result"

    runtime.outbound = _Outbound(RuntimeError("fallback failed"))
    result = await _deliver(runtime, content="answer", stream=_Stream(finalize_error=failure))
    assert result["state"] == "delivery_failed"
    assert "fallback failed" in result["result"]["error"]

    runtime.outbound = _Outbound(available=False)
    result = await _deliver(runtime, content="answer", stream=_Stream(finalize_error=failure))
    assert result == {"ok": False, "state": "unroutable", "attempted": False}

    runtime.outbound = _Outbound()
    result = await _deliver(
        runtime,
        content="answer",
        stream=_Stream(finalize_error=failure),
        target=None,
    )
    assert result["state"] == "unroutable"


@pytest.mark.asyncio
async def test_generation_metrics_accept_usage_and_contain_invalid_telemetry() -> None:
    runtime = _runtime()
    result = await _deliver(
        runtime,
        content="answer",
        stream=None,
        model="",
        usage={"input_tokens": "4", "output_tokens": 5},
    )
    assert result["state"] == "delivered"
    assert runtime.metrics.gateway_tokens_in.increments == [4]
    assert runtime.metrics.gateway_tokens_out.increments == [5]
    assert runtime.metrics.gateway_requests.labels_seen == [
        {"model": "default/model", "status": "ok"}
    ]

    runtime.metrics.gateway_tokens_in = _Family(fail=True)
    result = await _deliver(
        runtime,
        content="answer",
        stream=None,
        usage={"input_tokens": "bad", "output_tokens": 0},
    )
    assert result["state"] == "delivered"


@pytest.mark.asyncio
async def test_emit_reply_covers_policy_sentinel_empty_route_and_adapter_failures() -> None:
    runtime = _runtime()
    emit = SimpleNamespace(decision="emit_reply")
    skip = SimpleNamespace(decision="store_only")

    assert (await runtime._emit_reply(_env(), skip, _response("answer")))[
        "state"
    ] == "not_requested"
    result = await runtime._emit_reply(_env(), emit, _response(""))
    assert result["state"] == "delivered"
    assert runtime.outbound.sent[-1][2].startswith("⚠️ Empty upstream response")
    assert (await runtime._emit_reply(_env(), emit, _response("NO_REPLY")))["state"] == (
        "intentionally_suppressed"
    )
    assert (await runtime._emit_reply(_env(), emit, _response("[[reply_to_current]]")))[
        "state"
    ] == "empty_reply"

    runtime.outbound = _Outbound(available=False)
    assert (await runtime._emit_reply(_env(), emit, _response("answer")))["state"] == "no_outbound"

    runtime.outbound = _Outbound()
    assert (
        await runtime._emit_reply(_env(source="a2a", channel_id=None), emit, _response("answer"))
    )["state"] == "no_target"

    runtime.outbound = _Outbound({"ok": False})
    result = await runtime._emit_reply(
        _env(source="a2a", channel_id=None, thread_id="thread"), emit, _response("answer")
    )
    assert result["state"] == "delivery_failed"
    assert result["attempted"] is True

    runtime.outbound = _Outbound(RuntimeError("network failed"))
    result = await runtime._emit_reply(_env(), emit, _response("answer"))
    assert result["state"] == "delivery_failed"
    assert result["result"]["error"].startswith("RuntimeError:")


@pytest.mark.asyncio
async def test_turn_delivery_mirrors_only_proven_discord_delivery() -> None:
    runtime = _runtime()
    runtime.agent_os_bridge = _Bridge()
    runtime.outbound = _Outbound({"ok": False})
    result = await _deliver(runtime, content="not delivered", stream=None)
    assert result["state"] == "delivery_failed"
    assert runtime.agent_os_bridge.messages == []

    runtime.outbound = _Outbound({"ok": True})
    result = await _deliver(runtime, content="delivered", stream=None)
    assert result["state"] == "delivered"
    assert runtime.agent_os_bridge.messages == ["delivered"]
    reply_event = runtime.events.rows[-1]
    assert reply_event[0] == "reply"
    assert reply_event[1]["delivered"] is True
    assert reply_event[2]["gen_ai.response.id"] == "request"
