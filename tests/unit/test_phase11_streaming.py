"""Phase 11c — typing indicator + streaming chat."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from norax.adapter.discord_in import (
    StreamingMessage,
    TypingKeepAlive,
    _split_discord_message,
)
from norax.brain import agent_loop
from norax.brain.output_verifier import OutputVerifier
from norax.gateway_client import GatewayClient, GatewayRequest, GatewayUpstreamError

# ---------------------------------------------------------------------------
# Typing keep-alive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_keepalive_triggers_immediately_and_stops_cleanly():
    # New TypingKeepAlive prefers channel._state.http.send_typing(id) over
    # the deprecated trigger_typing(). Mock the low-level HTTP path.
    channel = MagicMock()
    channel.id = 12345
    channel._state = MagicMock()
    channel._state.http = MagicMock()
    channel._state.http.send_typing = AsyncMock()
    keep = TypingKeepAlive(channel, interval=0.05)
    await keep.start()
    await asyncio.sleep(0.12)  # should re-trigger ~2 times
    await keep.stop()
    # Initial + at least one refresh
    assert channel._state.http.send_typing.await_count >= 2


# ---------------------------------------------------------------------------
# Streaming message — append + finalize (first_message provided)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_message_edits_in_place_and_finalizes():
    channel = MagicMock()
    msg = MagicMock()
    msg.id = 12345
    msg.edit = AsyncMock()
    sm = StreamingMessage(
        channel=channel,
        first_message=msg,
        min_edit_interval=0.0,
        edit_threshold_chars=1,
    )
    await sm.append("hello ")
    await sm.append("world")
    result = await sm.finalize("hello world!")
    # At least one edit during streaming + one in finalize
    assert msg.edit.await_count >= 2
    last_call = msg.edit.await_args
    assert last_call.kwargs.get("content") == "hello world!"
    assert result["ok"] is True
    assert result["chunks"] == 1


@pytest.mark.asyncio
async def test_streaming_message_chunks_long_finalize():
    channel = MagicMock()
    sent_msg = MagicMock()
    sent_msg.id = 99
    channel.send = AsyncMock(return_value=sent_msg)
    first = MagicMock()
    first.id = 1
    first.edit = AsyncMock()
    sm = StreamingMessage(channel=channel, first_message=first)

    big = "a" * 3000
    result = await sm.finalize(big)
    # Should have edited the first message AND sent one extra chunk.
    assert first.edit.await_count == 1
    assert channel.send.await_count == 1
    assert result["chunks"] == 2
    assert len(result["message_ids"]) == 2


@pytest.mark.asyncio
async def test_streaming_message_rate_limits_edits():
    channel = MagicMock()
    msg = MagicMock()
    msg.id = 1
    msg.edit = AsyncMock()
    sm = StreamingMessage(
        channel=channel,
        first_message=msg,
        min_edit_interval=10.0,  # effectively never during the test
        edit_threshold_chars=1,
    )
    for _ in range(20):
        await sm.append("x")
    # Initial fast burst should NOT have produced edits because the
    # interval gate hasn't elapsed.
    assert msg.edit.await_count == 0
    await sm.finalize()
    assert msg.edit.await_count == 1  # only finalize edits


# ---------------------------------------------------------------------------
# Streaming message — deferred mode (no first_message)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_message_deferred_first_message():
    """When first_message is None, _push() creates via send(); finalize edits."""
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 42
    sent.edit = AsyncMock()
    channel.send = AsyncMock(return_value=sent)
    sm = StreamingMessage(
        channel=channel, first_message=None, min_edit_interval=0.0, edit_threshold_chars=1
    )

    await sm.append("hello ")
    assert channel.send.await_count == 1
    assert channel.send.await_args.kwargs == {}
    assert sm.message_ids == ["42"]

    result = await sm.finalize("hello world!")
    # Already created via _push(), finalize should edit, not send again.
    assert sent.edit.await_count == 1
    assert channel.send.await_count == 1  # only the one from _push
    assert result["chunks"] == 1
    assert result["message_ids"] == ["42"]


@pytest.mark.asyncio
async def test_streaming_message_finalize_without_push():
    """If nothing was streamed, finalize sends the first message outright."""
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 77
    channel.send = AsyncMock(return_value=sent)
    sm = StreamingMessage(
        channel=channel, first_message=None, min_edit_interval=0.0, edit_threshold_chars=1
    )

    result = await sm.finalize("quick reply")
    assert channel.send.await_count == 1
    assert channel.send.await_args.kwargs == {}
    assert sm.message_ids == ["77"]
    assert result["chunks"] == 1


@pytest.mark.asyncio
async def test_streaming_message_empty_finalize_surfaces_diagnostic():
    channel = MagicMock()
    msg = MagicMock()
    msg.id = 2
    msg.edit = AsyncMock()
    sm = StreamingMessage(channel=channel, first_message=msg)
    result = await sm.finalize("")
    assert msg.edit.await_count == 1
    content = msg.edit.await_args.kwargs.get("content")
    assert "Empty Discord stream finalize" in content
    assert result["chunks"] == 1


@pytest.mark.asyncio
async def test_streaming_preview_chunks_live_output_without_continuation_stub():
    """Long live output is delivered as real chunks, never a dangling stub."""
    channel = MagicMock()
    first = MagicMock()
    first.id = 42
    first.edit = AsyncMock()
    second = MagicMock()
    second.id = 43
    second.edit = AsyncMock()
    channel.send = AsyncMock(side_effect=[first, second])
    sm = StreamingMessage(
        channel=channel, first_message=None, min_edit_interval=0.0, edit_threshold_chars=1
    )

    await sm.append("opening " + "x" * 2_100)

    assert channel.send.await_count == 2
    sent_chunks = [call.args[0] for call in channel.send.await_args_list]
    assert "".join(sent_chunks) == "opening " + "x" * 2_100
    assert all(len(chunk) <= 1_900 for chunk in sent_chunks)
    assert "response continues" not in "".join(sent_chunks)
    assert sm.message_ids == ["42", "43"]


@pytest.mark.asyncio
async def test_streaming_failover_finalize_removes_stale_preview_chunks():
    channel = MagicMock()
    first = MagicMock()
    first.id = 42
    first.edit = AsyncMock()
    first.delete = AsyncMock()
    second = MagicMock()
    second.id = 43
    second.edit = AsyncMock()
    second.delete = AsyncMock()
    channel.send = AsyncMock(side_effect=[first, second])
    sm = StreamingMessage(
        channel=channel, first_message=None, min_edit_interval=0.0, edit_threshold_chars=1
    )

    await sm.append("partial " + "x" * 2_100)
    result = await sm.finalize("complete fallback response")

    assert result["ok"] is True
    assert result["chunks"] == 1
    assert result["message_ids"] == ["42"]
    assert first.edit.await_args.kwargs["content"] == "complete fallback response"
    second.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_message_deferred_reference_passed():
    """When first_message is None, the reference is forwarded to the first send()."""
    channel = MagicMock()
    sent = MagicMock()
    sent.id = 88
    channel.send = AsyncMock(return_value=sent)
    ref = {"message_id": 1}
    sm = StreamingMessage(
        channel=channel,
        first_message=None,
        reference=ref,
        min_edit_interval=0.0,
        edit_threshold_chars=1,
    )
    await sm.append("hi")
    assert channel.send.await_count == 1
    assert channel.send.await_args.kwargs == {"reference": ref}
    assert sm.message_ids == ["88"]


# ---------------------------------------------------------------------------
# Gateway streaming client
# ---------------------------------------------------------------------------


def _sse(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


@pytest.mark.asyncio
@respx.mock
async def test_verifier_keeps_rejected_stream_draft_private() -> None:
    route = respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                text=_sse(
                    {"choices": [{"index": 0, "delta": {"content": "2 + 2 = 5"}}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ),
                headers={"content-type": "text/event-stream"},
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "2 + 2 = 4"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
        ]
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    published: list[str] = []

    async def on_delta(text: str) -> None:
        published.append(text)

    try:
        response, _trace, rounds, _state = await agent_loop.run_agent_loop(
            gateway=gateway,
            model="m",
            system_prompt="system",
            user_prompt="Calculate two plus two and check it.",
            allowed_tools=[],
            sender_tier="owner",
            event_log=_EventLogStub(),
            on_delta=on_delta,
            output_verifier=OutputVerifier(),
        )
    finally:
        await gateway.aclose()

    assert route.call_count == 2
    assert published == []
    assert response.content == "2 + 2 = 4"
    assert rounds == 2


@pytest.mark.asyncio
@respx.mock
async def test_gateway_chat_stream_yields_deltas_and_final():
    body = _sse(
        {"id": "r1", "model": "m", "choices": [{"index": 0, "delta": {"content": "Hello"}}]},
        {"choices": [{"index": 0, "delta": {"content": " world"}}]},
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )
    respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )
    )
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        deltas: list[str] = []
        final = None
        async for evt in gw.chat_stream(
            GatewayRequest(model="m", messages=[{"role": "user", "content": "hi"}])
        ):
            if evt.kind == "delta":
                deltas.append(evt.text)
            else:
                final = evt.response
        assert deltas == ["Hello", " world"]
        assert final is not None
        assert final.content == "Hello world"
        assert final.usage["input_tokens"] == 3
        assert final.usage["output_tokens"] == 2
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_chat_stream_assembles_tool_calls():
    body = _sse(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"pa'},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th":"x"}'}}]},
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    )
    respx.post("http://stub/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        final = None
        async for evt in gw.chat_stream(
            GatewayRequest(model="m", messages=[{"role": "user", "content": "x"}])
        ):
            if evt.kind == "final":
                final = evt.response
        assert final is not None
        assert final.tool_calls[0]["id"] == "tc1"
        assert final.tool_calls[0]["function"]["name"] == "read"
        assert final.tool_calls[0]["function"]["arguments"] == '{"path":"x"}'
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_ollama_native_jsonl_stream_yields_delta_and_final():
    body = (
        json.dumps(
            {
                "model": "gemma4:31b-cloud",
                "message": {"role": "assistant", "content": "ok"},
                "done": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "model": "gemma4:31b-cloud",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 4,
                "eval_count": 1,
            }
        )
        + "\n"
    )
    respx.post("http://stub/api/chat").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1", provider_kind="ollama")
    try:
        deltas: list[str] = []
        final = None
        async for evt in gw.chat_stream(
            GatewayRequest(
                model="gemma4:31b-cloud", messages=[{"role": "user", "content": "Respond ok"}]
            )
        ):
            if evt.kind == "delta":
                deltas.append(evt.text)
            if evt.kind == "final":
                final = evt.response
        assert deltas == ["ok"]
        assert final is not None
        assert final.content == "ok"
        assert final.usage == {"input_tokens": 4, "output_tokens": 1}
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_ollama_done_frame_deduplicates_full_echo():
    body = (
        json.dumps(
            {
                "model": "kimi-k3:cloud",
                "message": {"role": "assistant", "content": "answer"},
                "done": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "model": "kimi-k3:cloud",
                "message": {"role": "assistant", "content": "answer"},
                "done": True,
            }
        )
        + "\n"
    )
    respx.post("http://stub/api/chat").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1", provider_kind="ollama")
    try:
        deltas: list[str] = []
        final = None
        async for evt in gw.chat_stream(
            GatewayRequest(model="kimi-k3:cloud", messages=[{"role": "user", "content": "x"}])
        ):
            if evt.kind == "delta":
                deltas.append(evt.text)
            else:
                final = evt.response
        assert deltas == ["answer"]
        assert final is not None
        assert final.content == "answer"
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_ollama_keeps_content_sent_only_in_done_frame():
    body = (
        json.dumps(
            {
                "model": "kimi-k3:cloud",
                "message": {"role": "assistant", "content": "final-only answer"},
                "done": True,
            }
        )
        + "\n"
    )
    respx.post("http://stub/api/chat").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1", provider_kind="ollama")
    try:
        events = [
            evt
            async for evt in gw.chat_stream(
                GatewayRequest(model="kimi-k3:cloud", messages=[{"role": "user", "content": "x"}])
            )
        ]
        assert [evt.text for evt in events if evt.kind == "delta"] == ["final-only answer"]
        assert events[-1].response is not None
        assert events[-1].response.content == "final-only answer"
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_stream_suppresses_reasoning_tags_split_across_chunks():
    body = _sse(
        {"choices": [{"index": 0, "delta": {"content": "Visible<thi"}}]},
        {"choices": [{"index": 0, "delta": {"content": "nking>private chain"}}]},
        {"choices": [{"index": 0, "delta": {"content": "</think"}}]},
        {"choices": [{"index": 0, "delta": {"content": "ing> answer"}}]},
    )
    respx.post("http://stub/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        deltas: list[str] = []
        final = None
        async for evt in gw.chat_stream(
            GatewayRequest(model="m", messages=[{"role": "user", "content": "x"}])
        ):
            if evt.kind == "delta":
                deltas.append(evt.text)
            else:
                final = evt.response
        assert "private chain" not in "".join(deltas)
        assert "".join(deltas) == "Visible answer"
        assert final is not None
        assert final.content == "Visible answer"
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_ollama_native_jsonl_stream_assembles_tool_calls():
    body = (
        json.dumps(
            {
                "model": "glm-5.1:cloud",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "status", "arguments": {}}}],
                },
                "done": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "model": "glm-5.1:cloud",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 9,
                "eval_count": 2,
            }
        )
        + "\n"
    )
    respx.post("http://stub/api/chat").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1", provider_kind="ollama")
    try:
        final = None
        async for evt in gw.chat_stream(
            GatewayRequest(
                model="glm-5.1:cloud", messages=[{"role": "user", "content": "call status"}]
            )
        ):
            if evt.kind == "final":
                final = evt.response
        assert final is not None
        assert final.tool_calls == [
            {"function": {"name": "status", "arguments": "{}"}, "type": "function", "id": "call_0"}
        ]
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_stream_error_envelope_raises_instead_of_empty_final():
    body = _sse({"error": {"message": "no upstream channels", "type": "upstream"}})
    respx.post("http://stub/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(GatewayUpstreamError, match="no upstream channels"):
            async for _ in gw.chat_stream(
                GatewayRequest(model="m", messages=[{"role": "user", "content": "x"}])
            ):
                pass
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_stream_empty_success_raises_instead_of_empty_final():
    body = _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    respx.post("http://stub/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(GatewayUpstreamError, match="no content"):
            async for _ in gw.chat_stream(
                GatewayRequest(model="m", messages=[{"role": "user", "content": "x"}])
            ):
                pass
    finally:
        await gw.aclose()


# ---------------------------------------------------------------------------
# Agent loop with on_delta callback
# ---------------------------------------------------------------------------


class _EventLogStub:
    def __init__(self):
        self.events = []

    async def append(self, kind, payload, attrs=None):
        self.events.append((kind, payload))


@pytest.mark.asyncio
@respx.mock
async def test_agent_loop_on_delta_called_during_stream():
    body = _sse(
        {"choices": [{"index": 0, "delta": {"content": "Hi"}}]},
        {"choices": [{"index": 0, "delta": {"content": " there"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    respx.post("http://stub/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1")

    deltas: list[str] = []

    async def on_delta(t: str):
        deltas.append(t)

    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="m",
            system_prompt="s",
            user_prompt="u",
            allowed_tools=[],
            sender_tier="owner",
            event_log=_EventLogStub(),
            on_delta=on_delta,
        )
        assert deltas == ["Hi", " there"]
        assert resp.content == "Hi there"
        assert rounds == 1
        assert trace == []
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_agent_loop_falls_back_when_stream_fails():
    """If chat_stream breaks mid-flight, on_delta stops and we still get final."""
    body = _sse(
        {"choices": [{"index": 0, "delta": {"content": "Start"}}]},
    )  # missing [DONE]; will raise httpx.StreamConsumed or similar after last delta
    respx.post("http://stub/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
    gw = GatewayClient(base_url="http://stub/v1")

    deltas: list[str] = []

    async def on_delta(t: str):
        deltas.append(t)

    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="m",
            system_prompt="s",
            user_prompt="u",
            allowed_tools=[],
            sender_tier="owner",
            event_log=_EventLogStub(),
            on_delta=on_delta,
        )
        assert "Start" in resp.content
        # We should have gotten at least one delta before fallback.
        assert deltas == ["Start"]
        assert rounds == 1
        assert trace == []
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_agent_stream_transport_failure_uses_one_complete_recovery() -> None:
    """An ambiguous stream failure must not trigger three duplicate streams."""
    route = respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.ReadTimeout("stream response timed out"),
            httpx.Response(
                200,
                json={
                    "model": "m",
                    "choices": [
                        {"message": {"content": "Recovered once."}, "finish_reason": "stop"}
                    ],
                },
            ),
        ]
    )
    gateway = GatewayClient(base_url="http://stub/v1")

    async def on_delta(_text: str) -> None:
        return None

    try:
        response, _trace, _rounds, _state = await agent_loop.run_agent_loop(
            gateway=gateway,
            model="m",
            system_prompt="system",
            user_prompt="hello",
            allowed_tools=[],
            sender_tier="owner",
            event_log=_EventLogStub(),
            on_delta=on_delta,
        )
    finally:
        await gateway.aclose()

    assert response.content == "Recovered once."
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_split_discord_message_code_block_survives():
    md = "```python\nprint(1)\n```\n```bash\nls\n```"
    chunks = _split_discord_message(md, limit=200)
    assert chunks == [md]  # short enough, no split required


def test_split_discord_message_long_splits():
    text = "x " * 1500
    chunks = _split_discord_message(text, limit=50)
    for c in chunks:
        assert len(c) <= 50
