from __future__ import annotations

import base64
import copy
from typing import Any

import httpx
import pytest
import respx

import norax.gateway_client as gateway_module
from norax.gateway_client import (
    GatewayClient,
    GatewayRequest,
    GatewayRouter,
    GatewayUpstreamError,
    SpendGuard,
    SpendGuardTripped,
    _parse_openai,
    get_gateway,
)


def _response(content: str, **message_fields: Any) -> dict[str, Any]:
    return {
        "model": "local.gguf",
        "choices": [{"message": {"content": content, **message_fields}, "finish_reason": "stop"}],
    }


@pytest.mark.parametrize(
    ("base_url", "effort", "expected"),
    [
        (
            "http://127.0.0.1:11435/v1",
            "medium",
            {"enable_thinking": True, "reasoning_effort": "medium"},
        ),
        (
            "http://100.67.70.53:19136/v1",
            "high",
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
        ),
        ("http://127.0.0.1:11435/v1", "off", {"enable_thinking": False}),
    ],
)
@pytest.mark.asyncio
async def test_llama_cpp_effort_mapping_is_explicit_for_local_and_remote_relays(
    base_url: str,
    effort: str,
    expected: dict[str, Any],
) -> None:
    gateway = GatewayClient(base_url=base_url, provider_kind="openai")
    payload: dict[str, Any] = {"reasoning_effort": effort}
    try:
        gateway._apply_llama_cpp_thinking(payload, effort)
    finally:
        await gateway.aclose()

    assert payload["chat_template_kwargs"] == expected
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "https://llama.example.test/v1",
        "https://api.example.test/proxy/:11435/v1",
        "https://api.example.test:1913/v1?backend=:19136",
    ],
)
async def test_unrelated_openai_endpoint_is_not_treated_as_llama_cpp(base_url) -> None:
    gateway = GatewayClient(base_url=base_url, provider_kind="openai")
    payload: dict[str, Any] = {"reasoning_effort": "medium"}
    try:
        gateway._apply_llama_cpp_thinking(payload, "medium")
    finally:
        await gateway.aclose()

    assert payload == {"reasoning_effort": "medium"}


class _RouterClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.model_prefix = None
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"messages": {}},
        {"messages": [1]},
        {"messages": [{"role": 1}]},
        {"messages": [{"content": float("nan")}]},
        {"messages": [{1: "invalid key"}]},
        {"tools": {}},
        {"tools": [1]},
        {"tool_choice": 1},
        {"tool_choice": ""},
        {"metadata": []},
        {"metadata": {"reasoning_effort": 1}},
        {"metadata": {"_ollama_continuation_attempt": True}},
    ],
)
async def test_invalid_request_shapes_rejected_in_chat_and_stream(fields):
    gateway = GatewayClient(base_url="http://stub/v1")
    request = GatewayRequest(**({"model": "model", "messages": []} | fields))
    try:
        with pytest.raises((TypeError, ValueError)):
            await gateway.chat(request)
        with pytest.raises((TypeError, ValueError)):
            async for _ in gateway.chat_stream(request):
                pytest.fail("invalid requests must produce no stream events")
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_ollama_format_requires_explicit_constraint_and_opt_in() -> None:
    gateway = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    payload = {"model": "glm-5.1:cloud", "messages": []}
    schema = {"type": "object", "properties": {"score": {"type": "number"}}}
    try:
        assert "format" not in gateway._to_ollama_native_payload(payload)
        assert "format" not in gateway._to_ollama_native_payload(
            payload, metadata={"ollama_use_grammar": True}
        )
        for constraint in ("json", schema):
            actual = gateway._to_ollama_native_payload(
                payload, metadata={"ollama_use_grammar": True, "ollama_grammar": constraint}
            )
            assert actual["format"] == constraint
    finally:
        await gateway.aclose()


def test_router_rejects_invalid_topology_before_serving_requests() -> None:
    client = _RouterClient("http://default")

    with pytest.raises(ValueError, match="default provider"):
        GatewayRouter({"default": client}, [], "missing")
    with pytest.raises(ValueError, match="unknown provider"):
        GatewayRouter({"default": client}, [("other/*", "missing")], "default")


@pytest.mark.asyncio
async def test_router_owns_provider_snapshot_and_rejects_post_close_mutation() -> None:
    original = _RouterClient("http://original")
    providers = {"default": original}
    router = GatewayRouter(providers, [], "default")
    providers["default"] = _RouterClient("http://mutated-behind-router")

    assert router.route_for("model") == ("default", "http://original")
    await router.aclose()
    assert original.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        await router.remove_provider("other")


@pytest.mark.asyncio
async def test_standalone_gateway_defaults_to_loopback_ollama(monkeypatch) -> None:
    monkeypatch.delenv("NORAX_GATEWAY_URL", raising=False)
    gateway = get_gateway()
    try:
        assert gateway.base_url == "http://127.0.0.1:11434/v1"
        assert gateway.kind == "ollama"
    finally:
        await gateway.aclose()


def test_spend_guard_is_opt_in_and_zero_overhead_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("NORAX_LLM_MAX_CALLS_PER_MIN", raising=False)
    monkeypatch.delenv("NORAX_LLM_MAX_CALLS_PER_HOUR", raising=False)

    guard = SpendGuard()
    for _ in range(1_000):
        guard.record_and_check()

    assert guard.per_min == 0
    assert guard.per_hour == 0
    assert not guard._minute_stamps
    assert not guard._hour_stamps


def test_spend_guard_enforces_explicit_operator_limit(monkeypatch) -> None:
    monkeypatch.setenv("NORAX_LLM_MAX_CALLS_PER_MIN", "2")
    monkeypatch.setenv("NORAX_LLM_MAX_CALLS_PER_HOUR", "0")
    guard = SpendGuard()

    guard.record_and_check()
    guard.record_and_check()
    with pytest.raises(SpendGuardTripped) as exc_info:
        guard.record_and_check()

    assert exc_info.value.window == "minute"
    assert exc_info.value.count == 2


@pytest.mark.asyncio
@respx.mock
async def test_router_spend_guard_counts_actual_transport_once(monkeypatch) -> None:
    monkeypatch.setenv("NORAX_LLM_MAX_CALLS_PER_MIN", "1")
    monkeypatch.setenv("NORAX_LLM_MAX_CALLS_PER_HOUR", "0")
    monkeypatch.setattr(gateway_module, "_SPEND_GUARD", SpendGuard())
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response("ok"))
    )
    client = GatewayClient(base_url="http://stub/v1")
    router = GatewayRouter({"stub": client}, [], "stub")
    request = GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
    try:
        response = await router.chat(request)
        with pytest.raises(SpendGuardTripped):
            await router.chat(request)
    finally:
        await router.aclose()

    assert response.content == "ok"
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_non_idempotent_503_is_never_replayed() -> None:
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": {"message": "unavailable"}})
    )
    client = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(GatewayUpstreamError) as caught:
            await client.chat(
                GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
            )
    finally:
        await client.aclose()

    assert caught.value.status == 503
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_ambiguous_read_timeout_is_never_replayed() -> None:
    route = respx.post("http://stub/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("response timed out")
    )
    client = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(httpx.ReadTimeout):
            await client.chat(
                GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
            )
    finally:
        await client.aclose()

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_nonstream_response_body_is_bounded_before_json_allocation(monkeypatch) -> None:
    monkeypatch.setattr(gateway_module, "_MAX_CHAT_RESPONSE_BYTES", 64)
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b"{" + (b"x" * 128) + b"}")
    )
    client = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(GatewayUpstreamError, match="response exceeds 64 byte limit") as caught:
            await client.chat(
                GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
            )
    finally:
        await client.aclose()

    assert caught.value.status == 502
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_stream_line_and_total_allocation_are_bounded(monkeypatch) -> None:
    monkeypatch.setattr(gateway_module, "_MAX_STREAM_LINE_BYTES", 32)
    monkeypatch.setattr(gateway_module, "_MAX_STREAM_RESPONSE_BYTES", 128)
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b"data: " + (b"x" * 64) + b"\n")
    )
    client = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(GatewayUpstreamError, match="stream line exceeds 32 byte limit"):
            async for _event in client.chat_stream(
                GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
            ):
                pass
    finally:
        await client.aclose()

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_oversized_http_error_preserves_status_with_bounded_preview(monkeypatch) -> None:
    monkeypatch.setattr(gateway_module, "_MAX_UPSTREAM_ERROR_BYTES", 16)
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(503, content=b"unavailable:" + (b"x" * 1_000))
    )
    client = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(GatewayUpstreamError) as caught:
            await client.chat(
                GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
            )
    finally:
        await client.aclose()

    assert caught.value.status == 503
    assert len(caught.value.upstream_message) < 100
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_pre_send_connect_failure_gets_one_bounded_retry(monkeypatch) -> None:
    monkeypatch.setattr(gateway_module, "_SAFE_LLM_RETRY_DELAYS_MS", (0,))
    route = respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("connect failed"),
            httpx.Response(200, json=_response("ok")),
        ]
    )
    client = GatewayClient(base_url="http://stub/v1")
    try:
        response = await client.chat(
            GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
        )
    finally:
        await client.aclose()

    assert response.content == "ok"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_stream_required_503_is_never_replayed() -> None:
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": {"message": "unavailable"}})
    )
    client = GatewayClient(base_url="http://stub/v1", stream_required=True)
    try:
        with pytest.raises(GatewayUpstreamError) as caught:
            await client.chat(
                GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
            )
    finally:
        await client.aclose()

    assert caught.value.status == 503
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_ollama_image_normalization_is_bounded_native_and_non_mutating() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\nsmall-image"
    respx.get("https://cdn.example/image.png").mock(
        return_value=httpx.Response(200, content=image_bytes, headers={"content-type": "image/png"})
    )
    original = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://cdn.example/image.png"},
                },
            ],
        }
    ]
    snapshot = copy.deepcopy(original)
    client = GatewayClient(base_url="http://ollama/v1", provider_kind="ollama")
    try:
        normalized = await client._normalize_messages_for_provider(original, model="vision")
        native = client._to_ollama_native_payload(
            {"model": "vision", "messages": normalized, "stream": False}
        )
    finally:
        await client.aclose()

    assert original == snapshot
    assert native["messages"][0]["content"] == "inspect this"
    assert native["messages"][0]["images"] == [base64.b64encode(image_bytes).decode("ascii")]


@pytest.mark.asyncio
async def test_ollama_image_count_limit_fails_before_network() -> None:
    images = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,eA=="},
        }
        for _ in range(9)
    ]
    client = GatewayClient(base_url="http://ollama/v1", provider_kind="ollama")
    try:
        with pytest.raises(GatewayUpstreamError) as caught:
            await client._normalize_messages_for_provider(
                [{"role": "user", "content": images}], model="vision"
            )
    finally:
        await client.aclose()

    assert caught.value.status == 413


@pytest.mark.asyncio
@respx.mock
async def test_ollama_image_fetch_failure_is_explicit_not_silent_passthrough() -> None:
    respx.get("https://cdn.example/missing.png").mock(return_value=httpx.Response(404))
    client = GatewayClient(base_url="http://ollama/v1", provider_kind="ollama")
    try:
        with pytest.raises(GatewayUpstreamError) as caught:
            await client._normalize_messages_for_provider(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://cdn.example/missing.png"},
                            }
                        ],
                    }
                ],
                model="vision",
            )
    finally:
        await client.aclose()

    assert caught.value.status == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "failure_count"), [(400, 0), (503, 1)])
@respx.mock
async def test_stream_http_error_updates_circuit_exactly_once(
    status: int,
    failure_count: int,
) -> None:
    respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(status, json={"error": {"message": "upstream error"}})
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(GatewayUpstreamError):
            async for _event in gateway.chat_stream(
                GatewayRequest(model="model", messages=[{"role": "user", "content": "hi"}])
            ):
                pass
    finally:
        await gateway.aclose()

    assert len(gateway._breaker._failures) == failure_count


def test_openai_parser_does_not_allocate_a_throwaway_client(monkeypatch) -> None:
    def fail_init(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("parser must not construct an HTTP client")

    monkeypatch.setattr(GatewayClient, "__init__", fail_init)
    parsed = _parse_openai(
        _response(
            "",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": {"path": "README.md"}},
                }
            ],
        ),
        request_id="r1",
        fallback_model="local.gguf",
    )

    assert parsed.tool_calls[0]["function"]["arguments"] == '{"path": "README.md"}'


def test_text_tool_recovery_is_opt_in_declared_and_tool_only() -> None:
    tool_text = '<invoke name="read"><parameter name="path">README.md</parameter></invoke>'

    ordinary = _parse_openai(_response(tool_text), request_id="r1", fallback_model="local.gguf")
    recovered = _parse_openai(
        _response(tool_text),
        request_id="r2",
        fallback_model="local.gguf",
        text_tool_names={"read"},
    )
    quoted = _parse_openai(
        _response(f"Example: {tool_text}"),
        request_id="r3",
        fallback_model="local.gguf",
        text_tool_names={"read"},
    )
    undeclared = _parse_openai(
        _response(tool_text),
        request_id="r4",
        fallback_model="local.gguf",
        text_tool_names={"write"},
    )

    assert ordinary.tool_calls == []
    assert ordinary.content == tool_text
    assert recovered.content == ""
    assert recovered.tool_calls[0]["function"]["name"] == "read"
    assert quoted.tool_calls == []
    assert quoted.content.startswith("Example:")
    assert undeclared.tool_calls == []


def test_private_reasoning_is_removed_from_default_raw_response() -> None:
    parsed = _parse_openai(
        _response("Public answer", reasoning="private chain"),
        request_id="r1",
        fallback_model="m",
    )

    raw_message = parsed.raw["choices"][0]["message"]
    assert parsed.content == "Public answer"
    assert "reasoning" not in raw_message
    assert "private chain" not in str(parsed.raw)


def test_inline_private_reasoning_is_removed_from_default_raw_response() -> None:
    parsed = _parse_openai(
        _response("<think>private chain</think>Public answer"),
        request_id="r1",
        fallback_model="m",
    )

    assert parsed.content == "Public answer"
    assert "private chain" not in str(parsed.raw)
    assert parsed.raw["choices"][0]["message"]["content"] == "Public answer"


def test_openai_parser_preserves_finish_reason() -> None:
    response = _response("partial")
    response["choices"][0]["finish_reason"] = "length"

    parsed = _parse_openai(response, request_id="r1", fallback_model="m")

    assert parsed.metadata["finish_reason"] == "length"


@pytest.mark.asyncio
@respx.mock
async def test_stream_honors_explicit_token_budget_and_preserves_finish_reason() -> None:
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    final = None
    try:
        async for event in gateway.chat_stream(
            GatewayRequest(
                model="reasoning-model",
                messages=[{"role": "user", "content": "x"}],
                max_tokens=37,
                metadata={"reasoning_effort": "high"},
            )
        ):
            if event.kind == "final":
                final = event.response
    finally:
        await gateway.aclose()

    assert route.called
    assert route.calls[0].request.read()
    request_payload = __import__("json").loads(route.calls[0].request.content)
    assert request_payload["max_tokens"] == 37
    assert final is not None
    assert final.metadata["finish_reason"] == "length"


@pytest.mark.asyncio
@respx.mock
async def test_ollama_continuation_depth_is_bounded() -> None:
    route = respx.post("http://ollama/api/chat").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "model": "qwen3:8b",
                    "message": {"content": "part-1"},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                },
            ),
            httpx.Response(
                200,
                json={
                    "model": "qwen3:8b",
                    "message": {"content": "part-2"},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 11,
                    "eval_count": 3,
                },
            ),
            httpx.Response(
                200,
                json={
                    "model": "qwen3:8b",
                    "message": {"content": "part-3"},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 12,
                    "eval_count": 4,
                },
            ),
        ]
    )
    gateway = GatewayClient(
        base_url="http://ollama/v1",
        provider_kind="ollama",
    )
    try:
        response = await gateway.chat(
            GatewayRequest(
                model="qwen3:8b",
                messages=[{"role": "user", "content": "long answer"}],
            )
        )
    finally:
        await gateway.aclose()

    assert route.call_count == 3
    assert response.content == "part-1part-2part-3"
    assert response.metadata["continuation_attempt"] == 2
    assert response.metadata["done_reason"] == "length"
    assert response.usage == {"input_tokens": 33, "output_tokens": 9}


@pytest.mark.asyncio
@respx.mock
async def test_failed_ollama_continuation_never_replays_completed_first_call(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gateway_module, "_SAFE_LLM_RETRY_DELAYS_MS", (0,))
    route = respx.post("http://ollama/api/chat").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "model": "qwen3:8b",
                    "message": {"content": "already-generated"},
                    "done": True,
                    "done_reason": "length",
                },
            ),
            httpx.ConnectError("continuation connect failed"),
            httpx.ConnectError("continuation connect failed again"),
        ]
    )
    gateway = GatewayClient(base_url="http://ollama/v1", provider_kind="ollama")
    try:
        with pytest.raises(httpx.ConnectError):
            await gateway.chat(
                GatewayRequest(
                    model="qwen3:8b",
                    messages=[{"role": "user", "content": "long answer"}],
                )
            )
    finally:
        await gateway.aclose()

    # One completed generation plus two pre-send continuation attempts. A
    # fourth call here would be an expensive replay of the original request.
    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_gateway_rejects_unbounded_or_cyclic_requests_before_network(monkeypatch) -> None:
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response("should not run"))
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    try:
        monkeypatch.setattr(gateway_module, "_MAX_REQUEST_MESSAGES_BYTES", 32)
        with pytest.raises(ValueError, match="messages exceeds its byte limit"):
            await gateway.chat(
                GatewayRequest(
                    model="model",
                    messages=[{"role": "user", "content": "x" * 100}],
                )
            )

        monkeypatch.setattr(gateway_module, "_MAX_REQUEST_MESSAGES_BYTES", 64 * 1024 * 1024)
        cyclic: dict[str, Any] = {"role": "user"}
        cyclic["content"] = cyclic
        with pytest.raises(ValueError, match="reference cycle"):
            await gateway.chat(GatewayRequest(model="model", messages=[cyclic]))

        with pytest.raises(TypeError, match="non-JSON"):
            await gateway.chat(
                GatewayRequest(
                    model="model",
                    messages=[{"role": "user", "content": object()}],
                )
            )
    finally:
        await gateway.aclose()

    assert route.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_gateway_rejects_coerced_generation_controls_before_network() -> None:
    route = respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response("should not run"))
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    try:
        with pytest.raises(TypeError, match="reasoning_output"):
            await gateway.chat(
                GatewayRequest(
                    model="model",
                    messages=[],
                    metadata={"reasoning_output": "false"},
                )
            )
        with pytest.raises(ValueError, match="temperature"):
            await gateway.chat(
                GatewayRequest(model="model", messages=[], temperature=True)  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="max_tokens"):
            await gateway.chat(
                GatewayRequest(model="model", messages=[], max_tokens=True)  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="model"):
            await gateway.chat(GatewayRequest(model="bad model", messages=[]))
    finally:
        await gateway.aclose()

    assert route.call_count == 0


def test_gateway_router_rejects_invalid_model_before_route_matching() -> None:
    client = _RouterClient("http://default")
    router = GatewayRouter({"default": client}, [], "default")

    with pytest.raises(ValueError, match="model id"):
        router.route_for("")
    with pytest.raises(ValueError, match="model id"):
        router.route_for("bad model")
