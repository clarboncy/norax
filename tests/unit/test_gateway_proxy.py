import asyncio
import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from norax import gateway_proxy


def test_messages_proxy_preserves_named_sse_frames(monkeypatch):
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://upstream/v1")
    body = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n'
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n\n'
    )
    with respx.mock:
        respx.post("http://upstream/v1/messages").mock(
            return_value=httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )
        with TestClient(gateway_proxy.app) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert "event: message_start\ndata:" in response.text
    assert "\n\nevent: content_block_delta\ndata:" in response.text
    assert "event: message_start\n\ndata:" not in response.text


def test_chat_completions_aggregates_sse_for_non_stream_client(monkeypatch):
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://upstream/v1")
    sse_body = (
        'data: {"model":"gpt-5.5","choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
    )
    with respx.mock:
        respx.post("http://upstream/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )
        with TestClient(gateway_proxy.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.5",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "OK"


def test_messages_proxy_preserves_explicit_upstream_api_key(monkeypatch):
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://upstream/v1")
    monkeypatch.setenv("NORAX_GATEWAY_TOKEN", "fallback-token")
    with respx.mock:
        route = respx.post("http://upstream/v1/messages").mock(
            return_value=httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})
        )
        with TestClient(gateway_proxy.app) as client:
            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "dummy"},
                json={"model": "claude", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert response.status_code == 200
    assert route.calls.last.request.headers["x-api-key"] == "dummy"
    assert "authorization" not in route.calls.last.request.headers


def test_chat_proxy_uses_generic_fallback_token_without_rewriting_model(monkeypatch):
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://upstream/v1")
    monkeypatch.setenv("NORAX_GATEWAY_TOKEN", "provider-token")
    sse_body = b'data: {"choices":[{"delta":{"content":"OK"}}],"model":"claude-sonnet-4-6"}\n\ndata: [DONE]\n\n'
    with respx.mock:
        route = respx.post("http://upstream/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )
        with TestClient(gateway_proxy.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "provider/claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
    assert response.status_code == 200
    assert route.calls[0].request.headers["authorization"] == "Bearer provider-token"
    assert json.loads(route.calls[0].request.content)["model"] == "provider/claude-sonnet-4-6"
    assert response.json()["choices"][0]["message"]["content"] == "OK"


def test_health_alias_exists():
    with TestClient(gateway_proxy.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_models_rejects_obsolete_group_routing_without_calling_upstream(monkeypatch):
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://upstream/v1")
    with respx.mock:
        route = respx.get("http://upstream/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "claude-sonnet-4-6"}]})
        )
        with TestClient(gateway_proxy.app) as client:
            response = client.get("/v1/models?group=legacy")
    assert response.status_code == 400
    assert route.called is False


class _FakeStreamResponse:
    def __init__(self, chunks, *, status_code=200):
        self.status_code = status_code
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def aiter_bytes(self):
        async for chunk in self._chunks:
            yield chunk


class _FakeStreamClient:
    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.calls = 0

    def stream(self, method, url, **kwargs):
        self.calls += 1
        return self.response_factory()


@pytest.mark.asyncio
async def test_stream_response_forwards_first_chunk_before_upstream_finishes(monkeypatch):
    release = asyncio.Event()

    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
        await release.wait()
        yield b"data: [DONE]\n\n"

    fake_client = _FakeStreamClient(lambda: _FakeStreamResponse(chunks()))
    monkeypatch.setattr(gateway_proxy, "_get_stream_client", lambda: fake_client)
    response = await gateway_proxy._stream_response({}, b"{}", "model", 0.0)
    iterator = response.body_iterator.__aiter__()

    first = await asyncio.wait_for(anext(iterator), timeout=0.2)
    assert first.endswith(b'first"}}]}\n\n')
    assert not release.is_set()

    release.set()
    assert await asyncio.wait_for(anext(iterator), timeout=0.2) == b"data: [DONE]\n\n"
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


@pytest.mark.asyncio
async def test_stream_response_never_retries_after_delivering_bytes(monkeypatch):
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
        raise httpx.ReadError('provider said "broken"\non a new line')

    fake_client = _FakeStreamClient(lambda: _FakeStreamResponse(chunks()))
    monkeypatch.setattr(gateway_proxy, "_get_stream_client", lambda: fake_client)
    monkeypatch.setattr(gateway_proxy, "MAX_RETRIES", 2)
    response = await gateway_proxy._stream_response({}, b"{}", "model", 0.0)
    chunks_out = [chunk async for chunk in response.body_iterator]

    assert fake_client.calls == 1
    error_line = [line for line in b"".join(chunks_out).splitlines() if line.startswith(b"data:")][
        -1
    ]
    error = json.loads(error_line.removeprefix(b"data:").strip())
    assert error["error"]["type"] == "proxy_error"
    assert '"broken"' in error["error"]["message"]


def test_non_stream_sse_error_is_not_reported_as_empty_success(monkeypatch):
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://upstream/v1")
    with respx.mock:
        respx.post("http://upstream/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=b'event: error\ndata: {"error":{"message":"provider failed"}}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        )
        with TestClient(gateway_proxy.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.5",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "provider failed"


def test_request_body_limit_is_enforced_before_upstream(monkeypatch):
    monkeypatch.setattr(gateway_proxy, "MAX_REQUEST_BYTES", 64)
    with TestClient(gateway_proxy.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "x" * 200}],
            },
        )
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "request_too_large"


def test_context_trimming_keeps_assistant_tool_exchange_atomic(monkeypatch):
    monkeypatch.setitem(gateway_proxy._CONTEXT_LIMITS, "gpt", 700)
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "old" * 1_000},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_recent",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_recent", "content": "result"},
            {"role": "user", "content": "use that result"},
        ],
    }

    assert gateway_proxy._trim_payload_messages(payload) is True
    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["system", "system", "assistant", "tool", "user"]
    assert payload["messages"][2]["tool_calls"][0]["id"] == "call_recent"
    assert payload["messages"][3]["tool_call_id"] == "call_recent"


def test_required_context_that_cannot_fit_returns_413(monkeypatch):
    monkeypatch.setitem(gateway_proxy._CONTEXT_LIMITS, "gpt", 200)
    with TestClient(gateway_proxy.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "required" * 100},
                    {"role": "user", "content": "hi"},
                ],
            },
        )
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "context_length_exceeded"


def test_model_catalog_is_single_upstream_and_deduplicated(monkeypatch):
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://upstream/v1")
    monkeypatch.setenv("NORAX_GATEWAY_TOKEN", "provider-token")
    with respx.mock:
        route = respx.get("http://upstream/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "model-a"}, {"id": "model-a"}, {"id": "model-b"}]},
            )
        )
        with TestClient(gateway_proxy.app) as client:
            response = client.get("/v1/models")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["model-a", "model-b"]
    assert len(route.calls) == 1
    assert route.calls[0].request.headers["authorization"] == "Bearer provider-token"


def test_readyz_uses_authenticated_upstream_probe(monkeypatch):
    observed = []

    async def fake_fetch(client, headers):
        observed.append(headers)
        return []

    monkeypatch.setenv("NORAX_GATEWAY_TOKEN", "ready-token")
    monkeypatch.setattr(gateway_proxy, "_fetch_upstream_models", fake_fetch)
    monkeypatch.setattr(gateway_proxy, "_readiness_cache", None)
    with TestClient(gateway_proxy.app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert observed == [{"Authorization": "Bearer ready-token"}]


def test_health_does_not_leak_upstream_url(monkeypatch):
    """/health must not expose the internal upstream URL to public callers."""
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://secret-upstream:11434/v1")
    with TestClient(gateway_proxy.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "upstream" not in body
    assert "secret-upstream" not in json.dumps(body)


def test_readyz_does_not_leak_upstream_url(monkeypatch):
    """/readyz must not expose the internal upstream URL to public callers."""
    monkeypatch.setattr(gateway_proxy, "UPSTREAM", "http://secret-upstream:11434/v1")
    monkeypatch.setenv("NORAX_GATEWAY_TOKEN", "ready-token")

    async def fake_fetch(client, headers):
        return []

    monkeypatch.setattr(gateway_proxy, "_fetch_upstream_models", fake_fetch)
    monkeypatch.setattr(gateway_proxy, "_readiness_cache", None)
    with TestClient(gateway_proxy.app) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert "upstream" not in body
    assert "secret-upstream" not in json.dumps(body)
