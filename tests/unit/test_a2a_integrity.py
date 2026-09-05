"""Interoperability and lifecycle tests for the A2A boundary."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from norax.a2a.client import A2AClient, _task_result
from norax.a2a.server import (
    CANCELED,
    COMPLETED,
    FAILED,
    MAX_MESSAGE_CHARS,
    A2AError,
    A2ATask,
    NoraxA2AServer,
    create_a2a_app,
)
from norax.runtime.core import _a2a_advertised_url, _bounded_config_int, _is_loopback_host
from norax.runtime.outbound import OutboundRegistry


class _ReplyingRuntime:
    def __init__(self) -> None:
        self.outbound = OutboundRegistry()
        self.envelopes: list[Any] = []

    async def _handle_turn(self, env: Any) -> None:
        self.envelopes.append(env)
        assert env.thread_binding is not None
        result = await self.outbound.send(
            env.source,
            env.thread_binding.thread_id,
            f"answer: {env.body}",
        )
        assert result["ok"] is True


class _SilentRuntime:
    def __init__(self) -> None:
        self.outbound = OutboundRegistry()

    async def _handle_turn(self, _env: Any) -> None:
        return None


class _FailingRuntime:
    def __init__(self) -> None:
        self.outbound = OutboundRegistry()

    async def _handle_turn(self, _env: Any) -> None:
        raise RuntimeError("private filesystem detail")


class _SlowRuntime:
    def __init__(self) -> None:
        self.outbound = OutboundRegistry()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def _handle_turn(self, env: Any) -> None:
        self.started.set()
        try:
            await self.release.wait()
            assert env.thread_binding is not None
            await self.outbound.send(env.source, env.thread_binding.thread_id, "late answer")
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def _message(
    text: str = "inspect this",
    *,
    context_id: str = "",
    task_id: str = "",
    return_immediately: bool = False,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "messageId": "message-1",
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if context_id:
        message["contextId"] = context_id
    if task_id:
        message["taskId"] = task_id
    return {
        "message": message,
        "configuration": {"returnImmediately": return_immediately},
    }


def _state(body: dict[str, Any]) -> str:
    return str(body["task"]["status"]["state"])


@pytest.mark.asyncio
async def test_http_json_round_trip_captures_real_runtime_reply_and_isolates_context() -> None:
    runtime = _ReplyingRuntime()
    server = NoraxA2AServer(runtime, "http://agent.test/a2a")
    transport = httpx.ASGITransport(app=create_a2a_app(server))

    async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
        card_response = await client.get("/.well-known/agent-card.json")
        assert card_response.status_code == 200
        assert card_response.headers["content-type"].startswith("application/a2a+json")
        assert card_response.headers["cache-control"] == "public, max-age=300"
        card = card_response.json()
        assert card["capabilities"] == {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        }
        assert card["supportedInterfaces"][0] == {
            "url": "http://agent.test/a2a",
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }
        assert all(
            "policy" in skill["description"].lower() or skill["id"] == "research"
            for skill in card["skills"]
        )

        response = await client.post(
            "/message:send",
            json=_message("audit the code", context_id="context-one"),
        )
        assert response.status_code == 200
        body = response.json()
        assert _state(body) == COMPLETED
        assert body["task"]["contextId"] == "context-one"
        assert body["task"]["artifacts"][0]["parts"] == [{"text": "answer: audit the code"}]
        task_id = body["task"]["id"]

        fetched = await client.get(f"/tasks/{task_id}")
        assert fetched.json() == body
        listed = await client.get("/tasks", params={"contextId": "context-one"})
        assert listed.json()["tasks"] == [body["task"]]

    env = runtime.envelopes[0]
    assert env.source == "a2a"
    assert env.raw == {"channel_id": "context-one"}
    assert env.thread_binding.thread_id == task_id
    assert env.thread_binding.session_id == "context-one"
    assert env.sender.tier == "guest"
    assert env.sender.trust is False
    assert env.trusted is False
    assert not server._reply_futures


@pytest.mark.asyncio
async def test_bundled_client_and_server_interoperate_over_standard_routes() -> None:
    runtime = _ReplyingRuntime()
    server = NoraxA2AServer(runtime, "http://agent.test")
    transport = httpx.ASGITransport(app=create_a2a_app(server))
    client = A2AClient("http://agent.test")
    client._client = httpx.AsyncClient(transport=transport, base_url="http://agent.test")

    try:
        card = await client.get_agent_card()
        assert card["name"] == "Norax"
        sent = await client.send_task(
            "run tests",
            context_id="client-context",
            skill_id="code",
        )
        assert sent.status == "completed"
        assert sent.result == "answer: run tests"
        assert sent.context_id == "client-context"
        assert sent.id

        fetched = await client.get_task(sent.id)
        assert fetched == sent
        recancel = await client.cancel_task(sent.id)
        assert recancel.status == "error"
        assert "no longer cancelable" in recancel.error
    finally:
        await client.close()
    assert client._client is None


@pytest.mark.asyncio
async def test_nonblocking_task_can_be_polled_and_actually_canceled() -> None:
    runtime = _SlowRuntime()
    server = NoraxA2AServer(runtime, "http://agent.test")
    transport = httpx.ASGITransport(app=create_a2a_app(server))

    async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
        submitted = await client.post(
            "/message:send",
            json=_message(return_immediately=True),
        )
        assert submitted.status_code == 200
        task_id = submitted.json()["task"]["id"]
        await asyncio.wait_for(runtime.started.wait(), timeout=1)

        polled = await client.get(f"/tasks/{task_id}")
        assert _state(polled.json()) == "TASK_STATE_WORKING"
        canceled = await client.post(f"/tasks/{task_id}:cancel")
        assert canceled.status_code == 200
        assert _state(canceled.json()) == CANCELED
        await asyncio.wait_for(runtime.cancelled.wait(), timeout=1)
        assert task_id not in server._task_futures


@pytest.mark.asyncio
async def test_server_close_cancels_workers_rejects_new_work_and_detaches_outbound() -> None:
    runtime = _SlowRuntime()
    server = NoraxA2AServer(runtime, "http://agent.test")
    submitted = await server.send_message(_message(return_immediately=True))
    task_id = submitted["task"]["id"]
    await asyncio.wait_for(runtime.started.wait(), timeout=1)

    await server.close()
    await server.close()

    await asyncio.wait_for(runtime.cancelled.wait(), timeout=1)
    assert server._tasks[task_id].status == CANCELED
    assert not server._task_futures
    assert not server._reply_futures
    assert not runtime.outbound.has("a2a")
    with pytest.raises(A2AError, match="shutting down") as caught:
        await server.send_message(_message())
    assert caught.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", [_SilentRuntime(), _FailingRuntime()])
async def test_missing_or_failed_runtime_reply_is_truthfully_failed_without_detail_leak(
    runtime: Any,
) -> None:
    server = NoraxA2AServer(runtime, "http://agent.test")
    result = await server.send_message(_message())
    assert _state(result) == FAILED
    task = result["task"]
    assert "private filesystem detail" not in json.dumps(task)
    assert task["status"]["message"]["parts"] == [{"text": "Task execution failed"}]
    assert "artifacts" not in task


@pytest.mark.asyncio
async def test_authentication_is_advertised_and_enforced_with_constant_time_path() -> None:
    runtime = _ReplyingRuntime()
    server = NoraxA2AServer(runtime, "http://agent.test", auth_token="secret-token")
    transport = httpx.ASGITransport(app=create_a2a_app(server))

    async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
        card = (await client.get("/.well-known/agent-card.json")).json()
        assert "bearer" in card["securitySchemes"]
        assert card["securityRequirements"]

        missing = await client.post("/message:send", json=_message())
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"] == "Bearer"
        wrong = await client.post(
            "/message:send",
            json=_message(),
            headers={"Authorization": "Basic secret-token"},
        )
        assert wrong.status_code == 401
        accepted = await client.post(
            "/message:send",
            json=_message(),
            headers={"Authorization": "Bearer secret-token"},
        )
        assert accepted.status_code == 200
        unsupported_version = await client.post(
            "/message:send",
            json=_message(),
            headers={
                "Authorization": "Bearer secret-token",
                "A2A-Version": "0.3",
            },
        )
        assert unsupported_version.status_code == 400
        assert unsupported_version.json()["title"] == "Version Not Supported"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({}, 400),
        ({"message": []}, 400),
        ({"message": {"role": "ROLE_AGENT", "parts": [{"text": "x"}]}}, 400),
        ({"message": {"role": "ROLE_USER", "parts": []}}, 400),
        ({"message": {"role": "ROLE_USER", "parts": [{"url": "file"}]}}, 415),
        (_message("x" * (MAX_MESSAGE_CHARS + 1)), 413),
        ({"message": {"role": "ROLE_USER", "parts": [{"text": "x"}]}, "configuration": []}, 400),
    ],
)
async def test_http_boundary_rejects_invalid_or_unsupported_messages(
    payload: dict[str, Any], expected_status: int
) -> None:
    server = NoraxA2AServer(_ReplyingRuntime(), "http://agent.test")
    transport = httpx.ASGITransport(app=create_a2a_app(server))
    async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
        response = await client.post("/message:send", json=payload)
    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/problem+json")


@pytest.mark.asyncio
async def test_missing_tasks_and_unsupported_streaming_are_explicit_protocol_errors() -> None:
    server = NoraxA2AServer(_ReplyingRuntime(), "http://agent.test")
    transport = httpx.ASGITransport(app=create_a2a_app(server))
    async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
        missing = await client.get("/tasks/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["title"] == "Task Not Found"
        stream = await client.post("/message:stream", json=_message())
        assert stream.status_code == 501
        subscribe = await client.post("/tasks/abc:subscribe")
        assert subscribe.status_code == 501


@pytest.mark.asyncio
async def test_task_storage_evicts_old_terminal_records_and_bounds_active_capacity() -> None:
    server = NoraxA2AServer(_ReplyingRuntime(), "http://agent.test", max_tasks=2)
    first = (await server.send_message(_message("first")))["task"]["id"]
    second = (await server.send_message(_message("second")))["task"]["id"]
    third = (await server.send_message(_message("third")))["task"]["id"]
    assert set(server._tasks) == {second, third}
    assert first not in server._tasks

    slow = _SlowRuntime()
    busy_server = NoraxA2AServer(slow, "http://busy.test", max_tasks=1)
    initial = await busy_server.send_message(_message(return_immediately=True))
    task_id = initial["task"]["id"]
    await asyncio.wait_for(slow.started.wait(), timeout=1)
    with pytest.raises(Exception, match="capacity is currently exhausted"):
        await busy_server.send_message(_message("another", return_immediately=True))
    await busy_server.cancel_task(task_id)


@pytest.mark.asyncio
async def test_list_tasks_filters_and_uses_stable_cursor_pagination() -> None:
    server = NoraxA2AServer(_ReplyingRuntime(), "http://agent.test")
    await server.send_message(_message("one", context_id="shared"))
    await server.send_message(_message("two", context_id="shared"))
    await server.send_message(_message("other", context_id="different"))

    page_one = await server.list_tasks(context_id="shared", page_size=1)
    assert page_one["totalSize"] == 2
    assert page_one["pageSize"] == 1
    assert page_one["nextPageToken"]
    page_two = await server.list_tasks(
        context_id="shared",
        page_size=1,
        page_token=page_one["nextPageToken"],
    )
    assert page_two["pageSize"] == 1
    assert page_two["nextPageToken"] == ""
    assert page_two["tasks"][0]["id"] != page_one["tasks"][0]["id"]

    completed = await server.list_tasks(status=COMPLETED)
    assert completed["totalSize"] == 3
    with pytest.raises(Exception, match="pageSize"):
        await server.list_tasks(page_size=0)
    with pytest.raises(Exception, match="pageToken"):
        await server.list_tasks(page_token="not-a-cursor")


@pytest.mark.asyncio
async def test_current_and_legacy_jsonrpc_routes_work_without_fake_streaming() -> None:
    server = NoraxA2AServer(_ReplyingRuntime(), "http://agent.test")
    modern = await server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": _message("modern"),
        }
    )
    assert modern["result"]["task"]["status"]["state"] == COMPLETED
    task_id = modern["result"]["task"]["id"]

    fetched = await server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 2, "method": "GetTask", "params": {"id": task_id}}
    )
    assert fetched["result"]["task"]["id"] == task_id
    listed = await server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 3, "method": "ListTasks", "params": {}}
    )
    assert listed["result"]["totalSize"] == 1

    legacy = await server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tasks/send",
            "params": {
                "id": "client-chosen-id",
                "message": {"parts": [{"type": "text", "text": "legacy"}]},
            },
        }
    )
    assert legacy["result"]["status"] == "completed"
    assert legacy["result"]["id"] != "client-chosen-id"
    assert legacy["result"]["result"] == "answer: legacy"

    unsupported = await server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 5, "method": "SendStreamingMessage", "params": {}}
    )
    assert unsupported["error"]["data"]["reason"] == "UNSUPPORTED_OPERATION"
    unknown = await server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 6, "method": "NoSuchMethod", "params": {}}
    )
    assert unknown["error"]["code"] == -32601
    malformed = await server.handle_jsonrpc([])
    assert malformed["error"]["code"] == -32600


def test_client_validation_and_response_parsing_are_strict() -> None:
    for bad_url in ["", "ftp://agent.test", "http://user:pass@agent.test", "http://agent.test?q=1"]:
        with pytest.raises((TypeError, ValueError)):
            A2AClient(bad_url)
    for bad_timeout in [True, 0, -1, float("nan"), float("inf"), 3_601]:
        with pytest.raises(ValueError):
            A2AClient("http://agent.test", timeout=bad_timeout)

    client = A2AClient("http://agent.test")
    with pytest.raises(ValueError, match="message"):
        asyncio.run(client.send_task(" "))
    invalid = client._parse_sse_event("not-json", "task-1")
    assert invalid.status == "error"
    status = client._parse_sse_event(
        json.dumps(
            {
                "statusUpdate": {
                    "taskId": "task-1",
                    "contextId": "context-1",
                    "status": {"state": CANCELED},
                }
            }
        )
    )
    assert status.status == "canceled"
    artifact = client._parse_sse_event(
        json.dumps(
            {
                "artifactUpdate": {
                    "taskId": "task-1",
                    "artifact": {"parts": [{"text": "partial"}]},
                }
            }
        )
    )
    assert artifact.status == "working"
    assert artifact.result == "partial"


@pytest.mark.asyncio
async def test_client_agent_card_legacy_fallback_and_error_body_handling() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith(".json"):
            return httpx.Response(404, json={"detail": "missing"})
        if request.url.path.endswith("agent-card"):
            return httpx.Response(200, json={"name": "Legacy"})
        return httpx.Response(503, json={"detail": "busy"})

    client = A2AClient("http://agent.test", token="token")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await client.get_agent_card() == {"name": "Legacy"}
        result = await client.send_task("hello")
        assert result.status == "error"
        assert result.error == "busy"
    finally:
        await client.close()
    assert paths[:2] == [
        "/.well-known/agent-card.json",
        "/.well-known/agent-card",
    ]


@pytest.mark.asyncio
async def test_client_streaming_consumes_standard_sse_task_and_update_wrappers() -> None:
    stream_body = (
        'data: {"task":{"id":"task-1","contextId":"context-1",'
        '"status":{"state":"TASK_STATE_WORKING"}}}\n\n'
        ": keepalive\n\n"
        'data: {"artifactUpdate":{"taskId":"task-1","contextId":"context-1",'
        '"artifact":{"parts":[{"text":"partial"}]}}}\n\n'
        'data: {"statusUpdate":{"taskId":"task-1","contextId":"context-1",'
        '"status":{"state":"TASK_STATE_COMPLETED"}}}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/message:stream"
        assert request.headers["accept"] == "text/event-stream"
        assert request.headers["a2a-version"] == "1.0"
        return httpx.Response(
            200,
            content=stream_body,
            headers={"Content-Type": "text/event-stream"},
        )

    client = A2AClient("http://agent.test")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        events = [event async for event in client.send_task_streaming("work")]
    finally:
        await client.close()
    assert [(event.status, event.result) for event in events] == [
        ("working", ""),
        ("working", "partial"),
        ("completed", ""),
    ]
    assert all(event.id == "task-1" for event in events)
    assert all(event.context_id == "context-1" for event in events)


@pytest.mark.asyncio
async def test_client_streaming_bounds_malformed_events_and_surfaces_http_errors() -> None:
    oversized = "x" * 512_001
    responses = iter(
        [
            httpx.Response(
                200,
                content=f"data: not-json\n\ndata: {oversized}\n\n",
                headers={"Content-Type": "text/event-stream"},
            ),
            httpx.Response(501, json={"detail": "unsupported"}),
        ]
    )
    client = A2AClient("http://agent.test")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: next(responses))
    )
    try:
        events = [event async for event in client.send_task_streaming("work", task_id="task-1")]
        assert [event.error for event in events] == [
            "invalid JSON in A2A stream",
            "A2A stream event exceeded the size limit",
        ]
        with pytest.raises(httpx.HTTPStatusError):
            _ = [event async for event in client.send_task_streaming("work")]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_client_context_manager_and_all_public_argument_guards() -> None:
    async with A2AClient("http://agent.test") as client:
        assert client._client is not None
        with pytest.raises(TypeError, match="message"):
            await client.send_task(1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="character limit"):
            await client.send_task("x" * (MAX_MESSAGE_CHARS + 1))
        with pytest.raises(TypeError, match="task_id"):
            await client.send_task("x", task_id=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="task_id"):
            await client.send_task("x", task_id="")
        with pytest.raises(ValueError, match="context_id"):
            await client.send_task("x", context_id="x" * 257)
        with pytest.raises(ValueError, match="skill_id"):
            await client.send_task("x", skill_id=" ")
        with pytest.raises(TypeError, match="return_immediately"):
            await client.send_task("x", return_immediately=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="task_id"):
            await client.get_task("")
        with pytest.raises(ValueError, match="task_id"):
            await client.cancel_task("")
        with pytest.raises(ValueError, match="context_id"):
            _ = [event async for event in client.send_task_streaming("x", context_id="")]
    assert client._client is None


def test_client_normalizes_all_response_shapes_and_failure_messages() -> None:
    nested = _task_result(
        {
            "result": {
                "task": {
                    "id": "task-1",
                    "contextId": "context-1",
                    "status": {
                        "state": FAILED,
                        "message": {"parts": [{"text": "safe failure"}]},
                    },
                }
            }
        }
    )
    assert nested.status == "failed"
    assert nested.error == "safe failure"

    legacy = _task_result(
        {
            "result": {
                "id": "legacy",
                "status": "completed",
                "result": "legacy answer",
                "error": "",
            }
        }
    )
    assert legacy.result == "legacy answer"
    direct = _task_result(
        {
            "message": {
                "messageId": "direct",
                "contextId": "context",
                "parts": [{"text": "direct answer"}],
            }
        }
    )
    assert direct.status == "completed"
    assert direct.result == "direct answer"
    protocol_error = _task_result({"error": {"message": "denied"}}, fallback_id="fallback")
    assert protocol_error.error == "denied"
    assert _task_result([], fallback_id="fallback").error == "invalid A2A response body"
    assert _task_result({}, fallback_id="fallback").error == "unrecognized A2A response body"
    assert _task_result({"error": {}}, fallback_id="fallback").error == "A2A request failed"


@pytest.mark.asyncio
async def test_client_rejects_non_object_agent_card_and_handles_plain_text_http_error() -> None:
    responses = iter(
        [
            httpx.Response(200, json=[]),
            httpx.Response(502, text="upstream unavailable"),
        ]
    )
    client = A2AClient("http://agent.test")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: next(responses))
    )
    try:
        with pytest.raises(ValueError, match="JSON object"):
            await client.get_agent_card()
        result = await client.send_task("hello")
        assert result.status == "error"
        assert result.error == "A2A request failed with HTTP 502"
    finally:
        await client.close()


def test_task_serialization_and_runtime_listener_guards() -> None:
    task = A2ATask("task", "context")
    task.fail("safe failure")
    body = task.to_dict()
    assert body["status"]["state"] == FAILED
    assert body["status"]["timestamp"].endswith("Z")
    assert task.to_legacy_dict()["status"] == "failed"

    assert _is_loopback_host("127.0.0.1")
    assert _is_loopback_host("[::1]")
    assert _is_loopback_host("localhost")
    assert not _is_loopback_host("0.0.0.0")
    assert not _is_loopback_host("agent.internal")
    assert _a2a_advertised_url("127.0.0.1", 8766, None) == "http://127.0.0.1:8766"
    assert _a2a_advertised_url("::1", 8766, None) == "http://[::1]:8766"
    assert _a2a_advertised_url("0.0.0.0", 8766, "https://agent.example/a2a") == (
        "https://agent.example/a2a"
    )
    with pytest.raises(ValueError, match="base_url"):
        _a2a_advertised_url("0.0.0.0", 8766, None)
    assert _bounded_config_int("+8766", label="port", minimum=1, maximum=65_535) == 8766
    for bad_value in (True, 1.5, "1.5", "", 0, 65_536):
        with pytest.raises(ValueError, match="port"):
            _bounded_config_int(bad_value, label="port", minimum=1, maximum=65_535)


def test_server_constructor_rejects_invalid_limits_urls_and_adapter_collisions() -> None:
    with pytest.raises(ValueError):
        NoraxA2AServer(_ReplyingRuntime(), "http://agent.test", max_tasks=0)
    with pytest.raises(ValueError):
        NoraxA2AServer(_ReplyingRuntime(), "http://agent.test", max_tasks=True)
    with pytest.raises(ValueError):
        NoraxA2AServer(_ReplyingRuntime(), "http://agent.test", max_concurrent=0)
    with pytest.raises(ValueError):
        NoraxA2AServer(_ReplyingRuntime(), "http://agent.test", max_concurrent=True)
    with pytest.raises(ValueError):
        NoraxA2AServer(_ReplyingRuntime(), "file:///tmp/agent")

    runtime = _ReplyingRuntime()
    NoraxA2AServer(runtime, "http://agent.test")
    with pytest.raises(RuntimeError, match="already has"):
        NoraxA2AServer(runtime, "http://agent.test")
