"""Regression tests for HTTP ingress lifecycle ownership."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from norax.adapter.http_in import AgentOsChatBridge, HttpInAdapter
from norax.context.window import RollingWindow
from norax.observability.metrics import Metrics


class _FakeServer:
    instances: list[_FakeServer] = []

    def __init__(self, _config) -> None:
        self._should_exit = False
        self.started = asyncio.Event()
        self.stop_requested = asyncio.Event()
        type(self).instances.append(self)

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._should_exit = value
        if value:
            self.stop_requested.set()

    async def serve(self) -> None:
        self.started.set()
        await self.stop_requested.wait()


class _FakeSocket:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[str] = []

    async def send_text(self, payload: str) -> None:
        if self.fail:
            raise ConnectionError("disconnected")
        self.messages.append(payload)


class _BlockingSocket:
    async def send_text(self, _payload: str) -> None:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_agent_os_bridge_mirrors_discord_conversation_roles():
    bridge = AgentOsChatBridge()
    live = _FakeSocket()
    dead = _FakeSocket(fail=True)
    bridge.attach(live)
    bridge.attach(dead)

    inbound = await bridge.broadcast_inbound(sender_label="Colby", text="hello")
    outbound = await bridge.broadcast_outbound(text="Hi—Norax here.")

    assert inbound["delivered"] == 1
    assert outbound["delivered"] == 1
    assert bridge.connected == 1
    assert [json.loads(item) for item in live.messages] == [
        {
            "type": "message",
            "source": "discord",
            "role": "user",
            "sender": "Colby",
            "text": "hello",
            "ts": json.loads(live.messages[0])["ts"],
        },
        {
            "type": "message",
            "source": "discord",
            "role": "assistant",
            "sender": "Norax",
            "text": "Hi—Norax here.",
            "ts": json.loads(live.messages[1])["ts"],
        },
    ]


@pytest.mark.asyncio
async def test_agent_os_bridge_isolates_slow_clients_and_reports_real_delivery(monkeypatch):
    monkeypatch.setattr("norax.adapter.http_in._WEBSOCKET_SEND_TIMEOUT_SECONDS", 0.01)
    bridge = AgentOsChatBridge()
    slow = _BlockingSocket()
    live = _FakeSocket()
    assert bridge.attach(slow) is True
    assert bridge.attach(live) is True

    result = await bridge.send("thread", "answer")

    assert result["ok"] is True
    assert result["delivered"] == 1
    assert bridge.connected == 1
    assert json.loads(live.messages[0])["text"] == "answer"

    bridge.detach(live)
    unavailable = await bridge.send("thread", "answer")
    assert unavailable == {
        "ok": False,
        "message_id": unavailable["message_id"],
        "delivered": 0,
        "error": "no_dashboard_connection",
    }


@pytest.mark.asyncio
async def test_agent_os_working_snapshot_tracks_overlapping_turns():
    bridge = AgentOsChatBridge()
    live = _FakeSocket()
    bridge.attach(live)

    await bridge.broadcast_status("typing")
    await bridge.broadcast_status("typing")
    await bridge.broadcast_status("idle")
    snapshot = _FakeSocket()
    bridge.attach(snapshot)
    await bridge.send_status_snapshot(snapshot)
    assert json.loads(snapshot.messages[-1])["status"] == "working"

    await bridge.broadcast_status("idle")
    final = _FakeSocket()
    bridge.attach(final)
    await bridge.send_status_snapshot(final)
    assert json.loads(final.messages[-1])["status"] == "idle"


@pytest.mark.asyncio
async def test_http_adapter_stop_is_graceful_and_restart_safe(monkeypatch):
    _FakeServer.instances.clear()
    monkeypatch.setattr("norax.adapter.http_in.Server", _FakeServer)
    adapter = HttpInAdapter(host="127.0.0.1", port=0)

    await adapter.start()
    first_task = adapter._task
    first_server = _FakeServer.instances[-1]
    await first_server.started.wait()

    await adapter.start()
    assert adapter._task is first_task
    assert len(_FakeServer.instances) == 1

    await adapter.stop()
    assert first_server.should_exit is True
    assert first_task is not None and first_task.done()
    assert adapter._task is None
    assert adapter._server is None

    await adapter.start()
    second_server = _FakeServer.instances[-1]
    assert second_server is not first_server
    await adapter.stop()


@pytest.mark.asyncio
async def test_http_adapter_stop_before_start_is_idempotent():
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    await adapter.stop()
    await adapter.stop()


@pytest.mark.asyncio
async def test_http_start_fails_closed_when_listener_never_binds(monkeypatch):
    class _FailingServer:
        started = False
        should_exit = False

        def __init__(self, _config) -> None:
            pass

        async def serve(self) -> None:
            raise OSError("bind failed")

    monkeypatch.setattr("norax.adapter.http_in.Server", _FailingServer)
    adapter = HttpInAdapter(host="127.0.0.1", port=4101)

    with pytest.raises(OSError, match="bind failed"):
        await adapter.start()
    assert adapter._task is None
    assert adapter._server is None


@pytest.mark.asyncio
async def test_http_event_stream_surfaces_listener_failure_after_start(monkeypatch):
    class _FailingServer:
        def __init__(self, _config) -> None:
            self.started = False
            self.should_exit = False
            self.fail = asyncio.Event()

        async def serve(self) -> None:
            self.started = True
            await self.fail.wait()
            raise ConnectionError("listener crashed")

    monkeypatch.setattr("norax.adapter.http_in.Server", _FailingServer)
    adapter = HttpInAdapter(host="127.0.0.1", port=4101)
    await adapter.start()
    assert isinstance(adapter._server, _FailingServer)
    waiting = asyncio.create_task(anext(adapter.events()))
    adapter._server.fail.set()

    with pytest.raises(RuntimeError, match="listener failed") as caught:
        await waiting
    assert isinstance(caught.value.__cause__, ConnectionError)
    await adapter.stop()


@pytest.mark.asyncio
async def test_http_status_reports_readiness():
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        response = await client.get("/status")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "starting"}


@pytest.mark.asyncio
async def test_http_ingress_accepts_with_current_metrics_interface():
    adapter = HttpInAdapter(host="127.0.0.1", port=0, metrics=Metrics())
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        response = await client.post("/ingress/test", json={"body": "hello"})

    assert response.status_code == 200
    assert response.json()["ok"] is True
    event = await asyncio.wait_for(anext(adapter.events()), timeout=1)
    assert event.body == "hello"
    assert event.sender.tier == "guest"
    assert event.trusted is False


@pytest.mark.asyncio
async def test_public_test_ingress_requires_configured_bearer_token():
    adapter = HttpInAdapter(host="0.0.0.0", port=0)
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        denied = await client.post("/ingress/test", json={"body": "hello"})
        adapter.configure_owner_chat(owner_id="owner", chat_token="secret")
        accepted = await client.post(
            "/ingress/test",
            headers={"Authorization": "bearer secret"},
            json={"body": "hello"},
        )

    assert denied.status_code == 401
    assert denied.json() == {"ok": False, "error": "unauthorized"}
    assert accepted.status_code == 200
    assert accepted.json()["ok"] is True


@pytest.mark.asyncio
async def test_ingress_rejects_new_work_while_stopping():
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    await adapter.stop()
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        response = await client.post("/ingress/test", json={"body": "hello"})

    assert response.status_code == 503
    assert response.json() == {"ok": False, "error": "ingress_stopping"}


@pytest.mark.asyncio
async def test_agent_os_ingress_requires_bearer_token_and_emits_owner_event():
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter.configure_owner_chat(owner_id="owner-1", chat_token="test-secret")
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        denied = await client.post("/ingress/agent_os", json={"body": "hello"})
        accepted = await client.post(
            "/ingress/agent_os",
            headers={"Authorization": "Bearer test-secret"},
            json={"body": "  hello  ", "thread_id": "dashboard"},
        )

    assert denied.status_code == 401
    assert denied.json() == {"ok": False, "error": "unauthorized"}
    assert accepted.status_code == 200
    assert accepted.json()["ok"] is True
    event = await asyncio.wait_for(anext(adapter.events()), timeout=1)
    assert event.body == "  hello  "
    assert event.sender.id == "owner-1"
    assert event.sender.tier == "owner"
    assert event.trusted is True
    assert event.thread_binding is not None
    assert event.thread_binding.thread_id == "dashboard"


@pytest.mark.asyncio
async def test_agent_os_history_reports_real_roles_and_bounds_content():
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter.configure_owner_chat(owner_id="owner", chat_token="test-secret")
    window = RollingWindow()
    turn = window.start_turn()
    window.add_user("question", turn_id=turn)
    window.add_tool_call(call_id="call", content="{}", turn_id=turn, name="read")
    window.add_tool_result(call_id="call", content="result", turn_id=turn)
    window.add_assistant("a" * 20_000, turn_id=turn)
    adapter._runtime_ref = SimpleNamespace(_get_window=lambda _channel: window)

    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/history/agent_os",
            headers={"Authorization": "Bearer test-secret"},
        )

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "question"
    assert len(messages[1]["text"]) == 16_384


@pytest.mark.asyncio
async def test_agent_os_ingress_rejects_malformed_and_oversized_fields():
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter.configure_owner_chat(owner_id="owner-1", chat_token="test-secret")
    headers = {"Authorization": "Bearer test-secret"}
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        malformed = await client.post(
            "/ingress/agent_os",
            headers={**headers, "content-type": "application/json"},
            content="{",
        )
        wrong_body = await client.post(
            "/ingress/agent_os", headers=headers, json={"body": {"nested": True}}
        )
        long_thread = await client.post(
            "/ingress/agent_os",
            headers=headers,
            json={"body": "hello", "thread_id": "x" * 257},
        )

    assert malformed.status_code == 400
    assert malformed.json()["error"] == "invalid_json"
    assert wrong_body.status_code == 400
    assert wrong_body.json()["error"] == "body_must_be_string"
    assert long_thread.status_code == 400
    assert long_thread.json()["error"] == "invalid_thread_id"


@pytest.mark.asyncio
async def test_http_adapter_rejects_request_body_above_global_limit():
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        response = await client.post("/ingress/test", content=b"x" * 1_048_577)

    assert response.status_code == 413
    assert response.json() == {"ok": False, "error": "request_body_too_large"}
