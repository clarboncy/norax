"""MCP transport lifecycle ownership regressions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types

from norax.brain.agent_loop import _run_one_tool
from norax.mcp.client import MCPConnection, NoraxMCPClient
from norax.runtime.circuit_breaker import get_circuit_registry


class _Context:
    def __init__(self, value) -> None:
        self.value = value
        self.enter_task = None
        self.exit_task = None

    async def __aenter__(self):
        self.enter_task = asyncio.current_task()
        return self.value

    async def __aexit__(self, *_args):
        self.exit_task = asyncio.current_task()


class _Session(_Context):
    def __init__(self, _read, _write) -> None:
        super().__init__(self)

    async def initialize(self) -> None:
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=[])


class _EventLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def append(self, kind: str, payload: dict) -> None:
        self.events.append((kind, payload))


@pytest.mark.asyncio
async def test_stdio_transport_enters_and_exits_in_owner_task(monkeypatch):
    transport = _Context((object(), object()))
    sessions: list[_Session] = []

    def session_factory(read, write):
        session = _Session(read, write)
        sessions.append(session)
        return session

    monkeypatch.setattr("norax.mcp.client.stdio_client", lambda _params: transport)
    monkeypatch.setattr("norax.mcp.client.ClientSession", session_factory)
    client = NoraxMCPClient()

    await client.connect_stdio("filesystem", "fake-server")
    await client.disconnect_all()

    assert transport.enter_task is transport.exit_task
    assert sessions[0].enter_task is sessions[0].exit_task
    assert client.list_connections() == []


@pytest.mark.asyncio
async def test_mcp_protocol_error_cannot_report_success():
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text='{"ok": true, "message": "boom"}')],
                isError=True,
            )
        )
    )
    owner = asyncio.current_task()
    assert owner is not None
    connection = MCPConnection(
        "broken",
        session,
        close_event=asyncio.Event(),
        owner_task=owner,
    )

    result = await connection.call_tool("explode")

    assert result["ok"] is False
    assert result["is_error"] is True
    assert result["error"] == "mcp_tool_error"


@pytest.mark.asyncio
async def test_mcp_success_without_custom_ok_is_normalized():
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text='{"value": 42}')],
                isError=False,
            )
        )
    )
    owner = asyncio.current_task()
    assert owner is not None
    connection = MCPConnection(
        "healthy",
        session,
        close_event=asyncio.Event(),
        owner_task=owner,
    )

    result = await connection.call_tool("answer")

    assert result == {"value": 42, "is_error": False, "ok": True}


@pytest.mark.asyncio
async def test_mcp_transport_failure_and_recovery_emit_health_evidence():
    failures: list[tuple[str, BaseException | None]] = []
    session = SimpleNamespace(
        call_tool=AsyncMock(
            side_effect=[ConnectionError("offline"), types.CallToolResult(content=[])]
        ),
    )
    owner = asyncio.current_task()
    assert owner is not None
    connection = MCPConnection(
        "recovering",
        session,
        close_event=asyncio.Event(),
        owner_task=owner,
        health_callback=lambda name, error: failures.append((name, error)),
    )

    with pytest.raises(ConnectionError, match="offline"):
        await connection.call_tool("work")
    result = await connection.call_tool("work")

    assert result["ok"] is True
    assert isinstance(failures[0][1], ConnectionError)
    assert failures[-1] == ("recovering", None)


def test_mcp_client_health_snapshot_tracks_failures_and_recovery():
    client = NoraxMCPClient()
    snapshots: list[dict] = []
    pending = SimpleNamespace(done=lambda: False)
    client._owner_tasks["one"] = pending  # type: ignore[assignment]
    client.set_health_handler(snapshots.append)

    client._record_connection_health("one", TimeoutError("late"))
    assert snapshots[-1]["healthy"] == []
    assert snapshots[-1]["failures"] == {"one": "TimeoutError: late"}

    client._record_connection_health("one", None)
    assert snapshots[-1]["healthy"] == ["one"]
    assert snapshots[-1]["failures"] == {}


class _ToolConnection:
    def __init__(self, name: str, tool_name: str) -> None:
        self.name = name
        self.tool_name = tool_name
        self.calls: list[tuple[str, dict | None]] = []
        self.list_calls = 0

    async def list_tools(self):
        self.list_calls += 1
        return [types.Tool(name=self.tool_name, inputSchema={})]

    async def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        self.calls.append((name, arguments))
        return {"ok": True, "connection": self.name}


@pytest.mark.asyncio
async def test_mcp_sanitized_name_collisions_route_deterministically():
    client = NoraxMCPClient()
    first = _ToolConnection("a_b", "c")
    second = _ToolConnection("a", "b_c")
    client._connections = {"a_b": first, "a": second}  # type: ignore[assignment]
    owner = asyncio.current_task()
    assert owner is not None
    client._owner_tasks = {"a_b": owner, "a": owner}

    tools = await client.aggregate_tools()
    cached_tools = await client.aggregate_tools()

    assert len({tool["name"] for tool in tools}) == 2
    assert cached_tools == tools
    assert first.list_calls == 1
    assert second.list_calls == 1
    assert all(len(f"mcp_{tool['name']}") <= 64 for tool in tools)
    for tool in tools:
        result = await client.call_external_tool(tool["name"], {"x": 1})
        assert result["connection"] == tool["connection"]
    assert len(first.calls) == 1
    assert len(second.calls) == 1


@pytest.mark.asyncio
async def test_unqualified_ambiguous_mcp_tool_never_executes_speculatively():
    client = NoraxMCPClient()
    first = _ToolConnection("one", "publish")
    second = _ToolConnection("two", "publish")
    client._connections = {"one": first, "two": second}  # type: ignore[assignment]
    owner = asyncio.current_task()
    assert owner is not None
    client._owner_tasks = {"one": owner, "two": owner}

    result = await client.call_external_tool("publish", {"value": "release"})

    assert result["ok"] is False
    assert result["error"] == "ambiguous_tool"
    assert first.calls == []
    assert second.calls == []


@pytest.mark.asyncio
async def test_agent_dispatch_applies_risk_gate_before_mcp_execution():
    mcp = SimpleNamespace(call_external_tool=AsyncMock(return_value={"ok": True}))
    events = _EventLog()

    denied = await _run_one_tool(
        "mcp_store_publish",
        {"value": "release"},
        sender_tier="user",
        event_log=events,
        mcp_client=mcp,
    )
    assert denied["error"] == "risk_denied"
    mcp.call_external_tool.assert_not_awaited()

    allowed = await _run_one_tool(
        "mcp_store_publish",
        {"value": "release"},
        sender_tier="owner",
        event_log=events,
        mcp_client=mcp,
    )
    assert allowed["ok"] is True
    mcp.call_external_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcp_transport_failures_open_circuit_and_fail_fast():
    registry = get_circuit_registry()
    registry.reset_all()
    mcp = SimpleNamespace(call_external_tool=AsyncMock(side_effect=TimeoutError))
    events = _EventLog()
    name = "mcp_transport_test_publish"
    try:
        for _ in range(5):
            result = await _run_one_tool(
                name,
                {},
                sender_tier="owner",
                event_log=events,
                mcp_client=mcp,
            )
            assert result["error"] == "timeout"

        rejected = await _run_one_tool(
            name,
            {},
            sender_tier="owner",
            event_log=events,
            mcp_client=mcp,
        )
        assert rejected["error"] == "circuit_open"
        assert rejected["_not_executed"] is True
        assert mcp.call_external_tool.await_count == 5
    finally:
        registry.reset_all()
