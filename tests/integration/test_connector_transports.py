"""Real subprocess/TCP connector contracts, isolated from deployment state."""

from __future__ import annotations

import asyncio
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from sse_starlette.sse import AppStatus

from norax.a2a.client import A2AClient
from norax.a2a.server import NoraxA2AServer, create_a2a_app
from norax.mcp.client import NoraxMCPClient
from norax.runtime.outbound import OutboundRegistry


@asynccontextmanager
async def _listener(app):
    # sse-starlette keeps a process-global shutdown flag. These independent
    # fixture servers share a test process, unlike standalone deployments.
    previous_exit = AppStatus.should_exit
    AppStatus.should_exit = False
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("connector listener exited before startup")
                    await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=10)
            AppStatus.should_exit = previous_exit


@pytest.mark.asyncio
async def test_mcp_stdio_real_file_round_trip_and_error(tmp_path):
    client = NoraxMCPClient()
    artifact = tmp_path / "receipt.txt"
    root = Path(__file__).resolve().parents[2]
    try:
        async with asyncio.timeout(20):
            connection = await client.connect_stdio(
                "acceptance", sys.executable, "-m", "norax.mcp.server", cwd=str(root)
            )
            tools = {tool.name for tool in await connection.list_tools()}
            assert {"write", "read"} <= tools
            written = await connection.call_tool(
                "write", {"path": str(artifact), "content": "MCP_ROUND_TRIP"}
            )
            assert written["ok"] is True
            read = await connection.call_tool("read", {"path": str(artifact)})
            assert read["ok"] is True and read["content"] == "MCP_ROUND_TRIP"
            assert artifact.read_text() == "MCP_ROUND_TRIP"
            missing = await connection.call_tool("read", {"path": str(tmp_path / "missing")})
            assert missing["ok"] is False and missing["is_error"] is True
            unknown = await connection.call_tool("not_exposed", {})
            assert unknown["ok"] is False and unknown["is_error"] is True
    finally:
        await client.disconnect_all()
    assert client.get_connection("acceptance") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sse", "http"])
async def test_mcp_real_network_transport_concurrent_calls_and_errors(transport):
    fixture = FastMCP("connector-fixture", log_level="ERROR")

    @fixture.tool()
    def echo(value: str) -> dict:
        return {"ok": True, "value": value}

    @fixture.tool()
    def fail() -> dict:
        raise ValueError("intentional connector fixture failure")

    app = fixture.sse_app() if transport == "sse" else fixture.streamable_http_app()
    client = NoraxMCPClient()
    async with _listener(app) as base:
        try:
            connect = client.connect_sse if transport == "sse" else client.connect_http
            async with asyncio.timeout(20):
                connection = await connect(
                    "fixture", base + ("/sse" if transport == "sse" else "/mcp")
                )
                results = await asyncio.gather(
                    *(connection.call_tool("echo", {"value": str(i)}) for i in range(8))
                )
                assert [result["value"] for result in results] == [str(i) for i in range(8)]
                assert all(result["ok"] is True for result in results)
                failed = await connection.call_tool("fail", {})
                assert failed["ok"] is False and failed["is_error"] is True
        finally:
            await client.disconnect_all()


class _EchoRuntime:
    """Deterministic worker, not a simulated model-performance measurement."""

    def __init__(self):
        self.outbound = OutboundRegistry()

    async def _handle_turn(self, envelope):
        await self.outbound.send(
            envelope.source, envelope.thread_binding.thread_id, "receipt:" + envelope.body
        )


@pytest.mark.asyncio
async def test_a2a_real_tcp_auth_delivery_and_context_isolation():
    server = NoraxA2AServer(_EchoRuntime(), "http://127.0.0.1", auth_token="test-token")
    try:
        async with _listener(create_a2a_app(server)) as base:
            async with A2AClient(base, token="wrong-token") as denied:
                result = await denied.send_task("must not execute")
                assert result.status == "error"
            async with A2AClient(base, token="test-token") as client:
                results = await asyncio.gather(
                    *(client.send_task(str(i), context_id=f"context-{i}") for i in range(8))
                )
                assert [result.result for result in results] == [f"receipt:{i}" for i in range(8)]
                assert all(result.status == "completed" for result in results)
                assert len({result.id for result in results}) == 8
                assert [result.context_id for result in results] == [
                    f"context-{i}" for i in range(8)
                ]
                fetched = await client.get_task(results[0].id)
                assert fetched.result == "receipt:0"
    finally:
        await server.close()
