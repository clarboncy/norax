"""MCP Client — connect to external MCP servers for dynamic tool discovery.

Supports:
  - stdio transport (subprocess-based MCP servers)
  - SSE transport (HTTP Server-Sent Events)
  - streamable HTTP transport

Usage:
  client = NoraxMCPClient()
  await client.connect_stdio("npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp")
  tools = await client.list_tools()
  result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
  await client.disconnect()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

log = logging.getLogger("norax.mcp.client")


class MCPConnection:
    """A single MCP server connection."""

    def __init__(
        self,
        name: str,
        session: ClientSession,
        *,
        close_event: asyncio.Event,
        owner_task: asyncio.Task[None],
        health_callback: Callable[[str, BaseException | None], None] | None = None,
    ) -> None:
        self.name = name
        self.session = session
        self._close_event = close_event
        self._owner_task = owner_task
        self._health_callback = health_callback
        self._tools: list[types.Tool] | None = None
        self._resources: list[types.Resource] | None = None
        self._prompts: list[types.Prompt] | None = None

    def _report_health(self, error: BaseException | None) -> None:
        if self._health_callback is not None:
            self._health_callback(self.name, error)

    async def list_tools(self) -> list[types.Tool]:
        if self._tools is None:
            try:
                result = await self.session.list_tools()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._report_health(error)
                raise
            self._report_health(None)
            self._tools = result.tools
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict | None = None) -> dict:
        try:
            result = await self.session.call_tool(tool_name, arguments or {})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._report_health(error)
            raise
        self._report_health(None)
        # Prefer the protocol's structured payload. Text-only servers commonly
        # encode JSON in TextContent, so retain that compatibility without ever
        # allowing an MCP-level error to masquerade as ``ok: true``.
        texts: list[str] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                texts.append(block.text)
            elif hasattr(block, "data"):
                texts.append(str(block.data))
        text = "\n".join(texts) if texts else ""
        payload: Any = result.structuredContent
        if payload is None and text:
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                payload = None

        if isinstance(payload, dict):
            normalized = dict(payload)
        elif payload is not None:
            normalized = {"result": payload}
        else:
            normalized = {"text": text}

        is_error = bool(result.isError)
        normalized["is_error"] = is_error
        if is_error or normalized.get("error"):
            original_error = normalized.get("error")
            normalized["ok"] = False
            normalized["error"] = str(original_error or "mcp_tool_error")[:500]
            if text and "detail" not in normalized:
                normalized["detail"] = text
        else:
            normalized.setdefault("ok", True)
        return normalized

    async def list_resources(self) -> list[types.Resource]:
        if self._resources is None:
            try:
                result = await self.session.list_resources()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._report_health(error)
                raise
            self._report_health(None)
            self._resources = result.resources
        return self._resources

    async def read_resource(self, uri: str) -> str:
        from pydantic import AnyUrl

        try:
            result = await self.session.read_resource(AnyUrl(uri))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._report_health(error)
            raise
        self._report_health(None)
        texts = []
        for block in result.contents:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif hasattr(block, "blob"):
                texts.append(f"[binary: {len(block.blob)} bytes]")
        return "\n".join(texts)

    async def list_prompts(self) -> list[types.Prompt]:
        if self._prompts is None:
            try:
                result = await self.session.list_prompts()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._report_health(error)
                raise
            self._report_health(None)
            self._prompts = result.prompts
        return self._prompts

    async def get_prompt(self, name: str, arguments: dict | None = None) -> types.GetPromptResult:
        try:
            result = await self.session.get_prompt(name, arguments or {})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._report_health(error)
            raise
        self._report_health(None)
        return result

    async def close(self) -> None:
        """Ask the owning task to close its AnyIO contexts in-task."""
        self._close_event.set()
        if self._owner_task is asyncio.current_task():
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._owner_task), timeout=5.0)
        except TimeoutError:
            self._owner_task.cancel()
            await asyncio.gather(self._owner_task, return_exceptions=True)
        except asyncio.CancelledError:
            self._owner_task.cancel()
            await asyncio.gather(self._owner_task, return_exceptions=True)
            raise
        except Exception as e:
            log.debug("mcp.owner_close_failed name=%s error=%r", self.name, e)


class NoraxMCPClient:
    """Manage multiple MCP server connections and aggregate their tools."""

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._owner_tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        # Map sanitized tool names (dots→underscores) to original connection.tool names
        self._name_map: dict[str, str] = {}
        self._aggregate_cache: list[dict[str, Any]] | None = None
        self._aggregate_cache_key: tuple[str, ...] = ()
        self._connection_failures: dict[str, str] = {}
        self._health_handler: Callable[[dict[str, Any]], None] | None = None

    def set_health_handler(self, handler: Callable[[dict[str, Any]], None] | None) -> None:
        """Receive bounded health snapshots after transport success or failure."""
        self._health_handler = handler
        self._notify_health()

    def connection_tasks(self) -> dict[str, asyncio.Task[None]]:
        """Return a snapshot of connection-owner tasks for lifecycle supervision."""
        return dict(self._owner_tasks)

    def health(self) -> dict[str, Any]:
        active = sorted(name for name, task in self._owner_tasks.items() if not task.done())
        failures = {
            name: error
            for name, error in self._connection_failures.items()
            if name in self._owner_tasks
        }
        return {
            "active": active,
            "healthy": [name for name in active if name not in failures],
            "failures": failures,
        }

    def _notify_health(self) -> None:
        if self._health_handler is None:
            return
        try:
            self._health_handler(self.health())
        except Exception:  # noqa: BLE001
            log.exception("mcp.health_handler_failed")

    def _record_connection_health(self, name: str, error: BaseException | None) -> None:
        if error is None:
            self._connection_failures.pop(name, None)
        else:
            self._connection_failures[name] = f"{type(error).__name__}: {error}"[:500]
        self._notify_health()

    async def _connection_owner(
        self,
        name: str,
        transport_ctx: Any,
        ready: asyncio.Future[MCPConnection],
        stop_event: asyncio.Event,
    ) -> None:
        """Enter and exit transport/session contexts in one persistent task.

        MCP's AnyIO transports own a cancel scope and require ``__aexit__`` to
        run in the same task as ``__aenter__``. Keeping this task alive for the
        connection lifetime prevents noisy shutdown failures and leaked stdio
        subprocesses.
        """
        session: ClientSession | None = None
        transport_entered = False
        try:
            streams = await transport_ctx.__aenter__()
            transport_entered = True
            read_stream, write_stream = streams[0], streams[1]
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
            owner = asyncio.current_task()
            assert owner is not None
            conn = MCPConnection(
                name,
                session,
                close_event=stop_event,
                owner_task=owner,
                health_callback=self._record_connection_health,
            )
            if not ready.done():
                ready.set_result(conn)
            await stop_event.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                log.warning("mcp.connection_lost name=%s error=%r", name, exc)
                self._record_connection_health(name, exc)
                raise
        finally:
            if session is not None:
                try:
                    await session.__aexit__(None, None, None)
                except Exception as exc:
                    log.debug("mcp.session_close_failed name=%s error=%r", name, exc)
            if transport_entered:
                try:
                    await transport_ctx.__aexit__(None, None, None)
                except Exception as exc:
                    log.debug("mcp.transport_close_failed name=%s error=%r", name, exc)

    async def _connect_owned(
        self,
        name: str,
        transport_ctx: Any,
        *,
        description: str,
    ) -> MCPConnection:
        async with self._lock:
            if name in self._connections or name in self._owner_tasks:
                raise ValueError(f"Connection '{name}' already exists")
            loop = asyncio.get_running_loop()
            ready: asyncio.Future[MCPConnection] = loop.create_future()
            stop_event = asyncio.Event()
            owner = asyncio.create_task(
                self._connection_owner(name, transport_ctx, ready, stop_event),
                name=f"mcp-owner-{name}",
            )
            self._owner_tasks[name] = owner
            self._stop_events[name] = stop_event
            try:
                conn = await ready
                self._connections[name] = conn
                self._aggregate_cache = None
                tool_count = len(await conn.list_tools())
            except BaseException:
                stop_event.set()
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
                self._connections.pop(name, None)
                self._owner_tasks.pop(name, None)
                self._stop_events.pop(name, None)
                self._notify_health()
                raise
            self._notify_health()
            log.info(
                "mcp.connect name=%s transport=%s tools=%d",
                name,
                description,
                tool_count,
            )
            return conn

    async def connect_stdio(
        self,
        name: str,
        command: str,
        *args: str,
        env: dict | None = None,
        cwd: str | None = None,
    ) -> MCPConnection:
        """Connect to an MCP server via stdio (subprocess).

        Example:
          await client.connect_stdio("fs", "npx", "-y",
            "@modelcontextprotocol/server-filesystem", "/tmp")
        """
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        params = StdioServerParameters(
            command=command,
            args=list(args),
            env=full_env,
            cwd=cwd,
        )
        return await self._connect_owned(
            name,
            stdio_client(params),
            description=f"stdio:{command}",
        )

    async def connect_sse(self, name: str, url: str) -> MCPConnection:
        """Connect to an MCP server via SSE transport."""
        return await self._connect_owned(
            name,
            sse_client(url),
            description=f"sse:{url}",
        )

    async def connect_http(self, name: str, url: str) -> MCPConnection:
        """Connect to an MCP server via streamable HTTP transport."""
        return await self._connect_owned(
            name,
            streamablehttp_client(url),
            description=f"http:{url}",
        )

    async def disconnect(self, name: str) -> None:
        async with self._lock:
            conn = self._connections.pop(name, None)
            owner = self._owner_tasks.pop(name, None)
            stop_event = self._stop_events.pop(name, None)
            self._connection_failures.pop(name, None)
            self._aggregate_cache = None
            if stop_event is not None:
                stop_event.set()
        if conn is not None:
            await conn.close()
        elif owner is not None:
            await asyncio.gather(owner, return_exceptions=True)
        if conn is not None or owner is not None:
            log.info("mcp.disconnect name=%s", name)
        self._notify_health()

    async def disconnect_all(self) -> None:
        names = list(self._owner_tasks)
        if names:
            await asyncio.gather(*(self.disconnect(name) for name in names))

    def get_connection(self, name: str) -> MCPConnection | None:
        conn = self._connections.get(name)
        owner = self._owner_tasks.get(name)
        return conn if conn is not None and owner is not None and not owner.done() else None

    def list_connections(self) -> list[str]:
        return [name for name in self._connections if self.get_connection(name) is not None]

    async def aggregate_tools(self) -> list[dict]:
        """Return all tools from all connections as a flat list.

        Each tool is prefixed with the connection name to avoid collisions.
        The 'name' field uses underscores instead of dots (some providers like
        Moonshot reject dots in function names). The original connection.tool
        name is preserved in 'full_name' and registered in _name_map for routing.
          {"name": "fs_read_file", "full_name": "fs.read_file", ...}
        """
        all_tools: list[dict] = []
        active = tuple(self.list_connections())
        if self._aggregate_cache is not None and active == self._aggregate_cache_key:
            return list(self._aggregate_cache)
        self._name_map.clear()
        for conn_name in active:
            conn = self._connections[conn_name]
            try:
                tools = await conn.list_tools()
                for tool in tools:
                    full_name = f"{conn_name}.{tool.name}"
                    safe_name = _safe_tool_name(full_name, self._name_map)
                    self._name_map[safe_name] = full_name
                    all_tools.append(
                        {
                            "name": safe_name,
                            "full_name": full_name,
                            "connection": conn_name,
                            "tool_name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": tool.inputSchema or {},
                        }
                    )
            except Exception as e:
                log.warning("mcp.aggregate_tools conn=%s error=%r", conn_name, e)
        self._aggregate_cache = all_tools
        self._aggregate_cache_key = active
        return list(all_tools)

    async def call_external_tool(self, full_name: str, arguments: dict | None = None) -> dict:
        """Call a tool by its full name (connection.tool_name).

        Also accepts sanitized names (underscores instead of dots) and resolves
        them via _name_map.
        """
        # Resolve sanitized name to original if needed
        if "." not in full_name and full_name in self._name_map:
            full_name = self._name_map[full_name]
        if "." not in full_name:
            # Resolve before executing. Trying a mutating tool against several
            # servers until one succeeds can duplicate a partially completed
            # action after a transport error.
            matches: list[MCPConnection] = []
            for conn_name in self.list_connections():
                conn = self._connections[conn_name]
                try:
                    if any(tool.name == full_name for tool in await conn.list_tools()):
                        matches.append(conn)
                except Exception as exc:
                    log.debug(
                        "mcp.tool_discovery_failed conn=%s tool=%s error=%r",
                        conn.name,
                        full_name,
                        exc,
                    )
            if not matches:
                return {"ok": False, "error": "tool_not_found", "name": full_name}
            if len(matches) > 1:
                return {
                    "ok": False,
                    "error": "ambiguous_tool",
                    "name": full_name,
                    "connections": [conn.name for conn in matches],
                }
            return await matches[0].call_tool(full_name, arguments)

        conn_name, tool_name = full_name.split(".", 1)
        connection = self.get_connection(conn_name)
        if connection is None:
            return {"ok": False, "error": "connection_not_found", "name": conn_name}
        return await connection.call_tool(tool_name, arguments)

    async def read_external_resource(self, conn_name: str, uri: str) -> str:
        conn = self.get_connection(conn_name)
        if conn is None:
            raise ValueError(f"Connection '{conn_name}' not found")
        return await conn.read_resource(uri)


async def discover_tools(command: str, *args: str, timeout: float = 15.0) -> list[dict]:
    """One-shot: connect to an MCP server, list tools, disconnect.

    Useful for probing what tools a server exposes without keeping a
    persistent connection.
    """
    client = NoraxMCPClient()
    try:
        await asyncio.wait_for(
            client.connect_stdio("_probe", command, *args),
            timeout=timeout,
        )
        conn = client.get_connection("_probe")
        if conn is None:
            return []
        tools = await conn.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            for t in tools
        ]
    finally:
        await client.disconnect_all()


def _safe_tool_name(full_name: str, occupied: dict[str, str]) -> str:
    """Return a provider-safe, collision-free MCP name (before ``mcp_``).

    OpenAI-compatible providers commonly limit function names to 64 characters
    and the runtime adds a four-character ``mcp_`` prefix. Normal names retain
    the historical dots-to-underscores spelling; only invalid, long, or
    colliding names gain a stable hash suffix.
    """
    max_length = 60
    base = re.sub(r"[^A-Za-z0-9_-]", "_", full_name.replace(".", "_")) or "tool"
    candidate = base[:max_length]
    mapped = occupied.get(candidate)
    if len(base) <= max_length and (mapped is None or mapped == full_name):
        return candidate

    for digest_length in (8, 12, 16, 24, 32):
        digest = hashlib.sha256(full_name.encode("utf-8")).hexdigest()[:digest_length]
        stem_length = max_length - digest_length - 1
        candidate = f"{base[:stem_length]}_{digest}"
        mapped = occupied.get(candidate)
        if mapped is None or mapped == full_name:
            return candidate
    raise ValueError(f"unable to allocate unique MCP tool name for {full_name!r}")
