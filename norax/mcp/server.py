"""MCP Server — exposes Norax tools via Model Context Protocol.

Other MCP-compatible agents (Claude Code, Goose, OpenHands, Cline, etc.) can connect
to Norax as an MCP server and use our 22+ tools.

Transports:
  - stdio: for local subprocess integration (Claude Code, Cline)
  - streamable HTTP: for remote/network integration

Protocol: JSON-RPC 2.0 over stdio or HTTP SSE
Capabilities: tools (list + call), resources (list + read), prompts (list + get)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server

from ..dispatch.risk import check as risk_check
from ..dispatch.tools import REGISTRY as TOOL_REGISTRY
from ..dispatch.tools import normalize_tool_args, normalize_tool_name
from ..runtime.interrupts import InterruptManager, create_default_rules

log = logging.getLogger("norax.mcp.server")

# Tools that are safe to expose via MCP (exclude dangerous/internal ones)
_EXPOSED_TOOLS = {
    "read",
    "list_dir",
    "web_fetch",
    "web_search",
    "search_memory",
    "status",
    "write",
    "write_chunk",
    "edit",
    "append_memory",
    "exec",
    "shell",
    "browser",
    "repo_explore",
    "deep_research",
    "remote_enroll",
    "remote_list_nodes",
    "remote_exec",
    "remote_read",
    "remote_list",
    "remote_write",
    "message_send",
    "schedule_reminder",
    "gateway_config_patch",
    "memory_search",
}

# MCP resources — expose memory stores as readable resources
_RESOURCE_URIS = {
    "norax://memory/semantic": "Semantic memory — permanent facts and knowledge",
    "norax://memory/procedural": "Procedural memory — workflows and how-tos",
    "norax://memory/intel": "Intel memory — external research notes",
    "norax://memory/scratchpad": "Scratchpad — hot working state",
    "norax://status": "Runtime status",
}

_READ_ONLY_TOOLS = frozenset(
    {
        "read",
        "list_dir",
        "web_fetch",
        "web_search",
        "search_memory",
        "memory_search",
        "status",
        "repo_explore",
        "remote_read",
        "remote_list",
        "remote_list_nodes",
    }
)
_OPEN_WORLD_TOOLS = frozenset(
    {
        "web_fetch",
        "web_search",
        "browser",
        "exec",
        "shell",
        "message_send",
        "remote_enroll",
        "remote_list_nodes",
        "remote_exec",
        "remote_read",
        "remote_list",
        "remote_write",
        "deep_research",
    }
)

_MAX_RESOURCE_FILES = 128
_MAX_RESOURCE_SCAN_ENTRIES = 2_048
_MAX_RESOURCE_FILE_CHARS = 64_000
_MAX_RESOURCE_TOTAL_CHARS = 512_000

_MCP_SENSITIVE_POLICY = InterruptManager()
for _interrupt_rule in create_default_rules():
    _MCP_SENSITIVE_POLICY.add_rule(_interrupt_rule)


def _operator_trusts_mcp_host_approvals() -> bool:
    return os.environ.get("NORAX_MCP_TRUST_HOST_APPROVALS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _authorize_mcp_call(
    name: str,
    arguments: dict,
    *,
    sender_tier: str,
) -> tuple[dict | None, dict | None]:
    """Authorize an MCP call without inventing a non-resumable approval.

    Stdio/HTTP MCP servers commonly run in a process separate from the Discord
    runtime, so a process-local PendingInterrupt could never be approved from
    the owner channel. Approval-sensitive calls are therefore denied unless an
    operator explicitly delegates that boundary to the MCP host.
    """
    if name not in _EXPOSED_TOOLS:
        return None, {"ok": False, "error": "tool_not_exposed", "name": name}

    decision = risk_check(tool=name, args=arguments, sender_tier=sender_tier)
    if not decision.allowed:
        return None, {
            "ok": False,
            "error": "risk_denied",
            "reason": decision.reason,
        }

    should_pause, reason, description = _MCP_SENSITIVE_POLICY.should_interrupt(name, arguments)
    if should_pause and not _operator_trusts_mcp_host_approvals():
        return None, {
            "ok": False,
            "error": "authenticated_host_approval_required",
            "reason": reason.value,
            "description": description,
            "resumable": False,
            "hint": (
                "Use Norax's authenticated owner channel, or explicitly configure "
                "NORAX_MCP_TRUST_HOST_APPROVALS=1 when the MCP host enforces human approval."
            ),
        }
    return arguments, None


def _call_result(payload: dict[str, Any]) -> types.CallToolResult:
    """Return structured MCP output with a truthful protocol error flag."""
    text = json.dumps(payload, default=str)
    structured = json.loads(text)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=structured,
        isError=payload.get("ok") is not True,
    )


class NoraxMCPServer:
    """MCP server wrapping Norax's tool dispatch layer."""

    def __init__(
        self,
        *,
        sender_tier: str = "owner",
        memory_root: Path | None = None,
    ) -> None:
        self.server = Server("norax")
        self.sender_tier = sender_tier
        configured_memory = os.environ.get("NORAX_MEMORY_ROOT", "").strip()
        self.memory_root = (
            Path(configured_memory).expanduser()
            if memory_root is None and configured_memory
            else memory_root
            if memory_root is not None
            else Path.home() / "norax" / "memory"
        ).resolve(strict=False)
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            tools = []
            for name in sorted(_EXPOSED_TOOLS):
                spec = TOOL_REGISTRY.get(name)
                if spec is None:
                    continue
                # Build input schema from our flat schema dict
                props: dict[str, dict] = {}
                required: list[str] = []
                for arg_name, arg_type in spec.schema.items():
                    optional = arg_type.endswith("?")
                    t = arg_type.rstrip("?")
                    js_type = {
                        "str": "string",
                        "int": "integer",
                        "float": "number",
                        "bool": "boolean",
                        "dict": "object",
                        "list": "array",
                    }.get(t, "string")
                    prop: dict = {"type": js_type}
                    if js_type == "array":
                        prop["items"] = {"type": "string"}
                    props[arg_name] = prop
                    if not optional:
                        required.append(arg_name)

                tools.append(
                    types.Tool(
                        name=name,
                        description=spec.description,
                        inputSchema={
                            "type": "object",
                            "properties": props,
                            "required": required,
                        },
                        annotations=types.ToolAnnotations(
                            readOnlyHint=name in _READ_ONLY_TOOLS,
                            destructiveHint=name not in _READ_ONLY_TOOLS,
                            idempotentHint=name in _READ_ONLY_TOOLS,
                            openWorldHint=name in _OPEN_WORLD_TOOLS,
                        ),
                    )
                )
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
            name = normalize_tool_name(name)
            arguments = normalize_tool_args(name, arguments or {})

            # Normalize aliases before authorization so nested/script/cmd forms
            # cannot bypass command/path inspection. Reject names omitted from
            # the advertised surface even if they exist in the internal registry.
            arguments, denial = _authorize_mcp_call(
                name,
                arguments,
                sender_tier=self.sender_tier,
            )
            if denial is not None:
                return _call_result(denial)
            assert arguments is not None

            spec = TOOL_REGISTRY.get(name)
            if spec is None:
                return _call_result({"ok": False, "error": "unknown_tool", "name": name})

            try:
                import inspect

                sig = inspect.signature(spec.fn)
                accepted = {
                    k: v
                    for k, v in arguments.items()
                    if k in sig.parameters
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                }
                result = await asyncio.wait_for(spec.fn(**accepted), timeout=120.0)
                if not isinstance(result, dict):
                    result = {"ok": True, "result": result}
            except TimeoutError:
                result = {"ok": False, "error": "timeout", "timeout_s": 120}
            except Exception as e:
                result = {"ok": False, "error": str(e)}

            return _call_result(result)

        @self.server.list_resources()
        async def list_resources() -> list[types.Resource]:
            resources = []
            for uri, desc in _RESOURCE_URIS.items():
                resources.append(
                    types.Resource(
                        uri=types.AnyUrl(uri),
                        name=uri.split("//")[-1].replace("/", "."),
                        description=desc,
                        mimeType="text/plain",
                    )
                )
            return resources

        @self.server.read_resource()
        async def read_resource(uri: str) -> list[ReadResourceContents]:
            mem_root = self.memory_root
            uri_text = str(uri)

            if uri_text == "norax://status":
                from ..dispatch.tools import t_status

                r = await t_status()
                text = json.dumps(r, indent=2, default=str)
                return [ReadResourceContents(content=text, mime_type="application/json")]

            if uri_text == "norax://memory/scratchpad":
                p = mem_root / "scratchpad.md"
                return [
                    ReadResourceContents(content=self._read_text_file(p), mime_type="text/markdown")
                ]

            if uri_text == "norax://memory/semantic":
                text = self._read_memory_store(mem_root / "semantic")
                return [ReadResourceContents(content=text, mime_type="text/markdown")]

            if uri_text == "norax://memory/procedural":
                text = self._read_memory_store(mem_root / "procedural")
                return [ReadResourceContents(content=text, mime_type="text/markdown")]

            if uri_text == "norax://memory/intel":
                text = self._read_memory_store(mem_root / "intel")
                return [ReadResourceContents(content=text, mime_type="text/markdown")]

            raise ValueError(f"Unknown resource: {uri_text}")

        @self.server.list_prompts()
        async def list_prompts() -> list[types.Prompt]:
            return [
                types.Prompt(
                    name="norax_status",
                    description="Get Norax runtime status and memory stats",
                    arguments=[],
                ),
                types.Prompt(
                    name="norax_memory_search",
                    description="Search Norax memory stores",
                    arguments=[
                        types.PromptArgument(
                            name="query",
                            description="Search query",
                            required=True,
                        ),
                    ],
                ),
            ]

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
            if name == "norax_status":
                from ..dispatch.tools import t_status

                r = await t_status()
                return types.GetPromptResult(
                    description="Norax runtime status",
                    messages=[
                        types.PromptMessage(
                            role="assistant",
                            content=types.TextContent(
                                type="text", text=json.dumps(r, indent=2, default=str)
                            ),
                        ),
                    ],
                )
            if name == "norax_memory_search":
                query = (arguments or {}).get("query", "")
                from ..dispatch.tools import t_search_memory

                r = await t_search_memory(query=query, k=5)
                return types.GetPromptResult(
                    description=f"Memory search: {query}",
                    messages=[
                        types.PromptMessage(
                            role="assistant",
                            content=types.TextContent(
                                type="text", text=json.dumps(r, indent=2, default=str)
                            ),
                        ),
                    ],
                )
            raise ValueError(f"Unknown prompt: {name}")

    def _read_memory_store(self, path: Path) -> str:
        """Read a deterministic, bounded set of in-root Markdown resources."""
        if not path.exists():
            return ""
        root = path.resolve(strict=False)
        files: list[Path] = []
        stack = [root]
        scanned = 0
        truncated_scan = False
        while stack and len(files) < _MAX_RESOURCE_FILES:
            current = stack.pop()
            try:
                children = sorted(current.iterdir(), key=lambda item: item.name.casefold())
            except OSError:
                continue
            directories: list[Path] = []
            for child in children:
                scanned += 1
                if scanned > _MAX_RESOURCE_SCAN_ENTRIES:
                    truncated_scan = True
                    stack.clear()
                    break
                try:
                    if child.is_symlink():
                        continue
                    if child.is_dir():
                        directories.append(child)
                    elif child.is_file() and child.suffix.casefold() == ".md":
                        files.append(child)
                        if len(files) >= _MAX_RESOURCE_FILES:
                            truncated_scan = True
                            stack.clear()
                            break
                except OSError:
                    continue
            stack.extend(reversed(directories))

        parts: list[str] = []
        total = 0
        for file_path in files:
            body = self._read_text_file(file_path)
            part = f"--- {file_path.relative_to(root)} ---\n{body}\n"
            remaining = _MAX_RESOURCE_TOTAL_CHARS - total
            if remaining <= 0:
                truncated_scan = True
                break
            if len(part) > remaining:
                marker = "\n[resource output truncated]\n"
                prefix_chars = max(0, remaining - len(marker))
                parts.append(part[:prefix_chars] + marker[: remaining - prefix_chars])
                total = _MAX_RESOURCE_TOTAL_CHARS
                truncated_scan = True
                break
            parts.append(part)
            total += len(part)
        if truncated_scan and total < _MAX_RESOURCE_TOTAL_CHARS:
            marker = "[resource listing truncated]\n"
            parts.append(marker[: _MAX_RESOURCE_TOTAL_CHARS - total])
        return "\n".join(parts)

    @staticmethod
    def _read_text_file(path: Path) -> str:
        """Read one regular, non-symlink text resource with a hard cap."""
        try:
            if path.is_symlink() or not path.is_file():
                return ""
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(_MAX_RESOURCE_FILE_CHARS + 1)
        except OSError:
            return ""
        if len(text) > _MAX_RESOURCE_FILE_CHARS:
            return text[:_MAX_RESOURCE_FILE_CHARS] + "\n[file truncated]\n"
        return text

    async def run_stdio(self) -> None:
        """Run the MCP server over stdio (for local subprocess integration)."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    """Entry point for `python -m norax.mcp.server`"""
    server = NoraxMCPServer(sender_tier="owner")
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
