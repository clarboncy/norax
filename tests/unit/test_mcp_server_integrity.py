"""Protocol-level integrity checks for the standalone Norax MCP server."""

from __future__ import annotations

import json

import pytest
from mcp import types

from norax.dispatch.tools import REGISTRY as TOOL_REGISTRY
from norax.dispatch.tools import ToolSpec
from norax.mcp.server import NoraxMCPServer


async def _request(server: NoraxMCPServer, request):
    handler = server.server.request_handlers[type(request)]
    return (await handler(request)).root


def _json_content(result: types.CallToolResult) -> dict:
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    return json.loads(block.text)


@pytest.mark.asyncio
async def test_tool_manifest_is_schema_complete_and_truthfully_annotated(tmp_path) -> None:
    server = NoraxMCPServer(memory_root=tmp_path)
    result = await _request(server, types.ListToolsRequest())
    tools = {tool.name: tool for tool in result.tools}

    assert "deep_research" in tools
    assert tools["read"].annotations.readOnlyHint is True
    assert tools["read"].annotations.destructiveHint is False
    assert tools["deep_research"].annotations.readOnlyHint is False
    assert tools["deep_research"].annotations.openWorldHint is True
    assert tools["write"].inputSchema["required"] == ["path", "content"]
    assert tools["list_dir"].inputSchema["required"] == []


@pytest.mark.asyncio
async def test_tool_failures_set_mcp_error_flag_and_structured_content(
    tmp_path, monkeypatch
) -> None:
    async def failed_status() -> dict:
        return {"ok": False, "error": "backend_unavailable"}

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "status",
        ToolSpec("status", "test", {}, failed_status),
    )
    server = NoraxMCPServer(memory_root=tmp_path)
    request = types.CallToolRequest(params=types.CallToolRequestParams(name="status", arguments={}))
    result = await _request(server, request)

    assert isinstance(result, types.CallToolResult)
    assert result.isError is True
    assert result.structuredContent == {"ok": False, "error": "backend_unavailable"}
    assert _json_content(result) == result.structuredContent


@pytest.mark.asyncio
async def test_tool_success_filters_unknown_arguments_and_is_not_an_error(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []

    async def healthy_status() -> dict:
        calls.append("called")
        return {"ok": True, "state": "ready"}

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "status",
        ToolSpec("status", "test", {}, healthy_status),
    )
    server = NoraxMCPServer(memory_root=tmp_path)
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name="status",
            arguments={"unexpected": "discarded"},
        )
    )
    result = await _request(server, request)

    assert calls == ["called"]
    assert result.isError is False
    assert result.structuredContent == {"ok": True, "state": "ready"}


@pytest.mark.asyncio
async def test_truthy_non_boolean_tool_status_is_a_protocol_error(tmp_path, monkeypatch) -> None:
    async def malformed_status() -> dict:
        return {"ok": "false", "state": "not_ready"}

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "status",
        ToolSpec("status", "test", {}, malformed_status),
    )
    server = NoraxMCPServer(memory_root=tmp_path)
    request = types.CallToolRequest(params=types.CallToolRequestParams(name="status", arguments={}))
    result = await _request(server, request)

    assert result.isError is True
    assert result.structuredContent == {"ok": "false", "state": "not_ready"}


@pytest.mark.asyncio
async def test_unexposed_and_sensitive_tools_are_protocol_errors(tmp_path) -> None:
    server = NoraxMCPServer(memory_root=tmp_path)

    unknown = await _request(
        server,
        types.CallToolRequest(params=types.CallToolRequestParams(name="not_a_tool", arguments={})),
    )
    assert unknown.isError is True
    assert _json_content(unknown)["error"] == "tool_not_exposed"

    destructive = await _request(
        server,
        types.CallToolRequest(
            params=types.CallToolRequestParams(
                name="exec",
                arguments={"command": "rm -rf /tmp/norax-mcp-test"},
            )
        ),
    )
    assert destructive.isError is True
    assert _json_content(destructive)["error"] in {
        "risk_denied",
        "authenticated_host_approval_required",
    }


@pytest.mark.asyncio
async def test_tool_exception_is_a_bounded_protocol_error(tmp_path, monkeypatch) -> None:
    async def explode() -> dict:
        raise RuntimeError("boom")

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "status",
        ToolSpec("status", "test", {}, explode),
    )
    server = NoraxMCPServer(memory_root=tmp_path)
    result = await _request(
        server,
        types.CallToolRequest(params=types.CallToolRequestParams(name="status", arguments={})),
    )

    assert result.isError is True
    assert _json_content(result) == {"ok": False, "error": "boom"}


@pytest.mark.asyncio
async def test_memory_resources_are_bounded_and_do_not_follow_symlinks(tmp_path) -> None:
    memory_root = tmp_path / "memory"
    semantic = memory_root / "semantic"
    nested = semantic / "nested"
    nested.mkdir(parents=True)
    (semantic / "a.md").write_text("alpha", encoding="utf-8")
    (nested / "b.md").write_text("beta", encoding="utf-8")
    (semantic / "huge.md").write_text("x" * 70_000, encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("TOP_SECRET", encoding="utf-8")
    (semantic / "leak.md").symlink_to(outside)

    server = NoraxMCPServer(memory_root=memory_root)
    listed = await _request(server, types.ListResourcesRequest())
    assert {str(resource.uri) for resource in listed.resources} >= {
        "norax://memory/semantic",
        "norax://status",
    }

    result = await _request(
        server,
        types.ReadResourceRequest(
            params=types.ReadResourceRequestParams(uri="norax://memory/semantic")
        ),
    )
    text = result.contents[0].text
    assert "alpha" in text
    assert "beta" in text
    assert "[file truncated]" in text
    assert "TOP_SECRET" not in text
    assert len(text) <= 512_000


@pytest.mark.asyncio
async def test_prompts_execute_their_real_dispatch_contract(tmp_path, monkeypatch) -> None:
    async def fake_status() -> dict:
        return {"ok": True, "state": "ready"}

    async def fake_search_memory(*, query: str, k: int) -> dict:
        return {"ok": True, "query": query, "k": k, "results": []}

    monkeypatch.setattr("norax.dispatch.tools.t_status", fake_status)
    monkeypatch.setattr("norax.dispatch.tools.t_search_memory", fake_search_memory)
    server = NoraxMCPServer(memory_root=tmp_path)

    listed = await _request(server, types.ListPromptsRequest())
    assert {prompt.name for prompt in listed.prompts} == {
        "norax_status",
        "norax_memory_search",
    }

    status = await _request(
        server,
        types.GetPromptRequest(
            params=types.GetPromptRequestParams(name="norax_status", arguments={})
        ),
    )
    assert '"state": "ready"' in status.messages[0].content.text

    search = await _request(
        server,
        types.GetPromptRequest(
            params=types.GetPromptRequestParams(
                name="norax_memory_search",
                arguments={"query": "retry policy"},
            )
        ),
    )
    assert '"query": "retry policy"' in search.messages[0].content.text


def test_memory_root_honors_explicit_path_and_environment(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(configured))

    assert NoraxMCPServer().memory_root == configured.resolve()
    assert NoraxMCPServer(memory_root=explicit).memory_root == explicit.resolve()


def test_single_text_resource_is_regular_file_only_and_bounded(tmp_path) -> None:
    server = NoraxMCPServer(memory_root=tmp_path)
    missing = tmp_path / "missing.md"
    assert server._read_text_file(missing) == ""

    regular = tmp_path / "regular.md"
    regular.write_text("z" * 70_000, encoding="utf-8")
    text = server._read_text_file(regular)
    assert len(text) < 70_000
    assert text.endswith("[file truncated]\n")

    link = tmp_path / "link.md"
    link.symlink_to(regular)
    assert server._read_text_file(link) == ""
