"""Default/repair logic for tool arguments (weak-model + Cursor text tools)."""

import pytest

from norax.brain import agent_loop
from norax.brain.strong_model_scaffold import build_task_state, enforce_final_verification
from norax.dispatch.router import normalize_tool_name
from norax.dispatch.tools import normalize_tool_args, t_list_dir


def test_normalize_tool_args_list_dir_missing_path():
    assert normalize_tool_args("list_dir", {}) == {"path": "."}


def test_normalize_tool_args_list_dir_empty_path():
    assert normalize_tool_args("list_dir", {"path": ""}) == {"path": "."}


def test_normalize_tool_args_remote_list_missing_path():
    assert normalize_tool_args("remote_list", {}) == {"path": "."}


def test_normalize_tool_args_remote_node_alias():
    assert normalize_tool_args("remote_list", {"node": "worker", "path": "Desktop"}) == {
        "node_id": "worker",
        "path": "Desktop",
    }
    assert normalize_tool_args("remote_exec", {"node": "worker", "command": "pwd"}) == {
        "node_id": "worker",
        "command": "pwd",
    }


def test_normalize_tool_args_remote_read_preserves_path():
    assert normalize_tool_args(
        "remote_read",
        {"node_id": "worker", "path": "notes.txt"},
    ) == {"node_id": "worker", "path": "notes.txt"}


def test_normalize_tool_args_passthrough():
    assert normalize_tool_args("read", {"path": "x.py"}) == {"path": "x.py"}


def test_normalize_tool_args_read_coerces_numeric_strings():
    assert normalize_tool_args("read", {"path": "x.py", "offset": "10", "limit": "50"}) == {
        "path": "x.py",
        "offset": 10,
        "limit": 50,
    }


def test_normalize_tool_args_exec_coerces_timeout():
    assert normalize_tool_args("exec", {"command": "pwd", "timeout": "30"}) == {
        "command": "pwd",
        "timeout": 30.0,
    }


def test_normalize_tool_name_aliases():
    assert normalize_tool_name("Shell") == "shell"
    assert normalize_tool_name("run_terminal_cmd") == "shell"
    assert normalize_tool_name("execute") == "exec"
    assert normalize_tool_name("read") == "read"
    assert normalize_tool_name("Read") == "read"


def test_normalize_tool_name_does_not_guess_executable_from_typo():
    assert normalize_tool_name("writ") == "writ"
    assert normalize_tool_name("sandbox_exe") == "sandbox_exe"


def test_normalize_tool_args_read_drops_invalid_limit():
    assert normalize_tool_args("read", {"path": "x.py", "limit": "none"}) == {"path": "x.py"}


def test_normalize_tool_args_exec_list_command():
    assert normalize_tool_args("exec", {"command": ["echo", "hi"]}) == {
        "command": "echo hi",
    }


def test_normalize_tool_args_exec_list_is_shell_quoted():
    assert normalize_tool_args("exec", {"command": ["echo", "safe; rm -rf /"]}) == {
        "command": "echo 'safe; rm -rf /'",
    }


def test_normalize_tool_args_rejects_nonfinite_or_fractional_numeric_values():
    assert normalize_tool_args("exec", {"command": "pwd", "timeout": float("inf")}) == {
        "command": "pwd"
    }
    assert normalize_tool_args("read", {"path": "x", "limit": 1.5}) == {"path": "x"}
    assert normalize_tool_args("read", {"path": "x", "offset": True}) == {
        "path": "x",
        "offset": 0,
    }


def test_normalize_tool_args_exec_cmd_alias():
    assert normalize_tool_args("exec", {"cmd": "pwd"}) == {"command": "pwd"}


def test_normalize_tool_args_exec_value_alias():
    assert normalize_tool_args("exec", {"value": "echo hi"}) == {"command": "echo hi"}


def test_normalize_tool_args_exec_nested_parameters():
    original = {"parameters": {"command": "pwd", "timeout": 30}}
    assert normalize_tool_args("exec", original) == {
        "command": "pwd",
        "timeout": 30,
    }
    assert original == {"parameters": {"command": "pwd", "timeout": 30}}


def test_exec_args_valid():
    from norax.dispatch.tools import exec_args_valid, exec_command_is_placeholder

    assert exec_args_valid({"command": "echo hi"}) is True
    assert exec_args_valid({}) is False
    assert exec_args_valid({"cmd": "pwd"}) is True
    assert exec_args_valid({"command": "your command"}) is False
    assert exec_args_valid({"command": "..."}) is False
    assert exec_command_is_placeholder("your shell command here") is True
    assert exec_command_is_placeholder("echo ok") is False


def test_tool_calls_from_skips_exec_placeholder_command():
    from norax.brain.agent_loop import _tool_calls_from
    from norax.gateway_client import GatewayResponse

    resp = GatewayResponse(
        content='{"name":"exec","arguments":{"command":"your command"}}',
        tool_calls=[
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "exec", "arguments": '{"command":"your command"}'},
            }
        ],
    )
    assert _tool_calls_from(resp) == []


@pytest.mark.asyncio
async def test_run_one_tool_exec_rejects_placeholder():
    class _Log:
        async def append(self, *_a, **_k):
            return None

    result = await agent_loop._run_one_tool(
        "exec",
        {"command": "your command"},
        sender_tier="owner",
        event_log=_Log(),
    )
    assert result["ok"] is False
    assert result["error"] == "bad_arguments"
    assert "placeholder" in result["detail"]


@pytest.mark.asyncio
async def test_run_one_tool_exec_missing_command_hint():
    class _Log:
        async def append(self, *_a, **_k):
            return None

    result = await agent_loop._run_one_tool("exec", {}, sender_tier="owner", event_log=_Log())
    assert result["ok"] is False
    assert result["error"] == "bad_arguments"
    assert "command" in result["detail"]
    assert result.get("hint")


def test_cursor_hallucination_nudge_shell_was_rejected():
    state = build_task_state("fix exec", ["exec"])
    nudge = enforce_final_verification(
        state,
        "Shell was rejected. Trying exec instead.",
        model="composer/opus-4.6-thinking",
    )
    assert nudge is not None
    assert "no Cursor Shell tool" in nudge


def test_cursor_hallucination_nudge():
    state = build_task_state("fix exec", ["exec"])
    nudge = enforce_final_verification(
        state,
        "Shell was rejected by the user. I cannot execute commands.",
        model="composer/composer-2.5",
    )
    assert nudge is not None
    assert "no Cursor Shell tool" in nudge


def test_cursor_continue_nudge():
    state = build_task_state("run tests", ["exec"])
    nudge = enforce_final_verification(
        state,
        "Let me check the config first.",
        model="composer/composer-2.5",
    )
    assert nudge is not None
    assert "JSON" in nudge and "No narration" in nudge


def test_action_command_zero_tools_nudge():
    state = build_task_state("Ssh into staging-node and read the token", ["remote_exec"])
    nudge = enforce_final_verification(
        state,
        "I'll connect to staging-node now.",
        model="composer/opus-4.6-thinking",
    )
    assert nudge is not None
    assert "zero tools" in nudge.lower() or "remote_exec" in nudge


def test_nudge_for_dropped_exec_call():
    from norax.brain.strong_model_scaffold import nudge_for_dropped_tool_calls

    raw = [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "exec", "arguments": "{}"},
        }
    ]
    nudge = nudge_for_dropped_tool_calls(raw, [])
    assert nudge is not None
    assert "command" in nudge


def test_tool_calls_from_skips_exec_without_command():
    from norax.brain.agent_loop import _tool_calls_from
    from norax.gateway_client import GatewayResponse

    resp = GatewayResponse(
        content='{"name":"exec","arguments":{}}',
        tool_calls=[
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "exec", "arguments": "{}"},
            }
        ],
    )
    assert _tool_calls_from(resp) == []


def test_tool_calls_from_keeps_valid_exec():
    from norax.brain.agent_loop import _tool_calls_from
    from norax.gateway_client import GatewayResponse

    resp = GatewayResponse(
        content='{"name":"exec","arguments":{"command":"echo hello"}}',
        tool_calls=[
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "exec", "arguments": '{"command":"echo hello"}'},
            }
        ],
    )
    calls = _tool_calls_from(resp)
    assert len(calls) == 1
    assert calls[0]["args"]["command"] == "echo hello"


@pytest.mark.asyncio
async def test_t_read_string_offset_limit(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")
    from norax.dispatch.tools import t_read

    result = await t_read(path=str(f), offset="1", limit="2")
    assert result["ok"] is True
    assert result["content"] == "b\nc"


@pytest.mark.asyncio
async def test_t_read_marks_explicit_large_reads(tmp_path):
    f = tmp_path / "large.txt"
    f.write_text("line\n" * 100, encoding="utf-8")
    from norax.dispatch.tools import t_read

    result = await t_read(path=str(f), limit=80, large=True)
    assert result["ok"] is True
    assert result["_large_read"] is True


@pytest.mark.asyncio
async def test_list_dir_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    result = await t_list_dir()
    assert result["ok"] is True
    assert any(e["name"] == "a.txt" for e in result["entries"])


@pytest.mark.asyncio
async def test_run_one_tool_exec_with_string_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _Log:
        async def append(self, *_a, **_k):
            return None

    result = await agent_loop._run_one_tool(
        "exec",
        {"command": "echo exec_test_ok", "timeout": "30"},
        sender_tier="owner",
        event_log=_Log(),
    )
    assert result["ok"] is True
    assert "exec_test_ok" in result.get("stdout", "")


@pytest.mark.asyncio
async def test_run_one_tool_list_dir_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "b.txt").write_text("ok", encoding="utf-8")

    class _Log:
        async def append(self, *_a, **_k):
            return None

    result = await agent_loop._run_one_tool(
        "list_dir",
        {},
        sender_tier="owner",
        event_log=_Log(),
    )
    assert result["ok"] is True
    assert any(e["name"] == "b.txt" for e in result["entries"])
