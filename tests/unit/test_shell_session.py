"""Stateful shell session tests."""

import pytest

from norax.brain import agent_loop
from norax.dispatch.router import normalize_tool_name
from norax.dispatch.tools import normalize_tool_args, t_shell
from norax.shell.manager import get_shell_manager


@pytest.fixture(autouse=True)
def _reset_shell_sessions():
    get_shell_manager().reset("test-session")
    yield
    get_shell_manager().reset("test-session")


def test_shell_alias_routes_to_shell_tool():
    assert normalize_tool_name("Shell") == "shell"
    assert normalize_tool_name("bash") == "shell"


def test_normalize_tool_args_shell_cmd_alias():
    assert normalize_tool_args("shell", {"cmd": "pwd"}) == {"command": "pwd"}


@pytest.mark.asyncio
async def test_shell_session_persists_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "nested"
    sub.mkdir()

    r1 = await t_shell(command="cd nested", session_id="test-session")
    assert r1["ok"] is True
    assert r1["cwd"] == str(sub.resolve())

    r2 = await t_shell(command="pwd", session_id="test-session")
    assert r2["ok"] is True
    assert str(sub.resolve()) in r2["stdout"]


@pytest.mark.asyncio
async def test_exec_is_one_shot_no_cwd_persistence(tmp_path, monkeypatch):
    from norax.dispatch.tools import t_exec

    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "nested"
    sub.mkdir()

    await t_exec(command="cd nested")
    r = await t_exec(command="pwd")
    assert r["ok"] is True
    assert str(tmp_path.resolve()) in r["stdout"]


@pytest.mark.asyncio
async def test_run_one_tool_shell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _Log:
        async def append(self, *_a, **_k):
            return None

    result = await agent_loop._run_one_tool(
        "shell",
        {"command": "echo norax_shell_ok", "session_id": "test-session"},
        sender_tier="owner",
        event_log=_Log(),
    )
    assert result["ok"] is True
    assert "norax_shell_ok" in result.get("stdout", "")


@pytest.mark.asyncio
async def test_nonzero_exec_is_not_an_exec_circuit_failure():
    """A completed probe with a negative result must not disable the runner."""

    class _Log:
        async def append(self, *_a, **_k):
            return None

    from norax.runtime.circuit_breaker import get_circuit_registry

    registry = get_circuit_registry()
    registry.reset_all()
    result = await agent_loop._run_one_tool(
        "exec",
        {"command": "false"},
        sender_tier="owner",
        event_log=_Log(),
    )

    assert result["ok"] is False
    stats = next(item for item in registry.all_stats() if item["name"] == "exec")
    assert stats["consecutive_failures"] == 0
    assert stats["total_successes"] >= 1


@pytest.mark.asyncio
async def test_missing_files_do_not_poison_the_shared_read_circuit(tmp_path):
    """Domain-level negative results prove the read tool is still responsive."""

    class _Log:
        async def append(self, *_a, **_k):
            return None

    from norax.runtime.circuit_breaker import get_circuit_registry

    registry = get_circuit_registry()
    registry.reset_all()
    for index in range(8):
        result = await agent_loop._run_one_tool(
            "read",
            {"path": str(tmp_path / f"missing-{index}.txt")},
            sender_tier="owner",
            event_log=_Log(),
        )
        assert result["error"] == "file_not_found"

    stats = next(item for item in registry.all_stats() if item["name"] == "read")
    assert stats["state"] == "closed"
    assert stats["consecutive_failures"] == 0
    assert stats["total_failures"] == 0
