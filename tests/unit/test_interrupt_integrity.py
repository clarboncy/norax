from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from norax.brain import agent_loop
from norax.commands import ParsedCommand, handle
from norax.dispatch.tools import REGISTRY, ToolSpec
from norax.mcp.server import _authorize_mcp_call
from norax.runtime import interrupts as interrupt_mod
from norax.runtime.interrupts import (
    InterruptAction,
    InterruptManager,
    InterruptReason,
    InterruptRule,
    create_default_rules,
)


class _EventLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def append(self, kind: str, payload: dict) -> None:
        self.events.append((kind, payload))


def test_interrupt_rule_regexes_are_validated_at_configuration_time() -> None:
    with pytest.raises(re.error):
        InterruptRule(command_patterns=["["])


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "kill -9 $(pgrep norax)",
        "kill -TERM $(pgrep ollama)",
        "DROP TABLE users",
        "kill -9 $(pgrep norax)",
        "kill -TERM $(pgrep ollama)",
    ],
)
def test_default_rules_interrupt_destructive_commands(command: str) -> None:
    manager = InterruptManager()
    for rule in create_default_rules():
        manager.add_rule(rule)

    should_pause, reason, _description = manager.should_interrupt(
        "exec",
        {"command": command},
    )

    assert should_pause is True
    assert reason is InterruptReason.DANGEROUS_COMMAND


@pytest.mark.parametrize(
    "command",
    [
        "sudo systemctl stop ollama.service",
        "systemctl --user restart norax-ai.service",
        "sudo cp /tmp/bin/ollama /usr/local/bin/ollama",
        "docker compose down",
        "kill -TERM 1234",
        "ollama pull huge-model",
        "git status --short",
        "pip install requests",
        "npm run build",
    ],
)
def test_default_rules_allow_routine_operations(command: str) -> None:
    manager = InterruptManager()
    for rule in create_default_rules():
        manager.add_rule(rule)

    should_pause, _reason, _description = manager.should_interrupt(
        "exec",
        {"command": command},
    )

    assert should_pause is False


def test_default_rules_leave_safe_observation_commands_fast() -> None:
    manager = InterruptManager()
    for rule in create_default_rules():
        manager.add_rule(rule)

    assert manager.should_interrupt("exec", {"command": "git status --short"})[0] is False


def test_empty_approval_queue_skips_payload_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = InterruptManager()

    def unexpected_digest(*_args, **_kwargs):
        raise AssertionError("empty approval queue hashed a tool payload")

    monkeypatch.setattr(manager, "_call_digest", unexpected_digest)
    assert manager.consume_resolution("write", {"content": "x" * 1_000_000}) is None


def test_interrupt_is_deduplicated_and_approval_is_exact_and_single_use() -> None:
    manager = InterruptManager(max_pending=2)
    first = manager.create_interrupt(
        "write",
        {"path": "/etc/example", "content": "secret"},
        InterruptReason.SENSITIVE_PATH,
        "system path",
    )
    duplicate = manager.create_interrupt(
        "write",
        {"content": "secret", "path": "/etc/example"},
        InterruptReason.SENSITIVE_PATH,
        "system path",
    )
    assert duplicate.interrupt_id == first.interrupt_id
    assert not hasattr(first, "args")

    assert manager.resolve_interrupt(
        first.interrupt_id,
        InterruptAction.APPROVE,
        resolved_by="owner-id",
    )
    assert manager.consume_resolution("write", {"path": "/etc/other"}) is None
    resolution = manager.consume_resolution("write", {"path": "/etc/example", "content": "secret"})
    assert resolution == {
        "action": "approve",
        "args": {"path": "/etc/example", "content": "secret"},
    }
    assert (
        manager.consume_resolution("write", {"path": "/etc/example", "content": "secret"}) is None
    )


def test_interrupt_resolution_requires_authenticated_principal() -> None:
    manager = InterruptManager()
    pending = manager.create_interrupt(
        "remote_write", {}, InterruptReason.WRITE_OPERATION, "remote write"
    )
    with pytest.raises(ValueError, match="authenticated approver"):
        manager.resolve_interrupt(
            pending.interrupt_id,
            InterruptAction.APPROVE,
            resolved_by="",
        )


@pytest.mark.asyncio
async def test_owner_is_not_auto_approved_and_can_resume_exact_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = InterruptManager()
    manager.add_rule(InterruptRule(tools={"status"}, description="manual status check"))
    monkeypatch.setattr(interrupt_mod, "_manager", manager)
    calls = 0

    async def status() -> dict:
        nonlocal calls
        calls += 1
        return {"ok": True}

    monkeypatch.setitem(REGISTRY, "status", ToolSpec("status", "status", {}, status))
    event_log = _EventLog()

    interrupted = await agent_loop._run_one_tool(
        "status", {}, sender_tier="owner", event_log=event_log
    )
    assert interrupted["error"] == "interrupted"
    assert interrupted["interrupt_id"].startswith("interrupt_")
    assert calls == 0

    assert manager.resolve_interrupt(
        interrupted["interrupt_id"],
        InterruptAction.APPROVE,
        resolved_by="owner-id",
    )
    resumed = await agent_loop._run_one_tool("status", {}, sender_tier="owner", event_log=event_log)
    assert resumed["ok"] is True
    assert calls == 1

    interrupted_again = await agent_loop._run_one_tool(
        "status", {}, sender_tier="owner", event_log=event_log
    )
    assert interrupted_again["error"] == "interrupted"
    assert calls == 1


@pytest.mark.asyncio
async def test_runtime_mutation_requires_exact_owner_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = InterruptManager()
    for rule in create_default_rules():
        manager.add_rule(rule)
    monkeypatch.setattr(interrupt_mod, "_manager", manager)
    calls = 0

    async def fake_exec(*, command: str) -> dict:
        nonlocal calls
        calls += 1
        return {"ok": True, "command": command}

    monkeypatch.setitem(REGISTRY, "exec", ToolSpec("exec", "exec", {"command": "str"}, fake_exec))
    event_log = _EventLog()
    args = {"command": "kill -9 $(pgrep norax)"}

    interrupted = await agent_loop._run_one_tool(
        "exec",
        args,
        sender_tier="owner",
        event_log=event_log,
    )
    assert interrupted["error"] == "interrupted"
    assert calls == 0

    assert manager.resolve_interrupt(
        interrupted["interrupt_id"],
        InterruptAction.APPROVE,
        resolved_by="owner-id",
    )
    resumed = await agent_loop._run_one_tool(
        "exec",
        args,
        sender_tier="owner",
        event_log=event_log,
    )
    assert resumed["ok"] is True
    assert calls == 1


@pytest.mark.asyncio
async def test_interrupt_subsystem_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenManager:
        def consume_resolution(self, *_args, **_kwargs):
            raise ImportError("interrupt policy unavailable")

    monkeypatch.setattr(interrupt_mod, "_manager", BrokenManager())
    calls = 0

    async def status() -> dict:
        nonlocal calls
        calls += 1
        return {"ok": True}

    monkeypatch.setitem(REGISTRY, "status", ToolSpec("status", "status", {}, status))
    result = await agent_loop._run_one_tool(
        "status",
        {},
        sender_tier="owner",
        event_log=_EventLog(),
    )

    assert result["error"] == "interrupt_gate_error"
    assert result["_not_executed"] is True
    assert calls == 0


def test_mcp_rejects_hidden_and_unapproved_sensitive_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NORAX_MCP_TRUST_HOST_APPROVALS", raising=False)

    hidden_args, hidden_denial = _authorize_mcp_call(
        "sandbox_exec",
        {"command": "echo safe"},
        sender_tier="owner",
    )
    sensitive_args, sensitive_denial = _authorize_mcp_call(
        "exec",
        {"command": "kill -9 $(pgrep norax)"},
        sender_tier="owner",
    )
    safe_args, safe_denial = _authorize_mcp_call(
        "exec",
        {"command": "git status --short"},
        sender_tier="owner",
    )

    assert hidden_args is None
    assert hidden_denial is not None and hidden_denial["error"] == "tool_not_exposed"
    assert sensitive_args is None
    assert sensitive_denial is not None
    assert sensitive_denial["error"] == "authenticated_host_approval_required"
    assert sensitive_denial["resumable"] is False
    assert safe_args == {"command": "git status --short"}
    assert safe_denial is None

    monkeypatch.setenv("NORAX_MCP_TRUST_HOST_APPROVALS", "1")
    trusted_args, trusted_denial = _authorize_mcp_call(
        "exec",
        {"command": "kill -9 $(pgrep norax)"},
        sender_tier="owner",
    )
    assert trusted_args == {"command": "kill -9 $(pgrep norax)"}
    assert trusted_denial is None


@pytest.mark.asyncio
async def test_modified_approval_is_rechecked_by_risk_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = InterruptManager()
    manager.add_rule(InterruptRule(tools={"exec"}, description="manual exec check"))
    monkeypatch.setattr(interrupt_mod, "_manager", manager)
    event_log = _EventLog()
    original = {"command": "echo safe"}

    interrupted = await agent_loop._run_one_tool(
        "exec", original, sender_tier="owner", event_log=event_log
    )
    assert manager.resolve_interrupt(
        interrupted["interrupt_id"],
        InterruptAction.MODIFY,
        {"command": "rm -rf /"},
        resolved_by="owner-id",
    )

    result = await agent_loop._run_one_tool(
        "exec", original, sender_tier="owner", event_log=event_log
    )
    assert result["error"] == "risk_denied"
    assert result["_not_executed"] is True


@pytest.mark.asyncio
async def test_owner_only_approval_command_resolves_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = InterruptManager()
    monkeypatch.setattr(interrupt_mod, "_manager", manager)
    pending = manager.create_interrupt(
        "remote_write", {}, InterruptReason.WRITE_OPERATION, "remote write"
    )
    owner = SimpleNamespace(sender=SimpleNamespace(id="owner-id", tier="owner"))
    user = SimpleNamespace(sender=SimpleNamespace(id="user-id", tier="user"))
    command = ParsedCommand(
        name="approve",
        args=[pending.interrupt_id],
        raw=f"/approve {pending.interrupt_id}",
    )

    denied = await handle(command, user, object())  # type: ignore[arg-type]
    assert "Only the owner" in denied.reply
    assert manager.get_interrupt(pending.interrupt_id).resolved is False  # type: ignore[union-attr]

    approved = await handle(command, owner, object())  # type: ignore[arg-type]
    assert "Approved" in approved.reply
    assert manager.get_interrupt(pending.interrupt_id).resolved is True  # type: ignore[union-attr]
