"""Orchestrator plan parsing and round-cap helpers."""

import asyncio

import pytest

from norax.brain.orchestrator import FALLBACK_PLANNER_MODEL, Orchestrator
from norax.brain.strong_model_scaffold import (
    ORCHESTRATOR_PLANNER_MODEL,
    build_orchestrator_scaffold,
    resolve_planner_model,
)


def test_resolve_planner_model_opus_and_composer():
    from norax.brain.strong_model_scaffold import ORCHESTRATOR_OPUS_PLANNER

    assert resolve_planner_model("claude-opus-4-8-thinking-high") == "claude-opus-4-8-thinking-high"
    assert resolve_planner_model("composer/opus-4.6-thinking") == ORCHESTRATOR_OPUS_PLANNER
    assert resolve_planner_model("composer/composer-2.5") == ORCHESTRATOR_PLANNER_MODEL
    assert resolve_planner_model("qwen3-coder-next:cloud") == "qwen3-coder-next:cloud"


def test_build_orchestrator_scaffold_has_protocol():
    text = build_orchestrator_scaffold(planner_model="claude-opus-4-8-thinking-high")
    assert "ORCHESTRATOR_MODE" in text
    assert "DONE:" in text
    assert "Qwen Coder executes" in text
    assert "claude-opus-4-8-thinking-high" in text


def test_parse_done_marker():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    result = orch._parse_planner_output("DONE: all tests pass", ["read", "exec"])
    assert result.done is True
    assert result.final == "all tests pass"
    assert result.steps == []


def test_parse_done_marker_case_insensitive():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    result = orch._parse_planner_output("done: ok\nmore text", ["read"])
    assert result.done is True
    assert "ok" in result.final


def test_parse_plan_steps():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = (
        "THINK:\nNeed to inspect file first.\n\n"
        "PLAN:\n"
        '1. read path="foo.py" reason="inspect"\n'
        '2. exec command="pytest -q" reason="verify"\n'
    )
    result = orch._parse_planner_output(content, ["read", "exec"])
    assert result.done is False
    assert len(result.steps) == 2
    assert result.steps[0]["tool"] == "read"
    assert result.steps[0]["args"]["path"] == "foo.py"
    assert result.steps[1]["tool"] == "exec"
    assert result.steps[1]["args"]["command"] == "pytest -q"


def test_parse_plan_ignores_unknown_tools():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = 'PLAN:\n1. fly path="sky" reason="impossible"\n2. read path="x.py" reason="ok"\n'
    result = orch._parse_planner_output(content, ["read"])
    assert len(result.steps) == 1
    assert result.steps[0]["tool"] == "read"


def test_parse_plan_single_quoted_args():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = "PLAN:\n1. exec command='ls -la' reason='list files'\n"
    result = orch._parse_planner_output(content, ["exec"])
    assert len(result.steps) == 1
    assert result.steps[0]["args"]["command"] == "ls -la"


def test_parse_plan_unquoted_args():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = "PLAN:\n1. read path=config.json reason=inspect\n"
    result = orch._parse_planner_output(content, ["read"])
    assert len(result.steps) == 1
    assert result.steps[0]["args"]["path"] == "config.json"


def test_parse_json_tool_call_block():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = (
        "THINK: I need to read the file.\n"
        "```json\n"
        '{"name": "read", "arguments": {"path": "foo.py"}}\n'
        "```\n"
    )
    result = orch._parse_planner_output(content, ["read", "exec"])
    assert result.done is False
    assert len(result.steps) == 1
    assert result.steps[0]["tool"] == "read"
    assert result.steps[0]["args"]["path"] == "foo.py"


def test_parse_json_tool_call_block_function_format():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = (
        "```json\n"
        '{"function": {"name": "exec", "arguments": "{\\"command\\": \\"echo hi\\"}"}}\n'
        "```\n"
    )
    result = orch._parse_planner_output(content, ["exec"])
    assert len(result.steps) == 1
    assert result.steps[0]["tool"] == "exec"
    assert result.steps[0]["args"]["command"] == "echo hi"


def test_parse_no_plan_falls_to_final():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = "Everything looks good, no further actions needed."
    result = orch._parse_planner_output(content, ["read", "exec"])
    assert result.done is True
    assert result.final == content
    assert result.steps == []


def test_parse_plan_with_no_permitted_steps_is_incomplete():
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    result = orch._parse_planner_output(
        'PLAN:\n1. delete_everything target="/" reason="bad tool"\n',
        ["read"],
    )

    assert result.done is True
    assert result.completion_signal == "invalid_plan"
    assert "not completed" in result.final


def test_orchestrator_result_requires_verified_mutation() -> None:
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    orch._planner_model = "test-planner"
    trace = [
        {
            "name": "write",
            "tool": "write",
            "args": {"path": "feature.py", "content": "enabled = True\n"},
            "result": {"ok": True, "path": "feature.py"},
            "ok": True,
        }
    ]

    result = orch._finalize_run(
        content="I updated feature.py successfully.",
        trace=trace,
        rounds=1,
        user_prompt="Update feature.py",
        completion_signal="explicit_done",
    )

    assert result.complete is False
    assert result.verified_outcome is False
    assert result.status_reason in {"mutation_unverified", "output_verification_failed"}
    assert "withheld" in result.content


def test_orchestrator_result_accepts_target_verified_mutation_and_unpacks() -> None:
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    orch._planner_model = "test-planner"
    trace = [
        {
            "name": "write",
            "tool": "write",
            "args": {"path": "feature.py", "content": "enabled = True\n"},
            "result": {"ok": True, "path": "feature.py"},
            "ok": True,
        },
        {
            "name": "read",
            "tool": "read",
            "args": {"path": "feature.py"},
            "result": {"ok": True, "path": "feature.py", "content": "enabled = True\n"},
            "ok": True,
        },
    ]

    result = orch._finalize_run(
        content="Updated and verified feature.py.",
        trace=trace,
        rounds=2,
        user_prompt="Update feature.py",
        completion_signal="explicit_done",
    )
    content, unpacked_trace, rounds = result

    assert result.complete is True
    assert result.verified_outcome is True
    assert content == result.content
    assert unpacked_trace is trace
    assert rounds == 2


def test_round_limit_summary_never_claims_verified_completion() -> None:
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    orch._planner_model = "test-planner"

    result = orch._finalize_run(
        content="Here is the best available summary.",
        trace=[],
        rounds=24,
        user_prompt="Investigate the issue fully",
        completion_signal="round_limit",
    )

    assert result.complete is False
    assert result.verified_outcome is False
    assert result.status_reason == "round_limit"
    assert result.content.startswith("Incomplete")


def test_parse_done_before_plan_wins():
    """DONE: before any PLAN: wins."""
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = "DONE: got it\nPLAN:\n1. read path=x.py reason=inspect\n"
    result = orch._parse_planner_output(content, ["read"])
    assert result.done is True
    assert result.final == "got it"
    assert result.steps == []


def test_parse_plan_before_done_parses_plan():
    """PLAN: before DONE: marker parses the plan (planner is confused, but plan wins)."""
    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    content = "PLAN:\n1. read path=x.py reason=inspect\nDONE: got it\n"
    result = orch._parse_planner_output(content, ["read"])
    # PLAN found first — we trust it
    assert result.done is False
    assert len(result.steps) == 1
    assert result.steps[0]["tool"] == "read"


def test_compress_tool_result_strips_noise():
    result = {
        "tool": "exec",
        "ok": True,
        "result": {
            "ok": True,
            "exit_code": 0,
            "stdout": "hello\n",
            "stderr": "",
            "combined": "hello\n",
            "cwd": "/tmp",
            "started_at": 1234567890,
            "elapsed": 0.01,
        },
        "via": "direct",
        "extra_field": "should be stripped",
    }
    compressed = Orchestrator._compress_tool_result(result)
    assert "ok" in compressed
    assert "exit_code" in compressed
    assert "stdout" in compressed
    assert "via" in compressed
    # These should be stripped
    assert "extra_field" not in compressed
    assert "combined" not in compressed
    assert "cwd" not in compressed
    assert "started_at" not in compressed


def test_resolve_round_cap():
    from norax.brain.orchestrator import HARD_ORCH_ROUND_CAP, MAX_ORCH_ROUNDS

    assert Orchestrator._resolve_round_cap(0) == MAX_ORCH_ROUNDS
    assert Orchestrator._resolve_round_cap(12) == 12
    assert Orchestrator._resolve_round_cap(100) == 100
    assert Orchestrator._resolve_round_cap(250) == 250
    assert Orchestrator._resolve_round_cap(1000) == HARD_ORCH_ROUND_CAP


@pytest.mark.asyncio
async def test_planner_availability_failure_uses_distinct_fallback_without_replay(
    monkeypatch,
):
    from norax.gateway_client import GatewayResponse, GatewayUpstreamError

    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    orch._planner_model = "primary-planner"
    messages = [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": "do the task"},
    ]
    seen_models: list[str] = []

    async def request(req):
        seen_models.append(req.model)
        if len(seen_models) == 1:
            raise GatewayUpstreamError(503, "planner unavailable", req.model)
        return GatewayResponse(
            request_id="test",
            model=req.model,
            content="DONE: recovered",
            tool_calls=[],
            usage={},
            raw={},
        )

    monkeypatch.setattr(orch, "_plan_request", request)
    result = await orch._call_planner(messages, ["read"])

    assert result.done is True
    assert result.final == "recovered"
    assert seen_models == [
        "primary-planner",
        FALLBACK_PLANNER_MODEL,
    ]
    assert messages[0]["content"] == "original system"


@pytest.mark.asyncio
async def test_planner_request_error_is_terminal_without_model_hopping(monkeypatch) -> None:
    from norax.gateway_client import GatewayUpstreamError

    orch = Orchestrator(router=None)  # type: ignore[arg-type]
    orch._planner_model = "primary-planner"
    seen_models: list[str] = []

    async def request(req):
        seen_models.append(req.model)
        raise GatewayUpstreamError(403, "forbidden", req.model)

    monkeypatch.setattr(orch, "_plan_request", request)
    with pytest.raises(GatewayUpstreamError) as caught:
        await orch._call_planner(
            [{"role": "system", "content": "system"}],
            ["read"],
        )

    assert caught.value.status == 403
    assert seen_models == ["primary-planner"]


@pytest.mark.asyncio
async def test_orchestrator_result_reports_turn_wide_planner_usage() -> None:
    from norax.gateway_client import GatewayResponse

    class Client:
        async def chat(self, req):
            return GatewayResponse(
                request_id="planner-call",
                model=req.model,
                content="DONE: completed",
                usage={"input_tokens": 7, "output_tokens": 3},
            )

    client = Client()

    class Router:
        async def chat(self, req):
            return await client.chat(req)

    result = await Orchestrator(Router()).run(
        system_prompt="system",
        user_prompt="answer the question",
        allowed_tools=[],
        sender_tier="owner",
        planner_model="planner-model",
    )

    assert result.content == "completed"
    assert result.usage == {"input_tokens": 7, "output_tokens": 3}


@pytest.mark.asyncio
async def test_orchestrator_preserves_continuation_context() -> None:
    from norax.gateway_client import GatewayResponse

    class Router:
        async def chat(self, req):
            assert {"role": "user", "content": "Project target is parser.py"} in req.messages
            assert req.messages[-1] == {"role": "user", "content": "continue"}
            assert sum(m["role"] == "system" for m in req.messages) == 1
            return GatewayResponse(request_id="r", model=req.model, content="DONE: explained")

    prior = [
        {"role": "system", "content": "obsolete system"},
        {"role": "user", "content": "Project target is parser.py"},
    ]
    await Orchestrator(Router()).run(
        system_prompt="current system",
        user_prompt="continue",
        allowed_tools=[],
        sender_tier="owner",
        planner_model="planner-model",
        prior_messages=prior,
    )
    assert prior[0]["content"] == "obsolete system"


@pytest.mark.asyncio
async def test_orchestrator_deadline_cancels_planner_and_reports_incomplete() -> None:
    cancelled = asyncio.Event()

    class Router:
        async def chat(self, req):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    result = await Orchestrator(Router()).run(
        system_prompt="system",
        user_prompt="complete the task",
        allowed_tools=[],
        sender_tier="owner",
        timeout_seconds=0.02,
    )
    assert cancelled.is_set()
    assert result.complete is False
    assert result.verified_outcome is False
    assert result.status_reason == "flow_timeout"
    assert result.rounds == 1
