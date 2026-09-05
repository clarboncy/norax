"""Phase 11b — agentic tool-use loop."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC
from pathlib import Path

import httpx
import pytest
import respx

from norax.brain import agent_loop
from norax.brain.best_of_n import BestOfN
from norax.brain.output_verifier import OutputVerifier
from norax.brain.self_model import SelfModel
from norax.brain.strong_model_scaffold import build_task_state
from norax.context import CorrectionGate
from norax.dispatch import tools as tool_mod
from norax.gateway_client import GatewayClient, GatewayResponse


class _EventLogStub:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def append(self, kind, payload, attrs=None):
        self.events.append((kind, payload))


def _openai_response(*, content: str = "", tool_calls: list | None = None):
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "id": "r",
        "model": "fake",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


# ---------------------------------------------------------------------------
# Schema rendering
# ---------------------------------------------------------------------------


def test_render_tools_for_llm_shape():
    schemas = tool_mod.render_tools_for_llm(["read", "exec"])
    assert len(schemas) == 2
    s0 = schemas[0]
    assert s0["type"] == "function"
    assert s0["function"]["name"] == "read"
    params = s0["function"]["parameters"]
    assert params["type"] == "object"
    assert "path" in params["properties"]
    assert "path" in params["required"]


def test_render_tools_drops_unknown():
    schemas = tool_mod.render_tools_for_llm(["read", "doesnotexist"])
    names = [s["function"]["name"] for s in schemas]
    assert names == ["read"]


def test_tool_result_summary_requires_literal_boolean_success() -> None:
    assert agent_loop._summarize_tool_result({"ok": True})["ok"] is True
    assert agent_loop._summarize_tool_result({"ok": "false"})["ok"] is False
    assert agent_loop._summarize_tool_result({})["ok"] is False


# ---------------------------------------------------------------------------
# Agent loop: no tool calls → single round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_loop_no_tools_single_round():
    respx.post("http://stub/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_response(content="hello"))
    )
    gw = GatewayClient(base_url="http://stub/v1")
    evlog = _EventLogStub()
    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="sys",
            user_prompt="usr",
            allowed_tools=[],
            sender_tier="owner",
            event_log=evlog,
        )
        assert resp.content == "hello"
        assert trace == []
        assert rounds == 1
        assert resp.raw["accepted_outcome"] is True
        assert resp.raw["verified_outcome"] is False
        assert resp.usage == {"input_tokens": 1, "output_tokens": 1}
        assert resp.raw["usage_scope"] == "turn_total"
        trajectory = next(payload for kind, payload in evlog.events if kind == "agent_trajectory")
        assert trajectory["outcome"] == "accepted_unverified"
        assert trajectory["training_eligible"] is False
    finally:
        await gw.aclose()


@pytest.mark.asyncio
async def test_flow_deadline_cancels_hung_generation_and_reports_actual_limit():
    class HangingGateway:
        def __init__(self) -> None:
            self.cancelled = False

        def provider_kind_for_model(self, _model: str) -> str:
            return "openai"

        async def chat(self, _request):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    gateway = HangingGateway()
    events = _EventLogStub()
    started = time.monotonic()
    response, trace, rounds, _state = await agent_loop.run_agent_loop(
        gateway=gateway,
        model="fake",
        system_prompt="sys",
        user_prompt="usr",
        allowed_tools=[],
        sender_tier="owner",
        event_log=events,
        timeout_seconds=0.03,
    )

    assert time.monotonic() - started < 0.5
    assert gateway.cancelled is True
    assert trace == []
    assert rounds == 1
    assert response.raw["flow_timeout"] is True
    assert response.raw["limit_reason"] == "flow_timeout"
    assert "0.03-second task deadline" in response.content
    assert [kind for kind, _payload in events.events].count("flow_timeout") == 1


@pytest.mark.asyncio
async def test_outcome_learning_counts_one_turn_and_only_fresh_observations(tmp_path):
    events = _EventLogStub()
    self_model = SelfModel(tmp_path / "self-model.json")
    task_state = build_task_state("inspect and fix the code", ["read", "edit"])
    response = GatewayResponse(
        request_id="outcome",
        model="fake",
        content="Fixed and verified.",
        tool_calls=[],
        usage={},
        raw={
            "accepted_outcome": True,
            "verified_outcome": True,
            "objective_outcome_observed": True,
        },
    )
    trace = [
        {"name": "read", "args": {}, "result": {"ok": True, "dur_ms": 4}},
        {
            "name": "read",
            "args": {},
            "result": {"ok": True, "_cached": True, "dur_ms": 0},
        },
        {
            "name": "edit",
            "args": {},
            "result": {"ok": False, "_not_executed": True},
        },
    ]

    await agent_loop.record_agent_outcome(
        event_log=events,
        task_state=task_state,
        trace=trace,
        final_resp=response,
        rounds=1,
        model="fake",
        user_text="inspect and fix the code",
        self_model=self_model,
    )

    assert self_model.profile.total_turns == 1
    assert self_model.profile.total_successes == 1
    assert self_model.profile.total_failures == 0
    assert self_model.profile.tool_stats["read"].total_count == 1
    assert "edit" not in self_model.profile.tool_stats
    assert [kind for kind, _payload in events.events].count("agent_trajectory") == 1


@pytest.mark.asyncio
async def test_delivery_failure_is_negative_end_to_end_evidence(tmp_path):
    events = _EventLogStub()
    self_model = SelfModel(tmp_path / "self-model.json")
    task_state = build_task_state("inspect the code", ["read"])
    response = GatewayResponse(
        request_id="undelivered",
        model="fake",
        content="Verified result that transport never delivered.",
        tool_calls=[],
        usage={},
        raw={
            "accepted_outcome": True,
            "verified_outcome": True,
            "objective_outcome_observed": True,
            "delivery_succeeded": False,
        },
    )

    await agent_loop.record_agent_outcome(
        event_log=events,
        task_state=task_state,
        trace=[{"name": "read", "args": {}, "result": {"ok": True, "dur_ms": 4}}],
        final_resp=response,
        rounds=1,
        model="fake",
        user_text="inspect the code",
        self_model=self_model,
    )

    assert self_model.profile.total_successes == 0
    assert self_model.profile.total_failures == 1
    trajectory = next(payload for kind, payload in events.events if kind == "agent_trajectory")
    assert trajectory["outcome"] == "failure"
    assert trajectory["delivery_succeeded"] is False
    assert trajectory["end_to_end_verified"] is False
    assert trajectory["accepted_outcome"] is False
    assert trajectory["outcome_score"] <= 0.05


@pytest.mark.asyncio
async def test_flow_deadline_cancels_hung_tool_and_preserves_terminal_trace(monkeypatch):
    class ToolCallingGateway:
        def provider_kind_for_model(self, _model: str) -> str:
            return "openai"

        async def chat(self, _request):
            return GatewayResponse(
                request_id="tool-round",
                model="fake",
                content="",
                tool_calls=[
                    {
                        "id": "status-1",
                        "type": "function",
                        "function": {"name": "status", "arguments": "{}"},
                    }
                ],
                usage={},
                raw={},
            )

    cancelled = False

    async def hung_status() -> dict:
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    monkeypatch.setattr(tool_mod.REGISTRY["status"], "fn", hung_status)
    started = time.monotonic()
    response, trace, rounds, _state = await agent_loop.run_agent_loop(
        gateway=ToolCallingGateway(),
        model="fake",
        system_prompt="sys",
        user_prompt="check status",
        allowed_tools=["status"],
        sender_tier="owner",
        event_log=_EventLogStub(),
        timeout_seconds=0.04,
    )

    assert time.monotonic() - started < 0.5
    assert cancelled is True
    assert rounds == 1
    assert response.raw["limit_reason"] == "flow_timeout"
    assert len(trace) == 1
    assert trace[0]["name"] == "status"
    assert trace[0]["result"]["error"] == "flow_timeout"
    assert trace[0]["result"]["cancelled"] is True


@pytest.mark.asyncio
@respx.mock
async def test_opt_in_best_of_n_usage_includes_original_and_alternative() -> None:
    route = respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_openai_response(
                    content="The original answer contains enough detail to enter candidate selection."
                ),
            ),
            httpx.Response(
                200,
                json=_openai_response(
                    content="## Result\n\nThe alternative answer is concrete, structured, and complete."
                ),
            ),
        ]
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    try:
        response, _trace, rounds, _state = await agent_loop.run_agent_loop(
            gateway=gateway,
            model="fake",
            system_prompt="sys",
            user_prompt="Provide a detailed implementation result.",
            allowed_tools=[],
            sender_tier="owner",
            event_log=_EventLogStub(),
            best_of_n=BestOfN(gateway),
        )
    finally:
        await gateway.aclose()

    assert rounds == 1
    assert route.call_count == 2
    assert response.usage == {"input_tokens": 2, "output_tokens": 2}


@pytest.mark.asyncio
@respx.mock
async def test_correction_revision_reenters_remaining_quality_pipeline():
    class _Neuron:
        text = "EMAIL:owner@example.com|W5"
        weight = 1.3

    async def retrieve(query, k):
        return [(_Neuron(), 0.9, "local")]

    route = respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(content="Your email is wrong@example.com")),
            httpx.Response(200, json=_openai_response(content="Your email is owner@example.com")),
        ]
    )
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        resp, _, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="sys",
            user_prompt="What is my official email address?",
            allowed_tools=[],
            sender_tier="owner",
            event_log=_EventLogStub(),
            correction_gate=CorrectionGate(retrieve=retrieve),
        )

        assert rounds == 2
        assert resp.content == "Your email is owner@example.com"
        assert resp.usage == {"input_tokens": 2, "output_tokens": 2}
        assert resp.raw["verified_outcome"] is True
        revision_payload = json.loads(route.calls[1].request.content)
        assert any(
            message.get("role") == "assistant" and "wrong@example.com" in message.get("content", "")
            for message in revision_payload["messages"]
        )
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_output_revision_is_reverified_and_accepted_only_when_better():
    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(content="The result is 2 + 2 = 5.")),
            httpx.Response(200, json=_openai_response(content="The result is 2 + 2 = 4.")),
        ]
    )
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        resp, _, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="sys",
            user_prompt="Calculate two plus two and provide the checked result.",
            allowed_tools=[],
            sender_tier="owner",
            event_log=_EventLogStub(),
            output_verifier=OutputVerifier(),
        )

        assert rounds == 2
        assert resp.content == "The result is 2 + 2 = 4."
        assert resp.usage == {"input_tokens": 2, "output_tokens": 2}
        assert resp.raw["verified_outcome"] is True
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_output_revision_that_is_not_better_is_rejected():
    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(content="The result is 2 + 2 = 5.")),
            httpx.Response(200, json=_openai_response(content="The result is 3 + 3 = 9.")),
        ]
    )
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        resp, _, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="sys",
            user_prompt="Calculate two plus two and provide the checked result.",
            allowed_tools=[],
            sender_tier="owner",
            event_log=_EventLogStub(),
            output_verifier=OutputVerifier(),
        )

        assert rounds == 2
        assert "couldn't safely return" in resp.content
        assert "Arithmetic error" in resp.content
        assert resp.raw["incomplete"] is True
        assert resp.raw["verified_outcome"] is False
        assert resp.usage == {"input_tokens": 2, "output_tokens": 2}
    finally:
        await gw.aclose()


# ---------------------------------------------------------------------------
# Agent loop: one tool call, then final text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_loop_single_tool_then_final(tmp_path):
    # Create a file for `read` to succeed.
    import os

    os.environ["NORAX_WORKSPACE"] = str(tmp_path)
    f = tmp_path / "hi.txt"
    f.write_text("hello world")

    tool_call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "read", "arguments": f'{{"path":"{f}"}}'},
    }
    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(tool_calls=[tool_call])),
            httpx.Response(200, json=_openai_response(content="Read: hello world")),
        ]
    )
    gw = GatewayClient(base_url="http://stub/v1")
    evlog = _EventLogStub()
    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="sys",
            user_prompt="read hi.txt",
            allowed_tools=["read"],
            sender_tier="owner",
            event_log=evlog,
        )
        assert rounds == 2
        assert resp.content == "Read: hello world"
        assert len(trace) == 1
        assert trace[0]["name"] == "read"
        assert trace[0]["result"].get("ok") is True
        assert resp.usage == {"input_tokens": 2, "output_tokens": 2}
        # tool_call event logged
        kinds = [k for k, _ in evlog.events]
        assert "tool_call" in kinds
        assert "agent_round" in kinds
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_same_path_can_be_read_again_after_a_successful_write(tmp_path):
    import os

    os.environ["NORAX_WORKSPACE"] = str(tmp_path)
    path = tmp_path / "state.txt"
    path.write_text("before")

    def tool_call(call_id: str, name: str, arguments: dict) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_openai_response(
                    tool_calls=[tool_call("read-before", "read", {"path": str(path)})]
                ),
            ),
            httpx.Response(
                200,
                json=_openai_response(
                    tool_calls=[
                        tool_call(
                            "write",
                            "write",
                            {"path": str(path), "content": "after"},
                        )
                    ]
                ),
            ),
            httpx.Response(
                200,
                json=_openai_response(
                    tool_calls=[tool_call("read-after", "read", {"path": str(path)})]
                ),
            ),
            httpx.Response(200, json=_openai_response(content="Updated and verified state.txt.")),
        ]
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    try:
        response, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gateway,
            model="fake",
            system_prompt="system",
            user_prompt="Update state.txt to contain after and verify it.",
            allowed_tools=["read", "write"],
            sender_tier="owner",
            event_log=_EventLogStub(),
        )
    finally:
        await gateway.aclose()

    assert rounds == 4
    assert trace[2]["result"]["ok"] is True
    assert trace[2]["result"]["content"] == "after"
    assert response.raw["verified_outcome"] is True


@pytest.mark.asyncio
@respx.mock
async def test_successful_exec_invalidates_prior_read_cache(tmp_path, monkeypatch):
    path = tmp_path / "state.txt"
    path.write_text("before")
    read_calls = 0

    async def fake_read(*, path: str) -> dict:
        nonlocal read_calls
        read_calls += 1
        return {"ok": True, "path": path, "content": Path(path).read_text()}

    async def fake_exec(*, command: str) -> dict:
        assert command.startswith("printf after >")
        path.write_text("after")
        return {"ok": True, "exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setitem(
        tool_mod.REGISTRY,
        "read",
        tool_mod.ToolSpec("read", "read", {"path": "str"}, fake_read),
    )
    monkeypatch.setitem(
        tool_mod.REGISTRY,
        "exec",
        tool_mod.ToolSpec("exec", "exec", {"command": "str"}, fake_exec),
    )

    def tool_call(call_id: str, name: str, arguments: dict) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    route = respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json=_openai_response(
                    tool_calls=[tool_call("read-before", "read", {"path": str(path)})]
                ),
            ),
            httpx.Response(
                200,
                json=_openai_response(
                    tool_calls=[
                        tool_call(
                            "exec-write",
                            "exec",
                            {"command": f"printf after > {path}"},
                        )
                    ]
                ),
            ),
            httpx.Response(
                200,
                json=_openai_response(
                    tool_calls=[tool_call("read-after", "read", {"path": str(path)})]
                ),
            ),
            httpx.Response(200, json=_openai_response(content="Updated and verified state.txt.")),
        ]
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    try:
        response, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gateway,
            model="fake",
            system_prompt="system",
            user_prompt="Update state.txt through exec and verify it.",
            allowed_tools=["read", "exec"],
            sender_tier="owner",
            event_log=_EventLogStub(),
        )
    finally:
        await gateway.aclose()

    assert route.call_count == 4
    assert rounds == 4
    assert read_calls == 2
    assert trace[2]["result"]["content"] == "after"
    assert trace[2]["result"].get("_cached") is None
    assert response.raw["verified_outcome"] is True


# ---------------------------------------------------------------------------
# Risk denial: non-owner cannot call exec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_loop_risk_denial_logged_but_loop_continues():
    tool_call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "exec", "arguments": '{"cmd":"ls"}'},
    }
    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(tool_calls=[tool_call])),
            httpx.Response(200, json=_openai_response(content="not allowed, sorry")),
        ]
    )
    gw = GatewayClient(base_url="http://stub/v1")
    evlog = _EventLogStub()
    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="sys",
            user_prompt="run ls",
            allowed_tools=["exec"],
            sender_tier="user",  # non-owner
            event_log=evlog,
        )
        assert rounds == 2
        assert trace[0]["result"]["ok"] is False
        assert trace[0]["result"]["error"] == "risk_denied"
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_loop_rejects_hallucinated_tool_outside_turn_manifest(tmp_path):
    path = tmp_path / "must-not-exist.txt"
    tool_call = {
        "id": "tc1",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": json.dumps({"path": str(path), "content": "unsafe"}),
        },
    }
    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(tool_calls=[tool_call])),
            httpx.Response(200, json=_openai_response(content="The write was not allowed.")),
        ]
    )
    gateway = GatewayClient(base_url="http://stub/v1")
    try:
        _, trace, _, _ = await agent_loop.run_agent_loop(
            gateway=gateway,
            model="fake",
            system_prompt="sys",
            user_prompt="inspect only",
            allowed_tools=["read"],
            sender_tier="owner",
            event_log=_EventLogStub(),
        )
    finally:
        await gateway.aclose()

    assert trace[0]["result"]["error"] == "tool_not_allowed"
    assert trace[0]["result"]["_not_executed"] is True
    assert not path.exists()


# ---------------------------------------------------------------------------
# Duplicate call blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_loop_blocks_duplicate_calls_same_turn(tmp_path):
    import os

    os.environ["NORAX_WORKSPACE"] = str(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")

    dup = {
        "id": "tc",
        "type": "function",
        "function": {"name": "read", "arguments": f'{{"path":"{f}"}}'},
    }
    # Model tries the same call twice in row-1, then gives up.
    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(tool_calls=[dup, dup])),
            httpx.Response(200, json=_openai_response(content="done")),
        ]
    )
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="s",
            user_prompt="u",
            allowed_tools=["read"],
            sender_tier="owner",
            event_log=_EventLogStub(),
        )
        assert rounds == 2
        assert len(trace) == 2
        assert trace[0]["result"].get("ok") is True
        # With turn-cache active, the second identical read-safe call
        # returns the cached result instead of being blocked.
        assert trace[1]["result"].get("ok") is True
        assert trace[1]["result"].get("_cached") is True
    finally:
        await gw.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_loop_allows_same_file_with_different_read_ranges(tmp_path):
    f = tmp_path / "paged.txt"
    f.write_text("first\nsecond\nthird\n")

    def read_call(call_id: str, offset: int):
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "read",
                "arguments": f'{{"path":"{f}","offset":{offset},"limit":1}}',
            },
        }

    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(tool_calls=[read_call("tc1", 0)])),
            httpx.Response(200, json=_openai_response(tool_calls=[read_call("tc2", 1)])),
            httpx.Response(200, json=_openai_response(content="Read both ranges.")),
        ]
    )
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        _, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="s",
            user_prompt="inspect both lines",
            allowed_tools=["read"],
            sender_tier="owner",
            event_log=_EventLogStub(),
        )
        assert rounds == 3
        assert [item["result"]["ok"] for item in trace] == [True, True]
        assert trace[0]["result"]["content"] == "first"
        assert trace[1]["result"]["content"] == "second"
    finally:
        await gw.aclose()


# ---------------------------------------------------------------------------
# Max rounds safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_loop_max_rounds_returns_verified_incomplete(tmp_path):
    import os

    os.environ["NORAX_WORKSPACE"] = str(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")

    def tc(i):
        return {
            "id": f"tc{i}",
            "type": "function",
            "function": {"name": "read", "arguments": f'{{"path":"{f}","_i":{i}}}'},
        }

    # 3 rounds of tool calls, then a deterministic incomplete response.
    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(tool_calls=[tc(0)])),
            httpx.Response(200, json=_openai_response(tool_calls=[tc(1)])),
            httpx.Response(200, json=_openai_response(tool_calls=[tc(2)])),
            httpx.Response(200, json=_openai_response(content="summary")),
        ]
    )
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="s",
            user_prompt="u",
            allowed_tools=["read"],
            sender_tier="owner",
            event_log=_EventLogStub(),
            max_rounds=3,
        )
        # No extra model call is made to narrate unfinished work.
        assert rounds == 3
        assert "not verified" in resp.content
        assert resp.raw["limit_reason"] == "max_rounds"
    finally:
        await gw.aclose()


# ---------------------------------------------------------------------------
# Discord chunking
# ---------------------------------------------------------------------------


def test_split_discord_short_returns_one():
    from norax.adapter.discord_in import _split_discord_message

    assert _split_discord_message("hi") == ["hi"]


def test_split_discord_splits_on_paragraph():
    from norax.adapter.discord_in import _split_discord_message

    text = ("a" * 1000) + "\n\n" + ("b" * 1500)
    chunks = _split_discord_message(text, limit=1900)
    assert len(chunks) == 2
    assert all(len(c) <= 2000 for c in chunks)


def test_split_discord_preserves_code_fence():
    from norax.adapter.discord_in import _split_discord_message

    # Long code block that must be split
    code = "x\n" * 1500
    text = f"Here's code:\n```python\n{code}\n```\nend"
    chunks = _split_discord_message(text, limit=1900)
    assert len(chunks) >= 2
    # Every chunk's ``` count must be even (properly paired).
    for c in chunks:
        assert c.count("```") % 2 == 0, f"unpaired fence in chunk: {c[:80]!r}"


# ---------------------------------------------------------------------------
# Concatenated-JSON tool-call arguments (Gemma hallucination)
# ---------------------------------------------------------------------------


def test_tool_calls_split_concatenated_json_picks_best_schema_match():
    """Gemma sometimes emits `{...}{...}` in one arguments string with one
    tool name. We should pick the chunk whose keys best match the tool's
    schema, so the dispatch call gets valid args.
    """
    from norax.brain.agent_loop import _split_concatenated_json, _tool_calls_from
    from norax.gateway_client import GatewayResponse

    # Two intended calls squashed under name="read": one exec-style, one read-style.
    squashed = '{"command":"echo hi"}{"path":"/tmp/x.txt"}'
    assert len(_split_concatenated_json(squashed)) == 2

    resp = GatewayResponse(
        request_id="r1",
        model="gemma4",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": squashed},
            }
        ],
        usage={},
        raw={},
    )
    calls = _tool_calls_from(resp)
    assert len(calls) == 1
    assert calls[0]["name"] == "read"
    # Should have picked the `path` chunk (matches read's schema).
    assert calls[0]["args"] == {"path": "/tmp/x.txt"}


def test_tool_calls_normal_single_json_unchanged():
    """Regression: well-formed single JSON args should still parse normally."""
    from norax.brain.agent_loop import _tool_calls_from
    from norax.gateway_client import GatewayResponse

    resp = GatewayResponse(
        request_id="r1",
        model="x",
        content="",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"/etc/hostname"}'},
            }
        ],
        usage={},
        raw={},
    )
    calls = _tool_calls_from(resp)
    assert len(calls) == 1
    assert calls[0]["args"] == {"path": "/etc/hostname"}


def test_split_concatenated_json_handles_whitespace_and_single():
    from norax.brain.agent_loop import _split_concatenated_json

    assert _split_concatenated_json('{"a":1}') == ['{"a":1}']
    assert _split_concatenated_json('  {"a":1}  {"b":2}  ') == ['{"a":1}', '{"b":2}']
    assert _split_concatenated_json("") == []


# ---------------------------------------------------------------------------
# Image-only messages must not be silently dropped
# ---------------------------------------------------------------------------


def test_l8_plan_image_only_not_silent():
    """When body is empty but attachments exist (image-only message),
    l8_plan should set decision='emit_reply', not 'silent'."""
    from datetime import datetime

    from norax.brain.hot_path import BrainContext, l8_plan
    from norax.envelope import Principal, SensoryInput

    env = SensoryInput(
        channel="chat",
        source="discord",
        message_id="test1",
        timestamp=datetime.now(UTC),
        sender=Principal(id="1", label="U", trust=True, tier="owner"),
        body="",  # empty text
        attachments=[
            {
                "url": "https://cdn.discordapp.com/test.png",
                "content_type": "image/png",
                "filename": "screenshot.png",
                "size": 12345,
            }
        ],
        trusted=True,
    )
    ctx = BrainContext(env=env, decision="unknown", allowed_tools=set())
    ctx = l8_plan(ctx)
    assert ctx.decision == "emit_reply", f"expected emit_reply for image-only, got {ctx.decision}"


def test_l8_plan_empty_body_no_attachments_is_silent():
    """When body is empty AND no attachments, l8_plan should stay silent."""
    from datetime import datetime

    from norax.brain.hot_path import BrainContext, l8_plan
    from norax.envelope import Principal, SensoryInput

    env = SensoryInput(
        channel="chat",
        source="discord",
        message_id="test2",
        timestamp=datetime.now(UTC),
        sender=Principal(id="1", label="U", trust=True, tier="owner"),
        body="",
        attachments=[],
        trusted=True,
    )
    ctx = BrainContext(env=env, decision="unknown", allowed_tools=set())
    ctx = l8_plan(ctx)
    assert ctx.decision == "silent"


def test_render_user_message_image_only_default_prompt():
    """When body is empty but an image is attached, the text part should
    default to '(image attached)' so the model knows to look at it."""
    from datetime import datetime

    from norax.envelope import Principal, SensoryInput
    from norax.prompt.assembler import _render_user_message

    env = SensoryInput(
        channel="chat",
        source="discord",
        message_id="test3",
        timestamp=datetime.now(UTC),
        sender=Principal(id="1", label="U", trust=True, tier="owner"),
        body="",
        attachments=[
            {
                "url": "https://cdn.discordapp.com/test.png",
                "content_type": "image/png",
                "filename": "screenshot.png",
                "size": 12345,
            }
        ],
        trusted=True,
    )
    result = _render_user_message(env)
    # Should be a list of content parts
    assert isinstance(result, list), f"expected list for multimodal, got {type(result)}"
    text_part = result[0]
    assert text_part["type"] == "text"
    assert text_part["text"] == "(image attached)"
    assert any(p["type"] == "image_url" for p in result)


# ---------------------------------------------------------------------------
# Regression: no-progress stall detection (cached calls must not mask stall)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_no_progress_stall_forces_final(tmp_path):
    """When all tool calls are cached for NO_PROGRESS_ROUND_LIMIT rounds,
    the loop must force a final answer instead of spinning forever."""
    import os

    os.environ["NORAX_WORKSPACE"] = str(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")

    def tc(i):
        return {
            "id": f"tc{i}",
            "type": "function",
            "function": {"name": "read", "arguments": f'{{"path":"{f}"}}'},
        }

    # Round 0: read (novel, caches result)
    # Rounds 1-4: same read (all cached, zero novel calls)
    # Round 5: forced final
    responses = [httpx.Response(200, json=_openai_response(tool_calls=[tc(0)]))]
    for _ in range(4):
        responses.append(httpx.Response(200, json=_openai_response(tool_calls=[tc(0)])))
    responses.append(httpx.Response(200, json=_openai_response(content="final answer")))

    respx.post("http://stub/v1/chat/completions").mock(side_effect=responses)
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="s",
            user_prompt="u",
            allowed_tools=["read"],
            sender_tier="owner",
            event_log=_EventLogStub(),
            max_rounds=40,
        )
        # Should stop well before max_rounds due to no-progress stall
        assert rounds < 15, f"expected early termination, got {rounds} rounds"
        assert resp.content == "final answer"
    finally:
        await gw.aclose()


# ---------------------------------------------------------------------------
# Regression: per-turn tool-call ceiling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_tool_call_ceiling_forces_final(tmp_path):
    """When total_tool_calls exceeds MAX_TOOL_CALLS_PER_TURN, the loop
    must force a final answer."""
    import os

    os.environ["NORAX_WORKSPACE"] = str(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")

    # Each round emits 80 tool calls with unique args to avoid dedup.
    def make_calls(i):
        calls = []
        for j in range(80):
            calls.append(
                {
                    "id": f"tc{i}_{j}",
                    "type": "function",
                    "function": {"name": "read", "arguments": f'{{"path":"{f}","_i":{i}_{j}}}'},
                }
            )
        return calls

    # Round 0: 80 calls (total=80)
    # Round 1: 80 calls (total=160)
    # Round 2 requests 80 calls, but only the remaining 40 may execute before
    # the hard 200-call ceiling terminates the turn.
    responses = [
        httpx.Response(200, json=_openai_response(tool_calls=make_calls(0))),
        httpx.Response(200, json=_openai_response(tool_calls=make_calls(1))),
        httpx.Response(200, json=_openai_response(tool_calls=make_calls(2))),
        httpx.Response(200, json=_openai_response(tool_calls=make_calls(3))),
        httpx.Response(200, json=_openai_response(content="ceiling hit final")),
    ]

    route = respx.post("http://stub/v1/chat/completions").mock(side_effect=responses)
    gw = GatewayClient(base_url="http://stub/v1")
    try:
        resp, trace, rounds, _ = await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="s",
            user_prompt="u",
            allowed_tools=["read"],
            sender_tier="owner",
            event_log=_EventLogStub(),
            max_rounds=40,
        )
        assert "not verified" in resp.content
        assert resp.raw["limit_reason"] == "tool_call_ceiling"
        assert resp.raw["tool_calls_executed"] == agent_loop.MAX_TOOL_CALLS_PER_TURN
        assert resp.raw["tool_calls_rejected"] == 40
        assert len(trace) == agent_loop.MAX_TOOL_CALLS_PER_TURN
        # Two complete 80-call rounds plus only the remaining 40 calls. The
        # fourth provider response is never requested.
        assert rounds == 3
        assert route.call_count == 3
    finally:
        await gw.aclose()


# ---------------------------------------------------------------------------
# Regression: lowered round caps
# ---------------------------------------------------------------------------


def test_default_max_rounds_is_150():
    """Ensure DEFAULT_MAX_ROUNDS is 150 (bounded to prevent re-reading loops)."""
    assert agent_loop.DEFAULT_MAX_ROUNDS == 150


def test_hard_round_cap_leaves_default_headroom():
    """Ensure HARD_ROUND_CAP leaves headroom above the normal 150-round budget."""
    assert (
        agent_loop.HARD_ROUND_CAP == 200
        or agent_loop.HARD_ROUND_CAP > agent_loop.DEFAULT_MAX_ROUNDS
    )


def test_final_verification_nudge_limit_is_3():
    """Ensure a broken verification classifier cannot spin indefinitely."""
    assert agent_loop.FINAL_VERIFICATION_NUDGE_LIMIT == 3


def test_max_tool_calls_per_turn_defined():
    """Ensure MAX_TOOL_CALLS_PER_TURN ceiling exists and is reasonable."""
    assert hasattr(agent_loop, "MAX_TOOL_CALLS_PER_TURN")
    assert agent_loop.MAX_TOOL_CALLS_PER_TURN > 0
    assert agent_loop.MAX_TOOL_CALLS_PER_TURN == 200


def test_no_progress_round_limit_defined():
    """Ensure NO_PROGRESS_ROUND_LIMIT exists and is small."""
    assert hasattr(agent_loop, "NO_PROGRESS_ROUND_LIMIT")
    assert agent_loop.NO_PROGRESS_ROUND_LIMIT <= 10


def test_mutating_exec_calls_are_not_parallel_safe():
    assert not agent_loop._call_can_run_in_parallel(
        {"name": "exec", "args": {"command": "mv live-world live-world.backup"}}
    )
    assert agent_loop._call_can_run_in_parallel(
        {"name": "exec", "args": {"command": "docker compose ps"}}
    )


def test_context_eviction_keeps_external_task_state():
    from norax.brain.strong_model_scaffold import build_task_state

    state = build_task_state("restore Origins without replacing the Realm world", ["edit", "exec"])
    state.note_tool("edit", {"path": "compose.yml"}, {"ok": True, "path": "compose.yml"})
    state.note_tool(
        "exec",
        {"command": "cat compose.yml"},
        {"ok": True, "exit_code": 0, "stdout": "LEVEL_NAME=NoraxOrigins"},
    )
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": state.goal}]
    for index in range(5):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"compose.yml"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": '{"ok":true,"path":"compose.yml"}',
                },
            ]
        )
    compacted, _ = agent_loop._evict_middle_tool_traces(
        messages,
        max_messages=6,
        keep_recent_rounds=1,
        task_state=state,
    )
    marker = next(
        message["content"]
        for message in compacted
        if "EVICTED_CONTEXT" in message.get("content", "")
    )
    assert "restore Origins" in marker
    assert "compose.yml" in marker
    assert "verified_after_write" in marker


@pytest.mark.asyncio
@respx.mock
async def test_agent_progress_callback_receives_completed_round(tmp_path):
    path = tmp_path / "progress.txt"
    path.write_text("ready")
    tool_call = {
        "id": "progress-call",
        "type": "function",
        "function": {"name": "read", "arguments": f'{{"path":"{path}"}}'},
    }
    respx.post("http://stub/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_openai_response(tool_calls=[tool_call])),
            httpx.Response(200, json=_openai_response(content="done")),
        ]
    )
    snapshots = []

    async def on_progress(snapshot):
        snapshots.append(snapshot)

    gw = GatewayClient(base_url="http://stub/v1")
    try:
        await agent_loop.run_agent_loop(
            gateway=gw,
            model="fake",
            system_prompt="system",
            user_prompt="read progress",
            allowed_tools=["read"],
            sender_tier="owner",
            event_log=_EventLogStub(),
            on_progress=on_progress,
        )
    finally:
        await gw.aclose()
    assert snapshots
    assert snapshots[-1]["rounds"] >= 1
    assert snapshots[-1]["trace"][0]["name"] == "read"
    assert snapshots[-1]["task_state"]["tool_calls"] == 1
