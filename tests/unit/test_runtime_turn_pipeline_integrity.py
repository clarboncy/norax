"""Risk-focused tests for the production turn coordinator.

These tests deliberately exercise ``Runtime._handle_turn_inner`` as a unit of
orchestration.  Only external I/O is replaced; windowing, history assembly,
checkpoint decisions, telemetry, accounting, and reply routing remain real.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from norax.config.loader import Config
from norax.envelope import Principal, SensoryInput
from norax.gateway_client import GatewayResponse
from norax.runtime.core import Runtime


class _Events:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict, dict]] = []

    async def append(self, kind: str, payload: dict, **kwargs) -> None:
        self.rows.append((kind, payload, kwargs))

    @contextmanager
    def trace_scope(self):
        yield "trace-1"


class _FailingDeliveryEvents(_Events):
    async def append(self, kind: str, payload: dict, **kwargs) -> None:
        if kind in {"send", "reply"}:
            raise OSError("event sink unavailable")
        await super().append(kind, payload, **kwargs)


class _Outbound:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str, str | None]] = []

    def has(self, source: str) -> bool:
        return source == "test"

    async def send(
        self,
        source: str,
        target: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> dict:
        self.sent.append((source, target, text, reply_to))
        return {"ok": True, "message_id": "out-1"}


class _FailingOutbound(_Outbound):
    async def send(
        self,
        source: str,
        target: str,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> dict:
        self.sent.append((source, target, text, reply_to))
        return {"ok": "false", "error": "transport_unavailable"}


class _TraceCollector:
    def __init__(self) -> None:
        self.ended: list[tuple[str, int]] = []

    async def record_event(self, **_kwargs) -> None:
        return None

    async def end_turn(self, turn_id: str, *, rounds: int) -> None:
        self.ended.append((turn_id, rounds))


def _runtime(tmp_path) -> Runtime:
    (tmp_path / "state").mkdir()
    (tmp_path / "logs").mkdir()
    cfg = Config(
        raw={
            "http": {"bind": "127.0.0.1:0"},
            "owner": {"id": "owner"},
            "gateway": {"base_url": "http://127.0.0.1:1/v1"},
        },
        project_root=tmp_path,
    )
    runtime = Runtime.build(cfg)
    runtime.events = _Events()  # type: ignore[assignment]
    runtime.outbound = _Outbound()  # type: ignore[assignment]
    runtime.agent_os_bridge = None
    runtime._memory_store = None
    runtime._context_injector = None
    runtime._hybrid = None
    runtime._fast_ctx = None
    runtime._user_model = None
    runtime._episodic = None
    runtime._causal_graph = None
    runtime._temporal_graph = None
    runtime._memory_coordinator = None
    runtime._multi_agent_enabled = False
    runtime._planner_enabled = False
    runtime._output_verifier = None
    runtime._self_model = None
    runtime._active_inference = None
    runtime._metacognitive = None
    runtime._best_of_n = None
    runtime._curiosity = None
    runtime._domain_transfer = None
    runtime._analogy_engine = None
    runtime._ensure_cognitive_components = lambda: None  # type: ignore[method-assign]
    return runtime


def _env(*, body: str = "inspect the repository") -> SensoryInput:
    return SensoryInput(
        channel="chat",
        source="test",
        message_id="in-1",
        timestamp=datetime.now(UTC),
        sender=Principal(id="owner", label="Owner", trust=True, tier="owner"),
        body=body,
        trusted=True,
        raw={},
        thread_binding=SimpleNamespace(thread_id="thread-1"),
    )


@pytest.mark.asyncio
async def test_turn_pipeline_persists_context_accounts_usage_and_routes_reply(
    tmp_path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    collector = _TraceCollector()
    checkpoints: list[dict] = []

    ctx = SimpleNamespace(
        decision="emit_reply",
        allowed_tools=["read"],
        focus=SimpleNamespace(summary="repository audit"),
        metadata={},
        memory=SimpleNamespace(items=[]),
    )
    rendered = SimpleNamespace(static_hash="static-1", system="system", user="user")

    async def fake_plan_turn(_env, **_kwargs):
        return ctx, rendered

    async def fake_agent_loop(**kwargs):
        assert kwargs["model"] == runtime.default_model
        assert kwargs["allowed_tools"] == ["read"]
        assert kwargs["prior_messages"] == []
        assert kwargs["defer_outcome_recording"] is True
        trace = [
            {
                "name": "read",
                "args": {"path": "README.md"},
                "result": {"ok": True, "content": "Norax"},
            }
        ]
        return (
            GatewayResponse(
                request_id="req-1",
                model="test-model",
                content="[[reply_to_current]] audit complete",
                tool_calls=[],
                usage={"input_tokens": 11, "output_tokens": 7},
                raw={"verified_outcome": True},
            ),
            trace,
            1,
            None,
        )

    monkeypatch.setattr("norax.runtime.core.hot_path.plan_turn", fake_plan_turn)
    monkeypatch.setattr("norax.runtime.core.agent_loop.run_agent_loop", fake_agent_loop)
    monkeypatch.setattr("norax.runtime.checkpoint.load_latest_checkpoint", lambda _channel: None)
    monkeypatch.setattr(
        "norax.runtime.checkpoint.save_checkpoint", lambda **kwargs: checkpoints.append(kwargs)
    )
    monkeypatch.setattr("norax.observability.trace_ui.get_trace_collector", lambda: collector)

    await runtime._handle_turn(_env())

    outbound = runtime.outbound
    assert isinstance(outbound, _Outbound)
    assert outbound.sent == [("test", "thread-1", "audit complete", "in-1")]
    assert checkpoints and checkpoints[0]["status"] == "complete"
    assert checkpoints[0]["model"] == "test-model"
    assert checkpoints[0]["trace"][0]["name"] == "read"
    assert runtime._windows["chat"].body[-1].content == "audit complete"
    event_kinds = [kind for kind, _payload, _kwargs in runtime.events.rows]  # type: ignore[attr-defined]
    assert event_kinds[:2] == ["ingress", "brain"]
    assert "turn_telemetry" in event_kinds
    assert "reply" in event_kinds
    assert "send" in event_kinds
    assert collector.ended == [("turn-1-in-1", 1)]

    metrics = runtime.metrics.render()[0].decode("utf-8")
    assert 'model="test-model"' in metrics
    assert "norax_gateway_tokens_in_total" in metrics
    assert "norax_gateway_tokens_out_total" in metrics
    await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_reply_is_delivered_before_optional_post_turn_learning(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    order: list[str] = []
    outbound = runtime.outbound
    assert isinstance(outbound, _Outbound)
    original_send = outbound.send

    async def ordered_send(*args, **kwargs):
        order.append("reply.send")
        return await original_send(*args, **kwargs)

    outbound.send = ordered_send  # type: ignore[method-assign]

    class _UserModel:
        def render_for_prompt(self, _user_id: str) -> str:
            return ""

        def record_turn(self, **_kwargs) -> None:
            order.append("learning.user_model")

    runtime._user_model = _UserModel()  # type: ignore[assignment]
    runtime._release_current_turn_slot = lambda: order.append("slot.release") or True  # type: ignore[method-assign]
    ctx = SimpleNamespace(
        decision="emit_reply",
        allowed_tools=[],
        focus=SimpleNamespace(summary="fast response"),
        metadata={},
        memory=SimpleNamespace(items=[]),
    )
    rendered = SimpleNamespace(static_hash="static-1", system="system", user="user")

    async def fake_plan_turn(_env, **_kwargs):
        return ctx, rendered

    async def fake_agent_loop(**_kwargs):
        return (
            GatewayResponse(
                request_id="req-fast",
                model="test-model",
                content="done",
                tool_calls=[],
                usage={},
                raw={},
            ),
            [],
            1,
            SimpleNamespace(task_type="general"),
        )

    async def fake_record_agent_outcome(**kwargs):
        assert kwargs["user_text"] == "inspect the repository"
        order.append("learning.agent_outcome")

    monkeypatch.setattr("norax.runtime.core.hot_path.plan_turn", fake_plan_turn)
    monkeypatch.setattr("norax.runtime.core.agent_loop.run_agent_loop", fake_agent_loop)
    monkeypatch.setattr(
        "norax.runtime.core.agent_loop.record_agent_outcome", fake_record_agent_outcome
    )
    monkeypatch.setattr("norax.runtime.checkpoint.load_latest_checkpoint", lambda _channel: None)
    monkeypatch.setattr("norax.runtime.checkpoint.save_checkpoint", lambda **_kwargs: None)

    await runtime._handle_turn(_env())

    assert order == [
        "reply.send",
        "slot.release",
        "learning.agent_outcome",
        "learning.user_model",
    ]
    await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_failed_delivery_is_not_checkpointed_or_learned_as_completed(
    tmp_path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    runtime.outbound = _FailingOutbound()  # type: ignore[assignment]
    checkpoints: list[dict] = []
    recorded: list[dict] = []
    releases: list[str] = []
    runtime._release_current_turn_slot = lambda: releases.append("released") or True  # type: ignore[method-assign]
    ctx = SimpleNamespace(
        decision="emit_reply",
        allowed_tools=[],
        focus=SimpleNamespace(summary=""),
        metadata={},
        memory=SimpleNamespace(items=[]),
    )
    rendered = SimpleNamespace(static_hash="static-1", system="system", user="user")

    async def fake_plan_turn(_env, **_kwargs):
        return ctx, rendered

    async def fake_agent_loop(**_kwargs):
        return (
            GatewayResponse(
                request_id="req-undelivered",
                model="test-model",
                content="generated but not delivered",
                tool_calls=[],
                usage={},
                raw={"accepted_outcome": True, "verified_outcome": True},
            ),
            [],
            1,
            SimpleNamespace(
                task_type="general",
                to_persistable_dict=lambda: {"goal": "respond"},
            ),
        )

    async def fake_record_agent_outcome(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr("norax.runtime.core.hot_path.plan_turn", fake_plan_turn)
    monkeypatch.setattr("norax.runtime.core.agent_loop.run_agent_loop", fake_agent_loop)
    monkeypatch.setattr(
        "norax.runtime.core.agent_loop.record_agent_outcome", fake_record_agent_outcome
    )
    monkeypatch.setattr("norax.runtime.checkpoint.load_latest_checkpoint", lambda _channel: None)
    monkeypatch.setattr(
        "norax.runtime.checkpoint.save_checkpoint", lambda **kwargs: checkpoints.append(kwargs)
    )

    await runtime._handle_turn(_env())

    assert releases == ["released"]
    assert checkpoints[0]["status"] == "delivery_failed"
    assert checkpoints[0]["delivery"]["state"] == "delivery_failed"
    assert checkpoints[0]["response"]["content_chars"] == len("generated but not delivered")
    assert recorded[0]["final_resp"].raw["delivery_succeeded"] is False
    assert all(frame.kind != "assistant" for frame in runtime._windows["chat"].body)
    reply_event = next(
        payload
        for kind, payload, _kwargs in runtime.events.rows  # type: ignore[attr-defined]
        if kind == "reply"
    )
    assert reply_event["delivered"] is False
    assert reply_event["delivery_state"] == "delivery_failed"
    await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_failed_command_reply_blocks_post_send_effect(tmp_path, monkeypatch) -> None:
    from norax import commands as command_module

    runtime = _runtime(tmp_path)
    runtime.outbound = _FailingOutbound()  # type: ignore[assignment]
    effects: list[str] = []

    async def post_send() -> None:
        effects.append("ran")

    async def fake_handle(*_args, **_kwargs):
        return command_module.CommandResult(reply="restart queued", post_send=post_send)

    monkeypatch.setattr("norax.runtime.core.cmd_mod.handle", fake_handle)
    await runtime._handle_command(_env(), command_module.parse("/restart"))

    assert effects == []
    await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_delivery_telemetry_failure_does_not_rewrite_success_or_block_effect(
    tmp_path, monkeypatch
) -> None:
    from norax import commands as command_module

    runtime = _runtime(tmp_path)
    runtime.events = _FailingDeliveryEvents()  # type: ignore[assignment]
    effects: list[str] = []

    async def post_send() -> None:
        effects.append("ran")

    async def fake_handle(*_args, **_kwargs):
        return command_module.CommandResult(reply="restart queued", post_send=post_send)

    monkeypatch.setattr("norax.runtime.core.cmd_mod.handle", fake_handle)
    await runtime._handle_command(_env(), command_module.parse("/restart"))

    outbound = runtime.outbound
    assert isinstance(outbound, _Outbound)
    assert outbound.sent == [("test", "thread-1", "restart queued", "in-1")]
    assert effects == ["ran"]
    await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_planning_failure_returns_a_visible_terminal_reply(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)

    async def failed_plan(_env, **_kwargs):
        raise ValueError("invalid planner state")

    monkeypatch.setattr("norax.runtime.core.hot_path.plan_turn", failed_plan)

    await runtime._handle_turn(_env())

    outbound = runtime.outbound
    assert isinstance(outbound, _Outbound)
    assert len(outbound.sent) == 1
    assert "couldn't complete" in outbound.sent[0][2].lower()
    assert "valueerror" in outbound.sent[0][2].lower()
    await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_turn_pipeline_uses_verified_orchestrator_result_without_direct_replay(
    tmp_path, monkeypatch
) -> None:
    from norax.brain.orchestrator import OrchestratorRunResult

    runtime = _runtime(tmp_path)
    runtime._planner_enabled = True
    runtime.planning_mode = "orchestrator"
    ctx = SimpleNamespace(
        decision="emit_reply",
        allowed_tools=["read", "exec"],
        focus=SimpleNamespace(summary="orchestrated audit"),
        metadata={},
        memory=SimpleNamespace(items=[]),
    )
    rendered = SimpleNamespace(static_hash="static-1", system="system", user="user")
    direct_calls: list[dict] = []

    async def fake_plan_turn(_env, **_kwargs):
        return ctx, rendered

    async def direct_agent_loop(**kwargs):
        direct_calls.append(kwargs)
        raise AssertionError("direct loop must not replay a completed orchestrator result")

    async def fake_orchestrator_run(_self, **kwargs):
        assert kwargs["allowed_tools"] == ["read", "exec"]
        return OrchestratorRunResult(
            content="orchestrated completion",
            trace=[{"name": "read", "args": {}, "result": {"ok": True}}],
            rounds=2,
            complete=True,
            verified_outcome=True,
            status_reason="verified_complete",
            completion_signal="verified",
            planner_model="planner/test",
            usage={"input_tokens": 20, "output_tokens": 9},
            mutation_outcome={},
            blocking_categories=[],
        )

    monkeypatch.setattr("norax.runtime.core.hot_path.plan_turn", fake_plan_turn)
    monkeypatch.setattr("norax.runtime.core.agent_loop.run_agent_loop", direct_agent_loop)
    monkeypatch.setattr("norax.brain.orchestrator.Orchestrator.run", fake_orchestrator_run)
    monkeypatch.setattr("norax.runtime.checkpoint.load_latest_checkpoint", lambda _channel: None)
    monkeypatch.setattr("norax.runtime.checkpoint.save_checkpoint", lambda **_kwargs: None)

    await runtime._handle_turn(_env(body="plan and execute the audit"))

    assert direct_calls == []
    outbound = runtime.outbound
    assert isinstance(outbound, _Outbound)
    assert outbound.sent[0][2] == "orchestrated completion"
    telemetry = [
        payload
        for kind, payload, _kwargs in runtime.events.rows  # type: ignore[attr-defined]
        if kind == "turn_telemetry"
    ]
    assert telemetry[0]["path"] == "orchestrator"
    assert telemetry[0]["planner"] == "planner/test"
    await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_turn_pipeline_uses_multi_agent_fusion_without_direct_replay(
    tmp_path, monkeypatch
) -> None:
    from norax.brain.multi_agent import FusionResult, SubAgentResult

    runtime = _runtime(tmp_path)
    runtime._multi_agent_enabled = True
    runtime.autonomy_config = SimpleNamespace(multi_agent_auto=True)
    ctx = SimpleNamespace(
        decision="emit_reply",
        allowed_tools=["read"],
        focus=SimpleNamespace(summary="parallel audit"),
        metadata={},
        memory=SimpleNamespace(items=[]),
    )
    rendered = SimpleNamespace(static_hash="static-1", system="system", user="user")

    async def fake_plan_turn(_env, **_kwargs):
        return ctx, rendered

    async def direct_agent_loop(**_kwargs):
        raise AssertionError("direct loop must not replay a completed multi-agent result")

    async def fake_multi_agent_run(_self, **kwargs):
        assert kwargs["max_concurrent"] == runtime._multi_agent_max_concurrent
        return FusionResult(
            summary="parallel completion",
            sub_results=[
                SubAgentResult(
                    subtask_id="s1",
                    role="verify",
                    success=True,
                    output="verified",
                    tool_calls=[{"name": "read", "args": {}, "result": {"ok": True}}],
                    rounds=2,
                    usage={"input_tokens": 12, "output_tokens": 5},
                )
            ],
            total_tool_calls=1,
            total_rounds=2,
            usage={"input_tokens": 12, "output_tokens": 5},
        )

    monkeypatch.setattr("norax.runtime.core.hot_path.plan_turn", fake_plan_turn)
    monkeypatch.setattr("norax.runtime.core.agent_loop.run_agent_loop", direct_agent_loop)
    monkeypatch.setattr("norax.runtime.autonomy.should_trigger_multi_agent", lambda *_a: True)
    monkeypatch.setattr("norax.brain.multi_agent.MultiAgentOrchestrator.run", fake_multi_agent_run)
    monkeypatch.setattr("norax.runtime.checkpoint.load_latest_checkpoint", lambda _channel: None)
    monkeypatch.setattr("norax.runtime.checkpoint.save_checkpoint", lambda **_kwargs: None)

    await runtime._handle_turn(_env(body="audit these independent subsystems in parallel"))

    outbound = runtime.outbound
    assert isinstance(outbound, _Outbound)
    assert outbound.sent[0][2] == "parallel completion"
    telemetry = [
        payload
        for kind, payload, _kwargs in runtime.events.rows  # type: ignore[attr-defined]
        if kind == "turn_telemetry"
    ]
    assert telemetry[0]["path"] == "multi_agent"
    assert telemetry[0]["rounds"] == 2
    await runtime.gateway.aclose()


class _Typing:
    def __init__(self) -> None:
        self.stopped = 0

    async def stop(self, *, timeout: float) -> None:
        assert timeout == 1.5
        self.stopped += 1


class _StreamingMessage:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.finalized: list[str] = []
        self.deleted = 0

    async def append(self, text: str) -> None:
        self.deltas.append(text)

    async def finalize(self, text: str) -> dict:
        self.finalized.append(text)
        return {"ok": True, "message_id": "stream-1"}

    async def delete(self) -> None:
        self.deleted += 1


class _Discord:
    def __init__(self) -> None:
        self.typing = _Typing()
        self.message = _StreamingMessage()

    async def start_typing(self, channel: str) -> _Typing:
        assert channel == "discord-channel"
        return self.typing

    async def begin_streaming_message(self, channel: str, **kwargs) -> _StreamingMessage:
        assert channel == "discord-channel"
        assert kwargs["defer_initial"] is True
        return self.message


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_empty_response_diagnostic_never_sends_raw_provider_data(tmp_path, streamed):
    runtime = _runtime(tmp_path)
    stream = _StreamingMessage() if streamed else None
    response = GatewayResponse(
        request_id="empty",
        model="test-model",
        content="",
        tool_calls=[],
        usage={},
        raw={"private_provider_metadata": "private-value-must-not-be-delivered"},
    )
    try:
        result = await runtime._deliver_turn_response(
            env=_env(),
            ctx=SimpleNamespace(decision="emit_reply"),
            resp=response,
            trace=[],
            rounds=1,
            streaming_msg=stream,
            stream_delta_buf=[],
            target_channel="thread-1",
        )
        messages = stream.finalized if stream else [row[2] for row in runtime.outbound.sent]
        assert result["ok"] is True
        assert messages
        assert all("private-value-must-not-be-delivered" not in message for message in messages)
    finally:
        await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_generation_metrics_failure_cannot_prevent_reply_delivery(tmp_path):
    runtime = _runtime(tmp_path)

    class BrokenMetric:
        def labels(self, **kwargs):
            raise RuntimeError("metrics backend unavailable")

    runtime.metrics.gateway_requests = BrokenMetric()
    response = GatewayResponse(
        request_id="metrics",
        model="test-model",
        content="completed answer",
        tool_calls=[],
        usage={},
        raw={},
    )
    try:
        result = await runtime._deliver_turn_response(
            env=_env(),
            ctx=SimpleNamespace(decision="emit_reply"),
            resp=response,
            trace=[],
            rounds=1,
            streaming_msg=None,
            stream_delta_buf=[],
            target_channel="thread-1",
        )
        assert result["ok"] is True
        assert runtime.outbound.sent[0][2] == "completed answer"
    finally:
        await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_turn_pipeline_streams_and_finalizes_exactly_once(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    runtime.discord = _Discord()  # type: ignore[assignment]
    runtime.stream_replies = True
    ctx = SimpleNamespace(
        decision="emit_reply",
        allowed_tools=[],
        focus=SimpleNamespace(summary=""),
        metadata={},
        memory=SimpleNamespace(items=[]),
    )
    rendered = SimpleNamespace(static_hash="static-1", system="system", user="user")

    async def fake_plan_turn(_env, **_kwargs):
        return ctx, rendered

    async def fake_agent_loop(**kwargs):
        await kwargs["on_delta"]("streamed ")
        await kwargs["on_delta"]("answer")
        return (
            GatewayResponse(
                request_id="req-stream",
                model="test-model",
                content="streamed answer",
                tool_calls=[],
                usage={},
                raw={},
            ),
            [],
            1,
            None,
        )

    monkeypatch.setattr("norax.runtime.core.hot_path.plan_turn", fake_plan_turn)
    monkeypatch.setattr("norax.runtime.core.agent_loop.run_agent_loop", fake_agent_loop)
    monkeypatch.setattr("norax.runtime.checkpoint.load_latest_checkpoint", lambda _channel: None)
    monkeypatch.setattr("norax.runtime.checkpoint.save_checkpoint", lambda **_kwargs: None)

    env = _env()
    env.source = "discord"
    env.raw = {"channel_id": "discord-channel"}
    await runtime._handle_turn(env)

    discord = runtime.discord
    assert isinstance(discord, _Discord)
    assert discord.message.deltas == ["streamed ", "answer"]
    assert discord.message.finalized == ["streamed answer"]
    assert discord.message.deleted == 0
    assert discord.typing.stopped >= 1
    await runtime.gateway.aclose()


@pytest.mark.asyncio
async def test_turn_pipeline_converts_agent_failure_into_visible_reply(
    tmp_path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    ctx = SimpleNamespace(
        decision="emit_reply",
        allowed_tools=[],
        focus=SimpleNamespace(summary=""),
        metadata={},
        memory=SimpleNamespace(items=[]),
    )
    rendered = SimpleNamespace(static_hash="static-1", system="system", user="user")

    async def fake_plan_turn(_env, **_kwargs):
        return ctx, rendered

    async def failed_agent_loop(**_kwargs):
        raise TimeoutError("model request timed out")

    monkeypatch.setattr("norax.runtime.core.hot_path.plan_turn", fake_plan_turn)
    monkeypatch.setattr("norax.runtime.core.agent_loop.run_agent_loop", failed_agent_loop)
    monkeypatch.setattr("norax.runtime.checkpoint.load_latest_checkpoint", lambda _channel: None)

    await runtime._handle_turn(_env())

    outbound = runtime.outbound
    assert isinstance(outbound, _Outbound)
    # Non-Discord transports receive failures through their normal outbound
    # route, instead of silently dropping them.
    assert len(outbound.sent) == 1
    assert "couldn't complete" in outbound.sent[0][2].lower()
    assert "timeouterror" in outbound.sent[0][2].lower()
    event_kinds = [kind for kind, _payload, _kwargs in runtime.events.rows]  # type: ignore[attr-defined]
    assert "turn_failure" in event_kinds
    assert "error" in event_kinds
    await runtime.gateway.aclose()
