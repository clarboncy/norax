"""Regression tests for runtime ownership, cancellation, and shutdown ordering."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
from types import MethodType, SimpleNamespace
from typing import Any, Literal, cast

import pytest

from norax import commands as cmd_mod
from norax.brain import agent_loop
from norax.config.loader import Config
from norax.config.provider_store import ProviderStore
from norax.envelope import Principal, SensoryInput, ThreadBinding
from norax.observability.metrics import Metrics
from norax.runtime.core import Runtime
from norax.runtime.ingress_bus import IngressBus
from norax.runtime.outbound import OutboundRegistry


class _Events:
    def __init__(self, order: list[str] | None = None) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []
        self.order = order

    @contextmanager
    def trace_scope(self):
        yield "lifecycle-trace"

    async def append(self, kind: str, payload: dict[str, Any], **_kwargs: Any) -> None:
        self.rows.append((kind, payload))
        if self.order is not None:
            self.order.append(f"event:{kind}")


class _Gateway:
    base_url = "http://gateway.invalid/v1"
    provider_urls: dict[str, str] = {}

    def __init__(self, order: list[str] | None = None) -> None:
        self.close_count = 0
        self.order = order

    async def aclose(self) -> None:
        self.close_count += 1
        if self.order is not None:
            self.order.append("gateway.close")


class _IdleIngress:
    def __init__(self, order: list[str] | None = None) -> None:
        self._adapters: list[Any] = []
        self.order = order
        self.stop_count = 0
        self.release_turn: asyncio.Event | None = None

    async def start(self) -> None:
        if self.order is not None:
            self.order.append("ingress.start")

    async def stop(self, grace: float = 10.0) -> None:
        del grace
        self.stop_count += 1
        if self.order is not None:
            self.order.append("ingress.stop")
        if self.release_turn is not None:
            self.release_turn.set()

    async def stream(self):
        if False:
            yield None


def _runtime(
    *,
    ingress: Any | None = None,
    order: list[str] | None = None,
    max_tool_rounds: int | None = None,
) -> Runtime:
    cfg = SimpleNamespace(
        raw={},
        connectors={},
        shutdown_grace_seconds=0.25,
    )
    if max_tool_rounds is not None:
        cfg.max_tool_rounds = max_tool_rounds
    return Runtime(
        ingress=cast(IngressBus, ingress or _IdleIngress(order)),
        events=_Events(order),  # type: ignore[arg-type]
        cfg=cfg,  # type: ignore[arg-type]
        gateway=_Gateway(order),  # type: ignore[arg-type]
        outbound=OutboundRegistry(),
        metrics=Metrics(),
    )


def _env(
    body: str,
    *,
    tier: Literal["owner", "admin", "user", "guest"] = "owner",
    channel_id: str = "channel-1",
) -> SensoryInput:
    return SensoryInput(
        channel="chat",
        source="test",
        message_id=f"message-{body}",
        timestamp=datetime.now(UTC),
        sender=Principal(id=f"{tier}-id", label=tier, trust=tier == "owner", tier=tier),
        body=body,
        raw={"channel_id": channel_id},
        trusted=tier == "owner",
        thread_binding=ThreadBinding(
            channel="test",
            thread_id=channel_id,
            kind="custom",
            session_id=channel_id,
        ),
        metadata={},
    )


def test_runtime_setting_mutators_reject_invalid_direct_calls() -> None:
    runtime = _runtime()

    with pytest.raises(ValueError, match="thinking effort"):
        runtime.set_thinking_effort("infinite")
    with pytest.raises(ValueError, match="planning mode"):
        runtime.set_planning_mode("magic")
    with pytest.raises(ValueError, match="max_tool_rounds"):
        runtime.set_max_tool_rounds(100_000)
    with pytest.raises(TypeError, match="reasoning_output"):
        runtime.set_reasoning_output("false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="memory depth"):
        runtime.set_memory_depth("everything")
    with pytest.raises(ValueError, match="weak-model boost"):
        runtime.set_weak_model_boost("maybe")
    with pytest.raises(TypeError, match="stream_replies"):
        runtime.set_stream_replies("false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="response length"):
        runtime.set_response_length("novel")
    with pytest.raises(ValueError, match="tool activity"):
        runtime.set_tool_activity("constant")
    with pytest.raises(ValueError, match="model"):
        runtime.set_default_model("bad model")


def test_runtime_reports_the_effective_hard_capped_round_limit() -> None:
    runtime = _runtime(max_tool_rounds=agent_loop.HARD_ROUND_CAP + 50)

    assert runtime.max_tool_rounds == agent_loop.HARD_ROUND_CAP


@pytest.mark.asyncio
async def test_runtime_build_environment_model_overrides_persisted_selection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        ProviderStore,
        "runtime_settings",
        lambda _self: {"default_model": "persisted/model"},
    )
    monkeypatch.setenv("NORAX_DEFAULT_MODEL", "environment/model")
    cfg = Config(
        raw={
            "http": {"bind": "127.0.0.1:0"},
            "gateway": {"base_url": "http://127.0.0.1:1/v1"},
        },
        project_root=tmp_path,
    )
    runtime = Runtime.build(cfg)
    try:
        assert runtime.default_model == "environment/model"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_turn_boundary_cleans_cancelled_ui_helpers_and_stop_state() -> None:
    runtime = _runtime()
    started = asyncio.Event()
    typing = SimpleNamespace(stop_count=0)
    streaming = SimpleNamespace(delete_count=0)

    async def stop(*, timeout: float) -> None:
        assert timeout == 1.5
        typing.stop_count += 1

    async def delete() -> None:
        streaming.delete_count += 1

    typing.stop = stop
    streaming.delete = delete

    async def blocked_inner(self: Runtime, env: SensoryInput) -> None:
        del env
        owner_task = asyncio.current_task()
        assert owner_task is not None
        self._active_typing_handles[owner_task] = typing
        self._active_stream_messages[owner_task] = streaming
        started.set()
        await asyncio.Event().wait()

    runtime._handle_turn_inner = MethodType(blocked_inner, runtime)  # type: ignore[method-assign]
    runtime._stop_channels.add("channel-1")
    task = asyncio.create_task(runtime._handle_turn(_env("work")))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert typing.stop_count == 1
    assert streaming.delete_count == 1
    assert "channel-1" not in runtime._stop_channels
    assert not runtime._active_typing_handles
    assert not runtime._active_stream_messages


@pytest.mark.asyncio
async def test_completed_turn_tasks_are_released_and_cancel_reports_real_work() -> None:
    runtime = _runtime()

    completed = asyncio.create_task(asyncio.sleep(0))
    runtime._track_turn_task("done", completed)
    await completed
    await asyncio.sleep(0)
    assert "done" not in runtime._active_turn_tasks

    waiting = asyncio.create_task(asyncio.Event().wait())
    runtime._track_turn_task("active", waiting)
    assert runtime._cancel_channel("active") is True
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await asyncio.sleep(0)
    assert "active" not in runtime._active_turn_tasks
    assert runtime._cancel_channel("active") is False

    runtime._windows["inactive-window"] = SimpleNamespace()  # type: ignore[assignment]
    assert runtime._cancel_channel() is False
    assert "inactive-window" not in runtime._stop_channels


class _SequenceIngress(_IdleIngress):
    def __init__(self, messages: list[SensoryInput]) -> None:
        super().__init__()
        self.messages = messages
        self.finish = asyncio.Event()

    async def stop(self, grace: float = 10.0) -> None:
        await super().stop(grace)
        self.finish.set()

    async def stream(self):
        for message in self.messages:
            yield message
            await asyncio.sleep(0)
        await self.finish.wait()


class _CapturingOutbound:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.sent = asyncio.Event()

    async def send(
        self,
        target: str,
        text: str,
        *,
        reply_to: str | None = None,
        react_to: str | None = None,
        emoji: str | None = None,
        files: list[str] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        del target, reply_to, react_to, emoji, files, extra
        self.messages.append(text)
        self.sent.set()
        return {"ok": True, "message_id": "reply-1"}


@pytest.mark.asyncio
async def test_untrusted_stop_cannot_preempt_an_active_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal = _env("long work")
    untrusted_stop = _env("/stop", tier="guest")
    ingress = _SequenceIngress([normal, untrusted_stop])
    runtime = _runtime(ingress=ingress)
    outbound = _CapturingOutbound()
    runtime.outbound.register("test", outbound)
    normal_started = asyncio.Event()
    normal_release = asyncio.Event()
    normal_cancelled = asyncio.Event()
    normal_finished = asyncio.Event()

    async def controlled_inner(self: Runtime, env: SensoryInput) -> None:
        parsed = cmd_mod.parse(env.body)
        if parsed is not None and cmd_mod.is_known(parsed):
            await self._handle_command(env, parsed)
            return
        normal_started.set()
        try:
            await normal_release.wait()
            normal_finished.set()
        except asyncio.CancelledError:
            normal_cancelled.set()
            raise

    async def dormant() -> None:
        await asyncio.Event().wait()

    runtime._handle_turn_inner = MethodType(controlled_inner, runtime)  # type: ignore[method-assign]
    runtime._capability_probe_loop = dormant  # type: ignore[method-assign]
    runtime._idle_sleep_loop = dormant  # type: ignore[method-assign]
    runtime._warmup_caches = dormant  # type: ignore[method-assign]
    monkeypatch.setenv("NORAX_COMPLETION_PROBE_ENABLED", "false")
    monkeypatch.setenv("NORAX_AUTONOMY_ENABLED", "false")

    runner = asyncio.create_task(runtime.run())
    await asyncio.wait_for(normal_started.wait(), timeout=1)
    await asyncio.wait_for(outbound.sent.wait(), timeout=1)
    assert "owner" in outbound.messages[-1].lower()
    assert not normal_cancelled.is_set()
    assert not normal_finished.is_set()

    normal_release.set()
    await asyncio.wait_for(normal_finished.wait(), timeout=1)
    await runtime.shutdown()
    await asyncio.wait_for(runner, timeout=1)


@pytest.mark.asyncio
async def test_owner_stop_cleans_exact_turn_and_next_turn_is_not_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingress = _SequenceIngress([_env("long work"), _env("/stop"), _env("next work")])
    runtime = _runtime(ingress=ingress)
    outbound = _CapturingOutbound()
    runtime.outbound.register("test", outbound)
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    next_finished = asyncio.Event()
    streaming = SimpleNamespace(delete_count=0)

    async def delete() -> None:
        streaming.delete_count += 1

    streaming.delete = delete

    async def controlled_inner(self: Runtime, env: SensoryInput) -> None:
        parsed = cmd_mod.parse(env.body)
        if parsed is not None and cmd_mod.is_known(parsed):
            await self._handle_command(env, parsed)
            return
        if env.body == "next work":
            assert "channel-1" not in self._stop_channels
            next_finished.set()
            return
        owner_task = asyncio.current_task()
        assert owner_task is not None
        self._active_stream_messages[owner_task] = streaming
        first_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def dormant() -> None:
        await asyncio.Event().wait()

    runtime._handle_turn_inner = MethodType(controlled_inner, runtime)  # type: ignore[method-assign]
    runtime._capability_probe_loop = dormant  # type: ignore[method-assign]
    runtime._idle_sleep_loop = dormant  # type: ignore[method-assign]
    runtime._warmup_caches = dormant  # type: ignore[method-assign]
    monkeypatch.setenv("NORAX_COMPLETION_PROBE_ENABLED", "false")
    monkeypatch.setenv("NORAX_AUTONOMY_ENABLED", "false")

    runner = asyncio.create_task(runtime.run())
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await asyncio.wait_for(first_cancelled.wait(), timeout=1)
    await asyncio.wait_for(next_finished.wait(), timeout=1)
    assert any(message == "Stopping." for message in outbound.messages)
    assert streaming.delete_count == 1
    assert "channel-1" not in runtime._stop_channels

    await runtime.shutdown()
    await asyncio.wait_for(runner, timeout=1)


@pytest.mark.asyncio
async def test_same_channel_backlog_does_not_block_other_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingress = _SequenceIngress(
        [
            _env("a-first", channel_id="a"),
            _env("a-second", channel_id="a"),
            _env("b-first", channel_id="b"),
        ]
    )
    runtime = _runtime(ingress=ingress)
    a_first_started = asyncio.Event()
    release_a = asyncio.Event()
    b_finished = asyncio.Event()
    a_second_finished = asyncio.Event()

    async def controlled_inner(_self: Runtime, env: SensoryInput) -> None:
        if env.body == "a-first":
            a_first_started.set()
            await release_a.wait()
        elif env.body == "a-second":
            a_second_finished.set()
        elif env.body == "b-first":
            b_finished.set()

    async def dormant() -> None:
        await asyncio.Event().wait()

    runtime._handle_turn_inner = MethodType(controlled_inner, runtime)  # type: ignore[method-assign]
    runtime._capability_probe_loop = dormant  # type: ignore[method-assign]
    runtime._idle_sleep_loop = dormant  # type: ignore[method-assign]
    runtime._warmup_caches = dormant  # type: ignore[method-assign]
    monkeypatch.setenv("NORAX_COMPLETION_PROBE_ENABLED", "false")
    monkeypatch.setenv("NORAX_AUTONOMY_ENABLED", "false")

    runner = asyncio.create_task(runtime.run())
    await asyncio.wait_for(a_first_started.wait(), timeout=1)
    await asyncio.wait_for(b_finished.wait(), timeout=1)
    assert not a_second_finished.is_set()

    release_a.set()
    await asyncio.wait_for(a_second_finished.wait(), timeout=1)
    await runtime.shutdown()
    await asyncio.wait_for(runner, timeout=1)


@pytest.mark.asyncio
async def test_turn_queue_is_bounded_and_cancellation_releases_capacity() -> None:
    runtime = _runtime()
    runtime._max_pending_turns = 1
    runtime._max_pending_per_channel = 1
    runtime._turn_semaphore = asyncio.Semaphore(1)
    started = asyncio.Event()

    async def blocked_inner(_self: Runtime, _env: SensoryInput) -> None:
        started.set()
        await asyncio.Event().wait()

    runtime._handle_turn_inner = MethodType(blocked_inner, runtime)  # type: ignore[method-assign]
    assert runtime._queue_turn("one", _env("first", channel_id="one")) is True
    await asyncio.wait_for(started.wait(), timeout=1)
    assert runtime._queue_turn("two", _env("second", channel_id="two")) is False

    assert runtime._cancel_channel("one") is True
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert runtime._turn_work_count == 0
    assert runtime._queue_turn("two", _env("retry", channel_id="two")) is True
    assert runtime._cancel_channel("two") is True
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_delivered_turn_releases_global_slot_before_post_turn_work() -> None:
    runtime = _runtime()
    runtime._turn_semaphore = asyncio.Semaphore(1)
    first_in_post_turn = asyncio.Event()
    release_first = asyncio.Event()
    second_finished = asyncio.Event()

    async def controlled_inner(_self: Runtime, env: SensoryInput) -> None:
        if env.body == "first":
            assert runtime._release_current_turn_slot() is True
            first_in_post_turn.set()
            await release_first.wait()
        else:
            second_finished.set()

    runtime._handle_turn_inner = MethodType(controlled_inner, runtime)  # type: ignore[method-assign]
    assert runtime._queue_turn("one", _env("first", channel_id="one")) is True
    await asyncio.wait_for(first_in_post_turn.wait(), timeout=1)
    assert runtime._queue_turn("two", _env("second", channel_id="two")) is True

    await asyncio.wait_for(second_finished.wait(), timeout=1)
    assert not release_first.is_set()

    release_first.set()
    await asyncio.gather(*list(runtime._active_turn_tasks.values()))
    assert runtime._turn_work_count == 0


@pytest.mark.asyncio
async def test_shutdown_quiesces_writers_before_flush_and_is_idempotent(tmp_path) -> None:
    order: list[str] = []
    ingress = _IdleIngress(order)
    runtime = _runtime(ingress=ingress, order=order)
    release_turn = asyncio.Event()
    ingress.release_turn = release_turn

    async def active_turn() -> None:
        await release_turn.wait()
        order.append("turn.done")

    async def background() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            order.append("background.done")

    turn_task = asyncio.create_task(active_turn())
    runtime._track_turn_task("channel-1", turn_task)
    runtime._sleep_task = asyncio.create_task(background())
    await asyncio.sleep(0)

    class _Saver:
        def save(self) -> None:
            order.append("graph.save")

    class _UserModel:
        def flush(self) -> int:
            order.append("user.flush")
            return 1

    class _Hebbian:
        def flush(self, root) -> int:
            assert root == tmp_path
            order.append("hebbian.flush")
            return 1

    class _Reminders:
        async def stop(self) -> None:
            order.append("reminders.stop")

    runtime._causal_graph = _Saver()  # type: ignore[assignment]
    runtime._user_model = _UserModel()  # type: ignore[assignment]
    runtime._memory_store = SimpleNamespace(root=tmp_path)  # type: ignore[assignment]
    runtime._hebbian = _Hebbian()  # type: ignore[assignment]
    runtime.reminders = _Reminders()

    await runtime.shutdown()
    await runtime.shutdown()

    assert order.index("ingress.stop") < order.index("turn.done") < order.index("graph.save")
    assert order.index("background.done") < order.index("graph.save")
    assert order.index("reminders.stop") < order.index("graph.save")
    assert ingress.stop_count == 1
    gateway = cast(Any, runtime.gateway)
    events = cast(Any, runtime.events)
    assert gateway.close_count == 1
    assert [kind for kind, _ in events.rows].count("runtime.stop") == 1
    assert runtime._shutdown_complete is True


@pytest.mark.asyncio
async def test_uvicorn_readiness_waits_for_bound_listener_and_surfaces_failure() -> None:
    server = SimpleNamespace(started=False)

    async def start() -> None:
        await asyncio.sleep(0.01)
        server.started = True

    task = asyncio.create_task(start())
    await Runtime._await_uvicorn_started(server, task, timeout=0.2)
    await task

    async def fail() -> None:
        raise OSError("bind failed")

    failed = asyncio.create_task(fail())
    with pytest.raises(RuntimeError, match="failed during startup") as caught:
        await Runtime._await_uvicorn_started(SimpleNamespace(started=False), failed, timeout=0.2)
    assert isinstance(caught.value.__cause__, OSError)


def _quiet_runtime_background(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def dormant() -> None:
        await asyncio.Event().wait()

    runtime._capability_probe_loop = dormant  # type: ignore[method-assign]
    runtime._idle_sleep_loop = dormant  # type: ignore[method-assign]
    runtime._warmup_caches = dormant  # type: ignore[method-assign]
    monkeypatch.setenv("NORAX_COMPLETION_PROBE_ENABLED", "false")
    monkeypatch.setenv("NORAX_AUTONOMY_ENABLED", "false")


@pytest.mark.asyncio
async def test_a2a_connector_is_ready_only_after_listener_starts_and_closes_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    class _Config:
        def __init__(self, app: Any, **kwargs: Any) -> None:
            self.app = app
            self.kwargs = kwargs

    class _Server:
        instances: list[_Server] = []

        def __init__(self, config: _Config) -> None:
            self.config = config
            self.started = False
            self._should_exit = False
            self.exit = asyncio.Event()
            self.instances.append(self)

        @property
        def should_exit(self) -> bool:
            return self._should_exit

        @should_exit.setter
        def should_exit(self, value: bool) -> None:
            self._should_exit = value
            if value:
                self.exit.set()

        async def serve(self) -> None:
            await asyncio.sleep(0.01)
            self.started = True
            await self.exit.wait()

    monkeypatch.setattr(uvicorn, "Config", _Config)
    monkeypatch.setattr(uvicorn, "Server", _Server)
    ingress = _SequenceIngress([])
    runtime = _runtime(ingress=ingress)
    runtime._connectors["a2a_server"] = {
        "enabled": True,
        "required": True,
        "host": "127.0.0.1",
        "port": 8766,
    }
    _quiet_runtime_background(runtime, monkeypatch)

    runner = asyncio.create_task(runtime.run())
    for _ in range(100):
        if runtime.capabilities.is_ready("a2a_server"):
            break
        await asyncio.sleep(0.005)
    assert runtime.capabilities.is_ready("a2a_server")
    assert _Server.instances[0].started is True
    assert runtime.outbound.has("a2a")

    await runtime.shutdown()
    await asyncio.wait_for(runner, timeout=1)
    assert _Server.instances[0].should_exit is True
    assert runtime.a2a_server is None
    assert not runtime.outbound.has("a2a")


@pytest.mark.asyncio
async def test_failed_a2a_listener_never_reports_ready_or_leaves_outbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    class _Config:
        def __init__(self, app: Any, **_kwargs: Any) -> None:
            self.app = app

    class _Server:
        def __init__(self, _config: _Config) -> None:
            self.started = False
            self.should_exit = False

        async def serve(self) -> None:
            raise OSError("address already in use")

    monkeypatch.setattr(uvicorn, "Config", _Config)
    monkeypatch.setattr(uvicorn, "Server", _Server)
    ingress = _SequenceIngress([])
    runtime = _runtime(ingress=ingress)
    runtime._connectors["a2a_server"] = {
        "enabled": True,
        "required": False,
        "host": "127.0.0.1",
        "port": 8766,
    }
    _quiet_runtime_background(runtime, monkeypatch)

    runner = asyncio.create_task(runtime.run())
    for _ in range(100):
        state = runtime.capabilities.status().get("a2a_server", {}).get("state")
        if state == "failed":
            break
        await asyncio.sleep(0.005)
    assert runtime.capabilities.status()["a2a_server"]["state"] == "failed"
    assert not runtime.outbound.has("a2a")
    assert runtime.a2a_server is None

    await runtime.shutdown()
    await asyncio.wait_for(runner, timeout=1)


@pytest.mark.asyncio
async def test_partial_mcp_startup_disconnects_every_open_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from norax.mcp import client as mcp_module

    class _MCPClient:
        instances: list[_MCPClient] = []

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.disconnect_count = 0
            self.health_handler: Any = None
            self.instances.append(self)

        def set_health_handler(self, handler: Any) -> None:
            self.health_handler = handler

        async def connect_stdio(self, name: str, _command: str, *_args: str) -> None:
            self.calls.append(name)
            if name == "broken":
                raise RuntimeError("handshake failed")

        async def disconnect_all(self) -> None:
            self.disconnect_count += 1

    monkeypatch.setattr(mcp_module, "NoraxMCPClient", _MCPClient)
    ingress = _SequenceIngress([])
    runtime = _runtime(ingress=ingress)
    runtime._connectors["mcp_client"] = {
        "enabled": True,
        "required": False,
        "servers": [
            {"name": "working", "transport": "stdio", "command": "first"},
            {"name": "broken", "transport": "stdio", "command": "second"},
        ],
    }
    _quiet_runtime_background(runtime, monkeypatch)

    runner = asyncio.create_task(runtime.run())
    for _ in range(100):
        state = runtime.capabilities.status().get("mcp_client", {}).get("state")
        if state == "failed":
            break
        await asyncio.sleep(0.005)
    client = _MCPClient.instances[0]
    assert set(client.calls) == {"working", "broken"}
    assert client.disconnect_count == 1
    assert runtime.mcp_client is None
    assert runtime.capabilities.status()["mcp_client"]["state"] == "failed"

    await runtime.shutdown()
    await asyncio.wait_for(runner, timeout=1)
    assert client.disconnect_count == 1


@pytest.mark.asyncio
async def test_commerce_routes_mount_before_ingress_accepts_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from norax.commerce import api_endpoints

    order: list[str] = []
    ingress = _SequenceIngress([])
    ingress._adapters = [SimpleNamespace(app=object())]

    async def start() -> None:
        order.append("ingress.start")

    ingress.start = start  # type: ignore[method-assign]

    def mount(app: Any) -> None:
        assert app is ingress._adapters[0].app
        order.append("commerce.mount")

    monkeypatch.setattr(api_endpoints, "mount_commerce", mount)
    runtime = _runtime(ingress=ingress)
    runtime._connectors["commerce"] = {"enabled": True, "required": True}
    _quiet_runtime_background(runtime, monkeypatch)

    runner = asyncio.create_task(runtime.run())
    for _ in range(100):
        if "ingress.start" in order:
            break
        await asyncio.sleep(0.005)
    assert order[:2] == ["commerce.mount", "ingress.start"]
    assert runtime.capabilities.is_ready("commerce_api")

    await runtime.shutdown()
    await asyncio.wait_for(runner, timeout=1)


class _Adapter:
    def __init__(self, name: str, *, fail_start: bool = False) -> None:
        self.name = name
        self.fail_start = fail_start
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1
        if self.fail_start:
            raise RuntimeError("start failed")

    async def stop(self) -> None:
        self.stopped += 1

    async def events(self):
        await asyncio.Event().wait()
        if False:
            yield None


@pytest.mark.asyncio
async def test_ingress_stop_unblocks_stream_and_partial_start_rolls_back() -> None:
    empty = IngressBus([])
    await empty.start()

    async def next_message() -> SensoryInput:
        return await anext(empty.stream())

    waiting = asyncio.create_task(next_message())
    await asyncio.sleep(0)
    await empty.stop()
    with pytest.raises(StopAsyncIteration):
        await waiting

    first = _Adapter("first")
    second = _Adapter("second", fail_start=True)
    bus = IngressBus([first, second])
    with pytest.raises(RuntimeError, match="start failed"):
        await bus.start()
    assert first.started == 1
    assert first.stopped == 1
    assert second.started == 1
    assert second.stopped == 0
    assert not bus._tasks


@pytest.mark.asyncio
async def test_ingress_adapter_failure_reaches_runtime_consumer() -> None:
    class _FailingAdapter(_Adapter):
        async def events(self):
            raise ConnectionError("adapter broke")
            if False:
                yield None

    bus = IngressBus([_FailingAdapter("broken")])
    await bus.start()
    with pytest.raises(RuntimeError, match="ingress adapter failed: broken") as caught:
        await anext(bus.stream())
    assert isinstance(caught.value.__cause__, ConnectionError)
    await bus.stop()


@pytest.mark.asyncio
async def test_optional_ingress_failure_does_not_take_down_healthy_adapters() -> None:
    event = _env("still serving")

    class _OptionalFailingAdapter(_Adapter):
        failure_is_fatal = False

        async def events(self):
            raise ConnectionError("optional adapter broke")
            if False:
                yield None

    class _HealthyAdapter(_Adapter):
        async def events(self):
            yield event
            await asyncio.Event().wait()

    bus = IngressBus([_OptionalFailingAdapter("optional"), _HealthyAdapter("healthy")])
    await bus.start()

    assert await asyncio.wait_for(anext(bus.stream()), timeout=1) is event
    await bus.stop()


@pytest.mark.asyncio
async def test_ingress_bus_backpressures_without_dropping_accepted_events() -> None:
    first = _env("first")
    second = _env("second")

    class _BurstAdapter(_Adapter):
        async def events(self):
            yield first
            yield second
            await asyncio.Event().wait()

    bus = IngressBus([_BurstAdapter("burst")], maxsize=1)
    await bus.start()
    await asyncio.sleep(0)

    assert await asyncio.wait_for(anext(bus.stream()), timeout=1) is first
    assert await asyncio.wait_for(anext(bus.stream()), timeout=1) is second
    await bus.stop()


@pytest.mark.asyncio
async def test_ingress_adapters_share_one_concurrent_shutdown_budget() -> None:
    release = asyncio.Event()

    class _BlockingStopAdapter(_Adapter):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.stop_started = asyncio.Event()

        async def stop(self) -> None:
            self.stop_started.set()
            await release.wait()
            await super().stop()

    adapters = [_BlockingStopAdapter(f"adapter-{index}") for index in range(3)]
    bus = IngressBus(adapters)
    await bus.start()

    stopping = asyncio.create_task(bus.stop(grace=1.0))
    await asyncio.wait_for(
        asyncio.gather(*(adapter.stop_started.wait() for adapter in adapters)),
        timeout=0.25,
    )
    release.set()
    await asyncio.wait_for(stopping, timeout=0.25)
    assert [adapter.stopped for adapter in adapters] == [1, 1, 1]


@pytest.mark.asyncio
async def test_cancelled_optional_supervisor_is_marked_failed() -> None:
    runtime = _runtime()
    runtime.capabilities.register("optional")
    runtime.capabilities.mark_ok("optional")
    task = asyncio.create_task(asyncio.sleep(60))
    runtime._track_optional_task("optional", task)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    status = runtime.capabilities.status()["optional"]
    assert status["state"] == "failed"
    assert status["last_error"] == "RuntimeError: service cancelled unexpectedly"


@pytest.mark.asyncio
async def test_required_service_clean_exit_immediately_blocks_readiness() -> None:
    runtime = _runtime()
    runtime.capabilities.register("required_loop", required_for_readiness=True)
    runtime.capabilities.mark_ok("required_loop")

    async def exits() -> None:
        return None

    task = asyncio.create_task(exits())
    runtime._observe_service_task("required_loop", task)
    await task
    await asyncio.sleep(0)

    assert runtime.capabilities.status()["required_loop"]["state"] == "failed"
    assert runtime.capabilities.readiness_blockers() == ["required_loop"]


def test_mcp_capability_tracks_partial_failure_total_failure_and_recovery() -> None:
    runtime = _runtime()
    runtime.capabilities.register("mcp_client", required_for_readiness=True)

    runtime._record_mcp_health(
        {
            "active": ["filesystem", "browser"],
            "healthy": ["filesystem"],
            "failures": {"browser": "ConnectionError: offline"},
        }
    )
    partial = runtime.capabilities.status()["mcp_client"]
    assert partial["state"] == "degraded"
    assert partial["required_for_readiness"] is True
    assert partial["metadata"]["healthy"] == ["filesystem"]

    runtime._record_mcp_health(
        {
            "active": ["filesystem", "browser"],
            "healthy": [],
            "failures": {
                "filesystem": "TimeoutError: late",
                "browser": "ConnectionError: offline",
            },
        }
    )
    assert runtime.capabilities.status()["mcp_client"]["state"] == "failed"
    assert runtime.capabilities.readiness_blockers() == ["mcp_client"]

    runtime._record_mcp_health(
        {"active": ["filesystem"], "healthy": ["filesystem"], "failures": {}}
    )
    recovered = runtime.capabilities.status()["mcp_client"]
    assert recovered["state"] == "ready"
    assert recovered["metadata"]["failed"] == []


def test_outbound_unregister_is_identity_safe() -> None:
    registry = OutboundRegistry()
    first = _CapturingOutbound()
    replacement = _CapturingOutbound()
    registry.register("a2a", first)
    registry.register("a2a", replacement)

    assert registry.unregister("a2a", expected=first) is False
    assert registry.get("a2a") is replacement
    assert registry.unregister("a2a", expected=replacement) is True
    assert registry.unregister("a2a") is False
