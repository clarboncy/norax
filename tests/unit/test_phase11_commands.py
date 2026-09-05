"""Phase 11 — built-in slash commands."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from norax import commands as cmd_mod
from norax.brain import agent_loop
from norax.envelope import Principal, SensoryInput

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_basic():
    pc = cmd_mod.parse("/status")
    assert pc and pc.name == "status" and pc.args == []


def test_parse_with_args():
    pc = cmd_mod.parse("/model claude-opus-4.7")
    assert pc and pc.name == "model" and pc.args == ["claude-opus-4.7"]


def test_parse_case_insensitive():
    pc = cmd_mod.parse("/STATUS")
    assert pc and pc.name == "status"


def test_parse_non_command_returns_none():
    assert cmd_mod.parse("hello") is None
    assert cmd_mod.parse("// comment") is None  # double slash, not a cmd
    assert cmd_mod.parse("") is None
    assert cmd_mod.parse("/") is None


def test_parse_strips_leading_whitespace():
    pc = cmd_mod.parse("   /new   ")
    assert pc and pc.name == "new"


def test_is_known_only_for_whitelisted_names():
    assert cmd_mod.is_known(cmd_mod.parse("/status"))
    assert cmd_mod.is_known(cmd_mod.parse("/model"))
    assert cmd_mod.is_known(cmd_mod.parse("/restart"))
    assert cmd_mod.is_known(cmd_mod.parse("/think high"))
    assert cmd_mod.is_known(cmd_mod.parse("/reasoning on"))
    assert not cmd_mod.is_known(cmd_mod.parse("/yolo"))


# ---------------------------------------------------------------------------
# Handler fixtures
# ---------------------------------------------------------------------------


def _env(body: str, *, tier: str = "owner", sid: str = "owner-123"):
    return SensoryInput(
        channel="chat",
        source="discord",
        message_id="m1",
        timestamp=datetime.now(UTC),
        sender=Principal(id=sid, label="Tester", trust=True, tier=tier),
        body=body,
        raw={"channel_id": "c1"},
        trusted=True,
    )


class _FakeCounter:
    """Minimal prometheus counter stand-in for _sum_counter()."""

    def __init__(self, total: float = 0.0):
        self._total = total

    def collect(self):
        sample = SimpleNamespace(name="norax_fake_total", labels={}, value=self._total)
        metric = SimpleNamespace(samples=[sample])
        return [metric]


def _metrics():
    return SimpleNamespace(
        ingress_total=_FakeCounter(3),
        brain_turns=_FakeCounter(2),
        brain_errors=_FakeCounter(0),
        gateway_tokens_in=_FakeCounter(1234),
        gateway_tokens_out=_FakeCounter(56),
        gateway_requests=_FakeCounter(2),
    )


class _FakeEventLog:
    def __init__(self):
        self.events = []

    async def append(self, kind, payload, attrs=None):
        self.events.append((kind, payload))


def _handle(default_model="claude-sonnet-4.6", cleared=False):
    calls = {"set_model": [], "reset_window": []}

    def _set(m: str):
        calls["set_model"].append(m)

    def _reset(ch: str) -> bool:
        calls["reset_window"].append(ch)
        return cleared

    h = cmd_mod.RuntimeHandle(
        default_model=default_model,
        set_default_model=_set,
        started_at=0.0,
        wall_started_at=0.0,
        metrics=_metrics(),
        gateway_base_url="http://127.0.0.1:8899/v1",
        event_log=_FakeEventLog(),
        reset_window_for=_reset,
        outbound=None,
    )
    return h, calls


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_owner():
    h, _ = _handle()
    pc = cmd_mod.parse("/status")
    r = await cmd_mod.handle(pc, _env("/status"), h)
    assert "NORAX//STATUS" in r.reply
    assert "<@owner-123>" not in r.reply
    assert r.reply.count("```") == 2
    assert "claude-sonnet-4.6" in r.reply
    assert r.post_send is None


# ---------------------------------------------------------------------------
# /models, /model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_models_lists_default_marker(monkeypatch):
    async def _no_live_models(force_refresh=False):
        return []

    monkeypatch.setattr(cmd_mod, "fetch_ollama_models", _no_live_models)
    h, _ = _handle(default_model="kimi-k2.7-code:cloud")
    r = await cmd_mod.handle(cmd_mod.parse("/models"), _env("/models"), h)
    assert "kimi-k2.7-code:cloud" in r.reply
    assert "← current" in r.reply


@pytest.mark.asyncio
async def test_model_shows_current_when_no_arg():
    h, _ = _handle(default_model="claude-sonnet-4.6")
    r = await cmd_mod.handle(cmd_mod.parse("/model"), _env("/model"), h)
    assert "claude-sonnet-4.6" in r.reply
    assert "Current model" in r.reply
    assert r.data == {"model": "claude-sonnet-4.6"}


@pytest.mark.asyncio
async def test_model_set_owner():
    h, calls = _handle()
    r = await cmd_mod.handle(
        cmd_mod.parse("/model claude-opus-4.7"),
        _env("/model claude-opus-4.7", tier="owner"),
        h,
    )
    assert "model to" in r.reply.lower()
    assert "claude-opus-4.7" in r.reply
    assert calls["set_model"] == ["claude-opus-4.7"]


@pytest.mark.asyncio
async def test_model_set_unknown_still_applied_with_warning():
    h, calls = _handle()
    r = await cmd_mod.handle(
        cmd_mod.parse("/model some-wild-model"),
        _env("/model some-wild-model", tier="owner"),
        h,
    )
    assert "some-wild-model" in r.reply
    assert calls["set_model"] == ["some-wild-model"]


@pytest.mark.asyncio
async def test_model_live_ollama_verification_is_actually_awaited(monkeypatch):
    async def _live_models(force_refresh=False):
        await asyncio.sleep(0)
        return [("frontier:cloud", "Frontier Cloud", "🅞")]

    monkeypatch.setattr(cmd_mod, "fetch_ollama_models", _live_models)
    h, calls = _handle()

    result = await cmd_mod.handle(
        cmd_mod.parse("/model frontier:cloud"),
        _env("/model frontier:cloud", tier="owner"),
        h,
    )

    assert result.reply == "Model set to `frontier:cloud`."
    assert calls["set_model"] == ["frontier:cloud"]


@pytest.mark.asyncio
async def test_gateway_health_probes_run_concurrently(monkeypatch):
    started: list[str] = []
    both_started = asyncio.Event()

    async def _fake_get(self, url, *, headers=None):
        del self, headers
        started.append(url)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return SimpleNamespace(status_code=200, json=lambda: {})

    monkeypatch.setattr(cmd_mod.httpx.AsyncClient, "get", _fake_get)

    result = await asyncio.wait_for(
        cmd_mod._gateway_ping(
            {
                "one": "http://provider-one/v1",
                "two": "http://provider-two/v1",
            }
        ),
        timeout=0.5,
    )

    assert result == {"one": "ok", "two": "ok"}
    assert set(started) == {
        "http://provider-one/v1/models",
        "http://provider-two/v1/models",
    }


@pytest.mark.asyncio
async def test_sync_model_catalog_never_bridges_a_running_event_loop(monkeypatch):
    cached = [
        ("cached:latest", "Cached", "🖥️"),
        ("frontier:cloud", "Frontier Cloud", "🅞"),
    ]
    monkeypatch.setattr(cmd_mod, "_OLLAMA_MODEL_CACHE", (0.0, cached))

    async def _must_not_run(force_refresh=False):
        raise AssertionError("sync catalog attempted live network discovery")

    monkeypatch.setattr(cmd_mod, "fetch_ollama_models", _must_not_run)
    h, _ = _handle()

    catalog = cmd_mod.model_catalog_for_runtime(h)

    assert any(model == "kimi-k2.7-code:cloud" for model, _label in catalog["ollama"])
    assert ("cached:latest", "🖥️ Cached") not in catalog["ollama"]


def test_model_catalog_excludes_internal_and_unapproved_local_models():
    h, _ = _handle()
    discovered = [
        ("kimi-k2.7-code:cloud", "Kimi K2.7 Code", "☁️"),
        ("gemma4:12b", "Gemma 4 12B", "🖥️"),
        ("norax-embed-v3:latest", "Norax Embed V3", "📎"),
        ("kimi-k3:cloud", "Kimi K3 Cloud", "🅞"),
    ]

    catalog = cmd_mod.model_catalog_for_runtime(h, ollama_models=discovered)

    assert "kimi-k2.7-code:cloud" in {model for model, _ in catalog["ollama"]}
    assert "gemma4:12b" not in {model for models in catalog.values() for model, _ in models}
    assert "norax-embed-v3:latest" not in {
        model for models in catalog.values() for model, _ in models
    }
    assert len(catalog["ollama"]) <= 25


@pytest.mark.asyncio
async def test_ollama_discovery_is_single_flight_and_sanitized(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "models": [
                    {"name": "safe:latest"},
                    {"name": "safe:latest"},
                    {"name": "bad model\n```"},
                ]
            }

    async def _fake_get(self, url):
        nonlocal calls
        del self, url
        calls += 1
        started.set()
        await release.wait()
        return _Response()

    monkeypatch.setattr(cmd_mod, "_OLLAMA_MODEL_CACHE", None)
    monkeypatch.setattr(cmd_mod, "_OLLAMA_REFRESH_TASK", None)
    monkeypatch.setattr(cmd_mod.httpx.AsyncClient, "get", _fake_get)

    first = asyncio.create_task(cmd_mod.fetch_ollama_models())
    second = asyncio.create_task(cmd_mod.fetch_ollama_models())
    await started.wait()
    await asyncio.sleep(0)
    assert calls == 1
    release.set()

    one, two = await asyncio.gather(first, second)

    assert one == two == [("safe:latest", "Safe Latest", "🖥️")]


@pytest.mark.asyncio
async def test_settings_refreshes_live_cloud_catalog(monkeypatch):
    async def _live_models(force_refresh=False):
        return [("new-cloud:cloud", "New Cloud", "🅞")]

    monkeypatch.setattr(cmd_mod, "fetch_ollama_models", _live_models)
    h, _ = _handle()

    result = await cmd_mod.handle(cmd_mod.parse("/settings"), _env("/settings"), h)

    assert ("new-cloud:cloud", "🅞 New Cloud") in result.data["model_catalog"]["ollama"]


@pytest.mark.asyncio
async def test_model_set_non_owner_denied():
    h, calls = _handle()
    r = await cmd_mod.handle(
        cmd_mod.parse("/model claude-opus-4.7"),
        _env("/model claude-opus-4.7", tier="user", sid="xxx"),
        h,
    )
    assert "owner" in r.reply.lower()
    assert calls["set_model"] == []


# ---------------------------------------------------------------------------
# /think, /reasoning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_think_sets_owner():
    h, _ = _handle()
    seen = []
    h.set_thinking_effort = seen.append
    r = await cmd_mod.handle(cmd_mod.parse("/think high"), _env("/think high"), h)
    assert seen == ["high"]
    assert "high" in r.reply


@pytest.mark.asyncio
async def test_think_rejects_bad_level():
    h, _ = _handle()
    r = await cmd_mod.handle(cmd_mod.parse("/think turbo"), _env("/think turbo"), h)
    assert "off|low|medium|high|xhigh" in r.reply


@pytest.mark.asyncio
async def test_reasoning_sets_owner():
    h, _ = _handle()
    seen = []
    h.set_reasoning_output = seen.append
    r = await cmd_mod.handle(cmd_mod.parse("/reasoning on"), _env("/reasoning on"), h)
    assert seen == [True]
    assert "on" in r.reply


@pytest.mark.asyncio
async def test_reasoning_non_owner_denied():
    h, _ = _handle()
    seen = []
    h.set_reasoning_output = seen.append
    r = await cmd_mod.handle(
        cmd_mod.parse("/reasoning off"), _env("/reasoning off", tier="user", sid="u"), h
    )
    assert seen == []
    assert "owner" in r.reply.lower()


# ---------------------------------------------------------------------------
# /settings, /planning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_returns_structured_snapshot():
    h, _ = _handle(default_model="gpt-5.5")
    h.thinking_effort = "high"
    h.reasoning_output = True
    h.planning_mode = "orchestrator"
    r = await cmd_mod.handle(cmd_mod.parse("/settings"), _env("/settings"), h)
    assert "gpt-5.5" in r.reply
    assert "orchestrator" in r.reply
    assert "high" in r.reply
    assert "on" in r.reply
    expected = {
        "model": "gpt-5.5",
        "planning_mode": "orchestrator",
        "thinking_effort": "high",
        "reasoning_output": True,
        "max_tool_rounds": 0,
        "memory_depth": "auto",
        "weak_model_boost": "auto",
        "stream_replies": True,
        "response_length": "balanced",
        "tool_activity": "normal",
    }
    # Snapshot is allowed to grow (planner/executor/planning_active added later);
    # assert the core keys are present and correct (subset match).
    assert r.data is not None
    assert {k: r.data.get(k) for k in expected} == expected


@pytest.mark.asyncio
async def test_planning_sets_owner():
    h, _ = _handle()
    seen = []
    h.set_planning_mode = seen.append
    r = await cmd_mod.handle(
        cmd_mod.parse("/planning orchestrator"),
        _env("/planning orchestrator"),
        h,
    )
    assert seen == ["orchestrator"]
    assert "orchestrator" in r.reply


@pytest.mark.asyncio
async def test_planning_non_owner_denied():
    h, _ = _handle()
    seen = []
    h.set_planning_mode = seen.append
    r = await cmd_mod.handle(
        cmd_mod.parse("/planning direct"),
        _env("/planning direct", tier="user", sid="u"),
        h,
    )
    assert seen == []
    assert "owner" in r.reply.lower()


def test_settings_and_planning_are_known():
    assert cmd_mod.is_known(cmd_mod.parse("/settings"))
    assert cmd_mod.is_known(cmd_mod.parse("/planning"))


@pytest.mark.asyncio
async def test_rounds_auto_displays_the_effective_hard_cap(monkeypatch):
    monkeypatch.setattr(agent_loop, "HARD_ROUND_CAP", 40)
    h, _ = _handle()

    result = await cmd_mod.handle(cmd_mod.parse("/rounds"), _env("/rounds"), h)

    assert "auto (40)" in result.reply


@pytest.mark.asyncio
async def test_rounds_rejects_a_choice_above_the_process_hard_cap(monkeypatch):
    monkeypatch.setattr(agent_loop, "HARD_ROUND_CAP", 24)
    h, _ = _handle()
    seen = []
    h.set_max_tool_rounds = seen.append

    result = await cmd_mod.handle(cmd_mod.parse("/rounds 48"), _env("/rounds 48"), h)

    assert seen == []
    assert "hard safety cap of 24" in result.reply
    assert result.data == {"max_tool_rounds": 0, "hard_round_cap": 24}


# ---------------------------------------------------------------------------
# /new
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_cleared():
    h, calls = _handle(cleared=True)
    r = await cmd_mod.handle(cmd_mod.parse("/new"), _env("/new"), h)
    assert "reset" in r.reply.lower()
    assert calls["reset_window"] == ["c1"]


@pytest.mark.asyncio
async def test_new_no_window():
    h, calls = _handle(cleared=False)
    r = await cmd_mod.handle(cmd_mod.parse("/new"), _env("/new"), h)
    assert "acknowledged" in r.reply.lower() or "reset" in r.reply.lower()
    assert calls["reset_window"] == ["c1"]


# ---------------------------------------------------------------------------
# /stop, /restart — owner gate + post_send presence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_non_owner_denied():
    h, _ = _handle()
    r = await cmd_mod.handle(
        cmd_mod.parse("/stop"),
        _env("/stop", tier="user", sid="xxx"),
        h,
    )
    assert r.post_send is None
    assert "owner" in r.reply.lower()


@pytest.mark.asyncio
async def test_stop_owner_no_stream_does_not_shutdown():
    h, _ = _handle()
    r = await cmd_mod.handle(cmd_mod.parse("/stop"), _env("/stop"), h)
    # /stop is soft-interrupt only; never triggers process shutdown.
    assert r.post_send is None
    assert "nothing" in r.reply.lower() or "stop" in r.reply.lower()


@pytest.mark.asyncio
async def test_stop_owner_cancels_active_stream():
    cancelled = {"n": 0}

    def _cancel(channel_id: str = "") -> bool:
        cancelled["n"] += 1
        return True

    h, _ = _handle()
    h = cmd_mod.RuntimeHandle(
        default_model=h.default_model,
        set_default_model=h.set_default_model,
        started_at=h.started_at,
        wall_started_at=h.wall_started_at,
        metrics=h.metrics,
        gateway_base_url=h.gateway_base_url,
        event_log=h.event_log,
        reset_window_for=h.reset_window_for,
        outbound=h.outbound,
        cancel_stream=_cancel,
    )
    r = await cmd_mod.handle(cmd_mod.parse("/stop"), _env("/stop"), h)
    assert r.post_send is None
    assert cancelled["n"] == 1
    assert "stop" in r.reply.lower()


@pytest.mark.asyncio
async def test_restart_non_owner_denied():
    h, _ = _handle()
    r = await cmd_mod.handle(
        cmd_mod.parse("/restart"),
        _env("/restart", tier="user", sid="xxx"),
        h,
    )
    assert r.post_send is None


@pytest.mark.asyncio
async def test_restart_owner_returns_post_send():
    h, _ = _handle()
    r = await cmd_mod.handle(cmd_mod.parse("/restart"), _env("/restart"), h)
    assert r.post_send is not None


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_help_lists_core_commands():
    h, _ = _handle()
    r = await cmd_mod.handle(cmd_mod.parse("/help"), _env("/help"), h)
    for cmd in (
        "/status",
        "/models",
        "/model",
        "/settings",
        "/planning",
        "/think",
        "/reasoning",
        "/new",
        "/stop",
        "/restart",
    ):
        assert cmd in r.reply


# ---------------------------------------------------------------------------
# Runtime integration — commands intercepted before brain call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_routes_command_not_to_brain(monkeypatch, tmp_path):
    """A /status envelope should never reach hot_path.run_turn."""
    from norax.gateway_client import GatewayClient
    from norax.observability.log import EventLog
    from norax.observability.metrics import Metrics
    from norax.runtime.core import Runtime
    from norax.runtime.ingress_bus import IngressBus
    from norax.runtime.outbound import OutboundRegistry

    cfg = SimpleNamespace(
        shutdown_grace_seconds=5,
        event_log=tmp_path / "events.jsonl",
    )
    gw = GatewayClient(base_url="http://stub/v1")
    rt = Runtime(
        ingress=IngressBus([]),
        events=EventLog(cfg.event_log),
        cfg=cfg,
        gateway=gw,
        outbound=OutboundRegistry(),
        metrics=Metrics(),
    )

    brain_called = {"n": 0}

    async def _fake_plan_turn(*a, **kw):
        brain_called["n"] += 1
        return None, None

    monkeypatch.setattr("norax.brain.hot_path.plan_turn", _fake_plan_turn)

    sends: list[tuple[str, str, str | None]] = []

    class _FakeOut:
        async def send(self, target, text, *, reply_to=None):
            sends.append((target, text, reply_to))
            return {"ok": True, "message_id": "x"}

    rt.outbound.register("discord", _FakeOut())

    await rt._handle_turn_inner(_env("/status", tier="owner"))
    assert brain_called["n"] == 0
    assert sends, "expected at least one outbound reply from the command"
    assert "NORAX//STATUS" in sends[0][1]
    assert sends[0][2] is None

    await gw.aclose()


# ---------------------------------------------------------------------------
# Native Discord slash-dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_dispatch_returns_reply_and_logs(tmp_path):
    from norax.envelope import Principal
    from norax.gateway_client import GatewayClient
    from norax.observability.log import EventLog
    from norax.observability.metrics import Metrics
    from norax.runtime.core import Runtime
    from norax.runtime.ingress_bus import IngressBus
    from norax.runtime.outbound import OutboundRegistry

    cfg = SimpleNamespace(
        shutdown_grace_seconds=5,
        event_log=tmp_path / "events.jsonl",
    )
    gw = GatewayClient(base_url="http://stub/v1")
    rt = Runtime(
        ingress=IngressBus([]),
        events=EventLog(cfg.event_log),
        cfg=cfg,
        gateway=gw,
        outbound=OutboundRegistry(),
        metrics=Metrics(),
    )
    principal = Principal(id="owner-123", label="Colby", trust=True, tier="owner")
    out = await rt._slash_dispatch(
        "status",
        {},
        principal,
        {"channel_id": "c1", "interaction_id": "i1", "guild_id": None},
    )
    assert "NORAX//STATUS" in out["reply"]
    assert out["post_send"] is None
    await gw.aclose()


@pytest.mark.asyncio
async def test_slash_dispatch_model_args_passthrough(tmp_path):
    from norax.envelope import Principal
    from norax.gateway_client import GatewayClient
    from norax.observability.log import EventLog
    from norax.observability.metrics import Metrics
    from norax.runtime.core import Runtime
    from norax.runtime.ingress_bus import IngressBus
    from norax.runtime.outbound import OutboundRegistry

    cfg = SimpleNamespace(
        shutdown_grace_seconds=5,
        event_log=tmp_path / "events.jsonl",
    )
    gw = GatewayClient(base_url="http://stub/v1")
    rt = Runtime(
        ingress=IngressBus([]),
        events=EventLog(cfg.event_log),
        cfg=cfg,
        gateway=gw,
        outbound=OutboundRegistry(),
        metrics=Metrics(),
    )
    principal = Principal(id="owner-123", label="Colby", trust=True, tier="owner")
    out = await rt._slash_dispatch(
        "model",
        {"args": ["claude-opus-4.7"]},
        principal,
        {"channel_id": "c1", "interaction_id": "i2", "guild_id": None},
    )
    assert "claude-opus-4.7" in out["reply"]
    assert rt.default_model == "claude-opus-4.7"
    await gw.aclose()
