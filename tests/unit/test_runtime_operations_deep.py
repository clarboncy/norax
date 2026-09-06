from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from norax.brain import policy
from norax.dispatch import browser as browser_module
from norax.dispatch import sandbox as sandbox_module
from norax.runtime import operations as module
from norax.runtime.capability_registry import CapabilityRegistry


class _Response:
    def __init__(self, *, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"ok": True, "live_count": 1, "live_nodes": ["node"]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _HttpClient:
    searx_status = 200
    searx_error = False
    relay_payload = {"ok": True, "live_count": 1, "live_nodes": ["node"]}
    relay_error = False

    def __init__(self, *, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url):
        if url.endswith("/healthz"):
            if self.searx_error:
                raise RuntimeError("search offline")
            return _Response(status=self.searx_status)
        if self.relay_error:
            raise RuntimeError("relay offline")
        return _Response(payload=self.relay_payload)


class _Process:
    def __init__(self, *, returncode=0, output=b"ready", error=b""):
        self.returncode = returncode
        self.output = output
        self.error = error
        self.killed = False
        self.waited = False

    async def communicate(self):
        return self.output, self.error

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True


def _install_probe_fakes(monkeypatch, *, process=None):
    import httpx

    process = process or _Process()
    monkeypatch.setattr(httpx, "AsyncClient", _HttpClient)
    monkeypatch.setattr(
        module.asyncio,
        "create_subprocess_exec",
        lambda *_args, **_kwargs: _async_value(process),
    )
    monkeypatch.setattr(
        sandbox_module,
        "get_sandbox_manager",
        lambda: SimpleNamespace(runtime="podman"),
    )
    monkeypatch.setattr(
        browser_module,
        "probe_browser_backend",
        lambda **_kwargs: _async_value({"ok": True, "backend": "browser"}),
    )
    return process


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_operational_probe_exercises_every_real_backend_with_unique_display_path(monkeypatch):
    _HttpClient.searx_status = 200
    _HttpClient.searx_error = False
    _HttpClient.relay_error = False
    _HttpClient.relay_payload = {"ok": True, "live_count": 1, "live_nodes": ["node"]}
    _install_probe_fakes(monkeypatch)
    monkeypatch.setattr(
        module.dispatch_tools, "REGISTRY", {"memory_search": object(), "read": object()}
    )
    monkeypatch.setattr(policy, "ALL_OWNER_TOOLS", {"read"})
    screenshot_paths = []

    async def screenshot(*, action, path):
        assert action == "screenshot"
        screenshot_paths.append(Path(path))
        assert Path(path).is_file()
        Path(path).write_bytes(b"png")
        return {"ok": True, "path": path}

    monkeypatch.setattr(module.dispatch_tools, "t_computer_use", screenshot)
    observed = {}
    monkeypatch.setattr(
        module,
        "_record_operational_probe",
        lambda _caps, name, result: observed.setdefault(name, result),
    )
    runtime = SimpleNamespace(capabilities=SimpleNamespace(mark_failed=lambda *_args: None))
    await module.OperationalMixin._probe_operational_tools(runtime, include_interactive=True)

    assert set(observed) == {
        "tool_manifest",
        "sandbox_tool",
        "web_search_tool",
        "remote_relay",
        "browser_tool",
        "computer_tool",
    }
    assert all(result["ok"] is True for result in observed.values())
    assert screenshot_paths and not screenshot_paths[0].exists()
    assert observed["sandbox_tool"]["runtime"] == "podman"
    assert observed["remote_relay"]["live_count"] == 1


@pytest.mark.asyncio
async def test_operational_probe_reports_manifest_sandbox_and_optional_degradation(monkeypatch):
    import httpx

    _HttpClient.searx_error = True
    _HttpClient.relay_error = False
    _HttpClient.relay_payload = {"ok": True, "live_count": 0, "live_nodes": []}
    monkeypatch.setattr(httpx, "AsyncClient", _HttpClient)
    monkeypatch.setattr(
        sandbox_module,
        "get_sandbox_manager",
        lambda: SimpleNamespace(runtime=""),
    )
    monkeypatch.setattr(module.dispatch_tools, "REGISTRY", {"unexpected": object()})
    monkeypatch.setattr(policy, "ALL_OWNER_TOOLS", {"expected"})
    monkeypatch.setenv("NORAX_SERPER_API_KEY", "configured")
    caps = CapabilityRegistry()
    runtime = SimpleNamespace(capabilities=caps)

    await module.OperationalMixin._probe_operational_tools(runtime, include_interactive=False)

    assert caps.is_enabled("web_search_tool")
    assert caps.is_enabled("remote_relay")
    assert set(caps.failed_capabilities()) == {"tool_manifest", "sandbox_tool"}
    assert set(caps.degraded_capabilities()) == {"web_search_tool", "remote_relay"}


@pytest.mark.asyncio
async def test_sandbox_timeout_and_unconfigured_search_return_explicit_failures(monkeypatch):
    process = _install_probe_fakes(monkeypatch)
    _HttpClient.searx_error = False
    _HttpClient.searx_status = 503
    _HttpClient.relay_error = False
    monkeypatch.delenv("NORAX_SERPER_API_KEY", raising=False)
    monkeypatch.delenv("NORAX_TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(module.dispatch_tools, "REGISTRY", {})
    monkeypatch.setattr(policy, "ALL_OWNER_TOOLS", set())
    real_wait_for = module.asyncio.wait_for

    async def timeout_wait(awaitable, *, timeout):
        if timeout == 8.0:
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise TimeoutError
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(module.asyncio, "wait_for", timeout_wait)
    observed = {}
    monkeypatch.setattr(
        module,
        "_record_operational_probe",
        lambda _caps, name, result: observed.setdefault(name, result),
    )
    runtime = SimpleNamespace(capabilities=SimpleNamespace(mark_failed=lambda *_args: None))
    await module.OperationalMixin._probe_operational_tools(runtime, include_interactive=False)
    assert observed["sandbox_tool"] == {"ok": False, "error": "docker info timed out"}
    assert process.killed and process.waited
    assert observed["web_search_tool"] == {
        "ok": False,
        "backend": "none",
        "degraded_reason": "",
    }


@pytest.mark.asyncio
async def test_sandbox_nonzero_exit_preserves_bounded_diagnostic(monkeypatch):
    process = _Process(returncode=1, output=b"", error=b"daemon unavailable")
    _install_probe_fakes(monkeypatch, process=process)
    _HttpClient.searx_status = 200
    _HttpClient.searx_error = False
    monkeypatch.setattr(module.dispatch_tools, "REGISTRY", {})
    monkeypatch.setattr(policy, "ALL_OWNER_TOOLS", set())
    observed = {}
    monkeypatch.setattr(
        module,
        "_record_operational_probe",
        lambda _caps, name, result: observed.setdefault(name, result),
    )
    runtime = SimpleNamespace(capabilities=SimpleNamespace(mark_failed=lambda *_args: None))
    await module.OperationalMixin._probe_operational_tools(runtime, include_interactive=False)
    assert observed["sandbox_tool"] == {
        "ok": False,
        "runtime": "podman",
        "response_observed": False,
        "error": "daemon unavailable",
    }


@pytest.mark.asyncio
async def test_probe_wrapper_marks_backend_exception_failed(monkeypatch):
    _install_probe_fakes(monkeypatch)
    _HttpClient.relay_error = True
    _HttpClient.searx_error = False
    _HttpClient.searx_status = 200
    monkeypatch.setattr(module.dispatch_tools, "REGISTRY", {})
    monkeypatch.setattr(policy, "ALL_OWNER_TOOLS", set())
    caps = CapabilityRegistry()
    await module.OperationalMixin._probe_operational_tools(
        SimpleNamespace(capabilities=caps), include_interactive=False
    )
    assert caps.failed_capabilities() == ["remote_relay"]


@pytest.mark.asyncio
async def test_cache_warmup_handles_empty_success_partial_and_setup_failure(monkeypatch, caplog):
    caplog.set_level("INFO")
    runtime = SimpleNamespace(_hybrid=None, _context_injector=None)
    await module.OperationalMixin._warmup_caches(runtime)

    calls = []

    async def call(name, *, fail=False):
        calls.append(name)
        if fail:
            raise RuntimeError(name)

    embedder = SimpleNamespace(warmup=lambda: call("embed"))
    hybrid = SimpleNamespace(
        embedding=SimpleNamespace(embedder=embedder),
        ensure_index=lambda: call("index"),
    )
    context = SimpleNamespace(external=SimpleNamespace(refresh_if_stale=lambda: call("external")))
    runtime = SimpleNamespace(_hybrid=hybrid, _context_injector=context)
    await module.OperationalMixin._warmup_caches(runtime)
    assert calls == ["embed", "index", "external"]
    assert "caches warmed up" in caplog.text

    hybrid.ensure_index = lambda: call("index-failed", fail=True)
    await module.OperationalMixin._warmup_caches(runtime)
    assert "warmup_caches.partial" in caplog.text

    class Broken:
        @property
        def embedding(self):
            raise RuntimeError("setup failed")

    runtime = SimpleNamespace(_hybrid=Broken(), _context_injector=None)
    await module.OperationalMixin._warmup_caches(runtime)
    assert "first query will build lazily" in caplog.text


@pytest.mark.asyncio
async def test_cache_warmup_supports_embedder_without_warmup_and_absent_external():
    calls = []
    hybrid = SimpleNamespace(
        embedding=SimpleNamespace(embedder=object()),
        ensure_index=lambda: _record_async(calls, "index"),
    )
    runtime = SimpleNamespace(
        _hybrid=hybrid,
        _context_injector=SimpleNamespace(external=None),
    )
    await module.OperationalMixin._warmup_caches(runtime)
    assert calls == ["index"]


async def _record_async(calls, value):
    calls.append(value)


class _TransportGateway:
    def __init__(self, evidence_by_model, *, same_route=False):
        self.evidence_by_model = evidence_by_model
        self.same_route = same_route
        self.probed = []

    def route_for(self, model):
        return ("provider", "same" if self.same_route else model)

    async def transport_probe(self, model, *, timeout):
        assert timeout == 5.0
        self.probed.append(model)
        evidence = self.evidence_by_model[model]
        if isinstance(evidence, BaseException):
            raise evidence
        return evidence


@pytest.mark.asyncio
async def test_transport_probe_deduplicates_routes_and_fails_over_on_negative_evidence():
    gateway = _TransportGateway(
        {
            "primary": {"ok": False, "secret": "not evidence"},
            "fallback": {
                "ok": True,
                "provider": "cloud",
                "endpoint": "https://provider.test/models",
                "kind": "catalog",
                "status_code": 200,
                "ignored": "private",
            },
        }
    )
    runtime = SimpleNamespace(
        gateway=gateway,
        _effective_model="primary",
        failover_models=["primary", "fallback"],
        _completion_probe_mode="transport",
    )
    result = await module.OperationalMixin._run_completion_probe(runtime)
    assert gateway.probed == ["primary", "fallback"]
    assert result["ok"] is True and result["fallback"] is True
    assert result["provider"] == "cloud"
    assert result["evidence"] == {
        "endpoint": "https://provider.test/models",
        "kind": "catalog",
        "status_code": 200,
    }

    duplicate = _TransportGateway({"primary": {"ok": False}}, same_route=True)
    runtime.gateway = duplicate
    runtime.failover_models = ["fallback"]
    failed = await module.OperationalMixin._run_completion_probe(runtime)
    assert duplicate.probed == ["primary"]
    assert failed["ok"] is False and len(failed["failures"]) == 1


@pytest.mark.asyncio
async def test_transport_probe_requires_gateway_support_and_handles_no_candidates():
    runtime = SimpleNamespace(
        gateway=object(),
        _effective_model=None,
        failover_models=[],
        _completion_probe_mode="transport",
    )
    result = await module.OperationalMixin._run_completion_probe(runtime)
    assert result["ok"] is False and result["model"] == "?" and result["failures"] == []

    runtime._effective_model = "model"
    result = await module.OperationalMixin._run_completion_probe(runtime)
    assert "does not implement a transport probe" in result["failures"][0]


@pytest.mark.asyncio
async def test_completion_probe_propagates_cancellation():
    gateway = _TransportGateway({"model": asyncio.CancelledError()})
    runtime = SimpleNamespace(
        gateway=gateway,
        _effective_model="model",
        failover_models=[],
        _completion_probe_mode="transport",
    )
    with pytest.raises(asyncio.CancelledError):
        await module.OperationalMixin._run_completion_probe(runtime)


def test_completion_defer_reason_covers_noncompletion_done_tasks_and_no_prior_turn():
    runtime = SimpleNamespace(_completion_probe_mode="transport")
    assert module.OperationalMixin._completion_probe_defer_reason(runtime) is None
    runtime = SimpleNamespace(
        _completion_probe_mode="completion",
        _turn_work_count=0,
        _active_turn_tasks={"done": SimpleNamespace(done=lambda: True)},
        _last_user_turn_time=None,
    )
    assert module.OperationalMixin._completion_probe_defer_reason(runtime) is None


def test_completion_probe_age_handles_future_naive_and_default_clock():
    now = datetime.now(UTC)
    future = {"timestamp": (now + timedelta(seconds=5)).isoformat()}
    assert module.OperationalMixin._completion_probe_age(future, now=now) == 0.0
    assert module.OperationalMixin._completion_probe_age(
        {"timestamp": datetime.now().isoformat()}, now=now
    ) == float("inf")
    recent = {"timestamp": datetime.now(UTC).isoformat()}
    assert module.OperationalMixin._completion_probe_age(recent) < 2


@pytest.mark.asyncio
async def test_capability_probe_loop_marks_degraded_recovers_and_preserves_cancellation(
    monkeypatch,
):
    calls = []
    caps = SimpleNamespace(
        mark_ok=lambda name: calls.append(("ok", name)),
        mark_degraded=lambda name, reason: calls.append(("degraded", name, reason)),
    )
    attempts = 0

    async def probe(*, include_interactive):
        nonlocal attempts
        attempts += 1
        calls.append(("probe", include_interactive))
        if attempts == 1:
            raise RuntimeError("first failed")

    sleeps = 0

    async def sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    runtime = SimpleNamespace(capabilities=caps, _probe_operational_tools=probe)
    with pytest.raises(asyncio.CancelledError):
        await module.OperationalMixin._capability_probe_loop(runtime)
    assert ("probe", True) in calls and ("probe", False) in calls
    assert any(item[0] == "degraded" for item in calls)
    assert ("ok", "operational_probe_loop") in calls

    async def cancel_probe(*, include_interactive):
        raise asyncio.CancelledError

    runtime._probe_operational_tools = cancel_probe
    with pytest.raises(asyncio.CancelledError):
        await module.OperationalMixin._capability_probe_loop(runtime)


@pytest.mark.asyncio
async def test_completion_probe_loop_defers_then_records_success_failure_and_exception(monkeypatch):
    results = [
        {
            "ok": True,
            "model": "local",
            "latency_ms": 1,
            "fallback": False,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        {"ok": False, "failures": ["offline"], "timestamp": datetime.now(UTC).isoformat()},
    ]
    calls = []
    caps = SimpleNamespace(
        mark_ok=lambda name: calls.append(("ok", name)),
        mark_failed=lambda name, error: calls.append(("failed", name, str(error))),
    )
    defer_calls = 0

    def defer():
        nonlocal defer_calls
        defer_calls += 1
        return "active_turn" if defer_calls == 1 else None

    async def run_probe():
        if results:
            return results.pop(0)
        raise RuntimeError("probe crashed")

    sleep_calls = []

    async def sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 4:
            raise asyncio.CancelledError

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    runtime = SimpleNamespace(
        capabilities=caps,
        _completion_probe_interval_seconds=600,
        _last_probe_result={"timestamp": datetime.now(UTC).isoformat()},
        _completion_probe_defer_reason=defer,
        _completion_probe_age=lambda _result: 10.0,
        _run_completion_probe=run_probe,
    )
    with pytest.raises(asyncio.CancelledError):
        await module.OperationalMixin._completion_probe_loop(runtime)
    assert sleep_calls[0] == 15
    assert ("ok", "serving_probe_loop") in calls
    assert any(item[0] == "failed" and "offline" in item[2] for item in calls)
    assert any(item[0] == "failed" and "probe crashed" in item[2] for item in calls)
    assert runtime._last_probe_result["ok"] is False


@pytest.mark.asyncio
async def test_completion_probe_loop_propagates_probe_cancellation(monkeypatch):
    async def cancel():
        raise asyncio.CancelledError

    runtime = SimpleNamespace(
        capabilities=SimpleNamespace(),
        _completion_probe_defer_reason=lambda: None,
        _completion_probe_age=lambda _result: float("inf"),
        _run_completion_probe=cancel,
        _last_probe_result=None,
    )
    with pytest.raises(asyncio.CancelledError):
        await module.OperationalMixin._completion_probe_loop(runtime)
