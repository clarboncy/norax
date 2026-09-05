"""Contract tests for the capability registry and /readyz endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from norax.adapter.http_in import HttpInAdapter
from norax.runtime.capability_registry import CapabilityRegistry


def test_capability_registry_init():
    """CapabilityRegistry starts empty."""
    reg = CapabilityRegistry()
    assert reg.status() == {}
    assert reg.failed_capabilities() == []


def test_capability_registry_mark_ok():
    """mark_ok records a successful init."""
    reg = CapabilityRegistry()
    reg.mark_ok("test_cap")
    assert reg.is_enabled("test_cap")
    assert reg.status()["test_cap"]["enabled"] is True
    assert reg.status()["test_cap"]["last_error"] == ""
    assert reg.failed_capabilities() == []


def test_capability_registry_mark_failed():
    """mark_failed records the error and logs it."""
    reg = CapabilityRegistry()
    err = ValueError("something broke")
    reg.mark_failed("test_cap", err)
    assert not reg.is_enabled("test_cap")
    assert reg.status()["test_cap"]["enabled"] is False
    assert "ValueError" in reg.status()["test_cap"]["last_error"]
    assert "something broke" in reg.status()["test_cap"]["last_error"]
    assert "test_cap" in reg.failed_capabilities()


def test_capability_registry_try_init_success():
    """try_init returns the instance and marks ok on success."""
    reg = CapabilityRegistry()

    class Foo:
        pass

    result = reg.try_init("foo", Foo)
    assert isinstance(result, Foo)
    assert reg.is_enabled("foo")


def test_capability_registry_try_init_failure():
    """try_init returns None and marks failed on exception."""
    reg = CapabilityRegistry()

    def bad_factory() -> None:
        raise RuntimeError("boom")

    result = reg.try_init("bad", bad_factory)
    assert result is None
    assert not reg.is_enabled("bad")
    assert "bad" in reg.failed_capabilities()


def test_capability_registry_register():
    """register pre-registers a capability as not-yet-checked."""
    reg = CapabilityRegistry()
    reg.register("pending")
    assert "pending" in reg.status()
    assert not reg.is_enabled("pending")


def test_register_preserves_explicit_staleness_policy():
    reg = CapabilityRegistry()
    reg.register("heartbeat", stale_after_sec=30)
    reg.mark_initializing("heartbeat")
    reg.mark_ok("heartbeat")
    assert reg.status()["heartbeat"]["stale_after_sec"] == 30


def test_readiness_only_blocks_explicitly_required_capabilities():
    reg = CapabilityRegistry()
    reg.register("browser")
    reg.register("tool_manifest", required_for_readiness=True)
    reg.mark_failed("browser", RuntimeError("optional backend missing"))
    reg.mark_ok("tool_manifest")

    assert reg.failed_capabilities() == ["browser"]
    assert reg.failed_capabilities(required_only=True) == []
    assert reg.readiness_blockers() == []
    assert reg.status()["tool_manifest"]["required_for_readiness"] is True

    reg.mark_failed("tool_manifest", RuntimeError("manifest mismatch"))
    assert reg.readiness_blockers() == ["tool_manifest"]


def test_required_policy_and_probe_metadata_survive_state_changes():
    reg = CapabilityRegistry()
    reg.register("core", stale_after_sec=30, required_for_readiness=True)
    reg.update_metadata("core", evidence="manifest_match")
    reg.mark_initializing("core")
    reg.mark_ok("core")

    status = reg.status()["core"]
    assert status["required_for_readiness"] is True
    assert status["metadata"] == {"evidence": "manifest_match"}


class _ReadyMemory:
    def all_canonical(self):
        return []


class _ReadyGateway:
    base_url = "http://gateway/v1"

    def route_for(self, model: str) -> tuple[str, str]:
        return "provider", self.base_url


def _ready_runtime(tmp_path, capabilities: CapabilityRegistry):
    return SimpleNamespace(
        capabilities=capabilities,
        gateway=_ReadyGateway(),
        _effective_provider="provider",
        _effective_model="model",
        _memory_store=_ReadyMemory(),
        events=SimpleNamespace(_writable=True, _last_write_error=""),
        discord=None,
        _last_probe_result={
            "ok": True,
            "enabled": True,
            "mode": "transport",
            "transport_verified": True,
            "completion_verified": False,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        cfg=SimpleNamespace(state_dir=tmp_path),
    )


@pytest.mark.asyncio
async def test_readyz_reports_optional_failure_without_rejecting_traffic(tmp_path):
    capabilities = CapabilityRegistry()
    capabilities.register("browser")
    capabilities.mark_failed("browser", RuntimeError("not installed"))
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter._runtime_ref = _ready_runtime(tmp_path, capabilities)

    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["components"]["capabilities"]["failed"] == ["browser"]


@pytest.mark.asyncio
async def test_readyz_rejects_failed_required_capability(tmp_path):
    capabilities = CapabilityRegistry()
    capabilities.register("tool_manifest", required_for_readiness=True)
    capabilities.mark_failed("tool_manifest", RuntimeError("mismatch"))
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter._runtime_ref = _ready_runtime(tmp_path, capabilities)

    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["components"]["capabilities"]["readiness_blockers"] == ["tool_manifest"]


@pytest.mark.asyncio
async def test_public_status_bounds_and_scrubs_untrusted_probe_diagnostics(tmp_path):
    capabilities = CapabilityRegistry()
    secret = "sk-" + ("a" * 30)
    capabilities.mark_failed(
        "optional",
        RuntimeError(f"provider {secret} at https://internal.example.invalid/v1 failed"),
    )
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    runtime = _ready_runtime(tmp_path, capabilities)
    runtime._last_probe_result.update(
        {
            "ok": False,
            "enabled": "false",
            "failures": [f"{secret} https://private.example.invalid/" + ("x" * 1_000)],
            "base_url": "https://must-not-leak.example.invalid/v1",
            "rogue": {"arbitrary": object()},
        }
    )
    adapter._runtime_ref = runtime

    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        response = await client.get("/readyz")
        status_response = await client.get("/status")

    assert response.status_code == 503
    probe = response.json()["components"]["completion_probe"]
    assert "base_url" not in probe
    assert "rogue" not in probe
    assert "private.example.invalid" not in probe["failures"][0]
    assert "<REDACTED:openai_key>" in probe["failures"][0]
    assert len(probe["failures"][0]) <= 500
    capability = status_response.json()["capabilities"]["optional"]
    assert secret not in capability["last_error"]
