"""Regression tests for operational capability probe classification."""

from __future__ import annotations

import pytest

from norax.runtime.capability_registry import CapabilityRegistry
from norax.runtime.core import (
    _apply_configured_fleet_startup_fixes,
    _record_operational_probe,
    _refresh_tool_capability_evidence,
    _remote_relay_probe_result,
)


def test_healthy_relay_without_optional_nodes_is_degraded_not_failed() -> None:
    registry = CapabilityRegistry()
    result = _remote_relay_probe_result({"ok": True, "live_count": 0, "live_nodes": []})

    _record_operational_probe(registry, "remote_relay", result)

    assert registry.degraded_capabilities() == ["remote_relay"]
    assert registry.failed_capabilities() == []
    assert registry.is_enabled("remote_relay")


def test_relay_with_live_node_is_ready() -> None:
    registry = CapabilityRegistry()
    result = _remote_relay_probe_result({"ok": True, "live_count": 1, "live_nodes": ["n1"]})

    _record_operational_probe(registry, "remote_relay", result)

    assert registry.is_ready("remote_relay")


def test_unhealthy_operational_probe_raises_for_failure_path() -> None:
    with pytest.raises(RuntimeError, match="backend unavailable"):
        _record_operational_probe(
            CapabilityRegistry(),
            "remote_relay",
            {"ok": False, "error": "backend unavailable"},
        )


def test_private_fleet_startup_mutation_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.delenv("NORAX_APPLY_FLEET_STARTUP_FIXES", raising=False)
    monkeypatch.setattr(
        "norax.ops.fleet_healthcheck.apply_startup_fixes",
        lambda: calls.append(True),
    )

    assert _apply_configured_fleet_startup_fixes() is False
    assert calls == []

    monkeypatch.setenv("NORAX_APPLY_FLEET_STARTUP_FIXES", "true")
    assert _apply_configured_fleet_startup_fixes() is True
    assert calls == [True]


@pytest.mark.parametrize(
    ("tool_name", "capability"),
    [
        ("browser", "browser_tool"),
        ("computer_use", "computer_tool"),
        ("sandbox_exec", "sandbox_tool"),
        ("web_search", "web_search_tool"),
        ("deep_research", "web_search_tool"),
        ("remote_exec", "remote_relay"),
    ],
)
def test_successful_tool_receipt_refreshes_capability(tool_name: str, capability: str) -> None:
    registry = CapabilityRegistry()
    registry.register(capability, stale_after_sec=900)
    registry.mark_ok(capability)
    registry.mark_stale(capability)
    assert registry.status()[capability]["state"] == "stale"

    _refresh_tool_capability_evidence(
        registry,
        [{"name": tool_name, "result": {"ok": True}}],
    )

    assert registry.is_ready(capability)


def test_failed_or_nonliteral_tool_receipt_does_not_fake_freshness() -> None:
    registry = CapabilityRegistry()
    registry.register("browser_tool", stale_after_sec=900)
    registry.mark_ok("browser_tool")
    registry.mark_stale("browser_tool")

    _refresh_tool_capability_evidence(
        registry,
        [
            {"name": "browser", "result": {"ok": "true"}},
            {"name": "browser", "result": {"ok": False}},
            {"name": "unknown", "result": {"ok": True}},
        ],
    )

    assert registry.status()["browser_tool"]["state"] == "stale"
