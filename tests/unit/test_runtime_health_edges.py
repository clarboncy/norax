from __future__ import annotations

from norax.gateway_client import GatewayUpstreamError
from norax.runtime.capability_registry import CapabilityRegistry
from norax.runtime.health import (
    _deferred_runtime_lifecycle_action,
    _refresh_tool_capability_evidence,
    _remote_relay_probe_result,
    _turn_failure_message,
)


def test_refresh_capability_evidence_ignores_malformed_trace_items_and_names():
    registry = CapabilityRegistry()
    registry.register("browser_tool")
    _refresh_tool_capability_evidence(registry, None)
    _refresh_tool_capability_evidence(
        registry,
        ["bad", {"name": 4, "result": {"ok": True}}, {"name": "browser", "result": {}}],
    )
    assert registry.status()["browser_tool"]["last_success_ago"] is None


def test_remote_relay_probe_bounds_malformed_count_and_node_metadata():
    malformed = _remote_relay_probe_result(
        {"ok": True, "live_count": True, "live_nodes": "not-a-list"}
    )
    assert malformed["live_count"] == 0 and malformed["live_nodes"] == []
    assert malformed["degraded_reason"]

    bounded = _remote_relay_probe_result(
        {"ok": False, "live_count": -5, "live_nodes": list(range(150))}
    )
    assert bounded["live_count"] == 0 and len(bounded["live_nodes"]) == 100
    assert bounded["degraded_reason"] == ""


def test_provider_balance_detail_is_redacted_before_user_display():
    secret = "sk-proj-" + "A" * 24
    error = GatewayUpstreamError(402, f"balance empty for {secret}", "provider/model")
    message = _turn_failure_message(error)
    assert secret not in message
    assert "<REDACTED:openai_key>" in message


def test_deferred_lifecycle_ignores_malformed_and_unsuccessful_receipts():
    trace = [
        "bad",
        {"result": None},
        {"result": {"ok": False, "runtime_lifecycle_deferred": "restart"}},
        {"result": {"ok": True, "runtime_lifecycle_deferred": "invalid"}},
    ]
    assert _deferred_runtime_lifecycle_action(trace) is None  # type: ignore[arg-type]
