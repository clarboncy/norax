"""Runtime health-evidence and failure-classification helpers."""

from __future__ import annotations

from typing import Any

from ..gateway_client import GatewayUpstreamError, SpendGuardTripped
from ..safety.secrets import redact
from .capability_registry import CapabilityRegistry
from .validation import _explicit_result_ok

_TOOL_CAPABILITY = {
    "browser": "browser_tool",
    "computer_use": "computer_tool",
    "sandbox_exec": "sandbox_tool",
    "web_search": "web_search_tool",
    "deep_research": "web_search_tool",
}


def _record_operational_probe(
    capabilities: CapabilityRegistry,
    name: str,
    result: object,
) -> None:
    """Record a probe result without turning partial availability into failure."""
    if not isinstance(result, dict) or not _explicit_result_ok(result):
        raise RuntimeError(str(result)[:500])
    degraded_reason = result.get("degraded_reason")
    if degraded_reason:
        capabilities.mark_degraded(name, str(degraded_reason))
    else:
        capabilities.mark_ok(name)
    capabilities.update_metadata(
        name,
        **{
            str(key): value
            for key, value in result.items()
            if key not in {"ok", "degraded_reason", "error"}
        },
    )


def _refresh_tool_capability_evidence(
    capabilities: CapabilityRegistry,
    trace: object,
) -> None:
    """Refresh capability freshness only from explicit successful receipts."""
    if not isinstance(trace, list):
        return
    for item in trace:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        capability = "remote_relay" if name.startswith("remote_") else _TOOL_CAPABILITY.get(name)
        if capability and _explicit_result_ok(item.get("result")):
            capabilities.touch(capability)


def _remote_relay_probe_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate relay health into ready/degraded capability evidence."""
    relay_ok = payload.get("ok") is True
    raw_count = payload.get("live_count", 0)
    live_count = (
        max(0, int(raw_count))
        if not isinstance(raw_count, bool) and isinstance(raw_count, int | float)
        else 0
    )
    raw_nodes = payload.get("live_nodes", [])
    live_nodes = raw_nodes[:100] if isinstance(raw_nodes, list) else []
    return {
        "ok": relay_ok,
        "degraded_reason": (
            "relay is healthy but no optional remote nodes are connected"
            if relay_ok and live_count == 0
            else ""
        ),
        "live_count": live_count,
        "live_nodes": live_nodes,
    }


def _turn_failure_message(error: BaseException) -> str:
    """Build the user-visible reply for a failed prompted turn."""
    if isinstance(error, SpendGuardTripped):
        return (
            f"⏸️ **Spend guard activated** — {error.count} LLM calls in the last "
            f"{error.window} (limit {error.limit}). I've paused to protect your "
            "subscriptions. Try again in a minute or raise the limit with "
            "`NORAX_LLM_MAX_CALLS_PER_MIN` / `NORAX_LLM_MAX_CALLS_PER_HOUR`."
        )
    if isinstance(error, GatewayUpstreamError) and error.status == 402:
        detail = str(redact(error.upstream_message.strip()))
        detail_line = f"\nProvider detail: {detail[:500]}" if detail else ""
        return (
            "💳 **Provider balance alert (HTTP 402)** — the selected model could "
            "not continue because the account or API key lacks sufficient credit "
            "for this request. No fallback model was used, and your selected model "
            f"has not changed.\nModel: `{error.model}`{detail_line}\n"
            "Add provider credit, then send `Continue` or repeat the prompt."
        )
    if isinstance(error, GatewayUpstreamError):
        return (
            "⚠️ I hit an upstream error and couldn't complete that turn. "
            "The failure is logged — try again, or check "
            "`journalctl --user -u norax-ai` if it persists.\n"
            f"Error: Upstream {error.status} [{error.model}]"
        )
    return (
        "⚠️ I hit an internal error and couldn't complete that turn. "
        "The failure is logged — try again, or check "
        "`journalctl --user -u norax-ai` if it persists.\n"
        f"Error: {type(error).__name__}"
    )


def _deferred_runtime_lifecycle_action(trace: list[dict]) -> str | None:
    """Return the final validated self-lifecycle request from a tool trace."""
    action: str | None = None
    for item in trace:
        if not isinstance(item, dict):
            continue
        result = item.get("result")
        if not isinstance(result, dict) or not _explicit_result_ok(result):
            continue
        candidate = result.get("runtime_lifecycle_deferred")
        if candidate in {"restart", "stop"}:
            action = str(candidate)
    return action
