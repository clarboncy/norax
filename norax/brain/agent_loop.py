"""Agentic tool-use loop.

L10 gives us a first response. If that response contains `tool_calls`,
we execute them, feed results back, and keep looping until the model
emits a final text-only reply (or we hit limits).

This is the minimal OpenAI-style function-calling agent:

    messages: [system, user]
    loop:
        resp = gateway.chat(messages, tools=allowed_tools)
        if resp.tool_calls:
            messages.append(assistant_with_tool_calls)
            for each tool_call:
                result = run_tool(name, args)   # RiskGate + dispatch
                messages.append({"role":"tool", ...})
            continue
        else:
            return resp.content

Limits:
  - max_rounds (default 250): high-cap budget; emergency ceiling still prevents runaway loops
  - per-tool timeout (handled inside the tool fns)
  - RiskGate: all calls checked against sender tier + danger patterns
  - loop_guard: detects repeated and ping-pong call patterns without blocking
    legitimate post-mutation read-back

Events:
  - `tool_call` per invocation (name, args_redacted, ok, dur_ms, err)
  - `agent_round` at each outer iteration (round_idx, tools_count)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

import httpx

from ..dispatch import risk as risk_mod
from ..dispatch import tools as tool_mod
from ..dispatch.loop_guard import LoopDetected, LoopGuard
from ..gateway_client import (
    GatewayClient,
    GatewayRequest,
    GatewayResponse,
    GatewayRouter,
    GatewayUpstreamError,
    ReasoningTagFilter,
    SpendGuardTripped,
    strip_reasoning_blocks,
)
from ..memory.tool_experience import ToolExperienceMemory
from ..runtime.circuit_breaker import get_circuit_registry
from .active_inference import ActiveInference
from .best_of_n import BestOfN
from .goal_drift import DriftSeverity, GoalDriftMonitor
from .metacognitive import MetacognitiveCalibration
from .output_verifier import OutputVerifier
from .self_model import SelfModel, classify_domain
from .strong_model_scaffold import (
    TaskState,
    build_scaffold_prompt,
    build_task_state,
    compress_tool_result,
    enforce_final_verification,
    is_action_command,
    nudge_for_dropped_tool_calls,
    summarize_mutation_trace,
    tool_call_is_mutating,
    tool_call_is_successful_verification,
    tool_failure_is_blocking,
)
from .weak_model_boost import TurnToolCache, compose_weak_model_boosters

log = logging.getLogger("norax.agent_loop")


def _is_usage_limit(e: BaseException) -> bool:
    """True when the error is a terminal rate-limit (usage_limit_reached).

    These are not transient — the plan's quota is exhausted and retrying the
    same upstream will burn attempts for nothing.  The caller should skip
    straight to cross-model failover.
    """
    if not isinstance(e, GatewayUpstreamError):
        return False
    if e.status not in (429, 502):
        return False
    # Check both the extracted upstream_message and the full string repr,
    # since _extract_upstream_message may not always include the error type.
    haystack = f"{getattr(e, 'upstream_message', '')} {e}"
    return "usage_limit_reached" in haystack


_MODEL_FAILOVER_HTTP_STATUSES = frozenset({401, 404, 408, 425, 429, 500, 502, 503, 504})


def _parse_task_state_json(context_block: str) -> str:
    """Extract just the JSON payload from a context_block() string.

    context_block() returns 'TASK_STATE\\n{json}\\nDIRECTIVE: ...' — this
    returns only the JSON portion so json.loads() doesn't choke on the
    trailing directive text.
    """
    raw = context_block.split("\n", 1)[1] if "\n" in context_block else context_block
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        brace_end = raw.rfind("}")
        if brace_end > 0:
            return raw[: brace_end + 1]
        return raw


def _is_model_failover_error(e: BaseException | None) -> bool:
    """Return whether another configured model/provider may complete the call.

    These statuses describe provider credentials, a missing selected model,
    throttling, timeout, or upstream availability. Retrying the exact
    same route is either wasteful or less useful than moving to a separately
    configured fallback. Request-shape, permission, and policy failures remain
    terminal because switching models should not bypass them. HTTP 402 is also
    terminal: a balance failure must remain visible and must never silently
    switch the user's selected model.
    """
    if e is None or isinstance(e, SpendGuardTripped):
        return False
    if _is_usage_limit(e):
        return True
    if isinstance(e, GatewayUpstreamError):
        return e.status in _MODEL_FAILOVER_HTTP_STATUSES
    return isinstance(e, (httpx.TransportError, TimeoutError, ConnectionError, OSError))


# Sensible defaults; runtime can override.
# SAFETY (2026-07-17): every turn MUST run under a wall-clock deadline and a
# round ceiling. "Unlimited" settings previously burned through paid
# subscriptions in minutes — non-positive values now fall back to safe
# defaults instead of disabling the guard.
DEFAULT_MAX_ROUNDS = 250  # full-completion budget; anti-loop gates still stop unproductive work
_FLOW_TIMEOUT_FALLBACK = 4 * 60 * 60.0  # 4 hours — enough wall time to use the round budget
_MAX_FLOW_TIMEOUT_SECONDS = 24 * 60 * 60.0
try:
    _flow_timeout_env = float(os.environ.get("NORAX_FLOW_TIMEOUT_SECONDS", "") or 0)
except ValueError:
    _flow_timeout_env = 0.0
DEFAULT_TIMEOUT_SECONDS = (
    min(_flow_timeout_env, _MAX_FLOW_TIMEOUT_SECONDS)
    if math.isfinite(_flow_timeout_env) and _flow_timeout_env > 0
    else _FLOW_TIMEOUT_FALLBACK
)
# Hard emergency ceiling on LLM rounds per turn. Cannot be disabled: a
# non-positive env value falls back to the default instead of removing the cap.
# Keep a non-disableable emergency ceiling at or above the normal per-turn budget.
_HARD_ROUND_CAP_FALLBACK = 250
try:
    _hard_cap_env = int(
        os.environ.get("NORAX_AGENT_HARD_ROUND_CAP", "") or _HARD_ROUND_CAP_FALLBACK
    )
except ValueError:
    _hard_cap_env = _HARD_ROUND_CAP_FALLBACK
HARD_ROUND_CAP = min(_hard_cap_env, 1_000) if _hard_cap_env > 0 else _HARD_ROUND_CAP_FALLBACK
# Best-of-N is disabled by default — it doubles inference time which is
# painful on local models. Enable explicitly for cloud providers if needed.
try:
    _max_output_env = int(os.environ.get("NORAX_AGENT_MAX_OUTPUT_TOKENS", "16384"))
except ValueError:
    _max_output_env = 16384
# Preserve a full 16K default reservation for capable turns while honoring an
# operator's provider-specific setting exactly. If a provider cannot fund that
# reservation, a resulting 402 remains visible and never triggers failover.
AGENT_MAX_OUTPUT_TOKENS = _max_output_env if _max_output_env > 0 else 16384
CONSECUTIVE_FAIL_LIMIT = 3  # after N rounds where ALL tools fail, inject recovery
FINAL_VERIFICATION_NUDGE_LIMIT = 3
NO_TOOL_STREAK_LIMIT = 4
MAX_NO_TOOL_ROUNDS_TOTAL = 6
_MAX_TOOL_CALLS_FALLBACK = 1_000
try:
    _max_tool_calls_env = int(
        os.environ.get("NORAX_AGENT_MAX_TOOL_CALLS", str(_MAX_TOOL_CALLS_FALLBACK))
    )
except ValueError:
    _max_tool_calls_env = _MAX_TOOL_CALLS_FALLBACK
MAX_TOOL_CALLS_PER_TURN = (
    _max_tool_calls_env if _max_tool_calls_env > 0 else _MAX_TOOL_CALLS_FALLBACK
)
NO_PROGRESS_ROUND_LIMIT = 4

# Per-tool wall-clock ceilings. Applied on top of whatever the tool
# enforces internally — a stuck tool should never wedge the whole turn.
# The tool can still succeed faster; this is just the upper bound.
TOOL_TIMEOUTS: dict[str, float] = {
    # Local reads normally complete quickly; the timeout remains a real bound.
    "read": 10.0,
    "list_dir": 10.0,
    "search_memory": 10.0,
    "memory_search": 10.0,
    # writes are local but may touch large files
    "write": 20.0,
    "write_chunk": 15.0,
    "edit": 20.0,
    # network
    # web_fetch: direct fetch (20s) + optional Firecrawl fallback (20s)
    "web_fetch": 60.0,
    "web_search": 30.0,
    # deep_research runs a batch of searches + fetches internally; give it room
    "deep_research": 300.0,
    # exec carries its own timeout kwarg (default 30s); cap at 10 minutes so a
    # shelled `sleep 99999` cannot wedge the turn indefinitely.
    "exec": 600.0,
    "shell": 600.0,
    # remote nodes (relay round-trip; default was 60s → false timeouts)
    "remote_exec": 330.0,
    "remote_read": 90.0,
    "remote_list": 90.0,
    "remote_write": 90.0,
    "remote_list_nodes": 15.0,
    "remote_enroll": 30.0,
    # catch-all
    "_default": 60.0,
}


def _redact(obj: Any, max_chars: int = 400) -> Any:
    """Best-effort short representation for event logs."""
    try:
        s = json.dumps(obj, default=str)[:max_chars]
    except Exception:  # noqa: BLE001
        s = repr(obj)[:max_chars]
    return s


def _current_turn_anchor(user_text: str, *, active_goal: str) -> str:
    """Make the latest request authoritative over replayed history and memory."""
    request = str(user_text or "").strip()
    if len(request) > 6_000:
        request = request[:6_000] + "\n...[current request display truncated]"
    goal = str(active_goal or request).strip()
    if len(goal) > 2_000:
        goal = goal[:2_000] + "..."
    return (
        "CURRENT_TURN_PRIORITY (highest operational priority)\n"
        "The current user request supersedes conflicting older history, retrieved memory, "
        "prior plans, and stale task state. Use older context only as supporting background.\n"
        "Do not resume an older task unless this request explicitly asks to continue it. "
        "Answer and act on the current situation first.\n"
        "Follow the user's requested final-response format after verification; "
        "examples and default recap guidance do not override it. "
        "If work is incomplete, report the remaining work honestly.\n"
        f"ACTIVE_GOAL: {goal}\n"
        f"CURRENT_USER_REQUEST:\n{request}"
    )


def _fingerprint(name: str, args: dict) -> str:
    """Stable hash for duplicate-call detection (full args, not truncated)."""
    import hashlib

    body = json.dumps({"t": name, "a": args}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _extract_first_text(resp: GatewayResponse) -> str:
    return resp.content or ""


def _strip_thinking(content: str) -> str:
    """Evaluate only answer content, never inline private reasoning."""
    return strip_reasoning_blocks(content)


def _mark_incomplete(response: GatewayResponse, reason: str) -> GatewayResponse:
    """Attach machine-readable terminal status to a forced/incomplete answer."""
    response.raw = dict(response.raw or {})
    response.raw["incomplete"] = True
    # Preserve the first terminal cause.  A timeout returned by a forced-final
    # generation must not later be mislabeled as merely "no progress" or
    # "goal drift", because operators use this field for continuation policy.
    response.raw.setdefault("limit_reason", reason)
    return response


def _summarize_tool_result(result: Any) -> dict[str, Any]:
    """Preserve RHO-critical result structure without storing huge outputs."""
    if not isinstance(result, dict):
        return {"ok": False, "error": "non_dict_result", "detail": _redact(result, 300)}
    out: dict[str, Any] = {"ok": _tool_result_ok(result)}
    for key in ("error", "detail", "reason", "exit_code", "status", "content_type"):
        if key in result and result.get(key) is not None:
            out[key] = result.get(key)
    if "stdout" in result:
        out["stdout_len"] = len(str(result.get("stdout") or ""))
    if "stderr" in result:
        out["stderr_len"] = len(str(result.get("stderr") or ""))
    if "path" in result:
        out["path"] = result.get("path")
    return out


def _tool_result_ok(result: object) -> bool:
    """Treat only the protocol's literal boolean true as tool success."""
    return isinstance(result, dict) and result.get("ok") is True


def _boolean_or_default(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _split_concatenated_json(s: str) -> list[str]:
    """Split a string that may contain multiple JSON objects concatenated
    together (e.g. `{"a":1}{"b":2}`). Returns a list of individual JSON
    snippets. Used to handle Gemma/ollama tool-call hallucinations where
    multiple tool calls get squashed into one `arguments` string.
    """
    dec = json.JSONDecoder()
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        # skip leading whitespace
        while i < n and s[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            _, end = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            break
        out.append(s[i:end])
        i = end
    return out


def _extract_xmlish_tool_calls(content: str) -> list[dict[str, Any]]:
    """Recover tool calls from weak models that leak pseudo XML instead of
    native OpenAI tool_calls.

    Minimax-M3 commonly emits blocks like:
      <tool_call><invoke name="exec"><command>ls</command>...</invoke></tool_call>
    or fenced `<invoke ...>` fragments. Treat those as executable tool calls
    instead of sending the raw markup to the user.
    """
    if not content or "<invoke" not in content:
        return []
    calls: list[dict[str, Any]] = []
    for m in re.finditer(
        r'<invoke\s+name=["\']([^"\']+)["\']\s*>(.*?)</invoke>', content, re.I | re.S
    ):
        name = tool_mod.normalize_tool_name(m.group(1).strip())
        body = m.group(2) or ""
        args: dict[str, Any] = {}
        # JSON payload form: <arguments>{...}</arguments>
        am = re.search(r"<arguments\s*>(.*?)</arguments\s*>", body, re.I | re.S)
        if am:
            raw = am.group(1).strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args.update(parsed)
            except Exception:
                args["_raw"] = raw
        # Tag-per-argument form: <command>...</command><cwd>...</cwd>
        for km in re.finditer(r"<([A-Za-z_][\w-]*)\s*>(.*?)</\1\s*>", body, re.S):
            key = km.group(1)
            if key.lower() in {"arguments", "invoke", "tool_call"}:
                continue
            val = re.sub(r"\A\s*<!\[CDATA\[(.*?)\]\]>\s*\Z", r"\1", km.group(2).strip(), flags=re.S)
            # Strip minimax channel sentinels if they got copied into content.
            val = val.replace("]<]minimax[>[", "").strip()
            if key in {"timeout", "limit", "offset", "k"}:
                try:
                    args[key] = int(float(val))
                    continue
                except ValueError:
                    pass
            args[key] = val
        args = tool_mod.normalize_tool_args(name, args)
        if name in {"exec", "remote_exec", "shell"} and not tool_mod.exec_args_valid(args):
            continue
        if name:
            calls.append({"id": f"xmlish_{len(calls)}", "name": name, "args": args})
    return calls


def _tool_calls_from(resp: GatewayResponse) -> list[dict[str, Any]]:
    """Normalize tool_calls from either OpenAI format (choices[0].message.tool_calls)
    which our GatewayResponse already surfaces as `tool_calls`."""
    tcs = resp.tool_calls or []
    out: list[dict[str, Any]] = []
    for tc in tcs:
        # OpenAI: {"id","type":"function","function":{"name","arguments":"json-string"}}
        tid = tc.get("id") or tc.get("tool_call_id") or ""
        fn = tc.get("function") or {}
        name = tool_mod.normalize_tool_name(fn.get("name") or tc.get("name") or "")
        raw_args = fn.get("arguments")
        if raw_args is None:
            raw_args = tc.get("arguments") or {}
        if isinstance(raw_args, str):
            stripped = raw_args.strip()
            # Try splitting in case the provider concatenated multiple
            # JSON objects into one `arguments` string (Gemma/ollama
            # hallucination — sometimes two intended tool calls get
            # squashed into one with a single tool name).
            chunks = _split_concatenated_json(stripped) if stripped else []
            if len(chunks) > 1:
                # Score each chunk against the tool's expected schema;
                # keep the best-matching one. This recovers from squashed
                # concatenation without blindly dispatching foreign args.
                spec = tool_mod.REGISTRY.get(name)
                expected = set((spec.schema or {}).keys()) if spec else set()
                best_args: dict | None = None
                best_score = -1
                for chunk in chunks:
                    try:
                        candidate = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(candidate, dict):
                        continue
                    # Score = matching keys minus foreign keys.
                    have = set(candidate.keys())
                    score = len(have & expected) - len(have - expected)
                    if score > best_score:
                        best_score = score
                        best_args = candidate
                if best_args is not None:
                    best_args = tool_mod.normalize_tool_args(name, best_args)
                    if name in {"exec", "remote_exec", "shell"} and not tool_mod.exec_args_valid(
                        best_args
                    ):
                        continue
                    out.append({"id": tid, "name": name, "args": best_args})
                    continue
            try:
                args = json.loads(stripped) if stripped else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
        else:
            args = raw_args or {}
        args = tool_mod.normalize_tool_args(name, args if isinstance(args, dict) else {})
        if name in {"exec", "remote_exec", "shell"} and not tool_mod.exec_args_valid(args):
            continue
        out.append({"id": tid, "name": name, "args": args})
    return out


async def _run_one_tool(
    name: str,
    args: dict,
    *,
    sender_tier: str,
    event_log: Any,
    mcp_client: Any = None,
    allowed_tools: Collection[str] | None = None,
) -> dict:
    """RiskGate + dispatch. Always returns a dict; never raises."""
    name = tool_mod.normalize_tool_name(name)
    args = tool_mod.normalize_tool_args(name, args)
    t0 = time.perf_counter()

    if allowed_tools is not None and name not in allowed_tools:
        result = {
            "ok": False,
            "error": "tool_not_allowed",
            "name": name,
            "_not_executed": True,
        }
        await event_log.append(
            "tool_call",
            {
                "name": name,
                "args": _redact(args),
                "ok": False,
                "dur_ms": int((time.perf_counter() - t0) * 1000),
                "err": "tool_not_allowed",
            },
        )
        return result

    decision = risk_mod.check(tool=name, args=args, sender_tier=sender_tier)
    if not decision.allowed:
        dur = (time.perf_counter() - t0) * 1000
        result = {
            "ok": False,
            "error": "risk_denied",
            "reason": decision.reason,
            "tier": decision.tier,
            "_not_executed": True,
        }
        await event_log.append(
            "tool_call",
            {
                "name": name,
                "args": _redact(args),
                "ok": False,
                "dur_ms": int(dur),
                "err": decision.reason,
                "risk_tier": decision.tier,
            },
        )
        return result

    # Interrupt check: an approval is one-shot and bound to this exact call.
    try:
        from ..runtime.interrupts import get_interrupt_manager

        im = get_interrupt_manager()
        resolution = im.consume_resolution(name, args)
        if resolution is not None:
            action = resolution.get("action")
            if action == "reject":
                dur = (time.perf_counter() - t0) * 1000
                result = {
                    "ok": False,
                    "error": "interrupt_rejected",
                    "tool": name,
                    "_not_executed": True,
                }
                await event_log.append(
                    "tool_call",
                    {
                        "name": name,
                        "args": _redact(args),
                        "ok": False,
                        "dur_ms": int(dur),
                        "err": "interrupt_rejected",
                    },
                )
                return result
            if action == "modify":
                args = tool_mod.normalize_tool_args(name, resolution.get("args") or {})
                decision = risk_mod.check(tool=name, args=args, sender_tier=sender_tier)
                if not decision.allowed:
                    dur = (time.perf_counter() - t0) * 1000
                    result = {
                        "ok": False,
                        "error": "risk_denied",
                        "reason": decision.reason,
                        "tier": decision.tier,
                        "_not_executed": True,
                    }
                    await event_log.append(
                        "tool_call",
                        {
                            "name": name,
                            "args": _redact(args),
                            "ok": False,
                            "dur_ms": int(dur),
                            "err": decision.reason,
                            "risk_tier": decision.tier,
                        },
                    )
                    return result
            log.info("interrupt.resolution_consumed tool=%s action=%s", name, action)
        else:
            should_pause, _reason, _desc = im.should_interrupt(name, args)
            if should_pause:
                pending = im.create_interrupt(name, args, _reason, _desc)
                dur = (time.perf_counter() - t0) * 1000
                result = {
                    "ok": False,
                    "error": "interrupted",
                    "reason": _reason.value,
                    "description": _desc,
                    "interrupt_id": pending.interrupt_id,
                    "approval_command": f"/approve {pending.interrupt_id}",
                    "tool": name,
                    "_not_executed": True,
                }
                await event_log.append(
                    "tool_call",
                    {
                        "name": name,
                        "args": _redact(args),
                        "ok": False,
                        "dur_ms": int(dur),
                        "err": "interrupted",
                        "interrupt_id": pending.interrupt_id,
                    },
                )
                return result
    except Exception as e:
        log.exception("agent_loop.interrupt_gate_failed name=%s", name)
        dur = (time.perf_counter() - t0) * 1000
        result = {
            "ok": False,
            "error": "interrupt_gate_error",
            "detail": str(e)[:500],
            "tool": name,
            "_not_executed": True,
        }
        await event_log.append(
            "tool_call",
            {
                "name": name,
                "args": _redact(args),
                "ok": False,
                "dur_ms": int(dur),
                "err": "interrupt_gate_error",
            },
        )
        return result

    # MCP calls are external actions too: they pass through the same identity
    # RiskGate and interrupt policy above. Unknown MCP tools classify as T2,
    # so only owners can execute them unless a future manifest supplies a
    # narrower, explicit tier.
    if name.startswith("mcp_"):
        if mcp_client is None:
            result = {
                "ok": False,
                "error": "mcp_not_connected",
                "name": name,
                "_not_executed": True,
            }
        else:
            full_name = name[4:]
            mcp_circuit_registry = get_circuit_registry()
            cb_allowed, cb_reason = mcp_circuit_registry.check(name)
            if not cb_allowed:
                result = {
                    "ok": False,
                    "error": "circuit_open",
                    "reason": cb_reason,
                    "name": name,
                    "_not_executed": True,
                }
            else:
                try:
                    result = await asyncio.wait_for(
                        mcp_client.call_external_tool(full_name, args),
                        timeout=120,
                    )
                except asyncio.CancelledError:
                    mcp_circuit_registry.abandon_probe(name)
                    raise
                except TimeoutError:
                    result = {"ok": False, "error": "timeout", "name": name}
                    mcp_circuit_registry.record_failure(name, "timeout")
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "ok": False,
                        "error": "mcp_tool_error",
                        "detail": str(exc)[:2000],
                        "name": name,
                    }
                    mcp_circuit_registry.record_failure(name, "mcp_tool_error")
                else:
                    # The transport is healthy even when the remote tool
                    # returns an ordinary application-level error.
                    mcp_circuit_registry.record_success(name)
        if not isinstance(result, dict):
            result = {"ok": True, "result": result}
        dur = (time.perf_counter() - t0) * 1000
        await event_log.append(
            "tool_call",
            {
                "name": name,
                "args": _redact(args),
                "ok": _tool_result_ok(result),
                "dur_ms": int(dur),
                "err": result.get("error"),
                "mcp": True,
                "risk_tier": decision.tier,
            },
        )
        return result

    spec = tool_mod.REGISTRY.get(name)
    if spec is None:
        await event_log.append(
            "tool_call",
            {"name": name, "args": _redact(args), "ok": False, "err": "unknown_tool"},
        )
        return {
            "ok": False,
            "error": "unknown_tool",
            "name": name,
            "_not_executed": True,
        }

    if name in {"exec", "remote_exec", "shell"} and not tool_mod.exec_args_valid(args):
        cmd = args.get("command")
        if isinstance(cmd, str) and cmd.strip():
            detail = f"exec command is a placeholder or invalid: {cmd!r}"
        else:
            detail = "t_exec() missing 1 required keyword-only argument: 'command'"
        result = {
            "ok": False,
            "error": "bad_arguments",
            "detail": detail,
            "_not_executed": True,
            "hint": (
                'Emit {"name":"shell","arguments":{"command":"echo ok"}} '
                "with a real shell command — Norax has no Cursor Shell tool."
            ),
        }
        dur = (time.perf_counter() - t0) * 1000
        await event_log.append(
            "tool_call",
            {
                "name": name,
                "args": _redact(args),
                "ok": False,
                "dur_ms": int(dur),
                "err": "bad_arguments",
            },
        )
        return result

    cb_registry = None
    try:
        # Pass only the args the function actually accepts; drop extras
        # so a hallucinated kwarg doesn't crash us.
        sig = inspect.signature(spec.fn)
        accepted = {
            k: v
            for k, v in args.items()
            if k in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        }
        missing = [
            p.name
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.name not in accepted
        ]
        if missing:
            hint = ""
            if name in {"exec", "remote_exec", "shell"} and "command" in missing:
                hint = (
                    ' Emit {"name":"shell","arguments":{"command":"echo ok"}} or '
                    '{"name":"exec","arguments":{"command":"echo ok"}} with a real shell command'
                    " — Norax has no Cursor Shell tool."
                )
            elif name.startswith("remote_") and "node_id" in missing:
                hint = (
                    ' Emit {"name":"remote_list","arguments":{"node_id":"staging","path":"Desktop"}}'
                    " — use node_id, not node."
                )
            result = {
                "ok": False,
                "error": "bad_arguments",
                "detail": f"missing required: {', '.join(missing)}",
                "hint": hint.strip() or None,
                "_not_executed": True,
            }
            dur = (time.perf_counter() - t0) * 1000
            await event_log.append(
                "tool_call",
                {
                    "name": name,
                    "args": _redact(args),
                    "ok": False,
                    "dur_ms": int(dur),
                    "err": "bad_arguments",
                },
            )
            return result
        # Circuit breaker: check if tool is in open state
        cb_registry = get_circuit_registry()
        cb_allowed, cb_reason = cb_registry.check(name)
        if not cb_allowed:
            dur = (time.perf_counter() - t0) * 1000
            result = {
                "ok": False,
                "error": "circuit_open",
                "reason": cb_reason,
                "note": "Tool circuit breaker is open due to repeated failures. Will retry after cooldown.",
                "_not_executed": True,
            }
            await event_log.append(
                "tool_call",
                {
                    "name": name,
                    "args": _redact(args),
                    "ok": False,
                    "dur_ms": int(dur),
                    "err": "circuit_open",
                    "circuit_reason": cb_reason,
                },
            )
            return result

        # Wall-clock ceiling per tool so a hung tool doesn't wedge the turn.
        tool_budget = TOOL_TIMEOUTS.get(name, TOOL_TIMEOUTS["_default"])
        try:
            result = await asyncio.wait_for(spec.fn(**accepted), timeout=tool_budget)
        except TimeoutError:
            result = {
                "ok": False,
                "error": "tool_timeout",
                "timeout_s": tool_budget,
                "note": "tool exceeded harness wall-clock budget",
            }
            cb_registry.record_failure(name, "timeout")
        else:
            # A shell command can complete normally while returning a non-zero
            # exit status (for example, a probe finding a stopped service).
            # That is useful task evidence, not an outage of the exec tool.
            # Only infrastructure failures such as timeouts should contribute
            # to the tool circuit breaker.
            command_completed = (
                name in {"exec", "shell"} and isinstance(result, dict) and "exit_code" in result
            )
            if isinstance(result, dict) and (
                _tool_result_ok(result)
                or command_completed
                or not _is_infrastructure_tool_failure(result)
            ):
                cb_registry.record_success(name)
            else:
                err = str(result.get("error", "unknown") if isinstance(result, dict) else "unknown")
                cb_registry.record_failure(name, err)
        if not isinstance(result, dict):
            result = {"ok": True, "result": result}
    except asyncio.CancelledError:
        if cb_registry is not None:
            cb_registry.abandon_probe(name)
        raise
    except TypeError as e:
        result = {"ok": False, "error": "bad_arguments", "detail": str(e)}
        if cb_registry is not None:
            cb_registry.record_success(name)
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s raised", name)
        result = {"ok": False, "error": "tool_exception", "detail": repr(e)}
        if cb_registry is not None:
            cb_registry.record_failure(name, "tool_exception")

    dur = (time.perf_counter() - t0) * 1000
    await event_log.append(
        "tool_call",
        {
            "name": name,
            "args": _redact(args),
            "ok": _tool_result_ok(result),
            "dur_ms": int(dur),
            "err": result.get("error"),
            "risk_tier": decision.tier,
        },
    )
    return result


_INFRASTRUCTURE_TOOL_ERRORS = frozenset(
    {
        "connection_error",
        "connection_refused",
        "service_unavailable",
        "tool_exception",
        "tool_timeout",
        "transport_error",
    }
)


def _is_infrastructure_tool_failure(result: dict[str, Any]) -> bool:
    """Separate tool availability failures from ordinary negative results."""
    error = str(result.get("error") or "").strip().lower()
    if error in _INFRASTRUCTURE_TOOL_ERRORS:
        return True
    return any(
        marker in error
        for marker in ("connection refused", "service unavailable", "transport failure")
    )


def _call_can_run_in_parallel(call: dict[str, Any]) -> bool:
    name = str(call.get("name") or "")
    if name in {
        "read",
        "list_dir",
        "search_memory",
        "memory_search",
        "web_search",
        "web_fetch",
        "status",
    }:
        return True
    if name not in {"exec", "shell"}:
        return False
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    return not tool_call_is_mutating(name, args)


def _evict_middle_tool_traces(
    messages: list[dict],
    *,
    max_messages: int = 60,
    keep_recent_rounds: int = 8,
    task_state: TaskState | None = None,
) -> tuple[list[dict], int] | None:
    """Evict old tool_call/tool_result pairs from the middle of the messages list.

    The agent_loop builds its own `messages` list that grows unbounded during
    long tool loops. After 20+ rounds, the model processes 60K+ tokens of stale
    tool results every single round. This function trims old tool traces while
    preserving:
      - The system prompt (first message)
      - The original user prompt (second message)
      - The last `keep_recent_rounds` rounds of tool_call/tool_result pairs
      - Any non-tool messages (recovery nudges, goal drift) in the kept section

    Evicted tool pairs are replaced with a compact summary marker so the model
    knows earlier work happened without seeing the full results.

    Returns (new_messages, evicted_count) if eviction occurred, None otherwise.
    """
    if len(messages) <= max_messages:
        return None

    # Identify tool_call/tool_result pairs by tool_call_id
    tool_call_ids = set()
    tool_result_ids = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id", "")
                if cid:
                    tool_call_ids.add(cid)
        elif m.get("role") == "tool":
            rid = m.get("tool_call_id", "")
            if rid:
                tool_result_ids.add(rid)

    # Only evict if we have tool pairs to work with
    if not tool_call_ids:
        return None

    # Find the indices of the last `keep_recent_rounds` distinct tool_call_ids
    seen_recent: list[str] = []
    for m in reversed(messages):
        if m.get("role") == "tool":
            rid = m.get("tool_call_id", "")
            if rid and rid not in seen_recent:
                seen_recent.append(rid)
            if len(seen_recent) >= keep_recent_rounds:
                break
    protected_ids = set(seen_recent)

    # Split messages into: head (system+user), evictable middle, protected tail
    # Head = everything before the first tool_call
    head: list[dict] = []
    first_tool_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            first_tool_idx = i
            break
        head.append(m)

    if first_tool_idx is None:
        return None  # no tool calls at all

    # Partition the rest into evictable vs protected
    evictable: list[dict] = []
    protected_tail: list[dict] = []
    for m in messages[first_tool_idx:]:
        # A message is protected if it's a tool_result with a protected id,
        # or an assistant message whose tool_calls are all protected,
        # or a non-tool user/system message in the protected zone.
        is_protected = False
        if m.get("role") == "tool":
            if m.get("tool_call_id") in protected_ids:
                is_protected = True
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = m.get("tool_calls") or []
            if all(tc.get("id") in protected_ids for tc in tcs):
                is_protected = True
        else:
            # Non-tool messages (recovery nudges, etc) — protect if they
            # appear after the first protected tool result
            is_protected = bool(protected_tail)  # once we're in protected zone, stay

        if is_protected:
            protected_tail.append(m)
        else:
            evictable.append(m)

    if not evictable:
        return None

    evicted_count = len(evictable)

    # Build summary marker — include file summaries so the model has what it needs
    evicted_rounds = sum(
        1 for m in evictable if m.get("role") == "assistant" and m.get("tool_calls")
    )

    # Extract file summaries from evicted tool results before discarding them
    _evicted_file_summaries: dict[str, str] = {}
    for m in evictable:
        if m.get("role") == "tool":
            _content = m.get("content", "")
            try:
                _parsed = json.loads(_content) if isinstance(_content, str) else _content
                if isinstance(_parsed, dict) and _tool_result_ok(_parsed) and _parsed.get("path"):
                    _path = str(_parsed["path"])
                    _total = _parsed.get("total_lines")
                    _content_str = _parsed.get("content") or ""
                    _parts = []
                    if _total is not None:
                        _parts.append(f"lines={_total}")
                    _symbols = []
                    for line in (_content_str or "").splitlines():
                        _s = line.strip()
                        if _s.startswith(
                            (
                                "class ",
                                "def ",
                                "async def ",
                                "import ",
                                "from ",
                                "export ",
                                "function ",
                                "const ",
                                "interface ",
                                "type ",
                            )
                        ):
                            _symbols.append(_s[:80])
                        if len(_symbols) >= 12:
                            break
                    if _symbols:
                        _parts.append("symbols=" + "|".join(_symbols[:8]))
                    _head = [
                        ln.strip()[:80] for ln in (_content_str or "").splitlines() if ln.strip()
                    ][:5]
                    if _head:
                        _parts.append("head=" + " › ".join(_head))
                    if _parts:
                        _evicted_file_summaries[_path] = "; ".join(_parts)
            except (json.JSONDecodeError, TypeError):
                pass

    _files_msg = ""
    if _evicted_file_summaries:
        _files_lines = [f"  {p}: {s}" for p, s in list(_evicted_file_summaries.items())[:15]]
        _files_msg = "\nFILES_ALREADY_READ (use these summaries, do NOT re-read):\n" + "\n".join(
            _files_lines
        )

    _state_msg = f"\n\n{task_state.context_block()}" if task_state is not None else ""
    summary = (
        f"EVICTED_CONTEXT: {evicted_rounds} earlier tool rounds were executed "
        f"and their results processed.{_files_msg}{_state_msg}\n\n"
        f"TASK_STATE is compact prior evidence, not a substitute for current state. "
        f"Avoid exact redundant mutations. Use completed_actions and file summaries "
        f"when they still answer the question; if new evidence requires inspection, "
        f"read the narrowest relevant current range. Move to the next unresolved step."
    )

    new_messages = head + [{"role": "user", "content": summary}] + protected_tail
    return new_messages, evicted_count


async def _emit_agent_trajectory(
    *,
    event_log: Any,
    task_state: TaskState,
    trace: list[dict],
    final_resp: GatewayResponse,
    rounds: int,
    model: str,
) -> None:
    """Persist a completed turn as an auditable harness trajectory.

    Tool calls are already individually audited; this aggregate record makes
    retrospective harness optimization cheap and auditable.

    Outcome is computed at emit time from observable signals: errors,
    writes, verification, final length, tool activity.  This explicit label
    feeds the internal harness-evidence report directly.
    """
    try:
        mutation_outcome = summarize_mutation_trace(trace)
        raw = final_resp.raw or {}
        verified_outcome = raw.get("verified_outcome") is True
        delivery_succeeded = _boolean_or_default(raw.get("delivery_succeeded"), True)
        accepted_outcome = delivery_succeeded and _boolean_or_default(
            raw.get("accepted_outcome"),
            bool((final_resp.content or "").strip())
            and raw.get("incomplete") is not True
            and (mutation_outcome.supports_success_claim if mutation_outcome.attempted else True),
        )
        objective_outcome_observed = _boolean_or_default(
            raw.get("objective_outcome_observed"),
            bool(mutation_outcome.attempted)
            or any(
                isinstance(item.get("result"), dict)
                and item["result"].get("_not_executed") is not True
                for item in trace
            ),
        )
        # --- Compute outcome from the same signals score_outcome() uses ---
        _errors: list[str] = []
        _writes = mutation_outcome.succeeded
        _verified = (
            mutation_outcome.verified_after_last_mutation
            if mutation_outcome.attempted
            else verified_outcome
        )
        for item in trace:
            result = item.get("result") or {}
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {}
            if (
                isinstance(result, dict)
                and not _tool_result_ok(result)
                and tool_failure_is_blocking(str(item.get("name") or ""), result)
            ):
                _errors.append(str(result.get("error") or result.get("detail") or "failed"))
        if not delivery_succeeded:
            _errors.append("response_delivery_failed")
        _failures = len(_errors)
        _tool_calls = len(trace)
        _final_len = len(final_resp.content or "")

        # score_outcome() formula (keep in sync with harness_optimizer.py)
        _score = 0.55
        _score += 0.18 if _tool_calls == 0 and _final_len >= 20 else 0.0
        _score += 0.18 if _tool_calls > 0 and not _errors else 0.0
        _score += 0.22 if _writes > 0 and _verified else 0.0
        _score -= min(_failures, 8) * 0.10
        _score -= 0.22 if _errors else 0.0
        _score -= 0.25 if _writes > 0 and not _verified else 0.0
        _score -= 0.10 if _final_len < 20 and task_state.context_block() else 0.0
        _score = max(0.0, min(_score, 1.0))
        # A fluent final answer must not turn a failed or unverified objective
        # outcome into a positive training label. Conversely, ordinary prose
        # with no objective check is unknown evidence, not a training failure.
        interrupted = raw.get("stopped") is True
        terminal_limit_observed = raw.get("flow_timeout") is True or bool(
            str(raw.get("limit_reason") or "").strip()
        )
        training_eligible = not interrupted and (
            not delivery_succeeded or objective_outcome_observed or terminal_limit_observed
        )
        if not delivery_succeeded:
            _score = min(_score, 0.05)
        if training_eligible and not verified_outcome:
            _score = min(_score, 0.30)
        if interrupted:
            _label = "interrupted"
        elif not training_eligible:
            _label = "accepted_unverified" if accepted_outcome else "unassessed"
        else:
            _label = "success" if _score > 0.50 else "partial" if _score > 0.30 else "failure"

        await event_log.append(
            "agent_trajectory",
            {
                "model": model,
                "rounds": rounds,
                "tool_calls": _tool_calls,
                "final_len": _final_len,
                "outcome": _label,
                "outcome_score": round(_score, 4),
                "accepted_outcome": accepted_outcome,
                "verified_outcome": verified_outcome,
                "delivery_succeeded": delivery_succeeded,
                "end_to_end_verified": verified_outcome and delivery_succeeded,
                "objective_outcome_observed": objective_outcome_observed,
                "training_eligible": training_eligible,
                "mutation_outcome": {
                    "attempted": mutation_outcome.attempted,
                    "succeeded": mutation_outcome.succeeded,
                    "failed": mutation_outcome.failed,
                    "verified_after_last_mutation": (mutation_outcome.verified_after_last_mutation),
                },
                "task_state": json.loads(_parse_task_state_json(task_state.context_block())),
                "trace": [
                    {
                        "name": item.get("name"),
                        "args": _redact(item.get("args") or {}, max_chars=800),
                        "result": _summarize_tool_result(item.get("result") or {}),
                    }
                    for item in trace
                ],
            },
        )
    except Exception as e:  # noqa: BLE001
        log.warning("agent_trajectory.emit_failed: %r", e)


async def record_agent_outcome(
    *,
    event_log: Any,
    task_state: TaskState,
    trace: list[dict],
    final_resp: GatewayResponse,
    rounds: int,
    model: str,
    user_text: str,
    memory_root: Path | None = None,
    tool_experience: ToolExperienceMemory | None = None,
    self_model: SelfModel | None = None,
    active_inference: ActiveInference | None = None,
    metacognitive: MetacognitiveCalibration | None = None,
    verifier_score: float | None = None,
) -> None:
    """Record post-turn evidence without participating in response acceptance.

    Production invokes this after a bounded delivery attempt. Standalone
    callers retain the stronger default in :func:`run_agent_loop`, which
    awaits this recorder before returning unless explicitly asked to defer it.
    """
    verified_outcome = (final_resp.raw or {}).get("verified_outcome") is True
    delivery_succeeded = _boolean_or_default(
        (final_resp.raw or {}).get("delivery_succeeded"),
        True,
    )
    end_to_end_verified = verified_outcome and delivery_succeeded
    objective_outcome_observed = (final_resp.raw or {}).get("objective_outcome_observed") is True

    if tool_experience is None and memory_root is not None:
        try:
            tool_experience = ToolExperienceMemory(memory_root)
        except Exception as error:  # noqa: BLE001
            log.warning("tool_experience.init_for_record_failed: %r", error)

    # Persist wins and losses, but retrieval promotes only trajectories whose
    # outcome was independently verified. This prevents learning from confident
    # prose or a mutation that was never read back/tested.
    if tool_experience is not None and trace:
        try:
            tool_experience.record_outcome(user_text, trace, verified=verified_outcome)
        except Exception as error:  # noqa: BLE001
            log.warning("tool_experience.record_failed: %r", error)

    # Cached and explicitly rejected calls are not fresh observations. Feeding
    # them into capability models would make retry suppression look like tool
    # failure and make cache hits look like independent success evidence.
    observed_trace = [
        entry
        for entry in trace
        if isinstance(entry.get("result"), dict)
        and entry["result"].get("_not_executed") is not True
        and entry["result"].get("_cached") is not True
    ]

    if metacognitive is not None and final_resp.content and objective_outcome_observed:
        try:
            tool_success_rate = (
                sum(1 for entry in observed_trace if _tool_result_ok(entry["result"]))
                / max(1, len(observed_trace))
                if observed_trace
                else 0.7
            )
            confidence = verifier_score if verifier_score is not None else tool_success_rate
            calibrated = metacognitive.calibrate_confidence(confidence)
            metacognitive.record_prediction(
                confidence=calibrated,
                actual_success=end_to_end_verified,
                tool="agent_loop",
                domain=task_state.task_type,
            )
            metacognitive.save()
            await event_log.append(
                "metacognitive",
                {
                    "raw_confidence": round(confidence, 3),
                    "calibrated_confidence": round(calibrated, 3),
                    "actual_success": end_to_end_verified,
                },
            )
        except Exception as error:  # noqa: BLE001
            log.warning("metacognitive.error: %r", error)

    if self_model is not None and observed_trace:
        try:
            domain = classify_domain(user_text, observed_trace)
            self_model.record_turn_outcomes(
                [
                    (
                        str(entry.get("name") or ""),
                        _tool_result_ok(entry["result"]),
                        float(entry["result"].get("dur_ms", 0) or 0),
                    )
                    for entry in observed_trace
                ],
                domain=domain,
                task_type=task_state.task_type,
                turn_success=end_to_end_verified,
            )
            self_model.save()
        except Exception as error:  # noqa: BLE001
            log.warning("self_model.error: %r", error)

    if active_inference is not None and observed_trace:
        try:
            domain = classify_domain(user_text, observed_trace)
            for entry in observed_trace:
                result = entry["result"]
                tool_name = str(entry.get("name") or "")
                prediction = active_inference.predict_tool_outcome(tool_name, domain)
                active_inference.record_observation(
                    tool=tool_name,
                    domain=domain,
                    predicted_success=prediction.expected_success,
                    actual_success=_tool_result_ok(result),
                    predicted_duration_ms=prediction.expected_duration_ms,
                    actual_duration_ms=float(result.get("dur_ms", 0) or 0),
                )
            active_inference.save()
        except Exception as error:  # noqa: BLE001
            log.warning("active_inference.error: %r", error)

    await _emit_agent_trajectory(
        event_log=event_log,
        task_state=task_state,
        trace=trace,
        final_resp=final_resp,
        rounds=rounds,
        model=model,
    )


async def run_agent_loop(
    *,
    gateway: GatewayClient | GatewayRouter,
    model: str,
    system_prompt: str,
    user_prompt: str,
    allowed_tools: list[str],
    sender_tier: str,
    event_log: Any,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    on_delta: Any = None,
    on_progress: Any = None,
    prior_messages: list[dict] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    # --- Sprint A additions ---
    correction_gate: Any = None,  # CorrectionGate for post-output checking
    stop_check: Any = None,  # callable() -> bool; if True, abort turn
    reasoning_effort: str = "medium",  # off|low|medium|high|xhigh
    reasoning_output: bool = False,  # ask gateway/provider to expose reasoning text when supported
    weak_model_boost: str = "auto",  # auto|on|off — tool-call scaffolding for weak models
    response_length: str = "balanced",  # concise|balanced|detailed
    tool_activity: str = "normal",  # minimal|normal|verbose
    # --- Cognitive amplifiers ---
    output_verifier: OutputVerifier | None = None,
    self_model: SelfModel | None = None,
    active_inference: ActiveInference | None = None,
    metacognitive: MetacognitiveCalibration | None = None,
    best_of_n: BestOfN | None = None,
    failover_models: list[str] | None = None,
    mcp_client: Any = None,
    memory_root: Path | None = None,
    initial_task_state: dict[str, Any] | None = None,
    initial_task_status: str = "in_progress",
    defer_outcome_recording: bool = False,
) -> tuple[GatewayResponse, list[dict], int, TaskState | None]:
    """Run the agent loop.

    Returns `(final_response, trace, rounds, task_state)` where:
        final_response: the GatewayResponse whose `content` is the final text
        trace:          list of `{"name","args","result"}` records for every
                        tool invocation in order
        rounds:         number of agent reasoning rounds completed (>= 1)
        task_state:     the final TaskState (for checkpoint persistence), or None

    `on_delta` (optional) is an async callable invoked with each text
    delta from the streaming LLM response. Streaming is used opportun-
    istically: intermediate rounds (those that produce tool_calls) do
    not stream because we can't render mid-thought tool plans usefully.
    The first round that produces only text is streamed; if streaming
    fails, we transparently fall back to a non-streaming `chat()` call.
    """
    # SAFETY: "unlimited" (<=0) settings clamp to safe defaults. The hard
    # round cap and wall-clock deadline below are always active.
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds <= 0:
        max_rounds = DEFAULT_MAX_ROUNDS
    max_rounds = min(max_rounds, HARD_ROUND_CAP)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    timeout_seconds = min(float(timeout_seconds), _MAX_FLOW_TIMEOUT_SECONDS)
    # Start the turn budget before any remote discovery or optional setup.
    # Otherwise a stalled MCP server can consume unbounded time before the
    # supposedly absolute flow deadline even exists.
    t0 = time.monotonic()
    deadline = t0 + timeout_seconds

    # Gather MCP tool schemas if client is connected
    _mcp_tools_list: list[dict] = []
    if mcp_client is not None and any(name.startswith("mcp_") for name in allowed_tools):
        try:
            async with asyncio.timeout_at(deadline):
                _mcp_tools_list = await mcp_client.aggregate_tools()
        except TimeoutError:
            log.warning("mcp.aggregate_tools_deadline_exceeded")
        except Exception as e:
            log.warning("mcp.aggregate_tools_failed: %r", e)
    full_tools_schema = tool_mod.render_tools_for_llm(
        allowed_tools,
        mcp_tools=_mcp_tools_list,
    )
    allowed_tool_set = frozenset(tool_mod.normalize_tool_name(name) for name in allowed_tools)
    base_system_prompt = system_prompt
    user_text = user_prompt if isinstance(user_prompt, str) else str(user_prompt)
    task_state = build_task_state(
        user_text, allowed_tools, restore_from=initial_task_state, prior_status=initial_task_status
    )

    # Dedicated just-in-time tool memory stays separate from broad conversation
    # retrieval. Curated and learned wins are surfaced before planning; current
    # observations always remain authoritative.
    tool_experience: ToolExperienceMemory | None = None
    if memory_root is not None:
        try:
            tool_experience = ToolExperienceMemory(memory_root)
            pre_tool_guidance = tool_experience.render(
                user_text, allowed_tools, phase="pre", limit=4, max_chars=2600
            )
            if pre_tool_guidance:
                base_system_prompt = f"{base_system_prompt}\n\n{pre_tool_guidance}"
        except Exception as e:  # noqa: BLE001 -- memory must never block a turn
            log.warning("tool_experience.pre_retrieval_failed: %r", e)
            tool_experience = None

    _length_hints = {
        "concise": "LENGTH: Keep replies short — 1-2 paragraphs max unless the task requires detail.",
        "detailed": "LENGTH: Provide thorough answers with structure and context when helpful.",
    }
    _activity_hints = {
        "minimal": "TOOL_NARRATION: Call tools silently; no status narration between calls.",
        "verbose": "TOOL_NARRATION: Narrate each tool step clearly for user visibility.",
    }
    ux_hints = [
        _length_hints.get(response_length, ""),
        _activity_hints.get(tool_activity, ""),
    ]
    ux_block = "\n".join(h for h in ux_hints if h)

    def _provider_kind_for_model(target_model: str) -> str:
        resolver = getattr(gateway, "provider_kind_for_model", None)
        if callable(resolver):
            try:
                return str(resolver(target_model) or "unknown")
            except Exception:  # noqa: BLE001
                log.debug("gateway.provider_kind_for_model_failed", exc_info=True)
        return "unknown"

    def _harness_for_model(target_model: str) -> tuple[str, list[dict], dict, bool]:
        """Build the exact prompt/schema profile for one model.

        This intentionally runs again on failover. Reusing a Gemma request for
        Kimi/GLM leaked mandatory scratchpads, pruned schemas, and GBNF into
        strong cloud models.
        """
        boosters = compose_weak_model_boosters(
            task_type=task_state.task_type,
            model_id=target_model,
            tool_schemas=full_tools_schema,
            provider_kind=_provider_kind_for_model(target_model),
            force_boost=weak_model_boost,
        )
        is_weak = bool(boosters["is_weak"])
        target_tools = boosters["lean_schemas"] if is_weak else full_tools_schema
        system_parts = [base_system_prompt]
        if is_weak:
            system_parts.extend(
                [
                    build_scaffold_prompt(
                        task_state.task_type,
                        model=target_model,
                        user_prompt=user_text,
                    ),
                    boosters["system_additions"],
                    task_state.context_block(),
                ]
            )
        if ux_block:
            system_parts.append(ux_block)
        system_parts.append(_current_turn_anchor(user_text, active_goal=task_state.goal))

        metadata: dict[str, Any] = {}
        if reasoning_effort and reasoning_effort != "medium":
            metadata["reasoning_effort"] = reasoning_effort
        elif reasoning_effort == "medium":
            # llama.cpp servers ignore reasoning_effort; without an explicit
            # effort the payload builder leaves thinking at the model default
            # (xhigh), which burns the output budget on hidden reasoning.
            # Medium is the runtime default — pass it through so the payload
            # builder can map it to enable_thinking for llama.cpp endpoints.
            metadata["reasoning_effort"] = "medium"
        if reasoning_output:
            metadata["reasoning_output"] = True
        return (
            "\n\n".join(part for part in system_parts if part),
            target_tools,
            metadata,
            is_weak,
        )

    system_prompt, tools_schema, req_metadata, initial_model_is_weak = _harness_for_model(model)
    if initial_model_is_weak:
        log.info("weak_model_boost.active model=%s task=%s", model, task_state.task_type)

    # Per-turn tool result cache (dedup identical read-safe calls)
    turn_cache = TurnToolCache()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]
    # Inject prior conversation history (from rolling window).
    # Sanitize: ensure every tool_use has a matching tool_result and no duplicates.
    if prior_messages:
        # Pass 1: collect all declared call IDs and all result IDs
        _declared_ids: set[str] = set()
        _result_ids: set[str] = set()
        for _m in prior_messages:
            if _m.get("role") == "assistant":
                for _tc in _m.get("tool_calls") or []:
                    _declared_ids.add(_tc.get("id", ""))
            elif _m.get("role") == "tool":
                _result_ids.add(_m.get("tool_call_id", ""))
        # Only keep call IDs that have a matching result (and vice versa)
        _valid_ids = _declared_ids & _result_ids

        # Pass 2: rebuild messages, skipping any broken tool pairs
        _sanitized: list[dict] = []
        _seen_assistant_ids: set[str] = set()
        for _m in prior_messages:
            if _m.get("role") == "assistant" and (_m.get("tool_calls") or None):
                # Filter to only valid, non-duplicate call IDs
                _good_tcs = []
                for _tc in _m["tool_calls"]:
                    _cid = _tc.get("id", "")
                    if _cid in _valid_ids and _cid not in _seen_assistant_ids:
                        _seen_assistant_ids.add(_cid)
                        _good_tcs.append(_tc)
                if not _good_tcs:
                    continue  # drop entire assistant message if no valid calls remain
                _m = dict(_m)
                _m["tool_calls"] = _good_tcs
                _sanitized.append(_m)
            elif _m.get("role") == "tool":
                _ref = _m.get("tool_call_id", "")
                if _ref not in _valid_ids:
                    continue  # drop orphan result
                _sanitized.append(_m)
            else:
                _sanitized.append(_m)
        messages.extend(_sanitized)
    messages.append({"role": "user", "content": user_prompt})

    trace: list[dict] = []
    # Prevent a successful state-changing action from being applied twice.
    # Read calls are deliberately not globally deduplicated: after a mutation,
    # re-reading the same path is required verification evidence.
    successful_mutation_calls: set[str] = set()
    total_tool_calls = 0  # hard per-turn tool-call ceiling accounting
    consecutive_all_fail_rounds = 0  # progress-stall detector
    consecutive_no_tool_rounds = 0  # text-only response streak detector
    consecutive_no_progress_rounds = 0  # novel-progress stall detector
    loop_guard = LoopGuard(repeat_threshold=3, pingpong_min_cycles=3, window=12)
    goal_drift = GoalDriftMonitor(max_rounds=max_rounds, max_time_sec=timeout_seconds)
    goal_drift.set_task(user_prompt if isinstance(user_prompt, str) else str(user_prompt))

    final_resp: GatewayResponse | None = None
    rounds = 0
    turn_usage = {"input_tokens": 0, "output_tokens": 0}

    def _account_usage(usage: dict | None) -> None:
        """Accumulate every successful generation, not only the final round."""
        for key in ("input_tokens", "output_tokens"):
            try:
                value = int((usage or {}).get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                turn_usage[key] += value

    def _account_response(response: GatewayResponse) -> GatewayResponse:
        _account_usage(response.usage)
        return response

    final_verification_nudges = 0
    # Once a provider/model fails over successfully, keep using the healthy
    # model for the remainder of this turn. Long tasks make many model calls;
    # probing a known-rate-limited primary again on every round wastes quota
    # and can turn otherwise recoverable work into an abrupt failure.
    active_model = model
    unhealthy_models: set[str] = set()
    # A verifier cannot be a genuine pre-send gate if rejected draft tokens
    # have already reached the user. Keep provider streaming for transport
    # efficiency, but publish deltas only when no post-draft selector can
    # replace or reject the assembled answer.
    publish_stream_deltas = bool(
        on_delta is not None
        and correction_gate is None
        and output_verifier is None
        and best_of_n is None
    )

    async def _report_progress(status: str = "in_progress") -> None:
        if on_progress is None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            state_payload = json.loads(_parse_task_state_json(task_state.context_block()))
            value = on_progress(
                {
                    "status": status,
                    "rounds": rounds,
                    "model": active_model,
                    "messages": [dict(message) for message in messages],
                    "trace": [dict(item) for item in trace],
                    "task_state": state_payload,
                }
            )
            if inspect.isawaitable(value):
                # UI/checkpoint callbacks are advisory.  A stalled callback
                # must neither overrun the task deadline nor delay a completed
                # answer for more than a small bounded interval.
                async with asyncio.timeout(min(2.0, remaining)):
                    await value
        except TimeoutError:
            log.warning("agent.progress_callback_timed_out status=%s", status)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.progress_callback_failed: %r", exc)

    def _request_for_model(req: GatewayRequest, target_model: str) -> GatewayRequest:
        if req.model == target_model:
            return req
        target_system, target_tools, target_metadata, target_is_weak = _harness_for_model(
            target_model
        )
        target_messages = [dict(message) for message in req.messages]
        if target_messages and target_messages[0].get("role") == "system":
            target_messages[0]["content"] = target_system
        else:
            target_messages.insert(0, {"role": "system", "content": target_system})

        metadata = dict(req.metadata or {})
        metadata.pop("ollama_grammar", None)
        metadata.pop("ollama_use_grammar", None)
        metadata.update(target_metadata)
        if target_is_weak:
            log.info(
                "weak_model_boost.failover_profile model=%s task=%s",
                target_model,
                task_state.task_type,
            )
        else:
            log.info("strong_model_baseline.failover_profile model=%s", target_model)
        return GatewayRequest(
            model=target_model,
            messages=target_messages,
            tools=target_tools if req.tools is not None else None,
            tool_choice=req.tool_choice,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            metadata=metadata or None,
        )

    async def _chat_maybe_stream_inner(
        req: GatewayRequest, *, emit_deltas: bool = True
    ) -> GatewayResponse:
        """Try streaming if a delta callback is wired; on any failure
        fall back to a plain `chat()`. Returns a fully-assembled response.

        The gateway client owns same-provider retry/backoff. This layer owns
        cross-model failover and never stacks another retry storm on top.
        A failed partial stream is replaced by a complete fallback response
        instead of being misclassified as a finished answer.
        """
        nonlocal active_model
        req = _request_for_model(req, active_model)

        def _answer_response(response: GatewayResponse) -> GatewayResponse:
            stop_reason = str(
                response.metadata.get("finish_reason") or response.metadata.get("done_reason") or ""
            ).lower()
            if stop_reason in {"length", "max_tokens", "max_output_tokens", "token_limit"}:
                raise GatewayUpstreamError(
                    502,
                    f"upstream stopped before completing the answer ({stop_reason})",
                    response.model or req.model,
                )
            if reasoning_output:
                return response
            response.content = _strip_thinking(response.content or "")
            if not response.content and not response.tool_calls:
                raise GatewayUpstreamError(
                    502,
                    "upstream returned only hidden reasoning and no answer",
                    response.model or req.model,
                )
            return response

        if on_delta is None or not emit_deltas:
            last_exc: Exception | None = None
            try:
                return _answer_response(_account_response(await gateway.chat(req)))
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if isinstance(e, SpendGuardTripped):
                    raise
                if _is_model_failover_error(e):
                    unhealthy_models.add(req.model)
                    log.warning(
                        "agent.model_failover_required model=%s status=%s",
                        req.model,
                        getattr(e, "status", "transport"),
                    )
                else:
                    raise
            # Try failover models before giving up.
            fo_exc: Exception | None = last_exc
            for fo_model in failover_models or []:
                if fo_model == req.model or fo_model in unhealthy_models:
                    continue
                if not _is_model_failover_error(fo_exc):
                    continue
                log.warning(
                    "agent.failover from=%s to=%s status=%s",
                    req.model,
                    fo_model,
                    getattr(fo_exc, "status", "transport"),
                )
                fo_req = _request_for_model(req, fo_model)
                try:
                    response = _account_response(await gateway.chat(fo_req))
                    active_model = fo_model
                    log.warning("agent.failover_sticky model=%s", fo_model)
                    return _answer_response(response)
                except Exception as e:  # noqa: BLE001
                    fo_exc = e
                    if _is_model_failover_error(e):
                        unhealthy_models.add(fo_model)
                    log.warning("agent.failover_failed model=%s err=%s", fo_model, e)
            assert fo_exc is not None
            raise fo_exc

        last_exc = None
        collected: list[str] = []
        # A generation stream is not idempotent. Make exactly one streaming
        # attempt, then use a configured model failover or one complete
        # non-stream recovery; never replay the same stream blindly.
        for _attempt in range(1):
            try:
                assembled: GatewayResponse | None = None
                stream_filter = ReasoningTagFilter(expose=reasoning_output)
                async for evt in gateway.chat_stream(req):
                    if evt.kind == "delta" and evt.text:
                        delta_text = stream_filter.feed(evt.text)
                        if delta_text:
                            collected.append(delta_text)
                            if publish_stream_deltas:
                                try:
                                    await on_delta(delta_text)
                                except Exception:  # noqa: BLE001
                                    log.debug("agent.on_delta_failed", exc_info=True)
                    elif evt.kind == "final":
                        assembled = evt.response
                tail = stream_filter.finish()
                if tail:
                    collected.append(tail)
                    if publish_stream_deltas:
                        try:
                            await on_delta(tail)
                        except Exception:  # noqa: BLE001
                            log.debug("agent.on_delta_failed", exc_info=True)
                if assembled is not None:
                    return _answer_response(_account_response(assembled))
                # A stream without a final envelope is a protocol failure.
                # Never label accumulated partial text as a completed answer.
                if collected:
                    log.warning(
                        "agent.stream_no_final; recovering_complete_response deltas=%d",
                        len(collected),
                    )
                    last_exc = GatewayUpstreamError(
                        502,
                        "stream ended without a final response envelope",
                        req.model,
                    )
                    unhealthy_models.add(req.model)
                    break
                # Empty successful streams are provider failures, not a reason
                # to replay the same potentially billed generation repeatedly.
                raise GatewayUpstreamError(502, "empty_stream", req.model)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if isinstance(e, SpendGuardTripped):
                    raise
                if _is_model_failover_error(e):
                    unhealthy_models.add(req.model)
                    log.warning(
                        "agent.stream_failover_required model=%s status=%s partial_chars=%d",
                        req.model,
                        getattr(e, "status", "transport"),
                        sum(len(c) for c in collected),
                    )
                    break
                # Other 400s are permanent — don't waste more attempts.
                if isinstance(e, GatewayUpstreamError) and 400 <= e.status < 500:
                    raise
                # Unknown local/protocol defects are not evidence that another
                # paid generation will help. Surface them without amplification.
                raise
        # Recover a complete response through a healthy fallback model.
        fo_exc = last_exc
        attempted_failover = False
        for fo_model in failover_models or []:
            if fo_model == req.model or fo_model in unhealthy_models:
                continue
            if not _is_model_failover_error(fo_exc):
                continue
            log.warning(
                "agent.failover from=%s to=%s status=%s partial_chars=%d",
                req.model,
                fo_model,
                getattr(fo_exc, "status", "transport"),
                sum(len(c) for c in collected),
            )
            fo_req = _request_for_model(req, fo_model)
            attempted_failover = True
            try:
                response = _account_response(await gateway.chat(fo_req))
                active_model = fo_model
                log.warning("agent.failover_sticky model=%s", fo_model)
                return _answer_response(response)
            except Exception as e:  # noqa: BLE001
                fo_exc = e
                if _is_model_failover_error(e):
                    unhealthy_models.add(fo_model)
                log.warning("agent.failover_failed model=%s err=%s", fo_model, e)
        # Last-ditch non-stream fallback on the current model.
        log.exception(
            "agent.stream_failed_all_retries; final non-stream fallback", exc_info=last_exc
        )
        # If we actually tried a failover model and it also failed, re-raise
        # that failure. But when no failover models are configured (or none
        # were eligible), fo_exc == last_exc — don't skip the non-stream
        # attempt just because the error is "failover-eligible".
        if attempted_failover and _is_model_failover_error(fo_exc):
            assert fo_exc is not None
            raise fo_exc
        try:
            return _answer_response(_account_response(await gateway.chat(req)))
        except Exception:
            assert fo_exc is not None
            raise fo_exc from None

    deadline_event_emitted = False

    async def _flow_deadline_response() -> GatewayResponse:
        nonlocal deadline_event_emitted
        elapsed = time.monotonic() - t0
        log.warning(
            "agent.deadline_exceeded_stop rounds=%d elapsed=%.0fs limit=%.0fs",
            rounds,
            elapsed,
            timeout_seconds,
        )
        if not deadline_event_emitted:
            await event_log.append(
                "flow_timeout",
                {
                    "rounds": rounds,
                    "elapsed_seconds": int(elapsed),
                    "limit_seconds": int(timeout_seconds),
                    "tool_calls": len(trace),
                    "policy": "stop_and_alert_user",
                },
            )
            deadline_event_emitted = True
        return GatewayResponse(
            request_id="",
            model=model,
            content=(
                f"⏱️ Configured {timeout_seconds:g}-second task deadline reached. "
                "I am stopping instead of continuing the loop.\n\n"
                f"Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s\n"
                f"Rounds: {rounds}\n"
                f"Tool calls: {len(trace)}\n\n"
                "Please tell me whether to continue with more time, change approach, or stop here."
            ),
            tool_calls=[],
            usage={},
            raw={
                "flow_timeout": True,
                "elapsed_seconds": int(elapsed),
                "incomplete": True,
                "limit_reason": "flow_timeout",
            },
        )

    async def _chat_maybe_stream(
        req: GatewayRequest, *, emit_deltas: bool = True
    ) -> GatewayResponse:
        """Run the complete generation/failover attempt inside the turn deadline."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return await _flow_deadline_response()
        timer = asyncio.timeout(remaining)
        try:
            async with timer:
                return await _chat_maybe_stream_inner(req, emit_deltas=emit_deltas)
        except TimeoutError:
            if timer.expired():
                return await _flow_deadline_response()
            raise

    def _followup_budget_available() -> bool:
        return rounds < max_rounds and rounds < HARD_ROUND_CAP and time.monotonic() < deadline

    class _FlowDeadlineExceeded(TimeoutError):
        """A non-generation async gate consumed the turn's remaining budget."""

    async def _await_with_flow_deadline(awaitable: Any) -> Any:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise _FlowDeadlineExceeded
        timer = asyncio.timeout(remaining)
        try:
            async with timer:
                return await awaitable
        except TimeoutError as exc:
            if timer.expired():
                raise _FlowDeadlineExceeded from exc
            raise

    while True:
        # --- Intra-turn context eviction: prevent unbounded message growth ---
        # After many tool rounds, the messages list carries stale tool results
        # that bloat every subsequent LLM call. Evict old tool traces while
        # preserving system prompt, user prompt, and recent rounds.
        _evict_result = _evict_middle_tool_traces(
            messages,
            max_messages=80,
            keep_recent_rounds=8,
            task_state=task_state,
        )
        if _evict_result is not None:
            _new_msgs, _evicted_n = _evict_result
            log.info(
                "agent.context_evict evicted=%d msgs_before=%d msgs_after=%d rounds=%d",
                _evicted_n,
                len(messages),
                len(_new_msgs),
                rounds,
            )
            await event_log.append(
                "context_eviction",
                {
                    "evicted": _evicted_n,
                    "msgs_before": len(messages),
                    "msgs_after": len(_new_msgs),
                    "round": rounds,
                },
            )
            messages = _new_msgs

        # Check stop signal before each LLM call.
        if stop_check is not None and stop_check():
            log.info("agent.stopped_by_user rounds=%d", rounds)
            await event_log.append(
                "agent_round",
                {"round": rounds, "stopped": True},
            )
            if final_resp is None:
                final_resp = GatewayResponse(
                    request_id="",
                    model=model,
                    content="*(stopped)*",
                    tool_calls=[],
                    usage={},
                    raw={"stopped": True, "incomplete": True},
                )
            else:
                _mark_incomplete(final_resp, "user_stop")
                final_resp.raw["stopped"] = True
            await _emit_agent_trajectory(
                event_log=event_log,
                task_state=task_state,
                trace=trace,
                final_resp=final_resp,
                rounds=rounds,
                model=model,
            )
            return final_resp, trace, rounds, task_state

        if timeout_seconds > 0 and time.monotonic() > deadline:
            final_resp = await _flow_deadline_response()
            await _emit_agent_trajectory(
                event_log=event_log,
                task_state=task_state,
                trace=trace,
                final_resp=final_resp,
                rounds=rounds,
                model=model,
            )
            return final_resp, trace, rounds, task_state

        # Optional round cap: stop honestly when reached. Never ask the model to
        # turn an unfinished action into success-shaped prose after tools are removed.
        if max_rounds > 0 and rounds >= max_rounds:
            log.warning(
                "agent.max_rounds_exceeded rounds=%d cap=%d (incomplete)",
                rounds,
                max_rounds,
            )
            final_resp = GatewayResponse(
                request_id=f"limit-{rounds}",
                model=model,
                content=(
                    f"I couldn't complete this within the {max_rounds}-round tool budget. "
                    f"Completed tool calls: {task_state.tool_calls}; failures: {task_state.failures}. "
                    "The requested outcome is not verified."
                ),
                tool_calls=[],
                usage={},
                raw={"incomplete": True, "limit_reason": "max_rounds"},
            )
            await event_log.append(
                "agent_incomplete",
                {"reason": "max_rounds", "round": rounds, "task": task_state.goal[:240]},
            )
            break

        rounds += 1
        round_t0 = time.monotonic()
        req = GatewayRequest(
            model=model,
            messages=messages,
            tools=tools_schema if tools_schema else None,
            metadata=req_metadata or None,
            max_tokens=AGENT_MAX_OUTPUT_TOKENS,
        )
        resp = await _chat_maybe_stream(req)
        llm_ms = int((time.monotonic() - round_t0) * 1000)
        final_resp = resp
        calls: list[dict[str, Any]] = _tool_calls_from(resp)
        if (resp.raw or {}).get("flow_timeout"):
            break

        xmlish_calls: list[dict[str, Any]] = (
            [] if calls else _extract_xmlish_tool_calls(resp.content or "")
        )
        if xmlish_calls:
            calls = xmlish_calls
            # Do not echo pseudo tool markup back into the conversation/user-visible stream.
            final_resp = resp = GatewayResponse(
                request_id=resp.request_id,
                model=resp.model,
                content="",
                tool_calls=[
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["args"], default=str),
                        },
                    }
                    for c in calls
                ],
                usage=resp.usage,
                raw={**(resp.raw or {}), "xmlish_tool_calls_recovered": True},
            )

        # Enforce the tool ceiling *before* dispatch.  Checking only after a
        # round allowed one oversized model response to execute arbitrarily
        # many calls beyond the advertised safety limit.
        tools_requested = len(calls)
        remaining_tool_slots = max(0, MAX_TOOL_CALLS_PER_TURN - total_tool_calls)
        tools_rejected_by_budget = max(0, tools_requested - remaining_tool_slots)
        if tools_rejected_by_budget:
            calls = calls[:remaining_tool_slots]
            log.warning(
                "agent.tool_calls_truncated requested=%d accepted=%d remaining_budget=%d",
                tools_requested,
                len(calls),
                remaining_tool_slots,
            )

        usage = resp.usage or {}
        await event_log.append(
            "agent_round",
            {
                "round": rounds,
                "tools_called": len(calls),
                "tools_requested": tools_requested,
                "tools_rejected_by_budget": tools_rejected_by_budget,
                "content_len": len(resp.content or ""),
                "llm_ms": llm_ms,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "synthesized": (resp.raw or {}).get("synthesized") is True,
            },
        )

        if not calls:
            # --- No-tool-call streak counter (runs FIRST, before nudges) ---
            # Every round with zero tool calls increments the streak, regardless
            # of whether a nudge is issued.  This prevents the nudge loop from
            # bypassing the streak breaker (the old bug: nudges did `continue`
            # before the streak counter, so it never incremented).
            if is_action_command(task_state.goal):
                consecutive_no_tool_rounds += 1
            else:
                consecutive_no_tool_rounds = 0
            await _report_progress("finalizing")

            # Hard streak cap: even with nudges, if the model has produced zero
            # tool calls for MAX_NO_TOOL_ROUNDS_TOTAL consecutive rounds, break.
            # This is the ultimate backstop against text-only spin loops.
            if consecutive_no_tool_rounds >= MAX_NO_TOOL_ROUNDS_TOTAL:
                await event_log.append(
                    "no_tool_streak_breaker",
                    {
                        "round": rounds,
                        "streak": consecutive_no_tool_rounds,
                        "forced_final": True,
                        "reason": "max_total_no_tool",
                    },
                )
                log.warning(
                    "agent.no_tool_streak_total rounds=%d streak=%d (forcing finish)",
                    rounds,
                    consecutive_no_tool_rounds,
                )
                final_resp = _mark_incomplete(resp, "no_tool_streak")
                break

            nudge = nudge_for_dropped_tool_calls(resp.tool_calls or [], calls)
            # Always run verification + continue-signal detection during the
            # user's active turn, including before the first tool call. This is
            # task correctness, not background autonomy: text such as "I need
            # to inspect..." must not terminate a prompted long task merely
            # because post-task autonomous continuation is disabled.
            if not nudge:
                nudge = enforce_final_verification(
                    task_state,
                    _strip_thinking(resp.content or ""),
                    model=active_model,
                )
            if nudge:
                final_verification_nudges += 1
                if final_verification_nudges > FINAL_VERIFICATION_NUDGE_LIMIT:
                    await event_log.append(
                        "final_verification_gate",
                        {"round": rounds, "forced_final": True, "nudge": nudge[:240]},
                    )
                    final_resp = _mark_incomplete(resp, "final_verification_nudge_limit")
                    break
                await event_log.append(
                    "final_verification_gate",
                    {"round": rounds, "nudge": nudge[:240], "count": final_verification_nudges},
                )
                messages.append({"role": "user", "content": nudge})
                continue
            # Streak breaker for action tasks (lower threshold than the hard cap)
            if consecutive_no_tool_rounds >= NO_TOOL_STREAK_LIMIT:
                await event_log.append(
                    "no_tool_streak_breaker",
                    {"round": rounds, "streak": consecutive_no_tool_rounds, "forced_final": True},
                )
                log.warning(
                    "agent.no_tool_streak rounds=%d streak=%d (forcing finish)",
                    rounds,
                    consecutive_no_tool_rounds,
                )
                final_resp = _mark_incomplete(resp, "no_tool_streak")
                break
            break  # model produced verified final text — done

        # --- Hard round cap: emergency fallback against infinite loops ---
        if HARD_ROUND_CAP > 0 and rounds >= HARD_ROUND_CAP:
            log.warning(
                "agent.hard_round_cap rounds=%d (stopping without another generation)",
                rounds,
            )
            final_resp = GatewayResponse(
                request_id=f"hard-limit-{rounds}",
                model=model,
                content=(
                    f"I stopped at the hard safety ceiling of {HARD_ROUND_CAP} rounds. "
                    "The requested outcome is not verified; continue in a new turn if needed."
                ),
                tool_calls=[],
                usage={},
                raw={"incomplete": True, "limit_reason": "hard_round_cap"},
            )
            break

        # Append the assistant turn so the model sees its own tool_calls
        # in the next round (OpenAI requires this for correlation).
        # IMPORTANT: always re-serialize from the PARSED calls list (not
        # raw provider output) because:
        #   1. ollama returns function.arguments as a dict — spec requires
        #      a JSON string, or upstream 400s.
        #   2. Gemma can concatenate multiple JSON objects into a single
        #      arguments string (`{...}{...}`), which we split into N
        #      parsed calls. The raw list still has 1 entry — zipping
        #      would drop the rest AND echo back the invalid string.
        normalized_tcs: list[dict[str, Any]] = []
        for parsed in calls:
            # Stamp a unique fallback ID when the provider returned an empty one.
            # This ensures the assistant tool_calls and tool result tool_call_id match.
            if not parsed.get("id"):
                parsed["id"] = f"call_{rounds}_{len(normalized_tcs)}"
            try:
                args_str = json.dumps(parsed["args"] or {}, default=str)
            except Exception:  # noqa: BLE001
                args_str = "{}"
            normalized_tcs.append(
                {
                    "id": parsed["id"],
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": args_str,
                    },
                }
            )

        assistant_msg: dict = {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": normalized_tcs,
        }
        messages.append(assistant_msg)

        # Execute read-only calls in parallel and serialize every mutation.
        # Unique shell strings are not independent: they may still stop the
        # same service, rewrite the same file, or move the same live world.
        for _call in calls:
            _call["_parallel_ok"] = _call_can_run_in_parallel(_call)

        current_round = rounds

        async def _dispatch(
            call: dict, *, dispatch_round: int = current_round
        ) -> tuple[dict, dict]:
            # --- Turn cache: return cached result for identical read-safe calls ---
            cached = turn_cache.get(call["name"], call["args"])
            if cached is not None:
                return call, cached

            fp = _fingerprint(call["name"], call["args"])
            is_mutation = tool_call_is_mutating(call["name"], call["args"])
            if is_mutation and fp in successful_mutation_calls:
                result = {
                    "ok": False,
                    "error": "duplicate_mutation_blocked",
                    "_not_executed": True,
                    "note": (
                        "this identical state-changing call already succeeded in this turn; "
                        "inspect current state before taking another action"
                    ),
                }
            else:
                # LoopGuard: detect repeat/ping-pong patterns across rounds
                try:
                    # deep_research advances its own durable cursor and may be
                    # called with identical args until it returns saturated.
                    if call["name"] != "deep_research":
                        loop_guard.observe(call["name"], call["args"])
                except LoopDetected as ld:
                    log.warning("loop_guard.fired pattern=%s detail=%s", ld.pattern, ld.detail)
                    await event_log.append(
                        "loop_detected",
                        {"pattern": ld.pattern, "detail": ld.detail, "round": dispatch_round},
                    )
                    result = {
                        "ok": False,
                        "error": "loop_detected",
                        "pattern": ld.pattern,
                        "_not_executed": True,
                        "note": (
                            f"Loop detected ({ld.pattern}). You are repeating the same tool calls. "
                            "STOP and try a completely different approach: "
                            "1) re-read the target to check current state, "
                            "2) use a different tool, or "
                            "3) provide your final answer."
                        ),
                    }
                    return call, result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result = {
                        "ok": False,
                        "error": "flow_timeout",
                        "_not_executed": True,
                        "note": "turn deadline expired before tool dispatch",
                    }
                else:
                    try:
                        async with asyncio.timeout(remaining):
                            result = await _run_one_tool(
                                call["name"],
                                call["args"],
                                sender_tier=sender_tier,
                                event_log=event_log,
                                mcp_client=mcp_client,
                                allowed_tools=allowed_tool_set,
                            )
                    except TimeoutError:
                        # Cancellation can race a side effect after dispatch.
                        # Never invite an automatic retry by claiming the tool
                        # definitely did not execute.
                        result = {
                            "ok": False,
                            "error": "flow_timeout",
                            "cancelled": True,
                            "outcome_unknown": is_mutation,
                            "note": "turn deadline expired during tool dispatch",
                        }
                if is_mutation and _tool_result_ok(result):
                    # deep_research is deliberately progressive: an identical
                    # call advances durable query state until it reports
                    # saturation. Other exact mutations remain one-shot.
                    if call["name"] != "deep_research" or not result.get("continue"):
                        successful_mutation_calls.add(fp)
                    # Any successful state change can invalidate observations:
                    # exec may rewrite files, append_memory changes retrieval,
                    # and extension tools can mutate state outside their args.
                    turn_cache.invalidate_all()
                elif _tool_result_ok(result):
                    # Cache only successful bounded read-safe observations.
                    turn_cache.put(call["name"], call["args"], result)
            return call, result

        parallel_calls = [c for c in calls if c.get("_parallel_ok")]
        serial_calls = [c for c in calls if not c.get("_parallel_ok")]

        results: list[tuple[dict, dict]] = []
        if parallel_calls:
            results.extend(await asyncio.gather(*(_dispatch(c) for c in parallel_calls)))
        for call in serial_calls:
            results.append(await _dispatch(call))

        # Track total tool calls for RAS gate and update externalized task state.
        total_tool_calls += len(results)
        task_state.tool_rounds = rounds
        for call, result in results:
            task_state.note_tool(call["name"], call["args"], result)
        # NOTE: Anthropic only allows "system" as a top-level parameter, not mid-conversation.
        # Fold task_state updates into the tool result block instead of a separate system msg.

        # Preserve the model's original tool-call order before any terminal
        # gate can break the loop.  The old ordering dropped the final round
        # from the trace on deadline, no-progress, and tool-ceiling exits,
        # which made a real action invisible and could invite duplicate work.
        result_by_id = {id(c): r for c, r in results}
        for call in calls:
            result = result_by_id.get(id(call), {"ok": False, "error": "missing_result"})
            trace.append(
                {
                    "name": call["name"],
                    "args": call["args"],
                    "result": result,
                }
            )
            tool_result = compress_tool_result(call["name"], result)

            # Failure-conditioned recall gives the next planning round a proven
            # repair playbook without saturating the initial context.
            if tool_experience is not None and not _tool_result_ok(result):
                try:
                    recovery_guidance = tool_experience.render(
                        user_text,
                        [call["name"]],
                        phase="recovery",
                        failed_tool=call["name"],
                        error=str(result.get("error") or result)[:800],
                        limit=2,
                        max_chars=1500,
                    )
                    if recovery_guidance:
                        tool_result = dict(tool_result)
                        tool_result["tool_experience_memory"] = recovery_guidance
                except Exception as e:  # noqa: BLE001
                    log.warning("tool_experience.recovery_retrieval_failed: %r", e)
            result_json = json.dumps(tool_result, default=str)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": result_json,
                }
            )

        if time.monotonic() >= deadline:
            final_resp = await _flow_deadline_response()
            break

        # --- Progress-stall detection ---
        # A "novel" call is one that actually executed (not cached, not
        # duplicate-blocked).  Without this distinction, the model can spin
        # indefinitely on cached reads — every round returns ok:True from
        # the cache, resetting the failure counter, but no real work happens.
        round_ok = sum(1 for _, r in results if _tool_result_ok(r))
        novel_this_round = sum(
            1
            for _, r in results
            if not r.get("_cached")
            and r.get("error")
            not in {
                "circuit_open",
                "duplicate_call_blocked",  # compatibility with stored/provider results
                "duplicate_mutation_blocked",
                "loop_detected",
            }
        )
        if round_ok == 0:
            consecutive_all_fail_rounds += 1
        else:
            consecutive_all_fail_rounds = 0
        if novel_this_round > 0:
            consecutive_no_progress_rounds = 0
        else:
            consecutive_no_progress_rounds += 1
        consecutive_no_tool_rounds = 0  # tools were called — reset streak

        # --- No-progress stall: force final when zero novel calls for N rounds ---
        if consecutive_no_progress_rounds >= NO_PROGRESS_ROUND_LIMIT:
            log.warning(
                "agent.no_progress_stall rounds=%d consecutive=%d novel=0",
                rounds,
                consecutive_no_progress_rounds,
            )
            await event_log.append(
                "no_progress_stall",
                {
                    "round": rounds,
                    "consecutive": consecutive_no_progress_rounds,
                    "forced_final": True,
                },
            )
            if _followup_budget_available():
                # Use remaining round/deadline budget for a useful summary.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"SYSTEM: You have not made new progress for "
                            f"{consecutive_no_progress_rounds} rounds (repeated cached/"
                            f"duplicate tool calls). You MUST provide your final answer NOW. "
                            "Do not call any more tools. Summarize what you accomplished, "
                            "what remains, and any blockers. If the requested outcome was "
                            "not achieved, say so explicitly."
                        ),
                    }
                )
                req = GatewayRequest(
                    model=model,
                    messages=messages,
                    tools=None,
                    metadata=req_metadata or None,
                    max_tokens=AGENT_MAX_OUTPUT_TOKENS,
                )
                final_resp = await _chat_maybe_stream(req)
                _mark_incomplete(final_resp, "no_progress")
                rounds += 1
            else:
                final_resp = GatewayResponse(
                    request_id=f"stall-{rounds}",
                    model=model,
                    content=(
                        "I stopped after repeated rounds with no new progress. "
                        "The requested outcome is not verified."
                    ),
                    tool_calls=[],
                    usage={},
                    raw={"incomplete": True, "limit_reason": "no_progress"},
                )
            await event_log.append(
                "agent_incomplete",
                {"reason": "no_progress", "round": rounds, "task": task_state.goal[:240]},
            )
            break

        # --- Per-turn tool-call ceiling: hard stop on total executions ---
        if total_tool_calls >= MAX_TOOL_CALLS_PER_TURN:
            log.warning(
                "agent.tool_call_ceiling rounds=%d total=%d cap=%d",
                rounds,
                total_tool_calls,
                MAX_TOOL_CALLS_PER_TURN,
            )
            await event_log.append(
                "tool_call_ceiling",
                {
                    "round": rounds,
                    "total": total_tool_calls,
                    "cap": MAX_TOOL_CALLS_PER_TURN,
                    "requested_this_round": tools_requested,
                    "rejected_this_round": tools_rejected_by_budget,
                    "forced_final": True,
                },
            )
            final_resp = GatewayResponse(
                request_id=f"tool-limit-{rounds}",
                model=model,
                content=(
                    f"I couldn't complete the requested action before the {total_tool_calls}-call "
                    "tool limit. The requested outcome is not verified."
                ),
                tool_calls=[],
                usage={},
                raw={
                    "incomplete": True,
                    "limit_reason": "tool_call_ceiling",
                    "tool_calls_executed": total_tool_calls,
                    "tool_calls_rejected": tools_rejected_by_budget,
                },
            )
            await event_log.append(
                "agent_incomplete",
                {"reason": "tool_call_ceiling", "round": rounds, "task": task_state.goal[:240]},
            )
            break

        if consecutive_all_fail_rounds >= CONSECUTIVE_FAIL_LIMIT:
            log.warning(
                "agent.progress_stall rounds=%d consecutive_fail=%d",
                rounds,
                consecutive_all_fail_rounds,
            )
            # Adaptive recovery: analyze failure patterns and inject targeted guidance
            _recent_failures = [
                t
                for t in trace[-CONSECUTIVE_FAIL_LIMIT * 3 :]
                if isinstance(t.get("result"), dict) and not _tool_result_ok(t["result"])
            ]
            _failure_types: dict[str, int] = {}
            for tf in _recent_failures:
                err = tf["result"].get("error", "unknown")
                _failure_types[err] = _failure_types.get(err, 0) + 1

            # Build targeted recovery message based on failure pattern
            _recovery_parts = [
                f"SYSTEM: The last {CONSECUTIVE_FAIL_LIMIT} rounds of tool calls ALL failed."
            ]

            # Pattern-specific guidance
            if _failure_types.get("old_text_not_found", 0) >= 2:
                _recovery_parts.append(
                    "PATTERN: edit old_text mismatch. The file has changed since you last read it. "
                    "Re-read the file to get the CURRENT content, then retry the edit with the exact text."
                )
            elif _failure_types.get("file_not_found", 0) >= 2:
                _recovery_parts.append(
                    "PATTERN: file not found. Use list_dir to verify the path exists. "
                    "Check for typos or wrong directory."
                )
            elif _failure_types.get("bad_arguments", 0) >= 2:
                _recovery_parts.append(
                    "PATTERN: bad arguments. Check the tool schema. "
                    "Common issues: missing required params, wrong types, path vs file parameter."
                )
            elif _failure_types.get("tool_timeout", 0) >= 2:
                _recovery_parts.append(
                    "PATTERN: timeouts. The command is taking too long. "
                    "Try a narrower scope, shorter timeout, or a different approach."
                )
            elif _failure_types.get("loop_detected", 0) >= 1:
                _recovery_parts.append(
                    "PATTERN: loop detected. You are repeating the same calls. "
                    "STOP. Use a completely different tool or approach."
                )

            _recovery_parts.append(
                "ACTIONS: 1) Re-read the target to get current state, "
                "2) Verify arguments match the actual file/command, "
                "3) Try a fundamentally different approach, OR "
                "4) Provide your best answer with what you have. "
                "Do NOT repeat the same failing calls."
            )

            messages.append(
                {
                    "role": "user",
                    "content": " ".join(_recovery_parts),
                }
            )
            await event_log.append(
                "adaptive_recovery",
                {
                    "round": rounds,
                    "failure_types": _failure_types,
                    "pattern_targeted": any(k in str(_recovery_parts) for k in ("PATTERN:",)),
                },
            )
            consecutive_all_fail_rounds = 0  # reset after injection

        await _report_progress()

        # Goal drift check after tool execution
        drift_result = goal_drift.check_drift(
            round_num=rounds,
            recent_tool_calls=[{"name": t["name"], "args": t.get("args", {})} for t in trace[-5:]],
            draft_output=resp.content or "",
        )
        if drift_result.severity == DriftSeverity.SEVERE:
            log.warning(
                "goal_drift.severe score=%.2f recommendation=%s — %s",
                drift_result.score,
                drift_result.recommendation,
                "forcing abort"
                if drift_result.recommendation == "force_abort"
                else "forcing plan refresh",
            )
            await event_log.append(
                "goal_drift",
                {
                    "severity": drift_result.severity.value,
                    "score": drift_result.score,
                    "signals": drift_result.signals,
                    "round": rounds,
                    "recommendation": drift_result.recommendation,
                },
            )
            if drift_result.recommendation == "force_abort":
                if _followup_budget_available():
                    messages.append(
                        {
                            "role": "user",
                            "content": drift_result.refocus_message,
                        }
                    )
                    req = GatewayRequest(
                        model=model,
                        messages=messages,
                        tools=None,
                        metadata=req_metadata or None,
                        max_tokens=AGENT_MAX_OUTPUT_TOKENS,
                    )
                    final_resp = await _chat_maybe_stream(req)
                    _mark_incomplete(final_resp, "goal_drift_abort")
                    rounds += 1
                else:
                    final_resp = GatewayResponse(
                        request_id=f"drift-{rounds}",
                        model=model,
                        content=(
                            "I stopped because execution drifted from the requested goal. "
                            "The requested outcome is not verified."
                        ),
                        tool_calls=[],
                        usage={},
                        raw={"incomplete": True, "limit_reason": "goal_drift_abort"},
                    )
                await event_log.append(
                    "goal_drift_abort",
                    {"round": rounds, "score": drift_result.score, "signals": drift_result.signals},
                )
                break
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": drift_result.refocus_message,
                    }
                )
        elif drift_result.severity == DriftSeverity.MODERATE:
            messages.append(
                {
                    "role": "user",
                    "content": drift_result.refocus_message,
                }
            )
            await event_log.append(
                "goal_drift",
                {
                    "severity": drift_result.severity.value,
                    "score": drift_result.score,
                    "signals": drift_result.signals,
                    "round": rounds,
                },
            )

        # Continue: LLM processes tool results → next response

    assert final_resp is not None
    # Revision requests must include the draft they are supposed to improve.
    # The main loop does not append a text-only terminal response to history.
    messages = [*messages, {"role": "assistant", "content": final_resp.content or ""}]
    user_request_text = user_prompt if isinstance(user_prompt, str) else str(user_prompt)
    correction_blocking = False
    correction_evidence_count = 0
    correction_inconclusive = False
    ov_report = None

    # --- Response Gate (L32) on the agentic path ---
    if (
        correction_gate is not None
        and final_resp.content
        and (final_resp.raw or {}).get("incomplete") is not True
    ):
        try:
            gate_res = await _await_with_flow_deadline(
                correction_gate.check(_strip_thinking(final_resp.content))
            )
            correction_blocking = gate_res.needs_revision
            correction_evidence_count = len(gate_res.evidence)
            correction_inconclusive = gate_res.inconclusive
            await event_log.append(
                "response_gate",
                {
                    "ok": gate_res.ok,
                    "checked": gate_res.checked,
                    "total_claims": gate_res.total_claims,
                    "inconclusive": gate_res.inconclusive,
                    "errors": gate_res.errors[:3],
                    "contradictions": len(gate_res.contradictions),
                },
            )
            if gate_res.inconclusive:
                final_resp.raw = dict(final_resp.raw or {})
                final_resp.raw["correction_gate_inconclusive"] = True
            if gate_res.needs_revision and _followup_budget_available():
                issues = "; ".join(
                    f"{c['claim_subject']}: said '{c['claim_value']}' but stored '{c['stored_value']}'"
                    for c in gate_res.contradictions[:5]
                )
                log.warning("response_gate.revision_needed: %s", issues)
                revision_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            f"FACT-CHECK FAILED — correct these before replying:\n{issues}\n"
                            "Revise your answer. Do NOT repeat the errors."
                        ),
                    },
                ]
                req = GatewayRequest(
                    model=model,
                    messages=revision_messages,
                    tools=None,
                    metadata=req_metadata or None,
                    max_tokens=AGENT_MAX_OUTPUT_TOKENS,
                )
                revised = await _chat_maybe_stream(req, emit_deltas=False)
                rounds += 1
                if (revised.raw or {}).get("flow_timeout"):
                    final_resp = revised
                    correction_blocking = True
                    correction_inconclusive = True
                    raise _FlowDeadlineExceeded
                revised_text = _strip_thinking(revised.content or "")
                rechecked = await _await_with_flow_deadline(correction_gate.check(revised_text))
                accepted = bool(revised_text and rechecked.ok and not rechecked.inconclusive)
                await event_log.append(
                    "response_gate_revision",
                    {
                        "revised": True,
                        "accepted": accepted,
                        "content_len": len(revised.content or ""),
                        "contradictions_before": len(gate_res.contradictions),
                        "contradictions_after": len(rechecked.contradictions),
                    },
                )
                if accepted:
                    final_resp = revised
                    gate_res = rechecked
                    correction_blocking = rechecked.needs_revision
                    correction_evidence_count = len(rechecked.evidence)
                    correction_inconclusive = rechecked.inconclusive
                    messages = [
                        *revision_messages,
                        {"role": "assistant", "content": revised.content or ""},
                    ]
                else:
                    remaining = rechecked.contradictions or gate_res.contradictions
                    details = "; ".join(
                        f"{item['claim_subject']} is recorded as {item['stored_value']}"
                        for item in remaining[:3]
                    )
                    final_resp.content = (
                        "I couldn't safely return the generated answer because it still "
                        "conflicts with canonical memory. "
                        f"Current evidence: {details}. "
                        "The requested answer remains unverified; please confirm the canonical "
                        "value or ask me to investigate its source."
                    )
                    final_resp.raw = dict(final_resp.raw or {})
                    final_resp.raw.update(
                        {
                            "incomplete": True,
                            "verification_failure": "correction_gate",
                        }
                    )
                    log.warning("response_gate.revision_failed_closed")
            elif gate_res.needs_revision:
                details = "; ".join(
                    f"{item['claim_subject']} is recorded as {item['stored_value']}"
                    for item in gate_res.contradictions[:3]
                )
                final_resp.content = (
                    "I couldn't safely return the generated answer because it conflicts "
                    f"with canonical memory: {details}. The revision budget was exhausted, "
                    "so the requested answer remains unverified."
                )
                final_resp.raw = dict(final_resp.raw or {})
                final_resp.raw.update(
                    {
                        "incomplete": True,
                        "verification_failure": "correction_gate_revision_budget",
                    }
                )
                log.warning("response_gate.revision_skipped_budget")
        except _FlowDeadlineExceeded:
            if (final_resp.raw or {}).get("flow_timeout") is not True:
                final_resp = await _flow_deadline_response()
            correction_blocking = True
            correction_inconclusive = True
        except SpendGuardTripped:
            raise
        except Exception as e:
            log.warning("response_gate.error: %r", e)
            correction_inconclusive = True
            final_resp.raw = dict(final_resp.raw or {})
            final_resp.raw["correction_gate_inconclusive"] = True

    # --- Output Verifier: pre-send quality gate ---
    if (
        output_verifier is not None
        and final_resp.content
        and (final_resp.raw or {}).get("incomplete") is not True
    ):
        try:
            ov_report = output_verifier.verify(
                output=_strip_thinking(final_resp.content),
                user_request=user_request_text,
                tool_trace=trace,
                task_type=task_state.task_type,
            )
            await event_log.append(
                "output_verified",
                {
                    "score": round(ov_report.score, 3),
                    "summary": ov_report.summary,
                    "has_critical": ov_report.has_critical,
                    "issues": [
                        {
                            "severity": i.severity.value,
                            "category": i.category,
                            "description": i.description[:100],
                        }
                        for i in ov_report.issues[:5]
                    ],
                },
            )
            if ov_report.has_issues and _followup_budget_available():
                revision = ov_report.revision_prompt
                log.warning("output_verifier.issues: %s", ov_report.summary)
                revision_messages = [*messages, {"role": "user", "content": revision}]
                req = GatewayRequest(
                    model=model,
                    messages=revision_messages,
                    tools=None,
                    metadata=req_metadata or None,
                    max_tokens=AGENT_MAX_OUTPUT_TOKENS,
                )
                revised = await _chat_maybe_stream(req, emit_deltas=False)
                rounds += 1
                if (revised.raw or {}).get("flow_timeout"):
                    final_resp = revised
                    raise _FlowDeadlineExceeded
                revised_text = _strip_thinking(revised.content or "")
                revised_report = output_verifier.verify(
                    output=revised_text,
                    user_request=user_request_text,
                    tool_trace=trace,
                    task_type=task_state.task_type,
                )
                before_counts = (
                    sum(i.severity.value == "critical" for i in ov_report.issues),
                    sum(i.severity.value == "major" for i in ov_report.issues),
                )
                after_counts = (
                    sum(i.severity.value == "critical" for i in revised_report.issues),
                    sum(i.severity.value == "major" for i in revised_report.issues),
                )
                improved = bool(
                    revised_text
                    and (
                        after_counts < before_counts
                        or (
                            after_counts == before_counts and revised_report.score > ov_report.score
                        )
                    )
                )
                await event_log.append(
                    "output_verifier_revision",
                    {
                        "revised": True,
                        "accepted": improved,
                        "score_before": round(ov_report.score, 3),
                        "score_after": round(revised_report.score, 3),
                        "content_len": len(revised.content or ""),
                    },
                )
                if improved:
                    final_resp = revised
                    ov_report = revised_report
                    messages = [
                        *revision_messages,
                        {"role": "assistant", "content": revised.content or ""},
                    ]
                else:
                    blocking = [
                        issue
                        for issue in ov_report.issues
                        if issue.severity.value in {"critical", "major"}
                    ]
                    details = "; ".join(issue.description for issue in blocking[:3])
                    final_resp.content = (
                        "I couldn't safely return the generated draft because verification "
                        f"still found: {details}. The requested outcome remains unverified."
                    )
                    final_resp.raw = dict(final_resp.raw or {})
                    final_resp.raw.update(
                        {
                            "incomplete": True,
                            "verification_failure": "output_verifier",
                        }
                    )
                    log.warning("output_verifier.revision_failed_closed")
            elif ov_report.has_issues:
                blocking = [
                    issue
                    for issue in ov_report.issues
                    if issue.severity.value in {"critical", "major"}
                ]
                details = "; ".join(issue.description for issue in blocking[:3])
                final_resp.content = (
                    "I couldn't safely return the generated draft because verification "
                    f"found: {details}. The revision budget was exhausted, so the requested "
                    "outcome remains unverified."
                )
                final_resp.raw = dict(final_resp.raw or {})
                final_resp.raw.update(
                    {
                        "incomplete": True,
                        "verification_failure": "output_verifier_revision_budget",
                    }
                )
                log.warning("output_verifier.revision_skipped_budget")
        except _FlowDeadlineExceeded:
            if (final_resp.raw or {}).get("flow_timeout") is not True:
                final_resp = await _flow_deadline_response()
        except SpendGuardTripped:
            raise
        except Exception as e:
            log.warning("output_verifier.error: %r", e)

    # --- Best-of-N: generate alternatives and pick the best ---
    # Disabled by default for local models — doubles inference time for
    # marginal quality gain. Enable via NORAX_BEST_OF_N=1 if needed.
    if (
        best_of_n is not None
        and final_resp.content
        and len(final_resp.content) > 50
        and (final_resp.raw or {}).get("incomplete") is not True
        and _followup_budget_available()
    ):
        try:
            candidate_timeout = min(30.0, max(0.001, deadline - time.monotonic()))
            candidate = await _await_with_flow_deadline(
                best_of_n.generate(
                    model=model,
                    messages=messages,
                    user_request=user_prompt if isinstance(user_prompt, str) else str(user_prompt),
                    task_type=task_state.task_type,
                    tool_trace=trace,
                    n=1,
                    temperature=0.7,
                    timeout_seconds=candidate_timeout,
                )
            )
            _account_usage(candidate.usage)
            if candidate and candidate.content and candidate.score > 0:
                candidate_text = (
                    candidate.content if reasoning_output else _strip_thinking(candidate.content)
                )
                # Score the original output too so we only replace if better
                orig_score, _ = BestOfN.score_output(
                    final_resp.content,
                    user_request=user_request_text,
                    task_type=task_state.task_type,
                    tool_trace=trace,
                )
                candidate_safe = bool(candidate_text)
                candidate_report = None
                if candidate_safe and output_verifier is not None:
                    candidate_report = output_verifier.verify(
                        output=_strip_thinking(candidate_text),
                        user_request=user_request_text,
                        tool_trace=trace,
                        task_type=task_state.task_type,
                    )
                    candidate_safe = not candidate_report.has_issues and (
                        ov_report is None or candidate_report.score >= ov_report.score
                    )
                if candidate_safe and correction_gate is not None:
                    candidate_gate = await _await_with_flow_deadline(
                        correction_gate.check(_strip_thinking(candidate_text))
                    )
                    candidate_safe = candidate_gate.ok and not candidate_gate.inconclusive
                should_replace = bool(
                    candidate_safe
                    and candidate.score > orig_score
                    and candidate_text != final_resp.content
                )
                await event_log.append(
                    "best_of_n",
                    {
                        "score": round(candidate.score, 3),
                        "orig_score": round(orig_score, 3),
                        "content_len": len(candidate.content),
                        "safe": candidate_safe,
                        "replaced": should_replace,
                    },
                )
                if should_replace:
                    log.info(
                        "best_of_n: replacing output (score=%.3f > orig=%.3f)",
                        candidate.score,
                        orig_score,
                    )
                    final_resp = GatewayResponse(
                        request_id=final_resp.request_id,
                        model=final_resp.model,
                        content=candidate_text,
                        tool_calls=final_resp.tool_calls,
                        usage=final_resp.usage,
                        raw=final_resp.raw,
                    )
                    if candidate_report is not None:
                        ov_report = candidate_report
                    if correction_gate is not None:
                        correction_blocking = False
                        correction_evidence_count = len(candidate_gate.evidence)
                        correction_inconclusive = candidate_gate.inconclusive
                else:
                    log.info(
                        "best_of_n: keeping original (orig=%.3f >= cand=%.3f)",
                        orig_score,
                        candidate.score,
                    )
        except _FlowDeadlineExceeded:
            # Candidate generation is an optional enhancement.  The original
            # response is already complete, so keep it instead of converting a
            # valid answer into a timeout just because ranking ran out of time.
            log.warning("best_of_n.deadline_exceeded; keeping original")
        except SpendGuardTripped:
            raise
        except Exception as e:
            log.warning("best_of_n.error: %r", e)

    # The runtime records the returned response once, so expose turn-wide usage
    # here: tool rounds, revisions, failovers, continuations, and opt-in
    # best-of-N candidates have all been accounted above.
    final_resp.usage = dict(turn_usage)
    mutation_outcome = summarize_mutation_trace(trace)
    verifier_blocking = bool(ov_report and ov_report.has_issues)
    accepted_outcome = bool(
        not correction_blocking
        and not verifier_blocking
        and (final_resp.raw or {}).get("incomplete") is not True
        and bool((final_resp.content or "").strip())
        and (mutation_outcome.supports_success_claim if mutation_outcome.attempted else True)
    )
    successful_observation = any(
        tool_call_is_successful_verification(
            str(item.get("name") or ""),
            item.get("args") if isinstance(item.get("args"), dict) else {},
            item.get("result") if isinstance(item.get("result"), dict) else {},
        )
        for item in trace
        if not (
            isinstance(item.get("result"), dict) and item["result"].get("_not_executed") is True
        )
    )
    correction_supported = bool(
        correction_gate is not None
        and not correction_inconclusive
        and correction_evidence_count > 0
    )
    verifier_supported = bool(ov_report and ov_report.objectively_verified)
    verification_basis = [
        basis
        for basis, present in (
            ("mutation_receipt_or_check", mutation_outcome.supports_success_claim),
            ("successful_observation", successful_observation),
            ("canonical_evidence", correction_supported),
            ("deterministic_output_check", verifier_supported),
        )
        if present
    ]
    verified_outcome = bool(accepted_outcome and verification_basis)
    objective_outcome_observed = bool(
        mutation_outcome.attempted
        or any(
            isinstance(item.get("result"), dict) and item["result"].get("_not_executed") is not True
            for item in trace
        )
        or correction_supported
        or bool(ov_report and ov_report.objective_checked)
    )
    final_resp.raw = dict(final_resp.raw or {})
    final_resp.raw["accepted_outcome"] = accepted_outcome
    final_resp.raw["verified_outcome"] = verified_outcome
    final_resp.raw["objective_outcome_observed"] = objective_outcome_observed
    final_resp.raw["usage_scope"] = "turn_total"
    final_resp.raw["verification_basis"] = verification_basis
    final_resp.raw["mutation_outcome"] = {
        "attempted": mutation_outcome.attempted,
        "succeeded": mutation_outcome.succeeded,
        "failed": mutation_outcome.failed,
        "verified_after_last_mutation": mutation_outcome.verified_after_last_mutation,
    }
    verifier_score = ov_report.score if ov_report is not None else None
    if verifier_score is not None:
        final_resp.raw["output_verifier_score"] = round(verifier_score, 6)
    if not defer_outcome_recording:
        await record_agent_outcome(
            event_log=event_log,
            task_state=task_state,
            trace=trace,
            final_resp=final_resp,
            rounds=rounds,
            model=model,
            user_text=user_text,
            tool_experience=tool_experience,
            self_model=self_model,
            active_inference=active_inference,
            metacognitive=metacognitive,
            verifier_score=verifier_score,
        )
    await _report_progress("complete")
    return final_resp, trace, rounds, task_state
