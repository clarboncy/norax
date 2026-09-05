"""Runtime — the message loop. Wires brain hot_path, memory, adapters, and
cognitive amplifiers end-to-end with Discord in/out + outbound reply emission.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import os
import re
import subprocess
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import commands as cmd_mod
from ..adapter.cron_in import CronAdapter
from ..adapter.discord_in import DiscordInAdapter
from ..adapter.http_in import HttpInAdapter
from ..adapter.reply_tag import parse_reply_tag
from ..brain import agent_loop, hot_path
from ..brain.hot_path.basal_ganglia import BasalGanglia
from ..brain.hot_path.vta import VTA
from ..brain.policy import memory_k_for_turn
from ..commands import _is_gpt_56_sol, _supports_max_reasoning
from ..config.loader import Config
from ..context.window import Frame, RollingWindow
from ..dispatch import tools as dispatch_tools
from ..gateway_client import (
    GatewayClient,
    GatewayRouter,
    GatewayUpstreamError,
    SpendGuardTripped,
)
from ..memory.causal_graph import CausalGraph
from ..memory.entity_graph import EntityGraph
from ..memory.episodic import Episode, EpisodicBuffer
from ..memory.hebbian import HebbianLearner
from ..memory.injection import ContextInjector
from ..memory.retrievers.fast import FastContext
from ..memory.retrievers.multi_signal import MultiSignalRetriever
from ..memory.store import MemoryStore
from ..memory.temporal_graph import TemporalGraph
from ..memory.user_model import UserModel
from ..observability.log import EventLog
from ..observability.metrics import Metrics
from .capability_registry import CapabilityRegistry
from .graceful_degradation import get_degradation_manager
from .ingress_bus import IngressBus
from .memory_coordinator import MemoryCoordinator
from .outbound import OutboundRegistry

log = logging.getLogger("norax.runtime.core")

_MAX_PROVIDER_DISCOVERY_BYTES = 2 * 1024 * 1024
_MAX_PROVIDER_DISCOVERY_MODELS = 100


def _flag_enabled(value: object, *, default: bool = False) -> bool:
    """Parse an internal feature flag without truthy-string surprises."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _explicit_result_ok(value: object) -> bool:
    """Accept success only from the protocol's literal boolean ``true``."""
    return isinstance(value, dict) and value.get("ok") is True


def _runtime_choice(value: object, choices: set[str], *, default: str, label: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in choices:
            return normalized
    log.warning("runtime.invalid_choice key=%s value=%r default=%s", label, value, default)
    return default


def _model_identifier(value: object, *, label: str = "model") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    model = value.strip()
    if not model or len(model) > 512:
        raise ValueError(f"{label} must contain 1-512 characters")
    if any(not character.isprintable() or character.isspace() for character in model):
        raise ValueError(f"{label} must not contain whitespace or control characters")
    return model


def _model_identifiers(value: object, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    if len(value) > 32:
        raise ValueError(f"{label} must contain at most 32 models")
    return [_model_identifier(item, label=f"{label} entry") for item in value]


def _provider_identifier(value: object, *, label: str = "provider") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    provider = value.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", provider):
        raise ValueError(f"{label} contains unsupported characters")
    return provider


def _is_loopback_host(host: str) -> bool:
    """Return whether a listener host is restricted to the local machine."""
    normalized = host.strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _a2a_advertised_url(host: str, port: int, configured: object) -> str:
    """Build an honest Agent Card URL and reject ambiguous wildcard binds."""
    if configured is not None and str(configured).strip():
        return str(configured).strip()
    normalized = host.strip().strip("[]")
    if normalized in {"0.0.0.0", "::"}:
        raise ValueError("A2A base_url is required when binding to a wildcard address")
    url_host = f"[{normalized}]" if ":" in normalized else normalized
    return f"http://{url_host}:{port}"


def _bounded_config_int(value: object, *, label: str, minimum: int, maximum: int) -> int:
    """Parse a bounded integer setting without accepting booleans or decimals."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().lstrip("+").isdigit():
        parsed = int(value)
    else:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    return parsed


def _bounded_environment_int(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read an operator integer while keeping a bad override non-fatal."""
    raw = os.environ.get(name, str(default))
    try:
        return _bounded_config_int(raw, label=name, minimum=minimum, maximum=maximum)
    except ValueError:
        log.warning("runtime.invalid_environment key=%s value=%r default=%d", name, raw, default)
        return default


def _apply_configured_fleet_startup_fixes() -> bool:
    """Run the private fleet hook only after an explicit operator opt-in."""
    if not _flag_enabled(os.environ.get("NORAX_APPLY_FLEET_STARTUP_FIXES", "")):
        return False
    from ..ops.fleet_healthcheck import apply_startup_fixes

    apply_startup_fixes()
    return True


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


_TOOL_CAPABILITY = {
    "browser": "browser_tool",
    "computer_use": "computer_tool",
    "sandbox_exec": "sandbox_tool",
    "web_search": "web_search_tool",
    "deep_research": "web_search_tool",
}


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
    live_count = int(payload.get("live_count", 0))
    return {
        "ok": relay_ok,
        "degraded_reason": (
            "relay is healthy but no optional remote nodes are connected"
            if relay_ok and live_count == 0
            else ""
        ),
        "live_count": live_count,
        "live_nodes": payload.get("live_nodes", []),
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
        detail = error.upstream_message.strip()
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
        result = item.get("result")
        if not isinstance(result, dict) or not _explicit_result_ok(result):
            continue
        candidate = result.get("runtime_lifecycle_deferred")
        if candidate in {"restart", "stop"}:
            action = str(candidate)
    return action


def _coerce_tool_args_json(content: object) -> str:
    """Return a valid JSON-object string for a tool call's `arguments`.

    Strict providers (Alibaba/Qwen via OpenRouter) reject tool_calls whose
    `function.arguments` is empty, null, or non-JSON. Other providers
    (Anthropic, etc.) tolerate it. Coerce to a guaranteed JSON object string.
    """
    import json as _json

    if content is None:
        return "{}"
    if isinstance(content, (dict, list)):
        try:
            return _json.dumps(content)
        except Exception:  # noqa: BLE001
            return "{}"
    s = str(content).strip()
    if not s:
        return "{}"
    try:
        parsed = _json.loads(s)
    except Exception:  # noqa: BLE001
        # Non-JSON text args -> wrap so it stays a valid JSON object
        return _json.dumps({"_raw": s})
    if isinstance(parsed, dict):
        return s
    # Valid JSON but not an object (e.g. null, [], "str") -> wrap
    return _json.dumps({"_raw": parsed})


def _bounded_history_text(content: str, max_chars: int) -> str:
    """Keep both ends of an oversized historical message within budget."""
    text = str(content or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"\n...[historical message truncated; {len(text)} chars total]...\n"
    available = max(0, max_chars - len(marker))
    head = available // 2
    return text[:head] + marker + text[-(available - head) :]


def _clean_historical_assistant_text(content: str) -> str:
    """Remove private reasoning and leaked harness headers from replayed replies."""
    text = re.sub(r"<thinking>.*?</thinking>", "", str(content or ""), flags=re.DOTALL)
    lines = text.lstrip().splitlines()
    while lines and (not lines[0].strip() or re.match(r"^(?:GOAL|PLAN|RISK):", lines[0].strip())):
        lines.pop(0)
    return "\n".join(lines).strip()


def _history_turn_limit(user_prompt: str) -> int:
    """Use more history only when the user clearly asks to continue it."""
    text = " ".join(str(user_prompt or "").lower().split())
    words = re.findall(r"[a-z0-9_+-]+", text)
    if len(words) <= 8 and re.match(
        r"^(?:continue|resume|keep going|carry on|go ahead|finish(?: it)?|pick (?:it )?up|"
        r"do it|fix that|same thing|what about|and (?:then|now)|yes|ok|okay)\b",
        text,
    ):
        return 8
    return 3


def _candidate_prior_turn_ids(frames: list[Frame], user_prompt: str) -> set[int]:
    ordered = list(dict.fromkeys(frame.turn_id for frame in frames if frame.turn_id > 0))
    return set(ordered[-_history_turn_limit(user_prompt) :])


def _history_budget_for_model(
    model: str,
    *,
    provider_kind: str = "unknown",
    action_request: bool = False,
) -> int:
    """Reserve most context for the current request, tools, and response."""
    from ..gateway_client.ollama_profiles import resolve_profile

    profile = resolve_profile(model)
    context_tokens = int(profile.num_ctx or 131_072)
    window_override = int(getattr(profile, "window_tokens", 0) or 0)
    model_lower = (model or "").lower()
    local = (
        "freetoken" in model_lower
        or ".gguf" in model_lower
        or (provider_kind == "ollama" and ":cloud" not in model_lower)
    )
    if window_override:
        cap = 10_000 if action_request else 12_000
        fraction = 0.10 if action_request else 0.15
        return min(cap, max(2_048, int(window_override * fraction)))
    if local:
        context_tokens = min(context_tokens, 32_768)
        cap = 6_000 if action_request else 8_000
        return min(cap, max(2_048, int(context_tokens * 0.25)))
    # Cloud / non-local: ensure a large enough context floor so the fraction
    # yields the intended budget even when the profile is the local default.
    context_tokens = max(context_tokens, 262_144)
    fraction = 0.08 if action_request else 0.10
    cap = 12_000 if action_request else 16_000
    return min(cap, max(4_000, int(context_tokens * fraction)))


def _select_history_tail(
    frames: list[Frame],
    candidate_turn_ids: set[int],
    *,
    token_budget: int,
) -> tuple[set[int], set[str], int]:
    """Select a contiguous recent history tail under an estimated budget.

    Tool calls and results remain paired.  Within a very tool-heavy turn, the
    most recent pairs are retained because TaskState already carries compact
    evidence from earlier actions.
    """
    ordered_turns = list(
        dict.fromkeys(frame.turn_id for frame in frames if frame.turn_id in candidate_turn_ids)
    )
    text_cap_tokens = max(512, min(2_048, token_budget // 4))
    max_tool_pairs = max(1, min(12, token_budget // 1_600))

    valid_calls_by_turn: dict[int, set[str]] = {}
    for turn_id in ordered_turns:
        calls = [
            frame.call_id
            for frame in frames
            if frame.turn_id == turn_id and frame.kind == "tool_call" and frame.call_id
        ]
        results = {
            frame.call_id
            for frame in frames
            if frame.turn_id == turn_id and frame.kind == "tool_result" and frame.call_id
        }
        valid = [call_id for call_id in calls if call_id in results]
        valid_calls_by_turn[turn_id] = set(valid[-max_tool_pairs:])

    def turn_cost(turn_id: int) -> int:
        cost = 0
        valid_calls = valid_calls_by_turn[turn_id]
        for frame in frames:
            if frame.turn_id != turn_id:
                continue
            if frame.kind in {"tool_call", "tool_result"}:
                if frame.call_id not in valid_calls:
                    continue
                cap = 300 if frame.kind == "tool_call" else 500
                cost += min(frame.token_estimate(), cap) + 16
            else:
                cost += min(frame.token_estimate(), text_cap_tokens) + 16
        return cost

    selected_turns: set[int] = set()
    remaining = max(0, token_budget)
    for turn_id in reversed(ordered_turns):
        cost = turn_cost(turn_id)
        if cost > remaining:
            break
        selected_turns.add(turn_id)
        remaining -= cost

    selected_calls = (
        set().union(*(valid_calls_by_turn[turn_id] for turn_id in selected_turns))
        if selected_turns
        else set()
    )
    return selected_turns, selected_calls, text_cap_tokens * 4


def _checkpoint_status(task_state: Any, response: Any) -> str:
    """Classify a terminal turn without treating read-only work as unfinished."""
    if response is None:
        return "in_progress"
    raw = getattr(response, "raw", {}) or {}
    if raw.get("stopped") is True or raw.get("incomplete") is True:
        return "in_progress"
    if "accepted_outcome" in raw:
        return "complete" if raw.get("accepted_outcome") is True else "in_progress"
    if "verified_outcome" in raw:
        return "complete" if raw.get("verified_outcome") is True else "in_progress"
    # No execution path gets to infer completion from fluent prose. Callers
    # without TaskState must attach an explicit verified_outcome signal.
    return "in_progress"


def _is_coding_query(query: str) -> bool:
    """Heuristic for task_type passed to ContextInjector."""
    markers = (
        "code",
        "repo",
        "repository",
        "debug",
        "fix",
        "patch",
        "test",
        "pytest",
        "function",
        "class",
        "implement",
        "refactor",
        "error",
        "traceback",
        "exception",
        "tool",
        "file",
        "edit",
        "write",
        "read ",
        "search",
        "ollama",
        "gateway",
        "runtime",
        "server",
        "config",
        "agent loop",
        "memory",
        "retriev",
    )
    q = (query or "").lower()
    return any(m in q for m in markers)


class Runtime:
    def __init__(
        self,
        *,
        ingress: IngressBus,
        events: EventLog,
        cfg: Config,
        gateway: GatewayClient | GatewayRouter,
        default_model: str = "qwen3.8-27b-fast:latest",
        failover_models: list[str] | None = None,
        outbound: OutboundRegistry | None = None,
        metrics: Metrics | None = None,
        soul: Any | None = None,
    ) -> None:
        self.ingress = ingress
        self.events = events
        self.cfg = cfg
        self.gateway = gateway
        self.default_model = _model_identifier(default_model, label="default model")
        self.failover_models = _model_identifiers(failover_models, label="failover_models")
        self.thinking_effort = _runtime_choice(
            getattr(cfg, "thinking_effort", "medium"),
            set(cmd_mod.THINK_LABELS),
            default="medium",
            label="thinking_effort",
        )
        self.reasoning_output = _flag_enabled(getattr(cfg, "reasoning_output", False))
        self.planning_mode = _runtime_choice(
            getattr(cfg, "planning_mode", "direct"),
            cmd_mod.PLANNING_MODES,
            default="direct",
            label="planning_mode",
        )
        configured_rounds = getattr(cfg, "max_tool_rounds", 0) or 0
        if isinstance(configured_rounds, bool) or not isinstance(configured_rounds, int):
            raise TypeError("max_tool_rounds must be an integer")
        if configured_rounds < 0:
            raise ValueError("max_tool_rounds must be non-negative")
        self.max_tool_rounds = (
            min(configured_rounds, agent_loop.HARD_ROUND_CAP) if configured_rounds else 0
        )
        if self.max_tool_rounds != configured_rounds:
            log.warning(
                "configured max_tool_rounds=%s exceeds the hard safety cap; using %s",
                configured_rounds,
                self.max_tool_rounds,
            )
        self.memory_depth = _runtime_choice(
            getattr(cfg, "memory_depth", "auto"),
            cmd_mod.MEMORY_DEPTHS,
            default="auto",
            label="memory_depth",
        )
        self.weak_model_boost = _runtime_choice(
            getattr(cfg, "weak_model_boost", "auto"),
            cmd_mod.BOOST_MODES,
            default="auto",
            label="weak_model_boost",
        )
        self.stream_replies = _flag_enabled(getattr(cfg, "stream_replies", True), default=True)
        self.response_length = _runtime_choice(
            getattr(cfg, "response_length", "balanced"),
            cmd_mod.LENGTH_PREFS,
            default="balanced",
            label="response_length",
        )
        self.tool_activity = _runtime_choice(
            getattr(cfg, "tool_activity", "normal"),
            cmd_mod.ACTIVITY_MODES,
            default="normal",
            label="tool_activity",
        )
        self.outbound = outbound or OutboundRegistry()
        self.metrics = metrics or Metrics()
        self._soul = soul
        self.capabilities = CapabilityRegistry()
        self._config_hash: str = "unknown"
        self._effective_model: str = "?"
        self._effective_provider: str = "?"
        self._configured_provider: str = "?"
        self._provider_store: Any = None
        self._base_provider_names: set[str] = set()
        self._provider_mutation_lock = asyncio.Lock()
        self._gateway_timeout_seconds = 600.0
        self.discord: Any = None  # set by Runtime.build() if discord is enabled
        self.cron: Any = None  # set by Runtime.build() if cron is enabled
        self.agent_os_bridge: Any = None  # set by Runtime.build() if agent_os chat is wired
        self.reminders: Any = None  # persistent scheduler bound by Runtime.build()
        self.mcp_client: Any = None
        self.a2a_server: Any = None
        self._a2a_uvicorn: Any = None
        self._draining = False
        self._run_started = False
        self._shutdown_complete = False
        self._shutdown_lock: asyncio.Lock | None = None
        self._started_mono = time.monotonic()
        self._started_wall = time.time()
        self._last_user_turn_time: float | None = None
        # Stop signal: per-channel flag to abort active turns
        self._stop_channels: set[str] = set()
        self._active_turn_tasks: dict[str, asyncio.Task[Any]] = {}
        self._turn_slot_releasers: dict[asyncio.Task[Any], Callable[[], None]] = {}
        self._turn_queues: dict[str, deque[Any]] = {}
        self._turn_work_count = 0
        self._max_concurrent_turns = max(
            1,
            min(64, int(getattr(cfg, "max_concurrent_turns", 8) or 8)),
        )
        self._max_pending_turns = max(
            self._max_concurrent_turns,
            min(10_000, int(getattr(cfg, "max_pending_turns", 256) or 256)),
        )
        self._max_pending_per_channel = min(32, self._max_pending_turns)
        self._turn_semaphore = asyncio.Semaphore(self._max_concurrent_turns)
        self._active_typing_handles: dict[asyncio.Task[Any], Any] = {}
        self._active_stream_messages: dict[asyncio.Task[Any], Any] = {}
        self._capability_probe_task: asyncio.Task[None] | None = None
        self._sleep_task: asyncio.Task[None] | None = None
        self._warmup_task: asyncio.Task[None] | None = None
        self._probe_task: asyncio.Task[None] | None = None
        self._autonomy_task: asyncio.Task[Any] | None = None
        self._optional_tasks: list[asyncio.Task[Any]] = []
        self._maintenance_tasks: set[asyncio.Task[Any]] = set()
        self._default_model_persist_task: asyncio.Task[None] | None = None
        self._pending_default_model: str | None = None
        # Per-channel prompt-cache observability (Claude Code SEV pattern).
        self._last_static_hash: dict[str, str] = {}
        self._last_tools_hash: dict[str, str] = {}
        self._cache_break_times: dict[str, list[float]] = {}
        # Per-channel rolling-window: conversation history across turns.
        self._windows: dict[str, RollingWindow] = {}
        # Graceful degradation manager (singleton)
        self._degradation = get_degradation_manager()
        # Memory store + retriever
        mem_root = getattr(cfg, "memory_root", None)
        self._memory_root: Path | None = mem_root if isinstance(mem_root, Path) else None
        # Type annotations (assigned in both branches below)
        self._memory_store: MemoryStore | None = None
        self._memory_coordinator: MemoryCoordinator | None = None
        self._fast_ctx: FastContext | None = None
        self._episodic: EpisodicBuffer | None = None
        self._user_model: UserModel | None = None
        self._skill_learner: Any = None
        self._context_injector: ContextInjector | None = None
        self._causal_graph: CausalGraph | None = None
        self._temporal_graph: TemporalGraph | None = None
        self._graph_update_lock = asyncio.Lock()
        self._hybrid: MultiSignalRetriever | None = None
        self._last_graph_save: float = 0.0
        cognition_cfg = getattr(cfg, "cognition", {})
        if not isinstance(cognition_cfg, dict):
            cognition_cfg = {}
        configured_experiments = cognition_cfg.get("experimental_signals", False)
        self._experimental_cognitive_signals = _flag_enabled(
            configured_experiments
        ) or _flag_enabled(os.environ.get("NORAX_EXPERIMENTAL_COGNITIVE_SIGNALS", ""))
        configured_idle_learning = cognition_cfg.get("idle_learning", False)
        self._idle_learning_enabled = _flag_enabled(configured_idle_learning) or _flag_enabled(
            os.environ.get("NORAX_IDLE_LEARNING_ENABLED", "")
        )
        configured_harness_analysis = cognition_cfg.get("harness_analysis", False)
        self._harness_analysis_enabled = _flag_enabled(
            configured_harness_analysis
        ) or _flag_enabled(os.environ.get("NORAX_HARNESS_ANALYSIS_ENABLED", ""))
        vision_cfg = getattr(cfg, "vision", {})
        self._vision_config = dict(vision_cfg) if isinstance(vision_cfg, dict) else {}
        connectors_cfg = getattr(cfg, "connectors", {})
        if not isinstance(connectors_cfg, dict):
            connectors_cfg = {}
        self._connectors: dict[str, dict[str, Any]] = {
            str(name): dict(value)
            for name, value in connectors_cfg.items()
            if isinstance(value, dict)
        }
        self._planner_enabled = _flag_enabled(
            self._connectors.get("planner", {"enabled": True}).get("enabled", True)
        )
        self._multi_agent_enabled = _flag_enabled(
            self._connectors.get("multi_agent", {"enabled": True}).get("enabled", True)
        )
        try:
            self._multi_agent_max_concurrent = _bounded_config_int(
                self._connectors.get("multi_agent", {}).get("max_concurrent", 4),
                label="multi-agent concurrency",
                minimum=1,
                maximum=32,
            )
        except ValueError as error:
            log.warning("%s; using 4", error)
            self._multi_agent_max_concurrent = 4
        active_inference_cfg = connectors_cfg.get("active_inference", {})
        self._active_inference_enabled = _flag_enabled(active_inference_cfg.get("enabled", False))
        if not self._planner_enabled and self.planning_mode == "orchestrator":
            log.warning("planner connector is disabled; forcing planning_mode=direct")
            self.planning_mode = "direct"
        # Stateful, evidence-backed components are shared across turns.
        # Experimental prompt signals are opt-in and remain off the production
        # hot path until they have grounded provenance and live eval evidence.
        self._output_verifier: Any = None
        self._self_model: Any = None
        self._active_inference: Any = None
        self._metacognitive: Any = None
        self._best_of_n: Any = None
        self._curiosity: Any = None
        self._domain_transfer: Any = None
        self._analogy_engine: Any = None
        if mem_root and isinstance(mem_root, Path):
            self._memory_store = MemoryStore(root=mem_root)
            self._memory_store.refresh()
            self._memory_coordinator = MemoryCoordinator(
                mem_root,
                store=self._memory_store,
            )
            self._fast_ctx = FastContext(store=self._memory_store)
            # Multi-signal retriever: keyword + embedding + entity-link + FTS5 index
            entity_graph = EntityGraph(root=mem_root)
            # Persistent graph instances: shared with the retriever (which loads
            # them once at init) and mutated in-memory post-turn; Sprint E saves
            # debounced. Per-turn load+full-save of the 14MB causal JSON was the
            # hot path cost this replaces.
            causal_graph = CausalGraph(root=mem_root)
            temporal_graph = TemporalGraph(root=mem_root)
            self._causal_graph = causal_graph
            self._temporal_graph = temporal_graph
            sqlite_index = None
            try:
                from ..memory.retrievers.sqlite_index import SQLiteIndexRetriever

                sqlite_index = SQLiteIndexRetriever(memory_root=mem_root)
                # The projection is reconciled in run() before ingress starts.
                # Do not query it here: a fresh workspace has no schema yet.
                log.info("sqlite_index_retriever configured: %s", sqlite_index.db_path)
            except Exception as e:
                log.warning("sqlite_index_retriever.init_failed: %r", e)
            try:
                from ..memory.embeddings import OllamaEmbedder
                from ..memory.retrievers.local import LocalRetriever

                embedder = OllamaEmbedder.from_env()
                local = LocalRetriever(
                    store=self._memory_store,
                    embedder=embedder,
                    degradation_manager=self._degradation,
                )
                from ..memory.retrievers.cross_encoder import LLMJudgeReranker

                reranker_enabled = os.environ.get(
                    "NORAX_RERANK_ENABLED",
                    "false",
                ).strip().lower() in {"1", "true", "yes", "on"}
                reranker_model = os.environ.get("NORAX_RERANK_MODEL", "").strip()
                if reranker_enabled and not reranker_model:
                    log.warning("llm_judge_reranker disabled: NORAX_RERANK_MODEL is required")
                    reranker_enabled = False
                reranker = (
                    LLMJudgeReranker(model=reranker_model, enabled=True)
                    if reranker_enabled
                    else None
                )
                self._hybrid = MultiSignalRetriever(
                    keyword=self._fast_ctx,
                    embedding=local,
                    entity_graph=entity_graph,
                    sqlite_index=sqlite_index,
                    causal_graph=causal_graph,
                    temporal_graph=temporal_graph,
                    reranker=reranker,
                )
                log.info(
                    "multi_signal_retriever ready "
                    "(kw + embed + entity + idx + causal + tmp; llm_judge_rerank=%s)",
                    "enabled" if reranker_enabled else "disabled",
                )
            except Exception as e:
                log.warning("multi_signal_retriever.init_failed: %r (keyword-only)", e)
                self._hybrid = MultiSignalRetriever(
                    keyword=self._fast_ctx,
                    entity_graph=entity_graph,
                    sqlite_index=sqlite_index,
                    causal_graph=causal_graph,
                    temporal_graph=temporal_graph,
                )

            # Sprint F: smart context mixer over keyword + embedding + external
            local_retriever: LocalRetriever | None = None
            embedder_for_external = None
            if self._hybrid is not None:
                try:
                    local_retriever = self._hybrid.embedding
                except Exception:
                    local_retriever = None
            if local_retriever is not None:
                embedder_for_external = local_retriever.embedder
            if embedder_for_external is None:
                try:
                    from ..memory.embeddings import OllamaEmbedder

                    embedder_for_external = OllamaEmbedder.from_env()
                except Exception:
                    embedder_for_external = None
            try:
                from ..memory.retrievers.external import ExternalRetriever

                external = ExternalRetriever(
                    store=self._memory_store,
                    embedder=embedder_for_external,  # type: ignore[arg-type]
                )
                self._context_injector = ContextInjector(
                    fast=self._fast_ctx,
                    local=local_retriever,
                    external=external,
                )
                log.info("context_injector ready")
            except Exception as e:
                log.warning("context_injector.init_failed: %r", e)
            log.info(
                "memory loaded: %d hot, %d canonical, %d sleep",
                len(self._memory_store.hot),
                len(self._memory_store.all_canonical()),
                len(self._memory_store.sleep),
            )
            # Ensure hot identity lines are current (fixes stale Gen5 scratchpad header).
            try:
                from ..memory.hot_inject import refresh_hot_identity

                role = os.environ.get("NORAX_ROLE", "production")
                refresh_hot_identity(mem_root, generation=7, role=role)
                self._memory_store.refresh()
            except Exception:  # noqa: BLE001
                log.debug("hot_identity.refresh_failed", exc_info=True)
            self._turn_count = 0
            # Brain state (persistent across turns)
            state_dir = mem_root / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            self._vta = VTA(state_path=state_dir / "vta.json")
            self._basal_ganglia = BasalGanglia(state_path=state_dir / "basal_ganglia.json")
            self._hebbian = HebbianLearner()
            # Episodic replay and generated skills are experimental learning
            # paths.  Keeping them off by default avoids duplicate per-turn
            # persistence and prevents heuristic patterns from entering live
            # retrieval without an explicit deployment decision.
            if self._idle_learning_enabled or self._experimental_cognitive_signals:
                self._episodic = EpisodicBuffer(root=mem_root / "episodic")
            if self._idle_learning_enabled:
                from ..brain.skill_learner import SkillLearner

                self._skill_learner = SkillLearner(root=mem_root)
            # The user model directly contributes a bounded prompt block.
            self._user_model = UserModel(root=mem_root)
        else:
            self._vta = VTA()
            self._basal_ganglia = BasalGanglia()
            self._hebbian = HebbianLearner()

    @staticmethod
    def _channel_id(env: Any) -> str:
        raw = getattr(env, "raw", None)
        raw_channel = raw.get("channel_id") if isinstance(raw, dict) else None
        return str(raw_channel or getattr(env, "channel", None) or "default")

    def _track_turn_task(self, channel_id: str, task: asyncio.Task[Any]) -> None:
        """Own one channel task and release it as soon as it terminates."""
        self._active_turn_tasks[channel_id] = task

        def _finished(completed: asyncio.Task[Any]) -> None:
            if self._active_turn_tasks.get(channel_id) is completed:
                self._active_turn_tasks.pop(channel_id, None)
            queue = self._turn_queues.get(channel_id)
            if queue is not None and not queue:
                self._turn_queues.pop(channel_id, None)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                log.error(
                    "turn_task.crashed channel=%s error=%r",
                    channel_id,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(_finished)

    def _queue_turn(self, channel_id: str, env: Any) -> bool:
        """Queue one turn without ever blocking the global ingress consumer."""
        queue = self._turn_queues.setdefault(channel_id, deque())
        worker = self._active_turn_tasks.get(channel_id)
        channel_work = len(queue) + int(worker is not None and not worker.done())
        if (
            self._turn_work_count >= self._max_pending_turns
            or channel_work >= self._max_pending_per_channel
        ):
            log.warning(
                "turn_queue.full channel=%s channel_work=%d total_work=%d",
                channel_id,
                channel_work,
                self._turn_work_count,
            )
            if not queue:
                self._turn_queues.pop(channel_id, None)
            return False
        queue.append(env)
        self._turn_work_count += 1
        if worker is None or worker.done():
            worker = asyncio.create_task(
                self._run_channel_turns(channel_id),
                name=f"turn::{channel_id}",
            )
            self._track_turn_task(channel_id, worker)
        return True

    async def _run_channel_turns(self, channel_id: str) -> None:
        """Run one channel sequentially under the global turn concurrency cap."""
        queue = self._turn_queues[channel_id]
        try:
            while queue and not self._draining:
                env = queue.popleft()
                try:
                    await self._turn_semaphore.acquire()
                    slot_released = False
                    owner_task = asyncio.current_task()

                    def release_slot() -> None:
                        nonlocal slot_released
                        if not slot_released:
                            slot_released = True
                            self._turn_semaphore.release()

                    if owner_task is not None:
                        self._turn_slot_releasers[owner_task] = release_slot
                    try:
                        if self._draining:
                            return
                        await self._handle_turn(env)
                    finally:
                        if owner_task is not None:
                            self._turn_slot_releasers.pop(owner_task, None)
                        release_slot()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    log.exception("turn worker failed channel=%s", channel_id)
                finally:
                    self._turn_work_count = max(0, self._turn_work_count - 1)
        finally:
            if queue:
                self._turn_work_count = max(0, self._turn_work_count - len(queue))
                queue.clear()
            if self._turn_queues.get(channel_id) is queue:
                self._turn_queues.pop(channel_id, None)
            self._stop_channels.discard(channel_id)

    def _release_current_turn_slot(self) -> bool:
        """Release global turn capacity once the completed reply is delivered."""
        task = asyncio.current_task()
        if task is None:
            return False
        release = self._turn_slot_releasers.pop(task, None)
        if release is None:
            return False
        release()
        return True

    def _track_optional_task(self, capability: str, task: asyncio.Task[Any]) -> None:
        """Track a connector task and invalidate readiness if it exits."""
        if task not in self._optional_tasks:
            self._optional_tasks.append(task)

        self._observe_service_task(capability, task)

    def _observe_service_task(self, capability: str, task: asyncio.Task[Any]) -> None:
        """Turn an unexpected long-running task exit into capability evidence."""

        def _finished(completed: asyncio.Task[Any]) -> None:
            if self._draining:
                return
            if completed.cancelled():
                cancellation_error = RuntimeError("service cancelled unexpectedly")
                self.capabilities.mark_failed(capability, cancellation_error)
                log.error("subsystem cancelled unexpectedly: %s", capability)
                return
            task_error = completed.exception()
            if task_error is None:
                task_error = RuntimeError("service stopped unexpectedly")
                log.error("subsystem stopped unexpectedly: %s", capability)
            self.capabilities.mark_failed(capability, task_error)

        task.add_done_callback(_finished)

    def _record_mcp_health(self, status: dict[str, Any]) -> None:
        """Keep MCP capability state aligned with live connection evidence."""
        if self._draining:
            return
        active = [str(name) for name in status.get("active") or []]
        healthy = [str(name) for name in status.get("healthy") or []]
        failures = {
            str(name): str(error)[:500]
            for name, error in dict(status.get("failures") or {}).items()
        }
        if failures and healthy:
            self.capabilities.mark_degraded(
                "mcp_client",
                f"failed connections: {', '.join(sorted(failures))}",
            )
        elif failures:
            details = "; ".join(f"{name}: {error}" for name, error in sorted(failures.items()))
            self.capabilities.mark_failed(
                "mcp_client",
                RuntimeError(f"all MCP connections failed: {details}"),
            )
        elif active:
            self.capabilities.mark_ok("mcp_client")
        else:
            self.capabilities.mark_degraded("mcp_client", "no live MCP connections")
        self.capabilities.update_metadata(
            "mcp_client",
            active=active,
            healthy=healthy,
            failed=sorted(failures),
        )

    def _track_mcp_connection_task(
        self,
        client: Any,
        name: str,
        task: asyncio.Task[Any],
    ) -> None:
        if task not in self._optional_tasks:
            self._optional_tasks.append(task)

        def _finished(completed: asyncio.Task[Any]) -> None:
            if self._draining:
                return
            if completed.cancelled():
                detail = "connection owner cancelled unexpectedly"
            else:
                error = completed.exception()
                detail = (
                    f"{type(error).__name__}: {error}"
                    if error is not None
                    else "connection owner stopped unexpectedly"
                )
            status = dict(client.health())
            failures = dict(status.get("failures") or {})
            failures[name] = detail[:500]
            status["failures"] = failures
            status["healthy"] = [item for item in status.get("healthy") or [] if str(item) != name]
            self._record_mcp_health(status)

        task.add_done_callback(_finished)

    @staticmethod
    async def _await_uvicorn_started(
        server: Any,
        task: asyncio.Task[Any],
        *,
        timeout: float = 5.0,
    ) -> None:
        """Wait for a real bound listener instead of advertising readiness early."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not bool(getattr(server, "started", False)):
            if task.done():
                if task.cancelled():
                    raise RuntimeError("A2A listener was cancelled during startup")
                error = task.exception()
                if error is not None:
                    raise RuntimeError("A2A listener failed during startup") from error
                raise RuntimeError("A2A listener stopped before becoming ready")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("A2A listener did not become ready within 5 seconds")
            await asyncio.sleep(0.01)

    def _mount_commerce_connector(self) -> None:
        """Mount commerce routes before HTTP ingress begins accepting requests."""
        commerce_cfg = self._connectors.get("commerce", {})
        if not _flag_enabled(commerce_cfg.get("enabled")):
            return
        self.capabilities.register(
            "commerce_api",
            required_for_readiness=_flag_enabled(commerce_cfg.get("required")),
        )
        self.capabilities.mark_initializing("commerce_api")
        try:
            from ..commerce.api_endpoints import mount_commerce

            http_adapter = next(
                (adapter for adapter in self.ingress._adapters or [] if hasattr(adapter, "app")),
                None,
            )
            if http_adapter is None:
                raise RuntimeError("no HTTP adapter found")
            mount_commerce(http_adapter.app)
            self.capabilities.mark_ok("commerce_api")
            log.info("commerce_api mounted on HTTP adapter")
        except Exception as error:
            self.capabilities.mark_failed("commerce_api", error)
            if commerce_cfg.get("required"):
                raise RuntimeError("required commerce API failed to mount") from error

    def _ensure_cognitive_components(self) -> None:
        """Lazily initialize and then reuse stateful cognitive components.

        Turn handlers are concurrent across channels, but initialization and
        each component's record/save methods are synchronous. They therefore
        execute without an asyncio scheduling point and safely share one
        in-process state snapshot.
        """
        if self._output_verifier is None:
            self._output_verifier = self.capabilities.try_init(
                "output_verifier",
                lambda: __import__(
                    "norax.brain.output_verifier", fromlist=["OutputVerifier"]
                ).OutputVerifier(),
            )

        experimental = bool(getattr(self, "_experimental_cognitive_signals", False))

        if experimental and self._self_model is None:
            try:
                from ..brain.self_model import SelfModel

                self._self_model = (
                    SelfModel(self._memory_root / "self_model.json")
                    if self._memory_root
                    else SelfModel()
                )
                self._self_model.load()
                self.capabilities.mark_ok("self_model")
            except Exception as exc:  # noqa: BLE001
                self._self_model = None
                self.capabilities.mark_failed("self_model", exc)

        if (
            experimental or bool(getattr(self, "_active_inference_enabled", False))
        ) and self._active_inference is None:
            try:
                from ..brain.active_inference import ActiveInference

                self._active_inference = (
                    ActiveInference(self._memory_root / "active_inference.json")
                    if self._memory_root
                    else ActiveInference()
                )
                self._active_inference.load()
                self.capabilities.mark_ok("active_inference")
            except Exception as exc:  # noqa: BLE001
                self._active_inference = None
                self.capabilities.mark_failed("active_inference", exc)

        if experimental and self._metacognitive is None:
            try:
                from ..brain.metacognitive import MetacognitiveCalibration

                self._metacognitive = (
                    MetacognitiveCalibration(self._memory_root / "metacog.json")
                    if self._memory_root
                    else MetacognitiveCalibration()
                )
                self._metacognitive.load()
                self.capabilities.mark_ok("metacognitive")
            except Exception as exc:  # noqa: BLE001
                self._metacognitive = None
                self.capabilities.mark_failed("metacognitive", exc)

        best_of_n_enabled = os.environ.get("NORAX_BEST_OF_N", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if best_of_n_enabled and self._best_of_n is None:
            try:
                from ..brain.best_of_n import BestOfN

                self._best_of_n = BestOfN(gateway=self.gateway)
                self.capabilities.mark_ok("best_of_n")
            except Exception as exc:  # noqa: BLE001
                self._best_of_n = None
                self.capabilities.mark_failed("best_of_n", exc)

        if not experimental:
            return

        if self._curiosity is None:
            try:
                from ..brain.curiosity_engine import CuriosityEngine

                self._curiosity = (
                    CuriosityEngine(self._memory_root / "curiosity.json")
                    if self._memory_root
                    else CuriosityEngine()
                )
                self._curiosity.load()
                self.capabilities.mark_ok("curiosity_engine")
            except Exception as exc:  # noqa: BLE001
                self._curiosity = None
                self.capabilities.mark_failed("curiosity_engine", exc)

        if self._domain_transfer is None:
            try:
                from ..brain.domain_transfer import DomainTransfer

                self._domain_transfer = DomainTransfer(
                    self_model=self._self_model,
                    memory_root=self._memory_root,
                )
                self.capabilities.mark_ok("domain_transfer")
            except Exception as exc:  # noqa: BLE001
                self._domain_transfer = None
                self.capabilities.mark_failed("domain_transfer", exc)
        elif self._self_model is not None:
            self._domain_transfer.self_model = self._self_model

        if self._analogy_engine is None:
            try:
                from ..brain.analogy_engine import AnalogyEngine

                self._analogy_engine = AnalogyEngine(
                    memory_root=self._memory_root,
                    episodic=self._episodic,
                )
                self.capabilities.mark_ok("analogy_engine")
            except Exception as exc:  # noqa: BLE001
                self._analogy_engine = None
                self.capabilities.mark_failed("analogy_engine", exc)

    @classmethod
    def build(cls, cfg: Config) -> Runtime:
        host, port = cfg.http_bind
        from ..soul import load_soul

        soul = load_soul(
            root=cfg.soul_path.parent,
            soul_path=cfg.soul_path,
            identity_path=cfg.identity_path,
            user_path=cfg.user_path,
        )
        # The dashboard and monitors depend on this exact endpoint. Silent
        # port fallback creates a split-brain deployment that appears started
        # but is unreachable through its configured clients.
        log.info("http binding %s:%d", host, port)

        outbound = OutboundRegistry()
        metrics = Metrics()
        try:
            from ..gateway_client.cursor_observability import get_cursor_observability

            get_cursor_observability(metrics=metrics)
        except ImportError:
            pass
        adapters: list[Any] = [HttpInAdapter(host=host, port=port, metrics=metrics)]

        # Direct owner-chat bridge for the Agent OS dashboard. Registers the
        # bridge as an outbound adapter under channel key ``agent_os`` so the
        # runtime routes assistant replies back to the dashboard websocket, and
        # configures the token-authenticated ingress route.
        http_in = adapters[0]
        chat_token = cfg.agent_os_chat_token
        if chat_token:
            http_in.configure_owner_chat(
                owner_id=cfg.owner_id or "owner",
                owner_label=cfg.owner_label,
                chat_token=chat_token,
            )
            outbound.register("agent_os", http_in.agent_os_bridge)
            log.info("agent_os owner-chat bridge wired")
        else:
            log.info("agent_os chat_token not set; owner-chat bridge disabled")

        disc = None
        dcfg = cfg.discord
        if dcfg.enabled:
            if dcfg.token:
                disc = DiscordInAdapter(config=dcfg, owner_id=cfg.owner_id)
                adapters.append(disc)
                outbound.register("discord", disc)
                log.info("discord adapter wired")
            else:
                log.warning(
                    "discord enabled=true but no token configured; "
                    "adapter NOT started. Set NORAX_DISCORD_TOKEN or "
                    "channels.discord.token to enable."
                )
        # Bind outbound into the motor-tool registry so t_message_send is live.
        dispatch_tools.bind_outbound(outbound)

        ccfg = cfg.cron
        cron = None
        if ccfg.enabled and ccfg.jobs:
            cron = CronAdapter.from_config(ccfg.to_raw(), owner_id=cfg.owner_id)
            if cron.jobs:
                adapters.append(cron)
                log.info("cron adapter wired: %d enabled job(s)", len(cron.jobs))
            else:
                cron = None
                log.warning("cron enabled but no enabled jobs were configured")

        ingress = IngressBus(adapters)
        events = EventLog(cfg.event_log)
        gw_cfg = cfg.gateway
        import os as _os

        from ..config.provider_store import ProviderStore

        provider_store = ProviderStore()
        persisted_settings = provider_store.runtime_settings()
        persisted_model = persisted_settings.get("default_model")
        if persisted_model is not None:
            try:
                gw_cfg["default_model"] = _model_identifier(
                    persisted_model, label="persisted default model"
                )
            except ValueError as error:
                log.warning("ignoring invalid persisted default model: %s", error)
        # Environment is the final deployment authority, including over UI-
        # persisted settings from an earlier run.
        if _os.environ.get("NORAX_DEFAULT_MODEL"):
            gw_cfg["default_model"] = _model_identifier(
                _os.environ["NORAX_DEFAULT_MODEL"], label="NORAX_DEFAULT_MODEL"
            )
        if _os.environ.get("NORAX_DEFAULT_PROVIDER"):
            gw_cfg["default_provider"] = _provider_identifier(
                _os.environ["NORAX_DEFAULT_PROVIDER"], label="NORAX_DEFAULT_PROVIDER"
            )
        custom_providers = [
            row for row in provider_store.list_runtime() if row.get("enabled", True)
        ]
        timeout = gw_cfg["timeout_seconds"]

        configured_providers = gw_cfg.get("providers")
        providers_cfg = dict(configured_providers) if isinstance(configured_providers, dict) else {}
        base_provider_names = set(providers_cfg)
        accepted_custom_providers = []
        for custom in custom_providers:
            if custom["name"] in providers_cfg:
                log.warning("custom provider ignored because name is reserved: %s", custom["name"])
                continue
            accepted_custom_providers.append(custom)
            providers_cfg[custom["name"]] = {
                "base_url": custom["base_url"],
                "provider_kind": custom["provider_kind"],
                "api_key": custom.get("api_key") or None,
            }
        if providers_cfg:
            # Multi-provider mode: build a GatewayRouter.
            clients: dict[str, Any] = {}
            for pname, pcfg in providers_cfg.items():
                if not isinstance(pcfg, dict):
                    continue
                env_name = f"NORAX_{pname.upper().replace('-', '_')}_TOKEN"
                pk = pcfg.get("api_key") or _os.environ.get(env_name)
                provider_kind = str(pcfg.get("provider_kind") or pname).lower()
                if provider_kind == "codex_direct":
                    # Local Codex proxies on :4146/:4147 borrow OAuth from
                    # CODEX_HOME auth.json themselves. Do not attach a
                    # generic gateway bearer; it confuses local wrapping.
                    pk = None
                else:
                    pk = pk or _os.environ.get("NORAX_GATEWAY_TOKEN")
                if provider_kind == "ollama" or pname in ("ollama", "ollama_direct"):
                    from ..gateway_client.ollama_wrapper import OllamaGatewayClient

                    inner = GatewayClient(
                        base_url=pcfg.get("base_url", "http://127.0.0.1:11434/v1"),
                        timeout=timeout,
                        api_key=pk,
                        extra_headers=pcfg.get("extra_headers") or None,
                        stream_required=_flag_enabled(pcfg.get("stream_required", False)),
                        model_prefix=pcfg.get("model_prefix"),
                        provider_kind="ollama",
                        chat_path=pcfg.get("chat_path", "/chat/completions"),
                    )
                    clients[pname] = OllamaGatewayClient(inner, metrics=metrics)
                else:
                    clients[pname] = GatewayClient(
                        base_url=pcfg.get("base_url", "http://127.0.0.1:11434/v1"),
                        timeout=timeout,
                        api_key=pk,
                        extra_headers=pcfg.get("extra_headers") or None,
                        stream_required=_flag_enabled(pcfg.get("stream_required", False)),
                        model_prefix=pcfg.get("model_prefix"),
                        provider_kind=pcfg.get("provider_kind") or pname,
                        chat_path=pcfg.get("chat_path", "/chat/completions"),
                    )
                log.info("gateway.provider %s → %s", pname, pcfg.get("base_url"))
            routes_raw = gw_cfg["routes"]
            routes = [(f"{row['name']}/*", row["name"]) for row in accepted_custom_providers]
            routes.extend((r[0], r[1]) for r in routes_raw)
            default_prov = gw_cfg["default_provider"]
            if default_prov not in clients:
                raw_gateway = cfg.raw.get("gateway")
                explicit_default = (
                    isinstance(raw_gateway, dict) and "default_provider" in raw_gateway
                ) or bool(_os.environ.get("NORAX_DEFAULT_PROVIDER"))
                if explicit_default:
                    raise ValueError(
                        f"default gateway provider {default_prov!r} was not initialized"
                    )
                default_prov = next(iter(clients))
                gw_cfg["default_provider"] = default_prov
            gateway: Any = GatewayRouter(
                providers=clients,
                routes=routes,
                default_provider=default_prov,
            )
            log.info(
                "gateway.router ready: %d providers, %d routes, default=%s",
                len(clients),
                len(routes),
                default_prov,
            )
        else:
            # Legacy single-client mode.
            api_key = gw_cfg.get("api_key") or _os.environ.get("NORAX_GATEWAY_TOKEN")
            gateway = GatewayClient(  # type: ignore[no-redef]
                base_url=gw_cfg.get("base_url", "http://127.0.0.1:11434/v1"),
                timeout=timeout,
                api_key=api_key,
                extra_headers=gw_cfg.get("extra_headers") or None,
                provider_kind=gw_cfg.get("provider_kind", "ollama"),
                chat_path=gw_cfg.get("chat_path", "/chat/completions"),
            )
        rt_instance = cls(
            ingress=ingress,
            events=events,
            cfg=cfg,
            gateway=gateway,
            default_model=gw_cfg.get("default_model", "qwen3.8-27b-fast:latest"),
            outbound=outbound,
            failover_models=gw_cfg.get("failover_models") or [],
            metrics=metrics,
            soul=soul,
        )
        rt_instance._provider_store = provider_store
        rt_instance._base_provider_names = base_provider_names
        rt_instance._gateway_timeout_seconds = timeout
        if chat_token:
            rt_instance.agent_os_bridge = http_in.agent_os_bridge
        from .reminders import ReminderScheduler

        rt_instance.reminders = ReminderScheduler(
            root=cfg.memory_root / "reminders",
            outbound=outbound,
            default_channel="discord",
            default_target=cfg.owner_id,
        )
        dispatch_tools.bind_reminder_scheduler(rt_instance.reminders)
        # Wire runtime reference into HTTP adapter for /status and /readyz
        adapters[0]._runtime_ref = rt_instance
        # Wire live memory search into the search_memory tool
        if rt_instance._hybrid is not None:
            dispatch_tools.bind_memory_search(rt_instance._hybrid.search)
            log.info("search_memory tool wired to hybrid retriever")
        elif hasattr(rt_instance, "_fast_ctx") and rt_instance._fast_ctx is not None:
            dispatch_tools.bind_memory_search(rt_instance._fast_ctx.search)
            log.info("search_memory tool wired to keyword retriever")
        # Wire live status info into the status tool
        import time as _time

        _start_time = _time.time()
        # Compute a config hash for observability — redacts sensitive fields.
        import json as _json

        _cfg_for_hash = {k: v for k, v in cfg.raw.items() if k not in ("owner",)}
        _cfg_hash = hashlib.sha256(
            _json.dumps(_cfg_for_hash, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        rt_instance._config_hash = _cfg_hash
        rt_instance._effective_model = gw_cfg.get("default_model", "?")
        configured_provider = gw_cfg.get("default_provider", "?")
        rt_instance._configured_provider = configured_provider
        rt_instance._effective_provider = configured_provider
        route_for = getattr(gateway, "route_for", None)
        if callable(route_for):
            try:
                rt_instance._effective_provider = route_for(rt_instance._effective_model)[0]
            except Exception as exc:  # noqa: BLE001
                log.warning("gateway.effective_route_failed: %r", exc)

        def _live_status_info() -> dict[str, Any]:
            effective_provider = configured_provider
            live_route_for = getattr(gateway, "route_for", None)
            if callable(live_route_for):
                try:
                    effective_provider = live_route_for(rt_instance.default_model)[0]
                except Exception as exc:  # noqa: BLE001
                    log.warning("gateway.live_status_route_failed: %r", exc)
            rt_instance._effective_model = rt_instance.default_model
            rt_instance._effective_provider = effective_provider
            provider_urls = getattr(gateway, "provider_urls", {})
            providers = sorted(provider_urls) if isinstance(provider_urls, dict) else []
            return {
                "default_model": rt_instance.default_model,
                "default_provider": effective_provider,
                "configured_default_provider": configured_provider,
                "providers": providers,
                "discord_enabled": disc is not None,
                "start_time": _start_time,
                "config_hash": _cfg_hash,
                "version": cmd_mod.NORAX_VERSION,
            }

        dispatch_tools.bind_status_info(_live_status_info)
        # Wire slash-command handler on the discord adapter (if any).
        if disc is not None:
            disc.set_slash_handler(rt_instance._slash_dispatch)
            rt_instance.discord = disc
        if cron is not None:
            rt_instance.cron = cron
        return rt_instance

    async def run(self) -> None:
        if self._run_started:
            raise RuntimeError("Runtime.run() may only be started once")
        if self._shutdown_complete:
            raise RuntimeError("Runtime cannot run after shutdown")
        self._run_started = True
        # Deployment-specific host repair must never mutate a generic install.
        # Operators that own the referenced user services can explicitly opt
        # into the legacy fleet maintenance hook.
        _apply_configured_fleet_startup_fixes()
        # Reconcile canonical files and all derived projections before ingress
        # accepts turns. This prevents stale retrieval after an unclean stop or
        # an out-of-band memory edit while the service was offline.
        try:
            if self._memory_coordinator is not None:
                self._memory_coordinator.canonical_changed("runtime_start")
                await self._memory_coordinator.sync_projections()
        except Exception as e:
            log.warning("memory_projection_sync.startup_failed: %r", e)
        self._mount_commerce_connector()
        await self.ingress.start()
        for capability, adapter in (
            ("discord_ingress", self.discord),
            ("cron_scheduler", self.cron),
        ):
            if adapter is None:
                continue
            self.capabilities.register(capability)
            task = getattr(adapter, "_task", None)
            if not isinstance(task, asyncio.Task):
                self.capabilities.mark_failed(
                    capability,
                    RuntimeError(f"{adapter.name} did not start a supervisor task"),
                )
                continue
            self.capabilities.mark_ok(capability)
            self._track_optional_task(capability, task)
        reminders = getattr(self, "reminders", None)
        if reminders is not None:
            await reminders.start()
            self.capabilities.mark_ok("reminder_scheduler")
            reminder_task = getattr(reminders, "_task", None)
            if isinstance(reminder_task, asyncio.Task):
                self._track_optional_task("reminder_scheduler", reminder_task)
        for capability in (
            "browser_tool",
            "computer_tool",
            "sandbox_tool",
            "web_search_tool",
            "remote_relay",
        ):
            self.capabilities.register(capability, stale_after_sec=900.0)
        self.capabilities.register(
            "tool_manifest",
            stale_after_sec=900.0,
            required_for_readiness=True,
        )
        self.capabilities.register("operational_probe_loop", required_for_readiness=True)
        self.capabilities.mark_initializing("operational_probe_loop")
        self._capability_probe_task = asyncio.create_task(
            self._capability_probe_loop(), name="operational-capability-probe"
        )
        self._observe_service_task("operational_probe_loop", self._capability_probe_task)
        await self.events.append(
            "runtime.start",
            {
                "version": cmd_mod.NORAX_VERSION,
                "config_hash": getattr(self, "_config_hash", "unknown"),
                "effective_model": getattr(self, "_effective_model", "?"),
                "effective_provider": getattr(self, "_effective_provider", "?"),
                "build_commit": cmd_mod.BUILD_COMMIT,
                "build_date": cmd_mod.BUILD_DATE,
            },
        )
        # Sprint B: background sleep consolidation on idle
        self._last_turn_time = time.time()
        self.capabilities.register("idle_memory_maintenance")
        self.capabilities.mark_ok("idle_memory_maintenance")
        self._sleep_task = asyncio.create_task(self._idle_sleep_loop())
        self._observe_service_task("idle_memory_maintenance", self._sleep_task)
        # Pre-warm embedding caches from disk so first query doesn't pay cold-start cost
        self._warmup_task = asyncio.create_task(self._warmup_caches())
        # Synthetic completions consume provider quota, so deployments can
        # disable them independently from serving real user turns.
        self._completion_probe_enabled = os.environ.get(
            "NORAX_COMPLETION_PROBE_ENABLED", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        requested_probe_mode = os.environ.get("NORAX_COMPLETION_PROBE_MODE", "transport")
        self._completion_probe_mode = requested_probe_mode.strip().lower()
        if self._completion_probe_mode not in {"transport", "completion"}:
            log.warning(
                "invalid NORAX_COMPLETION_PROBE_MODE=%r; using transport",
                requested_probe_mode,
            )
            self._completion_probe_mode = "transport"
        # Completion probes are tiny but still consume model capacity. Keep
        # them infrequent, wait for an idle gap, and cap deferral below the
        # readiness freshness window so sustained traffic cannot hide an
        # unhealthy serving path forever.
        self._completion_probe_interval_seconds = _bounded_environment_int(
            "NORAX_COMPLETION_PROBE_INTERVAL_SECONDS",
            default=600,
            minimum=60,
            maximum=840,
        )
        self._completion_probe_idle_seconds = _bounded_environment_int(
            "NORAX_COMPLETION_PROBE_IDLE_SECONDS",
            default=30,
            minimum=0,
            maximum=300,
        )
        self._last_probe_result: dict[str, Any] | None = None
        if self._completion_probe_enabled:
            self.capabilities.register(
                "serving_probe_loop",
                stale_after_sec=900.0,
                required_for_readiness=True,
            )
            self.capabilities.mark_initializing("serving_probe_loop")
            self._probe_task = asyncio.create_task(self._completion_probe_loop())
            self._observe_service_task("serving_probe_loop", self._probe_task)
        else:
            self._probe_task = None
            self._last_probe_result = {
                "ok": True,
                "enabled": False,
                "mode": "disabled",
            }
            log.info("completion_probe disabled by config")
        # Autonomy engine: scratchpad sync, self-healing, prompt optimization, checkpoint resume
        # Register subsystems with graceful degradation manager
        try:
            dm = get_degradation_manager()
            log.info("graceful_degradation: %d subsystems registered", len(dm._subsystems))
        except Exception as e:
            log.warning("graceful_degradation.init_failed: %r", e)
        self.autonomy_config = None
        try:
            from .autonomy import AutonomyConfig, AutonomyEngine

            self.autonomy_config = AutonomyConfig()
            mem_root = getattr(self.cfg, "memory_root", None)
            if mem_root and isinstance(mem_root, Path) and self.autonomy_config.enabled:
                engine = AutonomyEngine(
                    memory_root=mem_root,
                    config=self.autonomy_config,
                    episodic=getattr(self, "_episodic", None),
                    gateway=self.gateway,
                    event_log=self.events,
                    default_model=getattr(self, "_effective_model", "") or "",
                )
                self._autonomy_task = engine.start()
                self.capabilities.register("autonomy_engine")
                self.capabilities.mark_ok("autonomy_engine")
                self._observe_service_task("autonomy_engine", self._autonomy_task)
                log.info("autonomy_engine started")
            elif mem_root and isinstance(mem_root, Path):
                log.info("autonomy_engine disabled by config")
        except Exception as e:
            log.warning("autonomy_engine.start_failed: %r", e)
        # Optional subsystems: wire behind config flags, default off
        # MCP client: discover external tools
        mcp_cfg = self._connectors.get("mcp_client", {})
        if _flag_enabled(mcp_cfg.get("enabled")):
            self.capabilities.register(
                "mcp_client",
                required_for_readiness=_flag_enabled(mcp_cfg.get("required")),
            )
            self.capabilities.mark_initializing("mcp_client")
            try:
                from ..mcp.client import NoraxMCPClient

                self.mcp_client = NoraxMCPClient()
                mcp_client = self.mcp_client
                mcp_client.set_health_handler(self._record_mcp_health)
                connections = []
                servers = mcp_cfg.get("servers") or []
                if not isinstance(servers, list) or not servers:
                    raise ValueError("enabled MCP client has no configured servers")
                for server_cfg in servers:
                    if not isinstance(server_cfg, dict):
                        raise ValueError("each MCP server must be an object")
                    if server_cfg.get("transport") != "stdio":
                        raise ValueError(
                            f"unsupported MCP transport: {server_cfg.get('transport')!r}"
                        )
                    command = str(server_cfg.get("command") or "").strip()
                    if not command:
                        raise ValueError("MCP stdio server requires a command")
                    sname = str(server_cfg.get("name") or "mcp-server")
                    connections.append(
                        mcp_client.connect_stdio(
                            sname,
                            command,
                            *(str(arg) for arg in (server_cfg.get("args") or [])),
                        )
                    )
                await asyncio.wait_for(asyncio.gather(*connections), timeout=30.0)
                self.capabilities.mark_ok("mcp_client")
                for name, owner_task in mcp_client.connection_tasks().items():
                    self._track_mcp_connection_task(mcp_client, name, owner_task)
                self._record_mcp_health(mcp_client.health())
                log.info("mcp_client ready: %d server(s)", len(connections))
            except Exception as e:
                if self.mcp_client is not None:
                    try:
                        await self.mcp_client.disconnect_all()
                    except Exception:  # noqa: BLE001
                        log.warning("mcp_client.startup_cleanup_failed", exc_info=True)
                    self.mcp_client = None
                self.capabilities.mark_failed("mcp_client", e)
                if mcp_cfg.get("required"):
                    raise RuntimeError("required MCP client failed to start") from e
        # A2A server: agent-to-agent HTTP server
        a2a_cfg = self._connectors.get("a2a_server", {})
        if _flag_enabled(a2a_cfg.get("enabled")):
            self.capabilities.register(
                "a2a_server",
                required_for_readiness=_flag_enabled(a2a_cfg.get("required")),
            )
            self.capabilities.mark_initializing("a2a_server")
            a2a_server = None
            a2a_task: asyncio.Task[Any] | None = None
            try:
                import uvicorn

                from ..a2a.server import NoraxA2AServer, create_a2a_app

                a2a_host = str(a2a_cfg.get("host", "127.0.0.1")).strip()
                a2a_port = _bounded_config_int(
                    a2a_cfg.get("port", 8766),
                    label="A2A port",
                    minimum=1,
                    maximum=65_535,
                )
                if not a2a_host:
                    raise ValueError("A2A host must not be empty")
                a2a_token = str(
                    os.environ.get("NORAX_A2A_TOKEN") or a2a_cfg.get("auth_token") or ""
                ).strip()
                if not _is_loopback_host(a2a_host) and not a2a_token:
                    raise ValueError("A2A bearer authentication is required for non-loopback binds")
                max_tasks = _bounded_config_int(
                    a2a_cfg.get("max_tasks", 1_000),
                    label="A2A max_tasks",
                    minimum=1,
                    maximum=100_000,
                )
                max_concurrent = _bounded_config_int(
                    a2a_cfg.get("max_concurrent", 4),
                    label="A2A max_concurrent",
                    minimum=1,
                    maximum=128,
                )
                advertised_url = _a2a_advertised_url(
                    a2a_host,
                    a2a_port,
                    a2a_cfg.get("base_url"),
                )

                a2a_server = NoraxA2AServer(
                    runtime=self,
                    base_url=advertised_url,
                    auth_token=a2a_token or None,
                    max_tasks=max_tasks,
                    max_concurrent=max_concurrent,
                )
                a2a_app = create_a2a_app(a2a_server)
                self.a2a_server = a2a_server
                a2a_config = uvicorn.Config(
                    a2a_app,
                    host=a2a_host,
                    port=a2a_port,
                    log_level="warning",
                )
                a2a_uvicorn = uvicorn.Server(a2a_config)
                self._a2a_uvicorn = a2a_uvicorn
                a2a_task = asyncio.create_task(a2a_uvicorn.serve(), name="a2a-server")
                self._optional_tasks.append(a2a_task)
                await self._await_uvicorn_started(a2a_uvicorn, a2a_task)
                self._track_optional_task("a2a_server", a2a_task)
                self.capabilities.mark_ok("a2a_server")
                log.info("a2a_server started on %s:%d", a2a_host, a2a_port)
            except Exception as e:
                if a2a_task is not None and not a2a_task.done():
                    a2a_task.cancel()
                    await asyncio.gather(a2a_task, return_exceptions=True)
                if a2a_server is not None:
                    await a2a_server.close()
                self.a2a_server = None
                self._a2a_uvicorn = None
                self.capabilities.mark_failed("a2a_server", e)
                if a2a_cfg.get("required"):
                    raise RuntimeError("required A2A server failed to start") from e
        try:
            async for env in self.ingress.stream():
                if self._draining:
                    return
                self._last_turn_time = time.time()
                self._last_user_turn_time = self._last_turn_time
                chan_id = self._channel_id(env)
                body = (env.body or "").strip().lower()

                # Cancellation belongs to the owner-gated command handler.
                # Doing it before authorization let an untrusted /stop abort
                # another user's active turn.
                if body in ("/stop", "!stop"):
                    await self._handle_turn(env)
                    continue

                if not self._queue_turn(chan_id, env):
                    await self._send_command_reply(
                        env,
                        "Norax is at its bounded turn capacity. Try again shortly.",
                        reply_to=env.message_id,
                    )
        except asyncio.CancelledError:
            log.info("runtime loop cancelled")
            for task in (
                self._sleep_task,
                self._probe_task,
                self._warmup_task,
                self._capability_probe_task,
                self._autonomy_task,
            ):
                if task is not None:
                    task.cancel()
            for task in getattr(self, "_optional_tasks", []):
                task.cancel()
            raise

    async def _capability_probe_loop(self) -> None:
        """Continuously refresh operational capability evidence."""
        include_interactive = True
        while True:
            try:
                await self._probe_operational_tools(include_interactive=include_interactive)
                self.capabilities.mark_ok("operational_probe_loop")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.capabilities.mark_degraded("operational_probe_loop", str(exc)[:500])
                log.warning("operational_capability_probe.failed: %r", exc)
            # Browser launch and desktop capture are genuine but heavyweight
            # and potentially disruptive. Prove them once at startup; normal
            # tool calls are the next source of live evidence. Their registry
            # entries become explicitly stale rather than being faked fresh.
            include_interactive = False
            await asyncio.sleep(300)

    async def _probe_operational_tools(self, *, include_interactive: bool = True) -> None:
        """Verify advertised owner capabilities against the deployed host.

        These are lightweight, non-destructive startup probes. A registered
        tool with a missing browser binary, dead display, or unavailable
        execution backend must be visible in /readyz instead of failing only
        after the model tries to use it.
        """

        async def probe(name: str, operation) -> None:
            try:
                result = await operation()
                _record_operational_probe(self.capabilities, name, result)
            except Exception as exc:  # noqa: BLE001
                self.capabilities.mark_failed(name, exc)

        async def tool_manifest() -> dict[str, Any]:
            from ..brain.policy import ALL_OWNER_TOOLS

            missing = sorted(
                set(dispatch_tools.REGISTRY) - {"memory_search"} - set(ALL_OWNER_TOOLS)
            )
            stale = sorted(set(ALL_OWNER_TOOLS) - set(dispatch_tools.REGISTRY))
            return {"ok": not missing and not stale, "missing": missing, "stale": stale}

        async def browser() -> dict[str, Any]:
            from ..dispatch.browser import probe_browser_backend

            return await probe_browser_backend(timeout=15)

        async def computer() -> dict[str, Any]:
            path = Path("/tmp/norax-startup-display-probe.png")
            try:
                return await dispatch_tools.t_computer_use(action="screenshot", path=str(path))
            finally:
                path.unlink(missing_ok=True)

        async def sandbox() -> dict[str, Any]:
            from ..dispatch.sandbox import get_sandbox_manager

            runtime = get_sandbox_manager().runtime
            if not runtime:
                return {"ok": False, "error": "no docker or podman executable found"}
            proc = await asyncio.create_subprocess_exec(
                runtime,
                "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=8.0)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return {"ok": False, "error": "docker info timed out"}
            return {
                "ok": proc.returncode == 0,
                "runtime": runtime,
                "response_observed": bool(out.strip()),
                "error": err.decode(errors="replace")[:300],
            }

        async def web_search() -> dict[str, Any]:
            import httpx

            url = os.environ.get("NORAX_SEARXNG_URL", "http://127.0.0.1:8889").rstrip("/")
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{url}/healthz")
                if 200 <= response.status_code < 300:
                    return {
                        "ok": True,
                        "backend": "searxng",
                        "status_code": response.status_code,
                    }
            except Exception:  # noqa: BLE001
                pass
            configured = bool(
                os.environ.get("NORAX_SERPER_API_KEY") or os.environ.get("NORAX_TAVILY_API_KEY")
            )
            return {
                "ok": configured,
                "backend": "api" if configured else "none",
                "degraded_reason": (
                    "local search is unavailable; configured API fallback was not live-probed"
                    if configured
                    else ""
                ),
            }

        async def remote_relay() -> dict[str, Any]:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://127.0.0.1:8765/health")
                response.raise_for_status()
                payload = response.json()
            return _remote_relay_probe_result(payload)

        operations = [
            ("tool_manifest", tool_manifest),
            ("sandbox_tool", sandbox),
            ("web_search_tool", web_search),
            ("remote_relay", remote_relay),
        ]
        if include_interactive:
            operations.extend(
                [
                    ("browser_tool", browser),
                    ("computer_tool", computer),
                ]
            )
        await asyncio.gather(*(probe(name, operation) for name, operation in operations))

    async def _warmup_caches(self) -> None:
        """Pre-load embedding caches from disk so first query is fast.

        Loads .npz caches for both LocalRetriever (canonical) and
        ExternalRetriever (sleep+intel) in parallel. Only embeds the delta
        (new/changed neurons since cache was written). If no cache exists,
        triggers a full build so subsequent queries avoid the cold-build cost.
        """
        try:
            tasks: list[Any] = []
            if self._hybrid is not None and self._hybrid.embedding is not None:
                # Pre-load the embed model into GPU memory so the first real
                # embed doesn't time out during cold load.
                embedder = self._hybrid.embedding.embedder
                if hasattr(embedder, "warmup"):
                    tasks.append(embedder.warmup())
                tasks.append(self._hybrid.ensure_index())
            if self._context_injector is not None and self._context_injector.external is not None:
                tasks.append(self._context_injector.external.refresh_if_stale())
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                log.info("embedding caches warmed up")
        except Exception as e:
            log.warning("warmup_caches.failed: %r (first query will build lazily)", e)

    async def _run_completion_probe(self) -> dict[str, Any]:
        """Probe the primary→failover serving chain with explicit evidence.

        The default transport mode performs an authenticated model-catalog
        request and generates no tokens.  Operators who need a synthetic
        inference can opt into ``NORAX_COMPLETION_PROBE_MODE=completion``.
        Results never conflate transport reachability with a completed model
        response.
        """
        from ..gateway_client import GatewayRequest

        primary = getattr(self, "_effective_model", None)
        probe_mode = getattr(self, "_completion_probe_mode", "transport")
        candidates = list(
            dict.fromkeys(
                [model for model in [primary, *getattr(self, "failover_models", [])] if model]
            )
        )
        failures: list[str] = []
        probed_routes: set[tuple[str, str]] = set()
        for model in candidates:
            t0 = time.monotonic()
            try:
                provider = "?"
                route_url = "?"
                route_for = getattr(self.gateway, "route_for", None)
                if callable(route_for):
                    provider, route_url = route_for(model)

                if probe_mode == "transport":
                    route_key = (str(provider), str(route_url))
                    if route_key in probed_routes:
                        continue
                    probed_routes.add(route_key)
                    transport_probe = getattr(self.gateway, "transport_probe", None)
                    if not callable(transport_probe):
                        raise RuntimeError("gateway does not implement a transport probe")
                    evidence = await transport_probe(model, timeout=5.0)
                    latency_ms = (time.monotonic() - t0) * 1000
                    if not _explicit_result_ok(evidence):
                        failures.append(f"{model}: {str(evidence)[:240]}")
                        continue
                    return {
                        "ok": True,
                        "enabled": True,
                        "mode": "transport",
                        "transport_verified": True,
                        "completion_verified": False,
                        "model": model,
                        "provider": evidence.get("provider", provider),
                        "fallback": model != primary,
                        "latency_ms": round(latency_ms, 1),
                        "evidence": {
                            key: value
                            for key, value in evidence.items()
                            if key in {"endpoint", "kind", "status_code"}
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

                req = GatewayRequest(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Reply with exactly OK and nothing else.",
                        }
                    ],
                    # Disable private reasoning for this binary liveness task.
                    # A 32-token ceiling accommodates provider wrappers while
                    # the expected visible completion remains only two tokens.
                    max_tokens=32,
                    temperature=0.0,
                    metadata={"reasoning_effort": "none", "synthetic_probe": True},
                )
                resp = await self.gateway.chat(req)
                latency_ms = (time.monotonic() - t0) * 1000
                observed = resp.content.strip()
                if observed != "OK":
                    digest = hashlib.sha256(observed.encode("utf-8")).hexdigest()[:12]
                    failures.append(
                        f"{model}: unexpected response chars={len(observed)} sha256={digest}"
                    )
                    continue
                return {
                    "ok": True,
                    "enabled": True,
                    "mode": "completion",
                    "transport_verified": True,
                    "completion_verified": True,
                    "model": model,
                    "provider": provider,
                    "fallback": model != primary,
                    "latency_ms": round(latency_ms, 1),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{model}: {type(exc).__name__}: {exc}"[:300])
        return {
            "ok": False,
            "enabled": True,
            "mode": probe_mode,
            "transport_verified": False,
            "completion_verified": False,
            "model": primary or "?",
            "failures": failures,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _completion_probe_defer_reason(self, *, now: float | None = None) -> str | None:
        """Return why a synthetic inference should yield to user traffic."""
        if getattr(self, "_completion_probe_mode", "transport") != "completion":
            return None
        if getattr(self, "_turn_work_count", 0) > 0:
            return "queued_or_active_turn"
        active = getattr(self, "_active_turn_tasks", {})
        if any(not task.done() for task in active.values()):
            return "active_turn"
        last_user_turn = getattr(self, "_last_user_turn_time", None)
        if last_user_turn is None:
            return None
        idle_seconds = (time.time() if now is None else now) - last_user_turn
        if idle_seconds < getattr(self, "_completion_probe_idle_seconds", 30):
            return "recent_turn"
        return None

    @staticmethod
    def _completion_probe_age(result: object, *, now: datetime | None = None) -> float:
        if not isinstance(result, dict):
            return float("inf")
        try:
            timestamp = datetime.fromisoformat(str(result["timestamp"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            return float("inf")
        current = now or datetime.now(UTC)
        try:
            return max(0.0, (current - timestamp).total_seconds())
        except TypeError:
            return float("inf")

    async def _completion_probe_loop(self) -> None:
        """Periodically verify serving without needlessly competing with turns."""
        probe_interval = getattr(self, "_completion_probe_interval_seconds", 600)
        busy_retry = 15
        max_deferral = 840
        while True:
            try:
                defer_reason = self._completion_probe_defer_reason()
                probe_age = self._completion_probe_age(getattr(self, "_last_probe_result", None))
                if defer_reason is not None and probe_age < max_deferral:
                    log.debug(
                        "completion_probe.deferred reason=%s evidence_age=%.1fs",
                        defer_reason,
                        probe_age,
                    )
                    await asyncio.sleep(min(busy_retry, max(1.0, max_deferral - probe_age)))
                    continue
                self._last_probe_result = await self._run_completion_probe()
                if self._last_probe_result.get("ok") is True:
                    self.capabilities.mark_ok("serving_probe_loop")
                    log.debug(
                        "completion_probe.ok model=%s latency=%sms fallback=%s",
                        self._last_probe_result["model"],
                        self._last_probe_result["latency_ms"],
                        self._last_probe_result["fallback"],
                    )
                else:
                    failures = "; ".join(self._last_probe_result.get("failures", []))
                    self.capabilities.mark_failed(
                        "serving_probe_loop",
                        RuntimeError(failures or "serving probe failed"),
                    )
                    log.warning(
                        "completion_probe.failed failures=%s",
                        self._last_probe_result.get("failures", []),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._last_probe_result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                self.capabilities.mark_failed("serving_probe_loop", exc)
                log.warning("completion_probe.failed error=%r", exc)
            await asyncio.sleep(probe_interval)

    async def _idle_sleep_loop(self) -> None:
        """Run sleep consolidation when idle for 5+ minutes."""
        IDLE_THRESHOLD = 300  # 5 minutes
        CHECK_INTERVAL = 120  # check every 2 minutes
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                idle_sec = time.time() - self._last_turn_time
                if idle_sec < IDLE_THRESHOLD:
                    continue
                if self._memory_store is None:
                    continue
                # Check if there's anything in sleep/ to consolidate
                sleep_dir = self._memory_store.root / "sleep"
                if not sleep_dir.exists():
                    continue
                buffers = list(sleep_dir.glob("buffer-*.md"))
                spills = list(sleep_dir.glob("spill-*.jsonl"))
                spill_mds = list(sleep_dir.glob("spill-*.md"))
                if not buffers and not spills and not spill_mds:
                    continue
                total_files = len(buffers) + len(spills) + len(spill_mds)
                log.info(
                    "sleep_consolidation: idle %.0fs, %d files (buffers=%d spills=%d spill_mds=%d)",
                    idle_sec,
                    total_files,
                    len(buffers),
                    len(spills),
                    len(spill_mds),
                )
                try:
                    if self._memory_coordinator is None:
                        continue
                    result = await self._memory_coordinator.consolidate_sleep()
                    if result["canonical_writes"]:
                        await self._memory_coordinator.sync_projections()
                    await self.events.append("sleep_consolidation", result)
                    log.info(
                        "sleep_consolidation: files=%d spills=%d candidates=%d writes=%d duplicates=%d",
                        result["files_processed"],
                        result["spills_processed"],
                        result["candidates_seen"],
                        result["canonical_writes"],
                        result["duplicates_skipped"],
                    )
                except Exception as e:
                    log.warning("sleep_consolidation.error: %r", e)

                # Verified trajectory replay is an explicit learning feature.
                if self._idle_learning_enabled and self._episodic is not None:
                    try:
                        from ..brain.sleep.replay import HippocampalReplay

                        replay = HippocampalReplay(
                            episodic=self._episodic,
                            memory_root=self._memory_store.root,
                        )
                        replay_result = await asyncio.to_thread(replay.run)
                        if replay_result.procedural_patterns or replay_result.failure_patterns:
                            if self._memory_coordinator is not None:
                                self._memory_coordinator.canonical_changed("hippocampal_replay")
                            await self.events.append(
                                "hippocampal_replay",
                                {
                                    "patterns": len(replay_result.procedural_patterns),
                                    "failures": len(replay_result.failure_patterns),
                                    "coactivation_pairs_observed": (
                                        replay_result.coactivation_pairs_observed
                                    ),
                                    "high_surprise_episodes_seen": (
                                        replay_result.high_surprise_episodes_seen
                                    ),
                                },
                            )
                            log.info(
                                "hippocampal_replay: %d patterns, %d failures",
                                len(replay_result.procedural_patterns),
                                len(replay_result.failure_patterns),
                            )
                            # Feed replay patterns back into ToolExperienceMemory
                            # so future tool retrieval can surface verified
                            # sequences and failure recovery guidance.
                            try:
                                from ..memory.tool_experience import ToolExperienceMemory

                                tem = ToolExperienceMemory(self._memory_store.root)
                                ingested = await asyncio.to_thread(
                                    tem.ingest_replay_patterns,
                                    self._memory_store.root / "procedural",
                                )
                                if ingested:
                                    log.info(
                                        "hippocampal_replay: %d patterns ingested into tool_experience",
                                        ingested,
                                    )
                            except Exception as e:
                                log.debug("tool_experience.ingest_replay.error: %r", e)
                    except Exception as e:
                        log.debug("hippocampal_replay.error: %r", e)

                # Sprint D: Skill acquisition — mine trajectories for patterns
                if self._skill_learner is not None:
                    try:
                        from ..brain.harness_optimizer import load_trajectories

                        event_log = getattr(self.events, "path", None)
                        if event_log and isinstance(event_log, Path) and event_log.exists():
                            trajectories = load_trajectories(event_log, limit=200)
                            if trajectories:
                                patterns = self._skill_learner.mine(trajectories)
                                if patterns:
                                    sl_result = self._skill_learner.generate(patterns)
                                    if sl_result.skills_created or sl_result.skills_updated:
                                        if self._memory_coordinator is not None:
                                            self._memory_coordinator.canonical_changed(
                                                "skill_learning"
                                            )
                                        await self.events.append(
                                            "skill_learning",
                                            {
                                                "created": sl_result.skills_created,
                                                "updated": sl_result.skills_updated,
                                                "skills": sl_result.skills,
                                            },
                                        )
                                        log.info(
                                            "skill_learning: created=%d updated=%d skills=%s",
                                            sl_result.skills_created,
                                            sl_result.skills_updated,
                                            sl_result.skills,
                                        )
                    except Exception as e:
                        log.debug("skill_learning.error: %r", e)

                if self._episodic is not None:
                    try:
                        await asyncio.to_thread(self._episodic.prune_old)
                    except Exception as e:
                        log.debug("episodic.prune.error: %r", e)

                # Incrementally sync only when an upstream component reported
                # a real canonical mutation.  Unconditionally marking this
                # dirty caused a full projection pass every idle poll.
                try:
                    if self._memory_coordinator is not None:
                        await self._memory_coordinator.sync_projections()
                except Exception as e:
                    log.debug("memory_projection_sync.error: %r", e)

                # The opt-in calibration component is already loaded and
                # updated on evidence-bearing turns.  Reuse it; never create a
                # second default-on telemetry path during idle maintenance.
                try:
                    if self._metacognitive is not None:
                        report = self._metacognitive.get_report()
                    else:
                        report = None
                    if report is not None and report.is_reliable and report.is_overconfident:
                        log.warning(
                            "metacognitive: OVERCONFIDENT bias=%.2f brier=%.3f — reducing confidence",
                            report.bias_magnitude,
                            report.brier_score,
                        )
                        await self.events.append(
                            "metacognitive_alert",
                            {
                                "bias": report.bias.value,
                                "magnitude": round(report.bias_magnitude, 3),
                                "brier": round(report.brier_score, 3),
                                "trend": report.trend,
                            },
                        )
                except Exception as e:
                    log.debug("metacognitive.idle.error: %r", e)

                # Retrospective harness analysis: mine live trajectories into
                # a reviewable report. This path never calls an LLM or edits
                # source code.
                try:
                    rho_last = getattr(self, "_rho_last_run", 0.0)
                    if self._harness_analysis_enabled and time.time() - rho_last > 1800:  # 30 min
                        from ..brain.harness_optimizer import analyze_harness

                        event_log_path = getattr(self.events, "path", None)
                        if (
                            event_log_path
                            and isinstance(event_log_path, Path)
                            and event_log_path.exists()
                        ):
                            event_mtime = event_log_path.stat().st_mtime_ns
                            if event_mtime != getattr(self, "_rho_last_event_mtime_ns", None):
                                out_dir = self._memory_store.root / "state" / "harness_optimizer"
                                rho_result = await asyncio.to_thread(
                                    analyze_harness,
                                    event_log_path,
                                    out_dir,
                                    k=10,
                                    limit=200,
                                )
                                await self.events.append(
                                    "rho_analysis",
                                    {
                                        "status": rho_result.get("status"),
                                        "source_mutations": 0,
                                        "proposal_count": rho_result.get("proposal_count", 0),
                                        "report_path": rho_result.get("report_path"),
                                    },
                                )
                                log.info(
                                    "rho_analysis: status=%s proposals=%s",
                                    rho_result.get("status"),
                                    rho_result.get("proposal_count", 0),
                                )
                                self._rho_last_event_mtime_ns = event_log_path.stat().st_mtime_ns
                        self._rho_last_run = time.time()
                except Exception as e:
                    log.debug("rho_analysis.error: %r", e)

            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("idle_sleep_loop.error", exc_info=True)
                await asyncio.sleep(60)

    async def _handle_turn(self, env) -> None:
        chan_id = self._channel_id(env)
        owner_task = asyncio.current_task()
        completed = False
        try:
            with self.events.trace_scope() as trace_id:
                if isinstance(env.metadata, dict):
                    env.metadata["trace_id"] = trace_id
                self.metrics.ingress_total.labels(
                    source=env.source,
                    channel=env.channel,
                    trusted=str(env.trusted).lower(),
                ).inc()
                with self.metrics.time_brain_turn():
                    await self._handle_turn_inner(env)
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception:
            self.metrics.agent_turn_failures.inc()
            raise
        finally:
            # Own UI helpers at the turn boundary so cancellation and early
            # returns cannot leak typing keep-alives or partial messages.
            streaming = (
                self._active_stream_messages.pop(owner_task, None)
                if owner_task is not None
                else None
            )
            if not completed and streaming is not None:
                try:
                    await streaming.delete()
                except Exception:  # noqa: BLE001
                    log.debug("stream.cancel_cleanup_failed", exc_info=True)
            typing_keep = (
                self._active_typing_handles.pop(owner_task, None)
                if owner_task is not None
                else None
            )
            if typing_keep is not None:
                try:
                    await typing_keep.stop(timeout=1.5)
                except Exception:  # noqa: BLE001
                    log.debug("typing.turn_cleanup_failed", exc_info=True)
            self._stop_channels.discard(chan_id)
            # Lifecycle is runtime-owned: disconnecting a browser never
            # cancels the turn, and every terminal path clears working.
            if self.agent_os_bridge is not None:
                try:
                    await self.agent_os_bridge.broadcast_status("idle")
                except Exception:
                    log.debug("agent_os idle_failed", exc_info=True)

    async def _handle_turn_inner(self, env) -> None:
        chan_id = self._channel_id(env)
        owner_task = asyncio.current_task()
        if self.agent_os_bridge is not None:
            try:
                await self.agent_os_bridge.broadcast_status("typing")
            except Exception:
                log.debug("agent_os typing_failed", exc_info=True)
        await self.events.append(
            "ingress",
            {
                "channel": env.channel,
                "source": env.source,
                "message_id": env.message_id,
                "sender_id": env.sender.id,
                "sender_tier": env.sender.tier,
                "trusted": env.trusted,
                "body": env.body[:2000],
            },
        )

        # ---- mirror inbound to Agent OS dashboard --------------------
        if (
            self.agent_os_bridge is not None
            and env.source == "discord"
            and env.sender.tier != "system"
        ):
            try:
                await self.agent_os_bridge.broadcast_inbound(
                    sender_label=env.sender.label or env.sender.id,
                    text=env.body or "",
                    source="discord",
                )
            except Exception:  # noqa: BLE001
                log.debug("agent_os mirror_failed", exc_info=True)

        # ---- slash-command interception ------------------------------
        pc = cmd_mod.parse(env.body or "")
        if pc is not None and cmd_mod.is_known(pc):
            await self._handle_command(env, pc)
            return

        # Start Discord typing before any memory or planning work.  Agent OS
        # receives its typing state above; keeping this adjacent prevents the
        # two surfaces from drifting when retrieval is slow or degraded.
        typing_keep = None
        target_channel: str | None = None
        if (
            env.source == "discord"
            and self.discord is not None
            and self.stream_replies
            and (env.raw or {}).get("channel_id")
        ):
            target_channel = str((env.raw or {}).get("channel_id"))
            try:
                typing_keep = await self.discord.start_typing(target_channel)
                if typing_keep is not None and owner_task is not None:
                    self._active_typing_handles[owner_task] = typing_keep
            except Exception:  # noqa: BLE001
                log.debug("typing.start_failed", exc_info=True)

        # Discover memory changes; unchanged files reuse parsed neurons.
        if self._memory_store is not None:
            try:
                await asyncio.to_thread(self._memory_store.refresh)
            except Exception:  # noqa: BLE001
                log.warning(
                    "memory.refresh_failed; continuing without refreshed state", exc_info=True
                )

        retrieval_k = self._memory_k_for_turn(env.body or "", env.sender.tier)

        async def _retrieve(query: str, k: int = 5):
            """Smart context retrieval: ContextInjector first, then hybrid, then keyword."""
            k = min(max(k, retrieval_k), 24)
            if self._context_injector is not None:
                try:
                    task_type = "coding" if _is_coding_query(query) else "general"
                    injected = await self._context_injector.run(
                        query,
                        k_each=min(k, 16),
                        task_type=task_type,
                    )
                    if injected.items:
                        return [(n, score, src) for n, score, src in injected.items]
                except Exception as e:
                    log.warning("context_injector.run.error query=%r: %r", query[:60], e)
            if self._hybrid is not None:
                hits = await asyncio.to_thread(self._hybrid.search_sync, query, k=k)
                return [(n, s, tag) for n, s, tag in hits]
            if self._fast_ctx is not None:
                return await asyncio.to_thread(self._fast_ctx.search, query, k=k)
            return []

        # Inject user model into metadata before plan_turn
        if self._user_model is not None and env.sender.id:
            try:
                user_model_block = self._user_model.render_for_prompt(env.sender.id)
            except Exception:  # noqa: BLE001
                user_model_block = ""
        else:
            user_model_block = ""

        try:
            ctx, rendered = await hot_path.plan_turn(
                env,
                runtime_info={
                    "model": self.default_model,
                    "channel": env.channel,
                    "capabilities": "agentic",
                },
                retrieve=_retrieve,
                memory_root=self._memory_root,
                user_model_block=user_model_block,
                soul=self._soul,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("brain plan failed")
            self.metrics.brain_errors.labels(where="brain").inc()
            self.metrics.agent_turn_failures.inc()
            await self.events.append("error", {"where": "brain.plan", "err": repr(e)})
            await self._send_command_reply(
                env,
                _turn_failure_message(e),
                reply_to=env.message_id,
            )
            return

        self.metrics.brain_turns.labels(decision=ctx.decision).inc()
        ch = env.channel or "default"
        prev_static = self._last_static_hash.get(ch)
        if prev_static and prev_static != rendered.static_hash:
            self.metrics.prompt_cache_breaks.labels(channel=ch).inc()
            now = time.time()
            cache_window = [t for t in self._cache_break_times.get(ch, []) if now - t < 300.0]
            cache_window.append(now)
            self._cache_break_times[ch] = cache_window
            self.metrics.prompt_cache_breaks_window.labels(channel=ch).set(len(cache_window))
            log.info(
                "prompt_cache_break channel=%s old=%s new=%s window_5m=%d",
                ch,
                prev_static,
                rendered.static_hash,
                len(cache_window),
            )
            if len(cache_window) > 2:
                log.warning(
                    "prompt_cache_break_alert channel=%s breaks_in_5m=%d static_hash=%s",
                    ch,
                    len(cache_window),
                    rendered.static_hash,
                )
        self._last_static_hash[ch] = rendered.static_hash
        tools_key = hashlib.sha256(",".join(sorted(ctx.allowed_tools or [])).encode()).hexdigest()[
            :16
        ]
        prev_tools = self._last_tools_hash.get(ch)
        if prev_tools and prev_tools != tools_key:
            self.metrics.prompt_tools_changes.labels(channel=ch).inc()
            log.info(
                "prompt_tools_change channel=%s old=%s new=%s tools=%s",
                ch,
                prev_tools,
                tools_key,
                ctx.allowed_tools,
            )
        self._last_tools_hash[ch] = tools_key
        await self.events.append(
            "brain",
            {
                "decision": ctx.decision,
                "static_hash": rendered.static_hash,
                "tools": ctx.allowed_tools,
                "focus": ctx.focus.summary,
            },
            attrs={
                "gen_ai.system": "gateway",
                "gen_ai.request.model": self.default_model,
            },
        )

        # Streaming setup (Discord only for now).  Typing already started
        # before memory/planning; now prepare the deferred streaming message.
        # If anything fails, we fall back to the classic _emit_reply path.
        # ------------------------------------------------------------------
        streaming_msg = None
        if target_channel is not None:
            try:
                streaming_msg = await self.discord.begin_streaming_message(
                    target_channel,
                    reply_to=env.message_id,
                    defer_initial=True,
                )
                if streaming_msg is not None and owner_task is not None:
                    self._active_stream_messages[owner_task] = streaming_msg
            except Exception:  # noqa: BLE001
                log.debug("stream.begin_failed", exc_info=True)

        if ctx.decision != "emit_reply":
            # Clean up typing before returning early
            if typing_keep is not None:
                try:
                    await typing_keep.stop(timeout=1.5)
                except Exception:  # noqa: BLE001
                    pass
            if streaming_msg is not None:
                try:
                    await streaming_msg.delete()
                except Exception:  # noqa: BLE001
                    pass
            return
        stream_delta_buf: list[str] = []

        _turn_t0 = time.monotonic()

        async def _on_stream_delta(delta: str) -> None:
            stream_delta_buf.append(delta)
            if streaming_msg is not None:
                await streaming_msg.append(delta)

        on_delta = _on_stream_delta if streaming_msg is not None else None

        # Build session history from the rolling window.
        window = self._get_window(chan_id)
        # Record user message in the window.
        turn_id = window.start_turn()
        # For image-only messages (empty body + attachments), store a
        # meaningful placeholder so the window context isn't blank.
        user_text_for_window = env.body or ""
        if not user_text_for_window and (env.attachments or []):
            user_text_for_window = "(image attached)"
        window.add_user(user_text_for_window, turn_id=turn_id)
        # Evict old frames if over budget.
        window.evict_if_needed()
        # Collect prior conversation turns (excluding the current user
        # message). Allocate by the selected model's context capacity: four
        # one-line turns are cheap, while one pasted log can be enormous.
        candidate_prior_turns = _candidate_prior_turn_ids(window.body[:-1], user_text_for_window)
        provider_kind = "unknown"
        provider_kind_for_model = getattr(self.gateway, "provider_kind_for_model", None)
        if callable(provider_kind_for_model):
            try:
                provider_kind = str(provider_kind_for_model(self._effective_model) or "unknown")
            except Exception:  # noqa: BLE001
                log.debug("history.provider_kind_resolution_failed", exc_info=True)
        history_budget = _history_budget_for_model(
            self._effective_model,
            provider_kind=provider_kind,
            action_request=agent_loop.is_action_command(user_text_for_window),
        )
        protected_prior_turns, selected_history_calls, history_text_cap = _select_history_tail(
            window.body[:-1],
            candidate_prior_turns,
            token_budget=history_budget,
        )

        # Build prior messages, grouping same-turn tool_calls into a single
        # assistant message (Claude API requires all parallel tool uses to share
        # one assistant message, otherwise IDs appear duplicate → 400 error).
        #
        # Single-pass design: collect tool_call/tool_result by turn_id, then
        # emit them in body order when the first tool_call for each turn is
        # encountered. User/assistant text frames pass through directly.
        prior: list[dict] = []
        _pending_tool_calls: dict[int, list] = {}  # turn_id -> list of tool_call dicts
        _pending_tool_results: dict[int, list] = {}  # turn_id -> list of tool_result dicts
        _seen_call_ids: set[str] = set()
        _emitted_tool_turns: set[int] = set()

        # Pass 1: collect all tool calls and results grouped by turn_id
        for f in window.body[:-1]:
            if f.turn_id not in protected_prior_turns:
                continue
            if f.kind == "tool_call" and f.call_id:
                if f.call_id not in selected_history_calls:
                    continue
                if f.call_id in _seen_call_ids:
                    continue
                _seen_call_ids.add(f.call_id)
                _pending_tool_calls.setdefault(f.turn_id, []).append(
                    {
                        "id": f.call_id,
                        "type": "function",
                        "function": {
                            "name": f.meta.get("name", "unknown"),
                            "arguments": _coerce_tool_args_json(f.content),
                        },
                    }
                )
            elif f.kind == "tool_result" and f.call_id:
                if f.call_id not in selected_history_calls:
                    continue
                if f.call_id not in _seen_call_ids:
                    continue
                _pending_tool_results.setdefault(f.turn_id, []).append(
                    {
                        "role": "tool",
                        "tool_call_id": f.call_id,
                        "content": _bounded_history_text(f.content, 2_000),
                    }
                )

        # Pass 2: emit messages in body order, flushing tool groups atomically
        for f in window.body[:-1]:
            if f.turn_id not in protected_prior_turns:
                continue
            if f.kind == "user":
                prior.append(
                    {"role": "user", "content": _bounded_history_text(f.content, history_text_cap)}
                )
            elif f.kind == "assistant":
                prior.append(
                    {
                        "role": "assistant",
                        "content": _bounded_history_text(
                            _clean_historical_assistant_text(f.content), history_text_cap
                        ),
                    }
                )
            elif f.kind == "tool_call" and f.call_id:
                if f.call_id not in selected_history_calls:
                    continue
                if f.turn_id in _emitted_tool_turns:
                    continue
                _emitted_tool_turns.add(f.turn_id)
                all_calls = _pending_tool_calls.get(f.turn_id, [])
                if all_calls:
                    prior.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": all_calls,
                        }
                    )
                    for res in _pending_tool_results.get(f.turn_id, []):
                        prior.append(res)
            elif f.kind == "tool_result":
                pass  # already emitted above

        # Build correction gate (L32) — non-LLM fact checker against memory.
        # Uses hybrid retriever (keyword + embedding) if available.
        correction_gate = None
        try:
            from ..context import CorrectionGate

            hybrid_retriever = self._hybrid
            fast_retriever = self._fast_ctx
            if hybrid_retriever is not None:

                async def _gate_retrieve(query, k=5):
                    # Post-draft fact checks use the already-warmed local
                    # keyword/graph/FTS signals. Embedding and reranker calls
                    # add seconds to every response without improving exact
                    # identifier/KV comparisons.
                    return await asyncio.to_thread(hybrid_retriever.search_sync, query, k=k)
            elif fast_retriever is not None:

                async def _gate_retrieve(query, k=5):
                    return fast_retriever.search(query, k=k)

            else:
                raise RuntimeError("no memory retriever is available for correction checks")

            correction_gate = CorrectionGate(retrieve=_gate_retrieve)
            self.capabilities.mark_ok("correction_gate")
        except Exception as e:  # noqa: BLE001
            self.capabilities.mark_failed("correction_gate", e)

        # --- Cognitive amplifiers and zero-call signal engines ---
        self._ensure_cognitive_components()
        output_verifier = self._output_verifier
        self_model = self._self_model
        active_inference = self._active_inference
        metacognitive = self._metacognitive
        best_of_n = self._best_of_n
        curiosity = self._curiosity
        domain_transfer = self._domain_transfer
        analogy_engine = self._analogy_engine

        # Pre-pass: compute the cognitive signals for THIS turn and fold the
        # hints into the system prompt before the loop runs.
        _cog_signal: dict[str, Any] = {}
        _turn_domain = "general"
        try:
            from ..brain.self_model import classify_domain

            _turn_text = env.body or ""
            _turn_domain = classify_domain(_turn_text)
            if curiosity is not None and _turn_text:
                _cur = curiosity.assess(_turn_text, domain=_turn_domain)
                _cog_signal["novelty"] = _cur.novelty
                if _cur.explore_hint:
                    rendered.system = f"{rendered.system}\n\n{_cur.explore_hint}"
                if _cur.novelty >= 0.40:
                    await self.events.append(
                        "curiosity_signal",
                        {
                            "novelty": _cur.novelty,
                            "surprise": _cur.surprise,
                            "new_domain": _cur.is_new_domain,
                            "new_entities": _cur.is_new_entity_mix,
                        },
                    )
            if domain_transfer is not None and _turn_text:
                _dt_hints = domain_transfer.transfer_hints(_turn_domain, _turn_text)
                if _dt_hints:
                    _cog_signal["transfers"] = len(_dt_hints)
                    rendered.system = (
                        f"{rendered.system}\n\n"
                        "TRANSFERRED TACTICS (proven in other domains, apply here):\n"
                        + "\n".join(f"- {h}" for h in _dt_hints)
                    )
            if analogy_engine is not None and _turn_text:
                _analogs = analogy_engine.find_analogies(
                    _turn_text,
                    task_type=_turn_domain,
                    limit=3,
                )
                _block = analogy_engine.render_hints(_analogs)
                if _block:
                    _cog_signal["analogies"] = len(_analogs)
                    rendered.system = f"{rendered.system}\n\n{_block}"
        except Exception as e:  # noqa: BLE001
            log.debug("cognitive_signals.error: %r", e)

        # Run the agent loop: L10 + tool-calls + iteration.
        # Initialise these before the optional multi-agent branch so a false
        # trigger (the normal case) cannot leave the single-agent gate reading
        # an unbound local.
        resp = None
        trace: list[dict[str, Any]] = []
        rounds = 0
        _task_state = None
        _restored_task_state: dict[str, Any] | None = None
        _restored_prior_status: str = "in_progress"

        # Offer the last task_state to build_task_state for crash recovery.
        # That builder restores it only when the new prompt is a confident
        # continuation; unrelated tasks receive a clean state.
        try:
            from ..runtime.checkpoint import load_latest_checkpoint

            _cp = load_latest_checkpoint(chan_id, memory_root=self._memory_root)
            if _cp and _cp.get("task_state"):
                _cp_status = _cp.get("status", "in_progress")
                _cp_task = _cp.get("task_state", {})
                _restored_task_state = _cp_task
                _restored_prior_status = _cp_status
                log.debug(
                    "checkpoint.restore channel=%s status=%s pending=%d completed=%d",
                    chan_id,
                    _cp_status,
                    len(_cp_task.get("pending_steps", [])),
                    len(_cp_task.get("completed_actions", [])),
                )
        except Exception as e:  # noqa: BLE001
            log.debug("checkpoint.restore_failed: %r", e)
        # Optional multi-agent trigger. Heuristic decomposition adds model calls
        # and can change task semantics, so it is explicitly opt-in.
        _multi_agent_started = False
        _multi_agent_execution_error: Exception | None = None
        try:
            if self._multi_agent_enabled and bool(
                getattr(self.autonomy_config, "multi_agent_auto", False)
            ):
                from .autonomy import should_trigger_multi_agent

                should_decompose = should_trigger_multi_agent(
                    env.body or "", list(ctx.allowed_tools or [])
                )
            else:
                should_decompose = False

            if should_decompose:
                from ..brain.multi_agent import MultiAgentOrchestrator
                from ..gateway_client import GatewayResponse

                log.info("multi_agent: auto-decomposing task")
                mao = MultiAgentOrchestrator(self.gateway, default_model=self.default_model)
                _multi_agent_started = True
                ma_result = await mao.run(
                    task=env.body or "",
                    system_prompt=rendered.system,
                    allowed_tools=list(ctx.allowed_tools or []),
                    sender_tier=env.sender.tier,
                    event_log=self.events,
                    max_concurrent=self._multi_agent_max_concurrent,
                )
                await self.events.append(
                    "multi_agent",
                    {
                        "subtasks": len(ma_result.sub_results),
                        "verified_outcome": ma_result.verified_outcome,
                        "failed_subtasks": sum(
                            1 for result in ma_result.sub_results if not result.success
                        ),
                        "total_tool_calls": ma_result.total_tool_calls,
                        "elapsed": round(ma_result.elapsed, 2),
                    },
                )
                resp = GatewayResponse(
                    request_id="",
                    model=self.default_model,
                    content=ma_result.summary,
                    tool_calls=[],
                    usage=dict(ma_result.usage),
                    raw={
                        "multi_agent": True,
                        "subtasks": len(ma_result.sub_results),
                        "verified_outcome": ma_result.verified_outcome,
                        "incomplete": not ma_result.verified_outcome,
                        "status_reason": (
                            "all_subtasks_verified"
                            if ma_result.verified_outcome
                            else "one_or_more_subtasks_unverified"
                        ),
                        "usage_scope": "turn_total",
                    },
                )
                trace = []
                for sr in ma_result.sub_results:
                    for tc in sr.tool_calls or []:
                        trace.append(
                            tc
                            if isinstance(tc, dict)
                            else {"name": str(tc), "result": {"ok": sr.success}}
                        )
                rounds = sum(r.rounds for r in ma_result.sub_results)
        except SpendGuardTripped as e:
            _multi_agent_execution_error = e
        except Exception as e:
            if _multi_agent_started:
                _multi_agent_execution_error = e
            else:
                log.debug("multi_agent.trigger_failed_before_execution: %r", e)

        try:
            if _multi_agent_execution_error is not None:
                raise _multi_agent_execution_error
            from ..brain.orchestrator import Orchestrator
            from ..gateway_client import GatewayResponse

            # A successful multi-agent run is already the execution result.
            # Do not immediately overwrite it with the single-agent path.
            if resp is None and self._planner_enabled and self.planning_mode == "orchestrator":
                from ..brain.orchestrator import EXECUTOR_MODEL
                from ..brain.strong_model_scaffold import resolve_planner_model

                planner = resolve_planner_model(self.default_model)
                log.info(
                    "orchestrator: planner=%s executor=%s",
                    planner,
                    EXECUTOR_MODEL,
                )
                orch = Orchestrator(self.gateway)
                orch_result = await orch.run(
                    system_prompt=rendered.system,
                    user_prompt=rendered.user,
                    allowed_tools=list(ctx.allowed_tools or []),
                    sender_tier=env.sender.tier,
                    event_log=self.events,
                    on_delta=on_delta,
                    max_rounds=self.max_tool_rounds,
                    planner_model=planner,
                )
                content, trace, rounds = orch_result
                resp = GatewayResponse(
                    request_id="",
                    model=self.default_model,
                    content=content or "",
                    tool_calls=[],
                    usage=dict(orch_result.usage),
                    raw={
                        "orchestrator": True,
                        "rounds": rounds,
                        "planner": orch_result.planner_model,
                        "executor": EXECUTOR_MODEL,
                        "session_model": self.default_model,
                        "verified_outcome": orch_result.verified_outcome,
                        "incomplete": not orch_result.complete,
                        "status_reason": orch_result.status_reason,
                        "completion_signal": orch_result.completion_signal,
                        "mutation_outcome": orch_result.mutation_outcome,
                        "blocking_categories": orch_result.blocking_categories,
                        "usage_scope": "turn_total",
                    },
                )
            elif resp is None:
                _tools = list(ctx.allowed_tools or [])
                # Inject MCP tools if the client is connected
                _mcp = getattr(self, "mcp_client", None)
                if _mcp is not None and env.sender.tier == "owner":
                    try:
                        _mcp_tools = await _mcp.aggregate_tools()
                        for _t in _mcp_tools:
                            _tools.append(f"mcp_{_t['name']}")
                    except Exception as e:
                        log.warning("mcp.aggregate_tools_failed: %r", e)
                # Vision: switch to VLM when image attachments are present
                _turn_model = self.default_model
                vision_enabled = _flag_enabled(self._vision_config.get("enabled"))
                if vision_enabled and (env.attachments or []):
                    _vlm = self._vision_config.get("model") or ""
                    if _vlm:
                        _turn_model = _vlm
                        log.info(
                            "vision.model_switch attachments=%d model=%s",
                            len(env.attachments or []),
                            _vlm,
                        )
                        self.capabilities.mark_ok("vision")
                    else:
                        self.capabilities.mark_degraded("vision", "no model configured")
                elif vision_enabled:
                    self.capabilities.mark_ok("vision")
                resp, trace, rounds, _task_state = await agent_loop.run_agent_loop(
                    gateway=self.gateway,  # type: ignore[arg-type]
                    model=_turn_model,
                    system_prompt=rendered.system,
                    user_prompt=rendered.user,
                    allowed_tools=_tools,
                    sender_tier=env.sender.tier,
                    event_log=self.events,
                    on_delta=on_delta,
                    prior_messages=prior,
                    correction_gate=correction_gate,
                    stop_check=lambda: chan_id in self._stop_channels,
                    reasoning_effort=self.thinking_effort,
                    reasoning_output=self.reasoning_output,
                    max_rounds=self.max_tool_rounds,
                    weak_model_boost=self.weak_model_boost,
                    response_length=self.response_length,
                    tool_activity=self.tool_activity,
                    output_verifier=output_verifier,
                    self_model=self_model,
                    active_inference=active_inference,
                    metacognitive=metacognitive,
                    best_of_n=best_of_n,
                    failover_models=self.failover_models,
                    mcp_client=getattr(self, "mcp_client", None),
                    memory_root=self._memory_root,
                    initial_task_state=_restored_task_state,
                    initial_task_status=_restored_prior_status,
                    defer_outcome_recording=True,
                )
        except Exception as e:  # noqa: BLE001
            log.exception("agent.turn_failed")
            self.metrics.brain_errors.labels(where="agent_loop").inc()
            self.metrics.agent_turn_failures.inc()
            await self.events.append(
                "turn_failure", {"error": str(e)[:500], "model": self.default_model}
            )
            await self.events.append("error", {"where": "agent_loop", "err": repr(e)})
            user_msg = _turn_failure_message(e)
            # Clean up typing/placeholder if they were created.
            if typing_keep is not None:
                try:
                    await typing_keep.stop(timeout=1.5)
                except Exception:  # noqa: BLE001
                    pass
            if streaming_msg is not None:
                try:
                    await streaming_msg.finalize(user_msg)
                except Exception:  # noqa: BLE001
                    pass
            elif env.source == "discord" and self.discord is not None:
                # No streaming message was set up — send the failure reply
                # directly to the channel so the user isn't left in silence.
                try:
                    ch_id = (env.raw or {}).get("channel_id")
                    if ch_id:
                        await self.discord.send(str(ch_id), user_msg)
                except Exception:  # noqa: BLE001
                    log.debug("agent.turn_failed_reply_send_failed", exc_info=True)
            else:
                # HTTP, Agent OS, and any future ingress adapters must receive
                # the same terminal failure signal.  Previously this branch
                # only logged the exception, leaving non-Discord callers
                # waiting for a response that could never arrive.
                await self._send_command_reply(env, user_msg, reply_to=env.message_id)
                return
            await self._mirror_discord_reply(env, user_msg)
            return

        # Stop typing as soon as we have a final response — finalize will
        # send/edit the actual content next.
        if typing_keep is not None:
            try:
                await typing_keep.stop(timeout=1.5)
            except Exception:  # noqa: BLE001
                pass

        # Record tool calls + results in the rolling window so future turns
        # have agentic context (what actions were taken and their outcomes).
        if trace:
            # Successful real tool use is stronger freshness evidence than a
            # synthetic heartbeat. This is an in-memory dictionary update and
            # does not add another browser, desktop, network, or sandbox probe.
            _refresh_tool_capability_evidence(self.capabilities, trace)
            for _t_idx, t in enumerate(trace[:20]):  # cap at 20 to preserve more agentic context
                call_id = f"fc_{turn_id}_{_t_idx}"
                import json as _json

                window.add_tool_call(
                    call_id=call_id,
                    content=_json.dumps(t.get("args", {}), default=str)[:800],
                    turn_id=turn_id,
                    name=t["name"],
                )
                result_preview = _json.dumps(t.get("result", {}), default=str)[:1200]
                window.add_tool_result(
                    call_id=call_id,
                    content=result_preview,
                    turn_id=turn_id,
                )

        # Deliver as soon as generation finishes. Window/checkpoint persistence
        # and telemetry happen afterward so local disk latency is never added to
        # the visible response tail.
        if resp is not None:
            try:
                delivery_outcome = await self._deliver_turn_response(
                    env=env,
                    ctx=ctx,
                    resp=resp,
                    trace=trace,
                    rounds=rounds,
                    streaming_msg=streaming_msg,
                    stream_delta_buf=stream_delta_buf,
                    target_channel=target_channel,
                )
            except Exception as error:  # noqa: BLE001
                log.warning("reply.delivery_pipeline_failed error=%r", error, exc_info=True)
                delivery_outcome = {
                    "ok": False,
                    "state": "delivery_pipeline_failed",
                    "attempted": True,
                    "error": f"{type(error).__name__}: {error}"[:500],
                }
            resp.raw = dict(resp.raw or {})
            resp.raw["delivery_succeeded"] = delivery_outcome.get("ok") is True
            resp.raw["delivery_state"] = delivery_outcome.get("state", "unknown")
            if delivery_outcome.get("ok") is not True:
                self.metrics.agent_turn_failures.inc()

            # A bounded delivery attempt is complete. Persistence and learning
            # must not retain scarce global execution capacity or typing state.
            self._release_current_turn_slot()
            if owner_task is not None:
                self._active_stream_messages.pop(owner_task, None)
                delivered_typing = self._active_typing_handles.pop(owner_task, None)
                if delivered_typing is not None:
                    try:
                        await delivered_typing.stop(timeout=1.5)
                    except Exception:  # noqa: BLE001
                        log.debug("typing.delivery_cleanup_failed", exc_info=True)

            # Only an actually delivered assistant message enters conversation
            # history. Suppressed or failed output must not become false shared
            # context on the next turn.
            if delivery_outcome.get("state") == "delivered" and resp.content:
                _clean = resp.content
                try:
                    from ..adapter.reply_tag import parse_reply_tag

                    _clean = parse_reply_tag(_clean).text
                except Exception:  # noqa: BLE001
                    pass
                _clean = _clean_historical_assistant_text(_clean)
                if _clean.strip():
                    window.add_assistant(_clean, turn_id=turn_id)

            # Persist history after delivery without blocking unrelated channel
            # work on serialization or filesystem I/O.
            try:

                def compact_and_persist_window() -> None:
                    window.compact_solved_turns()
                    self._persist_window(chan_id)

                await asyncio.to_thread(compact_and_persist_window)
            except Exception as error:  # noqa: BLE001
                log.warning("window_compaction.failed channel=%s error=%r", chan_id, error)

            # The checkpoint records delivery truth plus a bounded, hashed
            # response preview for operator/restart recovery when routing failed.
            try:
                from ..runtime.checkpoint import save_checkpoint

                _cp_status = (
                    _checkpoint_status(_task_state, resp)
                    if delivery_outcome.get("ok") is True
                    else "delivery_failed"
                )
                await asyncio.to_thread(
                    save_checkpoint,
                    channel=chan_id,
                    turn_id=str(turn_id),
                    messages=prior,
                    trace=trace or [],
                    task_state=_task_state.to_persistable_dict() if _task_state else None,
                    rounds=rounds,
                    model=resp.model or self.default_model,
                    status=_cp_status,
                    response={
                        "content_preview": (resp.content or "")[:2_000],
                        "content_chars": len(resp.content or ""),
                        "content_sha256": hashlib.sha256((resp.content or "").encode()).hexdigest(),
                        "model": resp.model or self.default_model,
                        "request_id": resp.request_id,
                    },
                    delivery=delivery_outcome,
                    memory_root=self._memory_root,
                )
            except Exception as error:  # noqa: BLE001
                log.warning(
                    "checkpoint.save_failed channel=%s turn=%s error=%r",
                    chan_id,
                    turn_id,
                    error,
                )

            try:
                raw_meta = resp.raw if isinstance(resp.raw, dict) else {}
                used_multi_agent = raw_meta.get("multi_agent") is True
                used_orchestrator = raw_meta.get("orchestrator") is True
                path = (
                    "multi_agent"
                    if used_multi_agent
                    else "orchestrator"
                    if used_orchestrator
                    else "agent_loop"
                )
                await self.events.append(
                    "turn_telemetry",
                    {
                        "turn_id": str(turn_id),
                        "channel": chan_id,
                        "path": path,
                        "model": resp.model or self.default_model,
                        "rounds": rounds,
                        "tools_used": [t["name"] for t in (trace or [])[:10]],
                        "multi_agent": used_multi_agent,
                        "planner": raw_meta.get("planner", ""),
                        "executor": raw_meta.get("executor", ""),
                        "content_len": len(resp.content or ""),
                        "delivered": delivery_outcome.get("ok") is True,
                        "delivery_state": delivery_outcome.get("state", "unknown"),
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        # Agent-loop learning and aggregate trajectory persistence are evidence
        # sinks, not acceptance gates. Keep them after delivery so a slow disk
        # cannot hold a completed answer hostage.
        if resp is not None and _task_state is not None:
            try:
                raw_verifier_score = (resp.raw or {}).get("output_verifier_score")
                verifier_score = (
                    float(raw_verifier_score)
                    if isinstance(raw_verifier_score, (int, float))
                    and not isinstance(raw_verifier_score, bool)
                    else None
                )
                await agent_loop.record_agent_outcome(
                    event_log=self.events,
                    task_state=_task_state,
                    trace=trace,
                    final_resp=resp,
                    rounds=rounds,
                    model=resp.model or self.default_model,
                    user_text=env.body or "",
                    memory_root=self._memory_root,
                    self_model=self_model,
                    active_inference=active_inference,
                    metacognitive=metacognitive,
                    verifier_score=verifier_score,
                )
            except Exception:  # noqa: BLE001
                log.warning("agent_outcome.record_failed", exc_info=True)

        # Post-action hooks: brain learns from every interaction.
        if resp is not None:
            self._turn_count = getattr(self, "_turn_count", 0) + 1
            trace = trace or []

            # Phase 2: Trace collector — record turn for observability UI
            try:
                from ..observability.trace_ui import get_trace_collector

                trace_collector = get_trace_collector()
                trace_turn_id = f"turn-{self._turn_count}-{env.message_id[:8]}"
                # Record tool calls
                for t in (trace or [])[:20]:
                    await trace_collector.record_event(
                        turn_id=trace_turn_id,
                        event_type="tool_call",
                        data={
                            "tool": t["name"],
                            "args": t.get("args", {}),
                            "result": t.get("result", {}),
                            "ok": _explicit_result_ok(t.get("result")),
                        },
                    )
                # Record memory hits
                for item in (ctx.memory.items or [])[:10]:
                    if isinstance(item, tuple) and len(item) >= 2:
                        n = item[0]
                        await trace_collector.record_event(
                            turn_id=trace_turn_id,
                            event_type="memory_retrieval",
                            data={
                                "neuron_id": getattr(n, "entity_id", ""),
                                "kind": getattr(n, "kind", ""),
                                "score": float(item[1]) if len(item) > 1 else 0.0,
                            },
                        )
                await trace_collector.end_turn(trace_turn_id, rounds=rounds)
            except Exception as e:  # noqa: BLE001
                log.debug("trace_collector.failed error=%r", e)

            from ..brain.strong_model_scaffold import tool_result_succeeded

            executed_trace = [
                item
                for item in trace
                if isinstance(item.get("result"), dict)
                and item["result"].get("_not_executed") is not True
                and item["result"].get("_cached") is not True
            ]

            def _outcome_key(item: dict[str, Any]) -> str:
                name = str(item.get("name") or "tool")
                raw_args = item.get("args")
                args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
                action = str(args.get("action") or "").strip().lower()
                return f"{name}:{action}" if action else name

            scored_outcomes = [
                (
                    _outcome_key(item),
                    10.0
                    if tool_result_succeeded(
                        str(item.get("name") or ""),
                        item["result"],
                        item.get("args") if isinstance(item.get("args"), dict) else {},
                    )
                    else 0.0,
                )
                for item in executed_trace
            ]
            action_summary = ", ".join(key for key, _ in scored_outcomes[:5])
            score = (
                sum(outcome for _, outcome in scored_outcomes) / len(scored_outcomes)
                if scored_outcomes
                else 0.0
            )
            rpe_result = None
            try:
                if scored_outcomes:
                    # VTA and action history learn only from calls that really
                    # executed. One turn is persisted as one batch.
                    rpe_result = self._vta.record_outcome(action_summary, score)
                    self._basal_ganglia.record_outcomes(scored_outcomes)

                if self._memory_store is not None and rpe_result is not None:
                    from ..memory.vta_writer import build_turn_summary, write_outcome

                    summary = build_turn_summary(
                        user_input=env.body or "",
                        response_preview=resp.content or "",
                        tool_calls=len(trace),
                        rounds=rounds,
                    )
                    thalamic = ctx.metadata.get("thalamic", {})
                    from ..memory.hot_inject import post_turn_hot_maintenance

                    role = os.environ.get("NORAX_ROLE", "production")
                    memory_root = self._memory_store.root
                    focus_summary = ctx.focus.summary or ""
                    user_body = env.body or ""
                    message_type = str(thalamic.get("type", ""))
                    pathway = str(thalamic.get("pathway", ""))
                    turn_count = self._turn_count

                    def persist_hot_turn() -> None:
                        write_outcome(
                            memory_root=memory_root,
                            route=rpe_result.route,
                            text=summary,
                            source="vta",
                        )
                        post_turn_hot_maintenance(
                            memory_root,
                            focus_summary=focus_summary,
                            user_body=user_body,
                            msg_type=message_type,
                            pathway=pathway,
                            turn_count=turn_count,
                            role=role,
                        )

                    # These bounded disk writes occur after reply delivery but
                    # still must not stall unrelated channels on the event loop.
                    await asyncio.to_thread(persist_hot_turn)
                    if self._memory_coordinator is not None:
                        self._memory_coordinator.canonical_changed("post_turn")
                    # Bound index staleness while coalescing writes between syncs.
                    if self._turn_count % 10 == 0:
                        try:
                            if self._memory_coordinator is not None:
                                await self._memory_coordinator.sync_projections()
                        except Exception:
                            log.debug("memory_projection_sync.post_turn_failed", exc_info=True)
                    await self.events.append(
                        "vta_write",
                        {
                            "route": rpe_result.route,
                            "rpe": round(rpe_result.rpe, 2),
                            "reason": rpe_result.reason,
                        },
                    )
            except Exception:  # noqa: BLE001
                log.warning("post_action.failed", exc_info=True)

        # Sprint B: Hebbian strengthening — co-fired neurons get weight boost
        if resp is not None and ctx.memory.items:
            try:
                retrieved_neurons = []
                for item in ctx.memory.items:
                    if isinstance(item, tuple) and len(item) >= 2:
                        n = item[0]
                        if hasattr(n, "entity_id"):
                            retrieved_neurons.append(n)
                if len(retrieved_neurons) >= 2:
                    self._hebbian.record_cofiring(retrieved_neurons)
                    # Flush to disk every 10 turns
                    if self._hebbian.turn_count % 10 == 0:
                        if self._memory_store is not None:
                            self._hebbian.flush(self._memory_store.root)
            except Exception:  # noqa: BLE001
                log.warning("hebbian.failed", exc_info=True)

        # Sprint C: Record episode to episodic buffer
        if resp is not None and self._episodic is not None:
            try:
                import hashlib as _hl

                retrieval_eids = []
                for item in ctx.memory.items or []:
                    if isinstance(item, tuple) and len(item) >= 2:
                        n = item[0]
                        if hasattr(n, "entity_id"):
                            retrieval_eids.append(n.entity_id)

                arousal_data = ctx.metadata.get("arousal")
                arousal_level = (
                    arousal_data.get("level", "normal")
                    if isinstance(arousal_data, dict)
                    else getattr(arousal_data, "level", "normal")
                    if arousal_data
                    else "normal"
                )

                # Get VTA result if we computed it
                vta_rpe = 0.0
                vta_route = "scratchpad"
                if resp is not None and rpe_result is not None:
                    # Re-use the rpe_result from above if available
                    try:
                        vta_rpe = rpe_result.rpe
                        vta_route = rpe_result.route
                    except NameError:
                        pass

                episode_outcomes = [
                    10.0
                    if tool_result_succeeded(
                        str(item.get("name") or ""),
                        item.get("result") if isinstance(item.get("result"), dict) else {},
                        item.get("args") if isinstance(item.get("args"), dict) else {},
                    )
                    else 0.0
                    for item in trace
                    if isinstance(item.get("result"), dict)
                    and item["result"].get("_not_executed") is not True
                    and item["result"].get("_cached") is not True
                ]
                score = sum(episode_outcomes) / len(episode_outcomes) if episode_outcomes else 5.0
                raw_outcome = resp.raw if isinstance(resp.raw, dict) else {}
                verified_episode = raw_outcome.get("verified_outcome") is True
                delivery_episode = raw_outcome.get("delivery_succeeded") is True
                objective_value = raw_outcome.get("objective_outcome_observed")
                objective_episode = (
                    objective_value is True if objective_value is not None else verified_episode
                )

                episode = Episode(
                    user_input=(env.body or "")[:200],
                    user_input_hash=_hl.sha256((env.body or "").encode()).hexdigest()[:12],
                    retrieval_hits=retrieval_eids[:20],
                    tool_calls=[
                        {
                            "name": t["name"],
                            "ok": tool_result_succeeded(
                                str(t.get("name") or ""),
                                (t.get("result") if isinstance(t.get("result"), dict) else {}),
                                t.get("args") if isinstance(t.get("args"), dict) else {},
                            ),
                        }
                        for t in trace[:15]
                    ],
                    response_preview=(resp.content or "")[:200],
                    arousal_level=arousal_level,
                    vta_rpe=vta_rpe,
                    vta_route=vta_route,
                    model=resp.model or self.default_model,
                    rounds=rounds,
                    task_type=getattr(_task_state, "task_type", "") if _task_state else "",
                    outcome_score=score,
                    accepted_outcome=(
                        delivery_episode
                        and (
                            raw_outcome.get("accepted_outcome") is True
                            if raw_outcome.get("accepted_outcome") is not None
                            else verified_episode
                        )
                    ),
                    verified_outcome=verified_episode,
                    objective_outcome_observed=objective_episode,
                    training_eligible=(
                        delivery_episode
                        and (
                            raw_outcome.get("training_eligible") is True
                            if raw_outcome.get("training_eligible") is not None
                            else objective_episode
                        )
                    ),
                )
                self._episodic.record(episode)
            except Exception:  # noqa: BLE001
                log.warning("episodic.record.failed", exc_info=True)

        # Curiosity engine: record this turn's fingerprint so future novelty
        # scoring has this input in its rolling window.
        if resp is not None and curiosity is not None:
            try:
                curiosity.record(env.body or "", domain=_turn_domain)
            except Exception:  # noqa: BLE001
                log.debug("curiosity.record.failed", exc_info=True)

        # Sprint D: record turn to UserModel + trigger SkillLearner
        if resp is not None and self._user_model is not None:
            try:
                tool_names = [t["name"] for t in trace[:20]] if trace else []
                self._user_model.record_turn(
                    user_id=env.sender.id,
                    label=env.sender.label,
                    tier=env.sender.tier,
                    body=env.body or "",
                    response_preview=resp.content or "",
                    tool_calls=tool_names,
                    model=resp.model or self.default_model,
                    command_name=getattr(pc, "name", "") if pc else "",
                )
            except Exception:  # noqa: BLE001
                log.warning("user_model.record.failed", exc_info=True)

        # Sprint E: Feed CausalGraph + TemporalGraph post-turn
        if (
            self._memory_store is not None
            and self._causal_graph is not None
            and self._temporal_graph is not None
        ):
            try:
                async with self._graph_update_lock:
                    causal_graph = self._causal_graph
                    if trace:
                        causal_graph.ingest_trajectory(
                            {
                                "trace": trace,
                                "trajectory_id": turn_id,
                                "timestamp": time.time(),
                            }
                        )
                    # TemporalGraph: record sequence of retrieved neuron IDs
                    temporal_graph = self._temporal_graph
                    if ctx.memory.items:
                        retrieved_ids = []
                        for item in ctx.memory.items[:30]:
                            if isinstance(item, tuple) and len(item) >= 2:
                                n = item[0]
                                if hasattr(n, "entity_id"):
                                    retrieved_ids.append(n.entity_id)
                        session_id = str(
                            (env.raw or {}).get("channel_id") or env.channel or "default"
                        )
                        for nid in retrieved_ids:
                            temporal_graph.record_access(nid, session_id=session_id)
                        if len(retrieved_ids) >= 2:
                            temporal_graph.record_sequence(retrieved_ids, session_id=session_id)
                    # Debounced persist: full JSON rewrites are ~14MB, so save at
                    # most once a minute; shutdown() flushes the tail. Serialize
                    # in worker threads while this async lock prevents a later
                    # post-turn mutation from racing the snapshots.
                    now = time.monotonic()
                    if now - self._last_graph_save >= 60.0:
                        await asyncio.gather(
                            asyncio.to_thread(causal_graph.save),
                            asyncio.to_thread(temporal_graph.save),
                        )
                        self._last_graph_save = now
            except Exception:  # noqa: BLE001
                log.warning("causal_temporal.record.failed", exc_info=True)

        if resp is not None:
            # A tool may request this runtime's own restart/stop. Executing it
            # inside the tool round used to kill the process before Discord
            # finalized long responses, leaving a permanent
            # "[…response continues]" preview. The shell tool now records the
            # intent; execute it only after the complete reply is delivered.
            lifecycle_action = _deferred_runtime_lifecycle_action(trace)
            if lifecycle_action is not None:
                await self.events.append(
                    "runtime_lifecycle_deferred",
                    {
                        "action": lifecycle_action,
                        "after_reply": True,
                        "tool_calls": len(trace),
                    },
                )
                cmd = [
                    "systemctl",
                    "--user",
                    lifecycle_action,
                    "--no-block",
                    "norax-ai.service",
                ]
                log.warning("runtime.lifecycle_after_reply: %s", " ".join(cmd))
                try:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("runtime.lifecycle_after_reply_failed")

    # ------------------------------------------------------------------
    # Slash-command plumbing
    # ------------------------------------------------------------------

    def custom_model_catalog(self) -> dict[str, list[tuple[str, str]]]:
        catalog: dict[str, list[tuple[str, str]]] = {}
        store = getattr(self, "_provider_store", None)
        if store is None:
            return catalog
        for provider in store.list_public():
            if provider.get("enabled") is not True:
                continue
            name = provider["name"]
            catalog[name] = [(f"{name}/{model}", model) for model in provider.get("models") or []]
        return catalog

    def _gateway_has_provider(self, name: str) -> bool:
        check = getattr(self.gateway, "has_provider", None)
        if callable(check):
            return bool(check(name))
        urls = getattr(self.gateway, "provider_urls", {})
        return isinstance(urls, dict) and name in urls

    def _custom_provider_usage(self, name: str) -> str | None:
        prefix = f"{name}/"
        model_references: list[tuple[str, object]] = [("default model", self.default_model)]
        model_references.extend(("failover model", model) for model in self.failover_models)
        model_references.append(("vision model", self._vision_config.get("model")))
        for label, model in model_references:
            if isinstance(model, str) and model.startswith(prefix):
                return label
        if getattr(self.gateway, "default_provider", None) == name:
            return "gateway default provider"
        return None

    async def _append_provider_event(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            await self.events.append(kind, payload)
        except Exception:  # noqa: BLE001
            # The provider transaction is already complete. Event-log health is
            # reported independently and must not turn a successful mutation
            # into a false 502 response that invites a duplicate retry.
            log.warning("%s.event_append_failed", kind, exc_info=True)

    async def _append_delivery_event(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """Record post-send telemetry without rewriting delivery truth.

        Once an outbound adapter has accepted a reply, an observability failure
        must not make callers believe delivery failed (or invite a duplicate
        retry).  Event-log health remains independently visible through logs and
        readiness checks.
        """
        try:
            await self.events.append(kind, payload, attrs=attrs)
        except Exception:  # noqa: BLE001
            log.error("%s.delivery_event_append_failed", kind, exc_info=True)

    def _record_outbound_metric(self, *, source: str, delivered: bool) -> None:
        """Keep metrics failures outside the user-visible delivery boundary."""
        try:
            self.metrics.outbound_sends.labels(
                channel=source,
                ok=str(delivered).lower(),
            ).inc()
        except Exception:  # noqa: BLE001
            log.warning("outbound.metric_record_failed source=%s", source, exc_info=True)

    async def _restore_custom_provider(
        self,
        name: str,
        previous: dict[str, Any] | None,
    ) -> None:
        if previous is None:
            await asyncio.to_thread(self._provider_store.remove, name)
            return
        public = {key: value for key, value in previous.items() if key != "api_key"}
        await asyncio.to_thread(
            self._provider_store.upsert,
            public,
            previous.get("api_key", ""),
        )

    @staticmethod
    async def _discover_custom_provider_models(
        *,
        client: GatewayClient,
    ) -> list[Any]:
        return await client.fetch_model_ids(
            max_response_bytes=_MAX_PROVIDER_DISCOVERY_BYTES,
            max_models=_MAX_PROVIDER_DISCOVERY_MODELS,
        )

    async def upsert_custom_provider(self, spec: dict[str, Any]) -> dict[str, Any]:
        from ..config.provider_store import validate_provider_api_key, validate_provider_spec

        if not isinstance(spec, dict):
            raise ValueError("provider specification must be an object")
        unknown = set(spec) - {
            "name",
            "base_url",
            "api_key",
            "provider_kind",
            "models",
            "enabled",
        }
        if unknown:
            raise ValueError(f"unsupported provider fields: {sorted(unknown)}")
        clean = validate_provider_spec(spec)
        supplied_key = spec.get("api_key")
        if supplied_key is not None:
            supplied_key = validate_provider_api_key(supplied_key)

        async with self._provider_mutation_lock:
            existing_rows = await asyncio.to_thread(self._provider_store.list_runtime)
            previous = {row["name"]: row for row in existing_rows}.get(clean["name"])
            if clean["name"] in self._base_provider_names:
                raise ValueError("provider name is reserved by the base runtime configuration")
            if previous is None and self._gateway_has_provider(clean["name"]):
                raise ValueError("provider name is already active outside the provider store")

            api_key = validate_provider_api_key(
                supplied_key if supplied_key is not None else (previous or {}).get("api_key", "")
            )
            if clean["enabled"] is not True:
                usage = self._custom_provider_usage(clean["name"])
                if usage:
                    raise ValueError(f"select a different {usage} before disabling this provider")
                was_live = self._gateway_has_provider(clean["name"])
                remove = getattr(self.gateway, "remove_provider", None)
                if was_live and not callable(remove):
                    raise RuntimeError("active gateway does not support dynamic provider removal")
                await asyncio.to_thread(
                    self._provider_store.upsert,
                    clean,
                    api_key if supplied_key is not None else None,
                )
                try:
                    if was_live and (not callable(remove) or not await remove(clean["name"])):
                        raise RuntimeError("active gateway refused to disable the provider")
                except Exception:
                    await self._restore_custom_provider(clean["name"], previous)
                    raise
                await self._append_provider_event("provider_disabled", {"provider": clean["name"]})
                return clean

            upsert = getattr(self.gateway, "upsert_provider", None)
            if not callable(upsert):
                raise RuntimeError("dynamic providers require multi-provider gateway mode")
            # Construct GatewayClient on the event loop. httpx.AsyncClient
            # initialization is lightweight and must not run in a worker
            # thread — the connection pool is bound to this loop and
            # cross-thread construction can stall during TLS setup.
            base_gateway_client = GatewayClient(
                base_url=clean["base_url"],
                timeout=self._gateway_timeout_seconds,
                api_key=api_key or None,
                provider_kind=clean["provider_kind"],
            )
            try:
                models = await self._discover_custom_provider_models(client=base_gateway_client)
                if models:
                    clean = validate_provider_spec({**clean, "models": models})
                if not clean["models"]:
                    raise ValueError(
                        "provider returned no models; provide an OpenAI-compatible /models endpoint"
                    )
            except Exception:
                await base_gateway_client.aclose()
                raise
            gateway_client: Any = base_gateway_client
            if clean["provider_kind"] == "ollama":
                from ..gateway_client.ollama_wrapper import OllamaGatewayClient

                gateway_client = OllamaGatewayClient(gateway_client, metrics=self.metrics)
            try:
                await asyncio.to_thread(
                    self._provider_store.upsert,
                    clean,
                    api_key if supplied_key is not None else None,
                )
            except Exception:
                await gateway_client.aclose()
                raise
            try:
                await upsert(
                    clean["name"],
                    gateway_client,
                    patterns=[f"{clean['name']}/*"],
                )
            except Exception:
                try:
                    await self._restore_custom_provider(clean["name"], previous)
                finally:
                    await gateway_client.aclose()
                raise
            await self._append_provider_event(
                "provider_updated",
                {"provider": clean["name"], "models": len(clean["models"])},
            )
            return clean

    async def remove_custom_provider(self, name: str) -> bool:
        try:
            clean_name = _provider_identifier(name)
        except ValueError:
            return False
        async with self._provider_mutation_lock:
            existing_rows = await asyncio.to_thread(self._provider_store.list_runtime)
            previous = {row["name"]: row for row in existing_rows}.get(clean_name)
            if previous is None:
                return False
            if clean_name in self._base_provider_names:
                raise ValueError("base runtime providers cannot be removed through this API")
            usage = self._custom_provider_usage(clean_name)
            if usage:
                raise ValueError(f"select a different {usage} before removing this provider")
            was_live = self._gateway_has_provider(clean_name)
            remove = getattr(self.gateway, "remove_provider", None)
            if was_live and not callable(remove):
                raise RuntimeError("active gateway does not support dynamic provider removal")
            removed_from_store = await asyncio.to_thread(self._provider_store.remove, clean_name)
            if not removed_from_store:
                return False
            try:
                if was_live and (not callable(remove) or not await remove(clean_name)):
                    raise RuntimeError("active gateway refused to remove the provider")
            except Exception:
                await self._restore_custom_provider(clean_name, previous)
                raise
            await self._append_provider_event("provider_removed", {"provider": clean_name})
            return True

    def set_default_model(self, model: str) -> bool:
        model = _model_identifier(model)
        self.default_model = model
        self._effective_model = model
        route_for = getattr(self.gateway, "route_for", None)
        if callable(route_for):
            try:
                self._effective_provider = route_for(model)[0]
            except Exception as exc:  # noqa: BLE001
                log.warning("gateway.model_selection_route_failed: %r", exc)
        log.info("default_model set to %s", model)
        # Schedule persistence off the event loop. The fsync inside
        # atomic_write_text can block for milliseconds; doing it inline
        # stalls every ingress stream and active turn on the same loop.
        persisted = self._persist_default_model_async(model)
        # max/ultra thinking is only valid for models that support it; downgrade
        # if the model changes away from a max-capable one.
        if self.thinking_effort == "ultra" and not _is_gpt_56_sol(model):
            self.thinking_effort = "medium"
            log.info("thinking_effort downgraded to medium (model is not GPT-5.6 Sol)")
        elif self.thinking_effort == "max" and not _supports_max_reasoning(model):
            self.thinking_effort = "medium"
            log.info("thinking_effort downgraded to medium (model does not support max)")
        return persisted

    def set_thinking_effort(self, level: str) -> None:
        if level not in cmd_mod.THINK_LABELS:
            raise ValueError(f"unsupported thinking effort: {level!r}")
        if level == "ultra" and not _is_gpt_56_sol(self.default_model):
            log.warning(
                "refusing to set thinking_effort=%s for non-GPT-5.6-Sol model %s",
                level,
                self.default_model,
            )
            return
        if level == "max" and not _supports_max_reasoning(self.default_model):
            log.warning(
                "refusing to set thinking_effort=%s for model %s (requires GPT-5.6 Sol or GLM-5.3)",
                level,
                self.default_model,
            )
            return
        self.thinking_effort = level
        log.info("thinking_effort set to %s", level)

    def set_reasoning_output(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("reasoning_output must be a boolean")
        self.reasoning_output = enabled
        log.info("reasoning_output set to %s", self.reasoning_output)

    def set_planning_mode(self, mode: str) -> bool:
        if mode not in cmd_mod.PLANNING_MODES:
            raise ValueError(f"unsupported planning mode: {mode!r}")
        if mode == "orchestrator" and not self._planner_enabled:
            log.warning("planning_mode=orchestrator rejected: planner connector is disabled")
            return False
        self.planning_mode = mode
        log.info("planning_mode set to %s", mode)
        return True

    def set_max_tool_rounds(self, cap: int) -> None:
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise TypeError("max_tool_rounds must be an integer")
        if not 0 <= cap <= agent_loop.HARD_ROUND_CAP:
            raise ValueError(f"max_tool_rounds must be between 0 and {agent_loop.HARD_ROUND_CAP}")
        self.max_tool_rounds = cap
        log.info("max_tool_rounds set to %s", self.max_tool_rounds)

    def set_memory_depth(self, mode: str) -> None:
        if mode not in cmd_mod.MEMORY_DEPTHS:
            raise ValueError(f"unsupported memory depth: {mode!r}")
        self.memory_depth = mode
        log.info("memory_depth set to %s", mode)

    def set_weak_model_boost(self, mode: str) -> None:
        if mode not in cmd_mod.BOOST_MODES:
            raise ValueError(f"unsupported weak-model boost: {mode!r}")
        self.weak_model_boost = mode
        log.info("weak_model_boost set to %s", mode)

    def set_stream_replies(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("stream_replies must be a boolean")
        self.stream_replies = enabled
        log.info("stream_replies set to %s", self.stream_replies)

    def set_response_length(self, pref: str) -> None:
        if pref not in cmd_mod.LENGTH_PREFS:
            raise ValueError(f"unsupported response length: {pref!r}")
        self.response_length = pref
        log.info("response_length set to %s", pref)

    def set_tool_activity(self, mode: str) -> None:
        if mode not in cmd_mod.ACTIVITY_MODES:
            raise ValueError(f"unsupported tool activity: {mode!r}")
        self.tool_activity = mode
        log.info("tool_activity set to %s", mode)

    def _memory_k_for_turn(self, body: str, sender_tier: str) -> int:
        depth_k = {"light": 3, "balanced": 5, "deep": 7}
        if self.memory_depth in depth_k:
            return depth_k[self.memory_depth]
        return memory_k_for_turn(body, sender_tier)

    def _persist_default_model(self, model: str) -> bool:
        try:
            self._provider_store.set_runtime_setting("default_model", model)
            log.info("persisted default_model=%s to private runtime settings", model)
            return True
        except Exception:  # noqa: BLE001
            log.exception("failed to persist default_model")
            return False

    def _persist_default_model_async(self, model: str) -> bool:
        """Schedule durable persistence off the event loop.

        Returns ``True`` immediately (the live state is already updated);
        the fsync happens in a worker thread so ingress and active turns
        are not stalled.  If the thread fails, the error is logged but
        the live selection remains active — the caller already received
        an honest "active for this process" response.
        """
        store = getattr(self, "_provider_store", None)
        if store is None:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._persist_default_model(model)

        self._pending_default_model = model
        current = getattr(self, "_default_model_persist_task", None)
        if current is not None and not current.done():
            return True

        async def _persist_pending() -> None:
            while self._pending_default_model is not None:
                selected = self._pending_default_model
                self._pending_default_model = None
                try:
                    await asyncio.to_thread(
                        store.set_runtime_setting,
                        "default_model",
                        selected,
                    )
                    log.info("persisted default_model=%s to private runtime settings", selected)
                except Exception:  # noqa: BLE001
                    log.exception("failed to persist default_model=%s", selected)

        task = loop.create_task(_persist_pending(), name="persist-default-model")
        self._default_model_persist_task = task
        maintenance_tasks = getattr(self, "_maintenance_tasks", None)
        if not isinstance(maintenance_tasks, set):
            maintenance_tasks = set()
            self._maintenance_tasks = maintenance_tasks
        maintenance_tasks.add(task)

        def _finished(completed: asyncio.Task[None]) -> None:
            self._maintenance_tasks.discard(completed)
            if self._default_model_persist_task is completed:
                self._default_model_persist_task = None
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                log.error("default-model persistence task crashed: %r", error)

        task.add_done_callback(_finished)
        return True

    def _window_path(self, channel_id: str) -> Path:
        """Disk location for persisted rolling window."""
        mem_root = getattr(self.cfg, "memory_root", None)
        base = (
            mem_root / "state" / "windows" if isinstance(mem_root, Path) else Path("state/windows")
        )
        # Sanitize channel_id to a filename-safe token.
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", channel_id):
            safe = channel_id
        else:
            slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in channel_id)[:64]
            digest = hashlib.sha256(channel_id.encode("utf-8")).hexdigest()[:16]
            safe = f"{slug or 'channel'}-{digest}"
        return base / f"{safe}.json"

    def _get_window(self, channel_id: str) -> RollingWindow:
        if channel_id not in self._windows:
            # Try to load persisted window from disk.
            path = self._window_path(channel_id)
            self._windows[channel_id] = RollingWindow.load(path, budget_tokens=256_000)
            self._windows[channel_id].protect_tail_turns = 8
        # Per-model window budget: profiles with `window_tokens` pin the
        # rolling window to that model's verified max context (e.g.
        # qwen3.8-27b-fast @ 80k) so the window never evicts before the
        # model's context would fill. Other models keep the 256k default.
        try:
            from ..gateway_client.ollama_profiles import resolve_profile

            prof = resolve_profile(getattr(self, "_effective_model", "") or "")
            wt = int(getattr(prof, "window_tokens", 0) or 0)
            self._windows[channel_id].budget_tokens = wt if wt else 256_000
        except Exception:  # noqa: BLE001
            pass
        return self._windows[channel_id]

    def _persist_window(self, channel_id: str) -> None:
        """Save the channel's window to disk. Fails quietly."""
        w = self._windows.get(channel_id)
        if w is None:
            return
        try:
            w.save(self._window_path(channel_id))
        except Exception as e:  # noqa: BLE001
            log.warning("window_persist.failed channel=%s err=%r", channel_id, e)

    def _reset_window(self, channel_id: str) -> bool:
        existed = channel_id in self._windows
        self._windows.pop(channel_id, None)
        # Also delete the persisted file.
        try:
            p = self._window_path(channel_id)
            if p.exists():
                p.unlink()
        except Exception as e:  # noqa: BLE001
            log.warning("window_reset.unlink_failed err=%r", e)
        return existed

    def _dump_window(self, channel_id: str, half: bool = False) -> dict | None:
        """Dump this channel's rolling body context to sleep/.

        The bootstrap/system head is preserved. Full mode clears all body
        frames; half mode clears the older 50% of body frames.
        """
        w = self._get_window(channel_id)
        before_frames = len(w.body)
        before_tokens = w.total_tokens()
        result = w.dump_to_sleep(half=half)
        if result is None:
            return None
        self._persist_window(channel_id)
        return {
            "mode": "half" if half else "full",
            "frames": len(result.evicted),
            "tokens_freed": result.tokens_freed,
            "spill_path": str(result.spill_path) if result.spill_path else None,
            "before_frames": before_frames,
            "remaining_frames": len(w.body),
            "before_tokens": before_tokens,
            "after_tokens": w.total_tokens(),
        }

    def _cancel_channel(self, channel_id: str = "") -> bool:
        """Signal any active turn on this channel to stop."""
        if not channel_id:
            active_tasks = [
                (cid, task) for cid, task in self._active_turn_tasks.items() if not task.done()
            ]
            for cid, task in active_tasks:
                self._stop_channels.add(cid)
                if self._active_turn_tasks.get(cid) is task:
                    self._active_turn_tasks.pop(cid, None)
                queue = self._turn_queues.pop(cid, None)
                if queue:
                    self._turn_work_count = max(0, self._turn_work_count - len(queue))
                    queue.clear()
                task.cancel()
            return bool(active_tasks)
        active_task = self._active_turn_tasks.get(channel_id)
        if active_task is not None and not active_task.done():
            self._stop_channels.add(channel_id)
            if self._active_turn_tasks.get(channel_id) is active_task:
                self._active_turn_tasks.pop(channel_id, None)
            queue = self._turn_queues.pop(channel_id, None)
            if queue:
                self._turn_work_count = max(0, self._turn_work_count - len(queue))
                queue.clear()
            active_task.cancel()
            return True
        if active_task is not None and self._active_turn_tasks.get(channel_id) is active_task:
            self._active_turn_tasks.pop(channel_id, None)
        self._stop_channels.discard(channel_id)
        return False

    def _get_window_stats(self, channel_id: str) -> dict | None:
        """Return context window stats for a channel."""
        w = self._windows.get(channel_id)
        if w is None:
            return None
        total = w.total_tokens()
        budget = w.budget_tokens
        head = w.head_tokens()
        body_count = len(w.body)
        return {
            "current_tokens": total,
            "budget_tokens": budget,
            "head_tokens": head,
            "body_frames": body_count,
            "usage_pct": round(total / max(budget, 1) * 100, 1),
        }

    def _get_brain_stats(self) -> dict[str, Any] | None:
        """Return brain subsystem stats."""
        stats: dict[str, Any] = {}
        if self._memory_store is not None:
            stats["memory"] = {
                "hot": len(self._memory_store.hot),
                "canonical": len(self._memory_store.all_canonical()),
                "sleep": len(self._memory_store.sleep),
            }
        if self._episodic is not None:
            stats["episodic"] = self._episodic.stats()
        if hasattr(self, "_hebbian"):
            stats["hebbian_turns"] = self._hebbian.turn_count
        return stats

    def _build_runtime_handle(self) -> cmd_mod.RuntimeHandle:
        prov_urls = getattr(self.gateway, "provider_urls", None)
        return cmd_mod.RuntimeHandle(
            default_model=self.default_model,
            set_default_model=self.set_default_model,
            started_at=self._started_mono,
            wall_started_at=self._started_wall,
            metrics=self.metrics,
            gateway_base_url=self.gateway.base_url,
            event_log=self.events,
            reset_window_for=self._reset_window,
            dump_window_for=self._dump_window,
            outbound=self.outbound,
            provider_urls=prov_urls,
            gateway=self.gateway,
            get_window_stats=self._get_window_stats,
            get_brain_stats=self._get_brain_stats,
            cancel_stream=self._cancel_channel,
            thinking_effort=self.thinking_effort,
            set_thinking_effort=self.set_thinking_effort,
            reasoning_output=self.reasoning_output,
            set_reasoning_output=self.set_reasoning_output,
            planning_mode=self.planning_mode,
            set_planning_mode=self.set_planning_mode,
            max_tool_rounds=self.max_tool_rounds,
            set_max_tool_rounds=self.set_max_tool_rounds,
            memory_depth=self.memory_depth,
            set_memory_depth=self.set_memory_depth,
            weak_model_boost=self.weak_model_boost,
            set_weak_model_boost=self.set_weak_model_boost,
            stream_replies=self.stream_replies,
            set_stream_replies=self.set_stream_replies,
            response_length=self.response_length,
            set_response_length=self.set_response_length,
            tool_activity=self.tool_activity,
            set_tool_activity=self.set_tool_activity,
            custom_model_catalog=self.custom_model_catalog,
        )

    async def _handle_command(self, env, pc: cmd_mod.ParsedCommand) -> None:
        """Run a slash command and emit its reply via the source channel."""
        rt = self._build_runtime_handle()
        try:
            result = await cmd_mod.handle(pc, env, rt)
        except Exception as e:  # noqa: BLE001
            log.exception("command.%s failed", pc.name)
            self.metrics.brain_errors.labels(where=f"cmd.{pc.name}").inc()
            await self.events.append(
                "cmd.error",
                {"name": pc.name, "err": repr(e), "user": env.sender.id},
            )
            result = cmd_mod.CommandResult(reply=f"Command `/{pc.name}` failed: {e}")

        await self.events.append(
            "cmd",
            {
                "name": pc.name,
                "args": pc.args,
                "user": env.sender.id,
                "tier": env.sender.tier,
                "channel_id": (env.raw or {}).get("channel_id"),
                "reply_preview": (result.reply or "")[:200],
                "post_send": result.post_send is not None,
            },
        )

        # Route the reply via the same outbound path as normal replies.
        delivered = await self._send_command_reply(
            env, result.reply, reply_to=None if pc.name == "status" else env.message_id
        )

        # Post-send effects (shutdown, restart) run after the reply is out.
        if result.post_send is not None and delivered:
            try:
                await result.post_send()
            except Exception:  # noqa: BLE001
                log.exception("command.post_send failed: %s", pc.name)
        elif result.post_send is not None:
            log.error("command.post_send suppressed because reply delivery failed: %s", pc.name)

    async def _send_command_reply(
        self,
        env,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        target = None
        if env.source == "discord":
            target = (env.raw or {}).get("channel_id")
        if not target and env.thread_binding is not None:
            target = env.thread_binding.thread_id
        if not self.outbound.has(env.source) or not target:
            log.debug("cmd.no_route source=%s target=%s", env.source, target)
            return False
        result = await self.outbound.send(env.source, target, text, reply_to=reply_to)
        delivery_ok = _explicit_result_ok(result)
        self._record_outbound_metric(source=env.source, delivered=delivery_ok)
        await self._append_delivery_event(
            "send",
            {
                "source": env.source,
                "target": target,
                "ok": delivery_ok,
                "result": result,
                "kind": "cmd",
            },
        )
        await self._mirror_discord_reply(env, text)
        return delivery_ok

    async def _mirror_discord_reply(self, env, text: str) -> None:
        """Mirror Discord output to Agent OS without routing Agent OS into Discord."""
        if env.source != "discord" or self.agent_os_bridge is None or not (text or "").strip():
            return
        try:
            await self.agent_os_bridge.broadcast_outbound(text=text.strip())
        except Exception:  # noqa: BLE001
            log.debug("agent_os reply_mirror_failed", exc_info=True)

    async def _slash_dispatch(
        self,
        name: str,
        args: dict,
        principal,
        ctx: dict,
    ) -> dict:
        """Handle a native Discord slash-command invocation.

        Returns a dict with:
            reply: str        — text to send as interaction followup
            post_send: Optional[awaitable] — coroutine to run after reply
        """
        from ..envelope import SensoryInput

        env = SensoryInput(
            channel="chat",
            source="discord",
            message_id=str(ctx.get("interaction_id") or ""),
            timestamp=datetime.now(UTC),
            sender=principal,
            body=f"/{name}",
            raw={
                "channel_id": ctx.get("channel_id"),
                "guild_id": ctx.get("guild_id"),
                "slash_command": True,
            },
            trusted=bool(principal.trust),
            metadata={},
        )

        # Reuse the text-mode ParsedCommand + handle() path.
        pc_args = args.get("args") or []
        # Drop None entries (optional params that weren't supplied).
        pc_args = [a for a in pc_args if a is not None and a != ""]
        pc = cmd_mod.ParsedCommand(name=name, args=list(pc_args), raw=f"/{name}")

        rt_handle = self._build_runtime_handle()
        try:
            result = await cmd_mod.handle(pc, env, rt_handle)
        except Exception as e:  # noqa: BLE001
            log.exception("slash_dispatch.%s failed", name)
            self.metrics.brain_errors.labels(where=f"slash.{name}").inc()
            return {"reply": f"Command `/{name}` failed: {e}", "post_send": None}

        await self.events.append(
            "cmd",
            {
                "name": name,
                "args": pc_args,
                "user": principal.id,
                "tier": principal.tier,
                "source": "slash",
                "channel_id": ctx.get("channel_id"),
                "reply_preview": (result.reply or "")[:200],
                "post_send": result.post_send is not None,
            },
        )
        return {
            "reply": result.reply,
            "post_send": result.post_send,
            "data": result.data,
        }

    async def _deliver_turn_response(
        self,
        *,
        env: Any,
        ctx: Any,
        resp: Any,
        trace: list[dict[str, Any]],
        rounds: int,
        streaming_msg: Any,
        stream_delta_buf: list[str],
        target_channel: str | None,
    ) -> dict[str, Any]:
        """Deliver a completed response and return explicit delivery evidence."""
        usage = resp.usage or {}
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        if in_tok:
            self.metrics.gateway_tokens_in.labels(model=resp.model or self.default_model).inc(
                in_tok
            )
        if out_tok:
            self.metrics.gateway_tokens_out.labels(model=resp.model or self.default_model).inc(
                out_tok
            )
        self.metrics.gateway_requests.labels(
            model=resp.model or self.default_model, status="ok"
        ).inc()

        content = (resp.content or "").strip()
        delivery: dict[str, Any]
        if streaming_msg is not None:
            result: dict[str, Any] | None = None
            if content in {"NO_REPLY", "HEARTBEAT_OK"}:
                try:
                    await streaming_msg.delete()
                    delivery = {
                        "ok": True,
                        "state": "intentionally_suppressed",
                        "attempted": False,
                    }
                except Exception:  # noqa: BLE001
                    log.debug("stream.sentinel_delete_failed", exc_info=True)
                    delivery = {
                        "ok": False,
                        "state": "suppression_cleanup_failed",
                        "attempted": True,
                    }
            else:
                if not content:
                    final_text = (
                        f"⚠️ Empty upstream response [{resp.model}] "
                        f"rounds={rounds} tool_calls={len(trace)} "
                        f"streamed_chars={sum(len(x) for x in stream_delta_buf)} "
                        f"usage={resp.usage} raw={str(resp.raw)[:1000]}"
                    )
                    reply_to = env.message_id
                elif content == "." and (resp.model or "").startswith("gemma"):
                    final_text = (
                        f"⚠️ Suspicious single-dot upstream response [{resp.model}] "
                        f"rounds={rounds} tool_calls={len(trace)} "
                        f"streamed_chars={sum(len(x) for x in stream_delta_buf)} "
                        f"usage={resp.usage} raw={str(resp.raw)[:1000]}"
                    )
                    reply_to = env.message_id
                else:
                    parsed = parse_reply_tag(content, current_message_id=env.message_id)
                    final_text = parsed.text
                    reply_to = parsed.reply_to
                try:
                    finalized = await streaming_msg.finalize(final_text)
                    result = (
                        finalized
                        if isinstance(finalized, dict)
                        else {
                            "ok": False,
                            "error": "invalid_stream_finalize_result",
                            "result": finalized,
                        }
                    )
                except Exception:  # noqa: BLE001
                    log.warning("stream.finalize_failed; using direct Discord send", exc_info=True)
                    if target_channel and self.outbound.has(env.source):
                        fallback = await self.outbound.send(
                            env.source,
                            target_channel,
                            final_text,
                            reply_to=reply_to,
                        )
                        result = (
                            fallback
                            if isinstance(fallback, dict)
                            else {
                                "ok": False,
                                "error": "invalid_outbound_result",
                                "result": fallback,
                            }
                        )
                if result is not None:
                    delivery_ok = _explicit_result_ok(result)
                    delivery = {
                        "ok": delivery_ok,
                        "state": "delivered" if delivery_ok else "delivery_failed",
                        "attempted": True,
                        "result": result,
                    }
                    self._record_outbound_metric(source=env.source, delivered=delivery_ok)
                    await self._append_delivery_event(
                        "send",
                        {
                            "source": env.source,
                            "target": target_channel,
                            "ok": delivery_ok,
                            "result": result,
                            "kind": "stream_finalize",
                        },
                    )
                else:
                    delivery = {
                        "ok": False,
                        "state": "unroutable",
                        "attempted": False,
                    }
        else:
            delivery = await self._emit_reply(env, ctx, resp)

        if content and content not in {"NO_REPLY", "HEARTBEAT_OK"}:
            parsed = parse_reply_tag(content, current_message_id=env.message_id)
            await self._mirror_discord_reply(env, parsed.text)

        await self._append_delivery_event(
            "reply",
            {
                "content_preview": (resp.content or "")[:500],
                "usage": resp.usage,
                "rounds": rounds,
                "tool_calls": len(trace),
                "streamed": streaming_msg is not None,
                "delivered": delivery.get("ok") is True,
                "delivery_state": delivery.get("state", "unknown"),
            },
            attrs={
                "gen_ai.response.id": resp.request_id,
                "gen_ai.response.model": resp.model,
                "gen_ai.usage.input_tokens": in_tok,
                "gen_ai.usage.output_tokens": out_tok,
            },
        )
        return delivery

    async def _emit_reply(self, env, ctx, resp) -> dict[str, Any]:
        """Route a brain reply back to the source channel when eligible."""
        if ctx.decision != "emit_reply":
            return {"ok": True, "state": "not_requested", "attempted": False}
        content = (resp.content or "").strip()
        if not content:
            content = (
                f"⚠️ Empty upstream response [{resp.model}] "
                f"usage={resp.usage} raw={str(resp.raw)[:1000]}"
            )
        if content in ("NO_REPLY", "HEARTBEAT_OK"):
            return {
                "ok": True,
                "state": "intentionally_suppressed",
                "attempted": False,
            }

        parsed = parse_reply_tag(content, current_message_id=env.message_id)
        text_out = parsed.text
        reply_to = parsed.reply_to
        if not text_out.strip():
            return {"ok": False, "state": "empty_reply", "attempted": False}

        target = None
        if env.source == "discord":
            target = (env.raw or {}).get("channel_id")
        if not target and env.thread_binding is not None:
            target = env.thread_binding.thread_id

        if not self.outbound.has(env.source):
            log.debug("emit_reply.no_outbound source=%s", env.source)
            return {"ok": False, "state": "no_outbound", "attempted": False}
        if not target:
            log.debug("emit_reply.no_target source=%s msg=%s", env.source, env.message_id)
            return {"ok": False, "state": "no_target", "attempted": False}

        result = await self.outbound.send(env.source, target, text_out, reply_to=reply_to)
        delivery_ok = _explicit_result_ok(result)
        self._record_outbound_metric(source=env.source, delivered=delivery_ok)
        await self._append_delivery_event(
            "send",
            {
                "source": env.source,
                "target": target,
                "ok": delivery_ok,
                "result": result,
            },
        )
        return {
            "ok": delivery_ok,
            "state": "delivered" if delivery_ok else "delivery_failed",
            "attempted": True,
            "result": result,
        }

    @staticmethod
    async def _settle_tasks(
        tasks: list[asyncio.Task[Any]],
        *,
        timeout: float,
        cancel: bool,
        label: str,
    ) -> set[asyncio.Task[Any]]:
        """Bounded task cleanup that consumes terminal exceptions."""
        current = asyncio.current_task()
        pending_set = {task for task in tasks if task is not current and not task.done()}
        if cancel:
            for task in pending_set:
                task.cancel()
        done: set[asyncio.Task[Any]] = {task for task in tasks if task.done()}
        pending: set[asyncio.Task[Any]] = set()
        if pending_set:
            newly_done, pending = await asyncio.wait(
                pending_set,
                timeout=max(0.0, timeout),
            )
            done.update(newly_done)
        for task in done:
            if task.cancelled():
                continue
            try:
                error = task.exception()
            except asyncio.CancelledError:
                continue
            if error is not None:
                log.warning("%s task failed: %r", label, error)
        if pending:
            log.warning("%s cleanup timed out with %d task(s) pending", label, len(pending))
        return pending

    async def _drain_active_turns(self, grace: float) -> None:
        active = [task for task in self._active_turn_tasks.values() if not task.done()]
        pending = await self._settle_tasks(
            active,
            timeout=grace,
            cancel=False,
            label="active turn drain",
        )
        if pending:
            still_pending = await self._settle_tasks(
                list(pending),
                timeout=min(2.0, max(0.1, grace)),
                cancel=True,
                label="active turn cancellation",
            )
            if still_pending:
                log.error(
                    "shutdown continuing after cancellation-resistant turn tasks: %d",
                    len(still_pending),
                )
        self._active_turn_tasks = {
            channel: task for channel, task in self._active_turn_tasks.items() if not task.done()
        }

    async def shutdown(self) -> None:
        if self._shutdown_lock is None:
            self._shutdown_lock = asyncio.Lock()
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._draining = True
            grace = self.cfg.shutdown_grace_seconds

            core_background = [
                task
                for task in (
                    self._capability_probe_task,
                    self._sleep_task,
                    self._warmup_task,
                    self._probe_task,
                    self._autonomy_task,
                )
                if task is not None
            ]
            for task in core_background:
                if not task.done():
                    task.cancel()

            # Ask the HTTP listener to leave its serve loop before using task
            # cancellation as the bounded fallback.
            if self._a2a_uvicorn is not None:
                self._a2a_uvicorn.should_exit = True

            try:
                await self.ingress.stop(grace=grace)
            except Exception:  # noqa: BLE001
                log.warning("ingress.shutdown_failed", exc_info=True)

            if self.a2a_server is not None:
                try:
                    await self.a2a_server.close()
                except Exception:  # noqa: BLE001
                    log.warning("a2a.shutdown_failed", exc_info=True)
                self.a2a_server = None

            reminders = self.reminders
            if reminders is not None:
                try:
                    await reminders.stop()
                except Exception:  # noqa: BLE001
                    log.warning("reminders.shutdown_failed", exc_info=True)

            await self._drain_active_turns(grace)
            await self._settle_tasks(
                list(self._maintenance_tasks),
                timeout=min(5.0, max(0.1, grace)),
                cancel=False,
                label="maintenance persistence",
            )
            await self._settle_tasks(
                core_background,
                timeout=min(2.0, max(0.1, grace)),
                cancel=True,
                label="background subsystem",
            )
            optional_pending = await self._settle_tasks(
                list(self._optional_tasks),
                timeout=min(2.0, max(0.1, grace)),
                cancel=False,
                label="optional subsystem",
            )
            if optional_pending:
                await self._settle_tasks(
                    list(optional_pending),
                    timeout=min(1.0, max(0.1, grace)),
                    cancel=True,
                    label="optional subsystem cancellation",
                )
            self._optional_tasks.clear()
            self._a2a_uvicorn = None
            self._capability_probe_task = None
            self._sleep_task = None
            self._warmup_task = None
            self._probe_task = None
            self._autonomy_task = None

            # A cancellation-resistant turn must not retain UI background
            # helpers beyond the process lifecycle.
            for streaming in list(self._active_stream_messages.values()):
                try:
                    await streaming.delete()
                except Exception:  # noqa: BLE001
                    log.debug("stream.shutdown_cleanup_failed", exc_info=True)
            self._active_stream_messages.clear()
            for typing_keep in list(self._active_typing_handles.values()):
                try:
                    await typing_keep.stop(timeout=1.5)
                except Exception:  # noqa: BLE001
                    log.debug("typing.shutdown_cleanup_failed", exc_info=True)
            self._active_typing_handles.clear()
            self._stop_channels.clear()

            # Active turns are drained, so these snapshots capture terminal
            # state instead of racing normal turn or idle-learning writers.
            curiosity = self._curiosity
            if curiosity is not None:
                try:
                    curiosity.save()
                except Exception:  # noqa: BLE001
                    log.warning("curiosity.shutdown_save_failed", exc_info=True)
            for graph in (self._causal_graph, self._temporal_graph):
                if graph is not None:
                    try:
                        graph.save()
                    except Exception:  # noqa: BLE001
                        log.warning("graph.shutdown_save_failed", exc_info=True)
            if self._user_model is not None:
                try:
                    self._user_model.flush()
                except Exception:  # noqa: BLE001
                    log.warning("user_model.shutdown_flush_failed", exc_info=True)
            if self._memory_store is not None:
                try:
                    self._hebbian.flush(self._memory_store.root)
                except Exception:  # noqa: BLE001
                    log.warning("hebbian.shutdown_flush_failed", exc_info=True)

            if self.mcp_client is not None:
                try:
                    await self.mcp_client.disconnect_all()
                except Exception:  # noqa: BLE001
                    log.warning("mcp.shutdown_failed", exc_info=True)
                self.mcp_client = None

            # Close the embedder's client before its owning gateway clients.
            if self._hybrid is not None:
                embedder = getattr(self._hybrid.embedding, "embedder", None)
                if embedder is not None and hasattr(embedder, "close"):
                    try:
                        await embedder.close()
                    except Exception:  # noqa: BLE001
                        log.debug("embedder.close_failed", exc_info=True)
            try:
                await self.gateway.aclose()
            except Exception:  # noqa: BLE001
                log.warning("gateway.shutdown_failed", exc_info=True)
            try:
                await self.events.append("runtime.stop", {})
            except Exception:  # noqa: BLE001
                log.warning("event_log.shutdown_append_failed", exc_info=True)
            self._shutdown_complete = True
