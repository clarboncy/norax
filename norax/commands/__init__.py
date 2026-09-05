"""Slash commands — intercepted before the brain hot-path.

Body starting with `/<word>` (and only that) routes here. Handlers are
side-effectful but synchronous-ish: they build a reply string and,
optionally, schedule post-reply effects (e.g. exec a restart).

Available commands (v1):
  /status         — process info, uptime, model, usage so far
  /models         — list known OpenAI-compatible model ids
  /model          — show current default model; `/model <id>` sets it
                    (owner only). Session-scoped; no config file write.
  /think          — show/set reasoning effort: off, low, medium, high, xhigh
  /reasoning      — show/toggle visible reasoning output: on/off
  /new            — reset this channel's rolling-window context
  /approvals      — list pending sensitive tool approvals (owner only)
  /approve        — approve one exact pending tool call (owner only)
  /reject         — reject one pending tool call (owner only)
  /stop           — graceful shutdown (owner only; no systemd restart)
  /restart        — graceful restart via systemd (owner only)
  /help           — list commands

Design notes:
  * Commands run in the runtime loop's context, so they have access to
    metrics, gateway, outbound registry, default_model, etc. via the
    `RuntimeHandle` passed to each handler.
  * Owner enforcement uses `env.sender.tier == "owner"`.
  * `/stop` and `/restart` must reply BEFORE tearing down; the handler
    returns a `deferred` coroutine that the runtime awaits after the
    reply send completes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import signal
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from ..version import BUILD_COMMIT as BUILD_COMMIT
from ..version import BUILD_DATE as BUILD_DATE
from ..version import __version__ as NORAX_VERSION

log = logging.getLogger("norax.commands")


# ---------------------------------------------------------------------------
# Runtime handle — what the handlers are allowed to touch
# ---------------------------------------------------------------------------


@dataclass
class RuntimeHandle:
    """Limited view of the Runtime passed to command handlers."""

    default_model: str  # mutable — setter sets it
    set_default_model: Callable[[str], bool | None]  # false means live-only; persistence failed
    started_at: float  # time.monotonic() at start
    wall_started_at: float  # time.time() at start
    metrics: Any
    gateway_base_url: str
    event_log: Any  # EventLog
    reset_window_for: Callable[[str], bool]  # (channel_id) -> bool
    outbound: Any
    dump_window_for: Callable[[str, bool], dict | None] = lambda _, __: None
    provider_urls: dict[str, str] | None = None  # {name: base_url}
    cancel_stream: Callable[[str], bool] = lambda _: False  # (channel_id) -> cancelled?
    gateway: Any = None  # GatewayClient/Router for preflight probes
    get_window_stats: Callable[[str], dict | None] = lambda _: None  # (chan_id) -> stats dict
    get_brain_stats: Callable[[], dict | None] = lambda: None  # -> brain stats dict
    thinking_effort: str = "medium"
    set_thinking_effort: Callable[[str], None] = lambda _: None
    reasoning_output: bool = False
    set_reasoning_output: Callable[[bool], None] = lambda _: None
    planning_mode: str = "direct"
    set_planning_mode: Callable[[str], bool | None] = lambda _: None
    max_tool_rounds: int = 0
    set_max_tool_rounds: Callable[[int], None] = lambda _: None
    memory_depth: str = "auto"
    set_memory_depth: Callable[[str], None] = lambda _: None
    weak_model_boost: str = "auto"
    set_weak_model_boost: Callable[[str], None] = lambda _: None
    stream_replies: bool = True
    set_stream_replies: Callable[[bool], None] = lambda _: None
    response_length: str = "balanced"
    set_response_length: Callable[[str], None] = lambda _: None
    tool_activity: str = "normal"
    set_tool_activity: Callable[[str], None] = lambda _: None
    custom_model_catalog: Callable[[], dict[str, list[tuple[str, str]]]] = lambda: {}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@dataclass
class ParsedCommand:
    name: str  # e.g. "status"
    args: list[str]  # shlex-split remainder
    raw: str  # full body verbatim


def parse(body: str) -> ParsedCommand | None:
    """Return ParsedCommand if `body` starts with a recognizable slash
    command, else None. Only the FIRST token is the command."""
    s = (body or "").strip()
    if not s.startswith("/"):
        return None
    # Strip leading slash, take the first whitespace-delimited token.
    head, _, tail = s[1:].partition(" ")
    head = head.strip()
    if not head:
        return None
    # Command name: lowercase, alphanumeric + underscore/dash only.
    cmd = head.lower()
    if not all(c.isalnum() or c in "_-" for c in cmd):
        return None
    try:
        args = shlex.split(tail) if tail.strip() else []
    except ValueError:
        args = tail.split()
    return ParsedCommand(name=cmd, args=args, raw=s)


# ---------------------------------------------------------------------------
# Handler result
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    reply: str
    # Coroutine to run AFTER the reply is sent (for stop/restart).
    post_send: Callable[[], Awaitable[None]] | None = None
    data: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _progress_bar(pct: float, width: int = 12) -> str:
    """Render a compact Discord-safe usage bar."""
    try:
        value = float(pct)
    except Exception:  # noqa: BLE001
        value = 0.0
    filled = max(0, min(width, round(value / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def _usage_line(label: str, pct: Any, *, reset_at: Any = None, note: str = "") -> str:
    """One aligned quota line for /status."""
    try:
        value = max(0.0, min(100.0, float(pct)))
        pct_s = f"{value:.0f}%"
    except Exception:  # noqa: BLE001
        value = 0.0
        pct_s = "?%"
    reset = f" reset {_fmt_reset(reset_at)}" if reset_at else ""
    tail = f" {note}" if note else ""
    return f"{label:<6} [{_progress_bar(value, 14)}] {pct_s:>4}{reset}{tail}"


def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _sum_counter(counter, labels: dict[str, str] | None = None) -> float:
    """Sum a Prometheus counter's samples; optionally filter by labels."""
    total = 0.0
    try:
        for metric in counter.collect():
            for sample in metric.samples:
                if not sample.name.endswith("_total"):
                    continue
                if labels:
                    if not all(sample.labels.get(k) == v for k, v in labels.items()):
                        continue
                total += sample.value
    except Exception:  # noqa: BLE001
        return 0.0
    return total


def _fmt_reset(ts: int | float | None) -> str:
    if not ts:
        return "unknown"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%b %-d %-I:%M %p")
    except Exception:  # noqa: BLE001
        return "unknown"


async def _gateway_ping(provider_urls: dict[str, str]) -> dict[str, str]:
    """Ping configured providers concurrently without blocking the runtime loop."""

    async def _ping_one(client: httpx.AsyncClient, name: str, base_url: str) -> tuple[str, str]:
        probe = base_url.rstrip("/") + "/models"
        token = os.environ.get(f"NORAX_{name.upper()}_TOKEN")
        token = token or os.environ.get("NORAX_GATEWAY_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        try:
            r = await client.get(probe, headers=headers)
            if r.status_code < 400:
                return name, "ok"
            try:
                payload = r.json()
                error = payload.get("error") if isinstance(payload, dict) else None
                msg = (error.get("message") if isinstance(error, dict) else "") or ""
            except Exception:  # noqa: BLE001
                msg = ""
            status = f"HTTP {r.status_code}" + (f": {str(msg).strip()[:60]}" if msg else "")
            return name, status
        except Exception as e:  # noqa: BLE001
            return name, type(e).__name__

    async with httpx.AsyncClient(timeout=3.0) as client:
        pairs = await asyncio.gather(
            *(_ping_one(client, name, base_url) for name, base_url in provider_urls.items())
        )
    return dict(pairs)


def _is_owner(env: Any) -> bool:
    return getattr(getattr(env, "sender", None), "tier", "") == "owner"


# ---------------------------------------------------------------------------
# Built-in command handlers
# ---------------------------------------------------------------------------

# Models we explicitly advertise. `/model <alias>` sends the alias through
# to the gateway router's `model` field; the router picks the provider.
# Discord renders provider + model as two dropdowns to avoid the 25-option
# single-select ceiling and make provider routing obvious.
# Price tiers:
#   💎 Premium  — highest quality, most expensive
#   ⭐ Standard  — solid quality, moderate cost
#   💰 Budget    — cheaper, faster
#   🆓 Free      — local/self-hosted, zero marginal cost
#
# The public release ships with Ollama and OpenRouter. Users add their own
# providers via the dashboard Settings tab or the /api/providers API.
MODEL_PROVIDERS: dict[str, list[tuple[str, str, str]]] = {
    "ollama": [
        # Cloud models only — local inference is listed under ollama_local.
        ("kimi-k2.7-code:cloud", "Kimi K2.7 Code", "☁️"),
        ("glm-5.3:cloud", "GLM 5.3", "☁️"),
        ("glm-5.2:cloud", "GLM 5.2", "☁️"),
        ("qwen3-coder-next:cloud", "Qwen3 Coder Next", "☁️"),
        ("qwen3.5:cloud", "Qwen 3.5", "☁️"),
        ("deepseek-v4-pro:cloud", "DeepSeek V4 Pro", "☁️"),
        ("deepseek-v4-flash:cloud", "DeepSeek V4 Flash", "☁️"),
        ("gemma4:31b-cloud", "Gemma 4 31B", "☁️"),
        ("nemotron-3-super:cloud", "Nemotron 3 Super", "☁️"),
    ],
    "ollama_local": [
        # Local inference — models pulled via `ollama pull` and run locally.
        ("qwen3.8-27b-fast:latest", "Qwen 3.8 27B Fast (local)", "�"),
        ("norax-gemma4-12b-agentic:latest", "Norax Gemma4 12B Agentic (local)", "🏠"),
    ],
    "codex_direct": [
        # Codex CLI OAuth proxy on :4146 — gpt-5.5+ family.
        ("gpt-5.6-sol", "GPT 5.6 Sol", "🔑"),
        ("gpt-5.6-terra", "GPT 5.6 Terra", "🔑"),
        ("gpt-5.6-luna", "GPT 5.6 Luna", "🔑"),
        ("gpt-5.5", "GPT 5.5", "🔑"),
        ("gpt-5.5-personal", "GPT 5.5 Personal", "🔑"),
        ("gpt-5.5-B", "GPT 5.5 B", "🔑"),
    ],
    "openrouter": [
        ("openrouter/anthropic/claude-sonnet-4", "Claude Sonnet 4", "⭐"),
        ("openrouter/openai/gpt-4o-mini", "GPT-4o Mini", "💰"),
        ("openrouter/google/gemini-flash-1.5", "Gemini Flash 1.5", "💰"),
        ("openrouter/meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B", "⭐"),
        ("openrouter/qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B", "⭐"),
        ("openrouter/deepseek/deepseek-chat", "DeepSeek Chat", "💰"),
    ],
}

PROVIDER_LABELS: dict[str, str] = {
    "ollama": "Ollama Cloud",
    "ollama_local": "Local Inference",
    "codex_direct": "Codex Direct",
    "openrouter": "OpenRouter",
}

BUILTIN_MODELS: list[tuple[str, str]] = [
    (mid, f"{label} · {PROVIDER_LABELS.get(provider, provider)}")
    for provider, models in MODEL_PROVIDERS.items()
    for mid, label, _price in models
]


def provider_for_model(model_id: str) -> str | None:
    for provider, models in MODEL_PROVIDERS.items():
        if any(mid == model_id for mid, _label, _price in models):
            return provider
    if model_id.startswith("openrouter/"):
        return "openrouter"
    if model_id.startswith("anthropic/"):
        return "openrouter"
    if model_id.startswith("moonshotai/"):
        return "openrouter"
    if model_id.startswith("openai-codex/") or model_id.startswith("gpt-5.") or model_id == "gpt-5":
        return "codex_direct"
    if ":cloud" in model_id or model_id.endswith("-cloud"):
        return "ollama"
    if model_id.endswith(":latest"):
        return "ollama_local"
    if ":" in model_id:
        return "ollama"
    return None


def provider_options_for_discord() -> list[tuple[str, str]]:
    return [(pid, PROVIDER_LABELS.get(pid, pid)) for pid in MODEL_PROVIDERS]


def model_options_for_provider(provider: str) -> list[tuple[str, str]]:
    """Return (model_id, display_label_with_price) for Discord dropdown."""
    rows = MODEL_PROVIDERS.get(provider) or []
    # Dedup by model id: Discord Select rejects duplicate option values with
    # HTTP 400 (error 50035), which silently breaks the whole /model & /settings
    # panels. Keep first occurrence and preserve order.
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for mid, label, price in rows:
        if mid in seen:
            continue
        seen.add(mid)
        out.append((mid, f"{price} {label}"))
    return out


def model_options_for_discord(provider: str | None = None) -> list[tuple[str, str]]:
    """Discord Select supports max 25 options per dropdown."""
    if provider:
        return model_options_for_provider(provider)[:25]
    return BUILTIN_MODELS[:25]


def _known_model_ids() -> set[str]:
    return {mid for mid, _ in BUILTIN_MODELS}


# ---- Live Ollama model discovery ----------------------------------------


_OLLAMA_MODEL_CACHE: tuple[float, list[tuple[str, str, str]]] | None = None
_OLLAMA_REFRESH_TASK: asyncio.Task[list[tuple[str, str, str]]] | None = None
_OLLAMA_CACHE_TTL = 30.0
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,199}$")


def _ollama_base_url() -> str:
    return os.environ.get("NORAX_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def _ollama_model_tier(model_id: str) -> str:
    """Pick a price-tier emoji based on model name heuristics."""
    mid = model_id.lower()
    if ":cloud" in mid:
        return "🅞"
    if any(x in mid for x in (":latest", ":q4", ":q5", ":q6", ":q8", ":iq4", ":fp16")):
        return "🖥️"
    if "embed" in mid:
        return "📎"
    return "🆓"


def _ollama_model_label(model_id: str) -> str:
    """Turn `llama3.2:3b` into `Llama 3.2 3B`."""
    name, _, tag = model_id.partition(":")
    parts = []
    for chunk in name.replace("-", " ").replace("_", " ").split():
        if chunk.isdigit():
            parts.append(chunk)
        else:
            parts.append(chunk.capitalize())
    label = " ".join(parts)
    if tag:
        tag_label = tag.replace("-", " ")
        tag_parts = " ".join(
            chunk.upper() if chunk.isdigit() else chunk.capitalize() for chunk in tag_label.split()
        )
        label += f" {tag_parts}"
    return label


async def fetch_ollama_models(force_refresh: bool = False) -> list[tuple[str, str, str]]:
    """Poll the local Ollama daemon for installed models.

    Returns a list of (model_id, label, price_tier) tuples. On failure or
    cache hit, returns the cached or empty list. This merges local models
    (pulled via `ollama pull`) and cloud models (e.g. `:cloud` tags).
    """
    global _OLLAMA_MODEL_CACHE, _OLLAMA_REFRESH_TASK
    now = time.monotonic()
    if (
        not force_refresh
        and _OLLAMA_MODEL_CACHE is not None
        and now - _OLLAMA_MODEL_CACHE[0] < _OLLAMA_CACHE_TTL
    ):
        return list(_OLLAMA_MODEL_CACHE[1])

    task = _OLLAMA_REFRESH_TASK
    if task is None or task.done():
        task = asyncio.create_task(_discover_ollama_models())
        _OLLAMA_REFRESH_TASK = task

        def _clear(completed: asyncio.Task[list[tuple[str, str, str]]]) -> None:
            global _OLLAMA_MODEL_CACHE, _OLLAMA_REFRESH_TASK
            if _OLLAMA_REFRESH_TASK is completed:
                _OLLAMA_REFRESH_TASK = None
            if not completed.cancelled() and completed.exception() is None:
                _OLLAMA_MODEL_CACHE = (time.monotonic(), completed.result())

        task.add_done_callback(_clear)

    discovered = await asyncio.shield(task)
    _OLLAMA_MODEL_CACHE = (time.monotonic(), discovered)
    return list(discovered)


async def _discover_ollama_models() -> list[tuple[str, str, str]]:
    """Perform one bounded Ollama discovery request."""

    base = _ollama_base_url()
    discovered: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base}/api/tags")
            resp.raise_for_status()
            payload = resp.json()
        models = payload.get("models") if isinstance(payload, dict) else None
        if isinstance(models, list):
            for entry in models:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                if not _MODEL_ID_RE.fullmatch(name) or name in seen:
                    continue
                seen.add(name)
                tier = _ollama_model_tier(name)
                label = _ollama_model_label(name)
                discovered.append((name, label, tier))
                if len(discovered) >= 500:
                    break
    except Exception:  # noqa: BLE001
        log.debug("fetch_ollama_models.failed", exc_info=True)
    return discovered


async def _fetch_ollama_names() -> set[str]:
    """Return just the model names from the local Ollama daemon."""
    models = await fetch_ollama_models()
    return {mid for mid, _label, _tier in models}


def ollama_models_sync(force_refresh: bool = False) -> list[tuple[str, str, str]]:
    """Fetch from synchronous code; return cache when already in an event loop.

    Blocking a running event loop to bridge async discovery stalls every active
    request and can deadlock. Async callers that require freshness must await
    :func:`fetch_ollama_models` directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(fetch_ollama_models(force_refresh))
    if _OLLAMA_MODEL_CACHE is None:
        return []
    return list(_OLLAMA_MODEL_CACHE[1])


def merged_ollama_options(
    discovered: list[tuple[str, str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Return cloud-tagged Ollama models (curated + live :cloud discoveries).

    Local models (:latest, bare tags) are NOT included here — they belong
    in the ollama_local section. This keeps the cloud provider list clean.
    """
    base = list(model_options_for_provider("ollama"))
    seen = {mid for mid, _ in base}
    live_models = ollama_models_sync() if discovered is None else discovered
    for mid, label, tier in live_models:
        if not mid.endswith(":cloud") or mid in seen:
            continue
        seen.add(mid)
        base.append((mid, f"{tier} {label}"))
    return base[:25]


def merged_ollama_local_options(
    discovered: list[tuple[str, str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Return local inference models (curated + live :latest discoveries).

    Only :latest models are merged from discovery — bare tags (e.g.
    `gemma4:12b`) and embedding models are excluded to keep the local
    list clean and avoid advertising internal artifacts. Cloud models
    (:cloud) belong in the ollama (cloud) section.
    """
    base = list(model_options_for_provider("ollama_local"))
    seen = {mid for mid, _ in base}
    live_models = ollama_models_sync() if discovered is None else discovered
    for mid, label, tier in live_models:
        if not mid.endswith(":latest") or mid in seen:
            continue
        # Skip embedding models — they're not chat models
        if "embed" in mid.lower():
            continue
        seen.add(mid)
        base.append((mid, f"{tier} {label}"))
    return base[:25]


def model_catalog_for_runtime(
    rt: RuntimeHandle,
    *,
    ollama_models: list[tuple[str, str, str]] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    catalog = {provider: model_options_for_provider(provider) for provider in MODEL_PROVIDERS}
    if "ollama" in catalog:
        catalog["ollama"] = merged_ollama_options(ollama_models)
    if "ollama_local" in catalog:
        catalog["ollama_local"] = merged_ollama_local_options(ollama_models)
    for provider, models in rt.custom_model_catalog().items():
        catalog[provider] = [(str(model), str(label)) for model, label in models]
    return catalog


COMMANDS = {
    "status",
    "models",
    "model",
    "settings",
    "planning",
    "think",
    "reasoning",
    "rounds",
    "memory",
    "boost",
    "stream",
    "length",
    "activity",
    "new",
    "dump",
    "approvals",
    "approve",
    "reject",
    "stop",
    "restart",
    "help",
}

PLANNING_MODES = {"direct", "orchestrator"}
PLANNING_LABELS = {
    "direct": "Direct — single-model agent loop",
    "orchestrator": "Orchestrator — strong model plans, fast model executes",
}

THINK_LABELS = {
    "off": "Off — fastest replies",
    "low": "Low — light reasoning",
    "medium": "Medium — balanced default",
    "high": "High — deeper analysis",
    "xhigh": "Extra high — heavy reasoning budget",
    "max": "Max — highest reasoning budget (GPT-5.6 Sol or GLM-5.3)",
    "ultra": "Ultra — maximum depth reasoning (GPT-5.6 Sol only)",
}


def _is_gpt_56_sol(model: str) -> bool:
    """Return True iff the model id is the GPT-5.6 Sol variant."""
    m = (model or "").lower().strip()
    return m == "gpt-5.6-sol" or m.endswith("/gpt-5.6-sol")


def _supports_max_reasoning(model: str) -> bool:
    """Return True iff the model supports the 'max' reasoning level.

    GPT-5.6 Sol and GLM-5.3 / GLM-5.3-Flash cloud models all support 'max'.
    """
    m = (model or "").lower().strip()
    if _is_gpt_56_sol(m):
        return True
    # Strip provider prefix (e.g. "ollama/glm-5.3:cloud" → "glm-5.3:cloud")
    if "/" in m:
        m = m.rsplit("/", 1)[1]
    return m in ("glm-5.3:cloud", "glm-5.3-flash:cloud", "kimi-k3:cloud")


ROUND_CAPS = (0, 6, 12, 24, 48)
ROUND_LABELS = {
    0: "Auto — use the agent loop's bounded default",
    6: "6 rounds — quick tasks",
    12: "12 rounds — moderate agent work",
    24: "24 rounds — heavy multi-step",
    48: "48 rounds — deep automation",
}


def _rounds_display(cap: int) -> str:
    if cap != 0:
        return str(cap)
    from ..brain.agent_loop import DEFAULT_MAX_ROUNDS, HARD_ROUND_CAP

    return f"auto ({min(DEFAULT_MAX_ROUNDS, HARD_ROUND_CAP)})"


def _hard_round_cap() -> int:
    from ..brain.agent_loop import HARD_ROUND_CAP

    return HARD_ROUND_CAP


MEMORY_DEPTHS = {"auto", "light", "balanced", "deep"}
MEMORY_DEPTH_LABELS = {
    "auto": "Auto — adapt to query complexity",
    "light": "Light — 3 memory hits (fast/cheap)",
    "balanced": "Balanced — 5 hits (default depth)",
    "deep": "Deep — 7 hits (research/audit)",
}

BOOST_MODES = {"auto", "on", "off"}
BOOST_LABELS = {
    "auto": "Auto — boost local/weak models only",
    "on": "On — always apply tool-call scaffolding",
    "off": "Off — never apply boosters",
}

STREAM_LABELS = {
    "on": "On — live token streaming in Discord",
    "off": "Off — single reply after completion",
}

LENGTH_PREFS = {"concise", "balanced", "detailed"}
LENGTH_LABELS = {
    "concise": "Concise — short replies, minimal filler",
    "balanced": "Balanced — default depth",
    "detailed": "Detailed — thorough explanations when useful",
}

ACTIVITY_MODES = {"minimal", "normal", "verbose"}
ACTIVITY_LABELS = {
    "minimal": "Minimal — silent tool calls",
    "normal": "Normal — brief multi-step narration",
    "verbose": "Verbose — narrate each tool step clearly",
}


def settings_snapshot(
    rt: RuntimeHandle,
    *,
    ollama_models: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Structured settings for Discord UI and /settings."""
    from ..brain.orchestrator import effective_planning_route

    route = effective_planning_route(rt.planning_mode, rt.default_model)
    return {
        "model": rt.default_model,
        "planning_mode": rt.planning_mode,
        "planning_active": route["active"],
        "planning_auto": route["auto"],
        "planner_model": route["planner"],
        "executor_model": route["executor"],
        "planning_route": route["label"],
        "thinking_effort": rt.thinking_effort,
        "reasoning_output": rt.reasoning_output,
        "max_tool_rounds": rt.max_tool_rounds,
        "memory_depth": rt.memory_depth,
        "weak_model_boost": rt.weak_model_boost,
        "stream_replies": rt.stream_replies,
        "response_length": rt.response_length,
        "tool_activity": rt.tool_activity,
        "model_catalog": model_catalog_for_runtime(rt, ollama_models=ollama_models),
    }


def format_settings_summary(rt: RuntimeHandle) -> str:
    snap = settings_snapshot(rt)
    reasoning = "on" if snap["reasoning_output"] else "off"
    rounds = snap["max_tool_rounds"]
    rounds_label = _rounds_display(rounds)
    route_line = snap.get("planning_route") or PLANNING_LABELS.get(
        snap["planning_mode"],
        snap["planning_mode"],
    )
    return (
        "**Norax settings**\n"
        f"Model: `{snap['model']}`\n"
        f"Planning: `{snap['planning_mode']}` — {route_line}\n"
        f"Planner: `{snap.get('planner_model', snap['model'])}`\n"
        f"Executor: `{snap.get('executor_model', snap['model'])}`\n"
        f"Thinking: `{snap['thinking_effort']}`\n"
        f"Reasoning output: `{reasoning}`\n"
        f"Tool rounds: `{rounds_label}`\n"
        f"Memory depth: `{snap['memory_depth']}`\n"
        f"Weak-model boost: `{snap['weak_model_boost']}`\n"
        f"Stream replies: `{'on' if snap['stream_replies'] else 'off'}`\n"
        f"Response length: `{snap['response_length']}`\n"
        f"Tool activity: `{snap['tool_activity']}`\n\n"
        "Use `/settings` in Discord for the unified selector panel.\n"
        "Set `/planning orchestrator` to force dual-model mode on any model."
    )


async def handle(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    name = pc.name
    if name == "help":
        return _cmd_help()
    if name == "status":
        return await _cmd_status(env, rt)
    if name == "models":
        return await _cmd_models(rt)
    if name == "model":
        return await _cmd_model(pc, env, rt)
    if name == "settings":
        return await _cmd_settings(rt)
    if name == "planning":
        return await _cmd_planning(pc, env, rt)
    if name == "think":
        return await _cmd_think(pc, env, rt)
    if name == "rounds":
        return await _cmd_rounds(pc, env, rt)
    if name == "memory":
        return await _cmd_memory(pc, env, rt)
    if name == "boost":
        return await _cmd_boost(pc, env, rt)
    if name == "stream":
        return await _cmd_stream(pc, env, rt)
    if name == "length":
        return await _cmd_length(pc, env, rt)
    if name == "activity":
        return await _cmd_activity(pc, env, rt)
    if name == "reasoning":
        return await _cmd_reasoning(pc, env, rt)
    if name == "new":
        return await _cmd_new(env, rt)
    if name == "dump":
        return await _cmd_dump(pc, env, rt)
    if name == "approvals":
        return _cmd_approvals(env)
    if name in {"approve", "reject"}:
        return _cmd_resolve_interrupt(pc, env)
    if name == "stop":
        return _cmd_stop(env, rt)
    if name == "restart":
        return _cmd_restart(env, rt)
    # Unrecognized: not ours; caller should route to brain.
    return CommandResult(reply="")  # sentinel: empty reply → let caller fall through


def is_known(pc: ParsedCommand) -> bool:
    return pc.name in COMMANDS


# ---- /help ---------------------------------------------------------------


def _cmd_help() -> CommandResult:
    lines = [
        "**Norax commands**",
        "`/status` — runtime status (uptime, model, tokens)",
        "`/models` — list available models",
        "`/model [id]` — show or set the default model (owner sets)",
        "`/settings` — unified settings panel (model, think, planning…)",
        "`/planning [direct|orchestrator]` — show/set planning mode",
        "`/think [off|low|medium|high|xhigh]` — show/set reasoning effort",
        "`/reasoning [on|off]` — show/toggle visible reasoning output",
        "`/new` — reset this channel's conversation context",
        "`/dump [half]` — dump context to sleep; full clears body, half clears older 50% (owner)",
        "`/approvals` — list pending sensitive tool calls (owner)",
        "`/approve <id>` — approve one exact pending call (owner)",
        "`/reject <id>` — reject one pending call (owner)",
        "`/stop` — graceful shutdown (owner)",
        "`/restart` — graceful restart (owner)",
        "`/help` — this text",
    ]
    return CommandResult(reply="\n".join(lines))


# ---- /approvals, /approve, /reject --------------------------------------


def _cmd_approvals(env: Any) -> CommandResult:
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can view pending approvals.")
    from ..runtime.interrupts import get_interrupt_manager

    pending = get_interrupt_manager().get_pending()
    if not pending:
        return CommandResult(reply="No tool calls are awaiting approval.")
    now = time.time()
    lines = ["**Pending tool approvals**"]
    for item in pending[:25]:
        age = max(0, int(now - item.timestamp))
        description = item.description or item.reason.value
        lines.append(f"`{item.interrupt_id}` — `{item.tool_name}` — {description} ({age}s ago)")
    lines.append("Use `/approve <id>` or `/reject <id>`.")
    return CommandResult(reply="\n".join(lines))


def _cmd_resolve_interrupt(pc: ParsedCommand, env: Any) -> CommandResult:
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can resolve tool approvals.")
    if len(pc.args) != 1:
        return CommandResult(reply=f"Usage: `/{pc.name} <interrupt_id>`.")

    from ..runtime.interrupts import InterruptAction, get_interrupt_manager

    interrupt_id = pc.args[0].strip()
    if not interrupt_id.startswith("interrupt_") or len(interrupt_id) > 128:
        return CommandResult(reply="Invalid interrupt ID.")
    action = InterruptAction.APPROVE if pc.name == "approve" else InterruptAction.REJECT
    resolved = get_interrupt_manager().resolve_interrupt(
        interrupt_id,
        action,
        resolved_by=str(env.sender.id),
    )
    if not resolved:
        return CommandResult(reply="That interrupt is missing, expired, or already resolved.")
    verb = "Approved" if action is InterruptAction.APPROVE else "Rejected"
    suffix = (
        " Retry the exact call or say continue; this approval is single-use."
        if action is InterruptAction.APPROVE
        else " The rejected call will not execute."
    )
    return CommandResult(reply=f"{verb} `{interrupt_id}`.{suffix}")


# ---- /status -------------------------------------------------------------


async def _cmd_status(env: Any, rt: RuntimeHandle) -> CommandResult:
    m = rt.metrics
    up = _fmt_dur(time.monotonic() - rt.started_at)
    ingress_total = _sum_counter(m.ingress_total)
    turns_total = _sum_counter(m.brain_turns)
    errors_total = _sum_counter(m.brain_errors)
    tokens_in = _sum_counter(m.gateway_tokens_in)
    tokens_out = _sum_counter(m.gateway_tokens_out)
    gw_requests = _sum_counter(m.gateway_requests)

    def _k(n: float) -> str:
        n = int(n)
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(n)

    def _fit(text: Any, width: int) -> str:
        value = str(text)
        if len(value) > width:
            return value[: max(0, width - 1)] + "…"
        return value.ljust(width)

    def _line(label: str, value: Any = "") -> str:
        return f"{_fit(label.upper(), 10)} {_fit(value, 48)}"

    def _section(name: str) -> str:
        return f"-- {name.upper()} "

    raw = getattr(env, "raw", None) or {}
    chan_id = raw.get("channel_id") or ""
    health = "DEGRADED" if errors_total else "ONLINE"
    providers = ", ".join((rt.provider_urls or {"default": ""}).keys())

    from ..brain.orchestrator import EXECUTOR_MODEL, should_use_orchestrator
    from ..brain.strong_model_scaffold import resolve_planner_model

    if should_use_orchestrator(rt.planning_mode, rt.default_model):
        planner = resolve_planner_model(rt.default_model)
        planning_line = f"orch · {planner} → {EXECUTOR_MODEL}"
    else:
        planning_line = f"{rt.planning_mode} · single-model"

    lines = [
        "NORAX//STATUS",
        "```ansi",
        _line("core", f"{health} · v{NORAX_VERSION}"),
        _line("model", rt.default_model),
        _line("planning", planning_line),
        _line("uptime", up),
        _line("provider", providers),
        "",
        _section("traffic"),
        _line(
            "ingress", f"{_k(ingress_total)} · turns {_k(turns_total)} · gw_req {_k(gw_requests)}"
        ),
        _line("tokens", f"in {_k(tokens_in)} · out {_k(tokens_out)} · err {_k(errors_total)}"),
        "",
        _section("gateway"),
    ]

    ping_urls = rt.provider_urls or {}
    if ping_urls:
        pings = await _gateway_ping(ping_urls)
        for pname, pstatus in pings.items():
            lines.append(_line(pname, pstatus))
    else:
        lines.append(_line("gateway", "no providers configured"))

    lines.extend(["", _section("context")])
    ws = rt.get_window_stats(chan_id) if chan_id else None
    if ws:
        lines.extend(
            [
                _line("window", f"[{_progress_bar(ws['usage_pct'], 14)}] {ws['usage_pct']}%"),
                _line(
                    "tokens",
                    f"{_k(ws['current_tokens'])} / {_k(ws['budget_tokens'])} · frames {ws['body_frames']} · head {_k(ws['head_tokens'])}",
                ),
            ]
        )
    else:
        lines.append(_line("window", "no active window"))

    bs = rt.get_brain_stats()
    if bs:
        lines.extend(["", _section("brain")])
        mem = bs.get("memory")
        if mem:
            lines.append(
                _line(
                    "memory",
                    f"hot {mem['hot']} · canonical {mem['canonical']} · sleep {mem['sleep']}",
                )
            )
        ep = bs.get("episodic")
        if ep and ep.get("days"):
            lines.append(_line("episodic", f"{ep['days']} days · {ep['total_size_kb']} KB"))
        ml = bs.get("meta_learning")
        if ml:
            drift_warn = " !" if ml["drift"] > ml["drift_limit"] * 0.7 else ""
            lines.append(
                _line(
                    "meta",
                    f"drift {ml['drift']:.3f}/{ml['drift_limit']}{drift_warn} · epochs {ml['epochs']} · frozen {len(ml['frozen'])}",
                )
            )
        heb = bs.get("hebbian_turns")
        if heb:
            lines.append(_line("hebbian", f"{heb} turns"))

    lines.extend(
        [
            "",
            _section("caller"),
            _line("user", f"{env.sender.label} · {env.sender.tier}"),
            "```",
        ]
    )

    return CommandResult(reply="\n".join(lines))


# ---- /models -------------------------------------------------------------


async def _cmd_models(rt: RuntimeHandle) -> CommandResult:
    cur = (rt.default_model or "").strip()
    cur_bare = cur.split("/")[-1]
    live_ollama = await fetch_ollama_models()
    catalog = model_catalog_for_runtime(rt, ollama_models=live_ollama)
    rows = [item for models in catalog.values() for item in models]
    has_exact = any(mid == cur for mid, _ in rows)
    marked = False
    lines = ["**Available models**"]
    for provider, models in catalog.items():
        if not models:
            continue
        lines.append(f"\n**{provider}**")
        for mid, desc in models:
            is_cur = (mid == cur) or (
                not has_exact
                and not marked
                and "/" not in cur
                and cur_bare
                and mid.split("/")[-1] == cur_bare
            )
            if is_cur:
                marked = True
            marker = " ← current" if is_cur else ""
            lines.append(f"• `{mid}` — {desc}{marker}")
    lines.append("\nSet with `/model <id>` (owner only).")
    return CommandResult(reply="\n".join(lines), data={"model_catalog": catalog})


# ---- /model --------------------------------------------------------------


async def _cmd_model(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        return CommandResult(
            reply=f"Current model: `{rt.default_model}`",
            data={"model": rt.default_model},
        )
    target = pc.args[0].strip()
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change the model.")

    # The dashboard UI sends `/model <provider>/<model>` (e.g. `ollama/llama3.2:3b`).
    # Strip the provider prefix for providers that use bare model ids when
    # talking to the upstream API (ollama, codex_direct, ollama_local).
    # openrouter keeps its prefix as part of the model id.
    if "/" in target and not target.startswith("openrouter/"):
        prefix, _, rest = target.partition("/")
        if prefix in MODEL_PROVIDERS or prefix in {"ollama"}:
            target = rest

    known = {mid for models in model_catalog_for_runtime(rt).values() for mid, _label in models}
    if target not in known:
        # Check live Ollama API for models that look like Ollama models
        # (contain ':' like glm-5.2:cloud, qwen3.5:latest, etc.)
        ollama_verified = False
        if ":" in target:
            try:
                ollama_names = await _fetch_ollama_names()
                ollama_verified = target in ollama_names
            except Exception:
                pass
        persisted = rt.set_default_model(target)
        persistence_note = (
            " Active for this process, but private settings persistence failed."
            if persisted is False
            else ""
        )
        if ollama_verified:
            return CommandResult(reply=f"Model set to `{target}`.{persistence_note}")
        return CommandResult(
            reply=(
                f"Set model to `{target}` (not in /models list — will forward as-is; "
                f"upstream may reject).{persistence_note}"
            )
        )
    persisted = rt.set_default_model(target)
    persistence_note = (
        " Active for this process, but private settings persistence failed."
        if persisted is False
        else ""
    )
    return CommandResult(reply=f"Model set to `{target}`.{persistence_note}")


# ---- /settings -----------------------------------------------------------


async def _cmd_settings(rt: RuntimeHandle) -> CommandResult:
    live_ollama = await fetch_ollama_models()
    snap = settings_snapshot(rt, ollama_models=live_ollama)
    return CommandResult(
        reply=format_settings_summary(rt),
        data=snap,
    )


# ---- /planning -----------------------------------------------------------


async def _cmd_planning(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        label = PLANNING_LABELS.get(rt.planning_mode, rt.planning_mode)
        return CommandResult(
            reply=(
                f"Planning mode: `{rt.planning_mode}` — {label}. "
                "Set with `/planning direct|orchestrator`."
            )
        )
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change planning mode.")
    mode = pc.args[0].strip().lower()
    aliases = {"orch": "orchestrator", "dual": "orchestrator", "single": "direct"}
    mode = aliases.get(mode, mode)
    if mode not in PLANNING_MODES:
        return CommandResult(reply="Usage: `/planning direct|orchestrator`.")
    changed = rt.set_planning_mode(mode)
    if changed is False:
        return CommandResult(
            reply="Planner connector is disabled; enable it before selecting orchestrator mode."
        )
    label = PLANNING_LABELS.get(mode, mode)
    return CommandResult(reply=f"Planning mode set to `{mode}` — {label}.")


# ---- /rounds -------------------------------------------------------------


async def _cmd_rounds(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    current = rt.max_tool_rounds
    label = _rounds_display(current)
    if not pc.args:
        return CommandResult(
            reply=(
                f"Tool round cap: `{label}`. "
                "Set with `/rounds auto|6|12|24|48`. "
                f"There is no unlimited mode — a hard safety cap of {_hard_round_cap()} rounds "
                "and the configured wall-clock deadline always apply."
            ),
            data={"max_tool_rounds": current},
        )
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change the tool round cap.")
    raw = pc.args[0].strip().lower()
    aliases = {"auto": 0, "unlimited": 0, "none": 0, "0": 0, "inf": 0}
    if raw in aliases:
        cap = aliases[raw]
    else:
        try:
            cap = int(raw)
        except ValueError:
            return CommandResult(reply="Usage: `/rounds auto|6|12|24|48`.")
    if cap not in ROUND_CAPS:
        return CommandResult(reply="Usage: `/rounds auto|6|12|24|48`.")
    hard_cap = _hard_round_cap()
    if cap > hard_cap:
        return CommandResult(
            reply=(
                f"`{cap}` exceeds this process's hard safety cap of {hard_cap} rounds. "
                "Choose a lower value or raise `NORAX_AGENT_HARD_ROUND_CAP` before startup."
            ),
            data={"max_tool_rounds": current, "hard_round_cap": hard_cap},
        )
    rt.set_max_tool_rounds(cap)
    out = _rounds_display(cap)
    return CommandResult(
        reply=f"Tool round cap set to `{out}`.",
        data={"max_tool_rounds": cap},
    )


# ---- /memory -------------------------------------------------------------


async def _cmd_memory(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        label = MEMORY_DEPTH_LABELS.get(rt.memory_depth, rt.memory_depth)
        return CommandResult(
            reply=(
                f"Memory depth: `{rt.memory_depth}` — {label}. "
                "Set with `/memory auto|light|balanced|deep`."
            ),
            data={"memory_depth": rt.memory_depth},
        )
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change memory depth.")
    mode = pc.args[0].strip().lower()
    aliases = {"default": "auto", "fast": "light", "normal": "balanced", "max": "deep"}
    mode = aliases.get(mode, mode)
    if mode not in MEMORY_DEPTHS:
        return CommandResult(reply="Usage: `/memory auto|light|balanced|deep`.")
    rt.set_memory_depth(mode)
    label = MEMORY_DEPTH_LABELS.get(mode, mode)
    return CommandResult(
        reply=f"Memory depth set to `{mode}` — {label}.",
        data={"memory_depth": mode},
    )


# ---- /boost --------------------------------------------------------------


async def _cmd_boost(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        label = BOOST_LABELS.get(rt.weak_model_boost, rt.weak_model_boost)
        return CommandResult(
            reply=(
                f"Weak-model boost: `{rt.weak_model_boost}` — {label}. "
                "Set with `/boost auto|on|off`."
            ),
            data={"weak_model_boost": rt.weak_model_boost},
        )
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change weak-model boost.")
    mode = pc.args[0].strip().lower()
    aliases = {"enable": "on", "disable": "off", "true": "on", "false": "off"}
    mode = aliases.get(mode, mode)
    if mode not in BOOST_MODES:
        return CommandResult(reply="Usage: `/boost auto|on|off`.")
    rt.set_weak_model_boost(mode)
    label = BOOST_LABELS.get(mode, mode)
    return CommandResult(
        reply=f"Weak-model boost set to `{mode}` — {label}.",
        data={"weak_model_boost": mode},
    )


# ---- /stream -------------------------------------------------------------


async def _cmd_stream(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        state = "on" if rt.stream_replies else "off"
        return CommandResult(
            reply=f"Stream replies: `{state}`. Set with `/stream on|off`.",
            data={"stream_replies": rt.stream_replies},
        )
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change stream replies.")
    val = pc.args[0].strip().lower()
    if val in {"on", "true", "1", "yes"}:
        enabled = True
    elif val in {"off", "false", "0", "no"}:
        enabled = False
    else:
        return CommandResult(reply="Usage: `/stream on|off`.")
    rt.set_stream_replies(enabled)
    return CommandResult(
        reply=f"Stream replies set to `{'on' if enabled else 'off'}`.",
        data={"stream_replies": enabled},
    )


# ---- /length -------------------------------------------------------------


async def _cmd_length(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        label = LENGTH_LABELS.get(rt.response_length, rt.response_length)
        return CommandResult(
            reply=(
                f"Response length: `{rt.response_length}` — {label}. "
                "Set with `/length concise|balanced|detailed`."
            ),
            data={"response_length": rt.response_length},
        )
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change response length.")
    mode = pc.args[0].strip().lower()
    aliases = {"short": "concise", "long": "detailed", "normal": "balanced"}
    mode = aliases.get(mode, mode)
    if mode not in LENGTH_PREFS:
        return CommandResult(reply="Usage: `/length concise|balanced|detailed`.")
    rt.set_response_length(mode)
    label = LENGTH_LABELS.get(mode, mode)
    return CommandResult(
        reply=f"Response length set to `{mode}` — {label}.",
        data={"response_length": mode},
    )


# ---- /activity -----------------------------------------------------------


async def _cmd_activity(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        label = ACTIVITY_LABELS.get(rt.tool_activity, rt.tool_activity)
        return CommandResult(
            reply=(
                f"Tool activity: `{rt.tool_activity}` — {label}. "
                "Set with `/activity minimal|normal|verbose`."
            ),
            data={"tool_activity": rt.tool_activity},
        )
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change tool activity.")
    mode = pc.args[0].strip().lower()
    aliases = {"quiet": "minimal", "silent": "minimal", "chatty": "verbose"}
    mode = aliases.get(mode, mode)
    if mode not in ACTIVITY_MODES:
        return CommandResult(reply="Usage: `/activity minimal|normal|verbose`.")
    rt.set_tool_activity(mode)
    label = ACTIVITY_LABELS.get(mode, mode)
    return CommandResult(
        reply=f"Tool activity set to `{mode}` — {label}.",
        data={"tool_activity": mode},
    )


# ---- /think --------------------------------------------------------------

_THINK_LEVELS = {"off", "low", "medium", "high", "xhigh", "max", "ultra"}
# 'ultra' is GPT-5.6 Sol only; 'max' is available on GPT-5.6 Sol AND GLM-5.3
# cloud models.
_THINK_LEVELS_ULTRA_ONLY = {"ultra"}
_THINK_LEVELS_REQUIRES_MAX_SUPPORT = {"max"}


async def _cmd_think(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        return CommandResult(
            reply=(
                f"Thinking effort: `{rt.thinking_effort}`. "
                "Set with `/think off|low|medium|high|xhigh|max|ultra`. "
                "`max` requires GPT-5.6 Sol or GLM-5.3. `ultra` requires GPT-5.6 Sol."
            )
        )
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change thinking effort.")
    level = pc.args[0].strip().lower()
    aliases = {"none": "off", "med": "medium", "extra": "xhigh", "extra-high": "xhigh"}
    level = aliases.get(level, level)
    if level not in _THINK_LEVELS:
        return CommandResult(
            reply="Usage: `/think off|low|medium|high|xhigh|max|ultra`. "
            "`max` requires GPT-5.6 Sol or GLM-5.3. `ultra` requires GPT-5.6 Sol."
        )
    if level in _THINK_LEVELS_ULTRA_ONLY and not _is_gpt_56_sol(rt.default_model):
        return CommandResult(
            reply=f"`{level}` thinking is only available with GPT-5.6 Sol "
            f"(current model: `{rt.default_model}`)."
        )
    if level in _THINK_LEVELS_REQUIRES_MAX_SUPPORT and not _supports_max_reasoning(
        rt.default_model
    ):
        return CommandResult(
            reply=f"`{level}` thinking requires GPT-5.6 Sol or GLM-5.3/5.3-Flash "
            f"(current model: `{rt.default_model}`)."
        )
    rt.set_thinking_effort(level)
    return CommandResult(reply=f"Thinking effort set to `{level}`.")


# ---- /reasoning ----------------------------------------------------------


async def _cmd_reasoning(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not pc.args:
        state = "on" if rt.reasoning_output else "off"
        return CommandResult(reply=f"Reasoning output: `{state}`. Set with `/reasoning on|off`.")
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can change reasoning output.")
    val = pc.args[0].strip().lower()
    if val in {"on", "true", "1", "yes", "show"}:
        enabled = True
    elif val in {"off", "false", "0", "no", "hide"}:
        enabled = False
    else:
        return CommandResult(reply="Usage: `/reasoning on|off`.")
    rt.set_reasoning_output(enabled)
    return CommandResult(reply=f"Reasoning output set to `{'on' if enabled else 'off'}`.")


# ---- /new ----------------------------------------------------------------


async def _cmd_new(env: Any, rt: RuntimeHandle) -> CommandResult:
    raw = getattr(env, "raw", None) or {}
    channel_id = raw.get("channel_id") or env.message_id
    cleared = False
    try:
        cleared = bool(rt.reset_window_for(str(channel_id)))
    except Exception as e:  # noqa: BLE001
        log.warning("reset_window_for failed: %r", e)
    await rt.event_log.append(
        "cmd.new",
        {"channel_id": str(channel_id), "cleared": cleared, "user": env.sender.id},
    )
    msg = "Conversation context reset." if cleared else "Reset acknowledged (no window was active)."
    return CommandResult(reply=msg)


# ---- /dump ---------------------------------------------------------------


async def _cmd_dump(pc: ParsedCommand, env: Any, rt: RuntimeHandle) -> CommandResult:
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can dump context.")

    mode = pc.args[0].lower() if pc.args else "full"
    if mode not in {"full", "half"}:
        return CommandResult(reply="Usage: `/dump` or `/dump half`.")

    raw = getattr(env, "raw", None) or {}
    channel_id = str(raw.get("channel_id") or env.message_id or "")
    try:
        result = rt.dump_window_for(channel_id, mode == "half")
    except Exception as e:  # noqa: BLE001
        log.warning("dump_window_for failed: %r", e)
        return CommandResult(reply=f"Dump failed: {e}")

    await rt.event_log.append(
        "cmd.dump",
        {"channel_id": channel_id, "mode": mode, "result": result, "user": env.sender.id},
    )
    if not result:
        return CommandResult(reply="No active context window to dump.")

    return CommandResult(
        reply=(
            f"Dumped {result.get('frames', 0)} frames "
            f"({result.get('tokens_freed', 0)} est tokens) to sleep. "
            f"Mode: {mode}. Remaining body frames: {result.get('remaining_frames', 0)}."
        )
    )


# ---- /stop ---------------------------------------------------------------


def _cmd_stop(env: Any, rt: RuntimeHandle) -> CommandResult:
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can stop me.")

    raw = getattr(env, "raw", None) or {}
    chan_id = str(raw.get("channel_id") or "")
    cancelled = rt.cancel_stream(chan_id)
    if cancelled:
        return CommandResult(reply="Stopping.")

    return CommandResult(reply="Nothing active to stop.")


# ---- /restart ------------------------------------------------------------


def _cmd_restart(env: Any, rt: RuntimeHandle) -> CommandResult:
    if not _is_owner(env):
        return CommandResult(reply="Only the owner can restart me.")

    async def _do_restart() -> None:
        await asyncio.sleep(1.0)
        # Use --no-block so systemd doesn't hold us hostage. This runs as
        # the `norax` user session; --user is required.
        cmd = ["systemctl", "--user", "restart", "--no-block", "norax-ai.service"]
        log.warning("cmd.restart: %s", " ".join(cmd))
        try:
            # Detach so we survive until systemd kills us.
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            log.exception("systemctl restart failed; falling back to SIGTERM")
            os.kill(os.getpid(), signal.SIGTERM)

    return CommandResult(
        reply="Restarting. Back in a moment.",
        post_send=_do_restart,
    )
