"""Operator-maintained request defaults for selected Ollama model tags.

Profiles configure request shape only. They do not advertise a model, prove it
is installed, establish a benchmark result, or guarantee that an upstream
accepts a context/reasoning option. Live availability comes from Ollama's model
catalog and unsupported options remain observable upstream errors.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Any


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


# Known cloud profiles may opt into a larger configured request window. Unknown
# model tags use the conservative local default to avoid surprise VRAM/RAM use.
OLLAMA_DEFAULT_CTX = _bounded_env_int("NORAX_OLLAMA_CTX", 262144, 2048, 2_000_000)
OLLAMA_LOCAL_CTX = _bounded_env_int("NORAX_LOCAL_OLLAMA_CTX", 32768, 2048, 1_000_000)

# ── Model profiles ───────────────────────────────────────────────────


@dataclass(frozen=True)
class OllamaProfile:
    """Tuned inference + agentic settings for one Ollama model tag."""

    model: str
    temperature: float = 0.3
    top_p: float = 0.95
    num_predict: int = 8192
    num_ctx: int | None = OLLAMA_DEFAULT_CTX
    num_batch: int | None = None  # per-model batch size (VRAM-tuned for local drivers)
    keep_alive: str | int | None = None  # e.g. -1 = keep resident (no cold-reload latency)
    think: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    clear_thinking: bool | None = None
    tools_enabled: bool = True
    role: str = "general"  # agentic | planner | executor | verifier | fast | weak
    fallback: str | None = None
    tool_system_hint: str | None = None
    tool_temperature: float | None = None  # lower temp when tools present (weak models)
    continuation_attempts: int = 2  # auto-continue when done_reason == "length"
    window_tokens: int | None = None  # per-model rolling-window budget (align to max ctx)
    continue_prompt: str | None = None  # per-role nudge to finish, not ramble


OLLAMA_PROFILES: dict[str, OllamaProfile] = {
    # Explicit operator profiles; upstream capability is still discovered live.
    "kimi-k3:cloud": OllamaProfile(
        model="kimi-k3:cloud",
        temperature=1.0,
        top_p=0.95,
        think=True,
        reasoning_efforts=("low", "medium", "high", "max"),
        num_predict=8192,
        role="executor",
        tool_system_hint="You are the coding/tool specialist. Call tools directly for repo/debug/exec/research tasks.",
        continue_prompt="Continue from where you left off. Finish the current file or thought. Do not restart.",
    ),
    "kimi-k2.7-code:cloud": OllamaProfile(
        model="kimi-k2.7-code:cloud",
        temperature=1.0,
        top_p=0.95,
        think=True,
        num_predict=8192,
        role="executor",
        fallback="kimi-k2.6:cloud",
        tool_system_hint="You are the coding/tool specialist. Call tools directly for repo/debug/exec/research tasks.",
        continue_prompt="Continue from where you left off. Finish the current file or thought. Do not restart.",
    ),
    # Kimi fallback profiles.
    "kimi-k2.6:cloud": OllamaProfile(
        model="kimi-k2.6:cloud",
        temperature=1.0,
        top_p=0.95,
        think=True,
        num_predict=8192,
        role="agentic",
        fallback="kimi-k2.5:cloud",
        tool_system_hint="You have tools. Use them for repo/debug/exec/research tasks.",
    ),
    "kimi-k2.5:cloud": OllamaProfile(
        model="kimi-k2.5:cloud",
        temperature=1.0,
        top_p=0.95,
        think=True,
        role="agentic",
    ),
    # GLM profiles.
    "glm-5.3:cloud": OllamaProfile(
        model="glm-5.3:cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        reasoning_efforts=("low", "medium", "high", "max"),
        clear_thinking=True,
        role="executor",
        fallback="glm-5.3-flash:cloud",
        continue_prompt="Continue from where you left off. Complete the current task without restarting.",
    ),
    "glm-5.3-flash:cloud": OllamaProfile(
        model="glm-5.3-flash:cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        reasoning_efforts=("low", "medium", "high", "max"),
        role="agentic",
        fallback="glm-5.2:cloud",
        continue_prompt="Continue. Wrap up your current thought concisely.",
    ),
    "glm-5.2:cloud": OllamaProfile(
        model="glm-5.2:cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        role="agentic",
        fallback="glm-5.1:cloud",
        continue_prompt="Continue. Wrap up your current thought concisely.",
    ),
    "glm-5.1:cloud": OllamaProfile(
        model="glm-5.1:cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        role="agentic",
        fallback="glm-5:cloud",
    ),
    "glm-5:cloud": OllamaProfile(
        model="glm-5:cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        role="agentic",
    ),
    # Qwen coder profile.
    "qwen3-coder-next:cloud": OllamaProfile(
        model="qwen3-coder-next:cloud",
        temperature=0.2,
        top_p=0.9,
        think=False,
        num_predict=8192,
        role="executor",
        tool_system_hint="Call tools directly. Do not describe — execute.",
        continue_prompt="Continue. Complete the current code block or file.",
    ),
    # DeepSeek profiles.
    "deepseek-v4-pro:cloud": OllamaProfile(
        model="deepseek-v4-pro:cloud",
        temperature=1.0,
        top_p=1.0,
        think=True,
        num_predict=16384,
        role="planner",
        fallback="deepseek-v4-flash:cloud",
        continue_prompt="Continue your plan. Finish the current step, then stop.",
    ),
    "deepseek-v4-flash:cloud": OllamaProfile(
        model="deepseek-v4-flash:cloud",
        temperature=0.3,
        top_p=0.95,
        think=False,
        num_predict=8192,
        role="fast",
        tool_system_hint="Call tools directly. Do not describe — execute.",
    ),
    "deepseek-v3.1:671b-cloud": OllamaProfile(
        model="deepseek-v3.1:671b-cloud",
        temperature=1.0,
        top_p=1.0,
        think=True,
        role="planner",
    ),
    # General cloud and explicitly tuned local profiles.
    "qwen3.8-27b-fast:latest": OllamaProfile(
        model="qwen3.8-27b-fast:latest",
        temperature=1.0,
        tool_temperature=1.0,
        top_p=0.95,
        think=True,
        reasoning_efforts=("low", "medium", "high"),
        num_predict=8192,
        num_ctx=65536,
        window_tokens=65536,
        num_batch=1024,
        keep_alive=-1,  # keep the 27B resident — avoids 14.5GB cold-reload latency
        role="agentic",
        fallback="glm-5.3:cloud",
        tool_system_hint=(
            "Call tools directly. Do not describe — execute. Read before acting. "
            "Emit valid tool_calls JSON only."
        ),
    ),
    "qwen3.8:27b": OllamaProfile(
        model="qwen3.8:27b",
        temperature=0.3,
        top_p=0.95,
        think=True,
        num_predict=12288,
        num_ctx=262144,
        role="agentic",
        fallback="qwen3.5:cloud",
        tool_system_hint="Call tools directly. Do not describe — execute. Read before acting.",
    ),
    "hf.co/jpetrina/Qwen3.8-27B-IQ4_XS-pure-GGUF:latest": OllamaProfile(
        model="hf.co/jpetrina/Qwen3.8-27B-IQ4_XS-pure-GGUF:latest",
        temperature=0.3,
        top_p=0.95,
        think=True,
        num_predict=12288,
        num_ctx=262144,
        role="agentic",
        fallback="qwen3.8:27b",
        tool_system_hint="Call tools directly. Do not describe — execute. Read before acting.",
    ),
    "qwen3.5:cloud": OllamaProfile(
        model="qwen3.5:cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        role="agentic",
    ),
    "qwen3.5:397b-cloud": OllamaProfile(
        model="qwen3.5:397b-cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        role="agentic",
    ),
    "nemotron-3-super:cloud": OllamaProfile(
        model="nemotron-3-super:cloud",
        temperature=0.2,
        top_p=0.9,
        think=False,
        role="agentic",
    ),
    "minimax-m3:cloud": OllamaProfile(
        model="minimax-m3:cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        num_predict=8192,
        role="agentic",
        fallback="minimax-m2.7:cloud",
        tool_system_hint="Call tools directly. Do not describe — execute.",
        continue_prompt="Continue. Finish the current task, then stop.",
    ),
    "minimax-m2.7:cloud": OllamaProfile(
        model="minimax-m2.7:cloud",
        temperature=0.3,
        top_p=0.95,
        think=True,
        role="agentic",
    ),
    "gemma4:31b-cloud": OllamaProfile(
        model="gemma4:31b-cloud",
        temperature=0.0,
        top_p=1.0,
        think=False,
        num_predict=256,
        tools_enabled=False,
        num_ctx=8192,
        role="verifier",
    ),
    "fastcontext-1.0-4b-sft:latest": OllamaProfile(
        model="fastcontext-1.0-4b-sft:latest",
        temperature=0.2,
        top_p=0.9,
        think=False,
        num_predict=4096,
        tools_enabled=False,
        # The helper receives a bounded grep/path evidence block, not an entire
        # repository. A 262k KV cache only adds latency/VRAM pressure here.
        num_ctx=16384,
        role="retriever",
        continuation_attempts=0,
        tool_system_hint="You are FastContext, a repository exploration subagent. Read files, run glob, and grep. Return file:line citations.",
    ),
    "fastcontext-1.0-4b-rl-q4_k_m:latest": OllamaProfile(
        model="fastcontext-1.0-4b-rl-q4_k_m:latest",
        temperature=0.2,
        top_p=0.9,
        think=False,
        num_predict=4096,
        tools_enabled=False,
        num_ctx=16384,
        role="retriever",
        continuation_attempts=0,
        tool_system_hint="You are FastContext, a repository exploration subagent. Read files, run glob, and grep. Return file:line citations.",
    ),
    # Weak local models — ultra-low tool temperature
    "deepseek-r1:14b": OllamaProfile(
        model="deepseek-r1:14b",
        temperature=0.15,
        tool_temperature=0.05,
        top_p=0.9,
        think=True,
        num_ctx=OLLAMA_LOCAL_CTX,
        role="weak",
        tool_system_hint="Emit valid tool_calls JSON only. No prose before tools.",
    ),
    # Base Gemma 4 12B — general-purpose local model, no agentic tuning.
    # Good for chat/light coding; tool calls work but less reliable than agentic variant.
    "gemma4:12b": OllamaProfile(
        model="gemma4:12b",
        temperature=0.7,
        tool_temperature=0.2,
        top_p=0.95,
        think=False,
        num_predict=4096,
        num_ctx=OLLAMA_LOCAL_CTX,
        role="general",
        tool_system_hint=(
            "Call tools directly. Do not describe — execute. Emit valid tool_calls JSON only."
        ),
    ),
    # Local Gemma 4 12B agentic (Fable5 + Composer2.5 v2, Q6_K)
    # Fine-tuned for coding/terminal/tool-use. Author recommends temp 1.0,
    # rep_pen 1.1. We use lower tool_temperature for reliable tool JSON.
    "norax-gemma4-12b-agentic:latest": OllamaProfile(
        model="norax-gemma4-12b-agentic:latest",
        temperature=1.0,
        tool_temperature=0.3,
        top_p=0.95,
        think=True,
        num_predict=8192,
        num_ctx=OLLAMA_LOCAL_CTX,
        role="agentic",
        fallback="glm-5.2:cloud",
        tool_system_hint=(
            "Call tools directly. Do not describe — execute. Read before acting. "
            "When multiple independent tools are needed, emit ALL tool_calls in this turn. "
            "Do not stop after one — list every tool call that is needed."
        ),
    ),
    # Q6_K quant of the same agentic fine-tune — same weights, different tag
    "xentriom/gemma-4-12B-agentic-fable5-composer2.5-v2:Q6_K": OllamaProfile(
        model="xentriom/gemma-4-12B-agentic-fable5-composer2.5-v2:Q6_K",
        temperature=1.0,
        tool_temperature=0.3,
        top_p=0.95,
        think=True,
        num_predict=8192,
        num_ctx=OLLAMA_LOCAL_CTX,
        role="agentic",
        fallback="glm-5.2:cloud",
        tool_system_hint=(
            "Call tools directly. Do not describe — execute. Read before acting. "
            "When multiple independent tools are needed, emit ALL tool_calls in this turn. "
            "Do not stop after one — list every tool call that is needed."
        ),
    ),
    # FreeToken-served local GGUF models (Ornith / Qwen-AgentWorld MoE)
    "Ornith-1.5-35B-IQ3_S.gguf": OllamaProfile(
        model="Ornith-1.5-35B-IQ3_S.gguf",
        temperature=0.6,
        top_p=0.95,
        think=False,
        num_predict=8192,
        num_ctx=32768,
        role="agentic",
        tool_system_hint=(
            "Call tools directly. Do not describe — execute. Read before acting. "
            "When multiple independent tools are needed, emit ALL tool_calls in this turn. "
            "Do not stop after one — list every tool call that is needed."
        ),
    ),
    "Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf": OllamaProfile(
        model="Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf",
        temperature=0.6,
        top_p=0.95,
        think=False,
        num_predict=8192,
        num_ctx=32768,
        role="agentic",
        tool_system_hint=(
            "Call tools directly. Do not describe — execute. Read before acting. "
            "When multiple independent tools are needed, emit ALL tool_calls in this turn. "
            "Do not stop after one — list every tool call that is needed."
        ),
    ),
}

# Prefix fallbacks for unknown tags
_PREFIX_DEFAULTS: list[tuple[str, str]] = [
    ("fastcontext-1.0-4b-rl", "fastcontext-1.0-4b-rl-q4_k_m:latest"),
    ("fastcontext", "fastcontext-1.0-4b-rl-q4_k_m:latest"),
    ("kimi-k2.7-code", "kimi-k2.7-code:cloud"),
    ("kimi-", "kimi-k2.7-code:cloud"),
    ("glm-5.3-flash", "glm-5.3-flash:cloud"),
    ("glm-5.3", "glm-5.3:cloud"),
    ("glm-", "glm-5.2:cloud"),
    ("qwen3-coder", "qwen3-coder-next:cloud"),
    ("deepseek-v4-pro", "deepseek-v4-pro:cloud"),
    ("deepseek-v4-flash", "deepseek-v4-flash:cloud"),
    ("minimax-m3", "minimax-m3:cloud"),
    ("minimax-m2", "minimax-m2.7:cloud"),
    ("qwen3.8-xs", "hf.co/jpetrina/Qwen3.8-27B-IQ4_XS-pure-GGUF:latest"),
    ("qwen3.8-27b-fast", "qwen3.8-27b-fast:latest"),
    ("qwen3.8", "qwen3.8:27b"),
    ("qwen3.5", "qwen3.5:cloud"),
    ("deepseek-r1", "deepseek-r1:14b"),
    ("norax-gemma4", "norax-gemma4-12b-agentic:latest"),
    ("gemma4:12b", "gemma4:12b"),
    ("gemma4-12b", "gemma4:12b"),
]

_DEFAULT_PROFILE = OllamaProfile(
    model="default",
    temperature=0.3,
    top_p=0.95,
    think=False,
    num_predict=4096,
    num_ctx=OLLAMA_LOCAL_CTX,
    role="general",
    tool_temperature=0.15,
)

_MATH_PROMPT = re.compile(
    r"\b(calculate|compute|what is|evaluate|solve|arithmetic|sum of|product of)\b",
    re.I,
)
_HAS_ARITHMETIC = re.compile(r"\d+\s*[\*x×/+\-]\s*\d+")
# Trivial arithmetic that a competent model can answer without exec: a single
# operation on two single-digit integers (e.g. "2+2", "7-3", "3*4"). Forcing
# exec for these wastes a round, risks tool-call loops, and can duplicate the
# answer. Only flag expressions that involve larger operands, decimals,
# division, or multiple operations where mental arithmetic is error-prone.
_TRIVIAL_ARITHMETIC = re.compile(r"(?<![\w.])\d\s*[+\-*×]\s*\d(?![\w.])")
_HAS_DIVISION = re.compile(r"\d+\s*[/÷]\s*\d+")
_HAS_DECIMAL = re.compile(r"\d+\.\d+")
_HAS_MULTIPLE_OPS = re.compile(r"\d+\s*[\*x×/+\-]\s*\d+\s*[\*x×/+\-]\s*\d+")


def normalize_model_name(model: str) -> str:
    """Strip provider prefix (ollama/kimi-.../freetoken) from routed model ids."""
    m = (model or "").strip()
    for prefix in ("ollama/", "ollama-direct/", "freetoken/"):
        if m.startswith(prefix):
            m = m[len(prefix) :]
    return m


def resolve_profile(model: str) -> OllamaProfile:
    """Return tuned profile for a model tag; falls back to prefix heuristics."""
    bare = normalize_model_name(model)
    if bare in OLLAMA_PROFILES:
        return OLLAMA_PROFILES[bare]
    lower = bare.lower()
    for prefix, key in _PREFIX_DEFAULTS:
        if lower.startswith(prefix):
            return OLLAMA_PROFILES[key]
    return replace(_DEFAULT_PROFILE, model=bare)


def is_ollama_agentic_model(model: str) -> bool:
    p = resolve_profile(model)
    return p.role in {"agentic", "executor", "planner", "weak"}


def is_ollama_retriever_model(model: str) -> bool:
    p = resolve_profile(model)
    return p.role == "retriever"


def is_weak_ollama_model(model: str) -> bool:
    """True for small/local models that benefit from low tool temperature."""
    p = resolve_profile(model)
    if p.role == "weak":
        return True
    bare = normalize_model_name(model).lower()
    if ":cloud" in bare:
        return False
    size_match = re.search(r":(\d+(?:\.\d+)?)b(?:[-_:]|$)", bare)
    if size_match:
        return float(size_match.group(1)) <= 32.0
    return ":latest" in bare and bare.startswith(("norax-gemma", "fastcontext"))


def effective_temperature(profile: OllamaProfile, *, has_tools: bool) -> float:
    """Use tool_temperature for tool-calling turns on weak models."""
    if has_tools and profile.tool_temperature is not None:
        return profile.tool_temperature
    return profile.temperature


def reasoning_effort_for_profile(profile: OllamaProfile, effort: Any) -> str | None:
    """Map Norax effort names to the levels accepted by an Ollama model."""
    if not effort or not profile.reasoning_efforts:
        return None
    normalized = str(effort).strip().lower()
    if normalized in {"none", "minimal", "off"}:
        return None
    # If the model supports this level natively, use it directly.
    if normalized in profile.reasoning_efforts:
        return normalized
    # Fallback mapping for models that use a different effort vocabulary.
    candidates = {
        "medium": ("medium", "high"),
        "high": ("high", "xhigh", "max"),
        "xhigh": ("xhigh", "max", "high"),
        "max": ("max", "xhigh", "high"),
    }.get(normalized, ())
    return next((level for level in candidates if level in profile.reasoning_efforts), None)


def needs_exec_for_math(text: str) -> bool:
    t = text or ""
    has_arith = bool(_HAS_ARITHMETIC.search(t))
    if not has_arith and not _MATH_PROMPT.search(t):
        return False
    # Non-trivial signals: division, decimals, or multiple operations in the
    # full text. These are error-prone for mental arithmetic.
    has_division = bool(_HAS_DIVISION.search(t))
    has_decimal = bool(_HAS_DECIMAL.search(t))
    has_multiple_ops = bool(_HAS_MULTIPLE_OPS.search(t))
    if has_division or has_decimal or has_multiple_ops:
        return True
    # Explicit math verb (calculate/compute/solve/...) with arithmetic → exec.
    if _MATH_PROMPT.search(t) and has_arith:
        # But skip trivial single-op small-integer expressions like "2+2"
        # even when phrased as "what is 2+2" — a competent model knows these.
        arith_matches = _HAS_ARITHMETIC.findall(t)
        if arith_matches and all(_TRIVIAL_ARITHMETIC.fullmatch(m.strip()) for m in arith_matches):
            return False
        return True
    # Bare arithmetic (no math verb): only flag non-trivial expressions.
    # Trivial single-op small integers like "2+2" do not need exec.
    if has_arith:
        arith_matches = _HAS_ARITHMETIC.findall(t)
        if arith_matches and all(_TRIVIAL_ARITHMETIC.fullmatch(m.strip()) for m in arith_matches):
            return False
        return True
    return False


def build_exec_guard_nudge(user_prompt: str) -> str | None:
    if not needs_exec_for_math(user_prompt):
        return None
    return "\n".join(
        [
            "OLLAMA_EXEC_GUARD;weight=W5",
            "ARITHMETIC: Run python3 -c 'print(<expr>)' via exec/shell before answering.",
            "Base the non-trivial exact result on the command output.",
        ]
    )


def tools_for_profile(
    tools: list[dict] | None,
    *,
    enabled: bool,
) -> list[dict] | None:
    """Return the complete manifest when tools are enabled for a profile."""
    if not tools:
        return tools
    if not enabled:
        return None
    return tools


def trim_tools(
    tools: list[dict] | None,
    max_tools: int,
    *,
    model: str = "",
) -> list[dict] | None:
    """Compatibility wrapper for the retired numeric tool setting.

    A first-N slice is order-dependent and can silently remove the one tool a
    task needs. Until the runtime has a model-visible dynamic tool loader,
    every tool-capable profile receives the complete manifest. Weak-model
    prompt cost is reduced losslessly by compacting schemas upstream. Zero or
    a negative value disables tools and a positive value enables the complete
    manifest; the value is not a limit. New code should call
    :func:`tools_for_profile` with an explicit boolean.
    """
    del model
    return tools_for_profile(tools, enabled=max_tools > 0)


def profile_options(
    profile: OllamaProfile,
    *,
    override_temp: float | None = None,
    has_tools: bool = False,
) -> dict[str, Any]:
    temp = (
        override_temp
        if override_temp is not None
        else effective_temperature(profile, has_tools=has_tools)
    )
    opts: dict[str, Any] = {
        "temperature": temp,
        "top_p": profile.top_p,
        "num_predict": profile.num_predict,
    }
    if profile.num_ctx:
        opts["num_ctx"] = profile.num_ctx
    if getattr(profile, "num_batch", None):
        # Smaller batch shrinks CUDA graph buffers — critical for large-ctx
        # local drivers where VRAM headroom decides GPU vs CPU split.
        opts["num_batch"] = profile.num_batch
    return opts


def apply_profile_to_gateway_request(req: Any) -> Any:
    """Apply per-model inference defaults and explicit tool availability."""
    from . import GatewayRequest

    if not isinstance(req, GatewayRequest):
        return req
    profile = resolve_profile(req.model)
    has_tools = bool(req.tools)
    temp = (
        req.temperature
        if req.temperature is not None
        else effective_temperature(profile, has_tools=has_tools)
    )
    max_tok = req.max_tokens if req.max_tokens is not None else profile.num_predict
    tools = tools_for_profile(req.tools, enabled=profile.tools_enabled)
    meta = dict(req.metadata or {})
    meta.setdefault("ollama_profile", profile.role)
    meta.setdefault("ollama_think", profile.think)
    if profile.tool_system_hint and req.tools:
        meta.setdefault("ollama_tool_hint", profile.tool_system_hint)
    return replace(req, temperature=temp, max_tokens=max_tok, tools=tools, metadata=meta)


def build_ollama_chat_body(
    payload: dict[str, Any],
    model: str,
    *,
    stream: bool = False,
) -> dict[str, Any]:
    """Build native Ollama /api/chat body with per-model best practices."""
    profile = resolve_profile(model)
    messages = payload.get("messages") or []

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": profile.think,
    }
    if profile.clear_thinking is not None:
        body["clear_thinking"] = profile.clear_thinking

    has_tools = bool(payload.get("tools"))
    options = profile_options(profile, has_tools=has_tools)
    if payload.get("temperature") is not None:
        options["temperature"] = payload["temperature"]
    if payload.get("max_tokens") is not None:
        options["num_predict"] = int(payload["max_tokens"])
    if isinstance(payload.get("options"), dict):
        options.update(payload["options"])
    body["options"] = options

    effort = payload.get("reasoning_effort") or payload.get("reasoning")
    model_effort = reasoning_effort_for_profile(profile, effort)
    if model_effort:
        body["reasoning_effort"] = model_effort
    if payload.get("think") is not None:
        body["think"] = bool(payload["think"])
    if effort in ("none", "minimal", "off"):
        body["think"] = False
    elif effort in ("high", "xhigh", "max", "medium"):
        body["think"] = True

    tools = payload.get("tools")
    if tools:
        body["tools"] = tools_for_profile(tools, enabled=profile.tools_enabled)
    return body
