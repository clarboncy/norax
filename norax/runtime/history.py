"""Bounded conversation-history selection and turn classification."""

from __future__ import annotations

import re
from typing import Any

from ..context.window import Frame


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
