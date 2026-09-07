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

    def encode(value: object) -> str:
        try:
            return _json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (OverflowError, RecursionError, TypeError, ValueError):
            return "{}"

    if content is None:
        return "{}"
    if isinstance(content, (dict, list)):
        return encode(content if isinstance(content, dict) else {"_raw": content})
    try:
        s = str(content).strip()
    except Exception:  # noqa: BLE001
        return "{}"
    if not s:
        return "{}"
    try:
        parsed = _json.loads(s)
    except Exception:  # noqa: BLE001
        # Non-JSON text args -> wrap so it stays a valid JSON object
        return encode({"_raw": s})
    if isinstance(parsed, dict):
        # Re-encoding rejects Python's non-standard NaN/Infinity extensions
        # and compacts arguments before they consume provider context.
        return encode(parsed)
    # Valid JSON but not an object (e.g. null, [], "str") -> wrap
    return encode({"_raw": parsed})


def _bounded_history_text(content: str, max_chars: int) -> str:
    """Keep both ends of an oversized historical message within budget."""
    text = str(content or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = f"\n...[historical message truncated; {len(text)} chars total]...\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    available = max_chars - len(marker)
    head = available // 2
    tail = available - head
    return text[:head] + marker + text[-tail:]


def _clean_historical_assistant_text(content: str) -> str:
    """Remove private reasoning and leaked harness headers from replayed replies."""
    from ..gateway_client import strip_reasoning_blocks

    text = strip_reasoning_blocks(str(content or ""))
    lines = text.lstrip().splitlines()
    first_content = 0
    while first_content < len(lines) and (
        not lines[first_content].strip()
        or re.match(r"^(?:GOAL|PLAN|RISK):", lines[first_content].strip())
    ):
        first_content += 1
    return "\n".join(lines[first_content:]).strip()


def _history_turn_limit(user_prompt: str) -> int:
    """Return the candidate-turn ceiling before token-budget projection.

    The final history is still strictly bounded by ``_select_history_tail``.
    These wider ceilings preserve continuity across model switches and long
    projects without sending an unbounded prompt to any provider.
    """
    text = " ".join(str(user_prompt or "").lower().split())
    words = re.findall(r"[a-z0-9_+-]+", text)
    if len(words) <= 8 and re.match(
        r"^(?:continue|resume|keep going|carry on|go ahead|finish(?: it)?|pick (?:it )?up|"
        r"do it|fix that|same thing|what about|and (?:then|now)|yes|ok|okay)\b",
        text,
    ):
        return 64
    return 16


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
    if token_budget <= 0:
        return set(), set(), 0
    by_turn: dict[int, list[Frame]] = {}
    for frame in frames:
        if frame.turn_id in candidate_turn_ids:
            by_turn.setdefault(frame.turn_id, []).append(frame)
    ordered_turns = list(by_turn)
    text_cap_tokens = max(512, min(2_048, token_budget // 4))
    max_tool_pairs = max(1, min(12, token_budget // 1_600))

    valid_calls_by_turn: dict[int, list[str]] = {}
    for turn_id in ordered_turns:
        calls = [
            frame.call_id
            for frame in by_turn[turn_id]
            if frame.kind == "tool_call" and frame.call_id
        ]
        results = {
            frame.call_id
            for frame in by_turn[turn_id]
            if frame.kind == "tool_result" and frame.call_id
        }
        valid = [call_id for call_id in calls if call_id in results]
        valid_calls_by_turn[turn_id] = list(dict.fromkeys(valid))[-max_tool_pairs:]

    def turn_cost(turn_id: int) -> int:
        cost = 0
        valid_calls = set(valid_calls_by_turn[turn_id])
        for frame in by_turn[turn_id]:
            if frame.kind in {"tool_call", "tool_result"}:
                if frame.call_id not in valid_calls:
                    continue
                if frame.kind == "tool_call":
                    # Arguments are emitted as valid JSON, never truncated.
                    # Charging only 300 tokens here let historical writes send
                    # many thousands of unbudgeted tokens to a smaller model.
                    content = _coerce_tool_args_json(frame.content)
                    content += str(frame.meta.get("name", "unknown"))
                else:
                    content = _bounded_history_text(frame.content, 2_000)
                cost += (len(content) + len(frame.call_id or "") + 3) // 4 + 32
            else:
                cost += min((len(frame.content) + 3) // 4, text_cap_tokens) + 32
        return cost

    if ordered_turns:
        latest = ordered_turns[-1]
        # Preserve the latest user request and answer when its tool trace is
        # too large. Previously the first over-budget turn discarded *all*
        # history, including the task that a "continue" refers to.
        while valid_calls_by_turn[latest] and turn_cost(latest) > token_budget:
            valid_calls_by_turn[latest].pop(0)
        if turn_cost(latest) > token_budget:
            # Once all removable tool pairs are gone, any remaining cost comes
            # from at least one user/assistant text frame.
            text_frames = sum(
                frame.kind not in {"tool_call", "tool_result"} for frame in by_turn[latest]
            )
            text_cap_tokens = max(1, token_budget // text_frames - 32)

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
