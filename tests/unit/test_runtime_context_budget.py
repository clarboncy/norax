"""Regression tests for model-aware history and checkpoint classification."""

from __future__ import annotations

from types import SimpleNamespace

from norax.brain.agent_loop import _current_turn_anchor
from norax.context.window import Frame
from norax.runtime.core import (
    _bounded_history_text,
    _candidate_prior_turn_ids,
    _checkpoint_status,
    _clean_historical_assistant_text,
    _history_budget_for_model,
    _history_turn_limit,
    _select_history_tail,
)


def test_history_budget_uses_context_capacity_and_action_reserve() -> None:
    local_action = _history_budget_for_model(
        "freetoken/model.gguf", provider_kind="openai", action_request=True
    )
    local_chat = _history_budget_for_model(
        "freetoken/model.gguf", provider_kind="openai", action_request=False
    )
    cloud_chat = _history_budget_for_model(
        "test-cloud-model", provider_kind="openai", action_request=False
    )

    assert local_action == 6_000
    assert local_chat == 8_000
    assert cloud_chat == 16_000


def test_short_turns_are_not_discarded_by_a_fixed_turn_cap() -> None:
    frames: list[Frame] = []
    for turn_id in range(1, 9):
        frames.append(Frame(kind="user", content=f"question {turn_id}", turn_id=turn_id))
        frames.append(Frame(kind="assistant", content=f"answer {turn_id}", turn_id=turn_id))

    turns, calls, _cap = _select_history_tail(
        frames,
        set(range(1, 9)),
        token_budget=2_048,
    )

    assert turns == set(range(1, 9))
    assert calls == set()


def test_tool_heavy_history_keeps_recent_atomic_pairs() -> None:
    frames = [Frame(kind="user", content="continue", turn_id=1)]
    for index in range(5):
        call_id = f"call-{index}"
        frames.append(Frame(kind="tool_call", content='{"path":"x"}', turn_id=1, call_id=call_id))
        frames.append(Frame(kind="tool_result", content="x" * 4_000, turn_id=1, call_id=call_id))
    frames.append(Frame(kind="assistant", content="done", turn_id=1))

    turns, calls, _cap = _select_history_tail(frames, {1}, token_budget=3_200)

    assert turns == {1}
    assert calls == {"call-3", "call-4"}


def test_oversized_history_text_preserves_head_and_tail() -> None:
    original = "HEAD" + ("x" * 20_000) + "TAIL"
    bounded = _bounded_history_text(original, 1_000)

    assert len(bounded) == 1_000
    assert bounded.startswith("HEAD")
    assert bounded.endswith("TAIL")
    assert "historical message truncated" in bounded


def test_fresh_task_limits_history_to_three_recent_turns() -> None:
    frames = [
        Frame(kind="user", content=f"old request {turn_id}", turn_id=turn_id)
        for turn_id in range(1, 9)
    ]

    assert _history_turn_limit("Please fix the current authentication failure") == 3
    assert _candidate_prior_turn_ids(frames, "Please fix the current authentication failure") == {
        6,
        7,
        8,
    }


def test_explicit_continuation_can_use_a_larger_recent_tail() -> None:
    frames = [
        Frame(kind="user", content=f"request {turn_id}", turn_id=turn_id)
        for turn_id in range(1, 11)
    ]

    assert _history_turn_limit("continue") == 8
    assert _candidate_prior_turn_ids(frames, "continue") == set(range(3, 11))


def test_historical_assistant_scaffolding_is_not_replayed() -> None:
    content = (
        "<thinking>private stale reasoning</thinking>\n"
        "GOAL: old request\nPLAN: old plan\nRISK: none\n\nCurrent answer"
    )

    assert _clean_historical_assistant_text(content) == "Current answer"


def test_current_turn_anchor_makes_new_request_authoritative() -> None:
    anchor = _current_turn_anchor(
        "Fix the current authentication failure",
        active_goal="Fix the current authentication failure",
    )

    assert "supersedes conflicting older history" in anchor
    assert "Do not resume an older task" in anchor
    assert "Fix the current authentication failure" in anchor


def test_checkpoint_status_uses_terminal_verification_not_tool_kind() -> None:
    read_only_state = SimpleNamespace(tool_calls=2, writes=0)
    verified = SimpleNamespace(content="audit complete", raw={"verified_outcome": True})
    unverified = SimpleNamespace(content="changed it", raw={"verified_outcome": False})
    orchestrated = SimpleNamespace(
        content="completed synthesis",
        raw={"orchestrator": True, "verified_outcome": True},
    )
    prose_only = SimpleNamespace(content="looks complete", raw={"orchestrator": True})

    assert _checkpoint_status(read_only_state, verified) == "complete"
    assert _checkpoint_status(read_only_state, unverified) == "in_progress"
    assert _checkpoint_status(None, orchestrated) == "complete"
    assert _checkpoint_status(None, prose_only) == "in_progress"


def test_checkpoint_status_rejects_truthy_non_boolean_terminal_flags() -> None:
    malformed = SimpleNamespace(raw={"accepted_outcome": "true", "verified_outcome": "true"})
    interrupted = SimpleNamespace(raw={"stopped": "false", "verified_outcome": True})

    assert _checkpoint_status(None, malformed) == "in_progress"
    assert _checkpoint_status(None, interrupted) == "complete"
