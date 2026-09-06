"""Regression tests for model-aware history and checkpoint classification."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from norax.brain.agent_loop import _current_turn_anchor
from norax.context.window import Frame, RollingWindow
from norax.runtime.core import (
    _bounded_history_text,
    _candidate_prior_turn_ids,
    _checkpoint_status,
    _clean_historical_assistant_text,
    _history_budget_for_model,
    _history_turn_limit,
    _select_history_tail,
)
from norax.runtime.history import _coerce_tool_args_json
from norax.runtime.session import SessionMixin


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


@pytest.mark.parametrize("limit", [0, 1, 8, 64])
def test_tiny_history_cap_cannot_append_the_entire_original(limit) -> None:
    assert len(_bounded_history_text("x" * 20_000, limit)) <= limit


def test_oversized_latest_turn_preserves_task_without_overflow_or_mutation() -> None:
    huge_args = json.dumps({"path": "app.py", "content": "x" * 50_000})
    frames = [
        Frame(kind="user", content="USER_GOAL" + "x" * 20_000, turn_id=1),
        Frame(kind="tool_call", content=huge_args, turn_id=1, call_id="write-1"),
        Frame(kind="tool_result", content='{"ok": true}', turn_id=1, call_id="write-1"),
        Frame(kind="assistant", content="NEXT_STEP" + "y" * 20_000, turn_id=1),
    ]
    turns, calls, cap = _select_history_tail(frames, {1}, token_budget=1_024)
    assert turns == {1}
    assert calls == set()
    assert 2 * ((cap + 3) // 4 + 32) <= 1_024
    assert _bounded_history_text(frames[0].content, cap).startswith("USER_GOAL")
    assert _bounded_history_text(frames[-1].content, cap).startswith("NEXT_STEP")
    assert frames[1].content == huge_args


def test_large_tool_arguments_are_charged_at_emitted_size() -> None:
    frames = [
        Frame(kind="user", content="Implement the parser", turn_id=1),
        Frame(kind="tool_call", content=json.dumps({"code": "x" * 12_000}), turn_id=1, call_id="c"),
        Frame(kind="tool_result", content='{"ok": true}', turn_id=1, call_id="c"),
        Frame(kind="assistant", content="Verify the saved parser next", turn_id=1),
    ]
    turns, calls, _ = _select_history_tail(frames, {1}, token_budget=2_048)
    assert turns == {1}
    assert calls == set()


def test_historical_list_tool_arguments_become_a_json_object() -> None:
    assert json.loads(_coerce_tool_args_json([1, 2])) == {"_raw": [1, 2]}


def test_model_switch_history_does_not_replay_alternate_thinking_tags() -> None:
    assert _clean_historical_assistant_text("<think>private</think>\nSaved the file") == (
        "Saved the file"
    )


def test_fresh_task_keeps_sixteen_candidate_turns_before_token_projection() -> None:
    frames = [
        Frame(kind="user", content=f"old request {turn_id}", turn_id=turn_id)
        for turn_id in range(1, 21)
    ]

    assert _history_turn_limit("Please fix the current authentication failure") == 16
    assert _candidate_prior_turn_ids(frames, "Please fix the current authentication failure") == {
        *range(5, 21),
    }


def test_explicit_continuation_can_use_a_sixty_four_turn_tail() -> None:
    frames = [
        Frame(kind="user", content=f"request {turn_id}", turn_id=turn_id)
        for turn_id in range(1, 81)
    ]

    assert _history_turn_limit("continue") == 64
    assert _candidate_prior_turn_ids(frames, "continue") == set(range(17, 81))


def test_model_switch_does_not_shrink_the_canonical_window(tmp_path) -> None:
    channel = "switch-test"
    path = tmp_path / "state" / "windows" / f"{channel}.json"
    persisted = RollingWindow(budget_tokens=65_536)
    persisted.add_user("retain this exact context")
    persisted.save(path)

    session = SessionMixin()
    session.cfg = SimpleNamespace(memory_root=tmp_path)
    session._windows = {}
    session._effective_model = "qwen3.8-27b-fast:latest"

    local_window = session._get_window(channel)
    assert local_window.budget_tokens == 256_000
    assert [frame.content for frame in local_window.body] == ["retain this exact context"]

    session._effective_model = "glm-5.3:cloud"
    cloud_window = session._get_window(channel)
    assert cloud_window is local_window
    assert cloud_window.budget_tokens == 256_000
    assert [frame.content for frame in cloud_window.body] == ["retain this exact context"]


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
