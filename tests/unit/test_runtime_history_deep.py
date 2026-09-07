"""Branch-complete protocol and budget contracts for conversation history."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from norax.context.window import Frame
from norax.runtime.history import (
    _bounded_history_text,
    _candidate_prior_turn_ids,
    _checkpoint_status,
    _clean_historical_assistant_text,
    _coerce_tool_args_json,
    _history_budget_for_model,
    _history_turn_limit,
    _is_coding_query,
    _select_history_tail,
)


def test_tool_arguments_are_always_compact_standard_json_objects() -> None:
    assert _coerce_tool_args_json(None) == "{}"
    assert _coerce_tool_args_json({"name": "東", "count": 1}) == '{"name":"東","count":1}'
    assert _coerce_tool_args_json([1, 2]) == '{"_raw":[1,2]}'
    assert _coerce_tool_args_json("  ") == "{}"
    assert json.loads(_coerce_tool_args_json("plain text")) == {"_raw": "plain text"}
    assert _coerce_tool_args_json(' { "value": 2 } ') == '{"value":2}'
    assert json.loads(_coerce_tool_args_json("null")) == {"_raw": None}
    assert json.loads(_coerce_tool_args_json('["one"]')) == {"_raw": ["one"]}


def test_tool_arguments_fail_closed_on_cycles_nonfinite_values_and_bad_stringification() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert _coerce_tool_args_json(cyclic) == "{}"
    assert _coerce_tool_args_json({"value": float("nan")}) == "{}"
    assert _coerce_tool_args_json('{"value":NaN}') == "{}"
    assert _coerce_tool_args_json({("unsupported",): "key"}) == "{}"

    class BadString:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    assert _coerce_tool_args_json(BadString()) == "{}"


def test_bounded_history_text_handles_zero_small_exact_and_split_budgets() -> None:
    assert _bounded_history_text("content", 0) == ""
    assert _bounded_history_text("content", -1) == ""
    assert _bounded_history_text("content", 7) == "content"
    assert _bounded_history_text("x" * 1_000, 10) == "x" * 10

    bounded = _bounded_history_text("HEAD" + ("x" * 1_000) + "TAIL", 100)
    assert len(bounded) == 100
    assert bounded.startswith("HEAD")
    assert bounded.endswith("TAIL")
    assert "historical message truncated" in bounded


def test_historical_assistant_cleanup_is_linear_and_removes_only_leading_scaffold() -> None:
    raw = (
        "<think>private</think>\n\nGOAL: old\nPLAN: old\nRISK: low\n\n"
        "Answer\nPLAN: this line is content"
    )
    assert _clean_historical_assistant_text(raw) == "Answer\nPLAN: this line is content"
    assert _clean_historical_assistant_text("\nGOAL: only header\n") == ""
    assert _clean_historical_assistant_text("") == ""


def test_continuation_detection_and_candidate_order_are_bounded() -> None:
    assert _history_turn_limit("continue") == 64
    assert _history_turn_limit("yes please do that") == 64
    assert (
        _history_turn_limit("continue this detailed task with many additional specific words now")
        == 16
    )
    assert _history_turn_limit("brief unrelated request") == 16

    frames = [
        Frame(kind="system", content="system", turn_id=0),
        Frame(kind="user", content="one", turn_id=1),
        Frame(kind="assistant", content="one", turn_id=1),
        Frame(kind="user", content="two", turn_id=2),
    ]
    assert _candidate_prior_turn_ids(frames, "continue") == {1, 2}


@pytest.mark.parametrize("action_request", [False, True])
def test_history_budget_covers_profile_override_local_and_cloud_paths(action_request: bool) -> None:
    qwen = _history_budget_for_model(
        "qwen3.8-27b-fast:latest",
        provider_kind="ollama",
        action_request=action_request,
    )
    assert qwen == (6_553 if action_request else 9_830)

    assert _history_budget_for_model(
        "freetoken/model", provider_kind="unknown", action_request=action_request
    ) == (6_000 if action_request else 8_000)
    assert _history_budget_for_model(
        "custom.gguf", provider_kind="unknown", action_request=action_request
    ) == (6_000 if action_request else 8_000)
    assert _history_budget_for_model(
        "custom-local", provider_kind="ollama", action_request=action_request
    ) == (6_000 if action_request else 8_000)
    assert _history_budget_for_model(
        "custom:cloud", provider_kind="ollama", action_request=action_request
    ) == (12_000 if action_request else 16_000)


def test_history_selection_handles_zero_budget_missing_candidates_and_contiguous_tail() -> None:
    frames = [
        Frame(kind="user", content="x" * 4_000, turn_id=1),
        Frame(kind="user", content="recent", turn_id=2),
        Frame(kind="user", content="not selected", turn_id=3),
    ]
    assert _select_history_tail(frames, {1, 2}, token_budget=0) == (set(), set(), 0)
    assert _select_history_tail(frames, set(), token_budget=1_000) == (set(), set(), 2_048)

    turns, calls, cap = _select_history_tail(frames, {1, 2}, token_budget=100)
    assert turns == {2}
    assert calls == set()
    assert cap >= 4


def test_history_selection_keeps_only_complete_recent_unique_tool_pairs() -> None:
    frames = [
        Frame(kind="user", content="task", turn_id=1),
        Frame(kind="tool_call", content='{"x":1}', turn_id=1, call_id="old"),
        Frame(kind="tool_result", content="old result", turn_id=1, call_id="old"),
        Frame(kind="tool_call", content='{"x":2}', turn_id=1, call_id="repeat"),
        Frame(kind="tool_call", content='{"x":3}', turn_id=1, call_id="repeat"),
        Frame(kind="tool_result", content="r" * 4_000, turn_id=1, call_id="repeat"),
        Frame(kind="tool_call", content='{"x":4}', turn_id=1, call_id="unclosed"),
        Frame(kind="assistant", content="done", turn_id=1),
    ]
    turns, calls, _cap = _select_history_tail(frames, {1}, token_budget=3_000)
    assert turns == {1}
    assert calls <= {"old", "repeat"}
    assert "unclosed" not in calls


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (None, "in_progress"),
        (SimpleNamespace(raw=None), "in_progress"),
        (SimpleNamespace(raw={"stopped": True}), "in_progress"),
        (SimpleNamespace(raw={"stopped": False, "incomplete": True}), "in_progress"),
        (SimpleNamespace(raw={"accepted_outcome": True}), "complete"),
        (SimpleNamespace(raw={"accepted_outcome": False}), "in_progress"),
        (SimpleNamespace(raw={"verified_outcome": True}), "complete"),
        (SimpleNamespace(raw={"verified_outcome": False}), "in_progress"),
    ],
)
def test_checkpoint_status_requires_explicit_verified_evidence(
    response: object,
    expected: str,
) -> None:
    assert _checkpoint_status(object(), response) == expected


def test_coding_query_heuristic_has_positive_and_negative_paths() -> None:
    assert _is_coding_query("Please debug the gateway runtime") is True
    assert _is_coding_query("Recall our vacation itinerary") is False
    assert _is_coding_query("") is False
