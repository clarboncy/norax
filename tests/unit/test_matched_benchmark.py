"""Independent benchmark grading rejects wrong artifacts and unsafe code."""

from __future__ import annotations

import json

import pytest

from benchmarks.matched_agent_bench import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_ROUND_LIMIT,
    DEFAULT_TURN_BUDGET_SECONDS,
    Task,
)


def test_matched_benchmark_defaults_allow_full_completion() -> None:
    assert DEFAULT_ROUND_LIMIT == 250
    assert DEFAULT_TURN_BUDGET_SECONDS == 14_400
    assert DEFAULT_REQUEST_TIMEOUT_SECONDS == 600


@pytest.mark.parametrize("name", ["repair_clamp", "aggregate", "transient_recovery"])
def test_missing_or_incorrect_artifact_never_passes(tmp_path, name):
    task = Task(tmp_path, name, 42)
    assert task.check() is False
    task.call("bench_write", {"content": "DONE"})
    assert task.call("bench_check", {})["ok"] is False


def test_clamp_is_checked_against_boundary_cases_not_substrings(tmp_path):
    task = Task(tmp_path, "repair_clamp", 42)
    task.call("bench_write", {"content": task.call("bench_read", {})["content"]})
    assert task.check() is False
    task.call(
        "bench_write",
        {"content": "def clamp(value, lower, upper):\n    return max(lower, min(upper, value))\n"},
    )
    assert task.check() is True
    task.call(
        "bench_write",
        {
            "content": "import os\ndef clamp(value, lower, upper):\n    return max(lower, min(upper, value))\n"
        },
    )
    assert task.check() is False


def test_aggregate_uses_only_paid_rows_and_exact_numeric_types(tmp_path):
    task = Task(tmp_path, "aggregate", 42)
    task.rows = [
        {"region": "east", "amount": 1, "status": "paid"},
        {"region": "east", "amount": 9, "status": "void"},
    ]
    for wrong in ({"east": 10}, {"east": True}, {"east": "1"}, {"east": 1, "west": 0}):
        task.call("bench_write", {"content": json.dumps(wrong)})
        assert task.check() is False
    task.call("bench_write", {"content": '{"east": 1}'})
    assert task.check() is True


def test_retry_failure_requires_a_real_followup_read(tmp_path):
    task = Task(tmp_path, "transient_recovery", 42)
    assert task.call("bench_read", {}) == {
        "ok": False,
        "error": "temporary_read_failure",
        "retryable": True,
    }
    receipt = task.call("bench_read", {})
    assert receipt["ok"] is True
    task.call("bench_write", {"content": receipt["content"] + "\n"})
    assert task.check() is False
    task.call("bench_write", {"content": receipt["content"]})
    assert task.check() is True
