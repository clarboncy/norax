from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

import pytest

from benchmarks import ollama_power_bench, perf_bench


def test_live_benchmark_receives_only_its_final_thread_reply():
    events = iter(
        [
            {"type": "status", "status": "working"},
            {"type": "reply", "thread_id": "other", "text": "unrelated"},
            {"type": "delta", "thread_id": "ours", "text": "partial"},
            {"type": "reply", "thread_id": "ours", "text": '{"answer": 42}'},
        ]
    )

    class Socket:
        def recv(self, **kwargs):
            return json.dumps(next(events))

    assert perf_bench.receive_reply(Socket(), "ours", 1) == '{"answer": 42}'


def test_live_benchmark_reply_timeout_is_not_a_pass():
    class Socket:
        def recv(self, **kwargs):
            raise TimeoutError

    reply = perf_bench.receive_reply(Socket(), "ours", 1)
    assert perf_bench.score_task({"expect_exact": "PONG"}, reply) == (0.0, ["empty_reply"])


def test_perf_required_constraints_are_hard_failures() -> None:
    task = {"must_contain": ["alpha", "beta"]}

    score, problems = perf_bench.score_task(task, "alpha only")

    assert score == 0.5
    assert problems == ["missing:beta"]


def test_perf_exact_arithmetic_cannot_pass_on_substring() -> None:
    task = next(task for task in perf_bench.TASKS if task["name"] == "reasoning_arithmetic")

    score, problems = perf_bench.score_task(task, "15")

    assert score == 0.0
    assert problems == ["exact_mismatch"]


def test_perf_measurement_detects_background_counter_pollution() -> None:
    result = {"task": "one", "score": 1.0, "ok": True}

    measurement = perf_bench.assess_measurement(
        [result],
        {"brain_turns": 2.0, "brain_seconds_count": 2.0, "gateway_requests": 2.0},
        expected_turns=1,
    )

    assert measurement["valid"] is False
    assert measurement["reasons"]


def test_perf_measurement_detects_synthetic_or_retry_request_pollution() -> None:
    measurement = perf_bench.assess_measurement(
        [{"task": "one", "score": 1.0, "ok": True}],
        {"brain_turns": 1.0, "brain_seconds_count": 1.0, "gateway_requests": 2.0},
        expected_turns=1,
    )

    assert measurement["valid"] is False
    assert measurement["reasons"] == ["gateway_requests=2, expected=1"]


def test_perf_labels_cannot_escape_run_directory() -> None:
    with pytest.raises(ValueError):
        perf_bench._run_path("../../outside")


def _perf_run(*, valid: bool = True, score: float = 1.0) -> dict[str, Any]:
    return {
        "schema": "norax.perf_bench.run.v2",
        "model": "model-a",
        "provider": "provider-a",
        "measurement": {"valid": valid},
        "task_results": [{"task": "exact", "repetition": 1, "ok": score == 1.0, "score": score}],
        "summary": {
            "distinct_tasks": 1,
            "repetitions": 1,
            "samples": 1,
            "passed": int(score == 1.0),
            "mean_score": score,
            "tokens_in_delta": 100,
            "tokens_out_delta": 20,
            "mean_brain_seconds": 2.0,
        },
    }


def test_perf_diff_refuses_contaminated_input() -> None:
    report = perf_bench.build_diff_report(_perf_run(valid=False), _perf_run(valid=True))

    assert report["comparison_valid"] is False
    assert report["verdict"] == ["INVALID COMPARISON: do not interpret performance deltas."]


def test_nonstream_reasoning_is_not_used_as_public_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_power_bench,
        "post_json",
        lambda *_args, **_kwargs: {
            "choices": [
                {
                    "message": {"content": None, "reasoning": "PONG"},
                    "finish_reason": "stop",
                }
            ]
        },
    )

    observation = ollama_power_bench.nonstream_chat("http://local/v1", "model", "prompt", 10, 0.0)

    assert observation.content == ""
    assert observation.reasoning_chars == 4
    assert ollama_power_bench.score_content(
        {"expect": "PONG"}, observation.content, observation.finish_reasons
    ) == ["empty_public_content", "exact_mismatch"]


def test_power_bench_strips_embedded_reasoning_and_reports_honest_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = ollama_power_bench.ChatObservation(
        content="<think>private chain</think>PONG",
        reasoning_chars=3,
        seconds_total=2.0,
        seconds_first_any_field=0.5,
        seconds_first_content_field=0.5,
        usage={"prompt_tokens": 10, "completion_tokens": 4},
        metadata={},
        finish_reasons=["stop"],
    )
    monkeypatch.setattr(ollama_power_bench, "stream_chat", lambda *_args, **_kwargs: observation)
    args = Namespace(
        stream=True,
        base="http://local/v1",
        model="model",
        max_tokens=10,
        temperature=0.0,
    )

    result = ollama_power_bench.run_task(
        args, {"name": "exact", "prompt": "p", "expect": "PONG"}, 1
    )

    assert result.ok is True
    assert result.visible_chars == 4
    assert result.reasoning_chars > 3
    assert result.completion_tok_per_sec_e2e == 2.0
    assert result.completion_tok_per_sec_after_first == 2.0


def test_power_bench_treats_length_finish_as_failure() -> None:
    assert ollama_power_bench.score_content({"expect": "PONG"}, "PONG", ["length"]) == ["truncated"]


@pytest.mark.parametrize(
    "option,value",
    [
        ("--tasks", ","),
        ("--tasks", ""),
        ("--temperature", "nan"),
        ("--temperature", "inf"),
        ("--repetitions", "101"),
        ("--max-tokens", "0"),
    ],
)
def test_power_benchmark_rejects_empty_or_invalid_runs_before_inference(monkeypatch, option, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid benchmark must not invoke a model")

    monkeypatch.setattr(ollama_power_bench, "run_task", forbidden)
    with pytest.raises(SystemExit) as error:
        ollama_power_bench.main(["--model", "fixture", option, value])
    assert error.value.code == 2


def test_keyword_only_answer_does_not_verify_coding_task_completion(monkeypatch):
    observation = ollama_power_bench.ChatObservation(
        content="class LRUCache pytest def test",
        reasoning_chars=0,
        seconds_total=1,
        seconds_first_any_field=0.1,
        seconds_first_content_field=0.1,
        usage={},
        metadata={},
        finish_reasons=["stop"],
    )
    monkeypatch.setattr(ollama_power_bench, "stream_chat", lambda *args: observation)
    args = Namespace(
        stream=True, base="http://fixture/v1", model="fixture", max_tokens=100, temperature=0
    )
    task = next(task for task in ollama_power_bench.TASKS if task["name"] == "coding_pytest")
    result = ollama_power_bench.run_task(args, task, 1)
    assert result.ok
    assert result.validation_scope == "keyword_presence_only"
    assert result.task_completion_verified is False
