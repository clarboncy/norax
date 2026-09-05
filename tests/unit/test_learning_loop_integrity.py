from __future__ import annotations

from pathlib import Path

import pytest

from norax.brain import learning_loop as learning_module
from norax.brain.learning_loop import CURRICULUM, LearningConfig, LearningLoop, LearningResult


def _loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config_overrides) -> LearningLoop:
    monkeypatch.setattr(learning_module, "LEARNING_DIR", tmp_path / "intel")
    monkeypatch.setattr(learning_module, "LEARNING_LOG", tmp_path / "intel" / "learning.jsonl")
    return LearningLoop(LearningConfig(**config_overrides))


def test_curriculum_scripts_are_syntactically_valid() -> None:
    for topic in CURRICULUM:
        compile(topic["test_code"], topic["test_script"], "exec")


def test_learning_test_requires_staging_or_explicit_local_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(
        tmp_path,
        monkeypatch,
        staging_host="",
        allow_local_tests=False,
    )

    ok, stdout, stderr, location = loop._run_test_on_staging("raise AssertionError")

    assert ok is False
    assert stdout == ""
    assert "local learning tests are disabled" in stderr
    assert location == "not_run"


def test_explicit_local_learning_test_reports_its_real_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(
        tmp_path,
        monkeypatch,
        staging_host="",
        allow_local_tests=True,
        max_test_time_sec=5,
    )

    ok, stdout, stderr, location = loop._run_test_on_staging(
        'print("VERDICT: GOOD (measured=true)")'
    )

    assert ok is True
    assert "VERDICT: GOOD" in stdout
    assert stderr == ""
    assert location == "local_opt_in"


@pytest.mark.parametrize(
    ("output", "expected_verdict", "healthy"),
    [
        ("VERDICT: STABLE (errors=0)", "STABLE", True),
        ("VERDICT: DOWN (status=503)", "DOWN", False),
        ("process exited cleanly", "INCONCLUSIVE", False),
    ],
)
def test_health_verdict_is_distinct_from_process_exit(
    output: str,
    expected_verdict: str,
    healthy: bool,
) -> None:
    assert LearningLoop._classify_verdict(output) == (expected_verdict, healthy)


@pytest.mark.asyncio
async def test_cycle_does_not_label_unhealthy_zero_exit_as_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path, monkeypatch, staging_host="staging")
    monkeypatch.setattr(
        loop,
        "_next_topic",
        lambda: {
            "topic": "health_check",
            "search_queries": [],
            "test_code": "print('VERDICT: DOWN')",
        },
    )
    monkeypatch.setattr(
        loop,
        "_run_test_on_staging",
        lambda _code: (True, "VERDICT: DOWN (status=503)", "", "ssh:staging"),
    )
    monkeypatch.setattr(loop, "_persist_result", lambda _result: None)

    result = await loop.run_one_cycle()

    assert result.test_executed is True
    assert result.test_passed is False
    assert result.test_verdict == "DOWN"
    assert result.test_location == "ssh:staging"


def test_persisted_report_records_actual_execution_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop(tmp_path, monkeypatch)
    result = LearningResult(
        topic="audit",
        timestamp=1_700_000_000.0,
        test_executed=True,
        test_passed=False,
        test_verdict="UNSTABLE",
        test_location="local_opt_in",
        test_output="VERDICT: UNSTABLE",
    )

    loop._persist_result(result)

    report = next((tmp_path / "intel").glob("learned_audit_*.md")).read_text()
    assert "**Tested on:** local_opt_in" in report
    assert "**Verdict:** UNSTABLE" in report
