from __future__ import annotations

import pytest

from norax.brain.quality_monitor import QualityLevel, QualityMonitor


def _record(
    monitor: QualityMonitor,
    *,
    success: float = 1.0,
    errors: int = 0,
    latency: float = 1.0,
    length: int = 500,
    text: str = "specific output with concrete evidence",
    workload: str = "coding",
) -> None:
    monitor.record(
        output_length=length,
        tool_success_rate=success,
        error_count=errors,
        response_time_sec=latency,
        tool_count=2,
        output_text=text,
        workload=workload,
        model="model-a",
    )


def test_style_and_length_changes_are_advisory_not_quality_failures():
    monitor = QualityMonitor()
    for _ in range(5):
        _record(monitor, length=1_000, text="many distinct technical words with long detail")
    for _ in range(5):
        _record(monitor, length=20, text="done")

    report = monitor.check()
    assert report.level is QualityLevel.HEALTHY
    assert report.score == 0.0
    assert "output_length_ratio_advisory" in report.trend
    assert report.signals == []


def test_objective_failures_and_latency_trigger_degradation():
    monitor = QualityMonitor()
    for _ in range(5):
        _record(monitor, success=1.0, errors=0, latency=1.0)
    for _ in range(5):
        _record(monitor, success=0.2, errors=3, latency=3.0)

    report = monitor.check()
    assert report.level is QualityLevel.DEGRADED
    assert report.score == 1.0
    assert any("tool_success_declined" in signal for signal in report.signals)
    assert any("errors_increased" in signal for signal in report.signals)
    assert any("latency_increased" in signal for signal in report.signals)
    assert "prompt" not in report.recommendation


def test_mixed_workloads_do_not_form_a_false_baseline():
    monitor = QualityMonitor(window_size=20)
    for index in range(12):
        _record(monitor, workload="coding" if index % 2 else "research")

    report = monitor.check()
    assert report.level is QualityLevel.HEALTHY
    assert report.recommendation == "insufficient_comparable_data"
    assert report.sample_size == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tool_success_rate": 1.1},
        {"tool_success_rate": float("nan")},
        {"error_count": -1},
        {"response_time_sec": float("inf")},
        {"tool_count": True},
    ],
)
def test_invalid_metrics_are_rejected(kwargs):
    values = {
        "output_length": 10,
        "tool_success_rate": 1.0,
        "error_count": 0,
        "response_time_sec": 1.0,
        "tool_count": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        QualityMonitor().record(**values)
