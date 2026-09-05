"""Advisory monitor for measurable operational degradation.

This module does not attempt to infer answer correctness from prose length,
formatting, or vocabulary. Those values are retained only as diagnostics.
Alerts are based on observed tool failures, errors, and latency within the same
workload/model cohort, and recommendations require investigation rather than
automatic prompt mutation or model replacement.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger("norax.brain.quality_monitor")


class QualityLevel(Enum):
    HEALTHY = "healthy"
    DEGRADING = "degrading"
    DEGRADED = "degraded"


@dataclass
class TurnMetric:
    timestamp: float
    output_length: int
    tool_success_rate: float
    error_count: int
    response_time_sec: float
    tool_count: int
    keyword_density: float = 0.0
    workload: str = "general"
    model: str = ""


@dataclass
class QualityReport:
    level: QualityLevel
    score: float
    signals: list[str] = field(default_factory=list)
    recommendation: str = ""
    trend: dict[str, float] = field(default_factory=dict)
    sample_size: int = 0


class QualityMonitor:
    """Detect objective operational regressions in comparable recent turns."""

    def __init__(
        self,
        *,
        window_size: int = 20,
        degradation_threshold: float = 0.6,
        degraded_threshold: float = 0.8,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 10:
            raise ValueError("window_size must be an integer of at least 10")
        if not 0 < degradation_threshold < degraded_threshold <= 1:
            raise ValueError("quality thresholds must satisfy 0 < degrading < degraded <= 1")
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.degraded_threshold = degraded_threshold
        self._history: deque[TurnMetric] = deque(maxlen=window_size)
        self._baseline: dict[str, float] = {}

    def record(
        self,
        *,
        output_length: int,
        tool_success_rate: float,
        error_count: int,
        response_time_sec: float,
        tool_count: int,
        output_text: str = "",
        workload: str = "general",
        model: str = "",
    ) -> None:
        """Record one completed turn after validating metric bounds."""
        integer_metrics = (output_length, error_count, tool_count)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_metrics
        ):
            raise ValueError("length, error_count, and tool_count must be non-negative integers")
        if (
            isinstance(tool_success_rate, bool)
            or not isinstance(tool_success_rate, (int, float))
            or not math.isfinite(float(tool_success_rate))
            or not 0 <= float(tool_success_rate) <= 1
        ):
            raise ValueError("tool_success_rate must be finite and between 0 and 1")
        if (
            isinstance(response_time_sec, bool)
            or not isinstance(response_time_sec, (int, float))
            or not math.isfinite(float(response_time_sec))
            or response_time_sec < 0
        ):
            raise ValueError("response_time_sec must be a finite non-negative number")

        words = output_text.lower().split()
        keyword_density = len(set(words)) / len(words) if words else 0.0
        self._history.append(
            TurnMetric(
                timestamp=time.time(),
                output_length=output_length,
                tool_success_rate=float(tool_success_rate),
                error_count=error_count,
                response_time_sec=float(response_time_sec),
                tool_count=tool_count,
                keyword_density=keyword_density,
                workload=str(workload or "general")[:100],
                model=str(model or "")[:200],
            )
        )

    def _cohort(self) -> list[TurnMetric]:
        if not self._history:
            return []
        latest = self._history[-1]
        return [
            metric
            for metric in self._history
            if metric.workload == latest.workload
            and (not latest.model or metric.model == latest.model)
        ]

    def check(self) -> QualityReport:
        """Compare the latest five cohort turns with preceding cohort history."""
        cohort = self._cohort()
        if len(cohort) < 10:
            return QualityReport(
                level=QualityLevel.HEALTHY,
                score=0.0,
                recommendation="insufficient_comparable_data",
                sample_size=len(cohort),
            )

        baseline, recent = cohort[:-5], cohort[-5:]
        signals: list[str] = []
        trend: dict[str, float] = {}
        score = 0.0

        baseline_length = sum(metric.output_length for metric in baseline) / len(baseline)
        recent_length = sum(metric.output_length for metric in recent) / len(recent)
        if baseline_length > 0:
            trend["output_length_ratio_advisory"] = round(recent_length / baseline_length, 2)
        baseline_density = sum(metric.keyword_density for metric in baseline) / len(baseline)
        recent_density = sum(metric.keyword_density for metric in recent) / len(recent)
        trend["lexical_diversity_advisory"] = round(recent_density, 2)
        trend["lexical_diversity_baseline_advisory"] = round(baseline_density, 2)

        baseline_tool_turns = [metric for metric in baseline if metric.tool_count > 0]
        recent_tool_turns = [metric for metric in recent if metric.tool_count > 0]
        if baseline_tool_turns and recent_tool_turns:
            baseline_success = sum(
                metric.tool_success_rate for metric in baseline_tool_turns
            ) / len(baseline_tool_turns)
            recent_success = sum(metric.tool_success_rate for metric in recent_tool_turns) / len(
                recent_tool_turns
            )
            success_drop = baseline_success - recent_success
            trend["tool_success_rate"] = round(recent_success, 2)
            trend["tool_success_baseline"] = round(baseline_success, 2)
            if success_drop >= 0.30:
                score += 0.5
                signals.append(
                    f"tool_success_declined (recent={recent_success:.2f} "
                    f"baseline={baseline_success:.2f})"
                )
            elif success_drop >= 0.15:
                score += 0.25
                signals.append(
                    f"tool_success_soft_decline (recent={recent_success:.2f} "
                    f"baseline={baseline_success:.2f})"
                )

        baseline_errors = sum(metric.error_count for metric in baseline) / len(baseline)
        recent_errors = sum(metric.error_count for metric in recent) / len(recent)
        error_increase = recent_errors - baseline_errors
        trend["error_count"] = round(recent_errors, 2)
        trend["error_count_baseline"] = round(baseline_errors, 2)
        if error_increase >= 2:
            score += 0.3
            signals.append(
                f"errors_increased (recent={recent_errors:.2f} baseline={baseline_errors:.2f})"
            )
        elif error_increase >= 0.5:
            score += 0.15
            signals.append(
                f"errors_soft_increase (recent={recent_errors:.2f} baseline={baseline_errors:.2f})"
            )

        baseline_latency = sum(metric.response_time_sec for metric in baseline) / len(baseline)
        recent_latency = sum(metric.response_time_sec for metric in recent) / len(recent)
        if baseline_latency > 0:
            latency_ratio = recent_latency / baseline_latency
            trend["response_time_ratio"] = round(latency_ratio, 2)
            if latency_ratio >= 2.5:
                score += 0.2
                signals.append(f"latency_increased (ratio={latency_ratio:.2f})")
            elif latency_ratio >= 1.75:
                score += 0.1
                signals.append(f"latency_soft_increase (ratio={latency_ratio:.2f})")

        score = min(1.0, score)
        if score >= self.degraded_threshold:
            level = QualityLevel.DEGRADED
            recommendation = "investigate_tool_errors_and_provider_health"
        elif score >= self.degradation_threshold:
            level = QualityLevel.DEGRADING
            recommendation = "inspect_recent_failures_before_changing_runtime"
        else:
            level = QualityLevel.HEALTHY
            recommendation = "continue"
        if level is not QualityLevel.HEALTHY:
            log.warning(
                "quality_monitor level=%s score=%.2f signals=%s", level.value, score, signals
            )

        self._baseline = {
            "tool_success_rate": trend.get("tool_success_baseline", 0.0),
            "error_count": baseline_errors,
            "response_time_sec": baseline_latency,
        }
        return QualityReport(
            level=level,
            score=round(score, 3),
            signals=signals,
            recommendation=recommendation,
            trend=trend,
            sample_size=len(cohort),
        )

    def reset(self) -> None:
        self._history.clear()
        self._baseline.clear()

    def stats(self) -> dict[str, Any]:
        report = self.check()
        return {
            "window_size": len(self._history),
            "comparable_sample_size": report.sample_size,
            "baseline": dict(self._baseline),
            "level": (
                report.level.value
                if report.recommendation != "insufficient_comparable_data"
                else "insufficient_data"
            ),
        }
