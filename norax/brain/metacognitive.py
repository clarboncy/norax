"""Metacognitive Calibration — confidence calibration curve.

Detects overconfidence and underconfidence patterns. Enables:
  - Route adjustment: overconfident agent on weak tasks → escalate
  - Output calibration: inject confidence disclaimers when overconfident
  - Self-awareness: know when to trust own output vs. verify more
  - Trend tracking: is confidence getting better or worse over time?

Zero LLM calls. Uses Brier score (meteorology standard) and calibration
curve (reliability diagram) to measure how well confidence matches reality.

Usage:
    mc = MetacognitiveCalibration(Path("~/norax/memory/metacog.json"))
    mc.load()
    mc.record_prediction(confidence=0.9, actual_success=True)
    mc.record_prediction(confidence=0.3, actual_success=False)
    report = mc.get_report()
    # report.brier_score → 0.0 (perfect) to 1.0 (worst)
    # report.calibration_error → how far off confidence is from reality
    # report.bias → "overconfident" | "underconfident" | "calibrated"
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..atomic import atomic_write_text, read_bounded_text

log = logging.getLogger("norax.brain.metacognitive")

MIN_PREDICTIONS = 10  # need at least 10 predictions before trusting calibration
CALIBRATION_BINS = 10  # 10 bins of 0.1 width: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
DECAY_HALF_LIFE = 14 * 24 * 3600  # 14 days
_MAX_STATE_BYTES = 2 * 1024 * 1024
_MAX_PREDICTIONS = 500
_MAX_LABEL_CHARS = 128


def _finite_number(
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float = float("inf"),
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def _label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split())[:_MAX_LABEL_CHARS]


class ConfidenceBias(Enum):
    CALIBRATED = "calibrated"
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"


@dataclass
class CalibrationBin:
    """One bin of the calibration curve."""

    confidence_range: tuple[float, float]
    predicted_confidence: float = 0.0  # avg confidence in this bin
    actual_success_rate: float = 0.0  # actual success rate in this bin
    count: int = 0

    @property
    def gap(self) -> float:
        """How far off the confidence is from reality in this bin."""
        return self.predicted_confidence - self.actual_success_rate


@dataclass
class CalibrationReport:
    brier_score: float = 0.0  # 0=perfect, 1=worst
    calibration_error: float = 0.0  # avg |gap| across bins
    bias: ConfidenceBias = ConfidenceBias.CALIBRATED
    bias_magnitude: float = 0.0  # signed: + = overconfident, - = underconfident
    bins: list[CalibrationBin] = field(default_factory=list)
    total_predictions: int = 0
    trend: str = "stable"  # improving, degrading, stable
    recommendation: str = ""

    @property
    def is_reliable(self) -> bool:
        return self.total_predictions >= MIN_PREDICTIONS

    @property
    def is_overconfident(self) -> bool:
        return self.bias == ConfidenceBias.OVERCONFIDENT

    @property
    def is_underconfident(self) -> bool:
        return self.bias == ConfidenceBias.UNDERCONFIDENT


@dataclass
class PredictionRecord:
    confidence: float
    actual_success: bool
    timestamp: float
    tool: str = ""
    domain: str = ""


class MetacognitiveCalibration:
    """Confidence calibration tracker with decay and trend analysis."""

    def __init__(self, path: Path | str = "~/norax/memory/metacog.json"):
        self.path = Path(path).expanduser()
        self.predictions: deque[PredictionRecord] = deque(maxlen=_MAX_PREDICTIONS)
        self._loaded = False

    def load(self) -> None:
        try:
            data = json.loads(read_bounded_text(self.path, max_bytes=_MAX_STATE_BYTES))
            if not isinstance(data, dict):
                raise ValueError("metacognitive state root must be an object")
            raw_predictions = data.get("predictions", [])
            if not isinstance(raw_predictions, list):
                raise ValueError("metacognitive predictions must be a list")
            if len(raw_predictions) > _MAX_PREDICTIONS:
                raise ValueError("metacognitive state contains too many predictions")

            predictions: deque[PredictionRecord] = deque(maxlen=_MAX_PREDICTIONS)
            for raw_prediction in raw_predictions:
                if not isinstance(raw_prediction, dict):
                    continue
                confidence = _finite_number(
                    raw_prediction.get("confidence"),
                    maximum=1.0,
                )
                timestamp = _finite_number(raw_prediction.get("timestamp", 0.0))
                actual_success = raw_prediction.get("actual_success")
                tool = _label(raw_prediction.get("tool", ""))
                domain = _label(raw_prediction.get("domain", ""))
                if (
                    confidence is None
                    or timestamp is None
                    or not isinstance(actual_success, bool)
                    or tool is None
                    or domain is None
                ):
                    continue
                predictions.append(
                    PredictionRecord(
                        confidence=confidence,
                        actual_success=actual_success,
                        timestamp=timestamp,
                        tool=tool,
                        domain=domain,
                    )
                )
            self.predictions = predictions
            log.info("metacognitive loaded: %d predictions", len(self.predictions))
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as e:
            log.warning("metacognitive load failed: %r", e)
        self._loaded = True

    def save(self) -> None:
        data = {
            "predictions": [
                {
                    "confidence": p.confidence,
                    "actual_success": p.actual_success,
                    "timestamp": p.timestamp,
                    "tool": p.tool,
                    "domain": p.domain,
                }
                for p in self.predictions
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(data, separators=(",", ":"), allow_nan=False)
        if len(serialized.encode("utf-8")) > _MAX_STATE_BYTES:
            raise ValueError(f"metacognitive state exceeds {_MAX_STATE_BYTES} bytes")
        atomic_write_text(self.path, serialized, mode=0o600)

    def record_prediction(
        self,
        confidence: float,
        actual_success: bool,
        tool: str = "",
        domain: str = "",
        now: float | None = None,
    ) -> None:
        """Record a prediction and its outcome."""
        if not isinstance(actual_success, bool):
            raise TypeError("actual_success must be a boolean")
        normalized_confidence = _finite_number(confidence, maximum=1.0)
        timestamp = _finite_number(time.time() if now is None else now)
        normalized_tool = _label(tool)
        normalized_domain = _label(domain)
        if normalized_confidence is None:
            raise ValueError("confidence must be between 0 and 1")
        if timestamp is None:
            raise ValueError("now must be a finite non-negative number")
        if normalized_tool is None or normalized_domain is None:
            raise TypeError("tool and domain must be strings")
        self.predictions.append(
            PredictionRecord(
                confidence=normalized_confidence,
                actual_success=actual_success,
                timestamp=timestamp,
                tool=normalized_tool,
                domain=normalized_domain,
            )
        )

    def _apply_decay(self, now: float) -> list[tuple[PredictionRecord, float]]:
        """Weight recent predictions more than old ones."""
        weighted: list[tuple[PredictionRecord, float]] = []
        for p in self.predictions:
            elapsed = now - p.timestamp
            if elapsed <= 0:
                weight = 1.0
            else:
                weight = 0.5 ** (elapsed / DECAY_HALF_LIFE)
            if weight > 0.01:  # skip very old predictions
                weighted.append((p, weight))
        return weighted

    def get_report(
        self,
        now: float | None = None,
        *,
        tool: str = "",
        domain: str = "",
    ) -> CalibrationReport:
        """Compute calibration report."""
        timestamp = _finite_number(time.time() if now is None else now)
        if timestamp is None:
            raise ValueError("now must be a finite non-negative number")
        normalized_tool = _label(tool)
        normalized_domain = _label(domain)
        if normalized_tool is None or normalized_domain is None:
            raise TypeError("tool and domain must be strings")
        weighted = self._apply_decay(timestamp)
        if normalized_tool:
            weighted = [
                (record, weight) for record, weight in weighted if record.tool == normalized_tool
            ]
        if normalized_domain:
            weighted = [
                (record, weight)
                for record, weight in weighted
                if record.domain == normalized_domain
            ]

        if len(weighted) < MIN_PREDICTIONS:
            return CalibrationReport(
                total_predictions=len(weighted),
                recommendation="Insufficient data for calibration analysis",
            )

        # Brier score: mean((confidence - actual)^2)
        # actual is 1.0 for success, 0.0 for failure
        brier = sum(
            w * (p.confidence - (1.0 if p.actual_success else 0.0)) ** 2 for p, w in weighted
        ) / sum(w for _, w in weighted)

        # Calibration curve (reliability diagram)
        bins: list[CalibrationBin] = [
            CalibrationBin(confidence_range=(i / 10, (i + 1) / 10)) for i in range(CALIBRATION_BINS)
        ]

        binned: list[list[tuple[PredictionRecord, float]]] = [[] for _ in range(CALIBRATION_BINS)]
        for p, w in weighted:
            bin_idx = min(int(p.confidence * 10), 9)
            binned[bin_idx].append((p, w))
        bin_weights: dict[int, float] = {}
        for index, records in enumerate(binned):
            if not records:
                continue
            weight_sum = sum(weight for _, weight in records)
            if weight_sum <= 0:
                continue
            bin_weights[index] = weight_sum
            bins[index].count = len(records)
            bins[index].predicted_confidence = (
                sum(record.confidence * weight for record, weight in records) / weight_sum
            )
            bins[index].actual_success_rate = (
                sum((1.0 if record.actual_success else 0.0) * weight for record, weight in records)
                / weight_sum
            )

        # Only consider bins with enough samples
        valid_bins = [b for b in bins if b.count >= 2]

        if not valid_bins:
            return CalibrationReport(
                brier_score=brier,
                total_predictions=len(weighted),
                recommendation="Insufficient bin samples for calibration curve",
            )

        # Calibration error: weighted mean of |gap|
        valid_indexes = [index for index, item in enumerate(bins) if item.count >= 2]
        total_weight = sum(bin_weights[index] for index in valid_indexes) or 1.0
        calibration_error = (
            sum(abs(bins[index].gap) * bin_weights[index] for index in valid_indexes) / total_weight
        )

        # Bias: signed mean gap
        bias_magnitude = (
            sum(bins[index].gap * bin_weights[index] for index in valid_indexes) / total_weight
        )

        if bias_magnitude > 0.1:
            bias = ConfidenceBias.OVERCONFIDENT
        elif bias_magnitude < -0.1:
            bias = ConfidenceBias.UNDERCONFIDENT
        else:
            bias = ConfidenceBias.CALIBRATED

        # Trend analysis: compare first half vs second half of predictions
        sorted_preds = sorted(weighted, key=lambda x: x[0].timestamp)
        mid = len(sorted_preds) // 2
        if mid > 0:
            first_half_brier = sum(
                w * (p.confidence - (1.0 if p.actual_success else 0.0)) ** 2
                for p, w in sorted_preds[:mid]
            ) / sum(w for _, w in sorted_preds[:mid])
            second_half_brier = sum(
                w * (p.confidence - (1.0 if p.actual_success else 0.0)) ** 2
                for p, w in sorted_preds[mid:]
            ) / sum(w for _, w in sorted_preds[mid:])
            if second_half_brier < first_half_brier - 0.02:
                trend = "improving"
            elif second_half_brier > first_half_brier + 0.02:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Recommendation
        if bias == ConfidenceBias.OVERCONFIDENT:
            recommendation = (
                f"OVERCONFIDENT by {abs(bias_magnitude):.2f}. "
                "Reduce confidence on outputs, add verification steps, "
                "and escalate to stronger models more often."
            )
        elif bias == ConfidenceBias.UNDERCONFIDENT:
            recommendation = (
                f"UNDERCONFIDENT by {abs(bias_magnitude):.2f}. "
                "Trust outputs more, reduce unnecessary verification, "
                "and use weaker models more often to save cost."
            )
        else:
            recommendation = "Well calibrated. Confidence matches reality."

        return CalibrationReport(
            brier_score=brier,
            calibration_error=calibration_error,
            bias=bias,
            bias_magnitude=bias_magnitude,
            bins=valid_bins,
            total_predictions=len(weighted),
            trend=trend,
            recommendation=recommendation,
        )

    def calibrate_confidence(
        self, raw_confidence: float, tool: str = "", domain: str = ""
    ) -> float:
        """Adjust a raw confidence score based on calibration history."""
        normalized = _finite_number(raw_confidence, maximum=1.0)
        if normalized is None:
            raise ValueError("raw_confidence must be between 0 and 1")
        report = self.get_report(tool=tool, domain=domain)
        if not report.is_reliable:
            return normalized  # not enough data to calibrate

        # Apply bias correction
        if report.bias == ConfidenceBias.OVERCONFIDENT:
            # Reduce confidence
            adjusted = normalized - abs(report.bias_magnitude) * 0.5
        elif report.bias == ConfidenceBias.UNDERCONFIDENT:
            # Increase confidence
            adjusted = normalized + abs(report.bias_magnitude) * 0.5
        else:
            return normalized

        return max(0.05, min(0.95, adjusted))

    def summary(self) -> str:
        r = self.get_report()
        return (
            f"Metacognitive: {r.total_predictions} predictions, "
            f"brier={r.brier_score:.3f}, "
            f"bias={r.bias.value} ({r.bias_magnitude:+.2f}), "
            f"trend={r.trend}"
        )
