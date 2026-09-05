"""L25 VTA — Reward Prediction Error (RPE) + D-MEM Memory Routing.

The brain's learning signal. Computes the difference between expected
and actual reward. High surprise → flashbulb memory. Low surprise →
scratchpad only.

D-MEM routing (Song & Xin, UCSD/CMU):
  |RPE| < 2  → scratchpad only (fast, volatile)
  |RPE| 2-4  → sleep buffer (consolidate later)
  |RPE| > 4  → immediate flashbulb to semantic memory

Norax hot-path module.
Norax native: state persisted via memory store.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from ...atomic import atomic_write_text, read_bounded_text

log = logging.getLogger("norax.brain.vta")

MemoryRoute = Literal["scratchpad", "sleep_buffer", "immediate_flashbulb"]
_MAX_STATE_BYTES = 1_048_576
_MAX_EXPECTATIONS = 512
_MAX_HISTORY = 200
_MAX_ACTION_CHARS = 80


def _action_key(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("action must be a string")
    return value.strip().lower()[:_MAX_ACTION_CHARS]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class RPEResult:
    action: str
    expected: float
    actual: float
    rpe: float  # actual - expected
    abs_rpe: float
    route: MemoryRoute
    reason: str
    confidence: float  # 0-1


@dataclass(frozen=True)
class PredictionResult:
    action: str
    expected_reward: float
    confidence: float
    trend: str  # "up", "down", "stable", "unknown"
    history_count: int


@dataclass
class _HistoryEntry:
    action: str
    expected: float
    actual: float
    rpe: float
    timestamp: float


@dataclass
class VTA:
    """Ventral Tegmental Area — reward prediction and learning."""

    state_path: Path | None = None
    _expectations: dict[str, float] = field(default_factory=dict)
    _history: list[_HistoryEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state_path and self.state_path.exists():
            self._load()

    def _load(self) -> None:
        if not self.state_path:
            return
        try:
            data = json.loads(read_bounded_text(self.state_path, max_bytes=_MAX_STATE_BYTES))
            if not isinstance(data, dict):
                raise ValueError("VTA state root must be an object")
            raw_expectations = data.get("expectations", {})
            raw_history = data.get("history", [])
            if not isinstance(raw_expectations, dict) or not isinstance(raw_history, list):
                raise ValueError("VTA state collections are malformed")

            expectations: dict[str, float] = {}
            for raw_action, raw_expected in islice(raw_expectations.items(), _MAX_EXPECTATIONS):
                try:
                    action = _action_key(raw_action)
                except TypeError:
                    continue
                expected = _finite_number(raw_expected)
                if action and expected is not None:
                    expectations[action] = min(10.0, max(0.0, expected))

            history: list[_HistoryEntry] = []
            for h in raw_history[-_MAX_HISTORY:]:
                if not isinstance(h, dict):
                    continue
                try:
                    action = _action_key(h.get("action"))
                except TypeError:
                    continue
                expected = _finite_number(h.get("expected"))
                actual = _finite_number(h.get("actual"))
                timestamp = _finite_number(h.get("timestamp", 0.0))
                if not action or expected is None or actual is None or timestamp is None:
                    continue
                expected = min(10.0, max(0.0, expected))
                actual = min(10.0, max(0.0, actual))
                history.append(
                    _HistoryEntry(
                        action=action,
                        expected=expected,
                        actual=actual,
                        rpe=actual - expected,
                        timestamp=max(0.0, timestamp),
                    )
                )
            self._expectations = expectations
            self._history = history
        except Exception:
            log.warning("vta.load_failed", exc_info=True)

    def _save(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "expectations": self._expectations,
                "history": [
                    {
                        "action": h.action,
                        "expected": h.expected,
                        "actual": h.actual,
                        "rpe": h.rpe,
                        "timestamp": h.timestamp,
                    }
                    for h in self._history[-200:]
                ],
            }
            atomic_write_text(
                self.state_path,
                json.dumps(data, indent=2, allow_nan=False),
                mode=0o600,
            )
        except Exception:
            log.warning("vta.save_failed", exc_info=True)

    def predict(self, action: str) -> PredictionResult:
        """Predict expected reward for an action."""
        key = _action_key(action)
        expected = self._expectations.get(key, 5.0)  # neutral prior

        relevant = [h for h in self._history if h.action == key]
        if len(relevant) >= 3:
            last3 = relevant[-3:]
            trend_val = sum(h.rpe for h in last3) / len(last3)
            trend = "up" if trend_val > 0.5 else "down" if trend_val < -0.5 else "stable"
            # Confidence from variance
            rewards = [h.actual for h in relevant[-10:]]
            variance = sum((r - expected) ** 2 for r in rewards) / len(rewards)
            confidence = max(0.1, min(1.0, 1.0 - math.sqrt(variance) / 5))
        else:
            trend = "unknown"
            confidence = 0.3

        return PredictionResult(
            action=key,
            expected_reward=expected,
            confidence=confidence,
            trend=trend,
            history_count=len(relevant),
        )

    def record_outcome(self, action: str, actual_score: float) -> RPEResult:
        """Record outcome and compute RPE + D-MEM routing."""
        key = _action_key(action)
        if isinstance(actual_score, bool) or not isinstance(actual_score, (int, float)):
            raise TypeError("actual_score must be a number")
        actual_score = float(actual_score)
        if not math.isfinite(actual_score) or not 0.0 <= actual_score <= 10.0:
            raise ValueError("actual_score must be finite and between 0 and 10")
        expected = self._expectations.get(key, 5.0)
        rpe = actual_score - expected
        abs_rpe = abs(rpe)

        # D-MEM routing
        if abs_rpe < 2:
            route: MemoryRoute = "scratchpad"
            reason = "low_surprise"
        elif abs_rpe < 4:
            route = "sleep_buffer"
            reason = "moderate_surprise"
        else:
            route = "immediate_flashbulb"
            reason = "high_surprise"

        # Update expectation (EMA)
        alpha = 0.3
        self._expectations[key] = alpha * actual_score + (1 - alpha) * expected

        # Record history
        self._history.append(
            _HistoryEntry(
                action=key,
                expected=expected,
                actual=actual_score,
                rpe=rpe,
                timestamp=time.time(),
            )
        )
        # Trim
        if len(self._history) > 200:
            self._history = self._history[-200:]

        self._save()

        return RPEResult(
            action=key,
            expected=expected,
            actual=actual_score,
            rpe=rpe,
            abs_rpe=abs_rpe,
            route=route,
            reason=reason,
            confidence=min(1.0, len([h for h in self._history if h.action == key]) / 10),
        )
