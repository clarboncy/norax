"""Active Inference — prediction→observation→error→update loop.

Core learning mechanism that doesn't require weight updates. On every
turn, the agent:
  1. Predicts what will happen (tool outcome, user satisfaction, task completion)
  2. Observes what actually happens
  3. Computes prediction error (surprisal)
  4. Updates internal models to reduce future error

This is the Friston free-energy principle applied to agent operations.
Zero LLM calls. Pure statistical learning.

Usage:
    ai = ActiveInference(Path("~/norax/memory/active_inference.json"))
    ai.load()

    # Before tool execution
    prediction = ai.predict_tool_outcome("exec", "coding", {"command": "ls"})
    # prediction.expected_success = 0.85, prediction.expected_duration_ms = 200

    # After tool execution
    ai.record_observation(
        tool="exec", domain="coding",
        predicted_success=0.85, actual_success=True,
        predicted_duration_ms=200, actual_duration_ms=150,
    )

    # Get prediction errors for learning
    errors = ai.get_recent_errors()
    # High error → agent should pay more attention to this tool/domain
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text, read_bounded_text

log = logging.getLogger("norax.brain.active_inference")

DECAY_HALF_LIFE = 7 * 24 * 3600  # 7 days
MIN_OBSERVATIONS = 5
_MAX_STATE_BYTES = 4 * 1024 * 1024
_MAX_MODELS = 2_048
_MAX_COMPONENT_CHARS = 128
_MAX_MODEL_KEY_CHARS = _MAX_COMPONENT_CHARS * 2 + 1
_MAX_COUNTER = 1_000_000_000
_MAX_DURATION_MS = 365 * 24 * 60 * 60 * 1_000.0


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


def _bounded_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_COUNTER:
        return None
    return value


def _component(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    normalized = " ".join(value.split())[:_MAX_COMPONENT_CHARS]
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if ":" in normalized:
        raise ValueError(f"{label} must not contain ':'")
    return normalized


@dataclass
class ToolPrediction:
    """Prediction for a tool execution outcome."""

    expected_success: float = 0.5
    expected_duration_ms: float = 1000.0
    confidence: float = 0.3  # confidence in the prediction itself
    basis: str = "prior"  # prior | learned | recent


@dataclass
class Observation:
    tool: str
    domain: str
    predicted_success: float
    actual_success: bool
    predicted_duration_ms: float
    actual_duration_ms: float
    prediction_error: float  # |predicted - actual|
    duration_error: float
    timestamp: float
    surprise: float = 0.0  # information-theoretic surprise


@dataclass
class ModelStats:
    """Running statistics for a tool/domain combination."""

    success_weight: float = 0.0
    failure_weight: float = 0.0
    total_count: int = 0
    last_updated: float = 0.0
    avg_duration_ms: float = 0.0
    duration_variance: float = 0.0
    prediction_error_sum: float = 0.0  # sum of |predicted - actual|
    prediction_error_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_weight + self.failure_weight
        if total < 0.01:
            return 0.5
        return self.success_weight / total

    @property
    def avg_prediction_error(self) -> float:
        if self.prediction_error_count == 0:
            return 0.5  # unknown → moderate error
        return self.prediction_error_sum / self.prediction_error_count

    @property
    def is_reliable(self) -> bool:
        return self.total_count >= MIN_OBSERVATIONS

    def record(
        self,
        success: bool,
        duration_ms: float,
        predicted_success: float,
        now: float | None = None,
    ) -> tuple[float, float]:
        """Record an observation. Returns (prediction_error, surprise)."""
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        duration = _finite_number(duration_ms, maximum=_MAX_DURATION_MS)
        prediction = _finite_number(predicted_success, maximum=1.0)
        timestamp = _finite_number(time.time() if now is None else now)
        if duration is None:
            raise ValueError("duration_ms must be a finite non-negative number")
        if prediction is None:
            raise ValueError("predicted_success must be between 0 and 1")
        if timestamp is None:
            raise ValueError("now must be a finite non-negative number")
        now = timestamp
        self._decay(now)
        weight = 1.0
        if success:
            self.success_weight = min(float(_MAX_COUNTER), self.success_weight + weight)
        else:
            self.failure_weight = min(float(_MAX_COUNTER), self.failure_weight + weight)
        self.total_count = min(_MAX_COUNTER, self.total_count + 1)
        self.last_updated = now

        # Duration stats (exponential moving average)
        if self.total_count == 1:
            self.avg_duration_ms = duration
        else:
            alpha = 0.15
            self.avg_duration_ms = (1 - alpha) * self.avg_duration_ms + alpha * duration

        # Prediction error
        actual_val = 1.0 if success else 0.0
        pred_error = abs(prediction - actual_val)
        self.prediction_error_sum = min(
            float(_MAX_COUNTER),
            self.prediction_error_sum + pred_error,
        )
        self.prediction_error_count = min(_MAX_COUNTER, self.prediction_error_count + 1)

        # Surprise: binary cross-entropy of prediction vs actual
        p = max(0.01, min(0.99, prediction))
        if success:
            surprise = -math.log(p)
        else:
            surprise = -math.log(1 - p)

        return pred_error, surprise

    def _decay(self, now: float) -> None:
        if self.last_updated == 0:
            return
        elapsed = now - self.last_updated
        if elapsed <= 0:
            return
        factor = 0.5 ** (elapsed / DECAY_HALF_LIFE)
        self.success_weight *= factor
        self.failure_weight *= factor


class ActiveInference:
    """Prediction→observation→error→update loop for agent learning."""

    def __init__(self, path: Path | str = "~/norax/memory/active_inference.json"):
        self.path = Path(path).expanduser()
        self.models: dict[str, ModelStats] = {}  # key = "tool:domain"
        self.recent_observations: deque[Observation] = deque(maxlen=100)
        self.total_surprise: float = 0.0
        self.total_observations: int = 0
        self._loaded = False

    def _key(self, tool: str, domain: str) -> str:
        return f"{_component(tool, label='tool')}:{_component(domain, label='domain')}"

    def load(self) -> None:
        try:
            data = json.loads(read_bounded_text(self.path, max_bytes=_MAX_STATE_BYTES))
            if not isinstance(data, dict):
                raise ValueError("active inference state root must be an object")
            raw_models = data.get("models", {})
            if not isinstance(raw_models, dict):
                raise ValueError("active inference models must be an object")
            if len(raw_models) > _MAX_MODELS:
                raise ValueError("active inference state contains too many models")

            models: dict[str, ModelStats] = {}
            for key, raw_stats in islice(raw_models.items(), _MAX_MODELS):
                if (
                    not isinstance(key, str)
                    or not key
                    or len(key) > _MAX_MODEL_KEY_CHARS
                    or not isinstance(raw_stats, dict)
                ):
                    continue
                tool, separator, domain = key.partition(":")
                if not separator:
                    continue
                try:
                    normalized_key = self._key(tool, domain)
                except (TypeError, ValueError):
                    continue
                success_weight = _finite_number(
                    raw_stats.get("success_weight", 0.0),
                    maximum=float(_MAX_COUNTER),
                )
                failure_weight = _finite_number(
                    raw_stats.get("failure_weight", 0.0),
                    maximum=float(_MAX_COUNTER),
                )
                total_count = _bounded_count(raw_stats.get("total_count", 0))
                last_updated = _finite_number(raw_stats.get("last_updated", 0.0))
                avg_duration = _finite_number(
                    raw_stats.get("avg_duration_ms", 0.0),
                    maximum=_MAX_DURATION_MS,
                )
                duration_variance = _finite_number(
                    raw_stats.get("duration_variance", 0.0),
                    maximum=_MAX_DURATION_MS**2,
                )
                error_sum = _finite_number(
                    raw_stats.get("prediction_error_sum", 0.0),
                    maximum=float(_MAX_COUNTER),
                )
                error_count = _bounded_count(raw_stats.get("prediction_error_count", 0))
                if any(
                    value is None
                    for value in (
                        success_weight,
                        failure_weight,
                        total_count,
                        last_updated,
                        avg_duration,
                        duration_variance,
                        error_sum,
                        error_count,
                    )
                ):
                    continue
                assert success_weight is not None
                assert failure_weight is not None
                assert total_count is not None
                assert last_updated is not None
                assert avg_duration is not None
                assert duration_variance is not None
                assert error_sum is not None
                assert error_count is not None
                models[normalized_key] = ModelStats(
                    success_weight=success_weight,
                    failure_weight=failure_weight,
                    total_count=total_count,
                    last_updated=last_updated,
                    avg_duration_ms=avg_duration,
                    duration_variance=duration_variance,
                    prediction_error_sum=error_sum,
                    prediction_error_count=error_count,
                )

            total_surprise = _finite_number(
                data.get("total_surprise", 0.0),
                maximum=float(_MAX_COUNTER),
            )
            total_observations = _bounded_count(data.get("total_observations", 0))
            if total_surprise is None or total_observations is None:
                raise ValueError("active inference aggregate counters are malformed")

            self.models = models
            self.total_surprise = total_surprise
            self.total_observations = total_observations
            log.info(
                "active_inference loaded: %d models, %d observations",
                len(self.models),
                self.total_observations,
            )
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as e:
            log.warning("active_inference load failed: %r", e)
        self._loaded = True

    def save(self) -> None:
        if len(self.models) > _MAX_MODELS:
            retained = sorted(
                self.models.items(),
                key=lambda item: (item[1].last_updated, item[0]),
                reverse=True,
            )[:_MAX_MODELS]
            self.models = dict(retained)
        data = {
            "models": {k: asdict(v) for k, v in self.models.items()},
            "total_surprise": self.total_surprise,
            "total_observations": self.total_observations,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(data, separators=(",", ":"), allow_nan=False)
        if len(serialized.encode("utf-8")) > _MAX_STATE_BYTES:
            raise ValueError(f"active inference state exceeds {_MAX_STATE_BYTES} bytes")
        atomic_write_text(self.path, serialized, mode=0o600)

    def predict_tool_outcome(
        self,
        tool: str,
        domain: str = "general",
        args: dict | None = None,
    ) -> ToolPrediction:
        """Predict the outcome of a tool execution."""
        key = self._key(tool, domain)
        stats = self.models.get(key)

        if stats is None or not stats.is_reliable:
            # Use priors based on tool type
            priors = {
                "read": (0.95, 50),
                "list_dir": (0.98, 20),
                "search_memory": (0.90, 100),
                "exec": (0.80, 500),
                "shell": (0.80, 500),
                "write": (0.85, 100),
                "edit": (0.75, 200),
                "web_search": (0.70, 3000),
                "web_fetch": (0.65, 5000),
            }
            base_success, base_duration = priors.get(tool, (0.75, 1000))
            return ToolPrediction(
                expected_success=base_success,
                expected_duration_ms=base_duration,
                confidence=0.2,
                basis="prior",
            )

        # Use learned statistics
        confidence = min(0.95, 0.3 + stats.total_count * 0.05)
        return ToolPrediction(
            expected_success=stats.success_rate,
            expected_duration_ms=stats.avg_duration_ms,
            confidence=confidence,
            basis="learned",
        )

    def record_observation(
        self,
        tool: str,
        domain: str,
        predicted_success: float,
        actual_success: bool,
        predicted_duration_ms: float = 0,
        actual_duration_ms: float = 0,
        now: float | None = None,
    ) -> Observation:
        """Record an observation and compute prediction error."""
        if not isinstance(actual_success, bool):
            raise TypeError("actual_success must be a boolean")
        predicted = _finite_number(predicted_success, maximum=1.0)
        predicted_duration = _finite_number(
            predicted_duration_ms,
            maximum=_MAX_DURATION_MS,
        )
        actual_duration = _finite_number(actual_duration_ms, maximum=_MAX_DURATION_MS)
        timestamp = _finite_number(time.time() if now is None else now)
        if predicted is None:
            raise ValueError("predicted_success must be between 0 and 1")
        if predicted_duration is None or actual_duration is None:
            raise ValueError("durations must be finite non-negative numbers")
        if timestamp is None:
            raise ValueError("now must be a finite non-negative number")
        now = timestamp
        key = self._key(tool, domain)

        if key not in self.models:
            if len(self.models) >= _MAX_MODELS:
                oldest = min(
                    self.models,
                    key=lambda model_key: (
                        self.models[model_key].last_updated,
                        model_key,
                    ),
                )
                del self.models[oldest]
            self.models[key] = ModelStats()

        pred_error, surprise = self.models[key].record(
            success=actual_success,
            duration_ms=actual_duration,
            predicted_success=predicted,
            now=now,
        )

        duration_error = abs(predicted_duration - actual_duration) if predicted_duration > 0 else 0

        obs = Observation(
            tool=tool,
            domain=domain,
            predicted_success=predicted,
            actual_success=actual_success,
            predicted_duration_ms=predicted_duration,
            actual_duration_ms=actual_duration,
            prediction_error=pred_error,
            duration_error=duration_error,
            timestamp=now,
            surprise=surprise,
        )
        self.recent_observations.append(obs)
        self.total_surprise = min(float(_MAX_COUNTER), self.total_surprise + surprise)
        self.total_observations = min(_MAX_COUNTER, self.total_observations + 1)

        # Log high-surprise events
        if surprise > 2.0:
            log.info(
                "active_inference.high_surprise tool=%s domain=%s surprise=%.2f pred=%.2f actual=%s",
                tool,
                domain,
                surprise,
                predicted,
                actual_success,
            )

        return obs

    def get_recent_errors(self, limit: int = 10) -> list[Observation]:
        """Get recent high-error observations for learning focus."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return []
        sorted_obs = sorted(
            self.recent_observations, key=lambda o: o.prediction_error, reverse=True
        )
        return sorted_obs[: min(limit, len(self.recent_observations))]

    def get_high_surprise_areas(self) -> list[tuple[str, float]]:
        """Tool/domain combos with highest average surprise — areas to learn more about."""
        surprises: dict[str, list[float]] = {}
        for obs in self.recent_observations:
            key = self._key(obs.tool, obs.domain)
            surprises.setdefault(key, []).append(obs.surprise)

        avg_surprises = [(key, sum(s) / len(s)) for key, s in surprises.items() if len(s) >= 2]
        avg_surprises.sort(key=lambda x: x[1], reverse=True)
        return avg_surprises[:5]

    def get_learning_priorities(self) -> list[dict[str, Any]]:
        """Get areas where the agent should focus learning (high prediction error)."""
        priorities: list[dict[str, Any]] = []
        for key, stats in self.models.items():
            if stats.is_reliable and stats.avg_prediction_error > 0.3:
                tool, domain = key.split(":", 1)
                priorities.append(
                    {
                        "tool": tool,
                        "domain": domain,
                        "avg_error": stats.avg_prediction_error,
                        "success_rate": stats.success_rate,
                        "samples": stats.total_count,
                    }
                )
        priorities.sort(key=lambda x: x["avg_error"], reverse=True)
        return priorities[:5]

    def avg_prediction_error(self) -> float:
        """Overall average prediction error across all models."""
        total_error = sum(s.prediction_error_sum for s in self.models.values())
        total_count = sum(s.prediction_error_count for s in self.models.values())
        if total_count == 0:
            return 0.5
        return total_error / total_count

    def avg_surprise(self) -> float:
        """Average surprise per observation."""
        if self.total_observations == 0:
            return 0.0
        return self.total_surprise / self.total_observations

    def summary(self) -> str:
        return (
            f"ActiveInference: {self.total_observations} observations, "
            f"{len(self.models)} models, "
            f"avg_error={self.avg_prediction_error():.3f}, "
            f"avg_surprise={self.avg_surprise():.3f}"
        )
