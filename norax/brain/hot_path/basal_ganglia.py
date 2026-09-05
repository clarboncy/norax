"""Persistent advisory history for action outcomes.

The historical ``gate()`` API returns a recommendation; it is not an
authorization boundary and the runtime never blocks a requested action from
this heuristic. Tool identity/risk checks remain authoritative. Runtime
outcomes are recorded from actual tool results, not prose quality.
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

log = logging.getLogger("norax.brain.basal_ganglia")

Decision = Literal["go", "nogo", "habit", "explore"]

# Thresholds
HABIT_THRESHOLD = 5  # consecutive successes to become habit
NOGO_THRESHOLD = 3.0  # expected reward below this → nogo
EXPLORE_THRESHOLD = 2  # fewer than this many outcomes → explore
MAX_ACTION_RECORDS = 512
_MAX_STATE_BYTES = 1_048_576
_MAX_ACTION_CHARS = 80
_MAX_OUTCOME_COUNT = 1_000_000_000


@dataclass
class GateResult:
    decision: Decision
    action_key: str
    expected_reward: float
    confidence: float  # 0-1
    history_count: int
    reason: str


@dataclass
class ActionRecord:
    key: str
    total_positive: int = 0
    total_negative: int = 0
    consecutive_success: int = 0
    expected_reward: float = 5.0  # neutral prior
    last_score: float = 0.0
    last_time: float = 0.0

    @property
    def total(self) -> int:
        return self.total_positive + self.total_negative

    @property
    def is_habit(self) -> bool:
        return self.consecutive_success >= HABIT_THRESHOLD

    @property
    def is_nogo(self) -> bool:
        return self.expected_reward < NOGO_THRESHOLD and self.total >= EXPLORE_THRESHOLD


@dataclass
class BasalGanglia:
    """Advisory success/failure history with bounded persistence."""

    state_path: Path | None = None
    _actions: dict[str, ActionRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_path and self.state_path.exists():
            self._load()

    def _normalize_key(self, action: str) -> str:
        if not isinstance(action, str):
            raise TypeError("action must be a string")
        return action.strip().lower()[:_MAX_ACTION_CHARS]

    def _load(self) -> None:
        if not self.state_path:
            return
        try:
            data = json.loads(read_bounded_text(self.state_path, max_bytes=_MAX_STATE_BYTES))
            if not isinstance(data, dict):
                raise ValueError("basal ganglia state root must be an object")
            actions = data.get("actions", {})
            if not isinstance(actions, dict):
                raise ValueError("actions must be an object")
            loaded: dict[str, ActionRecord] = {}
            for k, v in islice(actions.items(), MAX_ACTION_RECORDS * 4):
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                key = self._normalize_key(k)
                if not key:
                    continue
                counts = (
                    v.get("total_positive", 0),
                    v.get("total_negative", 0),
                    v.get("consecutive_success", 0),
                )
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in counts
                ):
                    continue
                positive, negative, consecutive = (
                    min(value, _MAX_OUTCOME_COUNT) for value in counts
                )
                numeric = (
                    v.get("expected_reward", 5.0),
                    v.get("last_score", 0.0),
                    v.get("last_time", 0.0),
                )
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in numeric
                ):
                    continue
                reward, last_score, last_time = (float(value) for value in numeric)
                if not all(math.isfinite(value) for value in (reward, last_score, last_time)):
                    continue
                loaded[key] = ActionRecord(
                    key=key,
                    total_positive=positive,
                    total_negative=negative,
                    consecutive_success=min(consecutive, positive),
                    expected_reward=min(10.0, max(0.0, reward)),
                    last_score=min(10.0, max(0.0, last_score)),
                    last_time=max(0.0, last_time),
                )
                if len(loaded) >= MAX_ACTION_RECORDS:
                    break
            self._actions = loaded
        except Exception:
            log.warning("basal_ganglia.load_failed", exc_info=True)

    def _save(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {"actions": {}}
            for k, a in self._actions.items():
                data["actions"][k] = {
                    "total_positive": a.total_positive,
                    "total_negative": a.total_negative,
                    "consecutive_success": a.consecutive_success,
                    "expected_reward": a.expected_reward,
                    "last_score": a.last_score,
                    "last_time": a.last_time,
                }
            atomic_write_text(
                self.state_path,
                json.dumps(data, indent=2, allow_nan=False),
                mode=0o600,
            )
        except Exception:
            log.warning("basal_ganglia.save_failed", exc_info=True)

    def gate(self, action: str) -> GateResult:
        """Return an advisory history classification for an action."""
        key = self._normalize_key(action)
        rec = self._actions.get(key)

        if rec is None or rec.total < EXPLORE_THRESHOLD:
            return GateResult(
                decision="explore",
                action_key=key,
                expected_reward=rec.expected_reward if rec else 5.0,
                confidence=0.15 * rec.total if rec else 0.0,
                history_count=rec.total if rec else 0,
                reason="unknown action — proceed with monitoring",
            )

        if rec.is_nogo:
            return GateResult(
                decision="nogo",
                action_key=key,
                expected_reward=rec.expected_reward,
                confidence=min(1.0, rec.total / 10),
                history_count=rec.total,
                reason=f"negative history: {rec.total_positive}+ / {rec.total_negative}-",
            )

        if rec.is_habit:
            return GateResult(
                decision="habit",
                action_key=key,
                expected_reward=rec.expected_reward,
                confidence=min(1.0, rec.consecutive_success / 10),
                history_count=rec.total,
                reason=f"habit: {rec.consecutive_success} consecutive successes",
            )

        return GateResult(
            decision="go",
            action_key=key,
            expected_reward=rec.expected_reward,
            confidence=min(1.0, rec.total / 10),
            history_count=rec.total,
            reason=f"positive history: reward={rec.expected_reward:.1f}",
        )

    def record_outcome(self, action: str, score: float) -> None:
        """Record an action outcome. Score 0-10 (5=neutral)."""
        self._record_outcome(action, score)
        self._prune()
        self._save()

    def record_outcomes(self, outcomes: list[tuple[str, float]]) -> None:
        """Record a batch and persist once (the runtime's per-turn path)."""
        if not outcomes:
            return
        for action, score in outcomes:
            self._record_outcome(action, score)
        self._prune()
        self._save()

    def _record_outcome(self, action: str, score: float) -> None:
        try:
            value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError("score must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError("score must be a finite number")
        value = min(10.0, max(0.0, value))
        key = self._normalize_key(action)
        if not key:
            raise ValueError("action must not be empty")
        rec = self._actions.get(key)
        if rec is None:
            rec = ActionRecord(key=key)
            self._actions[key] = rec

        if value >= 5:
            rec.total_positive += 1
            rec.consecutive_success += 1
        else:
            rec.total_negative += 1
            rec.consecutive_success = 0

        # Exponential moving average for expected reward
        alpha = 0.3
        rec.expected_reward = alpha * value + (1 - alpha) * rec.expected_reward
        rec.last_score = value
        rec.last_time = time.time()

    def _prune(self) -> None:
        overflow = len(self._actions) - MAX_ACTION_RECORDS
        if overflow <= 0:
            return
        oldest = sorted(self._actions.values(), key=lambda record: record.last_time)[:overflow]
        for record in oldest:
            self._actions.pop(record.key, None)
