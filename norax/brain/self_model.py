"""Self-Model — persistent capability profile.

Tracks what the agent is good/bad at across sessions. Enables:
  - Routing: avoid tools/domains where success rate is low
  - Escalation: know when to ask for help vs. push through
  - Improvement: identify weak areas for skill mining
  - Confidence: calibrate output confidence based on track record

Zero LLM calls. Pure statistical tracking with exponential decay
(recent performance matters more than ancient history).

Usage:
    sm = SelfModel(Path("~/norax/memory/self_model.json"))
    sm.load()
    sm.record_outcome("exec", "coding", success=True, duration_ms=120)
    sm.record_outcome("web_search", "research", success=False, duration_ms=5000)
    profile = sm.get_profile()
    # profile.tool_success["exec"] -> 0.92
    # profile.domain_success["coding"] -> 0.88
    # profile.confidence("exec", "coding") -> 0.90
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from ..atomic import atomic_write_text, read_bounded_text

log = logging.getLogger("norax.brain.self_model")

DECAY_HALF_LIFE = 7 * 24 * 3600  # 7 days — recent performance weighted higher
MIN_SAMPLES = 3  # need at least 3 outcomes before trusting stats
CONFIDENCE_FLOOR = 0.3  # never go below 30% confidence even with failures
CONFIDENCE_CEIL = 0.95  # never claim 100% confidence
_MAX_STATE_BYTES = 4 * 1024 * 1024
_MAX_STATS_PER_CATEGORY = 2_048
_MAX_OUTCOMES_PER_TURN = 256
_MAX_LABEL_CHARS = 128
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


def _label(value: object, *, field_name: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = " ".join(value.split())[:_MAX_LABEL_CHARS]
    if required and not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _skill_stats_from_raw(value: object) -> SkillStats | None:
    if not isinstance(value, dict):
        return None
    success_weight = _finite_number(
        value.get("success_weight", 0.0),
        maximum=float(_MAX_COUNTER),
    )
    failure_weight = _finite_number(
        value.get("failure_weight", 0.0),
        maximum=float(_MAX_COUNTER),
    )
    total_count = _bounded_count(value.get("total_count", 0))
    last_updated = _finite_number(value.get("last_updated", 0.0))
    avg_duration = _finite_number(
        value.get("avg_duration_ms", 0.0),
        maximum=_MAX_DURATION_MS,
    )
    raw_best = value.get("best_duration_ms")
    best_duration: float | None
    if raw_best is None or raw_best == float("inf"):
        best_duration = float("inf")
    else:
        best_duration = _finite_number(raw_best, maximum=_MAX_DURATION_MS)
    worst_duration = _finite_number(
        value.get("worst_duration_ms", 0.0),
        maximum=_MAX_DURATION_MS,
    )
    if any(
        field is None
        for field in (
            success_weight,
            failure_weight,
            total_count,
            last_updated,
            avg_duration,
            best_duration,
            worst_duration,
        )
    ):
        return None
    assert success_weight is not None
    assert failure_weight is not None
    assert total_count is not None
    assert last_updated is not None
    assert avg_duration is not None
    assert best_duration is not None
    assert worst_duration is not None
    return SkillStats(
        success_weight=success_weight,
        failure_weight=failure_weight,
        total_count=total_count,
        last_updated=last_updated,
        avg_duration_ms=avg_duration,
        best_duration_ms=best_duration,
        worst_duration_ms=worst_duration,
    )


def _skill_stats_payload(stats: SkillStats) -> dict[str, float | int | None]:
    return {
        "success_weight": stats.success_weight,
        "failure_weight": stats.failure_weight,
        "total_count": stats.total_count,
        "last_updated": stats.last_updated,
        "avg_duration_ms": stats.avg_duration_ms,
        "best_duration_ms": (
            stats.best_duration_ms if math.isfinite(stats.best_duration_ms) else None
        ),
        "worst_duration_ms": stats.worst_duration_ms,
    }


@dataclass
class SkillStats:
    """Exponentially-decayed statistics for a tool or domain."""

    success_weight: float = 0.0  # decayed sum of successes
    failure_weight: float = 0.0  # decayed sum of failures
    total_count: int = 0  # raw count (for sample size checks)
    last_updated: float = 0.0
    avg_duration_ms: float = 0.0
    best_duration_ms: float = float("inf")
    worst_duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_weight + self.failure_weight
        if total < 0.01:
            return 0.5  # no data → neutral
        return self.success_weight / total

    @property
    def sample_size(self) -> int:
        return self.total_count

    @property
    def is_reliable(self) -> bool:
        return self.total_count >= MIN_SAMPLES

    def record(self, success: bool, duration_ms: float = 0.0, now: float | None = None) -> None:
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        duration = _finite_number(duration_ms, maximum=_MAX_DURATION_MS)
        timestamp = _finite_number(time.time() if now is None else now)
        if duration is None:
            raise ValueError("duration_ms must be a finite non-negative number")
        if timestamp is None:
            raise ValueError("now must be a finite non-negative number")
        now = timestamp
        # Apply decay to existing weights
        self._decay(now)
        # Add new observation
        weight = 1.0
        if success:
            self.success_weight = min(float(_MAX_COUNTER), self.success_weight + weight)
        else:
            self.failure_weight = min(float(_MAX_COUNTER), self.failure_weight + weight)
        self.total_count = min(_MAX_COUNTER, self.total_count + 1)
        self.last_updated = now
        # Update duration stats
        if duration > 0:
            if self.total_count == 1:
                self.avg_duration_ms = duration
            else:
                # Running average with slight recency bias
                self.avg_duration_ms = 0.9 * self.avg_duration_ms + 0.1 * duration
            self.best_duration_ms = min(self.best_duration_ms, duration)
            self.worst_duration_ms = max(self.worst_duration_ms, duration)

    def _decay(self, now: float) -> None:
        if self.last_updated == 0:
            return
        elapsed = now - self.last_updated
        if elapsed <= 0:
            return
        decay_factor = 0.5 ** (elapsed / DECAY_HALF_LIFE)
        self.success_weight *= decay_factor
        self.failure_weight *= decay_factor


@dataclass
class SelfProfile:
    """Snapshot of the agent's self-assessed capabilities."""

    tool_stats: dict[str, SkillStats] = field(default_factory=dict)
    domain_stats: dict[str, SkillStats] = field(default_factory=dict)
    task_type_stats: dict[str, SkillStats] = field(default_factory=dict)
    total_turns: int = 0
    total_successes: int = 0
    total_failures: int = 0
    last_updated: float = 0.0

    @property
    def overall_success_rate(self) -> float:
        total = self.total_successes + self.total_failures
        if total == 0:
            return 0.5
        return self.total_successes / total

    def tool_success_rate(self, tool: str) -> float:
        s = self.tool_stats.get(tool)
        return s.success_rate if s else 0.5

    def domain_success_rate(self, domain: str) -> float:
        s = self.domain_stats.get(domain)
        return s.success_rate if s else 0.5

    def confidence(self, tool: str = "", domain: str = "", task_type: str = "") -> float:
        """Compute confidence for a given tool/domain/task combination."""
        confidences = []
        if tool and tool in self.tool_stats:
            s = self.tool_stats[tool]
            if s.is_reliable:
                confidences.append(s.success_rate)
        if domain and domain in self.domain_stats:
            s = self.domain_stats[domain]
            if s.is_reliable:
                confidences.append(s.success_rate)
        if task_type and task_type in self.task_type_stats:
            s = self.task_type_stats[task_type]
            if s.is_reliable:
                confidences.append(s.success_rate)
        if not confidences:
            return 0.5  # no data → neutral confidence
        avg = sum(confidences) / len(confidences)
        return max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEIL, avg))

    def weak_tools(self, min_samples: int = MIN_SAMPLES) -> list[tuple[str, float]]:
        """Tools with success rate below 0.6."""
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            return []
        return [
            (name, s.success_rate)
            for name, s in self.tool_stats.items()
            if s.sample_size >= min_samples and s.success_rate < 0.6
        ]

    def strong_tools(self, min_samples: int = MIN_SAMPLES) -> list[tuple[str, float]]:
        """Tools with success rate above 0.8."""
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            return []
        return [
            (name, s.success_rate)
            for name, s in self.tool_stats.items()
            if s.sample_size >= min_samples and s.success_rate > 0.8
        ]

    def routing_hint(self, tool: str, domain: str = "") -> str:
        """Generate a routing hint for the orchestrator."""
        tool_rate = self.tool_success_rate(tool)
        domain_rate = self.domain_success_rate(domain) if domain else 0.5
        combined = (tool_rate + domain_rate) / 2 if domain else tool_rate
        if combined < 0.4:
            return "escalate"  # use stronger model
        elif combined < 0.6:
            return "cautious"  # use current model but with verification
        else:
            return "normal"  # proceed normally


# Domain classification keywords
_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "coding": {
        "exec",
        "shell",
        "read",
        "write",
        "edit",
        "list_dir",
        "write_chunk",
        "code",
        "python",
        "bash",
        "function",
        "class",
        "bug",
        "fix",
        "implement",
        "debug",
        "compile",
        "syntax",
        "refactor",
        "patch",
        "deploy",
    },
    "research": {
        "web_search",
        "web_fetch",
        "search_memory",
        "search",
        "find",
        "research",
        "investigate",
        "analyze",
        "compare",
        "study",
        "look up",
        "query",
    },
    "memory": {
        "search_memory",
        "append_memory",
        "read",
        "remember",
        "recall",
        "store",
        "consolidate",
        "forget",
        "learn",
        "memory",
        "entity",
        "graph",
    },
    "communication": {
        "message_send",
        "send",
        "reply",
        "notify",
        "alert",
        "discord",
        "channel",
        "user",
        "message",
        "chat",
        "respond",
    },
    "ops": {
        "exec",
        "shell",
        "status",
        "systemctl",
        "service",
        "deploy",
        "restart",
        "config",
        "gateway",
        "server",
        "port",
        "process",
        "monitor",
        "health",
    },
    "planning": {
        "schedule_reminder",
        "plan",
        "schedule",
        "task",
        "decompose",
        "orchestrate",
        "coordinate",
        "strategy",
        "prioritize",
        "organize",
    },
}


def classify_domain(user_request: str, tool_trace: list[dict] | None = None) -> str:
    """Classify the request into a domain based on keywords and tool usage."""
    text = str(user_request or "")[:32_768].lower()
    tool_trace = tool_trace or []
    tool_names = [
        name.lower()
        for item in tool_trace[:_MAX_OUTCOMES_PER_TURN]
        if isinstance(item, dict) and isinstance((name := item.get("name")), str)
    ]

    scores: dict[str, float] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            if kw in text:
                score += 1.0
            if kw in tool_names:
                score += 0.5
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"
    return max(scores, key=lambda k: scores[k])


class SelfModel:
    """Persistent self-capability model with exponential decay."""

    def __init__(self, path: Path | str = "~/norax/memory/self_model.json"):
        self.path = Path(path).expanduser()
        self.profile = SelfProfile()
        self._loaded = False

    def load(self) -> None:
        try:
            data = json.loads(read_bounded_text(self.path, max_bytes=_MAX_STATE_BYTES))
            if not isinstance(data, dict):
                raise ValueError("self model state root must be an object")
            successes = _bounded_count(data.get("total_successes", 0))
            failures = _bounded_count(data.get("total_failures", 0))
            last_updated = _finite_number(data.get("last_updated", 0.0))
            if successes is None or failures is None or last_updated is None:
                raise ValueError("self model aggregate fields are malformed")
            loaded_categories: dict[str, dict[str, SkillStats]] = {}
            for key in ("tool_stats", "domain_stats", "task_type_stats"):
                raw_category = data.get(key, {})
                if not isinstance(raw_category, dict):
                    raise ValueError(f"self model {key} must be an object")
                if len(raw_category) > _MAX_STATS_PER_CATEGORY:
                    raise ValueError(f"self model {key} contains too many entries")
                stats: dict[str, SkillStats] = {}
                for raw_name, raw_stats in islice(
                    raw_category.items(),
                    _MAX_STATS_PER_CATEGORY,
                ):
                    try:
                        name = _label(raw_name, field_name=key)
                    except (TypeError, ValueError):
                        continue
                    parsed = _skill_stats_from_raw(raw_stats)
                    if parsed is not None:
                        stats[name] = parsed
                loaded_categories[key] = stats

            self.profile = SelfProfile(
                tool_stats=loaded_categories["tool_stats"],
                domain_stats=loaded_categories["domain_stats"],
                task_type_stats=loaded_categories["task_type_stats"],
                total_turns=successes + failures,
                total_successes=successes,
                total_failures=failures,
                last_updated=last_updated,
            )
            log.info(
                "self_model loaded: %d turns, %d tools tracked",
                self.profile.total_turns,
                len(self.profile.tool_stats),
            )
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as e:
            log.warning("self_model load failed: %r, preserving current state", e)
        self._loaded = True

    def save(self) -> None:
        self.profile.last_updated = time.time()
        for category_name in ("tool_stats", "domain_stats", "task_type_stats"):
            category: dict[str, SkillStats] = getattr(self.profile, category_name)
            if len(category) > _MAX_STATS_PER_CATEGORY:
                retained = sorted(
                    category.items(),
                    key=lambda item: (item[1].last_updated, item[0]),
                    reverse=True,
                )[:_MAX_STATS_PER_CATEGORY]
                setattr(self.profile, category_name, dict(retained))
        data = {
            "total_turns": self.profile.total_turns,
            "total_successes": self.profile.total_successes,
            "total_failures": self.profile.total_failures,
            "last_updated": self.profile.last_updated,
            "tool_stats": {
                key: _skill_stats_payload(stats) for key, stats in self.profile.tool_stats.items()
            },
            "domain_stats": {
                key: _skill_stats_payload(stats) for key, stats in self.profile.domain_stats.items()
            },
            "task_type_stats": {
                key: _skill_stats_payload(stats)
                for key, stats in self.profile.task_type_stats.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(data, separators=(",", ":"), allow_nan=False)
        if len(serialized.encode("utf-8")) > _MAX_STATE_BYTES:
            raise ValueError(f"self model state exceeds {_MAX_STATE_BYTES} bytes")
        atomic_write_text(self.path, serialized, mode=0o600)

    def record_outcome(
        self,
        tool: str,
        domain: str,
        success: bool,
        duration_ms: float = 0.0,
        task_type: str = "",
        now: float | None = None,
    ) -> None:
        """Record one standalone outcome as one turn (compatibility API)."""
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        timestamp = _finite_number(time.time() if now is None else now)
        duration = _finite_number(duration_ms, maximum=_MAX_DURATION_MS)
        if timestamp is None:
            raise ValueError("now must be a finite non-negative number")
        if duration is None:
            raise ValueError("duration_ms must be a finite non-negative number")
        self._record_dimensions(
            tool=_label(tool, field_name="tool"),
            domain=_label(domain, field_name="domain"),
            success=success,
            duration_ms=duration,
            task_type=_label(task_type, field_name="task_type", required=False),
            now=timestamp,
        )
        self._record_global_turn(success)

    def record_turn_outcomes(
        self,
        outcomes: list[tuple[str, bool, float]],
        *,
        domain: str,
        task_type: str = "",
        turn_success: bool,
        now: float | None = None,
    ) -> None:
        """Record many tool observations while counting one completed turn."""
        if not isinstance(outcomes, list):
            raise TypeError("outcomes must be a list")
        if not outcomes:
            return
        if not isinstance(turn_success, bool):
            raise TypeError("turn_success must be a boolean")
        observed_at = _finite_number(time.time() if now is None else now)
        if observed_at is None:
            raise ValueError("now must be a finite non-negative number")
        normalized_domain = _label(domain, field_name="domain")
        normalized_task_type = _label(task_type, field_name="task_type", required=False)
        normalized_outcomes: list[tuple[str, bool, float]] = []
        for item in outcomes[:_MAX_OUTCOMES_PER_TURN]:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("each outcome must be a (tool, success, duration_ms) tuple")
            tool, success, duration_ms = item
            if not isinstance(success, bool):
                raise TypeError("outcome success must be a boolean")
            duration = _finite_number(duration_ms, maximum=_MAX_DURATION_MS)
            if duration is None:
                raise ValueError("duration_ms must be a finite non-negative number")
            normalized_outcomes.append((_label(tool, field_name="tool"), success, duration))
        for tool, success, duration_ms in normalized_outcomes:
            self._record_dimensions(
                tool=tool,
                domain=normalized_domain,
                success=success,
                duration_ms=duration_ms,
                task_type=normalized_task_type,
                now=observed_at,
            )
        self._record_global_turn(turn_success)

    def _record_dimensions(
        self,
        *,
        tool: str,
        domain: str,
        success: bool,
        duration_ms: float,
        task_type: str,
        now: float,
    ) -> None:
        # Tool stats
        if tool not in self.profile.tool_stats:
            self._make_room(self.profile.tool_stats)
            self.profile.tool_stats[tool] = SkillStats()
        self.profile.tool_stats[tool].record(success, duration_ms, now)
        # Domain stats
        if domain not in self.profile.domain_stats:
            self._make_room(self.profile.domain_stats)
            self.profile.domain_stats[domain] = SkillStats()
        self.profile.domain_stats[domain].record(success, duration_ms, now)
        # Task type stats
        if task_type:
            if task_type not in self.profile.task_type_stats:
                self._make_room(self.profile.task_type_stats)
                self.profile.task_type_stats[task_type] = SkillStats()
            self.profile.task_type_stats[task_type].record(success, duration_ms, now)

    @staticmethod
    def _make_room(category: dict[str, SkillStats]) -> None:
        if len(category) < _MAX_STATS_PER_CATEGORY:
            return
        oldest = min(
            category,
            key=lambda name: (category[name].last_updated, name),
        )
        del category[oldest]

    def _record_global_turn(self, success: bool) -> None:
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        self.profile.total_turns = min(_MAX_COUNTER, self.profile.total_turns + 1)
        if success:
            self.profile.total_successes = min(
                _MAX_COUNTER,
                self.profile.total_successes + 1,
            )
        else:
            self.profile.total_failures = min(
                _MAX_COUNTER,
                self.profile.total_failures + 1,
            )

    def get_profile(self) -> SelfProfile:
        return self.profile

    def confidence(self, tool: str = "", domain: str = "", task_type: str = "") -> float:
        return self.profile.confidence(tool, domain, task_type)

    def routing_hint(self, tool: str, domain: str = "") -> str:
        return self.profile.routing_hint(tool, domain)

    def weak_areas(self) -> list[tuple[str, float]]:
        """Return weak tools and domains."""
        return self.profile.weak_tools()

    def strong_areas(self) -> list[tuple[str, float]]:
        """Return strong tools and domains."""
        return self.profile.strong_tools()

    def summary(self) -> str:
        p = self.profile
        weak = p.weak_tools()
        strong = p.strong_tools()
        return (
            f"SelfModel: {p.total_turns} turns, "
            f"success_rate={p.overall_success_rate:.1%}, "
            f"tools_tracked={len(p.tool_stats)}, "
            f"strong={len(strong)}, weak={len(weak)}"
        )
