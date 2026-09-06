"""Prediction Network — prospective retrieval and context pre-fetching.

Predicts likely next topics/questions and pre-fetches relevant memory
before the user even asks. Makes retrieval feel prescient and reduces
the cognitive load on the model by having context ready.

Zero LLM calls. Uses:
  1. Conversation trajectory analysis (what topics are escalating?)
  2. Tool usage patterns (what tools are being used → what comes next?)
  3. Temporal patterns (what time of day → what kind of request?)
  4. Entity co-occurrence (entities mentioned together → likely next entities)
  5. Task transition matrix (coding → testing → deployment)

Usage:
    pn = PredictionNetwork(memory_store, entity_graph)
    pn.load()
    predictions = pn.predict_next(user_request, tool_trace, conversation_history)
    # predictions: [{"topic": "testing", "confidence": 0.8, "entities": ["pytest", "coverage"]}]
    prefetched = pn.prefetch(predictions)
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text, read_bounded_text

log = logging.getLogger("norax.brain.prediction_network")

_MAX_STATE_BYTES = 16 * 1024 * 1024
_MAX_TRANSITION_COUNT = 1_000_000_000


@dataclass
class Prediction:
    topic: str
    confidence: float
    entities: list[str] = field(default_factory=list)
    reasoning: str = ""
    memory_keys: list[str] = field(default_factory=list)


@dataclass
class TransitionStats:
    """Tracks transitions from one task type to another."""

    transitions: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    total: int = 0

    def record(self, from_task: str, to_task: str) -> None:
        self.transitions[from_task][to_task] += 1
        self.total += 1

    def predict(self, from_task: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Given current task, predict likely next tasks."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            return []
        if from_task not in self.transitions:
            return []
        total_from = sum(self.transitions[from_task].values())
        if total_from == 0:
            return []
        ranked = [
            (to_task, count / total_from) for to_task, count in self.transitions[from_task].items()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# Task type classification (simplified — reuses self_model's domain logic)
_TASK_KEYWORDS: dict[str, set[str]] = {
    "coding": {
        "code",
        "implement",
        "function",
        "class",
        "bug",
        "fix",
        "debug",
        "error",
        "handling",
        "python",
        "bash",
        "script",
        "refactor",
        "patch",
        "deploy",
        "compile",
    },
    "testing": {
        "test",
        "pytest",
        "unittest",
        "verify",
        "validation",
        "coverage",
        "assert",
        "mock",
        "fixture",
        "integration",
    },
    "research": {
        "search",
        "find",
        "research",
        "investigate",
        "analyze",
        "compare",
        "study",
        "look up",
        "query",
        "web",
    },
    "memory": {
        "remember",
        "recall",
        "store",
        "consolidate",
        "forget",
        "learn",
        "entity",
        "graph",
        "memory",
    },
    "ops": {
        "deploy",
        "restart",
        "service",
        "systemctl",
        "config",
        "server",
        "port",
        "process",
        "monitor",
        "health",
        "status",
    },
    "planning": {
        "plan",
        "schedule",
        "task",
        "decompose",
        "orchestrate",
        "strategy",
        "prioritize",
        "organize",
        "roadmap",
    },
    "communication": {
        "send",
        "message",
        "notify",
        "alert",
        "discord",
        "channel",
        "reply",
        "chat",
        "respond",
    },
}


def _classify_task(text: str) -> str:
    text_lower = (text or "").lower()
    tokens = set(re.findall(r"[a-z0-9_]+", text_lower))
    # Common inflections should not turn "run tests" or "fix bugs" into an
    # unrelated general task. Keep the original tokens and add a lightweight
    # singular form without introducing a model dependency.
    tokens.update(token[:-1] for token in tuple(tokens) if len(token) > 3 and token.endswith("s"))
    scores: dict[str, int] = {}
    for task, keywords in _TASK_KEYWORDS.items():
        score = sum(
            1
            for keyword in keywords
            if (keyword in text_lower if " " in keyword else keyword in tokens)
        )
        if score > 0:
            scores[task] = score
    if not scores:
        return "general"
    return max(scores, key=lambda k: scores[k])


# Task transition matrix — learned patterns
_DEFAULT_TRANSITIONS = {
    "coding": {"testing": 0.35, "coding": 0.25, "ops": 0.20, "research": 0.10, "planning": 0.10},
    "testing": {"coding": 0.30, "ops": 0.25, "testing": 0.20, "planning": 0.15, "research": 0.10},
    "research": {
        "coding": 0.30,
        "planning": 0.25,
        "research": 0.20,
        "memory": 0.15,
        "communication": 0.10,
    },
    "memory": {"coding": 0.20, "research": 0.20, "planning": 0.20, "memory": 0.20, "ops": 0.20},
    "ops": {"testing": 0.25, "ops": 0.25, "coding": 0.20, "planning": 0.15, "communication": 0.15},
    "planning": {
        "coding": 0.35,
        "ops": 0.20,
        "planning": 0.15,
        "research": 0.15,
        "communication": 0.15,
    },
    "communication": {
        "coding": 0.20,
        "research": 0.20,
        "planning": 0.20,
        "ops": 0.20,
        "communication": 0.20,
    },
}


class PredictionNetwork:
    """Prospective retrieval and context pre-fetching."""

    def __init__(
        self,
        memory_store: Any = None,
        entity_graph: Any = None,
        state_path: Path | str = "~/norax/memory/prediction_network.json",
    ):
        self.memory_store = memory_store
        self.entity_graph = entity_graph
        self.state_path = Path(state_path).expanduser()
        self.transition_stats = TransitionStats()
        self.entity_cooccurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.topic_history: deque[str] = deque(maxlen=20)
        self.last_episode_timestamp: float = 0.0
        self._loaded = False

    def load(self) -> None:
        try:
            raw = read_bounded_text(self.state_path, max_bytes=_MAX_STATE_BYTES)
        except FileNotFoundError:
            self._loaded = True
            return
        except (OSError, UnicodeError, ValueError) as exc:
            log.warning("prediction_network load failed: %r", exc)
            self._loaded = True
            return
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("prediction network state root must be an object")

            transition_stats = TransitionStats()
            transitions = data.get("transitions", {})
            if not isinstance(transitions, dict):
                transitions = {}
            for from_task, to_dict in transitions.items():
                if not isinstance(from_task, str) or not from_task or not isinstance(to_dict, dict):
                    continue
                for to_task, count in to_dict.items():
                    if (
                        not isinstance(to_task, str)
                        or not to_task
                        or isinstance(count, bool)
                        or not isinstance(count, int | float)
                        or not math.isfinite(float(count))
                    ):
                        continue
                    parsed_count = max(0, min(_MAX_TRANSITION_COUNT, int(count)))
                    if parsed_count:
                        transition_stats.transitions[from_task[:80]][to_task[:80]] = parsed_count
                        transition_stats.total += parsed_count

            entity_cooccurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            entity_rows = data.get("entity_cooccurrence", {})
            if not isinstance(entity_rows, dict):
                entity_rows = {}
            for entity, related in entity_rows.items():
                if not isinstance(entity, str) or not entity or not isinstance(related, dict):
                    continue
                for related_entity, count in related.items():
                    if (
                        not isinstance(related_entity, str)
                        or not related_entity
                        or isinstance(count, bool)
                        or not isinstance(count, int | float)
                        or not math.isfinite(float(count))
                    ):
                        continue
                    parsed_count = max(0, min(_MAX_TRANSITION_COUNT, int(count)))
                    if parsed_count:
                        entity_cooccurrence[entity[:160]][related_entity[:160]] = parsed_count

            topic_history: deque[str] = deque(maxlen=20)
            topics = data.get("topic_history", [])
            if isinstance(topics, list):
                topic_history.extend(str(topic)[:80] for topic in topics if isinstance(topic, str))
            timestamp_value = data.get("last_episode_timestamp", 0.0)
            if isinstance(timestamp_value, bool) or not isinstance(timestamp_value, int | float):
                timestamp = 0.0
            else:
                timestamp = float(timestamp_value)
                if not math.isfinite(timestamp) or timestamp < 0:
                    timestamp = 0.0

            self.transition_stats = transition_stats
            self.entity_cooccurrence = entity_cooccurrence
            self.topic_history = topic_history
            self.last_episode_timestamp = timestamp
            log.info(
                "prediction_network loaded: %d transitions, %d entities",
                self.transition_stats.total,
                len(self.entity_cooccurrence),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("prediction_network load failed: %r", exc)
        self._loaded = True

    def save(self) -> None:
        data = {
            "transitions": {k: dict(v) for k, v in self.transition_stats.transitions.items()},
            "entity_cooccurrence": {k: dict(v) for k, v in self.entity_cooccurrence.items()},
            "topic_history": list(self.topic_history),
            "last_episode_timestamp": self.last_episode_timestamp,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.state_path, json.dumps(data, indent=2), mode=0o600)

    def record_turn(
        self,
        user_request: str,
        task_type: str = "",
        entities: list[str] | None = None,
    ) -> None:
        """Record a completed turn for learning."""
        task = task_type or _classify_task(user_request)
        # Record transition from previous task
        if self.topic_history:
            prev_task = self.topic_history[-1]
            if prev_task != task:
                self.transition_stats.record(prev_task, task)
        self.topic_history.append(task)
        # Record entity co-occurrence
        normalized_entities = list(
            dict.fromkeys(
                entity.strip()[:160]
                for entity in (entities or [])
                if isinstance(entity, str) and entity.strip()
            )
        )
        if len(normalized_entities) >= 2:
            for i, e1 in enumerate(normalized_entities):
                for e2 in normalized_entities[i + 1 :]:
                    self.entity_cooccurrence[e1][e2] += 1
                    self.entity_cooccurrence[e2][e1] += 1

    def record_episodes(self, episodes: list[Any]) -> int:
        """Record only episodes newer than the durable replay watermark."""
        unseen: list[tuple[float, Any]] = []
        for episode in episodes:
            raw_timestamp = getattr(episode, "timestamp", 0.0)
            if isinstance(raw_timestamp, bool):
                continue
            try:
                timestamp = float(raw_timestamp or 0.0)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(timestamp) and timestamp > self.last_episode_timestamp:
                unseen.append((timestamp, episode))
        unseen.sort(key=lambda pair: pair[0])
        for _, episode in unseen:
            self.record_turn(
                getattr(episode, "user_input", "") or "",
                task_type=getattr(episode, "task_type", "") or "",
            )
        if unseen:
            self.last_episode_timestamp = unseen[-1][0]
        return len(unseen)

    def predict_next(
        self,
        user_request: str = "",
        tool_trace: list[dict] | None = None,
        conversation_history: list[dict] | None = None,
        top_k: int = 3,
    ) -> list[Prediction]:
        """Predict likely next topics and pre-fetchable entities."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            return []
        predictions: list[Prediction] = []
        tool_trace = tool_trace or []
        conversation_history = conversation_history or []

        # 1. Task transition prediction
        current_task = (
            _classify_task(user_request)
            if user_request
            else (self.topic_history[-1] if self.topic_history else "general")
        )

        # Use learned transitions if we have enough data, otherwise use defaults
        if self.transition_stats.total >= 10 and current_task in self.transition_stats.transitions:
            transitions = self.transition_stats.predict(current_task, top_k)
        else:
            # Use default transition matrix
            defaults = _DEFAULT_TRANSITIONS.get(current_task, {})
            transitions = sorted(defaults.items(), key=lambda x: x[1], reverse=True)[:top_k]

        for next_task, prob in transitions:
            predictions.append(
                Prediction(
                    topic=next_task,
                    confidence=prob,
                    reasoning=f"task_transition: {current_task}→{next_task}",
                )
            )

        # 2. Entity co-occurrence prediction
        # Extract entities from current request
        current_entities = self._extract_entities(user_request, tool_trace)
        if current_entities and self.entity_cooccurrence:
            for entity in current_entities[:3]:
                if entity in self.entity_cooccurrence:
                    related = sorted(
                        self.entity_cooccurrence[entity].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    for related_entity, count in related:
                        cooccurrence_prob = min(1.0, count / 10)
                        predictions.append(
                            Prediction(
                                topic=f"entity:{related_entity}",
                                confidence=cooccurrence_prob
                                * 0.6,  # lower confidence than task transitions
                                entities=[related_entity],
                                reasoning=f"co-occurrence: {entity}→{related_entity} ({count}x)",
                            )
                        )

        # 3. Tool pattern prediction
        if tool_trace:
            tool_names = [t.get("name", "") for t in tool_trace if isinstance(t, dict)]
            # If we see read+edit, predict testing next
            if (
                "read" in tool_names
                and "edit" in tool_names
                and "testing" not in [p.topic for p in predictions]
            ):
                predictions.append(
                    Prediction(
                        topic="testing",
                        confidence=0.5,
                        reasoning="tool_pattern: read+edit→testing",
                    )
                )
            # If we see web_search, predict coding next
            if "web_search" in tool_names and "coding" not in [p.topic for p in predictions]:
                predictions.append(
                    Prediction(
                        topic="coding",
                        confidence=0.4,
                        reasoning="tool_pattern: web_search→coding",
                    )
                )

        # 4. Conversation trajectory analysis
        if len(conversation_history) >= 3:
            recent_tasks = [
                _classify_task(m.get("content", "") if isinstance(m, dict) else str(m))
                for m in conversation_history[-3:]
            ]
            # If we're in a sequence of the same task, likely to continue
            if len(set(recent_tasks)) == 1 and recent_tasks[0] != "general":
                predictions.append(
                    Prediction(
                        topic=recent_tasks[0],
                        confidence=0.7,
                        reasoning=f"trajectory: 3 consecutive {recent_tasks[0]} turns",
                    )
                )

        # Sort by confidence and deduplicate
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        seen: set[str] = set()
        unique: list[Prediction] = []
        for p in predictions:
            if p.topic not in seen:
                seen.add(p.topic)
                unique.append(p)
        return unique[:top_k]

    def prefetch(self, predictions: list[Prediction]) -> dict[str, Any]:
        """Pre-fetch memory for predicted topics."""
        results: dict[str, Any] = {}
        if not self.memory_store:
            return results

        for pred in predictions:
            try:
                # Search memory for predicted topic
                topic = pred.topic.replace("entity:", "")
                # Use memory store's search if available
                if hasattr(self.memory_store, "search"):
                    hits = self.memory_store.search(topic, k=3)
                    if hits:
                        results[pred.topic] = hits
                elif hasattr(self.memory_store, "semantic"):
                    # Fallback: scan semantic memory for topic keywords
                    hits = []
                    for line in self.memory_store.semantic:
                        if isinstance(line, str) and topic.lower() in line.lower():
                            hits.append(line)
                    if hits:
                        results[pred.topic] = hits[:3]
            except Exception as e:
                log.debug("prefetch failed for %s: %r", pred.topic, e)
        return results

    def _extract_entities(self, text: str, tool_trace: list[dict]) -> list[str]:
        """Extract entity-like terms from text and tool trace."""
        entities: list[str] = []
        # File paths
        for m in re.finditer(r"\b([a-z_][a-z0-9_]*\.(?:py|js|ts))\b", text, re.I):
            entities.append(m.group(1))
        # Capitalized terms (likely proper nouns / entity names)
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
            entities.append(m.group(1))
        # Tool names from trace
        for t in tool_trace:
            if not isinstance(t, dict):
                continue
            name = t.get("name", "")
            if isinstance(name, str) and name:
                entities.append(name[:160])
        return list(dict.fromkeys(entities))  # dedupe preserving order

    def summary(self) -> str:
        return (
            f"PredictionNetwork: {self.transition_stats.total} transitions recorded, "
            f"{len(self.entity_cooccurrence)} entities tracked, "
            f"topic_history={list(self.topic_history)[-5:]}"
        )
