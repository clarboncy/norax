"""Analogy Engine — "this problem is like that solved problem."

Given a new task, finds lexically similar successful past requests with a
task-type bonus and surfaces their compact tool sequence. The current request
has no executed tool shape yet, so this module deliberately does not claim
structural similarity before a plan or trace exists.

Zero LLM calls. Uses the episodic store's recorded episodes: each has a
goal, tool trace summary, and outcome. We index compact structural
sketches and rank by sketch overlap + outcome quality.

Usage:
    ae = AnalogyEngine(memory_root=Path("~/norax/memory"), episodic=eps)
    analogs = ae.find_analogies(user_text, task_type="ops", limit=3)
    # -> [Analogy(summary=..., approach=..., similarity=0.71, outcome=...)]
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("norax.brain.analogy_engine")

MAX_INDEX = 400  # keep the sketch index bounded
MIN_SIMILARITY = 0.25
REBUILD_TTL = 1800  # rebuild the index at most every 30 min

_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "it",
    "is",
    "are",
    "be",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "my",
    "your",
    "our",
    "me",
    "him",
    "her",
    "them",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOP and len(w) > 2}


def _tool_shape(trace: Any) -> tuple[str, ...]:
    """Abstract a tool trace into its shape: sequence of tool *kinds*."""
    if not trace:
        return ()
    shape: list[str] = []
    for step in trace[:24]:
        name = ""
        if isinstance(step, dict):
            name = str(step.get("name") or step.get("tool") or "")
        else:
            name = str(getattr(step, "name", "") or getattr(step, "tool", ""))
        if not name:
            continue
        # collapse repeats: read read read -> read
        if not shape or shape[-1] != name:
            shape.append(name)
    return tuple(shape)


def _shape_overlap(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Order-aware overlap: shared adjacent pairs weighted higher."""
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    base = len(set_a & set_b) / max(1, len(set_a | set_b))
    pairs_a = set(zip(a, a[1:], strict=False))
    pairs_b = set(zip(b, b[1:], strict=False))
    pair_bonus = 0.0
    if pairs_a and pairs_b:
        pair_bonus = len(pairs_a & pairs_b) / max(1, len(pairs_a | pairs_b))
    return min(1.0, base * 0.6 + pair_bonus * 0.4)


@dataclass
class EpisodeSketch:
    goal: str
    words: set[str] = field(default_factory=set)
    shape: tuple[str, ...] = ()
    task_type: str = ""
    outcome_success: bool = False
    approach: str = ""  # bounded tool-sequence summary; never prior model prose
    ts: float = 0.0


@dataclass
class Analogy:
    summary: str
    approach: str
    similarity: float
    task_type: str
    outcome_success: bool


class AnalogyEngine:
    """Structural-similarity retrieval over past episodes."""

    def __init__(self, memory_root: Path | str | None = None, episodic: Any = None):
        self.memory_root = Path(memory_root).expanduser() if memory_root else None
        self.episodic = episodic
        self._index: list[EpisodeSketch] = []
        self._built_at: float = 0.0

    # -------------------------------------------------------------- index
    def rebuild_index(self, force: bool = False) -> int:
        now = time.time()
        if not force and now - self._built_at < REBUILD_TTL and self._index:
            return len(self._index)

        sketches: list[EpisodeSketch] = []
        episodes: list[Any] = []
        try:
            if self.episodic is not None:
                episodes = list(self.episodic.recent_episodes(days=30, limit=MAX_INDEX))
        except Exception as e:  # noqa: BLE001
            log.debug("analogy_engine.episodic_fetch_failed: %r", e)

        for ep in episodes:
            try:
                goal = str(
                    getattr(ep, "user_input", "")
                    or (ep.get("user_input") if isinstance(ep, dict) else "")
                    or getattr(ep, "goal", "")
                    or (ep.get("goal") if isinstance(ep, dict) else "")
                    or ""
                )[:400]
                if len(goal) < 12:
                    continue
                trace = (
                    getattr(ep, "tool_calls", None)
                    or (ep.get("tool_calls") if isinstance(ep, dict) else None)
                    or getattr(ep, "tool_trace", None)
                    or (ep.get("tool_trace") if isinstance(ep, dict) else None)
                    or []
                )
                # Episode stores outcome_score 0-10; >=5 counts as success.
                score = (
                    getattr(ep, "outcome_score", None)
                    if not isinstance(ep, dict)
                    else ep.get("outcome_score")
                )
                if score is not None:
                    outcome = bool(float(score) >= 5.0)
                else:
                    outcome = bool(
                        getattr(ep, "success", False)
                        or (ep.get("success") if isinstance(ep, dict) else False)
                    )
                if not outcome:
                    continue
                ttype = str(
                    getattr(ep, "task_type", "")
                    or (ep.get("task_type") if isinstance(ep, dict) else "")
                    or ""
                )
                shape = _tool_shape(trace)
                summary = "tools: " + " → ".join(shape[:8]) if shape else ""
                ts = float(
                    getattr(ep, "timestamp", 0.0)
                    or (ep.get("timestamp") if isinstance(ep, dict) else 0.0)
                    or getattr(ep, "ts", 0.0)
                    or (ep.get("ts") if isinstance(ep, dict) else 0.0)
                    or 0.0
                )
                sketches.append(
                    EpisodeSketch(
                        goal=goal,
                        words=_content_words(goal),
                        shape=shape,
                        task_type=ttype,
                        outcome_success=bool(outcome),
                        approach=summary,
                        ts=ts,
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        self._index = sketches[-MAX_INDEX:]
        self._built_at = now
        log.info("analogy_engine index built: %d episode sketches", len(self._index))
        return len(self._index)

    # ------------------------------------------------------------ retrieve
    def find_analogies(
        self,
        user_text: str,
        task_type: str = "",
        limit: int = 3,
    ) -> list[Analogy]:
        self.rebuild_index()
        if not self._index:
            return []

        q_words = _content_words(user_text)
        if not q_words:
            return []

        scored: list[tuple[float, EpisodeSketch]] = []
        for sk in self._index:
            if not sk.words:
                continue
            word_sim = len(q_words & sk.words) / max(1, len(q_words | sk.words))
            if word_sim < 0.05:
                continue
            # Type match is a bonus, not a requirement — cross-domain
            # analogies are the whole point.
            type_bonus = 0.15 if task_type and sk.task_type == task_type else 0.0
            sim = word_sim * 0.85 + type_bonus
            if sim >= MIN_SIMILARITY:
                scored.append((sim, sk))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[Analogy] = []
        for sim, sk in scored[:limit]:
            out.append(
                Analogy(
                    summary=sk.goal[:160],
                    approach=sk.approach
                    or ("tools: " + " → ".join(sk.shape[:8]) if sk.shape else ""),
                    similarity=round(sim, 3),
                    task_type=sk.task_type,
                    outcome_success=sk.outcome_success,
                )
            )
        return out

    def render_hints(self, analogies: list[Analogy]) -> str:
        """Render analogies as a prompt block, or "" when none are useful."""
        if not analogies:
            return ""
        lines = ["PAST EPISODES (similar successful requests; tool sequence is evidence):"]
        for a in analogies:
            status = "succeeded" if a.outcome_success else "failed — avoid repeating"
            lines.append(
                f"- [{a.similarity:.2f}, {status}] {a.summary}"
                + (f" | approach: {a.approach}" if a.approach else "")
            )
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "index_size": len(self._index),
            "built_at": self._built_at,
            "success_share": (
                sum(1 for s in self._index if s.outcome_success) / len(self._index)
                if self._index
                else 0.0
            ),
        }
