"""ContextInjector — the smart mixer.

Runs FastContext + LocalRetriever + ExternalRetriever in parallel, merges
their results, dedupes on entity_id, de-duplicates redundant text with
a cheap MMR (maximal marginal relevance), applies per-source floor so
each retriever contributes at least `min_per_source`, and fits everything
into a char budget.

Scoring improvements (v2):
  - Recency boost: neurons updated within 7 days get a mild lift; within
    24 h get a stronger lift. Older items are gently penalized.
  - Kind weighting: task_type biases which memory kinds surface first.
    coding/debugging → procedural up, sleep down
    research → intel up
    general → no bias (flat)

Returns a list[(neuron, score, source)] ready for the assembler's
MEMORY block.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass

from .retrievers.external import ExternalRetriever
from .retrievers.fast import FastContext
from .retrievers.local import LocalRetriever
from .store import Neuron

log = logging.getLogger("norax.memory.injection")


# ---------------------------------------------------------------------------
# Recency + kind weighting helpers
# ---------------------------------------------------------------------------

# How much recency can boost/penalize the base similarity score.
_RECENCY_HOT_BOOST = 0.12  # updated < 24 h
_RECENCY_WEEK_BOOST = 0.05  # updated 1–7 days
_RECENCY_COLD_PENALTY = 0.04  # updated > 30 days

# Per-task-type kind multipliers.  Keys match memory Neuron.kind values.
# 1.0 = neutral, >1.0 = boost, <1.0 = suppress.
_KIND_WEIGHTS: dict[str, dict[str, float]] = {
    "coding": {
        "procedural": 1.20,
        "semantic": 1.00,
        "intel": 0.85,
        "scratchpad": 1.10,
        "focus": 1.10,
        "sleep": 0.70,
    },
    "debugging": {
        "procedural": 1.25,
        "semantic": 1.00,
        "intel": 0.80,
        "scratchpad": 1.10,
        "focus": 1.10,
        "sleep": 0.70,
    },
    "research": {
        "intel": 1.25,
        "semantic": 1.10,
        "procedural": 0.85,
        "scratchpad": 0.90,
        "focus": 1.00,
        "sleep": 0.90,
    },
    "general": {},  # empty → all multipliers default to 1.0
}


def _recency_delta(neuron: Neuron, now: float) -> float:
    """Return a score delta based on how recently the neuron was written."""
    age_h = (now - neuron.mtime) / 3600.0
    if age_h < 24:
        return _RECENCY_HOT_BOOST
    if age_h < 24 * 7:
        return _RECENCY_WEEK_BOOST
    if age_h > 24 * 30:
        return -_RECENCY_COLD_PENALTY
    return 0.0


def _kind_multiplier(neuron: Neuron, task_type: str) -> float:
    """Return a score multiplier based on neuron kind and task type."""
    weights = _KIND_WEIGHTS.get(task_type, {})
    return weights.get(neuron.kind, 1.0)


def _adjusted_score(neuron: Neuron, base_score: float, task_type: str, now: float) -> float:
    """Apply recency and kind weighting on top of the retriever's base score."""
    score = base_score * neuron.weight  # existing weight tag (W1-W5)
    score *= _kind_multiplier(neuron, task_type)
    score += _recency_delta(neuron, now)
    return max(0.0, score)


@dataclass
class InjectionResult:
    items: list[tuple[Neuron, float, str]]  # (neuron, score, source_label)
    total_chars: int
    counts: dict[str, int]  # how many came from each source


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class ContextInjector:
    fast: FastContext
    local: LocalRetriever | None = None
    external: ExternalRetriever | None = None
    budget_chars: int = 32000  # ~8k tokens; scales with 256k rolling window
    min_per_source: int = 2  # guarantee each retriever contributes
    mmr_lambda: float = 0.75  # higher = prefer relevance over diversity

    async def run(
        self,
        query: str,
        *,
        k_each: int = 12,
        task_type: str = "general",
    ) -> InjectionResult:
        """Retrieve and mix memory items for injection.

        `task_type` biases which memory kinds surface first (coding →
        procedural up; research → intel up; general → flat).
        Recency is always applied: items written in the last 24 h get a
        score lift; items older than 30 days get a mild penalty.
        """
        now = time.time()
        coros = [self._run_one("fast", self.fast.search, query, k_each)]
        shared_embedder = (
            self.local is not None
            and self.external is not None
            and self.local.embedder is self.external.embedder
        )
        if shared_embedder:
            coros.append(self._run_shared_embedding(query, k_each))
        else:
            if self.local is not None:
                coros.append(self._run_one("local", self.local.search, query, k_each))
            if self.external is not None:
                coros.append(self._run_one("external", self.external.search, query, k_each))

        gathered = await asyncio.gather(*coros, return_exceptions=True)
        groups: list[BaseException | tuple[str, list[tuple[Neuron, float]]]] = []
        for group in gathered:
            if isinstance(group, list):
                groups.extend(group)
            else:
                groups.append(group)

        # Apply recency + kind weighting to every retrieved item upfront
        def _adjust(items: list[tuple[Neuron, float]], src: str) -> list[tuple[Neuron, float]]:
            return [(n, _adjusted_score(n, s, task_type, now)) for n, s in items]

        merged: list[tuple[Neuron, float, str]] = []
        counts = {"fast": 0, "local": 0, "external": 0}
        selected_ids: set[str] = set()

        # First pass: guarantee min_per_source top-N (deduped by entity_id)
        adjusted_groups: list[BaseException | tuple[str, list[tuple[Neuron, float]]]] = []
        for g in groups:
            if isinstance(g, BaseException):
                log.warning("retriever.error: %r", g)
                adjusted_groups.append(g)
                continue
            src, items = g
            adjusted_groups.append((src, _adjust(items, src)))

        for g in adjusted_groups:
            if isinstance(g, BaseException):
                continue
            src, items = g
            # Re-sort after score adjustment
            items_sorted = sorted(items, key=lambda x: -x[1])
            taken = 0
            for n, s in items_sorted:
                if taken >= self.min_per_source:
                    break
                if n.entity_id in selected_ids:
                    continue
                merged.append((n, s, src))
                selected_ids.add(n.entity_id)
                counts[src] = counts.get(src, 0) + 1
                taken += 1

        # Second pass: fill remaining slots from all sources, score-sorted
        pool: list[tuple[Neuron, float, str]] = []
        for g in adjusted_groups:
            if isinstance(g, BaseException):
                continue
            src, items = g
            items_sorted = sorted(items, key=lambda x: -x[1])
            for n, s in items_sorted[self.min_per_source :]:
                pool.append((n, s, src))
        pool.sort(key=lambda x: -x[1])

        # MMR dedupe against already-merged + against each new candidate
        chars = sum(len(n.text) for n, _, _ in merged)

        for n, s, src in pool:
            if n.entity_id in selected_ids:
                continue
            max_sim = 0.0
            for sn, _, _ in merged:
                j = _jaccard(n.text, sn.text)
                if j > max_sim:
                    max_sim = j
            mmr = self.mmr_lambda * s - (1 - self.mmr_lambda) * max_sim
            if mmr <= 0:
                continue
            if chars + len(n.text) + 2 > self.budget_chars:
                continue
            merged.append((n, mmr, src))
            selected_ids.add(n.entity_id)
            counts[src] = counts.get(src, 0) + 1
            chars += len(n.text) + 2

        merged.sort(key=lambda x: -x[1])
        return InjectionResult(items=merged, total_chars=chars, counts=counts)

    async def _run_one(self, label: str, coro_or_fn, query: str, k: int):
        if inspect.iscoroutinefunction(coro_or_fn):
            out = coro_or_fn(query, k=k)
        else:
            out = await asyncio.to_thread(coro_or_fn, query, k=k)
        if inspect.isawaitable(out):
            res = await out
        else:
            res = out
        # Normalize retriever outputs: some return (Neuron, score) and some
        # return (Neuron, score, tag). Strip the tag so downstream logic can
        # trust the 2-tuple shape.
        normalized: list[tuple[Neuron, float]] = []
        for row in res:
            if isinstance(row, tuple):
                if len(row) >= 2:
                    normalized.append((row[0], float(row[1])))
        return label, normalized

    async def _run_shared_embedding(
        self, query: str, k: int
    ) -> list[tuple[str, list[tuple[Neuron, float]]]]:
        """Refresh two indexes in parallel and embed their common query once."""
        local = self.local
        external = self.external
        if local is None or external is None:
            return []
        await asyncio.gather(
            local.refresh_if_stale(),
            external.refresh_if_stale(),
            return_exceptions=True,
        )
        query_vector = await local.embed_query(query)
        if query_vector is None:
            return [("local", []), ("external", [])]
        return [
            ("local", local.search_vector(query_vector, k=k)),
            ("external", external.search_vector(query_vector, k=k)),
        ]
