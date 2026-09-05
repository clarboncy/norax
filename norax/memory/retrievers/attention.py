"""Attention Heads — parallel query routing for retrieval.

Instead of sending every query through the same retriever, attention heads
classify the query and route to the optimal source:

  Head 0 (identity):   wallet/email/id queries   → keyword exact match
  Head 1 (semantic):   conceptual/factual queries → embedding similarity
  Head 2 (procedural): how-to/workflow queries    → procedural store
  Head 3 (temporal):   recent/time queries        → episodic buffer

Results from active heads are merged with weighted RRF.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..store import Neuron

log = logging.getLogger("norax.memory.retrievers.attention")

# Query classifiers (lightweight regex — no ML needed)
_IDENTITY_RX = re.compile(r"\b(wallet|address|email|0x[a-f0-9]{6,}|id|token|contract|key)\b", re.I)
_PROCEDURAL_RX = re.compile(
    r"\b(how to|steps|workflow|process|procedure|when .+ do|pattern|avoid)\b", re.I
)
_TEMPORAL_RX = re.compile(
    r"\b(recent(?:ly)?|latest|last|today|yesterday|earlier|just now|minute|hour ago)\b", re.I
)

_RRF_K = 60


@dataclass
class AttentionHead:
    """A single retrieval head with a name, classifier, and searcher."""

    name: str
    weight: float = 1.0
    # searcher: async (query, k) -> list[(Neuron, float)] or list[(Neuron, float, str)]
    searcher: Any = None

    async def search(self, query: str, k: int) -> list[tuple[Neuron, float]]:
        if self.searcher is None:
            return []
        try:
            result = self.searcher(query, k=k)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as e:
            log.warning("attention.head.%s.error: %r", self.name, e)
            return []


@dataclass
class AttentionRouter:
    """Routes queries to appropriate heads and merges results."""

    heads: list[AttentionHead] = field(default_factory=list)

    def classify(self, query: str) -> list[str]:
        """Determine which heads should fire for this query."""
        active = []
        if _IDENTITY_RX.search(query):
            active.append("identity")
        if _PROCEDURAL_RX.search(query):
            active.append("procedural")
        if _TEMPORAL_RX.search(query):
            active.append("temporal")
        if not active:
            active.append("semantic")  # default
        # Always include semantic as a fallback if others fire
        if "semantic" not in active and len(active) < 3:
            active.append("semantic")
        return active

    async def search(self, query: str, *, k: int = 6) -> list[tuple[Neuron, float, str]]:
        """Route query to appropriate heads, merge with RRF."""
        active_names = self.classify(query)
        active_heads = [h for h in self.heads if h.name in active_names]

        if not active_heads:
            # Fallback: use all heads
            active_heads = self.heads

        # Fire all active heads in parallel
        tasks = {h.name: asyncio.ensure_future(h.search(query, k * 2)) for h in active_heads}

        results_by_head: dict[str, list[tuple[Neuron, float]]] = {}
        for name, task in tasks.items():
            try:
                results_by_head[name] = await asyncio.wait_for(task, timeout=5.0)
            except (TimeoutError, Exception) as e:
                log.warning("attention.head.%s.timeout_or_error: %r", name, e)
                results_by_head[name] = []

        # RRF fusion across heads
        entity_scores: dict[str, tuple[Neuron, float, list[str]]] = {}
        for head in active_heads:
            head_results = results_by_head.get(head.name, [])
            for rank, item in enumerate(head_results):
                n = item[0]
                rrf_score = head.weight / (_RRF_K + rank + 1)
                if n.entity_id in entity_scores:
                    existing_n, existing_score, sources = entity_scores[n.entity_id]
                    entity_scores[n.entity_id] = (
                        existing_n,
                        existing_score + rrf_score,
                        sources + [head.name],
                    )
                else:
                    entity_scores[n.entity_id] = (n, rrf_score, [head.name])

        fused = [
            (n, score, "+".join(sorted(set(sources))))
            for n, score, sources in entity_scores.values()
        ]
        fused.sort(key=lambda x: -x[1])
        return fused[:k]
