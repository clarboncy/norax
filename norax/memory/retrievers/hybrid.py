"""Hybrid Retriever — Sprint A prefrontal context merger.

Fuses keyword (FastContext) + embedding (LocalRetriever) results in parallel.
RRF (Reciprocal Rank Fusion) merges the two ranked lists into one.

This is the L12 prefrontal equivalent: it owns the "what goes into context"
decision by combining multiple retrieval signals.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..store import Neuron
from .fast import FastContext
from .local import LocalRetriever

log = logging.getLogger("norax.memory.retrievers.hybrid")

# RRF constant (standard value from Cormack et al. 2009)
_RRF_K = 60


@dataclass
class HybridRetriever:
    """Parallel keyword + embedding retrieval with RRF fusion."""

    keyword: FastContext
    embedding: LocalRetriever | None = None
    keyword_weight: float = 1.0
    embedding_weight: float = 1.2  # slight boost for semantic matches
    _embed_ready: bool = field(default=False, repr=False)

    async def ensure_index(self) -> None:
        """Build/refresh the embedding index if stale. Call once at startup."""
        if self.embedding is not None:
            try:
                refreshed = await self.embedding.refresh_if_stale()
                self._embed_ready = True
                if refreshed:
                    log.info("hybrid.embed_index refreshed")
            except Exception as e:
                log.warning("hybrid.embed_index.error: %r (keyword-only fallback)", e)
                self._embed_ready = False

    async def search(self, query: str, *, k: int = 6) -> list[tuple[Neuron, float, str]]:
        """Returns list of (Neuron, fused_score, source_tag)."""

        # Always run keyword; run embedding in parallel if available
        kw_task = asyncio.ensure_future(asyncio.to_thread(self.keyword.search, query, k=k * 2))

        embed_results: list[tuple[Neuron, float]] = []
        if self.embedding is not None and self._embed_ready:
            try:
                embed_results = await asyncio.wait_for(
                    self.embedding.search(query, k=k * 2),
                    timeout=5.0,
                )
            except TimeoutError:
                log.warning("hybrid.embed_timeout query=%r", query[:60])
            except Exception as e:
                log.warning("hybrid.embed_error: %r", e)

        kw_results = await kw_task

        # Build per-entity ranked lists
        kw_ranked: dict[str, tuple[Neuron, int]] = {}
        for rank, (n, _score) in enumerate(kw_results):
            kw_ranked[n.entity_id] = (n, rank)

        em_ranked: dict[str, tuple[Neuron, int]] = {}
        for rank, (n, _score) in enumerate(embed_results):
            em_ranked[n.entity_id] = (n, rank)

        # RRF fusion
        all_entities = set(kw_ranked.keys()) | set(em_ranked.keys())
        fused: list[tuple[Neuron, float, str]] = []

        for eid in all_entities:
            sources = []
            score = 0.0
            if eid in kw_ranked:
                n, rank = kw_ranked[eid]
                score += self.keyword_weight / (_RRF_K + rank + 1)
                sources.append("kw")
            if eid in em_ranked:
                n, rank = em_ranked[eid]
                score += self.embedding_weight / (_RRF_K + rank + 1)
                sources.append("em")
            # Prefer the neuron object from embedding (richer) if available
            neuron = em_ranked.get(eid, kw_ranked.get(eid, (None, 0)))[0]
            if neuron is None:
                continue
            fused.append((neuron, score, "+".join(sources)))

        fused.sort(key=lambda x: -x[1])
        return fused[:k]
