"""Multi-signal retriever — 6-signal RRF: keyword + embedding + entity + FTS5 + causal + temporal.

MAGMA 2026 pattern: orthogonal graph layers across semantic, temporal, causal, and entity.
Norax implementation:
  - Embedding (weight 1.0) — semantic similarity via LocalRetriever
  - Entity (weight 0.9) — Jaccard entity overlap across full graph
  - Keyword (weight 0.8) — FastContext term matching
  - FTS5 index (weight 0.85) — BM25 persistent index via SQLiteIndexRetriever
  - Causal (weight 0.75) — tool-call → outcome tracking (MAGMA causal graph)
  - Temporal (weight 0.7) — sequential before/after edges + recency boost (MAGMA temporal graph)

Results tagged: ``kw``, ``kw+ent+causal``, ``kw+em+ent+idx+tmp``, etc.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ...gateway_client import SpendGuardTripped
from ..causal_graph import CausalGraph
from ..entity_graph import EntityGraph
from ..hot_inject import merge_pinned, pin_hot_neurons
from ..store import Neuron
from ..temporal_graph import TemporalGraph
from .cross_encoder import LLMJudgeReranker
from .fast import FastContext
from .local import LocalRetriever
from .sqlite_index import SQLiteIndexRetriever

log = logging.getLogger("norax.memory.retrievers.multi_signal")

_RRF_K = 60


@runtime_checkable
class SignalProtocol(Protocol):
    """Uniform async search protocol every retrieval signal must satisfy.

    Returns a list of (Neuron, score, source_tag) triples.
    """

    async def search(self, query: str, *, k: int = 6) -> list[tuple[Neuron, float, str]]: ...


def _idx_to_neuron(row: dict) -> Neuron:
    """Convert a SQLiteIndexRetriever result row into a Neuron."""
    return Neuron(
        text=row.get("text", ""),
        path=Path(row.get("path", "")),
        line=row.get("start_line", 0),
        weight=row.get("weight", 1.0),
        kind=row.get("kind", "semantic"),
        entity_id=row.get("entity_id", ""),
    )


@dataclass
class MultiSignalRetriever:
    """Parallel 6-signal retrieval with weighted RRF (MAGMA-inspired)."""

    keyword: FastContext
    embedding: LocalRetriever | None = None
    entity_graph: EntityGraph | None = None
    sqlite_index: SQLiteIndexRetriever | None = None
    causal_graph: CausalGraph | None = None
    temporal_graph: TemporalGraph | None = None
    keyword_weight: float = 0.8
    embedding_weight: float = 1.0
    entity_weight: float = 0.9
    index_weight: float = 0.85
    causal_weight: float = 0.75
    temporal_weight: float = 0.7
    _embed_ready: bool = field(default=False, repr=False)
    _entity_refresh_task: asyncio.Task[None] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    reranker: LLMJudgeReranker | None = None

    def validate_signals(self) -> list[str]:
        """Check every injected signal against SignalProtocol at construction.

        Returns a list of warnings for signals that don't conform.
        Call this right after construction to catch protocol drift early
        instead of at first query time.
        """
        warnings: list[str] = []
        signals = [
            ("keyword", self.keyword),
            ("embedding", self.embedding),
            ("sqlite_index", self.sqlite_index),
        ]
        for name, sig in signals:
            if sig is None:
                continue
            if not hasattr(sig, "search"):
                warnings.append(
                    f"{name}: missing search() method — does not conform to SignalProtocol"
                )
        return warnings

    async def ensure_index(self, *, background_refresh: bool = False) -> None:
        """Build/refresh all indexes and load graph sidecars.

        When ``background_refresh`` is True, a stale entity graph is refreshed
        in a fire-and-forget background task instead of blocking the caller.
        The stale graph is still usable for the current turn — it just doesn't
        include the most recent memory writes. This is the right tradeoff for
        latency-sensitive turn paths.
        """
        store = self.keyword.store
        neurons = store.all_canonical() + store.hot

        async def ensure_entity_graph() -> None:
            if self.entity_graph is None:
                return
            if background_refresh:
                if self._entity_refresh_task is None or self._entity_refresh_task.done():
                    # Snapshot the neuron list so a concurrent store refresh
                    # cannot change what this fingerprint/rebuild represents.
                    snapshot = list(neurons)
                    self._entity_refresh_task = asyncio.create_task(
                        self._refresh_entity_graph(snapshot),
                        name="entity-graph-refresh",
                    )
                else:
                    log.debug("multi_signal.entity_graph refresh already running")
                return

            current = await asyncio.to_thread(self.entity_graph.ensure, neurons)
            if not current:
                log.warning("multi_signal.entity_graph foreground refresh remained stale")

        # Sidecars can be many megabytes. Load independent graph layers in
        # workers and in parallel so startup/warmup never blocks the event loop.
        graph_loads: list[Awaitable[Any]] = [ensure_entity_graph()]
        if self.causal_graph is not None:
            graph_loads.append(asyncio.to_thread(self.causal_graph.load))
        if self.temporal_graph is not None:
            graph_loads.append(asyncio.to_thread(self.temporal_graph.load))
        await asyncio.gather(*graph_loads)

        if self.embedding is not None:
            try:
                refreshed = await self.embedding.refresh_if_stale()
                self._embed_ready = True
                if refreshed:
                    log.info("multi_signal.embed_index refreshed")
            except Exception as e:
                log.warning("multi_signal.embed_index.error: %r (fallback)", e)
                self._embed_ready = False

        if self.sqlite_index is not None:
            try:
                stats = self.sqlite_index.stats()
                log.info("multi_signal.sqlite_index ready: %s", stats)
            except Exception as e:
                log.warning("multi_signal.sqlite_index.error: %r", e)

    async def _refresh_entity_graph(self, neurons: list[Neuron]) -> None:
        """Refresh a stale entity graph off the event loop exactly once."""
        if self.entity_graph is None:
            return
        try:
            current = await asyncio.to_thread(self.entity_graph.ensure, neurons)
            if current:
                log.info("multi_signal.entity_graph background refresh complete")
            else:
                log.warning("multi_signal.entity_graph background refresh remained stale")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("multi_signal.entity_graph background refresh failed: %r", exc)

    def search_sync(self, query: str, *, k: int = 6) -> list[tuple[Neuron, float, str]]:
        """Sync path: keyword + entity + FTS5 + causal + temporal (for plan assembly)."""
        neurons = self.keyword.store.all_canonical() + self.keyword.store.hot
        kw_results = self.keyword.search(query, k=k * 2)
        ent_results: list[tuple[Neuron, float]] = []
        if self.entity_graph is not None:
            ent_results = self.entity_graph.search(query, neurons, k=k * 2)
        idx_results: list[tuple[Neuron, float]] = []
        if self.sqlite_index is not None:
            try:
                rows = self.sqlite_index.search_keyword(query, k=k * 2)
                idx_results = [(_idx_to_neuron(r), -r.get("rank", 0)) for r in rows]
            except Exception as e:
                log.warning("multi_signal.sqlite_index.search_sync.error: %r", e)
        causal_results: list[tuple[Neuron, float]] = []
        if self.causal_graph is not None:
            causal_results = self.causal_graph.search(query, neurons, k=k * 2)
        temporal_results: list[tuple[Neuron, float]] = []
        if self.temporal_graph is not None:
            temporal_results = self.temporal_graph.search(query, neurons, k=k * 2)
        fused = self._fuse(
            kw_results, [], ent_results, idx_results, causal_results, temporal_results, k=k
        )
        pinned = pin_hot_neurons(self.keyword.store)
        return merge_pinned(fused, pinned, k=k)

    async def search(self, query: str, *, k: int = 6) -> list[tuple[Neuron, float, str]]:
        """Full async 6-signal search."""
        kw_task = asyncio.ensure_future(asyncio.to_thread(self.keyword.search, query, k=k * 2))

        embed_results: list[tuple[Neuron, float]] = []
        if self.embedding is not None and self._embed_ready:
            try:
                embed_results = await asyncio.wait_for(
                    self.embedding.search(query, k=k * 2),
                    timeout=5.0,
                )
            except TimeoutError:
                log.warning("multi_signal.embed_timeout query=%r", query[:60])
            except Exception as e:
                log.warning("multi_signal.embed_error: %r", e)

        neurons = self.keyword.store.all_canonical() + self.keyword.store.hot

        ent_results: list[tuple[Neuron, float]] = []
        if self.entity_graph is not None:
            ent_results = self.entity_graph.search(query, neurons, k=k * 2)

        idx_results: list[tuple[Neuron, float]] = []
        if self.sqlite_index is not None:
            try:
                rows = await asyncio.to_thread(self.sqlite_index.search_keyword, query, k=k * 2)
                idx_results = [(_idx_to_neuron(r), -r.get("rank", 0)) for r in rows]
            except Exception as e:
                log.warning("multi_signal.sqlite_index.search.error: %r", e)

        causal_results: list[tuple[Neuron, float]] = []
        if self.causal_graph is not None:
            causal_results = self.causal_graph.search(query, neurons, k=k * 2)

        temporal_results: list[tuple[Neuron, float]] = []
        if self.temporal_graph is not None:
            temporal_results = self.temporal_graph.search(query, neurons, k=k * 2)

        kw_results = await kw_task
        fused = self._fuse(
            kw_results,
            embed_results,
            ent_results,
            idx_results,
            causal_results,
            temporal_results,
            k=k,
        )
        pinned = pin_hot_neurons(self.keyword.store)
        merged = merge_pinned(fused, pinned, k=k)
        # Optional generative-judge reranking for precision boost.
        if self.reranker is not None and merged:
            try:
                merged = await self.reranker.rerank(query, merged, top_k=k)
            except (asyncio.CancelledError, SpendGuardTripped):
                raise
            except Exception as e:
                log.warning("multi_signal.rerank.error: %r", e)
        return merged

    def _fuse(
        self,
        kw_results: list[tuple[Neuron, float]],
        embed_results: list[tuple[Neuron, float]],
        ent_results: list[tuple[Neuron, float]],
        idx_results: list[tuple[Neuron, float]],
        causal_results: list[tuple[Neuron, float]],
        temporal_results: list[tuple[Neuron, float]],
        *,
        k: int,
    ) -> list[tuple[Neuron, float, str]]:
        kw_ranked: dict[str, tuple[Neuron, int]] = {}
        for rank, (n, _) in enumerate(kw_results):
            kw_ranked[n.entity_id] = (n, rank)

        em_ranked: dict[str, tuple[Neuron, int]] = {}
        for rank, (n, _) in enumerate(embed_results):
            em_ranked[n.entity_id] = (n, rank)

        ent_ranked: dict[str, tuple[Neuron, int]] = {}
        for rank, (n, _) in enumerate(ent_results):
            ent_ranked[n.entity_id] = (n, rank)

        idx_ranked: dict[str, tuple[Neuron, int]] = {}
        for rank, (n, _) in enumerate(idx_results):
            idx_ranked[n.entity_id] = (n, rank)

        causal_ranked: dict[str, tuple[Neuron, int]] = {}
        for rank, (n, _) in enumerate(causal_results):
            causal_ranked[n.entity_id] = (n, rank)

        temporal_ranked: dict[str, tuple[Neuron, int]] = {}
        for rank, (n, _) in enumerate(temporal_results):
            temporal_ranked[n.entity_id] = (n, rank)

        all_entities = (
            set(kw_ranked)
            | set(em_ranked)
            | set(ent_ranked)
            | set(idx_ranked)
            | set(causal_ranked)
            | set(temporal_ranked)
        )
        fused: list[tuple[Neuron, float, str]] = []

        for eid in all_entities:
            sources: list[str] = []
            score = 0.0
            neuron: Neuron | None = None

            if eid in kw_ranked:
                n, rank = kw_ranked[eid]
                neuron = n
                score += self.keyword_weight / (_RRF_K + rank + 1)
                sources.append("kw")
            if eid in em_ranked:
                n, rank = em_ranked[eid]
                neuron = neuron or n
                score += self.embedding_weight / (_RRF_K + rank + 1)
                sources.append("em")
            if eid in ent_ranked:
                n, rank = ent_ranked[eid]
                neuron = neuron or n
                score += self.entity_weight / (_RRF_K + rank + 1)
                sources.append("ent")
            if eid in idx_ranked:
                n, rank = idx_ranked[eid]
                neuron = neuron or n
                score += self.index_weight / (_RRF_K + rank + 1)
                sources.append("idx")
            if eid in causal_ranked:
                n, rank = causal_ranked[eid]
                neuron = neuron or n
                score += self.causal_weight / (_RRF_K + rank + 1)
                sources.append("causal")
            if eid in temporal_ranked:
                n, rank = temporal_ranked[eid]
                neuron = neuron or n
                score += self.temporal_weight / (_RRF_K + rank + 1)
                sources.append("tmp")

            if neuron is None:
                continue
            age = neuron.age_days()
            recency = 1.0 / (1.0 + 0.01 * age) if age > 0 else 1.0
            score *= recency
            fused.append((neuron, score, "+".join(sources)))

        fused.sort(key=lambda x: -x[1])
        return fused[:k]
