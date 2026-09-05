"""ExternalRetriever — embedding kNN over sleep/ buffer + intel/.

Exists separately because these sources have different volatility and
freshness characteristics than canonical stores:
  - sleep/   : hot, changes on every roll; re-index cheap (small N)
  - intel/   : bursty updates; external research notes

Scored with a recency bonus (sleep content is time-sensitive by design).

Uses IdMapIndex for O(1) add/remove/update of individual neurons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..embeddings import Embedder
from ..index import IdMapIndex
from ..store import MemoryStore, Neuron

log = logging.getLogger("norax.memory.retrievers.external")


@dataclass
class ExternalRetriever:
    store: MemoryStore
    embedder: Embedder
    index: IdMapIndex = None  # type: ignore[assignment]
    recency_half_life_days: float = 2.0  # sleep content decays fast
    cache_path: Path | None = None
    _built_once: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = IdMapIndex()
        if self.cache_path is None:
            self.cache_path = self.store.root / "embed_cache_external.npz"
        self.index.cache_path = self.cache_path

    async def refresh_if_stale(self) -> bool:
        """Incremental refresh: only embed new/changed neurons.

        On first call, tries loading cached vectors from disk. On subsequent
        calls, uses sync_from() for O(delta) embedding cost.
        """
        neurons = self.store.all_external()

        # First-time build: index is empty
        if self.index.vectors.shape[0] == 0 and not self._built_once:
            # Try disk cache first
            loaded = self.index.load(self.embedder, neurons, path=self.cache_path)
            if loaded:
                self._built_once = True
                delta = await self.index.embed_delta(self.embedder, neurons)
                if delta:
                    log.info(
                        "external_retriever: cache loaded + delta embedded (%d neurons total)",
                        len(neurons),
                    )
                else:
                    log.info(
                        "external_retriever: cache loaded, all vectors matched (%d neurons)",
                        len(neurons),
                    )
                return True
            # Cache miss — full build
            try:
                await self.index.build(self.embedder, neurons)
                self._built_once = True
                log.info("external_retriever: initial index built (%d neurons)", len(neurons))
                return True
            except Exception as e:
                log.warning("external.refresh_if_stale.build_failed: %r", e)
                return False

        # Incremental sync
        try:
            return await self.index.sync_from(self.embedder, neurons)
        except Exception as e:
            log.warning("external.refresh_if_stale.sync_failed: %r", e)
            return False

    async def search(self, query: str, *, k: int = 5) -> list[tuple[Neuron, float]]:
        await self.refresh_if_stale()
        qvec_batch = await self.embedder.embed([query])
        if qvec_batch.shape[0] == 0:
            return []
        return self.search_vector(qvec_batch[0], k=k)

    def search_vector(self, query_vector: np.ndarray, *, k: int = 5) -> list[tuple[Neuron, float]]:
        """Search the current index with an already-computed query vector."""
        if query_vector.size == 0 or np.allclose(query_vector, 0):
            return []
        raw = self.index.search(query_vector, k=k * 2)  # over-fetch for recency rerank
        # Apply recency decay: newer → bigger bonus. Half-life days.
        rescored: list[tuple[Neuron, float]] = []
        for n, s in raw:
            age = n.age_days()
            decay = 0.5 ** (age / self.recency_half_life_days)
            # 70% semantic + 30% recency
            rescored.append((n, 0.7 * s + 0.3 * decay))
        rescored.sort(key=lambda x: -x[1])
        return rescored[:k]
