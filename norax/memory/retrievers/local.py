"""LocalRetriever — embedding kNN over canonical memory (semantic/procedural/intel).

Uses `IdMapIndex` with cached vectors. Index is rebuilt lazily when the
underlying neuron set's fingerprint changes. One build on startup,
re-embed only the delta on refresh.

Resilience (2026-06-10): all embed calls wrapped in try/except so that
Ollama outages degrade to empty results instead of crashing the retriever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..embeddings import Embedder
from ..index import IdMapIndex
from ..store import MemoryStore, Neuron

if TYPE_CHECKING:
    from ...runtime.graceful_degradation import GracefulDegradation

log = logging.getLogger("norax.memory.retrievers.local")


@dataclass
class LocalRetriever:
    store: MemoryStore
    embedder: Embedder
    index: IdMapIndex = None  # type: ignore[assignment]
    _built_once: bool = field(default=False, repr=False)
    cache_path: Path | None = None  # override for embed cache location
    degradation_manager: GracefulDegradation | None = (
        None  # GracefulDegradation instance for failure reporting
    )

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = IdMapIndex()
        # Default cache path follows the store's memory root
        if self.cache_path is None:
            self.cache_path = self.store.root / "embed_cache.npz"
        self.index.cache_path = self.cache_path

    async def refresh_if_stale(self) -> bool:
        """Incremental refresh: only embed new/changed neurons.

        Uses entity_id (content hash) to skip unchanged neurons entirely.
        On first call (empty index), tries loading cached vectors from disk.
        On subsequent calls, uses sync_from() for O(delta) embedding cost.
        """
        neurons = self.store.all_canonical()

        # First-time build: index is empty
        if self.index.vectors.shape[0] == 0 and not self._built_once:
            # Try disk cache first
            loaded = self.index.load(self.embedder, neurons, path=self.cache_path)
            if loaded:
                self._built_once = True
                # Embed any unmatched neurons (delta from cache)
                delta = await self.index.embed_delta(self.embedder, neurons)
                if delta:
                    log.info(
                        "local_retriever: cache loaded + delta embedded (%d neurons total)",
                        len(neurons),
                    )
                else:
                    log.info(
                        "local_retriever: cache loaded, all vectors matched (%d neurons)",
                        len(neurons),
                    )
                return True
            # Cache miss — full build
            try:
                await self.index.build(self.embedder, neurons)
                self._built_once = True
                log.info("local_retriever: initial index built (%d neurons)", len(neurons))
                return True
            except Exception as e:
                log.warning("local_retriever.refresh_if_stale.build_failed: %r", e)
                return False

        # Incremental sync: only embed the delta
        try:
            changed = await self.index.sync_from(self.embedder, neurons)
            if changed:
                log.debug("local_retriever: incremental sync (delta embedded)")
            return changed
        except Exception as e:
            log.warning("local_retriever.refresh_if_stale.sync_failed: %r — keeping stale index", e)
            return False

    async def search(self, query: str, *, k: int = 5) -> list[tuple[Neuron, float]]:
        try:
            await self.refresh_if_stale()
        except Exception as e:
            log.warning("local_retriever.search.refresh_failed: %r", e)

        query_vector = await self.embed_query(query)
        if query_vector is None:
            return []
        return self.search_vector(query_vector, k=k)

    async def embed_query(self, query: str) -> np.ndarray | None:
        """Embed one query with consistent degradation accounting."""
        try:
            qvec_batch = await self.embedder.embed([query])
        except Exception as e:
            log.warning("local_retriever.search.embed_failed: %r — returning empty", e)
            if self.degradation_manager is not None:
                self.degradation_manager.report_failure(
                    "embedder",
                    str(e),
                    observed_fallback="omit_semantic_results",
                )
            return None

        if qvec_batch.shape[0] == 0:
            return None
        if np.allclose(qvec_batch[0], 0):
            return None
        if self.degradation_manager is not None:
            self.degradation_manager.report_recovery("embedder")
        return qvec_batch[0]

    def search_vector(self, query_vector: np.ndarray, *, k: int = 5) -> list[tuple[Neuron, float]]:
        """Search the current index with an already-computed query vector."""
        if query_vector.size == 0 or np.allclose(query_vector, 0):
            return []
        return self.index.search(query_vector, k=k)
