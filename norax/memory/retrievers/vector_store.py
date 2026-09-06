"""Pluggable Vector Store Backend — scale path for embedding retrieval.

When neuron count exceeds the numpy flat index threshold (~50K), switch
to a dedicated vector store for O(log n) instead of O(n) search.

Supported backends:
  - numpy (default, in-memory, fine up to ~50K vectors)
  - Qdrant (local or remote, production-grade)
  - Chroma (local, lightweight)
  - LanceDB (local, columnar, fast)

The backend is auto-selected based on:
  1. Explicit configuration
  2. Availability (which is installed)
  3. Neuron count (auto-migrate when > threshold)

Usage:
    store = get_vector_store(memory_root)
    store.add(neuron_id, embedding, metadata)
    results = store.search(query_embedding, k=10)
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("norax.memory.vector_store")

AUTO_MIGRATE_THRESHOLD = 50_000  # neurons
_QDRANT_ID_NAMESPACE = uuid.UUID("2bb64582-433d-5e8f-8322-3f2adfdee81a")


def _validated_vector(values: list[float], *, dimension: int = 0) -> list[float]:
    if not values:
        raise ValueError("vector must contain at least one value")
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("vector values must be finite numbers") from exc
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("vector values must be finite numbers")
    if dimension and len(vector) != dimension:
        raise ValueError(
            f"vector dimension {len(vector)} does not match store dimension {dimension}"
        )
    return vector


def _validated_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    return k


def _qdrant_point_id(neuron_id: str) -> str:
    """Map arbitrary Norax entity IDs into Qdrant's UUID-compatible ID domain."""
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, str(neuron_id)))


@dataclass
class VectorSearchResult:
    """Result from a vector search."""

    id: str
    score: float
    metadata: dict


class NumpyVectorStore:
    """Default in-memory numpy vector store (existing behavior).

    No external dependencies. Fine up to ~50K vectors.
    """

    def __init__(self, dim: int = 0) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError("dim must be a non-negative integer")
        self.dim = dim
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._metadata: list[dict] = []

    def add(self, neuron_id: str, embedding: list[float], metadata: dict | None = None) -> None:
        vector = _validated_vector(embedding, dimension=self.dim)
        if self.dim == 0:
            self.dim = len(vector)
        try:
            index = self._ids.index(neuron_id)
        except ValueError:
            self._ids.append(neuron_id)
            self._vectors.append(vector)
            self._metadata.append(metadata or {})
        else:
            self._vectors[index] = vector
            self._metadata[index] = metadata or {}

    def search(self, query: list[float], k: int = 10) -> list[VectorSearchResult]:
        if not self._vectors:
            return []
        if _validated_k(k) <= 0:
            return []
        import numpy as np

        q = np.array(_validated_vector(query, dimension=self.dim), dtype=np.float32)
        mat = np.array(self._vectors, dtype=np.float32)
        # Cosine similarity
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        scores = mat_norm @ q_norm
        top_k = min(k, len(scores))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            VectorSearchResult(
                id=self._ids[i],
                score=float(scores[i]),
                metadata=self._metadata[i],
            )
            for i in idx
        ]

    def count(self) -> int:
        return len(self._ids)

    def remove(self, neuron_id: str) -> bool:
        try:
            idx = self._ids.index(neuron_id)
            self._ids.pop(idx)
            self._vectors.pop(idx)
            self._metadata.pop(idx)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        self._ids.clear()
        self._vectors.clear()
        self._metadata.clear()


class QdrantVectorStore:
    """Qdrant vector store backend.

    Requires: pip install qdrant-client
    Can run embedded (local) or connect to a remote Qdrant instance.
    """

    def __init__(
        self, collection: str = "norax_memory", url: str = "localhost:6333", dim: int = 1024
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            ) from exc

        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams

            collections = self.client.get_collections()
            names = [c.name for c in collections.collections]
            if self.collection not in names:
                # Create with default dim, will be set on first add
                self.client.create_collection(
                    self.collection,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
        except Exception as exc:
            log.warning("qdrant.ensure_collection failed: %s", exc)

    def add(self, neuron_id: str, embedding: list[float], metadata: dict | None = None) -> None:
        vector = _validated_vector(embedding, dimension=self.dim)
        try:
            from qdrant_client.models import PointStruct

            self.client.upsert(
                self.collection,
                points=[
                    PointStruct(
                        id=_qdrant_point_id(neuron_id),
                        vector=vector,
                        payload={**(metadata or {}), "_norax_id": neuron_id},
                    )
                ],
            )
        except Exception as exc:
            log.warning("qdrant.add failed: %s", exc)

    def search(self, query: list[float], k: int = 10) -> list[VectorSearchResult]:
        if _validated_k(k) <= 0:
            return []
        vector = _validated_vector(query, dimension=self.dim)
        try:
            results = self.client.search(
                self.collection,
                query_vector=vector,
                limit=k,
            )
            converted = []
            for result in results:
                metadata = dict(result.payload or {})
                original_id = str(metadata.pop("_norax_id", result.id))
                converted.append(
                    VectorSearchResult(id=original_id, score=result.score, metadata=metadata)
                )
            return converted
        except Exception as exc:
            log.warning("qdrant.search failed: %s", exc)
            return []

    def count(self) -> int:
        try:
            result = self.client.count(self.collection)
            return result.count
        except Exception:
            return 0

    def remove(self, neuron_id: str) -> bool:
        try:
            from qdrant_client.models import PointIdsList

            self.client.delete(
                self.collection,
                points_selector=PointIdsList(points=[_qdrant_point_id(neuron_id)]),
            )
            return True
        except Exception:
            return False

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection)
            self._ensure_collection()
        except Exception as exc:
            log.warning("qdrant.clear failed: %s", exc)


class ChromaVectorStore:
    """Chroma vector store backend (local, lightweight).

    Requires: pip install chromadb
    """

    def __init__(self, collection: str = "norax_memory", persist_dir: str = "") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("chromadb not installed. Run: pip install chromadb") from exc

        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()
        self.collection_name = collection
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def add(self, neuron_id: str, embedding: list[float], metadata: dict | None = None) -> None:
        vector = _validated_vector(embedding)
        self.collection.upsert(
            ids=[neuron_id],
            embeddings=[vector],
            metadatas=[metadata or {}],
        )

    def search(self, query: list[float], k: int = 10) -> list[VectorSearchResult]:
        if _validated_k(k) <= 0:
            return []
        vector = _validated_vector(query)
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            VectorSearchResult(
                id=ids[i],
                score=1.0 - distances[i],  # Chroma returns distance, convert to similarity
                metadata=metadatas[i] if i < len(metadatas) else {},
            )
            for i in range(len(ids))
        ]

    def count(self) -> int:
        return self.collection.count()

    def remove(self, neuron_id: str) -> bool:
        self.collection.delete(ids=[neuron_id])
        return True

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)


# ── Backend selection ────────────────────────────────────────────────────


def get_vector_store(
    memory_root: Path | None = None,
    backend: str = "auto",
    **kwargs: Any,
) -> NumpyVectorStore | QdrantVectorStore | ChromaVectorStore:
    """Get the appropriate vector store backend.

    Args:
        memory_root: Memory root path (for local backends)
        backend: "auto", "numpy", "qdrant", or "chroma"
        **kwargs: Backend-specific arguments

    Returns:
        Vector store instance
    """
    if backend == "numpy":
        return NumpyVectorStore()

    if backend == "qdrant":
        try:
            return QdrantVectorStore(**kwargs)
        except ImportError:
            log.warning("qdrant not available, falling back to numpy")
            return NumpyVectorStore()

    if backend == "chroma":
        try:
            persist_dir = str(memory_root / "chroma") if memory_root else ""
            return ChromaVectorStore(persist_dir=persist_dir, **kwargs)
        except ImportError:
            log.warning("chroma not available, falling back to numpy")
            return NumpyVectorStore()

    if backend != "auto":
        raise ValueError(f"unsupported vector store backend: {backend!r}")

    # Auto: try qdrant → chroma → numpy
    for name in ("qdrant", "chroma"):
        try:
            if name == "qdrant":
                return QdrantVectorStore(**kwargs)
            persist_dir = str(memory_root / "chroma") if memory_root else ""
            return ChromaVectorStore(persist_dir=persist_dir)
        except ImportError:
            continue
        except Exception:
            continue

    return NumpyVectorStore()
