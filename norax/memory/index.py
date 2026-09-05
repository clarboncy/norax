"""Numpy-backed kNN indexes — FlatIndex (batch) + IdMapIndex (incremental).

Not ANN — just dense matmul. For our expected scale (≤ 100k neurons)
this is faster than building/maintaining FAISS, and zero deps beyond numpy.
Breaks down somewhere around 1M vectors — problem for a later phase.

IdMapIndex is the production default. It maps entity_id → row index so
individual neurons can be added, removed, or updated without a full rebuild.
Deleted rows are NaN-padded and skipped during search; compaction runs
lazily when dead rows exceed 20%.
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
import zipfile
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .embeddings import EmbeddingError
from .store import Neuron

log = logging.getLogger("norax.memory.index")

_COMPACT_THRESHOLD = 0.20  # compact when >20% dead rows
_MAX_CACHE_BYTES = 1024 * 1024 * 1024


@contextmanager
def _cache_reader(path: Path) -> Iterator[BinaryIO]:
    """Validate archive and array sizes before NumPy allocates cached arrays."""
    if path.parent.is_symlink():
        raise ValueError("cache directory must not be a symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("cache must be a regular unlinked file")
        if info.st_size > _MAX_CACHE_BYTES:
            raise ValueError("cache exceeds byte limit")
        with zipfile.ZipFile(handle) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(members) > 16 or len(set(names)) != len(names):
                raise ValueError("cache has excessive or duplicate entries")
            if sum(member.file_size for member in members) > _MAX_CACHE_BYTES:
                raise ValueError("expanded cache exceeds byte limit")
            for name in ("vectors.npy", "entity_ids.npy", "dim.npy"):
                member = archive.getinfo(name)
                with archive.open(member) as array:
                    version = np.lib.format.read_magic(array)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(array)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(array)
                    else:
                        raise ValueError("unsupported cache array format")
                    if dtype.hasobject or dtype.itemsize <= 0:
                        raise ValueError("cache array has unsafe dtype")
                    size = dtype.itemsize
                    for dimension in shape:
                        size *= dimension
                    if size > _MAX_CACHE_BYTES or size != member.file_size - array.tell():
                        raise ValueError("cache array size does not match its header")
                    if name == "vectors.npy" and (
                        len(shape) != 2 or dtype.kind not in "fi" or shape[1] > 16384
                    ):
                        raise ValueError("invalid cache vector layout")
                    if name == "entity_ids.npy" and (len(shape) != 1 or dtype.kind not in "US"):
                        raise ValueError("invalid cache ID layout")
                    if name == "dim.npy" and (shape != () or dtype.kind not in "iu"):
                        raise ValueError("invalid cache dimension layout")
        handle.seek(0)
        yield handle


@contextmanager
def _cache_writer(path: Path) -> Iterator[BinaryIO]:
    """Publish a complete private cache, preserving the old file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError("cache directory must not be a symlink")
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("cache must be a regular unlinked file")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            yield handle
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _validated_embedding_batch(
    vectors: object,
    *,
    expected_rows: int,
    expected_dim: int,
    operation: str,
) -> np.ndarray:
    """Validate an embedding response before it can mutate an index."""
    try:
        batch = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(f"{operation}: embedding output is not a numeric array") from exc
    if batch.ndim != 2:
        raise EmbeddingError(f"{operation}: expected a 2D embedding batch, got shape {batch.shape}")
    if batch.shape != (expected_rows, expected_dim):
        raise EmbeddingError(
            f"{operation}: expected embedding shape {(expected_rows, expected_dim)}, "
            f"got {batch.shape}"
        )
    if not np.isfinite(batch).all():
        raise EmbeddingError(f"{operation}: embedding output contains non-finite values")
    if expected_rows and np.any(np.all(np.isclose(batch, 0.0), axis=1)):
        raise EmbeddingError(f"{operation}: embedding output contains a zero vector")
    return batch


def _default_cache_path() -> Path:
    """Resolve cache path at call-time (not import-time) so NORAX_MEMORY_ROOT
    changes (e.g. set by runtime before instantiating MemoryStore) are respected."""
    root = Path(os.environ.get("NORAX_MEMORY_ROOT", Path.cwd() / "memory")).expanduser()
    return root / "embed_cache.npz"


@dataclass
class FlatIndex:
    """Batch-built index. Rebuilds entirely on any change. Kept for backward compat."""

    neurons: list[Neuron] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _fingerprint: str = ""

    def fingerprint(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for n in self.neurons:
            h.update(f"{n.path}|{n.line}|{n.entity_id}".encode())
        return h.hexdigest()[:16]

    async def build(self, embedder, neurons: list[Neuron]) -> None:
        dim = int(getattr(embedder, "dim", 384))
        if dim <= 0:
            raise EmbeddingError(f"build: invalid embedding dimension {dim}")
        if not neurons:
            vectors = np.zeros((0, dim), dtype=np.float32)
        else:
            vectors = _validated_embedding_batch(
                await embedder.embed([n.text for n in neurons]),
                expected_rows=len(neurons),
                expected_dim=dim,
                operation="flat build",
            )
        self.neurons = list(neurons)
        self.vectors = vectors
        self._fingerprint = self.fingerprint()

    def search(self, qvec: np.ndarray, k: int = 5) -> list[tuple[Neuron, float]]:
        if not self.neurons or self.vectors.shape[0] == 0:
            return []
        query = np.asarray(qvec, dtype=np.float32)
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all() or k <= 0:
            return []
        scores = self.vectors @ query
        k = min(k, len(self.neurons))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.neurons[i], float(scores[i])) for i in top_idx]


@dataclass
class IdMapIndex:
    """Index with constant-time row lookup via entity_id mapping.

    Appending copies the dense matrix; deletion occasionally compacts it.

    neurons[i] and vectors[i] are paired. _id_to_idx maps entity_id → i.
    Deleted rows get vectors[i] = NaN and are skipped during search.
    Compaction (re-pack without dead rows) runs lazily when dead_ratio > 20%.
    """

    neurons: list[Neuron] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    _id_to_idx: dict[str, int] = field(default_factory=dict)
    _dead_count: int = 0
    _fingerprint: str = ""
    _dim: int = 384
    cache_path: Path | None = None  # if set, save()/load() use this path

    # ── build ──────────────────────────────────────────────────────────

    async def build(self, embedder, neurons: list[Neuron]) -> None:
        """Full rebuild from a neuron list. Clears all prior state."""
        dim = int(getattr(embedder, "dim", 384))
        if dim <= 0:
            raise EmbeddingError(f"build: invalid embedding dimension {dim}")
        if not neurons:
            vectors = np.zeros((0, dim), dtype=np.float32)
        else:
            vectors = _validated_embedding_batch(
                await embedder.embed([n.text for n in neurons]),
                expected_rows=len(neurons),
                expected_dim=dim,
                operation="build",
            )
        self.neurons = list(neurons)
        self.vectors = vectors
        self._dim = dim
        self._id_to_idx = {n.entity_id: i for i, n in enumerate(self.neurons)}
        self._dead_count = 0
        self._fingerprint = self._compute_fingerprint()
        self.save()

    # ── disk persistence ───────────────────────────────────────────────

    def save(self, path: Path | None = None) -> bool:
        """Persist vectors + neuron metadata to .npz for cold-start recovery.

        Stores: vectors (N×D float32), entity_ids (N str), paths (N str),
        lines (N int), weights (N float), mtimes (N float), kinds (N str),
        and the fingerprint. Returns True on success.
        """
        path = path or self.cache_path or _default_cache_path()
        try:
            if self._dead_count:
                # Tombstones are an in-memory optimization, not reusable embeddings.
                self._compact()
            with _cache_writer(path) as handle:
                np.savez(
                    handle,
                    vectors=self.vectors.astype(np.float32, copy=False),
                    entity_ids=np.array([n.entity_id for n in self.neurons], dtype=str),
                    paths=np.array([str(n.path) for n in self.neurons], dtype=str),
                    lines=np.array([n.line for n in self.neurons], dtype=np.int32),
                    weights=np.array([n.weight for n in self.neurons], dtype=np.float32),
                    mtimes=np.array([n.mtime for n in self.neurons], dtype=np.float64),
                    kinds=np.array([n.kind for n in self.neurons], dtype=str),
                    fingerprint=np.array(self.fingerprint(), dtype=str),
                    dim=np.array(self._dim, dtype=np.int32),
                )
            log.info("idmap_index.save: %d vectors → %s", len(self.neurons), path)
            return True
        except Exception as e:
            log.warning("idmap_index.save.failed: %r", e)
            return False

    def load(self, embedder, neurons: list[Neuron], path: Path | None = None) -> bool:
        """Load cached vectors from disk and reconcile with current neurons.

        Reuses vectors for neurons whose entity_id matches (content hash =
        same text). Only embeds the delta (new/changed neurons) via the
        embedder. Returns True if the cache was usable (even partially).

        This is a *sync* method — it does not call the embedder for matched
        neurons. Unmatched neurons are collected and returned to the caller
        for batch embedding via the embedder.
        """
        path = path or self.cache_path or _default_cache_path()
        if not path.exists():
            return False

        try:
            with _cache_reader(path) as handle, np.load(handle, allow_pickle=False) as data:
                cached_vectors = np.asarray(data["vectors"], dtype=np.float32)
                cached_ids = [str(value) for value in data["entity_ids"]]
                cached_dim = int(data["dim"])
        except Exception as e:
            log.warning("idmap_index.load.failed: %r", e)
            return False

        if cached_dim != getattr(embedder, "dim", 384):
            log.warning(
                "idmap_index.load: dim mismatch (cache=%d, embedder=%d) — skipping",
                cached_dim,
                getattr(embedder, "dim", 384),
            )
            return False

        if (
            cached_dim <= 0
            or cached_vectors.ndim != 2
            or cached_vectors.shape[1] != cached_dim
            or cached_vectors.shape[0] != len(cached_ids)
            or not np.isfinite(cached_vectors).all()
        ):
            log.warning("idmap_index.load: malformed or non-finite vector cache — skipping")
            return False

        if cached_vectors.shape[0] == 0:
            return False

        # Build lookup: entity_id → list of (position, vector) for duplicates
        cached_map: dict[str, deque[np.ndarray]] = {}
        for i, eid in enumerate(cached_ids):
            cached_map.setdefault(str(eid), deque()).append(cached_vectors[i])

        new_vectors = np.zeros((len(neurons), cached_dim), dtype=np.float32)
        matched = 0
        unmatched: list[int] = []

        for i, n in enumerate(neurons):
            candidates = cached_map.get(n.entity_id)
            if candidates:
                new_vectors[i] = candidates.popleft()
                matched += 1
            else:
                unmatched.append(i)

        if matched == 0:
            log.info("idmap_index.load: 0/%d matched — full rebuild needed", len(neurons))
            return False

        self.neurons = list(neurons)
        self.vectors = new_vectors
        self._dim = cached_dim
        self._id_to_idx = {n.entity_id: i for i, n in enumerate(self.neurons)}
        self._dead_count = 0
        self._fingerprint = self._compute_fingerprint()

        log.info(
            "idmap_index.load: %d/%d vectors recovered from cache, %d need embedding",
            matched,
            len(neurons),
            len(unmatched),
        )

        self._unmatched_indices = unmatched
        return True

    @property
    def unmatched_indices(self) -> list[int]:
        """Indices of neurons that still need embedding after a load()."""
        return getattr(self, "_unmatched_indices", [])

    async def embed_delta(self, embedder, neurons: list[Neuron]) -> bool:
        """Embed only the unmatched neurons after a load() call.

        Returns True if any embeddings were made.
        """
        unmatched = self.unmatched_indices
        if not unmatched:
            return False

        to_embed = [neurons[i] for i in unmatched]
        try:
            vecs = _validated_embedding_batch(
                await embedder.embed([n.text for n in to_embed]),
                expected_rows=len(to_embed),
                expected_dim=self._dim,
                operation="delta embed",
            )
        except Exception as e:
            log.warning("idmap_index.embed_delta.failed: %r", e)
            return False

        for idx, vec in zip(unmatched, vecs, strict=False):
            self.vectors[idx] = vec

        self._unmatched_indices = []
        self._fingerprint = ""  # invalidate
        self.save()
        log.info("idmap_index.embed_delta: embedded %d neurons, saved to cache", len(unmatched))
        return True

    # ── mutate ─────────────────────────────────────────────────────────

    async def add(self, embedder, neuron: Neuron) -> None:
        """Append one neuron + its embedding, copying the existing dense matrix.

        Raises EmbeddingError if the embedder fails — never persists a zero vector.
        """
        if neuron.entity_id in self._id_to_idx:
            # Already exists — update instead
            await self.update(embedder, neuron.entity_id, neuron)
            return
        embedder_dim = int(getattr(embedder, "dim", self._dim))
        if not self.neurons and self.vectors.shape[0] == 0:
            if embedder_dim <= 0:
                raise EmbeddingError(f"add: invalid embedding dimension {embedder_dim}")
            self._dim = embedder_dim
        vec = _validated_embedding_batch(
            await embedder.embed([neuron.text]),
            expected_rows=1,
            expected_dim=self._dim,
            operation=f"add {neuron.entity_id}",
        )
        if self.vectors.shape[0] == 0:
            self.vectors = vec
        else:
            self.vectors = np.vstack([self.vectors, vec])
        idx = len(self.neurons)
        self.neurons.append(neuron)
        self._id_to_idx[neuron.entity_id] = idx
        self._fingerprint = ""  # invalidate

    def remove(self, entity_id: str) -> bool:
        """Mark a neuron deleted, occasionally compacting. False if not found."""
        idx = self._id_to_idx.pop(entity_id, None)
        if idx is None:
            return False
        self.vectors[idx] = np.nan
        self._dead_count += 1
        self._fingerprint = ""
        if self._dead_ratio() > _COMPACT_THRESHOLD:
            self._compact()
        return True

    async def update(self, embedder, entity_id: str, new_neuron: Neuron) -> bool:
        """Replace neuron text + vector in-place. O(1). Returns False if not found.

        Raises EmbeddingError if the embedder fails — never persists a zero vector.
        """
        idx = self._id_to_idx.get(entity_id)
        if idx is None:
            return False
        if new_neuron.entity_id != entity_id and new_neuron.entity_id in self._id_to_idx:
            raise ValueError(f"duplicate entity_id: {new_neuron.entity_id}")
        vec = _validated_embedding_batch(
            await embedder.embed([new_neuron.text]),
            expected_rows=1,
            expected_dim=self._dim,
            operation=f"update {entity_id}",
        )
        self.vectors[idx] = vec[0]
        self.neurons[idx] = new_neuron
        # entity_id may have changed
        if new_neuron.entity_id != entity_id:
            del self._id_to_idx[entity_id]
            self._id_to_idx[new_neuron.entity_id] = idx
        self._fingerprint = ""
        return True

    # ── incremental sync ────────────────────────────────────────────────

    async def sync_from(self, embedder, neurons: list[Neuron]) -> bool:
        """Incremental sync: only embed new/changed neurons, remove gone ones.

        Uses entity_id (content hash) to detect changes. Reuses cached
        vectors for unchanged neurons. Returns True if any embed calls were made.

        Embedding work scales with the delta: on a 10k neuron store with 2-3
        changes, this embeds 2-3 vectors. Reconciliation, matrix copying, and
        persistence still scale with the full index.
        """
        new_ids = {n.entity_id for n in neurons}
        old_ids = set(self._id_to_idx.keys())

        # Compute the delta without mutating the current searchable snapshot.
        # If embedding fails, callers retain the complete previous index.
        gone = old_ids - new_ids
        to_embed: list[Neuron] = []
        for n in neurons:
            idx = self._id_to_idx.get(n.entity_id)
            if idx is None:
                to_embed.append(n)

        vecs: np.ndarray | None = None
        if to_embed:
            log.debug(
                "idmap_index.sync_from: embedding %d new neurons (of %d total)",
                len(to_embed),
                len(neurons),
            )
            try:
                vecs = _validated_embedding_batch(
                    await embedder.embed([n.text for n in to_embed]),
                    expected_rows=len(to_embed),
                    expected_dim=self._dim,
                    operation="incremental sync",
                )
            except Exception as e:
                log.warning("idmap_index.sync_from.embed_failed: %r", e)
                return False

        # Commit only after the entire embedding batch has validated.
        for eid in gone:
            self.remove(eid)
        for neuron in neurons:
            idx = self._id_to_idx.get(neuron.entity_id)
            if idx is not None:
                # entity_id is a content hash, so a match reuses the vector;
                # path/line/mtime metadata may still have changed.
                self.neurons[idx] = neuron

        if vecs is not None:
            # Append the batch once; copying the whole matrix for each new row
            # made large deltas quadratic despite using a batched embed request.
            self.vectors = vecs if self.vectors.shape[0] == 0 else np.vstack([self.vectors, vecs])
            for neuron in to_embed:
                self.neurons.append(neuron)
                self._id_to_idx[neuron.entity_id] = len(self.neurons) - 1

            self._fingerprint = ""  # invalidate
            self.save()
            return True

        # No new neurons to embed — just update fingerprint
        self._fingerprint = ""
        return False

    # ── search ─────────────────────────────────────────────────────────

    def search(self, qvec: np.ndarray, k: int = 5) -> list[tuple[Neuron, float]]:
        """kNN via matmul. Short-circuits NaN logic when no dead rows (common case)."""
        if not self.neurons or self.vectors.shape[0] == 0:
            return []
        query = np.asarray(qvec, dtype=np.float32)
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all() or k <= 0:
            return []
        scores = self.vectors @ query
        if self._dead_count > 0:
            dead_mask = np.isnan(self.vectors[:, 0])
            scores[dead_mask] = -np.inf
            alive = len(self.neurons) - self._dead_count
        else:
            alive = len(self.neurons)
        k = min(k, alive)
        if k <= 0:
            return []
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        if self._dead_count > 0:
            dead_mask = np.isnan(self.vectors[:, 0])
            return [(self.neurons[i], float(scores[i])) for i in top_idx if not dead_mask[i]]
        return [(self.neurons[i], float(scores[i])) for i in top_idx]

    # ── helpers ────────────────────────────────────────────────────────

    def _dead_ratio(self) -> float:
        total = len(self.neurons)
        return self._dead_count / total if total > 0 else 0.0

    def _compact(self) -> None:
        """Re-pack arrays without dead rows. Rebuilds _id_to_idx."""
        dead_mask = np.isnan(self.vectors[:, 0])
        alive = ~dead_mask
        if not alive.any():
            self.neurons = []
            self.vectors = np.zeros((0, self._dim), dtype=np.float32)
            self._id_to_idx = {}
            self._dead_count = 0
            return
        self.neurons = [n for n, a in zip(self.neurons, alive, strict=False) if a]
        self.vectors = self.vectors[alive]
        self._id_to_idx = {n.entity_id: i for i, n in enumerate(self.neurons)}
        self._dead_count = 0
        log.debug("idmap_index.compact: %d neurons after compaction", len(self.neurons))

    def _compute_fingerprint(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for n in self.neurons:
            h.update(f"{n.path}|{n.line}|{n.entity_id}".encode())
        return h.hexdigest()[:16]

    def fingerprint(self) -> str:
        if not self._fingerprint:
            self._fingerprint = self._compute_fingerprint()
        return self._fingerprint

    @property
    def alive_count(self) -> int:
        return len(self.neurons) - self._dead_count
