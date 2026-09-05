"""Integrity and persistence tests for production memory vector indexes."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from norax.memory.embeddings import EmbeddingError
from norax.memory.index import FlatIndex, IdMapIndex
from norax.memory.store import Neuron


class _Embedder:
    dim = 3

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = [self.vectors.get(text, [float(len(text)), 1.0, 0.5]) for text in texts]
        return np.asarray(rows, dtype=np.float32)


class _FixedEmbedder:
    dim = 3

    def __init__(self, value: object) -> None:
        self.value = value

    async def embed(self, _texts: list[str]):
        return self.value


@pytest.mark.asyncio
async def test_incremental_batch_copies_matrix_once_and_reloads_after_deletion(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache.npz"
    original = [_neuron(f"old-{i}") for i in range(20)]
    added = [_neuron(f"new-{i}") for i in range(50)]
    index = IdMapIndex(cache_path=cache)
    embedder = _Embedder()
    await index.build(embedder, original)
    calls = []
    real_vstack = np.vstack

    def measured_vstack(arrays):
        calls.append(len(arrays))
        return real_vstack(arrays)

    monkeypatch.setattr(np, "vstack", measured_vstack)
    current = original[1:] + added
    assert await index.sync_from(embedder, current)
    assert calls == [2]
    assert embedder.calls[-1] == [n.text for n in added]
    recovered = IdMapIndex(cache_path=cache)
    assert recovered.load(embedder, current)
    assert recovered.unmatched_indices == []
    assert np.isfinite(recovered.vectors).all()
    assert original[0].entity_id not in recovered._id_to_idx


@pytest.mark.asyncio
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
async def test_cache_load_rejects_linked_file(tmp_path, link_kind):
    source = tmp_path / "source.npz"
    neuron = _neuron("original")
    await IdMapIndex(cache_path=source).build(_Embedder(), [neuron])
    linked = tmp_path / "linked.npz"
    if link_kind == "symlink":
        linked.symlink_to(source)
    else:
        linked.hardlink_to(source)
    index = IdMapIndex(cache_path=linked)
    assert index.load(_Embedder(), [neuron]) is False
    assert index.neurons == []


def test_cache_rejects_forged_array_size_before_numpy_load(tmp_path, monkeypatch):
    cache = tmp_path / "forged.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 3)}
    )
    with zipfile.ZipFile(cache, "w") as archive:
        archive.writestr("vectors.npy", header.getvalue())

    def forbidden(*args, **kwargs):
        pytest.fail("NumPy must not allocate before archive validation")

    monkeypatch.setattr(np, "load", forbidden)
    assert IdMapIndex(cache_path=cache).load(_Embedder(), [_neuron("original")]) is False


def test_cache_byte_limit_checked_before_archive_decode(tmp_path, monkeypatch):
    import norax.memory.index as module

    cache = tmp_path / "oversized.npz"
    cache.write_bytes(b"x" * 128)
    monkeypatch.setattr(module, "_MAX_CACHE_BYTES", 64)
    assert IdMapIndex(cache_path=cache).load(_Embedder(), [_neuron("original")]) is False


@pytest.mark.asyncio
async def test_failed_cache_save_preserves_previous_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache.npz"
    index = IdMapIndex(cache_path=cache)
    await index.build(_Embedder(), [_neuron("original")])
    previous = cache.read_bytes()

    def interrupted(handle, **kwargs):
        handle.write(b"partial archive")
        raise OSError("interrupted")

    monkeypatch.setattr(np, "savez", interrupted)
    assert index.save() is False
    assert cache.read_bytes() == previous
    assert list(tmp_path.glob(".cache.npz.*")) == []
    assert cache.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_cache_save_rejects_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"keep")
    cache = tmp_path / "cache.npz"
    cache.symlink_to(target)
    assert IdMapIndex(cache_path=cache).save() is False
    assert target.read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_duplicate_cache_recovery_consumes_each_vector_once(tmp_path):
    cache = tmp_path / "cache.npz"
    neurons = [_neuron("same", line=i) for i in range(100)]
    index = IdMapIndex(cache_path=cache)
    await index.build(_Embedder(), neurons)
    recovered = IdMapIndex(cache_path=cache)
    assert recovered.load(_Embedder(), neurons + [_neuron("same", line=101)])
    assert recovered.unmatched_indices == [100]


def _neuron(text: str, *, line: int = 1, entity_id: str = "") -> Neuron:
    return Neuron(
        text=text,
        path=Path("semantic/facts.md"),
        line=line,
        entity_id=entity_id,
    )


@pytest.mark.asyncio
async def test_flat_index_build_search_and_invalid_queries_are_bounded() -> None:
    neurons = [_neuron("alpha"), _neuron("beta", line=2)]
    embedder = _Embedder({"alpha": [1, 0, 0], "beta": [0, 1, 0]})
    index = FlatIndex()

    await index.build(embedder, neurons)

    assert [n.text for n, _score in index.search(np.array([0.9, 0.1, 0]), k=2)] == [
        "alpha",
        "beta",
    ]
    assert index.search(np.array([1, 0]), k=1) == []
    assert index.search(np.array([np.nan, 0, 0]), k=1) == []
    assert index.search(np.array([1, 0, 0]), k=0) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([[1, 2]], dtype=np.float32),
        np.array([[0, 0, 0]], dtype=np.float32),
        np.array([[np.nan, 1, 2]], dtype=np.float32),
        "not-an-array",
    ],
)
async def test_invalid_build_is_rejected_without_corrupting_existing_index(invalid) -> None:
    original = _neuron("original")
    index = IdMapIndex(
        neurons=[original],
        vectors=np.array([[1, 0, 0]], dtype=np.float32),
        _id_to_idx={original.entity_id: 0},
        _dim=3,
    )

    with pytest.raises(EmbeddingError):
        await index.build(_FixedEmbedder(invalid), [_neuron("replacement")])

    assert index.neurons == [original]
    assert np.array_equal(index.vectors, np.array([[1, 0, 0]], dtype=np.float32))
    assert index._id_to_idx == {original.entity_id: 0}


@pytest.mark.asyncio
async def test_cache_round_trip_is_pickle_free_and_delta_only(tmp_path) -> None:
    cache = tmp_path / "index.npz"
    first = _neuron("first")
    second = _neuron("second", line=2)
    initial_embedder = _Embedder({"first": [1, 0, 0], "second": [0, 1, 0]})
    built = IdMapIndex(cache_path=cache)
    await built.build(initial_embedder, [first, second])

    # The cache must be loadable with pickle disabled.
    with np.load(cache, allow_pickle=False) as payload:
        assert payload["entity_ids"].dtype.kind in {"U", "S"}
        assert payload["paths"].dtype.kind in {"U", "S"}

    third = _neuron("third", line=3)
    delta_embedder = _Embedder({"third": [0, 0, 1]})
    recovered = IdMapIndex(cache_path=cache)
    assert recovered.load(delta_embedder, [first, third]) is True
    assert recovered.unmatched_indices == [1]

    assert await recovered.embed_delta(delta_embedder, [first, third]) is True
    assert delta_embedder.calls == [["third"]]
    assert recovered.unmatched_indices == []
    assert recovered.search(np.array([0, 0, 1], dtype=np.float32), k=1)[0][0].text == "third"


@pytest.mark.parametrize(
    "vectors,ids,dim",
    [
        (np.array([[1, 2]], dtype=np.float32), np.array(["a"]), 3),
        (np.array([[1, 2, 3]], dtype=np.float32), np.array(["a", "b"]), 3),
        (np.array([[np.inf, 0, 0]], dtype=np.float32), np.array(["a"]), 3),
    ],
)
def test_malformed_vector_cache_is_rejected(tmp_path, vectors, ids, dim) -> None:
    cache = tmp_path / "malformed.npz"
    np.savez(cache, vectors=vectors, entity_ids=ids, dim=np.array(dim, dtype=np.int32))

    index = IdMapIndex(cache_path=cache)

    assert index.load(_Embedder(), [_neuron("first")]) is False
    assert index.neurons == []
    assert index.vectors.shape == (0, 0)


def test_legacy_pickle_cache_is_rejected_instead_of_deserialized(tmp_path) -> None:
    cache = tmp_path / "legacy.npz"
    np.savez(
        cache,
        vectors=np.array([[1, 0, 0]], dtype=np.float32),
        entity_ids=np.array(["legacy"], dtype=object),
        dim=np.array(3, dtype=np.int32),
    )

    assert IdMapIndex(cache_path=cache).load(_Embedder(), [_neuron("first")]) is False


@pytest.mark.asyncio
async def test_add_update_remove_and_compaction_keep_id_map_consistent() -> None:
    neurons = [_neuron(f"n{i}", entity_id=f"id-{i}") for i in range(4)]
    embedder = _Embedder({n.text: [float(i + 1), 1, 1] for i, n in enumerate(neurons)})
    index = IdMapIndex()
    await index.build(embedder, neurons)

    replacement = _neuron("replacement", entity_id="replacement-id")
    embedder.vectors["replacement"] = [9, 1, 1]
    assert await index.update(embedder, "id-0", replacement) is True
    assert "id-0" not in index._id_to_idx
    assert index._id_to_idx["replacement-id"] == 0

    duplicate = _neuron("duplicate", entity_id="id-1")
    with pytest.raises(ValueError, match="duplicate entity_id"):
        await index.update(embedder, "replacement-id", duplicate)

    # One deletion out of four crosses the 20% threshold and compacts.
    assert index.remove("id-3") is True
    assert index.remove("missing") is False
    assert index._dead_count == 0
    assert index.alive_count == 3
    assert index.vectors.shape == (3, 3)
    assert set(index._id_to_idx) == {"replacement-id", "id-1", "id-2"}


@pytest.mark.asyncio
async def test_add_to_empty_index_adopts_embedder_dimension() -> None:
    index = IdMapIndex()
    neuron = _neuron("first")

    await index.add(_Embedder({"first": [1, 2, 3]}), neuron)

    assert index._dim == 3
    assert index.vectors.shape == (1, 3)
    assert index.alive_count == 1


@pytest.mark.asyncio
async def test_failed_incremental_batch_leaves_previous_snapshot_intact(tmp_path) -> None:
    old = _neuron("old", entity_id="old")
    index = IdMapIndex(
        neurons=[old],
        vectors=np.array([[1, 0, 0]], dtype=np.float32),
        _id_to_idx={"old": 0},
        _dim=3,
        cache_path=tmp_path / "index.npz",
    )
    new = _neuron("new", entity_id="new")

    # Two requested neurons but one returned row: the whole update is rejected.
    changed = await index.sync_from(
        _FixedEmbedder(np.array([[0, 1, 0]], dtype=np.float32)),
        [new, _neuron("another", entity_id="another")],
    )

    assert changed is False
    assert index.neurons == [old]
    assert index._id_to_idx == {"old": 0}
    assert np.array_equal(index.vectors, np.array([[1, 0, 0]], dtype=np.float32))


@pytest.mark.asyncio
async def test_incremental_sync_reuses_existing_vectors_and_embeds_only_new(tmp_path) -> None:
    old = _neuron("old", line=1, entity_id="old")
    index = IdMapIndex(
        neurons=[old],
        vectors=np.array([[1, 0, 0]], dtype=np.float32),
        _id_to_idx={"old": 0},
        _dim=3,
        cache_path=tmp_path / "index.npz",
    )
    moved = _neuron("old", line=99, entity_id="old")
    new = _neuron("new", entity_id="new")
    embedder = _Embedder({"new": [0, 1, 0]})

    assert await index.sync_from(embedder, [moved, new]) is True

    assert embedder.calls == [["new"]]
    assert index.neurons[index._id_to_idx["old"]].line == 99
    assert set(index._id_to_idx) == {"old", "new"}
    assert index.search(np.array([0, 1, 0], dtype=np.float32), k=1)[0][0].entity_id == "new"
