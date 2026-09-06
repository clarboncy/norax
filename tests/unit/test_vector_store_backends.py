from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from norax.memory.retrievers import vector_store as stores


def test_numpy_vector_store_is_dimension_safe_and_upserts():
    store = stores.NumpyVectorStore()
    assert store.search([1, 0]) == []
    store.add("a", [1, 0], {"version": 1})
    store.add("b", [0, 1], {"version": 1})
    store.add("a", [0.8, 0.2], {"version": 2})
    assert store.count() == 2 and store.dim == 2
    results = store.search([1, 0], k=10)
    assert [result.id for result in results] == ["a", "b"]
    assert results[0].metadata == {"version": 2}
    assert store.search([1, 0], k=0) == []
    with pytest.raises(TypeError, match="integer"):
        store.search([1, 0], k=True)
    with pytest.raises(ValueError, match="dimension"):
        store.search([1])
    with pytest.raises(ValueError, match="dimension"):
        store.add("c", [1])
    for invalid in ([], [float("nan")], [float("inf")], ["bad"]):
        with pytest.raises(ValueError, match="vector"):
            stores.NumpyVectorStore().add("bad", invalid)
    assert store.remove("a") is True
    assert store.remove("missing") is False
    store.clear()
    assert store.count() == 0
    for invalid_dim in (-1, True, 1.5):
        with pytest.raises(ValueError, match="dim"):
            stores.NumpyVectorStore(invalid_dim)


class _QdrantClient:
    collection_names: list[str] = []

    def __init__(self, url):
        self.url = url
        self.created = []
        self.upserts = []
        self.deletes = []
        self.deleted_collections = []
        self.fail = set()

    def get_collections(self):
        if "collections" in self.fail:
            raise RuntimeError("offline")
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collection_names]
        )

    def create_collection(self, name, vectors_config):
        self.created.append((name, vectors_config))

    def upsert(self, name, points):
        if "upsert" in self.fail:
            raise RuntimeError("write failed")
        self.upserts.append((name, points))

    def search(self, name, query_vector, limit):
        if "search" in self.fail:
            raise RuntimeError("query failed")
        return [
            SimpleNamespace(id="uuid", score=0.9, payload={"_norax_id": "entity", "kind": "fact"}),
            SimpleNamespace(id="legacy", score=0.5, payload=None),
        ]

    def count(self, name):
        if "count" in self.fail:
            raise RuntimeError("count failed")
        return SimpleNamespace(count=7)

    def delete(self, name, points_selector):
        if "delete" in self.fail:
            raise RuntimeError("delete failed")
        self.deletes.append((name, points_selector))

    def delete_collection(self, name):
        if "clear" in self.fail:
            raise RuntimeError("clear failed")
        self.deleted_collections.append(name)


class _Model:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def fake_qdrant(monkeypatch):
    package = types.ModuleType("qdrant_client")
    models = types.ModuleType("qdrant_client.models")
    package.QdrantClient = _QdrantClient
    models.Distance = SimpleNamespace(COSINE="cosine")
    models.VectorParams = _Model
    models.PointStruct = _Model
    models.PointIdsList = _Model
    monkeypatch.setitem(sys.modules, "qdrant_client", package)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)
    _QdrantClient.collection_names = []
    return package, models


def test_qdrant_backend_maps_ids_and_handles_success(fake_qdrant):
    store = stores.QdrantVectorStore(collection="facts", url="http://qdrant", dim=2)
    assert store.client.url == "http://qdrant"
    assert store.client.created[0][1].size == 2
    store.add("arbitrary/entity/id", [1, 0], {"kind": "fact", "_norax_id": "spoof"})
    point = store.client.upserts[0][1][0]
    assert point.id == stores._qdrant_point_id("arbitrary/entity/id")
    assert point.payload["_norax_id"] == "arbitrary/entity/id"
    results = store.search([1, 0], k=2)
    assert results[0] == stores.VectorSearchResult("entity", 0.9, {"kind": "fact"})
    assert results[1] == stores.VectorSearchResult("legacy", 0.5, {})
    assert store.count() == 7
    assert store.remove("arbitrary/entity/id") is True
    selector = store.client.deletes[0][1]
    assert selector.points == [stores._qdrant_point_id("arbitrary/entity/id")]
    store.clear()
    assert store.client.deleted_collections == ["facts"]
    assert store.search([1, 0], k=0) == []
    with pytest.raises(TypeError, match="integer"):
        store.search([1, 0], k=True)
    with pytest.raises(ValueError, match="dimension"):
        store.search([1], k=1)
    with pytest.raises(ValueError, match="dimension"):
        store.add("invalid", [1])
    for invalid_dim in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="dim"):
            stores.QdrantVectorStore(dim=invalid_dim)


@pytest.mark.parametrize(
    ("failure", "operation", "expected"),
    [
        ("collections", lambda store: store._ensure_collection(), None),
        ("upsert", lambda store: store.add("id", [1, 0]), None),
        ("search", lambda store: store.search([1, 0]), []),
        ("count", lambda store: store.count(), 0),
        ("delete", lambda store: store.remove("id"), False),
        ("clear", lambda store: store.clear(), None),
    ],
)
def test_qdrant_backend_degrades_truthfully(fake_qdrant, failure, operation, expected):
    _QdrantClient.collection_names = ["facts"]
    store = stores.QdrantVectorStore(collection="facts", dim=2)
    store.client.fail.add(failure)
    assert operation(store) == expected


def test_missing_qdrant_dependency_reports_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(ImportError, match="qdrant-client not installed"):
        stores.QdrantVectorStore()


class _ChromaCollection:
    def __init__(self):
        self.upserts = []
        self.deletes = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query(self, **kwargs):
        return {"ids": [["a", "b"]], "distances": [[0.1, 0.4]], "metadatas": [[{"x": 1}]]}

    def count(self):
        return 2

    def delete(self, **kwargs):
        self.deletes.append(kwargs)


class _ChromaClient:
    def __init__(self, path=None):
        self.path = path
        self.collections = {}
        self.deleted = []

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, _ChromaCollection())

    def delete_collection(self, name):
        self.deleted.append(name)
        self.collections.pop(name, None)


@pytest.fixture
def fake_chroma(monkeypatch):
    package = types.ModuleType("chromadb")
    package.Client = lambda: _ChromaClient()
    package.PersistentClient = lambda path: _ChromaClient(path)
    monkeypatch.setitem(sys.modules, "chromadb", package)
    return package


@pytest.mark.parametrize("persistent", [False, True])
def test_chroma_backend_lifecycle(fake_chroma, tmp_path, persistent):
    store = stores.ChromaVectorStore(persist_dir=str(tmp_path) if persistent else "")
    assert bool(store.client.path) is persistent
    store.add("a", [1, 0], {"x": 1})
    assert store.collection.upserts[0]["ids"] == ["a"]
    assert store.search([1, 0]) == [
        stores.VectorSearchResult("a", 0.9, {"x": 1}),
        stores.VectorSearchResult("b", 0.6, {}),
    ]
    assert store.count() == 2
    assert store.remove("a") is True
    original = store.collection
    store.clear()
    assert store.client.deleted == ["norax_memory"]
    assert store.collection is not original
    assert store.search([1, 0], k=0) == []
    with pytest.raises(TypeError, match="integer"):
        store.search([1, 0], k=False)
    with pytest.raises(ValueError, match="vector"):
        store.add("bad", [float("nan")])


def test_missing_chroma_dependency_reports_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(ImportError, match="chromadb not installed"):
        stores.ChromaVectorStore()


def test_backend_selection_explicit_and_fallbacks(monkeypatch, tmp_path):
    assert isinstance(stores.get_vector_store(backend="numpy"), stores.NumpyVectorStore)

    marker = object()
    monkeypatch.setattr(stores, "QdrantVectorStore", lambda **kwargs: marker)
    assert stores.get_vector_store(backend="qdrant", dim=2) is marker

    def unavailable(**kwargs):
        raise ImportError

    monkeypatch.setattr(stores, "QdrantVectorStore", unavailable)
    assert isinstance(stores.get_vector_store(backend="qdrant"), stores.NumpyVectorStore)
    monkeypatch.setattr(stores, "ChromaVectorStore", lambda **kwargs: ("chroma", kwargs))
    selected = stores.get_vector_store(tmp_path, backend="chroma", collection="facts")
    assert selected == ("chroma", {"persist_dir": str(tmp_path / "chroma"), "collection": "facts"})

    monkeypatch.setattr(stores, "ChromaVectorStore", unavailable)
    assert isinstance(stores.get_vector_store(tmp_path, backend="chroma"), stores.NumpyVectorStore)
    with pytest.raises(ValueError, match="unsupported"):
        stores.get_vector_store(backend="typo")


def test_auto_backend_order_and_total_fallback(monkeypatch, tmp_path):
    qdrant = object()
    monkeypatch.setattr(stores, "QdrantVectorStore", lambda **kwargs: qdrant)
    assert stores.get_vector_store(tmp_path) is qdrant

    def missing(**kwargs):
        raise ImportError

    chroma = object()
    monkeypatch.setattr(stores, "QdrantVectorStore", missing)
    monkeypatch.setattr(stores, "ChromaVectorStore", lambda **kwargs: chroma)
    assert stores.get_vector_store(tmp_path) is chroma
    assert stores.get_vector_store() is chroma

    def broken(**kwargs):
        raise RuntimeError("broken backend")

    monkeypatch.setattr(stores, "QdrantVectorStore", broken)
    monkeypatch.setattr(stores, "ChromaVectorStore", broken)
    assert isinstance(stores.get_vector_store(tmp_path), stores.NumpyVectorStore)
