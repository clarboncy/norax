"""Phase 5 (memory layer) — stores, retrievers, injection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from norax.memory import (
    ContextInjector,
    ExternalRetriever,
    FastContext,
    LocalRetriever,
    MemoryStore,
    StubEmbedder,
)
from norax.memory.index import FlatIndex
from norax.memory.store import Neuron, _parse_weight

# ---------- store / parsing ------------------------------------------------


def test_weight_parsing():
    assert _parse_weight("OWNER:foo|W5") == 1.3
    assert _parse_weight("STYLE:bar") == 1.0
    assert _parse_weight("note:x|W1") == 0.4


def test_sleep_archive_not_indexed(tmp_path: Path):
    (tmp_path / "sleep").mkdir()
    (tmp_path / "sleep" / "archive").mkdir()
    (tmp_path / "sleep" / "processed_dumps").mkdir()
    (tmp_path / "sleep" / "spill-active.jsonl").write_text('{"content":"ACTIVE:note"}\n')
    (tmp_path / "sleep" / "archive" / "spill-old.jsonl").write_text(
        '{"content":"ARCHIVED:trace"}\n'
    )
    (tmp_path / "sleep" / "processed_dumps" / "intel_sleep-flush.md").write_text(
        "RAW tool trace junk\n"
    )
    store = MemoryStore(root=tmp_path)
    store.refresh()
    texts = [n.text for n in store.sleep]
    assert any("ACTIVE" in t for t in texts)
    assert not any("ARCHIVED" in t for t in texts)
    assert not any("tool trace junk" in t for t in texts)


def test_store_scans_and_parses(tmp_path: Path):
    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic" / "facts.md").write_text(
        "OWNER:Colby|W5\nTZ:America/New_York|W4\n# comment\n"
    )
    (tmp_path / "scratchpad.md").write_text("HOT:norax\nbuilding memory layer\n")
    store = MemoryStore(root=tmp_path)
    store.refresh()
    assert len(store.semantic) == 2
    assert len(store.hot) == 2
    assert store.semantic[0].weight == 1.3
    assert store.semantic[0].kind == "semantic"
    assert store.hot[0].kind == "scratchpad"


# ---------- FastContext ----------------------------------------------------


def test_fast_context_keyword_match(tmp_path: Path):
    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic" / "facts.md").write_text(
        "RUNTIME:Python 3.12;event_sourced|W3\nWALLET:EVM=0xecCBe...|W3\nOWNER:Colby|W5\n"
    )
    store = MemoryStore(root=tmp_path)
    store.refresh()
    fc = FastContext(store=store)
    hits = fc.search("tell me about my wallet", k=3)
    assert hits, "expected at least one hit for 'wallet'"
    assert "WALLET" in hits[0][0].text


# ---------- embeddings + index --------------------------------------------


@pytest.mark.asyncio
async def test_stub_embedder_is_deterministic():
    e = StubEmbedder()
    a = await e.embed(["hello world"])
    b = await e.embed(["hello world"])
    c = await e.embed(["different"])
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert a.shape == (1, 384)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-3


@pytest.mark.asyncio
async def test_flat_index_builds_and_searches():
    ns = [
        Neuron(text="OWNER:Colby|W5", path=Path("/x"), line=1, weight=1.3),
        Neuron(text="RUNTIME:Python", path=Path("/x"), line=2, weight=1.0),
        Neuron(text="WALLET:EVM", path=Path("/x"), line=3, weight=1.0),
    ]
    idx = FlatIndex()
    e = StubEmbedder()
    await idx.build(e, ns)
    q = await e.embed(["OWNER:Colby|W5"])
    hits = idx.search(q[0], k=2)
    assert len(hits) == 2
    assert hits[0][0].text == "OWNER:Colby|W5"  # exact-match dominates


# ---------- LocalRetriever --------------------------------------------------


@pytest.mark.asyncio
async def test_local_retriever_end_to_end(tmp_path: Path):
    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic" / "facts.md").write_text(
        "OWNER:Colby|W5\nRUNTIME:Python 3.12;event_sourced|W3\nWALLET:EVM=0xecCBe...|W3\n"
    )
    store = MemoryStore(root=tmp_path)
    store.refresh()
    lr = LocalRetriever(store=store, embedder=StubEmbedder())
    hits = await lr.search("OWNER:Colby|W5", k=2)
    assert hits and hits[0][0].text == "OWNER:Colby|W5"
    # Second call should NOT rebuild (fingerprint unchanged)
    built = await lr.refresh_if_stale()
    assert built is False


# ---------- ExternalRetriever ----------------------------------------------


@pytest.mark.asyncio
async def test_external_retriever_prefers_recent(tmp_path: Path):
    (tmp_path / "sleep").mkdir()
    (tmp_path / "sleep" / "recent.md").write_text("FRESH:hot_note\n")
    (tmp_path / "intel").mkdir()
    (tmp_path / "intel" / "old.md").write_text("STALE:old_note\n")
    # Artificially age the intel file
    import os
    import time

    ancient = time.time() - 7 * 86400
    os.utime(tmp_path / "intel" / "old.md", (ancient, ancient))
    store = MemoryStore(root=tmp_path)
    store.refresh()
    er = ExternalRetriever(store=store, embedder=StubEmbedder(), recency_half_life_days=1.0)
    # Query unrelated to either — recency should tie-break toward FRESH
    hits = await er.search("anything", k=2)
    assert hits
    top_text = hits[0][0].text
    assert top_text == "FRESH:hot_note"


# ---------- ContextInjector -----------------------------------------------


@pytest.mark.asyncio
async def test_context_injector_merges_and_dedupes(tmp_path: Path):
    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic" / "a.md").write_text("OWNER:Colby|W5\n")
    (tmp_path / "sleep").mkdir()
    (tmp_path / "sleep" / "b.md").write_text("OWNER:Colby|W5\n")  # dup
    (tmp_path / "sleep" / "c.md").write_text("FOCUS:building_memory|W4\n")
    (tmp_path / "scratchpad.md").write_text("HOT:norax phase 5\n")

    store = MemoryStore(root=tmp_path)
    store.refresh()
    emb = StubEmbedder()
    injector = ContextInjector(
        fast=FastContext(store=store),
        local=LocalRetriever(store=store, embedder=emb),
        external=ExternalRetriever(store=store, embedder=emb),
        budget_chars=500,
    )
    res = await injector.run("norax phase 5", k_each=3)
    # Dedup: "OWNER:Colby|W5" should appear exactly once even though
    # it lives in both semantic and sleep.
    texts = [n.text for n, _s, _src in res.items]
    assert texts.count("OWNER:Colby|W5") <= 1
    assert res.total_chars <= 500
    # At least one source contributed
    assert sum(res.counts.values()) > 0


@pytest.mark.asyncio
async def test_context_injector_embeds_shared_query_once(tmp_path: Path):
    class CountingEmbedder(StubEmbedder):
        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts):
            self.calls += 1
            return await super().embed(texts)

    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic" / "facts.md").write_text("FACT:canonical memory\n")
    (tmp_path / "sleep").mkdir()
    (tmp_path / "sleep" / "recent.md").write_text("FACT:recent memory\n")
    store = MemoryStore(root=tmp_path)
    store.refresh()
    embedder = CountingEmbedder()
    local = LocalRetriever(store=store, embedder=embedder)
    external = ExternalRetriever(store=store, embedder=embedder)
    await local.refresh_if_stale()
    await external.refresh_if_stale()
    embedder.calls = 0

    result = await ContextInjector(
        fast=FastContext(store=store),
        local=local,
        external=external,
    ).run("memory", k_each=3)

    assert result.items
    assert embedder.calls == 1
