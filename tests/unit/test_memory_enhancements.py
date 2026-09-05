"""Memory enhancements — entity graph, consolidator, multi-signal retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from norax.memory.consolidator import MemoryConsolidator
from norax.memory.entity_graph import EntityGraph, extract_entities, jaccard_overlap
from norax.memory.hot_inject import (
    capture_directive,
    merge_pinned,
    pin_hot_neurons,
    refresh_hot_identity,
    update_active_focus,
)
from norax.memory.retrievers.fast import FastContext
from norax.memory.retrievers.multi_signal import MultiSignalRetriever
from norax.memory.store import MemoryStore, Neuron

# ---------- entity_graph -----------------------------------------------------


def test_extract_entities_proper_nouns_and_tech():
    ents = extract_entities("Norax uses kimi-k2.6:cloud on localhost")
    assert any("norax" in e.lower() or "Norax" in e for e in ents)
    assert any("kimi" in e.lower() for e in ents)


def test_extract_entities_filters_stopwords():
    ents = extract_entities("yeah ok best done")
    assert not ents or all(e.lower() not in {"yeah", "ok", "best", "done"} for e in ents)


def test_jaccard_overlap():
    a = {"norax", "ollama", "kimi"}
    b = {"norax", "kimi", "glm"}
    assert jaccard_overlap(a, b) == pytest.approx(2 / 4)


def test_entity_graph_build_and_search(tmp_path: Path):
    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic" / "facts.md").write_text(
        "OLLAMA:port=11434;host=localhost|W4\n"
        "NORAX:kimi-k2.6 and glm-5.1 route through direct ollama|W4\n"
    )
    store = MemoryStore(root=tmp_path)
    store.refresh()
    neurons = store.all_canonical()
    graph = EntityGraph.from_store(tmp_path, neurons)
    assert graph.stats()["entities"] > 0
    hits = graph.search("tell me about kimi-k2.6 and glm-5.1", neurons, k=3)
    assert hits
    graph.save()
    assert (tmp_path / "entity_graph.json").exists()


# ---------- consolidator -----------------------------------------------------


def test_consolidator_add_only_and_dedup(tmp_path: Path):
    (tmp_path / "sleep").mkdir()
    (tmp_path / "semantic").mkdir()
    (tmp_path / "procedural").mkdir()
    spill = tmp_path / "sleep" / "spill-test.jsonl"
    # Realistic tool-trace spill (what the rolling window actually writes).
    spill.write_text(
        '{"kind":"tool_call","content":"{}","meta":{"name":"read"}}\n'
        '{"kind":"tool_result","content":"{\\"path\\":\\"config/runtime.jsonc\\",\\"ok\\":true}"}\n'
        '{"kind":"tool_call","content":"{}","meta":{"name":"search_memory"}}\n'
        '{"kind":"tool_result","content":"{\\"totalMatches\\":34,\\"ok\\":true,\\"source\\":\\"ollama\\"}"}\n'
    )
    # Backdate so min_age passes
    import os
    import time

    old = time.time() - 600
    os.utime(spill, (old, old))

    c = MemoryConsolidator(memory_root=tmp_path, min_age_sec=300.0)
    r1 = c.consolidate()
    assert r1.facts_written + r1.procedural_written > 0

    # Second run: identical content → dedup
    spill2 = tmp_path / "sleep" / "spill-test2.jsonl"
    spill2.write_text(
        '{"kind":"tool_call","content":"{}","meta":{"name":"search_memory"}}\n'
        '{"kind":"tool_result","content":"{\\"totalMatches\\":34,\\"ok\\":true,\\"source\\":\\"ollama\\"}"}\n'
    )
    os.utime(spill2, (old, old))
    r2 = c.consolidate()
    assert r2.skipped_dup >= 1
    assert not spill2.exists()
    assert any((tmp_path / "sleep" / "archive").glob("spill-test2*.jsonl"))


def test_consolidator_archives_empty_processed_buffer(tmp_path: Path):
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    empty = sleep / "buffer-empty.md"
    empty.write_text("# no durable facts\n")

    c = MemoryConsolidator(memory_root=tmp_path, min_age_sec=0)
    result = c.consolidate()

    assert result.files_processed == 1
    assert len(result.archived) == 1
    assert not empty.exists()
    assert (sleep / "archive" / empty.name).exists()


def test_consolidator_dry_run_does_not_poison_live_dedup(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    sleep = root / "sleep"
    sleep.mkdir(parents=True)
    source = sleep / "buffer-audit.md"
    source.write_text("AUDIT_RESULT:all capability receipts verified|W5\n")
    consolidator = MemoryConsolidator(root, min_age_sec=0)

    preview = consolidator.consolidate(dry_run=True)
    committed = consolidator.consolidate()

    assert preview.facts_written == 1
    assert committed.facts_written == 1
    assert not source.exists()
    assert "AUDIT_RESULT" in next((root / "semantic").glob("consolidated-*.md")).read_text()


def test_consolidator_preserves_prior_archive_on_name_collision(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    sleep = root / "sleep"
    archive = sleep / "archive"
    archive.mkdir(parents=True)
    source = sleep / "buffer-repeat.md"
    source.write_text("NEW_FACT:second buffer incarnation is retained|W4\n")
    prior = archive / source.name
    prior.write_text("prior audit artifact\n")

    result = MemoryConsolidator(root, min_age_sec=0).consolidate()

    assert result.archived == [str(source)]
    assert prior.read_text() == "prior audit artifact\n"
    versions = list(archive.glob("buffer-repeat*.md"))
    assert len(versions) == 2
    assert any("NEW_FACT" in path.read_text() for path in versions)


def test_consolidator_ignores_linked_input(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    sleep = root / "sleep"
    sleep.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("FORGED_FACT:do not ingest linked authority|W5\n")
    (sleep / "buffer-linked.md").symlink_to(outside)

    result = MemoryConsolidator(root, min_age_sec=0).consolidate()

    assert result.files_processed == 0
    assert outside.exists()
    assert not (root / "semantic").exists()


# ---------- multi_signal ---------------------------------------------------


def test_multi_signal_fusion_tags(tmp_path: Path):
    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic" / "facts.md").write_text(
        "WALLET:EVM=0x0000000000000000000000000000000000000001|W5\n"
        "RUNTIME:Python 3.12;event_sourced|W3\n"
    )
    store = MemoryStore(root=tmp_path)
    store.refresh()
    fc = FastContext(store=store)
    graph = EntityGraph.from_store(tmp_path, store.all_canonical())
    ms = MultiSignalRetriever(keyword=fc, entity_graph=graph)
    hits = ms.search_sync("wallet EVM address", k=3)
    assert hits
    assert any("WALLET" in n.text for n, _, _ in hits)
    tags = {tag for _, _, tag in hits}
    assert any("kw" in t or "ent" in t for t in tags)


@pytest.mark.asyncio
async def test_multi_signal_async_without_embedder(tmp_path: Path):
    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic" / "facts.md").write_text("OWNER:Colby|W5\n")
    store = MemoryStore(root=tmp_path)
    store.refresh()
    fc = FastContext(store=store)
    graph = EntityGraph.from_store(tmp_path, store.all_canonical())
    ms = MultiSignalRetriever(keyword=fc, entity_graph=graph)
    await ms.ensure_index()
    hits = await ms.search("Colby owner", k=2)
    assert hits
    assert hits[0][2]  # source tag present


# ---------- hot_inject -----------------------------------------------------


def test_pin_hot_neurons_always_includes_state(tmp_path: Path):
    (tmp_path / "scratchpad.md").write_text(
        "SCRATCHPAD;updated=2026-06-08;type=hot_memory\n"
        "IDENTITY:Norax production|W5\n"
        "TURN:12:00|in=hello|out=hi|tools=0|rounds=1\n"
    )
    (tmp_path / "active-focus.md").write_text("FOCUS:sync staging with prod\n")
    store = MemoryStore(root=tmp_path)
    store.refresh()
    pinned = pin_hot_neurons(store)
    texts = {n.text for n, _, _ in pinned}
    assert any(t.startswith("IDENTITY:") for t in texts)
    assert any(t.startswith("TURN:") for t in texts)


def test_merge_pinned_prepends_hot(tmp_path: Path):
    n1 = Neuron(text="IDENTITY:Norax", path=tmp_path / "scratchpad.md", line=2, kind="scratchpad")
    n2 = Neuron(
        text="WALLET:EVM=0xabc", path=tmp_path / "semantic" / "x.md", line=1, kind="semantic"
    )
    pinned = [(n1, 1.3, "hot")]
    hits = [(n2, 0.5, "kw")]
    merged = merge_pinned(hits, pinned, k=2)
    assert merged[0][0].text.startswith("IDENTITY:")


def test_capture_directive_writes_semantic(tmp_path: Path):
    (tmp_path / "semantic").mkdir()
    p = capture_directive(
        tmp_path, "Remember that port 4101 is Norax runtime", pathway="procedural_update"
    )
    assert p is not None
    assert "DIRECTIVE:" in p.read_text()


def test_refresh_hot_identity_replaces_stale_gen5(tmp_path: Path):
    (tmp_path / "scratchpad.md").write_text(
        "SCRATCHPAD;updated=2026-04-28;type=hot_memory\n"
        "IDENTITY:Norax Gen5 — old identity line\n"
        "TURN:01:00|in=ok|out=done|tools=0|rounds=1\n"
    )
    refresh_hot_identity(tmp_path, generation=7, role="production")
    text = (tmp_path / "scratchpad.md").read_text()
    assert "Norax" in text
    assert "Gen5" not in text
    assert "TURN:01:00" in text
    assert "health=unverified" in text
    assert " active" not in text.lower()


def test_update_active_focus_persists(tmp_path: Path):
    update_active_focus(tmp_path, "[command] sync staging-node staging")
    body = (tmp_path / "active-focus.md").read_text()
    assert "sync staging-node staging" in body


def test_update_active_focus_replaces_stale_current(tmp_path: Path):
    (tmp_path / "active-focus.md").write_text(
        "# active-focus;updated=old\nFOCUS:old task\nCURRENT:old task\nNEXT:old step\n"
    )
    update_active_focus(tmp_path, "[command] repair context lifecycle")
    body = (tmp_path / "active-focus.md").read_text()
    assert "CURRENT:[command] repair context lifecycle" in body
    assert "CURRENT:old task" not in body
    assert "NEXT:old step" not in body


def test_is_tool_dump_recognizes_tool_io():
    from norax.memory.consolidator import is_tool_dump

    # Raw tool I/O — must be rejected
    assert is_tool_dump('{"ok": true, "exit_code": 0, "stdout": "foo\\nbar"}')
    assert is_tool_dump('{"path": "/srv/agent/x.py", "limit": 240, "offset": 1200}')
    assert is_tool_dump('{"query": "memory search", "k": 5}')
    assert is_tool_dump('["a", "b"]')
    assert is_tool_dump('{"command": "ls -la", "ok": true')  # truncated fragment

    # Real facts — must survive
    assert not is_tool_dump("DIRECTIVE:12:00|always prefer local models|W5")
    assert not is_tool_dump("TURN:01:46|in=x-ray goggles|out=Pack loaded")
    assert not is_tool_dump("STATUS:norax-ai live; norax-ai active|W5")
    assert not is_tool_dump("Deployed fix for {path traversal} in server.py|W4")
    assert not is_tool_dump("")


def test_extract_from_text_rejects_tool_dumps():
    from norax.memory.consolidator import _extract_from_text

    text = (
        '{"ok": true, "exit_code": 0, "stdout": "norax/core.py\\nnorax/run.py"}\n'
        '{"path": "/srv/agent/norax/server.py", "limit": 240, "offset": 0}\n'
        "PORTS:gateway=8899 runtime=4101 embed=11436|W5\n"
    )
    semantic, procedural = _extract_from_text(text)
    all_lines = semantic + procedural
    assert not any('"ok"' in line or '"path"' in line for line in all_lines)
    assert any("PORTS:" in line for line in all_lines)


def test_strip_tool_dumps_preserves_facts(tmp_path: Path):
    from norax.memory.consolidator import strip_tool_dumps

    f = tmp_path / "consolidated-2026-06-20.md"
    f.write_text(
        "# consolidated 2026-06-20\n"
        "FACT:embeddings run locally on :11436|W4\n"
        '{"ok": true, "path": "/x.py", "content": "..."}\n'
        '{"command": "find . -name *.py", "ok": true}\n'
        "VERIFY:memory index rebuilt after sweep|W3\n"
    )
    removed = strip_tool_dumps(f)
    assert removed == 2
    body = f.read_text()
    assert "FACT:embeddings" in body
    assert "VERIFY:memory index" in body
    assert '"ok"' not in body
    assert '"command"' not in body

    # Clean file is untouched (no rewrite, returns 0)
    before = f.read_text()
    assert strip_tool_dumps(f) == 0
    assert f.read_text() == before
