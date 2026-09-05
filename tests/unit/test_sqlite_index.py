"""Tests for SQLite FTS5 memory index and retriever."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from norax.memory.build_index import build_memory_index
from norax.memory.retrievers.sqlite_index import SQLiteIndexRetriever


@pytest.fixture
def memory_root(tmp_path: Path) -> Path:
    """Create a minimal memory tree for testing."""
    sem = tmp_path / "semantic"
    sem.mkdir()
    (sem / "test_facts.md").write_text(
        "# test facts\nWALLET_EVM:0xTest123\nEMAIL:test@example.com\n"
    )
    proc = tmp_path / "procedural"
    proc.mkdir()
    (proc / "test_procs.md").write_text(
        "# test procs\nRULE:always verify after write\nFAILURE:edit without read first\n"
    )
    intel = tmp_path / "intel"
    intel.mkdir()
    (intel / "test_intel.md").write_text("MEDIA_SERVER:hostname=testhost;ip=10.0.0.1\n")
    (tmp_path / "scratchpad.md").write_text("HOT:current task is testing\n")
    (tmp_path / "active-focus.md").write_text("FOCUS:test focus\n")
    # Entity graph
    graph = {
        "fingerprint": "test",
        "entity_count": 2,
        "link_count": 1,
        "neuron_count": 2,
        "entities": {"wallet": ["abc123"], "email": ["def456"]},
        "neuron_entities": {},
        "links": [["wallet", "email"]],
    }
    (tmp_path / "entity_graph.json").write_text(json.dumps(graph))
    return tmp_path


class TestBuildIndex:
    def test_build_creates_db(self, memory_root: Path) -> None:
        meta = build_memory_index(memory_root)
        db_path = memory_root / "index" / "norax_memory.sqlite"
        assert db_path.exists()
        assert meta["stats"]["chunks_added"] > 0
        assert meta["entities"] == 2
        assert meta["relations"] == 1

    def test_incremental_skips_unchanged(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        meta2 = build_memory_index(memory_root)
        assert meta2["stats"]["files_skipped"] > 0
        assert meta2["stats"]["chunks_added"] == 0 or meta2["stats"]["chunks_added"] <= 2

    def test_meta_written(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        meta_path = memory_root / "index" / "index_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "built_at" in meta
        assert meta["fts5_available"] is True

    def test_subsecond_change_is_not_mistaken_for_unchanged(self, memory_root: Path) -> None:
        source = memory_root / "semantic" / "same_tick.md"
        source.write_text("STATE:before\n", encoding="utf-8")
        build_memory_index(memory_root)
        source.write_text("STATE:after!\n", encoding="utf-8")

        meta = build_memory_index(memory_root)
        with sqlite3.connect(memory_root / "index" / "norax_memory.sqlite") as conn:
            indexed = conn.execute(
                "SELECT text FROM chunks WHERE path = ?",
                ("semantic/same_tick.md",),
            ).fetchall()

        assert meta["stats"]["files_indexed"] >= 1
        assert indexed == [("STATE:after!",)]

    def test_deleted_graph_entities_are_removed_from_projection(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        graph_path = memory_root / "entity_graph.json"
        graph_path.write_text(
            json.dumps({"entities": {"wallet": ["abc123"]}, "links": []}),
            encoding="utf-8",
        )

        meta = build_memory_index(memory_root)
        with sqlite3.connect(memory_root / "index" / "norax_memory.sqlite") as conn:
            names = conn.execute("SELECT name FROM entities ORDER BY name").fetchall()
            relation_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

        assert meta["entities"] == 1
        assert names == [("wallet",)]
        assert relation_count == 0

    def test_linked_source_cannot_remain_as_stale_index_authority(
        self,
        memory_root: Path,
    ) -> None:
        source = memory_root / "semantic" / "linked.md"
        source.write_text("TRUSTED_FACT:original\n", encoding="utf-8")
        build_memory_index(memory_root)
        source.unlink()
        outside = memory_root / "outside.md"
        outside.write_text("FORGED_FACT:external\n", encoding="utf-8")
        source.symlink_to(outside)

        meta = build_memory_index(memory_root)
        with sqlite3.connect(memory_root / "index" / "norax_memory.sqlite") as conn:
            rows = conn.execute(
                "SELECT text FROM chunks WHERE path = ?",
                ("semantic/linked.md",),
            ).fetchall()

        assert rows == []
        assert meta["stats"]["chunks_deleted"] >= 1
        assert outside.read_text(encoding="utf-8") == "FORGED_FACT:external\n"

    def test_linked_database_is_rejected_without_touching_target(self, tmp_path: Path) -> None:
        index = tmp_path / "index"
        index.mkdir()
        outside = tmp_path / "outside.sqlite"
        outside.write_text("unchanged", encoding="utf-8")
        (index / "norax_memory.sqlite").symlink_to(outside)

        with pytest.raises(ValueError, match="singly linked regular file"):
            build_memory_index(tmp_path)

        assert outside.read_text(encoding="utf-8") == "unchanged"

    def test_index_artifacts_are_private(self, memory_root: Path) -> None:
        build_memory_index(memory_root)

        assert (memory_root / "index").stat().st_mode & 0o777 == 0o700
        assert (memory_root / "index" / "norax_memory.sqlite").stat().st_mode & 0o777 == 0o600
        assert (memory_root / "index" / "index_meta.json").stat().st_mode & 0o777 == 0o600


class TestSQLiteIndexRetriever:
    def test_keyword_search(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        retriever = SQLiteIndexRetriever(memory_root)
        results = retriever.search_keyword("wallet EVM")
        assert len(results) > 0
        assert any("wallet" in r["text"].lower() for r in results)
        retriever.close()

    def test_keyword_search_fallback(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        retriever = SQLiteIndexRetriever(memory_root)
        # Query with special chars should not crash
        results = retriever.search_keyword("test (verify)")
        assert isinstance(results, list)
        retriever.close()

    def test_entity_search(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        retriever = SQLiteIndexRetriever(memory_root)
        entities = retriever.search_entities("wallet")
        assert len(entities) > 0
        retriever.close()

    def test_entity_neighbors(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        retriever = SQLiteIndexRetriever(memory_root)
        entities = retriever.search_entities("wallet")
        if entities:
            neighbors = retriever.entity_neighbors(entities[0]["entity_id"])
            assert isinstance(neighbors, list)
        retriever.close()

    def test_stats(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        retriever = SQLiteIndexRetriever(memory_root)
        stats = retriever.stats()
        assert stats["chunks"] > 0
        assert stats["entities"] > 0
        assert stats["db_size_kb"] > 0
        retriever.close()

    def test_missing_index_is_a_real_empty_fallback(self, tmp_path: Path) -> None:
        retriever = SQLiteIndexRetriever(tmp_path)

        assert retriever.search_keyword("anything") == []
        assert retriever.search_entities("anything") == []
        assert retriever.entity_neighbors("missing") == []
        assert retriever.stats()["available"] is False
        retriever.close()

    def test_query_and_traversal_controls_are_strictly_bounded(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        retriever = SQLiteIndexRetriever(memory_root)

        with pytest.raises(ValueError, match="k"):
            retriever.search_keyword("wallet", k=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="query"):
            retriever.search_keyword("x" * 16_001)
        with pytest.raises(ValueError, match="depth"):
            retriever.entity_neighbors("entity", depth=6)
        with pytest.raises(ValueError, match="entity_id"):
            retriever.entity_neighbors("")
        retriever.close()

    def test_entity_search_treats_wildcards_as_literal_text(self, memory_root: Path) -> None:
        build_memory_index(memory_root)
        retriever = SQLiteIndexRetriever(memory_root)

        assert retriever.search_entities("wallet")
        assert retriever.search_entities("%") == []
        assert retriever.search_entities("_") == []
        retriever.close()

    def test_retriever_rejects_linked_or_corrupt_database(self, tmp_path: Path) -> None:
        index = tmp_path / "index"
        index.mkdir()
        outside = tmp_path / "outside.sqlite"
        outside.write_text("unchanged", encoding="utf-8")
        db_path = index / "norax_memory.sqlite"
        db_path.symlink_to(outside)
        linked = SQLiteIndexRetriever(tmp_path)

        with pytest.raises(ValueError, match="singly linked regular file"):
            linked.search_keyword("anything")
        assert outside.read_text(encoding="utf-8") == "unchanged"

        db_path.unlink()
        db_path.write_text("not sqlite", encoding="utf-8")
        corrupt = SQLiteIndexRetriever(tmp_path)
        assert corrupt.search_keyword("anything") == []
        assert corrupt.stats()["available"] is False
