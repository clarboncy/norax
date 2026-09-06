from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from norax.memory import decay
from norax.memory.fact_evolution import FactEvolutionTracker, FactHistory, FactVersion


def _aged(path: Path, days: float) -> None:
    timestamp = decay.time.time() - days * 86_400
    os.utime(path, (timestamp, timestamp))


@pytest.mark.parametrize(
    ("weight", "age", "accesses", "kind", "expected"),
    [
        (1.3, 99_999, 0, "intel", 1.0),
        (2.0, 0, 0, "unknown", 1.0),
        (-1.0, 0, 0, "semantic", 0.0),
        (0.4, 1_000, 0, "intel", pytest.approx(0.4 / 21)),
    ],
)
def test_compute_importance_boundaries(weight, age, accesses, kind, expected):
    assert decay.compute_importance(weight, age, accesses, kind) == expected


@pytest.mark.parametrize("payload", ["not json", "[]", '{"ok": 1, "bad": -1, "flag": true}'])
def test_access_log_recovers_and_filters_invalid_state(tmp_path: Path, payload: str):
    path = tmp_path / "index" / "access_log.json"
    path.parent.mkdir()
    path.write_text(payload, encoding="utf-8")
    loaded = decay._load_access_log(tmp_path)
    if payload.startswith("{"):
        assert loaded == {"ok": 1}
    else:
        assert loaded == {}


def test_record_access_accumulates_and_uses_private_atomic_file(tmp_path: Path):
    decay.record_access(tmp_path, ["a", "a", "b"])
    decay.record_access(tmp_path, ["a"])
    path = tmp_path / "index" / "access_log.json"
    assert json.loads(path.read_text()) == {"a": 3, "b": 1}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_decay_pass_archives_weak_files_without_name_collisions(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(decay.time, "time", lambda: 2_000_000_000.0)
    for parent, text in (
        (tmp_path / "intel" / "one", "old alpha |W1"),
        (tmp_path / "intel" / "two", "old beta |W1"),
    ):
        parent.mkdir(parents=True)
        file = parent / "same.md"
        file.write_text(text, encoding="utf-8")
        os.utime(file, (1, 1))
    permanent = tmp_path / "semantic" / "owner.md"
    permanent.parent.mkdir()
    permanent.write_text("owner fact |W5", encoding="utf-8")
    fresh = tmp_path / "procedural" / "fresh.md"
    fresh.parent.mkdir()
    fresh.write_text("recent workflow |W4", encoding="utf-8")
    os.utime(fresh, (2_000_000_000, 2_000_000_000))

    result = decay.run_decay_pass(tmp_path)

    archived = sorted((tmp_path / "sleep" / "archive").glob("decayed_*.md"))
    assert result == decay.DecayResult(scanned=3, archived=2, skipped_permanent=1)
    assert len(archived) == 2
    assert {path.read_text() for path in archived} == {"old alpha |W1", "old beta |W1"}
    assert permanent.exists() and fresh.exists()


def test_decay_archive_failure_never_deletes_source(tmp_path: Path, monkeypatch):
    source = tmp_path / "intel" / "old.md"
    source.parent.mkdir(parents=True)
    source.write_text("old |W1", encoding="utf-8")
    _aged(source, 50_000)

    def fail_link(self, target):
        raise OSError("archive unavailable")

    monkeypatch.setattr(Path, "hardlink_to", fail_link)
    result = decay.run_decay_pass(tmp_path)
    assert result.errors == 1 and result.archived == 0
    assert source.read_text() == "old |W1"


def test_archive_allocator_handles_collision_and_unlink_failure(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.md"
    archive = tmp_path / "archive"
    archive.mkdir()
    source.write_text("preserve me")
    prefix = "decayed_intel_source_1000000_entity"
    (archive / f"{prefix}.md").write_text("existing")
    moved = decay._archive_file(source, archive, "intel", "entity", 1.0)
    assert moved.name == f"{prefix}_1.md"
    assert moved.read_text() == "preserve me"
    assert not source.exists()

    source.write_text("still here")
    original_unlink = Path.unlink

    def fail_source_unlink(self, *args, **kwargs):
        if self == source:
            raise OSError("cannot remove source")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source_unlink)
    with pytest.raises(OSError, match="cannot remove source"):
        decay._archive_file(source, archive, "intel", "other", 2.0)
    assert source.read_text() == "still here"
    assert not list(archive.glob("*other*"))


def test_archive_allocator_rejects_non_file_and_exhaustion(tmp_path: Path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    directory = tmp_path / "directory.md"
    directory.mkdir()
    with pytest.raises(OSError, match="regular file"):
        decay._archive_file(directory, archive, "intel", "entity", 1.0)

    source = tmp_path / "source.md"
    source.write_text("data")

    def collide(*_args, **_kwargs):
        raise FileExistsError

    monkeypatch.setattr(Path, "hardlink_to", collide)
    with pytest.raises(OSError, match="unique"):
        decay._archive_file(source, archive, "intel", "entity", 1.0)


def test_decay_scan_error_and_missing_stores_are_nonfatal(tmp_path: Path, monkeypatch):
    source = tmp_path / "semantic" / "bad.md"
    source.parent.mkdir()
    source.write_text("fact", encoding="utf-8")
    original = Path.read_text

    def selective_failure(self, *args, **kwargs):
        if self == source:
            raise OSError("unreadable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", selective_failure)
    result = decay.run_decay_pass(tmp_path)
    assert result.errors == 1 and result.scanned == 0


def test_decay_defaults_to_environment_and_stats_cover_all_kinds(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(tmp_path))
    monkeypatch.setattr(decay.time, "time", lambda: 2_000_000_000.0)
    semantic = tmp_path / "semantic" / "permanent.md"
    semantic.parent.mkdir()
    semantic.write_text("fact |W5", encoding="utf-8")
    procedural = tmp_path / "procedural" / "weak.md"
    procedural.parent.mkdir()
    procedural.write_text("workflow |W1\nwithout marker", encoding="utf-8")
    os.utime(procedural, (1, 1))
    broken = tmp_path / "intel" / "broken.md"
    broken.parent.mkdir()
    broken.write_text("intel |W2", encoding="utf-8")
    os.utime(broken, (1, 1))
    archive = tmp_path / "sleep" / "archive"
    archive.mkdir(parents=True)
    (archive / "decayed_existing.md").write_text("old")

    original = Path.read_text

    def selective_failure(self, *args, **kwargs):
        if self == broken:
            raise OSError("gone")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", selective_failure)
    stats = decay.get_decay_stats()
    assert stats["total_neurons"] == 2
    assert stats["permanent_count"] == 1
    assert stats["at_risk"] == 1
    assert stats["archived_count"] == 1
    assert stats["by_kind"]["semantic"]["avg_importance"] == 1.0
    assert stats["by_kind"]["procedural"]["count"] == 1
    assert stats["by_kind"]["intel"]["count"] == 0
    assert 0 < stats["avg_importance"] < 1


def test_decay_default_pass_and_empty_stats(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(tmp_path))
    assert decay.run_decay_pass() == decay.DecayResult()
    assert decay.get_decay_stats(tmp_path / "never-created") == {
        "total_neurons": 0,
        "at_risk": 0,
        "archived_count": 0,
        "permanent_count": 0,
        "by_kind": {},
        "avg_importance": 0.0,
    }


def test_decay_non_marker_and_strong_nonpermanent_paths(tmp_path: Path):
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    (semantic / "plain.md").write_text("no explicit weight")
    result = decay.run_decay_pass(tmp_path)
    assert result.scanned == 1 and result.archived == 0
    stats = decay.get_decay_stats(tmp_path)
    assert stats["at_risk"] == 0
    assert stats["by_kind"]["semantic"]["avg_importance"] > 0.9


def test_fact_history_helpers_are_temporal_and_deterministic():
    empty = FactHistory("empty")
    assert empty.current_value() == "" and empty.value_at(100) == ""
    history = FactHistory(
        "version",
        [FactVersion("version", "one", 10), FactVersion("version", "two", 20)],
    )
    assert history.current_value() == "two"
    assert history.value_at(9) == ""
    assert history.value_at(10) == "one"
    assert history.value_at(99) == "two"
    assert history.has_contradiction() is True
    assert history.to_dict()["current"] == "two"


def test_fact_tracker_round_trip_search_and_contradictions(tmp_path: Path, monkeypatch):
    ticks = iter([10.0, 20.0, 30.0])
    monkeypatch.setattr("norax.memory.fact_evolution.time.time", lambda: next(ticks))
    tracker = FactEvolutionTracker(tmp_path)
    assert tracker.record("model_version", "same", "same") is None
    first = tracker.record(
        "model_version", "", "one", source="owner", episode_id="ep1", metadata={"x": 1}
    )
    second = tracker.record("model_version", "wrong-old", "two", confidence=0.8)
    tracker.record("location", "", "New York")
    assert first and second
    assert tracker.get_current("model_version") == "two"
    assert tracker.get_current("missing") == ""
    assert tracker.get_state_at("model_version", 15) == "one"
    assert tracker.get_state_at("missing", 15) == ""
    assert tracker.list_facts() == ["model_version", "location"]
    assert tracker.search_facts("model")[0]["fact_id"] == "model_version"
    assert tracker.search_facts("new york")[0]["fact_id"] == "location"
    assert tracker.search_facts("absent") == []
    contradiction = tracker.get_contradictions()
    assert contradiction == [
        {"fact_id": "model_version", "versions": 2, "values": ["one", "two"], "current": "two"}
    ]
    assert stat.S_IMODE((tmp_path / "fact_evolution.jsonl").stat().st_mode) == 0o600

    reloaded = FactEvolutionTracker(tmp_path)
    assert reloaded.get_history("model_version").versions[0].source == "owner"
    assert reloaded.get_history("model_version").versions[0].metadata == {"x": 1}
    assert reloaded.get_current("model_version") == "two"
    # Idempotent load: querying twice does not duplicate entries.
    assert len(reloaded.get_history("model_version").versions) == 2


def test_fact_loader_skips_corruption_and_sorts_later_valid_rows(tmp_path: Path, caplog):
    path = tmp_path / "fact_evolution.jsonl"
    rows = [
        {"fact_id": "x", "value": "later", "timestamp": 20},
        "",
        "not-json",
        {"missing": "required fields"},
        {"fact_id": "x", "value": "earlier", "timestamp": 10},
    ]
    path.write_text(
        "\n".join(json.dumps(row) if isinstance(row, dict) else row for row in rows),
        encoding="utf-8",
    )
    tracker = FactEvolutionTracker(tmp_path)
    assert [v.value for v in tracker.get_history("x").versions] == ["earlier", "later"]
    assert "skipped line" in caplog.text


def test_fact_load_and_persist_failures_are_truthful_and_non_destructive(
    tmp_path: Path, monkeypatch, caplog
):
    outside = tmp_path / "outside"
    outside.write_text("do not touch\n", encoding="utf-8")
    (tmp_path / "fact_evolution.jsonl").symlink_to(outside)
    tracker = FactEvolutionTracker(tmp_path)
    assert tracker.get_current("x") == ""
    version = tracker.record("x", "", "new")
    assert version is not None
    assert outside.read_text() == "do not touch\n"
    assert "load failed" in caplog.text and "persist failed" in caplog.text
