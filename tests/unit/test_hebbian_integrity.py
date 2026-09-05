"""Tests for durable, bounded Hebbian weight updates."""

from __future__ import annotations

from pathlib import Path

import pytest

import norax.memory.hebbian as hebbian_module
from norax.memory.hebbian import HebbianLearner, _weight_to_tag
from norax.memory.store import MemoryStore, Neuron


def _memory_file(root: Path, text: str) -> Path:
    path = root / "semantic" / "facts.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _neurons(root: Path) -> list[Neuron]:
    store = MemoryStore(root)
    store.refresh()
    return store.all_canonical()


@pytest.mark.parametrize(
    ("weight", "tag"),
    [
        (-1.0, "|W1"),
        (0.54, "|W1"),
        (0.55, "|W2"),
        (0.75, "|W3"),
        (0.95, "|W4"),
        (1.25, "|W5"),
        (10.0, "|W5"),
    ],
)
def test_weight_bands_have_explicit_boundaries(weight: float, tag: str) -> None:
    assert _weight_to_tag(weight) == tag


def test_record_cofiring_deduplicates_entities_and_requires_two_distinct_memories(
    tmp_path: Path,
) -> None:
    neurons = [
        Neuron("same", tmp_path / "x.md", 1),
        Neuron("same", tmp_path / "y.md", 1),
        Neuron("different", tmp_path / "x.md", 2),
    ]
    learner = HebbianLearner()
    learner.record_cofiring(neurons[:2])
    assert learner.turn_count == 0
    assert learner._pending_strengthens == {}

    learner.record_cofiring(neurons)
    assert learner.turn_count == 1
    assert set(learner._pending_strengthens) == {
        neurons[0].entity_id,
        neurons[2].entity_id,
    }
    assert all(delta == pytest.approx(0.06) for delta in learner._pending_strengthens.values())


def test_flush_groups_atomic_file_updates_and_preserves_unrelated_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _memory_file(
        tmp_path,
        "first fact|W1\nsecond fact|W2\nuntagged fact\n",
    )
    neurons = _neurons(tmp_path)
    learner = HebbianLearner()
    for _ in range(3):
        learner.record_cofiring(neurons[:2])

    writes: list[Path] = []
    real_atomic_write = hebbian_module.atomic_write_text

    def tracked_write(target: Path, text: str) -> None:
        writes.append(target)
        real_atomic_write(target, text)

    monkeypatch.setattr(hebbian_module, "atomic_write_text", tracked_write)
    assert learner.flush(tmp_path) == 2
    assert writes == [path]
    assert path.read_text(encoding="utf-8") == ("first fact|W2\nsecond fact|W3\nuntagged fact\n")
    assert learner._pending_strengthens == {}
    assert learner.flush(tmp_path) == 0


def test_failed_atomic_write_retains_pending_updates_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _memory_file(tmp_path, "first|W1\nsecond|W1\n")
    neurons = _neurons(tmp_path)
    learner = HebbianLearner()
    for _ in range(3):
        learner.record_cofiring(neurons)

    real_atomic_write = hebbian_module.atomic_write_text

    def fail_write(_target: Path, _text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(hebbian_module, "atomic_write_text", fail_write)
    assert learner.flush(tmp_path) == 0
    assert len(learner._pending_strengthens) == 2
    assert path.read_text(encoding="utf-8") == "first|W1\nsecond|W1\n"

    monkeypatch.setattr(hebbian_module, "atomic_write_text", real_atomic_write)
    assert learner.flush(tmp_path) == 2
    assert learner._pending_strengthens == {}
    assert "|W2" in path.read_text(encoding="utf-8")


def test_missing_memories_do_not_leave_unbounded_pending_entries(tmp_path: Path) -> None:
    learner = HebbianLearner(_pending_strengthens={"missing": 1.0})
    assert learner.flush(tmp_path) == 0
    assert learner._pending_strengthens == {}


def test_homeostatic_decay_really_moves_discrete_bands_but_preserves_w1_and_w5(
    tmp_path: Path,
) -> None:
    path = _memory_file(
        tmp_path,
        "floor|W1\nlow|W2\nmedium|W3\nhigh|W4\npermanent|W5\nuntagged\n",
    )
    learner = HebbianLearner(
        turn_count=50,
        _pending_strengthens={"missing": 0.06},
    )
    assert learner.flush(tmp_path) == 3
    assert path.read_text(encoding="utf-8") == (
        "floor|W1\nlow|W1\nmedium|W2\nhigh|W3\npermanent|W5\nuntagged\n"
    )


def test_decay_write_failure_is_non_destructive(tmp_path: Path, monkeypatch) -> None:
    path = _memory_file(tmp_path, "one|W2\ntwo|W3\n")

    def fail_write(_target: Path, _text: str) -> None:
        raise OSError("readonly")

    monkeypatch.setattr(hebbian_module, "atomic_write_text", fail_write)
    learner = HebbianLearner()
    assert learner._decay(tmp_path) == 0
    assert path.read_text(encoding="utf-8") == "one|W2\ntwo|W3\n"
