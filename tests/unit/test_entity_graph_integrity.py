"""Integrity, resource-bound, and transactional tests for the entity graph."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import norax.memory.entity_graph as entity_module
from norax.memory.entity_graph import EntityGraph, extract_entities
from norax.memory.store import Neuron


def _neurons() -> list[Neuron]:
    return [
        Neuron(
            text="Norax routes kimi-k2.6 through GatewayRouter",
            path=Path("semantic/runtime.md"),
            line=1,
        ),
        Neuron(
            text="GatewayRouter reads config/runtime.jsonc",
            path=Path("procedural/runtime.md"),
            line=2,
        ),
    ]


def test_sidecar_round_trip_is_private_and_metadata_is_derived(tmp_path: Path) -> None:
    graph = EntityGraph.from_store(tmp_path, _neurons())
    graph.save()

    assert stat.S_IMODE(graph.sidecar_path.stat().st_mode) == 0o600
    restored = EntityGraph(tmp_path)
    assert restored.load() is True
    assert restored.entities == graph.entities
    assert restored.neuron_entities == graph.neuron_entities
    assert restored.links == graph.links
    assert restored.stats() == graph.stats()

    payload = json.loads(graph.sidecar_path.read_text(encoding="utf-8"))
    payload["entity_count"] = 999_999
    payload["link_count"] = 999_999
    graph.sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    derived = EntityGraph(tmp_path)
    assert derived.load() is True
    assert derived.stats()["entities"] == len(derived.entities)
    assert derived.stats()["links"] == len(derived.links)


def test_load_is_transactional_and_rejects_inconsistent_or_symlinked_state(
    tmp_path: Path,
) -> None:
    graph = EntityGraph(tmp_path)
    graph.entities = {"existing": {"id"}}
    graph.neuron_entities = {"id": {"existing"}}
    graph.links = [("existing", "existing")]
    graph.sidecar_path.write_text(
        json.dumps(
            {
                "fingerprint": "bad",
                "entities": {"invented": ["id"]},
                "neuron_entities": {"id": ["existing"]},
                "links": [],
            }
        ),
        encoding="utf-8",
    )

    assert graph.load() is False
    assert graph.entities == {"existing": {"id"}}
    assert graph.neuron_entities == {"id": {"existing"}}
    assert graph.links == [("existing", "existing")]

    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    graph.sidecar_path.unlink()
    graph.sidecar_path.symlink_to(target)
    assert graph.load() is False


def test_oversized_sidecar_and_entity_explosion_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(entity_module, "_MAX_SIDECAR_BYTES", 64)
    graph = EntityGraph(tmp_path)
    graph.sidecar_path.write_text("x" * 65, encoding="utf-8")
    assert graph.load() is False

    monkeypatch.setattr(entity_module, "_MAX_ENTITIES_PER_TEXT", 3)
    monkeypatch.setattr(entity_module, "_MAX_GRAPH_LINKS", 2)
    entities = extract_entities('"alpha" "bravo" "charlie" "delta" "echo"')
    assert len(entities) == 3
    bounded = EntityGraph.from_store(
        tmp_path,
        [Neuron(text='"alpha" "bravo" "charlie" "delta" "echo"', path=Path("x"), line=1)],
    )
    assert len(bounded.links) <= 2
    assert bounded.stats()["truncated"] is True
