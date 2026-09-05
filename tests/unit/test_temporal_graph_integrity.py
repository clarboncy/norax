"""Tests for bounded and query-anchored temporal memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import norax.memory.temporal_graph as temporal_module
from norax.memory.store import Neuron
from norax.memory.temporal_graph import TemporalEdge, TemporalGraph


def _neuron(text: str, *, weight: float = 1.0) -> Neuron:
    return Neuron(text=text, path=Path("/memory/test.md"), line=1, weight=weight)


def test_record_access_accepts_epoch_zero_validates_numbers_and_bounds_history(
    tmp_path: Path,
) -> None:
    graph = TemporalGraph(tmp_path)
    graph.record_access("memory-a", ts=0.0, session_id="session")
    for timestamp in range(1, 8):
        graph.record_access("memory-a", ts=float(timestamp), session_id="session")
    assert graph.access_log["memory-a"] == [
        (3.0, "session"),
        (4.0, "session"),
        (5.0, "session"),
        (6.0, "session"),
        (7.0, "session"),
    ]
    assert graph.stats()["nodes"] == 1

    for bad in [True, -1, float("nan"), float("inf"), "1"]:
        with pytest.raises(ValueError, match="ts"):
            graph.record_access("memory-b", ts=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="neuron_id"):
        graph.record_access("", ts=1)


def test_sequence_records_real_delta_skips_self_edges_and_tracks_sessions(
    tmp_path: Path,
) -> None:
    graph = TemporalGraph(tmp_path)
    graph.record_access("a", ts=10, session_id="s1")
    graph.record_access("b", ts=13.5, session_id="s1")
    graph.record_access("c", ts=12, session_id="s1")
    assert graph.record_sequence(["a", "a", "b", "c"], session_id="s1") == 2
    assert graph.edges == [
        TemporalEdge("a", "b", 3.5, "s1"),
        TemporalEdge("b", "c", 0.0, "s1"),
    ]
    stats = graph.stats()
    assert stats["nodes"] == 3
    assert stats["edges"] == 2
    assert stats["sessions"] == 1
    assert stats["fingerprint"]
    with pytest.raises(TypeError, match="list"):
        graph.record_sequence("a,b")  # type: ignore[arg-type]


def test_sequence_uses_accesses_from_the_requested_session_only(tmp_path: Path) -> None:
    graph = TemporalGraph(tmp_path)
    graph.record_access("a", ts=1, session_id="old")
    graph.record_access("b", ts=100, session_id="old")
    graph.record_access("a", ts=10, session_id="current")
    graph.record_access("b", ts=12, session_id="current")
    graph.record_sequence(["a", "b"], session_id="current")
    assert graph.edges[0].delta_sec == 2

    graph.record_sequence(["a", "missing"], session_id="current")
    assert graph.edges[-1].delta_sec == 0


def test_save_and_load_preserve_access_types_deltas_and_metadata(tmp_path: Path) -> None:
    graph = TemporalGraph(tmp_path)
    graph.record_access("a", ts=10, session_id="session")
    graph.record_access("b", ts=12, session_id="session")
    graph.record_sequence(["a", "b"], session_id="session")
    graph.save()

    raw = json.loads(graph.sidecar_path.read_text(encoding="utf-8"))
    assert raw["edges"][0]["deltaSec"] == 2
    restored = TemporalGraph(tmp_path)
    assert restored.load() is True
    assert restored.access_log == {
        "a": [(10.0, "session")],
        "b": [(12.0, "session")],
    }
    assert restored.edges == [TemporalEdge("a", "b", 2.0, "session")]
    assert restored.stats() == graph.stats()


def test_load_filters_malformed_records_edges_and_legacy_delta(tmp_path: Path) -> None:
    graph = TemporalGraph(tmp_path)
    graph.sidecar_path.write_text(
        json.dumps(
            {
                "access_log": {
                    "a": [[1, "s"], [float("nan"), "s"], [2], "bad"],
                    "b": [[2, "s"]],
                    "empty": [],
                    "bad": "not-a-list",
                },
                "edges": [
                    {"before": "a", "after": "b", "session": "s"},
                    {"before": "missing", "after": "b", "deltaSec": 1},
                    {"before": "a", "after": "b", "deltaSec": -1},
                    {"before": "a", "after": "b", "deltaSec": float("nan")},
                    "bad",
                ],
            }
        ),
        encoding="utf-8",
    )
    assert graph.load() is True
    assert graph.access_log == {"a": [(1.0, "s")], "b": [(2.0, "s")]}
    assert graph.edges == [TemporalEdge("a", "b", 0.0, "s")]


def test_failed_load_is_transactional_and_symlinks_are_rejected(tmp_path: Path) -> None:
    graph = TemporalGraph(tmp_path)
    graph.access_log = {"existing": [(1, "s")]}
    graph.sidecar_path.write_text("{", encoding="utf-8")
    assert graph.load() is False
    assert graph.access_log == {"existing": [(1, "s")]}

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    graph.sidecar_path.unlink()
    graph.sidecar_path.symlink_to(target)
    assert graph.load() is False


def test_recency_score_is_bounded_for_future_and_invalid_clock_values(tmp_path: Path) -> None:
    graph = TemporalGraph(tmp_path)
    assert graph.recency_score("missing", now=100) == 0
    graph.record_access("a", ts=100)
    assert graph.recency_score("a", now=50) == 1
    assert graph.recency_score("a", now=100) == 1
    assert graph.recency_score("a", now=100 + 24 * 3_600) == pytest.approx(0.5)
    assert graph.recency_score("a", now=float("nan")) == 0
    assert graph.recency_score("a", now=True) == 0


def test_search_is_query_anchored_and_only_returns_temporal_neighbors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor = _neuron("pytest runtime failure", weight=1.0)
    after = _neuron("apply verified fix", weight=1.3)
    before = _neuron("inspect repository", weight=1.0)
    unrelated = _neuron("unrelated recent gardening", weight=5.0)
    graph = TemporalGraph(tmp_path)
    for index, neuron in enumerate([before, anchor, after, unrelated]):
        graph.record_access(neuron.entity_id, ts=100 + index, session_id="s")
    graph.record_sequence(
        [before.entity_id, anchor.entity_id, after.entity_id],
        session_id="s",
    )
    monkeypatch.setattr(temporal_module.time, "time", lambda: 104.0)

    hits = graph.search("pytest runtime", [unrelated, before, after, anchor])
    assert {neuron.entity_id for neuron, _score in hits} == {
        anchor.entity_id,
        before.entity_id,
        after.entity_id,
    }
    assert unrelated not in [neuron for neuron, _score in hits]
    assert hits[0][0] in [anchor, after]
    assert graph.search("", [anchor]) == []
    assert graph.search("no matching token", [anchor]) == []
    assert graph.search("pytest", [anchor], k=0) == []
    assert graph.score_neuron(anchor.entity_id, set()) == 0


def test_search_rejects_nonfinite_or_nonpositive_weights(tmp_path: Path) -> None:
    good = _neuron("runtime pytest", weight=1.0)
    nan = _neuron("runtime pytest nan", weight=float("nan"))
    negative = _neuron("runtime pytest other", weight=-1)
    graph = TemporalGraph(tmp_path)
    graph.record_access(good.entity_id, ts=1)
    graph.record_access(nan.entity_id, ts=1)
    graph.record_access(negative.entity_id, ts=1)
    hits = graph.search("runtime pytest", [good, nan, negative])
    assert [neuron for neuron, _score in hits] == [good]


def test_node_and_edge_pruning_keeps_recent_referentially_valid_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(temporal_module, "_MAX_NODES", 2)
    monkeypatch.setattr(temporal_module, "_MAX_EDGES", 1)
    graph = TemporalGraph(tmp_path)
    graph.record_access("old", ts=1)
    graph.record_access("middle", ts=2)
    graph.record_sequence(["old", "middle"])
    graph.record_access("new", ts=3)
    graph.record_sequence(["middle", "new"])

    assert set(graph.access_log) == {"middle", "new"}
    assert len(graph.edges) <= 1
    assert all(
        edge.before_id in graph.access_log and edge.after_id in graph.access_log
        for edge in graph.edges
    )
    graph.save()
    assert graph.stats()["nodes"] == 2
