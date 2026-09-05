"""Behavioral tests for causal-memory ingestion, persistence, and retrieval."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import norax.memory.causal_graph as causal_module
from norax.memory.causal_graph import CausalGraph, CausalNode, _result_outcome
from norax.memory.store import Neuron


def _trace() -> list[dict]:
    return [
        {
            "name": "read",
            "args": {"path": "src/runtime.py", "ignored": "value"},
            "result": {"ok": True},
        },
        {
            "name": "exec",
            "args": '{"command":"pytest tests/runtime"}',
            "result": '{"ok":false,"error":"runtime test failed"}',
        },
        {
            "name": "edit",
            "args": {"path": "src/runtime.py"},
            "result": {"exit_code": 0},
        },
    ]


def _neuron(text: str, *, weight: float = 1.0) -> Neuron:
    return Neuron(text=text, path=Path("/memory/test.md"), line=1, weight=weight)


def test_ingestion_is_deterministic_and_records_truthful_outcome_edges(tmp_path: Path) -> None:
    graph = CausalGraph(tmp_path)
    payload = {"trajectory_id": "turn-1", "timestamp": 100.0, "trace": _trace()}
    assert graph.ingest_trajectory(payload) == 3
    assert graph.ingest_trajectory(payload) == 0
    assert len(graph.nodes) == 3
    assert [edge[2] for edge in graph.edges] == [
        "caused_failure",
        "recovered_after_failure",
    ]
    nodes = list(graph.nodes.values())
    assert nodes[0].args_summary == "path=src/runtime.py"
    assert nodes[1].result_ok is False
    assert nodes[1].error == "runtime test failed"
    assert nodes[2].result_ok is True
    assert all(node.source_trajectory == "turn-1" for node in nodes)
    assert [node.ts for node in nodes] == sorted(node.ts for node in nodes)
    assert graph.stats()["fingerprint"]


def test_distinct_trajectory_ids_keep_repeated_real_world_calls(tmp_path: Path) -> None:
    graph = CausalGraph(tmp_path)
    assert graph.ingest_trajectory({"trajectory_id": "turn-1", "trace": _trace()[:1]}) == 1
    assert graph.ingest_trajectory({"trajectory_id": "turn-2", "trace": _trace()[:1]}) == 1
    assert len(graph.nodes) == 2

    anonymous = CausalGraph(tmp_path / "anonymous")
    assert anonymous.ingest_trajectory({"trace": _trace()[:1]}) == 1
    assert anonymous.ingest_trajectory({"trace": _trace()[:1]}) == 0


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (None, (False, "invalid tool result")),
        ({"ok": True}, (True, "")),
        ({"ok": "yes"}, (False, "invalid tool outcome")),
        ({"ok": "false"}, (False, "invalid tool outcome")),
        ({"ok": "garbage"}, (False, "invalid tool outcome")),
        ({"exit_code": 0}, (True, "")),
        ({"exit_code": 2, "detail": "failed"}, (False, "failed")),
        ({"error": "failed"}, (False, "failed")),
        ({}, (False, "missing tool outcome")),
    ],
)
def test_tool_outcome_parsing_does_not_treat_false_strings_as_success(
    result: object, expected: tuple[bool, str]
) -> None:
    assert _result_outcome(result) == expected


def test_ingestion_skips_malformed_items_and_bounds_stored_fields(tmp_path: Path) -> None:
    graph = CausalGraph(tmp_path)
    trace = [
        None,
        {},
        {"name": 1},
        {
            "name": "x" * 200,
            "args": "not-json-" + "a" * 500,
            "result": {"ok": False, "error": "e" * 2_000},
        },
        {
            "name": "list",
            "args": ["bad"],
            "result": "plain output",
        },
    ]
    assert graph.ingest_trajectory({"trace": trace, "timestamp": float("nan")}) == 2
    first, second = graph.nodes.values()
    assert len(first.tool_name) == 128
    assert len(first.error) == 1_000
    assert first.args_summary == ""
    assert second.result_ok is False
    assert second.error == "invalid tool result"
    assert math.isfinite(first.ts)
    assert graph.ingest_trajectory({"trace": "not-a-list"}) == 0
    assert graph.ingest_trajectory({"trace": []}) == 0


def test_persistence_retains_argument_evidence_and_filters_invalid_records(tmp_path: Path) -> None:
    graph = CausalGraph(tmp_path)
    graph.ingest_trajectory({"trajectory_id": "turn-1", "timestamp": 10, "trace": _trace()})
    graph.save()

    restored = CausalGraph(tmp_path)
    assert restored.load() is True
    assert restored.edges == graph.edges
    assert restored.stats() == graph.stats()
    exec_node = next(node for node in restored.nodes.values() if node.tool_name == "exec")
    assert exec_node.args_summary == "command=pytest tests/runtime"
    assert exec_node.source_trajectory == "turn-1"

    data = json.loads(restored.sidecar_path.read_text(encoding="utf-8"))
    data["nodes"]["bad-ts"] = {"tool_name": "bad", "ts": float("nan")}
    data["nodes"][""] = {"tool_name": "bad", "ts": 1}
    data["edges"].extend(
        [
            ["missing", exec_node.node_id, "bad"],
            ["too", "short"],
            "not-an-edge",
        ]
    )
    restored.sidecar_path.write_text(json.dumps(data), encoding="utf-8")
    filtered = CausalGraph(tmp_path)
    assert filtered.load() is True
    assert "bad-ts" not in filtered.nodes
    assert "" not in filtered.nodes
    assert all(len(edge) == 3 for edge in filtered.edges)
    assert all(edge[0] in filtered.nodes and edge[1] in filtered.nodes for edge in filtered.edges)


def test_failed_load_is_transactional_and_rejects_symlinks(tmp_path: Path) -> None:
    graph = CausalGraph(tmp_path)
    graph.nodes["existing"] = CausalNode("existing", "tool_call", "read", "", True, "", 1)
    graph.sidecar_path.write_text("{", encoding="utf-8")
    assert graph.load() is False
    assert set(graph.nodes) == {"existing"}

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    graph.sidecar_path.unlink()
    graph.sidecar_path.symlink_to(target)
    assert graph.load() is False


def test_causal_search_requires_shared_query_and_memory_evidence(tmp_path: Path) -> None:
    graph = CausalGraph(tmp_path)
    graph.ingest_trajectory({"trajectory_id": "turn-1", "trace": _trace()})
    relevant = _neuron("pytest runtime failure remediation", weight=1.3)
    weaker = _neuron("pytest runtime invocation", weight=0.4)
    unrelated = _neuron("gardening calendar", weight=1.3)

    hits = graph.search("why did the runtime pytest fail", [unrelated, weaker, relevant])
    assert [neuron for neuron, _score in hits] == [relevant, weaker]
    assert hits[0][1] > hits[1][1] > 0
    assert graph.search("database migration", [relevant]) == []
    assert graph.score_neuron(relevant.entity_id, set(), relevant.text) == 0
    assert graph.search("pytest", [relevant], k=0) == []
    assert graph.search("pytest", [_neuron("pytest", weight=float("nan"))]) == []


def test_query_causes_follows_two_hops_and_ignores_dangling_edges(tmp_path: Path) -> None:
    graph = CausalGraph(tmp_path)
    graph.ingest_trajectory({"trajectory_id": "turn-1", "trace": _trace()})
    graph.edges.append(("missing", "also-missing", "bad"))
    downstream = graph.query_causes("read")
    assert [node.tool_name for node in downstream] == ["exec", "edit"]
    assert graph.query_causes("unknown") == []


def test_pruning_bounds_nodes_and_edges_without_dangling_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(causal_module, "_MAX_NODES", 2)
    monkeypatch.setattr(causal_module, "_MAX_EDGES", 1)
    graph = CausalGraph(tmp_path)
    assert graph.ingest_trajectory({"trajectory_id": "turn-1", "trace": _trace()}) == 3
    assert len(graph.nodes) == 2
    assert len(graph.edges) <= 1
    assert all(edge[0] in graph.nodes and edge[1] in graph.nodes for edge in graph.edges)
    assert graph.stats()["nodes"] == 2


def test_argument_summary_and_node_id_helpers_are_bounded_and_stable() -> None:
    assert CausalGraph._summarize_args('{"url":"https://example.test","query":"term"}') == (
        "url=https://example.test query=term"
    )
    assert CausalGraph._summarize_args("not-json") == ""
    assert CausalGraph._summarize_args({}) == ""
    args = {"path": "x" * 200, "command": "run", "other": "ignored"}
    assert len(CausalGraph._summarize_args(args)) <= 512
    first = CausalGraph._make_node_id("read", args, "turn:1")
    second = CausalGraph._make_node_id("read", args, "turn:1")
    assert first == second
    assert len(first) == 16
