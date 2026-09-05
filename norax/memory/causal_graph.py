"""Bounded causal memory over tool trajectories.

The graph records deterministic tool-call nodes and sequential outcome edges.
Search only boosts a memory when both the query and that memory overlap with
the same observed causal node; merely existing in the graph is not relevance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..atomic import atomic_write_text, read_bounded_text
from .store import Neuron

log = logging.getLogger("norax.memory.causal_graph")

_TOKEN_RX: Final = re.compile(r"[a-z0-9_./-]{3,}")
_MAX_NODES: Final = 50_000
_MAX_EDGES: Final = 50_000
_MAX_SIDECAR_CHARS: Final = 64 * 1024 * 1024
_MAX_ERROR_CHARS: Final = 1_000
_MAX_SUMMARY_CHARS: Final = 512
_MAX_SOURCE_CHARS: Final = 256


@dataclass
class CausalNode:
    """One observed tool call in a causal trajectory."""

    node_id: str
    kind: str
    tool_name: str
    args_summary: str
    result_ok: bool
    error: str
    ts: float
    source_trajectory: str = ""


def _result_outcome(result: Any) -> tuple[bool, str]:
    """Interpret common tool result shapes without truthy-string mistakes."""
    if not isinstance(result, dict):
        return False, "invalid tool result"
    error = str(result.get("error") or result.get("detail") or "")[:_MAX_ERROR_CHARS]
    if "ok" in result:
        raw_ok = result["ok"]
        if isinstance(raw_ok, bool):
            return raw_ok, error
        return False, error or "invalid tool outcome"
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return exit_code == 0, error
    if "raw" in result:
        return False, error or "invalid tool result"
    return False, error or "missing tool outcome"


def _normalized_args(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value[:200]}
    return value if isinstance(value, dict) else {}


@dataclass
class CausalGraph:
    """In-memory causal graph with bounded JSON sidecar persistence."""

    root: Path
    nodes: dict[str, CausalNode] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    _fingerprint: str = ""
    _node_count: int = 0

    @property
    def sidecar_path(self) -> Path:
        return self.root / "causal_graph.json"

    def ingest_trajectory(self, payload: dict[str, Any]) -> int:
        """Ingest one trajectory, deterministically deduplicating repeated events."""
        trace = payload.get("trace")
        if not isinstance(trace, list) or not trace:
            return 0
        source = str(
            payload.get("trajectory_id")
            or payload.get("trace_id")
            or payload.get("session_id")
            or ""
        ).strip()[:_MAX_SOURCE_CHARS]
        if not source:
            canonical = json.dumps(trace, sort_keys=True, default=str, separators=(",", ":"))
            source = hashlib.sha256(canonical.encode()).hexdigest()[:24]
        raw_ts = payload.get("timestamp")
        base_ts = (
            float(raw_ts)
            if isinstance(raw_ts, (int, float))
            and not isinstance(raw_ts, bool)
            and math.isfinite(float(raw_ts))
            else time.time()
        )

        added = 0
        prev_node_id: str | None = None
        prev_ok: bool | None = None
        for index, item in enumerate(trace):
            if not isinstance(item, dict):
                continue
            name_value = item.get("name")
            if not isinstance(name_value, str) or not name_value.strip():
                continue
            name = name_value.strip()[:128]
            args = _normalized_args(item.get("args"))
            result = item.get("result")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"raw": result[:200]}
            ok, error = _result_outcome(result)
            node_id = self._make_node_id(name, args, f"{source}:{index}")

            if node_id not in self.nodes:
                self.nodes[node_id] = CausalNode(
                    node_id=node_id,
                    kind="tool_call",
                    tool_name=name,
                    args_summary=self._summarize_args(args),
                    result_ok=ok,
                    error=error,
                    ts=base_ts + index / 1_000_000,
                    source_trajectory=source,
                )
                added += 1
                if prev_node_id is not None:
                    relation = "succeeded" if prev_ok else "recovered_after_failure"
                    if not ok:
                        relation = "caused_failure"
                    edge = (prev_node_id, node_id, relation)
                    if edge not in self.edges:
                        self.edges.append(edge)
            prev_node_id = node_id
            prev_ok = ok

        self._prune()
        self._refresh_metadata()
        return added

    def _prune(self) -> None:
        if len(self.nodes) > _MAX_NODES:
            keep = {
                node.node_id
                for node in sorted(
                    self.nodes.values(),
                    key=lambda item: (item.ts, item.node_id),
                    reverse=True,
                )[:_MAX_NODES]
            }
            self.nodes = {node_id: node for node_id, node in self.nodes.items() if node_id in keep}
            self.edges = [edge for edge in self.edges if edge[0] in keep and edge[1] in keep]
        if len(self.edges) > _MAX_EDGES:
            self.edges = self.edges[-_MAX_EDGES:]

    def _refresh_metadata(self) -> None:
        self._fingerprint = hashlib.sha256(
            json.dumps(sorted(self.nodes), separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self._node_count = len(self.nodes)

    def save(self) -> None:
        self._prune()
        self._refresh_metadata()
        data = {
            "fingerprint": self._fingerprint,
            "node_count": self._node_count,
            "edge_count": len(self.edges),
            "nodes": {
                node_id: {
                    "kind": node.kind,
                    "tool_name": node.tool_name,
                    "args_summary": node.args_summary,
                    "result_ok": node.result_ok,
                    "error": node.error,
                    "ts": node.ts,
                    "source_trajectory": node.source_trajectory,
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [list(edge) for edge in self.edges],
        }
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.sidecar_path,
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        )

    def load(self) -> bool:
        try:
            raw = read_bounded_text(
                self.sidecar_path,
                max_bytes=_MAX_SIDECAR_CHARS,
            )
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("causal graph sidecar root must be an object")
            raw_nodes = data.get("nodes", {})
            raw_edges = data.get("edges", [])
            if not isinstance(raw_nodes, dict) or not isinstance(raw_edges, list):
                raise ValueError("causal graph nodes and edges must be collections")

            loaded_nodes: dict[str, CausalNode] = {}
            for raw_id, details in list(raw_nodes.items())[-_MAX_NODES:]:
                if (
                    not isinstance(raw_id, str)
                    or not raw_id
                    or len(raw_id) > 128
                    or not isinstance(details, dict)
                ):
                    continue
                ts = details.get("ts", 0.0)
                if (
                    isinstance(ts, bool)
                    or not isinstance(ts, (int, float))
                    or not math.isfinite(float(ts))
                ):
                    continue
                loaded_nodes[raw_id] = CausalNode(
                    node_id=raw_id,
                    kind=str(details.get("kind") or "tool_call")[:32],
                    tool_name=str(details.get("tool_name") or "")[:128],
                    args_summary=str(details.get("args_summary") or "")[:_MAX_SUMMARY_CHARS],
                    result_ok=details.get("result_ok") is True,
                    error=str(details.get("error") or "")[:_MAX_ERROR_CHARS],
                    ts=float(ts),
                    source_trajectory=str(details.get("source_trajectory") or "")[
                        :_MAX_SOURCE_CHARS
                    ],
                )

            loaded_edges: list[tuple[str, str, str]] = []
            for edge in raw_edges[-_MAX_EDGES:]:
                if (
                    isinstance(edge, (list, tuple))
                    and len(edge) == 3
                    and all(isinstance(value, str) for value in edge)
                    and edge[0] in loaded_nodes
                    and edge[1] in loaded_nodes
                ):
                    loaded_edges.append((edge[0], edge[1], edge[2][:64]))
            self.nodes = loaded_nodes
            self.edges = loaded_edges
            self._refresh_metadata()
            return True
        except FileNotFoundError:
            return False
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("causal_graph.load_failed: %r", exc)
            return False

    def query_causes(self, tool_name: str) -> list[CausalNode]:
        """Find nodes up to two causal hops downstream from a tool type."""
        downstream: set[str] = set()
        for from_id, to_id, _relation in self.edges:
            source = self.nodes.get(from_id)
            if source and source.tool_name == tool_name:
                downstream.add(to_id)
        first_hop = set(downstream)
        for from_id, to_id, _relation in self.edges:
            if from_id in first_hop:
                downstream.add(to_id)
        return sorted(
            (self.nodes[node_id] for node_id in downstream if node_id in self.nodes),
            key=lambda node: (node.ts, node.node_id),
        )

    def score_neuron(
        self,
        neuron_id: str,
        query_entities: set[str],
        neuron_text: str = "",
    ) -> float:
        """Return a bounded causal relevance boost for one memory."""
        query_tokens = {token.lower() for token in query_entities if len(token) >= 3}
        if not query_tokens:
            return 0.0
        memory_tokens = set(_TOKEN_RX.findall((neuron_text or "").lower()))
        best = 0.0
        for node in self.nodes.values():
            causal_text = f"{node.tool_name} {node.args_summary} {node.error}".lower()
            causal_tokens = set(_TOKEN_RX.findall(causal_text))
            query_overlap = query_tokens & causal_tokens
            if not query_overlap:
                continue
            id_link = bool(neuron_id) and neuron_id.lower() in causal_text
            memory_overlap = memory_tokens & causal_tokens
            if not id_link and not memory_overlap:
                continue
            query_ratio = len(query_overlap) / max(1, len(query_tokens))
            memory_ratio = 1.0 if id_link else len(memory_overlap) / max(1, len(memory_tokens))
            failure_boost = 0.05 if not node.result_ok else 0.0
            score = min(0.4, 0.12 + 0.16 * query_ratio + 0.07 * memory_ratio + failure_boost)
            best = max(best, score)
        return best

    def search(
        self,
        query: str,
        neurons: list[Neuron],
        *,
        k: int = 10,
    ) -> list[tuple[Neuron, float]]:
        """Rank memories supported by causal nodes relevant to the query."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            return []
        query_tokens = set(_TOKEN_RX.findall((query or "").lower()))
        hits: list[tuple[Neuron, float]] = []
        for neuron in neurons:
            score = self.score_neuron(neuron.entity_id, query_tokens, neuron.text)
            if score > 0:
                weight = float(neuron.weight)
                if math.isfinite(weight) and weight > 0:
                    hits.append((neuron, score * weight))
        hits.sort(key=lambda item: (-item[1], item[0].entity_id))
        return hits[:k]

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": self._node_count,
            "edges": len(self.edges),
            "relation_types": Counter(relation for _, _, relation in self.edges).most_common(10),
            "fingerprint": self._fingerprint,
        }

    @staticmethod
    def _make_node_id(name: str, args: dict[str, Any], scope: object) -> str:
        body = json.dumps(
            {"name": name, "args": args, "scope": scope},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    @staticmethod
    def _summarize_args(args: dict[str, Any] | str) -> str:
        normalized = _normalized_args(args)
        if not normalized:
            return ""
        parts: list[str] = []
        for key, value in normalized.items():
            if key in {"command", "path", "query", "url"}:
                parts.append(f"{key}={str(value)[:120]}")
        return " ".join(parts[:4])[:_MAX_SUMMARY_CHARS]
