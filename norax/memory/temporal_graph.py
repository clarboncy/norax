"""Bounded temporal relationships between retrieved memory neurons."""

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

log = logging.getLogger("norax.memory.temporal_graph")

TEMPORAL_HALF_LIFE_HOURS: Final = 24.0
_TOKEN_RX: Final = re.compile(r"[a-z0-9_./-]{3,}")
_MAX_NODES: Final = 50_000
_MAX_EDGES: Final = 20_000
_MAX_ACCESSES_PER_NODE: Final = 5
_MAX_SIDECAR_CHARS: Final = 64 * 1024 * 1024
_MAX_IDENTIFIER_CHARS: Final = 256


@dataclass(frozen=True)
class TemporalEdge:
    """A temporal relationship between two memory accesses."""

    before_id: str
    after_id: str
    delta_sec: float
    session_id: str


def _identifier(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:_MAX_IDENTIFIER_CHARS]


@dataclass
class TemporalGraph:
    """In-memory temporal graph with bounded JSON sidecar persistence."""

    root: Path
    access_log: dict[str, list[tuple[float, str]]] = field(default_factory=dict)
    edges: list[TemporalEdge] = field(default_factory=list)
    _fingerprint: str = ""
    _node_count: int = 0

    @property
    def sidecar_path(self) -> Path:
        return self.root / "temporal_graph.json"

    def record_access(
        self,
        neuron_id: str,
        *,
        ts: float | None = None,
        session_id: str = "",
    ) -> None:
        """Record a validated access, retaining only the recent bounded tail."""
        neuron_id = _identifier(neuron_id)
        session_id = _identifier(session_id)
        if not neuron_id:
            raise ValueError("neuron_id must be a non-empty string")
        timestamp = time.time() if ts is None else ts
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or timestamp < 0
        ):
            raise ValueError("ts must be a finite non-negative number")
        accesses = self.access_log.setdefault(neuron_id, [])
        accesses.append((float(timestamp), session_id))
        if len(accesses) > _MAX_ACCESSES_PER_NODE:
            del accesses[:-_MAX_ACCESSES_PER_NODE]
        if len(self.access_log) > _MAX_NODES:
            self._prune_nodes()
        self._node_count = len(self.access_log)

    def _latest_access(self, neuron_id: str, session_id: str) -> float | None:
        matches = [
            timestamp
            for timestamp, recorded_session in self.access_log.get(neuron_id, [])
            if not session_id or recorded_session == session_id
        ]
        return max(matches) if matches else None

    def record_sequence(self, neuron_ids: list[str], *, session_id: str = "") -> int:
        """Record consecutive distinct accesses and their observed time deltas."""
        if not isinstance(neuron_ids, list):
            raise TypeError("neuron_ids must be a list")
        normalized = [_identifier(neuron_id) for neuron_id in neuron_ids]
        normalized = [neuron_id for neuron_id in normalized if neuron_id]
        session_id = _identifier(session_id)
        added = 0
        for before, after in zip(normalized, normalized[1:], strict=False):
            if before == after:
                continue
            before_ts = self._latest_access(before, session_id)
            after_ts = self._latest_access(after, session_id)
            delta = (
                max(0.0, after_ts - before_ts)
                if before_ts is not None and after_ts is not None
                else 0.0
            )
            self.edges.append(
                TemporalEdge(
                    before_id=before,
                    after_id=after,
                    delta_sec=delta,
                    session_id=session_id,
                )
            )
            added += 1
        if len(self.edges) > _MAX_EDGES:
            self.edges = self.edges[-_MAX_EDGES:]
        self._refresh_metadata()
        return added

    def _prune_nodes(self) -> None:
        keep = {
            neuron_id
            for neuron_id, _accesses in sorted(
                self.access_log.items(),
                key=lambda item: (
                    max((timestamp for timestamp, _ in item[1]), default=0.0),
                    item[0],
                ),
                reverse=True,
            )[:_MAX_NODES]
        }
        self.access_log = {
            neuron_id: accesses
            for neuron_id, accesses in self.access_log.items()
            if neuron_id in keep
        }
        self.edges = [
            edge for edge in self.edges if edge.before_id in keep and edge.after_id in keep
        ][-_MAX_EDGES:]

    def _refresh_metadata(self) -> None:
        self._fingerprint = hashlib.sha256(
            json.dumps(sorted(self.access_log), separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self._node_count = len(self.access_log)

    def save(self) -> None:
        self._prune_nodes()
        self._refresh_metadata()
        data = {
            "fingerprint": self._fingerprint,
            "node_count": self._node_count,
            "edge_count": len(self.edges),
            "access_log": {
                neuron_id: accesses[-_MAX_ACCESSES_PER_NODE:]
                for neuron_id, accesses in self.access_log.items()
            },
            "edges": [
                {
                    "before": edge.before_id,
                    "after": edge.after_id,
                    "deltaSec": edge.delta_sec,
                    "session": edge.session_id,
                }
                for edge in self.edges[-_MAX_EDGES:]
            ],
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
                raise ValueError("temporal graph sidecar root must be an object")
            raw_accesses = data.get("access_log", {})
            raw_edges = data.get("edges", [])
            if not isinstance(raw_accesses, dict) or not isinstance(raw_edges, list):
                raise ValueError("temporal graph collections are malformed")

            loaded_accesses: dict[str, list[tuple[float, str]]] = {}
            for raw_id, records in list(raw_accesses.items())[-_MAX_NODES:]:
                neuron_id = _identifier(raw_id)
                if not neuron_id or not isinstance(records, list):
                    continue
                valid: list[tuple[float, str]] = []
                for record in records[-_MAX_ACCESSES_PER_NODE:]:
                    if not isinstance(record, (list, tuple)) or len(record) != 2:
                        continue
                    timestamp, session = record
                    if (
                        isinstance(timestamp, bool)
                        or not isinstance(timestamp, (int, float))
                        or not math.isfinite(float(timestamp))
                        or timestamp < 0
                    ):
                        continue
                    valid.append((float(timestamp), _identifier(session)))
                if valid:
                    loaded_accesses[neuron_id] = valid

            loaded_edges: list[TemporalEdge] = []
            for raw_edge in raw_edges[-_MAX_EDGES:]:
                if not isinstance(raw_edge, dict):
                    continue
                before = _identifier(raw_edge.get("before"))
                after = _identifier(raw_edge.get("after"))
                delta = raw_edge.get("deltaSec", raw_edge.get("delta_sec", 0.0))
                if (
                    not before
                    or not after
                    or before not in loaded_accesses
                    or after not in loaded_accesses
                    or isinstance(delta, bool)
                    or not isinstance(delta, (int, float))
                    or not math.isfinite(float(delta))
                    or delta < 0
                ):
                    continue
                loaded_edges.append(
                    TemporalEdge(
                        before_id=before,
                        after_id=after,
                        delta_sec=float(delta),
                        session_id=_identifier(raw_edge.get("session")),
                    )
                )
            self.access_log = loaded_accesses
            self.edges = loaded_edges
            self._refresh_metadata()
            return True
        except FileNotFoundError:
            return False
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("temporal_graph.load_failed: %r", exc)
            return False

    def recency_score(self, neuron_id: str, *, now: float | None = None) -> float:
        """Return a bounded exponential recency score."""
        accesses = self.access_log.get(neuron_id, [])
        if not accesses:
            return 0.0
        current = time.time() if now is None else now
        if (
            isinstance(current, bool)
            or not isinstance(current, (int, float))
            or not math.isfinite(float(current))
        ):
            return 0.0
        latest_ts = max(timestamp for timestamp, _ in accesses)
        age_hours = max(0.0, (float(current) - latest_ts) / 3_600.0)
        return min(1.0, 2.0 ** (-age_hours / TEMPORAL_HALF_LIFE_HOURS))

    def score_neuron(
        self,
        neuron_id: str,
        context_ids: set[str],
        *,
        now: float | None = None,
    ) -> float:
        """Score only query-anchored memories and their temporal neighbors."""
        if not context_ids:
            return 0.0
        recency_multiplier = 0.7 + 0.3 * self.recency_score(neuron_id, now=now)
        if neuron_id in context_ids:
            return recency_multiplier
        for edge in reversed(self.edges):
            if edge.before_id in context_ids and edge.after_id == neuron_id:
                return 0.7 * recency_multiplier
            if edge.after_id in context_ids and edge.before_id == neuron_id:
                return 0.5 * recency_multiplier
        return 0.0

    def search(
        self,
        query: str,
        neurons: list[Neuron],
        *,
        k: int = 10,
    ) -> list[tuple[Neuron, float]]:
        """Rank query matches and memories adjacent to them in access history."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            return []
        query_tokens = set(_TOKEN_RX.findall((query or "").lower()))
        if not query_tokens:
            return []
        unique = {neuron.entity_id: neuron for neuron in neurons}
        context_ids = {
            neuron.entity_id
            for neuron in unique.values()
            if query_tokens & set(_TOKEN_RX.findall(neuron.text.lower()))
        }
        if not context_ids:
            return []

        hits: list[tuple[Neuron, float]] = []
        now = time.time()
        for neuron_id, neuron in unique.items():
            score = self.score_neuron(neuron_id, context_ids, now=now)
            weight = float(neuron.weight)
            if score > 0 and math.isfinite(weight) and weight > 0:
                hits.append((neuron, score * weight))
        hits.sort(key=lambda item: (-item[1], item[0].entity_id))
        return hits[:k]

    def stats(self) -> dict[str, Any]:
        edges_by_session = Counter(edge.session_id for edge in self.edges)
        return {
            "nodes": self._node_count,
            "edges": len(self.edges),
            "sessions": len(edges_by_session),
            "top_sessions": edges_by_session.most_common(5),
            "fingerprint": self._fingerprint,
        }
