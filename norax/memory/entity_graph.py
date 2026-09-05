"""Entity graph — regex-based entity linking across memory neurons.

Local-first design (Mem0 2026 / A-MEM patterns):
  - Extract entities from markdown neurons (proper nouns, CamelCase, tech IDs,
    emails, versions, file paths)
  - Link neurons that share entities
  - Score queries via Jaccard overlap between query entities and neuron entities
  - Persist JSON sidecar at ``memory/entity_graph.json``; rebuild when the
    neuron fingerprint changes

No external graph DB — flat markdown + JSON sidecar only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..atomic import atomic_write_text, read_bounded_text
from .store import Neuron

log = logging.getLogger("norax.memory.entity_graph")

_MAX_ENTITY_TEXT_CHARS: Final = 32_768
_MAX_ENTITIES_PER_TEXT: Final = 128
_MAX_ENTITY_CHARS: Final = 128
_MAX_NEURON_ID_CHARS: Final = 128
_MAX_GRAPH_NEURONS: Final = 200_000
_MAX_GRAPH_ENTITIES: Final = 100_000
_MAX_GRAPH_MEMBERSHIPS: Final = 250_000
_MAX_GRAPH_LINKS: Final = 50_000
_MAX_SIDECAR_BYTES: Final = 64 * 1024 * 1024

# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

_EMAIL_RX = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b")
_VERSION_RX = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-_a-z0-9]+)?\b", re.I)
_PATH_RX = re.compile(
    r"(?:~?/[\w./-]+|(?:norax|memory|config|scripts)/[\w./-]+|\w+\.(?:py|md|jsonc?|sh|yaml|yml|toml))\b",
    re.I,
)
_CAMEL_RX = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_TECH_ID_RX = re.compile(r"\b[a-z][a-z0-9]*\d[a-z0-9_-]{2,}\b")
_PROPER_RX = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_QUOTED_RX = re.compile(r'"([^"]{2,40})"')
_MODEL_RX = re.compile(
    r"\b(?:kimi-k2\.6|glm-5\.1|deepseek-v4|qwen3|gemma4|claude-opus|gpt-5|llama3.2|qwen3)\S*\b",
    re.I,
)

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "yes",
        "yeah",
        "ok",
        "done",
        "best",
        "good",
        "also",
        "just",
        "like",
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "will",
        "what",
        "when",
        "where",
        "which",
        "into",
        "than",
        "then",
        "them",
        "they",
        "your",
        "here",
        "there",
        "about",
        "after",
        "before",
        "being",
        "each",
        "more",
        "most",
        "some",
        "such",
        "only",
        "over",
        "very",
        "well",
        "back",
        "make",
        "made",
        "need",
        "want",
        "know",
        "think",
        "still",
        "should",
        "could",
        "would",
        "does",
        "doing",
        "reply",
        "current",
        "tools",
        "rounds",
        "turn",
        "assistant",
        "user",
    }
)


def extract_entities(text: str) -> set[str]:
    """Fast regex entity extraction from a neuron line."""
    if not text or len(text) < 3:
        return set()
    text = text[:_MAX_ENTITY_TEXT_CHARS]

    found: set[str] = set()

    for rx in (_EMAIL_RX, _MODEL_RX):
        for m in rx.finditer(text):
            found.add(m.group(0).lower())

    for m in _CAMEL_RX.finditer(text):
        found.add(m.group(0))

    for m in _TECH_ID_RX.finditer(text):
        tok = m.group(0)
        if len(tok) >= 4 and not tok.isdigit():
            found.add(tok)

    for m in _PROPER_RX.finditer(text):
        phrase = m.group(0)
        words = phrase.split()
        if all(w.lower() not in _STOPWORDS for w in words):
            found.add(phrase)

    for m in _VERSION_RX.finditer(text):
        tok = m.group(0)
        if len(tok) >= 3:
            found.add(tok.lower())

    for m in _PATH_RX.finditer(text):
        found.add(m.group(0).lower())

    for m in _QUOTED_RX.finditer(text):
        q = m.group(1).strip()
        if len(q) >= 3 and q.lower() not in _STOPWORDS:
            found.add(q)

    # Filter stopwords and noise
    clean: set[str] = set()
    for e in found:
        low = e.lower().strip()
        if len(low) < 2 or low in _STOPWORDS:
            continue
        if low.isdigit():
            continue
        normalized = e if e != low else low
        clean.add(normalized[:_MAX_ENTITY_CHARS])
    # A single adversarial line must not turn co-occurrence construction into
    # an unbounded O(n²) operation. Sorting keeps truncation deterministic.
    return set(sorted(clean, key=lambda item: (item.lower(), item))[:_MAX_ENTITIES_PER_TEXT])


def jaccard_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    # Case-insensitive Jaccard
    la = {x.lower() for x in a}
    lb = {x.lower() for x in b}
    inter = len(la & lb)
    union = len(la | lb)
    return inter / union if union else 0.0


def _neuron_fingerprint(neurons: list[Neuron]) -> str:
    h = hashlib.sha256()
    for n in neurons:
        h.update(f"{n.path}|{n.line}|{n.entity_id}".encode())
    return h.hexdigest()[:16]


@dataclass
class EntityGraph:
    """In-memory entity graph with JSON sidecar persistence."""

    root: Path
    entities: dict[str, set[str]] = field(default_factory=dict)  # entity → neuron_ids
    neuron_entities: dict[str, set[str]] = field(default_factory=dict)  # neuron_id → entities
    links: list[tuple[str, str]] = field(default_factory=list)  # co-occurrence pairs
    _fingerprint: str = ""
    _entity_count: int = 0
    _link_count: int = 0
    _truncated: bool = False

    @classmethod
    def from_store(cls, store_root: Path, neurons: list[Neuron]) -> EntityGraph:
        graph = cls(root=store_root)
        graph.rebuild(neurons)
        return graph

    @property
    def sidecar_path(self) -> Path:
        return self.root / "entity_graph.json"

    def rebuild(self, neurons: list[Neuron], *, force: bool = False) -> bool:
        """Rebuild graph from neurons. Returns True if rebuilt."""
        fp = _neuron_fingerprint(neurons)
        if not force and fp == self._fingerprint and self.neuron_entities:
            return False

        # Build into local structures, then publish one coherent snapshot.
        # Background refreshes may run while searches read the previous graph;
        # clearing the live dictionaries first exposed partially rebuilt data.
        entities: dict[str, set[str]] = {}
        neuron_entities: dict[str, set[str]] = {}
        graph_links: set[tuple[str, str]] = set()
        membership_count = 0
        truncated = False

        for neuron_index, n in enumerate(neurons):
            if neuron_index >= _MAX_GRAPH_NEURONS or membership_count >= _MAX_GRAPH_MEMBERSHIPS:
                truncated = True
                break
            ents = extract_entities(n.text)
            if not ents:
                continue
            indexed: set[str] = set()
            for e in sorted(ents, key=lambda item: (item.lower(), item)):
                el = e.lower()
                if el not in entities and len(entities) >= _MAX_GRAPH_ENTITIES:
                    truncated = True
                    continue
                if membership_count >= _MAX_GRAPH_MEMBERSHIPS:
                    truncated = True
                    break
                entities.setdefault(el, set()).add(n.entity_id)
                indexed.add(e)
                membership_count += 1
            if not indexed:
                continue
            neuron_entities[n.entity_id] = indexed
            # Co-occurrence links within neuron
            if len(graph_links) < _MAX_GRAPH_LINKS:
                elist = sorted(indexed, key=str.lower)
                for i, a in enumerate(elist):
                    for b in elist[i + 1 :]:
                        graph_links.add((a.lower(), b.lower()))
                        if len(graph_links) >= _MAX_GRAPH_LINKS:
                            truncated = True
                            break
                    if len(graph_links) >= _MAX_GRAPH_LINKS:
                        break

        links = sorted(graph_links)
        self.entities = entities
        self.neuron_entities = neuron_entities
        self.links = links
        self._fingerprint = fp
        self._entity_count = len(entities)
        self._link_count = len(links)
        self._truncated = truncated
        return True

    def save(self) -> None:
        data = {
            "fingerprint": self._fingerprint,
            "entity_count": self._entity_count,
            "link_count": self._link_count,
            "neuron_count": len(self.neuron_entities),
            "truncated": self._truncated,
            "entities": {k: sorted(v) for k, v in self.entities.items()},
            "neuron_entities": {k: sorted(v) for k, v in self.neuron_entities.items()},
            "links": [list(p) for p in self.links[:_MAX_GRAPH_LINKS]],
        }
        serialized = json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        if len(serialized.encode("utf-8")) > _MAX_SIDECAR_BYTES:
            raise ValueError(f"entity graph sidecar exceeds {_MAX_SIDECAR_BYTES} bytes")
        atomic_write_text(
            self.sidecar_path,
            serialized,
            mode=0o600,
        )

    def load(self) -> bool:
        try:
            data = json.loads(
                read_bounded_text(
                    self.sidecar_path,
                    max_bytes=_MAX_SIDECAR_BYTES,
                )
            )
            if not isinstance(data, dict):
                raise ValueError("entity graph sidecar root must be an object")
            raw_entities = data.get("entities", {})
            raw_neuron_entities = data.get("neuron_entities", {})
            raw_links = data.get("links", [])
            if (
                not isinstance(raw_entities, dict)
                or not isinstance(raw_neuron_entities, dict)
                or not isinstance(raw_links, list)
            ):
                raise ValueError("entity graph collections are malformed")
            if len(raw_entities) > _MAX_GRAPH_ENTITIES:
                raise ValueError("entity graph contains too many entities")
            if len(raw_neuron_entities) > _MAX_GRAPH_NEURONS:
                raise ValueError("entity graph contains too many neurons")
            if len(raw_links) > _MAX_GRAPH_LINKS:
                raise ValueError("entity graph contains too many links")

            neuron_entities: dict[str, set[str]] = {}
            expected_entities: dict[str, set[str]] = {}
            memberships = 0
            for raw_id, raw_values in raw_neuron_entities.items():
                if (
                    not isinstance(raw_id, str)
                    or not raw_id
                    or len(raw_id) > _MAX_NEURON_ID_CHARS
                    or not isinstance(raw_values, list)
                    or len(raw_values) > _MAX_ENTITIES_PER_TEXT
                    or any(
                        not isinstance(value, str) or not value or len(value) > _MAX_ENTITY_CHARS
                        for value in raw_values
                    )
                ):
                    raise ValueError("entity graph contains an invalid neuron mapping")
                memberships += len(raw_values)
                if memberships > _MAX_GRAPH_MEMBERSHIPS:
                    raise ValueError("entity graph contains too many memberships")
                neuron_entities[raw_id] = set(raw_values)
                for value in raw_values:
                    expected_entities.setdefault(value.lower(), set()).add(raw_id)

            entities: dict[str, set[str]] = {}
            reverse_memberships = 0
            for raw_entity, raw_ids in raw_entities.items():
                if (
                    not isinstance(raw_entity, str)
                    or not raw_entity
                    or len(raw_entity) > _MAX_ENTITY_CHARS
                    or not isinstance(raw_ids, list)
                    or any(
                        not isinstance(neuron_id, str)
                        or not neuron_id
                        or len(neuron_id) > _MAX_NEURON_ID_CHARS
                        or neuron_id not in neuron_entities
                        for neuron_id in raw_ids
                    )
                ):
                    raise ValueError("entity graph contains an invalid entity mapping")
                entity = raw_entity.lower()
                if entity in entities:
                    raise ValueError("entity graph contains duplicate normalized entities")
                reverse_memberships += len(raw_ids)
                if reverse_memberships > _MAX_GRAPH_MEMBERSHIPS:
                    raise ValueError("entity graph contains too many reverse memberships")
                entities[entity] = set(raw_ids)
            if entities != expected_entities:
                raise ValueError("entity graph forward and reverse mappings disagree")

            links: list[tuple[str, str]] = []
            for raw_link in raw_links:
                if (
                    not isinstance(raw_link, list)
                    or len(raw_link) != 2
                    or not all(isinstance(value, str) for value in raw_link)
                ):
                    raise ValueError("entity graph contains an invalid link")
                before, after = raw_link
                before = before.lower()
                after = after.lower()
                if (
                    not before
                    or not after
                    or len(before) > _MAX_ENTITY_CHARS
                    or len(after) > _MAX_ENTITY_CHARS
                    or before not in entities
                    or after not in entities
                ):
                    raise ValueError("entity graph contains a dangling link")
                links.append((before, after))

            fingerprint = data.get("fingerprint", "")
            if not isinstance(fingerprint, str) or len(fingerprint) > 128:
                raise ValueError("entity graph fingerprint is invalid")

            # Publish only after the complete sidecar has passed validation.
            self._fingerprint = fingerprint
            self._entity_count = len(entities)
            self._link_count = len(links)
            self._truncated = data.get("truncated") is True
            self.entities = entities
            self.neuron_entities = neuron_entities
            self.links = links
            return True
        except FileNotFoundError:
            return False
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as e:
            log.warning("entity_graph.load_failed: %r", e)
            return False

    def ensure(self, neurons: list[Neuron], *, rebuild_stale: bool = True) -> bool:
        """Load the sidecar and optionally rebuild when stale.

        Returns ``True`` when the loaded graph is current.  Callers on a
        latency-sensitive path can pass ``rebuild_stale=False`` and refresh it
        in a worker without blocking message handling.
        """
        fp = _neuron_fingerprint(neurons)
        loaded = self.load()
        if loaded and self._fingerprint == fp:
            return True
        if not rebuild_stale:
            return False
        if self.rebuild(neurons, force=True):
            self.save()
            log.info(
                "entity_graph rebuilt entities=%d links=%d neurons=%d",
                self._entity_count,
                self._link_count,
                len(self.neuron_entities),
            )
        return self._fingerprint == fp

    def query_entities(self, query: str) -> set[str]:
        return extract_entities(query)

    def score_neuron(self, neuron_id: str, query_entities: set[str]) -> float:
        """Jaccard overlap between query entities and neuron's entities."""
        n_ents = self.neuron_entities.get(neuron_id)
        if not n_ents or not query_entities:
            return 0.0
        return jaccard_overlap(query_entities, n_ents)

    def search(
        self,
        query: str,
        neurons: list[Neuron],
        *,
        k: int = 10,
    ) -> list[tuple[Neuron, float]]:
        """Entity-link search over the full neuron set (not candidate-limited)."""
        q_ents = self.query_entities(query)
        if not q_ents:
            # Fallback: token overlap on entity names in graph
            q_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9_./-]{3,}", query)}
            q_ents = q_tokens

        by_id = {n.entity_id: n for n in neurons}
        hits: list[tuple[Neuron, float]] = []
        for nid, n_ents in self.neuron_entities.items():
            n = by_id.get(nid)
            if n is None:
                continue
            score = jaccard_overlap(q_ents, n_ents)
            if score > 0:
                hits.append((n, score * n.weight))
        hits.sort(key=lambda x: -x[1])
        return hits[:k]

    def stats(self) -> dict:
        return {
            "entities": self._entity_count,
            "links": self._link_count,
            "neurons_linked": len(self.neuron_entities),
            "fingerprint": self._fingerprint,
            "truncated": self._truncated,
        }
