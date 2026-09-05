"""Community Detection in Entity Graph — Louvain-style clustering.

Graphiti builds communities of related entities for higher-level knowledge
queries ("what do I know about the X ecosystem?"). This module adds the
same capability to Norax's entity graph.

Implementation:
  - Label propagation algorithm (simpler than full Louvain, O(n) per pass)
  - Multi-pass: 5 iterations max, converges when no node changes community
  - Community summaries: top entities + relation count per community
  - 7th retrieval signal: community-based search boost
  - Incremental: re-runs only when entity graph changes

Output:
  - memory/index/communities.json — {community_id: {entities, size, summary}}
  - Integrated into MultiSignalRetriever as community_weight signal
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text

log = logging.getLogger("norax.memory.community")


def _load_entity_graph(memory_root: Path) -> tuple[dict[str, list], list[list]]:
    """Load entity graph from sidecar JSON. Returns (entities, links)."""
    p = memory_root / "entity_graph.json"
    if not p.exists():
        return {}, []
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        entities = data.get("entities", {})
        links = data.get("links", [])
        return entities, links
    except (json.JSONDecodeError, OSError):
        return {}, []


def _build_adjacency(links: list[list]) -> dict[str, set[str]]:
    """Build adjacency list from links."""
    adj: dict[str, set[str]] = defaultdict(set)
    for link in links:
        if len(link) >= 2:
            src, tgt = str(link[0]), str(link[1])
            adj[src].add(tgt)
            adj[tgt].add(src)
    return adj


def label_propagation(
    nodes: list[str],
    adj: dict[str, set[str]],
    *,
    max_passes: int = 5,
) -> dict[str, int]:
    """Run label propagation community detection.

    Returns {node_name: community_id}.
    """
    # Initialize: each node gets its own community
    labels: dict[str, int] = {node: i for i, node in enumerate(nodes)}

    for pass_num in range(max_passes):
        changed = False
        # Process nodes in random-ish order (sorted for determinism)
        for node in sorted(nodes):
            neighbors = adj.get(node, set())
            if not neighbors:
                continue

            # Count labels among neighbors
            label_counts: dict[int, int] = defaultdict(int)
            for nb in neighbors:
                if nb in labels:
                    label_counts[labels[nb]] += 1

            if not label_counts:
                continue

            # Pick the most common label (ties broken by lowest ID)
            best_label = max(
                label_counts,
                key=lambda label: (label_counts[label], -label),
            )

            if labels[node] != best_label:
                labels[node] = best_label
                changed = True

        if not changed:
            log.debug("community.converged pass=%d", pass_num + 1)
            break

    # Relabel communities to be contiguous (0, 1, 2, ...)
    unique_labels = sorted(set(labels.values()))
    relabel_map = {old: new for new, old in enumerate(unique_labels)}
    return {node: relabel_map[old] for node, old in labels.items()}


def detect_communities(memory_root: Path | None = None) -> dict[str, Any]:
    """Run community detection and save results.

    Returns stats dict with community count, sizes, and timing.
    """
    if memory_root is None:
        memory_root = Path(
            os.environ.get(
                "NORAX_MEMORY_ROOT",
                Path.home() / "norax" / "memory",
            )
        )

    t0 = time.monotonic()

    entities, links = _load_entity_graph(memory_root)
    if not entities:
        log.info("community: no entities found, skipping")
        return {
            "community_count": 0,
            "entity_count": 0,
            "link_count": 0,
            "elapsed_sec": round(time.monotonic() - t0, 4),
        }

    # Get all entity names
    all_entities = list(entities.keys())
    adj = _build_adjacency(links)

    # Run label propagation
    labels = label_propagation(all_entities, adj, max_passes=5)

    # Build community structures
    communities: dict[int, list[str]] = defaultdict(list)
    for entity, comm_id in labels.items():
        communities[comm_id].append(entity)

    # Build output with summaries
    output: dict[str, Any] = {
        "built_at": time.time(),
        "algorithm": "label_propagation",
        "entity_count": len(all_entities),
        "link_count": len(links),
        "community_count": len(communities),
        "communities": {},
    }

    for comm_id, members in sorted(communities.items()):
        # Sort members by ref count (from entities dict)
        member_info: list[dict[str, int | str]] = []
        for name in members:
            ref_count = len(entities.get(name, [])) if isinstance(entities.get(name), list) else 1
            member_info.append({"name": name, "ref_count": ref_count})
        member_info.sort(key=lambda x: x["ref_count"], reverse=True)

        output["communities"][str(comm_id)] = {
            "size": len(members),
            "members": member_info[:20],  # top 20 by ref count
            "top_entities": [m["name"] for m in member_info[:5]],
        }

    elapsed = time.monotonic() - t0
    output["elapsed_sec"] = round(elapsed, 4)

    # Save the complete projection atomically enough for readers: elapsed and
    # schema metadata must be present in both the return value and sidecar.
    index_dir = memory_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    out_path = index_dir / "communities.json"
    atomic_write_text(out_path, json.dumps(output, indent=2, ensure_ascii=False))

    log.info(
        "community.detect: entities=%d links=%d communities=%d (%.3fs)",
        len(all_entities),
        len(links),
        len(communities),
        elapsed,
    )
    return output


def get_community_for_entity(entity_name: str, memory_root: Path | None = None) -> dict | None:
    """Get the community that an entity belongs to."""
    if memory_root is None:
        memory_root = Path(
            os.environ.get(
                "NORAX_MEMORY_ROOT",
                Path.home() / "norax" / "memory",
            )
        )

    p = memory_root / "index" / "communities.json"
    if not p.exists():
        return None

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    for comm_id, comm in data.get("communities", {}).items():
        members = [m["name"] if isinstance(m, dict) else m for m in comm.get("members", [])]
        if entity_name in members:
            return {
                "community_id": comm_id,
                "size": comm.get("size", 0),
                "members": members,
                "top_entities": comm.get("top_entities", []),
            }
    return None


def search_by_community(
    query: str, memory_root: Path | None = None, k: int = 6
) -> list[tuple[str, float]]:
    """Find communities whose top entities match the query.

    Returns list of (entity_name, score) for entities in matching communities.
    """
    if memory_root is None:
        memory_root = Path(
            os.environ.get(
                "NORAX_MEMORY_ROOT",
                Path.home() / "norax" / "memory",
            )
        )

    p = memory_root / "index" / "communities.json"
    if not p.exists():
        return []

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    query_lower = query.lower()
    query_terms = set(query_lower.split())
    results: list[tuple[str, float]] = []

    for comm in data.get("communities", {}).values():
        members = comm.get("members", [])
        top = comm.get("top_entities", [])

        # Score community by how many top entities match query terms
        match_score = 0.0
        for entity_name in top:
            name_lower = entity_name.lower()
            name_terms = set(name_lower.split())
            overlap = len(query_terms & name_terms)
            if overlap > 0:
                match_score += overlap / len(query_terms)

        if match_score > 0:
            # Return all members of matching communities, scored
            for member in members:
                name = member["name"] if isinstance(member, dict) else member
                results.append((name, match_score * 0.5))  # community signal is weaker

    results.sort(key=lambda x: -x[1])
    return results[:k]


def load_communities(memory_root: Path | None = None) -> dict | None:
    """Load the communities JSON file. Returns None if not found."""
    if memory_root is None:
        memory_root = Path(
            os.environ.get(
                "NORAX_MEMORY_ROOT",
                Path.home() / "norax" / "memory",
            )
        )

    p = memory_root / "index" / "communities.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
