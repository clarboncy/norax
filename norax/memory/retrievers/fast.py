"""FastContext — keyword/substring retriever over HOT memory + cwd scan.

No embeddings. Uses in-process term matching and fuzzy overlap.
What it's good at:
  - matches exact tokens from the user's message (filenames, IDs, technical terms)
  - surfaces scratchpad and active-focus entries
  - scans working directory for referenced filenames

Fast keyword/context retriever for Norax memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..store import MemoryStore, Neuron

_TOKEN_RX = re.compile(r"[A-Za-z0-9_./-]{3,}")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RX.findall(text)}


@dataclass
class FastContext:
    store: MemoryStore

    def search(self, query: str, *, k: int = 6) -> list[tuple[Neuron, float]]:
        q = _tokens(query)
        if not q:
            return []
        hits: list[tuple[Neuron, float]] = []

        # Score every hot neuron by overlap × weight (recency-weighted for hot)
        for n in self.store.hot:
            overlap = len(q & _tokens(n.text))
            if overlap == 0:
                continue
            score = (overlap / max(len(q), 1)) * n.weight
            hits.append((n, score))

        # Also scan the canonical stores for literal substring matches
        for n in self.store.all_canonical():
            low = n.text.lower()
            sub_hits = sum(1 for t in q if t in low)
            if sub_hits == 0:
                continue
            score = 0.7 * (sub_hits / max(len(q), 1)) * n.weight
            hits.append((n, score))

        # Dedup by entity_id keeping max
        by_ent: dict[str, tuple[Neuron, float]] = {}
        for n, s in hits:
            cur = by_ent.get(n.entity_id)
            if cur is None or s > cur[1]:
                by_ent[n.entity_id] = (n, s)

        ranked = sorted(by_ent.values(), key=lambda x: -x[1])
        return ranked[:k]
