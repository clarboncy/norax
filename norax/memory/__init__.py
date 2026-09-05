"""Memory layer — multi-signal retrieval with smart context injection.

Retrievers:
  - FastContext — keyword/substring over *hot* memory (scratchpad + focus)
                  and the current working directory. No embeddings.
  - LocalRetriever — embedding kNN over canonical stores
                  (semantic/, procedural/, intel/). ~100ms cold, cached after.
  - MultiSignalRetriever — 3-signal RRF fusion: keyword + embedding + entity-link
  - ExternalRetriever — embedding kNN over the rolling sleep/ buffer
                  (what sleep-flush is about to consolidate) and external docs.

Entity linking:
  - EntityGraph — regex entity extraction + Jaccard scoring; JSON sidecar

Consolidation:
  - MemoryConsolidator — ADD-only sleep → canonical distillation pipeline

Injection:
  - ContextInjector — runs all retrievers in parallel, merges with MMR
                  de-dupe, applies recency decay + tier weights, and fits
                  everything into a per-turn token budget.

Norax memory design choices:
  - norax-embed-v3 (384d) via Ollama :11436 (pluggable; stub for offline tests)
  - Compressed KV neuron format (one fact per line, W-tag weighted)
  - Store = flat directory of Markdown files, NOT a DB (grep-friendly, git-diffable)
  - Index = numpy .npz sidecar rebuilt on-demand when mtimes change
"""

from .consolidator import ConsolidateResult, MemoryConsolidator
from .embeddings import Embedder, OllamaEmbedder, StubEmbedder
from .entity_graph import EntityGraph, extract_entities, jaccard_overlap
from .hot_inject import (
    capture_directive,
    compact_scratchpad,
    merge_pinned,
    pin_hot_neurons,
    post_turn_hot_maintenance,
    refresh_hot_identity,
    update_active_focus,
)
from .injection import ContextInjector, InjectionResult
from .retrievers.external import ExternalRetriever
from .retrievers.fast import FastContext
from .retrievers.local import LocalRetriever
from .retrievers.multi_signal import MultiSignalRetriever
from .retrievers.sqlite_index import SQLiteIndexRetriever
from .store import MemoryStore, Neuron

__all__ = [
    "MemoryStore",
    "Neuron",
    "Embedder",
    "OllamaEmbedder",
    "StubEmbedder",
    "EntityGraph",
    "extract_entities",
    "jaccard_overlap",
    "pin_hot_neurons",
    "merge_pinned",
    "refresh_hot_identity",
    "update_active_focus",
    "capture_directive",
    "post_turn_hot_maintenance",
    "compact_scratchpad",
    "MemoryConsolidator",
    "ConsolidateResult",
    "FastContext",
    "LocalRetriever",
    "MultiSignalRetriever",
    "ExternalRetriever",
    "SQLiteIndexRetriever",
    "ContextInjector",
    "InjectionResult",
]
