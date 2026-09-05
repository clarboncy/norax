"""Runtime ownership boundary for durable memory projections.

Canonical files remain the source of truth. This coordinator serializes and
coalesces refreshes of rebuildable projections (SQLite FTS/entity index) so
turn handling, sleep consolidation, and maintenance cannot race each other.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class MemoryCoordinator:
    """Coordinate canonical writes and derived-index synchronization."""

    def __init__(self, memory_root: Path, *, store: Any | None = None) -> None:
        self.root = Path(memory_root)
        self.store = store
        self._projection_lock = asyncio.Lock()
        self._dirty = False
        self._dirty_reasons: set[str] = set()
        self._generation = 0
        self._indexed_generation = 0
        self._last_stats: dict[str, Any] = {}

    def canonical_changed(self, reason: str) -> int:
        """Record a canonical mutation and return its generation.

        This method is intentionally I/O-free so callers on the runtime event
        loop can cheaply publish mutations. Store refresh and projection work
        happen together under ``sync_projections``' serialization boundary.
        """
        self._generation += 1
        self._dirty = True
        self._dirty_reasons.add(reason)
        return self._generation

    async def run_blocking(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run maintenance work without blocking the runtime event loop."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def consolidate_sleep(self, *, min_age_sec: float = 300.0) -> dict[str, Any]:
        """Drain mature sleep inputs through one ordered consolidation pipeline.

        JSONL rolling-window spills go through ``SleepFlusher`` first because it
        preserves action/outcome pairing and route scoring. Remaining buffer and
        markdown inputs then go through ``MemoryConsolidator``. Both stages run
        in one worker thread, so they cannot archive each other's inputs midway
        through a cycle. Projection synchronization remains a separate,
        coalesced step owned by this coordinator.
        """

        def _drain() -> dict[str, Any]:
            from ..context.sleep_flush import SleepFlusher
            from ..memory.consolidator import MemoryConsolidator
            from ..memory.process_lock import memory_process_lock

            with memory_process_lock(self.root):
                flush = SleepFlusher(
                    memory_root=self.root,
                    min_age_sec=min_age_sec,
                )._flush_unlocked()
                consolidated = MemoryConsolidator(
                    memory_root=self.root,
                    min_age_sec=min_age_sec,
                )._consolidate_unlocked()
            return {
                "spills_processed": flush.spills_processed,
                "candidates_seen": flush.candidates_seen,
                "flush_written": dict(flush.written),
                "files_processed": consolidated.files_processed,
                "facts_written": consolidated.facts_written,
                "procedural_written": consolidated.procedural_written,
                "duplicates_skipped": consolidated.skipped_dup,
            }

        result = await self.run_blocking(_drain)
        flush_written = sum(result["flush_written"].values())
        canonical_writes = flush_written + result["facts_written"] + result["procedural_written"]
        result["canonical_writes"] = canonical_writes
        if canonical_writes:
            self.canonical_changed("sleep_consolidation")
        return result

    async def sync_projections(self, *, force: bool = False) -> dict[str, Any]:
        """Incrementally synchronize all derived retrieval projections.

        Concurrent callers collapse behind one lock. A generation snapshot
        prevents a write arriving during indexing from being incorrectly
        marked synchronized.
        """
        if not force and not self._dirty:
            return self._last_stats

        async with self._projection_lock:
            if not force and not self._dirty:
                return self._last_stats

            target_generation = self._generation
            reasons = sorted(self._dirty_reasons)

            def _rebuild() -> dict[str, Any]:
                from ..memory.build_index import _build_memory_index_unlocked
                from ..memory.process_lock import memory_process_lock

                with memory_process_lock(self.root):
                    result = _build_memory_index_unlocked(self.root)
                    if self.store is not None:
                        self.store.refresh()
                    return result

            result = await asyncio.to_thread(_rebuild)
            self._last_stats = result
            self._indexed_generation = target_generation

            if self._generation == target_generation:
                self._dirty = False
                self._dirty_reasons.clear()
            else:
                self._dirty_reasons.difference_update(reasons)

            delta = result.get("stats", {})
            if delta.get("files_indexed") or delta.get("chunks_deleted"):
                log.info(
                    "memory_projection_sync: reasons=%s files=%d chunks_added=%d chunks_deleted=%d",
                    ",".join(reasons) or "forced",
                    delta.get("files_indexed", 0),
                    delta.get("chunks_added", 0),
                    delta.get("chunks_deleted", 0),
                )
            return result
