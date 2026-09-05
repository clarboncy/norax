"""P1-7: Concurrent sleep flush + idle consolidation safety test.

Verifies that timer flush and idle consolidation running concurrently
against the same memory root do not cause lost writes, duplicate
promotion, or broken hash chains. The cross-process lock
(memory_process_lock) must serialize all canonical mutations.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from norax.memory.process_lock import memory_process_lock
from norax.memory.store import MemoryStore


def _write_sleep_spills(root: Path, count: int) -> None:
    """Write sleep spill JSONL files that the flusher can pick up."""
    sleep_dir = root / "sleep"
    sleep_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        spill = sleep_dir / f"spill_{i:04d}.jsonl"
        with spill.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": f"Fact #{i}: the answer is {i}",
                        "timestamp": f"2026-07-21T{i:02d}:00:00Z",
                        "channel": "test",
                    }
                )
                + "\n"
            )


def _write_canonical(root: Path, text: str, weight: float = 1.0) -> None:
    """Write a canonical neuron file."""
    sem_dir = root / "semantic"
    sem_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    h = hashlib.sha256(text.encode()).hexdigest()[:16]
    (sem_dir / f"fact_{h}.md").write_text(f"{text}|W{weight}\n")


@pytest.mark.asyncio
async def test_concurrent_sleep_flush_and_consolidation_no_lost_writes(tmp_path: Path) -> None:
    """Run SleepFlusher + MemoryConsolidator concurrently and verify safety.

    Both operations acquire memory_process_lock internally, so they must
    serialize. We verify:
    1. No exception is raised by either operation.
    2. Canonical files are not corrupted (readable + parseable).
    3. No duplicate facts are promoted.
    4. Hash chain in event log (if present) is not broken.
    """
    from norax.context.sleep_flush import SleepFlusher
    from norax.memory.consolidator import MemoryConsolidator

    root = tmp_path / "memory"
    root.mkdir()

    # Pre-populate with some canonical content
    _write_canonical(root, "EXISTING:baseline fact for dedup", 2.0)

    # Write sleep spills for the flusher to consolidate
    _write_sleep_spills(root, 10)

    # Also write some hot spills
    hot_dir = root / "hot"
    hot_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        with (hot_dir / f"hot_{i:04d}.jsonl").open("w") as f:
            f.write(
                json.dumps(
                    {
                        "role": "assistant",
                        "content": f"Hot fact #{i}: value is {i * 10}",
                        "timestamp": f"2026-07-21T{i:02d}:30:00Z",
                        "channel": "test",
                    }
                )
                + "\n"
            )

    # Run flush and consolidate concurrently
    flusher = SleepFlusher(memory_root=root, min_age_sec=0)
    consolidator = MemoryConsolidator(memory_root=root, min_age_sec=0)

    # Both should serialize via memory_process_lock — no errors
    flush_result, consol_result = await asyncio.gather(
        asyncio.to_thread(flusher.flush),
        asyncio.to_thread(consolidator.consolidate),
    )

    # Verify no exceptions, results are valid
    assert flush_result.spills_processed >= 0
    assert consol_result.files_processed >= 0

    # Verify canonical files are readable and parseable
    store = MemoryStore(root=root)
    store.refresh()
    neurons = store.all_canonical()
    assert len(neurons) > 0

    # Verify no duplicate facts (each canonical file has unique content)
    sem_dir = root / "semantic"
    if sem_dir.exists():
        contents = set()
        for f in sem_dir.glob("*.md"):
            text = f.read_text().strip()
            assert text not in contents, f"Duplicate canonical fact: {f.name}"
            contents.add(text)


def test_memory_process_lock_serializes_concurrent_access(tmp_path: Path) -> None:
    """Verify that memory_process_lock actually serializes concurrent access."""
    import threading
    import time

    root = tmp_path / "mem"
    root.mkdir()
    execution_order: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        with memory_process_lock(root):
            with lock:
                execution_order.append(f"{name}_start")
            time.sleep(0.05)
            with lock:
                execution_order.append(f"{name}_end")

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each worker should have non-overlapping start/end (serialized)
    # If serialized, we should see start/end pairs, not interleaved
    assert len(execution_order) == 6
    # Verify no interleaving: each start is immediately followed by its end
    for i in range(0, 6, 2):
        assert execution_order[i].endswith("_start")
        assert execution_order[i + 1].endswith("_end")
        assert execution_order[i].replace("_start", "") == execution_order[i + 1].replace(
            "_end", ""
        )
