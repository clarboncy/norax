"""Runtime memory coordinator: projection ownership and race handling."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from norax.memory.process_lock import memory_process_lock
from norax.runtime.memory_coordinator import MemoryCoordinator


class _Store:
    def __init__(self) -> None:
        self.refreshes = 0

    def refresh(self) -> None:
        self.refreshes += 1


@pytest.mark.asyncio
async def test_clean_sync_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def build(_root: Path):
        nonlocal calls
        calls += 1
        return {"stats": {}}

    monkeypatch.setattr("norax.memory.build_index._build_memory_index_unlocked", build)
    coordinator = MemoryCoordinator(tmp_path)

    assert await coordinator.sync_projections() == {}
    assert calls == 0


def test_change_notification_is_io_free(tmp_path: Path) -> None:
    store = _Store()
    coordinator = MemoryCoordinator(tmp_path, store=store)

    generation = coordinator.canonical_changed("turn")

    assert generation == 1
    assert coordinator._dirty is True
    assert store.refreshes == 0


@pytest.mark.asyncio
async def test_concurrent_syncs_coalesce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def build(_root: Path):
        nonlocal calls
        calls += 1
        return {"stats": {"files_indexed": 1}}

    monkeypatch.setattr("norax.memory.build_index._build_memory_index_unlocked", build)
    store = _Store()
    coordinator = MemoryCoordinator(tmp_path, store=store)
    coordinator.canonical_changed("turn")

    results = await asyncio.gather(*[coordinator.sync_projections() for _ in range(8)])

    assert calls == 1
    assert store.refreshes == 1
    assert all(result == results[0] for result in results)
    assert coordinator._dirty is False


@pytest.mark.asyncio
async def test_write_during_build_remains_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_to_thread(func, *args, **kwargs):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"stats": {}}

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    coordinator = MemoryCoordinator(tmp_path)
    coordinator.canonical_changed("first")

    task = asyncio.create_task(coordinator.sync_projections())
    await started.wait()
    coordinator.canonical_changed("second")
    release.set()
    await task

    assert coordinator._dirty is True
    await coordinator.sync_projections()
    assert calls == 2
    assert coordinator._dirty is False


@pytest.mark.asyncio
async def test_sleep_pipeline_flushes_jsonl_before_generic_consolidation(
    tmp_path: Path,
) -> None:
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    spill = sleep / "spill-20260101-000000-000000.jsonl"
    spill.write_text(
        '{"role":"user","content":"DEPLOYMENT_POLICY: verification uses health checks |W5"}\n',
        encoding="utf-8",
    )

    coordinator = MemoryCoordinator(tmp_path)
    result = await coordinator.consolidate_sleep(min_age_sec=0)

    assert result["spills_processed"] == 1
    assert result["candidates_seen"] == 1
    assert result["canonical_writes"] == 1
    assert coordinator._dirty is True
    assert not spill.exists()
    assert any((sleep / "archive").glob("spill-*.jsonl"))
    canonical = list((tmp_path / "semantic").glob("sleep-flush-*.md"))
    assert canonical
    assert "DEPLOYMENT_POLICY" in canonical[0].read_text(encoding="utf-8")


def test_memory_process_lock_is_reentrant_across_sequential_owners(tmp_path: Path) -> None:
    lock_path = tmp_path / ".maintenance.lock"

    with memory_process_lock(tmp_path):
        assert lock_path.exists()
    with memory_process_lock(tmp_path):
        assert lock_path.exists()


def test_memory_process_lock_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.lock"
    outside.write_text("unchanged", encoding="utf-8")
    (tmp_path / ".maintenance.lock").symlink_to(outside)

    with pytest.raises(OSError):
        with memory_process_lock(tmp_path):
            pass

    assert outside.read_text(encoding="utf-8") == "unchanged"
