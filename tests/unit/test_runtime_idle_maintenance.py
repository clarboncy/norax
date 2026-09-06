"""Idle maintenance must run independently of spills and yield to user work."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from norax.runtime.core import Runtime


def _runtime(tmp_path, monkeypatch):
    runtime = Runtime.__new__(Runtime)
    runtime._last_turn_time = 0
    runtime._turn_work_count = 0
    runtime._active_turn_tasks = {}
    runtime._memory_store = SimpleNamespace(root=tmp_path)
    runtime._memory_coordinator = SimpleNamespace(
        consolidate_sleep=AsyncMock(
            return_value={
                "canonical_writes": 1,
                "files_processed": 1,
                "spills_processed": 0,
                "candidates_seen": 1,
                "duplicates_skipped": 0,
            }
        ),
        sync_projections=AsyncMock(),
    )
    runtime.events = SimpleNamespace(append=AsyncMock())
    runtime._idle_learning_enabled = False
    runtime._episodic = SimpleNamespace(prune_old=Mock())
    runtime._skill_learner = None
    runtime._metacognitive = None
    runtime._harness_analysis_enabled = False
    checks = 0

    async def one_poll(_seconds):
        nonlocal checks
        checks += 1
        if checks > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr("norax.runtime.cognition.asyncio.sleep", one_poll)
    return runtime


@pytest.mark.asyncio
async def test_idle_pruning_and_projection_sync_do_not_require_sleep_files(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch)
    await runtime._idle_sleep_loop()

    runtime._memory_coordinator.consolidate_sleep.assert_not_awaited()
    runtime._memory_coordinator.sync_projections.assert_awaited_once()
    runtime._episodic.prune_old.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_kind", ["queued", "active"])
async def test_long_running_turn_prevents_idle_maintenance(tmp_path, monkeypatch, busy_kind):
    runtime = _runtime(tmp_path, monkeypatch)
    (tmp_path / "sleep").mkdir()
    (tmp_path / "sleep" / "buffer-1.md").write_text("pending memory\n")
    if busy_kind == "queued":
        runtime._turn_work_count = 1
    else:
        runtime._active_turn_tasks = {"channel": SimpleNamespace(done=lambda: False)}
    await runtime._idle_sleep_loop()

    runtime._memory_coordinator.consolidate_sleep.assert_not_awaited()
    runtime._memory_coordinator.sync_projections.assert_not_awaited()
    runtime._episodic.prune_old.assert_not_called()


@pytest.mark.asyncio
async def test_idle_sleep_writes_still_refresh_memory_and_record_evidence(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch)
    (tmp_path / "sleep").mkdir()
    (tmp_path / "sleep" / "buffer-1.md").write_text("pending memory\n")
    await runtime._idle_sleep_loop()

    runtime._memory_coordinator.consolidate_sleep.assert_awaited_once()
    assert runtime._memory_coordinator.sync_projections.await_count >= 1
    runtime.events.append.assert_awaited_once()
    assert runtime.events.append.call_args.args[0] == "sleep_consolidation"


@pytest.mark.asyncio
async def test_skill_file_work_runs_off_the_serving_event_loop(tmp_path, monkeypatch):
    import threading

    runtime = _runtime(tmp_path, monkeypatch)
    runtime.events.path = tmp_path / "events.jsonl"
    runtime.events.path.write_text("")
    runtime._memory_coordinator.canonical_changed = Mock()
    serving_thread = threading.get_ident()
    observed: list[tuple[str, int]] = []

    def load(*args, **kwargs):
        observed.append(("load", threading.get_ident()))
        return ["verified trajectory"]

    def mine(trajectories):
        assert trajectories == ["verified trajectory"]
        observed.append(("mine", threading.get_ident()))
        return ["verified pattern"]

    def generate(patterns):
        assert patterns == ["verified pattern"]
        observed.append(("generate", threading.get_ident()))
        return SimpleNamespace(skills_created=1, skills_updated=0, skills=["verified skill"])

    monkeypatch.setattr("norax.brain.harness_optimizer.load_trajectories", load)
    runtime._skill_learner = SimpleNamespace(mine=mine, generate=generate)
    await runtime._idle_sleep_loop()

    assert [name for name, _ in observed] == ["load", "mine", "generate"]
    assert all(thread != serving_thread for _, thread in observed)
    runtime._memory_coordinator.canonical_changed.assert_called_once_with("skill_learning")
