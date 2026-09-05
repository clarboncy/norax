"""Background entity refresh must be real, non-blocking, and coalesced."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from norax.memory.retrievers.multi_signal import MultiSignalRetriever


class _Graph:
    def __init__(self) -> None:
        self.foreground_checks = 0
        self.rebuilds = 0

    def ensure(self, neurons, *, rebuild_stale: bool = True):
        if not rebuild_stale:
            self.foreground_checks += 1
            return False
        self.rebuilds += 1
        return True


@pytest.mark.asyncio
async def test_background_entity_refresh_returns_immediately_and_coalesces(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_to_thread(function, *args, **kwargs):
        started.set()
        await release.wait()
        return function(*args, **kwargs)

    monkeypatch.setattr(
        "norax.memory.retrievers.multi_signal.asyncio.to_thread",
        controlled_to_thread,
    )
    graph = _Graph()
    keyword = SimpleNamespace(store=SimpleNamespace(all_canonical=lambda: [], hot=[]))
    retriever = MultiSignalRetriever(keyword=keyword, entity_graph=graph)  # type: ignore[arg-type]

    await retriever.ensure_index(background_refresh=True)
    first_task = retriever._entity_refresh_task
    assert first_task is not None
    await started.wait()

    await retriever.ensure_index(background_refresh=True)
    assert retriever._entity_refresh_task is first_task
    # Even sidecar freshness checks stay off the latency-sensitive caller.
    assert graph.foreground_checks == 0
    assert graph.rebuilds == 0

    release.set()
    await first_task
    assert graph.rebuilds == 1
