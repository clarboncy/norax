"""IngressBus — bounded queue, back-pressure, structured concurrency."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Final

from ..envelope import SensoryInput

log = logging.getLogger("norax.runtime.ingress_bus")
_STOP: Final = object()


@dataclass(frozen=True)
class _IngressFailure:
    adapter: str
    error: BaseException


class IngressBus:
    """Multiplex N adapters onto a single bounded queue."""

    def __init__(self, adapters: list, *, maxsize: int = 256) -> None:
        self._adapters = adapters
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._tasks: list[asyncio.Task[Any]] = []
        self._started_adapters: list[Any] = []
        self._started = False
        self._stopped = False

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("IngressBus.start() may only be called once")
        if self._stopped:
            raise RuntimeError("IngressBus cannot restart after stop()")
        self._started = True
        try:
            for ad in self._adapters:
                await ad.start()
                self._started_adapters.append(ad)
                t = asyncio.create_task(self._pump(ad), name=f"ingress::{ad.name}")
                t.add_done_callback(self._on_task_done)
                self._tasks.append(t)
        except BaseException:
            await self.stop()
            raise

    async def stop(self, grace: float = 10.0) -> None:
        if self._stopped:
            return
        self._stopped = True

        async def stop_adapter(adapter: Any) -> None:
            try:
                await adapter.stop()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                log.warning("adapter %s stop failed: %s", adapter.name, error)

        # All adapters share one shutdown budget. Applying ``grace`` to each
        # adapter serially made total shutdown time N * grace and let one stuck
        # adapter delay every other cleanup hook.
        stop_tasks = [
            asyncio.create_task(stop_adapter(adapter), name=f"ingress-stop::{adapter.name}")
            for adapter in reversed(self._started_adapters)
        ]
        if stop_tasks:
            try:
                _done, pending = await asyncio.wait(stop_tasks, timeout=max(0.0, grace))
            except BaseException:
                for task in stop_tasks:
                    task.cancel()
                await asyncio.gather(*stop_tasks, return_exceptions=True)
                raise
            for task in pending:
                task.cancel()
                log.warning(
                    "adapter %s stop timed out", task.get_name().removeprefix("ingress-stop::")
                )
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._started_adapters.clear()
        # Wake a stream blocked on an empty queue. If the queue is full, its
        # existing item already wakes the consumer, which then observes stop.
        try:
            self._queue.put_nowait(_STOP)
        except asyncio.QueueFull:
            pass

    async def _pump(self, adapter) -> None:
        configured_fatal = getattr(adapter, "failure_is_fatal", True)
        failure_is_fatal = configured_fatal if isinstance(configured_fatal, bool) else True
        if not isinstance(configured_fatal, bool):
            log.warning(
                "adapter %s has non-boolean failure_is_fatal=%r; treating it as fatal",
                getattr(adapter, "name", type(adapter).__name__),
                configured_fatal,
            )
        try:
            async for env in adapter.events():
                if self._stopped:
                    return
                # The adapter-facing queues are bounded and reject overload at
                # ingress. Once an event is accepted, preserve it here instead
                # of silently dropping it between two bounded queue layers.
                await self._queue.put(env)
            if not self._stopped:
                raise RuntimeError(f"adapter event stream ended unexpectedly: {adapter.name}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.exception("adapter pump crashed: %s", adapter.name)
            if not self._stopped:
                if failure_is_fatal:
                    await self._queue.put(_IngressFailure(str(adapter.name), error))
                    raise
                log.error(
                    "optional ingress disabled after failure: %s error=%r",
                    adapter.name,
                    error,
                )

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("task_crash %s: %r", task.get_name(), exc)

    async def stream(self) -> AsyncIterator[SensoryInput]:
        while True:
            item = await self._queue.get()
            if item is _STOP or self._stopped:
                return
            if isinstance(item, _IngressFailure):
                raise RuntimeError(f"ingress adapter failed: {item.adapter}") from item.error
            yield item
