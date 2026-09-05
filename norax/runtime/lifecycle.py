"""Turn ownership, service supervision, draining, and shutdown."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from ._mixin import RuntimeAccessMixin
from .validation import _flag_enabled

log = logging.getLogger("norax.runtime.core")


class LifecycleMixin(RuntimeAccessMixin):
    _a2a_uvicorn: Any
    _active_turn_tasks: dict[str, asyncio.Task[Any]]
    _autonomy_task: asyncio.Task[Any] | None
    _capability_probe_task: asyncio.Task[None] | None
    _draining: bool
    _probe_task: asyncio.Task[None] | None
    _shutdown_complete: bool
    _shutdown_lock: asyncio.Lock | None
    _sleep_task: asyncio.Task[None] | None
    _turn_work_count: int
    _warmup_task: asyncio.Task[None] | None
    a2a_server: Any
    mcp_client: Any

    @staticmethod
    def _channel_id(env: Any) -> str:
        raw = getattr(env, "raw", None)
        raw_channel = raw.get("channel_id") if isinstance(raw, dict) else None
        return str(raw_channel or getattr(env, "channel", None) or "default")

    def _track_turn_task(self, channel_id: str, task: asyncio.Task[Any]) -> None:
        """Own one channel task and release it as soon as it terminates."""
        self._active_turn_tasks[channel_id] = task

        def _finished(completed: asyncio.Task[Any]) -> None:
            if self._active_turn_tasks.get(channel_id) is completed:
                self._active_turn_tasks.pop(channel_id, None)
            queue = self._turn_queues.get(channel_id)
            if queue is not None and not queue:
                self._turn_queues.pop(channel_id, None)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                log.error(
                    "turn_task.crashed channel=%s error=%r",
                    channel_id,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(_finished)

    def _queue_turn(self, channel_id: str, env: Any) -> bool:
        """Queue one turn without ever blocking the global ingress consumer."""
        queue = self._turn_queues.setdefault(channel_id, deque())
        worker = self._active_turn_tasks.get(channel_id)
        channel_work = len(queue) + int(worker is not None and not worker.done())
        if (
            self._turn_work_count >= self._max_pending_turns
            or channel_work >= self._max_pending_per_channel
        ):
            log.warning(
                "turn_queue.full channel=%s channel_work=%d total_work=%d",
                channel_id,
                channel_work,
                self._turn_work_count,
            )
            if not queue:
                self._turn_queues.pop(channel_id, None)
            return False
        queue.append(env)
        self._turn_work_count += 1
        if worker is None or worker.done():
            worker = asyncio.create_task(
                self._run_channel_turns(channel_id),
                name=f"turn::{channel_id}",
            )
            self._track_turn_task(channel_id, worker)
        return True

    async def _run_channel_turns(self, channel_id: str) -> None:
        """Run one channel sequentially under the global turn concurrency cap."""
        queue = self._turn_queues[channel_id]
        try:
            while queue and not self._draining:
                env = queue.popleft()
                try:
                    await self._turn_semaphore.acquire()
                    slot_released = False
                    owner_task = asyncio.current_task()

                    def release_slot() -> None:
                        nonlocal slot_released
                        if not slot_released:
                            slot_released = True
                            self._turn_semaphore.release()

                    if owner_task is not None:
                        self._turn_slot_releasers[owner_task] = release_slot
                    try:
                        if self._draining:
                            return
                        await self._handle_turn(env)
                    finally:
                        if owner_task is not None:
                            self._turn_slot_releasers.pop(owner_task, None)
                        release_slot()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    log.exception("turn worker failed channel=%s", channel_id)
                finally:
                    self._turn_work_count = max(0, self._turn_work_count - 1)
        finally:
            if queue:
                self._turn_work_count = max(0, self._turn_work_count - len(queue))
                queue.clear()
            if self._turn_queues.get(channel_id) is queue:
                self._turn_queues.pop(channel_id, None)
            self._stop_channels.discard(channel_id)

    def _release_current_turn_slot(self) -> bool:
        """Release global turn capacity once the completed reply is delivered."""
        task = asyncio.current_task()
        if task is None:
            return False
        release = self._turn_slot_releasers.pop(task, None)
        if release is None:
            return False
        release()
        return True

    def _track_optional_task(self, capability: str, task: asyncio.Task[Any]) -> None:
        """Track a connector task and invalidate readiness if it exits."""
        if task not in self._optional_tasks:
            self._optional_tasks.append(task)

        self._observe_service_task(capability, task)

    def _observe_service_task(self, capability: str, task: asyncio.Task[Any]) -> None:
        """Turn an unexpected long-running task exit into capability evidence."""

        def _finished(completed: asyncio.Task[Any]) -> None:
            if self._draining:
                return
            if completed.cancelled():
                cancellation_error = RuntimeError("service cancelled unexpectedly")
                self.capabilities.mark_failed(capability, cancellation_error)
                log.error("subsystem cancelled unexpectedly: %s", capability)
                return
            task_error = completed.exception()
            if task_error is None:
                task_error = RuntimeError("service stopped unexpectedly")
                log.error("subsystem stopped unexpectedly: %s", capability)
            self.capabilities.mark_failed(capability, task_error)

        task.add_done_callback(_finished)

    def _record_mcp_health(self, status: dict[str, Any]) -> None:
        """Keep MCP capability state aligned with live connection evidence."""
        if self._draining:
            return
        active = [str(name) for name in status.get("active") or []]
        healthy = [str(name) for name in status.get("healthy") or []]
        failures = {
            str(name): str(error)[:500]
            for name, error in dict(status.get("failures") or {}).items()
        }
        if failures and healthy:
            self.capabilities.mark_degraded(
                "mcp_client",
                f"failed connections: {', '.join(sorted(failures))}",
            )
        elif failures:
            details = "; ".join(f"{name}: {error}" for name, error in sorted(failures.items()))
            self.capabilities.mark_failed(
                "mcp_client",
                RuntimeError(f"all MCP connections failed: {details}"),
            )
        elif active:
            self.capabilities.mark_ok("mcp_client")
        else:
            self.capabilities.mark_degraded("mcp_client", "no live MCP connections")
        self.capabilities.update_metadata(
            "mcp_client",
            active=active,
            healthy=healthy,
            failed=sorted(failures),
        )

    def _track_mcp_connection_task(
        self,
        client: Any,
        name: str,
        task: asyncio.Task[Any],
    ) -> None:
        if task not in self._optional_tasks:
            self._optional_tasks.append(task)

        def _finished(completed: asyncio.Task[Any]) -> None:
            if self._draining:
                return
            if completed.cancelled():
                detail = "connection owner cancelled unexpectedly"
            else:
                error = completed.exception()
                detail = (
                    f"{type(error).__name__}: {error}"
                    if error is not None
                    else "connection owner stopped unexpectedly"
                )
            status = dict(client.health())
            failures = dict(status.get("failures") or {})
            failures[name] = detail[:500]
            status["failures"] = failures
            status["healthy"] = [item for item in status.get("healthy") or [] if str(item) != name]
            self._record_mcp_health(status)

        task.add_done_callback(_finished)

    @staticmethod
    async def _await_uvicorn_started(
        server: Any,
        task: asyncio.Task[Any],
        *,
        timeout: float = 5.0,
    ) -> None:
        """Wait for a real bound listener instead of advertising readiness early."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not bool(getattr(server, "started", False)):
            if task.done():
                if task.cancelled():
                    raise RuntimeError("A2A listener was cancelled during startup")
                error = task.exception()
                if error is not None:
                    raise RuntimeError("A2A listener failed during startup") from error
                raise RuntimeError("A2A listener stopped before becoming ready")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("A2A listener did not become ready within 5 seconds")
            await asyncio.sleep(0.01)

    def _mount_commerce_connector(self) -> None:
        """Mount commerce routes before HTTP ingress begins accepting requests."""
        commerce_cfg = self._connectors.get("commerce", {})
        if not _flag_enabled(commerce_cfg.get("enabled")):
            return
        self.capabilities.register(
            "commerce_api",
            required_for_readiness=_flag_enabled(commerce_cfg.get("required")),
        )
        self.capabilities.mark_initializing("commerce_api")
        try:
            from ..commerce.api_endpoints import mount_commerce

            http_adapter = next(
                (adapter for adapter in self.ingress._adapters or [] if hasattr(adapter, "app")),
                None,
            )
            if http_adapter is None:
                raise RuntimeError("no HTTP adapter found")
            mount_commerce(http_adapter.app)
            self.capabilities.mark_ok("commerce_api")
            log.info("commerce_api mounted on HTTP adapter")
        except Exception as error:
            self.capabilities.mark_failed("commerce_api", error)
            if commerce_cfg.get("required"):
                raise RuntimeError("required commerce API failed to mount") from error

    @staticmethod
    async def _settle_tasks(
        tasks: list[asyncio.Task[Any]],
        *,
        timeout: float,
        cancel: bool,
        label: str,
    ) -> set[asyncio.Task[Any]]:
        """Bounded task cleanup that consumes terminal exceptions."""
        current = asyncio.current_task()
        pending_set = {task for task in tasks if task is not current and not task.done()}
        if cancel:
            for task in pending_set:
                task.cancel()
        done: set[asyncio.Task[Any]] = {task for task in tasks if task.done()}
        pending: set[asyncio.Task[Any]] = set()
        if pending_set:
            newly_done, pending = await asyncio.wait(
                pending_set,
                timeout=max(0.0, timeout),
            )
            done.update(newly_done)
        for task in done:
            if task.cancelled():
                continue
            try:
                error = task.exception()
            except asyncio.CancelledError:
                continue
            if error is not None:
                log.warning("%s task failed: %r", label, error)
        if pending:
            log.warning("%s cleanup timed out with %d task(s) pending", label, len(pending))
        return pending

    async def _drain_active_turns(self, grace: float) -> None:
        active = [task for task in self._active_turn_tasks.values() if not task.done()]
        pending = await self._settle_tasks(
            active,
            timeout=grace,
            cancel=False,
            label="active turn drain",
        )
        if pending:
            still_pending = await self._settle_tasks(
                list(pending),
                timeout=min(2.0, max(0.1, grace)),
                cancel=True,
                label="active turn cancellation",
            )
            if still_pending:
                log.error(
                    "shutdown continuing after cancellation-resistant turn tasks: %d",
                    len(still_pending),
                )
        self._active_turn_tasks = {
            channel: task for channel, task in self._active_turn_tasks.items() if not task.done()
        }

    async def shutdown(self) -> None:
        if self._shutdown_lock is None:
            self._shutdown_lock = asyncio.Lock()
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._draining = True
            grace = self.cfg.shutdown_grace_seconds

            core_background = [
                task
                for task in (
                    self._capability_probe_task,
                    self._sleep_task,
                    self._warmup_task,
                    self._probe_task,
                    self._autonomy_task,
                )
                if task is not None
            ]
            for task in core_background:
                if not task.done():
                    task.cancel()

            # Ask the HTTP listener to leave its serve loop before using task
            # cancellation as the bounded fallback.
            if self._a2a_uvicorn is not None:
                self._a2a_uvicorn.should_exit = True

            try:
                await self.ingress.stop(grace=grace)
            except Exception:  # noqa: BLE001
                log.warning("ingress.shutdown_failed", exc_info=True)

            if self.a2a_server is not None:
                try:
                    await self.a2a_server.close()
                except Exception:  # noqa: BLE001
                    log.warning("a2a.shutdown_failed", exc_info=True)
                self.a2a_server = None

            reminders = self.reminders
            if reminders is not None:
                try:
                    await reminders.stop()
                except Exception:  # noqa: BLE001
                    log.warning("reminders.shutdown_failed", exc_info=True)

            await self._drain_active_turns(grace)
            await self._settle_tasks(
                list(self._maintenance_tasks),
                timeout=min(5.0, max(0.1, grace)),
                cancel=False,
                label="maintenance persistence",
            )
            await self._settle_tasks(
                core_background,
                timeout=min(2.0, max(0.1, grace)),
                cancel=True,
                label="background subsystem",
            )
            optional_pending = await self._settle_tasks(
                list(self._optional_tasks),
                timeout=min(2.0, max(0.1, grace)),
                cancel=False,
                label="optional subsystem",
            )
            if optional_pending:
                await self._settle_tasks(
                    list(optional_pending),
                    timeout=min(1.0, max(0.1, grace)),
                    cancel=True,
                    label="optional subsystem cancellation",
                )
            self._optional_tasks.clear()
            self._a2a_uvicorn = None
            self._capability_probe_task = None
            self._sleep_task = None
            self._warmup_task = None
            self._probe_task = None
            self._autonomy_task = None

            # A cancellation-resistant turn must not retain UI background
            # helpers beyond the process lifecycle.
            for streaming in list(self._active_stream_messages.values()):
                try:
                    await streaming.delete()
                except Exception:  # noqa: BLE001
                    log.debug("stream.shutdown_cleanup_failed", exc_info=True)
            self._active_stream_messages.clear()
            for typing_keep in list(self._active_typing_handles.values()):
                try:
                    await typing_keep.stop(timeout=1.5)
                except Exception:  # noqa: BLE001
                    log.debug("typing.shutdown_cleanup_failed", exc_info=True)
            self._active_typing_handles.clear()
            self._stop_channels.clear()

            # Active turns are drained, so these snapshots capture terminal
            # state instead of racing normal turn or idle-learning writers.
            curiosity = self._curiosity
            if curiosity is not None:
                try:
                    curiosity.save()
                except Exception:  # noqa: BLE001
                    log.warning("curiosity.shutdown_save_failed", exc_info=True)
            for graph in (self._causal_graph, self._temporal_graph):
                if graph is not None:
                    try:
                        graph.save()
                    except Exception:  # noqa: BLE001
                        log.warning("graph.shutdown_save_failed", exc_info=True)
            if self._user_model is not None:
                try:
                    self._user_model.flush()
                except Exception:  # noqa: BLE001
                    log.warning("user_model.shutdown_flush_failed", exc_info=True)
            if self._memory_store is not None:
                try:
                    self._hebbian.flush(self._memory_store.root)
                except Exception:  # noqa: BLE001
                    log.warning("hebbian.shutdown_flush_failed", exc_info=True)

            if self.mcp_client is not None:
                try:
                    await self.mcp_client.disconnect_all()
                except Exception:  # noqa: BLE001
                    log.warning("mcp.shutdown_failed", exc_info=True)
                self.mcp_client = None

            # Close the embedder's client before its owning gateway clients.
            if self._hybrid is not None:
                embedder = getattr(self._hybrid.embedding, "embedder", None)
                if embedder is not None and hasattr(embedder, "close"):
                    try:
                        await embedder.close()
                    except Exception:  # noqa: BLE001
                        log.debug("embedder.close_failed", exc_info=True)
            try:
                await self.gateway.aclose()
            except Exception:  # noqa: BLE001
                log.warning("gateway.shutdown_failed", exc_info=True)
            try:
                await self.events.append("runtime.stop", {})
            except Exception:  # noqa: BLE001
                log.warning("event_log.shutdown_append_failed", exc_info=True)
            self._shutdown_complete = True
