"""Watchdog Timer — detects hung tasks and agent stalls.

Research (Zylos 2026):
  "An agent may silently loop for 35 minutes, spawn redundant subprocesses
   that contend for shared resources, accumulate context until the model
   halts, or take an irreversible action before a human can intervene."

  "The result was a 35-minute hang until an external watchdog sent a
   process signal to exit."

Solution: per-task watchdog timer. If a task exceeds its time budget,
the watchdog fires, cancelling the task and logging the stall.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

log = logging.getLogger("norax.runtime.watchdog")

T = TypeVar("T")


@dataclass
class WatchdogEntry:
    name: str
    started_at: float
    timeout_sec: float


class Watchdog:
    """Monitors tasks for hangs and stalls.

    Usage:
        watchdog = Watchdog()
        result = await watchdog.run("memory_sync", memory_sync(), timeout=300)
    """

    def __init__(self) -> None:
        self._entries: dict[object, WatchdogEntry] = {}
        self._total_timeouts: int = 0
        self._total_completions: int = 0

    async def run(
        self,
        name: str,
        coro: Awaitable[T],
        *,
        timeout: float = 300.0,
        on_timeout: Callable[[str, float], None] | None = None,
    ) -> T:
        """Run a coroutine with a watchdog timer.

        If the coroutine doesn't complete within `timeout` seconds,
        it's cancelled and a TimeoutError is raised.

        Args:
            name: task name for logging
            coro: coroutine to run
            timeout: max seconds before cancellation
            on_timeout: callback(name, elapsed) called on timeout

        Raises:
            asyncio.TimeoutError: if task exceeds timeout
        """
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise ValueError("watchdog name must contain 1-160 characters")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("watchdog timeout must be a positive finite number")
        entry = WatchdogEntry(
            name=name.strip(),
            started_at=time.monotonic(),
            timeout_sec=float(timeout),
        )
        token = object()
        self._entries[token] = entry

        try:
            result = await asyncio.wait_for(coro, timeout=float(timeout))
            self._total_completions += 1
            return result
        except TimeoutError:
            self._total_timeouts += 1
            elapsed = time.monotonic() - entry.started_at
            log.error("watchdog: %s timed out after %.1fs", entry.name, elapsed)
            if on_timeout:
                try:
                    on_timeout(entry.name, elapsed)
                except Exception:  # noqa: BLE001
                    log.exception("watchdog timeout callback failed for %s", entry.name)
            raise
        finally:
            self._entries.pop(token, None)

    def active_tasks(self) -> list[dict[str, Any]]:
        """Return info about tasks being watched."""
        now = time.monotonic()
        return [
            {
                "name": e.name,
                "elapsed_sec": round(now - e.started_at, 1),
                "timeout_sec": e.timeout_sec,
                "remaining_sec": round(e.timeout_sec - (now - e.started_at), 1),
            }
            for e in self._entries.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "active": len(self._entries),
            "total_completions": self._total_completions,
            "total_timeouts": self._total_timeouts,
        }


# Global singleton
_watchdog: Watchdog | None = None


def get_watchdog() -> Watchdog:
    global _watchdog
    if _watchdog is None:
        _watchdog = Watchdog()
    return _watchdog
