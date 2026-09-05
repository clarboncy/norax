"""Idempotency guard for duplicate concurrent task execution.

Callers acquire a deterministic name/argument key before spawning work and
release it in ``finally``. Optional stale reaping is disabled by default: an
arbitrary one-hour lease can duplicate a legitimate long-running task.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("norax.runtime.idempotency")


@dataclass
class RunningTask:
    key: str
    name: str
    started_at: float
    task: asyncio.Task | None = None
    args_hash: str = ""


class IdempotencyGuard:
    """Prevents duplicate concurrent tasks.

    Tracks running tasks by a key (usually tool+args hash). A duplicate
    acquisition returns ``(False, existing_key)``.
    """

    def __init__(self, *, stale_timeout_sec: float | None = None) -> None:
        self._running: dict[str, RunningTask] = {}
        self.stale_timeout_sec = (
            float(stale_timeout_sec)
            if stale_timeout_sec is not None and float(stale_timeout_sec) > 0
            else None
        )
        self._total_blocked: int = 0
        self._total_allowed: int = 0

    def _make_key(self, name: str, args: dict) -> str:
        """Create a deterministic key from task name + args."""
        # Normalize args for hashing
        normalized = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(f"{name}:{normalized}".encode()).hexdigest()[:16]

    def is_running(self, name: str, args: dict | None = None) -> bool:
        """Check if a task with the same name+args is already running."""
        key = self._make_key(name, args or {})
        self._cleanup_stale()
        return key in self._running

    def acquire(
        self,
        name: str,
        args: dict | None = None,
        *,
        task: asyncio.Task | None = None,
    ) -> tuple[bool, str]:
        """Try to acquire a lock for a task.

        Returns (acquired, key). If acquired=False, the task is already running.
        """
        key = self._make_key(name, args or {})
        self._cleanup_stale()

        if key in self._running:
            self._total_blocked += 1
            existing = self._running[key]
            elapsed = time.monotonic() - existing.started_at
            log.info("idempotency: blocked duplicate %s (running for %.0fs)", name, elapsed)
            return False, key

        self._running[key] = RunningTask(
            key=key,
            name=name,
            started_at=time.monotonic(),
            task=task,
            args_hash=key,
        )
        self._total_allowed += 1
        return True, key

    def release(self, key: str) -> None:
        """Release a task lock."""
        if key in self._running:
            del self._running[key]

    def _cleanup_stale(self) -> None:
        """Remove entries for tasks that have been running too long."""
        now = time.monotonic()
        stale_keys = [
            key
            for key, task in self._running.items()
            if (task.task is not None and task.task.done())
            or (
                self.stale_timeout_sec is not None
                and task.task is None
                and now - task.started_at > self.stale_timeout_sec
            )
        ]
        for key in stale_keys:
            task = self._running[key]
            log.warning(
                "idempotency: cleaning stale task %s (ran for %.0fs)",
                task.name,
                now - task.started_at,
            )
            del self._running[key]

    def running_tasks(self) -> list[dict[str, Any]]:
        """Return info about currently running tasks."""
        self._cleanup_stale()
        now = time.monotonic()
        return [
            {
                "name": t.name,
                "key": t.key,
                "elapsed_sec": round(now - t.started_at, 1),
            }
            for t in self._running.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "running": len(self._running),
            "total_allowed": self._total_allowed,
            "total_blocked": self._total_blocked,
        }


# Global singleton
_guard: IdempotencyGuard | None = None


def get_idempotency_guard() -> IdempotencyGuard:
    global _guard
    if _guard is None:
        _guard = IdempotencyGuard()
    return _guard
