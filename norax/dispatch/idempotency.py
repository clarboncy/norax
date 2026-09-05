"""Bounded completed-call cache for scoped dispatcher idempotency.

LLMs occasionally re-issue identical tool_calls across retries. Any tool
call that provides a request ID gets its result cached. The dispatcher binds
that ID to the authenticated caller, tier, tool, and normalized arguments;
reuse for a different operation is an explicit conflict, not a cache hit.
"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class _Entry:
    result: dict
    ts: float
    scope: str | None


class IdempotencyConflict(ValueError):
    """Raised when a request ID is reused for a different scoped operation."""


@dataclass
class IdempotencyCache:
    ttl_seconds: float = 300.0
    max_entries: int = 4096
    _store: OrderedDict[str, _Entry] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        self.ttl_seconds = min(86_400.0, max(1.0, float(self.ttl_seconds)))
        self.max_entries = min(100_000, max(1, int(self.max_entries)))

    def _live_entry(self, request_id: str, now: float) -> _Entry | None:
        """Return one live entry without sweeping the bounded cache.

        The previous implementation scanned and allocated a list across every
        cached request on each get and put. Expiry only matters when a key is
        reused; capacity eviction already bounds retained stale entries, so a
        lazy key-local check preserves semantics and keeps the hot path O(1).
        """
        entry = self._store.get(request_id)
        if entry is not None and now - entry.ts > self.ttl_seconds:
            self._store.pop(request_id, None)
            return None
        return entry

    def get(self, request_id: str, *, scope: str | None = None) -> dict | None:
        now = time.monotonic()
        e = self._live_entry(request_id, now)
        if e is None:
            return None
        if e.scope != scope:
            raise IdempotencyConflict("request_id was already used for a different operation")
        self._store.move_to_end(request_id)
        return copy.deepcopy(e.result)

    def put(self, request_id: str, result: dict, *, scope: str | None = None) -> None:
        now = time.monotonic()
        existing = self._live_entry(request_id, now)
        if existing is not None and existing.scope != scope:
            raise IdempotencyConflict("request_id was already used for a different operation")
        if existing is not None:
            return
        while len(self._store) >= self.max_entries:
            self._store.popitem(last=False)
        self._store[request_id] = _Entry(
            result=copy.deepcopy(result),
            ts=now,
            scope=scope,
        )
