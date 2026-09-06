"""Hash-chained, schema-versioned, secret-scrubbed event log."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import ulid

_ulid_new_compat = getattr(ulid, "new", None) or (lambda: str(ulid.ULID()))

from ..safety.secrets import redact  # noqa: E402

log = logging.getLogger("norax.observability.log")
SCHEMA_VERSION = "v1"
_GENESIS_HASH = "0" * 64
_MAX_EVENT_LINE_BYTES = 4 * 1024 * 1024
_MAX_EVENT_STRING_CHARS = 128 * 1024
_MAX_EVENT_VALUE_CHARS = 1 * 1024 * 1024
_MAX_EVENT_ITEMS = 20_000
_MAX_CONTAINER_ITEMS = 2_048
_MAX_EVENT_DEPTH = 12
_MAX_REPLAY_BYTES = 64 * 1024 * 1024
_MAX_IDENTIFIER_CHARS = 256


@dataclass
class _EventBudget:
    items: int = _MAX_EVENT_ITEMS
    chars: int = _MAX_EVENT_VALUE_CHARS


def _bounded_identifier(value: object, *, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    return value.strip()[:_MAX_IDENTIFIER_CHARS] or fallback


def _bounded_event_value(
    value: Any,
    *,
    budget: _EventBudget,
    depth: int = 0,
    active: set[int] | None = None,
) -> Any:
    """Produce bounded, JSON-safe telemetry without recursively blocking a turn."""
    if budget.items <= 0 or budget.chars <= 0:
        return "<truncated:event_budget>"
    budget.items -= 1
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= 4_096 else "<integer_too_large>"
    if isinstance(value, float):
        return value if math.isfinite(value) else f"<{value}>"
    if isinstance(value, str):
        limit = min(_MAX_EVENT_STRING_CHARS, budget.chars)
        bounded = value[:limit]
        budget.chars -= len(bounded)
        if len(value) > limit:
            bounded += "…<truncated>"
        return redact(bounded)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "bytes": len(value)}
    if depth >= _MAX_EVENT_DEPTH:
        return "<truncated:event_depth>"

    active = active if active is not None else set()
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            return "<cycle>"
        active.add(identity)
        try:
            out: dict[str, Any] = {}
            for raw_key, item in islice(value.items(), _MAX_CONTAINER_ITEMS):
                try:
                    rendered_key = str(raw_key)
                except Exception:  # noqa: BLE001 - telemetry key hooks are untrusted
                    rendered_key = f"<{type(raw_key).__name__}>"
                key = str(_bounded_event_value(rendered_key, budget=budget))[:512]
                out[key] = _bounded_event_value(
                    item,
                    budget=budget,
                    depth=depth + 1,
                    active=active,
                )
                if budget.items <= 0 or budget.chars <= 0:
                    break
            if len(value) > len(out):
                out.setdefault("_truncated_items", len(value) - len(out))
            return out
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in active:
            return "<cycle>"
        active.add(identity)
        try:
            items = (
                value[:_MAX_CONTAINER_ITEMS]
                if isinstance(value, (list, tuple))
                else list(islice(value, _MAX_CONTAINER_ITEMS))
            )
            sequence_out = [
                _bounded_event_value(
                    item,
                    budget=budget,
                    depth=depth + 1,
                    active=active,
                )
                for item in items
                if budget.items > 0 and budget.chars > 0
            ]
            if len(value) > len(sequence_out):
                sequence_out.append({"_truncated_items": len(value) - len(sequence_out)})
            return sequence_out
        finally:
            active.remove(identity)
    try:
        rendered = str(value)
    except Exception:  # noqa: BLE001 - telemetry must tolerate hostile repr/str hooks
        rendered = f"<{type(value).__name__}>"
    return _bounded_event_value(rendered, budget=budget, depth=depth + 1, active=active)


def _open_regular(path: Path, flags: int, *, mode: int = 0o600) -> int:
    descriptor = os.open(
        path,
        flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"event-log path is not a regular file: {path}")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@dataclass
class _TraceScope:
    trace_id: str
    root_span_id: str | None = None


_TRACE_SCOPE: ContextVar[_TraceScope | None] = ContextVar(
    "norax_event_trace_scope",
    default=None,
)


class EventLog:
    def __init__(self, path: Path, *, rotate_bytes: int = 25_000_000) -> None:
        if isinstance(rotate_bytes, bool) or not isinstance(rotate_bytes, int):
            raise TypeError("rotate_bytes must be an integer")
        if rotate_bytes < 0:
            raise ValueError("rotate_bytes must be non-negative")
        self.path = Path(path)
        self.rotate_bytes = rotate_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._writable = True
        self._last_write_error = ""
        # A writer may be rotating or halfway through a multi-write append.
        # Read the initial tail under the same lock as subsequent appends.
        with self._process_lock():
            self._prev_hash = self._load_tail_hash()

    @contextmanager
    def _process_lock(self):
        import fcntl

        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        descriptor = _open_regular(lock_path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(descriptor, "a+") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def _empty_tail_hash(self) -> str:
        """Allow genesis only for a new log, never after lost current data."""
        anchor = self.path.with_suffix(self.path.suffix + ".anchor")
        if (
            getattr(self, "_prev_hash", _GENESIS_HASH) != _GENESIS_HASH
            or anchor.exists()
            or anchor.is_symlink()
            or next(self.path.parent.glob(f"{self.path.stem}-*{self.path.suffix}"), None)
            is not None
        ):
            raise ValueError("event-log current file is missing or empty after existing history")
        return _GENESIS_HASH

    @contextmanager
    def trace_scope(self, trace_id: str | None = None):
        """Correlate every event emitted by one logical operation.

        Context variables propagate through awaited calls and child asyncio
        tasks, so agent-loop, orchestrator, and tool events inherit the turn
        trace without every internal API needing a trace parameter.  The first
        event becomes the root span; later events point to that root.
        """
        scope = _TraceScope(trace_id=trace_id or str(_ulid_new_compat()))
        token = _TRACE_SCOPE.set(scope)
        try:
            yield scope.trace_id
        finally:
            _TRACE_SCOPE.reset(token)

    def _load_tail_hash(self, *, ignore_cache: bool = False) -> str:
        # Prefer the in-memory tail when we have already appended this session.
        # Caller may bypass the cache (e.g. under the cross-process write lock)
        # to re-read the authoritative on-disk tail.
        if not ignore_cache and getattr(self, "_prev_hash", None):
            return self._prev_hash
        try:
            descriptor = _open_regular(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return self._empty_tail_hash()
        with os.fdopen(descriptor, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return self._empty_tail_hash()
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                raise ValueError("event-log tail is incomplete: unterminated record")
            # Read the entire file if small enough, otherwise grow a window
            # from the tail until we capture a complete non-empty last line.
            # A line is complete when it is followed by \n or when we've
            # read from the start of the file.
            window = min(size, 4_096)
            last_line = b""
            while True:
                fh.seek(max(0, size - window))
                data = fh.read(window)
                lines = [line for line in data.splitlines() if line.strip()]
                if lines:
                    # If we read from the start, the first line is complete.
                    # Otherwise the first element may be a fragment — skip it
                    # unless it's the only element.
                    if size <= window or len(lines) >= 2 or data.startswith((b"\n", b"\r")):
                        last_line = lines[-1]
                        break
                if window >= size or window >= _MAX_EVENT_LINE_BYTES + 1:
                    break
                window = min(size, _MAX_EVENT_LINE_BYTES + 1, window * 2)
            if not last_line:
                if size > _MAX_EVENT_LINE_BYTES:
                    raise ValueError("event-log tail line exceeds the size limit")
                raise ValueError("event-log tail contains no complete record")
            try:
                record = json.loads(last_line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("event-log tail is not valid JSON") from exc
            tail_hash = record.get("hash") if isinstance(record, dict) else None
            if (
                not isinstance(tail_hash, str)
                or len(tail_hash) != 64
                or any(char not in "0123456789abcdef" for char in tail_hash)
            ):
                raise ValueError("event-log tail has an invalid hash")
            return tail_hash

    async def append(
        self,
        kind: str,
        payload: dict,
        *,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> str:
        ts = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        scope = _TRACE_SCOPE.get()
        resolved_trace_id = (
            trace_id or (scope.trace_id if scope else None) or str(_ulid_new_compat())
        )
        resolved_span_id = span_id or str(_ulid_new_compat())
        resolved_parent_span_id = parent_span_id
        if resolved_parent_span_id is None and scope is not None:
            resolved_parent_span_id = scope.root_span_id
            if scope.root_span_id is None:
                scope.root_span_id = resolved_span_id
        # The process-local asyncio lock is not enough: development tools and
        # overlapping runtime processes can each instantiate EventLog.  Build
        # and append the record under an OS file lock so every writer advances
        # the same tail atomically.
        async with self._lock:
            await asyncio.to_thread(
                self._append_locked,
                ts,
                kind,
                payload,
                resolved_trace_id,
                resolved_span_id,
                resolved_parent_span_id,
                attrs or {},
            )
        return resolved_trace_id

    def _append_locked(
        self,
        ts: str,
        kind: str,
        payload: dict,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        attrs: dict[str, Any],
    ) -> None:
        """Append one record with an advisory process-wide lock."""
        with self._process_lock():
            try:
                # Under the fcntl lock, always re-read the on-disk tail: another
                # process may have appended since our cached _prev_hash was set.
                prev_hash = self._load_tail_hash(ignore_cache=True)
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "ts": ts,
                    "kind": _bounded_identifier(kind, fallback="unknown"),
                    "trace_id": _bounded_identifier(trace_id, fallback="unknown"),
                    "span_id": _bounded_identifier(span_id, fallback="unknown"),
                    "parent_span_id": (
                        _bounded_identifier(parent_span_id) if parent_span_id is not None else None
                    ),
                    "prev_hash": prev_hash,
                    "attrs": _bounded_event_value(attrs, budget=_EventBudget()),
                    "payload": _bounded_event_value(payload, budget=_EventBudget()),
                }
                body = json.dumps(
                    record,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                body_bytes = body.encode()
                if len(body_bytes) > _MAX_EVENT_LINE_BYTES - 256:
                    record["payload"] = {
                        "_truncated": True,
                        "reason": "event_size_limit",
                        "normalized_record_bytes": len(body_bytes),
                        "normalized_record_sha256": hashlib.sha256(body_bytes).hexdigest(),
                    }
                    record["attrs"] = {"payload_truncated": True}
                    body = json.dumps(record, separators=(",", ":"), sort_keys=True)
                record_hash = hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()
                record["hash"] = record_hash
                line = json.dumps(record, separators=(",", ":")) + "\n"
                if len(line.encode()) > _MAX_EVENT_LINE_BYTES:
                    raise ValueError("bounded event record still exceeds the line limit")
                self._write_line(line)
                self._prev_hash = record_hash
                self._writable = True
                self._last_write_error = ""
            except Exception as exc:
                self._writable = False
                self._last_write_error = f"{type(exc).__name__}: {exc}"
                raise

    def _write_line(self, line: str) -> None:
        payload = line.encode()
        self._rotate_if_needed(len(payload))
        descriptor = _open_regular(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while appending event record")
                view = view[written:]
        finally:
            os.close(descriptor)

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if self.rotate_bytes <= 0:
            return
        try:
            descriptor = _open_regular(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return
        try:
            current_size = os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)
        if current_size == 0 or current_size + incoming_bytes <= self.rotate_bytes:
            return
        anchor = self._load_tail_hash(ignore_cache=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        rotated = self.path.with_name(f"{self.path.stem}-{ts}{self.path.suffix}")
        if rotated.exists() or rotated.is_symlink():
            rotated = self.path.with_name(
                f"{self.path.stem}-{ts}-{str(_ulid_new_compat())}{self.path.suffix}"
            )
        self.path.rename(rotated)
        anchor_path = self.path.with_suffix(self.path.suffix + ".anchor")
        from ..atomic import atomic_write_text

        atomic_write_text(
            anchor_path,
            json.dumps({"rotated": rotated.name, "tail_hash": anchor}, separators=(",", ":"))
            + "\n",
            mode=0o600,
        )

    async def replay(self):
        try:
            records = await asyncio.to_thread(self._read_replay_records)
        except FileNotFoundError:
            return
        for record in records:
            yield record

    def _read_replay_records(self) -> list[dict[str, Any]]:
        from ..atomic import read_bounded_text

        raw = read_bounded_text(self.path, max_bytes=_MAX_REPLAY_BYTES)
        records: list[dict[str, Any]] = []
        for raw_line in raw.splitlines():
            if not raw_line.strip():
                continue
            if len(raw_line.encode()) > _MAX_EVENT_LINE_BYTES:
                raise ValueError("event-log replay line exceeds the size limit")
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError("event-log replay record is not an object")
            records.append(record)
        return records
