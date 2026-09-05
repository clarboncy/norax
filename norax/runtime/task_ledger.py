"""Durable task-outcome ledger for failed-task and frustration resolution.

The ledger is a bounded, atomically replaced JSONL snapshot. Mutations are
serialized across threads and worker processes so prompt focus never reflects
a half-written or stale-success update.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text, read_bounded_text

log = logging.getLogger("norax.task_ledger")

VALID_STATES = {"open", "blocked", "failed", "repairing", "verified", "user_confirmed", "abandoned"}
_ACTIVE_STATES = frozenset({"open", "blocked", "failed", "repairing"})
_RESOLVED_STATES = frozenset({"verified", "user_confirmed", "abandoned"})
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_MAX_LEDGER_BYTES = 64 * 1024 * 1024
_MAX_LEDGER_TASKS = 100_000
_MAX_LINE_BYTES = 256 * 1024
_MAX_METADATA_NODES = 1_024


class TaskLedgerPersistenceError(RuntimeError):
    """Task ledger could not be restored or committed safely."""


def _clean_id(value: object, *, field_name: str, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned and not required:
        return ""
    if not _ID_RE.fullmatch(cleaned):
        raise ValueError(f"invalid {field_name}")
    return cleaned


def _clean_prompt_text(value: object, *, field_name: str, max_chars: int = 2_000) -> str:
    if not isinstance(value, str) or len(value) > max_chars or "\x00" in value:
        raise ValueError(f"invalid {field_name}")
    if any(ord(character) < 0x20 and character not in "\n\r\t" for character in value):
        raise ValueError(f"invalid {field_name}")
    return " ".join(value.split())


def _normalize_metadata(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    remaining = [_MAX_METADATA_NODES]
    visiting: set[int] = set()

    def normalize(item: object, *, depth: int) -> Any:
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 6:
            raise ValueError("metadata exceeds its structural limit")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int):
            if abs(item) > 10**18:
                raise ValueError("metadata integer is outside the supported range")
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("metadata numbers must be finite")
            return item
        if isinstance(item, str):
            if len(item) > 4_000 or "\x00" in item:
                raise ValueError("metadata text exceeds its limit")
            return item
        if isinstance(item, dict):
            identity = id(item)
            if identity in visiting or len(item) > 64:
                raise ValueError("metadata contains a cycle or too many keys")
            visiting.add(identity)
            try:
                output: dict[str, Any] = {}
                for key, child in item.items():
                    if (
                        not isinstance(key, str)
                        or not 1 <= len(key) <= 128
                        or any(ord(character) < 0x20 for character in key)
                    ):
                        raise ValueError("metadata contains an invalid key")
                    output[key] = normalize(child, depth=depth + 1)
                return output
            finally:
                visiting.remove(identity)
        if isinstance(item, list):
            identity = id(item)
            if identity in visiting or len(item) > 128:
                raise ValueError("metadata contains a cycle or oversized list")
            visiting.add(identity)
            try:
                return [normalize(child, depth=depth + 1) for child in item]
            finally:
                visiting.remove(identity)
        raise ValueError("metadata contains a non-JSON value")

    normalized = normalize(value, depth=0)
    assert isinstance(normalized, dict)
    return normalized


@dataclass
class TaskOutcome:
    """A single task outcome record in the ledger."""

    task_id: str
    turn_id: str = ""
    event_id: str = ""
    description: str = ""
    state: str = "open"
    failure_cause: str = ""
    repair_attempts: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0
    owner_confirmed: bool = False
    valence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskOutcome:
        if not isinstance(data, dict):
            raise ValueError("task outcome must be an object")
        task_id = _clean_id(data.get("task_id"), field_name="task_id", required=True)
        turn_id = _clean_id(data.get("turn_id", ""), field_name="turn_id")
        event_id = _clean_id(data.get("event_id", ""), field_name="event_id")
        description = _clean_prompt_text(data.get("description", ""), field_name="description")
        failure_cause = _clean_prompt_text(
            data.get("failure_cause", ""),
            field_name="failure_cause",
        )
        state_value = data.get("state", "open")
        if not isinstance(state_value, str) or state_value not in VALID_STATES:
            raise ValueError("invalid task state")
        repair_attempts = data.get("repair_attempts", 0)
        if (
            isinstance(repair_attempts, bool)
            or not isinstance(repair_attempts, int)
            or not 0 <= repair_attempts <= 1_000_000
        ):
            raise ValueError("invalid repair attempt count")
        now = time.time()
        timestamps: list[float] = []
        for field_name, default in (
            ("created_at", now),
            ("updated_at", now),
            ("resolved_at", 0.0),
        ):
            raw_value = data.get(field_name, default)
            if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
                raise ValueError(f"invalid {field_name}")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0 or value > now + 300:
                raise ValueError(f"invalid {field_name}")
            timestamps.append(value)
        created_at, updated_at, resolved_at = timestamps
        if created_at <= 0 or updated_at < created_at:
            raise ValueError("invalid task timestamp ordering")
        if state_value in _RESOLVED_STATES and resolved_at == 0:
            resolved_at = updated_at
        if resolved_at and resolved_at < created_at:
            raise ValueError("invalid task resolution timestamp")
        owner_confirmed = data.get("owner_confirmed", False)
        if not isinstance(owner_confirmed, bool):
            raise ValueError("owner_confirmed must be a boolean")
        if state_value == "user_confirmed":
            owner_confirmed = True
        raw_valence = data.get("valence", 0.0)
        if isinstance(raw_valence, bool) or not isinstance(raw_valence, int | float):
            raise ValueError("invalid task valence")
        valence = float(raw_valence)
        if not math.isfinite(valence) or not -1.0 <= valence <= 1.0:
            raise ValueError("task valence must be between -1 and 1")
        return cls(
            task_id=task_id,
            turn_id=turn_id,
            event_id=event_id,
            description=description,
            state=state_value,
            failure_cause=failure_cause,
            repair_attempts=repair_attempts,
            created_at=created_at,
            updated_at=updated_at,
            resolved_at=resolved_at,
            owner_confirmed=owner_confirmed,
            valence=valence,
            metadata=_normalize_metadata(data.get("metadata", {})),
        )


class TaskOutcomeLedger:
    """Cross-worker-safe durable ledger stored under ``memory_root``."""

    def __init__(self, memory_root: Path) -> None:
        self._root = Path(memory_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "task_outcomes.jsonl"
        self._lock_path = self._root / ".task_outcomes.lock"
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskOutcome] = {}
        self._load_error = ""
        self._loaded_signature: tuple[int, int, int, int, int] | None = None
        self._had_persisted_state = False
        with self._transaction():
            self._load(force=True)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            flags = (
                os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(self._lock_path, flags, 0o600)
            try:
                lock_stat = os.fstat(descriptor)
                if not stat.S_ISREG(lock_stat.st_mode):
                    raise TaskLedgerPersistenceError("task ledger lock is not a regular file")
                os.fchmod(descriptor, 0o600)
                try:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
                os.close(descriptor)

    def _file_signature(self) -> tuple[int, int, int, int, int] | None:
        try:
            current = self._path.lstat()
        except FileNotFoundError:
            return None
        return (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )

    def _load(self, *, force: bool = False) -> None:
        signature = self._file_signature()
        if not force and signature == self._loaded_signature and not self._load_error:
            return
        if signature is None:
            if self._had_persisted_state:
                self._tasks = {}
                self._load_error = "task ledger disappeared"
                return
            self._tasks = {}
            self._loaded_signature = None
            self._load_error = ""
            return
        try:
            raw = read_bounded_text(self._path, max_bytes=_MAX_LEDGER_BYTES)
            loaded: dict[str, TaskOutcome] = {}
            records = 0
            for raw_line in raw.splitlines():
                if not raw_line.strip():
                    continue
                if len(raw_line.encode("utf-8")) > _MAX_LINE_BYTES:
                    raise ValueError("task ledger contains an oversized record")
                records += 1
                if records > _MAX_LEDGER_TASKS:
                    raise ValueError("task ledger exceeds its task limit")
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ValueError("task ledger record must be an object")
                task = TaskOutcome.from_dict(payload)
                loaded[task.task_id] = task
            self._tasks = loaded
            self._loaded_signature = signature
            self._had_persisted_state = True
            self._load_error = ""
        except Exception as exc:  # noqa: BLE001
            self._tasks = {}
            self._loaded_signature = signature
            self._load_error = str(exc)
            log.warning("task_ledger.load_failed error=%r", exc)

    def _require_healthy(self) -> None:
        if self._load_error:
            raise TaskLedgerPersistenceError(
                f"task ledger could not be restored safely: {self._load_error}"
            )

    @staticmethod
    def _clone(task: TaskOutcome) -> TaskOutcome:
        return TaskOutcome.from_dict(copy.deepcopy(task.to_dict()))

    def _persist_tasks(self, tasks: dict[str, TaskOutcome]) -> None:
        if len(tasks) > _MAX_LEDGER_TASKS:
            raise TaskLedgerPersistenceError("task ledger exceeds its task limit")
        lines: list[str] = []
        for task in tasks.values():
            canonical = TaskOutcome.from_dict(task.to_dict())
            encoded = json.dumps(
                canonical.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if len(encoded.encode("utf-8")) > _MAX_LINE_BYTES:
                raise TaskLedgerPersistenceError("task ledger record exceeds its byte limit")
            lines.append(encoded)
        body = "\n".join(lines) + ("\n" if lines else "")
        if len(body.encode("utf-8")) > _MAX_LEDGER_BYTES:
            raise TaskLedgerPersistenceError("task ledger exceeds its byte limit")
        atomic_write_text(self._path, body, durable=True, mode=0o600)
        self._tasks = tasks
        self._loaded_signature = self._file_signature()
        self._had_persisted_state = True

    def _rewrite(self) -> None:
        with self._transaction():
            self._load()
            self._require_healthy()
            self._persist_tasks(dict(self._tasks))

    def record(
        self,
        *,
        task_id: str,
        turn_id: str = "",
        event_id: str = "",
        description: str = "",
        state: str = "open",
        failure_cause: str = "",
        valence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskOutcome:
        """Create or durably update a task outcome."""
        clean_task_id = _clean_id(task_id, field_name="task_id", required=True)
        if state not in VALID_STATES:
            raise ValueError(f"invalid state {state!r}; must be one of {VALID_STATES}")
        clean_turn_id = _clean_id(turn_id, field_name="turn_id")
        clean_event_id = _clean_id(event_id, field_name="event_id")
        clean_description = _clean_prompt_text(description, field_name="description")
        clean_failure = _clean_prompt_text(failure_cause, field_name="failure_cause")
        clean_metadata = _normalize_metadata(metadata)
        if valence is not None and (
            isinstance(valence, bool)
            or not isinstance(valence, int | float)
            or not math.isfinite(float(valence))
            or not -1 <= float(valence) <= 1
        ):
            raise ValueError("valence must be a finite number between -1 and 1")

        with self._transaction():
            self._load()
            self._require_healthy()
            now = time.time()
            existing = self._tasks.get(clean_task_id)
            if existing is None:
                candidate = TaskOutcome.from_dict(
                    {
                        "task_id": clean_task_id,
                        "turn_id": clean_turn_id,
                        "event_id": clean_event_id,
                        "description": clean_description,
                        "state": state,
                        "failure_cause": clean_failure,
                        "valence": 0.0 if valence is None else float(valence),
                        "metadata": clean_metadata,
                        "repair_attempts": 1 if state == "repairing" else 0,
                        "created_at": now,
                        "updated_at": now,
                        "resolved_at": now if state in _RESOLVED_STATES else 0.0,
                        "owner_confirmed": state == "user_confirmed",
                    }
                )
            else:
                payload = existing.to_dict()
                payload.update(
                    {
                        "state": state,
                        "updated_at": now,
                        "resolved_at": now if state in _RESOLVED_STATES else 0.0,
                        "owner_confirmed": existing.owner_confirmed or state == "user_confirmed",
                        "repair_attempts": existing.repair_attempts
                        + (1 if state == "repairing" else 0),
                    }
                )
                if clean_turn_id:
                    payload["turn_id"] = clean_turn_id
                if clean_event_id:
                    payload["event_id"] = clean_event_id
                if clean_description:
                    payload["description"] = clean_description
                if clean_failure:
                    payload["failure_cause"] = clean_failure
                if valence is not None:
                    payload["valence"] = float(valence)
                merged_metadata = copy.deepcopy(existing.metadata)
                merged_metadata.update(clean_metadata)
                payload["metadata"] = merged_metadata
                candidate = TaskOutcome.from_dict(payload)
            next_tasks = dict(self._tasks)
            next_tasks[clean_task_id] = candidate
            self._persist_tasks(next_tasks)
            return self._clone(candidate)

    def transition(
        self,
        task_id: str,
        new_state: str,
        *,
        failure_cause: str = "",
    ) -> TaskOutcome | None:
        """Durably transition an existing task; return ``None`` when absent."""
        clean_task_id = _clean_id(task_id, field_name="task_id", required=True)
        if new_state not in VALID_STATES:
            raise ValueError(f"invalid state {new_state!r}")
        clean_failure = _clean_prompt_text(failure_cause, field_name="failure_cause")
        with self._transaction():
            self._load()
            self._require_healthy()
            existing = self._tasks.get(clean_task_id)
            if existing is None:
                return None
            now = time.time()
            payload = existing.to_dict()
            payload.update(
                {
                    "state": new_state,
                    "updated_at": now,
                    "resolved_at": now if new_state in _RESOLVED_STATES else 0.0,
                    "owner_confirmed": existing.owner_confirmed or new_state == "user_confirmed",
                    "repair_attempts": existing.repair_attempts
                    + (1 if new_state == "repairing" else 0),
                }
            )
            if clean_failure:
                payload["failure_cause"] = clean_failure
            candidate = TaskOutcome.from_dict(payload)
            next_tasks = dict(self._tasks)
            next_tasks[clean_task_id] = candidate
            self._persist_tasks(next_tasks)
            return self._clone(candidate)

    def unresolved(self, limit: int = 5) -> list[TaskOutcome]:
        """Return the highest-priority unresolved task snapshots."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        with self._transaction():
            self._load()
            self._require_healthy()
            candidates = [task for task in self._tasks.values() if task.state in _ACTIVE_STATES]
            candidates.sort(key=lambda task: (task.valence, task.updated_at))
            return [self._clone(task) for task in candidates[:limit]]

    def all(self) -> list[TaskOutcome]:
        """Return snapshots of all tasks."""
        with self._transaction():
            self._load()
            self._require_healthy()
            return [self._clone(task) for task in self._tasks.values()]

    def get(self, task_id: str) -> TaskOutcome | None:
        clean_task_id = _clean_id(task_id, field_name="task_id", required=True)
        with self._transaction():
            self._load()
            self._require_healthy()
            task = self._tasks.get(clean_task_id)
            return self._clone(task) if task is not None else None

    def expire_stale(self, max_age_sec: float = 86400 * 7) -> int:
        """Durably abandon unresolved tasks older than ``max_age_sec``."""
        if (
            isinstance(max_age_sec, bool)
            or not isinstance(max_age_sec, int | float)
            or not math.isfinite(float(max_age_sec))
            or max_age_sec <= 0
        ):
            raise ValueError("max_age_sec must be a positive finite number")
        with self._transaction():
            self._load()
            self._require_healthy()
            now = time.time()
            next_tasks = dict(self._tasks)
            expired = 0
            for task_id, task in self._tasks.items():
                if task.state not in _ACTIVE_STATES or now - task.created_at <= max_age_sec:
                    continue
                payload = task.to_dict()
                metadata = copy.deepcopy(task.metadata)
                metadata["expiry_reason"] = "stale_timeout"
                payload.update(
                    {
                        "state": "abandoned",
                        "updated_at": now,
                        "resolved_at": now,
                        "metadata": metadata,
                    }
                )
                next_tasks[task_id] = TaskOutcome.from_dict(payload)
                expired += 1
            if expired:
                self._persist_tasks(next_tasks)
                log.info("task_ledger.expired %d stale tasks", expired)
            return expired

    def focus_context(self, limit: int = 3) -> str:
        """Generate a bounded, single-line-per-task prompt focus block."""
        items = self.unresolved(limit=limit)
        if not items:
            return ""
        lines = ["UNRESOLVED TASKS (owner attention required):"]
        for task in items:
            lines.append(
                f"  [{task.state}] {task.description or task.task_id}: "
                f"{task.failure_cause or 'cause unknown'}"
            )
        return "\n".join(lines)
