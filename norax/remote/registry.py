"""Persistent registry for remote nodes and job audit records."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text, read_bounded_text
from ..safety.secrets import redact
from .protocol import new_token, now_ms, token_hash

DEFAULT_CAPABILITIES = ["exec", "read", "list", "write"]
VALID_CAPABILITIES = frozenset(DEFAULT_CAPABILITIES)
_SENSITIVE_KEYS = frozenset(
    {"authorization", "credential", "credentials", "password", "private_key", "secret", "token"}
)
_MAX_REGISTRY_BYTES = 16 * 1024 * 1024
_MAX_NODES = 4_096
_MAX_ROOTS = 32
_MAX_ROOT_CHARS = 4_096
_MAX_META_KEYS = 100
_MAX_AUDIT_LINE_BYTES = 1 * 1024 * 1024
_MAX_AUDIT_FILE_BYTES = 256 * 1024 * 1024
_MAX_AUDIT_ITEMS = 100
_MAX_AUDIT_DEPTH = 8
_MAX_AUDIT_NODES = 10_000
_MAX_AUDIT_CONTENT_CHARS = 1_000_000
_NODE_ID_RX = re.compile(r"[a-z0-9][a-z0-9_-]{0,119}")
_TOKEN_HASH_RX = re.compile(r"[a-f0-9]{64}")


class RegistryCorruptError(RuntimeError):
    """Raised when persisted authorization state cannot be trusted."""


@dataclass
class NodeInfo:
    node_id: str
    name: str
    token_hash: str
    capabilities: list[str] = field(default_factory=lambda: list(DEFAULT_CAPABILITIES))
    roots: list[str] = field(default_factory=list)
    enabled: bool = True
    created_ms: int = field(default_factory=now_ms)
    last_seen_ms: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _millisecond_value(value: object, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 9_223_372_036_854_775_807
    ):
        raise RegistryCorruptError("remote registry contains an invalid timestamp")
    return value


def _validated_node(node_id: object, value: object) -> NodeInfo:
    if not isinstance(node_id, str) or _NODE_ID_RX.fullmatch(node_id) is None:
        raise RegistryCorruptError("remote registry contains an invalid node id")
    if not isinstance(value, dict):
        raise RegistryCorruptError("remote registry node must be an object")
    stored_id = value.get("node_id")
    if stored_id != node_id:
        raise RegistryCorruptError("remote registry node id does not match its key")
    name = value.get("name")
    token_digest = value.get("token_hash")
    capabilities = value.get("capabilities", list(DEFAULT_CAPABILITIES))
    roots = value.get("roots", [])
    enabled = value.get("enabled", True)
    meta = value.get("meta", {})
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise RegistryCorruptError("remote registry contains an invalid node name")
    if not isinstance(token_digest, str) or _TOKEN_HASH_RX.fullmatch(token_digest) is None:
        raise RegistryCorruptError("remote registry contains an invalid token hash")
    if (
        not isinstance(capabilities, list)
        or len(capabilities) > len(VALID_CAPABILITIES)
        or any(not isinstance(item, str) or item not in VALID_CAPABILITIES for item in capabilities)
        or len(set(capabilities)) != len(capabilities)
    ):
        raise RegistryCorruptError("remote registry contains invalid capabilities")
    if (
        not isinstance(roots, list)
        or len(roots) > _MAX_ROOTS
        or any(
            not isinstance(root, str) or not root.strip() or len(root) > _MAX_ROOT_CHARS
            for root in roots
        )
    ):
        raise RegistryCorruptError("remote registry contains invalid roots")
    if not isinstance(enabled, bool):
        raise RegistryCorruptError("remote registry enabled must be a boolean")
    if not isinstance(meta, dict) or len(meta) > _MAX_META_KEYS:
        raise RegistryCorruptError("remote registry metadata is malformed")
    audited_meta = _audit_value(meta)
    if not isinstance(audited_meta, dict):
        raise RegistryCorruptError("remote registry metadata is malformed")
    created_ms = _millisecond_value(value.get("created_ms", 0))
    last_seen_ms = _millisecond_value(value.get("last_seen_ms"), optional=True)
    assert created_ms is not None
    return NodeInfo(
        node_id=node_id,
        name=" ".join(name.split()),
        token_hash=token_digest,
        capabilities=list(capabilities),
        roots=[root.strip() for root in roots],
        enabled=enabled,
        created_ms=created_ms,
        last_seen_ms=last_seen_ms,
        meta=audited_meta,
    )


def _validated_registry(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegistryCorruptError("remote registry root must be an object")
    raw_nodes = value.get("nodes", {})
    if not isinstance(raw_nodes, dict):
        raise RegistryCorruptError("remote registry nodes must be an object")
    if len(raw_nodes) > _MAX_NODES:
        raise RegistryCorruptError("remote registry contains too many nodes")
    nodes = {
        node_id: asdict(_validated_node(node_id, raw_node))
        for node_id, raw_node in raw_nodes.items()
    }
    return {"nodes": nodes}


class RemoteRegistry:
    def __init__(self, root: Path | str | None = None) -> None:
        memory_root = Path(
            os.environ.get("NORAX_MEMORY_ROOT", Path.home() / ".local/share/norax/memory")
        )
        base = Path(root) if root is not None else memory_root / "remote"
        self.root = base
        self.nodes_path = self.root / "nodes.json"
        self.jobs_path = self.root / "jobs.jsonl"
        self.lock_path = self.root / ".registry.lock"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Serialize registry read/modify/write across threads and processes."""
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX)
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
                os.close(fd)

    def _load(self) -> dict[str, Any]:
        with self._transaction():
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, Any]:
        try:
            raw = read_bounded_text(
                self.nodes_path,
                max_bytes=_MAX_REGISTRY_BYTES,
            )
        except FileNotFoundError:
            return {"nodes": {}}
        except (OSError, UnicodeError, ValueError) as exc:
            raise RegistryCorruptError("remote registry is corrupt or unsafe") from exc
        try:
            data = json.loads(raw)
            return _validated_registry(data)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise RegistryCorruptError("remote registry is corrupt or unsafe") from exc

    def _save(self, data: dict[str, Any]) -> None:
        with self._transaction():
            self._save_unlocked(data)

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        validated = _validated_registry(data)
        serialized = (
            json.dumps(
                validated,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        if len(serialized.encode("utf-8")) > _MAX_REGISTRY_BYTES:
            raise ValueError(f"remote registry exceeds {_MAX_REGISTRY_BYTES} bytes")
        atomic_write_text(
            self.nodes_path,
            serialized,
            durable=True,
            mode=0o600,
        )

    def enroll(
        self, name: str, *, roots: list[str] | None = None, capabilities: list[str] | None = None
    ) -> tuple[NodeInfo, str]:
        if not isinstance(name, str):
            raise TypeError("node name must be a string")
        clean_name = " ".join(name.split())
        if not clean_name or len(clean_name) > 80:
            raise ValueError("node name must contain 1-80 characters")
        if capabilities is not None and not isinstance(capabilities, list):
            raise TypeError("remote capabilities must be a list")
        selected_capabilities = list(DEFAULT_CAPABILITIES) if capabilities is None else capabilities
        if any(not isinstance(capability, str) for capability in selected_capabilities):
            raise TypeError("remote capabilities must contain strings")
        selected_capabilities = list(dict.fromkeys(selected_capabilities))
        unknown = set(selected_capabilities) - VALID_CAPABILITIES
        if unknown:
            raise ValueError(f"unknown remote capabilities: {sorted(unknown)}")
        if roots is not None and not isinstance(roots, list):
            raise TypeError("remote roots must be a list")
        if any(not isinstance(root, str) for root in (roots or [])):
            raise TypeError("remote roots must contain strings")
        selected_roots = [root.strip() for root in (roots or []) if root.strip()]
        if len(selected_roots) > _MAX_ROOTS or any(
            len(root) > _MAX_ROOT_CHARS for root in selected_roots
        ):
            raise ValueError("remote roots exceed configured limits")
        slug = (
            "".join(c if c.isalnum() or c in "-_" else "-" for c in clean_name.lower()).strip("-")
            or "node"
        )[:100]
        with self._transaction():
            data = self._load_unlocked()
            if len(data["nodes"]) >= _MAX_NODES:
                raise ValueError("remote registry node limit reached")
            node_id = slug
            i = 2
            while node_id in data["nodes"]:
                suffix = f"-{i}"
                node_id = f"{slug[: 120 - len(suffix)]}{suffix}"
                i += 1
            token = new_token()
            info = NodeInfo(
                node_id=node_id,
                name=clean_name,
                token_hash=token_hash(token),
                roots=selected_roots,
                capabilities=selected_capabilities,
            )
            data["nodes"][node_id] = asdict(info)
            self._save_unlocked(data)
        return info, token

    def list_nodes(self) -> list[NodeInfo]:
        with self._transaction():
            return [NodeInfo(**v) for v in self._load_unlocked().get("nodes", {}).values()]

    def get(self, node_id: str) -> NodeInfo | None:
        if not isinstance(node_id, str) or _NODE_ID_RX.fullmatch(node_id) is None:
            return None
        with self._transaction():
            raw = self._load_unlocked().get("nodes", {}).get(node_id)
            return NodeInfo(**raw) if isinstance(raw, dict) else None

    def authenticate(self, node_id: str, token: str) -> NodeInfo | None:
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 4_096
            or any(character.isspace() for character in token)
        ):
            return None
        node = self.get(node_id)
        supplied_hash = token_hash(token)
        if node and node.enabled and secrets.compare_digest(node.token_hash, supplied_hash):
            return node
        return None

    def touch(self, node_id: str, *, meta: dict[str, Any] | None = None) -> None:
        with self._transaction():
            data = self._load_unlocked()
            raw = data.get("nodes", {}).get(node_id)
            if not isinstance(raw, dict):
                return
            raw["last_seen_ms"] = now_ms()
            if meta:
                raw.setdefault("meta", {}).update(_audit_value(meta))
            self._save_unlocked(data)

    def record_job(self, node_id: str, job: dict[str, Any], result: dict[str, Any]) -> None:
        rec = {
            "ts_ms": now_ms(),
            "node_id": str(node_id)[:120],
            "job": _audit_value(job),
            "result": _audit_value(result),
        }
        line = (
            json.dumps(
                rec,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        payload = line.encode("utf-8")
        if len(payload) > _MAX_AUDIT_LINE_BYTES:
            raise ValueError(f"remote audit record exceeds {_MAX_AUDIT_LINE_BYTES} bytes")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._transaction():
            self._append_job_line_unlocked(payload)

    def _append_job_line_unlocked(self, payload: bytes) -> None:
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(self.jobs_path, flags, 0o600)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("remote audit path is not a regular file")
            os.fchmod(descriptor, 0o600)
            if file_stat.st_size + len(payload) > _MAX_AUDIT_FILE_BYTES:
                os.close(descriptor)
                descriptor = -1
                archive = self.jobs_path.with_name(
                    f"{self.jobs_path.stem}-{time.time_ns()}{self.jobs_path.suffix}"
                )
                os.replace(self.jobs_path, archive)
                descriptor = os.open(self.jobs_path, flags, 0o600)
                os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while appending remote audit record")
                view = view[written:]
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _audit_value(
    value: Any,
    *,
    key: str = "",
    _depth: int = 0,
    _seen: set[int] | None = None,
    _budget: list[int] | None = None,
) -> Any:
    """Bound and redact remote audit fields without retaining file contents."""
    if _seen is None:
        _seen = set()
    if _budget is None:
        _budget = [_MAX_AUDIT_NODES]
    if _budget[0] <= 0:
        return "<TRUNCATED>"
    _budget[0] -= 1
    if _depth > _MAX_AUDIT_DEPTH:
        return "<MAX_DEPTH>"
    normalized_key = key.lower().replace("-", "_")
    if normalized_key in _SENSITIVE_KEYS or normalized_key.endswith(("_token", "_secret")):
        return "<REDACTED>"
    if isinstance(value, dict):
        identity = id(value)
        if identity in _seen:
            return "<CYCLE>"
        _seen.add(identity)
        try:
            return {
                str(item_key)[:120]: _audit_value(
                    item_value,
                    key=str(item_key)[:120],
                    _depth=_depth + 1,
                    _seen=_seen,
                    _budget=_budget,
                )
                for item_key, item_value in islice(value.items(), _MAX_AUDIT_ITEMS)
            }
        finally:
            _seen.discard(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in _seen:
            return "<CYCLE>"
        _seen.add(identity)
        try:
            return [
                _audit_value(
                    item,
                    _depth=_depth + 1,
                    _seen=_seen,
                    _budget=_budget,
                )
                for item in islice(value, _MAX_AUDIT_ITEMS)
            ]
        finally:
            _seen.discard(identity)
    if isinstance(value, str):
        if normalized_key == "content":
            bounded_content = value[:_MAX_AUDIT_CONTENT_CHARS]
            body = bounded_content.encode("utf-8", errors="replace")
            import hashlib

            return {
                "chars": len(value),
                "bytes": len(body),
                "hashed_chars": len(bounded_content),
                "sha256": hashlib.sha256(body).hexdigest(),
                "truncated": len(value) > len(bounded_content),
            }
        limit = 2000 if normalized_key in {"stdout", "stderr", "command"} else 1000
        bounded = value[:limit]
        if len(value) > limit:
            bounded += "…<truncated>"
        return redact(bounded)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    try:
        return redact(str(value)[:1_000])
    except Exception:  # noqa: BLE001
        return f"<{type(value).__name__}>"
