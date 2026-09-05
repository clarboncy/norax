"""Durable turn checkpoints for restart-aware continuation.

Serializes bounded messages, tool results, and TaskState at completed turn
boundaries. On restart, the runtime can offer that state to a related follow-up;
this is not an instruction-pointer snapshot or an automatic replay mechanism.

Design:
  - Checkpoint dir: memory/checkpoints/
  - One JSON file per turn: checkpoint_{turn_id}.json
  - Contains: messages, trace, task_state, rounds, timestamp
  - Auto-cleanup: keep last 10 checkpoints per channel
  - Resume API: load latest checkpoint, inject into agent loop
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import os
import re
import stat
import time
from itertools import islice
from pathlib import Path

from ..atomic import atomic_write_text

log = logging.getLogger("norax.runtime.checkpoint")

_MAX_CHECKPOINTS = 10  # keep last N per channel
_MAX_MESSAGES = 200
_MAX_TRACE_ENTRIES = 200
_MAX_CHECKPOINT_BYTES = 2_097_152
_MAX_CHECKPOINT_FILES_SCANNED = 1_024
_MAX_CHANNEL_DIRS_SCANNED = 1_024
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CHECKPOINT_FILENAME = re.compile(r"^checkpoint_[A-Za-z0-9_-]{1,128}\.json$")


def _safe_component(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > 512:
        raise ValueError(f"{label} must contain 1-512 characters")
    if _SAFE_COMPONENT.fullmatch(value):
        return value
    slug = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:64]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{slug or 'id'}-{digest}"


def _checkpoint_root(memory_root: Path | None = None) -> Path:
    if memory_root is not None:
        return Path(memory_root).expanduser().resolve() / "checkpoints"
    configured = os.environ.get("NORAX_CHECKPOINT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    configured_memory = os.environ.get("NORAX_MEMORY_ROOT")
    if configured_memory:
        return Path(configured_memory).expanduser().resolve() / "checkpoints"
    return (Path.home() / "norax" / "memory" / "checkpoints").resolve()


def _checkpoint_dir(
    channel: str,
    *,
    memory_root: Path | None = None,
    create: bool = True,
) -> Path:
    root = _checkpoint_root(memory_root)
    d = root / _safe_component(channel, label="channel")
    if d.is_symlink():
        raise OSError("checkpoint directories must not be symbolic links")
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            root.chmod(0o700)
            d.chmod(0o700)
        except OSError:
            pass
    return d


def _checkpoint_path(
    channel: str,
    turn_id: str,
    *,
    memory_root: Path | None = None,
    create: bool,
) -> Path:
    directory = _checkpoint_dir(channel, memory_root=memory_root, create=create)
    safe_turn = _safe_component(turn_id, label="turn_id")
    path = directory / f"checkpoint_{safe_turn}.json"
    if path.is_symlink():
        raise OSError("checkpoint files must not be symbolic links")
    return path


def _checkpoint_files(directory: Path, *, limit: int) -> list[Path]:
    """Return newest regular checkpoint files with bounded scan memory/work."""
    if directory.is_symlink() or not directory.is_dir():
        return []
    candidates: list[tuple[int, str, Path]] = []
    try:
        with os.scandir(directory) as entries:
            for scanned, entry in enumerate(entries, start=1):
                if scanned > _MAX_CHECKPOINT_FILES_SCANNED:
                    log.warning(
                        "checkpoint.scan_limit directory=%s limit=%d",
                        directory,
                        _MAX_CHECKPOINT_FILES_SCANNED,
                    )
                    break
                if not _CHECKPOINT_FILENAME.fullmatch(entry.name):
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    modified = entry.stat(follow_symlinks=False).st_mtime_ns
                except OSError:
                    continue
                candidates.append((modified, entry.name, Path(entry.path)))
    except OSError:
        return []
    return [
        path
        for _modified, _name, path in heapq.nlargest(
            max(0, limit), candidates, key=lambda item: (item[0], item[1])
        )
    ]


def _read_checkpoint(path: Path) -> dict | None:
    """Read one regular file without following a symlink or exceeding 2 MiB."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError("checkpoint is not a regular file")
            if file_stat.st_size > _MAX_CHECKPOINT_BYTES:
                raise ValueError("checkpoint exceeds the 2 MiB serialized limit")
            raw = handle.read(_MAX_CHECKPOINT_BYTES + 1)
        if len(raw) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds the 2 MiB serialized limit")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            return None
        bounded = _bounded_json_value(value, budget=[900])
        return bounded if isinstance(bounded, dict) else None
    except (json.JSONDecodeError, OSError, RecursionError, UnicodeError, ValueError) as error:
        log.warning("checkpoint.load_failed path=%s err=%r", path, error)
        return None


def _latest_in(directory: Path) -> dict | None:
    # A torn/corrupt newest file must not hide the previous durable checkpoint.
    for checkpoint_path in _checkpoint_files(directory, limit=_MAX_CHECKPOINTS):
        checkpoint = _read_checkpoint(checkpoint_path)
        if checkpoint is not None:
            return checkpoint
    return None


def _is_pending(checkpoint: dict, max_age_sec: float) -> bool:
    try:
        age = time.time() - float(checkpoint.get("timestamp", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        0 <= age <= max_age_sec
        and checkpoint.get("resumed") is not True
        and checkpoint.get("status", "in_progress") != "complete"
    )


def _bounded_json_value(
    value: object,
    *,
    budget: list[int],
    depth: int = 0,
    string_limit: int = 2_000,
) -> object:
    """Return a JSON-safe tree with a deterministic node/string budget."""
    if budget[0] <= 0:
        return "[checkpoint truncated]"
    budget[0] -= 1
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:string_limit]
    if depth >= 8:
        return "[checkpoint depth truncated]"
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for raw_key, item in islice(value.items(), 200):
            if budget[0] <= 0:
                out["_checkpoint_truncated"] = True
                break
            key = str(raw_key)[:128]
            out[key] = _bounded_json_value(
                item,
                budget=budget,
                depth=depth + 1,
                string_limit=string_limit,
            )
        return out
    if isinstance(value, (list, tuple)):
        out_list: list[object] = []
        for item in value[-200:]:
            if budget[0] <= 0:
                out_list.append({"_checkpoint_truncated": True})
                break
            out_list.append(
                _bounded_json_value(
                    item,
                    budget=budget,
                    depth=depth + 1,
                    string_limit=string_limit,
                )
            )
        return out_list
    return str(value)[:string_limit]


def save_checkpoint(
    *,
    channel: str,
    turn_id: str,
    messages: list[dict],
    trace: list[dict],
    task_state: dict | None,
    rounds: int,
    model: str,
    status: str = "in_progress",
    response: dict | None = None,
    delivery: dict | None = None,
    memory_root: Path | None = None,
) -> Path:
    """Save a checkpoint to disk. Returns the checkpoint path."""
    if not isinstance(messages, list) or not isinstance(trace, list):
        raise TypeError("messages and trace must be lists")
    if task_state is not None and not isinstance(task_state, dict):
        raise TypeError("task_state must be an object or None")
    if response is not None and not isinstance(response, dict):
        raise TypeError("response must be an object or None")
    if delivery is not None and not isinstance(delivery, dict):
        raise TypeError("delivery must be an object or None")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 0 <= rounds <= 1_000_000:
        raise ValueError("rounds must be an integer between 0 and 1000000")
    if not isinstance(model, str) or not model.strip() or len(model) > 512:
        raise ValueError("model must contain 1-512 characters")
    if not isinstance(status, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", status):
        raise ValueError("status contains unsupported characters")
    cp_dir = _checkpoint_dir(channel, memory_root=memory_root)
    cp_path = cp_dir / f"checkpoint_{_safe_component(turn_id, label='turn_id')}.json"
    if cp_path.is_symlink():
        raise OSError("checkpoint files must not be symbolic links")

    payload = _bounded_json_value(
        {
            "channel": channel,
            "turn_id": turn_id,
            "timestamp": time.time(),
            "model": model.strip(),
            "rounds": rounds,
            "status": status,
            "response": response,
            "delivery": delivery,
            # Preserve compact task intent before spending the remaining bound on
            # potentially bulky model/tool history.
            "task_state": task_state,
            "messages": _sanitize_messages(messages),
            "trace": _sanitize_trace(trace),
        },
        budget=[900],
    )
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint serialization produced an invalid payload")
    rendered = json.dumps(payload, allow_nan=False, ensure_ascii=False)
    if len(rendered.encode("utf-8")) > _MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds the 2 MiB serialized limit")

    atomic_write_text(
        cp_path,
        rendered,
        durable=True,
        mode=0o600,
    )

    # Cleanup old checkpoints
    _cleanup_old(cp_dir)

    log.debug(
        "checkpoint.saved channel=%s turn=%s rounds=%d trace=%d",
        channel,
        turn_id,
        rounds,
        len(trace),
    )
    return cp_path


def load_latest_checkpoint(channel: str, *, memory_root: Path | None = None) -> dict | None:
    """Load the most recent checkpoint for a channel. Returns None if none."""
    return _latest_in(_checkpoint_dir(channel, memory_root=memory_root, create=False))


def load_checkpoint(
    channel: str,
    turn_id: str,
    *,
    memory_root: Path | None = None,
) -> dict | None:
    """Load a specific checkpoint by turn_id."""
    cp_path = _checkpoint_path(
        channel,
        turn_id,
        memory_root=memory_root,
        create=False,
    )
    if not cp_path.exists():
        return None
    return _read_checkpoint(cp_path)


def list_checkpoints(channel: str, *, memory_root: Path | None = None) -> list[dict]:
    """List all checkpoints for a channel, newest first."""
    cp_dir = _checkpoint_dir(channel, memory_root=memory_root, create=False)
    if not cp_dir.is_dir():
        return []
    out = []
    for p in _checkpoint_files(cp_dir, limit=_MAX_CHECKPOINTS):
        try:
            data = _read_checkpoint(p)
            if data is None:
                continue
            trace = data.get("trace", [])
            messages = data.get("messages", [])
            out.append(
                {
                    "turn_id": data.get("turn_id", ""),
                    "timestamp": data.get("timestamp", 0),
                    "rounds": data.get("rounds", 0),
                    "model": data.get("model", ""),
                    "trace_count": len(trace) if isinstance(trace, list) else 0,
                    "message_count": len(messages) if isinstance(messages, list) else 0,
                    "path": str(p),
                }
            )
        except Exception as e:
            log.debug("checkpoint.load_failed path=%s error=%r", p, e)
            continue
    return out


def clear_checkpoints(channel: str, *, memory_root: Path | None = None) -> int:
    """Delete all checkpoints for a channel. Returns count deleted."""
    cp_dir = _checkpoint_dir(channel, memory_root=memory_root, create=False)
    if not cp_dir.is_dir():
        return 0
    count = 0
    for p in cp_dir.glob("checkpoint_*.json"):
        if p.is_symlink():
            continue
        try:
            p.unlink()
            count += 1
        except OSError:
            pass
    return count


def _cleanup_old(cp_dir: Path) -> None:
    """Keep only the last _MAX_CHECKPOINTS per channel."""
    checkpoints = _checkpoint_files(cp_dir, limit=_MAX_CHECKPOINT_FILES_SCANNED)
    if len(checkpoints) <= _MAX_CHECKPOINTS:
        return
    for p in checkpoints[_MAX_CHECKPOINTS:]:
        try:
            p.unlink()
        except OSError:
            pass


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Strip large content from messages to keep checkpoints manageable.

    Tool results can be huge (file contents, web pages). We keep the
    structure but truncate content > 2000 chars.
    """
    out = []
    for msg in messages[-_MAX_MESSAGES:]:
        if not isinstance(msg, dict):
            continue
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, str) and len(content) > 2000:
            m["content"] = content[:2000] + f"\n...[truncated, {len(content)} total chars]"
        # Truncate tool_calls arguments
        if "tool_calls" in m and isinstance(m["tool_calls"], list):
            tc_out = []
            for tc in m["tool_calls"]:
                tc_copy = dict(tc)
                fn = tc_copy.get("function", {})
                args = fn.get("arguments", "")
                if isinstance(args, str) and len(args) > 500:
                    fn_copy = dict(fn)
                    fn_copy["arguments"] = args[:500] + "...[truncated]"
                    tc_copy["function"] = fn_copy
                tc_out.append(tc_copy)
            m["tool_calls"] = tc_out
        out.append(m)
    return out


def _sanitize_trace(trace: list[dict]) -> list[dict]:
    """Strip large results from trace entries."""
    out = []
    for entry in trace[-_MAX_TRACE_ENTRIES:]:
        if not isinstance(entry, dict):
            continue
        e = dict(entry)
        result = e.get("result")
        if isinstance(result, dict):
            r = dict(result)
            for key in ("content", "stdout", "stderr", "text"):
                val = r.get(key)
                if isinstance(val, str) and len(val) > 1000:
                    r[key] = val[:1000] + f"...[truncated, {len(val)} total]"
            e["result"] = r
        elif isinstance(result, str) and len(result) > 1000:
            e["result"] = result[:1000] + "...[truncated]"
        out.append(e)
    return out


def has_pending_checkpoint(
    channel: str,
    max_age_sec: float = 3600,
    *,
    memory_root: Path | None = None,
) -> bool:
    """Check if there's a recent checkpoint that hasn't been resumed.

    A checkpoint is 'pending' if it's less than max_age_sec old, hasn't
    been marked as resumed, and its status is not ``complete``.
    """
    cp = load_latest_checkpoint(channel, memory_root=memory_root)
    if cp is None:
        return False
    return _is_pending(cp, max_age_sec)


def list_pending_checkpoints(memory_root: Path, max_age_sec: float = 3600) -> list[dict]:
    """Return pending checkpoints from one explicit memory root without mutations."""
    root = _checkpoint_root(memory_root)
    if not root.is_dir():
        return []
    pending: list[dict] = []
    try:
        with os.scandir(root) as directories:
            for scanned, entry in enumerate(directories, start=1):
                if scanned > _MAX_CHANNEL_DIRS_SCANNED:
                    log.warning(
                        "checkpoint.channel_scan_limit root=%s limit=%d",
                        root,
                        _MAX_CHANNEL_DIRS_SCANNED,
                    )
                    break
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                checkpoint = _latest_in(Path(entry.path))
                if checkpoint is not None and _is_pending(checkpoint, max_age_sec):
                    pending.append(checkpoint)
    except OSError:
        return []
    pending.sort(key=lambda item: float(item.get("timestamp", 0)), reverse=True)
    return pending


def mark_resumed(
    channel: str,
    turn_id: str,
    *,
    memory_root: Path | None = None,
) -> None:
    """Mark a checkpoint as resumed so it's not offered again."""
    cp_path = _checkpoint_path(
        channel,
        turn_id,
        memory_root=memory_root,
        create=False,
    )
    if not cp_path.exists():
        return
    try:
        data = _read_checkpoint(cp_path)
        if data is None:
            return
        data["resumed"] = True
        rendered = json.dumps(data, allow_nan=False, ensure_ascii=False)
        if len(rendered.encode("utf-8")) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds the 2 MiB serialized limit")
        atomic_write_text(
            cp_path,
            rendered,
            durable=True,
            mode=0o600,
        )
    except Exception as e:
        log.warning("checkpoint.mark_resumed_failed: %r", e)
