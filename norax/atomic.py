"""Small, dependency-free helpers for safe local state-file replacement."""

from __future__ import annotations

import os
import stat
import tempfile
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()


def read_bounded_bytes(
    path: Path,
    *,
    max_bytes: int,
    require_private: bool = False,
) -> bytes:
    """Read one regular file without following symlinks or exceeding a byte cap.

    Callers use this for state that can influence prompts or runtime decisions.
    Checking the opened descriptor closes the usual ``exists()``/``open()`` race,
    and the read limit remains authoritative even if a file grows concurrently.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    open_descriptor = descriptor
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"state path is not a regular file: {target}")
        if require_private and os.name == "posix" and stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise PermissionError(f"private state file is accessible by group or others: {target}")
        if file_stat.st_size > max_bytes:
            raise ValueError(f"state file exceeds {max_bytes} bytes: {target}")
        with os.fdopen(descriptor, "rb") as handle:
            open_descriptor = -1
            payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError(f"state file exceeds {max_bytes} bytes: {target}")
        return payload
    finally:
        if open_descriptor >= 0:
            os.close(open_descriptor)


def read_bounded_text(
    path: Path,
    *,
    max_bytes: int,
    encoding: str = "utf-8",
    errors: str = "strict",
    require_private: bool = False,
) -> str:
    """Text counterpart to :func:`read_bounded_bytes`."""
    return read_bounded_bytes(
        Path(path),
        max_bytes=max_bytes,
        require_private=require_private,
    ).decode(encoding, errors=errors)


def _lock_key(path: Path) -> str:
    return os.path.abspath(os.fspath(path.expanduser()))


@contextmanager
def path_lock(path: Path) -> Iterator[None]:
    """Serialize in-process read/modify/write operations for one path."""
    key = _lock_key(Path(path))
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.setdefault(key, threading.RLock())
    with lock:
        yield


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    durable: bool = False,
    mode: int | None = None,
) -> None:
    """Replace text atomically with a unique same-directory temporary file.

    ``durable=True`` also flushes the file and containing directory. It is
    intentionally opt-in because fsync on every cognitive-state update would
    add avoidable latency; financial and credential state should enable it.
    ``mode`` is applied to the temporary file before it becomes visible.
    """
    target = Path(path)
    if mode is not None and (
        isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o7777
    ):
        raise ValueError("mode must be an integer between 0 and 0o7777")
    target.parent.mkdir(parents=True, exist_ok=True)
    with path_lock(target):
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        open_fd = fd
        try:
            if mode is not None:
                os.fchmod(fd, mode)
            handle = os.fdopen(fd, "w", encoding=encoding)
            open_fd = -1
            with handle:
                handle.write(text)
                if durable:
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(tmp_name, target)
            if durable:
                dir_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            if open_fd >= 0:
                try:
                    os.close(open_fd)
                except OSError:
                    pass
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def atomic_create_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> bool:
    """Create a complete file exactly once without exposing partial contents.

    Returns ``True`` for the creator and ``False`` if another process already
    installed the target. A same-directory hard link is the atomic commit,
    which avoids the partial-read race of ``O_CREAT | O_EXCL`` followed by a
    write.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o7777:
        raise ValueError("mode must be an integer between 0 and 0o7777")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with path_lock(target):
        if target.exists():
            return False
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        open_fd = fd
        try:
            os.fchmod(fd, mode)
            handle = os.fdopen(fd, "wb")
            open_fd = -1
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp_name, target)
            except FileExistsError:
                return False
            dir_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            return True
        finally:
            if open_fd >= 0:
                try:
                    os.close(open_fd)
                except OSError:
                    pass
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def append_bounded_text(
    path: Path,
    text: str,
    *,
    max_bytes: int,
    encoding: str = "utf-8",
    durable: bool = False,
    mode: int = 0o600,
) -> int:
    """Securely append text to a regular file under a total byte ceiling.

    The opened descriptor is checked before use, symlinks and hard-linked
    files are rejected, and an advisory file lock serializes cooperating
    worker processes. The returned value is the resulting file size.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o7777:
        raise ValueError("mode must be an integer between 0 and 0o7777")
    encoded = text.encode(encoding)
    if len(encoded) > max_bytes:
        raise ValueError("appended text exceeds the file byte limit")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with path_lock(target):
        flags = (
            os.O_CREAT
            | os.O_WRONLY
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(target, flags, mode)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise ValueError(f"append target is not a singly linked regular file: {target}")
            os.fchmod(descriptor, mode)
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                pass
            current_size = os.fstat(descriptor).st_size
            if current_size + len(encoded) > max_bytes:
                raise ValueError(f"append would exceed {max_bytes} bytes: {target}")
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError(f"short write while appending to {target}")
                remaining = remaining[written:]
            if durable:
                os.fsync(descriptor)
            return current_size + len(encoded)
        finally:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                pass
            os.close(descriptor)
