"""Cross-process serialization for canonical memory mutations and projections."""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


@contextmanager
def memory_process_lock(memory_root: Path) -> Iterator[TextIO]:
    """Exclusively lock maintenance that mutates or indexes canonical memory.

    The lock is advisory and process-wide. Callers must acquire it from a
    worker thread when used by an asyncio runtime because acquisition may wait
    for the offline sleep worker.
    """
    root = Path(memory_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".maintenance.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    with handle:
        file_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ValueError("memory maintenance lock must be a singly linked regular file")
        os.fchmod(handle.fileno(), 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
