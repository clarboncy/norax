"""Secure local file operations shared by memory-maintenance pipelines."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def archive_regular_file(
    source: Path,
    archive_dir: Path,
    *,
    max_bytes: int,
    mode: int = 0o600,
) -> Path | None:
    """Move one regular file into an archive without overwriting a prior artifact.

    The source and archive directory are accessed through descriptors, and a
    hard-link commit gives collision-safe, same-filesystem archival. ``None``
    means the source disappeared before the transaction began.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o7777:
        raise ValueError("mode must be an integer between 0 and 0o7777")
    source = Path(source)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    source_dir_fd = os.open(source.parent, directory_flags | getattr(os, "O_NOFOLLOW", 0))
    archive_dir_fd = os.open(archive_dir, directory_flags | getattr(os, "O_NOFOLLOW", 0))
    source_fd = -1
    try:
        try:
            source_fd = os.open(
                source.name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=source_dir_fd,
            )
        except FileNotFoundError:
            return None
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
            raise ValueError(f"archive source is not a singly linked regular file: {source}")
        if source_stat.st_size > max_bytes:
            raise ValueError(f"archive source exceeds {max_bytes} bytes: {source}")
        os.fchmod(source_fd, mode)

        for collision in range(10_001):
            if collision == 0:
                candidate = source.name
            else:
                candidate = f"{source.stem}-{source_stat.st_mtime_ns}-{collision}{source.suffix}"
            try:
                os.link(
                    source.name,
                    candidate,
                    src_dir_fd=source_dir_fd,
                    dst_dir_fd=archive_dir_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                continue
            archived_stat = os.stat(candidate, dir_fd=archive_dir_fd, follow_symlinks=False)
            current_stat = os.stat(source.name, dir_fd=source_dir_fd, follow_symlinks=False)
            expected = (source_stat.st_dev, source_stat.st_ino)
            if (archived_stat.st_dev, archived_stat.st_ino) != expected or (
                current_stat.st_dev,
                current_stat.st_ino,
            ) != expected:
                os.unlink(candidate, dir_fd=archive_dir_fd)
                raise RuntimeError("archive source changed during commit")
            os.unlink(source.name, dir_fd=source_dir_fd)
            os.fsync(archive_dir_fd)
            os.fsync(source_dir_fd)
            return archive_dir / candidate
        raise FileExistsError(f"could not allocate an archive name for {source.name}")
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        os.close(archive_dir_fd)
        os.close(source_dir_fd)
