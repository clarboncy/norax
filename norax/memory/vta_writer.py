"""VTA-gated memory writer — routes turn outcomes to the correct memory tier.

VTA.evaluate() returns an RPEResult with a route:
  - scratchpad     → append to scratchpad.md (hot, ephemeral)
  - sleep_buffer   → append to memory/sleep/ (consolidate later)
  - immediate_flashbulb → write directly to semantic/ (high surprise = important)

This module does the actual writing.
"""

from __future__ import annotations

import logging
import os
import stat
import time
from pathlib import Path

from ..atomic import atomic_write_text, path_lock, read_bounded_text

log = logging.getLogger("norax.memory.vta_writer")

_MAX_OUTCOME_CHARS = 4_096
_MAX_SCRATCHPAD_BYTES = 256 * 1024
_MAX_DURABLE_FILE_BYTES = 16 * 1024 * 1024


def _one_line(value: str) -> str:
    """Normalize one memory record and prevent line/record injection."""
    return " ".join(part.strip() for part in str(value).splitlines() if part.strip())[
        :_MAX_OUTCOME_CHARS
    ]


def write_outcome(
    *,
    memory_root: Path,
    route: str,
    text: str,
    weight_tag: str = "",
    source: str = "vta",
) -> Path | None:
    """Write a line to the appropriate memory tier. Returns the file path written to."""

    line = _one_line(text)
    if not line:
        return None

    ts = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    normalized_weight = _one_line(weight_tag)[:32]
    if normalized_weight and normalized_weight not in line:
        line = f"{line}{normalized_weight}"[:_MAX_OUTCOME_CHARS]

    if route == "scratchpad":
        target = memory_root / "scratchpad.md"
        _append_line(
            target,
            line,
            header=f"SCRATCHPAD;updated={ts};type=hot_memory",
            deduplicate=True,
            max_lines=40,
            max_turn_lines=5,
        )
        log.info("vta_writer → scratchpad: %s", line[:80])
        return target

    elif route == "sleep_buffer":
        sleep_dir = memory_root / "sleep"
        sleep_dir.mkdir(parents=True, exist_ok=True)
        # One file per day to keep sleep dir tidy
        day = time.strftime("%Y-%m-%d")
        target = sleep_dir / f"buffer-{day}.md"
        _append_durable_line(target, line, header=f"# sleep-buffer {day}")
        log.info("vta_writer → sleep: %s", line[:80])
        return target

    elif route == "immediate_flashbulb":
        sem_dir = memory_root / "semantic"
        sem_dir.mkdir(parents=True, exist_ok=True)
        day = time.strftime("%Y-%m-%d")
        target = sem_dir / f"flashbulb-{day}.md"
        # Flashbulb events get W4 minimum
        if "|W" not in line:
            line = f"{line[: _MAX_OUTCOME_CHARS - 3]}|W4"
        _append_durable_line(target, line, header=f"# flashbulb events {day}")
        log.info("vta_writer → flashbulb: %s", line[:80])
        return target

    else:
        log.warning("vta_writer: unknown route %r, dropping", route)
        return None


def _append_line(
    path: Path,
    line: str,
    header: str = "",
    *,
    deduplicate: bool = False,
    max_lines: int | None = None,
    max_turn_lines: int | None = None,
) -> None:
    """Append one line with optional hot-scratchpad retention controls.

    Durable sleep-buffer and flashbulb files use the lossless defaults.  Only
    the prompt-facing scratchpad opts into deduplication and bounded history.
    The process lock prevents concurrent read/modify/write calls from losing a
    line, while atomic replacement prevents partial files after interruption.
    """
    normalized = line.strip()
    if not normalized:
        return

    with path_lock(path):
        try:
            lines = read_bounded_text(
                path,
                max_bytes=_MAX_SCRATCHPAD_BYTES,
                errors="replace",
            ).splitlines()
        except FileNotFoundError:
            lines = [header] if header else []

        if deduplicate and any(normalized == existing.strip() for existing in lines):
            return

        if max_turn_lines is not None and normalized.startswith("TURN:"):
            max_turn_lines = max(1, max_turn_lines)
            turn_indexes = [i for i, existing in enumerate(lines) if existing.startswith("TURN:")]
            excess = len(turn_indexes) - max_turn_lines + 1
            if excess > 0:
                remove_indexes = set(turn_indexes[:excess])
                lines = [existing for i, existing in enumerate(lines) if i not in remove_indexes]

        lines.append(normalized)

        if max_lines is not None:
            max_lines = max(1, max_lines)
            if len(lines) > max_lines:
                if header and lines:
                    lines = [lines[0], *lines[-(max_lines - 1) :]] if max_lines > 1 else [lines[0]]
                else:
                    lines = lines[-max_lines:]

        atomic_write_text(path, "\n".join(lines) + "\n")


def _append_durable_line(path: Path, line: str, *, header: str = "") -> None:
    """Append in O(1) without following links or rewriting the day's history."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    with path_lock(path):
        descriptor = os.open(path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"memory path is not a regular file: {path}")
            prefix = f"{header}\n" if header and file_stat.st_size == 0 else ""
            payload = f"{prefix}{line}\n".encode()
            if file_stat.st_size + len(payload) > _MAX_DURABLE_FILE_BYTES:
                raise ValueError(f"memory file exceeds {_MAX_DURABLE_FILE_BYTES} bytes: {path}")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while appending memory record")
                view = view[written:]
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def build_turn_summary(
    *,
    user_input: str,
    response_preview: str,
    tool_calls: int,
    rounds: int,
) -> str:
    """Build a compact memory line summarizing what happened in this turn."""
    ts = time.strftime("%H:%M")
    user_short = user_input[:60].replace("\n", " ")
    resp_short = response_preview[:60].replace("\n", " ")
    return f"TURN:{ts}|in={user_short}|out={resp_short}|tools={tool_calls}|rounds={rounds}"
