"""Memory store — flat directory of compressed-notation Markdown files.

Norax memory layout:
  memory/
    semantic/        — permanent knowledge (facts, entities, relationships)
    procedural/      — workflows, how-tos, policies
    intel/           — external intel, research notes
    sleep/           — rolling buffer awaiting consolidation (root only;
                       archive/ + processed_dumps/ are audit trail, not indexed)
    scratchpad.md    — hot state; read first on resume
    active-focus.md  — what matters NOW (attention gate)

Each file is parsed into a list of Neurons (one per line in compressed format,
or one per paragraph in free-form). A Neuron is the atomic retrieval unit.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import stat
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from ..atomic import read_bounded_text

log = logging.getLogger("norax.memory.store")

_MAX_MEMORY_FILE_BYTES = 16 * 1024 * 1024
_MAX_MEMORY_FILES_PER_AREA = 4_096
_MAX_NEURONS_PER_FILE = 50_000
_MAX_NEURONS_PER_AREA = 250_000
_MAX_NEURON_CHARS = 32_768


@dataclass
class Neuron:
    text: str
    path: Path
    line: int
    weight: float = 1.0  # from |W1-W5 tag if present
    mtime: float = 0.0
    kind: str = "semantic"  # semantic|procedural|intel|scratchpad|focus|sleep
    entity_id: str = ""  # sha256[:16] of canonical form

    def __post_init__(self):
        if not self.entity_id:
            canon = re.sub(r"\s+", " ", self.text).strip().lower()
            self.entity_id = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]

    def age_days(self, now: float | None = None) -> float:
        now = now or time.time()
        return max(0.0, (now - self.mtime) / 86400.0)


_WEIGHT_RX = re.compile(r"\|W([1-5])\b")


def _parse_weight(text: str) -> float:
    m = _WEIGHT_RX.search(text)
    if not m:
        return 1.0
    # W1=0.4, W2=0.6, W3=0.8, W4=1.0, W5=1.3
    return {"1": 0.4, "2": 0.6, "3": 0.8, "4": 1.0, "5": 1.3}[m.group(1)]


def _parse_file(path: Path, kind: str) -> list[Neuron]:
    try:
        text = read_bounded_text(
            path,
            max_bytes=_MAX_MEMORY_FILE_BYTES,
            encoding="utf-8",
            errors="replace",
        )
        mtime = path.stat(follow_symlinks=False).st_mtime
    except ValueError as error:
        log.warning("memory.file_ignored path=%s reason=%s", path, error)
        return []
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return []
    out: list[Neuron] = []

    # Lossless raw rolling-window spills are JSONL. Index each frame as a
    # retrievable sleep neuron so active context can stay small while raw
    # history remains recallable before sleep consolidation runs.
    if path.suffix == ".jsonl":
        for i, line in enumerate(text.splitlines()[:_MAX_NEURONS_PER_FILE], start=1):
            s = line.strip()[:_MAX_NEURON_CHARS]
            if not s:
                continue
            try:
                import json

                d = json.loads(s)
                if not isinstance(d, dict):
                    raise ValueError("JSONL frame must be an object")
                content = str(d.get("content", "")).strip()[:_MAX_NEURON_CHARS]
                if not content:
                    continue
                frame_kind = str(d.get("kind", "?"))[:64]
                turn_id = str(d.get("turn_id", "?"))[:128]
                call_id = str(d.get("call_id") or "-")[:128]
                prefix = f"RAW_FRAME;kind={frame_kind};turn={turn_id};call={call_id};"
                s = (prefix + " " + content)[:_MAX_NEURON_CHARS]
            except Exception:  # noqa: BLE001
                pass
            out.append(
                Neuron(text=s, path=path, line=i, weight=_parse_weight(s), mtime=mtime, kind=kind)
            )
        return out

    for i, line in enumerate(text.splitlines()[:_MAX_NEURONS_PER_FILE], start=1):
        s = line.strip()[:_MAX_NEURON_CHARS]
        if not s or s.startswith("#") or s.startswith("---"):
            continue
        out.append(
            Neuron(text=s, path=path, line=i, weight=_parse_weight(s), mtime=mtime, kind=kind)
        )
    return out


@dataclass
class MemoryStore:
    root: Path
    hot: list[Neuron] = field(default_factory=list)
    semantic: list[Neuron] = field(default_factory=list)
    procedural: list[Neuron] = field(default_factory=list)
    intel: list[Neuron] = field(default_factory=list)
    sleep: list[Neuron] = field(default_factory=list)
    _loaded_mtimes: dict[Path, tuple[int, int]] = field(default_factory=dict)
    _cached_neurons: dict[Path, list[Neuron]] = field(default_factory=dict)
    _refresh_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def from_env(cls) -> MemoryStore:
        root = Path(os.environ.get("NORAX_MEMORY_ROOT", Path.cwd() / "memory")).expanduser()
        return cls(root=root)

    def _load_file(self, path: Path, kind: str, seen: set[Path]) -> list[Neuron]:
        """Return cached neurons unless the file's timestamp or size changed."""
        seen.add(path)
        try:
            file_stat = path.stat(follow_symlinks=False)
        except (FileNotFoundError, PermissionError, OSError):
            self._loaded_mtimes.pop(path, None)
            self._cached_neurons.pop(path, None)
            return []
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > _MAX_MEMORY_FILE_BYTES:
            self._loaded_mtimes.pop(path, None)
            self._cached_neurons.pop(path, None)
            if file_stat.st_size > _MAX_MEMORY_FILE_BYTES:
                log.warning(
                    "memory.file_ignored path=%s reason=size limit (%d bytes)",
                    path,
                    _MAX_MEMORY_FILE_BYTES,
                )
            return []
        signature = (file_stat.st_mtime_ns, file_stat.st_size)
        if self._loaded_mtimes.get(path) == signature:
            return self._cached_neurons.get(path, [])
        parsed = _parse_file(path, kind)
        self._loaded_mtimes[path] = signature
        self._cached_neurons[path] = parsed
        return parsed

    @staticmethod
    def _bounded_paths(paths: Iterable[Path], *, area: str) -> list[Path]:
        """Materialize only a bounded discovery prefix before sorting it."""
        try:
            candidates = list(islice(paths, _MAX_MEMORY_FILES_PER_AREA + 1))
        except OSError as error:
            log.warning("memory.scan_failed area=%s error=%r", area, error)
            return []
        if len(candidates) > _MAX_MEMORY_FILES_PER_AREA:
            log.warning(
                "memory.file_limit area=%s limit=%d",
                area,
                _MAX_MEMORY_FILES_PER_AREA,
            )
            del candidates[_MAX_MEMORY_FILES_PER_AREA:]
        return sorted(candidates)

    def _append_file_neurons(
        self,
        out: list[Neuron],
        path: Path,
        kind: str,
        seen: set[Path],
    ) -> bool:
        """Append within the per-area neuron budget; return false when full."""
        remaining = _MAX_NEURONS_PER_AREA - len(out)
        if remaining <= 0:
            return False
        out.extend(self._load_file(path, kind, seen)[:remaining])
        return len(out) < _MAX_NEURONS_PER_AREA

    def _scan_sleep_dir(self, d: Path, seen: set[Path]) -> list[Neuron]:
        """Index only active root-level sleep buffer files.

        ``sleep/archive/`` and ``sleep/processed_dumps/`` hold post-consolidation
        tool traces — audit trail only, never injected or searched.
        """
        out: list[Neuron] = []
        if not d.exists():
            return out
        for p in self._bounded_paths(d.iterdir(), area="sleep"):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix in (".md", ".jsonl"):
                if not self._append_file_neurons(out, p, "sleep", seen):
                    log.warning("memory.neuron_limit area=sleep limit=%d", _MAX_NEURONS_PER_AREA)
                    break
        return out

    def _scan_dir(self, dir_name: str, kind: str, seen: set[Path]) -> list[Neuron]:
        d = self.root / dir_name
        out: list[Neuron] = []
        if not d.exists():
            return out
        if kind == "sleep":
            return self._scan_sleep_dir(d, seen)
        for p in self._bounded_paths(d.rglob("*.md"), area=dir_name):
            if not self._append_file_neurons(out, p, kind, seen):
                log.warning(
                    "memory.neuron_limit area=%s limit=%d",
                    dir_name,
                    _MAX_NEURONS_PER_AREA,
                )
                break
        return out

    def refresh(self) -> None:
        """Discover memory files and reparse only files that changed."""
        with self._refresh_lock:
            seen: set[Path] = set()
            self.semantic = self._scan_dir("semantic", "semantic", seen)
            self.procedural = self._scan_dir("procedural", "procedural", seen)
            self.intel = self._scan_dir("intel", "intel", seen)
            self.sleep = self._scan_dir("sleep", "sleep", seen)
            scratch = self.root / "scratchpad.md"
            focus = self.root / "active-focus.md"
            hot: list[Neuron] = []
            if scratch.exists():
                hot.extend(self._load_file(scratch, "scratchpad", seen))
            if focus.exists():
                hot.extend(self._load_file(focus, "focus", seen))
            self.hot = hot
            for stale in set(self._loaded_mtimes) - seen:
                self._loaded_mtimes.pop(stale, None)
                self._cached_neurons.pop(stale, None)

    def all_canonical(self) -> list[Neuron]:
        return self.semantic + self.procedural + self.intel

    def canonical_count(self) -> int:
        """Return the canonical neuron count without allocating a merged list."""
        return len(self.semantic) + len(self.procedural) + len(self.intel)

    def all_external(self) -> list[Neuron]:
        """What the external retriever searches — sleep buffer + intel."""
        return self.sleep + self.intel

    def iter_lines(self) -> Iterable[Neuron]:
        yield from self.hot
        yield from self.all_canonical()
        yield from self.sleep
