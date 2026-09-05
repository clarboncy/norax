"""Build a bounded SQLite FTS/entity projection from canonical memory.

Canonical files remain the source of truth. This module maintains a
rebuildable projection with exact file signatures, removes stale rows, and
never follows linked input or index files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text, read_bounded_text
from .process_lock import memory_process_lock

log = logging.getLogger("norax.memory.build_index")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'semantic',
    mtime REAL NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    entity_id TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 1.0,
    UNIQUE(path, start_line, end_line)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, path, kind,
    content='chunks', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS indexed_files (
    path TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    ctime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'unknown',
    first_seen TEXT NOT NULL DEFAULT '',
    last_seen TEXT NOT NULL DEFAULT '',
    ref_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS relations (
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL DEFAULT 'related_to',
    object_id TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 1.0,
    UNIQUE(subject_id, predicate, object_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, path, kind) VALUES (new.id, new.text, new.path, new.kind);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, path, kind)
    VALUES('delete', old.id, old.text, old.path, old.kind);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, path, kind)
    VALUES('delete', old.id, old.text, old.path, old.kind);
    INSERT INTO chunks_fts(rowid, text, path, kind) VALUES (new.id, new.text, new.path, new.kind);
END;
"""

_MAX_CHUNK_LINES = 6
_OVERLAP_STEP = 3
_MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_FILES = 10_000
_MAX_SOURCE_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_SOURCE_LINES = 250_000
_MAX_SOURCE_LINE_CHARS = 64 * 1024
_MAX_CHUNK_CHARS = 256 * 1024
_MAX_CHUNKS = 2_000_000
_MAX_DATABASE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ENTITY_GRAPH_BYTES = 64 * 1024 * 1024
_MAX_GRAPH_ENTITIES = 100_000
_MAX_GRAPH_LINKS = 500_000
_MAX_GRAPH_MEMBERSHIPS = 2_000_000
_MAX_ENTITY_NAME_CHARS = 512
_MAX_NEURON_ID_CHARS = 256


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def _entity_id_for(path: str, line: int) -> str:
    return hashlib.sha256(f"{path}|{line}".encode()).hexdigest()[:16]


def _chunks(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    if len(lines) > _MAX_SOURCE_LINES:
        raise ValueError("canonical memory file exceeds its line limit")
    chunks: list[tuple[int, int, str]] = []
    for offset in range(0, len(lines), _OVERLAP_STEP):
        selected = lines[offset : offset + _MAX_CHUNK_LINES]
        if any(len(line) > _MAX_SOURCE_LINE_CHARS for line in selected):
            raise ValueError("canonical memory contains an unbounded line")
        chunk = "\n".join(selected).strip()
        if chunk:
            if len(chunk) > _MAX_CHUNK_CHARS:
                raise ValueError("canonical memory chunk exceeds its character limit")
            chunks.append((offset + 1, offset + len(selected), chunk))
            if len(chunks) > _MAX_CHUNKS:
                raise ValueError("canonical memory exceeds its chunk limit")
    return chunks


def _signature(file_stat: os.stat_result) -> tuple[int, int, int]:
    return (file_stat.st_mtime_ns, file_stat.st_ctime_ns, file_stat.st_size)


def _safe_index_path(path: Path, *, kind: str) -> None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ValueError(f"{kind} must be a singly linked regular file: {path}")
    if file_stat.st_size > _MAX_DATABASE_BYTES:
        raise ValueError(f"{kind} exceeds its byte limit: {path}")
    if os.name == "posix":
        os.chmod(path, 0o600, follow_symlinks=False)


def _source_paths(memory_root: Path) -> list[tuple[str, Path, str]]:
    sources: list[tuple[str, Path, str]] = []
    for kind in ("semantic", "procedural", "intel"):
        directory = memory_root / kind
        try:
            directory_stat = directory.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError(f"canonical memory directory is not a real directory: {directory}")
        with os.scandir(directory) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if entry.name.endswith(".md") and entry.is_file(follow_symlinks=False)
            )
        for name in names:
            sources.append((kind, directory / name, f"{kind}/{name}"))

    for name in ("scratchpad.md", "active-focus.md"):
        path = memory_root / name
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(file_stat.st_mode):
            sources.append(("scratchpad", path, name))

    if len(sources) > _MAX_SOURCE_FILES:
        raise ValueError("canonical memory exceeds its file limit")
    return sources


def _read_stable_source(path: Path, initial_stat: os.stat_result) -> str:
    if not stat.S_ISREG(initial_stat.st_mode) or initial_stat.st_nlink != 1:
        raise ValueError(f"canonical memory source is not a singly linked regular file: {path}")
    if initial_stat.st_size > _MAX_SOURCE_FILE_BYTES:
        raise ValueError(f"canonical memory source exceeds its byte limit: {path}")
    text = read_bounded_text(path, max_bytes=_MAX_SOURCE_FILE_BYTES, errors="replace")
    final_stat = path.lstat()
    if (
        not stat.S_ISREG(final_stat.st_mode)
        or final_stat.st_nlink != 1
        or (initial_stat.st_dev, initial_stat.st_ino, *_signature(initial_stat))
        != (final_stat.st_dev, final_stat.st_ino, *_signature(final_stat))
    ):
        raise RuntimeError(f"canonical memory source changed while indexing: {path}")
    return text


def _validated_graph_rows(
    memory_root: Path,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str]]]:
    path = memory_root / "entity_graph.json"
    try:
        decoded = json.loads(read_bounded_text(path, max_bytes=_MAX_ENTITY_GRAPH_BYTES))
    except FileNotFoundError:
        return [], []
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        log.warning("memory index ignored invalid entity graph: %r", exc)
        return [], []
    try:
        if not isinstance(decoded, dict):
            raise ValueError("entity graph root must be an object")
        raw_entities = decoded.get("entities", {})
        raw_links = decoded.get("links", [])
        if not isinstance(raw_entities, dict) or not isinstance(raw_links, list):
            raise ValueError("entity graph collections are malformed")
        if len(raw_entities) > _MAX_GRAPH_ENTITIES or len(raw_links) > _MAX_GRAPH_LINKS:
            raise ValueError("entity graph exceeds its entry limits")

        entities: list[tuple[str, str, int]] = []
        normalized: dict[str, str] = {}
        memberships = 0
        for name, neuron_ids in raw_entities.items():
            if (
                not isinstance(name, str)
                or not name
                or len(name) > _MAX_ENTITY_NAME_CHARS
                or not isinstance(neuron_ids, list)
                or any(
                    not isinstance(neuron_id, str)
                    or not neuron_id
                    or len(neuron_id) > _MAX_NEURON_ID_CHARS
                    for neuron_id in neuron_ids
                )
            ):
                raise ValueError("entity graph contains an invalid entity")
            memberships += len(neuron_ids)
            if memberships > _MAX_GRAPH_MEMBERSHIPS:
                raise ValueError("entity graph exceeds its membership limit")
            lowered = name.lower()
            if lowered in normalized:
                raise ValueError("entity graph contains duplicate normalized entities")
            normalized[lowered] = name
            entities.append((hashlib.sha256(name.encode()).hexdigest()[:16], name, len(neuron_ids)))

        relations: list[tuple[str, str]] = []
        seen_relations: set[tuple[str, str]] = set()
        for link in raw_links:
            if (
                not isinstance(link, list | tuple)
                or len(link) != 2
                or not all(isinstance(value, str) for value in link)
            ):
                raise ValueError("entity graph contains an invalid link")
            source_name, target_name = link
            source = normalized.get(source_name.lower())
            target = normalized.get(target_name.lower())
            if source is None or target is None:
                raise ValueError("entity graph contains a dangling link")
            relation = (
                hashlib.sha256(source.encode()).hexdigest()[:16],
                hashlib.sha256(target.encode()).hexdigest()[:16],
            )
            if relation not in seen_relations:
                seen_relations.add(relation)
                relations.append(relation)
        return entities, relations
    except (TypeError, ValueError) as exc:
        log.warning("memory index ignored invalid entity graph: %s", exc)
        return [], []


def build_memory_index(memory_root: Path | None = None) -> dict[str, Any]:
    """Build the projection while serializing against canonical mutations."""
    root = (
        Path(memory_root)
        if memory_root is not None
        else Path(__file__).resolve().parents[2] / "memory"
    )
    with memory_process_lock(root):
        return _build_memory_index_unlocked(root)


def _build_memory_index_unlocked(memory_root: Path) -> dict[str, Any]:
    """Build one projection; caller must hold ``memory_process_lock``."""
    memory_root = Path(memory_root)
    index_dir = memory_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    index_stat = index_dir.lstat()
    if not stat.S_ISDIR(index_stat.st_mode):
        raise ValueError("memory index directory must be a real directory")
    if os.name == "posix":
        os.chmod(index_dir, 0o700)

    db_path = index_dir / "norax_memory.sqlite"
    meta_path = index_dir / "index_meta.json"
    for candidate, kind in (
        (db_path, "memory index database"),
        (db_path.with_name(f"{db_path.name}-wal"), "memory index WAL"),
        (db_path.with_name(f"{db_path.name}-shm"), "memory index shared state"),
    ):
        _safe_index_path(candidate, kind=kind)

    t0 = time.monotonic()
    sources = _source_paths(memory_root)
    total_source_bytes = 0
    for _, path, _ in sources:
        file_stat = path.lstat()
        total_source_bytes += file_stat.st_size
        if total_source_bytes > _MAX_SOURCE_TOTAL_BYTES:
            raise ValueError("canonical memory exceeds its total byte limit")

    stats = {
        "files_scanned": 0,
        "files_indexed": 0,
        "files_skipped": 0,
        "files_rejected": 0,
        "chunks_added": 0,
        "chunks_deleted": 0,
    }
    indexed_paths: set[str] = set()
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        if os.name == "posix":
            os.chmod(db_path, 0o600)
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
        conn.execute("BEGIN IMMEDIATE")
        existing_chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if existing_chunk_count > _MAX_CHUNKS:
            raise ValueError("existing memory index exceeds its chunk limit")

        existing_files = {
            row[0]: (row[1], int(row[2]), int(row[3]), int(row[4]))
            for row in conn.execute(
                "SELECT path, kind, mtime_ns, ctime_ns, size FROM indexed_files"
            )
        }

        for kind, path, rel_path in sources:
            stats["files_scanned"] += 1
            initial_stat = path.lstat()
            if not stat.S_ISREG(initial_stat.st_mode) or initial_stat.st_nlink != 1:
                stats["files_rejected"] += 1
                continue
            signature = _signature(initial_stat)
            if existing_files.get(rel_path) == (kind, *signature):
                indexed_paths.add(rel_path)
                stats["files_skipped"] += 1
                continue
            try:
                text = _read_stable_source(path, initial_stat)
                chunks = _chunks(text)
            except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
                stats["files_rejected"] += 1
                log.warning("memory index rejected %s: %s", rel_path, exc)
                continue

            deleted = conn.execute("DELETE FROM chunks WHERE path = ?", (rel_path,)).rowcount
            stats["chunks_deleted"] += max(0, deleted)
            for start_line, end_line, chunk in chunks:
                weight = 1.0
                weight_match = re.search(r"\|W([1-5])\b", chunk)
                if weight_match:
                    weight = int(weight_match.group(1)) / 5.0
                conn.execute(
                    "INSERT INTO chunks "
                    "(path, start_line, end_line, text, kind, mtime, content_hash, "
                    "entity_id, weight) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rel_path,
                        start_line,
                        end_line,
                        chunk,
                        kind,
                        initial_stat.st_mtime,
                        _content_hash(chunk),
                        _entity_id_for(rel_path, start_line),
                        weight,
                    ),
                )
                stats["chunks_added"] += 1
                if stats["chunks_added"] > _MAX_CHUNKS:
                    raise ValueError("memory index exceeds its chunk limit")
            conn.execute(
                "INSERT OR REPLACE INTO indexed_files "
                "(path, kind, mtime_ns, ctime_ns, size) VALUES (?, ?, ?, ?, ?)",
                (rel_path, kind, *signature),
            )
            indexed_paths.add(rel_path)
            stats["files_indexed"] += 1

        known_paths = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT path FROM chunks")
        } | set(existing_files)
        for stale_path in sorted(known_paths - indexed_paths):
            deleted = conn.execute("DELETE FROM chunks WHERE path = ?", (stale_path,)).rowcount
            conn.execute("DELETE FROM indexed_files WHERE path = ?", (stale_path,))
            stats["chunks_deleted"] += max(0, deleted)

        entities, relations = _validated_graph_rows(memory_root)
        conn.execute("DELETE FROM relations")
        conn.execute("DELETE FROM entities")
        date = datetime.now(UTC).date().isoformat()
        for entity_id, name, ref_count in entities:
            conn.execute(
                "INSERT INTO entities "
                "(entity_id, name, kind, first_seen, last_seen, ref_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entity_id, name, "entity", date, date, ref_count),
            )
        for source_id, target_id in relations:
            conn.execute(
                "INSERT INTO relations (subject_id, predicate, object_id, weight) "
                "VALUES (?, 'related_to', ?, 1.0)",
                (source_id, target_id),
            )

        final_chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if final_chunk_count > _MAX_CHUNKS:
            raise ValueError("memory index exceeds its total chunk limit")

        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()

    elapsed = time.monotonic() - t0
    meta = {
        "built_at": datetime.now(UTC).isoformat(),
        "elapsed_sec": round(elapsed, 3),
        "stats": stats,
        "entities": len(entities),
        "relations": len(relations),
        "schema_version": "v2",
        "fts5_available": True,
    }
    atomic_write_text(
        meta_path,
        json.dumps(meta, allow_nan=False, separators=(",", ":")) + "\n",
        mode=0o600,
    )
    log.info(
        "index built: files=%d indexed=%d chunks=%d entities=%d relations=%d (%.2fs)",
        stats["files_scanned"],
        stats["files_indexed"],
        stats["chunks_added"],
        len(entities),
        len(relations),
        elapsed,
    )
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(build_memory_index(), indent=2))
