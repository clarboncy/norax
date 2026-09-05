"""Bounded SQLite FTS5 and entity-graph retrieval projection."""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..store import Neuron

log = logging.getLogger("norax.memory.retrievers.sqlite_index")

DB_FILENAME = "norax_memory.sqlite"
_MAX_DB_BYTES = 4 * 1024 * 1024 * 1024
_MAX_QUERY_CHARS = 16_000
_MAX_ENTITY_CHARS = 512
_MAX_ENTITY_ID_CHARS = 256
_MAX_RESULT_TEXT_CHARS = 256 * 1024
_MAX_PATH_CHARS = 2_048
_MAX_RESULTS = 100
_MAX_NEIGHBORS = 512
_MAX_DEPTH = 5


def _bounded_count(name: str, value: Any, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_RESULTS:
        raise ValueError(f"{name} must be an integer between 1 and {_MAX_RESULTS}")
    return value


def _validate_db_artifact(path: Path) -> tuple[int, int] | None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ValueError(f"memory index artifact must be a singly linked regular file: {path}")
    if file_stat.st_size > _MAX_DB_BYTES:
        raise ValueError(f"memory index artifact exceeds its byte limit: {path}")
    return (file_stat.st_dev, file_stat.st_ino)


def _valid_result_row(row: tuple[Any, ...]) -> dict[str, Any] | None:
    if len(row) != 9:
        return None
    row_id, path, start, end, text, kind, entity_id, weight, rank = row
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 1:
        return None
    if not isinstance(path, str) or not path or len(path) > _MAX_PATH_CHARS:
        return None
    parsed_path = Path(path)
    if parsed_path.is_absolute() or ".." in parsed_path.parts:
        return None
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 1
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end < start
    ):
        return None
    if not isinstance(text, str) or not text or len(text) > _MAX_RESULT_TEXT_CHARS:
        return None
    if not isinstance(kind, str) or not kind or len(kind) > 64:
        return None
    if not isinstance(entity_id, str) or len(entity_id) > _MAX_ENTITY_ID_CHARS:
        return None
    if (
        isinstance(weight, bool)
        or not isinstance(weight, int | float)
        or not math.isfinite(float(weight))
        or not 0 <= float(weight) <= 10
        or isinstance(rank, bool)
        or not isinstance(rank, int | float)
        or not math.isfinite(float(rank))
    ):
        return None
    return {
        "id": row_id,
        "path": path,
        "start_line": start,
        "end_line": end,
        "text": text,
        "kind": kind,
        "entity_id": entity_id,
        "weight": float(weight),
        "rank": float(rank),
    }


@dataclass
class SQLiteIndexRetriever:
    """Thread-safe read-only access to the persistent retrieval projection."""

    memory_root: Path
    k: int = 8
    _conn: sqlite3.Connection | None = field(default=None, repr=False, init=False)
    _available: bool = field(default=False, repr=False, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, init=False)

    def __post_init__(self) -> None:
        self.memory_root = Path(self.memory_root)
        self.k = _bounded_count("k", self.k)

    @property
    def db_path(self) -> Path:
        return self.memory_root / "index" / DB_FILENAME

    def _empty_connection(self) -> sqlite3.Connection:
        from ..build_index import SCHEMA

        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.executescript(SCHEMA)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is not None:
                return self._conn
            path = self.db_path
            identity = _validate_db_artifact(path)
            if identity is None:
                log.warning("sqlite_index: db not found at %s; run build_index first", path)
                self._conn = self._empty_connection()
                self._available = False
                return self._conn
            for suffix in ("-wal", "-shm"):
                _validate_db_artifact(path.with_name(f"{path.name}{suffix}"))

            uri = f"{path.resolve(strict=True).as_uri()}?mode=ro"
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=5.0,
                check_same_thread=False,
            )
            try:
                current = path.lstat()
                if (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or (current.st_dev, current.st_ino) != identity
                ):
                    raise RuntimeError("memory index changed while opening")
                connection.execute("PRAGMA busy_timeout=5000")
                connection.execute("PRAGMA query_only=ON")
            except Exception:
                connection.close()
                raise
            self._conn = connection
            self._available = True
            return connection

    def search_keyword(self, query: str, *, k: int | None = None) -> list[dict[str, Any]]:
        """Run a bounded, literal-term FTS query using BM25 ranking."""
        if not isinstance(query, str) or len(query) > _MAX_QUERY_CHARS:
            raise ValueError(f"query must be text of at most {_MAX_QUERY_CHARS} characters")
        result_limit = _bounded_count("k", k, default=self.k)
        terms = query.replace('"', '""').split()
        if not terms:
            return []
        match_expr = " OR ".join(f'"{term}"' for term in terms[:8])
        statement = (
            "SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.kind, "
            "c.entity_id, c.weight, bm25(chunks_fts) AS rank "
            "FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?"
        )
        with self._lock:
            try:
                rows = self._connect().execute(statement, (match_expr, result_limit)).fetchall()
            except sqlite3.Error as exc:
                log.warning("sqlite_index.keyword_query_failed: %s", exc)
                return []
        results: list[dict[str, Any]] = []
        for row in rows:
            validated = _valid_result_row(row)
            if validated is not None:
                results.append(validated)
        return results

    def search_entities(self, name: str, *, k: int = 10) -> list[dict[str, Any]]:
        """Find entities by a bounded literal substring."""
        if not isinstance(name, str) or len(name) > _MAX_ENTITY_CHARS:
            raise ValueError(f"name must be text of at most {_MAX_ENTITY_CHARS} characters")
        if not name.strip():
            return []
        result_limit = _bounded_count("k", k)
        escaped = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            try:
                rows = (
                    self._connect()
                    .execute(
                        "SELECT entity_id, name, kind, ref_count FROM entities "
                        "WHERE name LIKE ? ESCAPE '\\' LIMIT ?",
                        (f"%{escaped}%", result_limit),
                    )
                    .fetchall()
                )
            except sqlite3.Error as exc:
                log.warning("sqlite_index.entity_query_failed: %s", exc)
                return []
        entities: list[dict[str, Any]] = []
        for entity_id, entity_name, kind, ref_count in rows:
            if (
                not isinstance(entity_id, str)
                or not entity_id
                or len(entity_id) > _MAX_ENTITY_ID_CHARS
                or not isinstance(entity_name, str)
                or not entity_name
                or len(entity_name) > _MAX_ENTITY_CHARS
                or not isinstance(kind, str)
                or len(kind) > 64
                or isinstance(ref_count, bool)
                or not isinstance(ref_count, int)
                or ref_count < 0
            ):
                continue
            entities.append(
                {
                    "entity_id": entity_id,
                    "name": entity_name,
                    "kind": kind,
                    "ref_count": ref_count,
                }
            )
        return entities

    def entity_neighbors(self, entity_id: str, depth: int = 1) -> list[dict[str, Any]]:
        """Traverse a bounded number of entity links up to five hops."""
        if not isinstance(entity_id, str) or not entity_id or len(entity_id) > _MAX_ENTITY_ID_CHARS:
            raise ValueError("entity_id must be bounded non-empty text")
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= _MAX_DEPTH:
            raise ValueError(f"depth must be an integer between 1 and {_MAX_DEPTH}")
        visited = {entity_id}
        frontier = [entity_id]
        neighbors: list[dict[str, Any]] = []
        statements = (
            (
                "SELECT r.object_id, r.predicate, e.name FROM relations r "
                "LEFT JOIN entities e ON e.entity_id = r.object_id WHERE r.subject_id = ? LIMIT ?",
                False,
            ),
            (
                "SELECT r.subject_id, r.predicate, e.name FROM relations r "
                "LEFT JOIN entities e ON e.entity_id = r.subject_id WHERE r.object_id = ? LIMIT ?",
                True,
            ),
        )
        with self._lock:
            connection = self._connect()
            try:
                for hop in range(1, depth + 1):
                    next_frontier: list[str] = []
                    for current_id in frontier[:_MAX_NEIGHBORS]:
                        for statement, _incoming in statements:
                            remaining = _MAX_NEIGHBORS - len(neighbors)
                            if remaining <= 0:
                                return neighbors
                            rows = connection.execute(statement, (current_id, remaining)).fetchall()
                            for neighbor_id, predicate, neighbor_name in rows:
                                if (
                                    not isinstance(neighbor_id, str)
                                    or not neighbor_id
                                    or len(neighbor_id) > _MAX_ENTITY_ID_CHARS
                                    or neighbor_id in visited
                                ):
                                    continue
                                safe_name = (
                                    neighbor_name
                                    if isinstance(neighbor_name, str)
                                    and 0 < len(neighbor_name) <= _MAX_ENTITY_CHARS
                                    else neighbor_id
                                )
                                safe_predicate = (
                                    predicate
                                    if isinstance(predicate, str) and len(predicate) <= 128
                                    else "related_to"
                                )
                                visited.add(neighbor_id)
                                next_frontier.append(neighbor_id)
                                neighbors.append(
                                    {
                                        "entity_id": neighbor_id,
                                        "name": safe_name,
                                        "predicate": safe_predicate,
                                        "depth": hop,
                                    }
                                )
                    frontier = next_frontier
                    if not frontier:
                        break
            except sqlite3.Error as exc:
                log.warning("sqlite_index.neighbor_query_failed: %s", exc)
        return neighbors

    def stats(self) -> dict[str, Any]:
        """Return retrieval projection status without crashing health endpoints."""
        with self._lock:
            try:
                connection = self._connect()
                chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                entities = int(connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
                relations = int(connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0])
                kinds = {
                    str(row[0]): int(row[1])
                    for row in connection.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind")
                }
            except (sqlite3.Error, ValueError, RuntimeError) as exc:
                log.warning("sqlite_index.stats_failed: %s", exc)
                return {
                    "available": False,
                    "chunks": 0,
                    "entities": 0,
                    "relations": 0,
                    "by_kind": {},
                    "db_size_kb": 0,
                }
            size = 0
            try:
                identity = _validate_db_artifact(self.db_path)
                if identity is not None:
                    size = self.db_path.lstat().st_size
            except (OSError, ValueError):
                return {
                    "available": False,
                    "chunks": chunks,
                    "entities": entities,
                    "relations": relations,
                    "by_kind": kinds,
                    "db_size_kb": 0,
                }
            return {
                "available": self._available,
                "chunks": chunks,
                "entities": entities,
                "relations": relations,
                "by_kind": kinds,
                "db_size_kb": size / 1024,
            }

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                self._available = False

    @staticmethod
    def _to_neuron(row: dict[str, Any]) -> Neuron:
        return Neuron(
            text=row["text"],
            path=Path(row["path"]),
            line=row["start_line"],
            weight=row["weight"],
            kind=row["kind"],
            entity_id=row["entity_id"],
        )

    async def search(self, query: str, *, k: int = 6) -> list[tuple[Neuron, float, str]]:
        """Return uniform ``(Neuron, score, source)`` retrieval triples."""
        rows = await asyncio.to_thread(self.search_keyword, query, k=k)
        return [(self._to_neuron(row), -row["rank"], "sqlite_index") for row in rows]
