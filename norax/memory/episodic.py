"""Bounded, append-only turn episodes for offline replay and learning.

Episodic memory is evidence, not an authority channel. Persisted rows are
validated strictly before they can influence replay, prediction, or learned
tool patterns. Normal writes remain a single buffered append; no fsync or
model call is added to the turn path.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import stat
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ..atomic import append_bounded_text, read_bounded_text

log = logging.getLogger("norax.memory.episodic")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_EPISODE_FILE_BYTES = 64 * 1024 * 1024
_MAX_EPISODE_RECORD_BYTES = 256 * 1024
_MAX_EPISODE_LINES = 250_000
_MAX_EPISODE_FILES = 3_650
_MAX_RETRIEVAL_HITS = 512
_MAX_TOOL_CALLS = 128
_MAX_TOOL_CALL_BYTES = 16 * 1024
_MAX_RECENT_EPISODES = 10_000
_MAX_STRING_LENGTHS = {
    "user_input": 16_000,
    "user_input_hash": 128,
    "response_preview": 8_000,
    "arousal_level": 64,
    "vta_route": 64,
    "model": 512,
    "task_type": 256,
}


def _validate_text(name: str, value: Any, *, max_chars: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if len(value) > max_chars:
        raise ValueError(f"{name} exceeds {max_chars} characters")
    return value


def _validate_number(
    name: str,
    value: Any,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return number


def _validate_date(value: str) -> str:
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise ValueError("date must use YYYY-MM-DD")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError("date must use canonical YYYY-MM-DD")
    return value


def _safe_tool_call(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("tool_calls entries must be objects")
    if any(not isinstance(key, str) or len(key) > 128 for key in value):
        raise TypeError("tool_calls keys must be bounded text")
    name = value.get("name", "")
    if not isinstance(name, str) or not name or len(name) > 256:
        raise ValueError("tool_calls entries need a bounded tool name")
    if "ok" in value and type(value["ok"]) is not bool:
        raise TypeError("tool_calls ok must be a boolean")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("tool_calls entries must contain finite JSON data") from exc
    if len(encoded) > _MAX_TOOL_CALL_BYTES:
        raise ValueError("tool_calls entry exceeds its byte limit")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guaranteed by encoding input
        raise TypeError("tool_calls entries must be objects")
    return decoded


@dataclass
class Episode:
    """One validated turn episode."""

    timestamp: float = field(default_factory=time.time)
    user_input: str = ""
    user_input_hash: str = ""
    retrieval_hits: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    response_preview: str = ""
    arousal_level: str = "normal"
    vta_rpe: float = 0.0
    vta_route: str = "scratchpad"
    model: str = ""
    rounds: int = 0
    task_type: str = ""
    outcome_score: float = 5.0
    accepted_outcome: bool = False
    verified_outcome: bool = False
    objective_outcome_observed: bool = False
    training_eligible: bool = False

    def __post_init__(self) -> None:
        self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, validated snapshot of this mutable dataclass."""
        timestamp = _validate_number(
            "timestamp",
            self.timestamp,
            minimum=0,
            maximum=time.time() + 86_400,
        )
        values: dict[str, Any] = {"timestamp": timestamp}
        for name, max_chars in _MAX_STRING_LENGTHS.items():
            values[name] = _validate_text(name, getattr(self, name), max_chars=max_chars)

        if not isinstance(self.retrieval_hits, list):
            raise TypeError("retrieval_hits must be a list")
        if len(self.retrieval_hits) > _MAX_RETRIEVAL_HITS:
            raise ValueError("retrieval_hits exceeds its entry limit")
        values["retrieval_hits"] = [
            _validate_text("retrieval hit", item, max_chars=256) for item in self.retrieval_hits
        ]

        if not isinstance(self.tool_calls, list):
            raise TypeError("tool_calls must be a list")
        if len(self.tool_calls) > _MAX_TOOL_CALLS:
            raise ValueError("tool_calls exceeds its entry limit")
        values["tool_calls"] = [_safe_tool_call(item) for item in self.tool_calls]

        if isinstance(self.rounds, bool) or not isinstance(self.rounds, int):
            raise TypeError("rounds must be an integer")
        if not 0 <= self.rounds <= 10_000:
            raise ValueError("rounds must be between 0 and 10000")
        values["rounds"] = self.rounds
        values["vta_rpe"] = _validate_number(
            "vta_rpe",
            self.vta_rpe,
            minimum=-1_000,
            maximum=1_000,
        )
        values["outcome_score"] = _validate_number(
            "outcome_score",
            self.outcome_score,
            minimum=0,
            maximum=10,
        )
        for name in (
            "accepted_outcome",
            "verified_outcome",
            "objective_outcome_observed",
            "training_eligible",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
            values[name] = value
        return values

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Episode:
        if not isinstance(value, dict):
            raise TypeError("episode row must be an object")
        known = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in known})


@dataclass
class EpisodicBuffer:
    """Bounded append-only episodic memory store."""

    root: Path
    max_days: int = 30
    _today_file: Path = field(default_factory=lambda: Path("."))
    _today_key: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.max_days, bool) or not isinstance(self.max_days, int):
            raise TypeError("max_days must be an integer")
        if not 1 <= self.max_days <= _MAX_EPISODE_FILES:
            raise ValueError(f"max_days must be between 1 and {_MAX_EPISODE_FILES}")
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_stat = self.root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("episodic root must be a real directory")
        if os.name == "posix":
            os.chmod(self.root, 0o700)

    def _get_today_file(self) -> Path:
        key = time.strftime("%Y-%m-%d")
        if key != self._today_key:
            self._today_key = key
            self._today_file = self.root / f"episodes-{key}.jsonl"
        return self._today_file

    def _episode_files(self) -> list[Path]:
        files: list[Path] = []
        with os.scandir(self.root) as entries:
            for entry in entries:
                if not entry.name.startswith("episodes-") or not entry.name.endswith(".jsonl"):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                files.append(self.root / entry.name)
                if len(files) > _MAX_EPISODE_FILES:
                    raise ValueError("episodic store exceeds its file limit")
        return sorted(files, key=lambda path: path.name)

    def record(self, episode: Episode) -> bool:
        """Append one episode, surfacing failure so the caller can report truthfully."""
        if not isinstance(episode, Episode):
            raise TypeError("episode must be an Episode")
        encoded = json.dumps(
            episode.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) + 1 > _MAX_EPISODE_RECORD_BYTES:
            raise ValueError("episode exceeds its record byte limit")
        append_bounded_text(
            self._get_today_file(),
            encoded + "\n",
            max_bytes=_MAX_EPISODE_FILE_BYTES,
            mode=0o600,
        )
        return True

    def load_day(self, date_str: str) -> list[Episode]:
        """Load valid episodes for one canonical YYYY-MM-DD date."""
        date_key = _validate_date(date_str)
        path = self.root / f"episodes-{date_key}.jsonl"
        try:
            payload = read_bounded_text(
                path,
                max_bytes=_MAX_EPISODE_FILE_BYTES,
                errors="strict",
            )
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError, ValueError) as exc:
            log.warning("episodic.load_day.error date=%s: %r", date_key, exc)
            return []

        episodes: list[Episode] = []
        invalid = 0
        for line_no, raw in enumerate(payload.splitlines(), start=1):
            if line_no > _MAX_EPISODE_LINES:
                log.warning("episodic.load_day.line_limit date=%s", date_key)
                break
            line = raw.strip()
            if not line:
                continue
            if len(line.encode("utf-8")) > _MAX_EPISODE_RECORD_BYTES:
                invalid += 1
                continue
            try:
                decoded = json.loads(line)
                if not isinstance(decoded, dict):
                    raise TypeError("episode row must be an object")
                episodes.append(Episode.from_dict(decoded))
            except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
                invalid += 1
        if invalid:
            log.warning("episodic.load_day.skipped_invalid date=%s rows=%d", date_key, invalid)
        return episodes

    def recent_episodes(self, days: int = 3, limit: int = 100) -> list[Episode]:
        """Load the newest bounded set of episodes across recent days."""
        if (
            isinstance(days, bool)
            or not isinstance(days, int)
            or not 1 <= days <= max(self.max_days, 365)
        ):
            raise ValueError("days must be a bounded positive integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _MAX_RECENT_EPISODES
        ):
            raise ValueError(f"limit must be between 1 and {_MAX_RECENT_EPISODES}")
        episodes: list[Episode] = []
        today = dt.date.today()
        for offset in range(days):
            date_value = today - dt.timedelta(days=offset)
            episodes.extend(self.load_day(date_value.isoformat()))
        episodes.sort(key=lambda episode: episode.timestamp, reverse=True)
        return episodes[:limit]

    def prune_old(self) -> int:
        """Delete only regular, canonically named episode files past retention."""
        cutoff = dt.date.today() - dt.timedelta(days=self.max_days)
        removed = 0
        for path in self._episode_files():
            try:
                date_key = path.stem.removeprefix("episodes-")
                file_date = dt.date.fromisoformat(_validate_date(date_key))
                file_stat = path.lstat()
                if stat.S_ISREG(file_stat.st_mode) and file_date < cutoff:
                    path.unlink()
                    removed += 1
            except (ValueError, OSError):
                continue
        if removed:
            log.info("episodic.prune: removed %d old files", removed)
        return removed

    def stats(self) -> dict[str, Any]:
        """Return bounded and deterministically ordered status data."""
        files = self._episode_files()
        total_size = 0
        valid_files: list[Path] = []
        for path in files:
            try:
                file_stat = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            total_size += file_stat.st_size
            valid_files.append(path)
        return {
            "days": len(valid_files),
            "total_size_kb": round(total_size / 1024, 1),
            "oldest": valid_files[0].stem if valid_files else None,
            "newest": valid_files[-1].stem if valid_files else None,
        }
