"""Sleep → canonical consolidator — ADD-only distillation pipeline.

Mem0 2026 / MemGPT patterns:
  - Never overwrites canonical memories; always appends new facts
  - Extracts semantic facts and procedural patterns from sleep buffer files
  - Writes ``semantic/consolidated-{date}.md`` and ``procedural/consolidated-{date}.md``
  - Archives processed sleep files to ``sleep/archive/``
  - Dedup across runs via content-hash cache
  - STRIPS secrets/tokens from env output before storing
  - Uses _safe_parse_result for malformed/truncated JSON in tool results

Complements ``SleepFlusher`` (raw spill → sleep-flush-*.md). This module
distills buffer content into durable consolidated canonical files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import stat
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..atomic import append_bounded_text, atomic_write_text, path_lock, read_bounded_text
from .file_ops import archive_regular_file
from .process_lock import memory_process_lock

log = logging.getLogger("norax.memory.consolidator")

_FACT_RX = re.compile(r"^[A-Z][A-Z0-9_]{1,}[:=]")
_PROC_RX = re.compile(
    r"\b(STEP|RULE|POLICY|RETRY|VERIFY|FAILURE|FLOW|WHEN|THEN|PROCEDURE)\b",
    re.I,
)
_URL_RX = re.compile(r"https?://[^\s<>\"']+")
_NOISE_RX = re.compile(r"^(ok|done|ack|yes|no|sure|got it|thanks?)\b[.!?\s]*$", re.I)
_ID_RX = re.compile(r"(0x[a-fA-F0-9]{40}|@\S+\.\S+)")
_MIN_FACT_LEN = 15
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_CANONICAL_FILE_BYTES = 32 * 1024 * 1024
_MAX_CANONICAL_FILES = 10_000
_MAX_DEDUP_ENTRIES = 2_000_000
_MAX_CANONICAL_OUTPUT_BYTES = 256 * 1024 * 1024

# Never-matching sentinel: used as a safe fallback so a single malformed
# noise pattern can NEVER crash the service on import.
_NEVER_RX = re.compile(r"(?!x)x")


def _safe_compile(pattern: str, flags: int = 0, *, name: str = "?") -> re.Pattern[str]:
    """Compile a noise/denylist regex defensively.

    The noise denylists below are hand-maintained by pasting observed junk
    strings. A stray unescaped quote, backslash, or control byte must NOT take
    down memory consolidation (and with it the whole runtime). On failure we
    log and fall back to a pattern that matches nothing.
    """
    try:
        return re.compile(pattern, flags)
    except re.error as exc:  # noqa: BLE001
        log.error("noise regex %s failed to compile (%s); using no-op fallback", name, exc)
        return _NEVER_RX


_NOISE_GLOB_RX = _safe_compile(
    r"(^total\s+\d+|^[-drwx]{10}\s|^---$|node_modules|"
    r'^\\"chat_path\\":|^\\"provider_kind\\":|^Environment=|^ExecStart=|'
    r"^The user wants|^The fact-check|^The corrected|^The revised|"
    r"^Now I understand|^\*\*Port correction|^2\. Create|"
    r"^- \*\*Jellyseerr|^\| \*\*|^1\. Go to|"
    r"^\d+:\s+// |^REF:https|^\| \*\*Hermes|^@self\._app\.|^@app\.|^href=.http|^\d+:@app\.|^JELLYFIN_URL|^http://host\.docker|^\d+:\s+// )",
    re.I,
    name="_NOISE_GLOB_RX",
)
_JSON_FRAGMENT_RX = _safe_compile(r'^["{\\][\s\\"{}[\]A-Za-z0-9_:,.-]+$', name="_JSON_FRAGMENT_RX")

_STDOUT_NOISE_RX = _safe_compile(
    r"(^Permission denied|^Traceback |^  File |^ModuleNotFoundError|"
    r"^The following packages|^libframe|^libqt|^nvidia-firmware|"
    r"^Use .sudo|^\d+ upgraded|^ssh: connect|^\d+\.\d+\.\d+\.\d+: |"
    r"^sent \d+|^total size is|^grep: |^Unit .* could not be found|"
    r"^===.*===|^sshpass|^Package:|^Architecture:|^Version:|^Priority:|"
    r"^Section:|^hint: |^note: |^\b0x[a-fA-F0-9]{40}\b|"
    r"^tests/unit/|^\"bot_public\"|^\"bot_require|^total \d+|"
    r"^drwx|^-rw|^\"default_provider|^\"default_model|^// |"
    r"^bash: |^/usr/share/|^/usr/lib/|^lrwx|^\d+-\d+-\d+ |"
    r"^staging\s+\d+|^\[|^File \"|^72034dc1f|"
    r"^httpx\.|^Main PID:|^Tasks: \d|^Memory: \d|^Active: |^CGroup: |"
    r"^Loaded:|^Docs: |^Process:|^\(code=\)|^/usr/bin/python3|"
    r"^Active: |^cursor-sdk|"
    r"^The user wants|^The fact-check|^The corrected|^The revised|"
    r"^Now I understand|^\*\*Port correction|^2\. Create|"
    r"^- \*\*Jellyseerr|^\| \*\*|^1\. Go to|"
    r"^\d+:\s+// |^REF:https|^\| \*\*Hermes|^@self\._app\.|^@app\.|^href=.http|^OpusHead|^A_OPUS|^@app\.on_event|^\d+:@app\.|^JELLYFIN_URL|^http://host\.docker|^\d+:\s+// |TOKEN_PRESENT:yes|old_host_count|localhost_count|MEMORY_VERIFY:|VERIFY:\d+)",
    re.I,
    name="_STDOUT_NOISE_RX",
)
_SECRET_RX = re.compile(
    r"(TOKEN|KEY|SECRET|PASSWORD|DISCORD_TOKEN)=[^\s]{8,}",
    re.I,
)
_ENV_BLACKLIST = re.compile(
    r"^(NORAX_|CURSOR_|CODEX_|ANTHROPIC_|OPENAI_|GEMINI_|BRAVE_|SEARXNG_)",
    re.I,
)

_TOOL_KEY_RX = re.compile(
    r'"(ok|exit_code|stdout|stderr|command|path|content|error|result|offset|limit|query|url|items|files|matches|data|response)":',
)


def is_tool_dump(s: str) -> bool:
    """True for lines that are raw tool I/O, not durable facts.

    A line that parses as a JSON object/array, or a JSON fragment carrying
    tool-call keys, is command input/output — never canonical memory.
    Legacy consolidated-*.md files (pre-filter era) accumulated ~5k of these.
    """
    s = s.strip()
    if not s or s[0] not in "{[":
        return False
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return bool(_TOOL_KEY_RX.search(s))


def strip_tool_dumps(path: Path) -> int:
    """Remove tool-I/O JSON lines from a canonical memory file in place.

    Returns the number of lines removed. Only pure-JSON / tool-key fragment
    lines are dropped; prose facts are never touched.
    """
    with path_lock(path):
        try:
            lines = read_bounded_text(
                path,
                max_bytes=_MAX_CANONICAL_FILE_BYTES,
                errors="replace",
            ).splitlines()
        except OSError:
            return 0
        kept: list[str] = []
        removed = 0
        for line in lines:
            if is_tool_dump(line):
                removed += 1
            else:
                kept.append(line)
        if removed:
            atomic_write_text(path, "\n".join(kept) + "\n", mode=0o600)
        log.info("consolidator.strip_tool_dumps: %s removed=%d", path.name, removed)
    return removed


def _content_hash(text: str) -> str:
    canon = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


@dataclass
class ConsolidateResult:
    files_processed: int = 0
    facts_written: int = 0
    procedural_written: int = 0
    archived: list[str] = field(default_factory=list)
    skipped_dup: int = 0


class _DedupCache:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def contains(self, text: str) -> bool:
        h = _content_hash(text)
        return h in self._seen

    def commit(self, texts: list[str]) -> None:
        for text in texts:
            self._seen.add(_content_hash(text))
            if len(self._seen) > _MAX_DEDUP_ENTRIES:
                raise ValueError("canonical dedup cache exceeds its entry limit")

    def seed_from_dir(self, directory: Path) -> None:
        if not directory.exists():
            return
        paths = list(directory.rglob("consolidated-*.md"))
        paths.extend(directory.rglob("sleep-flush-*.md"))
        unique_paths = sorted(set(paths))
        if len(unique_paths) > _MAX_CANONICAL_FILES:
            raise ValueError("canonical memory exceeds its file limit")
        for p in unique_paths:
            file_stat = p.lstat()
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"canonical memory is not a regular file: {p}")
            text = read_bounded_text(
                p,
                max_bytes=_MAX_CANONICAL_FILE_BYTES,
                errors="replace",
            )
            for line in text.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    self._seen.add(_content_hash(stripped))
                    if len(self._seen) > _MAX_DEDUP_ENTRIES:
                        raise ValueError("canonical dedup cache exceeds its entry limit")


def _extract_from_text(text: str) -> tuple[list[str], list[str]]:
    semantic: list[str] = []
    procedural: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("---"):
            continue
        if _NOISE_RX.match(s) or len(s) < _MIN_FACT_LEN:
            continue
        if _ENV_BLACKLIST.match(s) or _NOISE_GLOB_RX.search(s):
            continue
        if _JSON_FRAGMENT_RX.match(s) or _STDOUT_NOISE_RX.search(s):
            continue
        if is_tool_dump(s):
            continue
        is_fact = bool(_FACT_RX.match(s)) or bool(_ID_RX.search(s))
        is_proc = bool(_PROC_RX.search(s))
        if is_proc:
            procedural.append(s[:500])
        elif is_fact:
            semantic.append(s[:500])
        elif any(kw in s.lower() for kw in ("verified", "configured", "deployed", "active")):
            procedural.append(f"VERIFY:{s[:400]}")
        elif _URL_RX.search(s):
            semantic.append(f"REF:{s[:400]}")
    return semantic, procedural


def _strip_secrets(text: str) -> str:
    return _SECRET_RX.sub(
        lambda m: m.group(0).split("=")[0] + "=***",
        text,
    )


def _safe_parse_result(raw: str) -> dict:
    result: dict = {}
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            return d
    except (json.JSONDecodeError, TypeError):
        pass
    for key in ("ok", "exitCode", "totalMatches", "totalFiles", "referenceCount"):
        m = re.search(rf'"{key}"\s*:\s*((?:true|false|null|-?\d+(?:\.\d+)?))', raw)
        if m:
            val = m.group(1)
            if val == "true":
                result[key] = True
            elif val == "false":
                result[key] = False
            elif val == "null":
                result[key] = None
            else:
                try:
                    result[key] = int(val)
                except ValueError:
                    pass
    m = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m:
        result["path"] = m.group(1)
    m = re.search(r'"stdout"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m:
        result["stdout"] = m.group(1)
    m = re.search(r'"content"\s*:\s*"((?s:.)*)', raw)
    if m:
        content = m.group(1)
        content = re.sub(r"[\s,}\]]+$", "", content)
        result["content"] = content
    m = re.search(r'"source"\s*:\s*"([^"]*)"', raw)
    if m:
        result["source"] = m.group(1)
    return result if result else {}


def _stdout_line_ok(sline: str) -> bool:
    if sline.startswith("{") or len(sline) < _MIN_FACT_LEN:
        return False
    if _ENV_BLACKLIST.match(sline) or _NOISE_RX.match(sline):
        return False
    if (
        _STDOUT_NOISE_RX.search(sline)
        or _NOISE_GLOB_RX.search(sline)
        or _JSON_FRAGMENT_RX.match(sline)
    ):
        return False
    return True


def _extract_from_jsonl(path: Path) -> tuple[list[str], list[str]]:
    semantic: list[str] = []
    procedural: list[str] = []
    last_tool: str | None = None
    text = read_bounded_text(path, max_bytes=_MAX_INPUT_BYTES, errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        kind_value = data.get("kind", data.get("role", ""))
        content_value = data.get("content", "")
        if not isinstance(kind_value, str) or not isinstance(content_value, str):
            continue
        kind = kind_value
        content = content_value.strip()[:16_000]
        if not content or _NOISE_RX.match(content):
            continue

        if kind == "tool_call":
            meta = data.get("meta") or {}
            tool_value = meta.get("name", "") if isinstance(meta, dict) else ""
            tool_name = tool_value if isinstance(tool_value, str) else ""
            tool_name = re.sub(r"\s+", " ", tool_name).strip()[:128]
            last_tool = tool_name or last_tool
            if tool_name:
                procedural.append(f"AGENT_USED_TOOL:{tool_name}")

        elif kind == "tool_result":
            if content == "{}":
                continue
            inner = _safe_parse_result(content)
            if not inner:
                continue
            stdout = str(inner.get("stdout", "")).strip()
            if stdout:
                stdout = _strip_secrets(stdout)
                for stdout_line in stdout.splitlines():
                    stdout_line = stdout_line.strip()
                    if _stdout_line_ok(stdout_line):
                        semantic.append(f"RUNTIME_FACT:{stdout_line[:300]}")
            file_path = str(inner.get("path", "")).strip()
            if file_path and not file_path.startswith("{"):
                procedural.append(f"READ_FILE:{file_path[:120]}")
            inner_content = str(inner.get("content", "")).strip()
            if inner_content and 20 < len(inner_content) < 2000:
                sem, proc = _extract_from_text(_strip_secrets(inner_content))
                semantic.extend(sem)
                procedural.extend(proc)
            tool_prefix = last_tool or "tool"
            for key in ("totalMatches", "totalFiles", "referenceCount"):
                if key in inner:
                    procedural.append(f"RESULT:{tool_prefix}.{key}={inner[key]}")
            if "ok" in inner:
                procedural.append(f"RESULT:{tool_prefix}.ok={inner['ok']}")
            if "exitCode" in inner:
                procedural.append(f"RESULT:{tool_prefix}.exitCode={inner['exitCode']}")

        elif kind in ("user", "assistant"):
            sem, proc = _extract_from_text(content)
            semantic.extend(sem)
            procedural.extend(proc)
    return semantic, procedural


def _extract_from_spill_md(path: Path) -> tuple[list[str], list[str]]:
    semantic: list[str] = []
    procedural: list[str] = []
    last_tool: str | None = None

    text = read_bounded_text(path, max_bytes=_MAX_INPUT_BYTES, errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("SPILL"):
            continue

        if s.startswith("TOOL_CALL#"):
            json_str = s.split(":", 1)[1] if ":" in s else "{}"
            inner = _safe_parse_result(json_str)
            if "command" in inner:
                tool_name = "exec"
            else:
                tool_name = str(inner.get("name", ""))
            last_tool = tool_name if tool_name else None

        elif s.startswith("TOOL_RESULT#"):
            json_str = s.split(":", 1)[1] if ":" in s else "{}"
            inner = _safe_parse_result(json_str)
            if not inner:
                continue

            result_tool = str(inner.get("source", "")) or str(inner.get("name", ""))
            if result_tool:
                effective_tool = result_tool
            else:
                effective_tool = last_tool or "unknown"
            procedural.append(f"AGENT_USED_TOOL:{effective_tool}")

            stdout = str(inner.get("stdout", "")).strip()
            if stdout:
                stdout = _strip_secrets(stdout)
                for sline in stdout.splitlines():
                    sline = sline.strip()
                    if _stdout_line_ok(sline):
                        semantic.append(f"RUNTIME_FACT:{sline[:300]}")

            file_path = str(inner.get("path", "")).strip()
            if file_path and not file_path.startswith("{"):
                procedural.append(f"READ_FILE:{file_path[:120]}")

            inner_content = str(inner.get("content", "")).strip()
            if inner_content and 20 < len(inner_content) < 2000:
                inner_content = _strip_secrets(inner_content)
                sem, proc = _extract_from_text(inner_content)
                semantic.extend(sem)
                procedural.extend(proc)

            tp = f"{effective_tool}"
            for key in ("totalMatches", "totalFiles", "ok", "exitCode"):
                if key in inner:
                    procedural.append(f"RESULT:{tp}.{key}={inner[key]}")

    return semantic, procedural


@dataclass
class MemoryConsolidator:
    memory_root: Path
    min_age_sec: float = 300.0
    _dedup: _DedupCache = field(default_factory=_DedupCache)

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_age_sec, bool)
            or not isinstance(self.min_age_sec, int | float)
            or not math.isfinite(float(self.min_age_sec))
            or not 0 <= self.min_age_sec <= 365 * 86_400
        ):
            raise ValueError("min_age_sec must be a finite value between 0 and one year")
        self.memory_root = Path(self.memory_root)
        self.sleep_dir = self.memory_root / "sleep"
        self.archive_dir = self.sleep_dir / "archive"
        self.semantic_dir = self.memory_root / "semantic"
        self.procedural_dir = self.memory_root / "procedural"
        self._dedup.seed_from_dir(self.semantic_dir)
        self._dedup.seed_from_dir(self.procedural_dir)

    def _eligible_files(self) -> list[Path]:
        if not self.sleep_dir.exists():
            return []
        now = time.time()
        out: list[Path] = []
        seen_batch: set[str] = set()
        for p in sorted(self.sleep_dir.glob("spill-*.jsonl")):
            file_stat = p.lstat()
            if stat.S_ISREG(file_stat.st_mode) and (now - file_stat.st_mtime) >= self.min_age_sec:
                out.append(p)
                seen_batch.add(p.stem)
        for p in sorted(self.sleep_dir.glob("spill-*.md")):
            file_stat = p.lstat()
            if stat.S_ISREG(file_stat.st_mode) and (now - file_stat.st_mtime) >= self.min_age_sec:
                if p.stem not in seen_batch:
                    out.append(p)
        for p in sorted(self.sleep_dir.glob("buffer-*.md")):
            file_stat = p.lstat()
            if stat.S_ISREG(file_stat.st_mode) and (now - file_stat.st_mtime) >= self.min_age_sec:
                out.append(p)
        if len(out) > _MAX_CANONICAL_FILES:
            raise ValueError("too many sleep inputs in one consolidation cycle")
        return out

    def _archive_processed(self, files: list[Path], res: ConsolidateResult) -> None:
        """Archive every consumed input, including empty and duplicate-only files.

        Leaving a processed file in ``sleep/`` makes the idle loop consume it
        forever.  A same-named archive may already exist when a daily buffer is
        recreated, so collisions get a stable mtime suffix instead of being
        silently skipped.
        """
        if not files:
            return
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            if path.suffix == ".jsonl":
                archive_regular_file(
                    path.with_suffix(".md"),
                    self.archive_dir,
                    max_bytes=_MAX_INPUT_BYTES,
                )
            archived = archive_regular_file(
                path,
                self.archive_dir,
                max_bytes=_MAX_INPUT_BYTES,
            )
            if archived is not None:
                res.archived.append(str(path))

    def consolidate(self, *, dry_run: bool = False) -> ConsolidateResult:
        """Consolidate eligible inputs under the maintenance process lock."""
        with memory_process_lock(self.memory_root):
            return self._consolidate_unlocked(dry_run=dry_run)

    def _consolidate_unlocked(self, *, dry_run: bool = False) -> ConsolidateResult:
        res = ConsolidateResult()
        files = self._eligible_files()
        ts = datetime.now(UTC).strftime("%Y-%m-%d")
        sem_out = self.semantic_dir / f"consolidated-{ts}.md"
        proc_out = self.procedural_dir / f"consolidated-{ts}.md"
        sem_lines: list[str] = []
        proc_lines: list[str] = []
        staged_hashes: set[str] = set()
        consumed_files: list[Path] = []

        for path in files:
            if path.suffix == ".jsonl":
                sem, proc = _extract_from_jsonl(path)
            elif path.name.startswith("spill-") and path.suffix == ".md":
                sem, proc = _extract_from_spill_md(path)
            else:
                text = read_bounded_text(path, max_bytes=_MAX_INPUT_BYTES, errors="replace")
                sem, proc = _extract_from_text(text)
            consumed_files.append(path)
            res.files_processed += 1

            for line in sem:
                line_hash = _content_hash(line)
                if not self._dedup.contains(line) and line_hash not in staged_hashes:
                    sem_lines.append(line)
                    staged_hashes.add(line_hash)
                else:
                    res.skipped_dup += 1
            for line in proc:
                line_hash = _content_hash(line)
                if not self._dedup.contains(line) and line_hash not in staged_hashes:
                    proc_lines.append(line)
                    staged_hashes.add(line_hash)
                else:
                    res.skipped_dup += 1

        if dry_run:
            res.facts_written = len(sem_lines)
            res.procedural_written = len(proc_lines)
            return res

        if sem_lines:
            self.semantic_dir.mkdir(parents=True, exist_ok=True)
            try:
                is_empty = sem_out.lstat().st_size == 0
            except FileNotFoundError:
                is_empty = True
            prefix = f"# consolidated from sleep {ts}\n" if is_empty else ""
            append_bounded_text(
                sem_out,
                prefix + "".join(f"{line}\n" for line in sem_lines),
                max_bytes=_MAX_CANONICAL_OUTPUT_BYTES,
                durable=True,
                mode=0o600,
            )
            self._dedup.commit(sem_lines)
            res.facts_written = len(sem_lines)

        if proc_lines:
            self.procedural_dir.mkdir(parents=True, exist_ok=True)
            try:
                is_empty = proc_out.lstat().st_size == 0
            except FileNotFoundError:
                is_empty = True
            prefix = f"# consolidated from sleep {ts}\n" if is_empty else ""
            append_bounded_text(
                proc_out,
                prefix + "".join(f"{line}\n" for line in proc_lines),
                max_bytes=_MAX_CANONICAL_OUTPUT_BYTES,
                durable=True,
                mode=0o600,
            )
            self._dedup.commit(proc_lines)
            res.procedural_written = len(proc_lines)

        # A processed input is consumed even when it was empty, low-signal, or
        # entirely duplicate.  Keeping it live would retrigger the complete
        # replay/learning/index pipeline on every idle poll.
        self._archive_processed(consumed_files, res)

        log.info(
            "consolidator done files=%d sem=%d proc=%d dup=%d archived=%d",
            res.files_processed,
            res.facts_written,
            res.procedural_written,
            res.skipped_dup,
            len(res.archived),
        )
        return res

    def consolidate_and_reindex(self, *, dry_run: bool = False) -> ConsolidateResult:
        """Consolidate and index one consistent canonical-memory snapshot."""
        with memory_process_lock(self.memory_root):
            res = self._consolidate_unlocked(dry_run=dry_run)
            if not dry_run and (res.facts_written > 0 or res.procedural_written > 0):
                try:
                    from .build_index import _build_memory_index_unlocked

                    _build_memory_index_unlocked(self.memory_root)
                    log.info("consolidate_and_reindex: index rebuilt after consolidation")
                except Exception as e:
                    log.warning("consolidate_and_reindex: index rebuild failed: %r", e)
        return res
