"""Sleep-flush — idle-time consolidation of `memory/sleep/` into canonical stores.

Runs after the system has been idle for `idle_after_sec` (default: 3600).
The rolling window is in charge of *filling* sleep/ losslessly during
active work. This module is in charge of *cleaning* it during downtime.

Pipeline per flush:
  1. SCAN — find all spill-*.jsonl files in sleep/ older than
     `min_age_sec` (default: 300) so we never race the writer.
  2. PARSE — load every frame; drop obvious noise (acks, empty,
     system-level retries).
  3. EXTRACT — convert frames to candidate neurons:
        - user_msg / assistant_msg → statement neuron
        - tool_call+tool_result pair → action-outcome neuron
  4. DEDUP — build entity_ids; drop exact dupes against existing
     canonical files (semantic/, procedural/, intel/).
  5. SCORE — rank candidates by inferred importance:
        - ID/weight in source frame
        - whether it names identifiers (wallet, email, etc.)
        - whether it names the owner or system constants
  6. ROUTE — each candidate lands in:
        - semantic/   if it looks like a fact or permanent state
        - procedural/ if it looks like a workflow/policy
        - intel/      if it's external research
        - (discard)   if it's trivia or low-weight noise
  7. WRITE — append/merge into the target file, applying W-tag if
     the importance score clears a threshold.
  8. ARCHIVE — move the consumed spill-*.jsonl + .md sidecars into
     sleep/archive/ and retain them for the configured audit window.

The flush is idempotent: running it twice in a row is safe. It uses
(entity_id, path) keys for dedup so re-running can't produce dupes.
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
from datetime import UTC
from pathlib import Path

from ..atomic import append_bounded_text, read_bounded_text
from ..memory.file_ops import archive_regular_file
from ..memory.process_lock import memory_process_lock

log = logging.getLogger("norax.context.sleep_flush")

_MAX_SPILL_BYTES = 32 * 1024 * 1024
_MAX_SPILL_LINES = 100_000
_MAX_FRAME_CHARS = 16_000
_MAX_PENDING_CALLS = 4_096
_MAX_CANONICAL_FILE_BYTES = 32 * 1024 * 1024
_MAX_CANONICAL_FILES = 10_000
_MAX_CANONICAL_ENTITIES = 2_000_000
_MAX_CANONICAL_OUTPUT_BYTES = 256 * 1024 * 1024


# ---------- candidate extraction -------------------------------------------


@dataclass
class Candidate:
    text: str
    kind: str  # "statement" | "action_outcome" | "directive"
    weight: float  # W1..W5 → 0.4..1.3
    importance: float
    source_file: Path
    source_line: int = 0
    route: str = "discard"  # "semantic" | "procedural" | "intel" | "discard"

    @property
    def entity_id(self) -> str:
        canon = re.sub(r"\s+", " ", self.text).strip().lower()
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


# Patterns that promote a candidate's route
_ID_RX = re.compile(r"(0x[a-fA-F0-9]{40}|@\S+\.\S+|\$[a-zA-Z]\w{2,})")
_FACT_RX = re.compile(r"^[A-Z][A-Z0-9_]{2,}\s*[:=]")
_WORKFLOW_RX = re.compile(r"\b(STEP|RULE|POLICY|RETRY|FAIL|FLOW|WHEN|THEN)\b")
_INTEL_RX = re.compile(r"\b(SOURCE|URL|RESEARCH|BENCHMARK|RELEASE|CVE-\d)", re.I)
_NOISE_RX = re.compile(r"^(ok|done|ack|yes|no|sure|got it|thanks?)\b[.!?\s]*$", re.I)
# Tool-trace junk: raw JSON outcome blobs from action_outcome pairs that carry
# no narrative signal. Filtered at extraction so they never reach _route().
_JSON_BLOB_RX = re.compile(
    r'OUTCOME:\s*\{"(ok|command|stdout|stderr|exit_code|content|path|bytes|lines|diff_bytes)"'
)
_SECRET_RX = re.compile(
    r"(?i)\b(token|api[_-]?key|secret|password|authorization)\s*[:=]\s*([^\s,;]{8,})"
)


def _safe_memory_text(value: str, *, max_chars: int) -> str:
    collapsed = re.sub(r"\s+", " ", value).strip()[:max_chars]
    return _SECRET_RX.sub(lambda match: f"{match.group(1)}=***", collapsed)


def _importance(text: str, base_weight: float) -> float:
    score = base_weight
    if _ID_RX.search(text):
        score += 0.5
    if _FACT_RX.search(text):
        score += 0.3
    # Explicit workflows and research/URL intel are high-signal even when they
    # have no |W tag; keep them above the stricter sleep-flush threshold.
    if _WORKFLOW_RX.search(text):
        score += 0.2
    if _INTEL_RX.search(text):
        score += 0.2
    if re.search(r"\|W[4-5]\b", text):
        score += 0.4
    if len(text) < 8 or _NOISE_RX.match(text):
        score -= 1.0
    return score


def _route(text: str) -> str:
    if _NOISE_RX.match(text):
        return "discard"
    if _INTEL_RX.search(text):
        return "intel"
    if _WORKFLOW_RX.search(text):
        return "procedural"
    if _FACT_RX.search(text):
        return "semantic"
    # No matching signal → discard. Don't promote arbitrary "long + has colon"
    # content to intel; that produced 624 lines of tool-trace junk over 2 weeks.
    return "discard"


def _parse_weight(text: str) -> float:
    m = re.search(r"\|W([1-5])\b", text)
    if not m:
        return 1.0
    return {"1": 0.4, "2": 0.6, "3": 0.8, "4": 1.0, "5": 1.3}[m.group(1)]


def extract_candidates(spill_path: Path) -> list[Candidate]:
    try:
        raw_spill = read_bounded_text(
            spill_path,
            max_bytes=_MAX_SPILL_BYTES,
            errors="replace",
        )
    except FileNotFoundError:
        return []
    out: list[Candidate] = []
    # Buffer tool_call + tool_result pairs by call_id
    pending: dict[str, dict] = {}
    for line_no, raw in enumerate(raw_spill.splitlines(), start=1):
        if line_no > _MAX_SPILL_LINES:
            raise ValueError("sleep spill exceeds its line limit")
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(frame, dict):
            continue
        kind = frame.get("kind", frame.get("role"))
        raw_content = frame.get("content")
        if not isinstance(kind, str) or not isinstance(raw_content, str):
            continue
        content = _safe_memory_text(raw_content, max_chars=_MAX_FRAME_CHARS)
        if not content:
            continue
        call_id = frame.get("call_id")
        call_key = call_id if isinstance(call_id, str) and 1 <= len(call_id) <= 256 else None

        if kind == "tool_call" and call_key is not None:
            if len(pending) >= _MAX_PENDING_CALLS and call_key not in pending:
                pending.pop(next(iter(pending)))
            pending[call_key] = {"call": content, "line": line_no}
            continue
        if kind == "tool_result" and call_key is not None and call_key in pending:
            paired = pending.pop(call_key)
            combined = f"ACTION:{paired['call'][:80]} → OUTCOME:{content[:120]}"
            if _JSON_BLOB_RX.search(combined):
                continue
            weight = _parse_weight(combined)
            out.append(
                Candidate(
                    text=combined,
                    kind="action_outcome",
                    weight=weight,
                    importance=_importance(combined, weight),
                    source_file=spill_path,
                    source_line=paired["line"],
                    route=_route(combined),
                )
            )
            continue

        weight = _parse_weight(content)
        candidate_text = content[:400]
        candidate_kind = "statement" if kind in ("user", "assistant") else "directive"
        out.append(
            Candidate(
                text=candidate_text,
                kind=candidate_kind,
                weight=weight,
                importance=_importance(candidate_text, weight),
                source_file=spill_path,
                source_line=line_no,
                route=_route(candidate_text),
            )
        )
    return out


# ---------- canonical dedup ------------------------------------------------


def _load_existing_entity_ids(dir_path: Path) -> set[str]:
    if not dir_path.exists():
        return set()
    seen: set[str] = set()
    paths = list(dir_path.rglob("*.md"))
    if len(paths) > _MAX_CANONICAL_FILES:
        raise ValueError("canonical memory exceeds its file limit")
    for p in paths:
        file_stat = p.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"canonical memory is not a regular file: {p}")
        text = read_bounded_text(
            p,
            max_bytes=_MAX_CANONICAL_FILE_BYTES,
            errors="replace",
        )
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("---"):
                continue
            canon = re.sub(r"\s+", " ", s).lower()
            seen.add(hashlib.sha256(canon.encode()).hexdigest()[:16])
            if len(seen) > _MAX_CANONICAL_ENTITIES:
                raise ValueError("canonical memory exceeds its entity limit")
    return seen


# ---------- the flush ------------------------------------------------------


@dataclass
class FlushResult:
    spills_processed: int = 0
    candidates_seen: int = 0
    written: dict[str, int] = field(
        default_factory=lambda: {
            "semantic": 0,
            "procedural": 0,
            "intel": 0,
            "discard": 0,
            "dup": 0,
        }
    )
    archived: list[Path] = field(default_factory=list)


@dataclass
class SleepFlusher:
    memory_root: Path
    min_age_sec: float = 300.0
    min_importance: float = 1.1  # raised from 0.9 — borderline noise discards
    archive_retention_days: int = 14  # spill files older than this are deleted on flush
    batch_file: str = "sleep-flush-{date}.md"

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_age_sec, bool)
            or not isinstance(self.min_age_sec, int | float)
            or not math.isfinite(float(self.min_age_sec))
            or not 0 <= self.min_age_sec <= 365 * 86_400
        ):
            raise ValueError("min_age_sec must be a finite value between 0 and one year")
        if (
            isinstance(self.min_importance, bool)
            or not isinstance(self.min_importance, int | float)
            or not math.isfinite(float(self.min_importance))
            or not -10 <= self.min_importance <= 10
        ):
            raise ValueError("min_importance must be a finite value between -10 and 10")
        if (
            isinstance(self.archive_retention_days, bool)
            or not isinstance(self.archive_retention_days, int)
            or not 0 <= self.archive_retention_days <= 3_650
        ):
            raise ValueError("archive_retention_days must be between 0 and 3650")
        if (
            not isinstance(self.batch_file, str)
            or not self.batch_file
            or len(self.batch_file) > 255
            or Path(self.batch_file).name != self.batch_file
            or self.batch_file.replace("{date}", "").find("{") >= 0
            or self.batch_file.replace("{date}", "").find("}") >= 0
            or not self.batch_file.endswith(".md")
        ):
            raise ValueError("batch_file must be a markdown basename with optional {date}")
        self.sleep_dir = self.memory_root / "sleep"
        self.archive_dir = self.sleep_dir / "archive"
        self.semantic_dir = self.memory_root / "semantic"
        self.procedural_dir = self.memory_root / "procedural"
        self.intel_dir = self.memory_root / "intel"

    def _eligible_spills(self) -> list[Path]:
        if not self.sleep_dir.exists():
            return []
        now = time.time()
        out = []
        for p in sorted(self.sleep_dir.glob("spill-*.jsonl")):
            file_stat = p.lstat()
            if not stat.S_ISREG(file_stat.st_mode):
                continue
            age = now - file_stat.st_mtime
            if age >= self.min_age_sec:
                out.append(p)
        return out

    def flush(self, *, dry_run: bool = False) -> FlushResult:
        """Flush eligible spills under the cross-process maintenance lock."""
        with memory_process_lock(self.memory_root):
            return self._flush_unlocked(dry_run=dry_run)

    def _flush_unlocked(self, *, dry_run: bool = False) -> FlushResult:
        res = FlushResult()
        existing = {
            "semantic": _load_existing_entity_ids(self.semantic_dir),
            "procedural": _load_existing_entity_ids(self.procedural_dir),
            "intel": _load_existing_entity_ids(self.intel_dir),
        }
        # Collect across all spills first so we can dedup cross-file too
        all_candidates: list[Candidate] = []
        spills = self._eligible_spills()
        for s in spills:
            all_candidates.extend(extract_candidates(s))
        res.spills_processed = len(spills)
        res.candidates_seen = len(all_candidates)

        # Group by route; dedup within-run + against canonical
        kept: dict[str, list[Candidate]] = {"semantic": [], "procedural": [], "intel": []}
        run_seen: set[str] = set()
        for c in all_candidates:
            if c.route == "discard" or c.importance < self.min_importance:
                res.written["discard"] += 1
                continue
            if c.entity_id in existing[c.route] or c.entity_id in run_seen:
                res.written["dup"] += 1
                continue
            run_seen.add(c.entity_id)
            kept[c.route].append(c)

        if dry_run:
            for route, cs in kept.items():
                res.written[route] = len(cs)
            return res

        # Write each route to a dated flush-out.md so we don't thrash
        # existing canonical files; the user (or a later step) can merge.
        from datetime import datetime

        ts = datetime.now(UTC).strftime("%Y-%m-%d")
        output_name = self.batch_file.replace("{date}", ts)
        for route, cs in kept.items():
            if not cs:
                continue
            target_dir = getattr(self, f"{route}_dir")
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / output_name
            cs.sort(key=lambda c: -c.importance)
            lines: list[str] = []
            try:
                is_empty = out_path.lstat().st_size == 0
            except FileNotFoundError:
                is_empty = True
            if is_empty:
                lines.append(f"# sleep-flush {ts}\n")
            for candidate in cs:
                weight_tag = ""
                if candidate.importance >= 1.3 and "|W" not in candidate.text:
                    weight_tag = "|W4"
                elif candidate.importance >= 1.0 and "|W" not in candidate.text:
                    weight_tag = "|W3"
                lines.append(f"{candidate.text}{weight_tag}")
            append_bounded_text(
                out_path,
                "\n".join(lines).rstrip() + "\n",
                max_bytes=_MAX_CANONICAL_OUTPUT_BYTES,
                durable=True,
                mode=0o600,
            )
            res.written[route] += len(cs)

        # Archive consumed spill files
        if spills:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            for s in spills:
                md = s.with_suffix(".md")
                try:
                    archive_regular_file(
                        md,
                        self.archive_dir,
                        max_bytes=_MAX_SPILL_BYTES,
                    )
                except FileNotFoundError:
                    pass
                archived = archive_regular_file(
                    s,
                    self.archive_dir,
                    max_bytes=_MAX_SPILL_BYTES,
                )
                if archived is None:
                    continue
                res.archived.append(s)

        # Prune archive: delete spill files older than archive_retention_days.
        # 14 days is plenty for post-mortems; longer == unbounded growth.
        self._prune_archive()
        return res

    def _prune_archive(self) -> int:
        if not self.archive_dir.exists():
            return 0
        cutoff = time.time() - (self.archive_retention_days * 86400)
        removed = 0
        for p in self.archive_dir.glob("spill-*"):
            try:
                file_stat = p.lstat()
                if stat.S_ISREG(file_stat.st_mode) and file_stat.st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
        if removed:
            log.info(
                "sleep_flush archive prune: removed %d files older than %dd",
                removed,
                self.archive_retention_days,
            )
        return removed
