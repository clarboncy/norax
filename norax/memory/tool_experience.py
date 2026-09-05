"""Dedicated just-in-time memory for reliable tool use.

This memory is intentionally separate from broad conversational retrieval.  It
serves a small set of high-signal, verified tool-use lessons immediately before
planning and after a tool failure.  Successful trajectories are retained as
local episodic evidence; failed outcomes are retained for analytics but are
never presented as positive demonstrations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("norax.memory.tool_experience")

_TOKEN_RE = re.compile(r"[a-z0-9_./-]{2,}", re.IGNORECASE)
_SAFE_CODE_RE = re.compile(r"[^a-z0-9_.:-]+")
_MAX_EVENT_BYTES = 2_000_000

# Portable minimum when a deployment has not installed the curated dataset.
_BUILTIN: tuple[dict[str, Any], ...] = (
    {
        "id": "inspect-before-mutate",
        "intent": "Change existing state without acting on stale assumptions.",
        "tools": ["read", "list_dir", "repo_explore", "edit", "write", "exec"],
        "triggers": ["change", "edit", "fix", "update", "implement", "file"],
        "procedure": [
            "Locate the authoritative target.",
            "Inspect its current state before mutation.",
            "Apply one coherent minimal change.",
            "Verify the resulting state with an independent read or test.",
        ],
        "avoid": ["Guessing exact file contents", "Claiming success before verification"],
        "verification": "Read back changed state and run the narrowest meaningful check.",
        "phases": ["pre", "recovery"],
        "confidence": 1.0,
    },
    {
        "id": "failure-change-approach",
        "intent": "Recover from repeated tool failure without an echo loop.",
        "tools": ["read", "exec", "web_fetch", "web_search", "browser"],
        "triggers": ["error", "failed", "timeout", "loop", "not_found", "unavailable"],
        "procedure": [
            "Classify the concrete failure from the returned evidence.",
            "Correct arguments once when the cause is clear.",
            "After repetition, switch tool or approach rather than replaying the call.",
        ],
        "avoid": ["Repeating an identical failed call", "Replacing a real search with echo output"],
        "verification": "Require a novel observation before considering recovery successful.",
        "phases": ["recovery"],
        "confidence": 1.0,
    },
)


def _tokens(value: str | Iterable[str]) -> set[str]:
    text = value if isinstance(value, str) else " ".join(str(v) for v in value)
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


# Lightweight synonym expansion for tool-use vocabulary.
# Boosts lexical matching without requiring an embedding model.
_SYNONYMS: dict[str, frozenset[str]] = {
    "edit": frozenset({"modify", "change", "update", "patch", "fix", "alter"}),
    "write": frozenset({"create", "save", "store", "output", "produce"}),
    "read": frozenset({"inspect", "view", "check", "examine", "load", "fetch"}),
    "exec": frozenset({"run", "execute", "shell", "command", "bash"}),
    "search": frozenset({"find", "lookup", "query", "locate", "seek"}),
    "verify": frozenset({"confirm", "check", "validate", "test", "prove"}),
    "fail": frozenset({"error", "crash", "timeout", "break", "exception"}),
    "file": frozenset({"document", "path", "script", "code", "source"}),
    "memory": frozenset({"recall", "remember", "retrieve", "knowledge"}),
    "plan": frozenset({"strategy", "approach", "design", "outline", "prepare"}),
    "debug": frozenset({"troubleshoot", "diagnose", "trace", "investigate"}),
    "deploy": frozenset({"ship", "release", "publish", "launch", "rollout"}),
    "test": frozenset({"verify", "validate", "check", "assert", "confirm"}),
}
_SYNONYM_REVERSE: dict[str, str] = {}
for _canon, _syns in _SYNONYMS.items():
    for _syn in _syns:
        _SYNONYM_REVERSE.setdefault(_syn, _canon)
    _SYNONYM_REVERSE.setdefault(_canon, _canon)


def _expand_tokens(tokens: set[str]) -> set[str]:
    """Add canonical synonyms for each token to broaden matching."""
    expanded = set(tokens)
    for tok in tokens:
        canon = _SYNONYM_REVERSE.get(tok)
        if canon:
            expanded.add(canon)
            expanded |= _SYNONYMS.get(canon, frozenset())
    return expanded


def _safe_error(result: Any) -> str:
    if isinstance(result, dict) and result.get("ok") is True:
        return ""
    if not isinstance(result, dict):
        return "tool_error"
    raw = result.get("error") or result.get("error_type") or result.get("stderr") or "tool_error"
    if isinstance(raw, dict):
        raw = raw.get("code") or raw.get("type") or "tool_error"
    code = _SAFE_CODE_RE.sub("_", str(raw).strip().lower()).strip("_")
    return (code or "tool_error")[:80]


def _arg_shape(args: Any) -> list[str]:
    """Retain schema shape, never values (which may contain credentials)."""
    if not isinstance(args, dict):
        return []
    return sorted(str(key)[:64] for key in args)[:24]


@dataclass(frozen=True)
class ToolLesson:
    lesson_id: str
    intent: str
    tools: tuple[str, ...]
    triggers: tuple[str, ...]
    procedure: tuple[str, ...]
    avoid: tuple[str, ...]
    verification: str
    phases: tuple[str, ...]
    confidence: float = 1.0
    evidence_count: int = 1
    learned: bool = False

    @classmethod
    def from_seed(cls, row: dict[str, Any]) -> ToolLesson | None:
        lesson_id = str(row.get("id", "")).strip()[:96]
        intent = str(row.get("intent", "")).strip()[:320]
        procedure = tuple(str(x).strip()[:240] for x in row.get("procedure", []) if str(x).strip())
        if not lesson_id or not intent or not procedure:
            return None
        return cls(
            lesson_id=lesson_id,
            intent=intent,
            tools=tuple(str(x).strip()[:64] for x in row.get("tools", []) if str(x).strip()),
            triggers=tuple(str(x).strip()[:80] for x in row.get("triggers", []) if str(x).strip()),
            procedure=procedure[:8],
            avoid=tuple(str(x).strip()[:180] for x in row.get("avoid", []) if str(x).strip())[:6],
            verification=str(row.get("verification", "")).strip()[:280],
            phases=tuple(str(x).strip() for x in row.get("phases", ["pre"]))[:3],
            confidence=max(0.0, min(1.0, float(row.get("confidence", 1.0)))),
            evidence_count=max(1, int(row.get("evidence_count", 1))),
        )


class ToolExperienceMemory:
    """Retrieve curated wins and learn from evidence-backed tool trajectories."""

    def __init__(self, memory_root: Path, *, seed_path: Path | None = None) -> None:
        self.root = Path(memory_root)
        runtime_seed = self.root / "procedural" / "tool-use-wins.jsonl"
        packaged_seed = Path(__file__).with_name("data") / "tool-use-wins.jsonl"
        # A deployment-local corpus may override the versioned baseline. Clean
        # installs still receive the complete curated playbook dataset.
        self.seed_path = seed_path or (runtime_seed if runtime_seed.exists() else packaged_seed)
        self.events_path = self.root / "episodic" / "tool-experience.jsonl"
        self._lock = threading.RLock()
        self._seed_mtime_ns = -1
        self._seed_lessons: list[ToolLesson] = []
        self._events_mtime_ns = -1
        self._learned_lessons: list[ToolLesson] = []

    @staticmethod
    def _read_jsonl(path: Path, *, tail_bytes: int | None = None) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with path.open("rb") as handle:
                if tail_bytes and path.stat().st_size > tail_bytes:
                    handle.seek(-tail_bytes, os.SEEK_END)
                    handle.readline()  # discard a potentially partial first row
                data = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        rows: list[dict[str, Any]] = []
        for line in data.splitlines():
            try:
                row = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _load_seed(self) -> list[ToolLesson]:
        try:
            mtime = self.seed_path.stat().st_mtime_ns
        except OSError:
            mtime = -1
        if mtime == self._seed_mtime_ns and self._seed_lessons:
            return self._seed_lessons
        lessons = [
            ToolLesson.from_seed(row) for row in (*_BUILTIN, *self._read_jsonl(self.seed_path))
        ]
        self._seed_lessons = [lesson for lesson in lessons if lesson is not None]
        self._seed_mtime_ns = mtime
        return self._seed_lessons

    @staticmethod
    def _event_to_lesson(row: dict[str, Any]) -> ToolLesson | None:
        # Losses inform metrics, never positive in-context demonstrations.
        if not row.get("verified") or not row.get("sequence"):
            return None
        sequence = tuple(str(x)[:64] for x in row.get("sequence", []) if str(x))[:20]
        failures = row.get("failures") or []
        avoid = tuple(
            f"Do not repeat {str(item.get('tool', 'tool'))[:64]} after {str(item.get('error', 'error'))[:80]}"
            for item in failures
            if isinstance(item, dict)
        )[:4]
        sig = str(row.get("signature", "learned"))[:16]
        return ToolLesson(
            lesson_id=f"verified-trajectory-{sig}",
            intent="Reuse a similar locally verified tool sequence.",
            tools=tuple(dict.fromkeys(sequence)),
            triggers=tuple(str(x)[:80] for x in row.get("task_terms", []) if str(x))[:24],
            procedure=tuple(f"Call {name}" for name in sequence),
            avoid=avoid,
            verification="Finish with an observation that independently verifies the requested outcome.",
            phases=("pre", "recovery"),
            confidence=0.8,
            evidence_count=max(1, int(row.get("evidence_count", 1))),
            learned=True,
        )

    def _load_learned(self) -> list[ToolLesson]:
        try:
            mtime = self.events_path.stat().st_mtime_ns
        except OSError:
            mtime = -1
        if mtime == self._events_mtime_ns:
            return self._learned_lessons
        rows = self._read_jsonl(self.events_path, tail_bytes=_MAX_EVENT_BYTES)
        # Collapse equivalent trajectories; repeated verified wins increase rank.
        grouped: dict[str, dict[str, Any]] = {}
        # Track AVOID rows separately for recovery-phase lessons.
        avoid_grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            signature = str(row.get("signature", ""))
            if not signature:
                continue
            if not row.get("verified"):
                # Unverified rows from replay_avoid are failure-recovery
                # lessons, not positive demonstrations.  Load them for
                # the recovery phase only.
                if row.get("source") == "replay_avoid":
                    if signature not in avoid_grouped:
                        avoid_grouped[signature] = dict(row)
                        avoid_grouped[signature]["evidence_count"] = 1
                    else:
                        avoid_grouped[signature]["evidence_count"] += 1
                continue
            if signature not in grouped:
                grouped[signature] = dict(row)
                grouped[signature]["evidence_count"] = 1
            else:
                grouped[signature]["evidence_count"] += 1
        lessons = [self._event_to_lesson(row) for row in grouped.values()]
        lessons.extend(self._avoid_to_lesson(row) for row in avoid_grouped.values())
        self._learned_lessons = [lesson for lesson in lessons if lesson is not None]
        self._events_mtime_ns = mtime
        return self._learned_lessons

    @staticmethod
    def _avoid_to_lesson(row: dict[str, Any]) -> ToolLesson | None:
        """Convert a replay_avoid event into a recovery-phase lesson.

        These lessons are never positive demonstrations — they surface
        known failure modes and their recovery tools so that when a tool
        call fails with a familiar error, the agent gets actionable
        guidance instead of starting from scratch.
        """
        failures = row.get("failures") or []
        if not failures:
            return None
        fail = failures[0] if isinstance(failures[0], dict) else {}
        tool_name = str(fail.get("tool", "tool"))[:64]
        err_type = str(fail.get("error", "error"))[:80]
        recover_with = row.get("recover_with") or []
        recover_tools = tuple(str(t)[:64] for t in recover_with if str(t))[:4]
        task_terms = row.get("task_terms") or []
        triggers = tuple(str(x)[:80] for x in task_terms if str(x))[:24]
        # Include the error type as a trigger so retrieval matches on
        # the error string itself during recovery.
        triggers = (*triggers, err_type, tool_name)
        procedure = tuple(f"Retry with {t}" for t in recover_tools) or (
            f"Avoid {tool_name} when error is {err_type}",
        )
        return ToolLesson(
            lesson_id=f"replay-avoid-{tool_name}-{err_type}"[:48],
            intent=f"Recover from {tool_name} failure: {err_type}.",
            tools=(*recover_tools, tool_name) if recover_tools else (tool_name,),
            triggers=triggers,
            procedure=procedure,
            avoid=(f"Do not retry {tool_name} with the same arguments after {err_type}",),
            verification="Confirm the recovery tool succeeded before proceeding.",
            phases=("recovery",),
            confidence=0.7,
            evidence_count=max(1, int(row.get("evidence_count", 1))),
            learned=True,
        )

    @staticmethod
    def _score(
        lesson: ToolLesson,
        query_tokens: set[str],
        tool_set: set[str],
        phase: str,
    ) -> float:
        if phase not in lesson.phases:
            return -1.0
        lesson_tokens = _expand_tokens(_tokens((lesson.intent, *lesson.triggers, *lesson.tools)))
        overlap = len(query_tokens & lesson_tokens)
        lexical = overlap / math.sqrt(max(1, len(query_tokens)) * max(1, len(lesson_tokens)))
        tool_overlap = len(tool_set & set(lesson.tools)) / max(1, len(tool_set))
        evidence = min(0.35, math.log2(lesson.evidence_count + 1) * 0.08)
        baseline = 0.08 if not lesson.triggers else 0.0
        return lexical * 3.0 + tool_overlap * 1.6 + evidence + baseline + lesson.confidence * 0.15

    def retrieve(
        self,
        task: str,
        tools: Iterable[str],
        *,
        phase: str = "pre",
        failed_tool: str = "",
        error: str = "",
        limit: int = 4,
    ) -> list[ToolLesson]:
        tool_set = {str(tool) for tool in tools if str(tool)}
        if failed_tool:
            tool_set.add(failed_tool)
        query_tokens = _expand_tokens(_tokens((task, failed_tool, error, *tool_set)))
        with self._lock:
            candidates = [*self._load_seed(), *self._load_learned()]
        ranked = sorted(
            ((self._score(lesson, query_tokens, tool_set, phase), lesson) for lesson in candidates),
            key=lambda pair: (pair[0], pair[1].evidence_count, pair[1].lesson_id),
            reverse=True,
        )
        floor = 0.18 if phase == "pre" else 0.08
        return [lesson for score, lesson in ranked if score >= floor][: max(0, limit)]

    def render(
        self,
        task: str,
        tools: Iterable[str],
        *,
        phase: str = "pre",
        failed_tool: str = "",
        error: str = "",
        limit: int = 4,
        max_chars: int = 8000,
    ) -> str:
        lessons = self.retrieve(
            task,
            tools,
            phase=phase,
            failed_tool=failed_tool,
            error=error,
            limit=limit,
        )
        if not lessons:
            return ""
        heading = (
            "TOOL_EXPERIENCE_MEMORY: Relevant verified playbooks. Apply only when they fit "
            "the current evidence; current tool results override memory."
        )
        parts = [heading]
        for lesson in lessons:
            flow = " -> ".join(lesson.procedure)
            item = f"- {lesson.lesson_id}: {lesson.intent} FLOW: {flow}."
            if lesson.avoid:
                item += f" AVOID: {'; '.join(lesson.avoid)}."
            if lesson.verification:
                item += f" VERIFY: {lesson.verification}"
            if sum(len(part) + 1 for part in parts) + len(item) > max_chars:
                break
            parts.append(item)
        return "\n".join(parts) if len(parts) > 1 else ""

    def ingest_replay_patterns(self, procedural_dir: Path) -> int:
        """Ingest replay pattern files from sleep consolidation as learned lessons.

        Replay writes PATTERN: and AVOID: lines to procedural/replay-patterns-{date}.md
        and replay-avoid-{date}.md.  This method parses those files and appends
        synthetic tool-experience events so that future retrieve() calls can
        surface high-frequency verified sequences and failure recovery guidance.

        Returns the number of synthetic events appended.
        """
        if not procedural_dir.exists():
            return 0
        appended = 0
        # Process the most recent pattern file and avoid file
        pattern_files = sorted(procedural_dir.glob("replay-patterns-*.md"), reverse=True)
        avoid_files = sorted(procedural_dir.glob("replay-avoid-*.md"), reverse=True)

        rows: list[dict[str, Any]] = []

        for pf in pattern_files[:1]:  # only most recent
            try:
                text = pf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("PATTERN:"):
                    continue
                # PATTERN:id=hash|tool_seq=a→b→c|count=N|success_rate=X%|context=words|Wn
                parts = line[len("PATTERN:") :].split("|")
                meta: dict[str, Any] = {}
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        meta[k] = v
                seq = meta.get("tool_seq", "")
                if not seq or "→" not in seq:
                    continue
                tools = tuple(t.strip() for t in seq.split("→") if t.strip())
                if not tools:
                    continue
                count = int(meta.get("count", "1")) if meta.get("count", "").isdigit() else 1
                sr_str = meta.get("success_rate", "0%").rstrip("%")
                try:
                    success_rate = float(sr_str) / 100.0
                except ValueError:
                    success_rate = 0.0
                context = meta.get("context", "")
                seq_hash = meta.get("id", hashlib.sha256(seq.encode()).hexdigest()[:8])
                row = {
                    "schema": "norax.tool_experience.v1",
                    "ts": int(time.time()),
                    "signature": f"replay_{seq_hash}",
                    "verified": success_rate >= 0.8,
                    "task_terms": [w for w in context.split(",") if w][:10],
                    "sequence": list(tools),
                    "arg_shapes": {},
                    "failures": [],
                    "evidence_count": count,
                    "source": "replay_pattern",
                }
                rows.append(row)

        for af in avoid_files[:1]:  # only most recent
            try:
                text = af.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("AVOID:tool="):
                    continue
                # AVOID:tool=name|error=type|count=N|after=tool:c|context=words|recover_with=tool|Wn
                body = line[len("AVOID:tool=") :]
                parts = body.split("|")
                avoid_meta: dict[str, str] = {"tool": parts[0]}
                for p in parts[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        avoid_meta[k] = v
                tool_name = avoid_meta["tool"]
                err_type = avoid_meta.get("error", "tool_error")
                count = (
                    int(avoid_meta.get("count", "1"))
                    if avoid_meta.get("count", "").isdigit()
                    else 1
                )
                context = avoid_meta.get("context", "")
                recovery = avoid_meta.get("recover_with", "")
                # Build a synthetic failure event so _load_learned can surface it
                # as a recovery lesson.  We store it with verified=False so it
                # won't be promoted as a positive demonstration, but the
                # failure data is available for recovery-phase retrieval.
                row = {
                    "schema": "norax.tool_experience.v1",
                    "ts": int(time.time()),
                    "signature": f"replay_avoid_{tool_name}_{err_type}",
                    "verified": False,
                    "task_terms": [w for w in context.split(",") if w][:10],
                    "sequence": [tool_name],
                    "arg_shapes": {},
                    "failures": [{"tool": tool_name, "error": err_type}],
                    "evidence_count": count,
                    "source": "replay_avoid",
                    "recover_with": [w for w in recovery.split(",") if w][:3],
                }
                rows.append(row)

        if not rows:
            return 0
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                existing_signatures = {
                    str(event.get("signature", ""))
                    for event in self._read_jsonl(self.events_path)
                    if event.get("signature")
                }
                with self.events_path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        signature = str(row.get("signature", ""))
                        if not signature or signature in existing_signatures:
                            continue
                        payload = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        handle.write(payload)
                        existing_signatures.add(signature)
                        appended += 1
                    handle.flush()
                    os.fsync(handle.fileno())
            if appended:
                self._events_mtime_ns = -1  # force reload
                log.info("tool_experience: ingested %d replay patterns", appended)
        except OSError as e:
            log.warning("tool_experience.ingest_replay_patterns: %r", e)
        return appended

    def record_outcome(self, task: str, trace: list[dict[str, Any]], *, verified: bool) -> bool:
        """Append a privacy-conscious trajectory event; return whether it was written."""
        if not trace:
            return False
        sequence: list[str] = []
        failures: list[dict[str, str]] = []
        arg_shapes: dict[str, list[str]] = {}
        for entry in trace[:80]:
            name = str(entry.get("name", "unknown"))[:64]
            sequence.append(name)
            shape = _arg_shape(entry.get("args"))
            if shape:
                arg_shapes[name] = shape
            error = _safe_error(entry.get("result"))
            if error:
                failures.append({"tool": name, "error": error})
        task_terms = sorted(_tokens(task))[:40]
        signature_raw = json.dumps(
            {"terms": task_terms, "sequence": sequence, "failures": failures},
            sort_keys=True,
            separators=(",", ":"),
        )
        signature = hashlib.sha256(signature_raw.encode()).hexdigest()[:16]
        row = {
            "schema": "norax.tool_experience.v1",
            "ts": int(time.time()),
            "signature": signature,
            "verified": bool(verified),
            "task_terms": task_terms,
            "sequence": sequence,
            "arg_shapes": arg_shapes,
            "failures": failures,
        }
        payload = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
            self._events_mtime_ns = -1
            return True
        except OSError:
            return False
