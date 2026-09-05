"""Small skills directory for prompt-visible active context.

Always-on skills are intentionally tiny and listed every pass. Larger/task
procedures remain in memory/procedural/skills and are retrieved by context.

Learned skills (created by SkillLearner) are auto-discovered by scanning for
inject=learned in the SKILL header.
"""

from __future__ import annotations

import re as _re
import stat
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path

from ..atomic import read_bounded_text

_MAX_SKILL_BYTES = 262_144
_MAX_SKILL_BODY_CHARS = 16_384
_MAX_SKILL_FILES = 512
_MAX_SKILLS_PER_TURN = 16
_MAX_PROMPT_CHARS = 65_536
_MAX_TRIGGERS = 64
_MAX_TRIGGER_CHARS = 64
_MAX_SKILL_ID_CHARS = 128


@dataclass(frozen=True)
class _SkillRecord:
    skill_id: str
    scope: str
    inject: str
    body: str
    triggers: tuple[str, ...]
    priority: int


@lru_cache(maxsize=1_024)
def _parse_skill_revision(
    path_value: str,
    mtime_ns: int,
    ctime_ns: int,
    inode: int,
    size: int,
) -> _SkillRecord | None:
    del mtime_ns, ctime_ns, inode, size  # cache-key-only revision fields
    path = Path(path_value)
    try:
        raw = read_bounded_text(path, max_bytes=_MAX_SKILL_BYTES, encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return None
    lines = raw.splitlines()
    if not lines or not lines[0].strip().startswith("SKILL;"):
        return None
    fields: dict[str, str] = {}
    for part in lines[0][:4_096].split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()[:64]] = value.strip()[:256]
    skill_id = (fields.get("id") or path.stem).strip()[:_MAX_SKILL_ID_CHARS]
    if not skill_id or any(ord(char) < 32 for char in skill_id):
        return None
    inject = fields.get("inject", "retrieve").strip().lower()[:32]
    scope = fields.get("scope", "global").strip().lower()[:32] or "global"
    try:
        priority = max(-1_000, min(1_000, int(fields.get("priority", "0"))))
    except ValueError:
        priority = 0
    triggers: list[str] = []
    for line in lines[1:]:
        clean = line.strip()
        if not clean.startswith("TRIGGERS:"):
            continue
        for raw_trigger in clean.split(":", 1)[1].split(",")[:_MAX_TRIGGERS]:
            trigger = raw_trigger.strip().lower()[:_MAX_TRIGGER_CHARS]
            if trigger and trigger not in triggers:
                triggers.append(trigger)
        break
    return _SkillRecord(
        skill_id=skill_id,
        scope=scope,
        inject=inject,
        body="\n".join(lines[1:]).strip()[:_MAX_SKILL_BODY_CHARS],
        triggers=tuple(triggers),
        priority=priority,
    )


def _skill_record(path: Path) -> _SkillRecord | None:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError:
        return None
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > _MAX_SKILL_BYTES:
        return None
    return _parse_skill_revision(
        str(path),
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
        file_stat.st_ino,
        file_stat.st_size,
    )


def _skill_paths(skills_dir: Path, pattern: str) -> list[Path]:
    try:
        paths = list(islice(skills_dir.glob(pattern), _MAX_SKILL_FILES + 1))
    except OSError:
        return []
    return sorted(paths[:_MAX_SKILL_FILES])


def _ranked_entries(
    entries: list[tuple[int, tuple[str, str, str]]],
) -> list[tuple[str, str, str]]:
    ranked = sorted(entries, key=lambda item: (-item[0], item[1][0]))
    result: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for _priority, entry in ranked:
        if entry[0] in seen:
            continue
        seen.add(entry[0])
        result.append(entry)
        if len(result) >= _MAX_SKILLS_PER_TURN:
            break
    return result


def _parse_skill(path: Path) -> tuple[str, str, str] | None:
    record = _skill_record(path)
    if record is None:
        return None
    return (record.skill_id, record.scope, record.inject)


def _parse_skill_full(path: Path) -> tuple[str, str, str, str] | None:
    """Parse a skill file and return (id, scope, inject, body).

    body is the full file content (minus the SKILL header line) — used by
    retrieve_skills_by_trigger() to inject complete skill procedures into
    the prompt when trigger keywords match.
    """
    record = _skill_record(path)
    if record is None:
        return None
    return (record.skill_id, record.scope, record.inject, record.body)


def _parse_triggers(path: Path) -> list[str]:
    """Extract TRIGGERS from a skill file's first content line.

    TRIGGERS:browser,headed,headless,site,login,cookies,session,surf,interact
    → ["browser", "headed", "headless", "site", "login", "cookies", ...]
    """
    record = _skill_record(path)
    return list(record.triggers) if record is not None else []


def retrieve_skills_by_trigger(
    user_prompt: str,
    *,
    memory_root: Path | None = None,
    max_skills: int = 3,
) -> list[tuple[str, str, str, str]]:
    """Find retrieval-type skills whose TRIGGERS match the user prompt.

    Returns (id, scope, inject, body) tuples for skills with inject=retrieval
    that have trigger keywords present in the user prompt. This is the wiring
    that connects retrieval-type skills (browser.control, staging.sync,
    twitter.engagement, etc.) to the SKILLS block when the task matches.

    Matching uses word-boundary regex to avoid false positives (e.g. "site"
    matching inside "parasite"). Short triggers (<=4 chars) require exact
    word-boundary match; longer triggers use substring match.
    """
    root = memory_root or Path(__file__).resolve().parents[2] / "memory"
    skills_dir = root / "procedural" / "skills"
    if isinstance(max_skills, bool) or not isinstance(max_skills, int):
        raise TypeError("max_skills must be an integer")
    max_skills = max(0, min(_MAX_SKILLS_PER_TURN, max_skills))
    text = (user_prompt or "")[:_MAX_PROMPT_CHARS].lower()
    if not text:
        return []

    matches: list[tuple[int, tuple[str, str, str, str]]] = []
    for path in _skill_paths(skills_dir, "retrievable.*.md"):
        record = _skill_record(path)
        if record is None:
            continue
        sid, scope, inject = record.skill_id, record.scope, record.inject
        if inject != "retrieval":
            continue  # only retrieval-type skills are trigger-matched
        triggers = record.triggers
        if not triggers:
            continue
        hit_count = 0
        for trigger in triggers:
            if len(trigger) <= 4:
                if _re.search(r"\b" + _re.escape(trigger) + r"\b", text):
                    hit_count += 1
            else:
                if trigger in text:
                    hit_count += 1
        if hit_count == 0:
            continue
        full = (sid, scope, inject, record.body)
        # Score by number of trigger hits (more hits = more relevant)
        matches.append((hit_count, full))

    # Sort by hit count descending, then by name
    matches.sort(key=lambda x: (-x[0], x[1][0]))
    return [entry for _, entry in matches[:max_skills]]


def active_skills(memory_root: Path | None = None) -> list[tuple[str, str, str]]:
    """Returns all active (always-on + learned) skills for prompt injection."""
    root = memory_root or Path(__file__).resolve().parents[2] / "memory"
    skills_dir = root / "procedural" / "skills"
    entries: list[tuple[int, tuple[str, str, str]]] = []

    # Scan always_on.*.md files (inject=always)
    for path in _skill_paths(skills_dir, "always_on.*.md"):
        record = _skill_record(path)
        if record is None:
            continue
        parsed = (record.skill_id, record.scope, record.inject)
        entries.append((record.priority, parsed))

    # Scan retrievable.*.md files (inject=learned — auto-generated skills)
    for path in _skill_paths(skills_dir, "retrievable.*.md"):
        record = _skill_record(path)
        if record is None:
            continue
        sid, scope, inject = record.skill_id, record.scope, record.inject
        if inject not in ("learned", "always"):
            continue
        entries.append((record.priority, (sid, scope, inject)))

    return _ranked_entries(entries)


def learned_skills(memory_root: Path | None = None) -> list[str]:
    """Return names of auto-learned skills (for focus suggestions)."""
    root = memory_root or Path(__file__).resolve().parents[2] / "memory"
    skills_dir = root / "procedural" / "skills"
    names = []
    for path in _skill_paths(skills_dir, "retrievable.*.md"):
        parsed = _parse_skill(path)
        if parsed is None:
            continue
        sid, _, inject = parsed
        if inject == "learned":
            names.append(sid)
    return names


# Keyword sets for task-skill matching
_TASK_KEYWORDS: dict[str, set[str]] = {
    "coding": {
        "code",
        "edit",
        "write",
        "fix",
        "bug",
        "patch",
        "implement",
        "refactor",
        "file",
        "function",
        "class",
        "test",
        "pytest",
        "debug",
        "error",
        "traceback",
        "repo",
        "build",
        "compile",
        "deploy",
        "install",
        "package",
        "dependency",
    },
    "research": {
        "search",
        "research",
        "compare",
        "web",
        "find",
        "look up",
        "investigate",
        "analyze",
        "benchmark",
        "study",
        "explore",
        "discover",
    },
    "browser": {
        "browser",
        "login",
        "website",
        "site",
        "surf",
        "web automation",
        "headed",
        "headless",
        "cookies",
        "session",
        "cdp",
        "playwright",
    },
    "staging": {"staging", "clone", "sync", "deploy", "mirror", "remote", "ssh", "rsync"},
    "memory": {
        "memory",
        "recall",
        "remember",
        "forget",
        "consolidate",
        "sleep",
        "entity",
        "graph",
        "retrieval",
        "semantic",
        "procedural",
    },
    "ops": {
        "exec",
        "shell",
        "command",
        "run",
        "system",
        "process",
        "service",
        "systemd",
        "docker",
        "nginx",
        "config",
        "server",
        "restart",
        "status",
    },
    "twitter": {"twitter", "tweet", "post", "social", "engagement", "reply"},
}

# Skill name → task category mapping (derived from skill file TRIGGERS)
_SKILL_TASK_MAP: dict[str, set[str]] = {
    "edit-exec": {"coding", "ops"},
    "exec-edit": {"coding", "ops"},
    "exec-exec": {"ops"},
    "exec-read": {"coding", "ops", "research"},
    "read-exec": {"coding", "ops"},
    "read-read": {"coding", "research"},
    "list_dir-read": {"coding", "research"},
    "write-and-verify": {"coding"},
    "explore-and-read": {"coding", "research"},
    "build-and-test": {"coding", "ops"},
    "recall-and-review": {"memory", "research"},
    "exec-error-recovery": {"coding", "ops"},
    "browser.control": {"browser"},
    "staging.sync": {"staging", "ops"},
    "twitter.engagement": {"twitter"},
    "web_search-web_search": {"research"},
    "web_search-web_fetch": {"research"},
    "web_fetch-web_search": {"research"},
    "code.change": {"coding"},
    "context.efficiency": {"coding", "ops"},
    "gateway.providers": {"ops", "coding"},
    "goal-drift-awareness": {"coding", "ops"},
    "progress-stall-awareness": {"coding", "ops"},
    "response-gate-awareness": {"coding", "ops"},
    "learning-log-schema": {"memory"},
}

# Skills that should always be injected regardless of task match
_ALWAYS_INJECT = {"exec-error-recovery", "goal-drift-awareness", "progress-stall-awareness"}


def _classify_task_categories(user_prompt: str, task_type: str = "") -> set[str]:
    """Classify the current task into categories for skill matching.

    Uses word-boundary matching for short keywords (<=6 chars) to avoid
    false positives like 'system' matching inside 'ecosystem'.
    """
    import re as _re

    text = (user_prompt or "")[:_MAX_PROMPT_CHARS].lower()
    cats: set[str] = set()
    if task_type:
        cats.add(task_type)
    for cat, keywords in _TASK_KEYWORDS.items():
        for kw in keywords:
            if len(kw) <= 6:
                # Word-boundary match for short keywords
                if _re.search(r"\b" + _re.escape(kw) + r"\b", text):
                    cats.add(cat)
                    break
            else:
                # Substring match for longer keywords (phrases, multi-word)
                if kw in text:
                    cats.add(cat)
                    break
    return cats


def _skill_relevance(skill_name: str, task_cats: set[str]) -> float:
    """Score how relevant a skill is to the current task (0.0-1.0)."""
    # Strip any prefix (retrievable., always_on.) and normalize
    bare = skill_name
    for prefix in ("retrievable.", "always_on."):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    # Check both bare and full name for always-inject
    if bare in _ALWAYS_INJECT or skill_name in _ALWAYS_INJECT:
        return 1.0
    # Look up in skill map (try bare, then full, then with retrievable. prefix)
    skill_cats = (
        _SKILL_TASK_MAP.get(bare)
        or _SKILL_TASK_MAP.get(skill_name)
        or _SKILL_TASK_MAP.get(f"retrievable.{bare}")
        or set()
    )
    if not skill_cats:
        return 0.3  # unknown skill — low default
    overlap = skill_cats & task_cats
    if not overlap:
        return 0.0
    # Score by overlap ratio
    return len(overlap) / max(len(skill_cats), 1)


def active_skills_filtered(
    memory_root: Path | None = None,
    *,
    user_prompt: str = "",
    task_type: str = "",
    min_relevance: float = 0.15,
) -> list[tuple[str, str, str]]:
    """Return skills filtered by relevance to the current task.

    Always-on skills are always included. Learned skills are scored against
    the task and only included if they pass the relevance threshold.
    """
    root = memory_root or Path(__file__).resolve().parents[2] / "memory"
    skills_dir = root / "procedural" / "skills"
    task_cats = _classify_task_categories(user_prompt, task_type)
    entries: list[tuple[int, tuple[str, str, str]]] = []

    # Always-on skills — always included
    for path in _skill_paths(skills_dir, "always_on.*.md"):
        record = _skill_record(path)
        if record is None:
            continue
        parsed = (record.skill_id, record.scope, record.inject)
        entries.append((record.priority, parsed))

    # Learned skills — filtered by relevance
    # Match the same inject types as active_skills(): learned + always only.
    # retrieval-type skills are retrieved on-demand by the memory layer, not
    # injected into the prompt.
    for path in _skill_paths(skills_dir, "retrievable.*.md"):
        record = _skill_record(path)
        if record is None:
            continue
        sid, scope, inject = record.skill_id, record.scope, record.inject
        if inject not in ("learned", "always"):
            continue
        relevance = _skill_relevance(sid, task_cats)
        if relevance < min_relevance:
            continue
        # Boost priority by relevance so highly relevant skills sort first
        boosted_priority = record.priority + int(relevance * 10)
        entries.append((boosted_priority, (sid, scope, inject)))

    return _ranked_entries(entries)
