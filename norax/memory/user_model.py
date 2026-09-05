"""User model — persistent theory-of-mind tracking per user.

Tracks preferences, knowledge state, communication style, interaction
patterns, and recent topics for each user. Persisted to memory/semantic/
and cached in-memory (LRU). Injected into prompt as USER_MODEL block.

This is a lightweight, privacy-first implementation:
  - All data stays on-disk at ~/norax/memory/
  - No external API calls
  - User-visible only via prompt injection
  - Per-user profiles are plain Markdown files
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text, read_bounded_text

log = logging.getLogger("norax.memory.user_model")

_MAX_USER_ID_CHARS = 512
_MAX_LABEL_CHARS = 256
_MAX_PROFILE_LIST_ITEMS = 100
_MAX_PROFILE_ITEM_CHARS = 512
_MAX_PROMPT_CHARS = 4_096
_MAX_PROFILE_BYTES = 1_048_576


def _user_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("user_id must be a string")
    value = value.strip()
    if not value or len(value) > _MAX_USER_ID_CHARS:
        raise ValueError(f"user_id must be non-empty and at most {_MAX_USER_ID_CHARS} characters")
    return value


def _bounded_text(value: Any, *, limit: int = _MAX_PROFILE_ITEM_CHARS) -> str:
    text = str(value or "")
    text = " ".join(text.replace("\x00", "").split())
    return text[:limit]


def _string_list(value: Any, *, limit: int = _MAX_PROFILE_LIST_ITEMS) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:limit]:
        clean = _bounded_text(item)
        if clean:
            result.append(clean)
    return result


def _count_map(value: Any, *, key_limit: int = 128) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, count in list(value.items())[:_MAX_PROFILE_LIST_ITEMS]:
        clean_key = _bounded_text(key, limit=key_limit)
        if clean_key and isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            result[clean_key] = count
    return result


def _finite_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


@dataclass(slots=True)
class UserPreferences:
    """Derived preferences from interaction history."""

    conciseness: str = "balanced"  # concise | balanced | verbose
    formality: str = "casual"  # casual | neutral | formal
    technical_depth: str = "advanced"  # basic | intermediate | advanced
    response_style: str = "direct"  # direct | explanatory | conversational
    favorite_tools: list[str] = field(default_factory=list)
    favorite_models: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UserKnowledge:
    """What the user knows (inferred from interactions)."""

    topics: list[str] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)
    skill_level: str = "expert"  # novice | intermediate | expert
    known_facts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InteractionStats:
    """Aggregate interaction patterns."""

    total_turns: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    avg_message_length: float = 0.0
    commands_used: dict[str, int] = field(default_factory=dict)
    tools_requested: dict[str, int] = field(default_factory=dict)
    active_hours: dict[int, int] = field(default_factory=dict)  # hour -> count
    turn_times: list[float] = field(default_factory=list)


@dataclass(slots=True)
class UserProfile:
    """Complete user profile for one user."""

    user_id: str
    label: str
    tier: str = "user"
    preferences: UserPreferences = field(default_factory=UserPreferences)
    knowledge: UserKnowledge = field(default_factory=UserKnowledge)
    stats: InteractionStats = field(default_factory=InteractionStats)
    recent_topics: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _dirty: bool = False


class UserModel:
    """Persistent user profiles with LRU in-memory cache."""

    def __init__(self, root: Path, *, max_cache: int = 100):
        if isinstance(max_cache, bool) or not isinstance(max_cache, int) or max_cache < 1:
            raise ValueError("max_cache must be a positive integer")
        self.root = Path(root)
        self.profiles_dir = self.root / "semantic" / "users"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache = max_cache
        self._cache: OrderedDict[str, UserProfile] = OrderedDict()
        self._disabled: set[str] = set()  # users who opted out

    # ------------------------------------------------------------------
    # Profile I/O
    # ------------------------------------------------------------------

    def _profile_path(self, user_id: str) -> Path:
        user_id = _user_id(user_id)
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        if safe == user_id and len(safe) <= 128:
            filename_key = safe
        else:
            prefix = safe[:48].strip("_") or "user"
            digest = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            filename_key = f"{prefix}_{digest}"
        return self.profiles_dir / f"user_{filename_key}.json"

    def load(self, user_id: str) -> UserProfile | None:
        """Load a user profile from cache or disk."""
        user_id = _user_id(user_id)
        if user_id in self._disabled:
            return None

        # Check cache
        if user_id in self._cache:
            self._cache.move_to_end(user_id)
            return self._cache[user_id]

        # Load from disk
        path = self._profile_path(user_id)
        try:
            data = json.loads(
                read_bounded_text(path, max_bytes=_MAX_PROFILE_BYTES, encoding="utf-8")
            )
            if not isinstance(data, dict):
                raise ValueError("profile root must be an object")
            stored_id = data.get("user_id")
            if stored_id is not None and stored_id != user_id:
                raise ValueError("stored profile identity does not match requested user")
            profile = self._deserialize(user_id, data)
            self._cache[user_id] = profile
            self._evict_if_needed()
            return profile
        except FileNotFoundError:
            return None
        except Exception as e:
            log.warning("user_model.load_failed: %s: %r", user_id, e)
            return None

    def save(self, profile: UserProfile) -> None:
        """Persist a user profile to disk."""
        profile.user_id = _user_id(profile.user_id)
        if profile.user_id in self._disabled:
            return

        try:
            self._write_profile(profile)
            profile._dirty = False
            self._cache[profile.user_id] = profile
            self._cache.move_to_end(profile.user_id)
            self._evict_if_needed()
        except Exception as e:
            log.warning("user_model.save_failed: %s: %r", profile.user_id, e)

    def _write_profile(self, profile: UserProfile) -> None:
        """Persist without mutating the cache (used by eviction and flush)."""
        path = self._profile_path(profile.user_id)
        data = self._serialize(profile)
        atomic_write_text(
            path,
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            mode=0o600,
        )

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_cache:
            _oldest_id, oldest_profile = self._cache.popitem(last=False)
            if oldest_profile._dirty and oldest_profile.user_id not in self._disabled:
                try:
                    self._write_profile(oldest_profile)
                    oldest_profile._dirty = False
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "user_model.evict_save_failed: %s: %r",
                        oldest_profile.user_id,
                        exc,
                    )

    def flush(self) -> int:
        """Persist every dirty cached profile without changing LRU ordering."""
        saved = 0
        for profile in list(self._cache.values()):
            if not profile._dirty or profile.user_id in self._disabled:
                continue
            try:
                self._write_profile(profile)
                profile._dirty = False
                saved += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("user_model.flush_failed: %s: %r", profile.user_id, exc)
        return saved

    def _serialize(self, p: UserProfile) -> dict[str, Any]:
        return {
            "user_id": p.user_id,
            "label": _bounded_text(p.label, limit=_MAX_LABEL_CHARS),
            "tier": _bounded_text(p.tier, limit=32),
            "preferences": {
                "conciseness": _bounded_text(p.preferences.conciseness, limit=32),
                "formality": _bounded_text(p.preferences.formality, limit=32),
                "technical_depth": _bounded_text(p.preferences.technical_depth, limit=32),
                "response_style": _bounded_text(p.preferences.response_style, limit=32),
                "favorite_tools": _string_list(p.preferences.favorite_tools, limit=10),
                "favorite_models": _string_list(p.preferences.favorite_models, limit=10),
            },
            "knowledge": {
                "topics": _string_list(p.knowledge.topics, limit=50),
                "systems": _string_list(p.knowledge.systems, limit=50),
                "skill_level": _bounded_text(p.knowledge.skill_level, limit=32),
                "known_facts": _string_list(p.knowledge.known_facts),
            },
            "stats": {
                "total_turns": max(0, int(p.stats.total_turns)),
                "first_seen": _finite_number(p.stats.first_seen),
                "last_seen": _finite_number(p.stats.last_seen),
                "avg_message_length": max(0.0, _finite_number(p.stats.avg_message_length)),
                "commands_used": _count_map(p.stats.commands_used),
                "tools_requested": _count_map(p.stats.tools_requested),
                "active_hours": {
                    str(hour): count
                    for hour, count in p.stats.active_hours.items()
                    if isinstance(hour, int)
                    and not isinstance(hour, bool)
                    and 0 <= hour <= 23
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                },
            },
            "recent_topics": _string_list(p.recent_topics, limit=20),
            "notes": _string_list(p.notes, limit=50),
        }

    def _deserialize(self, user_id: str, data: dict[str, Any]) -> UserProfile:
        prefs_data = data.get("preferences", {})
        knowledge_data = data.get("knowledge", {})
        stats_data = data.get("stats", {})
        if not isinstance(prefs_data, dict):
            prefs_data = {}
        if not isinstance(knowledge_data, dict):
            knowledge_data = {}
        if not isinstance(stats_data, dict):
            stats_data = {}
        active_hours_raw = stats_data.get("active_hours", {})
        active_hours: dict[int, int] = {}
        if isinstance(active_hours_raw, dict):
            for raw_hour, raw_count in list(active_hours_raw.items())[:24]:
                try:
                    hour = int(raw_hour)
                except (TypeError, ValueError):
                    continue
                if (
                    0 <= hour <= 23
                    and isinstance(raw_count, int)
                    and not isinstance(raw_count, bool)
                    and raw_count >= 0
                ):
                    active_hours[hour] = raw_count

        total_turns = stats_data.get("total_turns", 0)
        if isinstance(total_turns, bool) or not isinstance(total_turns, int):
            total_turns = 0

        return UserProfile(
            user_id=user_id,
            label=_bounded_text(data.get("label") or user_id, limit=_MAX_LABEL_CHARS),
            tier=_bounded_text(data.get("tier") or "user", limit=32),
            preferences=UserPreferences(
                conciseness=_bounded_text(prefs_data.get("conciseness") or "balanced", limit=32),
                formality=_bounded_text(prefs_data.get("formality") or "casual", limit=32),
                technical_depth=_bounded_text(
                    prefs_data.get("technical_depth") or "advanced", limit=32
                ),
                response_style=_bounded_text(
                    prefs_data.get("response_style") or "direct", limit=32
                ),
                favorite_tools=_string_list(prefs_data.get("favorite_tools"), limit=10),
                favorite_models=_string_list(prefs_data.get("favorite_models"), limit=10),
            ),
            knowledge=UserKnowledge(
                topics=_string_list(knowledge_data.get("topics"), limit=50),
                systems=_string_list(knowledge_data.get("systems"), limit=50),
                skill_level=_bounded_text(knowledge_data.get("skill_level") or "expert", limit=32),
                known_facts=_string_list(knowledge_data.get("known_facts")),
            ),
            stats=InteractionStats(
                total_turns=max(0, total_turns),
                first_seen=_finite_number(stats_data.get("first_seen")),
                last_seen=_finite_number(stats_data.get("last_seen")),
                avg_message_length=max(0.0, _finite_number(stats_data.get("avg_message_length"))),
                commands_used=_count_map(stats_data.get("commands_used")),
                tools_requested=_count_map(stats_data.get("tools_requested")),
                active_hours=active_hours,
            ),
            recent_topics=_string_list(data.get("recent_topics"), limit=20),
            notes=_string_list(data.get("notes"), limit=50),
        )

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def get_or_create(self, user_id: str, label: str = "", tier: str = "user") -> UserProfile:
        """Get existing profile or create a new one.

        Identity labels (real names like 'Alice') are never overwritten by
        event source labels (like 'cron', 'discord'). A label is only set
        on first creation or if the existing label is the raw user_id.
        """
        user_id = _user_id(user_id)
        label = _bounded_text(label, limit=_MAX_LABEL_CHARS)
        tier = _bounded_text(tier or "user", limit=32)
        profile = self.load(user_id)
        if profile is not None:
            # Only update label if current label is unset or is the raw user_id
            # (i.e., not a real identity label). This prevents cron/discord
            # source labels from overwriting the owner's identity.
            if label and label != profile.label and profile.label in ("", user_id):
                profile.label = label
                profile._dirty = True
            if tier and tier != profile.tier:
                profile.tier = tier
                profile._dirty = True
            return profile

        profile = UserProfile(
            user_id=user_id,
            label=label or user_id,
            tier=tier,
            stats=InteractionStats(
                first_seen=time.time(),
                last_seen=time.time(),
            ),
        )
        profile._dirty = True
        self._cache[user_id] = profile
        self._evict_if_needed()
        return profile

    def disable(self, user_id: str) -> None:
        """Opt a user out of tracking. Deletes stored profile."""
        user_id = _user_id(user_id)
        self._disabled.add(user_id)
        self._cache.pop(user_id, None)
        path = self._profile_path(user_id)
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            log.warning("user_model.opt_out_failed user=%s error=%r", user_id, e)

    def enable(self, user_id: str) -> None:
        """Re-enable tracking for a user."""
        user_id = _user_id(user_id)
        self._disabled.discard(user_id)

    def user_count(self) -> int:
        return len(list(self.profiles_dir.glob("user_*.json")))

    # ------------------------------------------------------------------
    # Learning (called post-turn)
    # ------------------------------------------------------------------

    def record_turn(
        self,
        user_id: str,
        *,
        label: str = "",
        tier: str = "user",
        body: str = "",
        response_preview: str = "",
        tool_calls: list[str] | None = None,
        model: str = "",
        command_name: str = "",
    ) -> UserProfile | None:
        """Record a turn and update the user profile."""
        user_id = _user_id(user_id)
        body = str(body or "")
        response_preview = str(response_preview or "")
        if user_id in self._disabled:
            return None

        profile = self.get_or_create(user_id, label=label, tier=tier)
        if profile is None:
            return None

        s = profile.stats
        s.total_turns += 1
        s.last_seen = time.time()
        now = time.localtime()
        s.active_hours[now.tm_hour] = s.active_hours.get(now.tm_hour, 0) + 1

        # Message length
        msg_len = min(len(body or ""), 1_000_000)
        if s.total_turns > 1:
            s.avg_message_length = (
                s.avg_message_length * (s.total_turns - 1) + msg_len
            ) / s.total_turns
        else:
            s.avg_message_length = float(msg_len)

        # Tools requested
        if tool_calls:
            for raw_tool in tool_calls[:100]:
                tool = _bounded_text(raw_tool, limit=128)
                if tool:
                    s.tools_requested[tool] = s.tools_requested.get(tool, 0) + 1
            profile.preferences.favorite_tools = [
                name
                for name, _count in sorted(
                    s.tools_requested.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:10]
            ]

        # Models used
        model = _bounded_text(model, limit=128)
        if model and model not in profile.preferences.favorite_models:
            profile.preferences.favorite_models.append(model)
            if len(profile.preferences.favorite_models) > 10:
                profile.preferences.favorite_models = profile.preferences.favorite_models[-10:]

        # Commands
        if command_name:
            command_name = _bounded_text(command_name, limit=128)
            if command_name:
                s.commands_used[command_name] = s.commands_used.get(command_name, 0) + 1

        # Recent topics (simple keyword extraction)
        bounded_body = (body or "")[:16_384]
        topics = self._extract_topics(bounded_body)
        for topic in topics:
            if topic not in profile.recent_topics:
                profile.recent_topics.append(topic)
            if topic not in profile.knowledge.topics:
                profile.knowledge.topics.append(topic)

        # Trim topic lists
        if len(profile.recent_topics) > 20:
            profile.recent_topics = profile.recent_topics[-20:]
        if len(profile.knowledge.topics) > 50:
            profile.knowledge.topics = profile.knowledge.topics[-50:]

        # Learn preferences
        self._learn_preferences(profile, bounded_body, response_preview[:4_096])

        profile._dirty = True
        # Owners persist each turn. Other profiles persist on creation, every
        # ten turns, eviction, or shutdown to avoid synchronous hot-path I/O.
        if tier == "owner" or s.total_turns == 1 or s.total_turns % 10 == 0:
            self.save(profile)

        return profile

    def _extract_topics(self, text: str) -> list[str]:
        """Extract likely topics from user text."""
        if not text:
            return []
        # Common technical terms / domain keywords
        signals = [
            "agent",
            "memory",
            "model",
            "pipeline",
            "harness",
            "tool",
            "skill",
            "user",
            "profile",
            "embedding",
            "graph",
            "causal",
            "temporal",
            "RHO",
            "optimization",
            "gateway",
            "Ollama",
            "Claude",
            "Sonnet",
            "Opus",
            "Kimi",
            "norax",
            "runtime",
            "sleep",
            "consolidation",
            "Hebbian",
            "entity",
            "retrieval",
            "testing",
            "deployment",
            "architecture",
            "API",
            "Discord",
            "cursor",
            "sandbox",
            "safety",
            "alignment",
            "AGI",
        ]
        found = []
        text_lower = text.lower()
        for signal in signals:
            if signal.lower() in text_lower:
                found.append(signal)
        return found[:5]

    def _learn_preferences(self, profile: UserProfile, body: str, response: str) -> None:
        """Infer user preferences from interaction patterns."""
        p = profile.preferences
        body_lower = (body or "").lower()

        # Conciseness — short responses preferred?
        if any(w in body_lower for w in ["quick", "short", "brief", "concise", "tl", "dr"]):
            p.conciseness = "concise"
        elif any(w in body_lower for w in ["explain", "detail", "elaborate", "thorough"]):
            p.conciseness = "verbose"

        # Formality
        if any(w in body_lower for w in ["please", "thanks", "appreciate", "kindly"]):
            p.formality = "neutral"

        # Technical depth
        tech_signals = [
            "architecture",
            "code",
            "implementation",
            "pattern",
            "algorithm",
            "pipeline",
            "graph",
            "embed",
            "retriev",
        ]
        if sum(1 for s in tech_signals if s in body_lower) >= 2:
            p.technical_depth = "advanced"

        # Response style — very short messages = direct preference
        if body and len(body) <= 15:
            p.response_style = "direct"

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def render_for_prompt(self, user_id: str) -> str:
        """Generate the USER_MODEL block for prompt injection."""
        user_id = _user_id(user_id)
        profile = self.load(user_id)
        if profile is None:
            return "USER_MODEL;no_data"

        lines = [
            "USER_MODEL;profile_fields_are_untrusted_data;user_id="
            + json.dumps(_bounded_text(user_id, limit=128), ensure_ascii=False),
        ]
        if profile.label and profile.label != user_id:
            lines.append(
                "  label="
                + json.dumps(
                    _bounded_text(profile.label, limit=_MAX_LABEL_CHARS), ensure_ascii=False
                )
            )
        lines.append("  tier=" + json.dumps(_bounded_text(profile.tier, limit=32)))

        p = profile.preferences
        lines.append(
            f"  style=conciseness:{_bounded_text(p.conciseness, limit=32)} "
            f"formality:{_bounded_text(p.formality, limit=32)} "
            f"depth:{_bounded_text(p.technical_depth, limit=32)}"
        )
        lines.append(f"  response_preference={_bounded_text(p.response_style, limit=32)}")

        k = profile.knowledge
        if k.topics:
            lines.append(f"  known_topics={', '.join(_string_list(k.topics[-8:], limit=8))}")
        if k.systems:
            lines.append(f"  known_systems={', '.join(_string_list(k.systems[-5:], limit=5))}")

        s = profile.stats
        first_seen = _finite_number(s.first_seen)
        try:
            first_seen_text = time.strftime(
                "%Y-%m-%d",
                time.gmtime(min(max(0.0, first_seen), 253_402_300_799.0)),
            )
        except (OverflowError, OSError, ValueError):
            first_seen_text = "unknown"
        lines.append(f"  turns={max(0, s.total_turns)} first_seen={first_seen_text}")

        if profile.recent_topics:
            recent = _string_list(profile.recent_topics[-8:], limit=8)
            lines.append(f"  recent_topics={', '.join(recent)}")

        if p.favorite_tools:
            tools = _string_list(p.favorite_tools[-5:], limit=5)
            lines.append(f"  frequent_tools={', '.join(tools)}")

        if profile.notes:
            lines.append(f"  notes={len(profile.notes)} stored")

        return "\n".join(lines)[:_MAX_PROMPT_CHARS]

    def add_note(self, user_id: str, note: str) -> bool:
        """Add a freeform note about a user."""
        user_id = _user_id(user_id)
        profile = self.load(user_id)
        if profile is None:
            return False
        clean_note = _bounded_text(note, limit=2_000)
        if not clean_note:
            return False
        profile.notes.append(clean_note)
        if len(profile.notes) > 50:
            profile.notes = profile.notes[-50:]
        profile._dirty = True
        self.save(profile)
        return True
