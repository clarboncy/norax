"""Hot memory — always-on scratchpad/focus injection + immediate persistence.

Fixes the "forgets simple things" failure mode:
  - Hot neurons only surfaced when query tokens overlap (FastContext miss)
  - post_action skipped turns with no tool trace
  - scratchpad header/state lines stale (Gen5 identity from April)
  - directives detected by thalamus but never written to canonical memory
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from pathlib import Path

from ..atomic import atomic_write_text, path_lock, read_bounded_text
from .store import MemoryStore, Neuron

log = logging.getLogger("norax.memory.hot_inject")

_MAX_SCRATCHPAD_BYTES = 16 * 1024 * 1024
_MAX_FOCUS_BYTES = 256 * 1024
_MAX_FLASHBULB_BYTES = 16 * 1024 * 1024
_MAX_DIRECTIVE_CHARS = 240
_MAX_FOCUS_CHARS = 220
_MAX_PINNED_STATE_LINES = 64


def _one_line(value: object, *, max_chars: int) -> str:
    """Normalize prompt-facing state without allowing record injection."""
    return " ".join(str(value or "").split())[:max_chars]


def _bounded_positive_int(value: object, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


_STATE_PREFIXES = (
    "IDENTITY:",
    "CONTEXT:",
    "STATUS:",
    "FEATURES:",
    "FOCUS:",
    "GEN:",
    "MEMORY:",
    "VERIFY:",
    "CURRENT:",
    "NEXT:",
    "PORTS:",
    "PHASE:",
    "RECENT_WINS:",
    "SLEEP_CONSOLIDATION:",
    "STAGING:",
)
_TURN_RX = re.compile(r"^TURN:\d{2}:\d{2}\|")
_DIRECTIVE_RX = re.compile(
    r"\b(always|never|from now on|going forward|remember that|"
    r"priority|important|don't forget|do not forget)\b",
    re.I,
)


def is_state_line(text: str) -> bool:
    return any(text.startswith(p) for p in _STATE_PREFIXES)


def pin_hot_neurons(store: MemoryStore, *, max_turns: int = 10) -> list[tuple[Neuron, float, str]]:
    """Return hot scratchpad/focus neurons that must always appear in MEMORY."""
    max_turns = _bounded_positive_int(max_turns, name="max_turns", maximum=100)
    pinned: list[tuple[Neuron, float, str]] = []
    turns: deque[Neuron] = deque(maxlen=max_turns)

    for n in store.hot:
        if is_state_line(n.text) and len(pinned) < _MAX_PINNED_STATE_LINES:
            pinned.append((n, 1.35, "hot"))
        elif _TURN_RX.match(n.text):
            turns.append(n)

    for n in turns:
        pinned.append((n, 1.05, "hot+turn"))

    # Dedup by entity_id, keep highest score
    best: dict[str, tuple[Neuron, float, str]] = {}
    for n, s, tag in pinned:
        cur = best.get(n.entity_id)
        if cur is None or s > cur[1]:
            best[n.entity_id] = (n, s, tag)
    out = sorted(best.values(), key=lambda x: -x[1])
    return out


def merge_pinned(
    hits: list[tuple[Neuron, float, str]],
    pinned: list[tuple[Neuron, float, str]],
    *,
    k: int,
) -> list[tuple[Neuron, float, str]]:
    """Prepend pinned hot neurons, then fill remaining slots from fused hits.

    Pinned neurons are capped at floor(25%) of k so search results always
    have room. Without this cap, 18+ hot lines (state + FOCUS + TURN) can
    crowd out all search-hits, causing the \"forgets simple things\" bug.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        return []
    k = min(k, 10_000)
    max_pinned = max(2, k // 4)  # floor 25%, minimum 2 for identity + focus
    merged: dict[str, tuple[Neuron, float, str]] = {}
    for n, s, tag in pinned[:max_pinned]:
        merged[n.entity_id] = (n, s, tag)
    for n, s, tag in hits:
        if n.entity_id not in merged:
            merged[n.entity_id] = (n, s, tag)
    ranked = sorted(merged.values(), key=lambda x: -x[1])
    return ranked[:k]


def compact_scratchpad(
    memory_root: Path,
    *,
    max_turns: int = 96,
    max_lines: int = 240,
) -> bool:
    """Trim old TURN lines; keep state lines + recent turns."""
    max_turns = _bounded_positive_int(max_turns, name="max_turns", maximum=10_000)
    max_lines = _bounded_positive_int(max_lines, name="max_lines", maximum=20_000)
    path = memory_root / "scratchpad.md"
    with path_lock(path):
        try:
            lines = read_bounded_text(
                path,
                max_bytes=_MAX_SCRATCHPAD_BYTES,
                errors="replace",
            ).splitlines()
        except FileNotFoundError:
            return False
        if len(lines) <= max_lines:
            return False

        header = lines[0] if lines else f"SCRATCHPAD;updated={_now_tag()};type=hot_memory"
        state: list[str] = []
        turns: list[str] = []
        other: list[str] = []

        for line in lines[1:]:
            s = line.strip()
            if not s:
                continue
            if is_state_line(s):
                state.append(s)
            elif _TURN_RX.match(s):
                turns.append(s)
            else:
                other.append(s)

        kept_turns = turns[-max_turns:]
        new_lines = [header] + state + other[-8:] + kept_turns
        atomic_write_text(path, "\n".join(new_lines) + "\n", mode=0o600)
    log.info("scratchpad.compact: %d → %d lines", len(lines), len(new_lines))
    return True


def refresh_hot_identity(
    memory_root: Path,
    *,
    generation: int = 7,
    role: str = "production",
) -> None:
    """Refresh static identity and configured-provider facts in scratchpad.

    Environment configuration is not a health probe.  The generated status
    therefore records only whether a provider/endpoint was configured and
    leaves health explicitly unverified.
    """
    generation = _bounded_positive_int(generation, name="generation", maximum=1_000)
    path = memory_root / "scratchpad.md"
    ts = _now_tag()
    import socket

    host = _one_line(socket.gethostname() or "localhost", max_chars=255)
    # Detect active provider from environment to avoid stale references
    import os

    configured_provider = _one_line(
        os.environ.get("NORAX_DEFAULT_PROVIDER", ""),
        max_chars=128,
    )
    provider_label = configured_provider or "unspecified"
    endpoint_configured = bool(os.environ.get("NORAX_PROVIDER_URL", "").strip())
    role = _one_line(role, max_chars=64) or "production"
    state_lines = [
        f"SCRATCHPAD;updated={ts};type=hot_memory",
        f"IDENTITY:Norax — {role} agent on {host}",
        f"GEN:{generation}",
        "CONTEXT:Norax self-hosted agent runtime; multi-signal memory + entity graph + configurable model routing",
        f"STATUS:provider_config={provider_label};endpoint_configured={str(endpoint_configured).lower()};health=unverified",
        "FEATURES:entity linking + sleep consolidation + configurable model routing|W5",
    ]
    if role == "staging":
        state_lines.append(
            "STAGING:independent memory at ~/norax/memory; sync code from production|W5"
        )

    with path_lock(path):
        try:
            old = read_bounded_text(
                path,
                max_bytes=_MAX_SCRATCHPAD_BYTES,
                errors="replace",
            ).splitlines()
        except FileNotFoundError:
            atomic_write_text(path, "\n".join(state_lines) + "\n", mode=0o600)
            return
        kept: list[str] = []
        for line in old[1:]:
            s = line.strip()
            if not s:
                continue
            if is_state_line(s):
                continue
            kept.append(s)

        atomic_write_text(path, "\n".join(state_lines + kept) + "\n", mode=0o600)
    log.info("scratchpad.identity_refreshed gen=%d role=%s", generation, role)


def update_active_focus(memory_root: Path, summary: str, *, max_lines: int = 6) -> None:
    """Persist current turn focus so the next turn always recalls what we're doing.

    The new summary replaces any stale ``CURRENT:`` line and clears ``NEXT:``
    (the next step is re-derived on the following turn).  Prior ``FOCUS:``
    lines are retained (deduplicated, capped) for short-term context.
    """
    max_lines = _bounded_positive_int(max_lines, name="max_lines", maximum=100)
    text = _one_line(summary, max_chars=_MAX_FOCUS_CHARS)
    if not text:
        return
    path = memory_root / "active-focus.md"
    ts = _now_tag()
    new_current = f"CURRENT:{text}"
    new_focus = f"FOCUS:{text}"
    with path_lock(path):
        existing_focus: list[str] = []
        try:
            existing = read_bounded_text(
                path,
                max_bytes=_MAX_FOCUS_BYTES,
                errors="replace",
            )
        except FileNotFoundError:
            existing = ""
        if existing:
            for line in existing.splitlines():
                s = line.strip()
                if s.startswith("FOCUS:") and s != new_focus:
                    existing_focus.append(s)
                # Drop stale CURRENT: and NEXT: — they will be replaced/cleared.

        # Dedup recent focus lines, keep last few
        deduped: list[str] = []
        seen: set[str] = set()
        for line in reversed([new_focus] + existing_focus):
            if line in seen:
                continue
            seen.add(line)
            deduped.append(line)
            if len(deduped) >= max_lines:
                break
        deduped.reverse()

        body = "\n".join([f"# active-focus;updated={ts}"] + deduped + [new_current]) + "\n"
        atomic_write_text(path, body, mode=0o600)


def capture_directive(
    memory_root: Path,
    body: str,
    *,
    msg_type: str = "",
    pathway: str = "",
) -> Path | None:
    """Write owner directives to semantic memory immediately (W5)."""
    text = _one_line(body, max_chars=_MAX_DIRECTIVE_CHARS)
    if not text or len(text) < 8:
        return None
    if pathway != "procedural_update" and msg_type != "directive":
        if not _DIRECTIVE_RX.search(text):
            return None

    day = time.strftime("%Y-%m-%d")
    target = memory_root / "semantic" / f"flashbulb-{day}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%H:%M")
    line = f"DIRECTIVE:{ts}|{text}|W5"
    with path_lock(target):
        try:
            existing = read_bounded_text(
                target,
                max_bytes=_MAX_FLASHBULB_BYTES,
                errors="replace",
            )
        except FileNotFoundError:
            existing = ""
        if line in existing.splitlines():
            return None
        if existing:
            updated = existing.rstrip("\n") + "\n" + line + "\n"
        else:
            updated = f"# flashbulb events {day}\n{line}\n"
        atomic_write_text(target, updated, mode=0o600)
    log.info("directive.captured: %s", line[:80])
    return target


def post_turn_hot_maintenance(
    memory_root: Path,
    *,
    focus_summary: str = "",
    user_body: str = "",
    msg_type: str = "",
    pathway: str = "",
    turn_count: int = 0,
    role: str = "production",
) -> None:
    """Run after every turn: focus, directives, periodic scratchpad compact."""
    update_active_focus(memory_root, focus_summary)
    capture_directive(memory_root, user_body, msg_type=msg_type, pathway=pathway)
    if turn_count == 1:
        refresh_hot_identity(memory_root, generation=7, role=role)
    if turn_count > 0 and turn_count % 15 == 0:
        compact_scratchpad(memory_root)


def _now_tag() -> str:
    return time.strftime("%Y-%m-%dT%H:%M", time.localtime())
