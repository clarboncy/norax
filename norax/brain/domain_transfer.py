"""Domain Transfer — apply what works in one domain to another.

Watches SelfModel's per-domain success stats plus the procedural/semantic
memory kinds, and surfaces *transferable tactics*: patterns with strong
track records in a source domain whose structural signature matches the
current (possibly weak) target domain.

Examples it catches automatically:
  - "verify before claiming success" learned in `coding` transfers to
    `research` (claims need sources) and `ops` (changes need health checks)
  - "narrow the failing case first" from `debugging` transfers to
    `optimization` (measure before tuning)

Zero LLM calls. Pure keyword/signature overlap + success statistics.

Usage:
    dt = DomainTransfer(self_model, memory_root=Path("~/norax/memory"))
    hints = dt.transfer_hints(target_domain="research", user_text="...")
    # -> ["TRANSFER[coding→research]: verify-before-claim ..."]
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("norax.brain.domain_transfer")

MIN_SOURCE_SUCCESS = 0.75  # only transfer from domains we're actually good at
MIN_SOURCE_SAMPLES = 5
MAX_HINTS = 3
CACHE_TTL = 3600  # rebuild tactic library at most hourly

# Structural signatures: abstract problem shapes that recur across domains.
_SIGNATURE_KEYWORDS: dict[str, set[str]] = {
    "verify": {"verify", "confirm", "check", "validate", "test", "assert", "prove"},
    "decompose": {"split", "break", "decompose", "subtask", "step", "chunk", "stage"},
    "measure_first": {"measure", "benchmark", "profile", "baseline", "metric", "observe"},
    "rollback": {"backup", "snapshot", "checkpoint", "revert", "restore", "undo"},
    "iterate": {"retry", "iterate", "refine", "loop", "again", "improve"},
    "isolate": {"isolate", "narrow", "reproduce", "minimal", "bisect", "pinpoint"},
}


def _signatures_of(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return {sig for sig, kws in _SIGNATURE_KEYWORDS.items() if words & kws}


# Lines must look like a stated practice, not a log fragment or path dump.
_PRACTICE_RE = re.compile(
    r"(verify|confirm|check|test|read|measure|backup|snapshot|retry|iterate|"
    r"isolate|narrow|decompose|split|break|reproduce|revert|restore|undo|"
    r"profile|baseline|validate|assert|first|before|after|always|never)\b",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(r"(/home/|/[a-z]+/|:\d{2}:\d{2}|\.md|\.py|https?://|\|\||&&)")

# Log-style prefixes that indicate a captured transcript, not a practice.
_LOG_PREFIX_RE = re.compile(
    r"^(VERIFY|TOOL_SUMMARY|LOOP|ERROR|WARN|INFO|DEBUG|TRACE|AUDIT|CHECK|RESULT)"
    r"\s*[:|]",
    re.IGNORECASE,
)
# Strip markdown emphasis so the practice check sees the real sentence.
_MD_RE = re.compile(r"[*_`>]+")


# Code/shell heavy lines are not transferable practices.
_CODE_RE = re.compile(r"[(){}\[\]\"'`;=<>\\]|--[a-z]|\.add_argument|^\$|#!/")
_DOMAIN_TAG_RE = re.compile(
    r"(?:\[\s*domain\s*[:=]\s*([a-z0-9_-]+)\s*\]|\bdomain\s*[:=]\s*([a-z0-9_-]+)\b)",
    re.IGNORECASE,
)


def _is_practice(line: str) -> bool:
    """True if the line states a general practice vs. a log/path/code fragment."""
    if _LOG_PREFIX_RE.match(line) or _NOISE_RE.search(line):
        return False
    if len(_CODE_RE.findall(line)) > 2:
        return False
    clean = _MD_RE.sub(" ", line)
    return bool(_PRACTICE_RE.search(clean))


@dataclass
class Tactic:
    """A transferable pattern mined from memory."""

    source_domain: str
    summary: str  # one-line tactic statement
    signatures: set[str] = field(default_factory=set)
    success_rate: float = 0.5
    samples: int = 0


class DomainTransfer:
    """Mines strong-domain tactics and maps them onto the current domain."""

    def __init__(self, self_model: Any = None, memory_root: Path | str | None = None):
        self.self_model = self_model
        self.memory_root = Path(memory_root).expanduser() if memory_root else None
        self._library: list[Tactic] = []
        self._library_built_at: float = 0.0

    # ------------------------------------------------------------- library
    def _domain_success(self, domain: str) -> tuple[float, int]:
        if self.self_model is None:
            return 0.0, 0
        try:
            stats = self.self_model.profile.domain_stats.get(domain)
            if stats is None:
                return 0.0, 0
            # Production SelfModel uses success_weight/failure_weight/total_count;
            # synthetic/test models may expose success_rate/sample_size directly.
            rate = getattr(stats, "success_rate", None)
            n = getattr(stats, "sample_size", None)
            if rate is None and isinstance(stats, dict):
                rate = stats.get("success_rate")
                n = stats.get("sample_size")
            if rate is None:

                def stat_value(key: str, default: float | int = 0.0) -> Any:
                    if isinstance(stats, dict):
                        return stats.get(key, default)
                    return getattr(stats, key, default)

                sw = float(stat_value("success_weight", 0.0) or 0.0)
                fw = float(stat_value("failure_weight", 0.0) or 0.0)
                rate = sw / (sw + fw) if (sw + fw) > 0 else 0.0
                n = n if n is not None else stat_value("total_count", 0)
            return float(rate), int(n or 0)
        except Exception:  # noqa: BLE001
            return 0.0, 0

    def _memory_files(self) -> list[Path]:
        """Procedural/semantic memory lives in subdirs in production; flat
        files (procedural_memory.md) in test fixtures. Read both."""
        if self.memory_root is None:
            return []
        out: list[Path] = []
        for flat in ("procedural_memory.md", "semantic_memory.md"):
            p = self.memory_root / flat
            if p.exists():
                out.append(p)
        for sub in ("procedural", "semantic"):
            d = self.memory_root / sub
            if d.is_dir():
                out.extend(sorted(d.glob("*.md")))
        return out

    def build_library(self) -> int:
        """Mine procedural memory for tactics from strong domains.

        Tactics come from the procedural memory file: lines that state a
        practice ("verify before...", "read before edit...") tagged with a
        domain, from domains with success_rate >= MIN_SOURCE_SUCCESS.
        """
        now = time.time()
        if now - self._library_built_at < CACHE_TTL and self._library:
            return len(self._library)

        tactics: list[Tactic] = []
        for path in self._memory_files():
            try:
                for line in path.read_text(errors="replace").splitlines():
                    line = line.strip().lstrip("-*# \u203a").strip()
                    if len(line) < 25 or len(line) > 220:
                        continue
                    sigs = _signatures_of(line)
                    if not sigs or not _is_practice(line):
                        continue
                    # Provenance must be explicit. Assigning every tactic to
                    # the globally strongest domain fabricates transfer
                    # evidence unrelated to where the tactic was learned.
                    domain_match = _DOMAIN_TAG_RE.search(line)
                    source_domain = ""
                    if domain_match:
                        source_domain = (domain_match.group(1) or domain_match.group(2)).lower()
                        line = _DOMAIN_TAG_RE.sub("", line).strip(" -|:")
                    elif self.self_model is not None:
                        try:
                            known_domains = {
                                str(domain).lower()
                                for domain in self.self_model.profile.domain_stats
                            }
                            if path.stem.lower() in known_domains:
                                source_domain = path.stem.lower()
                        except Exception:  # noqa: BLE001
                            pass
                    if not source_domain:
                        continue
                    source_rate, source_n = self._domain_success(source_domain)
                    if source_n < MIN_SOURCE_SAMPLES or source_rate < MIN_SOURCE_SUCCESS:
                        continue
                    tactics.append(
                        Tactic(
                            source_domain=source_domain,
                            summary=line[:180],
                            signatures=sigs,
                            success_rate=source_rate,
                            samples=source_n,
                        )
                    )
            except Exception as e:  # noqa: BLE001
                log.debug("domain_transfer.mine_failed %s: %r", path, e)

        # Keep the strongest tactics only
        tactics.sort(key=lambda t: (t.success_rate, len(t.signatures)), reverse=True)
        self._library = tactics[:50]
        self._library_built_at = now
        log.info("domain_transfer library built: %d tactics", len(self._library))
        return len(self._library)

    # --------------------------------------------------------------- hints
    def transfer_hints(self, target_domain: str, user_text: str) -> list[str]:
        """Return up to MAX_HINTS transfer hints relevant to this request."""
        self.build_library()
        if not self._library:
            return []

        target_sigs = _signatures_of(user_text)
        if not target_sigs:
            return []

        target_rate, target_n = self._domain_success(target_domain)
        # Transfer matters most when the target domain is weak/unproven
        target_weak = target_n < MIN_SOURCE_SAMPLES or target_rate < 0.65

        scored: list[tuple[float, Tactic]] = []
        for t in self._library:
            if t.source_domain == target_domain:
                continue  # that's just normal practice, not transfer
            overlap = len(t.signatures & target_sigs)
            if overlap == 0:
                continue
            if t.samples >= MIN_SOURCE_SAMPLES and t.success_rate < MIN_SOURCE_SUCCESS:
                continue
            score = overlap * (1.0 + t.success_rate)
            if target_weak:
                score *= 1.5
            scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        hints = []
        seen: set[str] = set()
        for _, t in scored:
            key = t.summary[:60]
            if key in seen:
                continue
            seen.add(key)
            destination = target_domain or "current"
            hints.append(
                f"TRANSFER[{t.source_domain}→{destination}]: {t.summary} "
                f"(source success={t.success_rate:.2f})"
            )
            if len(hints) >= MAX_HINTS:
                break
        return hints

    def stats(self) -> dict:
        return {
            "library_size": len(self._library),
            "built_at": self._library_built_at,
            "source_domains": list({t.source_domain for t in self._library}),
        }
