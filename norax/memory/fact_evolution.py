"""Temporal Fact Evolution — track how facts change over time with provenance.

Each fact update records:
  - old_value, new_value, source_episode, timestamp
  - Query API: "what did I know about X on date Y?"
  - Contradiction detection across time

Architecture:
  - FactVersion: a single version of a fact
  - FactHistory: all versions of a fact, ordered by time
  - FactEvolutionTracker: manages fact histories

Usage:
    tracker = FactEvolutionTracker(memory_root)
    tracker.record("kimi_version", "2.6", "2.7", source="user_message", episode_id="ep_123")
    history = tracker.get_history("kimi_version")
    state = tracker.get_state_at("kimi_version", timestamp=1624000000)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("norax.memory.fact_evolution")


@dataclass
class FactVersion:
    """A single version of a fact."""

    fact_id: str
    value: str
    timestamp: float
    source: str = ""  # where the change came from
    episode_id: str = ""  # episodic buffer reference
    previous_value: str = ""  # what it was before
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FactHistory:
    """All versions of a fact, ordered by time."""

    fact_id: str
    versions: list[FactVersion] = field(default_factory=list)

    def current_value(self) -> str:
        """Get the latest value."""
        if not self.versions:
            return ""
        return self.versions[-1].value

    def value_at(self, timestamp: float) -> str:
        """Get the value that was current at the given timestamp."""
        result = ""
        for v in self.versions:
            if v.timestamp <= timestamp:
                result = v.value
            else:
                break
        return result

    def has_contradiction(self) -> bool:
        """Check if any version contradicts a later one."""
        values = [v.value for v in self.versions]
        return len(set(values)) > 1

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "versions": [v.to_dict() for v in self.versions],
            "current": self.current_value(),
            "contradiction": self.has_contradiction(),
        }


class FactEvolutionTracker:
    """Tracks fact evolution over time with provenance.

    Persists to memory/fact_evolution.jsonl (append-only log).
    """

    def __init__(self, memory_root: Path) -> None:
        self.memory_root = Path(memory_root)
        self.log_path = self.memory_root / "fact_evolution.jsonl"
        self._histories: dict[str, FactHistory] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load fact histories from the append-only log."""
        if self._loaded:
            return
        self._loaded = True
        if not self.log_path.exists():
            return
        try:
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                fv = FactVersion(
                    fact_id=entry["fact_id"],
                    value=entry["value"],
                    timestamp=entry["timestamp"],
                    source=entry.get("source", ""),
                    episode_id=entry.get("episode_id", ""),
                    previous_value=entry.get("previous_value", ""),
                    confidence=entry.get("confidence", 1.0),
                    metadata=entry.get("metadata", {}),
                )
                hist = self._histories.setdefault(fv.fact_id, FactHistory(fact_id=fv.fact_id))
                hist.versions.append(fv)
            # Sort versions by timestamp
            for hist in self._histories.values():
                hist.versions.sort(key=lambda v: v.timestamp)
        except Exception as exc:
            log.warning("fact_evolution.load failed: %s", exc)

    def record(
        self,
        fact_id: str,
        old_value: str,
        new_value: str,
        *,
        source: str = "",
        episode_id: str = "",
        confidence: float = 1.0,
        metadata: dict | None = None,
    ) -> FactVersion | None:
        """Record a fact change.

        If old_value matches the current value (or there's no current value),
        records the change. If old_value doesn't match, logs a contradiction.

        Returns the FactVersion if recorded, None if no change.
        """
        self._ensure_loaded()

        if old_value == new_value:
            return None  # No change

        hist = self._histories.get(fact_id)
        current = hist.current_value() if hist else ""

        # Check for contradiction
        if current and current != old_value and current != new_value:
            log.info(
                "fact_evolution: contradiction for %s — current=%s, old=%s, new=%s",
                fact_id,
                current,
                old_value,
                new_value,
            )

        fv = FactVersion(
            fact_id=fact_id,
            value=new_value,
            timestamp=time.time(),
            source=source,
            episode_id=episode_id,
            previous_value=old_value,
            confidence=confidence,
            metadata=metadata or {},
        )

        # Add to history
        if hist is None:
            hist = FactHistory(fact_id=fact_id)
            self._histories[fact_id] = hist
        hist.versions.append(fv)

        # Persist (append-only)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fv.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            log.warning("fact_evolution.persist failed: %s", exc)

        return fv

    def get_history(self, fact_id: str) -> FactHistory | None:
        """Get the full history of a fact."""
        self._ensure_loaded()
        return self._histories.get(fact_id)

    def get_current(self, fact_id: str) -> str:
        """Get the current value of a fact."""
        self._ensure_loaded()
        hist = self._histories.get(fact_id)
        return hist.current_value() if hist else ""

    def get_state_at(self, fact_id: str, timestamp: float) -> str:
        """Get the value of a fact at a specific time."""
        self._ensure_loaded()
        hist = self._histories.get(fact_id)
        return hist.value_at(timestamp) if hist else ""

    def get_contradictions(self) -> list[dict]:
        """Get all facts with contradictions."""
        self._ensure_loaded()
        results = []
        for fact_id, hist in self._histories.items():
            if hist.has_contradiction():
                results.append(
                    {
                        "fact_id": fact_id,
                        "versions": len(hist.versions),
                        "values": list(set(v.value for v in hist.versions)),
                        "current": hist.current_value(),
                    }
                )
        return results

    def list_facts(self) -> list[str]:
        """List all tracked fact IDs."""
        self._ensure_loaded()
        return list(self._histories.keys())

    def search_facts(self, query: str) -> list[dict]:
        """Search fact histories by query."""
        self._ensure_loaded()
        query_lower = query.lower()
        results = []
        for fact_id, hist in self._histories.items():
            if query_lower in fact_id.lower():
                results.append(
                    {
                        "fact_id": fact_id,
                        "current": hist.current_value(),
                        "versions": len(hist.versions),
                        "contradiction": hist.has_contradiction(),
                    }
                )
            else:
                # Search in values
                for v in hist.versions:
                    if query_lower in v.value.lower():
                        results.append(
                            {
                                "fact_id": fact_id,
                                "current": hist.current_value(),
                                "versions": len(hist.versions),
                                "contradiction": hist.has_contradiction(),
                            }
                        )
                        break
        return results
