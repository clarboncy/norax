"""Evidence-backed subsystem health and degradation bookkeeping.

This module records failures, recoveries, and fallbacks that a caller confirms
it actually used. It does not execute a fallback merely because one appears in
the policy table, and it never turns a sandbox failure into unsandboxed exec.

Degradation levels:
  FULL       — all systems operational
  DEGRADED   — some subsystems down, using fallbacks
  MINIMAL    — critical failures, read-only mode
  EMERGENCY  — only core identity + basic response, no tools/memory
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger("norax.runtime.graceful_degradation")


class DegradationLevel(Enum):
    FULL = "full"
    DEGRADED = "degraded"
    MINIMAL = "minimal"
    EMERGENCY = "emergency"


@dataclass
class SubsystemStatus:
    name: str
    # None means registered but not yet checked.  Readiness cannot be inferred
    # merely from the existence of a Python object.
    healthy: bool | None = None
    error: str = ""
    fallback_active: str | None = None
    failed_at: float = 0.0
    retry_after: float = 300.0  # retry subsystem after 5 min


@dataclass
class DegradationState:
    level: DegradationLevel = DegradationLevel.FULL
    failed_subsystems: list[str] = field(default_factory=list)
    unknown_subsystems: list[str] = field(default_factory=list)
    active_fallbacks: list[str] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    alert_message: str = ""


# Severity and truthful user-facing impact per subsystem. Fallback activation
# is supplied by the code path that actually performed it.
FALLBACKS: dict[str, dict[str, Any]] = {
    "embedder": {
        "description": "Embedding service unavailable — this retriever omitted semantic results",
        "degradation": DegradationLevel.DEGRADED,
    },
    "memory_store": {
        "description": "Memory store unavailable — persistent-memory operations are unavailable",
        "degradation": DegradationLevel.DEGRADED,
    },
    "entity_graph": {
        "description": "Entity graph unavailable — skipping entity-based retrieval",
        "degradation": DegradationLevel.DEGRADED,
    },
    "causal_graph": {
        "description": "Causal graph unavailable — skipping causal retrieval",
        "degradation": DegradationLevel.DEGRADED,
    },
    "sqlite_index": {
        "description": "SQLite FTS5 index unavailable — lexical index results are unavailable",
        "degradation": DegradationLevel.DEGRADED,
    },
    "episodic": {
        "description": "Episodic buffer unavailable — not recording episodes",
        "degradation": DegradationLevel.DEGRADED,
    },
    "ollama": {
        "description": "Ollama unavailable — local inference is unavailable",
        "degradation": DegradationLevel.DEGRADED,
    },
    "gateway": {
        "description": "Gateway unavailable — gateway-backed inference is unavailable",
        "degradation": DegradationLevel.MINIMAL,
    },
    "discord": {
        "description": "Discord unavailable — Discord delivery is unavailable",
        "degradation": DegradationLevel.MINIMAL,
    },
    "web_search": {
        "description": "Web search unavailable — live search results are unavailable",
        "degradation": DegradationLevel.DEGRADED,
    },
    "browser": {
        "description": "Browser unavailable — interactive browser operations are unavailable",
        "degradation": DegradationLevel.DEGRADED,
    },
    "sandbox": {
        "description": "Sandbox unavailable — sandbox-required execution remains disabled",
        "degradation": DegradationLevel.MINIMAL,
    },
}


class GracefulDegradation:
    """Manages subsystem degradation and fallback strategies."""

    def __init__(self) -> None:
        self._subsystems: dict[str, SubsystemStatus] = {}
        self._state = DegradationState()

    def register(self, name: str) -> None:
        """Register a subsystem for monitoring."""
        self._subsystems[name] = SubsystemStatus(name=name)
        self._recalculate_level()

    def report_failure(
        self,
        name: str,
        error: str = "",
        *,
        observed_fallback: str | None = None,
    ) -> dict[str, Any]:
        """Report a failure and, if supplied, a fallback already used by the caller."""
        if name not in self._subsystems:
            self._subsystems[name] = SubsystemStatus(name=name)

        sub = self._subsystems[name]
        sub.healthy = False
        sub.error = error
        sub.failed_at = time.time()

        fallback_info = FALLBACKS.get(
            name,
            {
                "description": f"{name} unavailable — no fallback was reported",
                "degradation": DegradationLevel.DEGRADED,
            },
        )

        sub.fallback_active = observed_fallback or None

        self._recalculate_level()

        log.warning(
            "graceful_degradation: %s failed — observed_fallback=%s (level=%s)",
            name,
            sub.fallback_active or "none",
            self._state.level.value,
        )

        return {
            "subsystem": name,
            "fallback": sub.fallback_active,
            "description": fallback_info["description"],
            "degradation_level": self._state.level.value,
            "alert": self._state.alert_message,
        }

    def report_recovery(self, name: str) -> dict[str, Any]:
        """Record a successful operation, recovering a failed subsystem."""
        if name not in self._subsystems:
            self._subsystems[name] = SubsystemStatus(name=name)
        sub = self._subsystems[name]
        # Only log an actual failed -> ready transition.  The first successful
        # call is evidence of readiness, not a recovery from an outage.
        was_unhealthy = sub.healthy is False or bool(sub.fallback_active)
        was_unknown = sub.healthy is None
        sub.healthy = True
        sub.error = ""
        sub.fallback_active = None
        if was_unhealthy:
            log.info("graceful_degradation: %s recovered", name)

        self._recalculate_level()
        return {
            "subsystem": name,
            "recovered": was_unhealthy,
            "first_verified": was_unknown,
            "healthy": True,
            "degradation_level": self._state.level.value,
        }

    def is_healthy(self, name: str) -> bool:
        """Return True only after a successful operation was observed."""
        sub = self._subsystems.get(name)
        return sub is not None and sub.healthy is True

    def should_retry(self, name: str) -> bool:
        """Check if enough time has passed to retry a failed subsystem."""
        sub = self._subsystems.get(name)
        if sub is None or sub.healthy is not False:
            return True
        elapsed = time.time() - sub.failed_at
        return elapsed >= sub.retry_after

    def get_fallback(self, name: str) -> str | None:
        """Get the fallback a caller reported actually using."""
        sub = self._subsystems.get(name)
        if sub is None or sub.healthy is not False:
            return None
        return sub.fallback_active

    def _recalculate_level(self) -> None:
        """Recalculate overall degradation level."""
        failed = [name for name, sub in self._subsystems.items() if sub.healthy is False]
        unknown = [name for name, sub in self._subsystems.items() if sub.healthy is None]
        fallbacks = [
            sub.fallback_active for sub in self._subsystems.values() if sub.fallback_active
        ]

        if not failed:
            level = DegradationLevel.FULL
            alert = ""
        else:
            # Find the worst degradation level among failed subsystems
            worst = DegradationLevel.FULL
            for name in failed:
                info = FALLBACKS.get(name, {})
                sub_level = info.get("degradation", DegradationLevel.DEGRADED)
                if sub_level == DegradationLevel.EMERGENCY:
                    worst = DegradationLevel.EMERGENCY
                elif sub_level == DegradationLevel.MINIMAL and worst != DegradationLevel.EMERGENCY:
                    worst = DegradationLevel.MINIMAL
                elif sub_level == DegradationLevel.DEGRADED and worst not in (
                    DegradationLevel.MINIMAL,
                    DegradationLevel.EMERGENCY,
                ):
                    worst = DegradationLevel.DEGRADED

            level = worst
            alert_parts = [
                FALLBACKS.get(n, {}).get("description", f"{n} unavailable") for n in failed
            ]
            alert = " | ".join(alert_parts[:3])

        self._state = DegradationState(
            level=level,
            failed_subsystems=failed,
            unknown_subsystems=unknown,
            active_fallbacks=fallbacks,
            last_update=time.time(),
            alert_message=alert,
        )

    def state(self) -> DegradationState:
        return self._state

    def status(self) -> dict[str, Any]:
        return {
            "level": self._state.level.value,
            "failed_subsystems": self._state.failed_subsystems,
            "unknown_subsystems": self._state.unknown_subsystems,
            "active_fallbacks": self._state.active_fallbacks,
            "alert": self._state.alert_message,
            "subsystems": {
                name: {
                    "healthy": sub.healthy,
                    "state": (
                        "ready"
                        if sub.healthy is True
                        else "failed"
                        if sub.healthy is False
                        else "unknown"
                    ),
                    "error": sub.error,
                    "fallback": sub.fallback_active,
                }
                for name, sub in self._subsystems.items()
            },
        }


# Global singleton
_degradation: GracefulDegradation | None = None


def get_degradation_manager() -> GracefulDegradation:
    global _degradation
    if _degradation is None:
        _degradation = GracefulDegradation()
        # Register all known subsystems
        for name in FALLBACKS:
            _degradation.register(name)
    return _degradation
