"""Circuit Breaker — prevents retry storms and cascading failures.

Research (Zylos 2026, AWS distributed systems):
  - Agents without circuit breakers enter unbounded retry loops
  - Exponential backoff with jitter reduces retry storms by 60-80%
  - Circuit breakers should open after sustained failure, stay open
    for a cooldown period, then allow a half-open probe

States:
  CLOSED    — normal operation, requests pass through
  OPEN      — all requests fail fast, no calls made
  HALF_OPEN — single probe request allowed; if it succeeds → CLOSED,
              if it fails → back to OPEN

Integration:
  - agent_loop checks circuit before each tool dispatch
  - if OPEN, returns error immediately without calling the tool
  - if HALF_OPEN, allows one probe; result determines next state
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger("norax.runtime.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitConfig:
    failure_threshold: int = 5  # consecutive failures before opening
    success_threshold: int = 2  # consecutive successes in half-open to close
    cooldown_sec: float = 60.0  # how long to stay open before half-open probe
    max_cooldown_sec: float = 600.0  # exponential backoff cap
    half_open_max_calls: int = 1  # max concurrent probe calls

    def __post_init__(self) -> None:
        for name in ("failure_threshold", "success_threshold", "half_open_max_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("cooldown_sec", "max_cooldown_sec"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
            setattr(self, name, normalized)
        if self.max_cooldown_sec < self.cooldown_sec:
            raise ValueError("max_cooldown_sec must be at least cooldown_sec")


@dataclass
class Circuit:
    """A single circuit breaker for one tool or operation."""

    name: str
    config: CircuitConfig = field(default_factory=CircuitConfig)
    state: CircuitState = CircuitState.CLOSED
    _consecutive_failures: int = 0
    _consecutive_successes: int = 0
    _opened_at: float = 0.0
    _cooldown: float = 0.0  # set on first open
    _half_open_in_flight: int = 0
    _total_failures: int = 0
    _total_successes: int = 0
    _total_rejected: int = 0

    def can_execute(self) -> tuple[bool, str]:
        """Check if execution is allowed. Returns (allowed, reason)."""
        now = time.monotonic()

        if self.state == CircuitState.OPEN:
            elapsed = now - self._opened_at
            if elapsed >= self._cooldown:
                # Transition to half-open
                self.state = CircuitState.HALF_OPEN
                self._half_open_in_flight = 0
                self._consecutive_successes = 0
                log.info("circuit[%s] OPEN→HALF_OPEN after %.1fs", self.name, elapsed)
            else:
                self._total_rejected += 1
                remaining = self._cooldown - elapsed
                return False, f"circuit_open cooldown={remaining:.0f}s"

        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_in_flight >= self.config.half_open_max_calls:
                self._total_rejected += 1
                return False, "circuit_half_open_probe_in_flight"

        if self.state == CircuitState.HALF_OPEN:
            self._half_open_in_flight += 1

        return True, "ok"

    def record_success(self) -> None:
        self._total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self._consecutive_failures = 0
                self._consecutive_successes = 0
                self._cooldown = 0.0
                log.info("circuit[%s] HALF_OPEN→CLOSED (recovered)", self.name)
        else:
            self._consecutive_failures = 0

    def record_failure(self, error: str = "") -> None:
        self._total_failures += 1
        self._consecutive_failures += 1

        # Concurrent calls that were already in flight may report after a
        # sibling opened the circuit. Count them, but do not repeatedly extend
        # the cooldown for the same outage wave.
        if self.state == CircuitState.OPEN:
            return

        if self.state == CircuitState.HALF_OPEN:
            # Probe failed — back to open with exponential backoff
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            self._open(error)
            log.warning("circuit[%s] HALF_OPEN→OPEN (probe failed: %s)", self.name, error[:100])
        elif self._consecutive_failures >= self.config.failure_threshold:
            self._open(error)
            log.warning(
                "circuit[%s] CLOSED→OPEN after %d failures (last: %s)",
                self.name,
                self._consecutive_failures,
                error[:100],
            )

    def abandon_probe(self) -> None:
        """Release a half-open slot when its caller is cancelled."""
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    def _open(self, reason: str = "") -> None:
        self.state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        # Exponential backoff: start at config.cooldown_sec, grow 1.5x each open
        if self._cooldown <= 0:
            self._cooldown = self.config.cooldown_sec
        else:
            self._cooldown = min(
                self._cooldown * 1.5,
                self.config.max_cooldown_sec,
            )
        # Add jitter (±20%)
        jitter = random.uniform(-0.2, 0.2)
        self._cooldown = min(
            self.config.max_cooldown_sec,
            max(self.config.cooldown_sec, self._cooldown * (1 + jitter)),
        )

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_in_flight = 0
        self._cooldown = 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "total_rejected": self._total_rejected,
            "cooldown_sec": round(self._cooldown, 1),
            "opened_at": self._opened_at,
        }


class CircuitBreakerRegistry:
    """Registry of circuit breakers, keyed by tool/operation name."""

    def __init__(self) -> None:
        self._circuits: dict[str, Circuit] = {}

    def get_or_create(self, name: str, config: CircuitConfig | None = None) -> Circuit:
        if name not in self._circuits:
            self._circuits[name] = Circuit(
                name=name,
                config=config or CircuitConfig(),
            )
        return self._circuits[name]

    def check(self, name: str) -> tuple[bool, str]:
        """Check if a tool/operation can execute."""
        circuit = self.get_or_create(name)
        return circuit.can_execute()

    def record_success(self, name: str) -> None:
        circuit = self.get_or_create(name)
        circuit.record_success()

    def record_failure(self, name: str, error: str = "") -> None:
        circuit = self.get_or_create(name)
        circuit.record_failure(error)

    def abandon_probe(self, name: str) -> None:
        circuit = self.get_or_create(name)
        circuit.abandon_probe()

    def all_stats(self) -> list[dict[str, Any]]:
        return [c.stats() for c in self._circuits.values()]

    def reset_all(self) -> None:
        for c in self._circuits.values():
            c.reset()


# Global singleton
_registry: CircuitBreakerRegistry | None = None


def get_circuit_registry() -> CircuitBreakerRegistry:
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry
