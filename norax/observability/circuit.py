"""Simple circuit breaker: sliding-window failure count → open/half-open/closed."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from enum import StrEnum

log = logging.getLogger("norax.observability.circuit")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        window_seconds: float = 30.0,
        failure_threshold: int = 5,
        open_seconds: float = 30.0,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("circuit name must be non-empty")
        if (
            isinstance(failure_threshold, bool)
            or not isinstance(failure_threshold, int)
            or failure_threshold < 1
        ):
            raise ValueError("failure_threshold must be a positive integer")
        for field_name, value, allow_zero in (
            ("window_seconds", window_seconds, False),
            ("open_seconds", open_seconds, True),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a finite non-negative number")
            normalized = float(value)
            if (
                not math.isfinite(normalized)
                or normalized < 0
                or (not allow_zero and normalized == 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{field_name} must be a finite {qualifier} number")
        self.name = name
        self.window = float(window_seconds)
        self.threshold = failure_threshold
        self.open_seconds = float(open_seconds)
        self._failures: deque[float] = deque()
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._half_open_probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        self._maybe_transition()
        return self._state

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    def _maybe_transition(self) -> None:
        now = time.monotonic()
        if self._state == CircuitState.OPEN and (now - self._opened_at) >= self.open_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_probe_in_flight = False
            log.info("circuit.half_open name=%s", self.name)

    def before_call(self) -> None:
        self._maybe_transition()
        if self._state == CircuitState.OPEN:
            raise CircuitOpen(f"{self.name} is open")
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_probe_in_flight:
                raise CircuitOpen(f"{self.name} half-open probe is already in flight")
            self._half_open_probe_in_flight = True

    def on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            log.info("circuit.close name=%s", self.name)
            self._state = CircuitState.CLOSED
            self._half_open_probe_in_flight = False
            self._failures.clear()

    def on_failure(self) -> None:
        now = time.monotonic()
        self._failures.append(now)
        self._prune(now)
        # Requests already in flight can fail after a sibling opened the
        # circuit. Count them in the window, but do not keep moving the
        # recovery deadline forward for the same outage wave.
        if self._state == CircuitState.OPEN:
            return
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_probe_in_flight = False
            self._trip(now)
            return
        if len(self._failures) >= self.threshold:
            self._trip(now)

    def on_abandoned(self) -> None:
        """Release a half-open probe cancelled before an outcome was observed."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_probe_in_flight = False

    def _trip(self, now: float) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = now
        self._half_open_probe_in_flight = False
        log.warning("circuit.open name=%s failures=%d", self.name, len(self._failures))
