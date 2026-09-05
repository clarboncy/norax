"""Capability registry — tracks initialization status of brain services and
optional connectors.

Each capability records:
  - name: short identifier (e.g. "self_model", "active_inference")
  - state: one of disabled, initializing, ready, degraded, failed, stale
  - enabled: whether the capability was successfully initialized
  - last_error: exception string if init failed, empty if ok
  - last_checked: monotonic timestamp of last init attempt
  - last_success: monotonic timestamp of last successful operation
  - error_class: category of error (auth, config, dependency, timeout, unknown)

This replaces silent ``except Exception: pass`` blocks in the turn path with
structured logging and makes the status queryable from /status and /readyz.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("norax.runtime.capability")

STATE_DISABLED = "disabled"
STATE_INITIALIZING = "initializing"
STATE_READY = "ready"
STATE_DEGRADED = "degraded"
STATE_FAILED = "failed"
STATE_STALE = "stale"

ERR_AUTH = "auth"
ERR_CONFIG = "config"
ERR_DEPENDENCY = "dependency"
ERR_TIMEOUT = "timeout"
ERR_UNKNOWN = "unknown"


def _classify_error(error: BaseException) -> str:
    err_name = type(error).__name__.lower()
    if "auth" in err_name or "login" in err_name or "token" in err_name:
        return ERR_AUTH
    if "config" in err_name or "setting" in err_name:
        return ERR_CONFIG
    if "timeout" in err_name or "timedout" in err_name:
        return ERR_TIMEOUT
    if "connect" in err_name or "dependency" in err_name or "import" in err_name:
        return ERR_DEPENDENCY
    return ERR_UNKNOWN


@dataclass
class CapabilityStatus:
    name: str
    required_for_readiness: bool = False
    enabled: bool = False
    state: str = STATE_DISABLED
    last_error: str = ""
    last_checked: float | None = None
    last_success: float | None = None
    error_class: str = ""
    stale_after_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CapabilityRegistry:
    """Tracks which optional brain/connector services initialized successfully."""

    def __init__(self) -> None:
        self._caps: dict[str, CapabilityStatus] = {}

    def register(
        self,
        name: str,
        *,
        stale_after_sec: float | None = None,
        required_for_readiness: bool = False,
    ) -> None:
        """Pre-register a capability and its readiness/freshness policy.

        Capabilities are optional by default.  A missing browser or remote
        relay should be visible to operators, but must not make a text-only
        serving process unready.  Required capabilities are an explicit,
        narrow contract rather than an inference from registration order.
        """
        if name not in self._caps:
            self._caps[name] = CapabilityStatus(
                name=name,
                stale_after_sec=stale_after_sec,
                required_for_readiness=required_for_readiness,
            )
        else:
            if stale_after_sec is not None:
                self._caps[name].stale_after_sec = stale_after_sec
            # Re-registration may promote an optional capability to required,
            # but an omitted/default flag must not silently demote one.
            if required_for_readiness:
                self._caps[name].required_for_readiness = True

    def mark_initializing(self, name: str) -> None:
        """Mark a capability as currently initializing."""
        prev = self._caps.get(name)
        self._caps[name] = CapabilityStatus(
            name=name,
            enabled=False,
            state=STATE_INITIALIZING,
            last_checked=time.monotonic(),
            required_for_readiness=prev.required_for_readiness if prev else False,
            stale_after_sec=prev.stale_after_sec if prev else None,
            metadata=prev.metadata if prev else {},
        )

    def mark_ok(self, name: str) -> None:
        """Record successful initialization."""
        now = time.monotonic()
        prev = self._caps.get(name)
        self._caps[name] = CapabilityStatus(
            name=name,
            enabled=True,
            state=STATE_READY,
            last_checked=now,
            last_success=now,
            required_for_readiness=prev.required_for_readiness if prev else False,
            stale_after_sec=prev.stale_after_sec if prev else None,
            metadata=prev.metadata if prev else {},
        )

    def mark_degraded(self, name: str, reason: str = "") -> None:
        """Mark a capability as partially functional."""
        prev = self._caps.get(name)
        self._caps[name] = CapabilityStatus(
            name=name,
            enabled=True,
            state=STATE_DEGRADED,
            last_error=reason,
            last_checked=time.monotonic(),
            last_success=prev.last_success if prev else None,
            required_for_readiness=prev.required_for_readiness if prev else False,
            stale_after_sec=prev.stale_after_sec if prev else None,
            metadata=prev.metadata if prev else {},
        )

    def mark_failed(self, name: str, error: BaseException) -> None:
        """Record failed initialization with structured logging."""
        err_str = f"{type(error).__name__}: {error}"
        err_class = _classify_error(error)
        prev = self._caps.get(name)
        self._caps[name] = CapabilityStatus(
            name=name,
            enabled=False,
            state=STATE_FAILED,
            last_error=err_str,
            last_checked=time.monotonic(),
            last_success=prev.last_success if prev else None,
            error_class=err_class,
            required_for_readiness=prev.required_for_readiness if prev else False,
            stale_after_sec=prev.stale_after_sec if prev else None,
            metadata=prev.metadata if prev else {},
        )
        log.warning(
            "capability.init_failed name=%s error_class=%s error=%r",
            name,
            err_class,
            error,
            exc_info=error,
        )

    def mark_stale(self, name: str) -> None:
        """Mark a capability as stale (no recent successful operation)."""
        prev = self._caps.get(name)
        if prev and prev.state == STATE_READY:
            prev.state = STATE_STALE
            prev.last_checked = time.monotonic()

    def touch(self, name: str) -> None:
        """Record a successful operation — refreshes last_success and clears stale."""
        cap = self._caps.get(name)
        if cap:
            cap.last_success = time.monotonic()
            if cap.state == STATE_STALE:
                cap.state = STATE_READY

    def update_metadata(self, name: str, **metadata: Any) -> None:
        """Attach bounded probe evidence to a registered capability."""
        cap = self._caps.get(name)
        if cap is None:
            self.register(name)
            cap = self._caps[name]
        cap.metadata.update(metadata)

    def try_init(
        self,
        name: str,
        factory: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Try to initialize a capability. Returns the instance or None.

        On success, marks the capability as ready and returns the instance.
        On failure, logs the error, marks as failed, and returns None.
        """
        self.mark_initializing(name)
        try:
            instance = factory(*args, **kwargs)
            self.mark_ok(name)
            return instance
        except Exception as e:
            self.mark_failed(name, e)
            return None

    def is_enabled(self, name: str) -> bool:
        cap = self._caps.get(name)
        return cap.enabled if cap else False

    def is_ready(self, name: str) -> bool:
        """True if capability is in ready state (not degraded/failed/stale)."""
        cap = self._caps.get(name)
        return cap.state == STATE_READY if cap else False

    def status(self) -> dict[str, Any]:
        """Return a dict suitable for /status or /readyz."""
        self._check_stale()
        return {
            name: {
                "enabled": cap.enabled,
                "state": cap.state,
                "required_for_readiness": cap.required_for_readiness,
                "last_error": cap.last_error,
                "error_class": cap.error_class,
                "last_checked_ago": (
                    round(max(0.0, time.monotonic() - cap.last_checked), 1)
                    if cap.last_checked is not None
                    else None
                ),
                "last_success_ago": (
                    round(max(0.0, time.monotonic() - cap.last_success), 1)
                    if cap.last_success is not None
                    else None
                ),
                "stale_after_sec": cap.stale_after_sec,
                "metadata": dict(cap.metadata),
            }
            for name, cap in self._caps.items()
        }

    def failed_capabilities(self, *, required_only: bool = False) -> list[str]:
        """Names of capabilities that failed to initialize."""
        return [
            name
            for name, cap in self._caps.items()
            if cap.state == STATE_FAILED and (not required_only or cap.required_for_readiness)
        ]

    def readiness_blockers(self) -> list[str]:
        """Required capabilities lacking current usable evidence."""
        self._check_stale()
        usable = {STATE_READY, STATE_DEGRADED}
        return [
            name
            for name, cap in self._caps.items()
            if cap.required_for_readiness and cap.state not in usable
        ]

    def degraded_capabilities(self) -> list[str]:
        """Names of capabilities that are degraded."""
        return [name for name, cap in self._caps.items() if cap.state == STATE_DEGRADED]

    def stale_capabilities(self) -> list[str]:
        """Names of capabilities that are stale."""
        self._check_stale()
        return [name for name, cap in self._caps.items() if cap.state == STATE_STALE]

    def _check_stale(self) -> None:
        """Mark ready capabilities as stale if last_success is too old."""
        now = time.monotonic()
        for cap in self._caps.values():
            if (
                cap.state == STATE_READY
                and cap.last_success is not None
                and cap.stale_after_sec is not None
            ):
                if (now - cap.last_success) > cap.stale_after_sec:
                    cap.state = STATE_STALE
