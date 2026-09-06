"""State-machine, recovery, and diagnostic safety tests for capabilities."""

from __future__ import annotations

import logging

import pytest

from norax.runtime import capability_registry as registry_module
from norax.runtime.capability_registry import (
    ERR_AUTH,
    ERR_CONFIG,
    ERR_DEPENDENCY,
    ERR_TIMEOUT,
    ERR_UNKNOWN,
    STATE_DEGRADED,
    STATE_DISABLED,
    STATE_FAILED,
    STATE_INITIALIZING,
    STATE_READY,
    STATE_STALE,
    CapabilityRegistry,
    _classify_error,
)


@pytest.mark.parametrize(
    ("exception_name", "expected"),
    [
        ("AuthenticationFailure", ERR_AUTH),
        ("LoginRejected", ERR_AUTH),
        ("TokenExpired", ERR_AUTH),
        ("ConfigurationError", ERR_CONFIG),
        ("SettingInvalid", ERR_CONFIG),
        ("RequestTimeout", ERR_TIMEOUT),
        ("TimedOut", ERR_TIMEOUT),
        ("ConnectionLost", ERR_DEPENDENCY),
        ("DependencyMissing", ERR_DEPENDENCY),
        ("ImportFailure", ERR_DEPENDENCY),
        ("OrdinaryFailure", ERR_UNKNOWN),
    ],
)
def test_error_classification_uses_exception_type_not_untrusted_message(
    exception_name: str,
    expected: str,
) -> None:
    exception_type = type(exception_name, (RuntimeError,), {})
    assert _classify_error(exception_type("auth token in a misleading message")) == expected


def test_registration_updates_policy_without_accidental_demotion() -> None:
    registry = CapabilityRegistry()
    registry.register("relay")
    registry.register("relay")
    registry.register("relay", stale_after_sec=30, required_for_readiness=True)
    registry.register("relay", required_for_readiness=False)

    status = registry.status()["relay"]
    assert status["state"] == STATE_DISABLED
    assert status["stale_after_sec"] == 30
    assert status["required_for_readiness"] is True


def test_state_replacements_preserve_policy_metadata_and_last_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [10.0]
    monkeypatch.setattr(registry_module.time, "monotonic", lambda: now[0])
    registry = CapabilityRegistry()
    registry.register("core", stale_after_sec=60, required_for_readiness=True)
    registry.update_metadata("core", source="probe")

    registry.mark_initializing("core")
    assert registry.status()["core"]["state"] == STATE_INITIALIZING
    now[0] = 20.0
    registry.mark_ok("core")
    now[0] = 30.0
    registry.mark_degraded("core", "partial")
    now[0] = 40.0
    registry.mark_failed("core", RuntimeError("offline"))

    now[0] = 60.0
    status = registry.status()["core"]
    assert status["state"] == STATE_FAILED
    assert status["last_success_ago"] == 40.0
    assert status["required_for_readiness"] is True
    assert status["stale_after_sec"] == 60
    assert status["metadata"] == {"source": "probe"}


def test_unregistered_state_transitions_get_safe_defaults() -> None:
    registry = CapabilityRegistry()
    registry.mark_initializing("initializing")
    registry.mark_ok("ready")
    registry.mark_degraded("degraded", "limited")
    registry.mark_failed("failed", RuntimeError("offline"))

    states = registry.status()
    assert states["initializing"]["state"] == STATE_INITIALIZING
    assert states["ready"]["state"] == STATE_READY
    assert states["degraded"]["state"] == STATE_DEGRADED
    assert states["failed"]["state"] == STATE_FAILED
    assert all(item["metadata"] == {} for item in states.values())


def test_successful_use_recovers_failed_degraded_and_stale_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module.time, "monotonic", lambda: 50.0)
    registry = CapabilityRegistry()
    registry.mark_failed("browser", RuntimeError("down"))
    registry.mark_degraded("search", "partial")
    registry.mark_ok("relay")
    registry.mark_stale("relay")

    for name in ("browser", "search", "relay"):
        registry.touch(name)
        status = registry.status()[name]
        assert status["enabled"] is True
        assert status["state"] == STATE_READY
        assert status["last_error"] == ""
        assert status["error_class"] == ""
        assert status["last_checked_ago"] == 0.0
        assert status["last_success_ago"] == 0.0
    registry.touch("not-registered")


def test_mark_stale_only_changes_an_existing_ready_capability() -> None:
    registry = CapabilityRegistry()
    registry.mark_stale("absent")
    registry.register("disabled")
    registry.mark_stale("disabled")
    registry.mark_ok("ready")
    registry.mark_stale("ready")

    assert registry.status()["disabled"]["state"] == STATE_DISABLED
    assert registry.status()["ready"]["state"] == STATE_STALE


def test_metadata_registers_on_demand_and_status_returns_a_copy() -> None:
    registry = CapabilityRegistry()
    registry.update_metadata("probe", count=1)
    registry.update_metadata("probe", source="live")
    status = registry.status()
    status["probe"]["metadata"]["count"] = 99
    assert registry.status()["probe"]["metadata"] == {"count": 1, "source": "live"}


def test_failure_diagnostics_are_redacted_bounded_and_safe_to_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-" + ("a" * 32)
    registry = CapabilityRegistry()
    with caplog.at_level(logging.WARNING, logger="norax.runtime.capability"):
        registry.mark_failed("provider", RuntimeError(f"key={secret} " + ("x" * 1_000)))
    registry.mark_degraded("relay", f"token={secret} " + ("y" * 1_000))

    failed = registry.status()["provider"]["last_error"]
    degraded = registry.status()["relay"]["last_error"]
    assert secret not in failed + degraded + " ".join(caplog.messages)
    assert "<REDACTED:openai_key>" in failed
    assert len(failed) <= 500
    assert len(degraded) <= 500


def test_query_helpers_cover_required_degraded_failed_and_stale_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(registry_module.time, "monotonic", lambda: now[0])
    registry = CapabilityRegistry()
    registry.register("required_failed", required_for_readiness=True)
    registry.mark_failed("required_failed", RuntimeError("down"))
    registry.mark_failed("optional_failed", RuntimeError("down"))
    registry.register("required_degraded", required_for_readiness=True)
    registry.mark_degraded("required_degraded", "slow")
    registry.register("fresh", stale_after_sec=10)
    registry.mark_ok("fresh")
    registry.register("no_success", stale_after_sec=10)
    registry.register("no_policy")
    registry.mark_ok("no_policy")

    now[0] = 105.0
    assert registry.is_enabled("missing") is False
    assert registry.is_ready("missing") is False
    assert registry.failed_capabilities() == ["required_failed", "optional_failed"]
    assert registry.failed_capabilities(required_only=True) == ["required_failed"]
    assert registry.degraded_capabilities() == ["required_degraded"]
    assert registry.readiness_blockers() == ["required_failed"]
    assert registry.stale_capabilities() == []

    now[0] = 111.0
    assert registry.stale_capabilities() == ["fresh"]
    assert registry.readiness_blockers() == ["required_failed"]


def test_try_init_passes_arguments_and_contains_factory_failures() -> None:
    registry = CapabilityRegistry()

    assert registry.try_init("sum", lambda left, *, right: left + right, 2, right=3) == 5
    assert registry.try_init("boom", lambda: (_ for _ in ()).throw(ValueError("bad"))) is None
    assert registry.is_ready("sum") is True
    assert registry.failed_capabilities() == ["boom"]
