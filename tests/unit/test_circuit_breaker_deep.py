from __future__ import annotations

import pytest

from norax.runtime import circuit_breaker as module


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cooldown_sec": True},
        {"max_cooldown_sec": "60"},
    ],
)
def test_circuit_config_rejects_non_numeric_cooldowns(kwargs):
    with pytest.raises(ValueError, match="finite non-negative"):
        module.CircuitConfig(**kwargs)


def test_closed_open_half_open_and_recovery_transitions(monkeypatch):
    now = 100.0
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    monkeypatch.setattr(module.random, "uniform", lambda *_args: 0.0)
    circuit = module.Circuit(
        "tool",
        module.CircuitConfig(
            failure_threshold=2,
            success_threshold=2,
            cooldown_sec=10,
            max_cooldown_sec=60,
            half_open_max_calls=1,
        ),
    )
    circuit.record_success()
    circuit.record_failure("first")
    assert circuit.state is module.CircuitState.CLOSED
    circuit.record_failure("second")
    assert circuit.state is module.CircuitState.OPEN

    circuit.record_failure("concurrent failure")
    allowed, reason = circuit.can_execute()
    assert allowed is False and reason == "circuit_open cooldown=10s"
    now += 10
    assert circuit.can_execute() == (True, "ok")
    assert circuit.can_execute() == (False, "circuit_half_open_probe_in_flight")
    circuit.record_success()
    assert circuit.state is module.CircuitState.HALF_OPEN
    assert circuit.can_execute() == (True, "ok")
    circuit.record_success()
    assert circuit.state is module.CircuitState.CLOSED
    assert circuit.stats()["cooldown_sec"] == 0


def test_failed_half_open_probe_reopens_with_bounded_backoff(monkeypatch):
    now = 10.0
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    monkeypatch.setattr(module.random, "uniform", lambda *_args: 0.0)
    circuit = module.Circuit(
        "tool",
        module.CircuitConfig(
            failure_threshold=1,
            cooldown_sec=10,
            max_cooldown_sec=12,
        ),
    )
    circuit.record_failure("initial")
    now += 10
    assert circuit.can_execute()[0]
    circuit.record_failure("probe failed")
    assert circuit.state is module.CircuitState.OPEN
    assert circuit.stats()["cooldown_sec"] == 12


def test_abandon_closed_probe_is_a_noop():
    circuit = module.Circuit("tool")
    circuit.abandon_probe()
    assert circuit.can_execute() == (True, "ok")


def test_registry_delegates_resets_and_does_not_share_mutable_default_config(monkeypatch):
    monkeypatch.setattr(module.random, "uniform", lambda *_args: 0.0)
    registry = module.CircuitBreakerRegistry()
    first = registry.get_or_create("first")
    assert registry.get_or_create("first", module.CircuitConfig(failure_threshold=1)) is first
    second = registry.get_or_create("second")
    assert first.config is not second.config
    first.config.failure_threshold = 1
    assert second.config.failure_threshold == 5

    assert registry.check("first") == (True, "ok")
    registry.record_failure("first", "bad")
    assert registry.check("first")[0] is False
    registry.record_success("second")
    registry.abandon_probe("second")
    stats = {row["name"]: row for row in registry.all_stats()}
    assert stats["first"]["total_failures"] == 1
    assert stats["second"]["total_successes"] == 1
    registry.reset_all()
    assert all(row["state"] == "closed" for row in registry.all_stats())


def test_global_registry_is_lazy_and_reused(monkeypatch):
    monkeypatch.setattr(module, "_registry", None)
    first = module.get_circuit_registry()
    assert module.get_circuit_registry() is first
