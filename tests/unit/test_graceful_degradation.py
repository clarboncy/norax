from __future__ import annotations

from norax.runtime.graceful_degradation import DegradationLevel, GracefulDegradation


def test_registered_and_unknown_subsystems_are_not_claimed_healthy():
    manager = GracefulDegradation()
    manager.register("embedder")

    assert manager.is_healthy("embedder") is False
    assert manager.is_healthy("never_registered") is False
    assert manager.state().level is DegradationLevel.FULL
    assert manager.status()["subsystems"]["embedder"]["state"] == "unknown"
    assert manager.status()["unknown_subsystems"] == ["embedder"]


def test_success_is_evidence_and_only_observed_fallback_is_active():
    manager = GracefulDegradation()
    manager.register("embedder")

    first = manager.report_recovery("embedder")
    assert first["first_verified"] is True
    assert first["recovered"] is False
    assert manager.is_healthy("embedder") is True

    failure = manager.report_failure(
        "embedder", "offline", observed_fallback="omit_semantic_results"
    )
    assert failure["fallback"] == "omit_semantic_results"
    assert manager.state().level is DegradationLevel.DEGRADED
    assert manager.get_fallback("embedder") == "omit_semantic_results"

    recovered = manager.report_recovery("embedder")
    assert recovered["recovered"] is True
    assert manager.state().level is DegradationLevel.FULL


def test_unobserved_fallback_is_not_claimed_active():
    manager = GracefulDegradation()
    failure = manager.report_failure("sandbox", "not installed")

    assert failure["fallback"] is None
    assert manager.get_fallback("sandbox") is None
    assert manager.state().level is DegradationLevel.MINIMAL
    assert "remains disabled" in failure["description"]
