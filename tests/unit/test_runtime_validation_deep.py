"""Branch-complete contracts for small runtime configuration validators."""

from __future__ import annotations

import logging

import pytest

from norax.runtime.validation import (
    _a2a_advertised_url,
    _apply_configured_fleet_startup_fixes,
    _bounded_config_int,
    _bounded_environment_int,
    _explicit_result_ok,
    _flag_enabled,
    _is_loopback_host,
    _model_identifier,
    _model_identifiers,
    _provider_identifier,
    _runtime_choice,
)


def test_flags_and_explicit_receipts_reject_truthy_lookalikes() -> None:
    assert _flag_enabled(True) is True
    assert _flag_enabled(False) is False
    assert _flag_enabled(" YES ") is True
    assert _flag_enabled("off") is False
    assert _flag_enabled("maybe", default=True) is True
    assert _flag_enabled(object()) is False

    assert _explicit_result_ok({"ok": True}) is True
    assert _explicit_result_ok({"ok": 1}) is False
    assert _explicit_result_ok("true") is False


def test_runtime_choice_normalizes_known_values_and_logs_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert _runtime_choice(" FAST ", {"fast", "safe"}, default="safe", label="mode") == "fast"
    with caplog.at_level(logging.WARNING, logger="norax.runtime.core"):
        assert _runtime_choice("turbo", {"fast", "safe"}, default="safe", label="mode") == "safe"
        assert _runtime_choice(1, {"fast", "safe"}, default="safe", label="mode") == "safe"
    assert caplog.messages == [
        "runtime.invalid_choice key=mode value='turbo' default=safe",
        "runtime.invalid_choice key=mode value=1 default=safe",
    ]


def test_model_identifier_accepts_real_names_and_rejects_ambiguous_values() -> None:
    assert _model_identifier(" qwen/model:latest ") == "qwen/model:latest"
    for value, message in (
        (None, "must be a string"),
        ("", "must contain 1-512 characters"),
        ("m" * 513, "must contain 1-512 characters"),
        ("two models", "must not contain whitespace"),
        ("model\x00name", "must not contain whitespace"),
    ):
        with pytest.raises(ValueError, match=message):
            _model_identifier(value, label="candidate")


def test_model_list_validation_is_bounded_and_validates_every_entry() -> None:
    assert _model_identifiers(None, label="models") == []
    assert _model_identifiers(("one", "two"), label="models") == ["one", "two"]
    with pytest.raises(ValueError, match="must be a list"):
        _model_identifiers("one", label="models")
    with pytest.raises(ValueError, match="at most 32"):
        _model_identifiers(["model"] * 33, label="models")
    with pytest.raises(ValueError, match="models entry must be a string"):
        _model_identifiers(["one", None], label="models")


def test_provider_identifiers_are_small_normalized_protocol_tokens() -> None:
    assert _provider_identifier(" Open_Router-2 ") == "open_router-2"
    for value in (None, "2provider", "provider/name", "p" * 65):
        with pytest.raises(ValueError):
            _provider_identifier(value)


def test_listener_helpers_cover_loopback_wildcard_and_ipv6_contracts() -> None:
    assert _is_loopback_host(" localhost ") is True
    assert _is_loopback_host("[::1]") is True
    assert _is_loopback_host("127.0.0.1") is True
    assert _is_loopback_host("agent.internal") is False

    assert _a2a_advertised_url("127.0.0.1", 8766, None) == "http://127.0.0.1:8766"
    assert _a2a_advertised_url("::1", 8766, " ") == "http://[::1]:8766"
    assert (
        _a2a_advertised_url("0.0.0.0", 8766, " https://agent.example/a2a ")
        == "https://agent.example/a2a"
    )
    with pytest.raises(ValueError, match="base_url"):
        _a2a_advertised_url("::", 8766, None)


def test_bounded_integer_parsing_has_no_boolean_decimal_or_range_coercion() -> None:
    assert _bounded_config_int(7, label="workers", minimum=1, maximum=8) == 7
    assert _bounded_config_int(" +8 ", label="workers", minimum=1, maximum=8) == 8
    for value in (True, 1.5, "1.5", "-1", 0, 9):
        with pytest.raises(ValueError, match="between 1 and 8"):
            _bounded_config_int(value, label="workers", minimum=1, maximum=8)


def test_environment_integer_falls_back_safely_and_reports_bad_override(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("NORAX_TEST_WORKERS", raising=False)
    assert _bounded_environment_int("NORAX_TEST_WORKERS", default=4, minimum=1, maximum=8) == 4
    monkeypatch.setenv("NORAX_TEST_WORKERS", "6")
    assert _bounded_environment_int("NORAX_TEST_WORKERS", default=4, minimum=1, maximum=8) == 6
    monkeypatch.setenv("NORAX_TEST_WORKERS", "unbounded")
    with caplog.at_level(logging.WARNING, logger="norax.runtime.core"):
        assert _bounded_environment_int("NORAX_TEST_WORKERS", default=4, minimum=1, maximum=8) == 4
    assert caplog.messages[-1] == (
        "runtime.invalid_environment key=NORAX_TEST_WORKERS value='unbounded' default=4"
    )


def test_private_fleet_hook_never_runs_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "norax.ops.fleet_healthcheck.apply_startup_fixes",
        lambda: calls.append("applied"),
    )
    monkeypatch.delenv("NORAX_APPLY_FLEET_STARTUP_FIXES", raising=False)
    assert _apply_configured_fleet_startup_fixes() is False
    assert calls == []

    monkeypatch.setenv("NORAX_APPLY_FLEET_STARTUP_FIXES", "on")
    assert _apply_configured_fleet_startup_fixes() is True
    assert calls == ["applied"]
