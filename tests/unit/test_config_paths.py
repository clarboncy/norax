"""Regression tests for config path resolution and startup validation."""

from __future__ import annotations

import pytest

from norax.config.loader import Config, _parse_discord, load_config, load_jsonc


def test_production_config_paths_exist():
    """All configured required paths must exist and resolve to soul/ directory."""
    cfg = load_config()
    assert cfg.soul_path.exists(), f"soul path missing: {cfg.soul_path}"
    assert cfg.identity_path.exists(), f"identity path missing: {cfg.identity_path}"
    assert cfg.user_path.exists(), f"user path missing: {cfg.user_path}"


def test_config_paths_resolve_to_soul_dir():
    """Paths should resolve to the soul/ subdirectory, not repo root."""
    cfg = load_config()
    assert cfg.soul_path.parent.name == "soul"
    assert cfg.identity_path.parent.name == "soul"
    assert cfg.user_path.parent.name == "soul"


def test_validate_required_paths_returns_empty_when_all_exist():
    cfg = load_config()
    missing = cfg.validate_required_paths()
    assert missing == []


def test_validate_required_paths_reports_missing(tmp_path):
    """When soul files don't exist, validation reports them."""
    raw = {
        "paths": {
            "soul": "./missing/SOUL.md",
            "identity": "./missing/IDENTITY.md",
            "user": "./missing/USER.md",
        }
    }
    cfg = Config(raw=raw, project_root=tmp_path)
    missing = cfg.validate_required_paths()
    assert len(missing) == 3
    assert any("soul:" in m for m in missing)


def test_load_config_raises_on_missing_paths(tmp_path):
    """load_config should raise FileNotFoundError when required paths are missing."""
    cfg_path = tmp_path / "runtime.jsonc"
    cfg_path.write_text(
        '{"paths": {"soul": "./nope.md", "identity": "./nope.md", "user": "./nope.md"}}'
    )
    with pytest.raises(FileNotFoundError, match="Required prompt files"):
        load_config(path=cfg_path)


def test_config_root_and_object_sections_are_validated(tmp_path, monkeypatch):
    monkeypatch.delenv("NORAX_MEMORY_ROOT", raising=False)
    cfg_path = tmp_path / "runtime.jsonc"
    cfg_path.write_text("[]")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        load_jsonc(cfg_path)

    with pytest.raises(ValueError, match="configuration root"):
        Config(raw=[], project_root=tmp_path)  # type: ignore[arg-type]

    cfg = Config(raw={"paths": ["not", "an", "object"]}, project_root=tmp_path)
    with pytest.raises(ValueError, match="config.paths must be an object"):
        _ = cfg.memory_root

    cfg = Config(raw={"runtime": "fast"}, project_root=tmp_path)
    with pytest.raises(ValueError, match="config.runtime must be an object"):
        _ = cfg.thinking_effort

    with pytest.raises(ValueError, match="unsupported sections"):
        Config(raw={"imaginary_runtime": {}}, project_root=tmp_path)


def test_dead_runtime_and_connector_settings_are_rejected(tmp_path):
    cfg = Config(raw={"runtime": {"context_length": 65_536}}, project_root=tmp_path)
    with pytest.raises(ValueError, match="unsupported settings"):
        _ = cfg.max_tool_rounds

    cfg = Config(
        raw={"connectors": {"planner": {"enabled": True, "required": True}}},
        project_root=tmp_path,
    )
    with pytest.raises(ValueError, match="unsupported settings"):
        _ = cfg.connectors


def test_nested_discord_and_cron_typos_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="channels.discord contains unsupported"):
        _parse_discord({"enabled": True, "allowBotsTypo": True})

    cfg = Config(
        raw={
            "cron": {
                "enabled": True,
                "jobs": [{"name": "job", "schedule": {"every_seconds": 5}, "jiter": 2}],
            }
        },
        project_root=tmp_path,
    )
    with pytest.raises(ValueError, match="unsupported settings"):
        _ = cfg.cron


def test_config_paths_resolve_relative_to_project_root(tmp_path):
    """Paths should resolve relative to project_root, not CWD."""
    (tmp_path / "soul").mkdir()
    (tmp_path / "soul" / "SOUL.md").write_text("soul")
    (tmp_path / "soul" / "IDENTITY.md").write_text("identity")
    (tmp_path / "soul" / "USER.md").write_text("user")
    raw = {
        "paths": {
            "soul": "./soul/SOUL.md",
            "identity": "./soul/IDENTITY.md",
            "user": "./soul/USER.md",
        }
    }
    cfg = Config(raw=raw, project_root=tmp_path)
    assert cfg.soul_path == (tmp_path / "soul" / "SOUL.md").resolve()
    assert cfg.validate_required_paths() == []


def test_runtime_path_environment_overrides_are_absolute(tmp_path, monkeypatch):
    data = tmp_path / "external"
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(data / "memory"))
    monkeypatch.setenv("NORAX_STATE_DIR", str(data / "state"))
    monkeypatch.setenv("NORAX_LOG_DIR", str(data / "logs"))

    cfg = Config(raw={"paths": {"memory_root": "./memory"}}, project_root=tmp_path)

    assert cfg.memory_root == (data / "memory").resolve()
    assert cfg.state_dir == (data / "state").resolve()
    assert cfg.log_dir == (data / "logs").resolve()
    assert cfg.event_log == (data / "state" / "events.jsonl").resolve()


def test_absolute_runtime_paths_do_not_get_prefixed(tmp_path, monkeypatch):
    monkeypatch.delenv("NORAX_MEMORY_ROOT", raising=False)
    monkeypatch.delenv("NORAX_STATE_DIR", raising=False)
    monkeypatch.delenv("NORAX_LOG_DIR", raising=False)
    data = tmp_path / "external"
    cfg = Config(
        raw={
            "paths": {
                "memory_root": str(data / "memory"),
                "state_dir": str(data / "state"),
                "log_dir": str(data / "logs"),
            }
        },
        project_root=tmp_path / "checkout",
    )

    assert cfg.memory_root == (data / "memory").resolve()
    assert cfg.state_dir == (data / "state").resolve()
    assert cfg.log_dir == (data / "logs").resolve()


def test_invalid_runtime_numeric_environment_uses_safe_default(tmp_path, monkeypatch):
    monkeypatch.setenv("NORAX_MAX_TOOL_ROUNDS", "not-a-number")
    cfg = Config(raw={"runtime": {"max_tool_rounds": 80}}, project_root=tmp_path)
    assert cfg.max_tool_rounds == 80


def test_runtime_numeric_config_rejects_nonfinite_and_negative_values(tmp_path, monkeypatch):
    monkeypatch.delenv("NORAX_MAX_TOOL_ROUNDS", raising=False)
    cfg = Config(
        raw={"runtime": {"max_tool_rounds": -1, "shutdown_grace_seconds": "nan"}},
        project_root=tmp_path,
    )
    assert cfg.max_tool_rounds == 8
    assert cfg.shutdown_grace_seconds == 10.0


def test_reasoning_output_parses_false_string_as_false(tmp_path):
    cfg = Config(
        raw={"runtime": {"reasoning_output": "false", "stream_replies": "false"}},
        project_root=tmp_path,
    )
    assert cfg.reasoning_output is False
    assert cfg.stream_replies is False


def test_runtime_choices_are_normalized_and_invalid_values_use_safe_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("NORAX_THINKING_EFFORT", "not-a-level")
    cfg = Config(
        raw={
            "runtime": {
                "planning_mode": "ORCHESTRATOR",
                "thinking_effort": "high",
                "memory_depth": "unknown",
                "weak_model_boost": "ON",
                "response_length": 12,
                "tool_activity": "VERBOSE",
            }
        },
        project_root=tmp_path,
    )
    assert cfg.planning_mode == "orchestrator"
    assert cfg.thinking_effort == "medium"
    assert cfg.memory_depth == "auto"
    assert cfg.weak_model_boost == "on"
    assert cfg.response_length == "balanced"
    assert cfg.tool_activity == "verbose"


def test_gateway_schema_normalizes_only_settings_runtime_consumes(tmp_path):
    cfg = Config(
        raw={
            "gateway": {
                "default_model": "example/model",
                "default_provider": "example",
                "timeout_seconds": "45",
                "failover_models": ["fallback/model"],
                "providers": {
                    "example": {
                        "base_url": "https://example.test/v1/",
                        "stream_required": "false",
                        "extra_headers": {"X-Test": "value"},
                    }
                },
                "routes": [["example/*", "example"]],
            }
        },
        project_root=tmp_path,
    )

    gateway = cfg.gateway
    assert gateway["timeout_seconds"] == 45.0
    assert gateway["failover_models"] == ["fallback/model"]
    assert gateway["providers"]["example"] == {
        "base_url": "https://example.test/v1",
        "provider_kind": "example",
        "api_key": None,
        "extra_headers": {"X-Test": "value"},
        "stream_required": False,
        "model_prefix": None,
        "chat_path": "/chat/completions",
    }
    assert gateway["routes"] == [["example/*", "example"]]


@pytest.mark.parametrize(
    ("gateway", "message"),
    [
        ({"timeout": 10}, "unsupported settings"),
        ({"failover_models": "model"}, "failover_models"),
        (
            {"providers": {"example": {"provider_kind": "openai"}}},
            "base_url is required",
        ),
        (
            {
                "default_provider": "missing",
                "providers": {"example": {"base_url": "https://example.test/v1"}},
            },
            "not a configured provider",
        ),
        (
            {
                "providers": {"example": {"base_url": "https://example.test/v1"}},
                "routes": [["example/*", "missing"]],
            },
            "unknown provider",
        ),
        (
            {
                "providers": {
                    "example": {
                        "base_url": "https://example.test/v1",
                        "extra_headers": {"X-Test": "bad\nvalue"},
                    }
                }
            },
            "printable ASCII",
        ),
        ({"api_key": "secret\nheader"}, "printable ASCII"),
        ({"stream_required": "maybe"}, "must be a boolean"),
    ],
)
def test_gateway_schema_rejects_dead_or_malformed_settings(tmp_path, gateway, message):
    cfg = Config(raw={"gateway": gateway}, project_root=tmp_path)
    with pytest.raises(ValueError, match=message):
        _ = cfg.gateway


def test_turn_concurrency_and_backlog_limits_are_bounded(tmp_path):
    cfg = Config(
        raw={
            "runtime": {
                "max_concurrent_turns": 1000,
                "max_pending_turns": 100_000,
            }
        },
        project_root=tmp_path,
    )
    assert cfg.max_concurrent_turns == 64
    assert cfg.max_pending_turns == 10_000

    invalid = Config(
        raw={
            "runtime": {
                "max_concurrent_turns": 0,
                "max_pending_turns": "nan",
            }
        },
        project_root=tmp_path,
    )
    assert invalid.max_concurrent_turns == 8
    assert invalid.max_pending_turns == 256


def test_http_bind_supports_ipv4_hostname_and_bracketed_ipv6(tmp_path, monkeypatch):
    monkeypatch.delenv("NORAX_HTTP_BIND", raising=False)
    monkeypatch.delenv("NORAX_HTTP_PORT", raising=False)
    assert Config(raw={}, project_root=tmp_path).http_bind == ("127.0.0.1", 4101)
    assert Config(raw={"http": {"bind": "localhost:4201"}}, project_root=tmp_path).http_bind == (
        "localhost",
        4201,
    )
    assert Config(raw={"http": {"bind": "[::1]:4301"}}, project_root=tmp_path).http_bind == (
        "::1",
        4301,
    )


def test_http_bind_environment_overrides_are_validated(tmp_path, monkeypatch):
    cfg = Config(raw={"http": {"bind": "127.0.0.1:4101"}}, project_root=tmp_path)
    monkeypatch.setenv("NORAX_HTTP_BIND", "0.0.0.0")
    monkeypatch.setenv("NORAX_HTTP_PORT", "4201")
    assert cfg.http_bind == ("0.0.0.0", 4201)

    monkeypatch.setenv("NORAX_HTTP_PORT", "nan")
    with pytest.raises(ValueError, match="NORAX_HTTP_PORT"):
        _ = cfg.http_bind


@pytest.mark.parametrize(
    "bind",
    ["", "127.0.0.1", "::1:4101", "bad host:4101", "127.0.0.1:65536"],
)
def test_http_bind_rejects_ambiguous_or_invalid_values(tmp_path, monkeypatch, bind):
    monkeypatch.delenv("NORAX_HTTP_BIND", raising=False)
    monkeypatch.delenv("NORAX_HTTP_PORT", raising=False)
    cfg = Config(raw={"http": {"bind": bind}}, project_root=tmp_path)
    with pytest.raises(ValueError, match="http.bind"):
        _ = cfg.http_bind


def test_identity_and_agent_os_secret_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("NORAX_OWNER_ID", "owner-from-env")
    monkeypatch.setenv("NORAX_OWNER_LABEL", "Public Owner")
    monkeypatch.setenv("NORAX_AGENT_OS_CHAT_TOKEN", "secret-from-env")
    cfg = Config(
        raw={
            "owner": {"id": "tracked-id", "label": "Tracked Label"},
            "agent_os": {"chat_token": "tracked-secret"},
        },
        project_root=tmp_path,
    )
    assert cfg.owner_id == "owner-from-env"
    assert cfg.owner_label == "Public Owner"
    assert cfg.agent_os_chat_token == "secret-from-env"


def test_discord_environment_configuration(monkeypatch):
    monkeypatch.setenv("NORAX_DISCORD_ENABLED", "true")
    monkeypatch.setenv(
        "NORAX_DISCORD_CONFIG_JSON",
        '{"allowBots":true,"dmPolicy":"allowlist","allowFrom":["owner"],"guilds":{}}',
    )
    cfg = _parse_discord({"enabled": False, "dmPolicy": "deny"})
    assert cfg.enabled is True
    assert cfg.allow_bots is True
    assert cfg.dm_policy == "allowlist"
    assert cfg.allow_from == {"owner"}


def test_discord_environment_configuration_rejects_non_object(monkeypatch):
    monkeypatch.setenv("NORAX_DISCORD_CONFIG_JSON", "[]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        _parse_discord({})


def test_jsonc_loader_rejects_unbounded_configuration_file(tmp_path):
    path = tmp_path / "runtime.jsonc"
    path.write_bytes(b'{"padding":"' + (b"x" * (2 * 1024 * 1024)) + b'"}')

    with pytest.raises(ValueError, match="configuration exceeds"):
        load_jsonc(path)


def test_vision_and_cognition_settings_have_consumed_schemas(tmp_path):
    cfg = Config(
        raw={
            "vision": {"enabled": "true", "model": "provider/vision-model"},
            "cognition": {
                "experimental_signals": False,
                "idle_learning": {"enabled": True},
                "harness_analysis": "false",
            },
        },
        project_root=tmp_path,
    )

    assert cfg.vision == {"enabled": True, "model": "provider/vision-model"}
    assert cfg.cognition == {
        "experimental_signals": False,
        "harness_analysis": False,
        "idle_learning": True,
    }

    with pytest.raises(ValueError, match="unsupported settings"):
        _ = Config(
            raw={"vision": {"enabled": True, "model": "provider/model", "fallback_model": "x"}},
            project_root=tmp_path,
        ).vision
