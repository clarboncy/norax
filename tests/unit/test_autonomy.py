from pathlib import Path

import pytest

from norax.runtime.autonomy import AutonomyConfig, check_subsystem_health, sync_scratchpad


@pytest.mark.parametrize(
    ("initialized", "with_episode", "expected_state"),
    [(True, True, None), (True, False, "ready_empty"), (False, False, "not_initialized")],
)
def test_health_check_observes_episodic_state_without_initializing_it(
    tmp_path: Path,
    initialized: bool,
    with_episode: bool,
    expected_state: str | None,
):
    episodic_dir = tmp_path / "episodic"
    if initialized:
        episodic_dir.mkdir()
    if with_episode:
        (episodic_dir / "episodes-2026-09-01.jsonl").write_text("{}\n")

    result = check_subsystem_health(tmp_path)
    episodic_issues = [issue for issue in result["issues"] if issue["subsystem"] == "episodic"]

    assert ("episodic" in result["healthy"]) is initialized
    assert episodic_issues == []
    assert episodic_dir.exists() is initialized
    if expected_state:
        assert {"subsystem": "episodic", "state": expected_state} in result["observations"]


def test_sync_scratchpad_preserves_model_and_provider_facts(tmp_path: Path):
    scratchpad = tmp_path / "scratchpad.md"
    scratchpad.write_text(
        "SCRATCHPAD;updated=old;type=hot_memory\n"
        "HISTORY:gpt-4o produced this artifact\n"
        "HISTORY:gpt-4o produced this artifact\n"
        "CONFIG:direct Ollama routing on a test host\n",
        encoding="utf-8",
    )

    result = sync_scratchpad(tmp_path)

    body = scratchpad.read_text(encoding="utf-8")
    assert result["changed"] is True
    assert "gpt-4o produced this artifact" in body
    assert body.count("gpt-4o produced this artifact") == 2
    assert "direct Ollama routing on a test host" in body
    assert "gpt-5.2" not in body
    assert "FreeToken routing" not in body

    second = sync_scratchpad(tmp_path)
    assert second == {"changed": False, "changes": []}


def test_sync_scratchpad_does_not_follow_symlinks(tmp_path: Path):
    target = tmp_path / "controlled.md"
    target.write_text("do not modify", encoding="utf-8")
    (tmp_path / "scratchpad.md").symlink_to(target)

    with pytest.raises(OSError):
        sync_scratchpad(tmp_path)

    assert target.read_text(encoding="utf-8") == "do not modify"
    health = check_subsystem_health(tmp_path)
    assert any(
        issue["subsystem"] == "scratchpad" and issue["issue"] == "not a regular file"
        for issue in health["issues"]
    )


def test_expensive_autonomy_features_are_opt_in(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "NORAX_AUTONOMY_PROMPT_OPT",
        "NORAX_AUTONOMY_LEARNING",
        "NORAX_AUTONOMY_MULTI_AGENT",
    ):
        monkeypatch.delenv(name, raising=False)

    config = AutonomyConfig()

    assert config.enabled is False
    assert config.prompt_opt_enabled is False
    assert config.learning_enabled is False
    assert config.multi_agent_auto is False


def test_expensive_autonomy_features_can_be_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NORAX_AUTONOMY_PROMPT_OPT", "1")
    monkeypatch.setenv("NORAX_AUTONOMY_LEARNING", "true")
    monkeypatch.setenv("NORAX_AUTONOMY_MULTI_AGENT", "yes")

    config = AutonomyConfig()

    assert config.prompt_opt_enabled is True
    assert config.learning_enabled is True
    assert config.multi_agent_auto is True
