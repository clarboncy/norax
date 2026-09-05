"""Global pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_norax_config(tmp_path, monkeypatch):
    """Prevent tests from reading or mutating any live Norax deployment.

    Test processes are sometimes launched by the running service's ``exec``
    tool and consequently inherit its production ``NORAX_*`` environment.
    Respecting those inherited values caused otherwise unit-scoped runtimes to
    append fake ingress/error events to the live chain and expose live Discord
    credentials to parser tests. Start every test from a clean environment and
    give it an isolated config, state, log, memory, checkpoint, and lock root.
    Individual tests remain free to set any variable after this fixture runs.
    """
    import os

    for key in tuple(os.environ):
        if key.startswith("NORAX_"):
            monkeypatch.delenv(key, raising=False)

    # Keep the isolation tree below a private namespace. Many unit tests use
    # ``tmp_path / "state"`` (and similar names) as their explicit fixture and
    # correctly expect to create that directory themselves.
    isolation = tmp_path / "_norax_isolation"
    project = isolation / "project"
    state = isolation / "state"
    logs = isolation / "logs"
    memory = isolation / "memory"
    checkpoints = memory / "checkpoints"
    locks = isolation / "locks"
    config_home = isolation / "config"
    for path in (project, state, logs, memory, checkpoints, locks, config_home):
        path.mkdir(parents=True, exist_ok=True)

    stub = project / "runtime.jsonc"
    stub.write_text('{"gateway": {"default_model": "claude-sonnet-4.6"}}\n')
    soul = project / "soul"
    soul.mkdir(parents=True, exist_ok=True)
    for name in ("SOUL.md", "IDENTITY.md", "USER.md"):
        (soul / name).write_text(f"# {name.removesuffix('.md')}\n")
    monkeypatch.setenv("NORAX_CONFIG", str(stub))
    monkeypatch.setenv("NORAX_PROJECT_ROOT", str(project))
    monkeypatch.setenv("NORAX_STATE_DIR", str(state))
    monkeypatch.setenv("NORAX_LOG_DIR", str(logs))
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(memory))
    monkeypatch.setenv("NORAX_CHECKPOINT_DIR", str(checkpoints))
    monkeypatch.setenv("NORAX_LOCK_DIR", str(locks))
    monkeypatch.setenv("NORAX_CONFIG_HOME", str(config_home))
    yield
