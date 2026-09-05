from __future__ import annotations

import asyncio
import os

import pytest

import norax
from norax.__main__ import _acquire_runtime_lock, _load_dotenv, _serve_runtime, main


def test_help_does_not_boot_runtime(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr()
    assert "usage: norax" in output.out
    assert output.err == ""


def test_version_reports_installed_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == norax.__version__


def test_unknown_argument_fails_without_booting(capsys):
    assert main(["--definitely-invalid"]) == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_runtime_lock_allows_only_one_process(tmp_path):
    first = _acquire_runtime_lock(tmp_path)
    assert first is not None
    assert _acquire_runtime_lock(tmp_path) is None
    first.close()
    second = _acquire_runtime_lock(tmp_path)
    assert second is not None
    second.close()


def test_explicit_project_does_not_fall_back_to_repository_dotenv(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("NORAX_CONFIG", str(config_dir / "runtime.jsonc"))
    monkeypatch.setenv("NORAX_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("NORAX_HTTP_PORT", raising=False)

    _load_dotenv()

    assert "NORAX_HTTP_PORT" not in os.environ


def test_explicit_project_loads_its_own_dotenv(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / ".env").write_text("NORAX_TEST_PROJECT_VALUE=isolated\n")
    monkeypatch.setenv("NORAX_CONFIG", str(config_dir / "runtime.jsonc"))
    monkeypatch.setenv("NORAX_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("NORAX_TEST_PROJECT_VALUE", raising=False)

    _load_dotenv()

    assert os.environ["NORAX_TEST_PROJECT_VALUE"] == "isolated"


@pytest.mark.asyncio
async def test_runtime_crash_returns_failure_after_shutdown():
    class _Runtime:
        shutdown_count = 0

        async def run(self):
            raise ConnectionError("ingress died")

        async def shutdown(self):
            self.shutdown_count += 1

    runtime = _Runtime()

    result = await _serve_runtime(runtime, install_signal_handlers=False)

    assert result == 1
    assert runtime.shutdown_count == 1


@pytest.mark.asyncio
async def test_unexpected_clean_runtime_exit_is_still_a_failure():
    class _Runtime:
        shutdown_count = 0

        async def run(self):
            return None

        async def shutdown(self):
            self.shutdown_count += 1

    runtime = _Runtime()

    result = await _serve_runtime(runtime, install_signal_handlers=False)

    assert result == 1
    assert runtime.shutdown_count == 1


@pytest.mark.asyncio
async def test_external_stop_remains_a_clean_exit():
    stop = asyncio.Event()
    stop.set()

    class _Runtime:
        shutdown_count = 0

        async def run(self):
            await asyncio.Event().wait()

        async def shutdown(self):
            self.shutdown_count += 1

    runtime = _Runtime()

    result = await _serve_runtime(
        runtime,
        stop=stop,
        install_signal_handlers=False,
    )

    assert result == 0
    assert runtime.shutdown_count == 1
