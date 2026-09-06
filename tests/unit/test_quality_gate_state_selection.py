"""Release checks must not silently skip explicitly selected deployment state."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("broken_script", ["second", "setup"])
def test_each_shell_script_is_syntax_checked(tmp_path, monkeypatch, broken_script):
    source = Path(__file__).resolve().parents[2] / "scripts" / "quality_gate.py"
    spec = importlib.util.spec_from_file_location("quality_gate_shell_test", source)
    assert spec and spec.loader
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required for shell syntax validation")
    units = tmp_path / "ops" / "systemd"
    units.mkdir(parents=True)
    (units / "example.service").touch()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "first.sh").write_text("exit 97\n")
    second = scripts / "second.sh"
    setup = tmp_path / "setup.sh"
    second.write_text("if then\n" if broken_script == "second" else "exit 98\n")
    setup.write_text("if then\n" if broken_script == "setup" else "exit 99\n")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate.shutil, "which", lambda name: bash if name == "bash" else name)
    monkeypatch.delenv("NORAX_VERIFY_STATE_DIR", raising=False)
    monkeypatch.delenv("NORAX_VERIFY_MEMORY_ROOT", raising=False)
    monkeypatch.delenv("NORAX_QUALITY_GATE_CONTINUE", raising=False)
    checked: list[str] = []

    def run(command):
        if command[:2] == [bash, "-n"]:
            checked.extend(command[2:])
            return subprocess.run(command, capture_output=True, timeout=5, check=False).returncode
        return 0

    monkeypatch.setattr(gate, "run", run)
    assert gate.main() == 1
    assert str(second if broken_script == "second" else setup) in checked


def test_selected_missing_event_log_is_still_verified(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts" / "quality_gate.py"
    spec = importlib.util.spec_from_file_location("quality_gate_state_test", source)
    assert spec and spec.loader
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    selected = tmp_path / "missing-state"
    monkeypatch.setenv("NORAX_VERIFY_STATE_DIR", str(selected))
    monkeypatch.delenv("NORAX_VERIFY_MEMORY_ROOT", raising=False)
    monkeypatch.delenv("NORAX_QUALITY_GATE_CONTINUE", raising=False)
    checks: list[list[str]] = []

    def run(command):
        checks.append(command)
        return int("norax.verify_event_chain" in command)

    monkeypatch.setattr(gate, "run", run)
    assert gate.main() == 1
    assert any(str(selected / "events.jsonl") in command for command in checks)
