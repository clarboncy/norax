"""Release checks must not silently skip explicitly selected deployment state."""

from __future__ import annotations

import importlib.util
from pathlib import Path


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
