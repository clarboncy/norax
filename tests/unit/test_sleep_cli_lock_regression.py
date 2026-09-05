"""The offline pipeline must not reacquire its own maintenance flock."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("dry_run", [False, True])
def test_sleep_cli_completes_without_nested_lock(tmp_path, dry_run):
    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / "memory"
    root.mkdir()
    cmd = [sys.executable, "-m", "norax.sleep", "--memory-root", str(root), "--json"]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["ok"] is True
    assert summary["dry_run"] is dry_run
