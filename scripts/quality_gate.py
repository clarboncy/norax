#!/usr/bin/env python3
"""Norax quality gate: deterministic local checks before shipping."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CHECK_TIMEOUT_SECONDS = 600
STATIC_ROOTS = ["norax", "agent_os", "tools", "tests", "scripts", "benchmarks", "training"]


def run(cmd: list[str], *, timeout: int = CHECK_TIMEOUT_SECONDS) -> int:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)  # noqa: S603
    try:
        return proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if sys.exc_info()[0] is KeyboardInterrupt:
            raise
        print(f"FAILED: check exceeded {timeout}s timeout", flush=True)
        return 124


def main() -> int:
    python = sys.executable
    ruff = str(Path(python).with_name("ruff"))
    mypy = str(Path(python).with_name("mypy"))
    uv = shutil.which("uv")
    if uv is None:
        print("FAILED: uv is required to verify the lockfile", flush=True)
        return 1
    systemd_analyze = shutil.which("systemd-analyze")
    if systemd_analyze is None:
        print("FAILED: systemd-analyze is required to validate deployment units", flush=True)
        return 1
    bash = shutil.which("bash")
    if bash is None:
        print("FAILED: bash is required to validate shell helpers", flush=True)
        return 1
    systemd_units = sorted(
        str(path)
        for pattern in ("*.service", "*.timer")
        for path in (ROOT / "ops" / "systemd").glob(pattern)
    )
    if not systemd_units:
        print("FAILED: no systemd deployment units found", flush=True)
        return 1
    shell_scripts = sorted(
        str(path)
        for directory in (ROOT / "ops", ROOT / "scripts", ROOT / "tools")
        for path in directory.rglob("*.sh")
    )
    setup_script = ROOT / "setup.sh"
    if setup_script.is_file():
        shell_scripts.append(str(setup_script))
    if not shell_scripts:
        print("FAILED: no shell helpers found", flush=True)
        return 1
    checks = [
        [uv, "lock", "--check"],
        [systemd_analyze, "--user", "verify", *systemd_units],
        [bash, "-n", *shell_scripts],
        [ruff, "check", *STATIC_ROOTS],
        [ruff, "format", "--check", *STATIC_ROOTS],
        [mypy, "norax", "agent_os", "tools"],
        [mypy, "scripts", "benchmarks", "training"],
        [python, "-m", "compileall", "-q", *STATIC_ROOTS],
        [python, "scripts/publication_check.py"],
        [python, "scripts/secret_scan.py"],
        [python, "scripts/verify_stack_e2e.py"],
        [
            python,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--cov=norax",
            "--cov-branch",
            "--cov-report=term",
            "--cov-report=json:coverage.json",
            "--cov-fail-under=64.5",
            "-m",
            "not host_integration and not subprocess_integration",
        ],
        [python, "scripts/coverage_guard.py", "coverage.json"],
        [
            python,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "tests/integration/test_phase1_acceptance.py",
            "tests/integration/test_phase8_runtime.py",
        ],
    ]
    # Mutable deployment state is only authoritative when the caller selects
    # it explicitly. The repository may contain legacy/dev event fixtures that
    # are intentionally outside the active service's chain.
    configured_state_dir = os.environ.get("NORAX_VERIFY_STATE_DIR")
    if configured_state_dir:
        events_file = Path(configured_state_dir) / "events.jsonl"
        checks.append(
            [
                python,
                "-m",
                "norax.verify_event_chain",
                "--file",
                str(events_file),
            ]
        )
    configured_memory_root = os.environ.get("NORAX_VERIFY_MEMORY_ROOT")
    if configured_memory_root:
        checks.append(
            [
                python,
                "scripts/memory_lint.py",
                "--memory-root",
                configured_memory_root,
            ]
        )
    continue_after_failure = os.environ.get("NORAX_QUALITY_GATE_CONTINUE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    failed = False
    for cmd in checks:
        rc = run(cmd)
        if rc:
            failed = True
            if not continue_after_failure:
                print("FAILED: stopping at first failed gate", flush=True)
                return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
