"""Shared subprocess helpers for exec (one-shot) and shell (session-backed)."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
from pathlib import Path

_MARKER = "__NORAX_END__"
_SELF_SYSTEMCTL_SEGMENT = re.compile(
    r"(?:^|&&|\|\||[;\n])\s*"
    r"(?P<segment>(?:sudo\s+)?(?:/[A-Za-z0-9_./-]+/)?systemctl\b[^;&|\n]*)",
    re.I,
)


def _search_command_no_matches(command: str, *, exit_code: int, stdout: str, stderr: str) -> bool:
    """Recognize grep/rg's documented exit=1 meaning 'no matches'."""
    if exit_code != 1 or stdout.strip() or stderr.strip():
        return False
    stripped = command.strip()
    if any(op in stripped for op in ("&&", "||", ";", "|")):
        return False
    return bool(re.match(r"^(?:command\s+)?(?:(?:git\s+)?grep|rg)(?:\s|$)", stripped))


def _self_runtime_lifecycle_request(command: str) -> tuple[str, bool] | None:
    """Detect a local stop/restart of the service executing this command.

    Returns ``(action, has_prefix)``. Commands inside ssh/remote wrappers do
    not match because the shell segment starts with ``ssh``, not ``systemctl``.
    """
    for match in _SELF_SYSTEMCTL_SEGMENT.finditer(command):
        segment = match.group("segment")
        if not re.search(r"\bnorax-ai(?:\.service)?\b", segment, re.I):
            continue
        action_match = re.search(r"\b(try-restart|restart|stop)\b", segment, re.I)
        if action_match is None:
            continue
        prefix = command[: match.start()].strip(" \t\r\n;&|")
        action = action_match.group(1).lower()
        return ("restart" if action == "try-restart" else action, bool(prefix))
    return None


def build_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for p in (
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
        str(Path.home() / ".local" / "bin"),
        str(Path.home() / "bin"),
    ):
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")
    if extra:
        env.update(extra)
    return env


def default_workspace() -> str:
    return str(Path(os.environ.get("NORAX_WORKSPACE", Path.cwd())).expanduser().resolve())


def clamp_timeout(timeout: float, *, lo: float = 1.0, hi: float | None = None) -> float:
    if hi is None:
        try:
            hi = float(os.environ.get("NORAX_COMMAND_TIMEOUT_MAX", "600"))
        except ValueError:
            hi = 600.0
        hi = max(lo, hi)
    return min(max(float(timeout), lo), hi)


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Terminate a timed-out shell and all descendants without leaking jobs."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        await proc.wait()


async def run_command(
    command: str,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
    track_cwd: bool = False,
) -> dict:
    """Run a shell command; optionally append cwd/exit marker for session tracking."""
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    elif not isinstance(command, str):
        command = str(command)

    lifecycle = _self_runtime_lifecycle_request(command)
    if lifecycle is not None:
        action, has_prefix = lifecycle
        if has_prefix:
            return {
                "ok": False,
                "error": "self_lifecycle_must_be_separate",
                "hint": (
                    "Run setup/verification commands separately, then request the Norax "
                    f"runtime {action} in its own exec call. The lifecycle action is deferred "
                    "until after the final response is delivered."
                ),
            }
        return {
            "ok": True,
            "exit_code": 0,
            "stdout": (
                f"Norax runtime {action} accepted and deferred until after the active "
                "task response is delivered."
            ),
            "runtime_lifecycle_deferred": action,
            "note": (
                "Do not call systemctl again or wait for the service in this turn. "
                "Finish verification that does not require the restart, then provide "
                "the final response."
            ),
        }

    timeout = clamp_timeout(timeout)
    run_cmd = command
    if track_cwd:
        # No subshell — bare `cd` must persist so pwd captures the new directory.
        run_cmd = (
            f"{command}; __ec=$?; printf '\\n{_MARKER}:%d:%s\\n' $__ec \"$(pwd -P)\"; exit $__ec"
        )

    requested_cwd = cwd
    effective_cwd: str | None = None
    cwd_note: str | None = None
    if requested_cwd:
        try:
            if Path(requested_cwd).is_dir():
                effective_cwd = requested_cwd
            else:
                cwd_note = (
                    f"requested cwd '{requested_cwd}' does not exist on this host; "
                    "falling back to process default. Set NORAX_WORKSPACE or pick a valid path."
                )
        except OSError as exc:
            cwd_note = f"cwd check failed for '{requested_cwd}': {exc!s}; using default"

    proc = await asyncio.create_subprocess_shell(
        run_cmd,
        cwd=effective_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=build_env(env),
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        await _terminate_process_group(proc)
        return {
            "ok": False,
            "error": "timeout",
            "timeout_s": timeout,
            "hint": (
                f"Command exceeded {timeout}s wall-clock limit. "
                "Use a shorter timeout or background the process."
            ),
        }
    except asyncio.CancelledError:
        await _terminate_process_group(proc)
        raise

    stdout_raw = out.decode(errors="replace")
    stderr_str = err.decode(errors="replace")[:2000]
    exit_code = proc.returncode if proc.returncode is not None else 1
    new_cwd: str | None = None

    if track_cwd:
        lines = stdout_raw.splitlines()
        stdout_lines = lines
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if line.startswith(f"{_MARKER}:"):
                parts = line.split(":", 2)
                if len(parts) == 3:
                    try:
                        exit_code = int(parts[1])
                    except ValueError:
                        pass
                    new_cwd = parts[2]
                stdout_lines = lines[:i]
                break
        stdout_str = "\n".join(stdout_lines).strip()[:8000]
    else:
        stdout_str = stdout_raw[:8000]

    no_matches = _search_command_no_matches(
        command, exit_code=exit_code, stdout=stdout_str, stderr=stderr_str
    )
    result: dict = {
        "ok": exit_code == 0 or no_matches,
        "exit_code": exit_code,
        "stdout": stdout_str,
    }
    if no_matches:
        result.update(
            {
                "no_matches": True,
                "note": "Search completed successfully; no matching lines were found.",
            }
        )
    if stderr_str:
        result["stderr"] = stderr_str
    if exit_code != 0 and stderr_str:
        result["combined"] = (stdout_str + "\n" + stderr_str).strip()[:8000]
    if new_cwd is not None:
        result["cwd"] = new_cwd
    if cwd_note:
        result["cwd_note"] = cwd_note
    return result
