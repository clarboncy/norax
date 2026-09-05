"""Stateful shell session — cwd persists across calls within a session."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .runner import clamp_timeout, default_workspace, run_command


@dataclass
class ShellSession:
    session_id: str
    cwd: str = field(default_factory=default_workspace)

    def reset_cwd(self, cwd: str) -> None:
        self.cwd = cwd

    async def run(self, command: str, *, timeout: float = 30.0) -> dict:
        timeout = clamp_timeout(timeout)
        # Test suites and runtime reloads can leave NORAX_WORKSPACE/session cwd
        # pointing at a removed temp dir. Stateful shell should recover to the
        # process cwd instead of running relative commands from a dead path.
        proc_cwd = Path.cwd().resolve()
        try:
            if not Path(self.cwd).is_dir():
                self.cwd = str(proc_cwd)
            parts = command.strip().split(maxsplit=1)
            if parts and parts[0] == "cd" and len(parts) == 2:
                target = parts[1].strip().strip("'\"")
                if target and not target.startswith(("/", "~", "-")):
                    if (proc_cwd / target).is_dir() and not (Path(self.cwd) / target).is_dir():
                        self.cwd = str(proc_cwd)
        except OSError:
            self.cwd = str(proc_cwd)
        result = await run_command(
            command,
            cwd=self.cwd,
            timeout=timeout,
            track_cwd=True,
        )
        if result.get("cwd"):
            self.cwd = result["cwd"]
        result["session_id"] = self.session_id
        result["cwd"] = self.cwd
        return result
