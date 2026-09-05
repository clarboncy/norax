"""Sandboxed container execution — Docker/Podman isolation for untrusted code.

Provides a safe execution environment for:
  - Running untrusted code from web sources
  - Testing code in isolated environments
  - Executing commands with filesystem/network isolation

Architecture:
  - SandboxManager: manages container lifecycle
  - Per-task sandbox with configurable:
    - Resource limits (CPU, memory, timeout)
    - Filesystem isolation (mount only approved paths)
    - Network isolation (allowlist domains)
    - Auto-cleanup after task completion

Usage:
    mgr = SandboxManager()
    result = await mgr.run(
        command="python3 -c 'print(1+1)'",
        image="python:3.12-slim",
        mounts=["./workspace"],
        network=False,
        timeout=30,
    )
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("norax.dispatch.sandbox")

MAX_COMMAND_CHARS = 100_000
MAX_MOUNTS = 16
MAX_STDOUT_BYTES = 1_048_576
MAX_STDERR_BYTES = 262_144

# ── Container runtime detection ──────────────────────────────────────────

_RUNTIME: str | None = None


def _detect_runtime() -> str | None:
    """Detect available container runtime (Docker or Podman)."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    for rt in ("docker", "podman"):
        if shutil.which(rt):
            _RUNTIME = rt
            log.info("sandbox: using %s runtime", rt)
            return rt
    log.warning("sandbox: no container runtime found (docker/podman)")
    return None


def is_available() -> bool:
    """Return whether a container-runtime executable is installed.

    This deliberately does not claim that the daemon is healthy. Startup
    readiness and each execution perform an operational check.
    """
    return _detect_runtime() is not None


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class SandboxConfig:
    """Configuration for a sandbox execution."""

    image: str = "python:3.12-slim"
    mounts: list[str] = field(default_factory=list)
    network: bool = False
    cpu_limit: str = "1.0"
    memory_limit: str = "512m"
    timeout: int = 60
    workdir: str = "/sandbox"
    env: dict[str, str] = field(default_factory=dict)
    read_only: bool = False


@dataclass
class SandboxResult:
    """Result from a sandbox execution."""

    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    elapsed: float = 0.0
    container_id: str = ""
    error: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


# ── SandboxManager ───────────────────────────────────────────────────────


class SandboxManager:
    """Manages sandboxed container executions."""

    # Default images for common languages
    DEFAULT_IMAGES = {
        "python": "python:3.12-slim",
        "python3": "python:3.12-slim",
        "node": "node:22-slim",
        "nodejs": "node:22-slim",
        "bash": "ubuntu:24.04",
        "sh": "ubuntu:24.04",
        "go": "golang:1.23-bookworm",
        "rust": "rust:1.82-slim",
    }

    def __init__(self) -> None:
        self.runtime: str | None = _detect_runtime()
        self._temp_dirs: list[Path] = []

    def _resolve_image(self, command: str, image: str | None = None) -> str:
        """Resolve image from command or explicit image."""
        if image:
            return image
        # Try to detect language from command
        cmd_lower = command.lower().strip()
        first_word = cmd_lower.split()[0] if cmd_lower else ""
        if first_word in self.DEFAULT_IMAGES:
            return self.DEFAULT_IMAGES[first_word]
        # Check for shebang-like patterns
        if "python" in cmd_lower:
            return self.DEFAULT_IMAGES["python"]
        if "node" in cmd_lower:
            return self.DEFAULT_IMAGES["node"]
        return self.DEFAULT_IMAGES["bash"]

    def _build_mount_args(self, mounts: list[str]) -> list[str]:
        """Build validated bind-mount arguments without mutating the host."""
        if len(mounts) > MAX_MOUNTS:
            raise ValueError(f"at most {MAX_MOUNTS} mounts are allowed")
        args: list[str] = []
        destinations: set[str] = set()
        for mount in mounts:
            p = Path(mount).expanduser().resolve()
            if not p.exists():
                raise ValueError(f"mount path does not exist: {p}")
            if p == Path(p.anchor):
                raise ValueError("mounting a host filesystem root is not allowed")
            if any(char in str(p) for char in ("\x00", "\n", "\r", ":")):
                raise ValueError(f"mount path contains unsupported characters: {p}")
            destination = f"/sandbox/{p.name}"
            if destination in destinations:
                raise ValueError(f"mount destination collision: {destination}")
            destinations.add(destination)
            args.extend(["-v", f"{p}:{destination}:rw"])
        return args

    def _build_run_args(
        self,
        command: str,
        config: SandboxConfig,
    ) -> list[str]:
        """Build the full container run command."""
        args: list[str] = [
            self.runtime or "docker",
            "run",
            "--rm",
            "--name",
            f"norax-sandbox-{uuid.uuid4().hex[:12]}",
            "--pids-limit",
            "128",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
        ]

        # Resource limits
        args.extend(["--cpus", config.cpu_limit])
        args.extend(["--memory", config.memory_limit])

        # Network isolation
        if not config.network:
            args.append("--network=none")

        # Read-only filesystem
        if config.read_only:
            args.append("--read-only")

        # Mounts
        args.extend(self._build_mount_args(config.mounts))

        # Environment variables
        for key, val in config.env.items():
            args.extend(["-e", f"{key}={val}"])

        # Workdir
        args.extend(["-w", config.workdir])

        # Image
        image = self._resolve_image(command, config.image)
        args.append(image)

        # Command to run
        args.extend(["sh", "-c", command])

        return args

    @staticmethod
    async def _read_limited(
        stream: asyncio.StreamReader | None,
        limit: int,
    ) -> tuple[bytes, bool]:
        """Drain a subprocess stream while retaining at most ``limit`` bytes."""
        if stream is None:
            return b"", False
        chunks: list[bytes] = []
        retained = 0
        truncated = False
        while True:
            chunk = await stream.read(65_536)
            if not chunk:
                break
            remaining = limit - retained
            if remaining > 0:
                kept = chunk[:remaining]
                chunks.append(kept)
                retained += len(kept)
            if len(chunk) > max(remaining, 0):
                truncated = True
        return b"".join(chunks), truncated

    async def _force_remove(self, container_name: str) -> None:
        """Best-effort removal of the exact timed-out container."""
        if not self.runtime:
            return
        try:
            cleanup = await asyncio.create_subprocess_exec(
                self.runtime,
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(cleanup.wait(), timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("sandbox.cleanup_failed container=%s error=%r", container_name, exc)

    async def run(
        self,
        command: str,
        *,
        image: str | None = None,
        mounts: list[str] | None = None,
        network: bool = False,
        cpu_limit: str = "1.0",
        memory_limit: str = "512m",
        timeout: int = 60,
        read_only: bool = False,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run a command in a sandboxed container.

        Args:
            command: Shell command to execute
            image: Container image (auto-detected if None)
            mounts: Host paths to mount into /sandbox/
            network: Allow network access
            cpu_limit: CPU limit (e.g. "1.0", "0.5")
            memory_limit: Memory limit (e.g. "512m", "1g")
            timeout: Execution timeout in seconds
            read_only: Read-only filesystem (except mounts)
            env: Environment variables

        Returns:
            SandboxResult with stdout, stderr, exit_code
        """
        if not self.runtime:
            return SandboxResult(
                ok=False,
                exit_code=-1,
                stdout="",
                stderr="",
                error="No container runtime available (install docker or podman)",
            )
        if not command.strip():
            return SandboxResult(
                ok=False,
                exit_code=-1,
                stdout="",
                stderr="",
                error="Sandbox command must not be empty",
            )
        if len(command) > MAX_COMMAND_CHARS:
            return SandboxResult(
                ok=False,
                exit_code=-1,
                stdout="",
                stderr="",
                error=f"Sandbox command exceeds {MAX_COMMAND_CHARS} characters",
            )
        timeout = max(1, min(int(timeout), 600))

        config = SandboxConfig(
            image=image or "",
            mounts=mounts or [],
            network=network,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            timeout=timeout,
            read_only=read_only,
            env=env or {},
        )

        try:
            args = self._build_run_args(command, config)
        except (OSError, RuntimeError, ValueError) as exc:
            return SandboxResult(
                ok=False,
                exit_code=-1,
                stdout="",
                stderr="",
                error=str(exc),
            )
        container_name = args[args.index("--name") + 1]
        t0 = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_task = asyncio.create_task(self._read_limited(proc.stdout, MAX_STDOUT_BYTES))
            stderr_task = asyncio.create_task(self._read_limited(proc.stderr, MAX_STDERR_BYTES))
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout + 5)
                (stdout_b, stdout_truncated), (stderr_b, stderr_truncated) = await asyncio.gather(
                    stdout_task, stderr_task
                )
            except TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                await self._force_remove(container_name)
                return SandboxResult(
                    ok=False,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    elapsed=time.time() - t0,
                    error=f"Sandbox timeout after {timeout}s",
                )

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            exit_code = proc.returncode if proc.returncode is not None else -1

            return SandboxResult(
                ok=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                elapsed=time.time() - t0,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )
        except Exception as exc:
            log.error("sandbox.run failed: %s", exc)
            return SandboxResult(
                ok=False,
                exit_code=-1,
                stdout="",
                stderr="",
                elapsed=time.time() - t0,
                error=str(exc),
            )

    async def run_python(
        self,
        code: str,
        *,
        mounts: list[str] | None = None,
        network: bool = False,
        timeout: int = 30,
    ) -> SandboxResult:
        """Run Python code in a sandbox."""
        with tempfile.TemporaryDirectory(prefix="norax-sandbox-") as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            script = tmpdir / "script.py"
            script.write_text(code, encoding="utf-8")
            requested_mounts = list(mounts or ())
            requested_mounts.append(str(tmpdir))
            return await self.run(
                f"python3 /sandbox/{tmpdir.name}/script.py",
                image="python:3.12-slim",
                mounts=requested_mounts,
                network=network,
                timeout=timeout,
            )

    async def run_script(
        self,
        script_path: str,
        *,
        language: str = "python",
        mounts: list[str] | None = None,
        network: bool = False,
        timeout: int = 60,
    ) -> SandboxResult:
        """Run a script file in a sandbox."""
        p = Path(script_path).expanduser().resolve()
        if not p.is_file():
            return SandboxResult(
                ok=False,
                exit_code=-1,
                stdout="",
                stderr="",
                error=f"Script not found: {p}",
            )

        requested_mounts = list(mounts or ())
        requested_mounts.append(str(p.parent))

        image = self.DEFAULT_IMAGES.get(language, "python:3.12-slim")
        runner = {
            "python": "python3",
            "node": "node",
            "go": "go run",
            "rust": "cargo run",
            "bash": "bash",
        }.get(language, "python3")

        return await self.run(
            f"{runner} {shlex.quote(f'/sandbox/{p.parent.name}/{p.name}')}",
            image=image,
            mounts=requested_mounts,
            network=network,
            timeout=timeout,
        )

    def cleanup(self) -> None:
        """Clean up temporary directories."""
        for d in self._temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        self._temp_dirs.clear()


# ── Singleton ────────────────────────────────────────────────────────────

_manager: SandboxManager | None = None


def get_sandbox_manager() -> SandboxManager:
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager


async def t_sandbox_exec(
    *,
    command: str,
    image: str | None = None,
    mounts: list[str] | None = None,
    network: bool = False,
    timeout: int = 60,
) -> dict:
    """Tool: Execute a command in a sandboxed container.

    Runs the command in an isolated Docker/Podman container with:
    - Resource limits (CPU, memory)
    - Filesystem isolation (only mounted paths accessible)
    - Optional network isolation
    - Auto-cleanup after execution

    Returns: {ok, exit_code, stdout, stderr, elapsed, error}
    """
    mgr = get_sandbox_manager()
    result = await mgr.run(
        command,
        image=image,
        mounts=mounts,
        network=network,
        timeout=timeout,
    )
    return {
        "ok": result.ok,
        "exit_code": result.exit_code,
        "stdout": result.stdout[:8000],
        "stderr": result.stderr[:4000],
        "elapsed": round(result.elapsed, 2),
        "error": result.error,
        "stdout_truncated": result.stdout_truncated or len(result.stdout) > 8000,
        "stderr_truncated": result.stderr_truncated or len(result.stderr) > 4000,
    }
