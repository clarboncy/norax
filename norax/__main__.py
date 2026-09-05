"""`python -m norax` — boots the Norax AI runtime."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from .config.loader import load_config

log = logging.getLogger("norax")

_USAGE = """usage: norax [--help] [--version]

Run the Norax AI agent runtime. Deployment requires NORAX_PROJECT_ROOT or
NORAX_CONFIG when configuration and soul files are outside the installed wheel.

options:
  -h, --help  show this help message and exit
  --version   show the installed Norax version and exit
"""


def _acquire_runtime_lock(base_dir: Path) -> Any:
    """Acquire an exclusive flock on a lock file inside *base_dir*.

    Returns the open file handle on success, or ``None`` if another Norax
    process already holds the lock.  The caller is responsible for calling
    ``.close()`` on the returned handle to release the lock.
    """
    lock_path = Path(base_dir) / ".norax.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        fd.close()
        return None
    fd.write(f"{os.getpid()}\n")
    fd.flush()
    return fd


def _load_dotenv() -> None:
    """Load the environment belonging to the selected runtime project.

    Explicit ``NORAX_CONFIG``/``NORAX_PROJECT_ROOT`` values define an
    isolation boundary. Falling back to the repository ``.env`` in that case
    can import production secrets and data paths into tests or alternate
    deployments, so fallback locations are used only without explicit context.
    Existing process environment values always win.
    """
    config_path = os.environ.get("NORAX_CONFIG")
    project_root = os.environ.get("NORAX_PROJECT_ROOT")
    if config_path or project_root:
        candidates = []
        if config_path:
            candidates.append(Path(config_path).expanduser().resolve().parent / ".env")
        if project_root:
            candidates.append(Path(project_root).expanduser().resolve() / ".env")
    else:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        break


async def _serve_runtime(
    runtime: Any,
    *,
    stop: asyncio.Event | None = None,
    install_signal_handlers: bool = True,
) -> int:
    """Run until signalled, preserving a non-zero exit on runtime failure."""
    loop = asyncio.get_running_loop()
    stop = stop or asyncio.Event()
    installed_signals: list[signal.Signals] = []
    if install_signal_handlers:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
            installed_signals.append(sig)

    log.info("norax %s starting", __import__("norax").__version__)
    runner_failure: BaseException | None = None

    def _on_runner_done(task: asyncio.Task) -> None:
        nonlocal runner_failure
        if stop.is_set():
            return
        if task.cancelled():
            runner_failure = RuntimeError("runtime.run() was cancelled unexpectedly")
        else:
            runner_failure = task.exception() or RuntimeError("runtime.run() exited unexpectedly")
        log.critical(
            "runtime.run() terminated unexpectedly: %r",
            runner_failure,
            exc_info=runner_failure,
        )
        stop.set()

    runner = asyncio.create_task(runtime.run())
    runner.add_done_callback(_on_runner_done)
    try:
        await stop.wait()
        if runner_failure is None:
            log.info("shutdown signal received; draining")
        else:
            log.error("runtime failure detected; draining owned resources")
        await runtime.shutdown()
        if not runner.done():
            runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass
        except Exception as error:  # noqa: BLE001
            if runner_failure is None:
                runner_failure = error
        if runner_failure is not None:
            log.error("norax stopped after runtime failure")
            return 1
        log.info("norax stopped cleanly")
        return 0
    finally:
        for sig in installed_signals:
            loop.remove_signal_handler(sig)


async def _async_main() -> int:
    from .observability.logging_setup import setup_logging
    from .runtime.core import Runtime

    setup_logging()
    _load_dotenv()
    cfg = load_config()
    runtime = Runtime.build(cfg)
    return await _serve_runtime(runtime)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        if args in (["-h"], ["--help"]):
            print(_USAGE, end="")
            return 0
        if args == ["--version"]:
            print(__import__("norax").__version__)
            return 0
        print(f"norax: unrecognized arguments: {' '.join(args)}", file=sys.stderr)
        print("Try 'norax --help' for usage.", file=sys.stderr)
        return 2
    # Acquire runtime lock to prevent duplicate Norax processes.
    default_lock_dir = (
        Path(os.environ["XDG_RUNTIME_DIR"]) / "norax"
        if os.environ.get("XDG_RUNTIME_DIR")
        else Path.home() / ".local" / "state" / "norax"
    )
    lock_dir = Path(os.environ.get("NORAX_LOCK_DIR") or default_lock_dir)
    lock = _acquire_runtime_lock(lock_dir)
    if lock is None:
        print(
            "norax: another Norax runtime is already running (lock held). "
            "Stop it first or use a different NORAX_LOCK_DIR.",
            file=sys.stderr,
        )
        return 1
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        return 130
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
