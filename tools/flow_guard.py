#!/usr/bin/env python3
"""
flow_guard.py — Cooperative flow deadline for standalone Norax helpers.

Usage in any Python script:
    from flow_guard import FlowGuard
    guard = FlowGuard("swagbucks_surveys")

    while some_work():
        if guard.check():
            break  # or return, raise, etc.
        do_work()

    guard.finish()

When time is exceeded, ``check()`` logs an alert and returns ``True``. This
module does not kill work on its own; async callers should also apply their own
operation timeout (``BrowserWrapper`` does).
"""

import os
import secrets
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_configured_flow_dir = os.environ.get("NORAX_FLOW_STATE_DIR", "").strip()
FLOW_DIR = (
    Path(_configured_flow_dir).expanduser()
    if _configured_flow_dir
    else Path(f"/tmp/norax-flow-{os.getuid()}")
)
FLOW_START_FILE = FLOW_DIR / "start"
FLOW_TASK_FILE = FLOW_DIR / "task"
FLOW_LOG_FILE = FLOW_DIR / "alerts.log"
FLOW_MAX_FILE = FLOW_DIR / "max_seconds"
FLOW_PID_FILE = FLOW_DIR / "pid"
FLOW_ID_FILE = FLOW_DIR / "id"
DEFAULT_MAX_MINUTES = 30


def _prepare_state_dir() -> None:
    directory = FLOW_START_FILE.parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = directory.stat()
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"flow state directory is owned by another user: {directory}")
    directory.chmod(0o700)


@contextmanager
def _state_lock() -> Iterator[None]:
    """Serialize marker snapshots and compare-and-clear across processes."""
    import fcntl

    _prepare_state_dir()
    lock_path = FLOW_START_FILE.parent / ".lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        lock_path.chmod(0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write(path: Path, value: str) -> None:
    _prepare_state_dir()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _clear_markers(*, expected_flow_id: str | None = None) -> bool:
    with _state_lock():
        if expected_flow_id is not None:
            try:
                if FLOW_ID_FILE.read_text(encoding="utf-8").strip() != expected_flow_id:
                    return False
            except OSError:
                return False
        for marker in (
            FLOW_START_FILE,
            FLOW_TASK_FILE,
            FLOW_MAX_FILE,
            FLOW_PID_FILE,
            FLOW_ID_FILE,
        ):
            marker.unlink(missing_ok=True)
    return True


class FlowGuard:
    def __init__(self, task_name: str, max_minutes: int = DEFAULT_MAX_MINUTES):
        if isinstance(max_minutes, bool) or not 0 < max_minutes <= 24 * 60:
            raise ValueError("max_minutes must be between 1 and 1440")
        self.task_name = task_name.replace("\n", " ").replace("\r", " ").strip()[:512] or "unnamed"
        self.max_seconds = float(max_minutes * 60)
        self.start_time = time.time()
        self._start_monotonic = time.monotonic()
        self._started_at = datetime.now(UTC).isoformat()
        self.flow_id = secrets.token_hex(12)
        self._stopped = False
        self._expired = False
        self._finished = False
        self._warned_thresholds: set[float] = set()

        # Persist for external monitoring
        with _state_lock():
            _atomic_write(FLOW_START_FILE, str(self.start_time))
            _atomic_write(FLOW_TASK_FILE, self.task_name)
            _atomic_write(FLOW_MAX_FILE, str(self.max_seconds))
            _atomic_write(FLOW_PID_FILE, str(os.getpid()))
            _atomic_write(FLOW_ID_FILE, self.flow_id)

        log = f"[{self._started_at}] [{self.task_name}] START max_minutes={max_minutes}\n"
        with FLOW_LOG_FILE.open("a") as f:
            f.write(log)
        print(
            f"[FLOW GUARD] Started: {self.task_name} at {self._started_at} (max {max_minutes}min)",
            file=sys.stderr,
        )

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_monotonic

    @property
    def active(self) -> bool:
        return not self._stopped

    @property
    def expired(self) -> bool:
        return self._expired

    def check(self) -> bool:
        """Returns True if HARD STOP should trigger. Caller MUST stop work immediately."""
        if self._stopped:
            return self._expired

        elapsed = self.elapsed_seconds()

        if elapsed >= self.max_seconds:
            now = datetime.now(UTC).isoformat()
            mins = int(elapsed // 60)
            msg = (
                f"\n{'=' * 60}\n"
                f"[FLOW GUARD] DEADLINE EXCEEDED\n"
                f"Task: {self.task_name}\n"
                f"Started: {self._started_at}\n"
                f"Now: {now}\n"
                f"Elapsed: {mins}m {int(elapsed % 60)}s\n"
                f"MAX ALLOWED: {self.max_seconds / 60:g} minutes\n"
                f"ACTION: STOP ALL WORK. Alert user immediately.\n"
                f"{'=' * 60}\n"
            )
            with FLOW_LOG_FILE.open("a") as f:
                f.write(f"[{now}] [{self.task_name}] HARD_STOP elapsed={int(elapsed)}s\n")
            print(msg, file=sys.stderr)
            self._stopped = True
            self._expired = True
            return True

        # Warn at 50% and 83% of time
        warn_at = [0.5 * self.max_seconds, 0.83 * self.max_seconds]
        for wa in warn_at:
            if elapsed >= wa and wa not in self._warned_thresholds:
                remaining = int(self.max_seconds - elapsed)
                print(
                    f"[FLOW GUARD] WARNING: {remaining}s remaining on {self.task_name}",
                    file=sys.stderr,
                )
                self._warned_thresholds.add(wa)
                break

        return False

    def status(self) -> dict:
        should_stop = self.check()
        elapsed = self.elapsed_seconds()
        return {
            "task": self.task_name,
            "started": self._started_at,
            "elapsed_seconds": round(elapsed, 1),
            "remaining_seconds": round(max(0, self.max_seconds - elapsed), 1),
            "active": self.active,
            "should_stop": should_stop,
            "expired": self.expired,
        }

    def finish(self, success: bool = False):
        """Call when flow completes (success or early stop)."""
        if self._finished:
            return
        elapsed = self.elapsed_seconds()
        now = datetime.now(UTC).isoformat()
        with FLOW_LOG_FILE.open("a") as f:
            f.write(
                f"[{now}] [{self.task_name}] FINISH success={success} elapsed={int(elapsed)}s\n"
            )
        print(
            f"[FLOW GUARD] Finished: {self.task_name} | Success={success} | {int(elapsed // 60)}m {int(elapsed % 60)}s",
            file=sys.stderr,
        )
        _clear_markers(expected_flow_id=self.flow_id)
        self._stopped = True
        self._finished = True


def current_flow_status() -> dict:
    """Check if any flow is running and its status."""
    with _state_lock():
        if not FLOW_START_FILE.exists():
            return {"active": False, "task": None}

        try:
            start = float(FLOW_START_FILE.read_text(encoding="utf-8").strip())
            task = (
                FLOW_TASK_FILE.read_text(encoding="utf-8").strip()
                if FLOW_TASK_FILE.exists()
                else "unknown"
            )
            max_seconds = (
                float(FLOW_MAX_FILE.read_text(encoding="utf-8").strip())
                if FLOW_MAX_FILE.exists()
                else float(DEFAULT_MAX_MINUTES * 60)
            )
            pid = (
                int(FLOW_PID_FILE.read_text(encoding="utf-8").strip())
                if FLOW_PID_FILE.exists()
                else None
            )
            flow_id = (
                FLOW_ID_FILE.read_text(encoding="utf-8").strip() if FLOW_ID_FILE.exists() else None
            )
        except (OSError, ValueError) as exc:
            return {"active": False, "task": None, "error": f"invalid flow marker: {exc}"}
    elapsed = max(0.0, time.time() - start)
    process_alive = True
    if pid is not None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            process_alive = False
        except PermissionError:
            process_alive = True

    return {
        "active": process_alive,
        "task": task,
        "pid": pid,
        "flow_id": flow_id,
        "tracking_scope": "latest_started_flow",
        "stale": not process_alive,
        "elapsed_seconds": round(elapsed, 1),
        "remaining_seconds": round(max(0, max_seconds - elapsed), 1),
        "over_limit": elapsed >= max_seconds,
    }


def finish_current_flow(task_name: str, *, success: bool = False) -> dict:
    """Finish the externally recorded flow without creating a fake new one."""
    status = current_flow_status()
    recorded_task = status.get("task")
    if recorded_task is None:
        return {"ok": False, "error": "no recorded flow"}
    if task_name != recorded_task:
        return {
            "ok": False,
            "error": f"recorded flow is {recorded_task!r}, not {task_name!r}",
        }
    flow_id = status.get("flow_id")
    if not isinstance(flow_id, str) or not flow_id:
        return {"ok": False, "error": "recorded flow has no identity marker"}
    now = datetime.now(UTC).isoformat()
    with FLOW_LOG_FILE.open("a") as log_file:
        log_file.write(
            f"[{now}] [{recorded_task}] FINISH_EXTERNAL success={success} "
            f"elapsed={int(float(status.get('elapsed_seconds', 0)))}s\n"
        )
    if not _clear_markers(expected_flow_id=flow_id):
        return {
            "ok": False,
            "error": "recorded flow was replaced before it could be finished",
        }
    return {"ok": True, "task": recorded_task, "success": success}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Norax Flow Guard")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start", help="Start a new flow")
    p.add_argument("task", help="Task name")
    p.add_argument("--max-minutes", type=int, default=DEFAULT_MAX_MINUTES)

    p = sub.add_parser("check", help="Check if hard stop needed")
    p.add_argument("--task", help="Task name to verify")

    p = sub.add_parser("status", help="Get current flow status")
    p = sub.add_parser("finish", help="Mark flow as complete")
    p.add_argument("task", help="Task name")
    p.add_argument("--success", action="store_true")

    args = parser.parse_args()

    if args.command == "start":
        guard = FlowGuard(args.task, args.max_minutes)
        print(guard.status())
    elif args.command == "check":
        status = current_flow_status()
        if status["active"] and status["over_limit"]:
            print(f"HARD STOP: {status['task']} exceeded its configured deadline!")
            sys.exit(2)
        print(status)
    elif args.command == "status":
        print(current_flow_status())
    elif args.command == "finish":
        result = finish_current_flow(args.task, success=args.success)
        print(result)
        if not result["ok"]:
            sys.exit(1)
