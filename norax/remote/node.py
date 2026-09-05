"""norax-node: small user-owned remote worker.

The node connects outbound to the Norax relay over WSS, so remote Linux PCs do
not need inbound firewall rules. File jobs are root-scoped, I/O is bounded, and
arbitrary shell execution requires an explicit operator opt-in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from ..dispatch.risk import check as check_risk
from .protocol import read_private_token

_MAX_COMMAND_CHARS = 65_536
_MAX_WRITE_BYTES = 1_000_000
_MAX_READ_BYTES = 1_000_000
_MAX_DIRECTORY_SCAN = 10_000
_MAX_JOB_BYTES = 1 * 1024 * 1024


def _under_root(path: Path, roots: list[Path]) -> bool:
    if not roots:
        return True
    try:
        rp = path.expanduser().resolve()
    except FileNotFoundError:
        rp = path.expanduser().parent.resolve() / path.name
    return any(rp == r or r in rp.parents for r in roots)


def _node_ws_url(relay: str, node_id: str) -> str:
    if not isinstance(relay, str) or not relay or len(relay) > 4_096:
        raise ValueError("relay URL must contain 1-4096 characters")
    if (
        not isinstance(node_id, str)
        or not node_id
        or len(node_id) > 120
        or any(ord(character) < 32 or ord(character) == 127 for character in node_id)
    ):
        raise ValueError("node_id must be a string of at most 120 printable characters")
    parsed = urlparse(relay)
    scheme_map = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}
    try:
        scheme = scheme_map[parsed.scheme.lower()]
    except KeyError as exc:
        raise ValueError("relay URL must use http(s) or ws(s)") from exc
    if not parsed.netloc:
        raise ValueError("relay URL must include a host")
    path = parsed.path.rstrip("/")
    if not path.endswith("/node"):
        path = f"{path}/node" if path else "/remote/node"
    query = urlencode({"node_id": node_id})
    return urlunparse((scheme, parsed.netloc, path, "", query, ""))


class NodeWorker:
    def __init__(
        self,
        *,
        roots: list[str] | None = None,
        max_output: int = 12000,
        allow_exec: bool = False,
        allow_unrestricted_paths: bool = False,
        inherit_env: bool = False,
    ) -> None:
        for name, value in (
            ("allow_exec", allow_exec),
            ("allow_unrestricted_paths", allow_unrestricted_paths),
            ("inherit_env", inherit_env),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if roots is not None and not isinstance(roots, list):
            raise TypeError("roots must be a list")
        if any(not isinstance(root, str) for root in (roots or [])):
            raise TypeError("roots must contain strings")
        if len(roots or []) > 32:
            raise ValueError("at most 32 roots may be configured")
        if isinstance(max_output, bool) or not isinstance(max_output, int):
            raise TypeError("max_output must be an integer")
        configured_roots = [Path(r).expanduser().resolve() for r in (roots or [])]
        if not configured_roots and not allow_unrestricted_paths:
            configured_roots = [Path.cwd().resolve()]
        self.roots = configured_roots
        self.max_output = min(max(int(max_output), 1024), 1_000_000)
        self.allow_exec = allow_exec
        self.allow_unrestricted_paths = allow_unrestricted_paths
        self.inherit_env = inherit_env

    def meta(self) -> dict[str, Any]:
        return {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cwd": os.getcwd(),
            "roots": [str(r) for r in self.roots],
            "exec_enabled": self.allow_exec,
            "unrestricted_paths": self.allow_unrestricted_paths,
            "pid": os.getpid(),
        }

    def _check_path(self, path: str) -> Path:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError("path must be a non-empty string without NUL bytes")
        expanded = Path(path).expanduser()
        candidate = Path(os.path.abspath(os.fspath(expanded)))
        if not _under_root(candidate, self.roots):
            raise PermissionError(f"path outside allowed roots: {path}")
        return candidate

    def _selected_root(self, path: str) -> tuple[Path, Path]:
        candidate = self._check_path(path)
        roots = self.roots or [Path(candidate.anchor or os.sep)]
        matches = [root for root in roots if candidate == root or root in candidate.parents]
        if not matches:
            raise PermissionError(f"path outside allowed roots: {path}")
        root = max(matches, key=lambda item: len(item.parts))
        return candidate, root

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )

    def _open_directory_fd(self, path: str, *, create: bool = False) -> tuple[int, Path]:
        candidate, root = self._selected_root(path)
        relative = candidate.relative_to(root)
        descriptor = os.open(root, self._directory_flags())
        try:
            for component in relative.parts:
                try:
                    child = os.open(component, self._directory_flags(), dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    child = os.open(component, self._directory_flags(), dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            file_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(file_stat.st_mode):
                raise NotADirectoryError(path)
            return descriptor, candidate
        except Exception:
            os.close(descriptor)
            raise

    def _open_file_fd(self, path: str, *, flags: int) -> tuple[int, Path]:
        candidate, _root = self._selected_root(path)
        parent_fd, _parent = self._open_directory_fd(os.fspath(candidate.parent))
        try:
            descriptor = os.open(
                candidate.name,
                flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        finally:
            os.close(parent_fd)
        return descriptor, candidate

    @staticmethod
    def _timeout(value: Any, *, maximum: float = 300.0) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timeout must be a number")
        timeout = float(value)
        if not math.isfinite(timeout):
            raise ValueError("timeout must be finite")
        return min(max(timeout, 1.0), maximum)

    async def run(self, job: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(job, dict):
            return {"ok": False, "error": "job must be an object"}
        action = job.get("action")
        try:
            if action == "exec":
                if not self.allow_exec:
                    return {
                        "ok": False,
                        "error": "exec_disabled",
                        "detail": "restart norax-node with --allow-exec to grant shell execution",
                    }
                command = job.get("command", "")
                cwd = job.get("cwd")
                if not isinstance(command, str) or (cwd is not None and not isinstance(cwd, str)):
                    return {"ok": False, "error": "command and cwd must be strings"}
                return await self.exec(command, cwd=cwd, timeout=job.get("timeout", 30))
            if action == "read":
                timeout = self._timeout(job.get("timeout", 30))
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(
                            self.read,
                            job["path"],
                            job.get("limit", 20000),
                            job.get("offset", 0),
                        ),
                        timeout=timeout,
                    )
                except TimeoutError:
                    return {"ok": False, "error": "timeout", "timeout_s": timeout}
            if action == "list":
                timeout = self._timeout(job.get("timeout", 30))
                return await asyncio.wait_for(
                    asyncio.to_thread(self.list_dir, job.get("path", ".")),
                    timeout=timeout,
                )
            if action == "write":
                timeout = self._timeout(job.get("timeout", 30))
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self.write,
                        job["path"],
                        job.get("content", ""),
                    ),
                    timeout=timeout,
                )
            if action == "hello":
                return {"ok": True, **self.meta()}
            return {"ok": False, "error": f"unknown action: {action}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "type": type(e).__name__}

    async def exec(
        self, command: str, *, cwd: str | None = None, timeout: float = 30
    ) -> dict[str, Any]:  # noqa: ASYNC109
        if len(command) > _MAX_COMMAND_CHARS:
            return {"ok": False, "error": "command_too_large", "max_chars": _MAX_COMMAND_CHARS}
        decision = check_risk(tool="remote_exec", args={"command": command}, sender_tier="owner")
        if not decision.allowed:
            return {"ok": False, "error": "blocked_dangerous_command", "detail": decision.reason}
        timeout = self._timeout(timeout)
        workdir_fd = -1
        process_kwargs: dict[str, Any] = {}
        workdir: str | None = None
        if cwd:
            if os.name == "posix" and Path("/proc/self/fd").is_dir():
                workdir_fd, _candidate = self._open_directory_fd(cwd)
                workdir = f"/proc/self/fd/{workdir_fd}"
                process_kwargs["pass_fds"] = (workdir_fd,)
            else:  # pragma: no cover - non-POSIX fallback
                workdir = str(self._check_path(cwd))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._exec_env(),
                start_new_session=os.name == "posix",
                **process_kwargs,
            )
        finally:
            if workdir_fd >= 0:
                os.close(workdir_fd)
        stdout_task = asyncio.create_task(_drain_stream(proc.stdout, self.max_output))
        stderr_task = asyncio.create_task(_drain_stream(proc.stderr, min(self.max_output, 4000)))
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            await _kill_process_tree(proc)
            out, out_truncated = await _finish_drain(stdout_task)
            err, err_truncated = await _finish_drain(stderr_task)
            return {
                "ok": False,
                "error": "timeout",
                "timeout_s": timeout,
                "stdout": out.decode(errors="replace"),
                "stderr": err.decode(errors="replace"),
                "stdout_truncated": out_truncated,
                "stderr_truncated": err_truncated,
            }
        out, out_truncated = await _finish_drain(stdout_task)
        err, err_truncated = await _finish_drain(stderr_task)
        stdout = out.decode(errors="replace")
        stderr = err.decode(errors="replace")
        res: dict[str, Any] = {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stdout_truncated": out_truncated,
            "stderr_truncated": err_truncated,
        }
        if stderr:
            res["stderr"] = stderr
        return res

    def _exec_env(self) -> dict[str, str]:
        if self.inherit_env:
            env = dict(os.environ)
        else:
            allowed = {
                "DISPLAY",
                "HOME",
                "LANG",
                "LOGNAME",
                "PATH",
                "SHELL",
                "TERM",
                "TMPDIR",
                "USER",
                "WAYLAND_DISPLAY",
                "XDG_RUNTIME_DIR",
            }
            env = {
                key: value
                for key, value in os.environ.items()
                if key in allowed or key.startswith("LC_")
            }
        env.pop("NORAX_NODE_TOKEN", None)
        env.pop("NORAX_REMOTE_CONTROL_TOKEN", None)
        return env

    def read(self, path: str, limit: int = 20000, offset: int = 0) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        limit = min(max(limit, 1), _MAX_READ_BYTES)
        descriptor, candidate = self._open_file_fd(path, flags=os.O_RDONLY)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("read target must be a regular file")
            os.lseek(descriptor, offset, os.SEEK_SET)
            body = os.read(descriptor, limit + 1)
            truncated = len(body) > limit
            body = body[:limit]
            total_bytes = file_stat.st_size
            return {
                "ok": True,
                "path": str(candidate),
                "content": body.decode(errors="replace"),
                "offset_bytes": offset,
                "total_bytes": total_bytes,
                "truncated": truncated or offset + len(body) < total_bytes,
            }
        finally:
            os.close(descriptor)

    def list_dir(self, path: str) -> dict[str, Any]:
        descriptor, candidate = self._open_directory_fd(path)
        children: list[Path] = []
        scan_truncated = False
        children_with_stats: list[tuple[Path, os.stat_result]] = []
        try:
            with os.scandir(descriptor) as entries:
                for index, entry in enumerate(entries):
                    if index >= _MAX_DIRECTORY_SCAN:
                        scan_truncated = True
                        break
                    child = candidate / entry.name
                    children.append(child)
                    children_with_stats.append((child, entry.stat(follow_symlinks=False)))
        finally:
            os.close(descriptor)
        items: list[dict[str, Any]] = []
        selected = sorted(
            children_with_stats,
            key=lambda pair: (not stat.S_ISDIR(pair[1].st_mode), pair[0].name.lower()),
        )[:500]
        for child, child_stat in selected:
            items.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": stat.S_ISDIR(child_stat.st_mode),
                    "is_symlink": stat.S_ISLNK(child_stat.st_mode),
                    "size": child_stat.st_size,
                }
            )
        return {
            "ok": True,
            "path": str(candidate),
            "items": items,
            "total_scanned": len(children),
            "scan_truncated": scan_truncated,
            "items_truncated": len(children) > len(items) or scan_truncated,
        }

    def write(self, path: str, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        candidate, _root = self._selected_root(path)
        size = len(content.encode("utf-8"))
        if size > _MAX_WRITE_BYTES:
            return {"ok": False, "error": "content_too_large", "max_bytes": _MAX_WRITE_BYTES}
        parent_fd, _parent = self._open_directory_fd(os.fspath(candidate.parent), create=True)
        temp_name = f".{candidate.name}.{uuid.uuid4().hex}.tmp"
        temp_fd = -1
        try:
            temp_fd = os.open(
                temp_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            view = memoryview(content.encode("utf-8"))
            while view:
                written = os.write(temp_fd, view)
                if written <= 0:
                    raise OSError("short write while writing remote file")
                view = view[written:]
            os.close(temp_fd)
            temp_fd = -1
            os.replace(
                temp_name,
                candidate.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            return {"ok": True, "path": str(candidate), "bytes": size}
        finally:
            if temp_fd >= 0:
                os.close(temp_fd)
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.close(parent_fd)


async def _drain_stream(stream: asyncio.StreamReader | None, limit: int) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    stored = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(65_536)
        if not chunk:
            break
        remaining = limit - len(stored)
        if remaining > 0:
            stored.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    return bytes(stored), truncated


async def _finish_drain(task: asyncio.Task[tuple[bytes, bool]]) -> tuple[bytes, bool]:
    try:
        return await asyncio.wait_for(task, timeout=2.0)
    except TimeoutError:
        task.cancel()
        return b"", True


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    await proc.wait()


async def run_websocket_node(
    *,
    relay: str,
    node_id: str,
    token: str,
    roots: list[str],
    allow_exec: bool = False,
    allow_unrestricted_paths: bool = False,
    inherit_env: bool = False,
) -> None:
    import websockets

    worker = NodeWorker(
        roots=roots,
        allow_exec=allow_exec,
        allow_unrestricted_paths=allow_unrestricted_paths,
        inherit_env=inherit_env,
    )
    url = _node_ws_url(relay, node_id)
    while True:
        try:
            async with websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {token}"},
                ping_interval=20,
                ping_timeout=20,
                max_size=2_000_000,
            ) as ws:
                await ws.send(json.dumps({"type": "hello", "meta": worker.meta()}))
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "job":
                        continue
                    job_id = str(msg.get("job_id", ""))
                    result = await worker.run(
                        msg.get("job") if isinstance(msg.get("job"), dict) else {}
                    )
                    await ws.send(
                        json.dumps({"type": "result", "job_id": job_id, "result": result})
                    )
        except Exception as e:  # noqa: BLE001
            print(f"norax-node reconnecting after error: {e}", flush=True)
            await asyncio.sleep(3)


async def _stdin_loop(worker: NodeWorker) -> None:
    while True:
        line = await asyncio.to_thread(input)
        res = await worker.run(json.loads(line))
        print(json.dumps(res), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Norax remote node worker")
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        help="Allowed filesystem root. Repeatable. Defaults to the current directory.",
    )
    ap.add_argument("--allow-exec", action="store_true", help="Allow arbitrary shell jobs.")
    ap.add_argument(
        "--allow-unrestricted-paths",
        action="store_true",
        help="Allow file jobs outside configured roots (high risk).",
    )
    ap.add_argument(
        "--inherit-env",
        action="store_true",
        help="Expose the node process environment to shell jobs (high risk).",
    )
    ap.add_argument("--once", help="Run one JSON job and print result")
    ap.add_argument(
        "--relay",
        default=os.environ.get("NORAX_RELAY", ""),
        help="Relay base URL, e.g. https://your-domain.example/remote",
    )
    ap.add_argument("--node-id", default=os.environ.get("NORAX_NODE_ID", ""))
    ap.add_argument("--token", default=os.environ.get("NORAX_NODE_TOKEN", ""))
    ap.add_argument(
        "--token-file",
        default=os.environ.get("NORAX_NODE_TOKEN_FILE", ""),
        help="Read the enrollment token from a private file instead of process arguments.",
    )
    args = ap.parse_args()
    token = str(args.token or "").strip()
    if args.relay and args.node_id and not token and args.token_file:
        try:
            token = read_private_token(args.token_file)
        except (OSError, UnicodeError, ValueError) as exc:
            ap.error(f"could not read --token-file: {exc}")
    if token and (len(token) > 4_096 or any(character.isspace() for character in token)):
        ap.error("node token must be one whitespace-free value of at most 4096 characters")
    worker = NodeWorker(
        roots=args.root,
        allow_exec=args.allow_exec,
        allow_unrestricted_paths=args.allow_unrestricted_paths,
        inherit_env=args.inherit_env,
    )
    if args.once:
        try:
            if args.once == "-":
                raw_job = sys.stdin.buffer.read(_MAX_JOB_BYTES + 1)
                if len(raw_job) > _MAX_JOB_BYTES:
                    ap.error(f"stdin job exceeds {_MAX_JOB_BYTES} bytes")
                parsed_job = json.loads(raw_job.decode("utf-8"))
            else:
                if len(args.once.encode("utf-8")) > _MAX_JOB_BYTES:
                    ap.error(f"--once job exceeds {_MAX_JOB_BYTES} bytes")
                parsed_job = json.loads(args.once)
            if not isinstance(parsed_job, dict):
                ap.error("--once job must be a JSON object")
        except (UnicodeError, json.JSONDecodeError) as exc:
            ap.error(f"invalid --once job: {exc}")
        print(json.dumps(asyncio.run(worker.run(parsed_job)), allow_nan=False))
    elif args.relay and args.node_id and token:
        asyncio.run(
            run_websocket_node(
                relay=args.relay,
                node_id=args.node_id,
                token=token,
                roots=args.root,
                allow_exec=args.allow_exec,
                allow_unrestricted_paths=args.allow_unrestricted_paths,
                inherit_env=args.inherit_env,
            )
        )
    else:
        asyncio.run(_stdin_loop(worker))


if __name__ == "__main__":
    main()
