"""Persistent reminder delivery with restart recovery and bounded retries."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import secrets
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text

log = logging.getLogger("norax.runtime.reminders")

_MAX_REMINDER_BYTES = 65_536
_MAX_REMINDER_FILES = 4_096
_MAX_REMINDER_TEXT = 16_384
_MAX_ROUTE_CHARS = 256
_MAX_RESULT_TEXT = 1_000
_EXTERNAL_RESCAN_SECONDS = 60.0
_REMINDER_FILENAME = re.compile(r"^reminder-[A-Za-z0-9_-]{1,192}\.json$")


def _bounded_text(value: object, *, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text or len(text) > maximum or any(not char.isprintable() for char in text):
        raise ValueError(f"{label} must contain 1-{maximum} printable characters")
    return text


def _read_payload(path: Path) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError("reminder is not a regular file")
            if file_stat.st_size > _MAX_REMINDER_BYTES:
                raise ValueError("reminder exceeds 64 KiB")
            raw = handle.read(_MAX_REMINDER_BYTES + 1)
        if len(raw) > _MAX_REMINDER_BYTES:
            raise ValueError("reminder exceeds 64 KiB")
        payload = json.loads(raw.decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError, RecursionError, UnicodeError, ValueError) as error:
        log.warning("reminder invalid path=%s error=%r", path, error)
        return None


def _reminder_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    try:
        with os.scandir(root) as entries:
            for scanned, entry in enumerate(entries, start=1):
                if scanned > _MAX_REMINDER_FILES:
                    log.warning("reminder scan capped root=%s limit=%d", root, _MAX_REMINDER_FILES)
                    break
                if not _REMINDER_FILENAME.fullmatch(entry.name):
                    continue
                try:
                    if entry.is_file(follow_symlinks=False):
                        paths.append(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        return []
    paths.sort(key=lambda path: path.name)
    return paths


def _safe_delivery_result(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"ok": False, "error": f"invalid outbound result: {type(value).__name__}"}
    result: dict[str, Any] = {"ok": value.get("ok") is True}
    for key in ("message_id", "error", "status", "detail"):
        item = value.get(key)
        if item is not None:
            result[key] = str(item)[:_MAX_RESULT_TEXT]
    return result


def _ensure_private_root(root: Path) -> None:
    if root.is_symlink():
        raise OSError("reminder root must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(root, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("reminder root is not a directory")
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(UTC)


class ReminderScheduler:
    """Persist reminders and deliver them through the live outbound registry."""

    def __init__(
        self,
        *,
        root: Path,
        outbound: Any,
        default_channel: str = "discord",
        default_target: str | None = None,
        poll_seconds: float = 1.0,
        delivery_timeout_seconds: float = 30.0,
    ) -> None:
        self.root = root
        self.outbound = outbound
        self.default_channel = default_channel
        self.default_target = default_target
        if isinstance(poll_seconds, bool):
            raise ValueError("poll_seconds must be a finite positive number")
        try:
            parsed_poll = float(poll_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("poll_seconds must be a finite positive number") from error
        if not math.isfinite(parsed_poll) or parsed_poll <= 0:
            raise ValueError("poll_seconds must be a finite positive number")
        self.poll_seconds = min(60.0, max(0.1, parsed_poll))
        if isinstance(delivery_timeout_seconds, bool):
            raise ValueError("delivery_timeout_seconds must be a finite positive number")
        try:
            parsed_delivery_timeout = float(delivery_timeout_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("delivery_timeout_seconds must be a finite positive number") from error
        if not math.isfinite(parsed_delivery_timeout) or parsed_delivery_timeout <= 0:
            raise ValueError("delivery_timeout_seconds must be a finite positive number")
        self.delivery_timeout_seconds = min(parsed_delivery_timeout, 300.0)
        self._task: asyncio.Task | None = None
        self._stopped = False
        self._delivery_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._next_due_at: datetime | None = None

    async def start(self) -> None:
        await asyncio.to_thread(_ensure_private_root, self.root)
        self._stopped = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="reminder-scheduler")

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def schedule(
        self,
        *,
        when_iso: str,
        text: str,
        channel: str | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(when_iso, str) or len(when_iso) > 128:
            return {"ok": False, "error": "invalid ISO datetime"}
        try:
            due = _utc(when_iso)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"invalid ISO datetime: {exc}"}
        try:
            reminder_text = _bounded_text(text, label="reminder text", maximum=_MAX_REMINDER_TEXT)
            resolved_channel = _bounded_text(
                channel or self.default_channel,
                label="reminder channel",
                maximum=_MAX_ROUTE_CHARS,
            )
        except ValueError as error:
            return {"ok": False, "error": str(error)}
        resolved_target_raw = target or self.default_target
        resolved_target = (
            str(resolved_target_raw).strip() if resolved_target_raw is not None else ""
        )
        if not resolved_target:
            return {"ok": False, "error": "reminder_target_required"}
        try:
            resolved_target = _bounded_text(
                resolved_target,
                label="reminder target",
                maximum=_MAX_ROUTE_CHARS,
            )
        except ValueError as error:
            return {"ok": False, "error": str(error)}
        if not self.outbound.has(resolved_channel):
            return {
                "ok": False,
                "error": "channel_not_registered",
                "channel": resolved_channel,
            }

        try:
            await asyncio.to_thread(_ensure_private_root, self.root)
        except OSError:
            return {"ok": False, "error": "reminder_root_is_symlink"}
        existing = await asyncio.to_thread(_reminder_paths, self.root)
        if len(existing) >= _MAX_REMINDER_FILES:
            return {"ok": False, "error": "reminder_capacity_reached"}
        reminder_id = f"reminder-{int(due.timestamp())}-{secrets.token_hex(4)}"
        payload = {
            "schema_version": 2,
            "id": reminder_id,
            "when_iso": due.isoformat(),
            "text": reminder_text,
            "channel": resolved_channel,
            "target": str(resolved_target),
            "created_at": datetime.now(UTC).isoformat(),
            "attempts": 0,
            "delivered": False,
            "next_attempt_at": due.isoformat(),
        }
        path = self.root / f"{reminder_id}.json"
        await asyncio.to_thread(self._write, path, payload)
        self._wake.set()
        return {
            "ok": True,
            "id": reminder_id,
            "scheduled_for": due.isoformat(),
            "channel": resolved_channel,
            "target": str(resolved_target),
            "durable": True,
        }

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        if path.is_symlink():
            raise OSError("reminder files must not be symbolic links")
        rendered = json.dumps(payload, allow_nan=False, indent=2) + "\n"
        if len(rendered.encode("utf-8")) > _MAX_REMINDER_BYTES:
            raise ValueError("reminder exceeds 64 KiB")
        atomic_write_text(path, rendered, durable=True, mode=0o600)

    async def _loop(self) -> None:
        while not self._stopped:
            self._wake.clear()
            tick_failed = False
            try:
                await self.deliver_due()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("reminder scheduler tick failed")
                tick_failed = True
            delay = self.poll_seconds if tick_failed else _EXTERNAL_RESCAN_SECONDS
            if not tick_failed and self._next_due_at is not None:
                delay = max(
                    0.05,
                    (self._next_due_at - datetime.now(UTC)).total_seconds(),
                )
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def deliver_due(self, *, now: datetime | None = None) -> int:
        """Deliver due reminders once. Exposed for deterministic tests."""
        async with self._delivery_lock:
            return await self._deliver_due_unlocked(now=now)

    async def _deliver_due_unlocked(self, *, now: datetime | None = None) -> int:
        now_utc = (now or datetime.now(UTC)).astimezone(UTC)
        delivered = 0
        self._next_due_at = None
        await asyncio.to_thread(_ensure_private_root, self.root)
        paths = await asyncio.to_thread(_reminder_paths, self.root)
        for path in paths:
            try:
                payload = await asyncio.to_thread(_read_payload, path)
                if payload is None:
                    continue
                if int(payload.get("schema_version", 1)) < 2:
                    due = _utc(payload["when_iso"])
                    payload["schema_version"] = 2
                    payload["channel"] = payload.get("channel") or self.default_channel
                    payload["target"] = payload.get("target") or self.default_target
                    attempts = payload.get("attempts", 0)
                    payload["attempts"] = (
                        min(1_000_000, max(0, attempts))
                        if isinstance(attempts, int) and not isinstance(attempts, bool)
                        else 0
                    )
                    delivered_value = payload.get("delivered", payload.get("fired", False))
                    payload["delivered"] = delivered_value is True
                    payload["next_attempt_at"] = payload.get("next_attempt_at") or due.isoformat()
                    # A legacy reminder that is already more than a day late
                    # was never deliverable under the old file-only stub. Do
                    # not surprise the owner with obsolete test/old alerts.
                    if not payload["delivered"] and due.timestamp() < now_utc.timestamp() - 86400:
                        payload["delivered"] = True
                        payload["expired"] = True
                        payload["expired_at"] = now_utc.isoformat()
                        payload["last_result"] = {"ok": False, "error": "expired_legacy_reminder"}
                    await asyncio.to_thread(self._write, path, payload)
                if payload.get("delivered") is True:
                    continue
                next_attempt = _utc(payload.get("next_attempt_at") or payload["when_iso"])
                if next_attempt > now_utc:
                    if self._next_due_at is None or next_attempt < self._next_due_at:
                        self._next_due_at = next_attempt
                    continue
                reminder_channel = _bounded_text(
                    payload.get("channel"),
                    label="reminder channel",
                    maximum=_MAX_ROUTE_CHARS,
                )
                reminder_target = _bounded_text(
                    payload.get("target"),
                    label="reminder target",
                    maximum=_MAX_ROUTE_CHARS,
                )
                reminder_text = _bounded_text(
                    payload.get("text"),
                    label="reminder text",
                    maximum=_MAX_REMINDER_TEXT,
                )
                try:
                    async with asyncio.timeout(self.delivery_timeout_seconds):
                        result = await self.outbound.send(
                            reminder_channel,
                            reminder_target,
                            f"Reminder: {reminder_text}",
                        )
                except asyncio.CancelledError:
                    raise
                except TimeoutError:
                    result = {"ok": False, "error": "delivery_timeout"}
                    log.warning("reminder delivery timed out id=%s", payload.get("id", path.name))
                except Exception as error:  # noqa: BLE001
                    result = {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                    log.warning(
                        "reminder delivery failed id=%s error=%r",
                        payload.get("id", path.name),
                        error,
                    )
                result = _safe_delivery_result(result)
                attempts = payload.get("attempts", 0)
                if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
                    attempts = 0
                payload["attempts"] = min(1_000_000, attempts + 1)
                payload["last_result"] = result
                if result["ok"] is True:
                    payload["delivered"] = True
                    payload["delivered_at"] = now_utc.isoformat()
                    delivered += 1
                else:
                    delay = min(300, 2 ** min(payload["attempts"], 8))
                    next_attempt = datetime.fromtimestamp(now_utc.timestamp() + delay, UTC)
                    payload["next_attempt_at"] = next_attempt.isoformat()
                    if self._next_due_at is None or next_attempt < self._next_due_at:
                        self._next_due_at = next_attempt
                await asyncio.to_thread(self._write, path, payload)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                log.warning("reminder invalid path=%s error=%r", path, exc)
        return delivered
