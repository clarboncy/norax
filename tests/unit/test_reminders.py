from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from norax.runtime.reminders import ReminderScheduler


class _Outbound:
    def __init__(self, results=None):
        self.results = list(results or [{"ok": True, "message_id": "1"}])
        self.sent = []

    def has(self, channel):
        return channel == "discord"

    async def send(self, channel, target, text, **kwargs):
        self.sent.append((channel, target, text, kwargs))
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_reminder_is_durable_and_delivered_after_restart(tmp_path):
    outbound = _Outbound()
    due = datetime.now(UTC) - timedelta(seconds=1)
    first = ReminderScheduler(root=tmp_path, outbound=outbound, default_target="owner")
    result = await first.schedule(when_iso=due.isoformat(), text="check harness")
    assert result["ok"] and result["durable"]

    restarted = ReminderScheduler(root=tmp_path, outbound=outbound, default_target="owner")
    assert await restarted.deliver_due() == 1
    assert outbound.sent[0][:3] == ("discord", "owner", "Reminder: check harness")
    saved = json.loads(next(tmp_path.glob("reminder-*.json")).read_text())
    assert saved["delivered"] is True


@pytest.mark.asyncio
async def test_reminder_retries_failed_delivery(tmp_path):
    outbound = _Outbound([{"ok": False, "error": "offline"}, {"ok": True}])
    now = datetime.now(UTC)
    scheduler = ReminderScheduler(root=tmp_path, outbound=outbound, default_target="owner")
    await scheduler.schedule(when_iso=(now - timedelta(seconds=1)).isoformat(), text="retry")
    assert await scheduler.deliver_due(now=now) == 0
    assert await scheduler.deliver_due(now=now + timedelta(minutes=10)) == 1


@pytest.mark.asyncio
async def test_reminder_transport_exception_is_persisted_with_backoff(tmp_path):
    class _FailingOutbound(_Outbound):
        async def send(self, channel, target, text, **kwargs):
            raise ConnectionError("offline")

    now = datetime.now(UTC)
    scheduler = ReminderScheduler(
        root=tmp_path,
        outbound=_FailingOutbound(),
        default_target="owner",
    )
    await scheduler.schedule(when_iso=(now - timedelta(seconds=1)).isoformat(), text="retry")

    assert await scheduler.deliver_due(now=now) == 0

    saved = json.loads(next(tmp_path.glob("reminder-*.json")).read_text())
    assert saved["attempts"] == 1
    assert saved["delivered"] is False
    assert saved["last_result"]["ok"] is False
    assert saved["last_result"]["error"] == "ConnectionError: offline"
    assert datetime.fromisoformat(saved["next_attempt_at"]) > now


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_reminder_poll_interval_rejects_invalid_values(tmp_path, value):
    with pytest.raises(ValueError, match="poll_seconds"):
        ReminderScheduler(root=tmp_path, outbound=_Outbound(), poll_seconds=value)


@pytest.mark.asyncio
async def test_reminder_rejects_missing_route(tmp_path):
    scheduler = ReminderScheduler(root=tmp_path, outbound=_Outbound(), default_target=None)
    result = await scheduler.schedule(when_iso=datetime.now(UTC).isoformat(), text="x")
    assert result == {"ok": False, "error": "reminder_target_required"}


@pytest.mark.asyncio
async def test_stale_legacy_reminder_is_migrated_without_late_delivery(tmp_path):
    now = datetime.now(UTC)
    legacy = tmp_path / "reminder-legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "id": "reminder-legacy",
                "when_iso": (now - timedelta(days=2)).isoformat(),
                "text": "obsolete",
                "fired": False,
            }
        )
    )
    outbound = _Outbound()
    scheduler = ReminderScheduler(root=tmp_path, outbound=outbound, default_target="owner")
    assert await scheduler.deliver_due(now=now) == 0
    migrated = json.loads(legacy.read_text())
    assert migrated["schema_version"] == 2
    assert migrated["expired"] is True
    assert migrated["delivered"] is True
    assert outbound.sent == []


@pytest.mark.asyncio
async def test_legacy_string_false_is_not_treated_as_delivered(tmp_path):
    now = datetime.now(UTC)
    legacy = tmp_path / "reminder-legacy-false.json"
    legacy.write_text(
        json.dumps(
            {
                "id": "reminder-legacy-false",
                "when_iso": (now - timedelta(seconds=1)).isoformat(),
                "text": "still due",
                "fired": "false",
            }
        )
    )
    outbound = _Outbound()
    scheduler = ReminderScheduler(root=tmp_path, outbound=outbound, default_target="owner")

    assert await scheduler.deliver_due(now=now) == 1
    assert outbound.sent[0][2] == "Reminder: still due"


@pytest.mark.asyncio
async def test_reminder_delivery_timeout_is_bounded_and_retried(tmp_path):
    class _HungOutbound(_Outbound):
        async def send(self, channel, target, text, **kwargs):
            await asyncio.Event().wait()

    now = datetime.now(UTC)
    scheduler = ReminderScheduler(
        root=tmp_path,
        outbound=_HungOutbound(),
        default_target="owner",
        delivery_timeout_seconds=0.02,
    )
    await scheduler.schedule(when_iso=(now - timedelta(seconds=1)).isoformat(), text="bounded")

    await asyncio.wait_for(scheduler.deliver_due(now=now), timeout=0.5)

    saved = json.loads(next(tmp_path.glob("reminder-*.json")).read_text())
    assert saved["delivered"] is False
    assert saved["last_result"] == {"ok": False, "error": "delivery_timeout"}


@pytest.mark.asyncio
async def test_idle_scheduler_does_not_rescan_every_poll_interval(tmp_path, monkeypatch):
    from norax.runtime import reminders as reminder_module

    scans = 0
    original = reminder_module._reminder_paths

    def counted(root):
        nonlocal scans
        scans += 1
        return original(root)

    monkeypatch.setattr(reminder_module, "_reminder_paths", counted)
    scheduler = ReminderScheduler(root=tmp_path, outbound=_Outbound(), poll_seconds=0.1)
    await scheduler.start()
    await asyncio.sleep(0.25)
    await scheduler.stop()

    assert scans == 1


@pytest.mark.asyncio
async def test_reminder_reader_ignores_oversized_files_and_symlinks(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"text":"must not load"}')
    os.symlink(outside, tmp_path / "reminder-link.json")
    (tmp_path / "reminder-oversized.json").write_bytes(b"x" * 65_537)
    scheduler = ReminderScheduler(root=tmp_path, outbound=_Outbound(), default_target="owner")

    assert await scheduler.deliver_due() == 0


@pytest.mark.asyncio
async def test_reminder_inputs_are_bounded(tmp_path):
    scheduler = ReminderScheduler(root=tmp_path, outbound=_Outbound(), default_target="owner")

    result = await scheduler.schedule(
        when_iso=datetime.now(UTC).isoformat(),
        text="x" * 16_385,
    )

    assert result["ok"] is False
    assert "reminder text" in result["error"]
