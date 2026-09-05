"""Phase 9 — CronAdapter + norax.sleep CLI.

Deterministic clock; no real asyncio sleeps in scheduling tests.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from norax.adapter.cron_in import CronAdapter, CronJob
from norax.config.loader import (
    Config,
    _parse_cron,
)


# ---------------------------------------------------------------------------
# Schedule math
# ---------------------------------------------------------------------------
class TestSchedule:
    def test_every_seconds_first_fire(self):
        j = CronJob(name="tick", schedule={"every_seconds": 60})
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        nxt = j.compute_next(now=now)
        assert nxt == now + timedelta(seconds=60)

    def test_every_seconds_advances(self):
        j = CronJob(name="tick", schedule={"every_seconds": 60})
        j._next_fire = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        now = datetime(2026, 4, 23, 12, 2, 30, tzinfo=UTC)
        # already past 12:00 and 12:01 — next is 12:03
        nxt = j.compute_next(now=now)
        assert nxt == datetime(2026, 4, 23, 12, 3, 0, tzinfo=UTC)

    def test_every_seconds_far_behind_catches_up_exactly(self):
        """Catch-up stays correct even when billions of windows were missed."""
        j = CronJob(name="tick", schedule={"every_seconds": 0.001})
        j._next_fire = datetime(2026, 1, 1, tzinfo=UTC)
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)

        nxt = j.compute_next(now=now)

        assert nxt == now + timedelta(milliseconds=1)

    @pytest.mark.parametrize("step", [0, -1, True, float("nan"), float("inf")])
    def test_every_seconds_rejects_non_positive_or_non_finite_values(self, step):
        j = CronJob(name="bad", schedule={"every_seconds": step})
        with pytest.raises(ValueError, match="every_seconds"):
            j.compute_next(now=datetime(2026, 4, 23, tzinfo=UTC))

    def test_at_same_day(self):
        j = CronJob(name="daily", schedule={"at": "14:30"})
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        nxt = j.compute_next(now=now)
        assert nxt == datetime(2026, 4, 23, 14, 30, 0, tzinfo=UTC)

    def test_at_rolls_to_next_day(self):
        j = CronJob(name="daily", schedule={"at": "06:00"})
        now = datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC)
        nxt = j.compute_next(now=now)
        assert nxt == datetime(2026, 4, 24, 6, 0, 0, tzinfo=UTC)

    def test_hourly_at_minute(self):
        j = CronJob(name="hr", schedule={"hourly_at_minute": 17})
        now = datetime(2026, 4, 23, 12, 5, 0, tzinfo=UTC)
        assert j.compute_next(now=now) == datetime(2026, 4, 23, 12, 17, 0, tzinfo=UTC)
        now = datetime(2026, 4, 23, 12, 30, 0, tzinfo=UTC)
        assert j.compute_next(now=now) == datetime(2026, 4, 23, 13, 17, 0, tzinfo=UTC)

    @pytest.mark.parametrize("minute", [-1, 60, 1.5, True, float("inf")])
    def test_hourly_rejects_invalid_minute(self, minute):
        j = CronJob(name="bad", schedule={"hourly_at_minute": minute})
        with pytest.raises(ValueError, match="hourly_at_minute"):
            j.compute_next(now=datetime(2026, 4, 23, tzinfo=UTC))

    def test_weekly(self):
        # Thursday 10:00 wanted; "now" is Thursday 09:00 → same-day 10:00
        now = datetime(2026, 4, 23, 9, 0, 0, tzinfo=UTC)  # Thu
        j = CronJob(name="wk", schedule={"weekly": {"day": "thu", "at": "10:00"}})
        assert j.compute_next(now=now) == datetime(2026, 4, 23, 10, 0, 0, tzinfo=UTC)

    def test_weekly_rolls_to_next_week(self):
        now = datetime(2026, 4, 23, 11, 0, 0, tzinfo=UTC)  # Thu 11:00
        j = CronJob(name="wk", schedule={"weekly": {"day": "thu", "at": "10:00"}})
        # already past, next Thu is 4/30
        assert j.compute_next(now=now) == datetime(2026, 4, 30, 10, 0, 0, tzinfo=UTC)

    def test_weekly_different_day(self):
        # Thu → next Mon
        now = datetime(2026, 4, 23, 11, 0, 0, tzinfo=UTC)  # Thu
        j = CronJob(name="wk", schedule={"weekly": {"day": "mon", "at": "09:00"}})
        assert j.compute_next(now=now) == datetime(2026, 4, 27, 9, 0, 0, tzinfo=UTC)

    def test_invalid_schedule_raises(self):
        j = CronJob(name="bad", schedule={"not_a_real_key": 1})
        with pytest.raises(ValueError):
            j.compute_next(now=datetime(2026, 4, 23, tzinfo=UTC))

    def test_ambiguous_schedule_raises(self):
        j = CronJob(name="bad", schedule={"every_seconds": 1, "at": "12:00"})
        with pytest.raises(ValueError, match="exactly one"):
            j.compute_next(now=datetime(2026, 4, 23, tzinfo=UTC))

    def test_invalid_weekday(self):
        j = CronJob(name="bad", schedule={"weekly": {"day": "funday", "at": "10:00"}})
        with pytest.raises(ValueError):
            j.compute_next(now=datetime(2026, 4, 23, tzinfo=UTC))

    def test_invalid_hhmm(self):
        j = CronJob(name="bad", schedule={"at": "25:00"})
        with pytest.raises(ValueError):
            j.compute_next(now=datetime(2026, 4, 23, tzinfo=UTC))

    @pytest.mark.parametrize(
        "schedule",
        [
            {"at": None},
            {"weekly": "monday"},
            {"weekly": {"day": "mon"}},
            {"weekly": {"day": "mon", "at": "09:00", "extra": True}},
        ],
    )
    def test_malformed_schedule_values_raise_value_error(self, schedule):
        j = CronJob(name="bad", schedule=schedule)
        with pytest.raises(ValueError):
            j.compute_next(now=datetime(2026, 4, 23, tzinfo=UTC))


# ---------------------------------------------------------------------------
# Adapter tick semantics (deterministic clock)
# ---------------------------------------------------------------------------
class _Clock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float):
        self.t = self.t + timedelta(seconds=seconds)


async def _null_sleep(_seconds):
    # yield to the event loop so tight loops can be cancelled
    await asyncio.sleep(0)


class TestCronAdapterTick:
    @pytest.mark.asyncio
    async def test_no_fire_before_next_fire_time(self):
        clk = _Clock(datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC))
        a = CronAdapter(
            jobs=[CronJob(name="t", schedule={"every_seconds": 60})],
            clock=clk,
            sleeper=_null_sleep,
        )
        a._schedule_initial()
        # start scheduled next_fire=12:01
        fires = await a.tick()
        assert fires == 0

    @pytest.mark.asyncio
    async def test_single_fire_at_exact_time(self):
        clk = _Clock(datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC))
        a = CronAdapter(
            jobs=[CronJob(name="t", schedule={"every_seconds": 60})],
            clock=clk,
            sleeper=_null_sleep,
        )
        a._schedule_initial()
        clk.advance(60)
        fires = await a.tick()
        assert fires == 1
        env = await asyncio.wait_for(a._queue.get(), timeout=0.5)
        assert env.channel == "schedule"
        assert env.source == "cron"
        assert env.sender.trust is True
        assert env.metadata["job_name"] == "t"
        assert env.metadata["job_kind"] == "tick"

    @pytest.mark.asyncio
    async def test_catches_up_missed_fires_once(self):
        """If we miss many windows (e.g. blocked loop), we fire once per
        job per tick — but the next_fire still marches forward."""
        clk = _Clock(datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC))
        a = CronAdapter(
            jobs=[CronJob(name="t", schedule={"every_seconds": 60})],
            clock=clk,
            sleeper=_null_sleep,
        )
        a._schedule_initial()
        clk.advance(300)  # 5 windows missed
        fires = await a.tick()
        assert fires == 1
        # next_fire must now be strictly after t
        j = a.jobs[0]
        assert j._next_fire is not None
        assert j._next_fire > clk()

    @pytest.mark.asyncio
    async def test_multiple_jobs_fire_independently(self):
        clk = _Clock(datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC))
        a = CronAdapter(
            jobs=[
                CronJob(name="a", schedule={"every_seconds": 60}),
                CronJob(name="b", schedule={"every_seconds": 120}),
            ],
            clock=clk,
            sleeper=_null_sleep,
        )
        a._schedule_initial()
        clk.advance(60)
        assert await a.tick() == 1  # only "a"
        clk.advance(60)
        assert await a.tick() == 2  # "a" and "b"

    @pytest.mark.asyncio
    async def test_envelope_has_owner_tier(self):
        clk = _Clock(datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC))
        a = CronAdapter(
            jobs=[CronJob(name="t", schedule={"every_seconds": 60})],
            owner_id="123",
            clock=clk,
            sleeper=_null_sleep,
        )
        a._schedule_initial()
        clk.advance(60)
        await a.tick()
        env = await asyncio.wait_for(a._queue.get(), timeout=0.5)
        assert env.sender.tier == "owner"
        assert env.sender.id == "123"

    @pytest.mark.asyncio
    async def test_no_jobs_start_is_noop(self):
        a = CronAdapter(jobs=[])
        await a.start()
        assert a._task is None
        await a.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_loop(self):
        clk = _Clock(datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC))

        # sleeper that actually sleeps so the loop is parked
        async def slow(_s):
            await asyncio.sleep(0.01)

        a = CronAdapter(
            jobs=[CronJob(name="t", schedule={"every_seconds": 60})],
            clock=clk,
            sleeper=slow,
            resolution_sec=0.01,
        )
        await a.start()
        await asyncio.sleep(0.05)
        await a.stop()
        assert a._stopped is True

    @pytest.mark.asyncio
    async def test_scheduler_failure_reaches_event_consumer(self):
        async def broken_sleep(_seconds):
            raise LookupError("clockwork broke")

        a = CronAdapter(
            jobs=[CronJob(name="t", schedule={"every_seconds": 60})],
            sleeper=broken_sleep,
        )
        await a.start()

        with pytest.raises(RuntimeError, match="cron scheduler failed") as caught:
            await anext(a.events())

        assert isinstance(caught.value.__cause__, LookupError)
        await a.stop()

    @pytest.mark.asyncio
    async def test_jitter_delays_fire_time_without_blocking_scheduler(self, monkeypatch):
        sleeps: list[float] = []

        async def recording_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr("norax.adapter.cron_in.random.uniform", lambda _low, _high: 7.0)
        clk = _Clock(datetime(2026, 4, 23, 12, 0, 0, tzinfo=UTC))
        a = CronAdapter(
            jobs=[CronJob(name="jittered", schedule={"every_seconds": 60}, jitter_seconds=10)],
            clock=clk,
            sleeper=recording_sleep,
        )

        a._schedule_initial()
        assert a.jobs[0]._next_fire == clk() + timedelta(seconds=67)
        clk.advance(67)
        assert await a.tick() == 1
        assert sleeps == []

    def test_invalid_job_prevents_partial_initial_schedule(self):
        good = CronJob(name="good", schedule={"every_seconds": 60})
        bad = CronJob(name="bad", schedule={"every_seconds": 0})
        a = CronAdapter(jobs=[good, bad])

        with pytest.raises(ValueError, match="bad"):
            a._schedule_initial()

        assert good._next_fire is None

    @pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
    def test_invalid_resolution_rejected(self, value):
        with pytest.raises(ValueError, match="resolution_sec"):
            CronAdapter(resolution_sec=value)

    @pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
    def test_invalid_jitter_rejected(self, value):
        with pytest.raises(ValueError, match="jitter_seconds"):
            CronAdapter(
                jobs=[CronJob(name="bad", schedule={"every_seconds": 1}, jitter_seconds=value)]
            )


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
class TestCronConfig:
    def test_empty_defaults(self):
        c = _parse_cron({})
        assert c.enabled is False
        assert c.jobs == []

    def test_disabled_jobs_dropped_in_from_config(self):
        raw = {
            "enabled": True,
            "jobs": [
                {"name": "a", "schedule": {"every_seconds": 10}, "enabled": True},
                {"name": "b", "schedule": {"every_seconds": 20}, "enabled": False},
            ],
        }
        a = CronAdapter.from_config(raw)
        assert [j.name for j in a.jobs] == ["a"]

    def test_disabled_block_returns_empty_adapter(self):
        a = CronAdapter.from_config(
            {"enabled": False, "jobs": [{"name": "x", "schedule": {"every_seconds": 1}}]}
        )
        assert a.jobs == []

    def test_full_parse(self):
        raw = {
            "enabled": True,
            "resolution_sec": 3.0,
            "jobs": [{"name": "morning", "schedule": {"at": "08:30"}, "kind": "checkin"}],
        }
        c = _parse_cron(raw)
        assert c.enabled is True
        assert c.resolution_sec == 3.0
        assert c.jobs[0].name == "morning"
        assert c.jobs[0].kind == "checkin"

    def test_parse_uses_real_boolean_and_finite_number_semantics(self):
        c = _parse_cron(
            {
                "enabled": "false",
                "resolution_sec": float("nan"),
                "jobs": [
                    {
                        "name": "disabled",
                        "schedule": {"every_seconds": 1},
                        "enabled": "false",
                        "jitter_seconds": float("inf"),
                    }
                ],
            }
        )

        assert c.enabled is False
        assert c.resolution_sec == 5.0
        assert c.jobs[0].enabled is False
        assert c.jobs[0].jitter_seconds == 0.0

    @pytest.mark.parametrize(
        "schedule",
        [
            {},
            {"every_seconds": 0},
            {"at": "25:00"},
            {"hourly_at_minute": 60},
            {"weekly": {"day": "noday", "at": "09:00"}},
            {"at": "09:00", "every_seconds": 60},
        ],
    )
    def test_invalid_schedule_is_rejected_during_config_loading(self, schedule):
        with pytest.raises(ValueError, match="invalid cron schedule"):
            _parse_cron(
                {
                    "enabled": True,
                    "jobs": [{"name": "bad", "schedule": schedule}],
                }
            )

    def test_duplicate_and_unbounded_job_sets_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate job name"):
            _parse_cron(
                {
                    "jobs": [
                        {"name": "same", "schedule": {"every_seconds": 1}},
                        {"name": "same", "schedule": {"every_seconds": 2}},
                    ]
                }
            )

        with pytest.raises(ValueError, match="at most 256"):
            _parse_cron(
                {
                    "jobs": [
                        {"name": f"job-{index}", "schedule": {"every_seconds": 1}}
                        for index in range(257)
                    ]
                }
            )


# ---------------------------------------------------------------------------
# Runtime wiring
# ---------------------------------------------------------------------------
class TestRuntimeCronWiring:
    def test_runtime_build_with_cron_disabled(self, tmp_path):
        from norax.runtime.core import Runtime

        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        cfg = Config(
            raw={"http": {"bind": "127.0.0.1:0"}, "owner": {"id": "1"}, "cron": {"enabled": False}},
            project_root=tmp_path,
        )
        rt = Runtime.build(cfg)
        adapters = rt.ingress._adapters
        assert not any(getattr(a, "name", "") == "cron" for a in adapters)

    def test_runtime_build_with_cron_enabled_and_jobs(self, tmp_path):
        from norax.runtime.core import Runtime

        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        cfg = Config(
            raw={
                "http": {"bind": "127.0.0.1:0"},
                "owner": {"id": "1"},
                "cron": {
                    "enabled": True,
                    "jobs": [{"name": "t", "schedule": {"every_seconds": 60}}],
                },
            },
            project_root=tmp_path,
        )
        rt = Runtime.build(cfg)
        adapters = rt.ingress._adapters
        crons = [a for a in adapters if getattr(a, "name", "") == "cron"]
        assert len(crons) == 1
        assert crons[0].jobs[0].name == "t"
        assert rt.cron is crons[0]

    def test_runtime_does_not_wire_cron_when_all_jobs_disabled(self, tmp_path):
        from norax.runtime.core import Runtime

        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        cfg = Config(
            raw={
                "http": {"bind": "127.0.0.1:0"},
                "owner": {"id": "1"},
                "cron": {
                    "enabled": True,
                    "jobs": [
                        {
                            "name": "disabled",
                            "schedule": {"every_seconds": 60},
                            "enabled": False,
                        }
                    ],
                },
            },
            project_root=tmp_path,
        )

        rt = Runtime.build(cfg)

        assert not any(getattr(a, "name", "") == "cron" for a in rt.ingress._adapters)
        assert rt.cron is None


# ---------------------------------------------------------------------------
# norax.sleep CLI
# ---------------------------------------------------------------------------
class TestSleepCLI:
    def test_cli_no_memory_root_exits_ok(self, tmp_path, monkeypatch, capsys):
        from norax import sleep as sleep_mod

        monkeypatch.setenv("NORAX_MEMORY_ROOT", str(tmp_path / "nope"))
        rc = sleep_mod.main(["--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out.strip())
        assert out["ok"] is True
        assert out["skipped"] == "no_memory_root"

    def test_cli_empty_memory_root_flush(self, tmp_path, monkeypatch, capsys):
        from norax import sleep as sleep_mod

        mem = tmp_path / "memory"
        (mem / "sleep").mkdir(parents=True)
        (mem / "semantic").mkdir()
        (mem / "procedural").mkdir()
        (mem / "intel").mkdir()
        monkeypatch.setenv("NORAX_MEMORY_ROOT", str(mem))
        rc = sleep_mod.main(["--json", "--min-age", "0"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out.strip())
        assert out["ok"] is True
        assert out["spills_processed"] == 0
        assert out["candidates_seen"] == 0

    def test_cli_dry_run_flag(self, tmp_path, monkeypatch, capsys):
        from norax import sleep as sleep_mod

        mem = tmp_path / "memory"
        (mem / "sleep").mkdir(parents=True)
        monkeypatch.setenv("NORAX_MEMORY_ROOT", str(mem))
        rc = sleep_mod.main(["--json", "--dry-run", "--min-age", "0"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out.strip())
        assert out["dry_run"] is True
