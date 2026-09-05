"""CronAdapter — in-process scheduler that emits SensoryInput envelopes.

Philosophy (per ongoing.md §6): cron is a sensory source, not a
background poller. A scheduled trigger fires an envelope into the same
ingress bus as Discord/HTTP; the brain decides what to do.

Supported schedule shapes (no external deps):
  - `{"every_seconds": 300}`           — every N seconds after start
  - `{"at": "HH:MM"}`                  — every day at wall-clock time (local)
  - `{"weekly": {"day": "mon", "at": "09:00"}}` — once a week at local time
  - `{"hourly_at_minute": 5}`          — every hour at minute N

Day names: sun|mon|tue|wed|thu|fri|sat (case-insensitive, 3-letter prefix OK).

Each job:
  - `name: str` (used as message_id prefix, and envelope body)
  - `schedule: dict` (one of the shapes above)
  - `kind: str` = "heartbeat" | "tick" | "check" | ... (free-form, goes
    into envelope.metadata.job_kind)
  - `jitter_seconds: float` = 0.0  (random delay added to fire time to
    avoid thundering-herd)

Stop semantics: `stop()` cancels the internal task; fires are not
attempted again after stop.

No wall-clock surprises: the scheduler is tick-based (wakes every
`resolution_sec` seconds, checks each job's next-fire). Default 5s
resolution is plenty for everything we'd want inside a personal agent.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import ulid

_ulid_new_compat = getattr(ulid, "new", None) or (lambda: str(ulid.ULID()))

from ..envelope import Principal, SensoryInput  # noqa: E402

log = logging.getLogger("norax.adapter.cron")


_DAY_IDX = {
    "sun": 6,
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    # Python's weekday(): Mon=0..Sun=6
}


def _norm_day(d: str) -> int:
    if not isinstance(d, str):
        raise ValueError(f"weekday must be a string, got {type(d).__name__}")
    key = d.strip().lower()[:3]
    if key not in _DAY_IDX:
        raise ValueError(f"invalid weekday: {d!r}")
    return _DAY_IDX[key]


def _parse_hhmm(s: str) -> tuple[int, int]:
    if not isinstance(s, str):
        raise ValueError(f"HH:MM must be a string, got {type(s).__name__}")
    try:
        h, m = s.strip().split(":", 1)
        hh, mm = int(h), int(m)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid HH:MM: {s!r}") from error
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"invalid HH:MM: {s!r}")
    return hh, mm


def _finite_number(value: Any, *, label: str, minimum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number >= {minimum}") from error
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    return parsed


def _minute(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("hourly_at_minute must be an integer from 0 to 59")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("hourly_at_minute must be an integer from 0 to 59") from error
    if not math.isfinite(numeric) or not numeric.is_integer() or not 0 <= numeric < 60:
        raise ValueError("hourly_at_minute must be an integer from 0 to 59")
    return int(numeric)


@dataclass
class CronJob:
    name: str
    schedule: dict
    kind: str = "tick"
    jitter_seconds: float = 0.0
    owner_id: str | None = None
    # Internal:
    _next_fire: datetime | None = None
    _next_nominal: datetime | None = None

    def compute_next(self, *, now: datetime) -> datetime:
        """Return the next wall-clock fire time at-or-after `now`."""
        return self._compute_next(now=now, previous=self._next_fire)

    def _compute_next(self, *, now: datetime, previous: datetime | None) -> datetime:
        s = self.schedule
        if not isinstance(s, dict) or len(s) != 1:
            raise ValueError(f"schedule must contain exactly one supported shape: {s!r}")

        if "every_seconds" in s:
            step = _finite_number(s["every_seconds"], label="every_seconds", minimum=0.001)
            delta = timedelta(seconds=step)
            if previous is None:
                return now + delta
            # Constant-time catch-up. The old while-loop could perform millions
            # of iterations after downtime when the interval was short.
            nxt = previous
            if nxt <= now:
                elapsed = (now - nxt).total_seconds()
                missed = math.floor(elapsed / step) + 1
                try:
                    nxt = nxt + (delta * missed)
                except OverflowError as error:
                    raise ValueError("every_seconds schedule exceeds datetime range") from error
                # Protect against float/timedelta rounding at the boundary.
                if nxt <= now:
                    nxt = nxt + delta
            return nxt

        if "at" in s:
            hh, mm = _parse_hhmm(s["at"])
            nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if nxt <= now:
                nxt = nxt + timedelta(days=1)
            return nxt

        if "hourly_at_minute" in s:
            mm = _minute(s["hourly_at_minute"])
            nxt = now.replace(minute=mm, second=0, microsecond=0)
            if nxt <= now:
                nxt = nxt + timedelta(hours=1)
            return nxt

        if "weekly" in s:
            w = s["weekly"]
            if not isinstance(w, dict) or set(w) != {"day", "at"}:
                raise ValueError("weekly schedule must contain exactly 'day' and 'at'")
            day_idx = _norm_day(w["day"])
            hh, mm = _parse_hhmm(w["at"])
            # Python: weekday() Mon=0..Sun=6
            days_ahead = (day_idx - now.weekday()) % 7
            candidate = (now + timedelta(days=days_ahead)).replace(
                hour=hh, minute=mm, second=0, microsecond=0
            )
            if candidate <= now:
                candidate = candidate + timedelta(days=7)
            return candidate

        raise ValueError(f"unrecognized schedule shape: {s!r}")


@dataclass
class CronAdapter:
    failure_is_fatal = False

    name: str = "cron"
    jobs: list[CronJob] = field(default_factory=list)
    resolution_sec: float = 5.0
    owner_id: str | None = None
    # For tests: inject a clock and a scheduler.
    clock: Any = None  # callable returning `datetime` aware-UTC
    sleeper: Any = None  # async callable(seconds) -> None

    _queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    _task: asyncio.Task | None = field(default=None, init=False)
    _stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.resolution_sec = _finite_number(
            self.resolution_sec,
            label="resolution_sec",
            minimum=0.001,
        )
        for job in self.jobs:
            job.jitter_seconds = _finite_number(
                job.jitter_seconds,
                label=f"cron job {job.name!r} jitter_seconds",
                minimum=0.0,
            )
        if self.clock is None:
            self.clock = lambda: datetime.now(UTC).astimezone()
        if self.sleeper is None:
            self.sleeper = asyncio.sleep

    # ------------------------------------------------------------------
    # Config loader
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, raw: dict, *, owner_id: str | None = None) -> CronAdapter:
        if not raw or not raw.get("enabled", False):
            return cls(jobs=[], owner_id=owner_id)
        jobs: list[CronJob] = []
        for j in raw.get("jobs") or []:
            if not j.get("enabled", True):
                continue
            jobs.append(
                CronJob(
                    name=str(j.get("name") or f"job-{len(jobs)}"),
                    schedule=j.get("schedule") or {},
                    kind=str(j.get("kind") or "tick"),
                    jitter_seconds=float(j.get("jitter_seconds") or 0.0),
                    owner_id=owner_id,
                )
            )
        return cls(
            jobs=jobs,
            resolution_sec=float(raw.get("resolution_sec", 5.0)),
            owner_id=owner_id,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("CronAdapter.start() may only be called once")
        if self._stopped:
            raise RuntimeError("CronAdapter cannot restart after stop()")
        if not self.jobs:
            log.info("cron.start skipped: no jobs")
            return
        self._schedule_initial()
        self._task = asyncio.create_task(self._loop(), name="cron-loop")

    def _schedule_initial(self) -> None:
        now = self.clock()
        scheduled: list[tuple[CronJob, datetime, datetime]] = []
        for j in self.jobs:
            try:
                nominal = j._compute_next(now=now, previous=None)
                scheduled.append((j, nominal, self._with_jitter(j, nominal)))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"invalid cron schedule for job {j.name!r}") from error
        for job, nominal, next_fire in scheduled:
            job._next_nominal = nominal
            job._next_fire = next_fire
            log.info("cron.job.scheduled name=%s next=%s", job.name, next_fire)

    @staticmethod
    def _with_jitter(job: CronJob, nominal: datetime) -> datetime:
        if job.jitter_seconds <= 0:
            return nominal
        delay = random.uniform(0, job.jitter_seconds)
        try:
            return nominal + timedelta(seconds=delay)
        except OverflowError as error:
            raise ValueError(f"cron jitter for job {job.name!r} exceeds datetime range") from error

    async def stop(self) -> None:
        self._stopped = True
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.warning("cron loop failed during shutdown", exc_info=True)
            finally:
                self._task = None

    async def events(self) -> AsyncIterator[SensoryInput]:
        while not self._stopped:
            loop_task = self._task
            if loop_task is None:
                return
            queued = asyncio.create_task(self._queue.get())
            try:
                done, _pending = await asyncio.wait(
                    {queued, loop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queued in done:
                    yield queued.result()
                    continue
                if self._stopped:
                    return
                if loop_task.cancelled():
                    raise RuntimeError("cron scheduler was cancelled unexpectedly")
                error = loop_task.exception()
                if error is not None:
                    raise RuntimeError("cron scheduler failed") from error
                raise RuntimeError("cron scheduler stopped unexpectedly")
            finally:
                if not queued.done():
                    queued.cancel()
                    await asyncio.gather(queued, return_exceptions=True)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    async def _loop(self) -> None:
        try:
            while not self._stopped:
                await self.tick()
                await self.sleeper(self.resolution_sec)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("cron._loop crashed")
            raise

    async def tick(self) -> int:
        """One scheduler tick. Returns number of fires. Exposed for tests."""
        now = self.clock()
        fires = 0
        for j in self.jobs:
            if j._next_fire is None:
                continue
            if now >= j._next_fire:
                await self._fire(j, at=now)
                nominal = j._compute_next(now=now, previous=j._next_nominal)
                j._next_nominal = nominal
                j._next_fire = self._with_jitter(j, nominal)
                fires += 1
        return fires

    async def _fire(self, job: CronJob, *, at: datetime) -> None:
        mid = f"cron-{job.name}-{_ulid_new_compat()}"
        principal = Principal(
            id=self.owner_id or "system",
            label="cron",
            trust=True,
            tier="owner" if self.owner_id else "admin",
        )
        env = SensoryInput(
            channel="schedule",
            source=self.name,
            message_id=mid,
            timestamp=at,
            sender=principal,
            body=f"[cron:{job.kind}] {job.name}",
            trusted=True,
            metadata={
                "job_name": job.name,
                "job_kind": job.kind,
                "fired_at": at.isoformat(),
                "schedule": job.schedule,
            },
        )
        try:
            self._queue.put_nowait(env)
        except asyncio.QueueFull:
            log.warning("cron.back_pressure_drop name=%s", job.name)
