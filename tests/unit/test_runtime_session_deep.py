"""Integrity contracts for context ownership, cancellation, and command dispatch."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from norax import commands as command_module
from norax.context.window import EvictionResult, Frame, RollingWindow
from norax.envelope import Principal
from norax.runtime import session as session_module
from norax.runtime.session import SessionMixin, _safe_command_error


class _Events:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []

    async def append(self, kind: str, payload: dict[str, Any]) -> None:
        self.rows.append((kind, payload))


class _BrainErrors:
    def __init__(self) -> None:
        self.labels_seen: list[str] = []
        self.count = 0

    def labels(self, *, where: str) -> _BrainErrors:
        self.labels_seen.append(where)
        return self

    def inc(self) -> None:
        self.count += 1


class _Metrics:
    def __init__(self) -> None:
        self.brain_errors = _BrainErrors()


class _Task:
    def __init__(self, *, done: bool) -> None:
        self._done = done
        self.cancelled = False

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancelled = True


def _noop(*_args: object) -> None:
    return None


def _session(tmp_path: Path) -> SessionMixin:
    runtime = SessionMixin()
    runtime.cfg = SimpleNamespace(memory_root=tmp_path)
    runtime._windows = {}
    runtime._active_turn_tasks = {}
    runtime._turn_queues = {}
    runtime._stop_channels = set()
    runtime._turn_work_count = 0
    runtime._memory_store = None
    runtime._episodic = None
    runtime.gateway = SimpleNamespace(
        base_url="http://gateway.test/v1",
        provider_urls={"local": "http://local.test/v1"},
    )
    runtime.default_model = "model"
    runtime.set_default_model = _noop
    runtime._started_mono = 10.0
    runtime._started_wall = 20.0
    runtime.metrics = _Metrics()
    runtime.events = _Events()
    runtime.outbound = object()
    runtime.thinking_effort = "high"
    runtime.set_thinking_effort = _noop
    runtime.reasoning_output = True
    runtime.set_reasoning_output = _noop
    runtime.planning_mode = "orchestrator"
    runtime.set_planning_mode = _noop
    runtime.max_tool_rounds = 250
    runtime.set_max_tool_rounds = _noop
    runtime.memory_depth = "deep"
    runtime.set_memory_depth = _noop
    runtime.weak_model_boost = "on"
    runtime.set_weak_model_boost = _noop
    runtime.stream_replies = True
    runtime.set_stream_replies = _noop
    runtime.response_length = "detailed"
    runtime.set_response_length = _noop
    runtime.tool_activity = "verbose"
    runtime.set_tool_activity = _noop
    runtime.custom_model_catalog = ["model"]
    runtime.sent = []
    runtime.delivery_succeeds = True

    async def send_command_reply(env: object, text: str, *, reply_to: str | None) -> bool:
        runtime.sent.append((env, text, reply_to))
        return runtime.delivery_succeeds

    runtime._send_command_reply = send_command_reply
    return runtime


def test_command_diagnostic_is_redacted_and_bounded() -> None:
    secret = "sk-" + ("a" * 32)
    diagnostic = _safe_command_error(RuntimeError(f"key={secret} " + ("x" * 1_000)))
    assert secret not in diagnostic
    assert "<REDACTED:openai_key>" in diagnostic
    assert diagnostic.startswith("RuntimeError:")
    assert len(diagnostic) == 500


def test_window_paths_are_stable_ascii_and_independent_of_model(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    assert runtime._window_path("channel_1") == tmp_path / "state/windows/channel_1.json"

    unsafe = runtime._window_path("room/東/alpha")
    empty = runtime._window_path("")
    assert unsafe.parent == tmp_path / "state/windows"
    assert unsafe.name.startswith("room___alpha-")
    assert unsafe.name.isascii()
    assert empty.name.startswith("channel-")

    runtime.cfg = SimpleNamespace(memory_root="relative")
    assert runtime._window_path("channel_1") == Path("state/windows/channel_1.json")


def test_get_and_persist_window_use_one_canonical_budget(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    window = runtime._get_window("chat")
    assert window.budget_tokens == 256_000
    assert window.protect_tail_turns == 8
    assert runtime._get_window("chat") is window

    window.add_user("remember this")
    runtime._persist_window("chat")
    restored = RollingWindow.load(runtime._window_path("chat"))
    assert [frame.content for frame in restored.body] == ["remember this"]
    runtime._persist_window("absent")


def test_persist_failure_is_contained_and_scrubbed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-" + ("b" * 32)

    class BrokenWindow:
        def save(self, _path: Path) -> None:
            raise OSError(f"cannot save with {secret}")

    runtime = _session(tmp_path)
    runtime._windows["chat"] = BrokenWindow()  # type: ignore[assignment]
    runtime._persist_window("chat")
    assert secret not in " ".join(caplog.messages)
    assert "<REDACTED:openai_key>" in " ".join(caplog.messages)


def test_reset_removes_memory_persistence_and_broken_symlinks(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    assert runtime._reset_window("missing") is False

    runtime._windows["memory-only"] = RollingWindow()
    assert runtime._reset_window("memory-only") is True
    assert "memory-only" not in runtime._windows

    persisted_path = runtime._window_path("disk-only")
    RollingWindow().save(persisted_path)
    assert runtime._reset_window("disk-only") is True
    assert not persisted_path.exists()

    broken_link = runtime._window_path("broken-link")
    broken_link.parent.mkdir(parents=True, exist_ok=True)
    broken_link.symlink_to(tmp_path / "does-not-exist")
    assert not broken_link.exists()
    assert runtime._reset_window("broken-link") is True
    assert not broken_link.is_symlink()


def test_failed_persisted_reset_retains_live_context(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    window = RollingWindow()
    runtime._windows["blocked"] = window
    path = runtime._window_path("blocked")
    path.mkdir(parents=True)

    assert runtime._reset_window("blocked") is False
    assert runtime._windows["blocked"] is window
    assert path.is_dir()


def test_dump_empty_full_half_and_no_spill_result(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    assert runtime._dump_window("empty") is None

    full = runtime._get_window("full")
    full.sleep_dir = tmp_path / "sleep-full"
    full.add_user("first")
    result = runtime._dump_window("full")
    assert result is not None
    assert result["mode"] == "full"
    assert result["frames"] == 1
    assert result["remaining_frames"] == 0
    assert result["spill_path"] is not None

    half = runtime._get_window("half")
    half.sleep_dir = tmp_path / "sleep-half"
    half.add_user("older")
    half.add_user("newer")
    result = runtime._dump_window("half", half=True)
    assert result is not None
    assert result["mode"] == "half"
    assert result["before_frames"] == 2
    assert result["remaining_frames"] == 1

    saved: list[Path] = []

    class NoSpillWindow:
        body = [Frame(kind="user", content="x")]

        def total_tokens(self) -> int:
            return len(self.body)

        def dump_to_sleep(self, *, half: bool) -> EvictionResult:
            assert half is False
            self.body.clear()
            return EvictionResult(
                evicted=[Frame(kind="user", content="x")], spill_path=None, tokens_freed=1
            )

        def save(self, path: Path) -> None:
            saved.append(path)

    runtime._windows["no-spill"] = NoSpillWindow()  # type: ignore[assignment]
    result = runtime._dump_window("no-spill")
    assert result is not None
    assert result["spill_path"] is None
    assert saved == [runtime._window_path("no-spill")]


def test_cancel_all_skips_finished_tasks_and_clears_queued_work(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    first = _Task(done=False)
    second = _Task(done=False)
    finished = _Task(done=True)
    runtime._active_turn_tasks = {"one": first, "two": second, "done": finished}
    runtime._turn_queues = {"one": deque([1, 2]), "two": deque()}
    runtime._turn_work_count = 1

    assert runtime._cancel_channel() is True
    assert first.cancelled and second.cancelled
    assert not finished.cancelled
    assert runtime._turn_work_count == 0
    assert runtime._stop_channels == {"one", "two"}
    assert runtime._active_turn_tasks == {"done": finished}
    assert runtime._turn_queues == {}

    assert runtime._cancel_channel() is False


def test_cancel_specific_active_finished_and_missing_channels(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    active = _Task(done=False)
    runtime._active_turn_tasks["active"] = active
    runtime._turn_queues["active"] = deque([1])
    runtime._turn_work_count = 4
    assert runtime._cancel_channel("active") is True
    assert active.cancelled
    assert runtime._turn_work_count == 3

    no_queue = _Task(done=False)
    runtime._active_turn_tasks["no-queue"] = no_queue
    assert runtime._cancel_channel("no-queue") is True

    finished = _Task(done=True)
    runtime._active_turn_tasks["finished"] = finished
    runtime._stop_channels.add("finished")
    assert runtime._cancel_channel("finished") is False
    assert "finished" not in runtime._active_turn_tasks
    assert "finished" not in runtime._stop_channels
    assert runtime._cancel_channel("missing") is False


def test_window_and_brain_stats_cover_empty_and_populated_components(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    assert runtime._get_window_stats("missing") is None

    window = RollingWindow(budget_tokens=0)
    window.add_system("system")
    window.add_user("body")
    runtime._windows["chat"] = window
    stats = runtime._get_window_stats("chat")
    assert stats == {
        "current_tokens": 2,
        "budget_tokens": 0,
        "head_tokens": 1,
        "body_frames": 1,
        "usage_pct": 200.0,
    }

    assert runtime._get_brain_stats() == {}
    runtime._memory_store = SimpleNamespace(
        hot=[1, 2],
        sleep=[1],
        all_canonical=lambda: [1, 2, 3],
    )
    runtime._episodic = SimpleNamespace(stats=lambda: {"episodes": 4})
    runtime._hebbian = SimpleNamespace(turn_count=5)
    assert runtime._get_brain_stats() == {
        "memory": {"hot": 2, "canonical": 3, "sleep": 1},
        "episodic": {"episodes": 4},
        "hebbian_turns": 5,
    }


def test_runtime_handle_exposes_complete_live_configuration(tmp_path: Path) -> None:
    runtime = _session(tmp_path)
    handle = runtime._build_runtime_handle()
    assert handle.default_model == "model"
    assert handle.provider_urls == {"local": "http://local.test/v1"}
    assert handle.max_tool_rounds == 250
    assert handle.thinking_effort == "high"
    assert handle.reset_window_for("missing") is False

    del runtime.gateway.provider_urls
    assert runtime._build_runtime_handle().provider_urls is None


def _env() -> SimpleNamespace:
    return SimpleNamespace(
        sender=SimpleNamespace(id="owner", tier="owner"),
        raw={"channel_id": "chat"},
        message_id="message-1",
    )


@pytest.mark.asyncio
async def test_command_dispatch_covers_delivery_and_post_send_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _session(tmp_path)
    effects: list[str] = []

    async def no_effect() -> command_module.CommandResult:
        return command_module.CommandResult(reply="")

    async def handle_no_effect(*_args: object) -> command_module.CommandResult:
        return await no_effect()

    monkeypatch.setattr(session_module.cmd_mod, "handle", handle_no_effect)
    await runtime._handle_command(_env(), command_module.ParsedCommand("status", [], "/status"))
    assert runtime.sent[-1][2] is None

    async def effect() -> None:
        effects.append("ran")

    async def handle_effect(*_args: object) -> command_module.CommandResult:
        return command_module.CommandResult(reply="queued", post_send=effect)

    monkeypatch.setattr(session_module.cmd_mod, "handle", handle_effect)
    await runtime._handle_command(_env(), command_module.ParsedCommand("restart", [], "/restart"))
    assert effects == ["ran"]
    assert runtime.sent[-1][2] == "message-1"

    async def broken_effect() -> None:
        raise RuntimeError("post-send failed")

    async def handle_broken_effect(*_args: object) -> command_module.CommandResult:
        return command_module.CommandResult(reply="queued", post_send=broken_effect)

    monkeypatch.setattr(session_module.cmd_mod, "handle", handle_broken_effect)
    await runtime._handle_command(_env(), command_module.ParsedCommand("restart", [], "/restart"))

    runtime.delivery_succeeds = False
    await runtime._handle_command(_env(), command_module.ParsedCommand("restart", [], "/restart"))
    assert effects == ["ran"]


@pytest.mark.asyncio
async def test_command_handler_failure_is_audited_redacted_and_delivered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _session(tmp_path)
    secret = "sk-" + ("c" * 32)

    async def fail(*_args: object) -> command_module.CommandResult:
        raise RuntimeError(f"provider key {secret}")

    monkeypatch.setattr(session_module.cmd_mod, "handle", fail)
    await runtime._handle_command(_env(), command_module.ParsedCommand("model", [], "/model"))

    error_event = next(payload for kind, payload in runtime.events.rows if kind == "cmd.error")
    assert secret not in error_event["err"]
    assert secret not in runtime.sent[-1][1]
    assert runtime.metrics.brain_errors.labels_seen == ["cmd.model"]
    assert runtime.metrics.brain_errors.count == 1


@pytest.mark.asyncio
async def test_slash_arguments_are_normalized_and_success_is_audited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _session(tmp_path)
    principal = Principal(id="owner", label="Owner", trust=True, tier="owner")
    seen: list[list[str]] = []

    async def handle(
        pc: command_module.ParsedCommand, *_args: object
    ) -> command_module.CommandResult:
        seen.append(pc.args)
        post_send = (lambda: None) if len(seen) == 2 else None
        return command_module.CommandResult(
            reply="ok" if len(seen) != 1 else "", post_send=post_send, data={"n": len(seen)}
        )

    monkeypatch.setattr(session_module.cmd_mod, "handle", handle)
    context = {"channel_id": "chat", "interaction_id": "interaction", "guild_id": None}
    outputs = [
        await runtime._slash_dispatch("status", {}, principal, context),
        await runtime._slash_dispatch("model", {"args": [None, "", 3]}, principal, context),
        await runtime._slash_dispatch("model", {"args": ("one",)}, principal, context),
        await runtime._slash_dispatch("model", {"args": "two"}, principal, context),
    ]

    assert seen == [[], ["3"], ["one"], ["two"]]
    assert outputs[-1]["data"] == {"n": 4}
    assert [kind for kind, _payload in runtime.events.rows] == ["cmd"] * 4
    assert runtime.events.rows[1][1]["post_send"] is True


@pytest.mark.asyncio
async def test_slash_failure_is_audited_and_cannot_leak_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _session(tmp_path)
    principal = Principal(id="owner", label="Owner", trust=False, tier="owner")
    secret = "sk-" + ("d" * 32)

    async def fail(*_args: object) -> command_module.CommandResult:
        raise RuntimeError(f"credential {secret}")

    monkeypatch.setattr(session_module.cmd_mod, "handle", fail)
    output = await runtime._slash_dispatch(
        "status",
        {},
        principal,
        {"channel_id": "chat", "interaction_id": None, "guild_id": "guild"},
    )

    assert secret not in output["reply"]
    assert output["post_send"] is None
    assert runtime.events.rows[0][0] == "cmd.error"
    assert runtime.events.rows[0][1]["source"] == "slash"
    assert secret not in runtime.events.rows[0][1]["err"]
    assert runtime.metrics.brain_errors.labels_seen == ["slash.status"]
