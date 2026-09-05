from __future__ import annotations

import time
from pathlib import Path

import pytest

import norax.memory.vta_writer as vta_writer
from norax.memory.vta_writer import write_outcome


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def test_sleep_buffer_is_not_truncated_or_deduplicated(tmp_path: Path):
    for index in range(55):
        write_outcome(memory_root=tmp_path, route="sleep_buffer", text=f"sleep-{index}")
    write_outcome(memory_root=tmp_path, route="sleep_buffer", text="sleep-54")

    lines = (tmp_path / "sleep" / f"buffer-{_today()}.md").read_text().splitlines()
    assert lines[0].startswith("# sleep-buffer")
    assert lines[1:] == [*[f"sleep-{index}" for index in range(55)], "sleep-54"]


def test_flashbulb_memory_is_not_truncated(tmp_path: Path):
    for index in range(55):
        write_outcome(memory_root=tmp_path, route="immediate_flashbulb", text=f"event-{index}")

    lines = (tmp_path / "semantic" / f"flashbulb-{_today()}.md").read_text().splitlines()
    assert len(lines) == 56
    assert lines[1] == "event-0|W4"
    assert lines[-1] == "event-54|W4"


def test_scratchpad_retains_only_recent_turns_and_stays_bounded(tmp_path: Path):
    for index in range(55):
        write_outcome(
            memory_root=tmp_path,
            route="scratchpad",
            text=f"TURN:12:{index:02d}|in=request-{index}|out=done",
        )

    lines = (tmp_path / "scratchpad.md").read_text().splitlines()
    turns = [line for line in lines if line.startswith("TURN:")]
    assert len(lines) <= 40
    assert len(turns) == 5
    assert "request-50" in turns[0]
    assert "request-54" in turns[-1]


def test_scratchpad_deduplicates_identical_lines(tmp_path: Path):
    for _ in range(2):
        write_outcome(memory_root=tmp_path, route="scratchpad", text="same outcome")

    lines = (tmp_path / "scratchpad.md").read_text().splitlines()
    assert lines.count("same outcome") == 1


def test_durable_writes_are_constant_time_and_normalize_one_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("durable append must not reread the whole daily file")

    monkeypatch.setattr(vta_writer, "read_bounded_text", forbidden_read)
    write_outcome(
        memory_root=tmp_path,
        route="sleep_buffer",
        text="first line\nsecond line" + "x" * 10_000,
    )

    lines = (tmp_path / "sleep" / f"buffer-{_today()}.md").read_text().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("first line second line")
    assert len(lines[1]) <= 4_096


def test_memory_writers_refuse_symlink_targets(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    scratch = tmp_path / "scratchpad.md"
    scratch.symlink_to(outside)
    with pytest.raises(OSError):
        write_outcome(memory_root=tmp_path, route="scratchpad", text="forged")
    assert outside.read_text(encoding="utf-8") == "private"

    sleep = tmp_path / "sleep"
    sleep.mkdir()
    target = sleep / f"buffer-{_today()}.md"
    target.symlink_to(outside)
    with pytest.raises(OSError):
        write_outcome(memory_root=tmp_path, route="sleep_buffer", text="forged")
    assert outside.read_text(encoding="utf-8") == "private"
