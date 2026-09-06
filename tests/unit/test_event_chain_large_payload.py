"""Tests for event hash-chain continuity with large (>4KB) payloads."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from norax.observability.log import EventLog


@pytest.mark.asyncio
async def test_large_payload_does_not_break_chain(tmp_path: Path):
    """Appending a >4KB event followed by a small event must preserve
    prev_hash continuity — the small event's prev_hash must equal the
    large event's hash."""
    log = EventLog(tmp_path / "events.jsonl")

    # First event: small
    await log.append("test", {"msg": "first"})

    # Second event: large (>4KB serialized line)
    big_payload = {"data": "x" * 8000}
    await log.append("test", big_payload)

    # Third event: small again
    await log.append("test", {"msg": "third"})

    # Read back all lines and verify chain continuity
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 3

    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    rec3 = json.loads(lines[2])

    # Each record's prev_hash must equal the previous record's hash
    assert rec2["prev_hash"] == rec1["hash"], "chain broken after first event"
    assert rec3["prev_hash"] == rec2["hash"], "chain broken after large >4KB event"

    # The large event line must actually be >4KB to validate the fix
    assert len(lines[1]) > 4096, (
        f"large event line only {len(lines[1])} bytes, test is not meaningful"
    )


@pytest.mark.asyncio
async def test_load_tail_hash_after_large_event(tmp_path: Path):
    """A fresh EventLog opened after a >4KB event was written must
    correctly load the last hash (not fall back to genesis)."""
    log1 = EventLog(tmp_path / "events.jsonl")
    await log1.append("test", {"msg": "small"})
    await log1.append("test", {"data": "y" * 8000})

    last_hash = log1._prev_hash

    # Simulate a new process — fresh EventLog, no in-memory _prev_hash
    log2 = EventLog(tmp_path / "events.jsonl")
    loaded = log2._load_tail_hash()

    assert loaded == last_hash, (
        f"tail hash mismatch: loaded={loaded[:16]}... expected={last_hash[:16]}..."
    )
    assert loaded != "0" * 64, "tail hash fell back to genesis after large event"


@pytest.mark.asyncio
async def test_multiple_large_events_chain_intact(tmp_path: Path):
    """Multiple consecutive >4KB events must all chain correctly."""
    log = EventLog(tmp_path / "events.jsonl")

    for i in range(5):
        await log.append("test", {"idx": i, "blob": "z" * 6000})

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 5

    for i in range(1, len(lines)):
        prev = json.loads(lines[i - 1])
        curr = json.loads(lines[i])
        assert curr["prev_hash"] == prev["hash"], f"chain broken at event {i}"


@pytest.mark.asyncio
async def test_trace_scope_correlates_nested_events(tmp_path: Path):
    log = EventLog(tmp_path / "events.jsonl")

    with log.trace_scope("trace-release-audit"):
        await log.append("ingress", {"message": "hello"})
        await log.append("brain", {"decision": "emit_reply"})
        await __import__("asyncio").create_task(
            log.append("tool_call", {"name": "status", "ok": True})
        )

    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {record["trace_id"] for record in records} == {"trace-release-audit"}
    root_span = records[0]["span_id"]
    assert records[0]["parent_span_id"] is None
    assert all(record["parent_span_id"] == root_span for record in records[1:])


@pytest.mark.asyncio
async def test_event_payloads_are_bounded_json_safe_redacted_and_cycle_safe(tmp_path: Path):
    log = EventLog(tmp_path / "events.jsonl")
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    secret = "sk-" + "a" * 30

    await log.append(
        "test",
        {
            "blob": secret + "x" * (256 * 1024),
            "cycle": cyclic,
            "items": list(range(3_000)),
            "not_finite": float("nan"),
            "bytes": b"private bytes",
        },
    )

    raw = (tmp_path / "events.jsonl").read_bytes()
    assert len(raw) < 4 * 1024 * 1024
    assert secret.encode() not in raw
    record = json.loads(raw)
    assert record["payload"]["cycle"]["self"] == "<cycle>"
    assert record["payload"]["blob"].endswith("…<truncated>")
    assert record["payload"]["not_finite"] == "<nan>"
    assert record["payload"]["bytes"] == {"type": "bytes", "bytes": 13}


def test_event_log_rejects_symlink_and_corrupt_existing_tail(tmp_path: Path):
    target = tmp_path / "target.jsonl"
    target.write_text("private\n", encoding="utf-8")
    alias = tmp_path / "events.jsonl"
    alias.symlink_to(target)

    with pytest.raises(OSError):
        EventLog(alias)

    alias.unlink()
    alias.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tail"):
        EventLog(alias)


@pytest.mark.asyncio
async def test_rotation_uses_collision_resistant_names_and_verifies(tmp_path: Path):
    from norax.verify_event_chain import verify_all_generations

    log = EventLog(tmp_path / "events.jsonl", rotate_bytes=1_000)
    await log.append("test", {"blob": "a" * 800})
    await log.append("test", {"blob": "b" * 800})

    rotations = list(tmp_path.glob("events-*.jsonl"))
    assert len(rotations) == 1
    assert verify_all_generations(tmp_path)["ok"] is True


@pytest.mark.asyncio
async def test_rotation_anchor_uses_latest_other_writer_tail(tmp_path: Path):
    from norax.verify_event_chain import verify_all_generations

    path = tmp_path / "events.jsonl"
    first = EventLog(path, rotate_bytes=0)
    second = EventLog(path, rotate_bytes=0)
    await first.append("first", {})
    await second.append("second", {})
    first.rotate_bytes = path.stat().st_size + 1
    await first.append("third", {})

    result = verify_all_generations(tmp_path)
    assert result["ok"] is True, result
    assert result["total_records"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_current", [False, True])
async def test_missing_current_after_rotation_cannot_restart_chain(tmp_path: Path, empty_current):
    path = tmp_path / "events.jsonl"
    writer = EventLog(path, rotate_bytes=1)
    await writer.append("first", {})
    await writer.append("second", {})
    original = path.read_bytes()
    path.unlink()
    if empty_current:
        path.touch()

    with pytest.raises(ValueError, match="missing|empty|rotation"):
        await writer.append("must_not_restart_at_genesis", {})
    assert not path.exists() or path.read_bytes() == b""
    assert original


@pytest.mark.asyncio
async def test_unterminated_record_cannot_be_concatenated_with_next_event(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    writer = EventLog(path)
    await writer.append("first", {})
    incomplete = path.read_bytes().rstrip(b"\n")
    path.write_bytes(incomplete)

    with pytest.raises(ValueError, match="unterminated|incomplete"):
        await writer.append("second", {})
    assert path.read_bytes() == incomplete


@pytest.mark.asyncio
async def test_empty_existing_file_does_not_create_empty_rotation(tmp_path: Path):
    from norax.verify_event_chain import verify_all_generations

    path = tmp_path / "events.jsonl"
    path.touch()
    writer = EventLog(path, rotate_bytes=1)
    await writer.append("first", {})

    result = verify_all_generations(tmp_path)
    assert result["ok"] is True, result
    assert result["generations"] == 1


@pytest.mark.asyncio
async def test_concurrent_writers_preserve_every_record_across_rotations(tmp_path: Path):
    from norax.verify_event_chain import verify_all_generations

    writers = [EventLog(tmp_path / "events.jsonl", rotate_bytes=1800) for _ in range(4)]
    await asyncio.gather(
        *(
            writers[index % len(writers)].append("concurrent", {"index": index})
            for index in range(80)
        )
    )

    result = verify_all_generations(tmp_path)
    assert result["ok"] is True, result
    assert result["total_records"] == 80
    assert result["generations"] > 1
