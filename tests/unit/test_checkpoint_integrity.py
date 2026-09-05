from __future__ import annotations

import os
import time
from pathlib import Path

from norax.runtime.checkpoint import (
    clear_checkpoints,
    has_pending_checkpoint,
    list_checkpoints,
    list_pending_checkpoints,
    load_checkpoint,
    load_latest_checkpoint,
    mark_resumed,
    save_checkpoint,
)


def _save(root: Path, *, channel: str = "chat", turn_id: str = "turn-1") -> Path:
    return save_checkpoint(
        channel=channel,
        turn_id=turn_id,
        messages=[{"role": "user", "content": "hello"}],
        trace=[],
        task_state={"goal": "answer"},
        rounds=1,
        model="model",
        memory_root=root,
    )


def test_explicit_memory_root_is_authoritative_and_reads_do_not_create(tmp_path, monkeypatch):
    configured = tmp_path / "configured"
    unrelated = tmp_path / "environment"
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(unrelated))
    monkeypatch.setenv("NORAX_CHECKPOINT_DIR", str(unrelated / "override"))

    assert load_latest_checkpoint("missing", memory_root=configured) is None
    assert not (configured / "checkpoints").exists()

    path = _save(configured)
    assert path.parent == configured / "checkpoints" / "chat"
    assert not unrelated.exists()
    loaded = load_checkpoint("chat", "turn-1", memory_root=configured)
    assert loaded is not None and loaded["model"] == "model"


def test_untrusted_identifiers_cannot_escape_checkpoint_root(tmp_path):
    channel = "../../outside/channel"
    turn_id = "../../outside-turn"
    path = _save(tmp_path, channel=channel, turn_id=turn_id)

    checkpoint_root = (tmp_path / "checkpoints").resolve()
    assert path.resolve().is_relative_to(checkpoint_root)
    assert path.parent != checkpoint_root / channel
    loaded = load_checkpoint(channel, turn_id, memory_root=tmp_path)
    assert loaded is not None and loaded["channel"] == channel
    assert not (tmp_path.parent / "outside").exists()


def test_checkpoint_state_lifecycle_and_permissions(tmp_path):
    path = _save(tmp_path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert has_pending_checkpoint("chat", memory_root=tmp_path)
    assert [item["turn_id"] for item in list_pending_checkpoints(tmp_path)] == ["turn-1"]
    assert list_checkpoints("chat", memory_root=tmp_path)[0]["turn_id"] == "turn-1"

    mark_resumed("chat", "turn-1", memory_root=tmp_path)
    assert not has_pending_checkpoint("chat", memory_root=tmp_path)
    assert clear_checkpoints("chat", memory_root=tmp_path) == 1
    assert clear_checkpoints("chat", memory_root=tmp_path) == 0


def test_checkpoint_persists_bounded_response_and_delivery_evidence(tmp_path):
    path = save_checkpoint(
        channel="chat",
        turn_id="delivery-failure",
        messages=[],
        trace=[],
        task_state={"goal": "reply"},
        rounds=1,
        model="model",
        status="delivery_failed",
        response={
            "content_preview": "x" * 10_000,
            "content_chars": 10_000,
            "content_sha256": "a" * 64,
        },
        delivery={"ok": False, "state": "delivery_failed"},
        memory_root=tmp_path,
    )

    loaded = load_checkpoint("chat", "delivery-failure", memory_root=tmp_path)
    assert loaded is not None
    assert loaded["status"] == "delivery_failed"
    assert loaded["delivery"] == {"ok": False, "state": "delivery_failed"}
    assert len(loaded["response"]["content_preview"]) == 2_000
    assert loaded["response"]["content_chars"] == 10_000
    assert path.stat().st_size <= 2_097_152


def test_unsafe_channel_names_do_not_collide(tmp_path):
    first = _save(tmp_path, channel="a/b", turn_id="one")
    second = _save(tmp_path, channel="a?b", turn_id="two")
    assert first.parent != second.parent


def test_checkpoint_reader_refuses_channel_symlink(tmp_path):
    root = tmp_path / "checkpoints"
    target = tmp_path / "elsewhere"
    root.mkdir()
    target.mkdir()
    os.symlink(target, root / "chat")

    try:
        load_latest_checkpoint("chat", memory_root=tmp_path)
    except OSError as exc:
        assert "symbolic links" in str(exc)
    else:
        raise AssertionError("checkpoint symlink should have been rejected")


def test_checkpoint_payload_has_real_count_and_size_bounds(tmp_path):
    path = save_checkpoint(
        channel="chat",
        turn_id="large-turn",
        messages=[{"role": "user", "content": "x" * 5_000, "index": index} for index in range(250)],
        trace=[
            {
                "name": "write",
                "args": {"text": "y" * 10_000},
                "result": {"content": "z" * 10_000},
            }
            for _ in range(250)
        ],
        task_state={"goal": "g" * 10_000},
        rounds=80,
        model="model",
        memory_root=tmp_path,
    )

    loaded = load_checkpoint("chat", "large-turn", memory_root=tmp_path)
    assert loaded is not None
    assert len(loaded["messages"]) <= 200
    assert len(loaded["trace"]) <= 200
    assert loaded["messages"][-1]["index"] == 249
    assert len(loaded["task_state"]["goal"]) <= 2_000
    assert path.stat().st_size <= 2_097_152


def test_latest_checkpoint_ignores_file_symlinks(tmp_path):
    real = _save(tmp_path, turn_id="real")
    outside = tmp_path / "outside.json"
    outside.write_text('{"turn_id":"outside","timestamp":9999999999}')
    os.symlink(outside, real.parent / "checkpoint_link.json")

    loaded = load_latest_checkpoint("chat", memory_root=tmp_path)
    assert loaded is not None
    assert loaded["turn_id"] == "real"


def test_checkpoint_readers_reject_oversized_legacy_files(tmp_path):
    real = _save(tmp_path, turn_id="real")
    oversized = real.parent / "checkpoint_oversized.json"
    oversized.write_bytes(b"{" + (b'"padding":"' + (b"x" * 2_097_152) + b'"}'))

    assert load_checkpoint("chat", "oversized", memory_root=tmp_path) is None
    assert [item["turn_id"] for item in list_checkpoints("chat", memory_root=tmp_path)] == ["real"]


def test_corrupt_newest_checkpoint_does_not_hide_previous_durable_state(tmp_path):
    durable = _save(tmp_path, turn_id="durable")
    corrupt = durable.parent / "checkpoint_corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    future = time.time() + 10
    os.utime(corrupt, (future, future))

    loaded = load_latest_checkpoint("chat", memory_root=tmp_path)

    assert loaded is not None
    assert loaded["turn_id"] == "durable"
