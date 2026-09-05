"""Tests for canonical event chain verifier (P2-1 reaudit fix)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from norax.verify_event_chain import _verify_single_file, verify_all_generations


def _make_record(prev_hash: str, body: dict, *, hash_override: str | None = None) -> str:
    """Create a valid hash-chained event record."""
    body_rec = dict(body)
    body_rec["prev_hash"] = prev_hash
    body_str = json.dumps(body_rec, separators=(",", ":"), sort_keys=True)
    h = hash_override or hashlib.sha256((prev_hash + body_str).encode("utf-8")).hexdigest()
    body_rec["hash"] = h
    return json.dumps(body_rec)


def _write_chain(path: Path, records: list[str]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(r + "\n")


def test_verify_single_file_valid(tmp_path: Path) -> None:
    f = tmp_path / "events.jsonl"
    r0 = _make_record("0" * 64, {"event": "start", "ts": 1})
    prev = json.loads(r0)["hash"]
    r1 = _make_record(prev, {"event": "turn", "ts": 2})
    _write_chain(f, [r0, r1])

    result = _verify_single_file(f)
    assert result["ok"] is True
    assert result["records"] == 2
    assert result["corrupt"] == 0


def test_verify_single_file_corrupt_hash(tmp_path: Path) -> None:
    f = tmp_path / "events.jsonl"
    r0 = _make_record("0" * 64, {"event": "start"})
    r1 = _make_record(json.loads(r0)["hash"], {"event": "turn"}, hash_override="bad")
    _write_chain(f, [r0, r1])

    result = _verify_single_file(f)
    assert result["ok"] is False
    assert result["corrupt"] == 1


def test_verify_all_generations_chained(tmp_path: Path) -> None:
    """Verify that rotated files are chained correctly across generations."""
    # Generation 1
    r0 = _make_record("0" * 64, {"event": "gen1_start"})
    r1 = _make_record(json.loads(r0)["hash"], {"event": "gen1_end"})
    _write_chain(tmp_path / "events-20260613T074738Z.jsonl", [r0, r1])

    # Generation 2 — first prev_hash must match gen1 tail
    gen1_tail = json.loads(r1)["hash"]
    r2 = _make_record(gen1_tail, {"event": "gen2_start"})
    r3 = _make_record(json.loads(r2)["hash"], {"event": "gen2_end"})
    _write_chain(tmp_path / "events-20260628T020608Z.jsonl", [r2, r3])

    # Current
    gen2_tail = json.loads(r3)["hash"]
    r4 = _make_record(gen2_tail, {"event": "current"})
    _write_chain(tmp_path / "events.jsonl", [r4])

    result = verify_all_generations(tmp_path)
    assert result["ok"] is True
    assert result["generations"] == 3
    assert result["total_records"] == 5
    assert result["total_corrupt"] == 0


def test_verify_all_generations_detects_cross_file_break(tmp_path: Path) -> None:
    """Detect when a rotated file's first prev_hash doesn't match predecessor's tail."""
    r0 = _make_record("0" * 64, {"event": "gen1"})
    _write_chain(tmp_path / "events-20260613T074738Z.jsonl", [r0])

    # Wrong prev_hash — doesn't match gen1 tail
    r1 = _make_record("0" * 64, {"event": "gen2"})
    _write_chain(tmp_path / "events-20260628T020608Z.jsonl", [r1])

    result = verify_all_generations(tmp_path)
    assert result["ok"] is False


def test_verify_all_generations_empty_dir(tmp_path: Path) -> None:
    result = verify_all_generations(tmp_path)
    assert result["ok"] is False
    assert "no_event_files_found" in result["error"]


def test_verify_single_file_missing(tmp_path: Path) -> None:
    result = _verify_single_file(tmp_path / "nonexistent.jsonl")
    assert result["ok"] is False
    assert result["error"] == "file_missing"


def test_verify_single_file_limit_is_bounded_and_validated(tmp_path: Path) -> None:
    f = tmp_path / "events.jsonl"
    first = _make_record("0" * 64, {"event": "first"})
    second = _make_record(json.loads(first)["hash"], {"event": "second"})
    _write_chain(f, [first, second])

    assert _verify_single_file(f, limit=1)["records"] == 1
    invalid = _verify_single_file(f, limit=0)
    assert invalid["ok"] is False
    assert invalid["error"] == "invalid_limit"


def test_verify_single_file_rejects_non_object_and_invalid_hash_types(tmp_path: Path) -> None:
    f = tmp_path / "events.jsonl"
    _write_chain(
        f,
        [
            json.dumps(["not", "an", "event"]),
            json.dumps({"prev_hash": 7, "hash": {"not": "a hash"}}),
        ],
    )

    result = _verify_single_file(f)

    assert result["ok"] is False
    assert result["records"] == 2
    assert result["corrupt"] == 2
    assert result["error"] == "record_not_object"


def test_writer_format_anchor_is_verified(tmp_path: Path) -> None:
    rotated = tmp_path / "events-20260613T074738Z.jsonl"
    first = _make_record("0" * 64, {"event": "rotated"})
    _write_chain(rotated, [first])
    tail = json.loads(first)["hash"]
    current = _make_record(tail, {"event": "current"})
    _write_chain(tmp_path / "events.jsonl", [current])
    (tmp_path / "events.jsonl.anchor").write_text(
        json.dumps({"rotated": rotated.name, "tail_hash": tail}),
        encoding="utf-8",
    )

    result = verify_all_generations(tmp_path)

    assert result["ok"] is True
    assert result["anchor"] == {
        "present": True,
        "ok": True,
        "rotated": rotated.name,
        "tail_hash": tail,
    }


@pytest.mark.parametrize(
    ("anchor_text", "expected_error"),
    [
        ("{broken", "invalid_anchor"),
        (json.dumps({"rotated": "events-20260613T074738Z.jsonl"}), "invalid_anchor"),
        (
            json.dumps({"rotated": "events-20260613T074738Z.jsonl", "tail_hash": "0" * 64}),
            "anchor_tail_hash_mismatch",
        ),
    ],
)
def test_corrupt_or_tampered_anchor_fails_closed(
    tmp_path: Path,
    anchor_text: str,
    expected_error: str,
) -> None:
    rotated = tmp_path / "events-20260613T074738Z.jsonl"
    _write_chain(rotated, [_make_record("0" * 64, {"event": "rotated"})])
    (tmp_path / "events.jsonl.anchor").write_text(anchor_text, encoding="utf-8")

    result = verify_all_generations(tmp_path)

    assert result["ok"] is False
    assert result["anchor"]["error"] == expected_error


def test_unorderable_rotation_filename_fails_closed(tmp_path: Path) -> None:
    _write_chain(tmp_path / "events-manual.jsonl", [_make_record("0" * 64, {"event": "x"})])

    result = verify_all_generations(tmp_path)

    assert result["ok"] is False
    assert result["error"] == "invalid_rotation_filename"


def test_verifier_rejects_oversized_lines_and_symlinks(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b"x" * (4 * 1024 * 1024 + 1) + b"\n")
    result = _verify_single_file(oversized)
    assert result["ok"] is False
    assert result["error"] == "line_too_large"
    assert result["corrupt"] == 1

    target = tmp_path / "target.jsonl"
    _write_chain(target, [_make_record("0" * 64, {"event": "private"})])
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(target)
    result = _verify_single_file(alias)
    assert result["ok"] is False
    assert result["error"] == "file_read_error"
