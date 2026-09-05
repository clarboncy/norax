"""Phase 7 — sleep-flush consolidation."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from norax.context import RollingWindow, SleepFlusher, extract_candidates


def _spill_with_content(tmp_path: Path, frames: list[dict]) -> Path:
    p = tmp_path / "spill-2026-01-01-000000-000000.jsonl"
    p.write_text("\n".join(json.dumps(f) for f in frames))
    # sidecar so retriever works
    md = p.with_suffix(".md")
    md.write_text(
        "SPILL;ts=x\n" + "\n".join(f"{f['kind'].upper()}#0:{f['content'][:200]}" for f in frames)
    )
    # Age it past min_age
    old = time.time() - 10_000
    os.utime(p, (old, old))
    os.utime(md, (old, old))
    return p


def test_extract_candidates_pairs_tool_call_result(tmp_path: Path):
    p = _spill_with_content(
        tmp_path,
        [
            {"kind": "tool_call", "content": "read file foo.md", "call_id": "c1", "turn_id": 1},
            {"kind": "tool_result", "content": "foo: bar\nbaz: qux", "call_id": "c1", "turn_id": 1},
        ],
    )
    cands = extract_candidates(p)
    assert len(cands) == 1
    assert cands[0].kind == "action_outcome"
    assert "ACTION:" in cands[0].text
    assert "OUTCOME:" in cands[0].text


def test_extract_candidates_skips_noise(tmp_path: Path):
    p = _spill_with_content(
        tmp_path,
        [
            {"kind": "user", "content": "ok", "turn_id": 1},
            {"kind": "assistant", "content": "thanks!", "turn_id": 1},
            {"kind": "user", "content": "RUNTIME:Python 3.12|W4", "turn_id": 2},
        ],
    )
    cands = extract_candidates(p)
    # All 3 extracted, but routing + importance will filter the noisy ones
    routes = [c.route for c in cands]
    assert "semantic" in routes  # RUNTIME:... is a fact
    assert routes.count("discard") >= 2  # "ok" and "thanks!"


def test_flusher_routes_by_content_shape(tmp_path: Path):
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    _spill_with_content(
        sleep,
        [
            {
                "kind": "user",
                "content": "WALLET:0x0000000000000000000000000000000000000001|W5",
                "turn_id": 1,
            },
            {"kind": "assistant", "content": "STEP 1: do X. STEP 2: RETRY on FAIL.", "turn_id": 1},
            {"kind": "user", "content": "URL:https://example.com/research paper", "turn_id": 2},
            {"kind": "user", "content": "ok", "turn_id": 3},
        ],
    )
    flusher = SleepFlusher(memory_root=tmp_path, min_age_sec=0.0)
    r = flusher.flush()
    assert r.spills_processed == 1
    assert r.written["semantic"] >= 1  # WALLET fact
    assert r.written["procedural"] >= 1  # STEP/RETRY workflow
    assert r.written["intel"] >= 1  # URL research
    assert r.written["discard"] >= 1  # "ok"
    # Outputs landed in the right directories
    sem_files = list((tmp_path / "semantic").glob("sleep-flush-*.md"))
    assert sem_files and "WALLET" in sem_files[0].read_text()


def test_flusher_dedupes_against_canonical(tmp_path: Path):
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    sem = tmp_path / "semantic"
    sem.mkdir()
    (sem / "existing.md").write_text("WALLET:0xabc|W5\n")
    _spill_with_content(
        sleep,
        [
            {"kind": "user", "content": "WALLET:0xabc|W5", "turn_id": 1},
        ],
    )
    r = SleepFlusher(memory_root=tmp_path, min_age_sec=0.0).flush()
    assert r.written["dup"] == 1
    assert r.written["semantic"] == 0


def test_flusher_dedupes_within_run(tmp_path: Path):
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    _spill_with_content(
        sleep,
        [
            {"kind": "user", "content": "RUNTIME:Python|W4", "turn_id": 1},
            {"kind": "assistant", "content": "RUNTIME:Python|W4", "turn_id": 2},
        ],
    )
    r = SleepFlusher(memory_root=tmp_path, min_age_sec=0.0).flush()
    assert r.written["semantic"] == 1
    assert r.written["dup"] == 1


def test_flusher_archives_consumed_spills(tmp_path: Path):
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    p = _spill_with_content(
        sleep,
        [
            {"kind": "user", "content": "FACT:something|W4", "turn_id": 1},
        ],
    )
    r = SleepFlusher(memory_root=tmp_path, min_age_sec=0.0).flush()
    assert not p.exists()
    assert (sleep / "archive" / p.name).exists()
    assert (sleep / "archive" / p.with_suffix(".md").name).exists()
    assert len(r.archived) == 1


def test_flusher_respects_min_age(tmp_path: Path):
    """Recently-written spills should NOT be touched (race prevention)."""
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    # Write with current mtime
    p = sleep / "spill-recent.jsonl"
    p.write_text(json.dumps({"kind": "user", "content": "NEW:thing|W4", "turn_id": 1}))
    # Don't age it
    flusher = SleepFlusher(memory_root=tmp_path, min_age_sec=3600.0)
    r = flusher.flush()
    assert r.spills_processed == 0
    assert p.exists()  # untouched


def test_flusher_is_idempotent(tmp_path: Path):
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    _spill_with_content(
        sleep,
        [
            {"kind": "user", "content": "CONST:value|W5", "turn_id": 1},
        ],
    )
    f = SleepFlusher(memory_root=tmp_path, min_age_sec=0.0)
    r1 = f.flush()
    # Add another spill with the same content
    sleep.mkdir(exist_ok=True)
    _spill_with_content(
        sleep,
        [
            {"kind": "user", "content": "CONST:value|W5", "turn_id": 1},
        ],
    )
    r2 = f.flush()
    assert r1.written["semantic"] == 1
    assert r2.written["semantic"] == 0
    assert r2.written["dup"] == 1


def test_flusher_integrated_with_rolling_window(tmp_path: Path):
    """End-to-end: window spills → flush consolidates."""
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    rw = RollingWindow(
        budget_tokens=200, protect_tail_turns=1, eviction_batch_pct=0.5, sleep_dir=sleep
    )
    rw.add_system("S" * 40)
    for i in range(30):
        tid = rw.start_turn()
        rw.add_user(f"FACT_{i}:value_{i}_payload|W4 " + "X" * 100, turn_id=tid)
        rw.add_assistant(f"ok {i} " + "Y" * 100, turn_id=tid)
        rw.evict_if_needed()
    # Age them all
    for p in sleep.glob("spill-*"):
        old = time.time() - 10_000
        os.utime(p, (old, old))
    r = SleepFlusher(memory_root=tmp_path, min_age_sec=0.0).flush()
    assert r.spills_processed > 0
    assert r.written["semantic"] > 0
    sem_file = next((tmp_path / "semantic").glob("sleep-flush-*.md"))
    assert "FACT_" in sem_file.read_text()


def test_flusher_honors_custom_batch_file_and_private_mode(tmp_path: Path) -> None:
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    _spill_with_content(
        sleep,
        [{"kind": "user", "content": "RUNTIME_POLICY:verify every deployment|W5"}],
    )

    result = SleepFlusher(
        memory_root=tmp_path,
        min_age_sec=0,
        batch_file="verified-{date}.md",
    ).flush()

    expected = tmp_path / "semantic" / f"verified-{datetime.now(UTC):%Y-%m-%d}.md"
    assert result.written["semantic"] == 1
    assert expected.is_file()
    assert expected.stat().st_mode & 0o777 == 0o600


def test_flusher_rejects_linked_canonical_output_without_touching_target(tmp_path: Path) -> None:
    sleep = tmp_path / "sleep"
    semantic = tmp_path / "semantic"
    sleep.mkdir()
    semantic.mkdir()
    _spill_with_content(
        sleep,
        [{"kind": "user", "content": "RUNTIME_POLICY:verify every deployment|W5"}],
    )
    outside = tmp_path / "outside.md"
    outside.write_text("keep me\n", encoding="utf-8")
    output = semantic / f"sleep-flush-{datetime.now(UTC):%Y-%m-%d}.md"
    output.symlink_to(outside)

    with pytest.raises((OSError, ValueError)):
        SleepFlusher(memory_root=tmp_path, min_age_sec=0).flush()

    assert outside.read_text(encoding="utf-8") == "keep me\n"
    assert any(sleep.glob("spill-*.jsonl"))


def test_flusher_ignores_linked_spill_input(tmp_path: Path) -> None:
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"kind":"user","content":"FACT:forged external input|W5"}\n')
    (sleep / "spill-linked.jsonl").symlink_to(outside)

    result = SleepFlusher(memory_root=tmp_path, min_age_sec=0).flush()

    assert result.spills_processed == 0
    assert outside.exists()
    assert not (tmp_path / "semantic").exists()
