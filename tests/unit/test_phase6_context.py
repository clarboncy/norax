"""Phase 6 — rolling context window + correction gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from norax.context import (
    CorrectionGate,
    RollingWindow,
    extract_claims,
)

# ---------- RollingWindow -------------------------------------------------


def test_rolling_window_basic_ingest(tmp_path: Path):
    rw = RollingWindow(budget_tokens=1000, sleep_dir=tmp_path)
    rw.add_system("SYSTEM_PROMPT_COMPRESSED_1234")
    tid = rw.start_turn()
    rw.add_user("hello", turn_id=tid)
    rw.add_assistant("hi there", turn_id=tid)
    assert rw.total_tokens() > 0
    assert not rw.needs_eviction()
    assert len(rw.messages()) == 3


def test_rolling_window_head_never_evicted(tmp_path: Path):
    rw = RollingWindow(budget_tokens=200, protect_tail_turns=2, sleep_dir=tmp_path)
    rw.add_system("S" * 200)  # 50 tokens — stays in head forever
    # Pile a bunch of turns to trigger eviction
    for _i in range(20):
        tid = rw.start_turn()
        rw.add_user("U" * 100, turn_id=tid)  # 25 tokens
        rw.add_assistant("A" * 100, turn_id=tid)  # 25 tokens
    r = rw.evict_if_needed()
    assert r is not None
    assert r.tokens_freed > 0
    # Head (system) is intact
    assert len(rw.head) == 1
    assert rw.head[0].content.startswith("S")
    # Last 2 turns preserved
    tail_ids = {f.turn_id for f in rw.body}
    assert max(tail_ids) == 20
    assert 19 in tail_ids


def test_rolling_window_tool_call_atomic(tmp_path: Path):
    """tool_call + matching tool_result must evict together, never apart."""
    rw = RollingWindow(budget_tokens=150, protect_tail_turns=1, sleep_dir=tmp_path)
    # Tons of tool_call/tool_result pairs in earlier turns
    for i in range(15):
        tid = rw.start_turn()
        rw.add_user(f"q{i}", turn_id=tid)
        rw.add_tool_call(call_id=f"c{i}", content=f"call {i}" + "X" * 40, turn_id=tid)
        rw.add_tool_result(call_id=f"c{i}", content=f"result {i}" + "Y" * 40, turn_id=tid)
        rw.add_assistant(f"ok {i}", turn_id=tid)
    r = rw.evict_if_needed()
    assert r is not None
    # Verify no orphan: every tool_call in body has its tool_result;
    # every tool_result has its tool_call.
    body_calls = {f.call_id for f in rw.body if f.kind == "tool_call"}
    body_results = {f.call_id for f in rw.body if f.kind == "tool_result"}
    assert body_calls == body_results
    # Same invariant on the evicted side
    ev_calls = {f.call_id for f in r.evicted if f.kind == "tool_call"}
    ev_results = {f.call_id for f in r.evicted if f.kind == "tool_result"}
    assert ev_calls == ev_results


def test_rolling_window_writes_jsonl_spill(tmp_path: Path):
    rw = RollingWindow(budget_tokens=150, protect_tail_turns=1, sleep_dir=tmp_path)
    rw.add_system("S" * 40)
    for _i in range(15):
        tid = rw.start_turn()
        rw.add_user("X" * 50, turn_id=tid)
        rw.add_assistant("Y" * 50, turn_id=tid)
    r = rw.evict_if_needed()
    assert r is not None and r.spill_path is not None
    assert r.spill_path.exists()
    # JSONL: each line parses to a frame
    lines = r.spill_path.read_text().strip().splitlines()
    assert len(lines) == len(r.evicted)
    parsed = [json.loads(line) for line in lines]
    assert all("kind" in p and "content" in p and "turn_id" in p for p in parsed)
    # Sidecar .md exists for the retriever
    md = r.spill_path.with_suffix(".md")
    assert md.exists() and md.read_text().startswith("SPILL;")


def test_rolling_window_no_eviction_when_under_budget(tmp_path: Path):
    rw = RollingWindow(budget_tokens=10_000, sleep_dir=tmp_path)
    rw.add_system("short")
    tid = rw.start_turn()
    rw.add_user("q", turn_id=tid)
    rw.add_assistant("a", turn_id=tid)
    assert rw.evict_if_needed() is None


def test_rolling_window_aggressive_batch_frees_breathing_room(tmp_path: Path):
    rw = RollingWindow(
        budget_tokens=400, protect_tail_turns=1, eviction_batch_pct=0.5, sleep_dir=tmp_path
    )
    for _i in range(40):
        tid = rw.start_turn()
        rw.add_user("U" * 40, turn_id=tid)
        rw.add_assistant("A" * 40, turn_id=tid)
    r = rw.evict_if_needed()
    assert r is not None
    assert rw.total_tokens() <= rw.budget_tokens
    assert r.tokens_freed >= rw.budget_tokens * 0.25


def test_rolling_window_many_evictions_are_lossless(tmp_path: Path):
    """Regression: multiple eviction cycles within the same second must each
    produce a distinct spill file — no overwrite, no data loss."""
    rw = RollingWindow(
        budget_tokens=200, protect_tail_turns=1, eviction_batch_pct=0.3, sleep_dir=tmp_path
    )
    rw.add_system("S" * 40)
    total_evicted = 0
    for _i in range(40):
        tid = rw.start_turn()
        rw.add_user("U" * 60, turn_id=tid)
        rw.add_assistant("A" * 60, turn_id=tid)
        ev = rw.evict_if_needed()
        if ev:
            total_evicted += len(ev.evicted)
    spills = sorted(tmp_path.glob("spill-*.jsonl"))
    disk_frames = sum(sum(1 for _ in p.open()) for p in spills)
    assert disk_frames == total_evicted, (
        f"lost frames: evicted={total_evicted} on_disk={disk_frames}"
    )


def test_rolling_window_load_rejects_malformed_frame_without_crashing(tmp_path: Path):
    path = tmp_path / "window.json"
    path.write_text(
        json.dumps(
            {
                "budget_tokens": 1000,
                "head": [],
                "body": [{"kind": "administrator", "content": "forged", "turn_id": 1}],
            }
        ),
        encoding="utf-8",
    )

    loaded = RollingWindow.load(path, budget_tokens=777)

    assert loaded.budget_tokens == 777
    assert loaded.body == []


def test_rolling_window_load_rejects_oversized_state_file(tmp_path: Path):
    path = tmp_path / "window.json"
    path.write_bytes(b" " * (16 * 1024 * 1024 + 1))

    loaded = RollingWindow.load(path, budget_tokens=888)

    assert loaded.budget_tokens == 888
    assert loaded.body == []


# ---------- Claim extraction ----------------------------------------------


def test_extract_claims_finds_identifiers():
    draft = (
        "your wallet is 0xabcdef1234567890abcdef1234567890abcdef12 "
        "and your email is owner@example.com"
    )
    claims = extract_claims(draft)
    kinds = {(c.kind, c.subject) for c in claims}
    assert ("identifier", "eth_addr") in kinds
    assert ("identifier", "email") in kinds


def test_extract_claims_finds_kv():
    draft = "RUNTIME:Python and PORT=11434 should be set."
    claims = extract_claims(draft)
    subjects = {c.subject for c in claims}
    assert "RUNTIME" in subjects
    assert "PORT" in subjects


# ---------- CorrectionGate -------------------------------------------------


class _FakeNeuron:
    def __init__(self, text, weight=1.0):
        self.text = text
        self.weight = weight


@pytest.mark.asyncio
async def test_correction_gate_passes_when_no_claims():
    async def retrieve(q, k):
        return []

    gate = CorrectionGate(retrieve=retrieve)
    r = await gate.check("how are you?")
    assert r.ok and not r.needs_revision
    assert r.checked == 0


@pytest.mark.asyncio
async def test_correction_gate_flags_wrong_email():
    """Draft claims wrong email; store has the right one."""
    stored = _FakeNeuron(
        text="EMAIL:owner@example.com|W5",
        weight=1.3,
    )

    async def retrieve(q, k):
        return [(stored, 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    r = await gate.check("your email is wrong@example.com")
    assert not r.ok
    assert r.needs_revision
    assert len(r.contradictions) == 1
    c = r.contradictions[0]
    assert c["claim_value"] == "wrong@example.com"
    assert "owner@example.com" in c["stored_value"]


@pytest.mark.asyncio
async def test_correction_gate_passes_matching_email():
    stored = _FakeNeuron(text="EMAIL:owner@example.com|W5", weight=1.3)

    async def retrieve(q, k):
        return [(stored, 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    r = await gate.check("your email is owner@example.com")
    assert r.ok and not r.needs_revision


@pytest.mark.asyncio
async def test_correction_gate_does_not_replace_third_party_email():
    stored = _FakeNeuron(text="EMAIL:owner@example.com|W5", weight=1.3)
    calls = 0

    async def retrieve(q, k):
        nonlocal calls
        calls += 1
        return [(stored, 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    result = await gate.check("Contact support@example.net if the vendor rejects the order.")

    assert result.ok
    assert not result.needs_revision
    assert calls == 0


@pytest.mark.asyncio
async def test_correction_gate_uses_one_retrieval_per_owned_identifier():
    stored = _FakeNeuron(text="EMAIL:owner@example.com|W5", weight=1.3)
    calls = 0

    async def retrieve(q, k):
        nonlocal calls
        calls += 1
        return [(stored, 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    result = await gate.check("Your official email is wrong@example.com.")

    assert result.needs_revision
    assert calls == 1


@pytest.mark.asyncio
async def test_correction_gate_flags_wrong_kv():
    stored = _FakeNeuron(text="RUNTIME:Python 3.12|W4", weight=1.0)

    async def retrieve(q, k):
        return [(stored, 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    r = await gate.check("RUNTIME:Ruby should be used for the core.")
    assert r.needs_revision
    block = r.as_correction_block()
    assert "CORRECTION;needs=true" in block
    assert "MISMATCH:RUNTIME" in block


@pytest.mark.asyncio
async def test_correction_gate_ignores_low_weight_mismatches():
    """If stored fact has |W1 weight, we don't hard-block on it."""
    stored = _FakeNeuron(text="RUNTIME:Python 3.12", weight=0.4)  # W1

    async def retrieve(q, k):
        return [(stored, 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve, hard_weight=1.0)
    r = await gate.check("RUNTIME:Ruby is fine.")
    # Not hard-blocked (weight below threshold), so ok=True
    assert r.ok


@pytest.mark.asyncio
async def test_correction_gate_ignores_identifiers_and_kv_inside_code_examples():
    calls = 0

    async def retrieve(q, k):
        nonlocal calls
        calls += 1
        return []

    gate = CorrectionGate(retrieve=retrieve)
    result = await gate.check(
        "Example configuration:\n```env\nEMAIL=placeholder@example.com\nAPI_URL=https://example.test\n```"
    )
    assert result.ok
    assert result.checked == 0
    assert calls == 0


@pytest.mark.asyncio
async def test_correction_gate_marks_retrieval_failure_inconclusive():
    async def retrieve(q, k):
        raise TimeoutError("memory unavailable")

    gate = CorrectionGate(retrieve=retrieve)
    result = await gate.check("Your official email is owner@example.com")
    assert result.ok
    assert not result.needs_revision
    assert result.inconclusive
    assert result.checked == 1
    assert result.errors == ["retrieval failed for email: TimeoutError"]


@pytest.mark.asyncio
async def test_correction_gate_does_not_treat_ordinary_prose_as_canonical_fact():
    calls = 0

    async def retrieve(q, k):
        nonlocal calls
        calls += 1
        return [(_FakeNeuron("runtime:Python|W5", 1.3), 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    result = await gate.check("The runtime is Ruby because this example targets JRuby.")

    assert result.ok
    assert result.checked == 0
    assert calls == 0


@pytest.mark.asyncio
async def test_correction_gate_retrieves_independent_kv_claims_concurrently():
    active = 0
    peak = 0

    async def retrieve(q, k):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return []

    gate = CorrectionGate(retrieve=retrieve)
    result = await gate.check("ALPHA=one BRAVO=two CHARLIE=three DELTA=four")

    assert result.ok
    assert result.checked == 4
    assert peak == 4


@pytest.mark.asyncio
async def test_third_party_urls_do_not_consume_owned_identifier_check_budget():
    stored = _FakeNeuron(text="EMAIL:owner@example.com|W5", weight=1.3)
    calls = 0

    async def retrieve(q, k):
        nonlocal calls
        calls += 1
        return [(stored, 0.9, "local")]

    citations = " ".join(f"https://example.test/source/{index}" for index in range(20))
    gate = CorrectionGate(retrieve=retrieve, max_claims=2)
    result = await gate.check(f"Sources: {citations}. Your official email is wrong@example.com.")

    assert result.needs_revision
    assert result.checked == 1
    assert calls == 1
