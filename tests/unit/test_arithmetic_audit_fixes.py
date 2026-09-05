"""Regression tests for the Codex-audit arithmetic/constant bug fixes.

Covers four root causes identified in the 2026-08-29 audit continuation:

1. Correction gate false positive on internal task-state markers
   (``GOAL: said 'answer' but stored 'find'``).
2. Goal drift force_abort on simple/numeric tasks with poor keyword overlap.
3. Exec guard nudge firing on trivial arithmetic (``2+2``) — wastes a round
   and can duplicate the answer.
4. Ollama streaming done-frame echoing the full response content, causing
   duplicated output (``4`` → ``4   4``).
"""

from __future__ import annotations

import pytest

from norax.brain.goal_drift import DriftSeverity, GoalDriftMonitor
from norax.context.correction import CorrectionGate, extract_claims
from norax.gateway_client.ollama_profiles import (
    build_exec_guard_nudge,
    needs_exec_for_math,
)

# ---------- 1. Correction gate: internal protocol markers -------------------


def test_extract_claims_ignores_internal_protocol_markers():
    """Internal task-state markers (GOAL:, TASK_STATE, etc.) must not be
    extracted as checkable KV claims — they cause false contradictions
    against stored neurons from prior turns."""
    draft = (
        'GOAL: The user asked "What is 2+2?". '
        'TASK_STATE: {"goal":"answer"}. '
        "COMPLETED_ACTIONS: none. "
        "PENDING_STEPS: answer. "
        "DIRECTIVE: respond. "
        "OLLAMA_EXEC_GUARD;weight=W5. "
        "ARITHMETIC: run exec. "
        "CORRECTION;needs=true. "
        "MISMATCH:GOAL:claimed=answer;stored=find. "
        "EVIDENCE: prior turn. "
        "RULE: prefer stored value. "
        "EVICTED_CONTEXT: 3 rounds. "
        "FEATURES: freetoken. "
        "TURN:09:07|in=What is 2+2."
    )
    claims = extract_claims(draft)
    subjects = {c.subject for c in claims if c.kind == "kv"}
    # None of the internal protocol markers should appear as KV claim subjects
    internal = {
        "GOAL",
        "TASK_STATE",
        "COMPLETED_ACTIONS",
        "PENDING_STEPS",
        "DIRECTIVE",
        "OLLAMA_EXEC_GUARD",
        "ARITHMETIC",
        "CORRECTION",
        "MISMATCH",
        "EVIDENCE",
        "RULE",
        "EVICTED_CONTEXT",
        "FEATURES",
        "TURN",
    }
    assert not (subjects & internal), f"Internal markers leaked: {subjects & internal}"


def test_extract_claims_still_finds_real_kv():
    """Real KV claims (RUNTIME:, PORT:) must still be extracted."""
    draft = "RUNTIME:Python and PORT=11434 should be set."
    claims = extract_claims(draft)
    subjects = {c.subject for c in claims if c.kind == "kv"}
    assert "RUNTIME" in subjects
    assert "PORT" in subjects


@pytest.mark.asyncio
async def test_correction_gate_passes_on_task_state_echo():
    """A draft that echoes task-state format (GOAL: ...) must not trigger
    a revision even when stored neurons contain the same markers with
    different values from prior turns."""
    stored = _FakeNeuron(text="GOAL: find the bug|W5", weight=1.3)

    async def retrieve(q, k):
        return [(stored, 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    r = await gate.check("GOAL: answer the question. The answer is 4.")
    assert r.ok, f"Should pass but got contradictions: {r.contradictions}"
    assert not r.needs_revision


class _FakeNeuron:
    def __init__(self, text, weight=1.0):
        self.text = text
        self.weight = weight


# ---------- 2. Goal drift: simple/numeric tasks ----------------------------


def test_goal_drift_skips_insufficient_keywords():
    """Tasks with < 2 extractable keywords (numeric-heavy) must not trigger
    drift — overlap is always ~0.0 and causes false force_aborts."""
    monitor = GoalDriftMonitor(max_rounds=30)
    # "Calculate 2+2" extracts only 1 keyword ("calculate") — below the
    # minimum of 2 needed for meaningful overlap comparison.
    monitor.set_task("Calculate 2+2")
    result = monitor.check_drift(
        round_num=5,
        recent_tool_calls=[{"name": "exec", "args": {"command": "python3 -c print(2+2)"}}],
        draft_output="4",
    )
    assert result.severity == DriftSeverity.NONE
    assert result.recommendation == "insufficient_keywords"


def test_goal_drift_skips_zero_keyword_tasks():
    """Pure-numeric tasks with 0 keywords must return no-task-set."""
    monitor = GoalDriftMonitor(max_rounds=30)
    monitor.set_task("2+2")
    result = monitor.check_drift(
        round_num=5,
        recent_tool_calls=[{"name": "exec", "args": {}}],
        draft_output="4",
    )
    assert result.severity == DriftSeverity.NONE


def test_goal_drift_grace_period_prevents_early_false_abort():
    """Early rounds (<= grace_rounds) must not trigger drift — the agent is
    still exploring and surface-keyword overlap is naturally low."""
    monitor = GoalDriftMonitor(max_rounds=30, grace_rounds=3)
    monitor.set_task("Fix the authentication bug in the login handler")
    for rnd in range(1, 4):
        result = monitor.check_drift(
            round_num=rnd,
            recent_tool_calls=[{"name": "read", "args": {"path": "/etc/hostname"}}],
            draft_output="Reading system configuration files.",
        )
        assert result.severity == DriftSeverity.NONE, f"Round {rnd} should be in grace"
        assert result.recommendation == "grace_period"


def test_goal_drift_skips_short_output_signal():
    """Very short outputs (<= 12 chars) on simple tasks must not trigger
    the low_output_relevance signal — a correct '4' has 0 keyword overlap
    without indicating drift."""
    monitor = GoalDriftMonitor(max_rounds=30, grace_rounds=0)
    monitor.set_task("What is 2+2? Just answer the number.")
    result = monitor.check_drift(
        round_num=5,
        recent_tool_calls=[],
        draft_output="4",
    )
    # "4" is too short for the output signal; with no tool calls, no signals
    # fire at all.
    assert result.severity == DriftSeverity.NONE
    assert "low_output_relevance" not in result.signals


def test_goal_drift_detects_real_off_track_after_grace():
    """After the grace period, genuinely off-track work must still drift."""
    monitor = GoalDriftMonitor(max_rounds=30, grace_rounds=2)
    monitor.set_task("Fix the authentication bug in the login handler")
    result = monitor.check_drift(
        round_num=5,
        recent_tool_calls=[{"name": "read", "args": {"path": "/etc/hostname"}}],
        draft_output="The weather is sunny today and the sky is clear.",
    )
    assert result.severity != DriftSeverity.NONE
    assert result.score >= 0.4


def test_keyword_drift_signal_never_forces_task_abort() -> None:
    monitor = GoalDriftMonitor(max_rounds=30, grace_rounds=0)
    monitor.set_task("Fix the authentication bug in the login handler")

    results = [
        monitor.check_drift(
            round_num=round_number,
            recent_tool_calls=[{"name": "read", "args": {"path": "/etc/hostname"}}],
            draft_output="The weather is sunny today and the sky is clear.",
        )
        for round_number in range(1, 9)
    ]

    assert any(result.severity != DriftSeverity.NONE for result in results)
    assert all(result.recommendation != "force_abort" for result in results)


# ---------- 3. Exec guard: trivial arithmetic ------------------------------


def test_exec_guard_skips_trivial_single_digit_arithmetic():
    """Trivial single-digit arithmetic (2+2, 7-3, 3*4) must not trigger the
    exec guard nudge — it wastes a round and can duplicate the answer."""
    assert not needs_exec_for_math("What is 2+2? Just answer the number.")
    assert not needs_exec_for_math("What is 7-3?")
    assert not needs_exec_for_math("What is 3*4?")
    assert not needs_exec_for_math("What is 8+1?")
    assert build_exec_guard_nudge("What is 2+2? Just answer the number.") is None


def test_exec_guard_flags_multi_operation_arithmetic():
    """Multi-operation expressions must still trigger the exec guard."""
    assert needs_exec_for_math("What is 17*23+41*19?")
    assert build_exec_guard_nudge("What is 17*23+41*19?") is not None


def test_exec_guard_flags_division():
    """Division must trigger the exec guard (error-prone for mental math)."""
    assert needs_exec_for_math("What is 100/7?")
    assert needs_exec_for_math("What is 100÷7?")


def test_exec_guard_flags_decimals():
    """Decimal operands must trigger the exec guard."""
    assert needs_exec_for_math("What is 3.14159 * 2.71828?")


def test_exec_guard_flags_large_operands():
    """Large integer operands (>= 10) must trigger the exec guard."""
    assert needs_exec_for_math("What is 12345 * 67890?")
    assert needs_exec_for_math("calculate 17*23")


def test_exec_guard_skips_non_arithmetic():
    """Non-arithmetic prompts must not trigger the exec guard."""
    assert not needs_exec_for_math("hello")
    assert not needs_exec_for_math("What is a hash collision?")
    assert not needs_exec_for_math("Fix the login bug.")


# ---------- 4. Ollama streaming: done-frame duplication --------------------


def test_ollama_done_frame_does_not_duplicate_content():
    """The ollama streaming done-frame must not append its content to
    already-accumulated deltas. This simulates the '4' → '4   4' bug."""
    from unittest.mock import AsyncMock, MagicMock

    from norax.gateway_client import GatewayRequest, GatewayResponse, StreamEvent
    from norax.gateway_client.ollama_wrapper import OllamaGatewayClient

    # Simulate an ollama server that sends "4" as a delta, then echoes the
    # full "4" again in the done frame (the bug that caused duplication).
    async def fake_chat_stream(req):
        yield StreamEvent(kind="delta", text="4")
        yield StreamEvent(
            kind="delta",
            text="",
        )  # intermediate empty delta
        # Done frame with message.content echoing the full answer
        yield StreamEvent(
            kind="final",
            response=GatewayResponse(
                request_id="r1",
                model="kimi-k3:cloud",
                content="4",
                tool_calls=[],
                usage={"input_tokens": 1, "output_tokens": 1},
                raw={"done": True},
            ),
        )

    inner = MagicMock()
    inner.base_url = "http://127.0.0.1:11434/v1"
    inner.stream_required = False
    inner.chat_path = "/chat/completions"
    inner.chat_stream = fake_chat_stream
    inner.aclose = AsyncMock()

    client = OllamaGatewayClient(inner)
    req = GatewayRequest(
        model="kimi-k3:cloud",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )

    import asyncio

    collected_deltas = []
    final_resp = None

    async def run():
        nonlocal final_resp
        async for evt in client.chat_stream(req):
            if evt.kind == "delta" and evt.text:
                collected_deltas.append(evt.text)
            elif evt.kind == "final":
                final_resp = evt.response

    asyncio.run(run())

    # The final response content should be "4", not "44" or "4   4"
    assert final_resp is not None
    assert final_resp.content.strip() == "4", (
        f"Expected '4' but got '{final_resp.content}' — done-frame duplication"
    )
