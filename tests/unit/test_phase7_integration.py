"""Phase 7 integration — run_turn with window + correction gate wired in."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from norax.brain import hot_path
from norax.context import CorrectionGate, RollingWindow
from norax.envelope import Principal, SensoryInput
from norax.gateway_client import GatewayResponse


class _FakeGateway:
    """Returns canned responses in order. Tracks request history."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.requests: list = []

    async def chat(self, req):
        self.requests.append(req)
        content = self._responses.pop(0) if self._responses else "ok"
        return GatewayResponse(
            model=req.model,
            content=content,
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            request_id="r1",
            tool_calls=[],
            raw={},
        )

    async def aclose(self):
        pass


def _env(body: str) -> SensoryInput:
    return SensoryInput(
        channel="chat",
        source="discord",
        message_id="mid-1",
        timestamp=datetime.now(UTC),
        body=body,
        sender=Principal(id="owner-123", label="Colby", trust=True, tier="owner"),
        trusted=True,
    )


@pytest.mark.asyncio
async def test_run_turn_with_window_records_frames(tmp_path: Path):
    rw = RollingWindow(budget_tokens=5000, sleep_dir=tmp_path)
    gw = _FakeGateway(["hello back"])
    ctx, rendered, resp = await hot_path.run_turn(
        _env("hi there"),
        gateway=gw,
        window=rw,
        runtime_info={"model": "test"},
    )
    assert resp.content == "hello back"
    # head got the system prompt; body got user + assistant
    assert len(rw.head) == 1
    assert len(rw.body) == 2
    assert rw.body[0].kind == "user" and "hi there" in rw.body[0].content
    assert rw.body[1].kind == "assistant" and "hello back" in rw.body[1].content
    # Both frames share a turn_id
    assert rw.body[0].turn_id == rw.body[1].turn_id


@pytest.mark.asyncio
async def test_run_turn_correction_gate_triggers_reprompt(tmp_path: Path):
    """Gate flags draft; run_turn re-prompts with the CORRECTION block."""

    # Fake retriever: always returns stored email |W5
    class _N:
        def __init__(self, text, weight):
            self.text = text
            self.weight = weight

    async def retrieve(q, k):
        return [(_N("EMAIL:owner@example.com|W5", 1.3), 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    # First response wrong; second correct
    gw = _FakeGateway(
        [
            "your email is wrong@example.com",
            "your email is owner@example.com",
        ]
    )
    ctx, rendered, resp = await hot_path.run_turn(
        _env("what is my email?"),
        gateway=gw,
        correction_gate=gate,
        max_correction_rounds=2,
    )
    # Should have re-prompted once; total 2 gateway calls
    assert len(gw.requests) == 2
    # Second request contains the CORRECTION block
    reprompt_text = gw.requests[1].messages[-1]["content"]
    assert "CORRECTION;needs=true" in reprompt_text
    assert "MISMATCH:email" in reprompt_text
    # Final response is the corrected one
    assert "owner@example.com" in resp.content


@pytest.mark.asyncio
async def test_run_turn_correction_gate_passes_on_clean_draft(tmp_path: Path):
    class _N:
        def __init__(self, text, weight):
            self.text = text
            self.weight = weight

    async def retrieve(q, k):
        return [(_N("EMAIL:owner@example.com|W5", 1.3), 0.9, "local")]

    gate = CorrectionGate(retrieve=retrieve)
    gw = _FakeGateway(["your email is owner@example.com"])
    ctx, rendered, resp = await hot_path.run_turn(
        _env("what is my email?"),
        gateway=gw,
        correction_gate=gate,
        max_correction_rounds=2,
    )
    # Only one call — no re-prompt
    assert len(gw.requests) == 1
    assert "owner@example.com" in resp.content


@pytest.mark.asyncio
async def test_run_turn_window_evicts_mid_conversation(tmp_path: Path):
    """Long conversation triggers eviction; tail (last turn) survives."""
    rw = RollingWindow(
        budget_tokens=1200, protect_tail_turns=2, eviction_batch_pct=0.4, sleep_dir=tmp_path
    )
    gw = _FakeGateway(["ack"] * 30)
    for i in range(25):
        await hot_path.run_turn(
            _env("payload " + "Q" * 200 + f" turn {i}"),
            gateway=gw,
            window=rw,
        )
    # Window stays near budget (head alone is ~500 tokens; allow 2x slack)
    assert rw.total_tokens() <= rw.budget_tokens * 2
    # Something ended up in sleep/
    spills = list(tmp_path.glob("spill-*.jsonl"))
    assert spills, "expected at least one spill file"
    # Head still intact
    assert len(rw.head) == 1
