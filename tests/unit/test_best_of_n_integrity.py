from __future__ import annotations

import asyncio

import pytest

from norax.brain.best_of_n import BestOfN
from norax.gateway_client import GatewayResponse


class _Gateway:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        return GatewayResponse(
            request_id="candidate",
            model=request.model,
            content=self.content,
            tool_calls=[],
            usage={"input_tokens": 2, "output_tokens": 3},
            raw={},
        )


@pytest.mark.asyncio
async def test_single_alternative_is_scored_instead_of_being_inert() -> None:
    gateway = _Gateway(
        "## Result\n\nThe parser fix is implemented in `parser.py` and the targeted tests pass."
    )
    selector = BestOfN(gateway)

    candidate = await selector.generate(
        model="test-model",
        messages=[{"role": "user", "content": "Fix the parser and test it"}],
        user_request="Fix the parser and test it",
        task_type="coding",
        n=1,
    )

    assert gateway.calls == 1
    assert candidate.score > 0
    assert candidate.signals
    assert candidate.usage == {"input_tokens": 2, "output_tokens": 3}


@pytest.mark.asyncio
async def test_parallel_candidate_usage_reports_every_generation() -> None:
    gateway = _Gateway(
        "## Result\n\nThe implementation and focused regression checks are complete."
    )
    selector = BestOfN(gateway)

    candidate = await selector.generate(
        model="test-model",
        messages=[{"role": "user", "content": "Complete the implementation"}],
        task_type="coding",
        n=3,
    )

    assert gateway.calls == 3
    assert candidate.usage == {"input_tokens": 6, "output_tokens": 9}


@pytest.mark.asyncio
async def test_candidate_count_is_bounded_before_any_generation() -> None:
    gateway = _Gateway("unused")
    selector = BestOfN(gateway)

    with pytest.raises(ValueError, match="between 1 and 4"):
        await selector.generate(
            model="test-model",
            messages=[{"role": "user", "content": "x"}],
            n=5,
        )

    assert gateway.calls == 0


@pytest.mark.asyncio
async def test_single_candidate_honors_generation_timeout() -> None:
    class HangingGateway:
        def __init__(self) -> None:
            self.cancelled = False

        async def chat(self, _request):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    gateway = HangingGateway()
    selector = BestOfN(gateway)

    with pytest.raises(TimeoutError):
        await selector.generate(
            model="test-model",
            messages=[{"role": "user", "content": "x"}],
            n=1,
            timeout_seconds=0.02,
        )

    assert gateway.cancelled is True


def test_generic_success_language_is_not_counted_as_tool_evidence() -> None:
    selector = BestOfN(_Gateway("unused"))
    trace = [
        {
            "name": "read",
            "args": {"path": "src/parser.py"},
            "result": {"ok": True, "path": "src/parser.py"},
        }
    ]

    _, generic = selector.score_output(
        "Everything was verified successfully and the result is confirmed.",
        tool_trace=trace,
    )
    _, grounded = selector.score_output(
        "The relevant implementation is in src/parser.py.",
        tool_trace=trace,
    )

    assert generic["tool_evidence"] == 0.0
    assert grounded["tool_evidence"] > generic["tool_evidence"]
