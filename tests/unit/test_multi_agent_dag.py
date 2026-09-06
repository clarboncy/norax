"""Tests for multi-agent DAG dependency ordering (P1-1 reaudit fix).

Verifies that:
- Research completes before execution starts
- Code completes before verification starts
- Failed dependencies block dependents
- Independent tasks remain concurrent
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from norax.brain.multi_agent import (
    ROLE_CODE,
    ROLE_RESEARCH,
    ROLE_VERIFY,
    MultiAgentOrchestrator,
    SubAgentResult,
    SubTask,
)
from norax.gateway_client import GatewayResponse, SpendGuardTripped


@pytest.mark.asyncio
async def test_decomposed_tasks_keep_the_requested_completion_budget(monkeypatch) -> None:
    from norax.brain import agent_loop

    received: list[int] = []

    async def run_loop(**kwargs):
        received.append(kwargs["max_rounds"])
        return (
            GatewayResponse(
                request_id="r", model="fake", content="verified", raw={"verified_outcome": True}
            ),
            [],
            1,
            None,
        )

    monkeypatch.setattr(agent_loop, "run_agent_loop", run_loop)
    orchestrator = MultiAgentOrchestrator(object(), default_model="fake")
    result = await orchestrator.run(
        task="implement the parser and verify it",
        system_prompt="system",
        allowed_tools=["read"],
        sender_tier="owner",
    )
    assert received == [250, 250]
    assert result.verified_outcome is True


@pytest.mark.asyncio
async def test_research_completes_before_execution() -> None:
    """Research phase must finish before execution phase starts."""
    execution_order: list[str] = []

    async def mock_run_sub_agent(
        self, subtask, system_prompt, allowed_tools, sender_tier, *, event_log=None
    ):
        execution_order.append(f"{subtask.id}_start")
        await asyncio.sleep(0.05)
        execution_order.append(f"{subtask.id}_end")
        return SubAgentResult(
            subtask_id=subtask.id,
            role=subtask.role,
            success=True,
            output=f"result from {subtask.id}",
            elapsed=0.05,
        )

    router = MagicMock()
    mao = MultiAgentOrchestrator(router)

    subtasks = [
        SubTask(id="s_research", description="Research", role=ROLE_RESEARCH),
        SubTask(id="s_execute", description="Execute", role=ROLE_CODE, depends_on=["s_research"]),
    ]

    with patch.object(MultiAgentOrchestrator, "_run_sub_agent", mock_run_sub_agent):
        await mao._run_dag(subtasks, "sys", [], "owner", 4)

    # Research must start AND end before execute starts
    assert execution_order.index("s_research_end") < execution_order.index("s_execute_start")


@pytest.mark.asyncio
async def test_code_completes_before_verification() -> None:
    """Code phase must finish before verification phase starts."""
    execution_order: list[str] = []

    async def mock_run_sub_agent(
        self, subtask, system_prompt, allowed_tools, sender_tier, *, event_log=None
    ):
        execution_order.append(f"{subtask.id}_start")
        await asyncio.sleep(0.05)
        execution_order.append(f"{subtask.id}_end")
        return SubAgentResult(
            subtask_id=subtask.id,
            role=subtask.role,
            success=True,
            output=f"result from {subtask.id}",
            elapsed=0.05,
        )

    router = MagicMock()
    mao = MultiAgentOrchestrator(router)

    subtasks = [
        SubTask(id="s_code", description="Code", role=ROLE_CODE),
        SubTask(id="s_verify", description="Verify", role=ROLE_VERIFY, depends_on=["s_code"]),
    ]

    with patch.object(MultiAgentOrchestrator, "_run_sub_agent", mock_run_sub_agent):
        await mao._run_dag(subtasks, "sys", [], "owner", 4)

    assert execution_order.index("s_code_end") < execution_order.index("s_verify_start")


@pytest.mark.asyncio
async def test_failed_dependency_blocks_dependent() -> None:
    """If a dependency fails, the dependent task should also fail."""

    async def mock_run_sub_agent(
        self, subtask, system_prompt, allowed_tools, sender_tier, *, event_log=None
    ):
        if subtask.id == "s_research":
            return SubAgentResult(
                subtask_id=subtask.id,
                role=subtask.role,
                success=False,
                output="",
                error="research failed",
                elapsed=0.01,
            )
        return SubAgentResult(
            subtask_id=subtask.id,
            role=subtask.role,
            success=True,
            output="should not happen",
            elapsed=0.01,
        )

    router = MagicMock()
    mao = MultiAgentOrchestrator(router)

    subtasks = [
        SubTask(id="s_research", description="Research", role=ROLE_RESEARCH),
        SubTask(id="s_execute", description="Execute", role=ROLE_CODE, depends_on=["s_research"]),
    ]

    with patch.object(MultiAgentOrchestrator, "_run_sub_agent", mock_run_sub_agent):
        results = await mao._run_dag(subtasks, "sys", [], "owner", 4)

    assert results[0].success is False
    assert results[1].success is False
    assert "dependency s_research failed" in results[1].error


@pytest.mark.asyncio
async def test_independent_tasks_run_concurrently() -> None:
    """Tasks without dependencies should run in parallel."""
    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}

    async def mock_run_sub_agent(
        self, subtask, system_prompt, allowed_tools, sender_tier, *, event_log=None
    ):
        start_times[subtask.id] = time.monotonic()
        await asyncio.sleep(0.1)
        end_times[subtask.id] = time.monotonic()
        return SubAgentResult(
            subtask_id=subtask.id,
            role=subtask.role,
            success=True,
            output=f"result from {subtask.id}",
            elapsed=0.1,
        )

    router = MagicMock()
    mao = MultiAgentOrchestrator(router)

    subtasks = [
        SubTask(id="s_a", description="Task A", role=ROLE_RESEARCH),
        SubTask(id="s_b", description="Task B", role=ROLE_RESEARCH),
    ]

    with patch.object(MultiAgentOrchestrator, "_run_sub_agent", mock_run_sub_agent):
        await mao._run_dag(subtasks, "sys", [], "owner", 4)

    # Both should start before either ends (concurrent)
    assert start_times["s_a"] < end_times["s_b"]
    assert start_times["s_b"] < end_times["s_a"]


@pytest.mark.asyncio
async def test_predecessor_output_injected_into_dependent() -> None:
    """Dependent tasks should receive predecessor output in context."""
    captured_contexts: dict[str, str] = {}

    async def mock_run_sub_agent(
        self, subtask, system_prompt, allowed_tools, sender_tier, *, event_log=None
    ):
        captured_contexts[subtask.id] = subtask.context
        return SubAgentResult(
            subtask_id=subtask.id,
            role=subtask.role,
            success=True,
            output=f"output from {subtask.id}",
            elapsed=0.01,
        )

    router = MagicMock()
    mao = MultiAgentOrchestrator(router)

    subtasks = [
        SubTask(id="s_research", description="Research", role=ROLE_RESEARCH),
        SubTask(
            id="s_execute",
            description="Execute",
            role=ROLE_CODE,
            context="Base context",
            depends_on=["s_research"],
        ),
    ]

    with patch.object(MultiAgentOrchestrator, "_run_sub_agent", mock_run_sub_agent):
        await mao._run_dag(subtasks, "sys", [], "owner", 4)

    # Execute should have research output injected
    assert "output from s_research" in captured_contexts["s_execute"]
    assert "Base context" in captured_contexts["s_execute"]


def test_auto_decompose_sets_dependencies() -> None:
    """Auto-decomposed research→execute and code→verify should have depends_on."""
    router = MagicMock()
    mao = MultiAgentOrchestrator(router)

    # Research + code task
    subtasks = mao._auto_decompose("Research the API and implement the client", ["read", "write"])
    ids = {st.id: st for st in subtasks}
    assert "s_research" in ids
    assert "s_execute" in ids
    assert "s_research" in ids["s_execute"].depends_on

    # Code + verify task
    subtasks = mao._auto_decompose("Implement the feature and verify with tests", ["read", "write"])
    ids = {st.id: st for st in subtasks}
    assert "s_code" in ids
    assert "s_verify" in ids
    assert "s_code" in ids["s_verify"].depends_on


@pytest.mark.asyncio
async def test_duplicate_subtask_ids_fail_explicitly() -> None:
    mao = MultiAgentOrchestrator(MagicMock())
    subtasks = [
        SubTask(id="same", description="First", role=ROLE_RESEARCH),
        SubTask(id="same", description="Second", role=ROLE_CODE),
    ]

    results = await mao._run_dag(subtasks, "sys", [], "owner", 4)

    assert len(results) == 2
    assert all(not result.success for result in results)
    assert all("duplicate subtask id" in result.error for result in results)


@pytest.mark.asyncio
async def test_sub_agent_nonempty_incomplete_response_is_not_success() -> None:
    mao = MultiAgentOrchestrator(MagicMock(), default_model="test-model")
    response = GatewayResponse(
        request_id="test",
        model="test-model",
        content="I reached the round limit, but here is a fluent summary.",
        tool_calls=[],
        usage={},
        raw={"incomplete": True, "limit_reason": "hard_round_cap", "verified_outcome": False},
    )

    with patch(
        "norax.brain.agent_loop.run_agent_loop",
        AsyncMock(return_value=(response, [], 48, MagicMock())),
    ):
        result = await mao._run_sub_agent(
            SubTask(id="s0", description="Finish task"),
            "system",
            [],
            "owner",
        )

    assert result.success is False
    assert result.error == "hard_round_cap"


@pytest.mark.asyncio
async def test_sub_agent_never_swallows_global_spend_guard() -> None:
    mao = MultiAgentOrchestrator(MagicMock(), default_model="test-model")

    with patch(
        "norax.brain.agent_loop.run_agent_loop",
        AsyncMock(side_effect=SpendGuardTripped("minute", 2, 2)),
    ):
        with pytest.raises(SpendGuardTripped):
            await mao._run_sub_agent(
                SubTask(id="s0", description="work"),
                "system",
                [],
                "owner",
            )


def test_fusion_is_verified_only_when_every_subtask_is_verified() -> None:
    mao = MultiAgentOrchestrator(MagicMock())
    success = SubAgentResult(
        "ok",
        ROLE_RESEARCH,
        True,
        "Evidence",
        usage={"input_tokens": 5, "output_tokens": 2},
    )
    failure = SubAgentResult(
        "bad",
        ROLE_VERIFY,
        False,
        "Partial",
        error="verification failed",
        usage={"input_tokens": 7, "output_tokens": 3},
    )

    fused = mao._fuse([success, failure])

    assert fused.verified_outcome is False
    assert "verification failed" in fused.summary
    assert fused.usage == {"input_tokens": 12, "output_tokens": 5}


def test_single_failed_subtask_produces_visible_failure_summary() -> None:
    mao = MultiAgentOrchestrator(MagicMock())
    failed = SubAgentResult("s0", ROLE_CODE, False, "", error="tool unavailable")

    fused = mao._fuse([failed])

    assert fused.verified_outcome is False
    assert "tool unavailable" in fused.summary


@pytest.mark.asyncio
async def test_multi_agent_subtask_count_is_bounded_before_execution() -> None:
    mao = MultiAgentOrchestrator(MagicMock(), default_model="m")
    subtasks = [SubTask(id=f"s{i}", description="work") for i in range(9)]

    with pytest.raises(ValueError, match="limit of 8"):
        await mao.run(
            task="work",
            system_prompt="system",
            allowed_tools=[],
            sender_tier="owner",
            subtasks=subtasks,
        )
