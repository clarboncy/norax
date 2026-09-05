from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from norax.brain.prompt_optimizer import BenchmarkTask, PromptOptimizer
from norax.runtime.autonomy import run_idle_prompt_optimization


class _Gateway:
    default_model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        return SimpleNamespace(
            content="objective answer",
            tool_calls=[],
            model=request.model,
            latency_ms=1.0,
        )


class _FailingGateway:
    default_model = "test-model"

    async def chat(self, request):
        del request
        raise ConnectionError("provider unavailable")


class _ToolGateway:
    default_model = "test-model"

    async def chat(self, request):
        assert request.tools[0]["function"]["name"] == "inspect_file"
        assert request.metadata["allow_text_tool_calls"] is True
        return SimpleNamespace(
            content="",
            tool_calls=[{"type": "function", "function": {"name": "inspect_file"}}],
            model=request.model,
            latency_ms=1.0,
        )


@pytest.mark.asyncio
async def test_optimizer_refuses_format_heuristics_without_live_evaluation(tmp_path):
    optimizer = PromptOptimizer(router=None, storage_dir=tmp_path)

    with pytest.raises(ValueError, match="live gateway"):
        await optimizer.optimize(
            "A long structured prompt with examples and constraints",
            [BenchmarkTask(input="question", expected_contains="answer")],
        )


@pytest.mark.asyncio
async def test_optimizer_requires_objective_assertions(tmp_path):
    gateway = _Gateway()
    optimizer = PromptOptimizer(router=gateway, storage_dir=tmp_path)

    with pytest.raises(ValueError, match="objective assertion"):
        await optimizer.optimize(
            "Answer accurately",
            [BenchmarkTask(input="question")],
        )

    assert gateway.calls == 0


@pytest.mark.asyncio
async def test_objective_score_is_normalized_over_configured_assertions(tmp_path):
    gateway = _Gateway()
    optimizer = PromptOptimizer(router=gateway, storage_dir=tmp_path)
    result = await optimizer.optimize(
        "Answer accurately",
        [
            BenchmarkTask(
                input="question",
                expected_contains=["objective", "answer"],
                expected_not_contains="fabricated",
            )
        ],
        iterations=1,
        variants_per_iter=1,
    )

    assert result.best_variant.score == 1.0
    assert result.best_variant.tested_on == 1
    assert result.iterations == 0
    assert gateway.calls == 1
    assert result.evaluations_attempted == 1
    assert result.evaluations_completed == 1
    assert result.evaluation_failures == 0


@pytest.mark.asyncio
async def test_tool_assertion_receives_a_real_tool_schema_for_local_or_cloud_models(tmp_path):
    optimizer = PromptOptimizer(router=_ToolGateway(), storage_dir=tmp_path)
    result = await optimizer.optimize(
        "Use the appropriate tool",
        [
            BenchmarkTask(
                input="inspect the file",
                tool_should_be_called="inspect_file",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "inspect_file",
                            "description": "Inspect a file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    }
                ],
            )
        ],
    )

    assert result.best_variant.score == 1.0
    assert result.evaluations_completed == 1


@pytest.mark.asyncio
async def test_tool_assertion_without_schema_is_rejected_before_provider_call(tmp_path):
    gateway = _Gateway()
    optimizer = PromptOptimizer(router=gateway, storage_dir=tmp_path)

    with pytest.raises(ValueError, match="supplied benchmark function tool"):
        await optimizer.optimize(
            "Use a tool",
            [BenchmarkTask(input="inspect", tool_should_be_called="inspect_file")],
        )

    assert gateway.calls == 0


@pytest.mark.asyncio
async def test_idle_prompt_evaluation_is_staged_not_deployed_and_deduplicated(tmp_path):
    storage = tmp_path / "state" / "prompt_opt"
    storage.mkdir(parents=True)
    benchmark = {
        "base_prompt": "Answer accurately",
        "tasks": [{"input": "question", "expected_contains": "objective answer"}],
    }
    (storage / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    gateway = _Gateway()

    first = await run_idle_prompt_optimization(tmp_path, gateway=gateway, min_episodes=1)
    first_call_count = gateway.calls
    second = await run_idle_prompt_optimization(tmp_path, gateway=gateway, min_episodes=1)

    assert first["status"] == "evaluated_not_deployed"
    assert first["deployed"] is False
    assert first["optimizations"] == 0
    assert first_call_count == 1
    assert second["status"] == "benchmark_unchanged"
    assert gateway.calls == first_call_count


@pytest.mark.asyncio
async def test_idle_prompt_gate_enforces_real_benchmark_minimum(tmp_path):
    storage = tmp_path / "state" / "prompt_opt"
    storage.mkdir(parents=True)
    (storage / "benchmark.json").write_text(
        json.dumps(
            {
                "base_prompt": "Answer accurately",
                "tasks": [{"input": "question", "expected_contains": "answer"}],
            }
        ),
        encoding="utf-8",
    )
    gateway = _Gateway()

    result = await run_idle_prompt_optimization(tmp_path, gateway=gateway, min_episodes=2)

    assert result["status"] == "insufficient_benchmark"
    assert result["required_tasks"] == 2
    assert gateway.calls == 0


@pytest.mark.asyncio
async def test_idle_prompt_benchmark_does_not_follow_symlinks(tmp_path):
    storage = tmp_path / "state" / "prompt_opt"
    storage.mkdir(parents=True)
    target = tmp_path / "controlled.json"
    target.write_text(
        json.dumps(
            {
                "base_prompt": "Untrusted",
                "tasks": [{"input": "question", "expected_contains": "answer"}],
            }
        ),
        encoding="utf-8",
    )
    (storage / "benchmark.json").symlink_to(target)

    result = await run_idle_prompt_optimization(
        tmp_path,
        gateway=_Gateway(),
        min_episodes=1,
    )

    assert result["status"] == "invalid_benchmark"


@pytest.mark.asyncio
async def test_optimizer_does_not_count_provider_failure_as_completed_evaluation(tmp_path):
    optimizer = PromptOptimizer(router=_FailingGateway(), storage_dir=tmp_path)

    with pytest.raises(RuntimeError, match="all prompt benchmark evaluations failed"):
        await optimizer.optimize(
            "Answer accurately",
            [BenchmarkTask(input="question", expected_contains="answer")],
        )

    assert not (tmp_path / "prompt_history.jsonl").exists()


@pytest.mark.asyncio
async def test_optimizer_history_appends_only_current_run_and_is_private(tmp_path):
    gateway = _Gateway()
    optimizer = PromptOptimizer(router=gateway, storage_dir=tmp_path)
    task = BenchmarkTask(input="question", expected_contains="objective answer")

    await optimizer.optimize("Answer accurately", [task])
    await optimizer.optimize("Answer accurately", [task])

    history_path = tmp_path / "prompt_history.jsonl"
    history = optimizer.load_history()
    assert len(history) == 2
    assert all(row["tested_on"] == 1 and row["failed_on"] == 0 for row in history)
    assert history_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_optimizer_history_does_not_follow_symlink(tmp_path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged\n", encoding="utf-8")
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "prompt_history.jsonl").symlink_to(outside)
    optimizer = PromptOptimizer(router=_Gateway(), storage_dir=storage)

    with pytest.raises(OSError):
        await optimizer.optimize(
            "Answer accurately",
            [BenchmarkTask(input="question", expected_contains="objective answer")],
        )

    assert outside.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.asyncio
async def test_optimizer_rejects_coerced_or_unbounded_controls(tmp_path):
    optimizer = PromptOptimizer(router=_Gateway(), storage_dir=tmp_path)
    task = BenchmarkTask(input="question", expected_contains="answer")

    with pytest.raises(ValueError, match="iterations"):
        await optimizer.optimize("Answer", [task], iterations=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model"):
        await optimizer.optimize("Answer", [task], model="x" * 257)
    with pytest.raises(ValueError, match="expected_contains"):
        await optimizer.optimize(
            "Answer",
            [BenchmarkTask(input="question", expected_contains=["x"] * 65)],
        )


@pytest.mark.asyncio
async def test_idle_benchmark_rejects_unsupported_fake_round_control(tmp_path):
    storage = tmp_path / "state" / "prompt_opt"
    storage.mkdir(parents=True)
    (storage / "benchmark.json").write_text(
        json.dumps(
            {
                "base_prompt": "Answer accurately",
                "tasks": [
                    {
                        "input": "question",
                        "expected_contains": "answer",
                        "max_rounds": 99,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = await run_idle_prompt_optimization(
        tmp_path,
        gateway=_Gateway(),
        min_episodes=1,
    )

    assert result["status"] == "invalid_benchmark"
    assert "max_rounds" in result["error"]
