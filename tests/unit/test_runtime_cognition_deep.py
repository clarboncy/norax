from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from norax.brain import active_inference as active_module
from norax.brain import analogy_engine as analogy_module
from norax.brain import best_of_n as best_module
from norax.brain import curiosity_engine as curiosity_module
from norax.brain import domain_transfer as transfer_module
from norax.brain import harness_optimizer
from norax.brain import metacognitive as metacognitive_module
from norax.brain import self_model as self_model_module
from norax.brain.sleep import replay as replay_module
from norax.memory import tool_experience as tool_experience_module
from norax.runtime import cognition as module


class _Capabilities:
    def __init__(self):
        self.ok = []
        self.failed = []

    def try_init(self, name, factory):
        try:
            value = factory()
        except Exception as exc:
            self.mark_failed(name, exc)
            return None
        self.mark_ok(name)
        return value

    def mark_ok(self, name):
        self.ok.append(name)

    def mark_failed(self, name, error):
        self.failed.append((name, type(error).__name__))


class _Loadable:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.loaded = False

    def load(self):
        self.loaded = True


def _cognitive_runtime(*, memory_root=None, experimental=True, active=False):
    return SimpleNamespace(
        _output_verifier=None,
        _self_model=None,
        _active_inference=None,
        _metacognitive=None,
        _best_of_n=None,
        _curiosity=None,
        _domain_transfer=None,
        _analogy_engine=None,
        _experimental_cognitive_signals=experimental,
        _active_inference_enabled=active,
        _memory_root=memory_root,
        _episodic=object(),
        gateway=object(),
        capabilities=_Capabilities(),
    )


def test_cognitive_components_initialize_without_memory_root_and_rebind_self_model(monkeypatch):
    monkeypatch.setenv("NORAX_BEST_OF_N", "yes")
    monkeypatch.setattr(self_model_module, "SelfModel", _Loadable)
    monkeypatch.setattr(active_module, "ActiveInference", _Loadable)
    monkeypatch.setattr(metacognitive_module, "MetacognitiveCalibration", _Loadable)
    monkeypatch.setattr(best_module, "BestOfN", _Loadable)
    monkeypatch.setattr(curiosity_module, "CuriosityEngine", _Loadable)
    monkeypatch.setattr(transfer_module, "DomainTransfer", _Loadable)
    monkeypatch.setattr(analogy_module, "AnalogyEngine", _Loadable)
    runtime = _cognitive_runtime(memory_root=None)

    module.CognitionMixin._ensure_cognitive_components(runtime)

    assert runtime._self_model.loaded
    assert runtime._active_inference.loaded
    assert runtime._metacognitive.loaded
    assert runtime._curiosity.loaded
    assert runtime._best_of_n.kwargs == {"gateway": runtime.gateway}
    assert runtime._domain_transfer.kwargs["self_model"] is runtime._self_model
    assert runtime._analogy_engine.kwargs["episodic"] is runtime._episodic
    assert set(runtime.capabilities.ok) >= {
        "self_model",
        "active_inference",
        "metacognitive",
        "best_of_n",
        "curiosity_engine",
        "domain_transfer",
        "analogy_engine",
    }

    replacement = object()
    runtime._self_model = replacement
    module.CognitionMixin._ensure_cognitive_components(runtime)
    assert runtime._domain_transfer.self_model is replacement


def test_cognitive_component_failures_are_isolated_and_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("NORAX_BEST_OF_N", "true")

    class Broken:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("unavailable")

    monkeypatch.setattr(self_model_module, "SelfModel", Broken)
    monkeypatch.setattr(active_module, "ActiveInference", Broken)
    monkeypatch.setattr(metacognitive_module, "MetacognitiveCalibration", Broken)
    monkeypatch.setattr(best_module, "BestOfN", Broken)
    monkeypatch.setattr(curiosity_module, "CuriosityEngine", Broken)
    monkeypatch.setattr(transfer_module, "DomainTransfer", Broken)
    monkeypatch.setattr(analogy_module, "AnalogyEngine", Broken)
    runtime = _cognitive_runtime(memory_root=tmp_path)

    module.CognitionMixin._ensure_cognitive_components(runtime)

    failed_names = {name for name, _ in runtime.capabilities.failed}
    assert failed_names >= {
        "self_model",
        "active_inference",
        "metacognitive",
        "best_of_n",
        "curiosity_engine",
        "domain_transfer",
        "analogy_engine",
    }
    assert all(
        getattr(runtime, name) is None
        for name in (
            "_self_model",
            "_active_inference",
            "_metacognitive",
            "_best_of_n",
            "_curiosity",
            "_domain_transfer",
            "_analogy_engine",
        )
    )


def test_disabled_experimental_components_return_after_optional_best_of_n(monkeypatch):
    monkeypatch.setenv("NORAX_BEST_OF_N", "on")
    monkeypatch.setattr(best_module, "BestOfN", _Loadable)
    runtime = _cognitive_runtime(experimental=False, active=False)
    module.CognitionMixin._ensure_cognitive_components(runtime)
    assert runtime._best_of_n is not None
    assert runtime._self_model is None and runtime._curiosity is None


def test_existing_domain_transfer_without_self_model_is_left_unchanged(monkeypatch):
    monkeypatch.delenv("NORAX_BEST_OF_N", raising=False)
    runtime = _cognitive_runtime(experimental=True)
    transfer = SimpleNamespace(self_model="existing")
    runtime._domain_transfer = transfer
    runtime._analogy_engine = object()

    class BrokenSelf:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("unavailable")

    monkeypatch.setattr(self_model_module, "SelfModel", BrokenSelf)
    module.CognitionMixin._ensure_cognitive_components(runtime)
    assert transfer.self_model == "existing"


def _idle_runtime(tmp_path):
    coordinator = SimpleNamespace(
        consolidate_sleep=AsyncMock(
            return_value={
                "canonical_writes": 0,
                "files_processed": 0,
                "spills_processed": 0,
                "candidates_seen": 0,
                "duplicates_skipped": 0,
            }
        ),
        sync_projections=AsyncMock(),
        canonical_changed=Mock(),
    )
    return SimpleNamespace(
        _last_turn_time=0.0,
        _turn_work_count=0,
        _active_turn_tasks={},
        _memory_store=SimpleNamespace(root=tmp_path),
        _memory_coordinator=coordinator,
        events=SimpleNamespace(append=AsyncMock(), path=None),
        _idle_learning_enabled=False,
        _episodic=None,
        _skill_learner=None,
        _metacognitive=None,
        _harness_analysis_enabled=False,
        _rho_last_run=0.0,
        _rho_last_event_mtime_ns=None,
    )


def _one_idle_poll(monkeypatch):
    calls = 0

    async def sleep(_seconds):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(module.asyncio, "sleep", sleep)


@pytest.mark.asyncio
async def test_idle_loop_skips_recent_activity_and_absent_memory(tmp_path, monkeypatch):
    _one_idle_poll(monkeypatch)
    recent = _idle_runtime(tmp_path)
    recent._last_turn_time = module.time.time()
    await module.CognitionMixin._idle_sleep_loop(recent)
    recent._memory_coordinator.sync_projections.assert_not_awaited()

    _one_idle_poll(monkeypatch)
    absent = _idle_runtime(tmp_path)
    absent._memory_store = None
    await module.CognitionMixin._idle_sleep_loop(absent)
    absent._memory_coordinator.sync_projections.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_consolidation_without_canonical_writes_records_but_does_not_pre_sync(
    tmp_path, monkeypatch
):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    (sleep / "spill-a.jsonl").write_text("{}\n")
    (sleep / "spill-a.md").write_text("SPILL:v1\n")
    await module.CognitionMixin._idle_sleep_loop(runtime)
    runtime._memory_coordinator.consolidate_sleep.assert_awaited_once()
    runtime._memory_coordinator.sync_projections.assert_awaited_once()
    runtime.events.append.assert_awaited_once()


@pytest.mark.asyncio
async def test_idle_consolidation_error_does_not_block_following_maintenance(
    tmp_path, monkeypatch, caplog
):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    sleep = tmp_path / "sleep"
    sleep.mkdir()
    (sleep / "buffer-a.md").write_text("pending\n")
    runtime._memory_coordinator.consolidate_sleep.side_effect = RuntimeError("consolidation failed")
    await module.CognitionMixin._idle_sleep_loop(runtime)
    runtime._memory_coordinator.sync_projections.assert_awaited_once()
    assert "sleep_consolidation.error" in caplog.text


@pytest.mark.asyncio
async def test_idle_replay_records_patterns_marks_memory_and_ingests_tool_experience(
    tmp_path, monkeypatch
):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._idle_learning_enabled = True
    runtime._episodic = object()
    replay_result = SimpleNamespace(
        procedural_patterns=["pattern"],
        failure_patterns=["failure"],
        coactivation_pairs_observed=3,
        high_surprise_episodes_seen=2,
    )

    class Replay:
        def __init__(self, **kwargs):
            assert kwargs["episodic"] is runtime._episodic
            assert kwargs["memory_root"] == tmp_path

        def run(self):
            return replay_result

    class Experience:
        def __init__(self, root):
            assert root == tmp_path

        def ingest_replay_patterns(self, directory):
            assert directory == tmp_path / "procedural"
            return 2

    monkeypatch.setattr(replay_module, "HippocampalReplay", Replay)
    monkeypatch.setattr(tool_experience_module, "ToolExperienceMemory", Experience)
    await module.CognitionMixin._idle_sleep_loop(runtime)

    runtime._memory_coordinator.canonical_changed.assert_called_once_with("hippocampal_replay")
    assert any(
        call.args[0] == "hippocampal_replay" for call in runtime.events.append.await_args_list
    )


@pytest.mark.asyncio
async def test_idle_replay_without_coordinator_and_new_ingestion_still_records(
    tmp_path, monkeypatch
):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._memory_coordinator = None
    runtime._idle_learning_enabled = True
    runtime._episodic = object()

    class Replay:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return SimpleNamespace(
                procedural_patterns=["pattern"],
                failure_patterns=[],
                coactivation_pairs_observed=0,
                high_surprise_episodes_seen=0,
            )

    class Experience:
        def __init__(self, _root):
            pass

        def ingest_replay_patterns(self, _directory):
            return 0

    monkeypatch.setattr(replay_module, "HippocampalReplay", Replay)
    monkeypatch.setattr(tool_experience_module, "ToolExperienceMemory", Experience)
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert any(
        call.args[0] == "hippocampal_replay" for call in runtime.events.append.await_args_list
    )


@pytest.mark.asyncio
async def test_idle_replay_handles_no_patterns_ingest_failure_and_replay_failure(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level("DEBUG")
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._idle_learning_enabled = True
    runtime._episodic = object()

    class EmptyReplay:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return SimpleNamespace(procedural_patterns=[], failure_patterns=[])

    monkeypatch.setattr(replay_module, "HippocampalReplay", EmptyReplay)
    await module.CognitionMixin._idle_sleep_loop(runtime)
    runtime._memory_coordinator.canonical_changed.assert_not_called()

    _one_idle_poll(monkeypatch)

    class PatternReplay(EmptyReplay):
        def run(self):
            return SimpleNamespace(
                procedural_patterns=["pattern"],
                failure_patterns=[],
                coactivation_pairs_observed=0,
                high_surprise_episodes_seen=0,
            )

    class BrokenExperience:
        def __init__(self, _root):
            raise RuntimeError("experience failed")

    monkeypatch.setattr(replay_module, "HippocampalReplay", PatternReplay)
    monkeypatch.setattr(tool_experience_module, "ToolExperienceMemory", BrokenExperience)
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert "tool_experience.ingest_replay.error" in caplog.text

    _one_idle_poll(monkeypatch)

    class BrokenReplay(EmptyReplay):
        def run(self):
            raise RuntimeError("replay failed")

    monkeypatch.setattr(replay_module, "HippocampalReplay", BrokenReplay)
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert "hippocampal_replay.error" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage", ["missing_log", "empty_trajectories", "empty_patterns", "no_changes"]
)
async def test_idle_skill_learning_noop_paths(tmp_path, monkeypatch, stage):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._skill_learner = SimpleNamespace(mine=Mock(), generate=Mock())
    if stage != "missing_log":
        runtime.events.path = tmp_path / "events.jsonl"
        runtime.events.path.write_text("")
    if stage == "empty_trajectories":
        monkeypatch.setattr(harness_optimizer, "load_trajectories", lambda *_args, **_kwargs: [])
    elif stage == "empty_patterns":
        monkeypatch.setattr(
            harness_optimizer, "load_trajectories", lambda *_args, **_kwargs: ["trajectory"]
        )
        runtime._skill_learner.mine = Mock(return_value=[])
    elif stage == "no_changes":
        monkeypatch.setattr(
            harness_optimizer, "load_trajectories", lambda *_args, **_kwargs: ["trajectory"]
        )
        runtime._skill_learner.mine = Mock(return_value=["pattern"])
        runtime._skill_learner.generate = Mock(
            return_value=SimpleNamespace(skills_created=0, skills_updated=0, skills=[])
        )
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert not any(
        call.args[0] == "skill_learning" for call in runtime.events.append.await_args_list
    )


@pytest.mark.asyncio
async def test_idle_skill_learning_exception_is_contained(tmp_path, monkeypatch, caplog):
    caplog.set_level("DEBUG")
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime.events.path = tmp_path / "events.jsonl"
    runtime.events.path.write_text("")
    runtime._skill_learner = object()
    monkeypatch.setattr(
        harness_optimizer,
        "load_trajectories",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert "skill_learning.error" in caplog.text


@pytest.mark.asyncio
async def test_idle_skill_learning_records_without_optional_coordinator(tmp_path, monkeypatch):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._memory_coordinator = None
    runtime.events.path = tmp_path / "events.jsonl"
    runtime.events.path.write_text("")
    monkeypatch.setattr(
        harness_optimizer, "load_trajectories", lambda *_args, **_kwargs: ["trajectory"]
    )
    runtime._skill_learner = SimpleNamespace(
        mine=lambda _items: ["pattern"],
        generate=lambda _patterns: SimpleNamespace(
            skills_created=0,
            skills_updated=1,
            skills=["improved"],
        ),
    )
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert any(call.args[0] == "skill_learning" for call in runtime.events.append.await_args_list)


@pytest.mark.asyncio
async def test_idle_prune_projection_and_metacognitive_failures_are_isolated(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level("DEBUG")
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._episodic = SimpleNamespace(prune_old=Mock(side_effect=RuntimeError("prune")))
    runtime._memory_coordinator.sync_projections.side_effect = RuntimeError("projection")
    runtime._metacognitive = SimpleNamespace(get_report=Mock(side_effect=RuntimeError("report")))
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert "episodic.prune.error" in caplog.text
    assert "memory_projection_sync.error" in caplog.text
    assert "metacognitive.idle.error" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "report",
    [
        None,
        SimpleNamespace(is_reliable=False, is_overconfident=True),
        SimpleNamespace(is_reliable=True, is_overconfident=False),
    ],
)
async def test_idle_metacognitive_nonalert_paths(tmp_path, monkeypatch, report):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._metacognitive = SimpleNamespace(get_report=lambda: report)
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert not any(
        call.args[0] == "metacognitive_alert" for call in runtime.events.append.await_args_list
    )


@pytest.mark.asyncio
async def test_idle_metacognitive_alert_records_bounded_evidence(tmp_path, monkeypatch):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    report = SimpleNamespace(
        is_reliable=True,
        is_overconfident=True,
        bias_magnitude=0.23456,
        brier_score=0.34567,
        bias=SimpleNamespace(value="overconfident"),
        trend="rising",
    )
    runtime._metacognitive = SimpleNamespace(get_report=lambda: report)
    await module.CognitionMixin._idle_sleep_loop(runtime)
    call = next(
        call
        for call in runtime.events.append.await_args_list
        if call.args[0] == "metacognitive_alert"
    )
    assert call.args[1] == {
        "bias": "overconfident",
        "magnitude": 0.235,
        "brier": 0.346,
        "trend": "rising",
    }


@pytest.mark.asyncio
async def test_idle_rho_analysis_runs_once_per_new_event_snapshot(tmp_path, monkeypatch):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._harness_analysis_enabled = True
    runtime.events.path = tmp_path / "events.jsonl"
    runtime.events.path.write_text("{}\n")
    monkeypatch.setattr(module.time, "time", lambda: 10_000.0)
    observed = []

    def analyze(event_path, out_dir, **kwargs):
        observed.append((event_path, out_dir, kwargs))
        return {"status": "ok", "proposal_count": 2, "report_path": "/report"}

    monkeypatch.setattr(harness_optimizer, "analyze_harness", analyze)
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert observed[0][0] == runtime.events.path
    assert observed[0][1] == tmp_path / "state" / "harness_optimizer"
    assert observed[0][2] == {"k": 10, "limit": 200}
    assert runtime._rho_last_run == 10_000.0
    assert runtime._rho_last_event_mtime_ns == runtime.events.path.stat().st_mtime_ns
    assert any(call.args[0] == "rho_analysis" for call in runtime.events.append.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["too_soon", "no_path", "wrong_path", "missing", "same"])
async def test_idle_rho_analysis_noop_paths(tmp_path, monkeypatch, state):
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._harness_analysis_enabled = True
    monkeypatch.setattr(module.time, "time", lambda: 10_000.0)
    if state == "too_soon":
        runtime._rho_last_run = 9_000.0
    elif state == "wrong_path":
        runtime.events.path = "not-a-path"
    elif state in {"missing", "same"}:
        runtime.events.path = tmp_path / "events.jsonl"
        if state == "same":
            runtime.events.path.write_text("{}\n")
            runtime._rho_last_event_mtime_ns = runtime.events.path.stat().st_mtime_ns
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert not any(call.args[0] == "rho_analysis" for call in runtime.events.append.await_args_list)


@pytest.mark.asyncio
async def test_idle_rho_analysis_exception_and_outer_loop_error_are_contained(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level("DEBUG")
    _one_idle_poll(monkeypatch)
    runtime = _idle_runtime(tmp_path)
    runtime._harness_analysis_enabled = True
    runtime.events.path = tmp_path / "events.jsonl"
    runtime.events.path.write_text("{}\n")
    monkeypatch.setattr(
        harness_optimizer,
        "analyze_harness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rho failed")),
    )
    await module.CognitionMixin._idle_sleep_loop(runtime)
    assert "rho_analysis.error" in caplog.text

    sleep_calls = []

    async def sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    broken = SimpleNamespace()
    with pytest.raises(asyncio.CancelledError):
        await module.CognitionMixin._idle_sleep_loop(broken)
    assert sleep_calls == [120, 60]
