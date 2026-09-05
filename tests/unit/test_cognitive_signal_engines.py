from __future__ import annotations

from types import SimpleNamespace

from norax.brain.analogy_engine import AnalogyEngine
from norax.brain.curiosity_engine import SAVE_EVERY, CuriosityEngine
from norax.brain.domain_transfer import DomainTransfer
from norax.memory.episodic import Episode, EpisodicBuffer
from norax.runtime.capability_registry import CapabilityRegistry
from norax.runtime.core import Runtime


def test_curiosity_persists_and_recognizes_a_repeated_turn(tmp_path) -> None:
    path = tmp_path / "curiosity.json"
    engine = CuriosityEngine(path)
    text = "Repair the failing deployment pipeline and verify the service health"

    first = engine.assess(text, entities=["pipeline"], domain="ops")
    assert first.novelty == 0.25
    assert first.is_new_domain is True
    assert first.explore_hint == ""

    for _ in range(SAVE_EVERY):
        engine.record(text, entities=["pipeline"], domain="ops")

    assert path.exists()
    restored = CuriosityEngine(path)
    restored.load()
    repeated = restored.assess(text, entities=["pipeline"], domain="ops")

    assert restored.total_turns == SAVE_EVERY
    assert repeated.surprise == 0.0
    assert repeated.is_new_domain is False
    assert repeated.is_new_entity_mix is False


def test_domain_transfer_uses_strong_source_domain_for_matching_practice(tmp_path) -> None:
    (tmp_path / "procedural_memory.md").write_text(
        "[domain:coding] Always verify outputs with focused tests before claiming success.\n"
    )
    self_model = SimpleNamespace(
        profile=SimpleNamespace(
            domain_stats={
                "coding": {"success_rate": 0.9, "sample_size": 20},
                "research": {"success_rate": 0.4, "sample_size": 8},
            }
        )
    )
    transfer = DomainTransfer(self_model=self_model, memory_root=tmp_path)

    hints = transfer.transfer_hints(
        "research",
        "Verify the research claims before publishing the report",
    )

    assert hints
    assert hints[0].startswith("TRANSFER[coding→research]")
    assert "verify outputs" in hints[0].lower()


def test_analogy_engine_retains_episode_task_type_and_successful_approach(tmp_path) -> None:
    episodes = [
        Episode(
            user_input="Repair the failing deployment service",
            tool_calls=[{"name": "read"}, {"name": "exec"}, {"name": "edit"}],
            response_preview="Inspected logs, patched the unit, and verified readiness.",
            task_type="ops",
            outcome_score=9.0,
        )
    ]
    episodic = SimpleNamespace(recent_episodes=lambda **_kwargs: episodes)
    engine = AnalogyEngine(memory_root=tmp_path, episodic=episodic)

    matches = engine.find_analogies(
        "Fix the failing deployment service",
        task_type="ops",
    )

    assert len(matches) == 1
    assert matches[0].task_type == "ops"
    assert matches[0].outcome_success is True
    assert matches[0].approach == "tools: read → exec → edit"


def test_runtime_reuses_stateful_cognitive_components(tmp_path) -> None:
    runtime = Runtime.__new__(Runtime)
    runtime._memory_root = tmp_path
    runtime._episodic = EpisodicBuffer(tmp_path / "episodic")
    runtime.gateway = SimpleNamespace()
    runtime.capabilities = CapabilityRegistry()
    for name in (
        "_output_verifier",
        "_self_model",
        "_active_inference",
        "_metacognitive",
        "_best_of_n",
        "_curiosity",
        "_domain_transfer",
        "_analogy_engine",
    ):
        setattr(runtime, name, None)

    runtime._experimental_cognitive_signals = False
    runtime._active_inference_enabled = False
    runtime._ensure_cognitive_components()
    verifier_identity = id(runtime._output_verifier)
    assert runtime._self_model is None
    assert runtime._active_inference is None
    assert runtime._metacognitive is None
    assert runtime._curiosity is None
    assert runtime._domain_transfer is None
    assert runtime._analogy_engine is None

    runtime._ensure_cognitive_components()

    assert id(runtime._output_verifier) == verifier_identity

    runtime._experimental_cognitive_signals = True
    runtime._ensure_cognitive_components()

    assert runtime._self_model is not None
    assert runtime._active_inference is not None
    assert runtime._metacognitive is not None
    assert runtime._curiosity is not None
    assert runtime._domain_transfer is not None
    assert runtime._analogy_engine is not None
    assert runtime._domain_transfer.self_model is runtime._self_model


def test_active_inference_telemetry_can_be_enabled_independently(tmp_path) -> None:
    runtime = Runtime.__new__(Runtime)
    runtime._memory_root = tmp_path
    runtime._episodic = EpisodicBuffer(tmp_path / "episodic")
    runtime.gateway = SimpleNamespace()
    runtime.capabilities = CapabilityRegistry()
    for name in (
        "_output_verifier",
        "_self_model",
        "_active_inference",
        "_metacognitive",
        "_best_of_n",
        "_curiosity",
        "_domain_transfer",
        "_analogy_engine",
    ):
        setattr(runtime, name, None)
    runtime._experimental_cognitive_signals = False
    runtime._active_inference_enabled = True

    runtime._ensure_cognitive_components()

    assert runtime._active_inference is not None
    assert runtime._self_model is None
    assert runtime._metacognitive is None


def test_domain_transfer_rejects_unattributed_tactics(tmp_path) -> None:
    (tmp_path / "procedural_memory.md").write_text(
        "Always verify outputs with focused tests before claiming success.\n"
    )
    self_model = SimpleNamespace(
        profile=SimpleNamespace(domain_stats={"coding": {"success_rate": 0.9, "sample_size": 20}})
    )

    transfer = DomainTransfer(self_model=self_model, memory_root=tmp_path)

    assert transfer.build_library() == 0
