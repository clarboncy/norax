"""Regression tests for idle-learning persistence and idempotency."""

from __future__ import annotations

import json
from pathlib import Path

from norax.brain.harness_optimizer import HarnessTrajectory
from norax.brain.meta_learning import PARAM_BOUNDS, MetaLearner
from norax.brain.prediction_network import PredictionNetwork
from norax.brain.skill_learner import SkillLearner
from norax.brain.sleep.replay import HippocampalReplay
from norax.memory.episodic import Episode
from norax.memory.tool_experience import ToolExperienceMemory


class _Episodes:
    def __init__(self, episodes: list[Episode]) -> None:
        self.episodes = episodes

    def recent_episodes(self, *, days: int, limit: int) -> list[Episode]:
        return self.episodes[:limit]


def _trajectory(trace_id: str, tools: list[str]) -> HarnessTrajectory:
    return HarnessTrajectory(
        trace_id=trace_id,
        ts="2026-07-22T00:00:00Z",
        goal="inspect, edit, and verify",
        task_type="coding",
        rounds=len(tools),
        tool_calls=len(tools),
        failures=0,
        writes=1,
        verified_after_write=True,
        final_len=20,
        tools=tools,
        outcome_score=1.0,
        outcome_label="success",
    )


def test_replay_upserts_generated_patterns_without_append_churn(tmp_path: Path):
    episodes = [
        Episode(
            timestamp=float(i),
            tool_calls=[{"name": "read"}, {"name": "exec"}],
            verified_outcome=True,
            objective_outcome_observed=True,
            training_eligible=True,
        )
        for i in range(1, 6)
    ]
    replay = HippocampalReplay(
        episodic=_Episodes(episodes),  # type: ignore[arg-type]
        memory_root=tmp_path,
        min_pattern_count=3,
    )

    first = replay.run()
    out = next((tmp_path / "procedural").glob("replay-patterns-*.md"))
    content = out.read_text()
    second = replay.run()

    assert len(first.procedural_patterns) == 1
    assert second.procedural_patterns == []
    assert out.read_text() == content
    assert content.count("PATTERN:") == 1


def test_replay_never_promotes_unverified_tool_success(tmp_path: Path):
    episodes = [
        Episode(
            timestamp=float(i),
            tool_calls=[{"name": "read", "ok": True}, {"name": "exec", "ok": True}],
            accepted_outcome=True,
            verified_outcome=False,
            objective_outcome_observed=False,
            training_eligible=False,
        )
        for i in range(1, 6)
    ]
    replay = HippocampalReplay(
        episodic=_Episodes(episodes),  # type: ignore[arg-type]
        memory_root=tmp_path,
        min_pattern_count=3,
    )

    result = replay.run()

    assert result.procedural_patterns == []
    assert not list((tmp_path / "procedural").glob("replay-patterns-*.md"))


def test_prediction_network_restores_counts_and_replays_episode_once(tmp_path: Path):
    state = tmp_path / "prediction.json"
    state.write_text(
        json.dumps(
            {
                "transitions": {"coding": {"testing": 7}},
                "topic_history": ["coding"],
                "last_episode_timestamp": 10.0,
            }
        )
    )
    network = PredictionNetwork(state_path=state)
    network.load()

    episodes = [
        Episode(timestamp=11.0, user_input="run the tests"),
        Episode(timestamp=12.0, user_input="deploy the service"),
    ]
    assert network.transition_stats.total == 7
    assert network.record_episodes(episodes) == 2
    assert network.record_episodes(episodes) == 0
    assert network.last_episode_timestamp == 12.0


def test_skill_mining_counts_distinct_trajectories_not_repeated_windows(tmp_path: Path):
    learner = SkillLearner(tmp_path, min_episodes=2)
    repeated = ["read", "edit", "read", "edit", "read"]

    assert learner.mine([_trajectory("one", repeated)]) == []
    patterns = learner.mine(
        [
            _trajectory("one", repeated),
            _trajectory("two", ["read", "edit", "read"]),
        ]
    )
    assert any(p.tools == ["read", "edit", "read"] and p.count == 2 for p in patterns)


def test_skill_generation_does_not_rewrite_unchanged_skill(tmp_path: Path):
    learner = SkillLearner(tmp_path, min_episodes=2)
    patterns = learner.mine(
        [
            _trajectory("one", ["read", "edit", "read"]),
            _trajectory("two", ["read", "edit", "read"]),
        ]
    )

    first = learner.generate(patterns)
    second = learner.generate(patterns)
    assert first.skills_created == 1
    assert second.skills_created == 0
    assert second.skills_updated == 0


def test_skill_mining_rejects_partial_or_unverified_trajectories(tmp_path: Path):
    learner = SkillLearner(tmp_path, min_episodes=2)
    first = _trajectory("one", ["read", "edit", "read"])
    second = _trajectory("two", ["read", "edit", "read"])
    first.outcome_label = "partial"
    second.verified_after_write = False

    assert learner.mine([first, second]) == []


def test_replay_pattern_ingestion_is_durable_and_idempotent(tmp_path: Path):
    procedural = tmp_path / "procedural"
    procedural.mkdir()
    (procedural / "replay-patterns-2026-08-01.md").write_text(
        "PATTERN:id=abc123|tool_seq=read→edit→read|count=4|"
        "success_rate=100%|context=code,change|W4\n",
        encoding="utf-8",
    )
    (procedural / "replay-avoid-2026-08-01.md").write_text(
        "AVOID:tool=edit|error=old_text_not_found|count=2|after=read:1|"
        "context=code|recover_with=read|W4\n",
        encoding="utf-8",
    )
    memory = ToolExperienceMemory(tmp_path / "memory")

    assert memory.ingest_replay_patterns(procedural) == 2
    assert memory.ingest_replay_patterns(procedural) == 0
    assert len(memory.events_path.read_text(encoding="utf-8").splitlines()) == 2


def test_meta_learning_boundary_noop_is_not_reported_as_adjustment(tmp_path: Path):
    learner = MetaLearner(tmp_path / "meta.json")
    learner._params["vta_alpha"] = PARAM_BOUNDS["vta_alpha"][1]

    result = learner.tune("vta_alpha", 0.02, "already at upper bound")

    assert result is None
    assert learner._epoch_count == 0
    assert not (tmp_path / "meta_learning_audit.jsonl").exists()
