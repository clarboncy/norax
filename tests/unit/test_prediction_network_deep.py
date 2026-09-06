from __future__ import annotations

import json
import stat
from collections import defaultdict
from types import SimpleNamespace

import pytest

from norax.brain import prediction_network as module


def test_transition_stats_validate_rank_and_handle_empty_sources():
    stats = module.TransitionStats()
    for invalid in (True, 1.5):
        with pytest.raises(TypeError, match="top_k"):
            stats.predict("coding", invalid)
    assert stats.predict("coding", 0) == []
    assert stats.predict("missing") == []
    stats.transitions["empty"] = defaultdict(int)
    assert stats.predict("empty") == []

    stats.record("coding", "testing")
    stats.record("coding", "testing")
    stats.record("coding", "ops")
    assert stats.total == 3
    assert stats.predict("coding", 2) == [
        ("testing", pytest.approx(2 / 3)),
        ("ops", pytest.approx(1 / 3)),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "general"),
        ("implement python code", "coding"),
        ("pytest coverage", "testing"),
        ("look up web research", "research"),
        ("remember this memory", "memory"),
        ("restart service", "ops"),
        ("plan the roadmap", "planning"),
        ("send a Discord message", "communication"),
        ("contest ascending", "general"),
        ("code test", "coding"),
    ],
)
def test_task_classification_uses_tokens_and_stable_ties(text, expected):
    assert module._classify_task(text) == expected


def test_load_missing_unsafe_oversized_and_malformed_state_fails_closed(
    tmp_path, monkeypatch, caplog
):
    missing = module.PredictionNetwork(state_path=tmp_path / "missing.json")
    missing.load()
    assert missing._loaded is True and missing.transition_stats.total == 0

    target = tmp_path / "target.json"
    target.write_text("{}")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    unsafe = module.PredictionNetwork(state_path=linked)
    unsafe.load()
    assert unsafe._loaded is True and unsafe.transition_stats.total == 0

    monkeypatch.setattr(module, "_MAX_STATE_BYTES", 1)
    oversized = module.PredictionNetwork(state_path=target)
    oversized.load()
    assert oversized._loaded is True
    monkeypatch.setattr(module, "_MAX_STATE_BYTES", 1024)

    target.write_text("{")
    malformed = module.PredictionNetwork(state_path=target)
    malformed.transition_stats.record("preserved", "state")
    malformed.load()
    assert malformed.transition_stats.total == 1

    target.write_text("[]")
    malformed.load()
    assert malformed.transition_stats.total == 1
    assert "load failed" in caplog.text


def test_load_validates_nested_state_is_idempotent_and_caps_counts(tmp_path):
    path = tmp_path / "prediction.json"
    path.write_text(
        json.dumps(
            {
                "transitions": {
                    "coding": {
                        "testing": 7,
                        "fraction": 2.9,
                        "negative": -1,
                        "zero": 0,
                        "bool": True,
                        "string": "3",
                        "nan": float("nan"),
                        "": 3,
                        4: 3,
                        "huge": module._MAX_TRANSITION_COUNT + 5,
                    },
                    "": {"testing": 2},
                    "bad": "not-a-map",
                    4: {"testing": 2},
                },
                "entity_cooccurrence": {
                    "Norax": {
                        "Python": 4,
                        "fraction": 2.9,
                        "negative": -1,
                        "zero": 0,
                        "bool": True,
                        "string": "3",
                        "nan": float("nan"),
                        "": 2,
                        5: 2,
                        "huge": module._MAX_TRANSITION_COUNT + 5,
                    },
                    "": {"Python": 1},
                    "bad": "not-a-map",
                    4: {"Python": 1},
                },
                "topic_history": ["coding", 3, "testing"],
                "last_episode_timestamp": 12.5,
            }
        )
    )
    network = module.PredictionNetwork(state_path=path)
    network.load()
    assert network.transition_stats.total == 7 + 2 + 3 + 2 + module._MAX_TRANSITION_COUNT
    assert network.transition_stats.transitions["coding"] == {
        "testing": 7,
        "fraction": 2,
        "4": 3,
        "huge": module._MAX_TRANSITION_COUNT,
    }
    assert network.entity_cooccurrence["Norax"] == {
        "Python": 4,
        "fraction": 2,
        "5": 2,
        "huge": module._MAX_TRANSITION_COUNT,
    }
    assert list(network.topic_history) == ["coding", "testing"]
    assert network.last_episode_timestamp == 12.5

    network.load()
    assert network.transition_stats.total == 7 + 2 + 3 + 2 + module._MAX_TRANSITION_COUNT


@pytest.mark.parametrize(
    ("state", "expected_timestamp"),
    [
        ({"transitions": [], "entity_cooccurrence": [], "topic_history": "bad"}, 0.0),
        ({"last_episode_timestamp": True}, 0.0),
        ({"last_episode_timestamp": "12"}, 0.0),
        ({"last_episode_timestamp": float("nan")}, 0.0),
        ({"last_episode_timestamp": float("inf")}, 0.0),
        ({"last_episode_timestamp": -1}, 0.0),
        ({"last_episode_timestamp": 3}, 3.0),
    ],
)
def test_load_normalizes_optional_collections_and_timestamps(tmp_path, state, expected_timestamp):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    network = module.PredictionNetwork(state_path=path)
    network.load()
    assert network.last_episode_timestamp == expected_timestamp


def test_save_is_private_and_round_trips_without_double_counting(tmp_path):
    path = tmp_path / "nested" / "prediction.json"
    network = module.PredictionNetwork(state_path=path)
    network.transition_stats.record("coding", "testing")
    network.entity_cooccurrence["Norax"]["Python"] = 3
    network.topic_history.extend(["coding", "testing"])
    network.last_episode_timestamp = 9.0
    network.save()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    restored = module.PredictionNetwork(state_path=path)
    restored.load()
    restored.load()
    assert restored.transition_stats.total == 1
    assert restored.entity_cooccurrence["Norax"]["Python"] == 3
    assert list(restored.topic_history) == ["coding", "testing"]


def test_record_turn_tracks_only_real_transitions_and_unique_entity_pairs(tmp_path):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    network.record_turn("implement code", entities=["Norax"])
    network.record_turn("more code", task_type="coding", entities=None)
    assert network.transition_stats.total == 0
    network.record_turn(
        "run tests",
        entities=["Norax", "Python", "Norax", "", None, "pytest"],  # type: ignore[list-item]
    )
    assert network.transition_stats.predict("coding") == [("testing", 1.0)]
    assert network.entity_cooccurrence["Norax"] == {"Python": 1, "pytest": 1}
    assert network.entity_cooccurrence["Python"]["pytest"] == 1
    assert "Norax" not in network.entity_cooccurrence["Norax"]


def test_record_episodes_skips_invalid_old_and_nonfinite_timestamps_and_sorts(tmp_path):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    network.last_episode_timestamp = 10.0
    episodes = [
        SimpleNamespace(timestamp=True, user_input="ignored"),
        SimpleNamespace(timestamp="bad", user_input="ignored"),
        SimpleNamespace(timestamp=float("nan"), user_input="ignored"),
        SimpleNamespace(timestamp=float("inf"), user_input="ignored"),
        SimpleNamespace(timestamp=9, user_input="ignored"),
        SimpleNamespace(timestamp=12, user_input="restart service", task_type="ops"),
        SimpleNamespace(timestamp=11, user_input="run test", task_type="testing"),
        SimpleNamespace(timestamp=None, user_input="ignored"),
    ]
    assert network.record_episodes(episodes) == 2
    assert list(network.topic_history) == ["testing", "ops"]
    assert network.last_episode_timestamp == 12.0
    assert network.record_episodes([]) == 0


@pytest.mark.parametrize("invalid", [True, 1.5])
def test_predict_next_rejects_noninteger_limits(tmp_path, invalid):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    with pytest.raises(TypeError, match="top_k"):
        network.predict_next("code", top_k=invalid)


def test_predict_next_zero_limit_and_unknown_context_do_no_work(tmp_path):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    assert network.predict_next("code", top_k=0) == []
    assert network.predict_next() == []
    network.topic_history.append("unknown")
    assert network.predict_next() == []


def test_predict_next_uses_default_and_learned_transitions_with_safe_fallback(tmp_path):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    defaults = network.predict_next("implement code", top_k=3)
    assert [item.topic for item in defaults] == ["testing", "coding", "ops"]

    for _ in range(8):
        network.transition_stats.record("coding", "ops")
    for _ in range(2):
        network.transition_stats.record("coding", "testing")
    learned = network.predict_next("implement code", top_k=2)
    assert [(item.topic, item.confidence) for item in learned] == [
        ("ops", pytest.approx(0.8)),
        ("testing", pytest.approx(0.2)),
    ]

    fallback = network.predict_next("run pytest coverage", top_k=1)
    assert fallback[0].topic == "coding"


def test_predict_next_combines_entities_tools_trajectory_and_deduplicates(tmp_path):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    network.entity_cooccurrence["Norax"].update(
        {"Python": 20, "Memory": 8, "Discord": 5, "Fourth": 1}
    )
    predictions = network.predict_next(
        "Norax uses Agent.py",
        tool_trace=["bad", {"name": "read"}, {"name": "edit"}, {"name": "web_search"}],  # type: ignore[list-item]
        conversation_history=[
            {"content": "implement code"},
            "fix python code",
            {"content": "debug function"},
        ],
        top_k=10,
    )
    by_topic = {item.topic: item for item in predictions}
    assert by_topic["entity:Python"].confidence == pytest.approx(0.6)
    assert by_topic["entity:Memory"].confidence == pytest.approx(0.48)
    assert by_topic["entity:Discord"].confidence == pytest.approx(0.3)
    assert "entity:Fourth" not in by_topic
    assert by_topic["testing"].confidence == 0.5
    assert by_topic["coding"].confidence == 0.7
    assert len([item for item in predictions if item.topic == "coding"]) == 1

    already_defaulted = network.predict_next(
        "implement code",
        tool_trace=[{"name": "read"}, {"name": "edit"}, {"name": "web_search"}],
        top_k=5,
    )
    assert len([item for item in already_defaulted if item.topic == "testing"]) == 1
    assert len([item for item in already_defaulted if item.topic == "coding"]) == 1


def test_predict_next_trajectory_requires_three_matching_non_general_turns(tmp_path):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    mixed = network.predict_next(
        conversation_history=[{"content": "code"}, {"content": "test"}, {"content": "restart"}],
        top_k=10,
    )
    assert mixed == []
    general = network.predict_next(
        conversation_history=[{"content": "hello"}, "world", {"content": "ordinary"}],
        top_k=10,
    )
    assert general == []


class _SearchStore:
    def __init__(self):
        self.calls = []

    def search(self, topic, k):
        self.calls.append((topic, k))
        if topic == "error":
            raise RuntimeError("offline")
        return [f"hit:{topic}"] if topic == "coding" else []


def test_prefetch_supports_search_semantic_no_store_and_per_prediction_failure(tmp_path):
    predictions = [
        module.Prediction("entity:coding", 1.0),
        module.Prediction("missing", 0.5),
        module.Prediction("error", 0.2),
    ]
    empty = module.PredictionNetwork(state_path=tmp_path / "state")
    assert empty.prefetch(predictions) == {}

    search_store = _SearchStore()
    network = module.PredictionNetwork(search_store, state_path=tmp_path / "state")
    assert network.prefetch(predictions) == {"entity:coding": ["hit:coding"]}
    assert search_store.calls == [("coding", 3), ("missing", 3), ("error", 3)]

    semantic_store = SimpleNamespace(
        semantic=[
            "coding one",
            4,
            "CODING two",
            "coding three",
            "coding four",
            "unrelated",
        ]
    )
    semantic_network = module.PredictionNetwork(semantic_store, state_path=tmp_path / "state")
    semantic_results = semantic_network.prefetch(
        [module.Prediction("missing", 0.5), module.Prediction("coding", 1.0)]
    )
    assert semantic_results["coding"] == [
        "coding one",
        "CODING two",
        "coding three",
    ]
    assert (
        module.PredictionNetwork(object(), state_path=tmp_path / "state").prefetch(predictions)
        == {}
    )


def test_extract_entities_deduplicates_files_names_and_valid_tool_names(tmp_path):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    entities = network._extract_entities(
        "Norax opened Agent2.PY and Norax checked helper.ts",
        ["bad", {"name": 3}, {"name": "read"}, {"name": "read"}],  # type: ignore[list-item]
    )
    assert entities == ["Agent2.PY", "helper.ts", "Norax", "read"]


def test_summary_reports_transition_entity_and_recent_topic_counts(tmp_path):
    network = module.PredictionNetwork(state_path=tmp_path / "state")
    network.transition_stats.record("coding", "testing")
    network.entity_cooccurrence["Norax"]["Python"] = 1
    network.topic_history.extend([f"topic-{index}" for index in range(7)])
    summary = network.summary()
    assert "1 transitions" in summary
    assert "1 entities" in summary
    assert "topic-2" in summary and "topic-6" in summary and "topic-1" not in summary
