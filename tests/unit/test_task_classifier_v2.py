from __future__ import annotations

from types import SimpleNamespace

import pytest

from norax.brain.hot_path import task_classifier_v2 as classifier


@pytest.mark.parametrize(
    ("message", "task_class", "depth"),
    [
        ("", "trivial_social", "fast"),
        ("rm -rf ./cache", "danger_sensitive", "deep"),
        ("research current best practices", "research", "deep"),
        ("implement a Python module", "coding", "deep"),
        ("restart the service", "ops_deploy", "deep"),
        ("investigate the timeout", "debug_investigate", "deep"),
        ("remember this preference", "memory_instruction", "deep"),
        ("design the architecture", "complex_planning", "deep"),
        ("hello!", "trivial_social", "fast"),
        ("thanks", "ack_feedback", "fast"),
        ("status?", "simple_status", "normal"),
        ("Why is the sky blue?", "simple_question", "normal"),
        ("unmapped intent", "ambiguous", "normal"),
        ("One? Two! Three. Four?", "complex_planning", "deep"),
        ("x" * 221, "complex_planning", "deep"),
    ],
)
def test_regex_classifier_covers_each_route(message, task_class, depth):
    result = classifier._regex_classify(message)
    assert (result.task_class, result.depth) == (task_class, depth)
    assert 0.0 <= result.confidence <= 1.0


def test_long_status_and_long_question_do_not_take_short_fast_lanes():
    status = "status " + "x" * 80
    question = "What " + "x" * 90 + "?"
    assert classifier._regex_classify(status).task_class == "ambiguous"
    assert classifier._regex_classify(question).task_class == "ambiguous"


def _semantic(
    task_class="coding",
    confidence=0.8,
    depth="deep",
    max_tool_calls=250,
    reason="semantic:test",
):
    return SimpleNamespace(
        task_class=task_class,
        confidence=confidence,
        depth=depth,
        max_tool_calls=max_tool_calls,
        reason=reason,
    )


def test_high_confidence_regex_does_not_invoke_semantic(monkeypatch):
    def should_not_run(_message):
        raise AssertionError("semantic fallback must not run")

    monkeypatch.setattr(classifier, "semantic_route", should_not_run)
    assert classifier.classify_task("hello").tier == "regex"


def test_semantic_failure_preserves_regex_result(monkeypatch):
    def fail(_message):
        raise RuntimeError("router unavailable")

    monkeypatch.setattr(classifier, "semantic_route", fail)
    expected = classifier._regex_classify("unmapped intent")
    assert classifier.classify_task("unmapped intent") == expected


@pytest.mark.parametrize(
    ("confidence", "expected_class", "expected_depth", "expected_tier"),
    [
        (0.80, "coding", "deep", "merged"),
        (0.60, "coding", "normal", "merged"),
        (0.45, "ambiguous", "normal", "merged"),
        (0.20, "ambiguous", "normal", "regex"),
    ],
)
def test_ambiguous_regex_uses_semantic_confidence_bands(
    monkeypatch, confidence, expected_class, expected_depth, expected_tier
):
    monkeypatch.setattr(
        classifier,
        "semantic_route",
        lambda _message: _semantic(confidence=confidence),
    )
    result = classifier.classify_task("unmapped intent")
    assert (result.task_class, result.depth, result.tier) == (
        expected_class,
        expected_depth,
        expected_tier,
    )
    if expected_tier == "merged":
        assert "semantic" in result.reason


def test_low_confidence_regex_is_boosted_when_semantic_agrees(monkeypatch):
    regex_result = classifier.TaskClassification(
        "simple_question", "normal", 0.78, 4, "short question"
    )
    monkeypatch.setattr(classifier, "_regex_classify", lambda _message: regex_result)
    monkeypatch.setattr(
        classifier,
        "semantic_route",
        lambda _message: _semantic(task_class="simple_question", confidence=0.9),
    )
    result = classifier.classify_task("question")
    assert result.confidence == pytest.approx(0.83)
    assert result.tier == "merged"
    assert result.reason.endswith("semantic_agrees")


def test_semantic_disagreement_never_downgrades_danger(monkeypatch):
    danger = classifier.TaskClassification("danger_sensitive", "deep", 0.84, 250, "danger")
    monkeypatch.setattr(classifier, "_regex_classify", lambda _message: danger)
    monkeypatch.setattr(classifier, "semantic_route", lambda _message: _semantic())
    assert classifier.classify_task("danger") == danger


@pytest.mark.parametrize("tool_budget", [None, 4])
def test_low_confidence_disagreement_hedges_without_reclassifying(monkeypatch, tool_budget):
    regex_result = classifier.TaskClassification(
        "simple_question", "deep", 0.79, tool_budget, "question"
    )
    monkeypatch.setattr(classifier, "_regex_classify", lambda _message: regex_result)
    monkeypatch.setattr(classifier, "semantic_route", lambda _message: _semantic())
    result = classifier.classify_task("question")
    assert result.task_class == "simple_question"
    assert result.depth == "normal"
    assert result.max_tool_calls == (tool_budget or 500)
    assert "semantic_disagrees:coding" in result.reason


def test_reasonably_confident_regex_survives_semantic_noise(monkeypatch):
    regex_result = classifier.TaskClassification("simple_status", "normal", 0.82, 4, "status")
    monkeypatch.setattr(classifier, "_regex_classify", lambda _message: regex_result)
    monkeypatch.setattr(classifier, "semantic_route", lambda _message: _semantic())
    assert classifier.classify_task("status") == regex_result
