"""Integrity tests for persistent statistical cognitive state."""

from __future__ import annotations

import json
import stat
import time
from pathlib import Path

import pytest

from norax.brain.active_inference import ActiveInference
from norax.brain.curiosity_engine import SAVE_EVERY, CuriosityEngine, _shingles
from norax.brain.metacognitive import ConfidenceBias, MetacognitiveCalibration
from norax.brain.self_model import SelfModel, SkillStats


def test_active_inference_state_is_bounded_transactional_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "active.json"
    engine = ActiveInference(path)
    observation = engine.record_observation(
        tool="read",
        domain="coding",
        predicted_success=0.9,
        actual_success=True,
        predicted_duration_ms=10,
        actual_duration_ms=8,
        now=0,
    )
    assert observation.timestamp == 0
    engine.save()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    restored = ActiveInference(path)
    restored.load()
    assert restored.models["read:coding"].total_count == 1
    assert restored.total_observations == 1

    with pytest.raises(TypeError, match="boolean"):
        restored.record_observation(
            "read",
            "coding",
            0.5,
            "false",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        restored.record_observation("read", "coding", float("nan"), False)

    path.write_text("{", encoding="utf-8")
    restored.load()
    assert restored.models["read:coding"].total_count == 1

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    path.unlink()
    path.symlink_to(outside)
    restored.load()
    assert restored.models["read:coding"].total_count == 1


def test_self_model_round_trip_strict_outcomes_and_real_sample_threshold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "self.json"
    model = SelfModel(path)
    model.record_outcome("read", "coding", True, duration_ms=4, now=0)
    model.save()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    restored = SelfModel(path)
    restored.load()
    assert restored.profile.total_turns == 1
    assert restored.profile.tool_stats["read"].best_duration_ms == 4

    before = restored.profile.total_turns
    with pytest.raises(TypeError, match="boolean"):
        restored.record_outcome("read", "coding", "false")  # type: ignore[arg-type]
    assert restored.profile.total_turns == before

    restored.profile.tool_stats["weak"] = SkillStats(
        success_weight=0,
        failure_weight=5,
        total_count=5,
    )
    assert restored.profile.weak_tools(min_samples=5) == [("weak", 0.0)]
    assert restored.profile.weak_tools(min_samples=6) == []

    path.write_text("[]", encoding="utf-8")
    restored.load()
    assert "read" in restored.profile.tool_stats


def test_metacognitive_state_rejects_truthy_strings_and_filters_real_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metacog.json"
    calibration = MetacognitiveCalibration(path)
    with pytest.raises(TypeError, match="boolean"):
        calibration.record_prediction(0.8, "false")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration.record_prediction(float("nan"), False)

    observed_at = time.time()
    for index in range(10):
        calibration.record_prediction(
            0.6,
            True,
            tool="reliable",
            domain="coding",
            now=observed_at + index,
        )
        calibration.record_prediction(
            0.6,
            False,
            tool="unreliable",
            domain="coding",
            now=observed_at + index,
        )

    reliable = calibration.get_report(now=observed_at + 10, tool="reliable")
    unreliable = calibration.get_report(now=observed_at + 10, tool="unreliable")
    assert reliable.bias is ConfidenceBias.UNDERCONFIDENT
    assert unreliable.bias is ConfidenceBias.OVERCONFIDENT
    assert calibration.calibrate_confidence(0.6, tool="reliable") > 0.6
    assert calibration.calibrate_confidence(0.6, tool="unreliable") < 0.6

    calibration.save()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["predictions"].append(
        {
            "confidence": 0.99,
            "actual_success": "false",
            "timestamp": 1,
            "tool": "forged",
            "domain": "coding",
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    # A truthy string is discarded rather than becoming a success.
    before = list(calibration.predictions)
    calibration.load()
    assert list(calibration.predictions) == before


def test_curiosity_state_is_bounded_private_and_retries_failed_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "curiosity.json"
    engine = CuriosityEngine(path)
    huge_text = " ".join(f"token-{index}" for index in range(10_000))
    assert len(_shingles(huge_text)) <= 512

    for _ in range(SAVE_EVERY):
        engine.record("verify deployment", entities=["Service"], domain="ops")
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["window"][0]["ts"] = float("nan")
    payload["window"][1]["entities"] = ["x"] * 21
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = CuriosityEngine(path)
    restored.load()
    assert len(restored.window) == SAVE_EVERY - 2

    attempts: list[str] = []
    monkeypatch.setattr(engine, "save", lambda: attempts.append("failed") or False)
    engine._since_save = SAVE_EVERY - 1
    engine.record("one more", domain="ops")
    assert attempts == ["failed"]
    assert engine._since_save == SAVE_EVERY

    with pytest.raises(TypeError, match="entities"):
        engine.assess("request", entities=("not", "a", "list"))  # type: ignore[arg-type]
