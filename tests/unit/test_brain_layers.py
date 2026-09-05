"""Tests for neuromorphic brain layers (L1, L14, L15, L25, L27)."""

import math

import pytest

from norax.brain.hot_path.amygdala import score
from norax.brain.hot_path.arousal import ArousalState, assess
from norax.brain.hot_path.basal_ganglia import BasalGanglia
from norax.brain.hot_path.thalamus import classify
from norax.brain.hot_path.vta import VTA

# ── L1 Thalamus ──────────────────────────────────────────────────────────


class TestThalamus:
    def test_classify_command(self):
        r = classify("fix the voice bot")
        assert r.msg_type == "command"
        assert r.pathway == "executive"

    def test_classify_question(self):
        r = classify("what is my wallet address?")
        assert r.msg_type == "question"
        assert r.pathway == "retrieval"

    def test_classify_feedback(self):
        r = classify("thanks that's perfect")
        assert r.msg_type == "feedback"

    def test_classify_social(self):
        r = classify("hey good morning")
        assert r.msg_type == "social"

    def test_classify_emergency_high_signal(self):
        r = classify("urgent: the server is down and broken")
        assert r.msg_type == "emergency"
        assert r.signal_strength >= 8

    def test_classify_empty(self):
        r = classify("")
        assert r.msg_type == "noise"
        assert r.signal_strength == 0

    def test_classify_directive(self):
        r = classify("from now on always use qwen3.8")
        assert r.msg_type == "directive"

    def test_complexity_scales(self):
        simple = classify("check status")
        complex_ = classify(
            "investigate the architecture and refactor the entire memory pipeline for better performance"
        )
        assert complex_.complexity > simple.complexity


# ── L27 RAS / Arousal ────────────────────────────────────────────────────


class TestArousal:
    def test_normal_message(self):
        r = assess("fix the voice bot", complexity=3, signal_strength=5)
        assert r.level_int >= 3
        assert "signal=" in r.reason

    def test_high_signal_boosts(self):
        low = assess("check status", signal_strength=3)
        high = assess("urgent server down", signal_strength=9)
        assert high.level_int > low.level_int

    def test_low_energy_reduces(self):
        r = assess(
            "hello", complexity=0, signal_strength=2, last_interaction_ago_sec=7200
        )  # 2 hours away
        assert r.level_int <= 4

    def test_returns_dataclass(self):
        r = assess("test")
        assert isinstance(r, ArousalState)
        assert 1 <= r.level_int <= 6


# ── L14 Amygdala ─────────────────────────────────────────────────────────


class TestAmygdala:
    def test_positive(self):
        r = score("great job, that's awesome!")
        assert r.valence > 0
        assert r.dominant == "positive"

    def test_negative(self):
        r = score("this is terrible, everything is broken")
        assert r.valence < 0
        assert r.dominant == "negative"

    def test_threat_detection(self):
        r = score("rm -rf / is dangerous")
        assert r.threat >= 5

    def test_neutral(self):
        r = score("please check the config file")
        assert abs(r.valence) <= 2

    def test_reward_pattern(self):
        r = score("we just shipped the new version and got our first payment!")
        assert r.valence > 2
        assert "reward" in r.tags

    def test_frustration_tags(self):
        r = score("I'm frustrated this keeps failing")
        assert "frustration" in r.tags


# ── L15 Basal Ganglia ────────────────────────────────────────────────────


class TestBasalGanglia:
    def test_explore_unknown(self):
        bg = BasalGanglia()
        r = bg.gate("deploy new service")
        assert r.decision == "explore"

    def test_go_after_positive_history(self, tmp_path):
        bg = BasalGanglia(state_path=tmp_path / "bg.json")
        for _ in range(5):
            bg.record_outcome("restart gateway", 8.0)
        r = bg.gate("restart gateway")
        assert r.decision in ("go", "habit")
        assert r.expected_reward > 5

    def test_nogo_after_failures(self, tmp_path):
        bg = BasalGanglia(state_path=tmp_path / "bg.json")
        for _ in range(5):
            bg.record_outcome("delete database", 1.0)
        r = bg.gate("delete database")
        assert r.decision == "nogo"
        assert r.expected_reward < 3

    def test_habit_formation(self, tmp_path):
        bg = BasalGanglia(state_path=tmp_path / "bg.json")
        for _ in range(8):
            bg.record_outcome("run tests", 9.0)
        r = bg.gate("run tests")
        assert r.decision == "habit"

    def test_persistence(self, tmp_path):
        state_file = tmp_path / "bg.json"
        bg1 = BasalGanglia(state_path=state_file)
        bg1.record_outcome("deploy", 9.0)
        bg1.record_outcome("deploy", 8.0)
        bg1.record_outcome("deploy", 9.0)
        # Reload from disk
        bg2 = BasalGanglia(state_path=state_file)
        r = bg2.gate("deploy")
        assert r.history_count == 3

    def test_batch_records_real_success_and_failure_once(self, tmp_path, monkeypatch):
        bg = BasalGanglia(state_path=tmp_path / "bg.json")
        saves = 0
        original_save = bg._save

        def counted_save():
            nonlocal saves
            saves += 1
            original_save()

        monkeypatch.setattr(bg, "_save", counted_save)
        bg.record_outcomes([("read", 10.0), ("write", 0.0), ("read", 10.0)])

        assert saves == 1
        assert bg.gate("read").expected_reward > 5
        assert bg.gate("write").expected_reward < 5

    @pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
    def test_nonfinite_outcome_is_rejected(self, score):
        bg = BasalGanglia()
        with pytest.raises(ValueError, match="finite"):
            bg.record_outcome("tool", score)


# ── L25 VTA ──────────────────────────────────────────────────────────────


class TestVTA:
    def test_neutral_prediction(self):
        vta = VTA()
        r = vta.predict("unknown action")
        assert r.expected_reward == 5.0
        assert r.trend == "unknown"

    def test_rpe_low_surprise(self, tmp_path):
        vta = VTA(state_path=tmp_path / "vta.json")
        # Set expectation near actual
        vta._expectations["deploy"] = 7.0
        r = vta.record_outcome("deploy", 7.5)
        assert r.abs_rpe < 2
        assert r.route == "scratchpad"

    def test_rpe_high_surprise_flashbulb(self, tmp_path):
        vta = VTA(state_path=tmp_path / "vta.json")
        # Expect 5, get 10 → big positive surprise
        vta._expectations["new feature"] = 5.0
        r = vta.record_outcome("new feature", 10.0)
        assert r.abs_rpe >= 4
        assert r.route == "immediate_flashbulb"

    def test_rpe_moderate_to_sleep(self, tmp_path):
        vta = VTA(state_path=tmp_path / "vta.json")
        vta._expectations["refactor"] = 5.0
        r = vta.record_outcome("refactor", 8.0)
        assert 2 <= r.abs_rpe < 4
        assert r.route == "sleep_buffer"

    def test_expectation_updates(self, tmp_path):
        vta = VTA(state_path=tmp_path / "vta.json")
        vta.record_outcome("task", 9.0)
        vta.record_outcome("task", 9.0)
        vta.record_outcome("task", 9.0)
        p = vta.predict("task")
        assert p.expected_reward > 7  # moved toward 9

    def test_persistence(self, tmp_path):
        state_file = tmp_path / "vta.json"
        vta1 = VTA(state_path=state_file)
        vta1.record_outcome("deploy", 9.0)
        # Reload
        vta2 = VTA(state_path=state_file)
        p = vta2.predict("deploy")
        assert p.history_count == 1
        assert p.expected_reward > 5
