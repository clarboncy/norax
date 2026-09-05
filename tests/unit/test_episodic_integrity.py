from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import norax.memory.episodic as episodic_module
from norax.memory.episodic import Episode, EpisodicBuffer


def test_episode_round_trip_preserves_strict_outcome_receipts(tmp_path: Path) -> None:
    buffer = EpisodicBuffer(tmp_path / "episodic")
    episode = Episode(
        user_input="repair and verify",
        retrieval_hits=["fact-1"],
        tool_calls=[{"name": "exec", "ok": True}],
        outcome_score=9,
        accepted_outcome=True,
        verified_outcome=True,
        objective_outcome_observed=True,
        training_eligible=True,
    )

    assert buffer.record(episode) is True
    restored = buffer.load_day(time.strftime("%Y-%m-%d"))

    assert len(restored) == 1
    assert restored[0].tool_calls == [{"name": "exec", "ok": True}]
    assert restored[0].verified_outcome is True
    path = buffer._get_today_file()
    assert path.stat().st_mode & 0o777 == 0o600


def test_loader_skips_truthy_or_nonfinite_learning_claims(tmp_path: Path) -> None:
    buffer = EpisodicBuffer(tmp_path / "episodic")
    day = "2026-09-05"
    path = buffer.root / f"episodes-{day}.jsonl"
    valid = Episode(timestamp=1, user_input="valid").to_dict()
    truthy = {**valid, "verified_outcome": 1}
    nonfinite = {**valid, "outcome_score": float("nan")}
    path.write_text(
        json.dumps(valid) + "\n" + json.dumps(truthy) + "\n" + json.dumps(nonfinite) + "\n"
    )

    loaded = buffer.load_day(day)

    assert len(loaded) == 1
    assert loaded[0].verified_outcome is False


def test_record_rejects_symlink_and_surfaces_capacity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = EpisodicBuffer(tmp_path / "episodic")
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged\n", encoding="utf-8")
    buffer._get_today_file().symlink_to(outside)

    with pytest.raises(OSError):
        buffer.record(Episode(user_input="must not escape"))
    assert outside.read_text(encoding="utf-8") == "unchanged\n"

    buffer._get_today_file().unlink()
    monkeypatch.setattr(episodic_module, "_MAX_EPISODE_FILE_BYTES", 8)
    with pytest.raises(ValueError, match="exceed"):
        buffer.record(Episode(user_input="capacity failure is visible"))


def test_load_is_bounded_and_date_cannot_escape_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = EpisodicBuffer(tmp_path / "episodic")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        buffer.load_day("../outside")

    path = buffer.root / "episodes-2026-09-05.jsonl"
    path.write_bytes(b"x" * 9)
    monkeypatch.setattr(episodic_module, "_MAX_EPISODE_FILE_BYTES", 8)
    assert buffer.load_day("2026-09-05") == []


def test_prune_and_stats_ignore_linked_files_and_order_dates(tmp_path: Path) -> None:
    buffer = EpisodicBuffer(tmp_path / "episodic", max_days=1)
    old = buffer.root / "episodes-2000-01-01.jsonl"
    new = buffer.root / "episodes-2099-01-01.jsonl"
    old.write_text("{}\n")
    new.write_text("{}\n")
    outside = tmp_path / "outside.jsonl"
    outside.write_text("keep\n")
    (buffer.root / "episodes-1999-01-01.jsonl").symlink_to(outside)

    stats = buffer.stats()
    removed = buffer.prune_old()

    assert stats["days"] == 2
    assert stats["oldest"] == "episodes-2000-01-01"
    assert stats["newest"] == "episodes-2099-01-01"
    assert removed == 1
    assert not old.exists()
    assert new.exists()
    assert outside.read_text() == "keep\n"


def test_episode_rejects_coerced_flags_and_unbounded_arguments(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="verified_outcome"):
        Episode(verified_outcome="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tool_calls ok"):
        Episode(tool_calls=[{"name": "exec", "ok": 1}])
    with pytest.raises(ValueError, match="outcome_score"):
        Episode(outcome_score=11)

    buffer = EpisodicBuffer(tmp_path / "episodic")
    with pytest.raises(ValueError, match="days"):
        buffer.recent_episodes(days=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        buffer.recent_episodes(limit=0)


def test_episodic_root_cannot_be_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory"):
        EpisodicBuffer(linked)


def test_concurrent_records_are_not_lost(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    buffer = EpisodicBuffer(tmp_path / "episodic")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda i: buffer.record(Episode(user_input=f"turn-{i}")), range(64))
        )

    assert all(result is True for result in results)
    loaded = buffer.load_day(time.strftime("%Y-%m-%d"))
    assert {episode.user_input for episode in loaded} == {f"turn-{i}" for i in range(64)}
