from __future__ import annotations

import asyncio
import copy
import os
import time
from pathlib import Path
from typing import Any, cast

import pytest

from norax.dispatch import deep_research as research
from norax.dispatch import tools


def _valid_state(path: Path) -> dict[str, Any]:
    now = time.time()
    return {
        "version": 1,
        "topic": "agent systems",
        "slug": path.stem.removesuffix(".state"),
        "rounds_done": 1,
        "seen_urls": ["https://docs.example/one"],
        "queries_run": ["agent systems evidence"],
        "tensions": ["However, validation remains open"],
        "created": now - 1,
        "updated": now,
    }


def test_research_lock_open_and_type_failures_close_descriptors(
    tmp_path: Path, monkeypatch
) -> None:
    lock_path = tmp_path / "research.lock"

    def fail_open(*_args: Any, **_kwargs: Any) -> int:
        raise OSError("denied")

    original_open = os.open
    monkeypatch.setattr(research.os, "open", fail_open)
    with pytest.raises(research.ResearchPersistenceError, match="cannot be opened"):
        research._acquire_research_file_lock(lock_path)

    monkeypatch.setattr(research.os, "open", original_open)
    monkeypatch.setattr(research.stat, "S_ISREG", lambda _mode: False)
    with pytest.raises(research.ResearchPersistenceError, match="regular file"):
        research._acquire_research_file_lock(lock_path)

    descriptor = original_open(lock_path, os.O_RDONLY)
    os.close(descriptor)


def test_research_url_domain_and_relevance_edges() -> None:
    assert research._host_of("http://[invalid") == ""
    assert research._domain_tier("") == 2
    assert research._domain_tier("news.quora.com") == 1
    assert research._domain_tier("docs.openai.com") == 3
    assert research._domain_tier("docs.example") == 2
    assert (
        research._relevance_score(
            {"url": "https://quora.com/agent", "title": "agent systems"},
            {"agent", "systems"},
        )
        == -1000
    )


def test_state_normalization_and_small_helpers_cover_invalid_shapes(tmp_path: Path) -> None:
    path = tmp_path / "topic.state.json"
    assert research._bounded_strings("bad", max_items=2, max_chars=4) == []
    assert research._bounded_strings([None, " ", "abcdef"], max_items=3, max_chars=4) == ["abcd"]
    fresh = research._normalize_state("bad", path)
    assert fresh["slug"] == "topic"
    assert fresh["seen_urls"] == []
    defaults = research._normalize_state(
        {
            "topic": 1,
            "slug": "",
            "rounds_done": True,
            "seen_urls": [],
            "queries_run": [],
            "tensions": [],
            "created": "bad",
            "updated": float("nan"),
        },
        path,
    )
    assert defaults["topic"] == ""
    assert defaults["slug"] == "topic"
    assert defaults["rounds_done"] == 0

    target = ["old", "keep"]
    research._extend_unique(target, ["KEEP", "", "new"], max_items=2)
    assert target == ["keep", "new"]

    assert research._bounded_positive_int(" 5 ", name="rounds", maximum=3) == (3, None)
    assert research._bounded_positive_int("bad", name="rounds", maximum=3) == (
        None,
        "rounds_must_be_a_positive_integer",
    )
    assert research._bounded_positive_int("0", name="rounds", maximum=3) == (
        None,
        "rounds_must_be_a_positive_integer",
    )


def test_loaded_state_validation_rejects_every_corrupt_field(tmp_path: Path) -> None:
    path = tmp_path / "topic.state.json"
    valid = _valid_state(path)
    assert research._validate_loaded_state(valid, path)["slug"] == "topic"

    invalid_values: list[object] = [None, [], "bad"]
    for value in invalid_values:
        with pytest.raises(research.ResearchPersistenceError):
            research._validate_loaded_state(value, path)

    missing = copy.deepcopy(valid)
    del missing["topic"]
    with pytest.raises(research.ResearchPersistenceError, match="missing required"):
        research._validate_loaded_state(missing, path)

    for value in (True, "1", 2):
        state = copy.deepcopy(valid)
        state["version"] = value
        with pytest.raises(research.ResearchPersistenceError, match="unsupported"):
            research._validate_loaded_state(state, path)

    for value in (None, "", "x" * (research._MAX_TOPIC_CHARS + 1)):
        state = copy.deepcopy(valid)
        state["topic"] = value
        with pytest.raises(research.ResearchPersistenceError, match="topic"):
            research._validate_loaded_state(state, path)

    for value in (None, "other"):
        state = copy.deepcopy(valid)
        state["slug"] = value
        with pytest.raises(research.ResearchPersistenceError, match="slug"):
            research._validate_loaded_state(state, path)

    for value in (True, "1", -1, 1_000_001):
        state = copy.deepcopy(valid)
        state["rounds_done"] = value
        with pytest.raises(research.ResearchPersistenceError, match="round count"):
            research._validate_loaded_state(state, path)

    for field_name in ("seen_urls", "queries_run", "tensions"):
        for value in (None, [""], ["bad\x00value"], [123]):
            state = copy.deepcopy(valid)
            state[field_name] = value
            with pytest.raises(research.ResearchPersistenceError, match=field_name):
                research._validate_loaded_state(state, path)

    for url in ("http://[invalid", "file:///etc/passwd"):
        state = copy.deepcopy(valid)
        state["seen_urls"] = [url]
        with pytest.raises(research.ResearchPersistenceError, match="URL"):
            research._validate_loaded_state(state, path)

    for value in (True, "1", float("nan"), -1, time.time() + 1_000):
        state = copy.deepcopy(valid)
        state["updated"] = value
        with pytest.raises(research.ResearchPersistenceError, match="updated"):
            research._validate_loaded_state(state, path)

    state = copy.deepcopy(valid)
    state["updated"] = state["created"] - 1
    with pytest.raises(research.ResearchPersistenceError, match="predates"):
        research._validate_loaded_state(state, path)


def test_state_size_limit_and_auto_query_fallback(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "topic.state.json"
    state = _valid_state(path)
    monkeypatch.setattr(research, "_MAX_STATE_BYTES", 1)
    with pytest.raises(research.ResearchPersistenceError, match="byte limit"):
        research._save_state(path, state)

    topic = "agent systems"
    standard = [
        topic,
        f"{topic} how it works",
        f"{topic} best practices",
        f"{topic} limitations trade-offs",
        f"{topic} recent developments",
        f"{topic}: unresolved validation",
    ]
    queries = research._auto_queries(
        topic,
        {
            "queries_run": standard,
            "tensions": ["unresolved validation", "unresolved validation"],
        },
    )
    assert queries == ["agent systems: unresolved validation"]


def test_passage_extraction_handles_empty_short_long_numeric_and_tension() -> None:
    assert research._extract_passages("", "https://docs.example") == ("", [])
    long_sentence = "x" * 401 + "."
    text = (
        "short. "
        f"{long_sentence} "
        "However, the verified agent system still has 3 unresolved integration risks."
    )
    lead, passages = research._extract_passages(text, "https://docs.example", {"agent", "system"})
    assert lead
    assert passages == [
        "However, the verified agent system still has 3 unresolved integration risks."
    ]


def test_append_report_is_idempotent_and_scrubs_untrusted_fields(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    state = _valid_state(tmp_path / "topic.state.json")
    secret = "sk-" + "q" * 30
    sources: list[dict] = [
        {
            "url": "https://docs.example/one",
            "title": secret,
            "passages": ["Verified evidence"],
        },
        {"url": "https://docs.example/two", "error": secret},
    ]
    research._append_report(report, state, 1, [secret], sources)
    before = report.read_bytes()
    research._append_report(report, state, 1, [secret], sources)
    assert report.read_bytes() == before
    assert secret.encode() not in before
    assert b"<REDACTED:openai_key>" in before


def test_append_report_rejects_type_size_and_write_failures(tmp_path: Path, monkeypatch) -> None:
    state = _valid_state(tmp_path / "topic.state.json")
    report = tmp_path / "report.md"

    monkeypatch.setattr(research.stat, "S_ISREG", lambda _mode: False)
    with pytest.raises(research.ResearchPersistenceError, match="regular file"):
        research._append_report(report, state, 1, ["query"], [])

    monkeypatch.undo()
    report.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(research, "_MAX_REPORT_BYTES", 1)
    with pytest.raises(research.ResearchPersistenceError, match="byte limit"):
        research._append_report(report, state, 1, ["query"], [])

    monkeypatch.undo()
    empty_report = tmp_path / "empty.md"
    monkeypatch.setattr(research, "_MAX_REPORT_BYTES", 1)
    with pytest.raises(research.ResearchPersistenceError, match="byte limit"):
        research._append_report(empty_report, state, 1, ["query"], [])

    monkeypatch.undo()
    short_report = tmp_path / "short.md"
    monkeypatch.setattr(research.os, "write", lambda *_args, **_kwargs: 0)
    with pytest.raises(OSError, match="short write"):
        research._append_report(short_report, state, 1, ["query"], [])


def test_intel_ingest_missing_root_empty_target_duplicates_and_safe_slug(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_root = tmp_path / "missing"
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(missing_root))
    assert research._memory_intel_dir() is None
    assert research._ingest_intel("topic", "slug", tmp_path / "report.md", [], []) is None

    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(memory_root))
    assert research._ingest_intel("topic", "slug", tmp_path / "report.md", [], []) is None

    sources = [
        {"url": "", "title": "empty"},
        {"url": "https://docs.example/one", "title": "one"},
        {"url": "https://docs.example/one", "title": "duplicate"},
    ]
    target_value = research._ingest_intel(
        "topic",
        "../../unsafe/name",
        tmp_path / "report\nname.md",
        sources,
        [],
    )
    assert target_value is not None
    target = Path(target_value)
    assert target.parent == memory_root.resolve() / "intel"
    assert target.name == "research_unsafe_name.md"
    assert target.read_text(encoding="utf-8").count("https://docs.example/one") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"topic": cast(Any, 1)}, "topic_must_be_a_string"),
        ({"topic": "   "}, "topic_required"),
        ({"topic": "x" * (research._MAX_TOPIC_CHARS + 1)}, "topic_too_long"),
        (
            {"topic": "agent systems", "queries": cast(Any, "bad")},
            "queries_must_be_a_list_of_strings",
        ),
        (
            {"topic": "agent systems", "queries": ["x" * (research._MAX_QUERY_CHARS + 1)]},
            "query_too_long",
        ),
        (
            {"topic": "agent systems", "max_chars_per_source": True},
            "max_chars_per_source_must_be_a_positive_integer",
        ),
    ],
)
async def test_deep_research_rejects_all_invalid_top_level_inputs(
    tmp_path: Path,
    kwargs: dict[str, Any],
    error: str,
) -> None:
    result = await research.t_deep_research(out_dir=str(tmp_path / "research"), **kwargs)
    assert result["error"] == error


@pytest.mark.asyncio
async def test_deep_research_bounds_and_deduplicates_query_and_seed_lists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(research, "_run_deep_research", fake_run)
    queries = ["", "same", "SAME"] + [f"query {index}" for index in range(20)]
    seeds = ["", "https://same.example", "https://same.example"] + [
        f"https://seed-{index}.example" for index in range(20)
    ]
    result = await research.t_deep_research(
        topic="agent systems",
        queries=queries,
        seeds=seeds,
        rounds=1,
        out_dir=str(tmp_path / "research"),
    )
    assert result["ok"] is True
    assert len(captured["queries"]) == research._MAX_QUERIES_PER_ROUND
    assert len(captured["seeds"]) == research._MAX_SEEDS


@pytest.mark.asyncio
async def test_deep_research_scrubs_output_directory_errors(tmp_path: Path, monkeypatch) -> None:
    secret = "sk-" + "d" * 30

    def fail_out_dir(_value: str | None) -> Path:
        raise OSError(secret)

    monkeypatch.setattr(research, "_out_dir", fail_out_dir)
    result = await research.t_deep_research(
        topic="agent systems",
        rounds=1,
        out_dir=str(tmp_path / "research"),
    )
    assert result["error"] == "output_directory_unavailable"
    assert secret not in result["detail"]
    assert "<REDACTED:openai_key>" in result["detail"]


@pytest.mark.asyncio
async def test_research_lock_cache_prunes_locked_and_stale_entries(
    tmp_path: Path, monkeypatch
) -> None:
    research._research_locks.clear()
    locked = research.asyncio.Lock()
    await locked.acquire()
    research._research_locks["locked"] = locked
    for index in range(research._RESEARCH_LOCKS_MAX):
        research._research_locks[f"stale-{index}"] = research.asyncio.Lock()

    async def fake_run(**_kwargs: Any) -> dict:
        return {"ok": True}

    monkeypatch.setattr(research, "_run_deep_research", fake_run)
    try:
        result = await research.t_deep_research(
            topic="agent systems",
            rounds=1,
            out_dir=str(tmp_path / "research"),
        )
        assert result["ok"] is True
        assert "locked" in research._research_locks
        assert len(research._research_locks) == research._RESEARCH_LOCKS_MAX
    finally:
        locked.release()
        research._research_locks.clear()


@pytest.mark.asyncio
async def test_research_lock_cache_remains_bounded_when_every_slot_is_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    research._research_locks.clear()
    active_locks = [research.asyncio.Lock() for _ in range(research._RESEARCH_LOCKS_MAX)]
    for index, lock in enumerate(active_locks):
        await lock.acquire()
        research._research_locks[f"active-{index}"] = lock

    async def fake_run(**_kwargs: Any) -> dict:
        return {"ok": True}

    monkeypatch.setattr(research, "_run_deep_research", fake_run)
    try:
        result = await research.t_deep_research(
            topic="overflow topic",
            rounds=1,
            out_dir=str(tmp_path / "research"),
        )
        assert result["ok"] is True
        assert len(research._research_locks) == research._RESEARCH_LOCKS_MAX
        assert all(key.startswith("active-") for key in research._research_locks)
    finally:
        for lock in active_locks:
            lock.release()
        research._research_locks.clear()


@pytest.mark.asyncio
async def test_run_research_isolates_seed_search_and_fetch_failures(
    tmp_path: Path, monkeypatch
) -> None:
    secret = "sk-" + "f" * 30

    async def failed_seeds(*_args: Any, **_kwargs: Any) -> tuple[list[dict], list[str]]:
        raise RuntimeError("seed failed")

    async def search(*_args: Any, **_kwargs: Any) -> dict:
        return {
            "ok": True,
            "items": [
                {
                    "url": "https://docs.example/agent",
                    "title": "agent systems",
                    "snippet": "agent systems verified evidence",
                }
            ],
        }

    async def fetch(*_args: Any, **_kwargs: Any) -> dict:
        raise RuntimeError(secret)

    monkeypatch.setattr(research, "_seed_sources", failed_seeds)
    monkeypatch.setattr(tools, "t_web_search", search)
    monkeypatch.setattr(tools, "t_web_fetch", fetch)
    result = await research.t_deep_research(
        topic="agent systems",
        queries=["agent systems"],
        seeds=["https://seed.example"],
        rounds=1,
        out_dir=str(tmp_path / "research"),
    )
    assert result["ok"] is True
    assert result["fetch_errors_this_call"] == 1
    round_result = result["rounds"][0]
    assert round_result["seed_notes"] == ["seed discovery failed: RuntimeError"]
    assert secret not in round_result["sources"][0]["error"]
    assert "<REDACTED:openai_key>" in round_result["sources"][0]["error"]


@pytest.mark.asyncio
async def test_run_research_merges_seed_and_search_sources(tmp_path: Path, monkeypatch) -> None:
    async def seeds(*_args: Any, **_kwargs: Any) -> tuple[list[dict], list[str]]:
        return [
            {
                "url": "https://seed.example/agent",
                "title": "agent systems",
                "_prefetched_text": "Agent systems use verified evidence for reliable work.",
            }
        ], []

    async def search(*_args: Any, **_kwargs: Any) -> dict:
        return {
            "ok": True,
            "items": [
                {
                    "url": "https://search.example/agent",
                    "title": "agent systems",
                    "snippet": "agent systems verified evidence",
                }
            ],
        }

    async def fetch(*_args: Any, **_kwargs: Any) -> dict:
        return {"ok": True, "text": "Agent systems use independent verified evidence."}

    monkeypatch.setattr(research, "_seed_sources", seeds)
    monkeypatch.setattr(tools, "t_web_search", search)
    monkeypatch.setattr(tools, "t_web_fetch", fetch)
    result = await research.t_deep_research(
        topic="agent systems",
        queries=["agent systems"],
        seeds=["https://seed.example"],
        rounds=1,
        max_sources=2,
        out_dir=str(tmp_path / "research"),
    )
    assert result["new_sources_this_call"] == 2
    assert {item["url"] for item in result["digest_sources"]} == {
        "https://seed.example/agent",
        "https://search.example/agent",
    }


@pytest.mark.asyncio
async def test_run_research_falls_back_to_topic_when_auto_queries_are_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    queries_seen: list[str] = []

    async def search(*, query: str, **_kwargs: Any) -> dict:
        queries_seen.append(query)
        return {"ok": True, "items": []}

    monkeypatch.setattr(research, "_auto_queries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools, "t_web_search", search)
    result = await research.t_deep_research(
        topic="agent systems",
        rounds=1,
        out_dir=str(tmp_path / "research"),
    )
    assert result["ok"] is True
    assert queries_seen == ["agent systems"]


@pytest.mark.asyncio
async def test_run_research_propagates_seed_cancellation(tmp_path: Path, monkeypatch) -> None:
    async def cancelled_seeds(*_args: Any, **_kwargs: Any) -> tuple[list[dict], list[str]]:
        raise asyncio.CancelledError

    async def search(**_kwargs: Any) -> dict:
        return {"ok": False, "error": "offline"}

    monkeypatch.setattr(research, "_seed_sources", cancelled_seeds)
    monkeypatch.setattr(tools, "t_web_search", search)
    with pytest.raises(asyncio.CancelledError):
        await research.t_deep_research(
            topic="agent systems",
            queries=["agent systems"],
            seeds=["https://seed.example"],
            rounds=1,
            out_dir=str(tmp_path / "research"),
        )


@pytest.mark.asyncio
async def test_run_research_cancels_seed_task_when_parent_stops_during_search(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed_started = asyncio.Event()
    seed_settled = asyncio.Event()
    search_started = asyncio.Event()

    async def seeds(*_args: Any, **_kwargs: Any) -> tuple[list[dict], list[str]]:
        seed_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            seed_settled.set()
        return [], []

    async def search(**_kwargs: Any) -> dict:
        search_started.set()
        await asyncio.Event().wait()
        return {"ok": False, "error": "unreachable"}

    monkeypatch.setattr(research, "_seed_sources", seeds)
    monkeypatch.setattr(tools, "t_web_search", search)
    task = asyncio.create_task(
        research.t_deep_research(
            topic="agent systems",
            queries=["agent systems"],
            seeds=["https://seed.example"],
            rounds=1,
            out_dir=str(tmp_path / "research"),
        )
    )
    await seed_started.wait()
    await search_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seed_settled.is_set()


@pytest.mark.asyncio
async def test_run_research_propagates_cancelled_search_and_records_other_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def cancelled_search(**_kwargs: Any) -> dict:
        raise asyncio.CancelledError

    monkeypatch.setattr(tools, "t_web_search", cancelled_search)
    with pytest.raises(asyncio.CancelledError):
        await research.t_deep_research(
            topic="agent systems",
            queries=["cancelled"],
            rounds=1,
            out_dir=str(tmp_path / "cancelled"),
        )

    async def mixed_search(*, query: str, **_kwargs: Any) -> object:
        if query == "raises":
            raise RuntimeError("offline")
        if query == "malformed":
            return {"ok": True, "items": {"bad": "shape"}}
        if query == "nondict":
            return "bad response shape"
        return {"ok": False, "error": "provider unavailable"}

    monkeypatch.setattr(tools, "t_web_search", mixed_search)
    result = await research.t_deep_research(
        topic="agent systems",
        queries=["raises", "malformed", "nondict", "failed"],
        rounds=1,
        out_dir=str(tmp_path / "mixed"),
    )
    assert result["ok"] is True
    assert result["saturated"] is True
    assert result["rounds"][0]["search_errors"] == [
        "raises: RuntimeError",
        "failed: provider unavailable",
    ]


@pytest.mark.asyncio
async def test_run_research_handles_empty_and_duplicate_seed_urls_in_digest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def seeds(*_args: Any, **_kwargs: Any) -> tuple[list[dict], list[str]]:
        text = "Agent systems use verified evidence for reliable execution."
        return [
            {"url": "", "title": "empty", "_prefetched_text": text},
            {
                "url": "https://seed.example/agent",
                "title": "one",
                "_prefetched_text": text,
            },
            {
                "url": "https://seed.example/agent",
                "title": "duplicate",
                "_prefetched_text": text,
            },
        ], []

    async def search(**_kwargs: Any) -> dict:
        return {"ok": False, "error": "offline"}

    monkeypatch.setattr(research, "_seed_sources", seeds)
    monkeypatch.setattr(tools, "t_web_search", search)
    result = await research.t_deep_research(
        topic="agent systems",
        queries=["agent systems"],
        seeds=["https://seed.example"],
        rounds=1,
        max_sources=3,
        out_dir=str(tmp_path / "research"),
    )
    assert result["new_sources_this_call"] == 3
    assert [item["url"] for item in result["digest_sources"]] == ["https://seed.example/agent"]


@pytest.mark.asyncio
async def test_run_research_enforces_seen_and_tension_caps(tmp_path: Path, monkeypatch) -> None:
    slug = research._slugify("agent systems")
    state = research._fresh_state(tmp_path / f"{slug}.state.json")
    state["topic"] = "agent systems"
    state["slug"] = slug
    state["seen_urls"] = [f"https://e.co/{index}" for index in range(research._MAX_SEEN_URLS)]
    state["tensions"] = [f"old tension {index}" for index in range(20)]

    async def seeds(*_args: Any, **_kwargs: Any) -> tuple[list[dict], list[str]]:
        return [
            {
                "url": "https://new.example/agent",
                "title": "agent systems",
                "_prefetched_text": (
                    "However, the verified agent system has a newly discovered integration risk."
                ),
            }
        ], []

    async def search(**_kwargs: Any) -> dict:
        return {"ok": False, "error": "offline"}

    monkeypatch.setattr(research, "_load_state", lambda _path: state)
    monkeypatch.setattr(research, "_seed_sources", seeds)
    monkeypatch.setattr(research, "_append_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(research, "_save_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(research, "_ingest_intel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tools, "t_web_search", search)
    result = await research._run_deep_research(
        topic="agent systems",
        queries=["agent systems"],
        seeds=["https://new.example"],
        rounds=1,
        max_sources=1,
        max_chars_per_source=2_000,
        base=tmp_path,
        slug=slug,
    )
    assert result["total_sources_read"] == research._MAX_SEEN_URLS
    assert len(state["tensions"]) == 20
    assert "newly discovered" in state["tensions"][-1]
