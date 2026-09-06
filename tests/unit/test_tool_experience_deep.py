from __future__ import annotations

import json
import stat
from types import SimpleNamespace

import pytest

from norax.memory import tool_experience as module


def _lesson(
    lesson_id="lesson",
    *,
    tools=("read",),
    triggers=("inspect",),
    procedure=("Call read",),
    avoid=("Do not guess",),
    verification="Verify the result",
    phases=("pre", "recovery"),
    confidence=0.8,
    evidence_count=1,
):
    return module.ToolLesson(
        lesson_id=lesson_id,
        intent="Inspect and verify a file",
        tools=tools,
        triggers=triggers,
        procedure=procedure,
        avoid=avoid,
        verification=verification,
        phases=phases,
        confidence=confidence,
        evidence_count=evidence_count,
        learned=True,
    )


def test_token_error_and_argument_helpers_cover_safe_shapes():
    assert module._tokens("Read FILE.py!") == {"read", "file.py"}
    assert module._tokens(["Run", "a command"]) == {"run", "command"}
    expanded = module._expand_tokens({"modify", "unmapped"})
    assert {"modify", "edit", "patch", "unmapped"} <= expanded

    assert module._safe_error({"ok": True}) == ""
    assert module._safe_error(None) == "tool_error"
    assert module._safe_error({}) == "tool_error"
    assert module._safe_error({"error": {"code": "NOT FOUND!"}}) == "not_found"
    assert module._safe_error({"error": {"type": "Timeout"}}) == "timeout"
    assert module._safe_error({"error": {}}) == "tool_error"
    assert module._safe_error({"stderr": "!"}) == "tool_error"
    assert len(module._safe_error({"error": "x" * 100})) == 80

    assert module._arg_shape(None) == []
    shaped = module._arg_shape({f"key-{index:02}": "secret" for index in range(30)})
    assert len(shaped) == 24
    assert all("secret" not in item for item in shaped)


def test_bounded_value_helpers_fail_closed_on_malformed_seed_data():
    assert module._bounded_strings("not-a-list", max_items=2, max_chars=3) == ()
    assert module._bounded_strings(["  abcdef ", "", "two"], max_items=2, max_chars=3) == (
        "abc",
        "two",
    )
    for value in (True, "invalid", None, float("nan"), float("inf")):
        assert module._confidence(value, default=0.4) == 0.4
    assert module._confidence(-1) == 0.0
    assert module._confidence(2) == 1.0
    assert module._confidence(0.3) == pytest.approx(0.3)

    for value in (True, "invalid", None, float("inf")):
        assert module._evidence_count(value) == 1
    assert module._evidence_count(-5) == 1
    assert module._evidence_count(module._MAX_EVIDENCE_COUNT + 1) == module._MAX_EVIDENCE_COUNT
    assert module._evidence_count("7") == 7


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"id": "id", "intent": "intent", "procedure": []},
        {"id": "id", "intent": "intent", "procedure": "one step"},
    ],
)
def test_seed_lesson_rejects_incomplete_or_wrongly_shaped_rows(row):
    assert module.ToolLesson.from_seed(row) is None


def test_seed_lesson_bounds_all_prompt_influencing_fields():
    row = {
        "id": "i" * 120,
        "intent": "n" * 400,
        "tools": ["t" * 80] * 30,
        "triggers": ["g" * 100] * 30,
        "procedure": ["p" * 300] * 10,
        "avoid": ["a" * 200] * 8,
        "verification": "v" * 400,
        "phases": ["recovery", "pre", "other", "ignored"],
        "confidence": "bad",
        "evidence_count": "bad",
    }
    lesson = module.ToolLesson.from_seed(row)
    assert lesson is not None
    assert len(lesson.lesson_id) == 96 and len(lesson.intent) == 320
    assert len(lesson.tools) == 24 and len(lesson.tools[0]) == 64
    assert len(lesson.triggers) == 24 and len(lesson.triggers[0]) == 80
    assert len(lesson.procedure) == 8 and len(lesson.procedure[0]) == 240
    assert len(lesson.avoid) == 6 and len(lesson.avoid[0]) == 180
    assert len(lesson.verification) == 280 and len(lesson.phases) == 3
    assert lesson.confidence == 1.0 and lesson.evidence_count == 1


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_jsonl_reader_rejects_invalid_tail_limits(tmp_path, invalid):
    with pytest.raises(ValueError, match="tail_bytes"):
        module.ToolExperienceMemory._read_jsonl(tmp_path / "missing", tail_bytes=invalid)


def test_jsonl_reader_handles_missing_bad_rows_links_directories_and_tail(tmp_path, monkeypatch):
    reader = module.ToolExperienceMemory._read_jsonl
    assert reader(tmp_path / "missing") == []

    path = tmp_path / "events.jsonl"
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    path.write_text("bad\n[]\n" + "".join(json.dumps(row) + "\n" for row in rows))
    assert reader(path) == rows
    last_line_size = len(json.dumps(rows[-1]) + "\n")
    assert reader(path, tail_bytes=last_line_size) == [rows[-1]]
    assert reader(path, tail_bytes=last_line_size + 3) == [rows[-1]]

    linked = tmp_path / "linked.jsonl"
    linked.symlink_to(path)
    assert reader(linked) == []
    directory = tmp_path / "directory"
    directory.mkdir()
    assert reader(directory) == []

    monkeypatch.setattr(module, "_MAX_SEED_BYTES", 2)
    assert reader(path) == []


def test_jsonl_reader_detects_growth_and_closes_descriptor_on_open_failure(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok":true}\n')
    real_fstat = module.os.fstat

    def stale_size(fd):
        current = real_fstat(fd)
        return SimpleNamespace(st_mode=current.st_mode, st_size=0)

    monkeypatch.setattr(module.os, "fstat", stale_size)
    monkeypatch.setattr(module, "_MAX_SEED_BYTES", 2)
    assert module.ToolExperienceMemory._read_jsonl(path) == []

    monkeypatch.setattr(module.os, "fstat", real_fstat)
    real_fdopen = module.os.fdopen

    def fail_fdopen(*_args, **_kwargs):
        raise OSError("fdopen failed")

    monkeypatch.setattr(module.os, "fdopen", fail_fdopen)
    assert module.ToolExperienceMemory._read_jsonl(path) == []
    monkeypatch.setattr(module.os, "fdopen", real_fdopen)


def test_seed_loading_prefers_runtime_override_reloads_and_caches(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    runtime_seed = root / "procedural" / "tool-use-wins.jsonl"
    runtime_seed.parent.mkdir(parents=True)
    runtime_seed.write_text(
        json.dumps({"id": "local", "intent": "Local lesson", "procedure": ["Verify local"]}) + "\n"
    )
    memory = module.ToolExperienceMemory(root)
    assert memory.seed_path == runtime_seed
    first = memory._load_seed()
    assert any(lesson.lesson_id == "local" for lesson in first)

    monkeypatch.setattr(memory, "_read_jsonl", lambda _path: pytest.fail("cache missed"))
    assert memory._load_seed() is first

    missing = module.ToolExperienceMemory(tmp_path / "blank", seed_path=tmp_path / "missing")
    assert len(missing._load_seed()) == len(module._BUILTIN)


def test_event_conversion_rejects_losses_and_malformed_sequences():
    assert module.ToolExperienceMemory._event_to_lesson({}) is None
    assert module.ToolExperienceMemory._event_to_lesson({"verified": True}) is None
    assert (
        module.ToolExperienceMemory._event_to_lesson(
            {"verified": True, "sequence": "read", "signature": "bad"}
        )
        is None
    )
    lesson = module.ToolExperienceMemory._event_to_lesson(
        {
            "verified": True,
            "sequence": ["read", "read", "edit"],
            "signature": "abcdefghijklmnop-extra",
            "task_terms": ["fix", "file"],
            "failures": ["bad", {"tool": "edit", "error": "not found"}],
            "evidence_count": 4,
        }
    )
    assert lesson is not None
    assert lesson.tools == ("read", "edit")
    assert lesson.procedure == ("Call read", "Call read", "Call edit")
    assert lesson.lesson_id.endswith("abcdefghijklmnop")
    assert lesson.evidence_count == 4
    assert len(lesson.avoid) == 1


def test_avoid_conversion_handles_absent_malformed_and_recovery_tools():
    convert = module.ToolExperienceMemory._avoid_to_lesson
    assert convert({}) is None
    assert convert({"failures": "bad"}) is None
    fallback = convert({"failures": ["bad"], "evidence_count": 2})
    assert fallback is not None
    assert fallback.tools == ("tool",)
    assert fallback.procedure == ("Avoid tool when error is error",)
    recovered = convert(
        {
            "failures": [{"tool": "edit", "error": "not_found"}],
            "recover_with": ["read", "search"],
            "task_terms": ["code"],
            "evidence_count": 5,
        }
    )
    assert recovered is not None
    assert recovered.tools == ("read", "search", "edit")
    assert recovered.procedure == ("Retry with read", "Retry with search")
    assert recovered.evidence_count == 5


def test_learned_loader_preserves_counts_groups_events_and_caches(tmp_path):
    memory = module.ToolExperienceMemory(tmp_path / "memory")
    memory.events_path.parent.mkdir(parents=True)
    rows = [
        {},
        {"signature": "loss", "verified": False, "sequence": ["read"]},
        {
            "signature": "avoid",
            "verified": False,
            "sequence": ["edit"],
            "source": "replay_avoid",
            "failures": [{"tool": "edit", "error": "missing"}],
            "evidence_count": 2,
        },
        {
            "signature": "avoid",
            "verified": False,
            "sequence": ["edit"],
            "source": "replay_avoid",
            "failures": [{"tool": "edit", "error": "missing"}],
            "evidence_count": 3,
        },
        {
            "signature": "win",
            "verified": True,
            "sequence": ["read", "edit"],
            "evidence_count": 4,
        },
        {
            "signature": "win",
            "verified": True,
            "sequence": ["read", "edit"],
            "evidence_count": 2,
        },
        {"signature": "invalid", "verified": True, "sequence": "bad"},
        {
            "signature": "empty-avoid",
            "verified": False,
            "sequence": ["edit"],
            "source": "replay_avoid",
            "failures": [],
        },
    ]
    memory.events_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    lessons = memory._load_learned()
    by_id = {lesson.lesson_id: lesson for lesson in lessons}
    assert by_id["verified-trajectory-win"].evidence_count == 6
    assert by_id["replay-avoid-edit-missing"].evidence_count == 5
    assert len(lessons) == 2
    assert memory._load_learned() is lessons

    absent = module.ToolExperienceMemory(tmp_path / "absent")
    assert absent._load_learned() == []


def test_learning_count_accumulation_is_safely_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_MAX_EVIDENCE_COUNT", 5)
    memory = module.ToolExperienceMemory(tmp_path)
    memory.events_path.parent.mkdir(parents=True)
    rows = [
        {"signature": "win", "verified": True, "sequence": ["read"], "evidence_count": 4},
        {"signature": "win", "verified": True, "sequence": ["read"], "evidence_count": 4},
        {
            "signature": "avoid",
            "verified": False,
            "sequence": ["edit"],
            "source": "replay_avoid",
            "failures": [{"tool": "edit", "error": "bad"}],
            "evidence_count": 4,
        },
        {
            "signature": "avoid",
            "verified": False,
            "sequence": ["edit"],
            "source": "replay_avoid",
            "failures": [{"tool": "edit", "error": "bad"}],
            "evidence_count": 4,
        },
    ]
    memory.events_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert {lesson.evidence_count for lesson in memory._load_learned()} == {5}


def test_scoring_respects_phase_synonyms_tools_evidence_and_baseline():
    lesson = _lesson(evidence_count=8)
    assert module.ToolExperienceMemory._score(lesson, {"inspect"}, {"read"}, "other") == -1
    relevant = module.ToolExperienceMemory._score(
        lesson,
        module._expand_tokens({"view", "file"}),
        {"read"},
        "pre",
    )
    baseline = module.ToolExperienceMemory._score(
        _lesson(triggers=(), tools=(), evidence_count=1), set(), set(), "pre"
    )
    assert relevant > baseline > 0


def test_retrieve_ranks_lessons_adds_failed_tool_and_applies_phase_floor(tmp_path, monkeypatch):
    memory = module.ToolExperienceMemory(tmp_path)
    pre = _lesson("pre", phases=("pre",), confidence=1.0)
    recovery = _lesson(
        "recovery",
        tools=("edit",),
        triggers=("timeout",),
        phases=("recovery",),
        confidence=1.0,
    )
    monkeypatch.setattr(memory, "_load_seed", lambda: [pre, recovery])
    monkeypatch.setattr(memory, "_load_learned", lambda: [])
    assert [lesson.lesson_id for lesson in memory.retrieve("inspect file", ["read"])] == ["pre"]
    assert [
        lesson.lesson_id
        for lesson in memory.retrieve(
            "recover",
            [],
            phase="recovery",
            failed_tool="edit",
            error="timeout",
            limit=1,
        )
    ] == ["recovery"]
    assert memory.retrieve("inspect", ["read"], limit=-1) == []


def test_render_handles_empty_full_optional_and_size_limited_lessons(tmp_path, monkeypatch):
    memory = module.ToolExperienceMemory(tmp_path)
    monkeypatch.setattr(memory, "retrieve", lambda *_args, **_kwargs: [])
    assert memory.render("task", []) == ""

    full = _lesson("full")
    plain = _lesson("plain", avoid=(), verification="")
    monkeypatch.setattr(memory, "retrieve", lambda *_args, **_kwargs: [full, plain])
    rendered = memory.render("task", ["read"])
    assert "TOOL_EXPERIENCE_MEMORY" in rendered
    assert "AVOID: Do not guess" in rendered
    assert "VERIFY: Verify the result" in rendered
    assert "plain" in rendered
    assert memory.render("task", [], max_chars=1) == ""


def test_replay_ingestion_handles_empty_malformed_unreadable_and_fallback_fields(
    tmp_path, monkeypatch
):
    memory = module.ToolExperienceMemory(tmp_path / "memory")
    assert memory.ingest_replay_patterns(tmp_path / "absent") == 0
    procedural = tmp_path / "procedural"
    procedural.mkdir()
    assert memory.ingest_replay_patterns(procedural) == 0

    target = tmp_path / "outside.md"
    target.write_text("PATTERN:id=x|tool_seq=read→edit|success_rate=100%\n")
    (procedural / "replay-patterns-9999.md").symlink_to(target)
    (procedural / "replay-avoid-9999.md").symlink_to(target)
    assert memory.ingest_replay_patterns(procedural) == 0
    (procedural / "replay-patterns-9999.md").unlink()
    (procedural / "replay-avoid-9999.md").unlink()

    (procedural / "replay-patterns-2026.md").write_text(
        "heading\n"
        "PATTERN:no-equals\n"
        "PATTERN:tool_seq=read|success_rate=100%\n"
        "PATTERN:tool_seq=→|success_rate=100%\n"
        "PATTERN:tool_seq=read→edit|count=bad|success_rate=bad|context=code\n"
        "PATTERN:tool_seq=read→write|count=2|success_rate=nan%|context=docs\n"
        "PATTERN:id=good|tool_seq=read→exec|count=3|success_rate=90%|context=ops,verify\n"
    )
    (procedural / "replay-avoid-2026.md").write_text(
        "heading\n"
        "AVOID:tool=edit|ignored|count=bad|context=code\n"
        "AVOID:tool=exec|error=timeout|count=2|context=ops|recover_with=read,search\n"
    )
    assert memory.ingest_replay_patterns(procedural) == 5
    events = memory._read_jsonl(memory.events_path)
    assert len(events) == 5
    good = next(row for row in events if row["signature"] == "replay_good")
    assert good["verified"] is True and good["evidence_count"] == 3
    assert memory.events_path.stat().st_mode & 0o077 == 0
    assert memory.ingest_replay_patterns(procedural) == 0

    monkeypatch.setattr(module, "_MAX_EVENT_FILE_BYTES", 1)
    fresh = module.ToolExperienceMemory(tmp_path / "full")
    assert fresh.ingest_replay_patterns(procedural) == 0


def test_record_outcome_is_private_bounded_and_contains_no_argument_values(tmp_path, monkeypatch):
    memory = module.ToolExperienceMemory(tmp_path / "memory")
    assert memory.record_outcome("task", [], verified=True) is False
    trace = [
        "malformed",
        {
            "name": "read",
            "args": {"path": "/secret/value", "token": "abcd"},
            "result": {"ok": True},
        },
        {"name": "edit", "args": None, "result": {"error_type": "Not Found"}},
        {"result": None},
    ]
    assert memory.record_outcome("Fix the file", trace, verified=True) is True  # type: ignore[arg-type]
    rows = memory._read_jsonl(memory.events_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["sequence"] == ["read", "edit", "unknown"]
    assert row["arg_shapes"]["read"] == ["path", "token"]
    assert "/secret/value" not in memory.events_path.read_text()
    assert row["failures"] == [
        {"tool": "edit", "error": "not_found"},
        {"tool": "unknown", "error": "tool_error"},
    ]
    assert stat.S_IMODE(memory.events_path.stat().st_mode) == 0o600

    monkeypatch.setattr(module, "_MAX_EVENT_FILE_BYTES", 1)
    assert memory.record_outcome("another", [{"name": "read"}], verified=False) is False
