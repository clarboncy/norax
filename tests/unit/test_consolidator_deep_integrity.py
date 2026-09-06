from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from norax.memory import consolidator as module


def test_safe_compile_hashing_and_tool_dump_edges(caplog):
    valid = module._safe_compile(r"^fact$", name="valid")
    invalid = module._safe_compile("[", name="broken")
    assert valid.match("fact")
    assert not invalid.search("anything")
    assert "broken" in caplog.text
    assert module._content_hash("  SAME\n fact ") == module._content_hash("same fact")
    assert not module.is_tool_dump("plain durable fact")
    assert module.is_tool_dump('{"stdout":"truncated"')


def test_strip_tool_dumps_missing_file_returns_zero(tmp_path):
    assert module.strip_tool_dumps(tmp_path / "missing.md") == 0


def test_dedup_cache_commit_is_atomic_at_limit(monkeypatch):
    monkeypatch.setattr(module, "_MAX_DEDUP_ENTRIES", 2)
    cache = module._DedupCache()
    cache.commit(["fact one", "fact one", "fact two"])
    assert cache.contains(" FACT one ")
    assert cache.contains_hash(module._content_hash("fact two"))
    before = cache._seen.copy()
    with pytest.raises(ValueError, match="entry limit"):
        cache.commit(["fact three"])
    assert cache._seen == before


def test_dedup_seed_handles_absent_headers_and_both_filename_families(tmp_path):
    cache = module._DedupCache()
    cache.seed_from_dir(tmp_path / "absent")
    (tmp_path / "nested").mkdir()
    (tmp_path / "consolidated-a.md").write_text("# heading\nFACT:first durable value\n")
    (tmp_path / "nested" / "sleep-flush-a.md").write_text("\nPROCEDURE:verify the stored index\n")
    cache.seed_from_dir(tmp_path)
    assert cache.contains("FACT:first durable value")
    assert cache.contains("PROCEDURE:verify the stored index")
    assert not cache.contains("# heading")


def test_dedup_seed_rejects_too_many_files_links_and_entries(tmp_path, monkeypatch):
    one = tmp_path / "consolidated-one.md"
    two = tmp_path / "consolidated-two.md"
    one.write_text("FACT:one durable value\n")
    two.write_text("FACT:two durable value\n")
    monkeypatch.setattr(module, "_MAX_CANONICAL_FILES", 1)
    with pytest.raises(ValueError, match="file limit"):
        module._DedupCache().seed_from_dir(tmp_path)

    monkeypatch.setattr(module, "_MAX_CANONICAL_FILES", 10)
    two.unlink()
    linked = tmp_path / "consolidated-link.md"
    linked.symlink_to(one)
    with pytest.raises(ValueError, match="regular file"):
        module._DedupCache().seed_from_dir(tmp_path)

    linked.unlink()
    one.write_text("FACT:one durable value\nFACT:two durable value\n")
    monkeypatch.setattr(module, "_MAX_DEDUP_ENTRIES", 1)
    with pytest.raises(ValueError, match="entry limit"):
        module._DedupCache().seed_from_dir(tmp_path)


def test_extract_from_text_classifies_signal_and_rejects_noise():
    long_fact = "LONG_FACT:" + "x" * 600
    text = "\n".join(
        [
            "",
            "# heading",
            "---",
            "ok",
            "tiny",
            "NORAX_TOKEN=should_not_survive",
            "total 123",
            '"provider_kind": "noise"',
            "Permission denied while reading data",
            '{"ok": true, "stdout": "raw tool output"}',
            "PORTS:gateway=8899 and runtime=4101",
            "Reach @operator.example for the durable owner identity",
            "PROCEDURE:VERIFY each deployment before promotion",
            "The production connector was verified successfully",
            "Documentation lives at https://example.test/reference",
            "This ordinary sentence has no durable classifier signal",
            long_fact,
        ]
    )
    semantic, procedural = module._extract_from_text(text)
    assert "PORTS:gateway=8899 and runtime=4101" in semantic
    assert any(line.startswith("Reach @operator") for line in semantic)
    assert any(line.startswith("REF:") for line in semantic)
    assert len(next(line for line in semantic if line.startswith("LONG_FACT:"))) == 500
    assert "PROCEDURE:VERIFY each deployment before promotion" in procedural
    assert any(line.startswith("VERIFY:The production") for line in procedural)
    joined = "\n".join(semantic + procedural)
    assert "should_not_survive" not in joined
    assert "raw tool output" not in joined


def test_secret_stripping_masks_values_and_preserves_nonsecrets():
    value = "TOKEN=abcdefgh PASSWORD=12345678 harmless=value"
    assert module._strip_secrets(value) == "TOKEN=*** PASSWORD=*** harmless=value"


def test_safe_parse_result_handles_complete_truncated_and_invalid_values():
    complete = {"ok": True, "stdout": "FACT:complete payload"}
    assert module._safe_parse_result(json.dumps(complete)) == complete
    assert module._safe_parse_result(None) == {}  # type: ignore[arg-type]
    assert module._safe_parse_result("[]") == {}

    truncated = (
        '{"ok":true,"exitCode":-2,"totalMatches":3.5,'
        '"totalFiles":null,"referenceCount":false,"path":"/tmp/x",'
        '"stdout":"a\\nb","content":"durable partial content   , } ]'
    )
    parsed = module._safe_parse_result(truncated)
    assert parsed["ok"] is True
    assert parsed["exitCode"] == -2
    assert parsed["totalFiles"] is None
    assert parsed["referenceCount"] is False
    assert "totalMatches" not in parsed
    assert parsed["path"] == "/tmp/x"
    assert parsed["stdout"] == r"a\nb"
    assert parsed["content"] == "durable partial content"

    false_and_source = '{"ok":false,"source":"local"'
    assert module._safe_parse_result(false_and_source) == {"ok": False, "source": "local"}
    assert module._safe_parse_result("not json") == {}


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("{raw json fragment that is long enough", False),
        ("short", False),
        ("OPENAI_TOKEN=abcdefgh", False),
        ("thanks", False),
        ("Permission denied while opening the durable path", False),
        ("total 42 files in this directory listing", False),
        ('"content": this is a long serialized fragment', False),
        ("RUNTIME:connector reached the expected endpoint", True),
    ],
)
def test_stdout_line_filtering(line, expected):
    assert module._stdout_line_ok(line) is expected


def test_extract_jsonl_covers_tool_user_and_malformed_records(tmp_path):
    path = tmp_path / "spill-all.jsonl"
    records = [
        "",
        "not-json",
        "[]",
        json.dumps({"kind": 7, "content": "FACT:wrong kind type"}),
        json.dumps({"kind": "user", "content": 7}),
        json.dumps({"kind": "user", "content": ""}),
        json.dumps({"kind": "user", "content": "ok"}),
        json.dumps({"kind": "tool_call", "content": "{}", "meta": {"name": "read"}}),
        json.dumps({"kind": "tool_call", "content": "{}", "meta": {"name": 4}}),
        json.dumps({"kind": "tool_call", "content": "{}", "meta": "bad"}),
        json.dumps({"kind": "tool_result", "content": "{}"}),
        json.dumps({"kind": "tool_result", "content": "not-json"}),
        json.dumps(
            {
                "kind": "tool_result",
                "content": json.dumps(
                    {
                        "stdout": (
                            "TOKEN=abcdefgh\n"
                            "RUNTIME:service is active on the expected endpoint\n"
                            "short"
                        ),
                        "path": "/srv/norax/config.json",
                        "content": (
                            "PORTS:gateway=8899 and runtime=4101\n"
                            "PROCEDURE:verify the connector response"
                        ),
                        "totalMatches": 3,
                        "totalFiles": 2,
                        "referenceCount": 1,
                        "ok": True,
                        "exitCode": 0,
                    }
                ),
            }
        ),
        json.dumps({"kind": "tool_result", "content": json.dumps({"path": "{invalid"})}),
        json.dumps({"role": "assistant", "content": "DEPLOYED:connector active and verified"}),
        json.dumps({"kind": "event", "content": "FACT:not a supported conversational kind"}),
    ]
    path.write_text("\n".join(records))

    semantic, procedural = module._extract_from_jsonl(path)
    joined = "\n".join(semantic + procedural)
    assert "AGENT_USED_TOOL:read" in procedural
    assert "RUNTIME_FACT:RUNTIME:service is active" in joined
    assert "READ_FILE:/srv/norax/config.json" in procedural
    assert "PORTS:gateway=8899" in joined
    assert "RESULT:read.totalMatches=3" in procedural
    assert "RESULT:read.totalFiles=2" in procedural
    assert "RESULT:read.referenceCount=1" in procedural
    assert "RESULT:read.ok=True" in procedural
    assert "RESULT:read.exitCode=0" in procedural
    assert "DEPLOYED:connector active and verified" in semantic
    assert "abcdefgh" not in joined
    assert "supported conversational kind" not in joined


def test_extract_spill_markdown_covers_tool_association_and_fallbacks(tmp_path):
    path = tmp_path / "spill-all.md"
    lines = [
        "",
        "SPILL:v1",
        "TOOL_CALL#1",
        'TOOL_RESULT#1:{"ok":true}',
        'TOOL_CALL#2:{"command":"pwd"}',
        (
            'TOOL_RESULT#2:{"stdout":"RUNTIME:working directory verified and active\\n'
            'RUNTIME:connector response verified and active\\nshort",'
            '"path":"/srv/norax","content":"PORTS:gateway=8899 is active and verified",'
            '"totalMatches":1,"totalFiles":2,"ok":true,"exitCode":0}'
        ),
        'TOOL_CALL#3:{"name":"search_memory"}',
        'TOOL_RESULT#3:{"name":"explicit","referenceCount":9}',
        'TOOL_CALL#4:{"name":""}',
        'TOOL_RESULT#4:{"source":"remote","ok":false}',
        "TOOL_RESULT#5:not-json",
        'TOOL_RESULT#6:{"path":"{invalid","content":"tiny"}',
        "ordinary text is ignored",
    ]
    path.write_text("\n".join(lines))
    semantic, procedural = module._extract_from_spill_md(path)
    joined = "\n".join(semantic + procedural)
    assert "AGENT_USED_TOOL:unknown" in procedural
    assert "AGENT_USED_TOOL:exec" in procedural
    assert "AGENT_USED_TOOL:explicit" in procedural
    assert "AGENT_USED_TOOL:remote" in procedural
    assert "RUNTIME_FACT:RUNTIME:working directory" in joined
    assert "READ_FILE:/srv/norax" in procedural
    assert "PORTS:gateway=8899" in joined
    assert "RESULT:exec.totalMatches=1" in procedural
    assert "RESULT:exec.totalFiles=2" in procedural
    assert "RESULT:exec.ok=True" in procedural
    assert "RESULT:exec.exitCode=0" in procedural


@pytest.mark.parametrize(
    "invalid",
    [True, "300", float("nan"), float("inf"), -0.1, 365 * 86_400 + 1],
)
def test_consolidator_rejects_invalid_minimum_age(tmp_path, invalid):
    with pytest.raises(ValueError, match="min_age_sec"):
        module.MemoryConsolidator(tmp_path, min_age_sec=invalid)


def _backdate(path: Path) -> None:
    old = time.time() - 600
    os.utime(path, (old, old), follow_symlinks=False)


def test_eligible_files_pairs_formats_age_links_and_limit(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    consolidator = module.MemoryConsolidator(root, min_age_sec=300)
    assert consolidator._eligible_files() == []
    sleep = root / "sleep"
    sleep.mkdir(parents=True)

    jsonl = sleep / "spill-pair.jsonl"
    paired = sleep / "spill-pair.md"
    standalone = sleep / "spill-standalone.md"
    young_spill = sleep / "spill-young.md"
    buffer_file = sleep / "buffer-state.md"
    young = sleep / "buffer-young.md"
    for path in (jsonl, paired, standalone, young_spill, buffer_file, young):
        path.write_text("FACT:this input has durable information\n")
    for path in (jsonl, paired, standalone, buffer_file):
        _backdate(path)
    outside = tmp_path / "outside.md"
    outside.write_text("FACT:outside data must not enter memory\n")
    (sleep / "buffer-linked.md").symlink_to(outside)
    (sleep / "spill-linked.jsonl").symlink_to(outside)

    eligible = consolidator._eligible_files()
    assert eligible == [jsonl, standalone, buffer_file]

    monkeypatch.setattr(module, "_MAX_CANONICAL_FILES", 2)
    with pytest.raises(ValueError, match="too many sleep inputs"):
        consolidator._eligible_files()


def test_archiver_handles_optional_companion_and_unsafe_companion(tmp_path):
    root = tmp_path / "memory"
    consolidator = module.MemoryConsolidator(root, min_age_sec=0)
    consolidator._archive_processed([], module.ConsolidateResult())
    sleep = root / "sleep"
    sleep.mkdir(parents=True)

    source = sleep / "spill-pair.jsonl"
    companion = sleep / "spill-pair.md"
    source.write_text("{}\n")
    companion.write_text("SPILL:v1\n")
    result = module.ConsolidateResult()
    consolidator._archive_processed([source], result)
    assert result.archived == [str(source)]
    assert not source.exists() and not companion.exists()

    source = sleep / "spill-unsafe.jsonl"
    source.write_text("{}\n")
    outside = tmp_path / "outside.md"
    outside.write_text("do not move\n")
    (sleep / "spill-unsafe.md").symlink_to(outside)
    result = module.ConsolidateResult()
    consolidator._archive_processed([sleep / "missing.md", source], result)
    assert result.archived == [str(source)]
    assert outside.exists()


def test_consolidation_appends_once_dedups_across_inputs_and_archives(tmp_path):
    root = tmp_path / "memory"
    sleep = root / "sleep"
    sleep.mkdir(parents=True)
    first = sleep / "buffer-first.md"
    second = sleep / "buffer-second.md"
    spill = sleep / "spill-third.md"
    duplicate = "FACT:the verified connector uses the durable local endpoint"
    first.write_text(duplicate + "\nPROCEDURE:verify every connector before release\n")
    second.write_text(duplicate + "\n")
    spill.write_text("SPILL:v1\n")
    consolidator = module.MemoryConsolidator(root, min_age_sec=0)
    result = consolidator.consolidate()
    assert (result.files_processed, result.facts_written, result.procedural_written) == (3, 1, 1)
    assert result.skipped_dup == 1
    assert len(result.archived) == 3

    third = sleep / "buffer-third.md"
    third.write_text("FACT:a second verified connector fact is durable\n")
    second_result = consolidator.consolidate()
    assert second_result.facts_written == 1
    semantic = next((root / "semantic").glob("consolidated-*.md")).read_text()
    assert semantic.count("# consolidated from sleep") == 1


def test_consolidate_and_reindex_runs_only_for_committed_signal(tmp_path, monkeypatch, caplog):
    from norax.memory import build_index

    root = tmp_path / "memory"
    sleep = root / "sleep"
    sleep.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        build_index,
        "_build_memory_index_unlocked",
        lambda memory_root: calls.append(memory_root),
    )
    consolidator = module.MemoryConsolidator(root, min_age_sec=0)

    empty = sleep / "buffer-empty.md"
    empty.write_text("# no signal\n")
    consolidator.consolidate_and_reindex()
    assert calls == []

    preview = sleep / "buffer-preview.md"
    preview.write_text("FACT:preview must not trigger a durable reindex\n")
    consolidator.consolidate_and_reindex(dry_run=True)
    assert calls == [] and preview.exists()

    committed = consolidator.consolidate_and_reindex()
    assert committed.facts_written == 1 and calls == [root]

    failing = sleep / "buffer-failing.md"
    failing.write_text("FACT:reindex failure does not lose consolidation output\n")

    def fail(_memory_root):
        raise RuntimeError("index offline")

    monkeypatch.setattr(build_index, "_build_memory_index_unlocked", fail)
    result = consolidator.consolidate_and_reindex()
    assert result.facts_written + result.procedural_written == 1
    assert "index rebuild failed" in caplog.text
