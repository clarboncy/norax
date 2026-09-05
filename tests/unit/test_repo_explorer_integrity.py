from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

import norax.dispatch.repo_explorer as repo_explorer
from norax.gateway_client import SpendGuardTripped


def test_query_tokenizer_boosts_real_identifiers_not_stopwords() -> None:
    tokens = repo_explorer._tokenize_query(
        "Where is GatewayClient enforce_spend_guard defined in the source"
    )

    assert tokens[:2] == ["gatewayclient", "enforce_spend_guard"]
    assert "where" not in tokens
    assert "defined" not in tokens


@pytest.mark.asyncio
@respx.mock
async def test_fastcontext_citations_are_bound_to_real_source_lines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "parser.py"
    source.parent.mkdir()
    source.write_text("header = True\ndef parse_token(value):\n    return value\n")

    async def _available() -> str:
        return "fastcontext-1.0-4b-sft:latest"

    monkeypatch.setattr(repo_explorer, "_fastcontext_model_available", _available)
    monkeypatch.setattr(repo_explorer, "OLLAMA_BASE", "http://ollama")
    route = respx.post("http://ollama/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "fastcontext-1.0-4b-sft:latest",
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"citations":['
                        '{"path":"src/parser.py","line":2,"snippet":"invented",'
                        '"confidence":0.9},'
                        '{"path":"src/parser.py","line":1,"snippet":"real but unobserved",'
                        '"confidence":1.0},'
                        '{"path":"../outside.py","line":1,"snippet":"bad",'
                        '"confidence":1.0},'
                        '{"path":"src/parser.py","line":999,"snippet":"bad",'
                        '"confidence":1.0}]}'
                    ),
                },
                "done": True,
                "done_reason": "stop",
            },
        )
    )

    result = await repo_explorer.explore_repo(
        "find parse_token",
        root=str(tmp_path),
        budget=512,
    )

    assert result["provider"] == "fastcontext"
    assert result["citations"] == [
        {
            "path": "src/parser.py",
            "line": 2,
            "snippet": "def parse_token(value):",
            "confidence": 0.9,
        }
    ]
    request = route.calls[0].request
    payload = json.loads(request.content)
    assert payload["format"] == "json"
    assert payload["options"]["num_ctx"] == 16_384
    assert payload["options"]["num_predict"] == 512


def test_repo_scan_skips_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-secret.py"
    outside.write_text("SECRET = True\n")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "safe.py").write_text("SAFE = True\n")
    (root / "linked.py").symlink_to(outside)

    scan = repo_explorer._find_files(root)

    assert [path.name for path in scan.files] == ["safe.py"]
    outside.unlink()


def test_repo_scan_includes_first_party_documentation(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    guide = docs / "architecture.md"
    guide.write_text("The dispatcher enforces exact-call authorization.\n")

    scan = repo_explorer._find_files(tmp_path)

    assert guide in scan.files


@pytest.mark.asyncio
async def test_fallback_searches_contents_when_filename_does_not_match(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "misc.py"
    source.write_text("def enforce_spend_guard():\n    return True\n")

    async def _unavailable() -> None:
        return None

    monkeypatch.setattr(repo_explorer, "_fastcontext_model_available", _unavailable)
    result = await repo_explorer.explore_repo(
        "where is enforce_spend_guard defined",
        root=str(tmp_path),
    )

    assert result["provider"] == "local_fallback"
    assert result["citations"][0]["path"] == "misc.py"
    assert result["citations"][0]["line"] == 1
    assert "enforce_spend_guard" in result["citations"][0]["snippet"]


def test_content_match_outranks_partial_filename_match(tmp_path: Path) -> None:
    (tmp_path / "loop_guard.py").write_text("class LoopGuard:\n    pass\n")
    (tmp_path / "transport.py").write_text("class SpendGuard:\n    pass\n")
    scan = repo_explorer._find_files(tmp_path)

    evidence = repo_explorer._search_evidence("where is the spend guard", tmp_path, scan.files)

    assert evidence.citations[0]["path"] == "transport.py"
    assert evidence.citations[0]["line"] == 1


@pytest.mark.asyncio
async def test_model_is_not_called_without_observed_evidence(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "misc.py").write_text("unrelated = True\n")

    async def _available() -> str:
        return "fastcontext-1.0-4b-sft:latest"

    async def _unexpected_model(*_args, **_kwargs):
        raise AssertionError("model should not rank an empty evidence set")

    monkeypatch.setattr(repo_explorer, "_fastcontext_model_available", _available)
    monkeypatch.setattr(repo_explorer, "_call_fastcontext", _unexpected_model)

    result = await repo_explorer.explore_repo(
        "symbol_that_does_not_exist",
        root=str(tmp_path),
    )

    assert result["provider"] == "local_fallback"
    assert result["citations"] == []
    assert "no evidence" in result["note"]


@pytest.mark.asyncio
async def test_repo_scan_runs_once_when_model_output_is_unusable(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "target.py").write_text("def target():\n    pass\n")
    real_scan = repo_explorer._find_files
    scan_calls = 0
    real_evidence = repo_explorer._search_evidence
    evidence_calls = 0

    def _counted_scan(root: Path):
        nonlocal scan_calls
        scan_calls += 1
        return real_scan(root)

    def _counted_evidence(query: str, root: Path, files: list[Path]):
        nonlocal evidence_calls
        evidence_calls += 1
        return real_evidence(query, root, files)

    async def _available() -> str:
        return "fastcontext-1.0-4b-sft:latest"

    async def _empty_model(*_args, **_kwargs):
        return []

    monkeypatch.setattr(repo_explorer, "_find_files", _counted_scan)
    monkeypatch.setattr(repo_explorer, "_search_evidence", _counted_evidence)
    monkeypatch.setattr(repo_explorer, "_fastcontext_model_available", _available)
    monkeypatch.setattr(repo_explorer, "_call_fastcontext", _empty_model)

    result = await repo_explorer.explore_repo("find target", root=str(tmp_path))

    assert result["provider"] == "local_fallback"
    assert scan_calls == 1
    assert evidence_calls == 1


@pytest.mark.asyncio
async def test_repo_explore_propagates_spend_guard(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "target.py").write_text("target = True\n")

    async def _available() -> str:
        return "fastcontext-1.0-4b-sft:latest"

    async def _blocked(*_args, **_kwargs):
        raise SpendGuardTripped("minute", 1, 1)

    monkeypatch.setattr(repo_explorer, "_fastcontext_model_available", _available)
    monkeypatch.setattr(repo_explorer, "_call_fastcontext", _blocked)

    with pytest.raises(SpendGuardTripped):
        await repo_explorer.explore_repo("find target", root=str(tmp_path))


@pytest.mark.asyncio
async def test_invalid_budget_fails_before_probe(monkeypatch, tmp_path: Path) -> None:
    async def _unexpected_probe() -> str | None:
        raise AssertionError("model catalog should not be queried")

    monkeypatch.setattr(repo_explorer, "_fastcontext_model_available", _unexpected_probe)
    result = await repo_explorer.explore_repo("find target", root=str(tmp_path), budget=50_000)

    assert result["ok"] is False
    assert "budget" in result["error"]
