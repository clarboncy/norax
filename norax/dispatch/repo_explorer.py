"""Repo exploration tool backed by Microsoft FastContext.

The helper performs one bounded deterministic source scan. When a FastContext
model is available it ranks that observed evidence; otherwise the same evidence
is returned directly. Every model citation is restricted to a source line that
was actually scanned and included in its prompt.

Contract for callers:
    Input:  natural-language query + root directory
    Output: list of {"path", "line", "snippet", "confidence"} sorted by relevance
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..gateway_client import GatewayClient, GatewayRequest, SpendGuardTripped

log = logging.getLogger("norax.dispatch.repo_explorer")

FASTCONTEXT_MODEL = os.environ.get("NORAX_FASTCONTEXT_MODEL", "FastContext-1.0-4B-SFT:latest")
OLLAMA_BASE = os.environ.get("NORAX_OLLAMA_URL", "http://127.0.0.1:11434")


def _default_budget() -> int:
    try:
        value = int(os.environ.get("NORAX_REPO_EXPLORE_BUDGET", "2500"))
    except (TypeError, ValueError):
        value = 2500
    return max(128, min(value, 4096))


DEFAULT_BUDGET = _default_budget()
MAX_GREP_HITS = 6
MAX_QUERY_CHARS = 2_000
MAX_SCAN_ENTRIES = 50_000
MAX_SOURCE_FILES = 20_000
MAX_SCAN_SECONDS = 2.0
MAX_CONTENT_SCAN_BYTES = 64 * 1024 * 1024
MAX_CONTENT_SCAN_SECONDS = 2.0
MAX_EVIDENCE_CANDIDATES = 256
MAX_PROMPT_CITATIONS = 96
MAX_EVIDENCE_CHARS = 32_000
MIN_MODEL_BUDGET = 128
MAX_MODEL_BUDGET = 4_096
MODEL_CATALOG_TTL = 60.0


@dataclass(frozen=True)
class _FileScan:
    files: list[Path]
    scanned_entries: int
    truncated: bool


@dataclass(frozen=True)
class _SearchEvidence:
    prompt: str
    citations: list[dict]
    allowed_citations: set[tuple[str, int]]
    scanned_files: int
    scanned_bytes: int
    truncated: bool


_model_catalog_cache: tuple[float, str | None] | None = None


async def _fastcontext_model_available() -> str | None:
    """Check whether the local Ollama has the FastContext model.

    Returns the matching model tag so callers can route to it.
    """
    global _model_catalog_cache
    now = time.monotonic()
    if _model_catalog_cache is not None and now - _model_catalog_cache[0] < MODEL_CATALOG_TTL:
        return _model_catalog_cache[1]
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.get(f"{OLLAMA_BASE}/api/tags")
            if r.status_code != 200:
                return None
            data = r.json()
        for m in data.get("models") or []:
            name = str(m.get("name") or m.get("model") or "")
            if "fastcontext" in name.lower():
                _model_catalog_cache = (now, name)
                return name
        _model_catalog_cache = (now, None)
        return None
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError):
        _model_catalog_cache = (now, None)
        return None


def _build_fastcontext_prompt(query: str, root: str, evidence: str) -> str:
    return (
        "You are FastContext, ranking already-observed repository evidence.\n"
        "Select only the supplied file:line observations that are relevant "
        "to the user's query. Never invent or alter a path or line number.\n"
        "Return your final answer as a single JSON object with key 'citations'.\n"
        "Each citation must have: path (relative string), line (int), "
        "snippet (string), confidence (0.0-1.0).\n\n"
        f"Repository root: {root}\n"
        f"User query: {query}\n\n"
        "Observations so far:\n"
        f"{evidence}\n\n"
        "Return ONLY valid JSON."
    )


async def _call_fastcontext(
    query: str,
    root: Path,
    budget: int,
    *,
    evidence: str,
    allowed_citations: set[tuple[str, int]],
    model_tag: str | None = None,
) -> list[dict]:
    """Run one accounted FastContext generation and verify every citation."""
    model = model_tag or FASTCONTEXT_MODEL
    gateway = GatewayClient(
        base_url=f"{OLLAMA_BASE.rstrip('/')}/v1",
        timeout=180.0,
        provider_kind="ollama",
    )
    request = GatewayRequest(
        model=model,
        messages=[
            {
                "role": "user",
                "content": _build_fastcontext_prompt(query, str(root), evidence),
            }
        ],
        temperature=0.2,
        max_tokens=budget,
        metadata={
            "repo_explore": True,
            "ollama_grammar": "json",
            "ollama_use_grammar": True,
        },
    )
    try:
        response = await gateway.chat(request)
    finally:
        await gateway.aclose()
    content = response.content
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return await asyncio.to_thread(
                _validated_citations,
                parsed.get("citations", []),
                root,
                allowed_citations,
            )
        return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _actual_line(path: Path, line_number: int) -> str | None:
    """Return one real source line without trusting model-supplied snippets."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle, start=1):
                if index == line_number:
                    return line.strip()[:200]
    except (OSError, UnicodeError):
        return None
    return None


def _validated_citations(
    raw: object,
    root: Path,
    allowed_citations: set[tuple[str, int]],
) -> list[dict]:
    """Bind model citations to observed evidence and replace snippets with source."""
    if not isinstance(raw, list):
        return []
    citations: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for item in raw[:32]:
        if not isinstance(item, dict):
            continue
        relative = str(item.get("path") or "").replace("\\", "/").lstrip("./")
        if not relative:
            continue
        raw_line = item.get("line")
        if isinstance(raw_line, bool) or not isinstance(raw_line, (int, str)):
            continue
        try:
            line_number = int(raw_line)
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError, OverflowError):
            continue
        if line_number < 1 or not math.isfinite(confidence):
            continue
        key = (relative, line_number)
        if key not in allowed_citations or key in seen:
            continue
        source_path = root / relative
        if source_path.is_symlink():
            continue
        actual = _actual_line(source_path, line_number)
        if actual is None:
            continue
        seen.add(key)
        citations.append(
            {
                "path": relative,
                "line": line_number,
                "snippet": actual,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
        if len(citations) >= 12:
            break
    citations.sort(key=lambda item: -item["confidence"])
    return citations


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "yet",
        "so",
        "if",
        "because",
        "although",
        "though",
        "where",
        "when",
        "why",
        "how",
        "what",
        "who",
        "which",
        "find",
        "locate",
        "located",
        "show",
        "tell",
        "defined",
        "definition",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "than",
    }
)


def _tokenize_query(query: str) -> list[str]:
    """Extract useful search tokens from a natural-language query."""
    q = re.sub(r"[^\w\s.]", " ", query)
    words = [w.lower() for w in q.split() if len(w) > 2 and w.lower() not in _STOPWORDS]
    # Boost exact identifiers (camelCase, snake_case, dot-paths)
    identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*", query)
    id_tokens = [
        identifier.lower()
        for identifier in identifiers
        if len(identifier) > 2
        and identifier.lower() not in _STOPWORDS
        and (
            "_" in identifier
            or "." in identifier
            or any(char.isupper() for char in identifier[1:])
            or any(char.isdigit() for char in identifier)
        )
    ]
    # Identifiers first, then regular words
    tokens = list(dict.fromkeys(id_tokens + words))
    return tokens[:8]


def _find_files(root: Path) -> _FileScan:
    """List source files with symlink, entry-count, and wall-clock bounds."""
    ignore = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        "target",
    }
    generated_or_vendored = {
        "vendor",
        "third_party",
        "site",
        "htmlcov",
        "coverage",
    }
    source_suffixes = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".java",
        ".kt",
        ".swift",
        ".rb",
        ".php",
        ".cs",
        ".sh",
        ".bash",
        ".zsh",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".md",
        ".txt",
        ".html",
        ".css",
        ".scss",
        ".sql",
    }
    started = time.monotonic()
    scanned_entries = 0
    truncated = False
    out: list[Path] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in ignore
            and name.lower() not in generated_or_vendored
            and not name.lower().startswith(("backup", "restore"))
            and not (Path(current) / name).is_symlink()
        )
        for name in sorted(filenames):
            scanned_entries += 1
            if (
                scanned_entries > MAX_SCAN_ENTRIES
                or len(out) >= MAX_SOURCE_FILES
                or time.monotonic() - started > MAX_SCAN_SECONDS
            ):
                truncated = True
                break
            path = Path(current) / name
            if path.suffix.lower() not in source_suffixes or path.is_symlink():
                continue
            try:
                if not path.is_file() or path.stat().st_size >= 500_000:
                    continue
            except OSError:
                continue
            out.append(path)
        if truncated:
            break
    return _FileScan(files=sorted(out), scanned_entries=scanned_entries, truncated=truncated)


def _score_file(p: Path, tokens: list[str]) -> float:
    """Score a file by name/path match to query tokens."""
    name = p.name.lower()
    parts = [part.lower() for part in p.parts]
    score = 0.0
    for t in tokens:
        if t in name:
            score += 0.75
        for part in parts:
            if t in part:
                score += 0.15
    return score


def _grep_text(text: str, tokens: list[str]) -> list[tuple[int, str, float]]:
    """Return scored source lines matching any query token."""
    hits: list[tuple[int, str, float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        matched_tokens = [token for token in tokens if token in lower]
        if not matched_tokens:
            continue
        score = float(len(matched_tokens))
        definition = re.search(
            r"\b(?:def|class|function|const|let|var|struct|enum|type)\s+"
            r"([a-zA-Z_][a-zA-Z0-9_]*)",
            line,
            re.IGNORECASE,
        )
        definition_name = (
            re.sub(r"[^a-z0-9]", "", definition.group(1).lower()) if definition else ""
        )
        for token in matched_tokens:
            normalized_token = re.sub(r"[^a-z0-9]", "", token)
            if normalized_token and normalized_token in definition_name:
                score += 2.0
        hits.append((line_number, line.strip(), score))
    return hits


def _search_evidence(query: str, root: Path, files: list[Path]) -> _SearchEvidence:
    """Run one bounded content scan reused by model and deterministic paths."""
    tokens = _tokenize_query(query)
    if not tokens:
        return _SearchEvidence(
            prompt="No usable search tokens.",
            citations=[],
            allowed_citations=set(),
            scanned_files=0,
            scanned_bytes=0,
            truncated=False,
        )

    ranked_files = sorted(
        ((path, _score_file(path, tokens)) for path in files),
        key=lambda item: (-item[1], str(item[0])),
    )
    started = time.monotonic()
    scanned_files = 0
    scanned_bytes = 0
    truncated = False
    candidates: list[tuple[float, str, int, str]] = []

    for path, path_score in ranked_files:
        if time.monotonic() - started > MAX_CONTENT_SCAN_SECONDS:
            truncated = True
            break
        try:
            size = path.stat().st_size
            if scanned_bytes + size > MAX_CONTENT_SCAN_BYTES:
                truncated = True
                break
            text = path.read_text(encoding="utf-8", errors="replace")
            relative = str(path.relative_to(root)).replace("\\", "/")
        except (OSError, UnicodeError, ValueError):
            continue
        scanned_files += 1
        scanned_bytes += size

        hits = _grep_text(text, tokens)
        hits.sort(key=lambda hit: (-hit[2], hit[0]))
        if not hits and path_score > 0:
            first_source_line = next(
                (
                    (index, line.strip())
                    for index, line in enumerate(text.splitlines(), 1)
                    if line.strip()
                ),
                None,
            )
            if first_source_line is not None:
                hits = [(first_source_line[0], first_source_line[1], 0.0)]

        for line_number, snippet, line_score in hits[:MAX_GREP_HITS]:
            combined_score = path_score + line_score
            candidates.append((combined_score, relative, line_number, snippet[:200]))
            # Bound temporary ranking memory without prematurely ending the
            # content scan; later files can still displace weaker candidates.
            if len(candidates) > MAX_EVIDENCE_CANDIDATES * 2:
                candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
                del candidates[MAX_EVIDENCE_CANDIDATES:]

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    candidates = candidates[:MAX_EVIDENCE_CANDIDATES]
    citations = [
        {
            "path": relative,
            "line": line_number,
            "snippet": snippet,
            "confidence": min(1.0, 0.25 + combined_score / 12.0),
        }
        for combined_score, relative, line_number, snippet in candidates[:12]
    ]

    prompt_lines = [
        f"Tokens: {tokens}",
        f"Content files scanned: {scanned_files}",
        f"Content bytes scanned: {scanned_bytes}",
    ]
    prompt_chars = sum(len(line) + 1 for line in prompt_lines)
    allowed_citations: set[tuple[str, int]] = set()
    for combined_score, relative, line_number, snippet in candidates[:MAX_PROMPT_CITATIONS]:
        line = f"{relative}:{line_number}: {snippet} (rank={combined_score:.1f})"
        if prompt_chars + len(line) + 1 > MAX_EVIDENCE_CHARS:
            break
        prompt_lines.append(line)
        prompt_chars += len(line) + 1
        allowed_citations.add((relative, line_number))

    return _SearchEvidence(
        prompt="\n".join(prompt_lines),
        citations=citations,
        allowed_citations=allowed_citations,
        scanned_files=scanned_files,
        scanned_bytes=scanned_bytes,
        truncated=truncated,
    )


async def explore_repo(query: str, root: str | None = None, budget: int | None = None) -> dict:
    """Public entry point for the repo_explore tool."""
    query = str(query or "").strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        return {
            "ok": False,
            "error": "query must contain 1-2000 characters",
        }
    try:
        cwd = Path(root or ".").expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return {"ok": False, "error": f"invalid_path: {exc}", "root": str(root or ".")}
    if not cwd.exists():
        return {"ok": False, "error": "path_not_found", "root": str(cwd)}
    if not cwd.is_dir():
        return {"ok": False, "error": "not_a_directory", "root": str(cwd)}

    try:
        parsed_budget = DEFAULT_BUDGET if budget is None else int(budget)
    except (TypeError, ValueError, OverflowError):
        return {"ok": False, "error": "budget must be an integer", "root": str(cwd)}
    if isinstance(budget, bool) or not MIN_MODEL_BUDGET <= parsed_budget <= MAX_MODEL_BUDGET:
        return {
            "ok": False,
            "error": f"budget must be between {MIN_MODEL_BUDGET} and {MAX_MODEL_BUDGET}",
            "root": str(cwd),
        }

    model_tag, scan = await asyncio.gather(
        _fastcontext_model_available(),
        asyncio.to_thread(_find_files, cwd),
    )
    evidence = await asyncio.to_thread(_search_evidence, query, cwd, scan.files)
    scan_metadata = {
        "scan_truncated": scan.truncated or evidence.truncated,
        "file_scan_truncated": scan.truncated,
        "content_scan_truncated": evidence.truncated,
        "scanned_entries": scan.scanned_entries,
        "content_scanned_files": evidence.scanned_files,
        "content_scanned_bytes": evidence.scanned_bytes,
    }
    fallback_reason = "FastContext model not available"
    if model_tag and evidence.allowed_citations:
        try:
            citations = await _call_fastcontext(
                query,
                cwd,
                parsed_budget,
                evidence=evidence.prompt,
                allowed_citations=evidence.allowed_citations,
                model_tag=model_tag,
            )
            if citations:
                return {
                    "ok": True,
                    "provider": "fastcontext",
                    "model": model_tag,
                    "root": str(cwd),
                    "citations": citations,
                    "budget": parsed_budget,
                    **scan_metadata,
                }
            fallback_reason = "FastContext returned no source-verified citations"
        except (asyncio.CancelledError, SpendGuardTripped):
            raise
        except Exception as e:  # noqa: BLE001 - deterministic fallback is the tool contract
            log.warning("fastcontext.explore.error: %r", e)
            fallback_reason = f"FastContext unavailable: {type(e).__name__}"
    elif model_tag:
        fallback_reason = "bounded source scan found no evidence for FastContext to rank"

    return {
        "ok": True,
        "provider": "local_fallback",
        "root": str(cwd),
        "citations": evidence.citations,
        "budget": parsed_budget,
        **scan_metadata,
        "note": f"{fallback_reason}; used one bounded deterministic source scan.",
    }
