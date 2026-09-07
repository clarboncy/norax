"""Motor tools registry — the 12 tools Norax can invoke.

Each tool is a Python async function with a declared schema. The
dispatcher resolves tool-name → callable, applies RiskGate, runs the
call, and emits a `tool_call` event.

12 tools (Phase 4 scope):
  T0: read, list_dir, web_fetch, search_memory, status
  T1: write, edit, append_memory, message_send, schedule_reminder
  T2: exec, gateway_config_patch

(Browser, sandbox, and remote tools are fully implemented.)
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
import shlex
import signal
import stat as stat_mod
import sys
import threading
import time
from collections import OrderedDict as _OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from ..atomic import atomic_write_text, path_lock, read_bounded_text
from ..remote.tools import (
    t_remote_enroll,
    t_remote_exec,
    t_remote_list,
    t_remote_list_nodes,
    t_remote_read,
    t_remote_write,
)
from ..safety.secrets import redact
from ..shell.manager import get_shell_manager
from ..shell.runner import run_command
from .browser import t_browser
from .deep_research import t_deep_research
from .firecrawl import (
    _FIRECRAWL_API_URL as _FIRECRAWL_API_URL,
)
from .firecrawl import (
    _FIRECRAWL_FALLBACK_ERRORS as _FIRECRAWL_FALLBACK_ERRORS,
)
from .firecrawl import (
    _FIRECRAWL_MIN_DIRECT_CHARS as _FIRECRAWL_MIN_DIRECT_CHARS,
)
from .firecrawl import (
    _FIRECRAWL_RESPONSE_MAX_BYTES as _FIRECRAWL_RESPONSE_MAX_BYTES,
)
from .firecrawl import (
    _firecrawl_api_key as _firecrawl_api_key,
)
from .firecrawl import (
    _firecrawl_scrape as _firecrawl_scrape,
)
from .input_coercion import _coerce_float as _coerce_float
from .input_coercion import _coerce_int as _coerce_int
from .memory_tool import bind_memory_search as bind_memory_search
from .memory_tool import t_search_memory as t_search_memory
from .repo_explorer import explore_repo
from .router import normalize_tool_name as normalize_tool_name  # re-exported for callers
from .sandbox import t_sandbox_exec
from .web_common import (
    _WEB_FETCH_MAX_BODY_BYTES as _WEB_FETCH_MAX_BODY_BYTES,
)
from .web_common import (
    _WEB_FETCH_MAX_CHARS as _WEB_FETCH_MAX_CHARS,
)
from .web_common import (
    _WEB_FETCH_MAX_REDIRECTS as _WEB_FETCH_MAX_REDIRECTS,
)
from .web_common import (
    _decode_web_body as _decode_web_body,
)
from .web_common import (
    _normalize_search_url as _normalize_search_url,
)
from .web_common import (
    _read_web_body as _read_web_body,
)
from .web_common import (
    _web_fetch_url_error as _web_fetch_url_error,
)

log = logging.getLogger("norax.dispatch.tools")


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict
    fn: Callable[..., Awaitable[dict]]


# --- T0 tools --------------------------------------------------------------

# Process-local LRU for reads keyed by (path, mtime_ns, offset, limit). It
# avoids redundant disk reads while file metadata supplies freshness across
# turns. Direct writer tools also invalidate eagerly. The cache is bounded so
# scans across many files cannot grow memory without limit.

_READ_CACHE: _OrderedDict[tuple, dict] = _OrderedDict()
_READ_CACHE_MAX = 64
_READ_AUTO_PAGE_BYTES = 200_000
_READ_EXACT_LINE_COUNT_MAX_BYTES = 8_000_000
_READ_OFFLOAD_BYTES = 1_000_000
_READ_MAX_PAGE_LINES = 20_000
_READ_MAX_LINE_CHARS = 64_000
_READ_MAX_PAGE_CHARS = 1_000_000
_LIST_DIR_DEFAULT_LIMIT = 500
_LIST_DIR_MAX_LIMIT = 2_000
_LIST_DIR_SCAN_LIMIT = 20_000


def _read_text_page(
    path: Path,
    *,
    offset: int,
    limit: int | None,
    count_to_eof: bool,
) -> tuple[list[str], int | None, int | None, bool, int | None, bool]:
    """Read one text page without scanning large files past the requested data."""
    if limit is None:
        text = path.read_text(encoding="utf-8", errors="replace")
        full_lines = text.splitlines()
        return full_lines, len(full_lines), None, False, None, False

    page_lines: list[str] = []
    observed_lines = 0
    has_more = False
    line_truncated_at: int | None = None
    page_char_limit_reached = False
    page_chars = 0
    end = offset + limit
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        index = 0
        while True:
            line = handle.readline(_READ_MAX_LINE_CHARS + 1)
            if not line:
                break
            observed_lines = index + 1
            line_too_long = len(line) > _READ_MAX_LINE_CHARS and not line.endswith("\n")
            if offset <= index < end:
                clean_line = line[:_READ_MAX_LINE_CHARS].rstrip("\r\n")
                separator_chars = 1 if page_lines else 0
                if page_chars + separator_chars + len(clean_line) > _READ_MAX_PAGE_CHARS:
                    has_more = True
                    page_char_limit_reached = True
                    break
                page_lines.append(clean_line)
                page_chars += separator_chars + len(clean_line)
            if line_too_long:
                line_truncated_at = index
                break
            if page_chars >= _READ_MAX_PAGE_CHARS:
                has_more = bool(handle.read(1))
                page_char_limit_reached = has_more
                break
            elif index >= end:
                has_more = True
                if not count_to_eof:
                    break
            index += 1

    total_lines = (
        observed_lines if line_truncated_at is None and (count_to_eof or not has_more) else None
    )
    total_lines_at_least = observed_lines if total_lines is None else None
    return (
        page_lines,
        total_lines,
        total_lines_at_least,
        has_more,
        line_truncated_at,
        page_char_limit_reached,
    )


def _tool_diagnostic(value: object, *, max_chars: int = 500) -> str:
    """Return a bounded, secret-scrubbed diagnostic for tool results."""
    return str(redact(str(value)))[:max_chars]


async def t_repo_explore(*, query: str, root: str | None = None, budget: int | None = None) -> dict:
    """Explore a repository and return relevant file:line citations.

    Uses Microsoft FastContext-1.0-4B-SFT if the model is available in the
    local Ollama registry; otherwise uses a deterministic local fallback
    (file/path ranking + grep). Returns citations sorted by confidence.
    """
    return await explore_repo(query, root=root, budget=budget)


async def t_read(
    *, path: str, limit: int | None = None, offset: int = 0, large: bool = False
) -> dict:
    # Defensive: weak models often emit offset/limit as strings.
    offset = _coerce_int(offset, default=0) or 0
    if limit is not None:
        limit = _coerce_int(limit)
    large = large is True or (
        isinstance(large, str) and large.strip().lower() in {"1", "true", "yes"}
    )
    offset = max(0, offset)
    if limit is not None:
        limit = min(_READ_MAX_PAGE_LINES, max(1, limit))
    p = Path(path).expanduser()
    try:
        file_stat = p.stat()
    except FileNotFoundError:
        return {
            "ok": False,
            "error": "file_not_found",
            "path": str(p),
            "suggestion": "Path does not exist — list_dir the parent to find the real path.",
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": "file_stat_failed",
            "path": str(p),
            "detail": _tool_diagnostic(exc),
        }
    if stat_mod.S_ISDIR(file_stat.st_mode):
        return {
            "ok": False,
            "error": "is_a_directory",
            "suggestion": "This is a directory — use list_dir instead of read.",
            "path": str(p),
            "hint": "use list_dir to list directory contents",
        }
    if not stat_mod.S_ISREG(file_stat.st_mode):
        return {
            "ok": False,
            "error": "unsupported_file_type",
            "path": str(p),
            "suggestion": "Use read only for regular files; use a purpose-built tool for devices or streams.",
        }
    mtime_ns = file_stat.st_mtime_ns
    size = file_stat.st_size
    key = (str(p), mtime_ns, int(offset), int(limit) if limit is not None else -1, large)
    cached = _READ_CACHE.get(key)
    if cached is not None:
        _READ_CACHE.move_to_end(key)
        return {**cached, "cached": True}
    # Transparently page large files instead of turning a normal read into a
    # hard failure. The model receives a next_offset and can keep reading.
    auto_paginated = False
    if limit is None and size > _READ_AUTO_PAGE_BYTES:
        limit = 2000
        auto_paginated = True
    count_to_eof = size <= _READ_EXACT_LINE_COUNT_MAX_BYTES
    try:
        if size > _READ_OFFLOAD_BYTES:
            page = await asyncio.to_thread(
                _read_text_page,
                p,
                offset=offset,
                limit=limit,
                count_to_eof=count_to_eof,
            )
        else:
            page = _read_text_page(
                p,
                offset=offset,
                limit=limit,
                count_to_eof=count_to_eof,
            )
    except OSError as exc:
        return {
            "ok": False,
            "error": "file_read_failed",
            "path": str(p),
            "detail": _tool_diagnostic(exc),
        }
    lines, total, total_at_least, has_more, line_truncated_at, page_char_limit_reached = page
    out = {
        "ok": True,
        "path": str(p),
        "content": "\n".join(lines),
        "total_lines": total,
    }
    if total_at_least is not None:
        out["total_lines_at_least"] = total_at_least
    if line_truncated_at is not None:
        out.update(
            {
                "truncated": True,
                "line_truncated_at": line_truncated_at,
                "line_char_limit": _READ_MAX_LINE_CHARS,
                "suggestion": "Use exec with a byte-oriented tool to inspect the remainder of this unusually long line.",
            }
        )
    if page_char_limit_reached:
        out["page_char_limit"] = _READ_MAX_PAGE_CHARS
    if limit is not None and (has_more or (total is not None and offset + len(lines) < total)):
        out.update(
            {
                "truncated": True,
                "next_offset": offset + len(lines),
                "page_limit": limit,
            }
        )
    if auto_paginated:
        out["auto_paginated"] = True
    if large:
        out["_large_read"] = True
    _READ_CACHE[key] = out.copy()
    if len(_READ_CACHE) > _READ_CACHE_MAX:
        _READ_CACHE.popitem(last=False)
    return out


_TOOL_META_KEYS = frozenset(
    {
        "name",
        "tool",
        "tool_name",
        "function",
        "arguments",
        "args",
        "input",
        "type",
        "id",
        "tool_calls",
        "tools",
        "calls",
    }
)

_EXEC_PLACEHOLDER_COMMANDS = frozenset(
    {
        "...",
        "..",
        ".",
        "your command",
        "your shell command here",
        "your shell command",
        "<command>",
        "<your command>",
        "your bash command",
        "actual command here",
        "command here",
        "shell command here",
    }
)


def exec_command_is_placeholder(command: str) -> bool:
    """True when the model copied a nudge/example instead of a real command."""
    normalized = (command or "").strip().lower()
    if not normalized:
        return True
    if normalized in _EXEC_PLACEHOLDER_COMMANDS:
        return True
    return normalized.startswith("your ") and "command" in normalized


def exec_args_valid(args: dict | None) -> bool:
    """True when normalized exec args include a non-empty, non-placeholder command."""
    normalized = normalize_tool_args("exec", args)
    cmd = normalized.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return False
    return not exec_command_is_placeholder(cmd)


def normalize_tool_args(tool: str, args: dict | None) -> dict:
    """Fill safe defaults for common model formatting mistakes."""
    out: dict[str, Any] = dict(args or {})
    if tool in {"remote_exec", "remote_read", "remote_list", "remote_write"}:
        if not out.get("node_id"):
            for alt in ("node", "nodeId", "node_name", "target", "host"):
                val = out.get(alt)
                if val and isinstance(val, str) and val.strip():
                    out["node_id"] = out.pop(alt)
                    break
    if tool in {"list_dir", "remote_list"}:
        path = out.get("path")
        if not path or (isinstance(path, str) and not str(path).strip()):
            out["path"] = "."
    if tool in {"read", "remote_read"}:
        if "offset" in out:
            out["offset"] = _coerce_int(out.get("offset"), default=0) or 0
        if "limit" in out:
            coerced = _coerce_int(out.get("limit"))
            if coerced is not None:
                out["limit"] = coerced
            else:
                out.pop("limit", None)
    if tool in {"exec", "remote_exec", "shell"}:
        if not out.get("command"):
            for alt in (
                "cmd",
                "shell",
                "script",
                "bash_command",
                "shell_command",
                "command_line",
                "terminal_command",
                "bash",
                "value",
            ):
                val = out.get(alt)
                if val and isinstance(val, str):
                    out["command"] = out.pop(alt)
                    break
            # Nested: {"parameters": {"command": "..."}} or {"input": {"command": "..."}}
            if not out.get("command"):
                for nest_key in ("parameters", "input", "args"):
                    nested_value = out.get(nest_key)
                    if isinstance(nested_value, dict) and nested_value.get("command"):
                        nested = dict(nested_value)
                        out["command"] = nested.pop("command")
                        for k, v in nested.items():
                            out.setdefault(k, v)
                        out.pop(nest_key, None)
                        break
                    if isinstance(nested_value, str) and nested_value.strip():
                        out["command"] = nested_value
                        out.pop(nest_key, None)
                        break
        cmd = out.get("command")
        if isinstance(cmd, list):
            if cmd and all(isinstance(part, str) for part in cmd):
                out["command"] = shlex.join(cmd)
            else:
                out.pop("command", None)
        elif cmd is not None and not isinstance(cmd, str):
            out.pop("command", None)
        if "timeout" in out:
            coerced_timeout = _coerce_float(out.get("timeout"))
            if coerced_timeout is not None:
                out["timeout"] = float(coerced_timeout)
            else:
                out.pop("timeout", None)
    if tool == "web_fetch" and "max_chars" in out:
        coerced = _coerce_int(out.get("max_chars"))
        if coerced is not None:
            out["max_chars"] = coerced
    if tool == "web_search" and "count" in out:
        coerced = _coerce_int(out.get("count"))
        if coerced is not None:
            out["count"] = coerced
    if tool == "web_search" and "recency_days" in out:
        coerced = _coerce_int(out.get("recency_days"))
        if coerced is not None:
            out["recency_days"] = coerced
        else:
            out.pop("recency_days", None)
    return out


def _directory_names(path: Path) -> tuple[list[str], bool]:
    """Return sorted names with a hard ceiling on pathological directories."""
    names: list[str] = []
    scan_truncated = False
    with os.scandir(path) as entries:
        for entry in entries:
            if len(names) >= _LIST_DIR_SCAN_LIMIT:
                scan_truncated = True
                break
            names.append(entry.name)
    names.sort()
    return names, scan_truncated


async def t_list_dir(
    *,
    path: str = ".",
    offset: int = 0,
    limit: int = _LIST_DIR_DEFAULT_LIMIT,
) -> dict:
    p = Path(path).expanduser()
    parsed_offset = _coerce_int(offset)
    parsed_limit = _coerce_int(limit)
    if parsed_offset is None or parsed_offset < 0:
        return {"ok": False, "error": "offset_must_be_a_non_negative_integer", "path": str(p)}
    if parsed_limit is None or parsed_limit < 1:
        return {"ok": False, "error": "limit_must_be_a_positive_integer", "path": str(p)}
    limit = min(parsed_limit, _LIST_DIR_MAX_LIMIT)
    offset = parsed_offset
    try:
        directory_stat = p.stat()
    except FileNotFoundError:
        return {
            "ok": False,
            "error": "path_not_found",
            "path": str(p),
            "suggestion": "List a parent directory to discover valid entries.",
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": "directory_stat_failed",
            "path": str(p),
            "detail": _tool_diagnostic(exc),
        }
    if not stat_mod.S_ISDIR(directory_stat.st_mode):
        return {
            "ok": False,
            "error": "not_a_directory",
            "path": str(p),
            "suggestion": "This is a file — use read instead of list_dir.",
        }
    try:
        names, scan_truncated = await asyncio.to_thread(_directory_names, p)
    except PermissionError:
        return {
            "ok": False,
            "error": "permission_denied",
            "path": str(p),
            "suggestion": "Pick a path inside the workspace or allowed directories.",
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": "directory_scan_failed",
            "path": str(p),
            "detail": _tool_diagnostic(exc),
        }

    entries: list[dict[str, Any]] = []
    page_names = names[offset : offset + limit]
    for name in page_names:
        child = p / name
        try:
            child_stat = child.stat()
            if stat_mod.S_ISDIR(child_stat.st_mode):
                kind = "dir"
                size: int | None = None
            elif stat_mod.S_ISREG(child_stat.st_mode):
                kind = "file"
                size = child_stat.st_size
            else:
                kind = "other"
                size = None
            entries.append({"name": name, "kind": kind, "size": size})
        except (FileNotFoundError, PermissionError, OSError):
            # Directory contents can change between scan and metadata lookup.
            entries.append({"name": name, "kind": "unknown", "size": None})

    more_scanned_entries = offset + len(page_names) < len(names)
    result: dict[str, Any] = {
        "ok": True,
        "path": str(p),
        "entries": entries,
        "offset": offset,
        "limit": limit,
        "total_entries": None if scan_truncated else len(names),
    }
    if scan_truncated:
        result.update(
            {
                "scan_truncated": True,
                "total_entries_at_least": _LIST_DIR_SCAN_LIMIT + 1,
                "suggestion": "Use repo_explore, search, or a narrower directory instead of listing every entry.",
            }
        )
    if more_scanned_entries:
        result.update({"truncated": True, "next_offset": offset + len(page_names)})
    elif scan_truncated:
        result["truncated"] = True
    return result


def _extract_readable_text(html: str, url: str) -> tuple[str, str]:
    """Extract main article text from HTML. Returns (text, extractor_name).

    Tries trafilatura first (best quality), falls back to regex strip.
    trafilatura is an optional dep — install with `pip install trafilatura`.
    """
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html,
            url=url,
            favor_recall=True,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if extracted and len(extracted) >= 50:
            return extracted, "trafilatura"
    except ImportError:
        pass
    except Exception:
        pass
    import html as _html
    import re as _re

    text = _re.sub(
        r"<(script|style|noscript|template)[^>]*>.*?</\1>",
        "",
        html,
        flags=_re.DOTALL | _re.IGNORECASE,
    )
    text = _re.sub(r"<!--.*?-->", "", text, flags=_re.DOTALL)
    text = _re.sub(r"<(br|/p|/div|/li|/tr|/h[1-6])[^>]*>", "\n", text, flags=_re.IGNORECASE)
    text = _re.sub(r"<[^>]+>", " ", text)
    text = _html.unescape(text)
    text = _re.sub(r"[ \t]+", " ", text)
    text = _re.sub(r"\n[ \t]*", "\n", text)
    text = _re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, "regex"


async def _read_search_json(response: httpx.Response) -> dict:
    """Decode one bounded search-provider response object."""
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = 0
        if declared_size > _SEARCH_RESPONSE_MAX_BYTES:
            raise ValueError("search provider response exceeds the byte limit")
    raw, truncated = await _read_web_body(response, _SEARCH_RESPONSE_MAX_BYTES)
    if truncated:
        raise ValueError("search provider response exceeds the byte limit")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("search provider response must be an object")
    return payload


async def _web_fetch_direct(*, url: str, max_chars: int = 24000) -> dict:
    """Direct HTTP fetch of a URL (no external fallback service).

    Uses trafilatura for clean article extraction when available; falls back
    to HTML tag-stripping regex otherwise.
    """
    requested_url = str(url or "").strip()
    parsed_limit = _coerce_int(max_chars, default=24_000) or 24_000
    max_chars = min(_WEB_FETCH_MAX_CHARS, max(1, parsed_limit))
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/json,text/plain;q=0.8,*/*;q=0.5",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    }
    redirects: list[str] = []
    current_url = requested_url
    try:
        async with _shared_client_cm() as c:
            for redirect_count in range(_WEB_FETCH_MAX_REDIRECTS + 1):
                unsafe_reason = await _web_fetch_url_error(current_url)
                if unsafe_reason:
                    return {
                        "ok": False,
                        "error": "url_not_allowed",
                        "url": current_url,
                        "detail": unsafe_reason,
                    }
                async with c.stream("GET", current_url, headers=headers) as r:
                    if 300 <= r.status_code < 400:
                        location = r.headers.get("location")
                        if not location:
                            return {
                                "ok": False,
                                "error": "redirect_without_location",
                                "status": r.status_code,
                                "url": current_url,
                            }
                        if redirect_count >= _WEB_FETCH_MAX_REDIRECTS:
                            return {
                                "ok": False,
                                "error": "too_many_redirects",
                                "url": current_url,
                                "redirects": redirects,
                            }
                        current_url = urljoin(str(r.url), location)
                        redirects.append(current_url)
                        continue

                    content_type = r.headers.get("content-type", "")
                    media_type = content_type.split(";", 1)[0].strip().lower()
                    text_like = (
                        not media_type
                        or media_type.startswith("text/")
                        or media_type
                        in {
                            "application/json",
                            "application/ld+json",
                            "application/xml",
                            "application/xhtml+xml",
                            "application/javascript",
                        }
                        or media_type.endswith("+json")
                        or media_type.endswith("+xml")
                    )
                    if not text_like:
                        return {
                            "ok": False,
                            "error": "unsupported_content_type",
                            "status": r.status_code,
                            "url": str(r.url),
                            "content_type": media_type,
                        }
                    if r.status_code >= 400:
                        raw_error, error_truncated = await _read_web_body(r, 4_000)
                        return {
                            "ok": False,
                            "error": "http_status",
                            "status": r.status_code,
                            "url": str(r.url),
                            "content_type": media_type,
                            "detail": _tool_diagnostic(
                                _decode_web_body(raw_error, r), max_chars=4_000
                            ),
                            "truncated": error_truncated,
                        }
                    body_limit = min(
                        _WEB_FETCH_MAX_BODY_BYTES,
                        max(256_000, max_chars * 4),
                    )
                    raw, body_truncated = await _read_web_body(r, body_limit)
                    final_url = str(r.url)
                    status = r.status_code
                    break
            else:  # pragma: no cover - loop returns on every exhausted path
                return {"ok": False, "error": "too_many_redirects", "url": current_url}
    except httpx.ConnectError:
        return {
            "ok": False,
            "error": "connection_failed",
            "suggestion": "Check the URL is reachable; retry once, then try web_search for an alternative source.",
            "url": current_url,
            "hint": "Could not connect to the server. Check the URL or try again.",
        }
    except httpx.TimeoutException:
        return {
            "ok": False,
            "error": "timeout",
            "url": current_url,
            "suggestion": "Retry once; if it times out again, try browser or web_search as a fallback.",
        }
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "error": "request_error",
            "url": current_url,
            "detail": _tool_diagnostic(exc),
            "suggestion": "If blocked (403/404), fall back to browser or web_search for another source.",
        }
    text = _decode_web_body(raw, r)
    extractor = "raw"
    if media_type == "text/html" and len(text) > 200:
        text, extractor = await asyncio.to_thread(_extract_readable_text, text, final_url)
    truncated = body_truncated or len(text) > max_chars
    text = text[:max_chars]
    return {
        "ok": True,
        "status": status,
        "url": final_url,
        "content_type": media_type,
        "extractor": extractor,
        "redirects": redirects,
        "truncated": truncated,
        "text": text,
    }


async def t_web_fetch(*, url: str, max_chars: int = 24000) -> dict:
    """Fetch a URL and return its content as text.

    Direct HTTP fetch first (fast, no API cost).  When the direct fetch is
    bot-blocked, times out, or yields a JS-rendered shell, transparently
    retries through the Firecrawl scrape API (renders JS, bypasses common
    anti-bot walls).  Private-network targets are never sent to Firecrawl —
    the direct path is the only path for them.
    """
    requested_url = str(url or "").strip()
    parsed_limit = _coerce_int(max_chars, default=24_000) or 24_000
    max_chars = min(_WEB_FETCH_MAX_CHARS, max(1, parsed_limit))

    direct = await _web_fetch_direct(url=requested_url, max_chars=max_chars)
    if not _firecrawl_api_key():
        return direct

    if direct.get("ok") is True:
        text = str(direct.get("text") or "")
        media = str(direct.get("content_type") or "")
        if media != "text/html" or len(text) >= min(_FIRECRAWL_MIN_DIRECT_CHARS, max_chars):
            return direct
        # Thin HTML page — likely a JS-rendered shell.  Retry through
        # Firecrawl; keep the direct result if Firecrawl cannot do better.
    elif str(direct.get("error") or "") not in _FIRECRAWL_FALLBACK_ERRORS:
        return direct

    # Check disclosure permission only when the fallback is actually needed.
    # A direct-fetch intranet opt-in never authorizes an external scraper.
    # Check the redirect destination too; a public URL may lead to an intranet.
    for target in dict.fromkeys((requested_url, str(direct.get("url") or requested_url))):
        if await _web_fetch_url_error(target, honor_private_opt_in=False) is not None:
            return direct

    fc = await _firecrawl_scrape(requested_url, max_chars)
    if fc.get("ok") is not True:
        if direct.get("ok") is True:
            return direct
        note = f"fallback_failed ({fc.get('error', 'unknown')})"
        if direct.get("error"):
            direct = dict(direct)
            direct["fallback"] = note
        return direct
    return {
        "ok": True,
        "status": 200,
        "url": str(fc.get("source_url") or requested_url),
        "content_type": "text/html",
        "extractor": "firecrawl",
        "truncated": bool(fc.get("truncated")),
        "text": str(fc.get("markdown") or ""),
    }


# ── web_search cache: 5-min TTL, capped 64 entries ──
_RRF_K = 60
_SEARCH_RESPONSE_MAX_BYTES = 4 * 1024 * 1024
_SEARCH_CACHE_MAX_BYTES = 16 * 1024 * 1024
_SEARCH_ITEM_TITLE_MAX_CHARS = 1_000
_SEARCH_ITEM_SNIPPET_MAX_CHARS = 8_000
_SEARCH_ITEM_URL_MAX_CHARS = 4_096


def _search_text(value: object, *, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value[:max_chars].split())


def _sanitize_search_item(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    raw_url = item.get("url")
    if not isinstance(raw_url, str) or not 1 <= len(raw_url) <= _SEARCH_ITEM_URL_MAX_CHARS:
        return None
    url = raw_url.strip()
    if any(ord(char) < 0x20 for char in url):
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    sanitized = {
        "title": _search_text(item.get("title"), max_chars=_SEARCH_ITEM_TITLE_MAX_CHARS),
        "url": url,
        "snippet": _search_text(
            item.get("snippet"),
            max_chars=_SEARCH_ITEM_SNIPPET_MAX_CHARS,
        ),
    }
    engine = _search_text(item.get("engine"), max_chars=128)
    if engine:
        sanitized["engine"] = engine
    return sanitized


def _item_richness(item: dict) -> int:
    return len(str(item.get("snippet") or "")) + len(str(item.get("title") or ""))


def _dedupe_rich(items: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for it in items:
        key = _normalize_search_url(str(it.get("url") or ""))
        if not key:
            continue
        if key not in best or _item_richness(it) > _item_richness(best[key]):
            best[key] = it
    return list(best.values())


def _rrf_merge(ranked_lists: list[list[dict]], *, limit: int) -> list[dict]:
    """Reciprocal Rank Fusion across heterogeneous search backends."""
    scores: dict[str, float] = {}
    items_by_url: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, it in enumerate(lst):
            url = _normalize_search_url(str(it.get("url") or ""))
            if not url:
                continue
            scores[url] = scores.get(url, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if url not in items_by_url or _item_richness(it) > _item_richness(items_by_url[url]):
                items_by_url[url] = it
    ordered = sorted(
        scores.keys(),
        key=lambda u: (-scores[u], -_item_richness(items_by_url[u])),
    )
    return [items_by_url[u] for u in ordered[:limit]]


def _diversify_domains(items: list[dict], *, max_per: int = 2) -> list[dict]:
    from urllib.parse import urlparse

    counts: dict[str, int] = {}
    out: list[dict] = []
    for it in items:
        try:
            dom = urlparse(str(it.get("url") or "")).netloc.lower()
        except Exception:
            dom = ""
        if counts.get(dom, 0) >= max_per:
            continue
        counts[dom] = counts.get(dom, 0) + 1
        out.append(it)
    return out


def _decompose_search_query(query: str) -> list[str]:
    """Split compare/versus/and queries into parallel sub-queries."""
    import re

    q = (query or "").strip()
    if not q:
        return []
    for sep in (" vs ", " versus ", " compare ", " and "):
        if sep in q.lower():
            parts = [p.strip() for p in re.split(re.escape(sep), q, flags=re.I) if p.strip()]
            if len(parts) >= 2 and all(len(p) > 8 for p in parts[:3]):
                return parts[:3]
    return [q]


def _finalize_search_items(items: list[dict], *, count: int) -> list[dict]:
    sanitized = [item for raw in items if (item := _sanitize_search_item(raw)) is not None]
    return _diversify_domains(_dedupe_rich(sanitized))[:count]


_WEB_SEARCH_CACHE: _OrderedDict[tuple, tuple[float, dict]] = _OrderedDict()
_WEB_SEARCH_CACHE_TTL = 600.0  # seconds
_WEB_SEARCH_CACHE_MAX = 128


# ── Shared pooled HTTP client (HTTP/2 where the peer supports it) ──────────
# One long-lived client per event loop replaces the per-call `httpx.AsyncClient`
# that used to be opened for every search/fetch. Reusing the client keeps
# connections warm (no repeated TLS handshake to remote backends like Serper)
# and lets HTTP/2 multiplex concurrent requests. Same requests, same results —
# pure transport speedup, zero quality change.
_HTTP_CLIENTS: dict[int, httpx.AsyncClient] = {}


def _shared_http_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _HTTP_CLIENTS.get(id(loop))
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=10.0),
            http2=True,
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
        _HTTP_CLIENTS[id(loop)] = client
    return client


@asynccontextmanager
async def _shared_client_cm() -> AsyncIterator[httpx.AsyncClient]:
    """Yield the shared pooled client inside an ``async with``.

    Lets existing ``async with httpx.AsyncClient(...) as c:`` call sites swap
    to the pooled client with no reindent — the context manager is a no-op
    wrapper (the client is process/loop-scoped, not per-call).
    """
    yield _shared_http_client()


# ── Durable web_search cache (survives restarts) ───────────────────────────
# In-memory LRU is the hot path; this JSON file is a durable mirror so repeated
# queries across restarts are instant. Same TTL as the in-memory cache, so
# freshness semantics are unchanged — it just persists.
_WEB_SEARCH_DISK_CACHE: dict[str, tuple[float, dict]] | None = None
_WEB_SEARCH_DISK_CACHE_SOURCE: Path | None = None
_WEB_SEARCH_DISK_CACHE_LOCK = threading.RLock()
_WEB_SEARCH_DISK_SAVE_PENDING = False
_WEB_SEARCH_DISK_SAVE_DIRTY = False


def _web_search_disk_cache_path() -> Path:
    configured = os.environ.get("NORAX_WEB_SEARCH_CACHE_PATH")
    if configured:
        return Path(configured).expanduser()
    state_root = os.environ.get("NORAX_STATE_DIR")
    if state_root:
        return Path(state_root).expanduser() / "web_search_cache.json"
    return Path.home() / ".norax" / "web_search_cache.json"


def _validated_cache_key(raw_key: object) -> tuple[str, int, int] | None:
    if not isinstance(raw_key, str) or len(raw_key) > 4_096:
        return None
    try:
        value = json.loads(raw_key)
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not isinstance(value[0], str)
        or not 1 <= len(value[0]) <= 2_000
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or not 1 <= value[1] <= 15
        or isinstance(value[2], bool)
        or not isinstance(value[2], int)
        or not 0 <= value[2] <= 365
    ):
        return None
    return value[0], value[1], value[2]


def _normalize_cached_search_payload(payload: object, key: tuple[str, int, int]) -> dict | None:
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    provider = payload.get("provider")
    query = payload.get("query")
    raw_items = payload.get("items")
    if (
        not isinstance(provider, str)
        or not 1 <= len(provider) <= 128
        or not isinstance(query, str)
        or not 1 <= len(query) <= 2_000
        or query.casefold() != key[0].casefold()
        or not isinstance(raw_items, list)
        or len(raw_items) > 15
    ):
        return None
    items = [item for raw in raw_items if (item := _sanitize_search_item(raw)) is not None]
    if not items or len(items) > key[1]:
        return None
    normalized: dict = {
        "ok": True,
        "provider": _search_text(provider, max_chars=128),
        "query": query,
        "items": items,
    }
    for field_name, max_items, max_chars in (
        ("engines", 64, 128),
        ("sub_queries", 3, 2_000),
    ):
        raw_values = payload.get(field_name)
        if raw_values is None:
            continue
        if (
            not isinstance(raw_values, list)
            or len(raw_values) > max_items
            or not all(isinstance(value, str) and len(value) <= max_chars for value in raw_values)
        ):
            return None
        normalized[field_name] = list(raw_values)
    time_range = payload.get("time_range")
    if time_range is not None:
        if time_range not in {"day", "week", "month", "year"}:
            return None
        normalized["time_range"] = time_range
    return normalized


def _load_disk_search_cache() -> dict[str, tuple[float, dict]]:
    global _WEB_SEARCH_DISK_CACHE, _WEB_SEARCH_DISK_CACHE_SOURCE
    with _WEB_SEARCH_DISK_CACHE_LOCK:
        cache_path = _web_search_disk_cache_path()
        if _WEB_SEARCH_DISK_CACHE is not None and _WEB_SEARCH_DISK_CACHE_SOURCE == cache_path:
            return _WEB_SEARCH_DISK_CACHE
        data: dict[str, tuple[float, dict]] = {}
        try:
            raw = json.loads(
                read_bounded_text(
                    cache_path,
                    max_bytes=_SEARCH_CACHE_MAX_BYTES,
                )
            )
            if not isinstance(raw, dict) or len(raw) > _WEB_SEARCH_CACHE_MAX:
                raise ValueError("invalid web-search cache")
            now = time.time()
            for raw_key, value in raw.items():
                key = _validated_cache_key(raw_key)
                if key is None or not isinstance(value, list | tuple) or len(value) != 2:
                    continue
                timestamp, payload = value
                if (
                    isinstance(timestamp, bool)
                    or not isinstance(timestamp, int | float)
                    or not math.isfinite(float(timestamp))
                    or not 0 <= now - float(timestamp) < _WEB_SEARCH_CACHE_TTL
                ):
                    continue
                normalized = _normalize_cached_search_payload(payload, key)
                if normalized is not None:
                    data[_disk_cache_key(key)] = (float(timestamp), normalized)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            data = {}
        _WEB_SEARCH_DISK_CACHE = data
        _WEB_SEARCH_DISK_CACHE_SOURCE = cache_path
        return data


def _save_disk_search_cache() -> None:
    try:
        with _WEB_SEARCH_DISK_CACHE_LOCK:
            snapshot = copy.deepcopy(_WEB_SEARCH_DISK_CACHE or {})
            cache_path = _WEB_SEARCH_DISK_CACHE_SOURCE or _web_search_disk_cache_path()
        serialized = json.dumps(
            snapshot,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(serialized.encode("utf-8")) > _SEARCH_CACHE_MAX_BYTES:
            return
        atomic_write_text(cache_path, serialized, mode=0o600)
    except Exception:
        pass  # cache is best-effort; never break a search on a cache write


def _flush_disk_search_cache() -> None:
    global _WEB_SEARCH_DISK_SAVE_DIRTY, _WEB_SEARCH_DISK_SAVE_PENDING
    while True:
        with _WEB_SEARCH_DISK_CACHE_LOCK:
            _WEB_SEARCH_DISK_SAVE_DIRTY = False
        _save_disk_search_cache()
        with _WEB_SEARCH_DISK_CACHE_LOCK:
            if not _WEB_SEARCH_DISK_SAVE_DIRTY:
                _WEB_SEARCH_DISK_SAVE_PENDING = False
                return


def _schedule_disk_search_cache_save() -> None:
    global _WEB_SEARCH_DISK_SAVE_DIRTY, _WEB_SEARCH_DISK_SAVE_PENDING
    with _WEB_SEARCH_DISK_CACHE_LOCK:
        _WEB_SEARCH_DISK_SAVE_DIRTY = True
        if _WEB_SEARCH_DISK_SAVE_PENDING:
            return
        _WEB_SEARCH_DISK_SAVE_PENDING = True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _flush_disk_search_cache()
    else:
        loop.run_in_executor(None, _flush_disk_search_cache)


def _cache_put_search(key: tuple, payload: dict) -> None:
    import time as _time

    normalized_key = _validated_cache_key(_disk_cache_key(key))
    normalized = (
        _normalize_cached_search_payload(payload, normalized_key)
        if normalized_key is not None
        else None
    )
    if normalized is None:
        return
    _WEB_SEARCH_CACHE[key] = (_time.monotonic(), copy.deepcopy(normalized))
    _WEB_SEARCH_CACHE.move_to_end(key)
    while len(_WEB_SEARCH_CACHE) > _WEB_SEARCH_CACHE_MAX:
        _WEB_SEARCH_CACHE.popitem(last=False)
    # Mirror to durable cache (same TTL, survives restarts).
    # Wall-clock (not monotonic) so entries stay correctly dated across reboots.
    with _WEB_SEARCH_DISK_CACHE_LOCK:
        disk = _load_disk_search_cache()
        disk[_disk_cache_key(key)] = (_time.time(), copy.deepcopy(normalized))
        while len(disk) > _WEB_SEARCH_CACHE_MAX:
            oldest = min(disk, key=lambda cache_key: disk[cache_key][0])
            disk.pop(oldest, None)
    _schedule_disk_search_cache_save()


def _disk_cache_key(key: tuple) -> str:
    return json.dumps(list(key), sort_keys=True)


def _disk_cache_get(key: tuple) -> dict | None:
    import time as _time

    cache_key = _disk_cache_key(key)
    with _WEB_SEARCH_DISK_CACHE_LOCK:
        disk = _load_disk_search_cache()
        entry = disk.get(cache_key)
        if entry is None:
            return None
        timestamp, payload = entry
        age = _time.time() - timestamp
        if 0 <= age < _WEB_SEARCH_CACHE_TTL:  # wall-clock: durable cache spans reboots
            return copy.deepcopy(payload)
        disk.pop(cache_key, None)
    _schedule_disk_search_cache_save()
    return None


def _search_provider_label(backends: set[str]) -> str:
    ordered = [name for name in ("tavily", "searxng", "serper") if name in backends]
    return "+".join(ordered) if ordered else "unknown"


_RECENCY_HINT_PATTERNS = (
    "latest",
    "today",
    "this week",
    "this month",
    "current",
    "just released",
    "newest",
    "recently",
    "breaking",
    "right now",
    "as of",
    "2026",
    "2025-",
)


def _infer_recency_days(query: str) -> int | None:
    """Detect time-sensitive queries and return suggested recency window."""
    q = query.lower()
    for needle in ("today", "right now", "breaking", "just released"):
        if needle in q:
            return 7
    for needle in ("this week", "newest", "latest", "recently"):
        if needle in q:
            return 30
    for needle in ("this month", "current", "2026"):
        if needle in q:
            return 90
    return None


def _searxng_time_range(days: int | None) -> str | None:
    if days is None:
        return None
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    return "year"


async def _tavily_search(
    query: str, count: int, api_key: str, recency_days: int | None
) -> dict | None:
    """Tavily research-grade search (Tier-0). Only used if NORAX_TAVILY_API_KEY set."""
    try:
        async with _shared_client_cm() as c:
            payload: dict = {
                "api_key": api_key,
                "query": query,
                "max_results": count,
                "search_depth": "basic",
                "include_answer": False,
            }
            if recency_days is not None:
                payload["days"] = max(1, min(recency_days, 365))
            async with c.stream("POST", "https://api.tavily.com/search", json=payload) as r:
                if r.status_code != 200:
                    return None
                data = await _read_search_json(r)
            results = data.get("results") or []
            if not isinstance(results, list) or not results:
                return None
            items = [
                {
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "snippet": it.get("content", ""),
                    "engine": "tavily",
                }
                for it in results[:count]
            ]
            return {"ok": True, "provider": "tavily", "query": query, "items": items}
    except Exception:
        return None


async def _fetch_searxng(
    query: str,
    count: int,
    *,
    searxng_url: str,
    recency_days: int | None,
) -> tuple[list[dict], list[str], str | None]:
    """Return (items, engines_used, time_range) from SearXNG.

    If a time_range filter yields zero results (many engines drop
    time-filtered queries from this host), retry without the filter so
    recency-inferred queries still return something rather than failing.
    """
    tr = _searxng_time_range(recency_days)

    async def _pages(
        client: httpx.AsyncClient, use_tr: bool
    ) -> tuple[list[dict], list, int | None]:
        items: list[dict] = []
        results: list = []
        last_status: int | None = None
        for page in (1, 2):
            params: dict = {"q": query, "format": "json", "pageno": page, "safesearch": "0"}
            if use_tr and tr:
                params["time_range"] = tr
            async with client.stream(
                "GET",
                f"{searxng_url}/search",
                params=params,
                headers={"Accept": "application/json"},
            ) as r:
                last_status = r.status_code
                if r.status_code != 200:
                    break
                data = await _read_search_json(r)
            results = data.get("results") or []
            if not isinstance(results, list):
                raise ValueError("searxng results must be a list")
            seen_urls: set[str] = set()
            for res in results[:250]:
                if not isinstance(res, dict):
                    continue
                url = res.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                items.append(
                    {
                        "title": res.get("title", ""),
                        "url": url,
                        "snippet": res.get("content", ""),
                        "engine": res.get("engine", "searxng"),
                    }
                )
                if len(items) >= count:
                    break
            if items:
                break
        return items, results, last_status

    async with _shared_client_cm() as c:
        items, results, last_status = await _pages(c, use_tr=True)
        if not items and tr:
            # time_range filter killed every engine — retry unfiltered
            items, results, last_status = await _pages(c, use_tr=False)
            tr = None
    if not items:
        if last_status == 200:
            raise RuntimeError("searxng: no results across page 1-2")
        raise RuntimeError(f"searxng: HTTP {last_status}")
    engines = sorted(
        {
            engine
            for row in results[:250]
            if isinstance(row, dict) and isinstance((engine := row.get("engine", "?")), str)
        }
    )
    return items, engines, tr


async def _fetch_serper(query: str, count: int, serper_key: str) -> list[dict]:
    async with _shared_client_cm() as c:
        async with c.stream(
            "POST",
            "https://google.serper.dev/search",
            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
            json={"q": query, "num": count},
        ) as r:
            if r.status_code != 200:
                raise RuntimeError(f"serper: HTTP {r.status_code}")
            data = await _read_search_json(r)
        organic = data.get("organic") or []
        if not isinstance(organic, list):
            raise ValueError("serper organic results must be a list")
        items = [
            {
                "title": it.get("title", ""),
                "url": it.get("link", ""),
                "snippet": it.get("snippet", ""),
                "engine": "serper",
            }
            for it in organic[:count]
            if isinstance(it, dict)
        ]
        if not items:
            raise RuntimeError("serper: no organic results")
        return items


async def t_web_search(
    *,
    query: str,
    count: int = 5,
    recency_days: int | None = None,
) -> dict:
    """Search the web using bounded, independently raced real backends.

    Tavily is optional (``NORAX_TAVILY_API_KEY``). SearXNG and optional Serper
    are queried together and RRF-ranked. Tavily races that tier so an outage in
    one provider never serially delays a healthy provider.

    Compare/versus/and queries decompose into parallel sub-queries (RRF merge).
    Results are URL-deduped (richest snippet wins) and domain-diversified.
    If every real search backend fails, this returns an explicit error; a text
    model is never presented as a search provider or allowed to invent URLs.
    """
    import time as _time

    searxng_url = os.environ.get("NORAX_SEARXNG_URL", "http://127.0.0.1:8889")
    serper_key = os.environ.get("NORAX_SERPER_API_KEY")
    tavily_key = os.environ.get("NORAX_TAVILY_API_KEY")
    if not isinstance(query, str):
        return {"ok": False, "error": "query_must_be_a_string"}
    normalized_query = query.strip()
    if not normalized_query:
        return {"ok": False, "error": "query_required"}
    if len(normalized_query) > 2_000:
        return {"ok": False, "error": "query_too_long", "max_chars": 2_000}
    if isinstance(count, bool):
        return {"ok": False, "error": "count_must_be_an_integer", "query": normalized_query}
    if isinstance(count, int):
        parsed_count = count
    elif isinstance(count, str) and count.strip().isascii() and count.strip().isdecimal():
        parsed_count = int(count.strip())
    else:
        return {"ok": False, "error": "count_must_be_an_integer", "query": normalized_query}
    count = max(1, min(parsed_count, 15))

    if recency_days is None:
        inferred = _infer_recency_days(normalized_query)
    else:
        if isinstance(recency_days, bool):
            return {
                "ok": False,
                "error": "recency_days_must_be_a_positive_integer",
                "query": normalized_query,
            }
        if isinstance(recency_days, int):
            inferred = recency_days
        elif (
            isinstance(recency_days, str)
            and recency_days.strip().isascii()
            and recency_days.strip().isdecimal()
        ):
            inferred = int(recency_days.strip())
        else:
            return {
                "ok": False,
                "error": "recency_days_must_be_a_positive_integer",
                "query": normalized_query,
            }
        if inferred <= 0:
            return {
                "ok": False,
                "error": "recency_days_must_be_a_positive_integer",
                "query": normalized_query,
            }
        inferred = min(inferred, 365)

    cache_key = (normalized_query.lower(), count, inferred or 0)
    now = _time.monotonic()
    cached = _WEB_SEARCH_CACHE.get(cache_key)
    if cached is not None:
        ts, payload = cached
        if now - ts < _WEB_SEARCH_CACHE_TTL:
            _WEB_SEARCH_CACHE.move_to_end(cache_key)
            result = copy.deepcopy(payload)
            result["cached"] = True
            return result
        _WEB_SEARCH_CACHE.pop(cache_key, None)
    # Disk I/O is never performed on the request event loop. This also checks
    # the durable cache after an in-memory entry expires.
    disk_payload = await asyncio.to_thread(_disk_cache_get, cache_key)
    if disk_payload is not None:
        result = copy.deepcopy(disk_payload)
        result["cached"] = True
        _WEB_SEARCH_CACHE[cache_key] = (time.monotonic(), copy.deepcopy(disk_payload))
        _WEB_SEARCH_CACHE.move_to_end(cache_key)
        return result

    errors: list[str] = []
    sub_queries = _decompose_search_query(normalized_query)
    parallel_mode = len(sub_queries) > 1

    async def _one_backend_search(
        q: str,
    ) -> tuple[list[list[dict]], list[str], set[str], str | None, list[str]]:
        local_errors: list[str] = []
        ranked: list[list[dict]] = []
        engines: list[str] = []
        backends: set[str] = set()
        tr_used: str | None = None

        async def _try_searxng() -> None:
            nonlocal tr_used
            try:
                sx_items, sx_engines, tr = await _fetch_searxng(
                    q,
                    count,
                    searxng_url=searxng_url,
                    recency_days=inferred,
                )
                ranked.append(sx_items)
                engines.extend(sx_engines)
                backends.add("searxng")
                tr_used = tr
            except Exception as e:
                local_errors.append(str(e))

        async def _try_serper() -> None:
            if not serper_key:
                return
            try:
                ranked.append(await _fetch_serper(q, count, serper_key))
                engines.append("serper")
                backends.add("serper")
            except Exception as e:
                local_errors.append(str(e))

        await asyncio.gather(_try_searxng(), _try_serper())
        return ranked, engines, backends, tr_used, local_errors

    async def _tier_one_search() -> tuple[dict | None, list[str]]:
        tier_errors: list[str] = []
        try:
            if parallel_mode:
                sub_results = await asyncio.gather(
                    *[_one_backend_search(sub_query) for sub_query in sub_queries]
                )
                all_ranked: list[list[dict]] = []
                all_engines: list[str] = []
                all_backends: set[str] = set()
                tr: str | None = None
                for ranked, engines, backends, tr_part, sub_errors in sub_results:
                    all_ranked.extend(ranked)
                    all_engines.extend(engines)
                    all_backends.update(backends)
                    tr = tr or tr_part
                    tier_errors.extend(sub_errors)
                merged = _finalize_search_items(
                    _rrf_merge(all_ranked, limit=count * 2),
                    count=count,
                )
                if merged:
                    payload: dict = {
                        "ok": True,
                        "provider": _search_provider_label(all_backends),
                        "query": normalized_query,
                        "sub_queries": sub_queries,
                        "engines": sorted(set(all_engines)),
                        "items": merged,
                    }
                    if tr:
                        payload["time_range"] = tr
                    return payload, tier_errors
            else:
                ranked, engines, backends, tr, sub_errors = await _one_backend_search(
                    normalized_query
                )
                tier_errors.extend(sub_errors)
                if ranked:
                    merged = _finalize_search_items(
                        _rrf_merge(ranked, limit=count * 2),
                        count=count,
                    )
                    payload = {
                        "ok": True,
                        "provider": _search_provider_label(backends),
                        "query": normalized_query,
                        "engines": sorted(set(engines)),
                        "items": merged,
                    }
                    if tr:
                        payload["time_range"] = tr
                    return payload, tier_errors
                if not sub_errors:
                    tier_errors.append("searxng: no results")
        except httpx.ConnectError:
            tier_errors.append("searxng: connection refused (is the container running?)")
        except httpx.TimeoutException:
            tier_errors.append("searxng: timeout")
        except Exception as exc:
            tier_errors.append(f"searxng: {type(exc).__name__}: {exc}")
        return None, tier_errors

    tier_one_task = asyncio.create_task(_tier_one_search())
    tavily_task = (
        asyncio.create_task(_tavily_search(normalized_query, count, tavily_key, inferred))
        if tavily_key
        else None
    )
    pending: set[asyncio.Task] = {tier_one_task}
    if tavily_task is not None:
        pending.add(tavily_task)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            # If both completed in the same event-loop turn, retain Tavily's
            # configured research-provider priority.
            if tavily_task is not None and tavily_task in done:
                tavily_result = await tavily_task
                if tavily_result and tavily_result.get("items"):
                    tavily_result["items"] = _finalize_search_items(
                        tavily_result["items"],
                        count=count,
                    )
                    _cache_put_search(cache_key, tavily_result)
                    return tavily_result
                errors.append("tavily: no result")

            if tier_one_task in done:
                tier_result, tier_errors = await tier_one_task
                errors.extend(tier_errors)
                if tier_result is not None:
                    _cache_put_search(cache_key, tier_result)
                    return tier_result
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    if not serper_key:
        errors.append("serper: no API key configured")

    # ── All tiers failed ──
    return {
        "ok": False,
        "error": "all_search_providers_failed",
        "query": normalized_query,
        "tried": errors,
    }


# Runtime info is patched in by Runtime.build() via bind_status_info().  A
# callable keeps mutable selections and provider hot-swaps truthful without a
# background refresh task.
_STATUS_INFO: dict[str, Any] | Callable[[], dict[str, Any]] = {}


def bind_status_info(info: dict[str, Any] | Callable[[], dict[str, Any]]) -> None:
    """Bind live runtime info for the status tool. Called by Runtime.build()."""
    global _STATUS_INFO
    _STATUS_INFO = info


async def t_status() -> dict:
    """Return runtime status including model, uptime, memory stats."""
    import time

    source = _STATUS_INFO() if callable(_STATUS_INFO) else _STATUS_INFO
    info = dict(source)
    info.setdefault("runtime", "norax")
    info.setdefault("phase", 4)
    info["ok"] = True
    info["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return info


# --- T1 tools --------------------------------------------------------------


def _invalidate_read_cache(p: Path) -> None:
    """Drop any cached reads for this path (e.g. after write/edit)."""
    key_prefix = str(p)
    for k in list(_READ_CACHE.keys()):
        if k[0] == key_prefix:
            _READ_CACHE.pop(k, None)


async def t_write(*, path: str, content: str) -> dict:
    p = Path(path).expanduser()
    await asyncio.to_thread(_write_user_text, p, content)
    _invalidate_read_cache(p)
    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return {"ok": True, "path": str(p), "bytes": len(content.encode("utf-8")), "lines": lines}


def _write_user_text(path: Path, content: str) -> None:
    """Atomically replace a user file while preserving its permission bits.

    Resolve symlinks to preserve the tool's existing follow-target behavior.
    RiskGate checks the resolved target before dispatch.
    """
    target = path.resolve()
    mode = stat_mod.S_IMODE(target.stat().st_mode) if target.exists() else 0o600
    atomic_write_text(target, content, mode=mode)


async def t_write_chunk(
    *,
    path: str,
    content: str,
    mode: str = "append",
    final: bool = False,
) -> dict:
    """Write a chunk to a file — use for large documents.

    Designed for LLMs that can't emit an entire large file in one tool
    call. Usage pattern:
      1. First call: `mode="start"` — truncates/creates the file, writes
         the first chunk. Prefer coherent chunks around 8-16KB when the
         provider can emit them.
      2. Subsequent calls: `mode="append"` — appends the next chunk.
      3. Last call: set `final=True` to confirm the document is
         complete (returns byte total for verification).

    Chunks should be natural segments (paragraphs / sections), not
    mid-sentence splits. The tool never imposes a size cap — only the
    LLM's own generation window limits per-chunk size.
    """
    if mode not in ("start", "append"):
        return {"ok": False, "error": f"invalid mode: {mode} (use start|append)"}
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if mode == "start":
        p.write_text(content, encoding="utf-8")
    else:
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
    _invalidate_read_cache(p)
    total = p.stat().st_size
    return {
        "ok": True,
        "path": str(p),
        "chunk_bytes": len(content),
        "total_bytes": total,
        "final": final,
    }


async def t_edit(*, path: str, old: str, new: str) -> dict:
    def edit_locked() -> dict:
        with path_lock(Path(path).expanduser().resolve()):
            return _edit_user_text(path=path, old=old, new=new)

    return await asyncio.to_thread(edit_locked)


def _edit_user_text(*, path: str, old: str, new: str) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        return {"ok": False, "error": "file_not_found", "path": str(p)}
    text = p.read_text(encoding="utf-8")
    if old not in text:
        # Provide context to help the model fix its next attempt
        # Find closest match via simple substring search
        old_stripped = old.strip()
        hint = ""
        if old_stripped and len(old_stripped) > 10:
            # Show a snippet from the file around where the match might be
            old_words = old_stripped.split()[:3]
            search = " ".join(old_words)
            idx = text.lower().find(search.lower())
            if idx >= 0:
                start = max(0, idx - 20)
                end = min(len(text), idx + len(old_stripped) + 20)
                hint = text[start:end].replace("\n", "\\n")
        total_lines = len(text.splitlines())
        return {
            "ok": False,
            "error": "old_text_not_found",
            "path": str(p),
            "total_lines": total_lines,
            "file_bytes": len(text),
            "hint": hint[:300]
            if hint
            else "Use `read` to see the current file content before retrying",
            "note": "The `old` text must EXACTLY match the file content including whitespace and newlines.",
        }
    new_text = text.replace(old, new, 1)
    _write_user_text(p, new_text)
    _invalidate_read_cache(p)
    return {
        "ok": True,
        "path": str(p),
        "diff_bytes": len(new.encode("utf-8")) - len(old.encode("utf-8")),
    }


async def t_append_memory(*, path: str, text: str) -> dict:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    return {"ok": True, "path": str(p)}


async def t_message_send(
    *,
    channel: str,
    target: str,
    text: str,
    files: list[str] | None = None,
) -> dict:
    # This function is replaced by ``bind_outbound`` during runtime wire-up.
    # If it is reached directly, no transport accepted the message: report that
    # fact rather than manufacturing a successful/queued result.
    log.warning(
        "message_send.unbound channel=%s target=%s text=%r files=%r",
        channel,
        target,
        text[:60],
        files,
    )
    return {
        "ok": False,
        "error": "outbound_not_bound",
        "channel": channel,
        "target": target,
        "queued": False,
    }


def make_message_send(outbound):
    """Factory that returns a `message_send` coroutine bound to an
    OutboundRegistry. Used by Runtime.build() to replace the stub with a
    live implementation.
    """

    async def _message_send(
        *,
        channel: str,
        target: str,
        text: str,
        reply_to: str | None = None,
        files: list[str] | None = None,
    ) -> dict:
        if outbound is None or not outbound.has(channel):
            return {
                "ok": False,
                "error": "channel_not_registered",
                "channel": channel,
                "target": target,
            }
        return await outbound.send(channel, target, text, reply_to=reply_to, files=files)

    return _message_send


def bind_outbound(outbound) -> None:
    """Replace the global `message_send` tool with a live, outbound-bound
    implementation. Safe to call multiple times; last-write wins.
    """
    REGISTRY["message_send"] = ToolSpec(
        "message_send",
        "Send a message to a channel target (live).",
        {"channel": "str", "target": "str", "text": "str", "reply_to": "str?", "files": "list?"},
        make_message_send(outbound),
    )


_REMINDER_SCHEDULER = None


def bind_reminder_scheduler(scheduler) -> None:
    """Bind the persistent runtime reminder scheduler."""
    global _REMINDER_SCHEDULER
    _REMINDER_SCHEDULER = scheduler


# --- OpenAI / provider function-tool schema rendering ---------------------

_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def _schema_for(spec: ToolSpec) -> dict:
    """Render a ToolSpec's mini-schema dict to a full JSON-schema
    compatible with OpenAI / Anthropic / OpenAI-compatible function tools.

    Our internal schema is a flat `{arg_name: "type"}` with optional `?`
    suffix for "not required". Everything else becomes an OBJECT schema.
    """
    props: dict[str, dict] = {}
    required: list[str] = []
    for name, typ in spec.schema.items():
        optional = typ.endswith("?")
        t = typ.rstrip("?")
        js_type = _TYPE_MAP.get(t, "string")
        prop: dict = {"type": js_type}
        # OpenAI strictly requires `items` on array schemas.
        if js_type == "array":
            prop["items"] = {"type": "string"}
        props[name] = prop
        if not optional:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def render_tools_for_llm(names: list[str], *, mcp_tools: list[dict] | None = None) -> list[dict]:
    """Return OpenAI-format tools array for the named specs.

    Unknown names are silently dropped. Order is preserved.
    MCP tools (prefixed with "mcp_") are rendered from the mcp_tools list
    if provided; each entry has name, description, and inputSchema.
    """
    out: list[dict] = []
    mcp_map: dict[str, dict] = {}
    if mcp_tools:
        for mt in mcp_tools:
            mcp_map[f"mcp_{mt['name']}"] = mt
    for n in names:
        if n.startswith("mcp_") and n in mcp_map:
            mt = mcp_map[n]
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": n,
                        "description": f"[MCP:{mt.get('connection', '')}] {mt.get('description', '')}",
                        "parameters": mt.get("inputSchema") or {"type": "object", "properties": {}},
                    },
                }
            )
            continue
        spec = REGISTRY.get(n)
        if spec is not None:
            out.append(_schema_for(spec))
    return out


async def t_schedule_reminder(
    *,
    when_iso: str,
    text: str,
    channel: str | None = None,
    target: str | None = None,
) -> dict:
    """Schedule a durable reminder through the live outbound adapter."""
    if _REMINDER_SCHEDULER is None:
        return {"ok": False, "error": "reminder_scheduler_not_initialized"}
    return await _REMINDER_SCHEDULER.schedule(
        when_iso=when_iso,
        text=text,
        channel=channel,
        target=target,
    )


# --- T2 tools --------------------------------------------------------------


async def t_exec(*, command: str, cwd: str | None = None, timeout: float = 30.0) -> dict:  # noqa: ASYNC109
    """Run a one-shot shell command via bash (no session state)."""
    coerced_timeout = _coerce_float(timeout, default=30.0)
    timeout = coerced_timeout if coerced_timeout is not None else 30.0
    return await run_command(command, cwd=cwd, timeout=timeout)


async def t_shell(
    *,
    command: str,
    session_id: str = "default",
    cwd: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Run a command in a stateful shell session (cwd persists across calls)."""
    coerced_timeout = _coerce_float(timeout, default=30.0)
    timeout = coerced_timeout if coerced_timeout is not None else 30.0
    session = get_shell_manager().get(session_id)
    if cwd:
        session.reset_cwd(cwd)
    return await session.run(command, timeout=timeout)


async def t_gateway_config_patch(*, patch: dict) -> dict:
    """Validate and atomically patch the active gateway configuration."""
    import json as _json
    import os as _os

    from ..atomic import atomic_write_text, path_lock
    from ..config.loader import DEFAULT_CONFIG, load_jsonc
    from ..config.loader import Config as RuntimeConfig

    if not isinstance(patch, dict) or not patch:
        return {"ok": False, "error": "patch_must_be_nonempty_object"}
    if set(patch) != {"gateway"} or not isinstance(patch.get("gateway"), dict):
        return {
            "ok": False,
            "error": "patch_must_contain_only_a_gateway_object",
        }
    try:
        patch_size = len(_json.dumps(patch, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError, OverflowError) as error:
        return {"ok": False, "error": f"patch_is_not_json_serializable: {error}"}
    if patch_size > 262_144:
        return {"ok": False, "error": "patch_too_large", "max_bytes": 262_144}

    cfg_path = Path(_os.environ.get("NORAX_CONFIG") or DEFAULT_CONFIG).expanduser().resolve()
    if not cfg_path.is_file():
        return {
            "ok": False,
            "error": "config_file_not_found",
            "searched": [str(cfg_path)],
        }

    # Deep-merge the patch into the config
    def _deep_merge(base: dict, overlay: dict, *, depth: int = 0) -> None:
        if depth > 8:
            raise ValueError("patch nesting exceeds 8 levels")
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _deep_merge(base[key], value, depth=depth + 1)
            else:
                base[key] = value

    def _apply_patch() -> dict:
        # Serialize the complete read/validate/write transaction so concurrent
        # tool calls cannot overwrite each other's updates.
        with path_lock(cfg_path):
            try:
                if cfg_path.stat().st_size > 2_097_152:
                    return {"ok": False, "error": "config_too_large", "max_bytes": 2_097_152}
                config = load_jsonc(cfg_path)
            except Exception as e:
                return {"ok": False, "error": f"config_parse_failed: {e}"}
            try:
                _deep_merge(config, patch)
                # Validate the exact schema Runtime.build consumes before
                # replacement. Misspelled/dead settings cannot report success.
                _ = RuntimeConfig(raw=config, project_root=cfg_path.parent).gateway
                rendered = _json.dumps(config, indent=2) + "\n"
                if len(rendered.encode("utf-8")) > 2_097_152:
                    return {
                        "ok": False,
                        "error": "resulting_config_too_large",
                        "max_bytes": 2_097_152,
                    }
                current_mode = stat_mod.S_IMODE(cfg_path.stat().st_mode)
                atomic_write_text(
                    cfg_path,
                    rendered,
                    durable=True,
                    mode=current_mode,
                )
            except ValueError as e:
                return {"ok": False, "error": f"config_validation_failed: {e}"}
            except Exception as e:
                return {"ok": False, "error": f"config_write_failed: {e}"}

        gateway_keys = sorted(patch["gateway"])
        log.info("gateway_config_patch applied: keys=%s", gateway_keys)
        return {
            "ok": True,
            "patched_keys": [f"gateway.{key}" for key in gateway_keys],
            "config_path": str(cfg_path),
            "restart_required": True,
        }

    return await asyncio.to_thread(_apply_patch)


# --- computer_use ----------------------------------------------------------


async def t_computer_use(
    *,
    action: str,
    x: int | None = None,
    y: int | None = None,
    x2: int | None = None,
    y2: int | None = None,
    button: str = "left",
    text: str | None = None,
    keys: str | None = None,
    path: str | None = None,
    scroll: int | None = None,
    delay: int = 12,
) -> dict:
    """Desktop automation via ydotool/xdotool + grim/scrot.

    Actions: screenshot, click, doubleclick, move, drag, scroll, type_text,
    key_press, clipboard_get, clipboard_set, info.

    Wayland-native (ydotool, grim) with X11 fallback (xdotool, scrot).
    Requires an explicitly configured or unambiguous desktop session.
    """
    source_tools = Path(__file__).resolve().parent.parent.parent / "tools"
    packaged_tools = Path(__file__).resolve().parent.parent / "_tools"
    packaged_backend = packaged_tools / "computer_use.py"
    source_backend = source_tools / "computer_use.py"
    # Installed wheels must not be shadowed by an unrelated top-level ``tools``
    # directory. A source checkout falls back only when no packaged backend is
    # present.
    backend = packaged_backend if packaged_backend.is_file() else source_backend
    if not backend.is_file():
        return {
            "ok": False,
            "error": "computer_use backend is not installed",
            "_not_executed": True,
        }

    # Invoke the synchronous desktop backend in a child process.  Forking its
    # commands from asyncio's worker threads can deadlock on some hosts, and a
    # cancelled ``to_thread`` call cannot stop the underlying operation.  This
    # boundary is independently timed, killable, and works from both a source
    # checkout and the packaged wheel.
    _btn_map = {"left": 1, "right": 2, "middle": 3, "1": 1, "2": 2, "3": 3}
    normalized_button = str(button or "left").lower()
    if normalized_button not in _btn_map:
        return {"ok": False, "error": f"unsupported mouse button: {button}"}
    cli_args: list[str]
    timeout = 15.0
    if action == "screenshot":
        cli_args = ["screenshot"]
        if path is not None:
            cli_args.extend(["--output", path])
        timeout = 25.0
    elif action == "click":
        if x is None or y is None:
            return {"ok": False, "error": "click requires x and y"}
        cli_args = ["click", str(x), str(y), "--button", str(_btn_map[normalized_button])]
    elif action == "doubleclick":
        if x is None or y is None:
            return {"ok": False, "error": "doubleclick requires x and y"}
        cli_args = ["doubleclick", str(x), str(y)]
    elif action == "move":
        if x is None or y is None:
            return {"ok": False, "error": "move requires x and y"}
        cli_args = ["move", str(x), str(y)]
    elif action == "drag":
        if None in (x, y, x2, y2):
            return {"ok": False, "error": "drag requires x, y, x2, and y2"}
        assert x is not None and y is not None and x2 is not None and y2 is not None
        cli_args = ["drag", str(x), str(y), str(x2), str(y2)]
        timeout = 25.0
    elif action == "scroll":
        cli_args = ["scroll", str(scroll if scroll is not None else 3)]
    elif action == "type_text":
        if text is None:
            return {"ok": False, "error": "type_text requires text"}
        if not 0 <= delay <= 1000:
            return {"ok": False, "error": "delay must be between 0 and 1000 milliseconds"}
        if len(text) > 16_384:
            return {
                "ok": False,
                "error": "text exceeds the 16384-character desktop input limit",
            }
        cli_args = ["type", text, "--delay", str(delay)]
        timeout = min(40.0, max(10.0, len(text) * delay / 1000 + 10.0))
    elif action == "key_press":
        if not keys:
            return {"ok": False, "error": "key_press requires keys"}
        cli_args = ["key", keys]
    elif action == "clipboard_get":
        cli_args = ["clipboard-get"]
    elif action == "clipboard_set":
        if text is None:
            return {"ok": False, "error": "clipboard_set requires text"}
        cli_args = ["clipboard-set", text]
    elif action == "info":
        cli_args = ["info"]
        timeout = 3.0
    else:
        return {"ok": False, "error": f"unknown action: {action}"}

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(backend),
            *cli_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            return {
                "ok": False,
                "error": f"computer_use {action} timed out after {timeout:g}s",
                "outcome_unknown": True,
            }
        except asyncio.CancelledError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            raise
    except OSError as exc:
        return {
            "ok": False,
            "error": f"computer_use backend could not start: {exc}",
            "_not_executed": True,
        }

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    try:
        result = json.loads(stdout_text)
    except json.JSONDecodeError:
        detail = (
            stderr_text or stdout_text.strip() or f"backend exited with status {proc.returncode}"
        )
        return {
            "ok": False,
            "error": f"computer_use backend returned invalid output: {detail[:500]}",
        }
    if not isinstance(result, dict):
        return {"ok": False, "error": "computer_use backend returned a non-object result"}
    if proc.returncode != 0 or result.get("ok") is not True:
        result["ok"] = False
        if not result.get("error"):
            result["error"] = stderr_text or (
                f"backend exited with {proc.returncode}"
                if proc.returncode != 0
                else "backend reported failure without an error"
            )
        return result
    return result


# --- registry --------------------------------------------------------------

REGISTRY: dict[str, ToolSpec] = {
    "repo_explore": ToolSpec(
        "repo_explore",
        "Explore a repository for files relevant to a query. Returns file:line citations. Uses Microsoft FastContext if available; falls back to deterministic local search.",
        {"query": "str", "root": "str?", "budget": "int?"},
        t_repo_explore,
    ),
    "read": ToolSpec(
        "read",
        "Read a text file. Relative paths start in the active workspace; do not search the whole filesystem for them. Returns content + total_lines. Use offset+limit for files >200KB. Set large=true only when a larger contiguous chunk is needed; its result gets a bounded 48KB reasoning budget and is still compacted before entering the rolling window. ALWAYS read before editing.",
        {"path": "str", "limit": "int?", "offset": "int?", "large": "bool?"},
        t_read,
    ),
    "list_dir": ToolSpec(
        "list_dir",
        "List a directory with deterministic bounded pagination. Defaults to the current directory and returns at most 500 entries; use offset/limit to continue.",
        {"path": "str?", "offset": "int?", "limit": "int?"},
        t_list_dir,
    ),
    "web_fetch": ToolSpec(
        "web_fetch",
        "Fetch a public HTTP(S) URL and return bounded text content. Strips HTML tags automatically; uses trafilatura for article pages. "
        "If the direct fetch is bot-blocked, times out, or returns a JS-rendered shell, it transparently retries through the Firecrawl scrape API (JS rendering). "
        "Default 24K chars; private-network targets require an explicit operator opt-in and are never proxied externally.",
        {"url": "str", "max_chars": "int?"},
        t_web_fetch,
    ),
    "web_search": ToolSpec(
        "web_search",
        "Search the web. Returns titles, URLs, snippets. Backends: Tavily (optional) → SearXNG+Serper parallel RRF merge → Ollama → Serper. "
        "Compare/versus/and queries fan out into parallel sub-searches. Results are deduped and domain-diversified. "
        "Optional recency_days narrows to last N days; auto-inferred for 'latest', 'today', '2026', etc. "
        "Identical queries within 5 minutes hit cache. Default 5 results, max 10.",
        {"query": "str", "count": "int?", "recency_days": "int?"},
        t_web_search,
    ),
    "deep_research": ToolSpec(
        "deep_research",
        "Endless, resumable deep research. Runs a batch of parallel web searches, fetches+extracts the best NEW sources, and appends findings to a durable markdown report with a JSON state file. "
        "Call repeatedly with the same `topic` to go deeper — state persists across calls so it never re-reads a source. "
        "Pass `queries` to steer; omit them to auto-generate the next query set from the topic and open tensions. "
        "Pass `seeds` (start URLs) to crawl a specific site: each seed is fetched, its same-site links are discovered via Firecrawl map, and the best pages are fetched; "
        "set `exhaustive` true to instead run a full Firecrawl crawl from each seed (slower, covers the whole site). "
        "Returns a bounded digest + report_path/state_path; read the report for full findings. "
        "Stop when `saturated` is true or you have enough to synthesize.",
        {
            "topic": "str",
            "queries": "list?",
            "seeds": "list?",
            "exhaustive": "bool?",
            "rounds": "int?",
            "max_sources": "int?",
            "max_chars_per_source": "int?",
            "out_dir": "str?",
        },
        t_deep_research,
    ),
    "search_memory": ToolSpec(
        "search_memory",
        "Search semantic/procedural/intel memory for stored facts, workflows, or knowledge. Returns top-k matches with relevance scores.",
        {"query": "str", "k": "int?"},
        t_search_memory,
    ),
    "status": ToolSpec("status", "Return runtime status.", {}, t_status),
    "write": ToolSpec(
        "write",
        "Write/overwrite a complete text file in one call when it fits the model output. Relative paths start in the active workspace. Creates parents. Use `write_chunk` only when the content cannot fit reliably in one tool call.",
        {"path": "str", "content": "str"},
        t_write,
    ),
    "write_chunk": ToolSpec(
        "write_chunk",
        "Write a large file in chunks when one complete write cannot fit. mode='start' truncates+writes the first chunk; mode='append' adds more. Prefer coherent 8-16KB chunks and set final=true on the last call.",
        {"path": "str", "content": "str", "mode": "str?", "final": "bool?"},
        t_write_chunk,
    ),
    "edit": ToolSpec(
        "edit",
        "Replace first occurrence of `old` with `new` in file. IMPORTANT: `old` must EXACTLY match the file content (including whitespace/newlines). Use `read` first to get the exact text.",
        {"path": "str", "old": "str", "new": "str"},
        t_edit,
    ),
    "append_memory": ToolSpec(
        "append_memory",
        "Append a line to a memory file.",
        {"path": "str", "text": "str"},
        t_append_memory,
    ),
    "message_send": ToolSpec(
        "message_send",
        "Send a message to a channel target. Optional file attachments via `files`.",
        {"channel": "str", "target": "str", "text": "str", "files": "list?"},
        t_message_send,
    ),
    "schedule_reminder": ToolSpec(
        "schedule_reminder",
        "Schedule a durable reminder at an ISO-8601 time. Defaults to the owner's Discord DM; optionally specify channel and target.",
        {"when_iso": "str", "text": "str", "channel": "str?", "target": "str?"},
        t_schedule_reminder,
    ),
    "exec": ToolSpec(
        "exec",
        "Run a one-shot shell command (bash) in the active workspace by default; use relative paths there and do not cd to ~ to find task files. No cwd persistence. Returns stdout/stderr/exit_code. Timeout: 1-600s (default 30s); background longer jobs and poll their output. PATH includes ~/.local/bin. T2 — dangerous patterns blocked by RiskGate.",
        {"command": "str", "cwd": "str?", "timeout": "float?"},
        t_exec,
    ),
    "shell": ToolSpec(
        "shell",
        "Run a command in a stateful shell session (cwd persists). Use for multi-step shell work; use exec for one-shot runs. Returns stdout/stderr/exit_code/cwd. Timeout: 1-600s; background longer jobs and poll their output. T2.",
        {"command": "str", "session_id": "str?", "cwd": "str?", "timeout": "float?"},
        t_shell,
    ),
    "gateway_config_patch": ToolSpec(
        "gateway_config_patch",
        "Atomically deep-patch the active runtime config (T2). A service restart is required to activate it.",
        {"patch": "dict"},
        t_gateway_config_patch,
    ),
    "remote_enroll": ToolSpec(
        "remote_enroll",
        "Create an enrollment token for a user-owned remote computer/node.",
        {"name": "str", "root": "str?"},
        t_remote_enroll,
    ),
    "remote_list_nodes": ToolSpec(
        "remote_list_nodes",
        "List enrolled remote computers/nodes and capabilities.",
        {},
        t_remote_list_nodes,
    ),
    "remote_exec": ToolSpec(
        "remote_exec",
        "Run a shell command on an enrolled remote node.",
        {"node_id": "str", "command": "str", "cwd": "str?", "timeout": "float?"},
        t_remote_exec,
    ),
    "remote_read": ToolSpec(
        "remote_read",
        "Read a file from an enrolled remote node.",
        {"node_id": "str", "path": "str", "limit": "int?", "offset": "int?"},
        t_remote_read,
    ),
    "remote_list": ToolSpec(
        "remote_list",
        "List a directory on an enrolled remote node.",
        {"node_id": "str", "path": "str"},
        t_remote_list,
    ),
    "remote_write": ToolSpec(
        "remote_write",
        "Write a file on an enrolled remote node.",
        {"node_id": "str", "path": "str", "content": "str"},
        t_remote_write,
    ),
    # Alias — some code references "memory_search" instead of "search_memory"
    "memory_search": ToolSpec(
        "memory_search",
        "Search semantic/procedural/intel memory (alias for search_memory).",
        {"query": "str", "k": "int?"},
        t_search_memory,
    ),
    "browser": ToolSpec(
        "browser",
        "Bounded Playwright browser automation. Actions: navigate, click, type, extract, screenshot, fill_form, scroll, wait, evaluate (JS), pdf, tabs_list, close. Up to 16 persistent sessions; optional operator-enabled stealth. Mutation results distinguish dispatch from observed readback.",
        {
            "action": "str",
            "url": "str?",
            "selector": "str?",
            "text": "str?",
            "fields": "dict?",
            "attribute": "str?",
            "session_id": "str?",
            "timeout": "float?",
            "wait_for": "str?",
            "wait_seconds": "float?",
            "extract_type": "str?",
            "scroll_amount": "int?",
            "js": "str?",
            "submit": "bool?",
        },
        t_browser,
    ),
    "sandbox_exec": ToolSpec(
        "sandbox_exec",
        "Execute a command in a sandboxed Docker/Podman container with resource limits, filesystem isolation, and optional network isolation. Safer than exec for untrusted code.",
        {
            "command": "str",
            "image": "str?",
            "mounts": "list?",
            "network": "bool?",
            "timeout": "int?",
        },
        t_sandbox_exec,
    ),
    "computer_use": ToolSpec(
        "computer_use",
        "Desktop automation via ydotool/xdotool + grim/scrot. Actions: screenshot, click, doubleclick, move, drag, scroll, type_text, key_press, clipboard_get, clipboard_set, info. Wayland-native with X11 fallback.",
        {
            "action": "str",
            "x": "int?",
            "y": "int?",
            "x2": "int?",
            "y2": "int?",
            "button": "str?",
            "text": "str?",
            "keys": "str?",
            "path": "str?",
            "scroll": "int?",
            "delay": "int?",
        },
        t_computer_use,
    ),
}
