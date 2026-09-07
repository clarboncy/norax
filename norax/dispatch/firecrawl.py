"""Bounded Firecrawl Map and Crawl protocol integration.

This module owns the asynchronous Firecrawl v2 job lifecycle. It deliberately
reuses the dispatch web-safety and bounded-response primitives so external
crawls cannot bypass the same URL and memory limits as direct web fetches.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from ..safety.secrets import redact
from .input_coercion import _coerce_int
from .web_common import _normalize_search_url, _read_web_body, _web_fetch_url_error

_FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"
_FIRECRAWL_API_URL = f"{_FIRECRAWL_BASE_URL}/scrape"
_FIRECRAWL_TIMEOUT = 20.0
_FIRECRAWL_RESPONSE_MAX_BYTES = 2 * 1024 * 1024
_FIRECRAWL_MIN_DIRECT_CHARS = 300
_FIRECRAWL_FALLBACK_ERRORS = frozenset(
    {
        "http_status",
        "timeout",
        "connection_failed",
        "request_error",
        "unsupported_content_type",
    }
)
_FIRECRAWL_MAP_MAX_LINKS = 50
_FIRECRAWL_CRAWL_PAGE_MAX = 40
_FIRECRAWL_CRAWL_DEFAULT_TIMEOUT = 180.0
_FIRECRAWL_POLL_INTERVAL = 3.0
_FIRECRAWL_CRAWL_MIN_TIMEOUT = 40.0
_FIRECRAWL_CRAWL_MAX_TIMEOUT = 270.0
_FIRECRAWL_PAGE_TEXT_MAX_CHARS = 24_000


def _firecrawl_error_detail(value: object) -> str:
    """Bound and scrub diagnostics originating outside the runtime."""
    return str(redact(str(value)))[:500]


def _firecrawl_api_key() -> str:
    return os.environ.get("FIRECRAWL_API_KEY", "").strip()


async def _firecrawl_scrape(url: str, max_chars: int) -> dict:
    """Fetch one URL through the bounded Firecrawl v2 scrape endpoint."""
    api_key = _firecrawl_api_key()
    if not api_key:
        return {"ok": False, "error": "firecrawl_not_configured"}
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_FIRECRAWL_TIMEOUT, connect=8.0),
        ) as client:
            async with client.stream(
                "POST",
                _FIRECRAWL_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                declared_length = response.headers.get("content-length")
                if declared_length:
                    try:
                        if int(declared_length) > _FIRECRAWL_RESPONSE_MAX_BYTES:
                            return {
                                "ok": False,
                                "error": "firecrawl_response_too_large",
                                "status": response.status_code,
                            }
                    except ValueError:
                        pass
                raw, truncated = await _read_web_body(
                    response,
                    _FIRECRAWL_RESPONSE_MAX_BYTES,
                )
                status_code = response.status_code
                is_success = response.is_success
    except httpx.TimeoutException:
        return {"ok": False, "error": "firecrawl_timeout"}
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "error": "firecrawl_request_error",
            "detail": _firecrawl_error_detail(exc),
        }
    if truncated:
        return {
            "ok": False,
            "error": "firecrawl_response_too_large",
            "status": status_code,
        }
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return {"ok": False, "error": "firecrawl_bad_response", "status": status_code}
    if not is_success or not isinstance(data, dict) or data.get("success") is not True:
        detail = data.get("error") if isinstance(data, dict) else None
        return {
            "ok": False,
            "error": "firecrawl_failed",
            "status": status_code,
            "detail": _firecrawl_error_detail(detail) if detail else None,
        }
    body = data.get("data") or {}
    if not isinstance(body, dict):
        return {"ok": False, "error": "firecrawl_bad_response", "status": status_code}
    markdown = body.get("markdown")
    if not isinstance(markdown, str):
        return {"ok": False, "error": "firecrawl_bad_response", "status": status_code}
    markdown = markdown.strip()
    if not markdown:
        return {"ok": False, "error": "firecrawl_empty"}
    metadata = body.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    title = str(metadata.get("title") or "").strip()
    source_url = str(metadata.get("sourceURL") or metadata.get("url") or url)
    out: dict[str, Any] = {
        "ok": True,
        "markdown": markdown[:max_chars],
        "truncated": len(markdown) > max_chars,
        "source_url": source_url,
    }
    if title:
        out["title"] = title
    return out


async def _firecrawl_pause(seconds: float) -> None:
    """Isolated poll wait so connector tests never patch global asyncio state."""
    await asyncio.sleep(seconds)


async def _firecrawl_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    api_key: str,
    payload: dict | None = None,
    require_success_field: bool,
) -> tuple[dict | None, dict | None]:
    """Perform one bounded Firecrawl request using an existing client."""
    if (
        not path
        or len(path) > 512
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_/" for char in path)
    ):
        return None, {"ok": False, "error": "firecrawl_invalid_api_path"}
    request_kwargs: dict[str, Any] = {
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
    }
    if payload is not None:
        request_kwargs["json"] = payload
        request_kwargs["headers"]["Content-Type"] = "application/json"
    try:
        async with client.stream(
            method,
            f"{_FIRECRAWL_BASE_URL}/{path}",
            **request_kwargs,
        ) as response:
            declared_length = response.headers.get("content-length")
            if declared_length:
                try:
                    if int(declared_length) > _FIRECRAWL_RESPONSE_MAX_BYTES:
                        return None, {
                            "ok": False,
                            "error": "firecrawl_response_too_large",
                            "status": response.status_code,
                        }
                except ValueError:
                    pass
            raw, truncated = await _read_web_body(response, _FIRECRAWL_RESPONSE_MAX_BYTES)
            status_code = response.status_code
            is_success = response.is_success
    except httpx.TimeoutException:
        return None, {"ok": False, "error": "firecrawl_timeout"}
    except httpx.RequestError as exc:
        return None, {
            "ok": False,
            "error": "firecrawl_request_error",
            "detail": _firecrawl_error_detail(exc),
        }
    if truncated:
        return None, {
            "ok": False,
            "error": "firecrawl_response_too_large",
            "status": status_code,
        }
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return None, {
            "ok": False,
            "error": "firecrawl_bad_response",
            "status": status_code,
        }
    if not is_success or not isinstance(data, dict):
        detail = data.get("error") if isinstance(data, dict) else None
        return None, {
            "ok": False,
            "error": "firecrawl_failed",
            "status": status_code,
            "detail": _firecrawl_error_detail(detail) if detail else None,
        }
    if require_success_field and data.get("success") is not True:
        return None, {
            "ok": False,
            "error": "firecrawl_failed",
            "status": status_code,
            "detail": _firecrawl_error_detail(data.get("error")) if data.get("error") else None,
        }
    return data, None


async def _firecrawl_post(path: str, payload: dict) -> tuple[dict | None, dict | None]:
    """Shared POST helper for Firecrawl API endpoints.

    Returns ``(data, error)``: exactly one is non-None.  Mirrors the error
    vocabulary of :func:`_firecrawl_scrape` so callers can surface failures
    without special-casing.
    """
    api_key = _firecrawl_api_key()
    if not api_key:
        return None, {"ok": False, "error": "firecrawl_not_configured"}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(_FIRECRAWL_TIMEOUT, connect=8.0),
    ) as client:
        return await _firecrawl_request(
            client,
            "POST",
            path,
            api_key=api_key,
            payload=payload,
            require_success_field=True,
        )


async def _firecrawl_map(url: str, limit: int = 50) -> dict:
    """List same-site links from a start URL via the Firecrawl map endpoint.

    Cheap discovery primitive: no full-page extraction, just the link graph
    visible from ``url``.  Used by deep_research to seed site-wide crawls.
    Returns ``{"ok": True, "links": [...]}`` or an error dict; never raises.
    """
    clean_url = str(url or "").strip()
    if not clean_url:
        return {"ok": False, "error": "firecrawl_map_url_required"}
    parsed_limit = _coerce_int(limit)
    if parsed_limit is None:
        return {"ok": False, "error": "firecrawl_map_limit_invalid"}
    if not _firecrawl_api_key():
        return {"ok": False, "error": "firecrawl_not_configured"}
    unsafe_reason = await _web_fetch_url_error(clean_url, honor_private_opt_in=False)
    if unsafe_reason is not None:
        return {
            "ok": False,
            "error": "url_not_allowed",
            "detail": unsafe_reason,
        }
    link_limit = max(1, min(_FIRECRAWL_MAP_MAX_LINKS, parsed_limit))
    payload = {
        "url": clean_url,
        "limit": link_limit,
        "search": "",
        "sitemap": "include",
        "includeSubdomains": False,
        "ignoreQueryParameters": True,
    }
    data, error = await _firecrawl_post("map", payload)
    if error is not None:
        return error
    assert data is not None
    raw_links = data.get("links")
    if not isinstance(raw_links, list):
        raw_links = []
    links: list[str] = []
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in raw_links:
        if isinstance(link, dict):
            value = str(link.get("url") or link.get("sourceURL") or "").strip()
            title = str(link.get("title") or "").strip()[:500]
            description = str(link.get("description") or "").strip()[:2_000]
        else:
            value = str(link or "").strip()
            title = ""
            description = ""
        if len(value) > 4_096:
            continue
        try:
            parsed = urlsplit(value)
        except ValueError:
            continue
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            continue
        if parsed.username is not None or parsed.password is not None:
            continue
        key = _normalize_search_url(value)
        if not key or key in seen:
            continue
        seen.add(key)
        links.append(value)
        items.append({"url": value, "title": title, "description": description})
        if len(links) >= link_limit:
            break
    return {"ok": True, "links": links, "items": items}


def _firecrawl_crawl_pages(value: object, *, page_limit: int) -> list[dict[str, str]]:
    """Normalize and bound page documents returned by crawl-status calls."""
    if not isinstance(value, list):
        return []
    pages: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in value:
        if not isinstance(page, dict):
            continue
        metadata = page.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        page_url = str(
            metadata.get("sourceURL")
            or metadata.get("url")
            or page.get("url")
            or page.get("sourceURL")
            or ""
        ).strip()
        if not page_url or len(page_url) > 4_096:
            continue
        try:
            parsed = urlsplit(page_url)
        except ValueError:
            continue
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            continue
        if parsed.username is not None or parsed.password is not None:
            continue
        key = _normalize_search_url(page_url)
        if not key or key in seen:
            continue
        seen.add(key)
        markdown = page.get("markdown")
        pages.append(
            {
                "url": page_url,
                "title": str(metadata.get("title") or page.get("title") or "").strip()[:500],
                "text": (
                    markdown.strip()[:_FIRECRAWL_PAGE_TEXT_MAX_CHARS]
                    if isinstance(markdown, str)
                    else ""
                ),
            }
        )
        if len(pages) >= page_limit:
            break
    return pages


async def _cancel_firecrawl_crawl(
    client: httpx.AsyncClient,
    crawl_id: str,
    *,
    api_key: str,
) -> None:
    """Best-effort remote cleanup for a timed-out or cancelled crawl."""
    await _firecrawl_request(
        client,
        "DELETE",
        f"crawl/{quote(crawl_id, safe='')}",
        api_key=api_key,
        require_success_field=False,
    )


async def _firecrawl_crawl(
    url: str,
    *,
    limit: int = 30,
    timeout: float | None = None,
) -> dict:
    """Crawl a site from a root URL via the Firecrawl crawl endpoint.

    Submits a crawl job, polls until it completes or the deadline expires,
    and returns the crawled pages.  Returns
    ``{"ok": True, "crawl_id", "status", "pages": [{"url", "title"}]}`` or an
    error dict; never raises.  A deadline expiry reports ``crawl_incomplete``
    with whatever pages finished so far.
    """
    clean_url = str(url or "").strip()
    if not clean_url:
        return {"ok": False, "error": "firecrawl_crawl_url_required"}
    parsed_limit = _coerce_int(limit)
    if parsed_limit is None:
        return {"ok": False, "error": "firecrawl_crawl_limit_invalid"}
    page_limit = max(1, min(_FIRECRAWL_CRAWL_PAGE_MAX, parsed_limit))
    if timeout is None:
        crawl_timeout = _FIRECRAWL_CRAWL_DEFAULT_TIMEOUT
    elif isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return {"ok": False, "error": "firecrawl_crawl_timeout_invalid"}
    else:
        crawl_timeout = float(timeout)
        if not math.isfinite(crawl_timeout):
            return {"ok": False, "error": "firecrawl_crawl_timeout_invalid"}
    api_key = _firecrawl_api_key()
    if not api_key:
        return {"ok": False, "error": "firecrawl_not_configured"}
    unsafe_reason = await _web_fetch_url_error(clean_url, honor_private_opt_in=False)
    if unsafe_reason is not None:
        return {
            "ok": False,
            "error": "url_not_allowed",
            "detail": unsafe_reason,
        }
    crawl_timeout = max(
        _FIRECRAWL_CRAWL_MIN_TIMEOUT,
        min(_FIRECRAWL_CRAWL_MAX_TIMEOUT, crawl_timeout),
    )
    payload = {
        "url": clean_url,
        "limit": page_limit,
        "sitemap": "include",
        "crawlEntireDomain": True,
        "allowExternalLinks": False,
        "allowSubdomains": False,
        "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
    }
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(_FIRECRAWL_TIMEOUT, connect=8.0),
    ) as client:
        data, error = await _firecrawl_request(
            client,
            "POST",
            "crawl",
            api_key=api_key,
            payload=payload,
            require_success_field=True,
        )
        if error is not None:
            return error
        assert data is not None
        crawl_id = str(data.get("id") or "").strip()
        if not crawl_id or len(crawl_id) > 256:
            return {
                "ok": False,
                "error": "firecrawl_crawl_no_id",
                "detail": "crawl response has no valid job id",
            }
        deadline = time.monotonic() + crawl_timeout
        last_status = "scraping"
        pages: list[dict[str, str]] = []
        should_cancel = True
        try:
            while last_status not in {"completed", "failed", "cancelled", "stopped"}:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await _firecrawl_pause(min(_FIRECRAWL_POLL_INTERVAL, remaining))
                if time.monotonic() >= deadline:
                    break
                poll, poll_error = await _firecrawl_request(
                    client,
                    "GET",
                    f"crawl/{quote(crawl_id, safe='')}",
                    api_key=api_key,
                    require_success_field=False,
                )
                if poll_error is not None:
                    return {
                        **poll_error,
                        "error": "firecrawl_crawl_poll_failed",
                        "crawl_id": crawl_id,
                    }
                assert poll is not None
                last_status = str(poll.get("status") or "unknown").strip().lower()
                polled_pages = _firecrawl_crawl_pages(poll.get("data"), page_limit=page_limit)
                if polled_pages:
                    pages = polled_pages
            if last_status == "completed":
                should_cancel = False
                return {
                    "ok": True,
                    "crawl_id": crawl_id,
                    "status": "completed",
                    "pages": pages,
                }
            if last_status == "failed":
                should_cancel = False
                return {
                    "ok": False,
                    "error": "firecrawl_crawl_failed",
                    "crawl_id": crawl_id,
                    "status": "failed",
                }
            if last_status in {"cancelled", "stopped"}:
                should_cancel = False
            return {
                "ok": True,
                "crawl_id": crawl_id,
                "status": last_status if not should_cancel else "timed_out",
                "pages": pages,
                "note": "crawl did not fully complete; partial pages returned",
            }
        finally:
            if should_cancel:
                cleanup = asyncio.create_task(
                    _cancel_firecrawl_crawl(client, crawl_id, api_key=api_key),
                    name="cancel-firecrawl-crawl",
                )
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    # Preserve the caller's cancellation while giving the
                    # already-bounded remote cleanup a chance to finish before
                    # this client's context closes underneath it.
                    await asyncio.gather(cleanup, return_exceptions=True)
                except Exception:
                    # A failed cleanup cannot replace the original crawl result
                    # or cancellation, but the request remains strictly bounded.
                    pass
