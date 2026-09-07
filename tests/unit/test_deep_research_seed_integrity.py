from __future__ import annotations

import asyncio
import logging
import stat
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from norax.dispatch import deep_research as research
from norax.dispatch import tools


@pytest.mark.asyncio
async def test_firecrawl_request_accepts_documented_top_level_shapes_and_scrubs_errors() -> None:
    secret = "sk-" + "x" * 30

    async def success_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer private"
        return httpx.Response(200, json={"status": "completed", "data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(success_handler)) as client:
        data, error = await tools._firecrawl_request(
            client,
            "GET",
            "crawl/job-id",
            api_key="private",
            require_success_field=False,
        )
    assert error is None
    assert data == {"status": "completed", "data": []}

    async def failure_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": secret})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failure_handler)) as client:
        data, error = await tools._firecrawl_request(
            client,
            "POST",
            "map",
            api_key="private",
            payload={},
            require_success_field=True,
        )
    assert data is None
    assert error is not None
    assert error["error"] == "firecrawl_failed"
    assert secret not in str(error)
    assert "<REDACTED:openai_key>" in error["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(200, content=b"not-json"), "firecrawl_bad_response"),
        (httpx.Response(200, json=["not", "an", "object"]), "firecrawl_failed"),
        (httpx.Response(200, json={"success": False}), "firecrawl_failed"),
        (
            httpx.Response(200, content=b"{}", headers={"content-length": "999999999"}),
            "firecrawl_response_too_large",
        ),
    ],
)
async def test_firecrawl_request_rejects_malformed_or_oversized_responses(
    response: httpx.Response,
    expected: str,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: response)
    ) as client:
        data, error = await tools._firecrawl_request(
            client,
            "POST",
            "map",
            api_key="private",
            payload={},
            require_success_field=True,
        )
    assert data is None
    assert error is not None
    assert error["error"] == expected


@pytest.mark.asyncio
async def test_firecrawl_request_rejects_invalid_path_timeout_and_transport_error() -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid API path must be rejected before transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as client:
        _, invalid = await tools._firecrawl_request(
            client,
            "GET",
            "../secret",
            api_key="private",
            require_success_field=False,
        )
    assert invalid == {"ok": False, "error": "firecrawl_invalid_api_path"}

    for failure, expected in (
        (httpx.ReadTimeout("slow"), "firecrawl_timeout"),
        (httpx.ConnectError("offline"), "firecrawl_request_error"),
    ):

        async def handler(_request: httpx.Request, error: Exception = failure) -> httpx.Response:
            raise error

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            data, error = await tools._firecrawl_request(
                client,
                "GET",
                "crawl/job",
                api_key="private",
                require_success_field=False,
            )
        assert data is None
        assert error is not None
        assert error["error"] == expected


@pytest.mark.asyncio
async def test_firecrawl_request_bounds_streams_and_tolerates_bad_length_header() -> None:
    async def bad_length_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "links": []},
            headers={"content-length": "not-an-integer"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(bad_length_handler)) as client:
        data, error = await tools._firecrawl_request(
            client,
            "POST",
            "map",
            api_key="private",
            payload={},
            require_success_field=True,
        )
    assert error is None
    assert data == {"success": True, "links": []}

    class OversizedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"x" * tools._FIRECRAWL_RESPONSE_MAX_BYTES
            yield b"x"

    async def oversized_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=OversizedStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(oversized_handler)) as client:
        data, error = await tools._firecrawl_request(
            client,
            "GET",
            "crawl/job",
            api_key="private",
            require_success_field=False,
        )
    assert data is None
    assert error is not None
    assert error["error"] == "firecrawl_response_too_large"


@pytest.mark.asyncio
async def test_firecrawl_post_handles_configuration_and_reuses_request_helper(monkeypatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    data, error = await tools._firecrawl_post("map", {})
    assert data is None
    assert error == {"ok": False, "error": "firecrawl_not_configured"}

    calls: list[tuple[str, str, dict | None, bool]] = []

    async def fake_request(
        _client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        api_key: str,
        payload: dict | None = None,
        require_success_field: bool,
    ) -> tuple[dict, None]:
        assert api_key == "private"
        calls.append((method, path, payload, require_success_field))
        return {"success": True}, None

    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")
    monkeypatch.setattr(tools, "_firecrawl_request", fake_request)
    data, error = await tools._firecrawl_post("map", {"url": "https://docs.example"})
    assert data == {"success": True}
    assert error is None
    assert calls == [("POST", "map", {"url": "https://docs.example"}, True)]


@pytest.mark.asyncio
async def test_firecrawl_map_never_proxies_private_target_and_normalizes_v2_links(
    monkeypatch,
) -> None:
    called = False
    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")

    async def forbidden_post(*_args: Any, **_kwargs: Any) -> tuple[None, None]:
        nonlocal called
        called = True
        return None, None

    monkeypatch.setattr(tools, "_firecrawl_post", forbidden_post)
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result="private target"),
    )
    result = await tools._firecrawl_map("http://127.0.0.1/private")
    assert result["error"] == "url_not_allowed"
    assert called is False

    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    assert (await tools._firecrawl_map("https://docs.example", limit=True))["error"] == (
        "firecrawl_map_limit_invalid"
    )

    async def fake_post(path: str, payload: dict) -> tuple[dict, None]:
        assert path == "map"
        assert payload == {
            "url": "https://docs.example",
            "limit": 2,
            "search": "",
            "sitemap": "include",
            "includeSubdomains": False,
            "ignoreQueryParameters": True,
        }
        return {
            "success": True,
            "links": [
                {
                    "url": "https://docs.example/guide",
                    "title": "Guide",
                    "description": "Reference",
                },
                "https://docs.example/guide",
                "file:///etc/passwd",
                "https://user:password@docs.example/private",
                "https://docs.example/api",
            ],
        }, None

    monkeypatch.setattr(tools, "_firecrawl_post", fake_post)
    result = await tools._firecrawl_map("https://docs.example", limit=2)
    assert result == {
        "ok": True,
        "links": ["https://docs.example/guide", "https://docs.example/api"],
        "items": [
            {
                "url": "https://docs.example/guide",
                "title": "Guide",
                "description": "Reference",
            },
            {"url": "https://docs.example/api", "title": "", "description": ""},
        ],
    }


@pytest.mark.asyncio
async def test_firecrawl_map_handles_empty_configuration_backend_and_malformed_links(
    monkeypatch,
) -> None:
    assert (await tools._firecrawl_map(""))["error"] == "firecrawl_map_url_required"
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert (await tools._firecrawl_map("https://docs.example"))["error"] == (
        "firecrawl_not_configured"
    )

    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )

    async def failed_post(*_args: Any, **_kwargs: Any) -> tuple[None, dict]:
        return None, {"ok": False, "error": "backend_down"}

    monkeypatch.setattr(tools, "_firecrawl_post", failed_post)
    assert (await tools._firecrawl_map("https://docs.example"))["error"] == "backend_down"

    async def malformed_post(*_args: Any, **_kwargs: Any) -> tuple[dict, None]:
        return {"links": "not-a-list"}, None

    monkeypatch.setattr(tools, "_firecrawl_post", malformed_post)
    assert await tools._firecrawl_map("https://docs.example") == {
        "ok": True,
        "links": [],
        "items": [],
    }

    async def mixed_post(*_args: Any, **_kwargs: Any) -> tuple[dict, None]:
        return {
            "links": [
                "https://docs.example/" + "x" * 4_100,
                "http://[invalid",
                "http://",
                None,
                "https://docs.example/valid",
            ]
        }, None

    monkeypatch.setattr(tools, "_firecrawl_post", mixed_post)
    result = await tools._firecrawl_map("https://docs.example")
    assert result["links"] == ["https://docs.example/valid"]


def test_firecrawl_crawl_pages_are_bounded_deduplicated_and_protocol_normalized() -> None:
    huge = "evidence " * 10_000
    pages = tools._firecrawl_crawl_pages(
        [
            {
                "markdown": huge,
                "metadata": {"sourceURL": "https://docs.example/a", "title": "A"},
            },
            {"url": "https://docs.example/a", "title": "duplicate"},
            {"sourceURL": "https://docs.example/b", "title": "B"},
            {"url": "file:///etc/passwd"},
            {"url": "https://user:pass@docs.example/private"},
            "bad-row",
        ],
        page_limit=2,
    )
    assert [page["url"] for page in pages] == [
        "https://docs.example/a",
        "https://docs.example/b",
    ]
    assert pages[0]["title"] == "A"
    assert len(pages[0]["text"]) == tools._FIRECRAWL_PAGE_TEXT_MAX_CHARS


def test_firecrawl_crawl_pages_rejects_malformed_container_and_rows() -> None:
    assert tools._firecrawl_crawl_pages({}, page_limit=2) == []
    pages = tools._firecrawl_crawl_pages(
        [
            "bad-row",
            {},
            {"url": "x" * 4_097},
            {"url": "http://[invalid"},
            {"url": "file:///etc/passwd"},
            {"url": "https://user:pass@docs.example/private"},
            {"url": "https://docs.example/valid", "markdown": 123},
        ],
        page_limit=2,
    )
    assert pages == [{"url": "https://docs.example/valid", "title": "", "text": ""}]


@pytest.mark.asyncio
async def test_firecrawl_crawl_uses_one_client_and_documented_v2_protocol(monkeypatch) -> None:
    original_client = httpx.AsyncClient
    created = 0
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"success": True, "id": "job-1"})
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "data": [
                    {
                        "markdown": "Agent systems use verified evidence and bounded tools.",
                        "metadata": {
                            "sourceURL": "https://docs.example/guide",
                            "title": "Guide",
                        },
                    }
                ],
            },
        )

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        nonlocal created
        created += 1
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(tools.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(tools, "_firecrawl_pause", lambda *_args: asyncio.sleep(0))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")

    result = await tools._firecrawl_crawl("https://docs.example", limit=3, timeout=40)

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["pages"][0]["url"] == "https://docs.example/guide"
    assert created == 1
    assert [request.method for request in requests] == ["POST", "GET"]
    assert b'"crawlEntireDomain":true' in requests[0].content
    assert b'"allowExternalLinks":false' in requests[0].content


@pytest.mark.asyncio
async def test_firecrawl_crawl_timeout_cancels_remote_job(monkeypatch) -> None:
    original_client = httpx.AsyncClient
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(200, json={"success": True, "id": "job-timeout"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"status": "cancelled"})
        raise AssertionError("deadline must expire before a poll")

    monkeypatch.setattr(
        tools.httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_client(
            *args, transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(tools, "_FIRECRAWL_CRAWL_MIN_TIMEOUT", 0.0)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")

    result = await tools._firecrawl_crawl("https://docs.example", timeout=0)

    assert result["ok"] is True
    assert result["status"] == "timed_out"
    assert methods == ["POST", "DELETE"]


@pytest.mark.asyncio
async def test_firecrawl_crawl_caller_cancellation_cancels_remote_job(monkeypatch) -> None:
    original_client = httpx.AsyncClient
    methods: list[str] = []
    sleeping = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(200, json={"success": True, "id": "job-cancel"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"status": "cancelled"})
        raise AssertionError("cancelled crawl must not poll")

    async def blocking_sleep(_seconds: float) -> None:
        sleeping.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        tools.httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_client(
            *args, transport=httpx.MockTransport(handler), **kwargs
        ),
    )

    async def safe_url(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(tools, "_web_fetch_url_error", safe_url)
    monkeypatch.setattr(tools, "_firecrawl_pause", blocking_sleep)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")

    task = asyncio.create_task(tools._firecrawl_crawl("https://docs.example", timeout=40))
    await sleeping.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert methods == ["POST", "DELETE"]


@pytest.mark.asyncio
async def test_firecrawl_crawl_repeated_cancellation_still_settles_cleanup(monkeypatch) -> None:
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    sleep_started = asyncio.Event()

    async def request(
        _client: httpx.AsyncClient,
        method: str,
        _path: str,
        **_kwargs: Any,
    ) -> tuple[dict, None]:
        assert method == "POST"
        return {"success": True, "id": "job-repeat-cancel"}, None

    async def pause(_seconds: float) -> None:
        sleep_started.set()
        await asyncio.Event().wait()

    async def cleanup(*_args: Any, **_kwargs: Any) -> None:
        cleanup_started.set()
        await release_cleanup.wait()

    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(tools, "_firecrawl_request", request)
    monkeypatch.setattr(tools, "_firecrawl_pause", pause)
    monkeypatch.setattr(tools, "_cancel_firecrawl_crawl", cleanup)

    task = asyncio.create_task(tools._firecrawl_crawl("https://docs.example", timeout=40))
    await sleep_started.wait()
    task.cancel()
    await cleanup_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_firecrawl_crawl_rejects_bad_inputs_without_network(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    assert (await tools._firecrawl_crawl(""))["error"] == "firecrawl_crawl_url_required"
    assert (await tools._firecrawl_crawl("https://docs.example", limit=True))["error"] == (
        "firecrawl_crawl_limit_invalid"
    )
    assert (await tools._firecrawl_crawl("https://docs.example", timeout=float("nan")))[
        "error"
    ] == "firecrawl_crawl_timeout_invalid"
    assert (await tools._firecrawl_crawl("https://docs.example", timeout=cast(Any, "40")))[
        "error"
    ] == ("firecrawl_crawl_timeout_invalid")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert (await tools._firecrawl_crawl("https://docs.example"))["error"] == (
        "firecrawl_not_configured"
    )


@pytest.mark.asyncio
async def test_firecrawl_crawl_rejects_unsafe_submission_errors_and_missing_ids(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result="private target"),
    )
    assert (await tools._firecrawl_crawl("http://127.0.0.1"))["error"] == "url_not_allowed"

    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )

    async def failed_submission(*_args: Any, **_kwargs: Any) -> tuple[None, dict]:
        return None, {"ok": False, "error": "submission_failed"}

    monkeypatch.setattr(tools, "_firecrawl_request", failed_submission)
    assert (await tools._firecrawl_crawl("https://docs.example"))["error"] == ("submission_failed")

    async def missing_id(*_args: Any, **_kwargs: Any) -> tuple[dict, None]:
        return {"success": True, "id": "x" * 257}, None

    monkeypatch.setattr(tools, "_firecrawl_request", missing_id)
    assert (await tools._firecrawl_crawl("https://docs.example"))["error"] == (
        "firecrawl_crawl_no_id"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "stopped"])
async def test_firecrawl_crawl_handles_terminal_remote_statuses(monkeypatch, status: str) -> None:
    calls: list[str] = []

    async def request(
        _client: httpx.AsyncClient,
        method: str,
        _path: str,
        **_kwargs: Any,
    ) -> tuple[dict, None]:
        calls.append(method)
        if method == "POST":
            return {"success": True, "id": f"job-{status}"}, None
        assert method == "GET"
        return {"status": status, "data": []}, None

    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(tools, "_firecrawl_pause", lambda *_args: asyncio.sleep(0))
    monkeypatch.setattr(tools, "_firecrawl_request", request)

    result = await tools._firecrawl_crawl("https://docs.example", timeout=40)

    assert calls == ["POST", "GET"]
    assert result["status"] == status
    if status == "failed":
        assert result["ok"] is False
        assert result["error"] == "firecrawl_crawl_failed"
    else:
        assert result["ok"] is True


@pytest.mark.asyncio
async def test_firecrawl_crawl_survives_intermediate_status_and_poll_error(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(tools, "_firecrawl_pause", lambda *_args: asyncio.sleep(0))
    statuses = iter(["queued", "completed"])

    async def completing_request(
        _client: httpx.AsyncClient,
        method: str,
        _path: str,
        **_kwargs: Any,
    ) -> tuple[dict, None]:
        if method == "POST":
            return {"success": True, "id": "job-progress"}, None
        return {"status": next(statuses), "data": []}, None

    monkeypatch.setattr(tools, "_firecrawl_request", completing_request)
    result = await tools._firecrawl_crawl("https://docs.example", timeout=40)
    assert result == {
        "ok": True,
        "crawl_id": "job-progress",
        "status": "completed",
        "pages": [],
    }

    methods: list[str] = []

    async def failing_poll(
        _client: httpx.AsyncClient,
        method: str,
        _path: str,
        **_kwargs: Any,
    ) -> tuple[dict | None, dict | None]:
        methods.append(method)
        if method == "POST":
            return {"success": True, "id": "job-poll-error"}, None
        if method == "GET":
            return None, {"ok": False, "error": "transport"}
        return {"status": "cancelled"}, None

    monkeypatch.setattr(tools, "_firecrawl_request", failing_poll)
    result = await tools._firecrawl_crawl("https://docs.example", timeout=40)
    assert result["error"] == "firecrawl_crawl_poll_failed"
    assert result["crawl_id"] == "job-poll-error"
    assert methods == ["POST", "GET", "DELETE"]


@pytest.mark.asyncio
async def test_firecrawl_crawl_deadline_after_pause_and_cleanup_failure_are_bounded(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "private")
    monkeypatch.setattr(
        tools,
        "_web_fetch_url_error",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    monkeypatch.setattr(tools, "_FIRECRAWL_CRAWL_MIN_TIMEOUT", 0.0)
    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(tools, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    methods: list[str] = []

    async def request(
        _client: httpx.AsyncClient,
        method: str,
        _path: str,
        **_kwargs: Any,
    ) -> tuple[dict, None]:
        methods.append(method)
        assert method == "POST"
        return {"success": True, "id": "job-deadline"}, None

    async def cleanup_failure(*_args: Any, **_kwargs: Any) -> None:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(tools, "_firecrawl_request", request)
    monkeypatch.setattr(tools, "_firecrawl_pause", lambda *_args: asyncio.sleep(0))
    monkeypatch.setattr(tools, "_cancel_firecrawl_crawl", cleanup_failure)

    result = await tools._firecrawl_crawl("https://docs.example", timeout=1)

    assert result["ok"] is True
    assert result["status"] == "timed_out"
    assert methods == ["POST"]


@pytest.mark.asyncio
async def test_firecrawl_pause_yields_without_blocking() -> None:
    await tools._firecrawl_pause(0)


@pytest.mark.asyncio
async def test_seed_discovery_is_parallel_deduplicated_and_falls_back(monkeypatch) -> None:
    active = 0
    maximum_active = 0

    async def fake_map(url: str, limit: int) -> dict:
        nonlocal active, maximum_active
        assert limit == research._CRAWL_PAGE_MAX
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        if "failed" in url:
            return {"ok": False, "error": "not configured"}
        return {
            "ok": True,
            "items": [
                {
                    "url": f"{url}/guide",
                    "title": "Agent systems guide",
                    "description": "bounded agent architecture",
                },
                {"url": "https://other.example/foreign"},
            ],
        }

    monkeypatch.setattr(tools, "_firecrawl_map", fake_map)
    candidates, notes = await research._seed_sources(
        [
            "https://one.example",
            "https://two.example",
            "https://failed.example",
            "file:///etc/passwd",
        ],
        {"seen_urls": ["https://one.example"]},
        4,
        "agent systems",
    )

    urls = {candidate["url"] for candidate in candidates}
    assert "https://one.example/guide" in urls
    assert "https://two.example/guide" in urls
    assert "https://failed.example" in urls
    assert all("other.example" not in url for url in urls)
    assert maximum_active > 1
    assert any("map failed" in note for note in notes)
    assert any("invalid public seed" in note for note in notes)


@pytest.mark.asyncio
async def test_seed_discovery_bounds_work_and_isolates_connector_failures(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_map(url: str, limit: int) -> dict:
        calls.append(url)
        assert limit == research._CRAWL_PAGE_MAX
        if "raise" in url:
            raise RuntimeError("connector exploded")
        return {"ok": True, "links": [f"{url}/agent-guide"]}

    monkeypatch.setattr(tools, "_firecrawl_map", fake_map)
    candidates, notes = await research._seed_sources(
        [
            "https://healthy.example",
            "https://raise.example",
            "https://not-run.example",
        ],
        {"seen_urls": []},
        2,
        "agent guide",
    )

    assert calls == ["https://healthy.example", "https://raise.example"]
    assert [item["url"] for item in candidates] == [
        "https://healthy.example/agent-guide",
        "https://healthy.example",
    ]
    assert any("beyond the source budget" in note for note in notes)
    assert any("seed discovery failed: RuntimeError" in note for note in notes)


@pytest.mark.asyncio
async def test_seed_discovery_handles_legacy_empty_and_malformed_map_results(monkeypatch) -> None:
    secret = "sk-" + "z" * 30

    async def fake_map(url: str, limit: int) -> dict:
        assert limit == research._CRAWL_PAGE_MAX
        if "legacy" in url:
            return {"ok": True, "links": [f"{url}/agent-reference"]}
        return {
            "ok": True,
            "items": ["bad", {"url": "https://other.example/foreign"}],
        }

    monkeypatch.setattr(tools, "_firecrawl_map", fake_map)
    candidates, notes = await research._seed_sources(
        [
            "http://[invalid",
            secret,
            "https://legacy.example",
            "https://empty.example",
        ],
        {"seen_urls": ["https://empty.example"]},
        4,
        "agent reference",
    )

    assert [item["url"] for item in candidates] == [
        "https://legacy.example/agent-reference",
        "https://legacy.example",
    ]
    assert any("no unread pages found" in note for note in notes)
    assert secret not in " ".join(notes)
    assert "<REDACTED:openai_key>" in " ".join(notes)


@pytest.mark.asyncio
async def test_exhaustive_seed_falls_back_and_keeps_partial_crawl_pages(monkeypatch) -> None:
    async def fake_crawl(url: str, **_kwargs: Any) -> Any:
        if "nondict" in url:
            return None
        if "failed" in url:
            return {"ok": False, "error": "backend unavailable"}
        return {
            "ok": True,
            "status": "scraping",
            "pages": [
                "bad-row",
                {
                    "url": f"{url}/agent-evidence",
                    "title": "Agent evidence",
                    "text": "evidence",
                },
            ],
        }

    monkeypatch.setattr(tools, "_firecrawl_crawl", fake_crawl)
    candidates, notes = await research._seed_sources(
        [
            "https://nondict.example",
            "https://failed.example",
            "https://partial.example",
        ],
        {"seen_urls": []},
        3,
        "agent evidence",
        exhaustive=True,
    )

    urls = {item["url"] for item in candidates}
    assert urls == {
        "https://nondict.example",
        "https://failed.example",
        "https://partial.example/agent-evidence",
    }
    partial = next(item for item in candidates if "partial" in item["url"])
    assert partial["_prefetched_text"] == "evidence"
    assert sum("crawl failed" in note for note in notes) == 2
    assert any("incomplete (scraping)" in note for note in notes)


@pytest.mark.asyncio
async def test_exhaustive_seed_reuses_prefetched_crawl_text_without_second_fetch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def fake_crawl(*_args: Any, **_kwargs: Any) -> dict:
        return {
            "ok": True,
            "status": "completed",
            "pages": [
                {
                    "url": "https://docs.example/agent-guide",
                    "title": "Agent systems evidence",
                    "text": (
                        "Agent systems require verified evidence and bounded execution. "
                        "However, independent validation remains necessary for reliable results."
                    ),
                }
            ],
        }

    async def fake_search(**_kwargs: Any) -> dict:
        return {"ok": False, "error": "offline"}

    async def forbidden_fetch(**_kwargs: Any) -> dict:
        raise AssertionError("prefetched crawl content must not be fetched twice")

    monkeypatch.setattr(tools, "_firecrawl_crawl", fake_crawl)
    monkeypatch.setattr(tools, "t_web_search", fake_search)
    monkeypatch.setattr(tools, "t_web_fetch", forbidden_fetch)

    result = await research.t_deep_research(
        topic="agent systems",
        queries=["agent systems evidence"],
        seeds=["https://docs.example"],
        exhaustive=True,
        rounds=1,
        max_sources=1,
        out_dir=str(tmp_path / "research"),
    )

    assert result["ok"] is True
    assert result["new_sources_this_call"] == 1
    assert result["digest_sources"][0]["url"] == "https://docs.example/agent-guide"
    assert Path(result["intel_path"]).is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"seeds": "https://docs.example"}, "seeds_must_be_a_list_of_urls"),
        ({"seeds": [1]}, "seeds_must_be_a_list_of_urls"),
        ({"seeds": ["https://docs.example/" + "x" * 4_096]}, "seed_url_too_long"),
        ({"exhaustive": "false"}, "exhaustive_must_be_a_boolean"),
    ],
)
async def test_deep_research_seed_controls_are_strict(
    tmp_path: Path,
    kwargs: dict[str, Any],
    error: str,
) -> None:
    result = await research.t_deep_research(
        topic="agent systems",
        rounds=1,
        out_dir=str(tmp_path / "research"),
        **kwargs,
    )
    assert result["ok"] is False
    assert result["error"] == error


def test_research_intel_ingest_is_private_bounded_and_does_not_erase_on_empty(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    monkeypatch.setenv("NORAX_MEMORY_ROOT", str(memory_root))
    report = tmp_path / "report.md"
    report.write_text("report", encoding="utf-8")
    slug = "agent-systems"
    secret = "sk-" + "s" * 30
    sources = [
        {
            "url": f"https://docs.example/{index}",
            "title": f"Evidence {index}",
            "lead": "é" * 1_000,
        }
        for index in range(200)
    ]
    sources[0]["lead"] = secret

    target_value = research._ingest_intel("agent systems", slug, report, sources, ["open thread"])
    assert target_value is not None
    target = Path(target_value)
    before = target.read_bytes()
    assert len(before) <= research._MAX_INTEL_BYTES
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert b"EXTERNAL_RESEARCH_EVIDENCE" in before
    assert secret.encode() not in before
    assert b"<REDACTED:openai_key>" in before
    assert research._ingest_intel("agent systems", slug, report, [], []) == str(target)
    assert target.read_bytes() == before

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(secret)

    monkeypatch.setattr(research, "atomic_write_text", fail_write)
    with caplog.at_level(logging.WARNING, logger="norax.dispatch.deep_research"):
        assert research._ingest_intel("agent systems", slug, report, sources[:1], []) is None
    assert secret not in caplog.text
    assert "<REDACTED:openai_key>" in caplog.text
