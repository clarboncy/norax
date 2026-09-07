from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import httpx
import pytest

from norax.dispatch import web_common


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com:bad", "invalid URL or port"),
        ("file:///tmp/a", "only http and https URLs are allowed"),
        ("https://user:pass@example.com", "credentials embedded in URLs are not allowed"),
        ("https://", "URL has no hostname"),
        ("https://localhost", "private or local network targets are disabled"),
        ("https://foo.localhost", "private or local network targets are disabled"),
        ("https://foo.local", "private or local network targets are disabled"),
        ("https://foo.internal", "private or local network targets are disabled"),
        (
            "http://127.0.0.1",
            "private, loopback, link-local, or reserved targets are disabled",
        ),
        ("https://8.8.8.8", None),
    ],
)
async def test_web_fetch_url_validation_static_shapes(url: str, expected: str | None) -> None:
    assert await web_common._web_fetch_url_error(url) == expected


@pytest.mark.asyncio
async def test_web_fetch_private_opt_in_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("NORAX_WEB_FETCH_ALLOW_PRIVATE", "yes")
    assert await web_common._web_fetch_url_error("http://127.0.0.1") is None
    assert (
        await web_common._web_fetch_url_error(
            "http://127.0.0.1",
            honor_private_opt_in=False,
        )
        is not None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("offline"), UnicodeError("bad host")])
async def test_web_fetch_dns_failures_are_closed(monkeypatch, failure: Exception) -> None:
    def fail(*_args):
        raise failure

    monkeypatch.setattr(web_common.socket, "getaddrinfo", fail)
    assert (
        await web_common._web_fetch_url_error("https://example.test")
        == "hostname could not be resolved"
    )


@pytest.mark.asyncio
async def test_web_fetch_dns_empty_invalid_private_and_global_results(monkeypatch) -> None:
    monkeypatch.setattr(web_common.socket, "getaddrinfo", lambda *_args: [])
    assert (
        await web_common._web_fetch_url_error("https://example.test")
        == "hostname resolved to no addresses"
    )

    monkeypatch.setattr(
        web_common.socket,
        "getaddrinfo",
        lambda *_args: [(2, 1, 6, "", ("not-an-address", 443))],
    )
    assert (
        await web_common._web_fetch_url_error("https://example.test")
        == "hostname resolved to an invalid address"
    )

    monkeypatch.setattr(
        web_common.socket,
        "getaddrinfo",
        lambda *_args: [(2, 1, 6, "", ("10.0.0.1", 443))],
    )
    assert "private" in str(await web_common._web_fetch_url_error("https://example.test"))

    captured: list[tuple] = []

    def global_result(*args):
        captured.append(args)
        return [(2, 1, 6, "", ("2606:4700:4700::1111%eth0", 80))]

    monkeypatch.setattr(web_common.socket, "getaddrinfo", global_result)
    assert await web_common._web_fetch_url_error("http://example.test") is None
    assert captured[0][1] == 80


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "limit", "expected", "truncated"),
    [
        ([b"ab", b"cd"], 10, b"abcd", False),
        ([b"abcd"], 2, b"ab", True),
        ([b"ab", b"cd"], 2, b"ab", True),
    ],
)
async def test_read_web_body_is_strictly_bounded(
    chunks: list[bytes],
    limit: int,
    expected: bytes,
    truncated: bool,
) -> None:
    response = httpx.Response(200, stream=_ChunkStream(chunks))
    assert await web_common._read_web_body(response, limit) == (expected, truncated)


def test_decode_web_body_and_normalize_url_fallbacks() -> None:
    response = httpx.Response(200)
    response.encoding = "made-up-codec"
    assert web_common._decode_web_body(b"hello", response) == "hello"
    default_response = httpx.Response(200)
    assert web_common._decode_web_body(b"hello", default_response) == "hello"
    assert web_common._normalize_search_url(" HTTPS://EXAMPLE.COM/ ") == "https://example.com"
    assert web_common._normalize_search_url(cast(str, None)) == ""
