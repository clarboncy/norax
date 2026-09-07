"""Shared bounded-response and SSRF defenses for outbound web connectors."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from urllib.parse import urlsplit

import httpx

_WEB_FETCH_MAX_CHARS = 1_000_000
_WEB_FETCH_MAX_BODY_BYTES = 4_000_000
_WEB_FETCH_MAX_REDIRECTS = 5
_WEB_FETCH_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google.internal",
        "kubernetes.default",
        "kubernetes.default.svc",
    }
)


def _web_fetch_private_allowed() -> bool:
    return os.environ.get("NORAX_WEB_FETCH_ALLOW_PRIVATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def _web_fetch_url_error(url: str, *, honor_private_opt_in: bool = True) -> str | None:
    """Return a reason when a URL is unsafe or cannot be resolved.

    Host resolution is repeated for every redirect. This prevents the common
    public-URL-to-metadata/private-network redirect bypass. Operators who
    intentionally use this tool for intranet URLs can opt in explicitly.
    """
    try:
        parsed = urlsplit(str(url or ""))
        port = parsed.port
    except ValueError:
        return "invalid URL or port"
    if parsed.scheme.lower() not in {"http", "https"}:
        return "only http and https URLs are allowed"
    if parsed.username is not None or parsed.password is not None:
        return "credentials embedded in URLs are not allowed"
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        return "URL has no hostname"
    if honor_private_opt_in and _web_fetch_private_allowed():
        return None
    if (
        host in _WEB_FETCH_BLOCKED_HOSTS
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host.endswith(".internal")
    ):
        return "private or local network targets are disabled"

    addresses: set[str] = set()
    try:
        addresses.add(str(ipaddress.ip_address(host)))
    except ValueError:
        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo,
                host,
                port or (443 if parsed.scheme.lower() == "https" else 80),
                0,
                socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError):
            return "hostname could not be resolved"
        addresses.update(str(info[4][0]).split("%", 1)[0] for info in infos if info[4])
    if not addresses:
        return "hostname resolved to no addresses"
    for address in addresses:
        try:
            if not ipaddress.ip_address(address).is_global:
                return "private, loopback, link-local, or reserved targets are disabled"
        except ValueError:
            return "hostname resolved to an invalid address"
    return None


async def _read_web_body(response: httpx.Response, byte_limit: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    truncated = False
    async for chunk in response.aiter_bytes():
        remaining = byte_limit - total
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            truncated = True
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), truncated


def _decode_web_body(raw: bytes, response: httpx.Response) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _normalize_search_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()
