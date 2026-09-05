"""Embedders — pluggable. Ollama for real, Stub for tests.

The only embedding model we use in production is `norax-embed-v3` (384d).
Pin it. Mixing embedding models destroys index coherence.

Resilience (2026-06-10): OllamaEmbedder now retries with exponential backoff
on transient errors (ConnectError, Timeout, 5xx). On total failure, raises
EmbeddingError instead of returning zero vectors — callers must handle
the failure explicitly, never silently poison the index.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np

log = logging.getLogger("norax.memory.embeddings")


class EmbeddingError(Exception):
    """Raised when embedding fails after all retries.

    Callers must handle this — never silently persist zero vectors.
    """


# Retrieval is on the user-facing turn path.  A dead embedding service must
# degrade to keyword retrieval quickly rather than holding the whole turn.
_RETRY_DELAYS_MS = (250, 750)
_RETRY_JITTER_MS = 200
_RETRIABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    ConnectionRefusedError,
    OSError,
)


class Embedder(Protocol):
    dim: int
    model: str

    async def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class OllamaEmbedder:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "norax-embed-v3"
    dim: int = 384
    timeout: float = 15.0
    circuit_cooldown: float = 30.0
    # Max texts per HTTP request. Large rebuilds (1000s of neurons) are split
    # into chunks so a single oversized text or payload cannot fail — and zero
    # out — the entire batch. Tunable via NORAX_EMBED_BATCH.
    max_batch: int = 512
    # The serving model (all-MiniLM-L6-v2, bert) has a 512-token context.
    # ~3.5 chars/token for typical English → ~1800 chars is a safe ceiling.
    # Pre-truncate before batching so a long text gets a real (prefix)
    # embedding instead of bouncing off the server's context-length check and
    # landing in the shrink path. Tunable via NORAX_EMBED_MAX_CHARS.
    # Scaled to 6000 to preserve more signal with the 256k rolling window.
    max_chars: int = 6_000
    # Ollama keep_alive: how long the model stays loaded in GPU memory after
    # the last request. Default 5m is too short — during idle periods the
    # model gets evicted and the next embed request times out during cold
    # load. 30m covers typical conversation gaps. Tunable via
    # NORAX_EMBED_KEEP_ALIVE.
    keep_alive: str = "30m"
    _consecutive_failures: int = field(default=0, repr=False)
    _last_success: float = field(default=0.0, repr=False)
    _circuit_open_until: float = field(default=0.0, repr=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> OllamaEmbedder:
        return cls(
            base_url=os.environ.get("NORAX_EMBED_URL", "http://127.0.0.1:11434"),
            model=os.environ.get("NORAX_EMBED_MODEL", "norax-embed-v3"),
            dim=int(os.environ.get("NORAX_EMBED_DIM", "384")),
            timeout=float(os.environ.get("NORAX_EMBED_TIMEOUT", "15")),
            circuit_cooldown=float(os.environ.get("NORAX_EMBED_CIRCUIT_COOLDOWN", "30")),
            max_batch=int(os.environ.get("NORAX_EMBED_BATCH", "512")),
            max_chars=int(os.environ.get("NORAX_EMBED_MAX_CHARS", "1800")),
            keep_alive=os.environ.get("NORAX_EMBED_KEEP_ALIVE", "30m"),
        )

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts, chunking large batches. Raises EmbeddingError on failure.

        Splits into `max_batch`-sized chunks. Each chunk retries on transient
        errors and, on a permanent 4xx (e.g. one oversized text), recursively
        splits down to per-item so a single bad text only fails itself — the
        rest of the batch keeps its real vectors. This prevents a single bad
        input from silently corrupting thousands of memory embeddings.
        """
        now = time.monotonic()
        if now < self._circuit_open_until:
            remaining = self._circuit_open_until - now
            log.warning("embed.circuit_open remaining=%.1fs", remaining)
            raise EmbeddingError(f"embed circuit open; retry in {remaining:.1f}s")

        texts = [text[: self.max_chars] for text in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if len(texts) <= self.max_batch:
            return await self._embed_chunk(texts)

        parts: list[np.ndarray] = []
        for i in range(0, len(texts), self.max_batch):
            parts.append(await self._embed_chunk(texts[i : i + self.max_batch]))
        return np.concatenate(parts, axis=0)

    async def _embed_chunk(self, texts: list[str]) -> np.ndarray:
        """Embed one chunk with retry; recursively split on permanent 4xx."""
        last_exc: Exception | None = None
        for attempt in range(len(_RETRY_DELAYS_MS) + 1):
            try:
                result = await self._embed_once(texts)
                # Post-embed validation: reject zero/invalid vectors regardless
                # of whether _embed_once is overridden by a subclass.
                if result.shape[0] != len(texts):
                    raise EmbeddingError(
                        f"embed count mismatch: expected {len(texts)} got {result.shape[0]}"
                    )
                if result.shape[1] != self.dim:
                    raise EmbeddingError(
                        f"embed dimension mismatch: expected {self.dim} got {result.shape[1]}"
                    )
                if not np.all(np.isfinite(result)):
                    raise EmbeddingError("embed contains non-finite values (inf/nan)")
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                zero_count = int(np.sum(norms.ravel() == 0))
                if zero_count > 0:
                    raise EmbeddingError(
                        f"embed produced {zero_count} zero-norm vectors out of {len(texts)}"
                    )
                if self._consecutive_failures > 0:
                    log.info(
                        "embed.recovered after %d consecutive failures",
                        self._consecutive_failures,
                    )
                self._consecutive_failures = 0
                self._last_success = time.monotonic()
                self._circuit_open_until = 0.0
                return result
            except _RETRIABLE as e:
                last_exc = e
                if attempt < len(_RETRY_DELAYS_MS):
                    delay = _RETRY_DELAYS_MS[attempt]
                    jitter = random.uniform(-_RETRY_JITTER_MS, _RETRY_JITTER_MS)
                    wait = max(0.05, (delay + jitter) / 1000.0)
                    log.warning(
                        "embed.retry attempt=%d/%d wait=%.2fs err=%r",
                        attempt + 1,
                        len(_RETRY_DELAYS_MS),
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as e:
                # 5xx = transient, retry; 4xx = permanent for this payload.
                if e.response.status_code >= 500:
                    last_exc = e
                    if attempt < len(_RETRY_DELAYS_MS):
                        delay = _RETRY_DELAYS_MS[attempt]
                        jitter = random.uniform(-_RETRY_JITTER_MS, _RETRY_JITTER_MS)
                        wait = max(0.05, (delay + jitter) / 1000.0)
                        log.warning(
                            "embed.retry_5xx attempt=%d/%d status=%d wait=%.2fs",
                            attempt + 1,
                            len(_RETRY_DELAYS_MS),
                            e.response.status_code,
                            wait,
                        )
                        await asyncio.sleep(wait)
                else:
                    # Permanent 4xx. If the chunk has >1 item, the culprit is
                    # likely one oversized/bad text — split and isolate it so
                    # the rest still get real vectors.
                    if len(texts) > 1:
                        mid = len(texts) // 2
                        log.warning(
                            "embed.split_on_4xx status=%d size=%d — bisecting",
                            e.response.status_code,
                            len(texts),
                        )
                        left = await self._embed_chunk(texts[:mid])
                        right = await self._embed_chunk(texts[mid:])
                        return np.concatenate([left, right], axis=0)
                    # Single text that 4xx'd. Safety net: if truncation
                    # didn't fit (env override pushed max_chars too high, or
                    # a dense-token input still overflows), progressively
                    # shrink and retry so we keep a real vector instead of
                    # dropping the memory entirely.
                    text = texts[0]
                    body = (e.response.text or "").lower()
                    if (
                        e.response.status_code == 400
                        and "context length" in body
                        and len(text) > 256
                    ):
                        try:
                            return await self._embed_chunk_shrink(text)
                        except EmbeddingError:
                            pass  # fall through to the raise below
                    log.error(
                        "embed.permanent_error status=%d body=%s (single text len=%d) — raising",
                        e.response.status_code,
                        e.response.text[:500] if e.response.text else "",
                        len(texts[0]) if texts else 0,
                    )
                    raise EmbeddingError(
                        f"permanent embed error status={e.response.status_code} "
                        f"for text len={len(texts[0]) if texts else 0}"
                    ) from e

        # Total failure for this chunk after all retries — open the circuit
        # breaker and raise so the caller degrades explicitly.
        self._consecutive_failures += 1
        self._circuit_open_until = time.monotonic() + self.circuit_cooldown
        log.error(
            "embed.failed texts=%d consecutive=%d circuit_cooldown=%.1fs err=%r — raising EmbeddingError",
            len(texts),
            self._consecutive_failures,
            self.circuit_cooldown,
            last_exc,
        )
        raise EmbeddingError(
            f"embed failed for {len(texts)} texts after {len(_RETRY_DELAYS_MS) + 1} attempts: {last_exc}"
        )

    async def _embed_chunk_shrink(self, text: str) -> np.ndarray:
        """Progressively shrink one oversized text until the model accepts it.

        Halves from max_chars down to a 64-char floor (below that an
        embedding is meaningless). Returns the first vector the model accepts.
        """
        limit = min(len(text), self.max_chars) // 2
        floor = 64
        while limit >= floor:
            shrunk = text[:limit]
            try:
                vec = await self._embed_once([shrunk])
                log.info(
                    "embed.shrunk_ok original_len=%d shrunk_to=%d",
                    len(text),
                    len(shrunk),
                )
                return vec
            except httpx.HTTPStatusError as e:
                if (
                    e.response.status_code == 400
                    and "context length" in (e.response.text or "").lower()
                ):
                    limit //= 2
                    continue
                raise
        raise EmbeddingError(
            f"embed permanent 4xx: could not shrink text of len={len(text)} to fit"
        )

    def _get_client(self) -> httpx.AsyncClient:
        """Return a pooled async client (reused across calls)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        return self._client

    async def warmup(self) -> None:
        """Send a throwaway embed request to pre-load the model into GPU memory.

        Call this at startup so the first real embed doesn't pay the
        cold-load penalty (which can exceed the timeout and trigger the
        circuit breaker).
        """
        try:
            await self._embed_once(["warmup"])
            log.info("embed.warmup ok model=%s keep_alive=%s", self.model, self.keep_alive)
        except Exception as e:
            log.warning("embed.warmup failed (will retry on first real call): %r", e)

    async def close(self) -> None:
        """Close the pooled HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _embed_once(self, texts: Sequence[str]) -> np.ndarray:
        """Single HTTP attempt — may raise. Validation is in _embed_chunk."""
        c = self._get_client()
        r = await c.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": list(texts),
                "keep_alive": self.keep_alive,
            },
        )
        r.raise_for_status()
        data = r.json()
        vecs = np.asarray(data["embeddings"], dtype=np.float32)
        # L2-normalize for cosine via dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # prevent NaN; _embed_chunk validates nonzero
        return vecs / norms

    @property
    def healthy(self) -> bool:
        """True only when a recent successful embed provides health evidence."""
        if time.monotonic() < self._circuit_open_until:
            return False
        if self._last_success == 0.0:
            return False
        return (time.monotonic() - self._last_success) < 120.0


@dataclass
class StubEmbedder:
    """Deterministic hash-based pseudo-embedder for tests.

    Produces a stable `dim`-vector from each string using SHA-256 bits. Not
    semantically meaningful — only preserves string identity → vector identity.
    Same string → same vector; different strings → different vectors. Good
    enough to exercise the retriever plumbing.
    """

    dim: int = 384
    model: str = "stub-embed-v0"

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Expand 32 bytes into dim floats by hashing with counters
            buf = b""
            counter = 0
            while len(buf) < self.dim * 4:
                buf += hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
                counter += 1
            vec = np.frombuffer(buf[: self.dim * 4], dtype=np.uint32).astype(np.float32)
            vec = (vec / 2**32).astype(np.float32) * 2 - 1  # [-1, 1]
            n = np.linalg.norm(vec)
            out[i] = vec / n if n > 0 else vec
        return out
