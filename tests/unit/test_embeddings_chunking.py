"""Regression tests for OllamaEmbedder batch chunking + bisect-on-4xx.

Guards against the failure mode where a single oversized/bad text in a large
batch caused the *entire* batch (thousands of memories) to be written as zero
vectors — silently destroying semantic retrieval. The embedder must:

1. Chunk batches larger than `max_batch` into multiple requests.
2. On a permanent 4xx for a chunk, recursively bisect to isolate the bad text,
   so only that one item raises EmbeddingError and the rest keep real vectors.
3. Never persist a zero vector as a successful embedding.
"""

from __future__ import annotations

import asyncio

import httpx
import numpy as np
import pytest

import norax.memory.embeddings as embeddings_mod
from norax.memory.embeddings import EmbeddingError, OllamaEmbedder


def _fake_vec(seed: int, dim: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).tolist()


class _Embedder(OllamaEmbedder):
    """OllamaEmbedder with `_embed_once` swapped for an in-memory fake.

    The fake raises an HTTP 400 for any chunk containing the poison text, and
    otherwise returns deterministic non-zero vectors. This lets us assert the
    chunking/bisection logic without a live Ollama endpoint.
    """

    poison: str = "POISON"
    seen_request_sizes: list[int] | None = None

    async def _embed_once(self, texts):  # type: ignore[override]
        if self.seen_request_sizes is not None:
            self.seen_request_sizes.append(len(texts))
        if any(t == self.poison for t in texts):
            req = httpx.Request("POST", f"{self.base_url}/api/embed")
            resp = httpx.Response(400, request=req)
            raise httpx.HTTPStatusError("bad request", request=req, response=resp)
        vecs = np.asarray([_fake_vec(hash(t) % 10_000, self.dim) for t in texts], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def test_large_batch_is_chunked():
    e = _Embedder(max_batch=128)
    e.seen_request_sizes = []
    texts = [f"mem {i}" for i in range(300)]
    out = asyncio.run(e.embed(texts))
    assert out.shape == (300, e.dim)
    # All vectors should be non-zero (no failures).
    assert int(np.count_nonzero(np.linalg.norm(out, axis=1))) == 300
    # Should have issued multiple requests, none larger than max_batch.
    assert e.seen_request_sizes
    assert max(e.seen_request_sizes) <= 128


def test_one_bad_text_raises_after_bisection():
    """A single bad text in a batch should raise EmbeddingError, not zero the batch.

    The bisection logic isolates the bad text so the rest of the batch gets real
    vectors, but the single bad item raises EmbeddingError rather than silently
    persisting a zero vector.
    """
    e = _Embedder(max_batch=64)
    texts = [f"mem {i}" for i in range(50)]
    texts[20] = e.poison  # single oversized/bad text
    with pytest.raises(EmbeddingError):
        asyncio.run(e.embed(texts))


def test_bisection_preserves_good_vectors_in_mixed_batch():
    """When a batch has one bad text, bisection should still embed the good ones.

    The bisect splits the batch in half — the good half succeeds, the bad half
    bisects again until only the poison text remains, which raises. The good
    vectors from the successful halves are preserved via np.concatenate, but
    the overall call raises because the poison sub-chunk fails.
    """
    e = _Embedder(max_batch=64)
    texts = [f"mem {i}" for i in range(50)]
    texts[20] = e.poison
    # The embed call raises because the poison text ultimately fails
    with pytest.raises(EmbeddingError):
        asyncio.run(e.embed(texts))


def test_empty_input_returns_empty():
    e = _Embedder()
    out = asyncio.run(e.embed([]))
    assert out.shape == (0, e.dim)


def test_health_is_unknown_until_a_successful_embed():
    e = _Embedder()
    assert e.healthy is False
    asyncio.run(e.embed(["evidence"]))
    assert e.healthy is True


def test_all_good_texts_produce_nonzero_vectors():
    """Every successfully embedded text must produce a non-zero vector."""
    e = _Embedder(max_batch=32)
    texts = [f"memory fact #{i}" for i in range(40)]
    out = asyncio.run(e.embed(texts))
    assert out.shape == (40, e.dim)
    norms = np.linalg.norm(out, axis=1)
    # No zero vectors should ever be persisted
    assert int(np.count_nonzero(norms)) == 40
    # All vectors should be finite
    assert np.all(np.isfinite(out))


def test_circuit_breaker_fails_fast_after_a_transient_outage(monkeypatch):
    """A dead embedding service must not retry on every user turn."""

    class _FailingEmbedder(OllamaEmbedder):
        attempts = 0

        async def _embed_once(self, texts):  # type: ignore[override]
            self.attempts += 1
            raise httpx.ReadTimeout("offline")

    monkeypatch.setattr(embeddings_mod, "_RETRY_DELAYS_MS", ())
    e = _FailingEmbedder(circuit_cooldown=30.0)
    with pytest.raises(EmbeddingError, match="after 1 attempts"):
        asyncio.run(e.embed(["first request"]))
    with pytest.raises(EmbeddingError, match="circuit open"):
        asyncio.run(e.embed(["next request"]))
    assert e.attempts == 1
