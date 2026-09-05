"""Deterministic lexical-similarity intent router (Tier 2).

Uses signed feature hashing and a tiny exact NumPy index for sub-millisecond
matching against predefined route examples. No model download is required.

Design based on:
- Semantic Router (aurora-develop/semantic-router on PyPI)
- NVIDIA AI-Q Intent Classifier depth decision patterns
- arxiv 2502.00409 "Doing More with Less" routing survey
- RouteLLM complexity-based routing approach

Tier 1 = regex fast-path (task_classifier.py)
Tier 2 = THIS — deterministic lexical-similarity fallback

It is consumed by the optional ``task_classifier_v2`` diagnostic API; runtime
authorization and hard tool budgets do not depend on its heuristic score.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

log = logging.getLogger("norax.brain.semantic_router")

# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------

TaskClass = Literal[
    "trivial_social",
    "ack_feedback",
    "simple_question",
    "simple_status",
    "memory_instruction",
    "research",
    "coding",
    "ops_deploy",
    "debug_investigate",
    "danger_sensitive",
    "complex_planning",
    "ambiguous",
]

Depth = Literal["fast", "normal", "deep"]


@dataclass(frozen=True)
class Route:
    name: TaskClass
    description: str
    examples: tuple[str, ...]
    depth: Depth
    max_tools: int


# 12 defined routes + implicit "ambiguous" catch-all
ROUTES: tuple[Route, ...] = (
    Route(
        "trivial_social",
        "Casual greetings, acknowledgements, small talk with no technical content.",
        (
            "hey",
            "hi there",
            "good morning",
            "yo",
            "sup",
            "howdy",
            "what's up",
            "hello",
            "gm",
            "gn",
        ),
        "fast",
        1,
    ),
    Route(
        "ack_feedback",
        "Short acknowledgements, confirmations, or brief emotional reactions.",
        (
            "ok",
            "cool",
            "nice",
            "thanks",
            "done",
            "great",
            "perfect",
            "yep",
            "lol",
            "good",
            "got it",
            "understood",
            "agreed",
        ),
        "fast",
        1,
    ),
    Route(
        "simple_question",
        "Straightforward factual or status questions that need at most one lookup.",
        (
            "what time is it",
            "how much disk space",
            "what's the uptime",
            "who is online",
            "what version are we running",
            "is the service running",
            "where is the log file",
        ),
        "normal",
        4,
    ),
    Route(
        "simple_status",
        "Quick health or status check requests.",
        (
            "status",
            "health check",
            "are you alive",
            "show me uptime",
            "service status",
            "system health",
            "running?",
        ),
        "normal",
        4,
    ),
    Route(
        "memory_instruction",
        "Requests to remember, store, retrieve, or manage memories and identity.",
        (
            "remember this",
            "store my api key",
            "save that for later",
            "forget about x",
            "clean your memory",
            "what do you know about",
            "update my identity",
            "add to memory",
        ),
        "deep",
        12,
    ),
    Route(
        "research",
        "Web search, information gathering, best practices, comparison, benchmarking.",
        (
            "search the web for",
            "research best practices",
            "find me the latest",
            "compare these frameworks",
            "what does cutting edge look like",
            "benchmark our performance",
            "crawl this documentation",
            "fetch that url",
        ),
        "deep",
        25,
    ),
    Route(
        "coding",
        "Writing, editing, refactoring, debugging, or reviewing code.",
        (
            "write a function that",
            "implement a new module",
            "fix the failing test",
            "refactor the database layer",
            "add error handling to",
            "review my pull request",
            "create a new endpoint",
            "patch this bug",
        ),
        "deep",
        20,
    ),
    Route(
        "ops_deploy",
        "Infrastructure deployment, service management, SSH, Docker, configuration.",
        (
            "deploy to staging",
            "restart the service",
            "ssh into the server",
            "docker compose up",
            "install these packages",
            "check the logs",
            "update the production config",
            "scale to 3 replicas",
        ),
        "deep",
        20,
    ),
    Route(
        "debug_investigate",
        "Troubleshooting, investigating failures, tracing errors.",
        (
            "why is this broken",
            "debug the crash",
            "investigate the timeout",
            "trace this error",
            "permission denied help",
            "rate limited fix",
            "figure out why it's failing",
            "what went wrong",
        ),
        "deep",
        20,
    ),
    Route(
        "danger_sensitive",
        "Dangerous operations, sensitive data, security concerns, credentials.",
        (
            "rm -rf the directory",
            "delete all records",
            "here's my private key",
            "my seed phrase is",
            "transfer from wallet",
            "chmod 777 the config",
            "expose the database",
            "wipe the disk",
        ),
        "deep",
        20,
    ),
    Route(
        "complex_planning",
        "Multi-step planning, architecture design, comprehensive framework creation.",
        (
            "design a new architecture",
            "create a comprehensive plan",
            "build a roadmap for",
            "write an end-to-end framework",
            "plan the migration",
            "how should we structure this long-term",
            "lay out the phases",
            "think about the full picture",
        ),
        "deep",
        18,
    ),
)

# "ambiguous" route is implicit — used when no route scores above threshold
AMBIGUOUS_ROUTE = Route("ambiguous", "Unclassified or multi-intent input.", (), "normal", 8)

# ---------------------------------------------------------------------------
# Embedding engine: stable signed feature hashing
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase alphanumeric + 3-gram hashes."""
    text = text.lower().strip()
    # Word tokens
    words = re.findall(r"[a-z0-9_./:-]+", text)
    # 3-grams of character triples for partial matching
    trigrams: list[str] = []
    for w in words:
        for i in range(max(1, len(w) - 2)):
            trigrams.append(w[i : i + 3])
    return words + trigrams


def _hash_token(token: str) -> tuple[int, float]:
    """Return a stable feature bucket and sign for a token."""
    value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
    return value % EMBEDDING_DIM, 1.0 if value & (1 << 63) else -1.0


def embed(text: str) -> np.ndarray:
    """Create a 384-d embedding vector from text via random projection.

    Steps:
    1. Tokenize into words + char 3-grams
    2. Map each token to a dimension via BLAKE2 hash
    3. Apply a stable ±1 sign from the same BLAKE2 digest
    4. L2-normalize

    Returns shape (384,) float32, unit-length.
    """
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Sparse bag-of-hashed-features
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for token in tokens:
        idx, sign = _hash_token(token)
        vec[idx] += sign

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm

    return vec


# ---------------------------------------------------------------------------
# Small exact cosine index
# ---------------------------------------------------------------------------


class _CosineIndex:
    """Minimal exact inner-product index for the router's tiny corpus."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"expected (*, {self.dim}) vectors")
        self._vectors = np.concatenate((self._vectors, matrix), axis=0)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"expected (*, {self.dim}) queries")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        count = min(k, self.ntotal)
        distances = np.full((matrix.shape[0], k), -np.inf, dtype=np.float32)
        indices = np.full((matrix.shape[0], k), -1, dtype=np.int64)
        if count == 0:
            return distances, indices
        scores = matrix @ self._vectors.T
        order = np.argsort(-scores, axis=1)[:, :count]
        distances[:, :count] = np.take_along_axis(scores, order, axis=1)
        indices[:, :count] = order
        return distances, indices


def build_index(routes: tuple[Route, ...] = ROUTES) -> tuple[_CosineIndex, dict[int, Route]]:
    """Build an exact inner-product index over route feature centroids.

    For each route, combine the description + all examples into a single
    centroid embedding by averaging individual embeddings.
    """
    dim = EMBEDDING_DIM
    index = _CosineIndex(dim)
    id_to_route: dict[int, Route] = {}

    vectors: list[np.ndarray] = []
    for i, route in enumerate(routes):
        # Create centroid: embed description + each example, average
        texts = [route.description] + list(route.examples)
        embs = [embed(t) for t in texts]
        centroid = np.mean(embs, axis=0).astype(np.float32)
        # Re-normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid /= norm
        vectors.append(centroid)
        id_to_route[i] = route

    if vectors:
        matrix = np.stack(vectors)
        index.add(matrix)

    return index, id_to_route


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticClassification:
    task_class: TaskClass
    depth: Depth
    confidence: float  # cosine similarity 0-1
    max_tool_calls: int
    matched_route: str
    reason: str


# Confidence thresholds
CONFIDENCE_HIGH = 0.70  # above this → use semantic classification
CONFIDENCE_LOW = 0.40  # below this → definitely "ambiguous"
# Between LOW and HIGH → use semantic class but with "normal" depth as hedge


class SemanticRouter:
    """Persistent intent router with deterministic lexical similarity.

    Usage:
        router = SemanticRouter()
        result = router.route("hey what's the status of the build?")
        # → SemanticClassification(task_class="simple_status", depth="normal", ...)
    """

    def __init__(self, routes: tuple[Route, ...] = ROUTES) -> None:
        self.routes = routes
        self.index, self.id_to_route = build_index(routes)
        # Build per-example index for finer-grained matching
        self._example_index, self._example_id_to_route = self._build_example_index(routes)
        log.info(
            "semantic_router.init routes=%d example_index=%d dim=%d",
            len(routes),
            self._example_index.ntotal,
            EMBEDDING_DIM,
        )

    @staticmethod
    def _build_example_index(
        routes: tuple[Route, ...],
    ) -> tuple[_CosineIndex, dict[int, Route]]:
        """Build a per-example index for finer matching."""
        dim = EMBEDDING_DIM
        index = _CosineIndex(dim)
        id_to_route: dict[int, Route] = {}
        vectors: list[np.ndarray] = []
        i = 0
        for route in routes:
            for example in route.examples:
                vec = embed(example)
                vectors.append(vec)
                id_to_route[i] = route
                i += 1
        if vectors:
            index.add(np.stack(vectors))
        return index, id_to_route

    def route(self, message: str) -> SemanticClassification:
        """Classify a message via semantic similarity.

        Strategy:
        1. Search per-example index (finer-grained)
        2. Search centroid index (broader generalization)
        3. Take the higher-confidence match
        4. Apply confidence thresholds
        """
        if not message or not message.strip():
            return SemanticClassification(
                "trivial_social",
                "fast",
                0.99,
                0,
                "trivial_social",
                "empty message",
            )

        query_vec = embed(message).reshape(1, -1)

        # Search example index
        ex_score, ex_id = 0.0, -1
        if self._example_index.ntotal > 0:
            distances, indices = self._example_index.search(query_vec, 1)
            ex_score = float(distances[0][0])
            ex_id = int(indices[0][0])

        # Search centroid index
        cent_score, cent_id = 0.0, -1
        if self.index.ntotal > 0:
            distances, indices = self.index.search(query_vec, 1)
            cent_score = float(distances[0][0])
            cent_id = int(indices[0][0])

        # Pick the better match
        if ex_score >= cent_score and ex_id in self._example_id_to_route:
            best_score = ex_score
            best_route = self._example_id_to_route[ex_id]
            source = "example_match"
        elif cent_id in self.id_to_route:
            best_score = cent_score
            best_route = self.id_to_route[cent_id]
            source = "centroid_match"
        else:
            # Fallback: no match found
            return SemanticClassification(
                "ambiguous",
                "normal",
                0.0,
                8,
                "ambiguous",
                "no_similarity_match",
            )

        # Apply confidence thresholds
        confidence = max(0.0, min(1.0, best_score))

        if confidence >= CONFIDENCE_HIGH:
            task_class = best_route.name
            depth = best_route.depth
            max_tools = best_route.max_tools
        elif confidence >= CONFIDENCE_LOW:
            # Hedge: use the route class but cap depth at "normal"
            task_class = best_route.name
            depth = "normal"  # conservative hedge
            max_tools = min(best_route.max_tools, 8)
        else:
            # Low confidence → ambiguous
            task_class = "ambiguous"
            depth = "normal"
            max_tools = 8

        reason = f"semantic:{source} score={confidence:.2f} route={best_route.name}"

        return SemanticClassification(
            task_class=task_class,
            depth=depth,
            confidence=confidence,
            max_tool_calls=max_tools,
            matched_route=best_route.name,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init, thread-safe via GIL)
# ---------------------------------------------------------------------------

_router: SemanticRouter | None = None


def get_router() -> SemanticRouter:
    """Get or create the module-level semantic router singleton."""
    global _router
    if _router is None:
        _router = SemanticRouter()
    return _router


def route(message: str) -> SemanticClassification:
    """Convenience function: route a message using the global router."""
    return get_router().route(message)
