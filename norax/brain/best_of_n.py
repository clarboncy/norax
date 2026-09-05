"""Best-of-N — generate N candidates, score with heuristics, return best.

Works under any model. It is opt-in because each alternative is another model
call and the local scoring signals are useful ranking hints, not a correctness
oracle.

Scoring is zero-LLM heuristic-based:
  1. Length appropriateness — not too short (lazy) or too long (rambling)
  2. Code presence — code blocks get bonus for coding tasks
  3. Structure — headers, lists, tables indicate organized thinking
  4. Specificity — concrete details > vague generalities
  5. Tool evidence — exact trace artifacts referenced in the answer get bonus
  6. No error markers — penalize TODO/FIXME/placeholder
  7. Completeness — addresses all parts of the request

Usage:
    bon = BestOfN(gateway=gateway)
    best = await bon.generate(
        model="your-model:latest",
        messages=messages,
        user_request="implement a web scraper",
        task_type="coding",
        n=3,
    )
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..gateway_client import GatewayRequest, SpendGuardTripped

log = logging.getLogger("norax.brain.best_of_n")
MAX_BEST_OF_N_CANDIDATES = 4


@dataclass
class Candidate:
    content: str
    score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0
    # For n>1 this is the aggregate usage of every generated candidate, not
    # merely the winner. Callers can therefore report the feature's real cost.
    usage: dict[str, int] = field(default_factory=dict)


# --- Scoring heuristics ---

_CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
_HEADER_RE = re.compile(r"^#{1,4}\s+\S+", re.MULTILINE)
_LIST_RE = re.compile(r"^[-*]\s+\S+|^\d+\.\s+\S+", re.MULTILINE)
_TABLE_RE = re.compile(r"\|.*\|.*\n\|[-:| ]+\|")
_ERROR_MARKERS = {"TODO", "FIXME", "placeholder", "not implemented", "WIP", "TBD", "coming soon"}
_SPECIFICITY_RE = re.compile(
    r"\b\d+(?:\.\d+)?\b|"
    r"\b[a-z_]+\.py\b|"
    r"\b[a-z_]+\.js\b|"
    r"\b[a-z_]+\.ts\b|"
    r"/[a-z/_-]+|"
    r"https?://\S+|"
    r"[A-Z][a-z]+Error|"
    r"def\s+\w+|class\s+\w+"
)
_VAGUE_RE = re.compile(
    r"\b(?:maybe|perhaps|might|could be|possibly|I think|not sure|probably)\b",
    re.IGNORECASE,
)


def _trace_evidence_tokens(tool_trace: list[dict]) -> set[str]:
    """Extract concrete trace artifacts that an answer can genuinely cite."""
    tokens: set[str] = set()
    for entry in tool_trace:
        raw_args = entry.get("args")
        raw_result = entry.get("result")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
        for mapping in (args, result):
            for key in ("path", "file", "target", "cwd"):
                value = str(mapping.get(key) or "").strip()
                if value:
                    tokens.add(value.lower())
                    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
                    if len(basename) >= 3:
                        tokens.add(basename.lower())
        command = str(args.get("command") or "").strip()
        if command:
            tokens.add(command[:120].lower())
        if result.get("exit_code") is not None:
            tokens.add(f"exit code {result['exit_code']}".lower())
            tokens.add(f"exit_code={result['exit_code']}".lower())
    return {token for token in tokens if len(token) >= 3}


def _score_candidate(
    content: str,
    user_request: str = "",
    task_type: str = "",
    tool_trace: list[dict] | None = None,
) -> tuple[float, dict[str, float]]:
    """Score a candidate output. Returns (score, signal_breakdown)."""
    signals: dict[str, float] = {}
    text = content.strip()
    text_lower = text.lower()
    word_count = len(text.split())

    # 1. Length appropriateness (bell curve peaking at ~200-800 words)
    if word_count < 10:
        signals["length"] = 0.0
    elif word_count < 50:
        signals["length"] = 0.3
    elif word_count < 200:
        signals["length"] = 0.7
    elif word_count <= 800:
        signals["length"] = 1.0
    elif word_count <= 1500:
        signals["length"] = 0.8
    elif word_count <= 3000:
        signals["length"] = 0.5
    else:
        signals["length"] = 0.3  # too long = rambling

    # 2. Code presence (bonus for coding tasks)
    code_blocks = _CODE_BLOCK_RE.findall(text)
    code_lines = sum(len(m[1].split("\n")) for m in code_blocks) if code_blocks else 0
    if task_type == "coding":
        if code_lines > 0:
            signals["code"] = min(1.0, code_lines / 30)
        else:
            signals["code"] = 0.0  # coding task with no code = bad
    else:
        # Non-coding: code is neutral, slight bonus if it adds specificity
        signals["code"] = min(0.3, code_lines / 100)

    # 3. Structure (headers, lists, tables)
    headers = len(_HEADER_RE.findall(text))
    lists = len(_LIST_RE.findall(text))
    tables = len(_TABLE_RE.findall(text))
    structure_count = headers + lists + tables
    signals["structure"] = min(1.0, structure_count / 8)

    # 4. Specificity (concrete details > vague language)
    specific_count = len(_SPECIFICITY_RE.findall(text))
    vague_count = len(_VAGUE_RE.findall(text))
    if word_count > 0:
        specificity_ratio = specific_count / max(1, word_count / 50)
        signals["specificity"] = min(1.0, specificity_ratio)
    else:
        signals["specificity"] = 0.0
    # Penalize vagueness
    if vague_count > 0:
        signals["specificity"] = max(0.0, signals["specificity"] - vague_count * 0.1)

    # 5. Tool evidence. Generic words such as "verified" or "success" are not
    # evidence and must not improve a candidate's rank.
    tool_trace = tool_trace or []
    if tool_trace:
        evidence_tokens = _trace_evidence_tokens(tool_trace)
        matched = sum(token in text_lower for token in evidence_tokens)
        signals["tool_evidence"] = min(1.0, matched / max(1, min(3, len(evidence_tokens))))
    else:
        signals["tool_evidence"] = 0.5  # neutral if no tools were used

    # 6. Error markers (penalize)
    error_count = sum(1 for m in _ERROR_MARKERS if m.lower() in text_lower)
    signals["no_errors"] = max(0.0, 1.0 - error_count * 0.3)

    # 7. Completeness (address all parts of request)
    if user_request and len(user_request) > 20:
        request_lower = user_request.lower()
        # Extract key terms
        terms = [
            w
            for w in re.findall(r"\b\w{4,}\b", request_lower)
            if w
            not in {
                "the",
                "that",
                "this",
                "with",
                "from",
                "have",
                "make",
                "sure",
                "need",
                "them",
                "they",
                "will",
                "then",
                "also",
                "into",
                "please",
                "things",
                "everything",
                "ensure",
                "fully",
                "might",
            }
        ]
        if terms:
            text_lower = text.lower()
            found = sum(1 for t in terms if t in text_lower)
            signals["completeness"] = found / len(terms)
        else:
            signals["completeness"] = 0.5
    else:
        signals["completeness"] = 0.5

    # 8. Markdown quality (unclosed code fences penalized)
    fence_count = text.count("```")
    if fence_count % 2 != 0:
        signals["markdown"] = 0.0
    else:
        signals["markdown"] = 1.0

    # Weighted sum
    weights = {
        "length": 0.10,
        "code": 0.20 if task_type == "coding" else 0.05,
        "structure": 0.10,
        "specificity": 0.20,
        "tool_evidence": 0.10,
        "no_errors": 0.15,
        "completeness": 0.10,
        "markdown": 0.05,
    }
    score = sum(signals.get(k, 0) * w for k, w in weights.items())
    return score, signals


class BestOfN:
    """Generate N candidates from the model, score them, return the best."""

    def __init__(self, gateway: Any):
        self.gateway = gateway

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict],
        user_request: str = "",
        task_type: str = "",
        tool_trace: list[dict] | None = None,
        n: int = 3,
        temperature: float = 0.7,
        timeout_seconds: float = 30.0,
        tools: list[dict] | None = None,
    ) -> Candidate:
        """Generate N candidates and return the best one.

        For n=1, generate and score one alternative.
        For n>1, generates in parallel and picks the best.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n < 1 or n > MAX_BEST_OF_N_CANDIDATES:
            raise ValueError(f"n must be between 1 and {MAX_BEST_OF_N_CANDIDATES}")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")

        if n == 1:
            # The runtime compares this one alternative against the already
            # generated final response, so it still needs a real score.
            req = GatewayRequest(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
            # Keep the same explicit wall-clock contract as the parallel path.
            # The common n=1 mode previously ignored ``timeout_seconds`` and
            # could delay an otherwise complete answer indefinitely.
            resp = await asyncio.wait_for(
                self.gateway.chat(req),
                timeout=timeout_seconds,
            )
            content = resp.content or ""
            score, signals = _score_candidate(
                content,
                user_request,
                task_type,
                tool_trace,
            )
            return Candidate(
                content=content,
                score=score,
                signals=signals,
                model=resp.model or model,
                latency_ms=getattr(resp, "latency_ms", 0),
                usage={
                    "input_tokens": int((resp.usage or {}).get("input_tokens") or 0),
                    "output_tokens": int((resp.usage or {}).get("output_tokens") or 0),
                },
            )

        # Generate N candidates in parallel
        async def _gen_one(idx: int) -> Candidate:
            # Vary temperature slightly for diversity
            temp = temperature + (idx * 0.1)
            temp = min(1.2, temp)
            req = GatewayRequest(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temp,
            )
            try:
                resp = await asyncio.wait_for(
                    self.gateway.chat(req),
                    timeout=timeout_seconds,
                )
                content = resp.content or ""
                score, signals = _score_candidate(
                    content,
                    user_request,
                    task_type,
                    tool_trace,
                )
                return Candidate(
                    content=content,
                    score=score,
                    signals=signals,
                    model=resp.model or model,
                    latency_ms=getattr(resp, "latency_ms", 0),
                    usage={
                        "input_tokens": int((resp.usage or {}).get("input_tokens") or 0),
                        "output_tokens": int((resp.usage or {}).get("output_tokens") or 0),
                    },
                )
            except SpendGuardTripped:
                raise
            except Exception as e:
                log.warning("best_of_n candidate %d failed: %r", idx, e)
                return Candidate(content="", score=0.0, model=model)

        tasks = [asyncio.create_task(_gen_one(i)) for i in range(n)]
        try:
            generated = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        candidates = [candidate for candidate in generated if candidate.content]

        if not candidates:
            log.warning("best_of_n: all %d candidates failed", n)
            return Candidate(content="", model=model)

        # Pick the best
        best = max(candidates, key=lambda c: c.score)
        best.usage = {
            key: sum(int((candidate.usage or {}).get(key) or 0) for candidate in generated)
            for key in ("input_tokens", "output_tokens")
        }
        log.info(
            "best_of_n: %d candidates, scores=[%s], best=%.3f",
            len(candidates),
            ", ".join(f"{c.score:.3f}" for c in candidates),
            best.score,
        )
        return best

    @staticmethod
    def score_output(
        content: str,
        user_request: str = "",
        task_type: str = "",
        tool_trace: list[dict] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score a single output without generating alternatives."""
        return _score_candidate(content, user_request, task_type, tool_trace)
