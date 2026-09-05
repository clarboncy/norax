#!/usr/bin/env python3
"""Benchmark an Ollama/OpenAI-compatible chat endpoint with hard rubrics.

The benchmark scores only the public assistant content. Provider reasoning or
thinking fields are counted for timing diagnostics but are never treated as an
answer or written to the result file. Throughput labels distinguish end-to-end
rate from a streaming decode-rate estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from norax.atomic import atomic_write_text  # noqa: E402


def _long_context_task() -> dict[str, Any]:
    target_indexes = (137, 613, 1189)
    target_codes: list[str] = []
    facts: list[str] = []
    for index in range(1, 1201):
        code = hashlib.sha256(f"norax-power-bench:{index}".encode()).hexdigest()[:10].upper()
        facts.append(
            f"FACT {index:04d}: subsystem_{index % 17} has audit code {code} "
            f"and retry ceiling {index % 5}."
        )
        if index in target_indexes:
            target_codes.append(code)
    expected = "|".join(target_codes)
    prompt = "\n".join(
        [
            "Read all numbered facts. Do not infer audit codes from subsystem names.",
            *facts,
            (
                "Question: return the audit codes from FACT 0137, FACT 0613, and "
                "FACT 1189 in that order, joined by |. Return only that string."
            ),
        ]
    )
    return {
        "name": "long_context_recall",
        "prompt": prompt,
        "max_tokens": 96,
        "expect": expected,
    }


TASKS: list[dict[str, Any]] = [
    {
        "name": "smoke_exact",
        "prompt": "Reply with exactly: PONG",
        "max_tokens": 64,
        "reasoning_effort": "off",
        "expect": "PONG",
    },
    {
        "name": "agentic_debug",
        "prompt": (
            "You are maintaining a Python AI gateway. A pytest test fails because an Ollama native "
            "payload sends OpenAI-only fields, hides model thinking, and truncates heavy tasks. "
            "Give a concise root-cause analysis and a minimal patch plan with tests."
        ),
        "max_tokens": 700,
        "must_contain": ["payload", "test"],
    },
    {
        "name": "coding_pytest",
        "prompt": (
            "Write a Python LRUCache class using OrderedDict with get/put/delete/clear, capacity "
            "validation, and pytest tests for eviction, update, delete missing, and capacity zero "
            "error. Include code blocks."
        ),
        "max_tokens": 900,
        "must_contain": ["class LRUCache", "pytest", "def test"],
    },
    {
        "name": "frontend_route",
        "prompt": (
            "Build a responsive React + Tailwind dashboard component with cards, modal, navbar, "
            "chart placeholder, dark mode toggle, and accessible labels. Return concise "
            "implementation code/notes."
        ),
        "max_tokens": 700,
        "must_contain": ["className", "dark", "aria"],
        "must_contain_any": ["chart", "recharts", "<svg"],
    },
    _long_context_task(),
]

_THINK_BLOCK_RE = re.compile(r"<(?P<tag>think|thinking)>.*?</(?P=tag)>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<(?:think|thinking)>", re.IGNORECASE)


@dataclass
class ChatObservation:
    content: str
    reasoning_chars: int
    seconds_total: float
    seconds_first_any_field: float | None
    seconds_first_content_field: float | None
    usage: dict[str, Any]
    metadata: dict[str, Any]
    finish_reasons: list[str]


@dataclass
class Result:
    task: str
    repetition: int
    ok: bool
    status: str
    seconds_total: float
    seconds_first_any_field: float | None
    seconds_first_content_field: float | None
    visible_chars: int
    reasoning_chars: int
    input_tokens: int | None
    completion_tokens: int | None
    completion_tok_per_sec_e2e: float | None
    completion_tok_per_sec_after_first: float | None
    has_reasoning: bool
    finish_reasons: list[str]
    routed_model: str | None = None
    specialist: str | None = None
    error: str | None = None
    validation_scope: str = "keyword_presence_only"
    task_completion_verified: bool = False


def _text_field(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _reasoning_field(message: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = _text_field(message.get(key))
        if value:
            return value
    return ""


def public_content(content: str) -> tuple[str, int]:
    """Remove embedded reasoning tags and return visible content plus hidden size."""
    hidden_chars = sum(len(match.group(0)) for match in _THINK_BLOCK_RE.finditer(content))
    visible = _THINK_BLOCK_RE.sub("", content)
    unmatched = _THINK_OPEN_RE.search(visible)
    if unmatched:
        hidden_chars += len(visible) - unmatched.start()
        visible = visible[: unmatched.start()]
    return visible.strip(), hidden_chars


def post_json(url: str, payload: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode())
    if not isinstance(decoded, dict):
        raise ValueError("chat endpoint returned a non-object response")
    return decoded


def _payload(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def stream_chat(
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None = None,
) -> ChatObservation:
    url = base.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(
            _payload(
                model,
                prompt,
                max_tokens,
                temperature,
                reasoning_effort,
                stream=True,
            )
        ).encode(),
        headers={"content-type": "application/json", "accept": "text/event-stream"},
    )
    start = time.perf_counter()
    first_any: float | None = None
    first_content: float | None = None
    content_parts: list[str] = []
    reasoning_chars = 0
    final_meta: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    finish_reasons: set[str] = set()
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            escalator = event.get("norax_escalator")
            if isinstance(escalator, dict):
                final_meta.update(escalator)
            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage.update(event_usage)
            for choice in event.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    finish_reasons.add(str(finish_reason))
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                content = _text_field(delta.get("content"))
                reasoning = _reasoning_field(delta)
                if content or reasoning:
                    now = time.perf_counter()
                    if first_any is None:
                        first_any = now
                    if content and first_content is None:
                        first_content = now
                if content:
                    content_parts.append(content)
                reasoning_chars += len(reasoning)
    total = time.perf_counter() - start
    return ChatObservation(
        content="".join(content_parts),
        reasoning_chars=reasoning_chars,
        seconds_total=total,
        seconds_first_any_field=first_any - start if first_any is not None else None,
        seconds_first_content_field=(first_content - start if first_content is not None else None),
        usage=usage,
        metadata=final_meta,
        finish_reasons=sorted(finish_reasons),
    )


def nonstream_chat(
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None = None,
) -> ChatObservation:
    url = base.rstrip("/") + "/chat/completions"
    start = time.perf_counter()
    data = post_json(
        url,
        _payload(
            model,
            prompt,
            max_tokens,
            temperature,
            reasoning_effort,
            stream=False,
        ),
    )
    total = time.perf_counter() - start
    choices = data.get("choices") or []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first_choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    usage = data.get("usage") or {}
    metadata = data.get("norax_escalator") or {}
    finish_reason = first_choice.get("finish_reason")
    return ChatObservation(
        content=_text_field(message.get("content")),
        reasoning_chars=len(_reasoning_field(message)),
        seconds_total=total,
        seconds_first_any_field=None,
        seconds_first_content_field=None,
        usage=usage if isinstance(usage, dict) else {},
        metadata=metadata if isinstance(metadata, dict) else {},
        finish_reasons=[str(finish_reason)] if finish_reason else [],
    )


def _usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in usage or usage[key] is None:
            continue
        try:
            return int(usage[key])
        except (TypeError, ValueError):
            continue
    return None


def score_content(task: dict[str, Any], content: str, finish_reasons: list[str]) -> list[str]:
    """Return failed hard requirements; an empty list is a pass."""
    problems: list[str] = []
    if not content:
        problems.append("empty_public_content")
    if any(reason.lower() in {"length", "max_tokens"} for reason in finish_reasons):
        problems.append("truncated")
    if task.get("expect") is not None and content.strip() != task["expect"]:
        problems.append("exact_mismatch")
    low = content.casefold()
    for required in task.get("must_contain", []):
        if required.casefold() not in low:
            problems.append(f"missing:{required}")
    required_any = task.get("must_contain_any") or []
    if required_any and not any(item.casefold() in low for item in required_any):
        problems.append("missing_any")
    return problems


def run_task(args: argparse.Namespace, task: dict[str, Any], repetition: int) -> Result:
    started = time.perf_counter()
    validation_scope = "exact_answer" if task.get("expect") is not None else "keyword_presence_only"
    try:
        function = stream_chat if args.stream else nonstream_chat
        observation = function(
            args.base,
            args.model,
            task["prompt"],
            task.get("max_tokens", args.max_tokens),
            args.temperature,
            task.get("reasoning_effort"),
        )
        content, embedded_reasoning_chars = public_content(observation.content)
        reasoning_chars = observation.reasoning_chars + embedded_reasoning_chars
        problems = score_content(task, content, observation.finish_reasons)
        output_tokens = _usage_int(observation.usage, "completion_tokens", "output_tokens")
        input_tokens = _usage_int(observation.usage, "prompt_tokens", "input_tokens")
        end_to_end_rate = (
            output_tokens / observation.seconds_total
            if output_tokens is not None and observation.seconds_total > 0
            else None
        )
        after_first_rate = None
        if (
            output_tokens is not None
            and output_tokens > 1
            and observation.seconds_first_any_field is not None
            and observation.seconds_total > observation.seconds_first_any_field
        ):
            after_first_rate = (output_tokens - 1) / (
                observation.seconds_total - observation.seconds_first_any_field
            )
        return Result(
            task=task["name"],
            repetition=repetition,
            ok=not problems,
            status="ok" if not problems else ",".join(problems),
            seconds_total=observation.seconds_total,
            seconds_first_any_field=observation.seconds_first_any_field,
            seconds_first_content_field=observation.seconds_first_content_field,
            visible_chars=len(content),
            reasoning_chars=reasoning_chars,
            input_tokens=input_tokens,
            completion_tokens=output_tokens,
            completion_tok_per_sec_e2e=end_to_end_rate,
            completion_tok_per_sec_after_first=after_first_rate,
            has_reasoning=reasoning_chars > 0,
            finish_reasons=observation.finish_reasons,
            routed_model=observation.metadata.get("routed_model"),
            specialist=observation.metadata.get("specialist"),
            validation_scope=validation_scope,
            task_completion_verified=not problems and validation_scope == "exact_answer",
        )
    except Exception as exc:
        return Result(
            task=task["name"],
            repetition=repetition,
            ok=False,
            status="error",
            seconds_total=time.perf_counter() - started,
            seconds_first_any_field=None,
            seconds_first_content_field=None,
            visible_chars=0,
            reasoning_chars=0,
            input_tokens=None,
            completion_tokens=None,
            completion_tok_per_sec_e2e=None,
            completion_tok_per_sec_after_first=None,
            has_reasoning=False,
            finish_reasons=[],
            error=repr(exc),
            validation_scope=validation_scope,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible base URL"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--tasks", help="Comma-separated task names to run; default all")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.repetitions <= 100:
        parser.error("--repetitions must be between 1 and 100")
    if not 1 <= args.max_tokens <= 32768:
        parser.error("--max-tokens must be between 1 and 32768")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        parser.error("--temperature must be finite and between 0 and 2")

    selected = TASKS
    if args.tasks is not None:
        wanted = {item.strip() for item in args.tasks.split(",") if item.strip()}
        selected = [task for task in TASKS if task["name"] in wanted]
        missing = wanted - {task["name"] for task in selected}
        if missing:
            parser.error(f"unknown task(s): {', '.join(sorted(missing))}")
    if not selected:
        parser.error("at least one task must be selected")

    results = [
        run_task(args, task, repetition)
        for repetition in range(1, args.repetitions + 1)
        for task in selected
    ]
    print(
        f"Benchmark model={args.model} base={args.base} stream={args.stream} "
        f"repetitions={args.repetitions}"
    )
    for result in results:
        detail = result.error or result.status
        first = (
            "-"
            if result.seconds_first_any_field is None
            else f"{result.seconds_first_any_field:.2f}s"
        )
        decode_rate = (
            "-"
            if result.completion_tok_per_sec_after_first is None
            else f"{result.completion_tok_per_sec_after_first:.1f} tok/s"
        )
        route = f" {result.specialist}->{result.routed_model}" if result.routed_model else ""
        print(
            f"{result.task:22} rep={result.repetition} "
            f"{'PASS' if result.ok else 'FAIL':4} total={result.seconds_total:.2f}s "
            f"first_field={first} out={result.completion_tokens} decode={decode_rate} "
            f"visible_chars={result.visible_chars} reasoning_chars={result.reasoning_chars}"
            f"{route} {detail}"
        )

    passed = sum(result.ok for result in results)
    totals = [result.seconds_total for result in results if result.seconds_total]
    summary = {
        "samples": len(results),
        "distinct_tasks": len(selected),
        "repetitions": args.repetitions,
        "passed": passed,
        "gate_passed": passed == len(results),
        "gate_scope": "declared_content_checks_only",
        "task_completions_verified": sum(result.task_completion_verified for result in results),
        "external_competitive_claim": False,
        "median_total_seconds": statistics.median(totals) if totals else None,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        payload = {
            "schema": "norax.ollama_power_bench.v2",
            "created_at": datetime.now(UTC).isoformat(),
            "model": args.model,
            "base": args.base,
            "stream": args.stream,
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        atomic_write_text(args.json_out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
