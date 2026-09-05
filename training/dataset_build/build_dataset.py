#!/usr/bin/env python3
"""Build a provenance-rich SFT dataset from verified Norax trajectories."""

import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from norax.atomic import atomic_write_text  # noqa: E402
from norax.brain.harness_optimizer import validate_event_log_hash_chain  # noqa: E402
from norax.dispatch.tools import REGISTRY, render_tools_for_llm  # noqa: E402
from norax.safety.secrets import redact  # noqa: E402

OUT = BASE / "training" / "dataset_build"
OUT.mkdir(parents=True, exist_ok=True)

TOOL_SCHEMAS = {
    schema["function"]["name"]: schema for schema in render_tools_for_llm(list(REGISTRY))
}

SYS_PROMPTS = {
    "offline_resilience": "You are Norax, an agent on a self-hosted runtime. Use a local or fallback model route only when it is configured and has current health evidence. Local tools and cached memory may remain available during an external outage, but never claim a route, remote node, or action works without verification. State genuine blockers precisely.",
    "error_recovery": "You are Norax, an autonomous AI agent. When you encounter errors, diagnose root cause by reading logs, checking configs, inspecting code. Apply fixes directly. Retry with modified approaches on failure. Never report success without verifying.",
    "system_operations": "You are Norax, managing multi-node AI infrastructure. SSH to remote nodes, manage systemd services, deploy updates, maintain runtime. Always verify service health after changes.",
    "remote_operations": "You are Norax, operating across multiple nodes. Use remote_exec, remote_read, remote_write, remote_list to manage remote machines. Verify connectivity before operations.",
    "web_research": "You are Norax, a research agent. Use current primary sources when needed, cite evidence, and apply findings only when the user's requested outcome includes a change. Verify any changes before claiming success.",
    "code_development": "You are Norax, a senior software engineer. Read code before editing. Write complete implementations. Test after changes.",
    "agentic_general": "You are Norax, an agent on a self-hosted runtime with multi-signal memory and bounded tool loops. Execute requested actions and answer questions directly. Never claim an action, capability, or result without evidence.",
    "domain_knowledge": "You are Norax. Answer questions about your architecture, procedures, and operational knowledge accurately.",
    "learned_patterns": "You are Norax. Apply learned patterns from operational experience.",
    "architecture_knowledge": "You are Norax. You have deep knowledge of your architecture, deployment, and operational procedures.",
}


def classify_conversation(turn):
    ui = turn["user_input"].lower()
    tools = turn.get("tool_calls", [])
    tool_names = [t["name"] for t in tools]
    if any(
        kw in ui
        for kw in [
            "internet",
            "offline",
            "api down",
            "api out",
            "unavailable",
            "fallback",
            "local model",
            "ollama",
            "can't use tools",
            "no tool binding",
            "rate limit",
            "403",
            "400 error",
            "upstream",
            "not available",
            "proxy",
        ]
    ):
        return "offline_resilience"
    if any(
        kw in ui
        for kw in [
            "fix",
            "bug",
            "error",
            "broken",
            "crash",
            "fail",
            "issue",
            "not working",
            "wrong",
            "investigate",
            "debug",
        ]
    ):
        return "error_recovery"
    if any(
        kw in ui
        for kw in [
            "deploy",
            "restart",
            "service",
            "systemd",
            "ssh",
            "staging",
            "server",
            "infrastructure",
            "config",
            "update",
            "upgrade",
            "install",
            "setup",
            "migrate",
            "cutover",
        ]
    ):
        return "system_operations"
    if "remote_" in str(tool_names):
        return "remote_operations"
    if "web_search" in tool_names or "web_fetch" in tool_names:
        return "web_research"
    if any(
        kw in ui
        for kw in [
            "code",
            "function",
            "class",
            "refactor",
            "implement",
            "write",
            "build",
            "create",
            "add",
        ]
    ):
        return "code_development"
    return "agentic_general"


def extract_conversations(
    state_dir: Path | None = None, *, require_integrity: bool = True
) -> list[dict[str, Any]]:
    """Join turn events by trace ID instead of adjacency in a shared log."""
    state_dir = state_dir or BASE / "state"
    event_files = sorted(glob.glob(str(state_dir / "events*.jsonl")))
    turns: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"user_input": "", "tool_calls": [], "reply": ""}
    )
    for fpath in event_files:
        event_path = Path(fpath)
        if not event_path.exists() or event_path.stat().st_size == 0:
            continue
        integrity = validate_event_log_hash_chain(event_path)
        if require_integrity and not integrity.get("ok"):
            raise ValueError(
                f"event log integrity failed for {event_path}: "
                f"{integrity.get('error') or integrity.get('first_error') or integrity}"
            )
        try:
            lines = event_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            trace_id = str(ev.get("trace_id") or "").strip()
            if not trace_id:
                continue
            kind = ev.get("kind")
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            turn = turns[trace_id]
            turn["trace_id"] = trace_id
            turn["ts"] = turn.get("ts") or ev.get("ts")
            if kind == "ingress":
                turn["user_input"] = str(payload.get("body") or "")
            elif kind == "tool_call":
                name = str(payload.get("name") or "")
                if name:
                    turn["tool_calls"].append(
                        {
                            "name": name,
                            "args": redact(payload.get("args") or {}),
                            "ok": payload.get("ok") is True,
                            "error": str(payload.get("err") or "")[:200],
                        }
                    )
            elif kind == "reply":
                turn["reply"] = str(payload.get("content_preview") or "")
            elif kind == "turn_telemetry":
                turn["content_len"] = int(payload.get("content_len") or 0)
            elif kind == "agent_trajectory":
                turn["trajectory"] = {
                    "outcome": payload.get("outcome"),
                    "outcome_score": payload.get("outcome_score"),
                    "accepted_outcome": payload.get("accepted_outcome") is True,
                    "training_eligible": payload.get("training_eligible", True) is True,
                    "verified_outcome": payload.get("verified_outcome") is True,
                    "mutation_outcome": payload.get("mutation_outcome") or {},
                }

    conversations = []
    for turn in turns.values():
        if not turn.get("user_input") or not turn.get("reply"):
            continue
        content_len = int(turn.get("content_len") or len(turn["reply"]))
        turn["reply_truncated"] = content_len > len(turn["reply"])
        conversations.append(turn)
    return sorted(conversations, key=lambda item: str(item.get("ts") or ""))


def filter_quality(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality = []
    for c in conversations:
        ui = c.get("user_input", "")
        reply = c.get("reply", "")
        tools = c.get("tool_calls", [])
        if ui.startswith("[cron:") or ui.startswith("LIVE_") or ui.startswith("NO_REPLY"):
            continue
        if len(ui) < 10 or not reply or len(reply) < 10:
            continue
        # Event logs intentionally store only a 500-character reply preview.
        # Training on a known prefix teaches abrupt, incomplete final answers.
        if c.get("reply_truncated"):
            continue
        trajectory = c.get("trajectory") or {}
        if trajectory.get("training_eligible", True) is not True:
            continue
        if trajectory.get("outcome") != "success":
            continue
        if float(trajectory.get("outcome_score") or 0.0) <= 0.5:
            continue
        if trajectory.get("verified_outcome") is not True:
            continue
        mutation = trajectory.get("mutation_outcome") or {}
        if mutation.get("attempted") and not mutation.get("verified_after_last_mutation"):
            continue
        if len(tools) > 32:
            continue
        if ui.startswith("Here's the honest") or ui.startswith("I just gave you"):
            continue
        if len(tools) == 0 and len(reply) < 50:
            continue
        quality.append(c)
    return quality


def build_chat_example(turn: dict[str, Any], category: str) -> dict[str, Any]:
    user_input = str(redact(turn["user_input"]))
    reply = str(redact(turn.get("reply", "")))
    tools = turn.get("tool_calls", [])
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYS_PROMPTS.get(category, SYS_PROMPTS["agentic_general"])},
        {"role": "user", "content": user_input},
    ]
    used: list[str] = []
    for index, tool in enumerate(tools):
        name = str(tool.get("name") or "")
        if name not in TOOL_SCHEMAS:
            continue
        args = tool.get("args") if isinstance(tool.get("args"), dict) else {}
        args = redact(args)
        call_id = f"trace-{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "name": name, "arguments": args}],
            }
        )
        tool_result: dict[str, Any] = {"ok": tool.get("ok") is True}
        if not tool_result["ok"]:
            tool_result["error"] = str(tool.get("error") or "tool_failed")[:200]
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": call_id,
                "content": json.dumps(tool_result, separators=(",", ":")),
            }
        )
        if name not in used:
            used.append(name)
    messages.append({"role": "assistant", "content": reply})
    return {
        "category": category,
        "source": "verified_runtime_trajectory",
        "source_id": turn.get("trace_id", ""),
        "messages": messages,
        "tools": render_tools_for_llm(used),
    }


def _bounded_source_text(raw: str, limit: int) -> str:
    """Redact and end source-derived answers at a textual boundary."""
    content = str(redact(raw)).replace("\x00", "").strip()
    if len(content) <= limit:
        return content
    prefix = content[:limit]
    for separator in ("\n\n", "\n", ". "):
        head, found, _tail = prefix.rpartition(separator)
        if found and len(head) >= limit // 2:
            return (head + found).rstrip()
    return ""


def extract_memory_knowledge(memory_dir: Path | None = None) -> list[dict[str, Any]]:
    """Extract self-generated memory only when explicitly requested."""
    examples = []
    memory_dir = memory_dir or BASE / "memory"
    for category, subdir, sys_key in [
        ("procedural", "procedural", "domain_knowledge"),
        ("semantic", "semantic", "domain_knowledge"),
        ("intel", "intel", "learned_patterns"),
    ]:
        for fpath in sorted(glob.glob(str(memory_dir / subdir / "**/*.md"), recursive=True)):
            try:
                content = Path(fpath).read_text(encoding="utf-8")
                if len(content) < 100:
                    continue
                content = _bounded_source_text(content, 2_000)
                if not content:
                    continue
                lines = content.split("\n")
                title = lines[0].replace("#", "").strip() if lines else Path(fpath).name
                examples.append(
                    {
                        "category": sys_key,
                        "source": f"memory:{category}",
                        "source_id": str(Path(fpath).relative_to(memory_dir)),
                        "messages": [
                            {"role": "system", "content": SYS_PROMPTS[sys_key]},
                            {"role": "user", "content": f"Explain: {title}"},
                            {"role": "assistant", "content": content},
                        ],
                        "tools": [],
                    }
                )
            except (OSError, UnicodeError):
                pass
    return examples


def extract_docs_knowledge(docs_dir: Path | None = None) -> list[dict[str, Any]]:
    examples = []
    docs_dir = docs_dir or BASE / "docs"
    for fpath in sorted(glob.glob(str(docs_dir / "**/*.md"), recursive=True)):
        try:
            content = Path(fpath).read_text(encoding="utf-8")
            if len(content) < 200:
                continue
            content = _bounded_source_text(content, 3_000)
            if not content:
                continue
            title = Path(fpath).stem.replace("_", " ")
            examples.append(
                {
                    "category": "architecture_knowledge",
                    "source": "versioned_docs",
                    "source_id": str(Path(fpath).relative_to(docs_dir)),
                    "messages": [
                        {"role": "system", "content": SYS_PROMPTS["architecture_knowledge"]},
                        {"role": "user", "content": f"Describe: {title}"},
                        {"role": "assistant", "content": content},
                    ],
                    "tools": [],
                }
            )
        except (OSError, UnicodeError):
            pass
    return examples


def generate_offline_examples():
    scenarios = [
        (
            "The internet is down and all cloud APIs are unavailable. Can you still operate?",
            "First inspect current capability and provider evidence. If a configured local route has a recent successful transport or completion probe, use it and report which level was verified. Local file and shell tools do not require internet, but a conversational turn still needs a working inference route to decide and describe new actions. If no model route is verified, say that the turn is blocked; do not invent an Ollama model, remote node, or successful fallback.",
        ),
        (
            "A tool call is failing repeatedly. How do you recover?",
            "Stop issuing the identical call. Preserve the exact error, re-read the target or inspect the relevant state, and change the arguments or approach based on that evidence. A different tool is useful only when it actually provides the required capability; changing `exec` to `shell` alone does not fix a bad command. After a mutation, verify the same target with a readback or recognized validation command. If the requested outcome remains unverified, report the blocker explicitly.",
        ),
    ]
    examples = []
    for user_msg, assistant_msg in scenarios:
        examples.append(
            {
                "category": "offline_resilience",
                "source": "curated_runtime_policy",
                "source_id": hashlib.sha256(user_msg.encode()).hexdigest()[:16],
                "messages": [
                    {"role": "system", "content": SYS_PROMPTS["offline_resilience"]},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "tools": [],
            }
        )
    return examples


def _example_user_key(example: dict[str, Any]) -> str:
    user = next(
        (
            str(message.get("content") or "")
            for message in example.get("messages", [])
            if message.get("role") == "user"
        ),
        "",
    )
    normalized = " ".join(user.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def deduplicate_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one answer per normalized user request to prevent contradictions."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example in examples:
        key = _example_user_key(example)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def split_examples(
    examples: list[dict[str, Any]], *, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic, category-stratified held-out split."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[str(example.get("category") or "unknown")].append(example)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for category in sorted(grouped):
        items = list(grouped[category])
        rng.shuffle(items)
        val_count = max(1, round(len(items) * 0.1)) if len(items) >= 5 else 0
        val.extend(items[:val_count])
        train.extend(items[val_count:])

    if not val and len(train) >= 2:
        val.append(train.pop())
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _jsonl(examples: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=BASE / "state")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--include-memory",
        action="store_true",
        help="Include self-generated canonical memory (off by default to avoid feedback loops).",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include the small curated runtime-policy set (off by default).",
    )
    parser.add_argument("--no-docs", action="store_true")
    parser.add_argument(
        "--allow-unverified-logs",
        action="store_true",
        help="Permit legacy/tampered event logs (unsafe; recorded in metadata).",
    )
    args = parser.parse_args(argv)
    out_dir: Path = args.out

    print("=" * 60)
    print("Norax Agentic Training Dataset Builder")
    print("=" * 60)
    all_examples: list[dict[str, Any]] = []

    print("\n[1] Extracting verified, complete conversations from state events...")
    try:
        convos = extract_conversations(
            args.state_dir,
            require_integrity=not args.allow_unverified_logs,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    quality = filter_quality(convos)
    for turn in quality:
        cat = classify_conversation(turn)
        all_examples.append(build_chat_example(turn, cat))
    print(f"  accepted={len(quality)} rejected={len(convos) - len(quality)} raw={len(convos)}")

    mem_ex: list[dict[str, Any]] = []
    if args.include_memory:
        print("\n[2] Extracting explicitly enabled self-generated memory...")
        mem_ex = extract_memory_knowledge()
        all_examples.extend(mem_ex)
        print(f"  {len(mem_ex)} memory examples")

    doc_ex: list[dict[str, Any]] = []
    if not args.no_docs:
        print("\n[3] Extracting version-controlled architecture docs...")
        doc_ex = extract_docs_knowledge()
        all_examples.extend(doc_ex)
        print(f"  {len(doc_ex)} documentation examples")

    synthetic_ex: list[dict[str, Any]] = []
    if args.include_synthetic:
        print("\n[4] Adding explicitly enabled curated policy examples...")
        synthetic_ex = generate_offline_examples()
        all_examples.extend(synthetic_ex)
        print(f"  {len(synthetic_ex)} curated examples")

    print("\n[5] Deduplicating by normalized full user request...")
    unique = deduplicate_examples(all_examples)
    print(f"  Before: {len(all_examples)} -> After: {len(unique)}")
    if len(unique) < 2:
        print("ERROR: fewer than two validated examples; refusing to emit train/val splits")
        return 1

    print("\n[6] Saving atomic dataset artifacts...")
    train, val = split_examples(unique)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "norax_agentic_dataset.jsonl"
    atomic_write_text(full_path, _jsonl(unique))
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in unique:
        by_cat[ex["category"]].append(ex)
    for cat, items in by_cat.items():
        atomic_write_text(out_dir / f"norax_{cat}.jsonl", _jsonl(items))
    atomic_write_text(out_dir / "norax_train.jsonl", _jsonl(train))
    atomic_write_text(out_dir / "norax_val.jsonl", _jsonl(val))

    total_chars = sum(len(json.dumps(ex)) for ex in unique)
    est_tokens = total_chars // 4
    cat_counts = Counter(str(example["category"]) for example in unique)
    source_counts = Counter(str(example.get("source") or "unknown") for example in unique)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total unique examples: {len(unique)}")
    print(f"Train: {len(train)} | Val: {len(val)}")
    print("\nBy category:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")
    print(f"\nEstimated tokens: {est_tokens:,}")
    print(f"Avg tokens/example: {est_tokens // max(len(unique), 1):,}")
    print(f"\nFiles in {out_dir}/:")
    for p in sorted(out_dir.glob("*.jsonl")):
        print(f"  {p.name}")
    schema_body = json.dumps(TOOL_SCHEMAS, sort_keys=True, separators=(",", ":"))
    meta = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_examples": len(unique),
        "train": len(train),
        "val": len(val),
        "categories": dict(cat_counts.most_common()),
        "estimated_tokens": est_tokens,
        "sources": dict(source_counts),
        "extraction": {
            "runtime_raw": len(convos),
            "runtime_verified_complete": len(quality),
            "event_log_integrity_required": not args.allow_unverified_logs,
            "memory_opt_in": bool(args.include_memory),
            "memory_extracted": len(mem_ex),
            "docs_extracted": len(doc_ex),
            "synthetic_opt_in": bool(args.include_synthetic),
            "synthetic_extracted": len(synthetic_ex),
        },
        "tool_schema_sha256": hashlib.sha256(schema_body.encode()).hexdigest(),
        "format": "OpenAI-style messages; convert_to_qwen35.py performs target-template validation",
    }
    atomic_write_text(
        out_dir / "dataset_metadata.json",
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
    )
    print("  dataset_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
