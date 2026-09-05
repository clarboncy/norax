#!/usr/bin/env python3
"""Convert and structurally validate the Norax dataset for Qwen3.5.

Actual chat-template rendering is performed by ``train_norax.py`` with the
target tokenizer. This stage refuses malformed roles, tool calls, schemas, or
arguments and never publishes a partially converted split.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from norax.atomic import atomic_write_text  # noqa: E402

_ROLES = {"system", "user", "assistant", "tool"}


def _validate_tools(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if tool.get("type") != "function" or not isinstance(function, dict):
            raise ValueError("tool must use OpenAI function schema")
        name = str(function.get("name") or "")
        parameters = function.get("parameters")
        if not name or not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError("tool is missing name/object parameters")
        properties = parameters.get("properties") or {}
        if not isinstance(properties, dict) or not all(
            isinstance(prop, dict) and isinstance(prop.get("type"), str)
            for prop in properties.values()
        ):
            raise ValueError(f"tool {name} has malformed JSON-schema properties")
        required = parameters.get("required") or []
        if not isinstance(required, list) or not set(required).issubset(properties):
            raise ValueError(f"tool {name} has invalid required fields")
        names.add(name)
    return names


def convert_example(ex: dict[str, Any]) -> dict[str, Any]:
    """Convert a single example to Qwen3.5-compatible format."""
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("example has no messages")
    tools = ex.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    declared_tools = _validate_tools(tools)
    fixed: list[dict[str, Any]] = []

    for index, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in _ROLES:
            raise ValueError(f"message {index} has invalid role")
        role = str(m["role"])
        m2: dict[str, Any] = {"role": role}

        # Content handling
        if m.get("content") is not None:
            m2["content"] = m["content"]
        elif role == "assistant" and m.get("tool_calls"):
            m2["content"] = ""  # Empty content is fine for tool-call-only messages
        else:
            m2["content"] = ""

        # Fix tool_calls: convert arguments from JSON string to dict
        if m.get("tool_calls"):
            fixed_tcs = []
            for tc in m["tool_calls"]:
                if not isinstance(tc, dict):
                    raise ValueError(f"message {index} has non-object tool call")
                raw_function = tc.get("function")
                function: dict[str, Any] = raw_function if isinstance(raw_function, dict) else tc
                name = str(function.get("name") or "")
                if not name or name not in declared_tools:
                    raise ValueError(f"message {index} calls undeclared tool {name!r}")
                tc2: dict[str, Any] = {"name": name}
                if tc.get("id"):
                    tc2["id"] = str(tc["id"])
                args = function.get("arguments")
                if isinstance(args, str):
                    try:
                        tc2["arguments"] = json.loads(args)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"message {index} has invalid tool JSON") from exc
                elif isinstance(args, dict):
                    tc2["arguments"] = args
                elif args is None:
                    tc2["arguments"] = {}
                else:
                    raise ValueError(f"message {index} has non-object tool arguments")
                fixed_tcs.append(tc2)
            m2["tool_calls"] = fixed_tcs

        # Tool role messages: only keep role + content (template uses content directly)
        if role == "tool":
            m2 = {"role": "tool", "content": m.get("content", "")}

        fixed.append(m2)

    result = {"messages": fixed}
    if tools:
        result["tools"] = tools
    if ex.get("category"):
        result["category"] = ex["category"]
    if ex.get("source"):
        result["source"] = ex["source"]
    if ex.get("source_id"):
        result["source_id"] = ex["source_id"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).parent)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    base: Path = args.base
    total_errors = 0
    converted_splits = 0

    for split in ["norax_train.jsonl", "norax_val.jsonl"]:
        src = base / split
        if not src.exists():
            message = f"MISSING: {src}"
            print(message, file=sys.stderr)
            if not args.allow_missing:
                total_errors += 1
            continue

        out_name = split.replace(".jsonl", "_qwen35.jsonl")
        dst = base / out_name

        converted = 0
        errors = 0
        rendered: list[str] = []

        with src.open(encoding="utf-8") as fin:
            for line in fin:
                try:
                    ex = json.loads(line)
                    fixed = convert_example(ex)
                    rendered.append(json.dumps(fixed, ensure_ascii=False) + "\n")
                    converted += 1
                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                    errors += 1
                    print(f"  ERROR in {split}: {e}", file=sys.stderr)

        if errors:
            print(
                f"{split}: validation failed ({errors} errors); {dst.name} not replaced",
                file=sys.stderr,
            )
            total_errors += errors
            continue
        atomic_write_text(dst, "".join(rendered))
        converted_splits += 1
        print(f"{split}: {converted} converted, 0 errors -> {dst.name}")

    if total_errors:
        return 1
    if converted_splits == 0:
        return 1
    print("\nDone. Target-tokenizer rendering remains a required training preflight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
