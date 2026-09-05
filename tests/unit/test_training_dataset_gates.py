from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from norax.observability.log import EventLog
from training.dataset_build import build_dataset, convert_to_qwen35


def _append(log: EventLog, trace_id: str, kind: str, payload: dict) -> None:
    asyncio.run(log.append(kind, payload, trace_id=trace_id))


def _successful_turn(
    log: EventLog,
    trace_id: str,
    *,
    body: str,
    reply: str,
    tool: str | None = None,
    verified: bool = True,
    attempted_mutation: bool = False,
) -> None:
    _append(log, trace_id, "ingress", {"body": body})
    if tool:
        _append(log, trace_id, "tool_call", {"name": tool, "args": {"path": "x"}, "ok": True})
    _append(
        log,
        trace_id,
        "agent_trajectory",
        {
            "outcome": "success" if verified else "failure",
            "outcome_score": 0.9 if verified else 0.2,
            "verified_outcome": verified,
            "mutation_outcome": {
                "attempted": attempted_mutation,
                "verified_after_last_mutation": verified and attempted_mutation,
            },
        },
    )
    _append(log, trace_id, "turn_telemetry", {"content_len": len(reply)})
    _append(log, trace_id, "reply", {"content_preview": reply})


def test_training_tool_schemas_are_runtime_schemas() -> None:
    message = build_dataset.TOOL_SCHEMAS["message_send"]["function"]["parameters"]
    edit = build_dataset.TOOL_SCHEMAS["edit"]["function"]["parameters"]
    remote = build_dataset.TOOL_SCHEMAS["remote_exec"]["function"]["parameters"]

    assert message["required"] == ["channel", "target", "text"]
    assert set(message["properties"]) >= {"channel", "target", "text", "files"}
    assert edit["required"] == ["path", "old", "new"]
    assert "node_id" in remote["required"]
    assert all(isinstance(prop, dict) for prop in message["properties"].values())


def test_event_extraction_joins_interleaved_traces(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    _append(log, "a", "ingress", {"body": "inspect alpha carefully"})
    _append(log, "b", "ingress", {"body": "inspect beta carefully"})
    _append(log, "a", "tool_call", {"name": "read", "args": {"path": "alpha"}, "ok": True})
    _append(log, "b", "tool_call", {"name": "read", "args": {"path": "beta"}, "ok": True})
    for trace_id, reply in (
        ("a", "alpha is verified and complete"),
        ("b", "beta is verified and complete"),
    ):
        _append(
            log,
            trace_id,
            "agent_trajectory",
            {
                "outcome": "success",
                "outcome_score": 0.9,
                "verified_outcome": True,
                "mutation_outcome": {"attempted": False},
            },
        )
        _append(log, trace_id, "turn_telemetry", {"content_len": len(reply)})
        _append(log, trace_id, "reply", {"content_preview": reply})

    turns = {turn["trace_id"]: turn for turn in build_dataset.extract_conversations(tmp_path)}

    assert turns["a"]["tool_calls"][0]["args"]["path"] == "alpha"
    assert turns["b"]["tool_calls"][0]["args"]["path"] == "beta"
    assert turns["a"]["reply"].startswith("alpha")
    assert turns["b"]["reply"].startswith("beta")


def test_quality_gate_rejects_truncated_and_unverified_examples() -> None:
    base = {
        "user_input": "please inspect this implementation",
        "reply": "verified complete response with enough useful detail for training",
        "tool_calls": [],
        "trajectory": {
            "outcome": "success",
            "outcome_score": 0.9,
            "verified_outcome": True,
            "mutation_outcome": {"attempted": False},
        },
    }
    truncated = {**base, "reply_truncated": True}
    unverified = {
        **base,
        "reply_truncated": False,
        "trajectory": {**base["trajectory"], "verified_outcome": False},
    }
    accepted = {**base, "reply_truncated": False}
    ineligible = {
        **base,
        "reply_truncated": False,
        "trajectory": {**base["trajectory"], "training_eligible": False},
    }

    assert build_dataset.filter_quality([truncated, unverified, ineligible, accepted]) == [accepted]


def test_chat_example_records_real_tool_status_and_current_schema() -> None:
    example = build_dataset.build_chat_example(
        {
            "trace_id": "trace",
            "user_input": "please inspect the file",
            "reply": "The file was not found, so no change was made.",
            "tool_calls": [
                {
                    "name": "read",
                    "args": {"path": "/missing"},
                    "ok": False,
                    "error": "file_not_found",
                }
            ],
        },
        "error_recovery",
    )

    tool_call = example["messages"][2]["tool_calls"][0]
    tool_result = json.loads(example["messages"][3]["content"])
    assert tool_call["arguments"] == {"path": "/missing"}
    assert tool_result == {"ok": False, "error": "file_not_found"}
    assert example["tools"][0] == build_dataset.TOOL_SCHEMAS["read"]


def test_converter_rejects_malformed_or_undeclared_tools() -> None:
    malformed = {
        "messages": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "read", "arguments": {"path": "x"}}],
            },
        ],
        "tools": [],
    }
    with pytest.raises(ValueError, match="undeclared tool"):
        convert_to_qwen35.convert_example(malformed)

    malformed["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {"type": "object", "properties": {"path": "string"}},
            },
        }
    ]
    with pytest.raises(ValueError, match="malformed JSON-schema"):
        convert_to_qwen35.convert_example(malformed)


def test_builder_metadata_uses_measured_sources(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    log = EventLog(state / "events.jsonl")
    _successful_turn(
        log,
        "one",
        body="explain alpha with enough context",
        reply="Alpha is a complete verified answer with enough detail for training.",
    )
    _successful_turn(
        log,
        "two",
        body="explain beta with enough context",
        reply="Beta is a separate verified answer with enough detail for training.",
    )
    out = tmp_path / "out"

    assert build_dataset.main(["--state-dir", str(state), "--out", str(out), "--no-docs"]) == 0

    metadata = json.loads((out / "dataset_metadata.json").read_text())
    assert metadata["total_examples"] == 2
    assert metadata["sources"] == {"verified_runtime_trajectory": 2}
    assert metadata["extraction"]["runtime_raw"] == 2
    assert metadata["extraction"]["runtime_verified_complete"] == 2
