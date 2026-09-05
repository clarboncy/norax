from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training import train_norax


def _example(user: str = "inspect the service", *, source_id: str = "trace-1") -> dict[str, Any]:
    tool = {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    return {
        "source": "verified_runtime_trajectory",
        "source_id": source_id,
        "messages": [
            {"role": "system", "content": "Use evidence."},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "read", "arguments": {"path": "/tmp/a"}}],
            },
            {"role": "tool", "content": '{"ok":true}'},
            {"role": "assistant", "content": "The service is healthy."},
        ],
        "tools": [tool],
    }


class _Tokenizer:
    def __init__(self, rendered: dict[str, Any]) -> None:
        self.rendered = rendered
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((messages, kwargs))
        return self.rendered


def test_tokenization_preserves_tools_and_masks_non_assistant_loss() -> None:
    example = _example()
    tokenizer = _Tokenizer(
        {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "assistant_masks": [0, 0, 1, 1],
        }
    )

    row = train_norax.tokenize_example(
        example, tokenizer, max_length=16, location="train example 0"
    )

    assert row["labels"] == [-100, -100, 12, 13]
    assert tokenizer.calls[0][1]["tools"] == example["tools"]
    assert tokenizer.calls[0][1]["return_assistant_tokens_mask"] is True
    assert tokenizer.calls[0][1]["add_generation_prompt"] is False


def test_tokenization_refuses_missing_assistant_mask() -> None:
    tokenizer = _Tokenizer({"input_ids": [1, 2], "attention_mask": [1, 1]})

    with pytest.raises(ValueError, match="refusing full-conversation loss"):
        train_norax.tokenize_example(
            _example(), tokenizer, max_length=16, location="train example 0"
        )


def test_tokenization_refuses_silent_truncation() -> None:
    tokenizer = _Tokenizer({"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]})

    with pytest.raises(ValueError, match="refusing silent truncation"):
        train_norax.tokenize_example(
            _example(), tokenizer, max_length=2, location="train example 0"
        )


def test_manifest_serializes_framework_scalars_as_strict_json(tmp_path: Path) -> None:
    class FrameworkScalar:
        def item(self) -> float:
            return 1.25

    path = tmp_path / "manifest.json"
    train_norax._write_manifest(
        path,
        {
            "metric": FrameworkScalar(),
            "non_finite": float("nan"),
            "artifact": Path("adapter"),
        },
    )

    assert json.loads(path.read_text()) == {
        "artifact": "adapter",
        "metric": 1.25,
        "non_finite": "nan",
    }


def test_split_audit_rejects_request_or_provenance_leakage() -> None:
    train = [_example("Inspect   the service", source_id="trace-a")]
    same_request = [_example(" inspect the SERVICE ", source_id="trace-b")]
    same_source = [_example("different request", source_id="trace-a")]

    with pytest.raises(ValueError, match="user_requests=1"):
        train_norax.audit_split_leakage(train, same_request)
    with pytest.raises(ValueError, match="source_ids=1"):
        train_norax.audit_split_leakage(train, same_source)


def test_structural_validation_does_not_import_training_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text(json.dumps(_example("train request", source_id="train")) + "\n")
    val_path.write_text(json.dumps(_example("validation request", source_id="val")) + "\n")

    def _unexpected_import() -> Any:
        raise AssertionError("training stack should not load")

    monkeypatch.setattr(train_norax, "_load_training_stack", _unexpected_import)
    assert (
        train_norax.main(
            [
                "--train",
                str(train_path),
                "--val",
                str(val_path),
                "--validate-data-only",
            ]
        )
        == 0
    )
