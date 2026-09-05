#!/usr/bin/env python3
"""Fine-tune the Norax tool-using model from validated conversational data.

The dataset is rendered by the target tokenizer with its declared tool schemas.
Loss is applied only to tokens marked as assistant output by the chat template.
The preflight refuses malformed data, split leakage, unsupported assistant masks,
and over-length examples instead of silently weakening the training signal.

Training dependencies are intentionally optional for the Norax runtime. Install
the CUDA-compatible PyTorch, Unsloth, Transformers, Datasets, and TRL stack in a
dedicated training environment before running this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from norax.atomic import atomic_write_text  # noqa: E402

MODEL_NAME = "Qwen/Qwen3.5-9B"
DATASET_PATH = Path(__file__).parent / "dataset_build" / "norax_train_qwen35.jsonl"
VAL_PATH = Path(__file__).parent / "dataset_build" / "norax_val_qwen35.jsonl"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "norax-9b-agentic-v1"
MAX_SEQ_LENGTH = 4096

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 4.0
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
SCHEDULER = "cosine"
SAVE_STEPS = 50
LOG_STEPS = 5
EVAL_STEPS = 50
SEED = 42


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_tool_names(tools: Any, *, location: str) -> set[str]:
    if not isinstance(tools, list):
        raise ValueError(f"{location}: tools must be a list")
    names: set[str] = set()
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        parameters = function.get("parameters") if isinstance(function, dict) else None
        name = str(function.get("name") or "") if isinstance(function, dict) else ""
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or not name
            or not isinstance(parameters, dict)
            or parameters.get("type") != "object"
        ):
            raise ValueError(f"{location}: tool {index} is not a valid function schema")
        if name in names:
            raise ValueError(f"{location}: duplicate tool declaration {name!r}")
        names.add(name)
    return names


def validate_example(example: Any, *, location: str) -> dict[str, Any]:
    """Validate the structure required by target-tokenizer rendering."""
    if not isinstance(example, dict):
        raise ValueError(f"{location}: example must be an object")
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{location}: messages must be a non-empty list")
    tools = example.get("tools") or []
    declared = _declared_tool_names(tools, location=location)
    assistant_messages = 0

    for index, message in enumerate(messages):
        item_location = f"{location} message {index}"
        if not isinstance(message, dict):
            raise ValueError(f"{item_location}: message must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{item_location}: invalid role {role!r}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"{item_location}: content must be a string")
        tool_calls = message.get("tool_calls") or []
        if tool_calls and role != "assistant":
            raise ValueError(f"{item_location}: only assistant messages may call tools")
        if not isinstance(tool_calls, list):
            raise ValueError(f"{item_location}: tool_calls must be a list")
        if role == "assistant":
            assistant_messages += 1
        for call_index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                raise ValueError(f"{item_location}: tool call {call_index} must be an object")
            raw_function = call.get("function")
            function = raw_function if isinstance(raw_function, dict) else call
            name = str(function.get("name") or "")
            arguments = function.get("arguments")
            if name not in declared:
                raise ValueError(f"{item_location}: call to undeclared tool {name!r}")
            if not isinstance(arguments, dict):
                raise ValueError(f"{item_location}: {name} arguments must be an object")

    if assistant_messages == 0:
        raise ValueError(f"{location}: no assistant response to train")
    return example


def load_examples(path: Path) -> list[dict[str, Any]]:
    """Read and validate one JSONL split without importing training packages."""
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
            examples.append(validate_example(parsed, location=location))
    if not examples:
        raise ValueError(f"{path}: split contains no validated examples")
    return examples


def _normalized_user_key(example: Mapping[str, Any]) -> str:
    messages = example.get("messages")
    if not isinstance(messages, list):
        return ""
    user_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )
    normalized = " ".join(user_text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _source_key(example: Mapping[str, Any]) -> tuple[str, str] | None:
    source = str(example.get("source") or "").strip()
    source_id = str(example.get("source_id") or "").strip()
    return (source, source_id) if source and source_id else None


def _source_keys(examples: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for example in examples:
        key = _source_key(example)
        if key is not None:
            keys.add(key)
    return keys


def audit_split_leakage(
    train_examples: Sequence[Mapping[str, Any]],
    val_examples: Sequence[Mapping[str, Any]],
) -> None:
    """Reject exact, request-level, or provenance overlap between held-out splits."""
    train_exact = {
        _canonical_json({"messages": ex.get("messages"), "tools": ex.get("tools")})
        for ex in train_examples
    }
    val_exact = {
        _canonical_json({"messages": ex.get("messages"), "tools": ex.get("tools")})
        for ex in val_examples
    }
    exact_overlap = train_exact & val_exact

    train_users = {key for ex in train_examples if (key := _normalized_user_key(ex))}
    val_users = {key for ex in val_examples if (key := _normalized_user_key(ex))}
    user_overlap = train_users & val_users

    train_sources = _source_keys(train_examples)
    val_sources = _source_keys(val_examples)
    source_overlap = train_sources & val_sources

    if exact_overlap or user_overlap or source_overlap:
        raise ValueError(
            "train/validation leakage detected: "
            f"exact={len(exact_overlap)}, user_requests={len(user_overlap)}, "
            f"source_ids={len(source_overlap)}"
        )


def _as_int_list(value: Any, *, field: str, location: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or any(isinstance(item, list) for item in value):
        raise ValueError(f"{location}: tokenizer returned invalid {field}")
    if not all(isinstance(item, (bool, int)) for item in value):
        raise ValueError(f"{location}: tokenizer returned non-integer {field}")
    return [int(item) for item in value]


def tokenize_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    location: str,
) -> dict[str, list[int]]:
    """Render one conversation and construct explicit assistant-only labels."""
    messages = example.get("messages")
    tools = example.get("tools") or []
    template_args: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
    }
    if tools:
        template_args["tools"] = tools
    rendered = tokenizer.apply_chat_template(messages, **template_args)
    if not isinstance(rendered, Mapping):
        raise ValueError(f"{location}: chat template did not return token fields")

    input_ids = _as_int_list(rendered.get("input_ids"), field="input_ids", location=location)
    raw_mask = rendered.get("assistant_masks")
    if raw_mask is None:
        # Some compatible tokenizer releases used the singular spelling.
        raw_mask = rendered.get("assistant_mask")
    if raw_mask is None:
        raise ValueError(
            f"{location}: chat template has no assistant token mask; "
            "refusing full-conversation loss"
        )
    assistant_mask = _as_int_list(raw_mask, field="assistant mask", location=location)
    if len(input_ids) != len(assistant_mask):
        raise ValueError(f"{location}: assistant mask length does not match input IDs")
    if not input_ids:
        raise ValueError(f"{location}: tokenizer emitted an empty sequence")
    if len(input_ids) > max_length:
        raise ValueError(
            f"{location}: {len(input_ids)} tokens exceed max length {max_length}; "
            "refusing silent truncation"
        )
    if not any(assistant_mask):
        raise ValueError(f"{location}: chat template marked no assistant tokens")

    raw_attention = rendered.get("attention_mask")
    attention_mask = (
        _as_int_list(raw_attention, field="attention_mask", location=location)
        if raw_attention is not None
        else [1] * len(input_ids)
    )
    if len(attention_mask) != len(input_ids):
        raise ValueError(f"{location}: attention mask length does not match input IDs")
    labels = [
        token if is_assistant else -100
        for token, is_assistant in zip(input_ids, assistant_mask, strict=True)
    ]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def prepare_rows(
    examples: Sequence[Mapping[str, Any]], tokenizer: Any, *, max_length: int, split: str
) -> tuple[list[dict[str, list[int]]], dict[str, float | int]]:
    rows: list[dict[str, list[int]]] = []
    lengths: list[int] = []
    trained_tokens = 0
    for index, example in enumerate(examples):
        row = tokenize_example(
            example,
            tokenizer,
            max_length=max_length,
            location=f"{split} example {index}",
        )
        rows.append(row)
        lengths.append(len(row["input_ids"]))
        trained_tokens += sum(label != -100 for label in row["labels"])
    return rows, {
        "examples": len(rows),
        "tokens": sum(lengths),
        "assistant_tokens": trained_tokens,
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 2),
    }


def _manifest_value(value: Any) -> Any:
    """Convert trainer/framework scalars into strict, portable JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _manifest_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_manifest_value(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _manifest_value(scalar())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = json.dumps(
        _manifest_value(manifest),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(path, payload + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("NORAX_TRAIN_MODEL", MODEL_NAME))
    parser.add_argument("--train", type=Path, default=DATASET_PATH)
    parser.add_argument("--val", type=Path, default=VAL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--load-bits", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--optimizer", default="adamw_8bit")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--allow-no-validation", action="store_true")
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="perform structural and split-leakage validation without ML dependencies",
    )
    parser.add_argument("--export-merged", action="store_true")
    parser.add_argument("--export-gguf", action="store_true")
    return parser


def _load_training_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        # Unsloth must be imported before Transformers/TRL so its runtime patches
        # are applied consistently in supported training environments.
        unsloth = importlib.import_module("unsloth")

        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install a compatible CUDA PyTorch, "
            "Unsloth, Transformers, Datasets, and TRL stack"
        ) from exc
    FastLanguageModel = unsloth.FastLanguageModel
    return FastLanguageModel, torch, Dataset, SFTConfig, SFTTrainer


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not args.train.is_file():
        raise ValueError(f"training split not found: {args.train}")
    if not args.val.is_file() and not args.allow_no_validation:
        raise ValueError(
            f"validation split not found: {args.val}; "
            "use --allow-no-validation explicitly to proceed"
        )


def run(args: argparse.Namespace) -> int:
    _validate_arguments(args)
    train_examples = load_examples(args.train)
    val_examples = load_examples(args.val) if args.val.is_file() else []
    if val_examples:
        audit_split_leakage(train_examples, val_examples)

    source_summary: dict[str, Any] = {
        "train": {
            "path": str(args.train.resolve()),
            "sha256": _sha256_file(args.train),
            "examples": len(train_examples),
        },
        "validation": None,
    }
    if val_examples:
        source_summary["validation"] = {
            "path": str(args.val.resolve()),
            "sha256": _sha256_file(args.val),
            "examples": len(val_examples),
        }

    print(
        f"Structurally validated {len(train_examples)} train and "
        f"{len(val_examples)} validation examples"
    )
    if args.validate_data_only:
        print("Structural-only validation passed; target-tokenizer rendering was not checked")
        return 0

    FastLanguageModel, torch, Dataset, SFTConfig, SFTTrainer = _load_training_stack()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; this Unsloth training configuration requires a CUDA GPU"
        )

    device = torch.cuda.get_device_properties(0)
    print(f"Model: {args.model}")
    print(f"GPU: {device.name} ({device.total_memory / 1e9:.1f} GB visible)")
    print(f"Output: {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "loading_model",
        "started_at": _utc_now(),
        "model": args.model,
        "sources": source_summary,
        "configuration": {
            "max_length": args.max_length,
            "load_bits": args.load_bits,
            "epochs": args.epochs,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "optimizer": args.optimizer,
            "seed": SEED,
            "export_merged": args.export_merged,
            "export_gguf": args.export_gguf,
        },
        "device": {"name": device.name, "visible_memory_bytes": device.total_memory},
    }
    _write_manifest(manifest_path, manifest)

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=args.load_bits == 4,
            load_in_8bit=args.load_bits == 8,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=LORA_TARGETS,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
            use_rslora=False,
            loftq_config=None,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            raise ValueError("target tokenizer has neither a pad token nor an EOS token")
        tokenizer.padding_side = "right"

        train_rows, train_stats = prepare_rows(
            train_examples, tokenizer, max_length=args.max_length, split="train"
        )
        val_rows: list[dict[str, list[int]]] = []
        val_stats: dict[str, float | int] | None = None
        if val_examples:
            val_rows, val_stats = prepare_rows(
                val_examples, tokenizer, max_length=args.max_length, split="validation"
            )
        manifest["dataset"] = {"train": train_stats, "validation": val_stats}
        manifest["status"] = "training"
        _write_manifest(manifest_path, manifest)

        train_data = Dataset.from_list(train_rows)
        val_data = Dataset.from_list(val_rows) if val_rows else None
        use_bf16 = bool(torch.cuda.is_bf16_supported())
        training_args = SFTConfig(
            output_dir=str(args.output),
            max_length=args.max_length,
            dataset_kwargs={"skip_prepare_dataset": True},
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=args.epochs,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=SCHEDULER,
            logging_steps=LOG_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=2,
            eval_strategy="steps" if val_data is not None else "no",
            eval_steps=EVAL_STEPS if val_data is not None else None,
            bf16=use_bf16,
            fp16=not use_bf16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=args.optimizer,
            seed=SEED,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
        )

        reserved_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        allocated_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(
            f"GPU memory before training: {reserved_memory:.0f} MB reserved, "
            f"{allocated_memory:.0f} MB allocated"
        )
        resume = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        trainer_stats = trainer.train(resume_from_checkpoint=resume)

        model.save_pretrained(str(args.output))
        tokenizer.save_pretrained(str(args.output))
        exports: dict[str, str] = {"adapters": str(args.output)}
        if args.export_merged:
            merged_dir = args.output.parent / f"{args.output.name}-merged"
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
            exports["merged_16bit"] = str(merged_dir)
        if args.export_gguf:
            gguf_dir = args.output.parent / f"{args.output.name}-gguf"
            model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q4_k_m")
            exports["gguf_q4_k_m"] = str(gguf_dir)

        metrics = dict(trainer_stats.metrics)
        manifest.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "global_step": trainer_stats.global_step,
                "training_loss": trainer_stats.training_loss,
                "metrics": metrics,
                "exports": exports,
            }
        )
        _write_manifest(manifest_path, manifest)
        print(
            f"Training complete: steps={trainer_stats.global_step}, "
            f"loss={trainer_stats.training_loss:.4f}"
        )
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
            }
        )
        _write_manifest(manifest_path, manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
