"""Structured-output parsing and validation helpers.

Callers can declare a Pydantic model or JSON Schema, validate a model response,
and use the returned retry prompt when they choose to make another LLM call.
This module performs no hidden model calls and is not a runtime-wide guarantee.

Design:
  - Per-task output schema declaration (Pydantic model or JSON schema dict)
  - Standards-compliant JSON Schema instance validation
  - Pydantic v2 model validation, including root models
  - A retry-prompt helper; orchestration remains the caller's responsibility
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("norax.brain.output_schema")


@dataclass
class ValidationResult:
    """Result of validating model output against a schema."""

    valid: bool
    data: Any | None = None
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""
    retries_used: int = 0
    retry_prompt: str = ""


def _extract_json_from_text(text: str) -> Any:
    """Try to extract JSON from model output text.

    Handles:
      - Pure JSON response
      - JSON in markdown code fences (```json ... ```)
      - JSON embedded in prose (first { to last })
    """
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try each code fence in order. A model may emit an example before the
    # actual payload, and one malformed fence must not hide a later valid one.
    for fence_match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            continue

    # Decode the first complete object/array embedded in prose. Slicing from
    # the first opening token to the last closing token incorrectly merges two
    # independent JSON values and rejects otherwise valid output.
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
            return value
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _validate_against_schema(data: Any, schema: dict) -> list[str]:
    """Validate an instance against a JSON Schema and return stable errors."""
    if not isinstance(schema, dict):
        return ["schema must be a dict"]
    try:
        from jsonschema.validators import validator_for

        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        validation_errors = sorted(
            validator_class(schema).iter_errors(data),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except Exception as exc:
        return [f"invalid JSON schema: {exc}"]

    errors: list[str] = []
    for error in validation_errors:
        path = "$"
        for part in error.absolute_path:
            path += f"[{part}]" if isinstance(part, int) else f".{part}"
        errors.append(f"{path}: {error.message}")
    return errors


def _validate_pydantic(data: Any, model_class: type) -> list[str]:
    """Validate data against a Pydantic model class."""
    try:
        model_validate = getattr(model_class, "model_validate", None)
        if callable(model_validate):
            model_validate(data)
        elif isinstance(data, dict):
            model_class(**data)
        else:
            model_class(data)
        return []  # valid
    except Exception as e:
        # Extract field-level errors
        errors: list[str] = []
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", str(e))
                errors.append(f"{loc}: {msg}")
        else:
            errors.append(str(e))
        return errors


def validate_output(
    raw_output: str,
    schema: dict | type | None = None,
) -> ValidationResult:
    """Validate model output against a schema.

    Args:
        raw_output: The model's text response
        schema: JSON schema dict OR Pydantic model class OR None (no validation)

    Returns:
        ValidationResult with valid flag, parsed data, and errors
    """
    if schema is None:
        return ValidationResult(valid=True, data={"text": raw_output}, raw_output=raw_output)

    # Extract JSON from text
    parsed = _extract_json_from_text(raw_output)
    if parsed is None:
        return ValidationResult(
            valid=False,
            errors=["could not extract JSON from output"],
            raw_output=raw_output,
        )

    # Validate
    if isinstance(schema, dict):
        errors = _validate_against_schema(parsed, schema)
    elif isinstance(schema, type):
        errors = _validate_pydantic(parsed, schema)
    else:
        errors = ["schema must be a dict or Pydantic class"]

    return ValidationResult(
        valid=len(errors) == 0,
        data=parsed if not errors else None,
        errors=errors,
        raw_output=raw_output,
    )


def build_retry_prompt(errors: list[str], original_output: str) -> str:
    """Build a retry prompt that tells the model what went wrong."""
    error_text = "\n".join(f"  - {e}" for e in errors)
    return (
        f"Your previous response failed validation:\n{error_text}\n\n"
        f"Please provide a corrected JSON response that matches the required schema. "
        f"Output ONLY valid JSON, no markdown or prose.\n\n"
        f"Your previous output was:\n{original_output[:500]}"
    )


def validate_with_retry(
    raw_output: str,
    schema: dict | type | None = None,
    *,
    max_retries: int = 2,
) -> ValidationResult:
    """Validate output and prepare a retry prompt if requested.

    Note: This function only validates and prepares retry prompts.
    This helper does not call an LLM. The caller decides whether and how to
    retry, and should increment ``retries_used`` after an actual retry.

    Returns ValidationResult with retries_used=0 (caller increments on retry).
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    result = validate_output(raw_output, schema)
    if result.valid or max_retries == 0:
        return result

    result.retry_prompt = build_retry_prompt(result.errors, raw_output)
    return result


# Pre-built schemas for common task types

SCHEMA_TASK_RESULT = {
    "type": "object",
    "required": ["ok", "summary"],
    "properties": {
        "ok": {"type": "boolean"},
        "summary": {"type": "string"},
        "details": {"type": "string"},
        "files_touched": {"type": "array", "items": {"type": "string"}},
        "errors": {"type": "array", "items": {"type": "string"}},
    },
}

SCHEMA_RESEARCH = {
    "type": "object",
    "required": ["findings"],
    "properties": {
        "findings": {"type": "array", "items": {"type": "string"}},
        "sources": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
}

SCHEMA_CODE_REVIEW = {
    "type": "object",
    "required": ["issues"],
    "properties": {
        "issues": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "string"},
        "suggestions": {"type": "array", "items": {"type": "string"}},
        "approved": {"type": "boolean"},
    },
}
