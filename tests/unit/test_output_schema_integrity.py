from __future__ import annotations

from pydantic import BaseModel, RootModel

from norax.brain.output_schema import validate_output, validate_with_retry


def test_json_schema_validates_nested_array_items():
    schema = {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["count"],
                    "properties": {"count": {"type": "integer", "minimum": 1}},
                },
            }
        },
    }

    result = validate_output('{"items": [{"count": 0}]}', schema)
    assert not result.valid
    assert any("$.items[0].count" in error for error in result.errors)


def test_json_schema_does_not_accept_boolean_as_integer():
    result = validate_output(
        '{"count": true}',
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        },
    )
    assert not result.valid


def test_embedded_json_decoder_does_not_merge_independent_objects():
    result = validate_output(
        'Result: {"ok": true}. Metadata: {"ignored": true}',
        {
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"const": True}},
        },
    )
    assert result.valid
    assert result.data == {"ok": True}


def test_invalid_schema_is_reported_not_silently_accepted():
    result = validate_output('{"x": 1}', {"type": "not-a-real-type"})
    assert not result.valid
    assert result.errors[0].startswith("invalid JSON schema:")


class _Payload(BaseModel):
    count: int


class _Names(RootModel[list[str]]):
    pass


def test_pydantic_models_and_root_models_use_model_validate():
    assert validate_output('{"count": 2}', _Payload).valid
    assert not validate_output('{"count": "not-an-int"}', _Payload).valid
    assert validate_output('["a", "b"]', _Names).valid


def test_retry_prompt_is_separate_from_validation_errors():
    result = validate_with_retry("not json", {"type": "object"})
    assert not result.valid
    assert result.errors == ["could not extract JSON from output"]
    assert "failed validation" in result.retry_prompt
