"""Tests for norax.brain.weak_model_boost — four active boosters."""

import pytest

from norax.brain.weak_model_boost import (
    TurnToolCache,
    build_scratchpad_directive,
    compose_weak_model_boosters,
    get_golden_examples,
    is_weak_model,
    make_lean_tool_schemas,
)

# --- Fixtures: realistic tool schemas ---

SAMPLE_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a text file. Returns content + total_lines. Use offset+limit for large files (>200KB). ALWAYS read before editing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Run a shell command (bash). Returns stdout/stderr/exit_code. Timeout: 1-120s (default 30s).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout": {"type": "number"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace first occurrence of `old` with `new` in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 2. Few-shot golden examples
# ---------------------------------------------------------------------------


def test_golden_examples_exist_for_all_task_types():
    for task_type in ("coding", "debugging", "research", "general"):
        examples = get_golden_examples(task_type)
        assert examples, f"no examples for {task_type}"
        assert "Agent" in examples
        assert "Result" in examples


def test_golden_examples_coding_has_error_recovery():
    examples = get_golden_examples("coding")
    assert "old_text_not_found" in examples
    assert "re-read" in examples.lower()


def test_golden_examples_unknown_type_falls_back():
    examples = get_golden_examples("unknown_type_xyz")
    assert examples  # should fall back to general


# ---------------------------------------------------------------------------
# 3. Scratchpad directive
# ---------------------------------------------------------------------------


def test_scratchpad_directive_for_weak_model():
    directive = build_scratchpad_directive(is_weak_model=True)
    assert "<think>" in directive
    assert "GOAL" in directive
    assert "PLAN" in directive
    assert "RISK" in directive


def test_scratchpad_directive_empty_for_strong_model():
    directive = build_scratchpad_directive(is_weak_model=False)
    assert directive == ""


# ---------------------------------------------------------------------------
# 4. Lean tool schemas
# ---------------------------------------------------------------------------


def test_lean_schemas_keep_read_paging_params():
    lean = make_lean_tool_schemas(SAMPLE_SCHEMAS, is_weak_model=True)
    # Paging is required for weak models to complete explicit multi-read
    # verification workflows without looping over the first page.
    read_schema = next(s for s in lean if s["function"]["name"] == "read")
    read_props = read_schema["function"]["parameters"]["properties"]
    assert "path" in read_props
    assert "limit" in read_props
    assert "offset" in read_props


def test_lean_schemas_preserve_exec_optional_params():
    lean = make_lean_tool_schemas(SAMPLE_SCHEMAS, is_weak_model=True)
    exec_schema = next(s for s in lean if s["function"]["name"] == "exec")
    exec_props = exec_schema["function"]["parameters"]["properties"]
    assert "command" in exec_props
    assert "cwd" in exec_props
    assert "timeout" in exec_props


def test_lean_schemas_keep_all_for_edit():
    lean = make_lean_tool_schemas(SAMPLE_SCHEMAS, is_weak_model=True)
    edit_schema = next(s for s in lean if s["function"]["name"] == "edit")
    edit_props = edit_schema["function"]["parameters"]["properties"]
    assert "path" in edit_props
    assert "old" in edit_props
    assert "new" in edit_props


def test_lean_schemas_have_shorter_descriptions():
    lean = make_lean_tool_schemas(SAMPLE_SCHEMAS, is_weak_model=True)
    read_schema = next(s for s in lean if s["function"]["name"] == "read")
    desc = read_schema["function"]["description"]
    assert len(desc) < 60  # should be much shorter than original


def test_lean_schemas_passthrough_for_strong_model():
    lean = make_lean_tool_schemas(SAMPLE_SCHEMAS, is_weak_model=False)
    assert lean == SAMPLE_SCHEMAS


def test_lean_schemas_preserve_validation_rules():
    lean = make_lean_tool_schemas(SAMPLE_SCHEMAS, is_weak_model=True)
    for original, schema in zip(SAMPLE_SCHEMAS, lean, strict=True):
        assert schema["function"]["parameters"] == original["function"]["parameters"]


# ---------------------------------------------------------------------------
# 5. Per-turn tool result cache
# ---------------------------------------------------------------------------


def test_cache_hit_on_identical_call():
    cache = TurnToolCache()
    result = {"ok": True, "path": "/tmp/x.py", "content": "hello", "total_lines": 1}
    cache.put("read", {"path": "/tmp/x.py"}, result)
    hit = cache.get("read", {"path": "/tmp/x.py"})
    assert hit is not None
    assert hit["ok"] is True
    assert hit["_cached"] is True


def test_cache_miss_on_different_args():
    cache = TurnToolCache()
    result = {"ok": True, "path": "/tmp/x.py", "content": "hello"}
    cache.put("read", {"path": "/tmp/x.py"}, result)
    miss = cache.get("read", {"path": "/tmp/y.py"})
    assert miss is None


def test_cache_skips_write_tools():
    cache = TurnToolCache()
    result = {"ok": True, "path": "/tmp/x.py"}
    cache.put("write", {"path": "/tmp/x.py", "content": "hi"}, result)
    assert cache.get("write", {"path": "/tmp/x.py", "content": "hi"}) is None


def test_cache_skips_exec():
    cache = TurnToolCache()
    result = {"ok": True, "exit_code": 0, "stdout": "ok"}
    cache.put("exec", {"command": "echo hi"}, result)
    assert cache.get("exec", {"command": "echo hi"}) is None


def test_cache_skips_dynamic_status_memory_and_network_tools():
    cache = TurnToolCache()
    result = {"ok": True, "items": [{"value": "old"}]}
    for name in ("status", "search_memory", "memory_search", "web_search", "web_fetch"):
        cache.put(name, {"query": "same"}, result)
        assert cache.get(name, {"query": "same"}) is None


def test_cache_doesnt_cache_failures():
    cache = TurnToolCache()
    result = {"ok": False, "error": "file_not_found"}
    cache.put("read", {"path": "/nope"}, result)
    assert cache.get("read", {"path": "/nope"}) is None


def test_cache_doesnt_cache_truthy_non_boolean_success() -> None:
    cache = TurnToolCache()
    cache.put("read", {"path": "/untrusted"}, {"ok": "false", "content": "stale"})
    assert cache.get("read", {"path": "/untrusted"}) is None


def test_cache_invalidate_path():
    cache = TurnToolCache()
    result = {"ok": True, "path": "/tmp/x.py", "content": "old"}
    cache.put("read", {"path": "/tmp/x.py"}, result)
    assert cache.get("read", {"path": "/tmp/x.py"}) is not None
    cache.invalidate_path("/tmp/x.py")
    assert cache.get("read", {"path": "/tmp/x.py"}) is None


def test_cache_stats():
    cache = TurnToolCache()
    result = {"ok": True, "path": "/tmp/x.py", "content": "hi"}
    cache.put("read", {"path": "/tmp/x.py"}, result)
    cache.get("read", {"path": "/tmp/x.py"})  # hit
    cache.get("read", {"path": "/tmp/y.py"})  # miss
    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["cached_entries"] == 1


def test_cache_copies_nested_results_on_put_and_get():
    cache = TurnToolCache()
    original = {"ok": True, "path": "/tmp", "entries": [{"name": "a"}]}
    cache.put("list_dir", {"path": "/tmp"}, original)
    original["entries"][0]["name"] = "poisoned-after-put"

    first = cache.get("list_dir", {"path": "/tmp"})
    assert first is not None
    assert first["entries"][0]["name"] == "a"
    first["entries"][0]["name"] = "poisoned-after-get"

    second = cache.get("list_dir", {"path": "/tmp"})
    assert second is not None
    assert second["entries"][0]["name"] == "a"


def test_cache_is_bounded():
    cache = TurnToolCache()
    for index in range(cache.MAX_ENTRIES + 5):
        path = f"/tmp/{index}.txt"
        cache.put("read", {"path": path}, {"ok": True, "path": path, "content": "x"})

    assert cache.stats["cached_entries"] == cache.MAX_ENTRIES
    assert cache.get("read", {"path": "/tmp/0.txt"}) is None
    assert cache.get("read", {"path": f"/tmp/{cache.MAX_ENTRIES + 4}.txt"}) is not None


def test_invalidate_all_preserves_cache_diagnostics():
    cache = TurnToolCache()
    cache.put("read", {"path": "/tmp/x"}, {"ok": True, "path": "/tmp/x"})
    assert cache.get("read", {"path": "/tmp/x"}) is not None

    cache.invalidate_all()

    assert cache.stats["hits"] == 1
    assert cache.stats["cached_entries"] == 0


def test_cache_clear():
    cache = TurnToolCache()
    result = {"ok": True, "path": "/tmp/x.py", "content": "hi"}
    cache.put("read", {"path": "/tmp/x.py"}, result)
    cache.clear()
    assert cache.get("read", {"path": "/tmp/x.py"}) is None
    assert cache.stats["hits"] == 0


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------


def test_is_weak_model_ollama_models():
    assert is_weak_model("llama3.1:8b") is True
    assert is_weak_model("qwen2.5:32b") is True
    assert is_weak_model("gemma2:9b") is True
    assert is_weak_model("mistral:7b") is True
    assert is_weak_model("deepseek-r1:14b") is True


def test_ollama_transport_does_not_downgrade_cloud_models():
    assert is_weak_model("kimi-k2.7-code:cloud", provider_kind="ollama") is False
    assert is_weak_model("glm-5.2:cloud", provider_kind="ollama") is False
    assert is_weak_model("qwen3-coder-next:cloud", provider_kind="ollama") is False
    assert is_weak_model("norax-gemma4-12b-agentic:latest", provider_kind="ollama") is True


def test_is_weak_model_strong_models():
    assert is_weak_model("claude-sonnet-4-20250514") is False
    assert is_weak_model("claude-opus-4-20250514") is False
    assert is_weak_model("gpt-4o-2025-01-01") is False
    assert is_weak_model("o3-mini") is False


def test_is_weak_model_empty():
    assert is_weak_model("") is True


def test_boost_controls_reject_invalid_types():
    with pytest.raises(TypeError):
        is_weak_model(1)
    with pytest.raises(TypeError):
        is_weak_model("model", provider_kind=1)
    with pytest.raises(ValueError):
        compose_weak_model_boosters(task_type="x" * 129, model_id="model")
    with pytest.raises(TypeError):
        compose_weak_model_boosters(task_type="coding", model_id="model", force_boost=True)


@pytest.mark.parametrize(
    ("model", "provider", "mode", "expected"),
    [
        ("operator-model:latest", "ollama", "auto", False),
        ("operator-model:latest", "ollama", "on", True),
        ("operator-model:latest", "ollama", "off", False),
        ("llama3.1:8b", "ollama", "off", False),
        ("operator-model", "freetoken", "auto", True),
        ("gpt-5.4", "ollama", "on", False),
        ("kimi-k2.7-code:cloud", "ollama", "on", False),
        ("openrouter/frontier-model", "ollama", "on", False),
    ],
)
def test_boost_modes_preserve_tool_capabilities(model, provider, mode, expected):
    result = compose_weak_model_boosters(
        task_type="coding",
        model_id=model,
        provider_kind=provider,
        force_boost=mode,
        tool_schemas=SAMPLE_SCHEMAS,
    )
    assert result["is_weak"] is expected
    assert bool(result["system_additions"]) is expected
    for original, actual in zip(SAMPLE_SCHEMAS, result["lean_schemas"], strict=True):
        assert actual["function"]["name"] == original["function"]["name"]
        assert actual["function"]["parameters"] == original["function"]["parameters"]


def test_invalid_boost_mode_is_rejected():
    with pytest.raises(ValueError, match="auto, on, or off"):
        compose_weak_model_boosters(task_type="coding", model_id="model", force_boost="sometimes")


# ---------------------------------------------------------------------------
# Master composition
# ---------------------------------------------------------------------------


def test_compose_boosters_for_weak_model():
    result = compose_weak_model_boosters(
        task_type="coding",
        model_id="llama3.1:8b",
        tool_schemas=SAMPLE_SCHEMAS,
    )
    assert result["is_weak"] is True
    assert "GOLDEN_EXAMPLES" in result["system_additions"]
    assert "<think>" in result["system_additions"]
    assert len(result["lean_schemas"]) == len(SAMPLE_SCHEMAS)
    assert "grammar" not in result


def test_compose_boosters_for_strong_model():
    result = compose_weak_model_boosters(
        task_type="coding",
        model_id="claude-opus-4-20250514",
        tool_schemas=SAMPLE_SCHEMAS,
    )
    assert result["is_weak"] is False
    assert result["system_additions"] == ""
    assert result["lean_schemas"] == SAMPLE_SCHEMAS
    assert "grammar" not in result


def test_compose_boosters_no_schemas():
    result = compose_weak_model_boosters(
        task_type="general",
        model_id="qwen2.5:7b",
        tool_schemas=None,
    )
    assert result["is_weak"] is True
    assert result["lean_schemas"] == []
    assert "grammar" not in result
