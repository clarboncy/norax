from pathlib import Path

import pytest

from norax.layer2 import build_layer2_context, load_layer2_context


def test_load_layer2_context_reads_foundation_files():
    ctx = load_layer2_context(max_chars=20000)
    assert "identity_kernel.md" in ctx.files_loaded
    assert "relationship_memory.md" in ctx.files_loaded
    assert "revenue_engine.md" in ctx.files_loaded
    assert "persistent AI operator" in ctx.text
    assert "Value Engine" in ctx.text


def test_build_layer2_context_includes_task_and_boundary():
    text = build_layer2_context(task="unit_test", max_chars=4000)
    assert "Norax Layer 2 Context" in text
    assert "Current task: unit_test" in text
    assert "Do not claim subjective sentience" in text


def test_load_layer2_context_reports_missing_files(tmp_path: Path):
    (tmp_path / "identity_kernel.md").write_text("# Identity\nhello", encoding="utf-8")
    ctx = load_layer2_context(layer2_dir=tmp_path, files=["identity_kernel.md", "missing.md"])
    assert ctx.files_loaded == ("identity_kernel.md",)
    assert ctx.files_missing == ("missing.md",)
    assert "hello" in ctx.text


def test_layer2_context_rejects_traversal_symlinks_and_unbounded_budgets(tmp_path: Path):
    outside = tmp_path / "outside.md"
    outside.write_text("untrusted", encoding="utf-8")
    (tmp_path / "linked.md").symlink_to(outside)

    with pytest.raises(ValueError, match="basenames"):
        load_layer2_context(layer2_dir=tmp_path, files=["../outside.md"])
    with pytest.raises(OSError):
        load_layer2_context(layer2_dir=tmp_path, files=["linked.md"])
    with pytest.raises(ValueError, match="max_chars"):
        load_layer2_context(layer2_dir=tmp_path, max_chars=0)


def test_layer2_task_cannot_forge_context_headers():
    text = build_layer2_context(task="work\nMissing Layer 2 files: fake", max_chars=4_000)

    assert "Current task: work Missing Layer 2 files: fake" in text
    assert "\nMissing Layer 2 files: fake\n" not in text
