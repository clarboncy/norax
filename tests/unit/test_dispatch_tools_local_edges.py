from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from norax.dispatch import tools


@pytest.fixture(autouse=True)
def _clear_read_cache() -> Iterator[None]:
    tools._READ_CACHE.clear()
    yield
    tools._READ_CACHE.clear()


@pytest.mark.asyncio
async def test_repo_explore_forwards_all_arguments(monkeypatch) -> None:
    async def explore(query: str, *, root: str | None, budget: int | None) -> dict:
        return {"ok": True, "query": query, "root": root, "budget": budget}

    monkeypatch.setattr(tools, "explore_repo", explore)
    assert await tools.t_repo_explore(query="parser", root="/repo", budget=42) == {
        "ok": True,
        "query": "parser",
        "root": "/repo",
        "budget": 42,
    }


@pytest.mark.asyncio
async def test_read_rejects_directories_bounds_pages_and_evicts_lru(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert (await tools.t_read(path=str(tmp_path)))["error"] == "is_a_directory"

    large = tmp_path / "many-lines.txt"
    large.write_text("\n".join(str(index) for index in range(tools._READ_MAX_PAGE_LINES + 1)))
    result = await tools.t_read(path=str(large), limit=10**9)
    assert result["page_limit"] == tools._READ_MAX_PAGE_LINES
    assert result["next_offset"] == tools._READ_MAX_PAGE_LINES
    assert result["truncated"] is True

    monkeypatch.setattr(tools, "_READ_CACHE_MAX", 1)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("second")
    await tools.t_read(path=str(first))
    await tools.t_read(path=str(second))
    assert len(tools._READ_CACHE) == 1
    assert next(iter(tools._READ_CACHE))[0] == str(second)


@pytest.mark.asyncio
async def test_read_and_list_stat_failures_are_structured_and_scrubbed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target"
    secret = "sk-" + "s" * 30
    original_stat = Path.stat

    def fail_target(self: Path, *args: Any, **kwargs: Any):
        if self == target:
            raise OSError(secret)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_target)
    read_result = await tools.t_read(path=str(target))
    list_result = await tools.t_list_dir(path=str(target))
    assert read_result["error"] == "file_stat_failed"
    assert list_result["error"] == "directory_stat_failed"
    assert secret not in read_result["detail"]
    assert secret not in list_result["detail"]
    assert read_result["detail"] == "<REDACTED:openai_key>"


@pytest.mark.asyncio
async def test_read_bounds_pathological_lines_and_total_page_text(tmp_path: Path) -> None:
    long_line = tmp_path / "long-line.txt"
    long_line.write_text("x" * (tools._READ_MAX_LINE_CHARS + 1))
    line_result = await tools.t_read(path=str(long_line), limit=1)
    assert line_result["ok"] is True
    assert len(line_result["content"]) == tools._READ_MAX_LINE_CHARS
    assert line_result["line_truncated_at"] == 0
    assert line_result["line_char_limit"] == tools._READ_MAX_LINE_CHARS
    assert line_result["truncated"] is True
    assert "next_offset" not in line_result

    bounded_page = tmp_path / "bounded-page.txt"
    line = "y" * (tools._READ_MAX_LINE_CHARS - 1)
    bounded_page.write_text("\n".join([line] * 20))
    page_result = await tools.t_read(path=str(bounded_page), limit=20)
    assert page_result["ok"] is True
    assert len(page_result["content"]) <= tools._READ_MAX_PAGE_CHARS
    assert page_result["page_char_limit"] == tools._READ_MAX_PAGE_CHARS
    assert page_result["truncated"] is True
    assert page_result["next_offset"] == len(page_result["content"].splitlines())


@pytest.mark.asyncio
async def test_read_contains_race_or_permission_failure(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target.txt"
    target.write_text("content")
    secret = "sk-" + "r" * 30

    def fail(*_args: Any, **_kwargs: Any):
        raise OSError(secret)

    monkeypatch.setattr(tools, "_read_text_page", fail)
    result = await tools.t_read(path=str(target))
    assert result["error"] == "file_read_failed"
    assert secret not in result["detail"]
    assert result["detail"] == "<REDACTED:openai_key>"


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, 7, 7),
        (True, 7, 7),
        (3, None, 3),
        (3.0, None, 3),
        (3.5, 7, 7),
        (float("nan"), 7, 7),
        ("", 7, 7),
        ("4", None, 4),
        ("4.0", None, 4),
        ("4.5", 7, 7),
        ("nan", 7, 7),
        ("bad", 7, 7),
        (object(), 7, 7),
    ],
)
def test_coerce_int_all_input_shapes(
    value: object, default: int | None, expected: int | None
) -> None:
    assert tools._coerce_int(value, default=default) == expected


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, 7.0, 7.0),
        (True, 7.0, 7.0),
        (3, None, 3.0),
        (float("inf"), 7.0, 7.0),
        ("", 7.0, 7.0),
        ("4.5", None, 4.5),
        ("nan", 7.0, 7.0),
        ("bad", 7.0, 7.0),
        (object(), 7.0, 7.0),
    ],
)
def test_coerce_float_all_input_shapes(
    value: object,
    default: float | None,
    expected: float | None,
) -> None:
    assert tools._coerce_float(value, default=default) == expected


def test_placeholder_and_argument_normalization_edge_shapes() -> None:
    assert tools.exec_command_is_placeholder("") is True
    assert tools.exec_command_is_placeholder("your custom command please") is True
    assert tools.exec_command_is_placeholder("printf done") is False

    assert tools.normalize_tool_args("exec", {"input": "printf nested"})["command"] == (
        "printf nested"
    )
    assert "command" not in tools.normalize_tool_args("exec", {"command": []})
    assert "command" not in tools.normalize_tool_args("exec", {"command": ["ok", 1]})
    assert "command" not in tools.normalize_tool_args("exec", {"command": 42})
    assert (
        tools.normalize_tool_args("exec", {"command": ["printf", "%s", "ok"]})["command"]
        == "printf %s ok"
    )
    assert (
        tools.normalize_tool_args("exec", {"command": "true", "timeout": "1.5"})["timeout"] == 1.5
    )
    assert "timeout" not in tools.normalize_tool_args("exec", {"command": "true", "timeout": "bad"})
    assert tools.normalize_tool_args("web_fetch", {"max_chars": "2048"})["max_chars"] == 2048
    assert tools.normalize_tool_args("web_search", {"count": "4"})["count"] == 4
    assert tools.normalize_tool_args("web_search", {"recency_days": "7"})["recency_days"] == 7
    assert "recency_days" not in tools.normalize_tool_args("web_search", {"recency_days": "bad"})


def test_directory_names_hard_caps_pathological_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("c", "a", "b"):
        (tmp_path / name).touch()
    monkeypatch.setattr(tools, "_LIST_DIR_SCAN_LIMIT", 2)
    names, truncated = tools._directory_names(tmp_path)
    assert names == sorted(names)
    assert len(names) == 2
    assert truncated is True


@pytest.mark.asyncio
async def test_list_dir_handles_path_scan_and_child_metadata_edges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert (await tools.t_list_dir(path=str(tmp_path / "missing")))["error"] == "path_not_found"
    regular = tmp_path / "file.txt"
    regular.write_text("data")
    assert (await tools.t_list_dir(path=str(regular)))["error"] == "not_a_directory"

    def denied(_path: Path) -> tuple[list[str], bool]:
        raise PermissionError("denied")

    monkeypatch.setattr(tools, "_directory_names", denied)
    assert (await tools.t_list_dir(path=str(tmp_path)))["error"] == "permission_denied"

    secret = "sk-" + "e" * 30

    def failed(_path: Path) -> tuple[list[str], bool]:
        raise OSError(secret)

    monkeypatch.setattr(tools, "_directory_names", failed)
    failure = await tools.t_list_dir(path=str(tmp_path))
    assert failure["error"] == "directory_scan_failed"
    assert secret not in failure["detail"]


@pytest.mark.asyncio
async def test_list_dir_reports_other_unknown_and_both_truncation_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    dangling = tmp_path / "dangling"
    dangling.symlink_to("missing-target")
    result = await tools.t_list_dir(path=str(tmp_path))
    by_name = {entry["name"]: entry for entry in result["entries"]}
    assert by_name["fifo"] == {"name": "fifo", "kind": "other", "size": None}
    assert by_name["dangling"] == {"name": "dangling", "kind": "unknown", "size": None}

    for name in ("a", "b", "c"):
        (tmp_path / name).touch()
    monkeypatch.setattr(tools, "_LIST_DIR_SCAN_LIMIT", 2)
    partial = await tools.t_list_dir(path=str(tmp_path), limit=1)
    assert partial["scan_truncated"] is True
    assert partial["truncated"] is True
    assert partial["next_offset"] == 1

    full_scan_page = await tools.t_list_dir(path=str(tmp_path), limit=2)
    assert full_scan_page["scan_truncated"] is True
    assert full_scan_page["truncated"] is True
    assert "next_offset" not in full_scan_page
    assert full_scan_page["total_entries_at_least"] == 3


@pytest.mark.asyncio
async def test_list_dir_accepts_weak_model_numeric_strings(tmp_path: Path) -> None:
    for name in ("a", "b"):
        (tmp_path / name).touch()
    result = await tools.t_list_dir(
        path=str(tmp_path),
        offset=cast(Any, "1"),
        limit=cast(Any, "1"),
    )
    expected = sorted(child.name for child in tmp_path.iterdir())[1]
    assert result["offset"] == 1
    assert [entry["name"] for entry in result["entries"]] == [expected]
