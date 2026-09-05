from __future__ import annotations

import json

import pytest

from norax.dispatch.tools import t_computer_use, t_gateway_config_patch, t_list_dir, t_read


@pytest.mark.asyncio
async def test_gateway_config_patch_uses_active_config_and_preserves_urls(tmp_path, monkeypatch):
    config = tmp_path / "runtime.jsonc"
    config.write_text(
        '{\n  // comment\n  "gateway": {"base_url": "https://example.test/v1", "timeout_seconds": 10}\n}\n'
    )
    monkeypatch.setenv("NORAX_CONFIG", str(config))
    result = await t_gateway_config_patch(patch={"gateway": {"timeout_seconds": 60}})
    assert result["ok"] and result["restart_required"]
    loaded = json.loads(config.read_text())
    assert loaded["gateway"] == {
        "base_url": "https://example.test/v1",
        "timeout_seconds": 60,
    }
    assert result["patched_keys"] == ["gateway.timeout_seconds"]


@pytest.mark.asyncio
async def test_gateway_config_patch_rejects_non_gateway_and_dead_settings(tmp_path, monkeypatch):
    config = tmp_path / "runtime.jsonc"
    original = '{"gateway":{"base_url":"https://example.test/v1"}}\n'
    config.write_text(original)
    monkeypatch.setenv("NORAX_CONFIG", str(config))

    wrong_scope = await t_gateway_config_patch(patch={"runtime": {"max_tool_rounds": 48}})
    dead_setting = await t_gateway_config_patch(patch={"gateway": {"timeout": 60}})
    invalid_value = await t_gateway_config_patch(
        patch={"gateway": {"stream_required": "definitely"}}
    )

    assert wrong_scope["ok"] is False
    assert dead_setting["ok"] is False
    assert "unsupported settings" in dead_setting["error"]
    assert invalid_value["ok"] is False
    assert "must be a boolean" in invalid_value["error"]
    assert config.read_text() == original


@pytest.mark.asyncio
async def test_computer_drag_accepts_start_and_end_coordinates(monkeypatch):
    captured: list[tuple[str, ...]] = []

    class FakeProcess:
        pid = 12345
        returncode = 0

        async def communicate(self):
            return b'{"ok": true}', b""

    async def fake_subprocess(*args, **_kwargs):
        captured.append(tuple(str(arg) for arg in args))
        return FakeProcess()

    monkeypatch.setattr("norax.dispatch.tools.asyncio.create_subprocess_exec", fake_subprocess)
    result = await t_computer_use(action="drag", x=1, y=2, x2=30, y2=40)
    assert result == {"ok": True}
    assert captured[0][-5:] == ("drag", "1", "2", "30", "40")


@pytest.mark.asyncio
async def test_computer_drag_rejects_incomplete_coordinates():
    result = await t_computer_use(action="drag", x=1, y=2)
    assert result["ok"] is False
    assert "x2" in result["error"]


@pytest.mark.asyncio
async def test_computer_mutations_reject_missing_or_ambiguous_inputs():
    assert not (await t_computer_use(action="click", x=1))["ok"]
    assert not (await t_computer_use(action="move", y=1))["ok"]
    assert not (await t_computer_use(action="key_press"))["ok"]
    assert not (await t_computer_use(action="click", x=1, y=2, button="side"))["ok"]


@pytest.mark.asyncio
async def test_large_read_auto_paginates_instead_of_blocking(tmp_path):
    target = tmp_path / "large.txt"
    target.write_text(
        "".join(f"line-{index:05d} payload payload payload\n" for index in range(15000))
    )

    first = await t_read(path=str(target))
    second = await t_read(path=str(target), offset=1000, limit=2)

    assert first["ok"] is True
    assert first["auto_paginated"] is True
    assert first["truncated"] is True
    assert first["next_offset"] == 2000
    assert second["ok"] is True
    assert second["content"].splitlines() == [
        "line-01000 payload payload payload",
        "line-01001 payload payload payload",
    ]


@pytest.mark.asyncio
async def test_very_large_read_does_not_scan_to_eof_for_first_page(tmp_path):
    target = tmp_path / "very-large.txt"
    target.write_text("line payload\n" * 700_000)

    result = await t_read(path=str(target), limit=2)

    assert result["ok"] is True
    assert result["content"] == "line payload\nline payload"
    assert result["total_lines"] is None
    assert result["total_lines_at_least"] == 3
    assert result["truncated"] is True
    assert result["next_offset"] == 2


@pytest.mark.asyncio
async def test_read_cache_cannot_be_poisoned_by_first_result_mutation(tmp_path):
    target = tmp_path / "cached.txt"
    target.write_text("original")

    first = await t_read(path=str(target))
    first["content"] = "poisoned"
    second = await t_read(path=str(target))

    assert second["content"] == "original"
    assert second["cached"] is True


@pytest.mark.asyncio
async def test_read_rejects_fifo_without_opening_it(tmp_path):
    target = tmp_path / "stream"
    target.parent.mkdir(parents=True, exist_ok=True)
    import os

    os.mkfifo(target)

    result = await t_read(path=str(target))

    assert result["ok"] is False
    assert result["error"] == "unsupported_file_type"


@pytest.mark.asyncio
async def test_list_dir_is_sorted_and_paginated(tmp_path):
    listing = tmp_path / "listing"
    listing.mkdir()
    for index in range(7):
        (listing / f"file-{index}.txt").write_text(str(index))

    first = await t_list_dir(path=str(listing), limit=3)
    second = await t_list_dir(path=str(listing), offset=3, limit=3)

    assert [entry["name"] for entry in first["entries"]] == [
        "file-0.txt",
        "file-1.txt",
        "file-2.txt",
    ]
    assert first["total_entries"] == 7
    assert first["next_offset"] == 3
    assert [entry["name"] for entry in second["entries"]] == [
        "file-3.txt",
        "file-4.txt",
        "file-5.txt",
    ]
    assert second["next_offset"] == 6


@pytest.mark.asyncio
async def test_list_dir_rejects_invalid_pagination(tmp_path):
    assert not (await t_list_dir(path=str(tmp_path), offset=-1))["ok"]
    assert not (await t_list_dir(path=str(tmp_path), limit=1.5))["ok"]
