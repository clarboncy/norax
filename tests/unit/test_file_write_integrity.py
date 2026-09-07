import asyncio
import stat

import pytest

from benchmarks.agent_capability_bench import confined_tool
from norax.dispatch import tools
from scripts.verify_agent_harness import _confined_exec_tool, _confined_file_tool


@pytest.mark.asyncio
async def test_write_preserves_modes_follows_alias_and_reports_utf8_bytes(tmp_path):
    target = tmp_path / "script"
    target.write_text("old")
    target.chmod(0o750)
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    result = await tools.t_write(path=str(alias), content="café\n")
    assert result["ok"] is True
    assert result["bytes"] == 6
    assert result["lines"] == 1
    assert target.read_text() == "café\n"
    assert target.stat().st_mode & 0o777 == 0o750
    assert alias.is_symlink()


@pytest.mark.asyncio
async def test_write_failure_preserves_old_contents(tmp_path, monkeypatch):
    import norax.atomic as atomic

    target = tmp_path / "existing"
    target.write_text("keep")

    def fail(*args):
        raise OSError("simulated commit failure")

    monkeypatch.setattr(atomic.os, "replace", fail)
    with pytest.raises(OSError, match="commit failure"):
        await tools.t_write(path=str(target), content="replacement")
    assert target.read_text() == "keep"
    assert list(tmp_path.glob(".existing.*")) == []


@pytest.mark.asyncio
async def test_live_benchmark_tool_cannot_write_outside_its_artifact(tmp_path):
    artifact = tmp_path / "proof.txt"
    tool = confined_tool(tools.t_write, artifact)
    other = tmp_path / "other.txt"
    assert (await tool(path=str(other), content="no"))["ok"] is False
    assert not other.exists()
    assert (await tool(path=str(artifact), content="yes"))["ok"] is True
    assert artifact.read_text() == "yes"


@pytest.mark.asyncio
async def test_harness_file_and_exec_tools_are_confined(tmp_path):
    artifact = (tmp_path / "proof.txt").resolve()
    other = tmp_path / "other.txt"
    file_tool = _confined_file_tool(tools.t_write, artifact)

    assert (await file_tool(path=str(other), content="no"))["ok"] is False
    assert (await file_tool(path=str(artifact), content="yes"))["ok"] is True

    calls = []

    async def fake_exec(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    exec_tool = _confined_exec_tool(fake_exec, allowed_commands={"printf OK"}, root=tmp_path)
    assert (await exec_tool(command="uname -a"))["ok"] is False
    assert (await exec_tool(command="printf OK", cwd=str(other)))["ok"] is False
    assert (await exec_tool(command="printf OK", cwd=str(tmp_path)))["ok"] is True
    assert calls == [{"command": "printf OK", "cwd": str(tmp_path)}]


@pytest.mark.asyncio
async def test_concurrent_edits_preserve_both_changes(tmp_path):
    target = tmp_path / "document"
    target.write_text("FIRST SECOND")
    results = await asyncio.gather(
        tools.t_edit(path=str(target), old="FIRST", new="café"),
        tools.t_edit(path=str(target), old="SECOND", new="done"),
    )
    assert all(result["ok"] is True for result in results)
    assert target.read_text() == "café done"
    assert results[0]["diff_bytes"] == 0


@pytest.mark.asyncio
async def test_chunked_write_is_private_serialized_and_reports_utf8_bytes(tmp_path):
    target = tmp_path / "nested" / "document"
    started = await tools.t_write_chunk(
        path=str(target),
        content="café",
        mode="start",
    )
    appended = await tools.t_write_chunk(
        path=str(target),
        content=" done",
        mode="append",
        final=True,
    )
    assert started["chunk_bytes"] == 5
    assert started["total_bytes"] == 5
    assert appended["chunk_bytes"] == 5
    assert appended["total_bytes"] == 10
    assert appended["final"] is True
    assert target.read_text() == "café done"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_concurrent_memory_appends_are_complete_and_private(tmp_path):
    target = tmp_path / "memory" / "notes.md"
    await asyncio.gather(
        *(tools.t_append_memory(path=str(target), text=f" record-{index}  ") for index in range(20))
    )
    assert set(target.read_text().splitlines()) == {f" record-{index}" for index in range(20)}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
