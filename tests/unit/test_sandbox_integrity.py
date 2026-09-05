from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from norax.dispatch.sandbox import SandboxConfig, SandboxManager, SandboxResult


def test_mount_validation_does_not_create_missing_host_path(tmp_path: Path) -> None:
    manager = SandboxManager()
    missing = tmp_path / "missing"

    with pytest.raises(ValueError, match="does not exist"):
        manager._build_mount_args([str(missing)])

    assert not missing.exists()


def test_mount_validation_rejects_destination_collisions(tmp_path: Path) -> None:
    manager = SandboxManager()
    first = tmp_path / "one" / "shared"
    second = tmp_path / "two" / "shared"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    with pytest.raises(ValueError, match="destination collision"):
        manager._build_mount_args([str(first), str(second)])


def test_container_arguments_drop_capabilities(tmp_path: Path) -> None:
    manager = SandboxManager()
    manager.runtime = "docker"
    args = manager._build_run_args("true", SandboxConfig(mounts=[str(tmp_path)]))

    assert "--cap-drop=ALL" in args
    assert "--security-opt=no-new-privileges" in args
    assert "--network=none" in args


@pytest.mark.asyncio
async def test_run_script_uses_actual_mounted_parent_and_preserves_input_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script = script_dir / "hello world.py"
    script.write_text("print('ok')", encoding="utf-8")
    supplied_mounts = [str(tmp_path)]
    observed: dict[str, object] = {}
    manager = SandboxManager()

    async def fake_run(command: str, **kwargs: object) -> SandboxResult:
        observed["command"] = command
        observed.update(kwargs)
        return SandboxResult(ok=True, exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(manager, "run", fake_run)
    result = await manager.run_script(str(script), mounts=supplied_mounts)

    assert result.ok is True
    assert supplied_mounts == [str(tmp_path)]
    assert observed["command"] == "python3 '/sandbox/scripts/hello world.py'"
    assert observed["mounts"] == [str(tmp_path), str(script_dir)]


@pytest.mark.asyncio
async def test_stream_reader_is_drained_but_retained_output_is_bounded() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"abcdefgh")
    reader.feed_eof()

    retained, truncated = await SandboxManager._read_limited(reader, 4)

    assert retained == b"abcd"
    assert truncated is True
