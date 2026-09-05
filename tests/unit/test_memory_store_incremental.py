from __future__ import annotations

from pathlib import Path

from norax.memory import store as store_module
from norax.memory.store import MemoryStore


def test_refresh_reuses_unchanged_parsed_files(tmp_path: Path, monkeypatch) -> None:
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    memory_file = semantic / "facts.md"
    memory_file.write_text("FACT:first|W5\n", encoding="utf-8")
    original = store_module._parse_file
    parsed: list[Path] = []

    def counting_parse(path: Path, kind: str):
        parsed.append(path)
        return original(path, kind)

    monkeypatch.setattr(store_module, "_parse_file", counting_parse)
    store = MemoryStore(root=tmp_path)

    store.refresh()
    store.refresh()
    assert parsed == [memory_file]
    assert store.semantic[0].text == "FACT:first|W5"

    memory_file.write_text("FACT:updated with a different size|W5\n", encoding="utf-8")
    store.refresh()
    assert parsed == [memory_file, memory_file]
    assert store.semantic[0].text == "FACT:updated with a different size|W5"

    memory_file.unlink()
    store.refresh()
    assert store.semantic == []
    assert memory_file not in store._cached_neurons
