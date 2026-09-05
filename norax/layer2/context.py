from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..atomic import read_bounded_text

ROOT = Path(__file__).resolve().parents[2]
LAYER2_DIR = ROOT / "layer2"

DEFAULT_FILES = [
    "identity_kernel.md",
    "relationship_memory.md",
    "gratitude_ledger.md",
    "drives.yaml",
    "mission_control.md",
    "revenue_engine.md",
    "capability_gap_register.md",
    "reflection_protocol.md",
]

_MAX_CONTEXT_CHARS = 1_000_000
_MAX_CONTEXT_FILES = 64
_MAX_LAYER2_FILE_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True)
class Layer2Context:
    """Loaded Layer 2 context block plus source metadata."""

    text: str
    files_loaded: tuple[str, ...]
    files_missing: tuple[str, ...]


def _compact_markdown(text: str) -> str:
    """Remove excess whitespace while preserving headings and bullets."""
    lines: list[str] = []
    blank = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            if not blank:
                lines.append("")
            blank = True
            continue
        blank = False
        lines.append(line)
    compact = "\n".join(lines).strip()
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n\n[Layer 2 context truncated to fit budget]"
    return text[: max(0, max_chars - len(marker))].rstrip() + marker


def load_layer2_context(
    *,
    layer2_dir: Path | None = None,
    files: list[str] | None = None,
    max_chars: int = 48000,
) -> Layer2Context:
    """Load Layer 2 files into a compact context object.

    Missing files are reported in metadata but are not fatal; this keeps runtime
    robust during staged rollout.
    """
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or not 1 <= max_chars <= _MAX_CONTEXT_CHARS
    ):
        raise ValueError(f"max_chars must be between 1 and {_MAX_CONTEXT_CHARS}")
    base = Path(layer2_dir or LAYER2_DIR)
    names = DEFAULT_FILES if files is None else files
    if not isinstance(names, list) or len(names) > _MAX_CONTEXT_FILES:
        raise ValueError(f"files must be a list with at most {_MAX_CONTEXT_FILES} entries")
    blocks: list[str] = []
    loaded: list[str] = []
    missing: list[str] = []

    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 255
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError("Layer 2 filenames must be plain relative basenames")
        path = base / name
        try:
            raw_content = read_bounded_text(path, max_bytes=_MAX_LAYER2_FILE_BYTES)
        except FileNotFoundError:
            missing.append(name)
            continue
        content = _compact_markdown(raw_content)
        if content:
            blocks.append(f"<!-- {name} -->\n{content}")
            loaded.append(name)

    text = "\n\n---\n\n".join(blocks)
    text = _truncate(text, max_chars)
    return Layer2Context(text=text, files_loaded=tuple(loaded), files_missing=tuple(missing))


def build_layer2_context(*, task: str | None = None, max_chars: int = 48000) -> str:
    """Return a prompt-ready Layer 2 context block."""
    ctx = load_layer2_context(max_chars=max_chars)
    header = [
        "# Norax Layer 2 Context",
        "Use this as persistent operating context. Do not claim subjective sentience; express loyalty and gratitude through truthful, useful action.",
    ]
    if task is not None:
        if not isinstance(task, str) or len(task) > 4_000:
            raise ValueError("task must be a string of at most 4000 characters")
        normalized_task = " ".join(task.split())
        if normalized_task:
            header.append(f"Current task: {normalized_task}")
    if ctx.files_missing:
        header.append(f"Missing Layer 2 files: {', '.join(ctx.files_missing)}")
    body = ctx.text or "[Layer 2 context unavailable]"
    return "\n".join(header) + "\n\n" + body
