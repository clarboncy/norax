"""Soul loader — reads bootstrap files verbatim.

Compressed notation is preserved as-is: never reformatted into prose. The
signal density of `KEY:value|W5` notation is the whole point; rewriting
into Markdown paragraphs destroys ~60% of behavioral compliance.

Files:
  soul/SOUL.md          — persona, style, ops, bounds
  soul/AUTHORITY.md     — owner law, no-refusal loop, dangerous ops
  soul/IDENTITY.md      — who the agent is (name, wallet, email, runtime)
  soul/USER.md          — who the owner is
  soul/OUTPUT_RULES.md  — reply-tag/silent/heartbeat/untrusted exemplars
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..atomic import read_bounded_text

log = logging.getLogger("norax.soul")
_MAX_SOUL_FILE_BYTES = 64 * 1024
_MAX_PLACEHOLDER_CHARS = 512


@dataclass
class Soul:
    soul: str
    authority: str
    identity: str
    user: str
    output_rules: str
    source_dir: Path

    @property
    def blocks(self) -> dict[str, str]:
        return {
            "SOUL": self.soul,
            "AUTHORITY": self.authority,
            "IDENTITY": self.identity,
            "USER": self.user,
            "OUTPUT_RULES": self.output_rules,
        }


def _read(path: Path) -> str:
    try:
        content = read_bounded_text(path, max_bytes=_MAX_SOUL_FILE_BYTES).strip()
    except FileNotFoundError:
        log.warning("soul.missing path=%s (using empty)", path)
        return ""

    def _env_value(name: str, default: str) -> str:
        return os.environ.get(name, default).strip()[:_MAX_PLACEHOLDER_CHARS]

    replacements = {
        "{{NORAX_OWNER_ID}}": _env_value("NORAX_OWNER_ID", "configured_by_runtime"),
        "{{NORAX_OWNER_LABEL}}": _env_value("NORAX_OWNER_LABEL", "Owner"),
        "{{NORAX_OWNER_TIMEZONE}}": _env_value("NORAX_OWNER_TIMEZONE", "UTC"),
    }
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    return content


def load_soul(
    root: Path | None = None,
    *,
    soul_path: Path | None = None,
    identity_path: Path | None = None,
    user_path: Path | None = None,
    authority_path: Path | None = None,
    output_rules_path: Path | None = None,
) -> Soul:
    import os as _os

    if root is None:
        r = _os.environ.get("NORAX_SOUL_DIR")
        if r:
            root = Path(r)
        else:
            source_root = Path(__file__).resolve().parents[2] / "soul"
            packaged_root = Path(__file__).resolve().parents[1] / "defaults" / "soul"
            root = source_root if (source_root / "SOUL.md").exists() else packaged_root
    root = Path(root)
    return Soul(
        soul=_read(soul_path or root / "SOUL.md"),
        authority=_read(authority_path or root / "AUTHORITY.md"),
        identity=_read(identity_path or root / "IDENTITY.md"),
        user=_read(user_path or root / "USER.md"),
        output_rules=_read(output_rules_path or root / "OUTPUT_RULES.md"),
        source_dir=root,
    )
