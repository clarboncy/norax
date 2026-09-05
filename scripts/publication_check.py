#!/usr/bin/env python3
"""Validate that the publishable tree contains no private runtime residue."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".jsonc",
    ".md",
    ".py",
    ".service",
    ".sh",
    ".toml",
    ".ts",
    ".tsv",
    ".timer",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_TOP_LEVEL = {
    ".env.example",
    ".github",
    ".gitignore",
    ".gitignore.private",
    ".python-version",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "agent_os",
    "architecture",
    "benchmarks",
    "config",
    "docs",
    "layer2",
    "norax",
    "ops",
    "patches",
    "pyproject.toml",
    "pytest.ini",
    "references",
    "research",
    "scripts",
    "setup.sh",
    "soul",
    "tests",
    "tools",
    "training",
    "uv.lock",
}
ALLOWED_TEXT_FILENAMES = {
    ".gitignore",
    ".gitignore.private",
    ".python-version",
    "LICENSE",
    "Modelfile.norax",
    "Modelfile.norax-gemma4-v2",
    "uv.lock",
}
FORBIDDEN_TEXT = {
    "legacy-host-name": re.compile(r"openclaw|\.openclaw|clawdbot|moltbot", re.I),
    "private-home-path": re.compile(r"/home/norax2?(?:/|\b)"),
    "private-discord-id": re.compile(
        r"1076402413134680114|1473045456098693283|1490086287003353269|1490124185186467951"
    ),
}
PRODUCTION_LEGACY_TEXT = re.compile(
    r"4118|norax_duo|ollama_duo|norax-duo|ollama-duo|duo_pipeline|duo pipeline|duo gateway",
    re.I,
)
PRODUCTION_PREFIXES = ("norax/", "config/", "ops/", "scripts/")
PRODUCTION_SUFFIXES = {".py", ".jsonc", ".json", ".sh", ".service", ".toml"}
FORBIDDEN_PREFIXES = ("memory/", "state/", "logs/", "datasets/", "agent_os/_backups/")
FORBIDDEN_ROOT_FILES = {"procedural", "semantic", "intel", "scratchpad.md", "active-focus.md"}
FORBIDDEN_SUFFIXES = (".db", ".npz", ".sqlite", ".token")
FORBIDDEN_PACKAGE_MEMORY_NAMES = {
    ".commerce_sessions.json.lock",
    "commerce_secret.key",
    "commerce_sessions.json",
    "intel.md",
}
REQUIRED = ("LICENSE", "README.md", "SECURITY.md", "CONTRIBUTING.md", ".env.example")


def publication_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return [
        Path(raw.decode(errors="surrogateescape"))
        for raw in proc.stdout.split(b"\0")
        if raw and (ROOT / raw.decode(errors="surrogateescape")).is_file()
    ]


def main() -> int:
    errors: list[str] = []
    files = publication_files()
    for required in REQUIRED:
        if not (ROOT / required).is_file():
            errors.append(f"missing required publication file: {required}")
    for rel in files:
        rel_text = rel.as_posix()
        if not rel.parts or rel.parts[0] not in ALLOWED_TOP_LEVEL:
            errors.append(f"unapproved top-level publication path: {rel_text}")
            continue
        if rel_text == "scripts/publication_check.py":
            continue
        if rel_text in FORBIDDEN_ROOT_FILES or rel_text.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"private runtime path included: {rel_text}")
            continue
        if rel_text.startswith("norax/") and any(part.startswith(".") for part in rel.parts[1:]):
            errors.append(f"hidden package file included: {rel_text}")
            continue
        if rel_text.startswith("norax/memory/") and rel.name in FORBIDDEN_PACKAGE_MEMORY_NAMES:
            errors.append(f"private package-tree state included: {rel_text}")
            continue
        if rel_text.endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"private/generated file included: {rel_text}")
            continue
        if (
            rel.suffix.lower() not in TEXT_SUFFIXES
            and rel.name not in REQUIRED
            and rel.name not in ALLOWED_TEXT_FILENAMES
        ):
            errors.append(f"unapproved non-text publication file: {rel_text}")
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"unreadable publication file: {rel_text} ({type(exc).__name__})")
            continue
        for label, pattern in FORBIDDEN_TEXT.items():
            # Allow OpenClaw references in README.md and THIRD_PARTY_NOTICES.md
            # (intentional credit to the project that inspired Norax)
            if label == "legacy-host-name" and rel_text in (
                "README.md",
                "THIRD_PARTY_NOTICES.md",
            ):
                continue
            if pattern.search(text):
                errors.append(f"{label}: {rel_text}")
        if (
            rel_text != "scripts/publication_check.py"
            and rel_text.startswith(PRODUCTION_PREFIXES)
            and rel.suffix.lower() in PRODUCTION_SUFFIXES
            and PRODUCTION_LEGACY_TEXT.search(text)
        ):
            errors.append(f"legacy-runtime-reference: {rel_text}")
    if errors:
        print("publication_check: failed")
        print("\n".join(sorted(set(errors))))
        return 1
    print(f"publication_check: ok ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
