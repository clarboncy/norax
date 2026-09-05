#!/usr/bin/env python3
"""Fail if durable/project files contain likely secrets.

Allows known public identifiers such as wallet addresses while catching API keys,
private keys, bearer tokens, Discord tokens, and password assignments.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = {
    ".github",
    "agent_os",
    "architecture",
    "benchmarks",
    "config",
    "docs",
    "gateway_provider",
    "layer2",
    "norax",
    "operations",
    "ops",
    "patches",
    "references",
    "research",
    "scripts",
    "soul",
    "tests",
    "tools",
    "training",
}
SCAN_ROOT_FILES = {
    ".env.example",
    ".gitignore",
    ".gitignore.private",
    ".python-version",
    "CONTRIBUTING.md",
    "LICENSE",
    "pyproject.toml",
    "pytest.ini",
    "README.md",
    "SECURITY.md",
    "setup.sh",
    "THIRD_PARTY_NOTICES.md",
    "uv.lock",
}
SKIP_PARTS = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "logs",
    "state",
}
PATTERNS = [
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)?PRIVATE KEY-----")),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*(['\"])(?!\*\*\*REDACTED)[^'\"\s]{12,}\1"
        ),
    ),
    (
        "assigned-secret-unquoted",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*"
            r"(?![\"']|\$\{|<|none\b|null\b|true\b|false\b)"
            r"[A-Za-z0-9_./+=:@-]{16,}\s*(?:#.*)?$"
        ),
    ),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("discord-token", re.compile(r"(?i)discord(?:_bot)?[_-]?token\s*[:=]\s*['\"]?[^'\"\s]{20,}")),
    ("api-key", re.compile(r"sk-[A-Za-z0-9_-]{24,}")),
]
_ENV_REFERENCE_ASSIGNMENT = re.compile(
    r"\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*[\"']?"
    r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}[\"']?\s*"
)


def iter_files():
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = Path(raw.decode(errors="surrogateescape"))
        if rel.name not in SCAN_ROOT_FILES and (not rel.parts or rel.parts[0] not in SCAN_DIRS):
            continue
        path = ROOT / rel
        if not path.is_file() or any(part in SKIP_PARTS for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".sqlite"}:
            continue
        yield path


def main() -> int:
    hits: list[str] = []
    try:
        files = list(iter_files())
    except subprocess.CalledProcessError as exc:
        print(f"secret_scan: failed to enumerate repository files (git exit {exc.returncode})")
        return 2
    for path in files:
        rel = path.relative_to(ROOT)
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError as exc:
            hits.append(f"{rel}: unreadable ({type(exc).__name__})")
            continue
        with handle:
            for i, line in enumerate(handle, 1):
                if _ENV_REFERENCE_ASSIGNMENT.fullmatch(line):
                    continue
                for label, rx in PATTERNS:
                    if rx.search(line):
                        # Never echo a matched source line: the scanner itself
                        # must not become a credential-exfiltration mechanism.
                        hits.append(f"{rel}:{i}: {label}")
                        break
    if hits:
        print("Potential secrets found:")
        print("\n".join(hits[:100]))
        return 1
    print("secret_scan: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
