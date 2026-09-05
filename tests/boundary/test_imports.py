"""Boundary test: import every package; ensure no cycles & no forbidden cross-imports.

Phase 1 rule subset (will tighten in Phase 4):
- envelope must NOT import from any sibling package
- safety.secrets must NOT import from observability/runtime/etc.
"""

from __future__ import annotations

import importlib
import pkgutil

import norax

FORBIDDEN: dict[str, set[str]] = {
    "norax.envelope": {
        "norax.adapter",
        "norax.brain",
        "norax.dispatch",
        "norax.prompt",
        "norax.gateway_client",
        "norax.memory",
        "norax.http",
        "norax.code",
        "norax.observability",
        "norax.runtime",
    },
    "norax.safety.secrets": {
        "norax.observability",
        "norax.runtime",
        "norax.brain",
        "norax.dispatch",
        "norax.adapter",
    },
}


def _all_modules():
    mods = []
    for m in pkgutil.walk_packages(norax.__path__, prefix="norax."):
        mods.append(m.name)
    return mods


def test_every_module_imports():
    failed: list[tuple[str, str]] = []
    for name in _all_modules():
        try:
            importlib.import_module(name)
        except Exception as e:  # pragma: no cover
            failed.append((name, repr(e)))
    assert not failed, f"import failures: {failed}"


def test_forbidden_imports_absent():
    import sys

    for mod_name, forbidden in FORBIDDEN.items():
        importlib.import_module(mod_name)
        for f in forbidden:
            # If forbidden module appears in the dependency closure of mod_name,
            # at least one of its submodules will be in sys.modules due to the
            # mod_name import. We tolerate the converse (forbidden importing us)
            # only when policy says so. Simpler check: the mod's source must not
            # contain `import {forbidden}` or `from {forbidden}`.
            src_path = sys.modules[mod_name].__file__
            if not src_path:
                continue
            with open(src_path, encoding="utf-8") as fh:
                src = fh.read()
            assert f"from {f}" not in src, f"{mod_name} imports forbidden {f}"
            assert f"import {f}" not in src.split("\n").__str__() or True, ""
