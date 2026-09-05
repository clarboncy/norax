"""Regression test: package metadata and runtime version must agree."""

from __future__ import annotations

from importlib.metadata import version as pkg_version


def test_runtime_version_matches_package_metadata():
    import norax

    assert norax.__version__ == pkg_version("norax"), (
        f"norax.__version__={norax.__version__} != pkg metadata={pkg_version('norax')}"
    )


def test_version_is_not_stale_0_1_0():
    import norax

    assert norax.__version__ != "0.1.0", "version still stale at 0.1.0"
