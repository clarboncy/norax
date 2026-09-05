"""Architecture invariants for the composed production runtime."""

from __future__ import annotations

import pytest

from norax.runtime._mixin import RuntimeAccessMixin
from norax.runtime.core import Runtime


@pytest.mark.parametrize(
    ("method", "owner"),
    [
        ("__init__", "norax.runtime.core"),
        ("build", "norax.runtime.core"),
        ("run", "norax.runtime.core"),
        ("_ensure_cognitive_components", "norax.runtime.cognition"),
        ("_deliver_turn_response", "norax.runtime.delivery"),
        ("_queue_turn", "norax.runtime.lifecycle"),
        ("shutdown", "norax.runtime.lifecycle"),
        ("upsert_custom_provider", "norax.runtime.model_management"),
        ("_run_completion_probe", "norax.runtime.operations"),
        ("_get_window", "norax.runtime.session"),
        ("_handle_command", "norax.runtime.session"),
        ("_handle_turn", "norax.runtime.turn_pipeline"),
        ("_handle_turn_inner", "norax.runtime.turn_pipeline"),
    ],
)
def test_runtime_behavior_is_owned_by_cohesive_modules(method: str, owner: str) -> None:
    implementation = getattr(Runtime, method)
    underlying = getattr(implementation, "__func__", implementation)

    assert underlying.__module__ == owner


def test_typing_mixin_does_not_mask_missing_runtime_state() -> None:
    assert "__getattr__" not in RuntimeAccessMixin.__dict__
    runtime = Runtime.__new__(Runtime)

    with pytest.raises(AttributeError):
        object.__getattribute__(runtime, "_definitely_missing_runtime_state")


def test_behavior_mixins_do_not_silently_shadow_each_other() -> None:
    owners: dict[str, type] = {}
    collisions: dict[str, tuple[type, type]] = {}
    for mixin in Runtime.__bases__:
        for name, value in mixin.__dict__.items():
            if name.startswith("__") or not callable(value):
                continue
            previous = owners.setdefault(name, mixin)
            if previous is not mixin:
                collisions[name] = (previous, mixin)

    assert collisions == {}
