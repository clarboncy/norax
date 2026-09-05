"""Orchestrator routing — explicit orchestrator mode only (no auto-routing)."""

from norax.brain.orchestrator import (
    EXECUTOR_MODEL,
    FALLBACK_PLANNER_MODEL,
    PLANNER_MODEL,
    effective_planning_route,
    should_use_orchestrator,
)
from norax.brain.strong_model_scaffold import (
    ORCHESTRATOR_OPUS_PLANNER,
    ORCHESTRATOR_PLANNER_MODEL,
    is_cursor_model,
    resolve_planner_model,
)


def test_is_cursor_model_prefixes():
    assert is_cursor_model("composer/composer-2.5")
    assert is_cursor_model("composer-opus-4.6-thinking")
    assert is_cursor_model("cursor/test-model")
    assert not is_cursor_model("qwen3-coder-next:cloud")
    assert not is_cursor_model("gpt-5.5")


def test_should_use_orchestrator_explicit_mode():
    """Orchestrator only activates for explicit 'orchestrator' mode."""
    assert should_use_orchestrator("orchestrator", "gpt-5.5")
    assert should_use_orchestrator("orchestrator", "composer/composer-2.5")
    assert not should_use_orchestrator("direct", "gpt-5.5")
    assert not should_use_orchestrator("direct", "composer/opus-4.6-thinking")
    assert not should_use_orchestrator("", "composer/composer-2.5")


def test_resolve_planner_model_opus_session_uses_opus():
    assert resolve_planner_model("claude-opus-4-8-thinking-high") == "claude-opus-4-8-thinking-high"
    assert (
        resolve_planner_model("claude-opus-4-8-thinking-xhigh") == "claude-opus-4-8-thinking-xhigh"
    )
    assert resolve_planner_model("opus-4.8") == ORCHESTRATOR_OPUS_PLANNER
    assert resolve_planner_model("composer/opus-4.6-thinking") == ORCHESTRATOR_OPUS_PLANNER


def test_resolve_planner_model_non_opus_cursor_uses_composer():
    assert resolve_planner_model("composer/composer-2.5") == ORCHESTRATOR_PLANNER_MODEL
    assert resolve_planner_model("cursor/test-model") == ORCHESTRATOR_PLANNER_MODEL


def test_resolve_planner_model_non_cursor_passthrough():
    assert resolve_planner_model("gpt-5.5") == "gpt-5.5"
    assert resolve_planner_model("qwen3-coder-next:cloud") == "qwen3-coder-next:cloud"


def test_default_planner_model_is_opus():
    assert PLANNER_MODEL == ORCHESTRATOR_OPUS_PLANNER


def test_effective_planning_route_explicit_orchestrator():
    route = effective_planning_route("orchestrator", "gpt-5.5")
    assert route["active"] is True
    assert route["auto"] is False
    assert route["planner"] == "gpt-5.5"
    assert route["executor"] == "qwen3-coder-next:cloud"
    assert route["fallback"] == FALLBACK_PLANNER_MODEL
    assert route["tertiary_fallback"] is None
    assert "Orchestrator" in route["label"]


def test_effective_planning_route_orchestrator_with_opus():
    route = effective_planning_route("orchestrator", "claude-opus-4-8-thinking-xhigh")
    assert route["active"] is True
    assert route["planner"] == "claude-opus-4-8-thinking-xhigh"
    assert route["executor"] == "qwen3-coder-next:cloud"
    assert "claude-opus-4-8-thinking-xhigh" in route["label"]


def test_effective_planning_route_orchestrator_with_composer():
    route = effective_planning_route("orchestrator", "composer/composer-2.5")
    assert route["active"] is True
    assert route["auto"] is False
    assert route["planner"] == ORCHESTRATOR_PLANNER_MODEL
    assert route["executor"] == "qwen3-coder-next:cloud"


def test_executor_model_is_qwen():
    """Verify EXECUTOR_MODEL routes directly to Qwen Coder Next."""
    assert EXECUTOR_MODEL == "qwen3-coder-next:cloud"


def test_effective_planning_route_direct():
    """Direct mode: orchestrator not active, model used as-is."""
    route = effective_planning_route("direct", "qwen3-coder-next:cloud")
    assert route["active"] is False
    assert route["planner"] == "qwen3-coder-next:cloud"
    assert route["executor"] == "qwen3-coder-next:cloud"
    assert route["fallback"] is None

    route2 = effective_planning_route("direct", "composer/opus-4.6-thinking")
    assert route2["active"] is False
    assert route2["executor"] == "composer/opus-4.6-thinking"
