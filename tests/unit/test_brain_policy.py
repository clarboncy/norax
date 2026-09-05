from norax.brain.policy import ALL_OWNER_TOOLS, memory_k_for_turn, tool_names_for_turn
from norax.dispatch.tools import REGISTRY


def test_owner_core_tools_always_present():
    tools = tool_names_for_turn("fix local code", "owner")
    for name in [
        "read",
        "list_dir",
        "search_memory",
        "status",
        "exec",
        "write",
        "write_chunk",
        "edit",
        "append_memory",
    ]:
        assert name in tools


def test_owner_always_has_complete_capability_manifest():
    expected = set(REGISTRY) - {"memory_search"}
    assert set(ALL_OWNER_TOOLS) == expected
    assert set(tool_names_for_turn("fix local code", "owner")) == expected
    assert set(tool_names_for_turn("ambiguous request", "owner")) == expected
    assert "computer_use" in expected


def test_memory_k_adaptive():
    assert memory_k_for_turn("ok", "owner") == 5
    assert (
        memory_k_for_turn(
            "normal implementation request with enough words to need context and some extra implementation detail",
            "owner",
        )
        == 10
    )
    assert memory_k_for_turn("audit our system end to end and research", "owner") == 16
