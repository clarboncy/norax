import pytest

from norax.shell.runner import _self_runtime_lifecycle_request, run_command


@pytest.mark.asyncio
async def test_grep_no_matches_is_successful_empty_result(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n")

    result = await run_command(f"grep -n definitely_missing {target}")

    assert result["ok"] is True
    assert result["exit_code"] == 1
    assert result["no_matches"] is True
    assert result["stdout"] == ""


@pytest.mark.asyncio
async def test_non_search_exit_one_remains_a_failure():
    result = await run_command("false")

    assert result["ok"] is False
    assert result["exit_code"] == 1


@pytest.mark.asyncio
async def test_self_restart_is_deferred_until_after_active_response():
    result = await run_command(
        "systemctl --user restart norax-ai.service && sleep 6 && "
        "systemctl --user is-active norax-ai.service"
    )

    assert result["ok"] is True
    assert result["runtime_lifecycle_deferred"] == "restart"
    assert "after the active task response" in result["stdout"]


@pytest.mark.asyncio
async def test_setup_before_self_restart_must_be_split_into_separate_tool_calls():
    result = await run_command("printf setup-complete && systemctl --user restart norax-ai.service")

    assert result["ok"] is False
    assert result["error"] == "self_lifecycle_must_be_separate"


def test_remote_runtime_restart_is_not_intercepted():
    assert (
        _self_runtime_lifecycle_request("ssh remote-host systemctl --user restart norax-ai.service")
        is None
    )


def test_runtime_lifecycle_trace_detection():
    from norax.runtime.core import _deferred_runtime_lifecycle_action

    trace = [
        {"result": {"ok": True}},
        {"result": {"ok": True, "runtime_lifecycle_deferred": "restart"}},
    ]

    assert _deferred_runtime_lifecycle_action(trace) == "restart"
