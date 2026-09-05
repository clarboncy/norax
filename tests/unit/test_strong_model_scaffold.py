import pytest

from norax.brain.strong_model_scaffold import (
    build_scaffold_prompt,
    build_task_state,
    compress_tool_result,
    enforce_final_verification,
    is_action_command,
    shell_command_is_mutating,
    summarize_mutation_trace,
    tool_call_has_authoritative_receipt,
    tool_call_is_mutating,
    tool_call_is_successful_verification,
    tool_failure_is_blocking,
    tool_result_succeeded,
)


@pytest.mark.parametrize(
    "text",
    [
        "Fix the parser and run its tests.",
        "Could you check the staging server?",
        "Please, go ahead and implement it.",
        "I need you to audit every helper.",
        "Continue",
        "Try again",
        "Read src/main.py",
    ],
)
def test_action_command_matches_direct_execution_intent(text):
    assert is_action_command(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "How do I fix the parser?",
        "What happens when the server restarts?",
        "Could you explain how to restart it?",
        "Tell me about staging and remote_exec.",
        "Should I install this package?",
        "Do not delete the cache.",
        "The audit says to check the logs.",
    ],
)
def test_action_command_ignores_mentions_questions_and_negations(text):
    assert is_action_command(text) is False


def test_coding_task_gets_checklist_and_constraints():
    state = build_task_state(
        "carefully implement the pipeline and test it", ["read", "edit", "exec"]
    )
    assert state.task_type == "coding"
    assert "inspect current files before editing" in state.constraints
    assert "post-write readback or targeted test/command succeeded" in state.done_criteria
    assert "done_criteria" in state.context_block()
    prompt = build_scaffold_prompt(state.task_type)
    assert "CODING_CHECKLIST" in prompt
    assert "VERIFY_GATE" in prompt


def test_write_before_read_triggers_escalation():
    state = build_task_state("fix code", ["edit"])
    state.note_tool("edit", {"path": "x.py"}, {"ok": True})
    assert state.writes == 1
    assert state.should_escalate
    assert any("write attempted before read" in r for r in state.escalation_reasons)


def test_verified_after_write_clears_final_gate():
    state = build_task_state("fix code", ["read", "edit", "exec"])
    state.note_tool("read", {"path": "x.py"}, {"ok": True, "path": "x.py", "total_lines": 3})
    state.note_tool("edit", {"path": "x.py"}, {"ok": True})
    assert enforce_final_verification(state, "done") is not None
    state.note_tool("exec", {"command": "pytest"}, {"ok": True, "exit_code": 0, "stdout": "ok"})
    assert enforce_final_verification(state, "done") is None


def test_tool_result_compression_preserves_signal_fields():
    """Large content is compressed with head+tail; status fields pass through."""
    big = "a\n" * 500  # 500 lines, ~1000 chars — triggers head+tail compression
    res = compress_tool_result(
        "read", {"ok": True, "path": "x.py", "total_lines": 500, "content": big}, max_chars=1000
    )
    assert res["ok"] is True
    assert res["path"] == "x.py"
    assert res["total_lines"] == 500
    # Content should be compressed, not passed through
    assert res["content"] != big
    assert res.get("_content_compressed") is True
    assert res.get("_content_original_lines") == 500
    # Head and tail should be preserved
    assert res["content"].startswith("a")
    assert res["content"].endswith("a")
    # Elision marker present
    assert "elided" in res["content"]


def test_explicit_large_read_gets_bounded_larger_budget():
    big = "0123456789" * 900  # 9KB: above normal budget, below large-read budget
    normal = compress_tool_result("read", {"ok": True, "content": big}, max_chars=1000)
    large = compress_tool_result(
        "read", {"ok": True, "content": big, "_large_read": True}, max_chars=1000
    )
    assert len(normal["content"]) < len(big)
    assert large["content"] == big
    assert large["_large_read"] is True


def test_three_failures_trigger_recovery_nudge():
    state = build_task_state("debug failing tests", ["exec"])
    for _ in range(3):
        state.note_tool("exec", {"command": "pytest"}, {"ok": False, "error": "failed"})
    nudge = enforce_final_verification(state, "I am not sure; tests failed")
    assert nudge is not None
    assert "Escalation trigger" in nudge


def test_recovered_shell_failure_is_not_a_permanent_blocker():
    state = build_task_state("inspect code", ["exec"])
    failed_search = {"ok": False, "exit_code": 1, "stdout": ""}
    assert tool_failure_is_blocking("exec", failed_search) is False
    state.note_tool("exec", {"command": "grep missing file"}, failed_search)
    assert state.blockers == []
    assert state.consecutive_failures == 1
    state.note_tool("exec", {"command": "rg symbol ."}, {"ok": True, "exit_code": 0})
    assert state.consecutive_failures == 0
    assert enforce_final_verification(state, "Inspection completed successfully.") is None


def test_need_to_inspect_is_treated_as_mid_task_narration():
    state = build_task_state("fix code", ["read", "exec"])
    state.note_tool("exec", {"command": "pwd"}, {"ok": True, "exit_code": 0})
    nudge = enforce_final_verification(state, "I need to inspect the implementation next.")
    assert nudge is not None
    assert "intent to continue" in nudge


def test_failed_mutating_exec_cannot_be_counted_as_success():
    args = {"command": "cp missing.file destination.file"}
    result = {"ok": True, "exit_code": 1, "stderr": "missing.file: No such file"}

    assert shell_command_is_mutating(args["command"]) is True
    assert tool_result_succeeded("exec", result, args) is False

    state = build_task_state("copy the file and verify it", ["exec"])
    state.note_tool("exec", args, result)
    assert state.mutation_attempts == 1
    assert state.mutations_succeeded == 0
    assert state.mutations_failed == 1
    assert state.writes == 0
    assert state.last_mutation_succeeded is False
    assert "latest state-changing tool call failed" in (
        enforce_final_verification(state, "The file was copied.") or ""
    )


def test_nonzero_shell_status_is_failure_except_explicit_search_no_match():
    assert not tool_result_succeeded(
        "shell",
        {"ok": True, "exit_code": 2, "stderr": "bad option"},
        {"command": "ls --bad-option"},
    )
    assert tool_result_succeeded(
        "exec",
        {"ok": True, "exit_code": 1, "stdout": "", "stderr": ""},
        {"command": "rg missing-pattern ."},
    )
    assert not tool_result_succeeded(
        "exec",
        {"ok": True, "exit_code": 1, "stdout": "", "stderr": ""},
        {"command": "rg missing-pattern .; false"},
    )


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("computer_use", {"action": "screenshot"}, False),
        ("computer_use", {"action": "click"}, True),
        ("browser", {"action": "extract"}, False),
        ("browser", {"action": "fill_form"}, True),
        ("append_memory", {"path": "notes.md"}, True),
        ("message_send", {"text": "hello"}, True),
        ("schedule_reminder", {"text": "hello"}, True),
        ("mcp_calendar_create_event", {}, True),
        ("mcp_calendar_get_event", {}, False),
    ],
)
def test_action_specific_mutation_classification(name, args, expected):
    assert tool_call_is_mutating(name, args) is expected


@pytest.mark.parametrize(
    "command",
    [
        "npm install lodash",
        "git push origin main",
        "kubectl apply -f deployment.yaml",
    ],
)
def test_package_vcs_and_deployment_commands_are_mutations(command):
    assert shell_command_is_mutating(command)


def test_api_receipt_completes_message_mutation_without_duplicate_send():
    result = {"ok": True, "message_id": "m-123", "queued": True}
    assert tool_call_has_authoritative_receipt("message_send", {"text": "hi"}, result)

    outcome = summarize_mutation_trace(
        [{"name": "message_send", "args": {"text": "hi"}, "result": result}]
    )
    assert outcome.verified_after_last_mutation is True
    assert outcome.supports_success_claim is True


def test_remote_read_verifies_matching_remote_write():
    state = build_task_state("update remote config", ["remote_read", "remote_write"])
    state.note_tool(
        "remote_read",
        {"node_id": "staging", "path": "/srv/app.ini"},
        {"ok": True, "path": "/srv/app.ini"},
    )
    state.note_tool(
        "remote_write",
        {"node_id": "staging", "path": "/srv/app.ini", "content": "x=1"},
        {"ok": True, "path": "/srv/app.ini"},
    )
    state.note_tool(
        "remote_read",
        {"node_id": "staging", "path": "/srv/app.ini"},
        {"ok": True, "path": "/srv/app.ini", "content": "x=1"},
    )

    assert state.verified_after_write is True


def test_new_mutation_invalidates_earlier_verification():
    state = build_task_state("fix code", ["read", "edit"])
    state.note_tool("read", {"path": "x.py"}, {"ok": True, "path": "x.py"})
    state.note_tool("edit", {"path": "x.py"}, {"ok": True, "path": "x.py"})
    state.note_tool("read", {"path": "x.py"}, {"ok": True, "path": "x.py"})
    assert state.verified_after_write is True

    state.note_tool("edit", {"path": "x.py"}, {"ok": True, "path": "x.py"})
    assert state.verified_after_write is False


def test_unrelated_read_does_not_verify_file_mutation():
    state = build_task_state("fix code", ["read", "edit"])
    state.note_tool("read", {"path": "x.py"}, {"ok": True, "path": "x.py"})
    state.note_tool("edit", {"path": "x.py"}, {"ok": True, "path": "x.py"})
    state.note_tool("read", {"path": "unrelated.py"}, {"ok": True, "path": "unrelated.py"})

    assert state.verified_after_write is False
    assert state.completed_actions == []


def test_direct_read_verifies_shell_write_to_explicit_same_path():
    state = build_task_state("update state file", ["exec", "read"])
    state.note_tool(
        "exec",
        {"command": "printf after > /tmp/state.txt"},
        {"ok": True, "exit_code": 0},
    )
    state.note_tool(
        "read",
        {"path": "/tmp/state.txt"},
        {"ok": True, "path": "/tmp/state.txt", "content": "after"},
    )

    assert state.verified_after_write is True


def test_unrelated_direct_read_does_not_verify_shell_write():
    state = build_task_state("update state file", ["exec", "read"])
    state.note_tool(
        "exec",
        {"command": "printf after > /tmp/state.txt"},
        {"ok": True, "exit_code": 0},
    )
    state.note_tool(
        "read",
        {"path": "/tmp/other.txt"},
        {"ok": True, "path": "/tmp/other.txt", "content": "after"},
    )

    assert state.verified_after_write is False


def test_shell_content_word_does_not_masquerade_as_output_target():
    state = build_task_state("update state file", ["exec", "read"])
    state.note_tool(
        "exec",
        {"command": "printf /tmp/other.txt > /tmp/state.txt"},
        {"ok": True, "exit_code": 0},
    )
    state.note_tool(
        "read",
        {"path": "/tmp/other.txt"},
        {"ok": True, "path": "/tmp/other.txt", "content": "old"},
    )

    assert state.verified_after_write is False


def test_same_target_read_records_mutation_as_completed():
    state = build_task_state("fix code", ["read", "edit"])
    state.note_tool("read", {"path": "x.py"}, {"ok": True, "path": "x.py"})
    state.note_tool("edit", {"path": "x.py"}, {"ok": True, "path": "x.py"})
    state.note_tool("read", {"path": "x.py"}, {"ok": True, "path": "x.py"})

    assert state.verified_after_write is True
    assert any(action.startswith("edit x.py") for action in state.completed_actions)


def test_mutation_and_check_in_same_shell_call_is_not_independent_verification():
    trace = [
        {
            "name": "exec",
            "args": {"command": "mkdir new-dir && ls -ld new-dir"},
            "result": {"ok": True, "exit_code": 0},
        }
    ]
    outcome = summarize_mutation_trace(trace)
    assert outcome.succeeded == 1
    assert outcome.verified_after_last_mutation is False
    assert outcome.supports_success_claim is False


def test_successful_independent_check_supports_mutation_claim():
    trace = [
        {
            "name": "shell",
            "args": {"command": "mkdir new-dir"},
            "result": {"ok": True, "exit_code": 0},
        },
        {
            "name": "shell",
            "args": {"command": "test -d new-dir"},
            "result": {"ok": True, "exit_code": 0},
        },
    ]
    outcome = summarize_mutation_trace(trace)
    assert tool_call_is_successful_verification(
        trace[1]["name"], trace[1]["args"], trace[1]["result"]
    )
    assert outcome.supports_success_claim is True


def test_unrelated_shell_check_does_not_verify_mutation():
    trace = [
        {
            "name": "shell",
            "args": {"command": "mkdir result-dir"},
            "result": {"ok": True, "exit_code": 0},
        },
        {
            "name": "shell",
            "args": {"command": "test -d other-dir"},
            "result": {"ok": True, "exit_code": 0},
        },
    ]

    assert summarize_mutation_trace(trace).verified_after_last_mutation is False


def test_failed_check_does_not_verify_mutation():
    trace = [
        {
            "name": "edit",
            "args": {"path": "x.py"},
            "result": {"ok": True, "path": "x.py"},
        },
        {
            "name": "exec",
            "args": {"command": "pytest -q"},
            "result": {"ok": False, "exit_code": 1},
        },
    ]
    assert summarize_mutation_trace(trace).verified_after_last_mutation is False


def test_unrelated_project_test_does_not_verify_non_code_mutation():
    trace = [
        {
            "name": "shell",
            "args": {"command": "mkdir result-dir"},
            "result": {"ok": True, "exit_code": 0},
        },
        {
            "name": "shell",
            "args": {"command": "pytest -q"},
            "result": {"ok": True, "exit_code": 0},
        },
    ]

    assert summarize_mutation_trace(trace).verified_after_last_mutation is False


def test_project_test_can_verify_identified_source_mutation():
    trace = [
        {
            "name": "edit",
            "args": {"path": "src/parser.py"},
            "result": {"ok": True, "path": "src/parser.py"},
        },
        {
            "name": "exec",
            "args": {"command": "pytest -q"},
            "result": {"ok": True, "exit_code": 0},
        },
    ]

    assert summarize_mutation_trace(trace).verified_after_last_mutation is True


def test_scripted_file_writes_are_mutations() -> None:
    assert shell_command_is_mutating("python -c \"Path('x').write_text('ok')\"")
    assert shell_command_is_mutating("node -e \"fs.writeFileSync('x', 'ok')\"")
    assert shell_command_is_mutating("curl -o artifact.bin https://example.test/file")


def test_stderr_fd_redirection_is_not_a_mutation():
    command = "docker compose logs bedrock2 --tail 20 2>&1 | tail -20"
    assert shell_command_is_mutating(command) is False


def test_read_only_docker_exec_verifies_after_write():
    state = build_task_state("update and verify server config", ["edit", "exec"])
    state.note_tool("edit", {"path": "compose.yml"}, {"ok": True, "path": "compose.yml"})
    state.note_tool(
        "exec",
        {"command": "docker exec minecraft-bedrock2 mc-monitor status-bedrock --host 127.0.0.1"},
        {"ok": True, "exit_code": 0, "stdout": "Bedrock server is healthy"},
    )
    assert state.verified_after_write is True
    assert enforce_final_verification(state, "Server is healthy.") is None


def test_mutating_docker_exec_still_counts_as_mutation():
    command = "docker exec app sh -c 'rm -f /tmp/stale && touch /tmp/fresh'"
    assert shell_command_is_mutating(command) is True


def test_rejected_mutation_is_not_treated_as_an_executed_action() -> None:
    trace = [
        {
            "name": "write",
            "args": {"path": "state.txt", "content": "after"},
            "result": {"ok": True, "path": "state.txt"},
        },
        {
            "name": "write",
            "args": {"path": "state.txt", "content": "after"},
            "result": {
                "ok": False,
                "error": "duplicate_mutation_blocked",
                "_not_executed": True,
            },
        },
        {
            "name": "read",
            "args": {"path": "state.txt"},
            "result": {"ok": True, "path": "state.txt", "content": "after"},
        },
    ]

    outcome = summarize_mutation_trace(trace)

    assert outcome.attempted == 1
    assert outcome.succeeded == 1
    assert outcome.failed == 0
    assert outcome.verified_after_last_mutation is True


def test_new_task_does_not_inherit_stale_operational_state():
    restored = {
        "goal": "fix the checkout parser",
        "completed_actions": ["edit parser.py (verified round 2)"],
        "file_summaries": {"parser.py": "lines=20"},
        "files_touched": ["parser.py"],
        "facts": ["old fact"],
        "pending_steps": ["run parser tests"],
    }

    state = build_task_state(
        "delete old backups", ["exec"], restore_from=restored, prior_status="in_progress"
    )

    assert state.goal == "delete old backups"
    assert state.completed_actions == []
    assert state.file_summaries == {}
    assert state.pending_steps == []


def test_same_project_new_issue_does_not_restore_stale_task_state():
    restored = {
        "goal": "fix norax provider routing",
        "completed_actions": ["edited gateway.py"],
        "pending_steps": ["restart provider"],
    }

    state = build_task_state(
        "fix norax context window",
        ["read", "edit", "exec"],
        restore_from=restored,
        prior_status="in_progress",
    )

    assert state.goal == "fix norax context window"
    assert state.completed_actions == []
    assert state.pending_steps == []


def test_explicit_continuation_restores_unfinished_task_state():
    restored = {
        "goal": "Fix the checkout parser",
        "completed_actions": ["edit parser.py (verified round 2)"],
        "file_summaries": {"parser.py": "lines=20"},
        "pending_steps": ["run parser tests"],
    }

    state = build_task_state(
        "continue the audit", ["exec"], restore_from=restored, prior_status="in_progress"
    )

    assert state.goal == "Fix the checkout parser"
    assert state.completed_actions == ["edit parser.py (verified round 2)"]
    assert state.pending_steps == ["run parser tests"]
    assert state.task_type == "coding"
    assert "run targeted verification after writes" in state.constraints


@pytest.mark.parametrize("prompt", ["continue", "resume", "keep going"])
def test_bare_continuation_preserves_goal_specific_verification(prompt):
    goal = "Research the competing explanations and compare sources"
    state = build_task_state(prompt, ["web_search"], restore_from={"goal": goal})
    assert state.goal == goal
    assert state.task_type == "research"
    assert "important disagreements or gaps are reported" in state.done_criteria
