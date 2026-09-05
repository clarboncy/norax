from norax.brain.output_verifier import (
    OutputVerifier,
    contains_channel_marker_garbage,
    find_channel_marker_garbage,
)


def test_objective_verification_requires_an_applicable_deterministic_check() -> None:
    verifier = OutputVerifier()

    prose = verifier.verify(output="A clear but uncheckable prose answer.")
    checked = verifier.verify(output="The result is 17 * 23 = 391.")

    assert prose.objectively_verified is False
    assert prose.objective_checked == 0
    assert checked.objectively_verified is True
    assert checked.objective_checked == 1


def _categories(report):
    return {issue.category for issue in report.issues}


def test_channel_protocol_markers_are_critical_output_garbage():
    output = "<|start_header_id|>assistant<|end_header_id|>\nDone."
    assert contains_channel_marker_garbage(output)
    assert find_channel_marker_garbage(output)

    report = OutputVerifier().verify(output=output, user_request="status?")
    assert report.has_critical
    assert "channel_marker" in _categories(report)


def test_internal_tool_recipient_marker_is_rejected():
    output = "assistant to=functions.exec_command\nThe task is complete."
    report = OutputVerifier().verify(output=output, user_request="status?")
    assert report.has_critical
    assert "channel_marker" in _categories(report)


def test_normal_use_of_analysis_word_is_not_a_channel_marker():
    output = "The analysis found no filesystem errors."
    assert not contains_channel_marker_garbage(output)


def test_honest_not_implemented_finding_does_not_trigger_revision():
    report = OutputVerifier().verify(
        output="The upstream API is not implemented yet, so that capability is unavailable.",
        user_request="Audit whether the upstream API capability really works.",
    )

    assert "incomplete" not in _categories(report)


def test_natural_conjunction_is_not_invented_into_deliverables():
    report = OutputVerifier().verify(
        output="The cache is safe and remains bounded.",
        user_request=(
            "Audit the cache and its lock behavior and ensure helpers do not hinder performance."
        ),
    )

    assert "completeness" not in _categories(report)


def test_terse_action_claim_without_mutation_is_blocked():
    report = OutputVerifier().verify(
        output="Created the production configuration.",
        user_request="Create the production configuration.",
        tool_trace=[],
    )

    assert report.has_critical
    assert "false_success" in _categories(report)


def test_nonzero_mutating_shell_exit_cannot_support_success_claim():
    trace = [
        {
            "name": "exec",
            "args": {"command": "cp missing.file destination.file"},
            "result": {"ok": True, "exit_code": 1, "stderr": "not found"},
        }
    ]
    report = OutputVerifier().verify(
        output="The file was copied successfully.",
        user_request="copy file",
        tool_trace=trace,
    )
    assert report.has_critical
    assert "claim_mismatch" in _categories(report)


def test_successful_mutation_requires_later_successful_check():
    mutation = {
        "name": "write",
        "args": {"path": "result.txt", "content": "ok"},
        "result": {"ok": True, "path": "result.txt"},
    }
    failed_check = {
        "name": "read",
        "args": {"path": "result.txt"},
        "result": {"ok": False, "error": "file_not_found"},
    }
    report = OutputVerifier().verify(
        output="Created the result.",
        user_request="create result",
        tool_trace=[mutation, failed_check],
    )
    assert report.has_critical
    assert "false_success" in _categories(report)


def test_mutation_and_check_in_one_command_does_not_self_verify():
    trace = [
        {
            "name": "shell",
            "args": {"command": "mkdir result && ls -ld result"},
            "result": {"ok": True, "exit_code": 0},
        }
    ]
    report = OutputVerifier().verify(
        output="Created the result directory.",
        user_request="create result",
        tool_trace=trace,
    )
    assert report.has_critical
    assert "false_success" in _categories(report)


def test_separate_successful_check_supports_success_claim():
    trace = [
        {
            "name": "shell",
            "args": {"command": "mkdir result"},
            "result": {"ok": True, "exit_code": 0},
        },
        {
            "name": "shell",
            "args": {"command": "test -d result"},
            "result": {"ok": True, "exit_code": 0},
        },
    ]
    report = OutputVerifier().verify(
        output="Created the result directory.",
        user_request="create result",
        tool_trace=trace,
    )
    assert "false_success" not in _categories(report)
    assert "claim_mismatch" not in _categories(report)


def test_arithmetic_rejects_small_but_real_error_and_division_by_zero():
    report = OutputVerifier().verify(
        output="The totals are 100 + 1 = 100 and 1 / 0 = 5.",
        user_request="Calculate both totals accurately.",
    )
    arithmetic = [issue for issue in report.issues if issue.category == "arithmetic"]
    assert len(arithmetic) == 2
    assert report.has_critical


def test_arithmetic_allows_result_rounded_to_claimed_precision():
    report = OutputVerifier().verify(
        output="Rounded to two decimals, 1 / 3 = 0.33.",
        user_request="Calculate one third.",
    )
    assert "arithmetic" not in _categories(report)


def test_unlabelled_pseudocode_is_not_guessed_to_be_python():
    report = OutputVerifier().verify(
        output="Example:\n```\nfunction value() { return ???; }\n```",
        user_request="Show pseudocode for the control flow.",
    )
    assert "code_syntax" not in _categories(report)


def test_placeholder_heuristic_is_advisory_not_a_rewrite_gate():
    report = OutputVerifier().verify(
        output='```python\nmessage = "not implemented"\n```',
        user_request="Show a test fixture containing the exact error text not implemented.",
    )
    incomplete = [issue for issue in report.issues if issue.category == "incomplete"]
    assert incomplete
    assert all(issue.severity.value == "minor" for issue in incomplete)
    assert not report.has_issues


def test_failed_read_does_not_substantiate_claimed_file_access():
    report = OutputVerifier().verify(
        output="I inspected /srv/app/config.py.",
        user_request="Inspect the application configuration file.",
        tool_trace=[
            {
                "name": "read",
                "args": {"path": "/srv/app/config.py"},
                "result": {"ok": False, "error": "permission_denied"},
            }
        ],
    )
    assert "path_hallucination" in _categories(report)
