"""Output Verifier — pre-send quality gate.

Catches errors before output reaches the user. Works under any model,
from small local models to frontier models. The checks are local and
rule-based and make no additional LLM calls unless the caller requests a
revision after a blocking finding.

Checks include explicitly labelled Python syntax, arithmetic expressions,
claimed file access, Markdown fence integrity, empty/trivial drafts, and action
success versus tool evidence. Lexical completeness and placeholder checks are
advisory because they cannot establish correctness on their own.

Usage:
    verifier = OutputVerifier()
    report = verifier.verify(
        output="Here is the code:\n```python\nprint('hello')\n```",
        user_request="Write a hello world script",
        tool_trace=[{"name": "write", "args": {"path": "/tmp/hello.py"}, "result": {"ok": True}}],
    )
    if report.has_issues:
        # inject issues as a revision nudge
        messages.append({"role": "user", "content": report.revision_prompt})
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path

from .strong_model_scaffold import summarize_mutation_trace, tool_result_succeeded

log = logging.getLogger("norax.brain.output_verifier")


class IssueSeverity(Enum):
    CRITICAL = "critical"  # output is wrong/dangerous, must revise
    MAJOR = "major"  # output has significant problems
    MINOR = "minor"  # cosmetic or low-impact issues
    INFO = "info"  # informational, no action needed


@dataclass
class Issue:
    severity: IssueSeverity
    category: str
    description: str
    evidence: str = ""
    fix_hint: str = ""


@dataclass
class VerificationReport:
    issues: list[Issue] = field(default_factory=list)
    checked: int = 0
    passed: int = 0
    objective_checked: int = 0
    objective_passed: int = 0

    @property
    def has_issues(self) -> bool:
        return any(i.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR) for i in self.issues)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    @property
    def score(self) -> float:
        """0.0 = many critical issues, 1.0 = clean output."""
        if not self.issues:
            return 1.0
        weights = {
            IssueSeverity.CRITICAL: 0.4,
            IssueSeverity.MAJOR: 0.2,
            IssueSeverity.MINOR: 0.05,
            IssueSeverity.INFO: 0.0,
        }
        penalty = sum(weights.get(i.severity, 0) for i in self.issues)
        return max(0.0, 1.0 - penalty)

    @property
    def objectively_verified(self) -> bool:
        """Whether at least one applicable deterministic check passed fully."""
        return self.objective_checked > 0 and self.objective_passed == self.objective_checked

    @property
    def revision_prompt(self) -> str:
        """Generate a revision nudge for the model."""
        critical = [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]
        major = [i for i in self.issues if i.severity == IssueSeverity.MAJOR]
        if not critical and not major:
            return ""
        lines = ["OUTPUT VERIFICATION FAILED — fix these before sending:"]
        for i in critical:
            lines.append(f"  ❌ [{i.category}] {i.description}")
            if i.fix_hint:
                lines.append(f"     Fix: {i.fix_hint}")
        for i in major:
            lines.append(f"  ⚠️ [{i.category}] {i.description}")
            if i.fix_hint:
                lines.append(f"     Fix: {i.fix_hint}")
        lines.append("Revise your answer. Do NOT repeat the errors.")
        return "\n".join(lines)

    @property
    def summary(self) -> str:
        c = sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)
        m = sum(1 for i in self.issues if i.severity == IssueSeverity.MAJOR)
        mn = sum(1 for i in self.issues if i.severity == IssueSeverity.MINOR)
        return f"{self.checked} checks, {self.passed} passed, {c} critical, {m} major, {mn} minor"


# --- Helpers ---

_CODE_FENCE_RE = re.compile(r"```(\w+)?\r?\n(.*?)```", re.DOTALL)
_ARITH_RE = re.compile(
    r"(?<![\w.])(-?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"([+\-*/×÷])\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*=\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))(?![\w.])"
)
_FILE_PATH_RE = re.compile(r'(?:^|\s|["\'])(/[^\s"\'<>|]+|~/[^\s"\'<>|]+|\.\.?/[^\s"\'<>|]+)')
_TRIVIAL_OUTPUTS = {
    "",
    " ",
    "checking...",
    "checking…",
    "let me check",
    "looking into it",
    "one moment",
    "wait",
    "thinking...",
    "thinking…",
    "processing...",
    "ok",
    "done",
    "sure",
    "yes",
    "no",  # these are fine only if they answer the question
}
_INCOMPLETE_MARKERS = [
    "TODO",
    "FIXME",
    "not implemented",
    "placeholder",
    "coming soon",
    "to be continued",
    "TBD",
    "WIP",
    "work in progress",
]
_CHANNEL_MARKER_PATTERNS = (
    re.compile(
        r"<\|(?:assistant|user|system|tool|analysis|commentary|final|channel|"
        r"recipient|constrain|im_start|im_end|start_header_id|end_header_id|eot_id)"
        r"(?:[^|>]*)\|>",
        re.I,
    ),
    re.compile(
        r"(?im)^\s*(?:assistant|analysis|commentary|final|tool)\s+"
        r"(?:to|recipient)\s*=\s*[^\s]+"
    ),
    re.compile(r"(?im)^\s*(?:assistant|analysis|commentary|final)/\w+\s*$"),
)
_ACTION_CLAIM_PATTERNS = (
    re.compile(r"\b(?:deleted|removed|purged|cleared|wiped|cleaned\s+up|erased)\b", re.I),
    re.compile(r"\b(?:created|made|generated|built|set\s+up|initialized)\b", re.I),
    re.compile(r"\b(?:moved|renamed|transferred|migrated|relocated)\b", re.I),
    re.compile(r"\b(?:copied|duplicated|backed\s+up|synced|synchronized)\b", re.I),
    re.compile(r"\b(?:installed|deployed|configured|enabled|activated)\b", re.I),
    re.compile(
        r"\b(?:fixed|resolved|repaired|patched|corrected|updated|modified|edited|"
        r"wrote|saved|changed|applied)\b",
        re.I,
    ),
    re.compile(r"\b(?:stopped|killed|terminated|restarted|reloaded|rebooted)\b", re.I),
    re.compile(r"\b(?:started|launched|brought\s+up|spun\s+up)\b", re.I),
)


def find_channel_marker_garbage(output: str) -> list[str]:
    """Return leaked model-protocol/channel markers found in user-facing text."""
    hits: list[str] = []
    for pattern in _CHANNEL_MARKER_PATTERNS:
        for match in pattern.finditer(output or ""):
            hit = match.group(0).strip()
            if hit and hit not in hits:
                hits.append(hit[:160])
    return hits


def contains_channel_marker_garbage(output: str) -> bool:
    """Cheap public predicate for adapters that want a pre-send guard."""
    return bool(find_channel_marker_garbage(output))


def claims_completed_action(output: str) -> bool:
    """Return whether text represents a completed state-changing action."""
    return any(pattern.search(output or "") for pattern in _ACTION_CLAIM_PATTERNS)


_FIRST_PERSON_ACTION_RE = re.compile(
    r"\b(?:I|I've|I have|I'd|I had|we|we've|we have)\s+"
    r"(?:deleted|removed|purged|cleared|wiped|cleaned\s+up|erased|"
    r"created|made|generated|built|set\s+up|initialized|"
    r"moved|renamed|transferred|migrated|relocated|"
    r"copied|duplicated|backed\s+up|synced|synchronized|"
    r"installed|deployed|configured|enabled|activated|"
    r"fixed|resolved|repaired|patched|corrected|updated|modified|edited|"
    r"wrote|saved|changed|applied|"
    r"stopped|killed|terminated|restarted|reloaded|rebooted|"
    r"started|launched|brought\s+up|spun\s+up)\b",
    re.IGNORECASE,
)
_LEADING_ACTION_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:successfully\s+)?"
    r"(?:deleted|removed|purged|cleared|wiped|created|generated|built|"
    r"moved|renamed|copied|backed\s+up|synced|installed|deployed|configured|"
    r"fixed|resolved|patched|updated|modified|edited|wrote|saved|changed|applied|"
    r"stopped|restarted|reloaded|rebooted|started|launched)\b",
    re.IGNORECASE,
)
_PASSIVE_SUCCESS_RE = re.compile(
    r"\b(?:file|directory|service|server|deployment|configuration|config|task|request)\b"
    r"[^.!?\n]{0,50}\b(?:was|were|has been|have been)\s+"
    r"(?:deleted|removed|created|moved|renamed|copied|installed|deployed|configured|"
    r"fixed|resolved|patched|updated|modified|edited|saved|changed|stopped|restarted|"
    r"reloaded|started)\b[^.!?\n]{0,30}\b(?:successfully|now|complete|completed)\b",
    re.IGNORECASE,
)


def _has_agent_action_claim(output: str) -> bool:
    """Recognize first-person, terse status, and explicit success claims."""
    text = output or ""
    return bool(
        _FIRST_PERSON_ACTION_RE.search(text)
        or _LEADING_ACTION_RE.search(text)
        or _PASSIVE_SUCCESS_RE.search(text)
    )


class OutputVerifier:
    """Pre-send output verification gate. Zero LLM calls."""

    def verify(
        self,
        *,
        output: str,
        user_request: str = "",
        tool_trace: list[dict] | None = None,
        task_type: str = "",
    ) -> VerificationReport:
        report = VerificationReport()
        tool_trace = tool_trace or []
        output_stripped = output.strip()

        # 1. Empty/trivial output check
        report.checked += 1
        if output_stripped.lower() in _TRIVIAL_OUTPUTS and len(output_stripped) < 30:
            # Only flag if the user asked for something substantive
            if len(user_request) > 20 and not _is_confirmation_question(user_request):
                report.issues.append(
                    Issue(
                        severity=IssueSeverity.CRITICAL,
                        category="trivial_output",
                        description=f"Output is trivial/empty ('{output_stripped}') for a substantive request",
                        fix_hint="Provide a complete answer to the user's request",
                    )
                )
            else:
                report.passed += 1
        else:
            report.passed += 1

        # 2. Leaked model-protocol/channel markers
        report.checked += 1
        marker_hits = find_channel_marker_garbage(output)
        if marker_hits:
            report.issues.append(
                Issue(
                    severity=IssueSeverity.CRITICAL,
                    category="channel_marker",
                    description="Output contains internal role/channel protocol markers",
                    evidence=", ".join(marker_hits[:3]),
                    fix_hint="Remove internal role/channel/tool routing markers and return only user-facing text",
                )
            )
        else:
            report.passed += 1

        # 3. Unclosed code fences
        report.checked += 1
        opened = output.count("```")
        if opened % 2 != 0:
            report.issues.append(
                Issue(
                    severity=IssueSeverity.MAJOR,
                    category="markdown",
                    description="Unclosed code block (odd number of ``` markers)",
                    fix_hint="Close the code block with a matching ```",
                )
            )
        else:
            report.passed += 1

        # 4. Code syntax validation
        report.checked += 1
        code_issues = self._check_code_syntax(output)
        python_blocks = sum(
            1
            for match in _CODE_FENCE_RE.finditer(output)
            if (match.group(1) or "").lower() in {"python", "py", "python3"}
        )
        report.objective_checked += python_blocks
        report.objective_passed += max(0, python_blocks - len(code_issues))
        if code_issues:
            report.issues.extend(code_issues)
        else:
            report.passed += 1

        # 5. Arithmetic verification
        report.checked += 1
        arith_issues = self._check_arithmetic(output)
        arithmetic_checks = len(_ARITH_RE.findall(output))
        report.objective_checked += arithmetic_checks
        report.objective_passed += max(0, arithmetic_checks - len(arith_issues))
        if arith_issues:
            report.issues.extend(arith_issues)
        else:
            report.passed += 1

        # 6. File path hallucination
        report.checked += 1
        path_issues = self._check_file_paths(output, tool_trace)
        if path_issues:
            report.issues.extend(path_issues)
        else:
            report.passed += 1

        # 7. Completeness check
        report.checked += 1
        completeness = self._check_completeness(output, user_request)
        if completeness:
            report.issues.extend(completeness)
        else:
            report.passed += 1

        # 8. Incomplete markers in delivered code. Honest prose such as "the
        # upstream feature is not implemented" is a finding, not a defect in
        # the answer and must not trigger an expensive rewrite.
        report.checked += 1
        code_blocks = [match.group(2) for match in _CODE_FENCE_RE.finditer(output)]
        for marker in _INCOMPLETE_MARKERS:
            lines_with_marker = [
                line
                for code in code_blocks
                for line in code.splitlines()
                if marker.lower() in line.lower()
                and not line.strip().startswith(("#", "//", "/*", "*"))
            ]
            if lines_with_marker:
                report.issues.append(
                    Issue(
                        severity=IssueSeverity.MINOR,
                        category="incomplete",
                        description=f"Delivered code contains incomplete marker '{marker}'",
                        evidence=lines_with_marker[0][:100],
                        fix_hint="Complete the implementation or remove the placeholder",
                    )
                )
                break
        else:
            report.passed += 1

        # 9. Claim-result mismatch (claims success but tools failed)
        report.checked += 1
        claim_issues = self._check_claim_result_mismatch(output, tool_trace)
        if claim_issues:
            report.issues.extend(claim_issues)
        else:
            report.passed += 1

        # 10. Unverified action claims (false-success guard)
        report.checked += 1
        unverified = self._check_unverified_action_claims(output, tool_trace)
        mutation_outcome = summarize_mutation_trace(tool_trace)
        if mutation_outcome.attempted and _has_agent_action_claim(output):
            report.objective_checked += 1
            if mutation_outcome.supports_success_claim:
                report.objective_passed += 1
        if unverified:
            report.issues.extend(unverified)
        else:
            report.passed += 1

        return report

    def _check_code_syntax(self, output: str) -> list[Issue]:
        """Parse Python code blocks and check for syntax errors."""
        issues: list[Issue] = []
        for match in _CODE_FENCE_RE.finditer(output):
            lang = (match.group(1) or "").lower()
            code = match.group(2)
            # Unlabelled fences may be pseudocode or another language. Parsing
            # them as Python (or counting shell quotes without a shell parser)
            # creates false blocking findings, so only explicit Python fences
            # receive syntax validation.
            if lang in ("python", "py", "python3"):
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    issues.append(
                        Issue(
                            severity=IssueSeverity.CRITICAL,
                            category="code_syntax",
                            description=f"Python syntax error in code block: {e.msg} (line {e.lineno})",
                            evidence=code.split("\n")[e.lineno - 1][:80]
                            if e.lineno and e.lineno <= len(code.split("\n"))
                            else "",
                            fix_hint=f"Fix the syntax error at line {e.lineno}",
                        )
                    )
        return issues

    def _check_arithmetic(self, output: str) -> list[Issue]:
        """Verify arithmetic expressions in the output."""
        issues: list[Issue] = []
        for match in _ARITH_RE.finditer(output):
            a, op, b, claimed = match.group(1), match.group(2), match.group(3), match.group(4)
            try:
                a_value, b_value, claimed_value = Decimal(a), Decimal(b), Decimal(claimed)
                op_map = {
                    "+": lambda x, y: x + y,
                    "-": lambda x, y: x - y,
                    "*": lambda x, y: x * y,
                    "/": lambda x, y: x / y,
                    "×": lambda x, y: x * y,
                    "÷": lambda x, y: x / y,
                }
                if op in {"/", "÷"} and b_value == 0:
                    issues.append(
                        Issue(
                            severity=IssueSeverity.CRITICAL,
                            category="arithmetic",
                            description=f"Arithmetic error: {a} {op} {b} is undefined",
                            fix_hint="Do not claim a finite result for division by zero",
                        )
                    )
                    continue
                actual = op_map[op](a_value, b_value)
                exponent = claimed_value.as_tuple().exponent
                decimal_places = max(0, -exponent) if isinstance(exponent, int) else 0
                rounding_tolerance = Decimal("0.5") * (Decimal(10) ** -decimal_places)
                if abs(actual - claimed_value) > rounding_tolerance:
                    actual_text = format(actual.normalize(), "f")
                    issues.append(
                        Issue(
                            severity=IssueSeverity.CRITICAL,
                            category="arithmetic",
                            description=f"Arithmetic error: {a} {op} {b} = {claimed} (actual: {actual_text})",
                            fix_hint=f"Correct the result to {actual_text} or state an explicit approximation",
                        )
                    )
            except (InvalidOperation, KeyError, ZeroDivisionError):
                pass
        return issues

    def _check_file_paths(self, output: str, tool_trace: list[dict]) -> list[Issue]:
        """Check if file paths mentioned in output were actually accessed."""
        issues: list[Issue] = []
        # Collect paths from tool trace
        accessed_paths: set[str] = set()
        successful_commands: list[str] = []
        for entry in tool_trace:
            name = str(entry.get("name", ""))
            raw_args = entry.get("args")
            raw_result = entry.get("result")
            args: dict = raw_args if isinstance(raw_args, dict) else {}
            result: dict = raw_result if isinstance(raw_result, dict) else {}
            if not tool_result_succeeded(name, result, args):
                continue
            if name in (
                "read",
                "write",
                "write_chunk",
                "edit",
                "exec",
                "shell",
                "remote_read",
                "remote_write",
                "remote_exec",
                "sandbox_exec",
            ):
                p = args.get("path", "") or args.get("cwd", "")
                if p:
                    accessed_paths.add(str(p))
            result_path = result.get("path")
            if result_path:
                accessed_paths.add(str(result_path))
            if name in {"list_dir", "remote_list"}:
                p = args.get("path", "")
                if p:
                    accessed_paths.add(str(p))
            # Also collect paths from exec/shell output
            if name in ("exec", "shell", "remote_exec", "sandbox_exec"):
                command = str(args.get("command") or args.get("cmd") or "")
                if command:
                    successful_commands.append(command)
                stdout = result.get("stdout", "")
                for m in _FILE_PATH_RE.finditer(str(stdout)):
                    accessed_paths.add(m.group(1))

        # Find paths in output that look like claims of having read/written them
        # Only flag if the output claims to have modified or read a specific file
        claim_patterns = [
            r"(?:wrote|created|modified|updated|saved|edited|patched)\s+(?:to\s+)?[`\"']?(/?[^\s`\"']+\.py|[^\s`\"']+/[^\s`\"']+)",
            r"(?:read|checked|verified|inspected)\s+[`\"']?(/?[^\s`\"']+/[^\s`\"']+)",
        ]
        claimed_paths: set[str] = set()
        for pattern in claim_patterns:
            for m in re.finditer(pattern, output, re.IGNORECASE):
                claimed_paths.add(m.group(1).rstrip(".,;:)]}"))

        # Check if claimed paths were actually accessed
        for cp in claimed_paths:
            cp_normalized = str(Path(cp).expanduser().resolve()) if not cp.startswith("$") else cp
            found = any(
                cp == ap
                or cp_normalized == str(Path(ap).expanduser().resolve())
                or (not Path(cp).is_absolute() and str(ap).replace("\\", "/").endswith("/" + cp))
                for ap in accessed_paths
                if ap
            )
            if not found:
                found = any(
                    re.search(rf"(?<![\w./]){re.escape(cp)}(?![\w./])", command)
                    for command in successful_commands
                )
            if not found and not cp.startswith("$"):
                issues.append(
                    Issue(
                        severity=IssueSeverity.MAJOR,
                        category="path_hallucination",
                        description=f"Output claims to have accessed '{cp}' but no tool call touched this path",
                        fix_hint=f"Actually read/write '{cp}' before claiming to have done so, or remove the claim",
                    )
                )
        return issues

    def _check_completeness(self, output: str, user_request: str) -> list[Issue]:
        """Check if output addresses all parts of the user request."""
        issues: list[Issue] = []
        if not user_request or len(user_request) < 20:
            return issues

        # Only explicit numbered/bulleted deliverables are reliable enough for
        # a blocking heuristic. Splitting ordinary prose on every "and" turns
        # constraints and compound nouns into invented tasks.
        tasks: list[str] = []
        numbered = re.findall(r"(?m)^\s*\d+[.)]\s+([^\n]+)", user_request)
        tasks.extend(numbered)
        bulleted = re.findall(r"(?m)^\s*[-*]\s+([^\n]+)", user_request)
        tasks.extend(bulleted)

        # Filter out negative instructions — "do not modify" is a constraint,
        # not a deliverable the output needs to "address"
        tasks = [
            t for t in tasks if not re.match(r"^(do\s+not|don't|without|never|avoid)", t.strip())
        ]

        # Filter out prerequisite tasks — "read X", "check Y", "look at Z" are
        # actions the agent performs to gather info, not deliverables the output
        # needs to address. The output should address "list", "summarize",
        # "report", "show" etc. — not repeat "I read the file".
        _PREREQUISITE_VERBS = {
            "read",
            "check",
            "look",
            "find",
            "search",
            "scan",
            "review",
            "inspect",
            "examine",
            "browse",
            "open",
            "load",
            "fetch",
            "go",
            "try",
            "test",
            "run",
            "verify",
        }
        tasks = [
            t for t in tasks if not any(t.strip().startswith(v + " ") for v in _PREREQUISITE_VERBS)
        ]

        if not tasks or len(tasks) <= 1:
            return issues

        # Path-like tokens (contain slashes or dots) are not deliverable terms
        _PATH_RE = re.compile(r"[/.]")

        # Check if each task keyword appears in the output
        output_lower = output.lower()
        unaddressed: list[str] = []
        for task in tasks:
            # Extract key terms from the task
            terms = [
                w
                for w in re.findall(r"\b\w{4,}\b", task.lower())
                if w
                not in {
                    "the",
                    "that",
                    "this",
                    "with",
                    "from",
                    "have",
                    "make",
                    "sure",
                    "need",
                    "them",
                    "they",
                    "will",
                    "then",
                    "also",
                    "into",
                    "anything",
                    "everything",
                    "something",
                }
                and not _PATH_RE.search(w)
            ]
            if not terms:
                continue
            # At least 30% of key terms should appear in output
            found = sum(1 for t in terms if t in output_lower)
            if found / len(terms) < 0.3:
                unaddressed.append(task[:60])

        if unaddressed:
            issues.append(
                Issue(
                    severity=IssueSeverity.MINOR,
                    category="completeness",
                    description=f"Output may not address all parts of the request. Unaddressed: {'; '.join(unaddressed[:3])}",
                    fix_hint="Address all parts of the user's request",
                )
            )
        return issues

    def _check_claim_result_mismatch(self, output: str, tool_trace: list[dict]) -> list[Issue]:
        """Check if output claims success but tools actually failed."""
        issues: list[Issue] = []
        if not tool_trace:
            return issues

        success_claims = _has_agent_action_claim(output)

        outcome = summarize_mutation_trace(tool_trace)
        if success_claims and outcome.attempted and outcome.last_mutation_succeeded is False:
            issues.append(
                Issue(
                    severity=IssueSeverity.CRITICAL,
                    category="claim_mismatch",
                    description="Output claims success but the latest mutating tool call failed",
                    evidence=(
                        f"Mutation attempts={outcome.attempted}, "
                        f"succeeded={outcome.succeeded}, failed={outcome.failed}"
                    ),
                    fix_hint="Retry and verify the failed mutation, or report the failure honestly",
                )
            )
        return issues

    def _check_unverified_action_claims(self, output: str, tool_trace: list[dict]) -> list[Issue]:
        """Detect false-success: output claims an action completed but no verification tool call confirms it.

        This catches the staging pattern: agent says "deleted those shows" without ever running
        a post-deletion ls/stat/read to confirm the files are actually gone.

        This is a local, zero-LLM heuristic.
        """
        issues: list[Issue] = []
        if not _has_agent_action_claim(output):
            return issues

        outcome = summarize_mutation_trace(tool_trace)
        if outcome.attempted == 0:
            issues.append(
                Issue(
                    severity=IssueSeverity.CRITICAL,
                    category="false_success",
                    description="Output claims the agent performed an action but no mutation tool call exists in the trace",
                    evidence="Agent action claim in output with zero matching tool calls",
                    fix_hint="Either call the appropriate tool to perform the action, or remove the success claim from the output",
                )
            )
            return issues

        # The mismatch check reports a failed latest mutation. Do not duplicate
        # it here as a second critical issue.
        if outcome.last_mutation_succeeded is False:
            return issues

        if not outcome.verified_after_last_mutation:
            issues.append(
                Issue(
                    severity=IssueSeverity.CRITICAL,
                    category="false_success",
                    description="Output claims actions completed but no post-action verification call (read/list_dir/exec-check) found in tool trace",
                    evidence=(
                        f"Last action tool at trace index {outcome.last_mutation_index}, "
                        "no successful independent verification call after it"
                    ),
                    fix_hint="After performing an action, call read/list_dir/exec(ls/stat/test) to verify the result before claiming success",
                )
            )

        return issues


def _is_confirmation_question(request: str) -> bool:
    """Check if the request is a simple yes/no confirmation."""
    r = request.lower().strip()
    return r in {"ok?", "done?", "yes?", "no?", "ready?", "go?", "sure?"} or len(r) < 10
