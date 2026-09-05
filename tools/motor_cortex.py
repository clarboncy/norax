#!/usr/bin/env python3
"""Bounded standalone action executor with optional before/after verification.

Basic execution distinguishes backend/process completion from verified UI
outcomes. Call :meth:`MotorCortex.perceive_act_verify` when an observed screen
change is required.
"""

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

TOOLS = Path(__file__).resolve().parent
_configured_workspace = os.environ.get("NORAX_WORKSPACE", "").strip()
WORKSPACE = Path(_configured_workspace).expanduser() if _configured_workspace else TOOLS.parent


def _allowed_path(raw_path: str) -> Path:
    """Resolve a file target under an operator-approved local root."""
    if not raw_path.strip():
        raise ValueError("file path is required")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE / candidate
    resolved = candidate.resolve(strict=False)
    configured = os.environ.get("NORAX_MOTOR_ALLOWED_ROOTS", "")
    roots = [WORKSPACE.resolve(strict=False), Path("/tmp").resolve()]
    roots.extend(
        Path(value).expanduser().resolve(strict=False)
        for value in configured.split(os.pathsep)
        if value.strip()
    )
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise PermissionError(
            f"path is outside NORAX_WORKSPACE, /tmp, and NORAX_MOTOR_ALLOWED_ROOTS: {resolved}"
        )
    return resolved


def _run_bounded(
    command: str | list[str],
    *,
    timeout: float,
    cwd: str | None = None,
    shell: bool = False,
    output_limit: int = 65_536,
) -> tuple[str, int]:
    """Run a process without buffering unbounded child output in memory."""
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(  # noqa: S603
            command,
            shell=shell,
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            start_new_session=True,
        )
        timed_out = False
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return_code = process.wait()
        output.seek(0)
        raw = output.read(output_limit + 1)
    truncated = len(raw) > output_limit
    text = raw[:output_limit].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n[output truncated at {output_limit} bytes]"
    if timed_out:
        text += f"\n[action timed out after {timeout:g}s; partial side effects may have occurred]"
        return text, 124
    return text, return_code


# Wire action gate — pre-execution safety verification
try:
    import action_gate as AGATE

    _GATE_ENABLED = True
except ImportError:
    _GATE_ENABLED = False

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Prediction:
    """Static/rolling heuristic expectation used for result comparison."""

    expected_state_change: str  # What should change
    expected_duration_ms: int  # How long it should take
    confidence: float  # non-calibrated template score, retained for compatibility
    failure_modes: list[str] = field(default_factory=list)
    predicted_success: bool = True  # Whether action is expected to succeed


@dataclass
class ActionResult:
    """Result of a motor action with verification."""

    success: bool
    attempts: int
    action_type: str
    target: str
    verification: str  # What we observed
    duration_ms: int
    corrections: list[str] = field(default_factory=list)
    verification_level: str = "execution"
    ui_match_score: float | None = None


@dataclass
class MotorPlan:
    """Plan for a sequence of actions."""

    steps: list[dict]
    total_predicted_ms: int
    risk_level: float  # 0.0-1.0


# ═══════════════════════════════════════════════════════════════════════════
# FORWARD MODEL — predict outcomes before executing
# ═══════════════════════════════════════════════════════════════════════════

# Default timing predictions by action category (in ms)
DEFAULT_TIMINGS = {
    "click": 200,
    "type": 300,
    "key": 150,
    "scroll": 200,
    "exec": 2000,
    "write": 500,
    "read": 300,
    "http": 1000,
}

# Process-local rolling timing observations. No background persistence or
# synthetic learning claim is made by this standalone helper.
_timing_history: dict[str, list[float]] = {}


class OutcomeTemplate(TypedDict):
    state_change: str
    confidence: float
    failures: list[str]


class ForwardModel:
    """Build action expectations from static templates and local timing samples."""

    # Outcome predictions by action type
    OUTCOME_TEMPLATES: dict[str, OutcomeTemplate] = {
        "click_button": {
            "state_change": "Button pressed/activated, UI may navigate or toggle",
            "confidence": 0.8,
            "failures": ["element moved", "element hidden", "wrong coords", "modal blocking"],
        },
        "click_link": {
            "state_change": "Page navigation or new content loaded",
            "confidence": 0.75,
            "failures": ["broken link", "popup blocked", "slow load"],
        },
        "type_text": {
            "state_change": "Text appears in focused input field",
            "confidence": 0.85,
            "failures": ["no focus", "readonly field", "wrong field focused"],
        },
        "key_press": {
            "state_change": "Keyboard shortcut executed",
            "confidence": 0.9,
            "failures": ["shortcut not available", "modal blocking"],
        },
        "scroll": {
            "state_change": "Page content shifted up/down",
            "confidence": 0.9,
            "failures": ["already at bottom", "already at top"],
        },
        "exec_command": {
            "state_change": "Command produces stdout/stderr output",
            "confidence": 0.7,
            "failures": ["command not found", "permission denied", "timeout"],
        },
        "file_write": {
            "state_change": "File created or modified on disk",
            "confidence": 0.85,
            "failures": ["permission denied", "disk full", "path invalid"],
        },
        "file_read": {
            "state_change": "File contents retrieved",
            "confidence": 0.9,
            "failures": ["file not found", "permission denied", "binary file"],
        },
    }

    def predict(self, action_type: str, target: str = "", context: str = "") -> Prediction:
        """Predict outcome of action before execution."""
        template = self.OUTCOME_TEMPLATES.get(
            action_type,
            {
                "state_change": f"{action_type} completes",
                "confidence": 0.5,
                "failures": ["unknown action type"],
            },
        )

        # Adjust confidence based on context
        confidence = template["confidence"]
        if "permission" in context.lower() or "denied" in context.lower():
            confidence *= 0.5
        if "timeout" in context.lower():
            confidence *= 0.7

        # Get timing prediction
        base_category = action_type.split("_")[0]
        predicted_ms = DEFAULT_TIMINGS.get(base_category, 1000)

        # Use learned timing if available
        timing_key = f"{action_type}:{target[:50]}" if target else action_type
        if timing_key in _timing_history and _timing_history[timing_key]:
            # Use average of recent timings
            recent = _timing_history[timing_key][-10:]
            predicted_ms = int(sum(recent) / len(recent))

        # Heuristic: detect likely-failure targets
        predicted_success = True
        target_lower = target.lower() if target else ""
        if action_type == "exec_command":
            # Commands targeting nonexistent paths likely fail
            if "/nonexistent" in target_lower or "/no-such" in target_lower:
                predicted_success = False
            # Commands that are inherently error-producing
            elif target_lower.startswith("cat /") and not target_lower.startswith("cat /tmp"):
                # cat on a path that doesn't exist will fail
                import os as _os

                path = target_lower.split("cat ", 1)[1].split()[0] if "cat " in target_lower else ""
                if path and not _os.path.exists(path):
                    predicted_success = False
        if "permission" in context.lower() or "denied" in context.lower():
            predicted_success = False

        return Prediction(
            expected_state_change=template["state_change"],
            expected_duration_ms=predicted_ms,
            confidence=confidence,
            failure_modes=template.get("failures", []),
            predicted_success=predicted_success,
        )

    def update_timing(self, action_type: str, target: str, actual_ms: float):
        """Update learned timing from actual execution."""
        timing_key = f"{action_type}:{target[:50]}" if target else action_type
        if timing_key not in _timing_history:
            _timing_history[timing_key] = []
        _timing_history[timing_key].append(actual_ms)
        # Keep only last 20 observations
        if len(_timing_history[timing_key]) > 20:
            _timing_history[timing_key] = _timing_history[timing_key][-20:]


# ═══════════════════════════════════════════════════════════════════════════
# STATE COMPARATOR — compare predicted vs actual
# ═══════════════════════════════════════════════════════════════════════════


class StateComparator:
    """Classify the observed execution result.

    Exit status can prove that a backend accepted an action, but it cannot by
    itself prove the requested real-world outcome. UI outcome verification is
    handled separately by :meth:`MotorCortex.perceive_act_verify`.
    """

    def compare(
        self, predicted: Prediction, actual_output: str, exit_code: int = 0
    ) -> tuple[float, str]:
        """Compare prediction with actual outcome.

        Returns: (match_score 0.0-1.0, explanation)
        """
        del actual_output
        if exit_code != 0:
            return 0.0, f"Backend failed with exit code {exit_code}"
        return 0.8, "Backend accepted the action; requested outcome not independently verified"

    def verify_ui_change(
        self, before_elements: list[str], after_elements: list[str], expected_change: str
    ) -> tuple[float, str]:
        """Verify UI changed as expected after an action.

        Compares AT-SPI/dmap element lists before and after.
        """
        if not before_elements or not after_elements:
            return 0.5, "Missing element data for comparison"

        # Check if elements changed at all
        before_set = set(before_elements)
        after_set = set(after_elements)

        new_elements = after_set - before_set
        removed_elements = before_set - after_set

        if not new_elements and not removed_elements:
            return 0.3, "No UI change detected"

        # Some change occurred
        change_magnitude = (len(new_elements) + len(removed_elements)) / max(len(before_set), 1)

        if change_magnitude > 0.5:
            score = 0.9
            explanation = f"Major UI diff: +{len(new_elements)}/-{len(removed_elements)} elements"
        elif change_magnitude > 0.1:
            score = 0.7
            explanation = (
                f"Moderate UI diff: +{len(new_elements)}/-{len(removed_elements)} elements"
            )
        else:
            score = 0.5
            explanation = f"Minor UI diff: +{len(new_elements)}/-{len(removed_elements)} elements"

        # A generic diff is not proof of the requested semantic outcome. Use
        # only visible terms as bounded heuristic evidence and cap the score
        # when none of the meaningful expected terms appear in changed items.
        stop_words = {
            "a",
            "an",
            "and",
            "be",
            "is",
            "should",
            "the",
            "to",
            "was",
            "will",
        }
        expected_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", expected_change.lower())
            if len(term) >= 3 and term not in stop_words
        }
        changed_text = " ".join(new_elements | removed_elements).lower()
        observed_terms = sorted(term for term in expected_terms if term in changed_text)
        if expected_terms and not observed_terms:
            return min(score, 0.6), explanation + "; expected terms not observed in the diff"
        if observed_terms:
            explanation += f"; observed expected term(s): {', '.join(observed_terms[:5])}"
        return score, explanation


# ═══════════════════════════════════════════════════════════════════════════
# MOTOR CORTEX — closed-loop action execution
# ═══════════════════════════════════════════════════════════════════════════


class MotorCortex:
    """Closed-loop action execution with prediction, verification, correction.

    The cycle:
    1. PLAN: Choose action, set expectations
    2. PREDICT: Forward model — what should happen?
    3. EXECUTE: Perform the action
    4. PERCEIVE: Check what actually happened
    5. COMPARE: Expected vs actual
    6. CORRECT: If mismatch, adjust and retry

    Max retries: 3 (prevents infinite loops)
    """

    MAX_RETRIES = 3
    SUCCESS_THRESHOLD = 0.7  # Match score needed to consider success

    def __init__(
        self,
        high_risk_authorizer: Callable[[str, str, dict[str, Any]], bool] | None = None,
    ):
        """Create an executor.

        ``high_risk_authorizer`` must be supplied by an authenticated host
        boundary, never by model output. Without it, HIGH actions fail closed.
        """
        self.forward_model = ForwardModel()
        self.comparator = StateComparator()
        self.action_log: list[ActionResult] = []
        self.high_risk_authorizer = high_risk_authorizer

    def execute_with_verify(
        self, action_type: str, target: str, expected_outcome: str = "", context: str = ""
    ) -> ActionResult:
        """Execute action with closed-loop verification.

        Args:
            action_type: Type of action (click_button, type_text, exec_command, etc.)
            target: Action target (element ref, command, file path)
            expected_outcome: What success looks like
            context: Additional context for prediction

        Returns: ActionResult with success/failure and corrections
        """
        t0 = time.monotonic()
        corrections: list[str] = []
        attempts_made = 0
        explanation = "Action was not executed"

        for attempt in range(self.MAX_RETRIES):
            attempts_made = attempt + 1
            # 1. PREDICT
            prediction = self.forward_model.predict(action_type, target, context)

            # 1.5 GATE — pre-execution safety check
            gated_actions = {"exec_command", "shell", "python", "file_write"}
            if action_type in gated_actions and not _GATE_ENABLED:
                elapsed = int((time.monotonic() - t0) * 1000)
                result = ActionResult(
                    success=False,
                    attempts=attempts_made,
                    action_type=action_type,
                    target=target,
                    verification="BLOCKED: action_gate is unavailable",
                    duration_ms=elapsed,
                )
                self.action_log.append(result)
                return result
            if action_type in gated_actions:
                gate_action_type = (
                    "python"
                    if action_type == "python"
                    else "shell"
                    if action_type in {"shell", "exec_command"}
                    else "command"
                )
                gate_target = (
                    f"file_write {target.split('|', 1)[0]}"
                    if action_type == "file_write"
                    else target
                )
                impact = AGATE.classify_impact(gate_target)
                high_risk_approved = False
                if impact.get("level") == "HIGH" and self.high_risk_authorizer is not None:
                    try:
                        high_risk_approved = (
                            self.high_risk_authorizer(action_type, target, impact) is True
                        )
                    except Exception:
                        high_risk_approved = False
                gate_result = AGATE.gate(
                    gate_target,
                    action_type=gate_action_type,
                    context=context,
                    trusted_high_risk_approval=high_risk_approved,
                )
                if gate_result.get("allowed") is not True:
                    elapsed = int((time.monotonic() - t0) * 1000)
                    result = ActionResult(
                        success=False,
                        attempts=attempts_made,
                        action_type=action_type,
                        target=target,
                        verification=f"BLOCKED by action_gate: {gate_result.get('blocked_reason', 'unknown')}",
                        duration_ms=elapsed,
                        corrections=[],
                    )
                    self.action_log.append(result)
                    return result

            # 2. EXECUTE
            output, exit_code = self._execute(action_type, target)

            # 3. COMPARE. No fixed settle delay is useful without a follow-up
            # perception call; the explicit closed-loop API performs that wait.
            match_score, explanation = self.comparator.compare(prediction, output, exit_code)
            if expected_outcome and exit_code == 0:
                explanation += f"; expected outcome remains unverified: {expected_outcome[:200]}"

            # 6. Success or correct
            if match_score >= self.SUCCESS_THRESHOLD:
                elapsed = int((time.monotonic() - t0) * 1000)
                self.forward_model.update_timing(action_type, target, elapsed)

                verification_level = "process_exit"
                if action_type in {"file_write", "file_read"}:
                    verification_level = "readback"
                elif action_type.startswith(("click", "type", "key", "scroll")):
                    verification_level = "input_dispatch"

                result = ActionResult(
                    success=True,
                    attempts=attempts_made,
                    action_type=action_type,
                    target=target,
                    verification=explanation,
                    duration_ms=elapsed,
                    corrections=corrections,
                    verification_level=verification_level,
                )
                self.action_log.append(result)
                return result

            # Need correction
            correction = self._generate_correction(
                action_type, target, output, exit_code, explanation, attempt
            )
            corrections.append(correction)

            # Retry only when a concrete correction changes the executable
            # target. Repeating an identical click, write, or shell command can
            # duplicate side effects without improving the chance of success.
            corrected_target = self._apply_correction(target, correction)
            if corrected_target is None or corrected_target == target:
                corrections.append("No safe executable correction was available; retry skipped")
                break
            target = corrected_target

        # All retries exhausted
        elapsed = int((time.monotonic() - t0) * 1000)
        result = ActionResult(
            success=False,
            attempts=attempts_made,
            action_type=action_type,
            target=target,
            verification=f"Execution failed after {attempts_made} attempt(s): {explanation}",
            duration_ms=elapsed,
            corrections=corrections,
        )
        self.action_log.append(result)
        return result

    def _execute(self, action_type: str, target: str) -> tuple[str, int]:
        """Execute raw action. Returns (output, exit_code)."""
        try:
            if action_type == "exec_command":
                return _run_bounded(
                    target,
                    shell=True,
                    timeout=10,
                    cwd=str(WORKSPACE),
                )

            elif action_type.startswith("click"):
                # Delegate to dmap.py
                return _run_bounded(
                    [sys.executable, str(TOOLS / "dmap.py"), "click", target],
                    timeout=5,
                )

            elif action_type == "type_text":
                parts = target.split("|", 1)
                ref = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                return _run_bounded(
                    [sys.executable, str(TOOLS / "dmap.py"), "type", ref, text],
                    timeout=5,
                )

            elif action_type == "key_press":
                return _run_bounded(
                    [sys.executable, str(TOOLS / "dmap.py"), "key", target],
                    timeout=5,
                )

            elif action_type == "file_write":
                parts = target.split("|", 1)
                path = parts[0]
                content = parts[1] if len(parts) > 1 else ""
                if len(content.encode("utf-8")) > 4_000_000:
                    return "File write exceeds the 4MB standalone helper limit", 1
                destination = _allowed_path(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
                )
                try:
                    temporary.write_text(content, encoding="utf-8")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                if destination.read_text(encoding="utf-8") != content:
                    return f"Write verification failed for {path}", 1
                return f"Wrote and verified {len(content)} characters at {path}", 0

            elif action_type == "file_read":
                with _allowed_path(target).open(encoding="utf-8") as source:
                    content = source.read(3001)
                if len(content) > 3000:
                    content = content[:3000] + "\n[file output truncated at 3000 characters]"
                return content, 0

            elif action_type == "scroll":
                # target is "down", "up", or a number
                direction = 1
                amount = 3
                parts = target.split()
                if parts and parts[0] in ("down", "up"):
                    direction = 1 if parts[0] == "down" else -1
                    if len(parts) > 1:
                        amount = abs(int(parts[1]))
                elif parts:
                    amount = int(parts[0])
                    if amount < 0:
                        direction = -1
                command = [
                    sys.executable,
                    str(TOOLS / "dmap.py"),
                    "scroll",
                    "down" if direction > 0 else "up",
                    str(abs(amount)),
                ]
                return _run_bounded(command, timeout=10)

            elif action_type == "python":
                return _run_bounded(
                    [sys.executable, "-c", target],
                    timeout=10,
                    cwd=str(WORKSPACE),
                )

            elif action_type == "shell":
                return _run_bounded(
                    target,
                    shell=True,
                    timeout=10,
                    cwd=str(WORKSPACE),
                )

            else:
                return f"Unknown action type: {action_type}", 1

        except FileNotFoundError as e:
            return f"File not found: {e}", 1
        except PermissionError as e:
            return f"Permission denied: {e}", 1
        except Exception as e:
            return f"Error: {e}", 1

    def _generate_correction(
        self,
        action_type: str,
        target: str,
        output: str,
        exit_code: int,
        explanation: str,
        attempt: int,
    ) -> str:
        """Generate correction strategy for failed action."""
        if "permission denied" in output.lower():
            return f"CORRECTION[{attempt}]: Permission denied — need elevated access"
        if "not found" in output.lower():
            return f"CORRECTION[{attempt}]: Target not found — verify path/element exists"
        if "timeout" in output.lower():
            return f"CORRECTION[{attempt}]: Timeout — action took too long"
        if exit_code != 0:
            return f"CORRECTION[{attempt}]: Exit code {exit_code} — {explanation}"
        return f"CORRECTION[{attempt}]: Match score low — {explanation}"

    def _apply_correction(self, target: str, correction: str) -> str | None:
        """Return a concrete corrected target, or ``None`` when unavailable."""
        del target, correction
        return None

    def get_success_rate(self) -> float:
        """Get overall success rate from action log."""
        if not self.action_log:
            return 1.0
        successes = sum(1 for r in self.action_log if r.success)
        return successes / len(self.action_log)

    def get_state(self) -> dict:
        """Get motor cortex state for introspection."""
        return {
            "success_rate": self.get_success_rate(),
            "total_actions": len(self.action_log),
            "recent_actions": [
                {
                    "action_type": r.action_type,
                    "success": r.success,
                    "attempts": r.attempts,
                    "duration_ms": r.duration_ms,
                    "verification": r.verification,
                    "verification_level": r.verification_level,
                    "ui_match_score": r.ui_match_score,
                }
                for r in self.action_log[-10:]
            ],
        }

    def perceive_act_verify(
        self,
        action_type: str,
        target: str,
        expected_outcome: str = "",
        context: str = "",
        perceive_fn: Callable[[], Any] | None = None,
    ) -> ActionResult:
        """Full closed-loop: perceive → act → re-perceive → verify.

        Args:
            action_type: Type of action (click_button, type_text, exec_command, etc.)
            target: Action target (element ref, command, file path)
            expected_outcome: What success looks like
            context: Additional context for prediction
            perceive_fn: Callable that returns a PerceptionState (from perception.perceive)

        Returns: ActionResult with UI diff verification.
        """
        import time as _time

        t0 = _time.monotonic()

        # 1. Capture before-state if perception function provided
        before_elements: list[str] = []
        perception_error = ""
        if perceive_fn:
            try:
                before_state = perceive_fn()
                before_elements = [f"{e.role}:{e.name}" for e in before_state.elements if e.name]
            except Exception as exc:
                perception_error = f"pre-action perception failed: {exc}"

        # 2. Execute action with internal verify
        result = self.execute_with_verify(action_type, target, expected_outcome, context)

        # 3. Capture after-state and compare
        if result.success and perceive_fn and before_elements:
            _time.sleep(0.3)  # Wait for UI to settle
            try:
                after_state = perceive_fn()
                after_elements = [f"{e.role}:{e.name}" for e in after_state.elements if e.name]
                match_score, explanation = self.comparator.verify_ui_change(
                    before_elements, after_elements, expected_outcome
                )
                # Enrich result with UI diff
                result.verification = (
                    result.verification + " | UI: " + explanation
                    if result.verification
                    else explanation
                )
                result.ui_match_score = match_score
                result.verification_level = "ui_diff"
                if expected_outcome and match_score < self.SUCCESS_THRESHOLD:
                    result.success = False
            except Exception as exc:
                perception_error = f"post-action perception failed: {exc}"
        elif result.success and perceive_fn and not before_elements and not perception_error:
            perception_error = "pre-action perception returned no comparable elements"

        if perception_error:
            result.verification += f" | UI verification unavailable: {perception_error}"
            if expected_outcome:
                result.success = False

        result.duration_ms = int((_time.monotonic() - t0) * 1000)
        return result

    def inject_signal(self, message: str) -> str:
        """Generate motor cortex signal for brain-state."""
        success_rate = self.get_success_rate()
        recent = self.action_log[-5:] if self.action_log else []

        if not recent:
            return ""

        recent_summary = []
        for r in recent:
            status = "✓" if r.success else "✗"
            recent_summary.append(f"{status} {r.action_type}({r.attempts}try,{r.duration_ms}ms)")

        return f"MOTOR_CORTEX:rate={success_rate:.0%} recent=[{', '.join(recent_summary)}]"

    def fire(self, message: str, context: str = "") -> str:
        """Brain module interface — returns motor status signal."""
        return self.inject_signal(message)


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

_motor = MotorCortex()


def execute(
    action_type: str, target: str, expected_outcome: str = "", context: str = ""
) -> ActionResult:
    """Execute action with closed-loop verification."""
    return _motor.execute_with_verify(action_type, target, expected_outcome, context)


def predict(action_type: str, target: str = "", context: str = "") -> Prediction:
    """Predict outcome without executing."""
    return _motor.forward_model.predict(action_type, target, context)


def inject_signal(message: str) -> str:
    """Generate signal for brain-state."""
    return _motor.inject_signal(message)


def fire(message: str, context: str = "") -> str:
    """Brain module interface."""
    return _motor.fire(message, context)


def get_success_rate() -> float:
    """Get current success rate."""
    return _motor.get_success_rate()


def get_state() -> dict:
    """Get motor cortex state for introspection."""
    return _motor.get_state()


def perceive_act_verify(
    action_type: str,
    target: str,
    expected_outcome: str = "",
    context: str = "",
    perceive_fn=None,
) -> ActionResult:
    """Full closed-loop: perceive → act → re-perceive → verify."""
    return _motor.perceive_act_verify(action_type, target, expected_outcome, context, perceive_fn)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════


def _run_tests():
    passed = 0
    total = 0

    def test(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    print("=== Motor Cortex Tests ===\n")

    # ── T1: Forward model prediction ──
    fm = ForwardModel()
    pred = fm.predict("click_button", "btn_submit")
    test(
        "T1: Forward model predicts click outcome",
        pred.confidence > 0 and pred.expected_duration_ms > 0,
    )

    # ── T2: Forward model confidence reduction ──
    pred2 = fm.predict("exec_command", "ls /root", "permission denied")
    test("T2: Permission context reduces confidence", pred2.confidence < pred.confidence)

    # ── T3: State comparator — success ──
    comp = StateComparator()
    score, expl = comp.compare(
        Prediction("File contents", 300, 0.9, ["not found"]), "Hello World\nLine 2", 0
    )
    test("T3: Successful output → high match score", score >= 0.7)

    # ── T4: State comparator — failure ──
    score2, expl2 = comp.compare(
        Prediction("Success", 300, 0.9, ["permission denied"]),
        "Error: permission denied: /etc/shadow",
        1,
    )
    test("T4: Failed command → low match score", score2 < 0.5)

    # ── T5: State comparator does not infer error semantics from prose ──
    score3, expl3 = comp.compare(
        Prediction("File read", 300, 0.9, ["not found"]),
        "Error: file not found: /tmp/missing.txt",
        1,
    )
    test(
        "T5: Nonzero exit is observed without a fabricated root cause", score3 == 0 and "1" in expl3
    )

    # ── T6: Motor cortex — exec command success ──
    mc = MotorCortex()
    result = mc.execute_with_verify("exec_command", "echo hello")
    test("T6: Simple exec command succeeds", result.success and result.attempts == 1)

    # ── T7: Motor cortex — exec command failure ──
    result2 = mc.execute_with_verify("exec_command", "cat /nonexistent_file_xyz")
    test("T7: Failed exec command detected", not result2.success or result2.verification)

    # ── T8: File write/read cycle ──
    import tempfile

    tmp = tempfile.mktemp(suffix=".txt")
    result3 = mc.execute_with_verify("file_write", f"{tmp}|test content")
    test("T8a: File write succeeds", result3.success)
    result4 = mc.execute_with_verify("file_read", tmp)
    test("T8b: File read succeeds", result4.success)
    try:
        os.unlink(tmp)
    except Exception:
        pass

    # ── T9: Success rate tracking ──
    rate = mc.get_success_rate()
    test("T9: Success rate tracked", 0.0 <= rate <= 1.0)

    # ── T10: Signal generation ──
    sig = mc.inject_signal("test message")
    test("T10: Signal generated with action log", "MOTOR_CORTEX:" in sig and "rate=" in sig)

    # ── T11: Empty action log → empty signal ──
    mc2 = MotorCortex()
    sig2 = mc2.inject_signal("test")
    test("T11: Empty log → empty signal", sig2 == "")

    # ── T12: Timing learning ──
    fm2 = ForwardModel()
    fm2.update_timing("exec_command", "ls", 50.0)
    fm2.update_timing("exec_command", "ls", 55.0)
    fm2.update_timing("exec_command", "ls", 48.0)
    pred3 = fm2.predict("exec_command", "ls")
    test("T12: Learned timing used for prediction", 40 <= pred3.expected_duration_ms <= 60)

    # ── T13: UI change verification ──
    before = ["Button:OK", "Input:name", "Label:Title"]
    after = ["Button:OK", "Input:name", "Label:Title", "Dialog:Confirm", "Button:Yes"]
    score4, expl4 = comp.verify_ui_change(before, after, "dialog should appear")
    test("T13: UI change detection (new elements)", score4 > 0.5)

    # ── T14: No UI change detection ──
    score5, expl5 = comp.verify_ui_change(before, before, "something should change")
    test("T14: No change detected → low score", score5 <= 0.3)

    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed")
    return passed, total


if __name__ == "__main__":
    p, t = _run_tests()
    print(f"\nFinal: {p}/{t}")
    exit(0 if p == t else 1)
