#!/usr/bin/env python3
"""
Norax Brain — conservative action risk and syntax gate.

Pipeline-level enforcement for callers that invoke :func:`gate` before an
action.  Pattern classification is not a proof of safety; critical matches are
therefore blocked for an external authority to review.

Based on:
- V&V Loops (muthu.co): Hierarchical checks, fail-fast on cheap checks
- PAG module: Destructive pattern detection (already exists)
- arXiv 2504.09923: Validate intermediate steps, not just final output

Impact levels:
  LOW      → cheap classification only (chat, lookups, reads)
  MEDIUM   → syntax/schema checks (file writes, config reads)
  HIGH     → blocked by default; a trusted host may approve and run checks
  CRITICAL → always blocked here; review/handling belongs to an outer layer

This helper is a conservative pattern/syntax gate, not an authentication or
authorization system. Only a trusted host boundary may set the high-risk
approval argument; never populate it from model output or an unauthenticated
request field.

Usage:
  python3 tools/action_gate.py classify "rm -rf /tmp/test"
  python3 tools/action_gate.py verify "restart ollama" --context "..."
  python3 tools/action_gate.py check-code "def foo(): return 1"
  python3 tools/action_gate.py check-response "The port is 11434" --context "..."
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_configured_workspace = os.environ.get("NORAX_WORKSPACE", "").strip()
WORKSPACE = (
    Path(_configured_workspace).expanduser()
    if _configured_workspace
    else Path(__file__).resolve().parent.parent
)
STATE_FILE = WORKSPACE / "memory" / ".action_gate.json"

# ── Impact Classification Patterns ──────────────────────────────────────────

CRITICAL_PATTERNS = [
    (
        r"(?:^|[\s;&|])(?:\S*/)?rm\s+(?=[^\n;&|]*(?:--recursive\b|-[A-Za-z]*[rR]))",
        "destructive_delete",
    ),
    (r"\bshutil\.rmtree\s*\(", "destructive_delete"),
    (r"\bfind\b[^\n;&|]*\s-delete\b", "destructive_delete"),
    (r"\bfind\b[^\n;&|]*-(?:exec|execdir)\b[^\n;&|]*\brm\b", "destructive_delete"),
    (r"\bxargs\b[^\n;&|]*\brm\b", "destructive_delete"),
    (r"\bDROP\s+(TABLE|DATABASE|INDEX)", "destructive_sql"),
    (
        r"\bdd\b[^\n;&|]*\bof=/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z]|xvd[a-z]|mmcblk\d+|mapper/)",
        "destructive_dd",
    ),
    (r"chmod\s+777\s+/(etc|boot|usr|var)", "dangerous_permissions"),
    (r"mkfs\b", "destructive_format"),
    (
        r">\s*/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z]|xvd[a-z]|mmcblk\d+|mapper/)",
        "destructive_overwrite",
    ),
    (r"truncate\s+-s\s*0", "destructive_truncate"),
    (r"\bgit\s+clean\b[^\n;&|]*(?:-[A-Za-z]*[xfd][A-Za-z]*|--force)\b", "git_clean"),
    (r"(?:^|[;&|\n]\s*)(?:sudo\s+)?(?:shred|wipefs)\b", "destructive_device"),
    (
        r"(?:^|[;&|\n]\s*)(?:sudo\s+)?(?:shutdown|reboot|poweroff|halt)\b",
        "host_lifecycle",
    ),
    (
        r"\bsystemctl\b[^\n;&|]*\b(?:reboot|poweroff|halt)\b",
        "host_lifecycle",
    ),
    (r"\bcurl\b[^\n|]*\|\s*(?:sh|bash)\b", "remote_code_execution"),
]

HIGH_PATTERNS = [
    (
        r"\bsystemctl\b(?:\s+--?[\w-]+)*\s+"
        r"(?:start|restart|stop|reload|try-restart|enable|disable|mask|unmask|daemon-reload)\b",
        "service_operation",
    ),
    (r"service\s+\S+\s+(restart|stop)", "service_operation"),
    (
        r"\bdocker\s+(?:(?:compose|container)\s+)?"
        r"(?:rm|stop|kill|down|restart|pause|unpause|update)\b",
        "container_operation",
    ),
    (r"(?:^|[;&|\n]\s*)(?:sudo\s+)?(?:kill|killall|pkill)\b", "process_operation"),
    (r"deploy|push\s+to\s+prod", "deployment"),
    (r"git\s+push\s+(-f|--force)", "force_push"),
    (r"git\s+reset\s+--hard", "git_destructive"),
    (r"pip\s+install|npm\s+install", "package_install"),
    (r"ollama\s+(rm|delete|pull|create)", "model_operation"),
    (r"iptables|ufw\s+(deny|delete|reset)", "firewall_change"),
    (r"\b(?:scp|rsync)\b[^\n;&|]*--delete", "remote_sync_delete"),
    (r"(?:^|[;&|\n]\s*)(?:sudo\s+)?(?:\S*/)?rm\b", "file_delete"),
    (
        r"\b(?:cp|mv|install|tee)\b[^\n;&|]*"
        r"(?:/usr/(?:local/)?bin/|/etc/systemd/)",
        "runtime_binary_or_unit_change",
    ),
    (r"(?:>|>>)\s*(?:/usr/(?:local/)?bin/|/etc/systemd/)", "runtime_file_change"),
    # Python-specific dangerous patterns
    (r"os\.system\s*\(", "py_os_system"),
    (r"subprocess\.(?:Popen|run|call|check_output)\s*\(", "py_subprocess"),
    (r"__import__\s*\(", "py_dynamic_import"),
    (r"exec\s*\(", "py_exec"),
    (r"eval\s*\(", "py_eval"),
    (r"open\s*\(.+[wa].*\)", "py_file_write"),
]

MEDIUM_PATTERNS = [
    (r"(cat|echo|tee)\s+.*>", "file_write"),
    (r">\s*/", "redirect_to_file"),
    (r"sed\s+-i", "in_place_edit"),
    (r"mv\s+", "file_move"),
    (r"cp\s+", "file_copy"),
    (r"chmod\s+", "permission_change"),
    (r"chown\s+", "ownership_change"),
    (r"git\s+(commit|merge|rebase|cherry-pick)", "git_mutation"),
    (r"curl\s+.*-X\s*(POST|PUT|DELETE|PATCH)", "api_mutation"),
    (r"sqlite3\s+.*\b(INSERT|UPDATE|DELETE)\b", "db_mutation"),
]

# ── State Management ────────────────────────────────────────────────────────


def _load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return _fresh_state()


def _fresh_state():
    return {
        "total_checks": 0,
        "blocked": 0,
        "warned": 0,
        "passed": 0,
        "last_check": None,
        "history": [],  # Last 50 checks
    }


def _save_state(state):
    temporary = STATE_FILE.with_name(f"{STATE_FILE.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, STATE_FILE)
    except Exception:
        pass
    finally:
        temporary.unlink(missing_ok=True)


# ── Impact Classification ───────────────────────────────────────────────────


def classify_impact(action_text):
    """Classify the impact level of an action.

    Returns: {"level": "LOW|MEDIUM|HIGH|CRITICAL", "tag": str, "pattern": str}
    """
    if not action_text:
        return {"level": "LOW", "tag": "empty", "pattern": ""}

    text = action_text.strip()

    for pattern, tag in CRITICAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return {"level": "CRITICAL", "tag": tag, "pattern": pattern}

    for pattern, tag in HIGH_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return {"level": "HIGH", "tag": tag, "pattern": pattern}

    for pattern, tag in MEDIUM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return {"level": "MEDIUM", "tag": tag, "pattern": pattern}

    return {"level": "LOW", "tag": "routine", "pattern": ""}


# ── Verification Functions ──────────────────────────────────────────────────


def verify_python(code_text):
    """Syntax-check Python code. Returns (valid, errors)."""
    errors = []
    try:
        compile(code_text, "<action_gate_check>", "exec")
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
    return (len(errors) == 0, errors)


def verify_shell(command_text):
    """Basic shell command validation. Returns (valid, warnings)."""
    warnings = []

    # Check for common dangerous patterns
    if "&&" in command_text and "rm" in command_text:
        warnings.append("Chained command includes rm — verify target carefully")

    if re.search(r"\$\{?\w+\}?\s*/", command_text):
        warnings.append("Variable expansion in path — ensure variable is set")

    # Run shellcheck when available. Built-in checks remain useful without it,
    # but the result explicitly describes the reduced verification depth.
    if shutil.which("shellcheck") is None:
        warnings.append("shellcheck unavailable; only built-in checks ran")
        return (len(warnings) == 0, warnings)

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(f"#!/bin/bash\n{command_text}\n")
            f.flush()
            temp_path = f.name
            result = subprocess.run(
                ["shellcheck", "-S", "warning", f.name], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                # Parse shellcheck output
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("In ") or "SC" in line:
                        continue
                    line = line.strip()
                    if line and not line.startswith("^"):
                        warnings.append(line[:200])
    except Exception as exc:
        warnings.append(f"shellcheck could not complete: {exc}")
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    return (len(warnings) == 0, warnings)


def verify_json(text):
    """Validate JSON. Returns (valid, errors)."""
    try:
        json.loads(text)
        return (True, [])
    except json.JSONDecodeError as e:
        return (False, [f"JSON error at line {e.lineno}: {e.msg}"])


def verify_service_state(service_name):
    """Check current state of a systemd service before modifying it.
    Returns service info dict."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name], capture_output=True, text=True, timeout=5
        )
        is_active = result.stdout.strip() == "active"

        result2 = subprocess.run(
            ["systemctl", "is-enabled", service_name], capture_output=True, text=True, timeout=5
        )
        is_enabled = result2.stdout.strip() == "enabled"

        return {
            "service": service_name,
            "active": is_active,
            "enabled": is_enabled,
            "status": "running" if is_active else "stopped",
        }
    except Exception as e:
        return {"service": service_name, "error": str(e)}


def verify_file_exists(path):
    """Check if a file exists before deletion or modification."""
    p = Path(path)
    return {
        "exists": p.exists(),
        "is_file": p.is_file() if p.exists() else False,
        "is_dir": p.is_dir() if p.exists() else False,
        "size": p.stat().st_size if p.exists() and p.is_file() else 0,
    }


# ── Response Verification ───────────────────────────────────────────────────


def verify_response_claims(response_text, context_text=""):
    """Check factual claims in a response against provided context.

    Returns list of claims with grounding status.
    """
    claims = []

    # Extract potential factual claims (numbers, ports, versions, paths)
    port_claims = re.findall(r"port\s+(\d+)", response_text, re.IGNORECASE)
    for port in port_claims:
        grounded = port in context_text if context_text else False
        claims.append(
            {
                "type": "port",
                "value": port,
                "grounded": grounded,
                "source": "context" if grounded else "unverified",
            }
        )

    version_claims = re.findall(
        r"(?:version|v)\s*(\d+\.\d+(?:\.\d+)?)", response_text, re.IGNORECASE
    )
    for ver in version_claims:
        grounded = ver in context_text if context_text else False
        claims.append(
            {
                "type": "version",
                "value": ver,
                "grounded": grounded,
                "source": "context" if grounded else "unverified",
            }
        )

    path_claims = re.findall(r"(/[\w./-]{5,})", response_text)
    path_claims = [p.rstrip(".,:;)") for p in path_claims]  # Strip trailing punctuation
    for path in path_claims:
        # Check if path actually exists OR is in context
        exists = Path(path).exists()
        in_context = path in context_text if context_text else False
        claims.append(
            {
                "type": "path",
                "value": path,
                "grounded": exists or in_context,
                "source": "filesystem" if exists else ("context" if in_context else "unverified"),
            }
        )

    ip_claims = re.findall(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", response_text)
    for ip in ip_claims:
        grounded = ip in context_text if context_text else False
        claims.append(
            {
                "type": "ip_address",
                "value": ip,
                "grounded": grounded,
                "source": "context" if grounded else "unverified",
            }
        )

    return claims


# ── Main Gate Function ──────────────────────────────────────────────────────


def gate(
    action_text,
    action_type="command",
    context="",
    sender_id=None,
    *,
    trusted_high_risk_approval=False,
):
    """Main verification gate. Call BEFORE executing any action.

    Args:
        action_text: The command/code/action to verify
        action_type: "command", "python", "shell", "config", "response"
        context: Retrieved context for grounding checks
        sender_id: Deprecated compatibility parameter. Identity supplied by a
            caller is not authentication and cannot bypass a critical block.
        trusted_high_risk_approval: Approval asserted by an authenticated host
            boundary. It applies only to HIGH actions; CRITICAL stays blocked.

    Returns:
        {
            "allowed": bool,
            "level": "LOW|MEDIUM|HIGH|CRITICAL",
            "checks": [...],          # List of check results
            "warnings": [...],        # Non-blocking warnings
            "blocked_reason": str,     # If blocked
            "verification_ms": float,  # Time taken
        }
    """
    t0 = time.monotonic()
    del sender_id

    impact = classify_impact(action_text)
    level = impact["level"]

    checks = []
    warnings = []
    blocked_reason = None
    allowed = True

    supported_action_types = {"command", "shell", "python", "config", "json", "response"}
    if action_type not in supported_action_types:
        allowed = False
        blocked_reason = f"Unsupported action type: {action_type}"
        checks.append({"check": "action_type", "result": "fail", "detail": blocked_reason})
    elif (
        action_type in {"command", "shell", "python", "config", "json"}
        and not str(action_text).strip()
    ):
        allowed = False
        blocked_reason = "Action payload is empty"
        checks.append({"check": "action_payload", "result": "fail", "detail": blocked_reason})

    # ── LOW: No verification needed ─────────────────────────────────────
    if level == "LOW":
        checks.append({"check": "impact_classification", "result": "pass", "detail": "low impact"})

    # ── MEDIUM: Syntax/schema checks ────────────────────────────────────
    elif level == "MEDIUM":
        checks.append(
            {
                "check": "impact_classification",
                "result": "warn",
                "detail": f"medium: {impact['tag']}",
            }
        )

    # ── HIGH: Full verification ─────────────────────────────────────────
    elif level == "HIGH":
        if trusted_high_risk_approval is not True:
            checks.append(
                {
                    "check": "impact_classification",
                    "result": "block",
                    "detail": f"high: {impact['tag']}; trusted host approval required",
                }
            )
            allowed = False
            blocked_reason = (
                f"HIGH action ({impact['tag']}) requires approval from an authenticated host"
            )
        else:
            checks.append(
                {
                    "check": "impact_classification",
                    "result": "warn",
                    "detail": f"high: {impact['tag']}; trusted host approval asserted",
                }
            )

        # Defer expensive validation probes until a trusted boundary approves.
        service_match = (
            re.search(r"(systemctl|service)\s+\S+\s+(\S+)", action_text)
            if trusted_high_risk_approval is True
            else None
        )
        if service_match is not None:
            svc_name = (
                service_match.group(2)
                if "systemctl" in service_match.group(1)
                else service_match.group(2)
            )
            # Try to find the actual service name
            svc_name2 = re.search(r"(?:restart|stop|start|disable)\s+(\S+)", action_text)
            if svc_name2:
                svc_name = svc_name2.group(1).removesuffix(".service")
            state = verify_service_state(svc_name)
            checks.append({"check": "service_state", "result": "info", "detail": state})
            if state.get("active"):
                warnings.append(
                    f"Service {svc_name} is currently ACTIVE — operation will affect running service"
                )

        # Shell command checks
        if trusted_high_risk_approval is True and action_type in ("shell", "command"):
            valid, warns = verify_shell(action_text)
            checks.append(
                {
                    "check": "shell_validation",
                    "result": "pass" if valid else "warn",
                    "warnings": warns,
                }
            )
            warnings.extend(warns)

        # File deletion checks
        del_match = (
            re.search(r"rm\s+(?:-\w+\s+)?(\S+)", action_text)
            if trusted_high_risk_approval is True
            else None
        )
        if del_match:
            file_info = verify_file_exists(del_match.group(1))
            checks.append({"check": "file_exists", "result": "info", "detail": file_info})
            if file_info.get("is_dir"):
                warnings.append(f"Target is a DIRECTORY: {del_match.group(1)}")

    # ── CRITICAL: Require owner approval ────────────────────────────────
    elif level == "CRITICAL":
        checks.append(
            {
                "check": "impact_classification",
                "result": "block",
                "detail": f"CRITICAL: {impact['tag']}",
            }
        )

        blocked_reason = (
            f"CRITICAL action ({impact['tag']}) requires approval in an authenticated outer layer"
        )
        allowed = False

    # Payload validation is determined by the declared type, not by whether a
    # risk regex happened to match. Malformed low-impact code/config must not
    # bypass syntax checks, and each payload is checked only once.
    if action_type == "python":
        valid, errs = verify_python(action_text)
        checks.append(
            {"check": "python_syntax", "result": "pass" if valid else "fail", "errors": errs}
        )
        if not valid:
            warnings.extend(errs)
            allowed = False
            blocked_reason = blocked_reason or "Python syntax validation failed"

    if action_type in {"config", "json"}:
        valid, errs = verify_json(action_text)
        checks.append(
            {"check": "json_valid", "result": "pass" if valid else "fail", "errors": errs}
        )
        if not valid:
            warnings.extend(errs)
            allowed = False
            blocked_reason = blocked_reason or "JSON validation failed"

    # ── Response verification (separate from action checks) ─────────────
    if action_type == "response" and context:
        claims = verify_response_claims(action_text, context)
        ungrounded = [c for c in claims if not c["grounded"]]
        if ungrounded:
            grounding_check: dict[str, Any] = {
                "check": "claim_grounding",
                "result": "warn",
                "ungrounded_count": len(ungrounded),
                "total_claims": len(claims),
                "ungrounded": ungrounded[:5],
            }
            checks.append(grounding_check)
            for c in ungrounded[:3]:
                warnings.append(f"[unverified] {c['type']}: {c['value']}")

    # Optional telemetry must not add file I/O to every action by default.
    if os.environ.get("NORAX_ACTION_GATE_TELEMETRY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        state = _load_state()
        state["total_checks"] += 1
        if not allowed:
            state["blocked"] += 1
        elif warnings:
            state["warned"] += 1
        else:
            state["passed"] += 1
        state["last_check"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        state["history"].append(
            {
                "ts": state["last_check"],
                "level": level,
                "tag": impact["tag"],
                "allowed": allowed,
                "warnings": len(warnings),
            }
        )
        state["history"] = state["history"][-50:]
        _save_state(state)

    elapsed_ms = (time.monotonic() - t0) * 1000

    return {
        "allowed": allowed,
        "level": level,
        "tag": impact["tag"],
        "checks": checks,
        "warnings": warnings,
        "blocked_reason": blocked_reason,
        "requires_external_approval": level in {"HIGH", "CRITICAL"} and not allowed,
        "authorization_basis": (
            "trusted_host_assertion"
            if level == "HIGH" and trusted_high_risk_approval is True
            else "not_evaluated"
        ),
        "assessment_kind": "pattern_and_syntax_heuristic",
        "verification_ms": round(elapsed_ms, 1),
    }


# ── Inject Signal (for brain-runner integration) ────────────────────────────


def inject_signal(action_text="", action_type="command", context=""):
    """Generate gate signal for injection into brain-runner output.

    Returns empty string if no concerns, or a warning/block signal.
    """
    if not action_text:
        return ""

    result = gate(action_text, action_type, context)

    if not result["allowed"]:
        return f"🛑 ACTION_BLOCKED: {result['blocked_reason']}"

    if result["warnings"]:
        warns = "; ".join(result["warnings"][:3])
        return f"⚠️ ACTION_GATE({result['level']}): {warns}"

    return ""


def stats():
    """Return gate statistics."""
    state = _load_state()
    return {
        "total": state.get("total_checks", 0),
        "blocked": state.get("blocked", 0),
        "warned": state.get("warned", 0),
        "passed": state.get("passed", 0),
        "last": state.get("last_check"),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "stats":
        print(json.dumps(stats(), indent=2))
        sys.exit(0)
    if len(sys.argv) < 3:
        print(
            "Usage: action_gate.py classify|verify|check-code|check-response 'text' [--context 'ctx']"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    text = sys.argv[2]
    ctx = ""
    if "--context" in sys.argv:
        idx = sys.argv.index("--context")
        if idx + 1 < len(sys.argv):
            ctx = sys.argv[idx + 1]

    if cmd == "classify":
        result = classify_impact(text)
        print(json.dumps(result, indent=2))
    elif cmd == "verify":
        result = gate(text, "command", ctx)
        print(json.dumps(result, indent=2))
        if not result["allowed"]:
            sys.exit(2)
    elif cmd == "check-code":
        valid, errs = verify_python(text)
        print(f"Valid: {valid}")
        if errs:
            for e in errs:
                print(f"  Error: {e}")
            sys.exit(2)
    elif cmd == "check-response":
        result = gate(text, "response", ctx)
        print(json.dumps(result, indent=2))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
