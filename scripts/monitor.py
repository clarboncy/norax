#!/usr/bin/env python3
"""Truthful operational health report for the active Norax deployment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

UNITS = (
    ("norax-ai.service", True),
    ("norax-remote-relay.service", True),
    ("norax-sleep.timer", True),
    ("norax-burnin.timer", True),
    ("norax-agent-os.service", False),
)
MAX_HEALTH_BYTES = 1024 * 1024
_SYSTEMD_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "Result",
    "NRestarts",
)
_SYSTEMD_REQUIRED_PROPERTIES = _SYSTEMD_PROPERTIES[:-1]


def systemd_status(unit: str, *, required: bool) -> dict[str, Any]:
    fields: dict[str, str | int] = {}
    query_error = ""
    query_ok = False
    command = [
        "systemctl",
        "--user",
        "show",
        unit,
        f"--property={','.join(_SYSTEMD_PROPERTIES)}",
    ]
    # A user-bus activation race can make one systemctl query fail even while
    # the service and its endpoint remain healthy. Retry once immediately;
    # persistent failures still fail closed and carry a bounded diagnostic.
    for _attempt in range(2):
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            fields = {}
            query_error = f"{type(exc).__name__}: {exc}"[:300]
            continue
        fields = {}
        for line in proc.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key] = int(value) if key == "NRestarts" and value.isdigit() else value
        # NRestarts exists for services but not all unit types (notably
        # timers), so it remains optional and defaults to zero below.
        query_ok = proc.returncode == 0 and all(
            key in fields for key in _SYSTEMD_REQUIRED_PROPERTIES
        )
        if query_ok:
            query_error = ""
            break
        detail = (proc.stderr or f"systemctl exit {proc.returncode}").strip()
        query_error = detail[:300]

    loaded = query_ok and fields.get("LoadState") == "loaded"
    enabled = fields.get("UnitFileState") in {"enabled", "enabled-runtime"}
    active = fields.get("ActiveState") == "active"
    expected_active = required or bool(enabled)
    ok = loaded and (active if expected_active else True)
    result = {
        "unit": unit,
        "required": required,
        "loaded": loaded,
        "enabled": enabled,
        "active": active,
        "substate": fields.get("SubState", "unknown"),
        "result": fields.get("Result", "unknown"),
        "restarts": fields.get("NRestarts", 0),
        "ok": ok,
    }
    if query_error:
        result["query_error"] = query_error
    return result


def json_health(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            raw = response.read(MAX_HEALTH_BYTES + 1)
            if len(raw) > MAX_HEALTH_BYTES:
                raise ValueError("health response exceeds 1 MiB")
            payload = json.loads(raw)
            return {
                "url": url,
                "http_status": response.status,
                "ok": response.status == 200
                and isinstance(payload, dict)
                and payload.get("ok") is True,
                "payload": payload if isinstance(payload, dict) else {},
            }
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
        return {
            "url": url,
            "http_status": 0,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "payload": {},
        }


def build_report(runtime_base: str, relay_base: str) -> dict[str, Any]:
    units = [systemd_status(name, required=required) for name, required in UNITS]
    runtime = json_health(runtime_base.rstrip("/") + "/readyz")
    relay = json_health(relay_base.rstrip("/") + "/health")
    required_units_ok = all(item["ok"] for item in units if item["required"])
    endpoints_ok = runtime["ok"] is True and relay["ok"] is True
    runtime_payload = runtime.pop("payload", {})
    relay_payload = relay.pop("payload", {})
    components = runtime_payload.get("components") if isinstance(runtime_payload, dict) else {}
    probe = components.get("completion_probe", {}) if isinstance(components, dict) else {}
    return {
        "schema": "norax.operational_health.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "ok": required_units_ok and endpoints_ok,
        "units": units,
        "endpoints": {"runtime": runtime, "remote_relay": relay},
        "serving_probe": {
            key: probe.get(key)
            for key in (
                "ok",
                "mode",
                "completion_verified",
                "model",
                "provider",
                "latency_ms",
                "age_seconds",
            )
            if isinstance(probe, dict) and key in probe
        },
        "remote_nodes": relay_payload.get("live_count")
        if isinstance(relay_payload, dict)
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-base", default="http://127.0.0.1:4101")
    parser.add_argument("--relay-base", default="http://127.0.0.1:8765")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = build_report(args.runtime_base, args.relay_base)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Norax operational health: {'PASS' if report['ok'] else 'FAIL'}")
        for unit in report["units"]:
            role = "required" if unit["required"] else "optional"
            state = f"{unit['substate']} / enabled={unit['enabled']}"
            print(f"  {'OK' if unit['ok'] else 'FAIL':4} {unit['unit']} ({role}): {state}")
        for name, endpoint in report["endpoints"].items():
            print(
                f"  {'OK' if endpoint['ok'] else 'FAIL':4} {name}: HTTP {endpoint['http_status']}"
            )
        if report["serving_probe"]:
            print(f"  serving probe: {json.dumps(report['serving_probe'], sort_keys=True)}")
        print(f"  remote nodes: {report['remote_nodes']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
