"""Opt-in live owner-chat and Discord delivery acceptance using deployed settings.

Sends one labeled private Discord receipt only when --discord-receipt is set.
Never starts, stops, or reconfigures the selected service. Credentials and
conversation history are not included in the result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True, choices=["norax-ai.service"])
    parser.add_argument("--discord-receipt", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    pid = subprocess.check_output(
        ["systemctl", "--user", "show", args.service, "-p", "MainPID", "--value"], text=True
    ).strip()
    if not pid.isdigit() or int(pid) < 1:
        raise RuntimeError("selected service is not running")
    service_env = dict(
        item.split("=", 1)
        for item in Path(f"/proc/{pid}/environ").read_text().split("\0")
        if "=" in item
    )
    # Resolve precisely the selected deployment; do not load a shell's .env.
    for key in list(os.environ):
        if key.startswith("NORAX_"):
            os.environ.pop(key)
    os.environ.update(service_env)
    from norax.__main__ import _load_dotenv
    from norax.atomic import atomic_write_text
    from norax.config.loader import load_config

    # /proc exposes the initial process environment, not Python's later dotenv
    # additions. Reapply the runtime's isolated dotenv selection as startup does.
    _load_dotenv()
    cfg = load_config()
    run_task_once = importlib.import_module("benchmarks.perf_bench").run_task_once
    host, port = cfg.http_bind
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("live acceptance requires an explicitly loopback HTTP listener")
    base = f"http://127.0.0.1:{port}"
    chat_token = cfg.agent_os_chat_token
    if not chat_token:
        raise RuntimeError("owner-chat bridge is not configured")
    nonce = secrets.token_hex(10)
    marker = f"CONNECTOR_OK_{nonce}"
    thread = f"connector-acceptance-{nonce}"
    receipt = f"Norax connector validation receipt: {nonce}"
    started = time.monotonic()
    target = None
    observed_discord = False
    with httpx.Client(timeout=15) as client:
        discord_headers = {"Authorization": f"Bot {cfg.discord.token}"}
        if args.discord_receipt:
            if not cfg.discord.enabled or not cfg.discord.token or not cfg.owner_id:
                raise RuntimeError("Discord owner delivery is not configured")
            dm = client.post(
                "https://discord.com/api/v10/users/@me/channels",
                headers=discord_headers,
                json={"recipient_id": str(cfg.owner_id)},
            )
            if dm.status_code != 200:
                raise RuntimeError(f"Discord DM resolution failed: HTTP {dm.status_code}")
            target = dm.json()["id"]
        prompt = (
            "This is a one-turn connector acceptance test, not a preference or memory update. "
            "Do not change files, settings, services, or memories. "
        )
        if target:
            prompt += (
                "Use message_send exactly once with channel=discord, "
                f"target={target}, text={receipt!r}. Verify the tool result. "
            )
        prompt += f"Then reply with exactly {marker} and nothing else."
        # A live websocket must be attached before ingress. A history query
        # alone is not an outbound recipient and cannot prove bridge delivery.
        turn = run_task_once(
            base,
            chat_token,
            {"name": "connector_receipt", "body": prompt, "expect_exact": marker, "timeout_s": 180},
            thread,
        )
        reply_seen = turn["ok"] is True
        if target:
            messages = client.get(
                f"https://discord.com/api/v10/channels/{target}/messages",
                headers=discord_headers,
                params={"limit": 25},
            )
            if messages.status_code == 200:
                observed_discord = any(row.get("content") == receipt for row in messages.json())
        ready = client.get(base + "/readyz")
        health_ok = ready.status_code == 200 and ready.json().get("ok") is True
    current_pid = subprocess.check_output(
        ["systemctl", "--user", "show", args.service, "-p", "MainPID", "--value"], text=True
    ).strip()
    result = {
        "scope": "live_owner_bridge_and_optional_discord_delivery",
        "owner_bridge_reply_verified": reply_seen,
        "discord_receipt_selected": args.discord_receipt,
        "discord_receipt_verified": observed_discord if target else None,
        "ready": health_ok,
        "service_identity_unchanged": current_pid == pid,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "ok": reply_seen and health_ok and current_pid == pid and (not target or observed_discord),
    }
    if args.json_out:
        atomic_write_text(args.json_out, json.dumps(result, indent=2) + "\n", mode=0o600)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
