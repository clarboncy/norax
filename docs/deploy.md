# Deploy

Single-host systemd-user deployment. No root service is required. Managed unit groups:

| Unit | Type | What |
|---|---|---|
| `norax-ai.service` | long-running | main runtime (Discord + HTTP + cron sensory) |
| `norax-remote-relay.service` | long-running | authenticated remote-node relay |
| `norax-agent-os.service` | optional long-running | local web dashboard/control plane |
| `norax-sleep.timer` + `norax-sleep.service` | oneshot on timer | offline sleep-flush |
| `norax-burnin.timer` + `norax-burnin.service` | oneshot on timer | 72-hour release evidence |

## Prereqs

- Ubuntu 24.04 / Pop!_OS 22.04 or later
- Python 3.12 (via `uv` recommended)
- `~/norax/` checkout with `.venv/` synced (`uv sync`)
- `~/norax/.env` with `DISCORD_TOKEN` (chmod 600)
- `systemctl --user` available (login lingering enabled if you want
  auto-start on boot without an active session:
  `sudo loginctl enable-linger $USER`)

## Install unit files

Canonical copies live in [`ops/systemd/`](../ops/systemd/). Use the deployment
helper so every repository unit is syntax-checked, managed files are copied with
known permissions, systemd is reloaded, and required units are enabled:

```bash
./ops/systemd/deploy-systemd-units.sh
./ops/systemd/deploy-systemd-units.sh --check
```

The helper installs but does not restart running services or start newly enabled
timers. `norax-agent-os.service` is installed as an optional unit and is not
enabled automatically.

## Start

```bash
systemctl --user enable --now norax-ai.service
systemctl --user enable --now norax-remote-relay.service
systemctl --user enable --now norax-sleep.timer
systemctl --user enable --now norax-burnin.timer
# Optional dashboard:
systemctl --user enable --now norax-agent-os.service
```

The included units expect the checkout at `~/norax` and use systemd's `%h`
home-directory specifier. For another checkout, use a drop-in and override
`WorkingDirectory`, `ExecStart`, and any `EnvironmentFile` path together. A
shell-style `${NORAX_HOME}` placeholder is not expanded in these directives.

Start a deliberately new burn-in campaign only after the audited runtime is
ready. This preserves the prior campaign under a timestamped private archive:

```bash
systemctl --user stop norax-burnin.timer
.venv/bin/python scripts/burnin_monitor.py --start-new-campaign post-deploy
systemctl --user start norax-burnin.timer
```

## Verify

```bash
systemctl --user status norax-ai.service
systemctl --user list-timers norax-sleep.timer norax-burnin.timer --no-pager
.venv/bin/python scripts/monitor.py --json
curl -s http://127.0.0.1:4101/healthz
curl -s http://127.0.0.1:4101/readyz
curl -s http://127.0.0.1:4101/metrics | head -20
tail -f ~/.local/state/norax/logs/norax-ai.log
```

Expected:
- `norax-ai.service` active (running), Discord gateway connected
- `norax-sleep.timer` NEXT within the hour, `:17` each hour
- `norax-burnin.timer` active (waiting), with a NEXT time near one minute
- `/readyz` reports an unexpired completion-verified model probe
- `/healthz` returns `{"ok":true,"uptime_seconds":...}`
- `/metrics` returns Prometheus text exposition with `norax_*` series

## Ports

| Port | Bound | What |
|---|---|---|
| 4101 | 127.0.0.1 | HTTP ingress + `/healthz` + `/metrics` |
| 8822 | 127.0.0.1 | Agent OS web dashboard (chat with Norax in your browser) |

### Port configuration

Norax fails startup clearly if its configured port is already occupied. Silent
port fallback is intentionally avoided because the dashboard and health checks
must agree on one endpoint. To select a different port, set the same explicit
port for the runtime and dashboard in `.env`:

```
NORAX_HTTP_PORT=4201       # runtime HTTP server
NORAX_OS_PORT=8922         # Agent OS dashboard
```

Only localhost by default. To scrape metrics from another host on a
Tailscale mesh, change `http.bind` in `config/runtime.jsonc` from
`"127.0.0.1:4101"` to `"0.0.0.0:4101"` (or your tailnet IP).
`/metrics` exposes **operational telemetry only** — no secrets, no
message bodies. Still, keep the port off the public internet.

## Environment

`.env` (chmod 600, in the checkout root):

```
NORAX_OWNER_ID=your-unique-id
NORAX_OWNER_LABEL=Your Name
NORAX_AGENT_OS_CHAT_TOKEN=<random string from: python3 -c "import secrets; print(secrets.token_urlsafe(32))">
# Optional:
# NORAX_DISCORD_TOKEN=...
# NORAX_DISCORD_ENABLED=true
# NORAX_GATEWAY_TOKEN=...
# NORAX_MEMORY_ROOT=~/.local/share/norax/memory
# NORAX_STATE_DIR=~/.local/state/norax
# NORAX_LOG_DIR=~/.local/state/norax/logs
# NORAX_AGENT_OS_STATE_DIR=~/.local/state/norax/agent-os
```

`norax/__main__.py` loads this file without overriding variables
already set in the process environment.

## Logs

- `~/.local/state/norax/logs/norax-ai.log` — main runtime, structured log lines
- `~/.local/state/norax/logs/norax-sleep.log` — one JSON summary per flush
- `~/.local/state/norax/events.jsonl` — hash-chained event log (canonical
  trace sink; see [`metrics.md`](metrics.md) § spans)
- `~/.local/state/norax/burnin/status.json` — current burn-in assessment
- `~/.local/state/norax/burnin/history.json` — bounded current-campaign samples
- `~/.local/state/norax/burnin/evidence.jsonl` — append-only burn-in samples
- `~/.local/state/norax/burnin/archive/` — preserved prior campaigns
- `~/.local/state/norax/agent-os/agent_os.db` — private dashboard transcript
  database (directory mode 0700, database mode 0600)

Rotation: not configured. For a long-running install add a
`~/.config/logrotate.d/norax` file or rely on systemd journal
(remove the `StandardOutput=append:` line from the unit file to fall
back to the journal).

## Upgrades

```bash
cd ~/norax
git pull
uv sync
./ops/systemd/deploy-systemd-units.sh
systemctl --user restart norax-ai.service
# The timer picks up new code on next firing automatically.
```

Zero-downtime upgrades are out of scope (single-process architecture).
Restarts are sub-second and Discord reconnects automatically.

## Uninstall

```bash
systemctl --user disable --now norax-ai.service
systemctl --user disable --now norax-remote-relay.service
systemctl --user disable --now norax-agent-os.service
systemctl --user disable --now norax-sleep.timer
systemctl --user disable --now norax-burnin.timer
rm ~/.config/systemd/user/norax-ai.service
rm ~/.config/systemd/user/norax-remote-relay.service
rm ~/.config/systemd/user/norax-agent-os.service
rm ~/.config/systemd/user/norax-sleep.service ~/.config/systemd/user/norax-sleep.timer
rm ~/.config/systemd/user/norax-burnin.service ~/.config/systemd/user/norax-burnin.timer
systemctl --user daemon-reload
```

The checkout remains in `~/norax/`; mutable runtime data remains under
`~/.local/share/norax/` and `~/.local/state/norax/`. Remove those paths
manually only after taking any required backup.
