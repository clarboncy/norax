#!/usr/bin/env bash
set -u

# Edge-triggered network recovery for long-running user services.
# A failed Discord gateway/API probe records an offline edge. The first
# successful probe after that edge restarts the agent, avoiding restart loops.
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/norax"
STATE_FILE="${CONNECTIVITY_STATE_FILE:-$STATE_DIR/connectivity.state}"
SERVICE="${CONNECTIVITY_TARGET_SERVICE:-norax-ai.service}"
PROBE_URL="${CONNECTIVITY_PROBE_URL:-https://discord.com/api/v10/gateway}"
mkdir -p "$STATE_DIR"

log() { printf 'connectivity_recovery %s\n' "$*"; }

online() {
  getent ahosts gateway.discord.gg >/dev/null 2>&1 || return 1
  curl --silent --show-error --fail --location \
    --connect-timeout 5 --max-time 10 \
    --output /dev/null "$PROBE_URL" >/dev/null 2>&1
}

if ! online; then
  printf 'offline %(%s)T\n' -1 >"$STATE_FILE.tmp"
  mv -f "$STATE_FILE.tmp" "$STATE_FILE"
  log "OFFLINE gateway probe failed; recovery armed"
  exit 0
fi

previous="unknown"
[[ -r "$STATE_FILE" ]] && read -r previous _ <"$STATE_FILE" || true
printf 'online %(%s)T\n' -1 >"$STATE_FILE.tmp"
mv -f "$STATE_FILE.tmp" "$STATE_FILE"

if [[ "${1:-}" == "--probe-only" ]]; then
  log "OK gateway reachable previous=$previous"
  exit 0
fi

if ! systemctl --user is-active --quiet "$SERVICE"; then
  log "RECOVER service=$SERVICE inactive; starting"
  systemctl --user reset-failed "$SERVICE" || true
  systemctl --user start "$SERVICE"
elif [[ "$previous" == "offline" ]]; then
  log "RECOVER service=$SERVICE connectivity restored; restarting"
  systemctl --user restart "$SERVICE"
else
  log "OK gateway reachable service=$SERVICE active"
fi
