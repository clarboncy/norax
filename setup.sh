#!/usr/bin/env bash
# Norax AI — interactive first-run setup.
#
# Walks you through: owner identity, chat token, Ollama check, optional API
# providers, optional Discord, and starts the runtime + Agent OS dashboard so
# you can chat with Norax locally in your browser before any external channel
# is wired.
#
# Usage:  bash setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── helpers ──────────────────────────────────────────────────────────────
BOLD="\033[1m"; DIM="\033[2m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; RESET="\033[0m"

info()  { printf "${CYAN}▸${RESET} %s\n" "$*"; }
ok()    { printf "${GREEN}✓${RESET} %s\n" "$*"; }
warn()  { printf "${YELLOW}⚠${RESET} %s\n" "$*"; }
err()   { printf "${RED}✗${RESET} %s\n" "$*" >&2; }
header(){ printf "\n${BOLD}%s${RESET}\n" "$*"; }
prompt(){ printf "${BOLD}%s${RESET}" "$*"; }

ask() {
  # ask "prompt" "default"  → prints user input or default to stdout
  # Prompt text goes to stderr so it doesn't pollute the captured value.
  local question="$1" default="${2:-}" reply=""
  prompt "$question" >&2
  [ -n "$default" ] && printf " ${DIM}[%s]${RESET}" "$default" >&2
  printf ": " >&2
  read -r reply
  echo "${reply:-$default}"
}

ask_yesno() {
  # ask_yesno "prompt" "default(y/n)" → returns 0/1
  local question="$1" default="${2:-n}" reply=""
  prompt "$question" >&2
  printf " ${DIM}[%s]${RESET}: " "$default" >&2
  read -r reply
  reply="${reply:-$default}"
  case "$reply" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

# ── preflight ────────────────────────────────────────────────────────────
header "Norax AI — First-Run Setup"

if [ ! -f "pyproject.toml" ]; then
  err "Run this script from the Norax repository root."
  exit 1
fi

# Check Python
if ! command -v python3 &>/dev/null; then
  err "Python 3 is required (3.12+). Install it first."
  exit 1
fi

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
info "Python: $PY_VER"

# Check uv
if ! command -v uv &>/dev/null; then
  warn "uv is not installed — installing now..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv available"

# ── venv ─────────────────────────────────────────────────────────────────
header "1/7  Virtual environment"

if [ ! -d ".venv" ]; then
  info "Creating virtual environment..."
  uv venv </dev/null
fi

info "Syncing dependencies..."
uv sync --all-extras </dev/null
ok "Dependencies synced"

# ── .env ─────────────────────────────────────────────────────────────────
header "2/7  Environment configuration"

if [ ! -f ".env" ]; then
  cp .env.example .env
  chmod 600 .env
  ok "Created .env from template"
else
  ok ".env already exists — will update missing values"
fi

# Owner identity
echo ""
info "Owner identity — this is how Norax recognizes you."
OWNER_LABEL="$(ask "  Your display name" "$(whoami)")"
OWNER_ID="$(ask "  Your owner ID (any unique string)" "$(whoami)")"
OWNER_TZ="$(ask "  Your timezone" "$(timedatectl show -p Timezone --value 2>/dev/null || echo UTC)")"

# Chat token — auto-generate if empty
echo ""
info "Chat token — authenticates the local web dashboard."
EXISTING_TOKEN="$(grep -E '^NORAX_AGENT_OS_CHAT_TOKEN=' .env | cut -d= -f2- || true)"
if [ -n "$EXISTING_TOKEN" ]; then
  ok "Existing chat token found — keeping it"
  CHAT_TOKEN="$EXISTING_TOKEN"
else
  CHAT_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  ok "Auto-generated secure chat token"
fi

# Write to .env (update or append each variable)
update_env() {
  local key="$1" val="$2"
  local line tmp replaced="false"
  if [[ ! "$key" =~ ^[A-Z][A-Z0-9_]*$ ]]; then
    err "Refusing invalid environment key"
    return 1
  fi
  if [[ "$val" == *$'\n'* || "$val" == *$'\r'* ]]; then
    err "Environment value for $key must fit on one line"
    return 1
  fi

  tmp="$(mktemp "${ROOT}/.env.tmp.XXXXXX")"
  chmod 600 "$tmp"
  while IFS= read -r line || [ -n "$line" ]; do
    if [[ "$line" == "$key="* ]]; then
      if [ "$replaced" = "false" ]; then
        printf '%s=%s\n' "$key" "$val"
        replaced="true"
      fi
    else
      printf '%s\n' "$line"
    fi
  done < .env > "$tmp"
  if [ "$replaced" = "false" ]; then
    printf '%s=%s\n' "$key" "$val" >> "$tmp"
  fi
  mv "$tmp" .env
}

env_value() {
  local key="$1" fallback="$2"
  python3 - "$key" "$fallback" <<'PY'
import os
import sys
from pathlib import Path

key, fallback = sys.argv[1:]
if key in os.environ:
    print(os.environ[key])
    raise SystemExit
for raw_line in Path(".env").read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    candidate, _, value = line.partition("=")
    if candidate.strip() == key:
        print(value.strip().strip('"').strip("'"))
        raise SystemExit
print(fallback)
PY
}

update_env "NORAX_OWNER_LABEL" "$OWNER_LABEL"
update_env "NORAX_OWNER_ID" "$OWNER_ID"
update_env "NORAX_OWNER_TIMEZONE" "$OWNER_TZ"
update_env "NORAX_AGENT_OS_CHAT_TOKEN" "$CHAT_TOKEN"
update_env "NORAX_RUNTIME_CHAT_TOKEN" "$CHAT_TOKEN"

# Lock directory — use a per-instance lock so multiple Norax installs
# (e.g. live runtime + test clone) don't conflict.
LOCK_INSTANCE="$(printf '%s\n%s' "$ROOT" "$OWNER_ID" | python3 -c \
  'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:16])')"
STATE_HOME="${XDG_STATE_HOME:-${HOME}/.local/state}"
if [[ "$STATE_HOME" != /* ]]; then
  err "XDG_STATE_HOME must be an absolute path"
  exit 1
fi
LOCK_DIR="${STATE_HOME}/norax/locks/${LOCK_INSTANCE}"
mkdir -p -m 700 "$LOCK_DIR"
chmod 700 "$LOCK_DIR"
update_env "NORAX_LOCK_DIR" "$LOCK_DIR"

ok ".env configured"
chmod 600 .env
ok ".env permissions set to 600"

# ── Ollama ───────────────────────────────────────────────────────────────
header "3/7  Model provider — Ollama (local)"

OLLAMA_OK=false
if curl -sf --max-time 3 http://127.0.0.1:11434/api/tags &>/dev/null; then
  ok "Ollama is running at http://127.0.0.1:11434"
  OLLAMA_OK=true
  echo ""
  info "Available models:"
  # Fetch to a temp file so python3 doesn't consume piped stdin
  OLLAMA_TAGS_TMP="$(mktemp)"
  curl -sf http://127.0.0.1:11434/api/tags -o "$OLLAMA_TAGS_TMP" 2>/dev/null
  python3 -c "
import sys, json
try:
    with open('$OLLAMA_TAGS_TMP') as f:
        data = json.load(f)
    for m in data.get('models', []):
        print(f'  • {m[\"name\"]}')
except Exception:
    print('  (could not parse)')
" </dev/null 2>/dev/null || echo "  (could not list)"
  rm -f "$OLLAMA_TAGS_TMP"
else
  warn "Ollama is not running at http://127.0.0.1:11434"
  echo ""
  echo "  Norax works with any OpenAI-compatible endpoint, but Ollama"
  echo "  is the easiest local option. Install it with:"
  echo ""
  echo "    curl -fsSL https://ollama.com/install.sh | sh"
  echo "    ollama pull llama3.2:3b"
  echo ""
  if ask_yesno "  Continue without Ollama? (you can add providers later)" "y"; then
    ok "Continuing — you can add providers via the dashboard or API"
  else
    err "Install Ollama and re-run this script."
    exit 1
  fi
fi

# Default model
if $OLLAMA_OK; then
  DEFAULT_MODEL="$(ask "  Default model" "llama3.2:3b")"
  update_env "NORAX_DEFAULT_MODEL" "$DEFAULT_MODEL"
  update_env "NORAX_DEFAULT_PROVIDER" "ollama"
fi

# ── Optional API providers ───────────────────────────────────────────────
header "4/7  Optional API providers"

echo "You can add any OpenAI-compatible API (OpenRouter, OpenAI, Together, etc.)"
echo "now or later via the dashboard Settings tab or the /api/providers endpoint."
echo ""

PROVIDER_PENDING=false
PROV_NAME=""
PROV_URL=""
PROV_KEY=""
PROV_KIND=""
if ask_yesno "  Add an API provider now?" "n"; then
  PROV_NAME="$(ask "    Provider name (e.g. openrouter, openai)" "openrouter")"
  PROV_URL="$(ask "    Base URL (e.g. https://openrouter.ai/api/v1)" "https://openrouter.ai/api/v1")"
  prompt "    API key: "
  read -rs PROV_KEY
  printf '\n'
  PROV_KIND="$(ask "    Provider kind (openai, ollama, openrouter)" "openai")"
  PROVIDER_PENDING=true
  ok "Provider will be added after this runtime passes its health check"
fi

# ── Discord (optional) ───────────────────────────────────────────────────
header "5/7  Discord (optional)"

echo "Discord is disabled by default. You can enable it now or later."
echo "The local web dashboard works without Discord."
echo ""

if ask_yesno "  Configure Discord now?" "n"; then
  DISCORD_TOKEN="$(ask "    Bot token" "")"
  if [ -n "$DISCORD_TOKEN" ]; then
    update_env "NORAX_DISCORD_TOKEN" "$DISCORD_TOKEN"
    update_env "NORAX_DISCORD_ENABLED" "true"
    ok "Discord configured — will connect on startup"
  else
    warn "No token provided — Discord stays disabled"
  fi
else
  ok "Discord skipped — use the local dashboard to chat with Norax"
fi

# ── Port conflict detection ──────────────────────────────────────────────
header "6/7  Port check"

# Read the same explicit values the launched processes will receive.
RUNTIME_PORT="$(env_value NORAX_HTTP_PORT 4101)"
DASHBOARD_PORT="$(env_value NORAX_OS_PORT 8822)"

validate_port() {
  local name="$1" port="$2"
  if [[ ! "$port" =~ ^[0-9]+$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
    err "$name must be an integer between 1 and 65535 (got: $port)"
    return 1
  fi
}

validate_port "NORAX_HTTP_PORT" "$RUNTIME_PORT"
validate_port "NORAX_OS_PORT" "$DASHBOARD_PORT"

check_port() {
  local port="$1"
  if python3 - "$port" <<'PY' 2>/dev/null
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
  then
    return 0  # available
  else
    return 1  # in use
  fi
}

if ! check_port "$RUNTIME_PORT"; then
  err "Configured runtime port $RUNTIME_PORT is already in use"
  err "Stop the existing listener or set NORAX_HTTP_PORT explicitly, then rerun setup."
  exit 1
else
  ok "Runtime port $RUNTIME_PORT is available"
fi

if ! check_port "$DASHBOARD_PORT"; then
  err "Configured dashboard port $DASHBOARD_PORT is already in use"
  err "Stop the existing listener or set NORAX_OS_PORT explicitly, then rerun setup."
  exit 1
else
  ok "Dashboard port $DASHBOARD_PORT is available"
fi

# ── Start ────────────────────────────────────────────────────────────────
header "7/7  Start Norax"

echo ""
echo "Configuration summary:"
echo "  Owner:       $OWNER_LABEL ($OWNER_ID)"
echo "  Timezone:    $OWNER_TZ"
echo "  Chat token:  ${CHAT_TOKEN:0:12}... (full token in .env)"
echo "  Runtime:     http://127.0.0.1:$RUNTIME_PORT"
echo "  Dashboard:   http://127.0.0.1:$DASHBOARD_PORT"
echo "  Discord:     $(grep '^NORAX_DISCORD_ENABLED=' .env | cut -d= -f2-)"
echo ""

if ask_yesno "  Start Norax now?" "y"; then
  info "Starting runtime (port $RUNTIME_PORT)..."
  uv run --env-file .env python -m norax &
  RUNTIME_PID=$!

  # Wait for runtime to be ready
  info "Waiting for runtime to be ready..."
  RUNTIME_READY=false
  for i in $(seq 1 30); do
    if curl -sf --max-time 2 "http://127.0.0.1:$RUNTIME_PORT/healthz" &>/dev/null; then
      ok "Runtime is ready"
      RUNTIME_READY=true
      break
    fi
    if ! kill -0 "$RUNTIME_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [ "$RUNTIME_READY" != "true" ]; then
    err "Runtime did not become healthy on the configured endpoint"
    kill "$RUNTIME_PID" 2>/dev/null || true
    wait "$RUNTIME_PID" 2>/dev/null || true
    exit 1
  fi

  if $PROVIDER_PENDING; then
    info "Adding provider '$PROV_NAME' to the verified runtime..."
    if printf '%s\0%s\0%s\0%s\0%s' \
      "$CHAT_TOKEN" "$PROV_NAME" "$PROV_URL" "$PROV_KEY" "$PROV_KIND" | \
      .venv/bin/python -c '
import json
import sys
import urllib.request
from urllib.parse import quote

parts = sys.stdin.buffer.read().split(b"\0")
if len(parts) != 5:
    raise SystemExit(2)
token, name, base_url, api_key, kind = (part.decode("utf-8") for part in parts)
payload = json.dumps(
    {"base_url": base_url, "api_key": api_key, "provider_kind": kind}
).encode("utf-8")
encoded_name = quote(name, safe="")
request = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/api/providers/{encoded_name}",
    data=payload,
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    method="PUT",
)
with urllib.request.urlopen(request, timeout=10) as response:
    if not 200 <= response.status < 300:
        raise SystemExit(1)
' "$RUNTIME_PORT" 2>/dev/null
    then
      ok "Provider added"
    else
      warn "Provider was not accepted; add it later through authenticated settings."
    fi
    PROV_KEY=""
  fi

  info "Starting Agent OS dashboard (port $DASHBOARD_PORT)..."
  uv run --env-file .env python agent_os/server.py &
  DASHBOARD_PID=$!

  # The setup succeeds only if the dashboard is actually reachable.
  DASHBOARD_READY=false
  for _ in $(seq 1 10); do
    if curl -sf --max-time 2 "http://127.0.0.1:$DASHBOARD_PORT/" &>/dev/null; then
      DASHBOARD_READY=true
      break
    fi
    if ! kill -0 "$DASHBOARD_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [ "$DASHBOARD_READY" != "true" ]; then
    err "Dashboard did not become healthy on the configured endpoint"
    kill "$DASHBOARD_PID" "$RUNTIME_PID" 2>/dev/null || true
    wait "$DASHBOARD_PID" 2>/dev/null || true
    wait "$RUNTIME_PID" 2>/dev/null || true
    exit 1
  fi
  ok "Dashboard is ready"

  echo ""
  header "Norax is live!"
  echo ""
  echo "  ${BOLD}Open your browser:${RESET} http://127.0.0.1:$DASHBOARD_PORT"
  echo ""
  echo "  The dashboard will ask for your chat token. It's in .env:"
  echo "    grep NORAX_AGENT_OS_CHAT_TOKEN .env"
  echo ""
  echo "  To stop Norax:"
  echo "    kill $RUNTIME_PID $DASHBOARD_PID"
  echo ""
  echo "  To start again later:"
  echo "    uv run --env-file .env python -m norax &"
  echo "    uv run --env-file .env python agent_os/server.py &"
  echo ""
  echo "  Ports are saved in .env (NORAX_HTTP_PORT / NORAX_OS_PORT)."
  echo ""

  # Tail runtime logs so the user sees what's happening
  info "Tailing runtime output (Ctrl+C to stop tailing — Norax keeps running)..."
  wait $RUNTIME_PID
else
  echo ""
  ok "Setup complete. Start Norax with:"
  echo ""
  echo "  ${BOLD}uv run --env-file .env python -m norax${RESET}        # runtime on :$RUNTIME_PORT"
  echo "  ${BOLD}uv run --env-file .env python agent_os/server.py${RESET}  # dashboard on :$DASHBOARD_PORT"
  echo ""
  echo "  Then open: http://127.0.0.1:$DASHBOARD_PORT"
  echo "  Chat token: grep NORAX_AGENT_OS_CHAT_TOKEN .env"
  echo ""
  echo "  Ports are saved in .env (NORAX_HTTP_PORT / NORAX_OS_PORT)."
  echo ""
fi
