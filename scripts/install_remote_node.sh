#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  install_remote_node.sh --node-id ID --token TOKEN [--relay ws://HOST:8765/remote] [--root PATH] [--allow-exec]

Installs a persistent user-level Norax remote node service on this computer.
Run on the target machine as the target user. No sudo required if systemd --user is available.
Shell execution is disabled unless --allow-exec is explicitly supplied.
USAGE
}

NODE_ID="${NORAX_NODE_ID:-}"
TOKEN="${NORAX_NODE_TOKEN:-}"
RELAY="${NORAX_RELAY:-ws://127.0.0.1:8765/remote}"
ROOT="${HOME}"
ALLOW_EXEC=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-id) NODE_ID="${2:-}"; shift 2 ;;
    --token) TOKEN="${2:-}"; shift 2 ;;
    --relay) RELAY="${2:-}"; shift 2 ;;
    --root) ROOT="${2:-}"; shift 2 ;;
    --allow-exec) ALLOW_EXEC=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$NODE_ID" || -z "$TOKEN" ]]; then
  echo "--node-id and --token are required" >&2
  usage
  exit 2
fi
if [[ ! "$NODE_ID" =~ ^[A-Za-z0-9_-]{1,120}$ ]]; then
  echo "--node-id must contain only letters, numbers, underscores, or hyphens" >&2
  exit 2
fi
if [[ "$RELAY" == *$'\n'* || "$ROOT" == *$'\n'* ]]; then
  echo "--relay and --root must not contain newlines" >&2
  exit 2
fi

PYTHON_BIN="$(command -v python3)"
NODE_DIR="$HOME/norax-node"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/norax-remote-node.service"
TOKEN_FILE="$NODE_DIR/node.token"
mkdir -p "$NODE_DIR" "$UNIT_DIR"
umask 077
printf '%s\n' "$TOKEN" > "$TOKEN_FILE"

if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import websockets
PY
then
  :
else
  "$PYTHON_BIN" -m pip install --user websockets
fi

if [[ ! -f "$NODE_DIR/node.py" ]]; then
  if command -v norax-node >/dev/null 2>&1; then
    cat > "$NODE_DIR/node.py" <<'PY'
from norax.remote.node import main
main()
PY
  else
    echo "No $NODE_DIR/node.py and norax-node is not installed. Copy norax/remote/node.py to $NODE_DIR/node.py first." >&2
    exit 3
  fi
fi

EXEC_FLAG=""
if [[ "$ALLOW_EXEC" == true ]]; then
  EXEC_FLAG=" --allow-exec"
fi

cat > "$UNIT" <<EOF
[Unit]
Description=Norax Remote Node Relay Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/norax-node
ExecStart="$PYTHON_BIN" %h/norax-node/node.py --relay "$RELAY" --node-id "$NODE_ID" --token-file %h/norax-node/node.token --root "$ROOT"$EXEC_FLAG
Restart=always
RestartSec=5
StandardOutput=append:%h/norax-node/node.log
StandardError=append:%h/norax-node/node.log

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now norax-remote-node.service
systemctl --user restart norax-remote-node.service
sleep 1
systemctl --user --no-pager --lines=20 status norax-remote-node.service
