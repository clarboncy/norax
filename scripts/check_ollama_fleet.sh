#!/usr/bin/env bash
# Fleet Ollama health — local + remote node registry.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log() { printf '[fleet] %s\n' "$*"; }

check() {
  local name="$1" url="$2"
  if curl -sS -m 6 "$url" >/dev/null 2>&1; then
    log "OK  ${name} ${url}"
  else
    log "FAIL ${name} ${url}"
    return 1
  fi
}

FAIL=0
log "=== Ollama entry points (local) ==="
check "ollama :11434" "http://127.0.0.1:11434/api/tags" || FAIL=1
check "norax runtime :4101" "http://127.0.0.1:4101/healthz" || FAIL=1

log "=== Remote relay nodes ==="
.venv/bin/python3 - <<'PY' || FAIL=1
import json
from pathlib import Path
from norax.remote.registry import RemoteRegistry

reg = RemoteRegistry(Path("memory/remote"))
nodes = reg.list_nodes()
live = [n for n in nodes if n.last_seen_ms]
stale = [n for n in nodes if not n.last_seen_ms]
print(f"enrolled={len(nodes)} live={len(live)} stale={len(stale)}")
for n in sorted(live, key=lambda x: x.last_seen_ms or 0, reverse=True)[:5]:
    print(f"  LIVE {n.node_id} last_seen_ms={n.last_seen_ms} meta={n.meta}")
if not live:
    print("  WARN no live remote nodes — check norax-node on enrolled hosts")
PY

log "=== Ollama context ==="
systemctl show ollama.service -p Environment 2>/dev/null | grep -o 'OLLAMA_CONTEXT_LENGTH=[^ ]*' || true

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
log "Fleet check complete"
