#!/usr/bin/env bash
# deploy-systemd-units.sh — idempotently install Norax systemd user units from ops/systemd/
#
# Usage:
#   ./ops/systemd/deploy-systemd-units.sh          # install + reload + enable
#   ./ops/systemd/deploy-systemd-units.sh --check   # diff only, no changes
#
# Exit codes:
#   0 — all units in sync (or successfully installed)
#   1 — drift detected (--check mode) or install failure

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/ops/systemd"
DST_DIR="$HOME/.config/systemd/user"

# Units managed by this script (service + timer files)
MANAGED_UNITS=(
  norax-ai.service
  norax-agent-os.service
  norax-sleep.service
  norax-sleep.timer
  norax-burnin.service
  norax-burnin.timer
  norax-remote-relay.service
)

# Drop-in directories managed by this script
MANAGED_DROPINS=(
)

# Services to enable
ENABLE_UNITS=(
  norax-ai.service
  norax-remote-relay.service
  norax-sleep.timer
  norax-burnin.timer
)

CHECK_ONLY=false
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=true

echo "Norax systemd unit deploy"
echo "  source:  $SRC_DIR"
echo "  target:  $DST_DIR"
echo "  mode:    $([[ $CHECK_ONLY == true ]] && echo 'check-only' || echo 'install')"
echo ""

drift_found=false

# Refuse to install any syntactically invalid repository unit, including
# optional templates not in the default managed set.
shopt -s nullglob
REPO_UNITS=("$SRC_DIR"/*.service "$SRC_DIR"/*.timer)
if ((${#REPO_UNITS[@]} == 0)); then
  echo "FAIL: no systemd units found in $SRC_DIR"
  exit 1
fi
if ! systemd-analyze --user verify "${REPO_UNITS[@]}"; then
  echo "FAIL: repository systemd unit validation failed"
  exit 1
fi
echo "  VALID: ${#REPO_UNITS[@]} repository unit(s)"

# --- Sync unit files ---
for unit in "${MANAGED_UNITS[@]}"; do
  src="$SRC_DIR/$unit"
  dst="$DST_DIR/$unit"

  if [[ ! -f "$src" ]]; then
    echo "  WARN: $unit missing from repo, skipping"
    continue
  fi

  if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
    drift_found=true
    if $CHECK_ONLY; then
      echo "  DRIFT: $unit"
      diff --unified=3 "$dst" "$src" 2>/dev/null || true
    else
      mkdir -p "$(dirname "$dst")"
      install -m 0644 "$src" "$dst"
      echo "  INSTALLED: $unit"
    fi
  else
    echo "  OK: $unit"
  fi
done

# --- Sync drop-in files ---
for dropin in "${MANAGED_DROPINS[@]}"; do
  src="$SRC_DIR/$dropin"
  dst="$DST_DIR/$dropin"

  if [[ ! -f "$src" ]]; then
    echo "  WARN: drop-in $dropin missing from repo, skipping"
    continue
  fi

  if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
    drift_found=true
    if $CHECK_ONLY; then
      echo "  DRIFT: $dropin"
      diff --unified=3 "$dst" "$src" 2>/dev/null || true
    else
      mkdir -p "$(dirname "$dst")"
      cp "$src" "$dst"
      echo "  INSTALLED: $dropin"
    fi
  else
    echo "  OK: $dropin"
  fi
done

if $CHECK_ONLY; then
  if $drift_found; then
    echo ""
    echo "FAIL: drift detected. Run without --check to sync."
    exit 1
  else
    echo ""
    echo "PASS: all units in sync."
    exit 0
  fi
fi

# --- Reload systemd ---
echo ""
echo "Reloading systemd daemon..."
systemctl --user daemon-reload

# --- Enable services ---
echo ""
for unit in "${ENABLE_UNITS[@]}"; do
  if systemctl --user is-enabled "$unit" >/dev/null 2>&1; then
    echo "  ENABLED: $unit (already)"
  else
    systemctl --user enable "$unit"
    echo "  ENABLED: $unit"
  fi
done

# --- Verify ---
echo ""
echo "Verification:"
for unit in "${MANAGED_UNITS[@]}"; do
  src="$SRC_DIR/$unit"
  dst="$DST_DIR/$unit"
  if cmp -s "$src" "$dst"; then
    echo "  MATCH: $unit"
  else
    echo "  MISMATCH: $unit"
  fi
done

echo ""
for unit in "${ENABLE_UNITS[@]}"; do
  # `is-enabled`/`is-active` intentionally use non-zero statuses for valid
  # states such as disabled or inactive. Capture those states without letting
  # `set -e` abort this reporting-only section.
  state=$(systemctl --user is-enabled "$unit" 2>&1 || true)
  active=$(systemctl --user is-active "$unit" 2>&1 || true)
  echo "  $unit: enabled=$state active=$active"
done

echo ""
echo "Done. Norax systemd units deployed from repository."
