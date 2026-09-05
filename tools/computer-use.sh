#!/usr/bin/env bash
# Compatibility wrapper for the canonical Python computer-use backend.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${NORAX_COMPUTER_USE_PYTHON:-python3}"
ACTION="${1:-}"

if [[ -z "$ACTION" ]]; then
    echo '{"ok":false,"error":"usage: computer-use.sh <action> [args...]"}'
    exit 2
fi
shift

case "$ACTION" in
    screenshot)
        output="${1:-/tmp/norax-screen.png}"
        mode="${2:-auto}"
        exec "$PYTHON_BIN" "$SCRIPT_DIR/computer_use.py" screenshot --output "$output" --mode "$mode"
        ;;
    click)
        if (( $# < 2 )); then
            echo '{"ok":false,"error":"click requires x and y"}'
            exit 2
        fi
        exec "$PYTHON_BIN" "$SCRIPT_DIR/computer_use.py" click "$1" "$2" --button "${3:-1}"
        ;;
    doubleclick|move|drag|scroll|key)
        exec "$PYTHON_BIN" "$SCRIPT_DIR/computer_use.py" "$ACTION" "$@"
        ;;
    type)
        exec "$PYTHON_BIN" "$SCRIPT_DIR/computer_use.py" type "$*"
        ;;
    clipboard-get|info)
        exec "$PYTHON_BIN" "$SCRIPT_DIR/computer_use.py" "$ACTION"
        ;;
    clipboard-set)
        exec "$PYTHON_BIN" "$SCRIPT_DIR/computer_use.py" clipboard-set "$*"
        ;;
    windows)
        echo '{"ok":false,"error":"windows is not provided by computer-use; use desktop-map-cdp.py"}'
        exit 2
        ;;
    *)
        echo '{"ok":false,"error":"unknown computer-use action"}'
        exit 2
        ;;
esac
