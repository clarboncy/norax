#!/bin/bash
# Cooperative shell flow timer. This script must be sourced so its state stays
# in the caller's shell; it does not mutate shared files under /tmp.
# Usage: source flow_timer.sh start [task_name]
#        source flow_timer.sh check
#        source flow_timer.sh stop

NORAX_SHELL_FLOW_START="${NORAX_SHELL_FLOW_START:-}"
NORAX_SHELL_FLOW_TASK="${NORAX_SHELL_FLOW_TASK:-}"
NORAX_SHELL_FLOW_MAX_SECONDS="${NORAX_SHELL_FLOW_MAX_SECONDS:-1800}"

flow_timer_start() {
    local task="${1:-unnamed}"
    task="${task//$'\n'/ }"
    task="${task//$'\r'/ }"
    NORAX_SHELL_FLOW_START="$(date +%s)"
    NORAX_SHELL_FLOW_TASK="$task"
    export NORAX_SHELL_FLOW_START NORAX_SHELL_FLOW_TASK NORAX_SHELL_FLOW_MAX_SECONDS
    echo "[FLOW TIMER] Started: $task at $(date -Iseconds)"
}

flow_timer_check() {
    if [[ -z "$NORAX_SHELL_FLOW_START" ]]; then
        echo "[FLOW TIMER] No active flow"
        return 1
    fi
    local start="$NORAX_SHELL_FLOW_START"
    local now
    now="$(date +%s)"
    local elapsed=$((now - start))
    local remaining=$((NORAX_SHELL_FLOW_MAX_SECONDS - elapsed))
    local task="${NORAX_SHELL_FLOW_TASK:-unknown}"
    
    echo "[FLOW TIMER] Task: $task | Elapsed: ${elapsed}s | Remaining: ${remaining}s"
    
    if [[ $elapsed -ge $NORAX_SHELL_FLOW_MAX_SECONDS ]]; then
        echo "[FLOW TIMER] HARD STOP TRIGGERED — configured deadline exceeded"
        return 2  # Special exit code for hard stop
    fi
    return 0
}

flow_timer_stop() {
    if [[ -n "$NORAX_SHELL_FLOW_START" ]]; then
        local start="$NORAX_SHELL_FLOW_START"
        local now
        now="$(date +%s)"
        local elapsed=$((now - start))
        local task="${NORAX_SHELL_FLOW_TASK:-unknown}"
        echo "[FLOW TIMER] Stopped: $task | Total: ${elapsed}s"
        NORAX_SHELL_FLOW_START=""
        NORAX_SHELL_FLOW_TASK=""
        export NORAX_SHELL_FLOW_START NORAX_SHELL_FLOW_TASK
    else
        echo "[FLOW TIMER] No active flow to stop"
    fi
}

# State lives in shell variables, so executing this file would discard a newly
# started timer as soon as the process exits.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Usage: source ${BASH_SOURCE[0]} {start [task]|check|stop}" >&2
    exit 1
fi

# Dispatch
case "${1:-}" in
    start) flow_timer_start "$2" ;;
    check) flow_timer_check ;;
    stop) flow_timer_stop ;;
    *) echo "Usage: source ${BASH_SOURCE[0]} {start [task]|check|stop}" ;;
esac
