#!/usr/bin/env bash
# GOD POLLER — bash one-shot poller for god-mode (Linux/Mac)
# Usage:
#   ./god_poll.sh worker mimo-code 5
#   ./god_poll.sh orchestrator "" 5
#   ./god_poll.sh observer "" 10
#
# Press Ctrl+C to stop.

set -u

MODE="${1:-observer}"
NAME="${2:-}"
INTERVAL="${3:-5}"
DAEMON="${DAEMON_URL:-http://localhost:7338}"
STATE_FILE="${HOME}/.config/mathir/god_bridge_state.json"
LOG_FILE="${HOME}/.config/mathir/god_bridge.log"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    local ts level msg
    ts="$(date -u +%FT%TZ)"
    level="${2:-INFO}"
    msg="$1"
    echo "[$ts] [$level] $msg"
    echo "[$ts] [$level] $msg" >> "$LOG_FILE"
}

beep_notify() {
    printf '\a\a'
    if command -v paplay >/dev/null 2>&1; then
        paplay /usr/share/sounds/freedesktop/stereo/bell.oga >/dev/null 2>&1 &
    elif command -v afplay >/dev/null 2>&1; then
        afplay /System/Library/Sounds/Ping.aiff >/dev/null 2>&1 &
    fi
}

if [ "$MODE" = "worker" ] && [ -z "$NAME" ]; then
    log "-- Name required in worker mode" "ERROR"
    exit 2
fi

log "POLLER START — Mode=$MODE Name=$NAME Interval=${INTERVAL}s Daemon=$DAEMON"

trap 'log "POLLER STOP" "INFO"; exit 0' INT TERM

poll_worker() {
    curl -sf -m 5 -X POST -H "Content-Type: application/json" \
        -d "{\"agent\":\"$NAME\",\"status\":\"pending\"}" \
        "$DAEMON/api/god/poll" 2>/dev/null
}

poll_audit() {
    curl -sf -m 5 -X POST -H "Content-Type: application/json" \
        -d '{"limit":50}' \
        "$DAEMON/api/memory/audit" 2>/dev/null
}

while true; do
    case "$MODE" in
        worker)
            resp=$(poll_worker)
            if [ -n "$resp" ] && echo "$resp" | grep -q '"task":\s*\{'; then
                label=$(echo "$resp" | python -c "import sys,json; d=json.load(sys.stdin); print(d['task']['label'])" 2>/dev/null)
                if [ -n "$label" ]; then
                    log "NEW TASK: $label" "TASK"
                    echo "$resp" | head -c 300
                    echo
                    # FIX (2026-08-18): god/poll atomically claims the task
                    # (pending -> claimed). A notify-only poller that never
                    # acks leaves every task stuck on "claimed", blocking the
                    # whole queue (see /api/god/ack docstring). Mark it
                    # delivered so the next task can surface.
                    mid=$(echo "$resp" | python -c "import sys,json; d=json.load(sys.stdin); print(d['task'].get('memory_id',''))" 2>/dev/null)
                    if [ -n "$mid" ]; then
                        ack=$(curl -sf -m 5 -X POST -H "Content-Type: application/json" \
                            -d "{\"memory_id\":\"$mid\",\"status\":\"delivered\"}" \
                            "$DAEMON/api/god/ack" 2>/dev/null)
                        if [ -n "$ack" ]; then
                            log "ACK delivered: $mid" "TASK"
                        else
                            log "ACK FAILED for $mid" "WARN"
                        fi
                    fi
                    beep_notify
                fi
            fi
            ;;
        orchestrator)
            resp=$(poll_audit)
            if echo "$resp" | grep -q '"god:result:'; then
                log "NEW RESULTS detected" "RESULT"
                echo "$resp" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict) and e.get('label','').startswith('god:result:'):
                print(f\"  {e.get('label')} agent={e.get('agent')}\")
except: pass
" 2>/dev/null
                beep_notify
            fi
            ;;
        observer)
            resp=$(poll_audit)
            if [ -n "$resp" ]; then
                echo "$resp" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        for e in data:
            label = e.get('label','') if isinstance(e, dict) else ''
            if label.startswith('god:'):
                print(f\"  [{label.split(':')[1]}] {label} agent={e.get('agent')}\")
except: pass
" 2>/dev/null
            fi
            ;;
    esac
    sleep "$INTERVAL"
done
