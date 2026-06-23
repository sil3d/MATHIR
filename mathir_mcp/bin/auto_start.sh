#!/usr/bin/env bash
# ============================================================================
# MATHIR Daemon Auto-Start (Linux / macOS)
# ----------------------------------------------------------------------------
# Bash equivalent of auto_start.bat. Launches the MATHIR cognitive memory
# daemon in the background, detached from the current shell, and verifies the
# daemon is actually listening on port 7338.
#
# Usage:   ./auto_start.sh
# Exit:    0 = daemon listening on port 7338
#          1 = failed to start after MAX_RETRIES attempts
#          2 = python not found / daemon script missing
# ============================================================================

set -u
set -o pipefail

# ---- Configuration ---------------------------------------------------------
MAX_RETRIES=3
WAIT_SECONDS=3
PORT=7338

# Resolve script directory (so we don't hardcode ~/.config/opencode/bin).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# python3 on PATH; if not present, try common locations.
if command -v python3 >/dev/null 2>&1; then
    PYTHON_PATH="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_PATH="$(command -v python)"
else
    echo "[FATAL] python3 not found on PATH"
    exit 2
fi

DAEMON_PATH="${DAEMON_PATH:-$SCRIPT_DIR/mathir_daemon.py}"
LOG_PATH="${LOG_PATH:-$SCRIPT_DIR/mathir_daemon.log}"

# ---- Logging helper --------------------------------------------------------
log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" >> "$LOG_PATH"
    echo "[$ts] $*"
}

# ---- Sanity checks ---------------------------------------------------------
if [ ! -f "$DAEMON_PATH" ]; then
    log "[FATAL] Daemon script not found at: $DAEMON_PATH"
    exit 2
fi

# Ensure log file is writable.
mkdir -p "$(dirname "$LOG_PATH")" 2>/dev/null || true
touch "$LOG_PATH" 2>/dev/null || {
    echo "[FATAL] Cannot write to log: $LOG_PATH"
    exit 2
}

log "================================================================================"
log "auto_start.sh invoked (pid=$$)"
log "Python:    $PYTHON_PATH"
log "Daemon:    $DAEMON_PATH"
log "Port:      $PORT"
log "Max tries: $MAX_RETRIES (wait ${WAIT_SECONDS}s between)"

# ---- Port check helper (POSIX, no /dev/tcp fallback assumption) ------------
port_open() {
    # Uses `nc` if present (BSD/macOS), otherwise /dev/tcp (bash only).
    local host=127.0.0.1
    local port=$1
    if command -v nc >/dev/null 2>&1; then
        nc -z -G 1 "$host" "$port" 2>/dev/null
        return $?
    fi
    # bash's /dev/tcp is non-POSIX but available on bash/zsh.
    (timeout 1 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null) && {
        exec 3<&- 3>&- 2>/dev/null
        return 0
    }
    return 1
}

# ---- Already-running check -------------------------------------------------
if port_open "$PORT"; then
    log "Daemon already listening on port $PORT, nothing to do."
    exit 0
fi

log "Port $PORT is not open, starting daemon..."

# ---- Retry loop ------------------------------------------------------------
for attempt in $(seq 1 "$MAX_RETRIES"); do
    log "Attempt $attempt/$MAX_RETRIES: launching daemon..."

    # nohup + & detaches the process from the controlling terminal so the
    # daemon survives shell exit. stdout+stderr are appended to the log.
    nohup "$PYTHON_PATH" "$DAEMON_PATH" >> "$LOG_PATH" 2>&1 &
    DAEMON_PID=$!
    disown "$DAEMON_PID" 2>/dev/null || true

    sleep "$WAIT_SECONDS"

    if port_open "$PORT"; then
        log "SUCCESS: daemon is listening on port $PORT (attempt $attempt, pid $DAEMON_PID)."
        exit 0
    fi

    log "Attempt $attempt failed -- port $PORT still closed."

    # Kill the zombie child before retrying (if any).
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
        sleep 1
    fi
done

log "FATAL: daemon failed to start after $MAX_RETRIES attempts."
log "Last 30 lines of log for diagnosis:"
tail -n 30 "$LOG_PATH" 2>/dev/null | while IFS= read -r line; do
    log "[log] $line"
done
exit 1
