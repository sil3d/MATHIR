#!/usr/bin/env bash
# MATHIR Daemon Direct Socket (bash, no Python)
# Usage: source ./mathir_daemon.sh
#   mathir_recall "Mycerise" 5
#   mathir_stats
#   mathir_save "hello world" "label" agent-name
#
# Cross-platform: macOS, Linux, WSL, Git Bash (MSYS).
# Wire protocol matches mathir_daemon.py: send one JSON request, daemon reads
# until EOF on the socket, then sends one JSON response and closes.
#
# Key insight: we must signal EOF after writing the request. With /dev/tcp we
# can't half-close cleanly, so the read is bounded with `read -t <timeout>`.
# We exploit the fact that json.dumps() output is always one line.
#
# Override daemon location: MATHIR_HOST / MATHIR_PORT env vars.

MATHIR_HOST="${MATHIR_HOST:-127.0.0.1}"
MATHIR_PORT="${MATHIR_PORT:-7338}"
MATHIR_TIMEOUT="${MATHIR_TIMEOUT:-10}"   # seconds

# /dev/tcp is a bash magic path (Bash 2.04+), not a real filesystem entry.
# `[ -e /dev/tcp ]` always returns false — use BASH_VERSION instead.
_mathir_supports_devtcp() {
    [ -n "${BASH_VERSION:-}" ]
}

# Detect command existence (POSIX-friendly).
_mathir_have() { command -v "$1" >/dev/null 2>&1; }

# ----------------------------------------------------------------------------
# Low-level: send request, read one-line JSON response, close.
# Args: $1 = JSON request string. Echoes response body to stdout.
# ----------------------------------------------------------------------------
mathir_call_raw() {
    local request="$1"

    # Prefer /dev/tcp (zero external deps, ~5ms startup). Fall back to nc
    # if bash lacks /dev/tcp support (e.g. sh, dash, some minimal builds).
    if _mathir_supports_devtcp && _mathir_devtcp_call "$request"; then
        return 0
    fi

    if _mathir_have nc; then
        # Try Linux-style nc -N first (closes socket after EOF on stdin)
        if printf '%s' "$request" | nc -N "$MATHIR_HOST" "$MATHIR_PORT" 2>/dev/null; then
            return 0
        fi
        # BSD/macOS nc -q 0 (linger 0s after EOF)
        if printf '%s' "$request" | nc -q 0 "$MATHIR_HOST" "$MATHIR_PORT" 2>/dev/null; then
            return 0
        fi
        # Last resort: nc + timeout (hangs the full timeout on most nc versions)
        if _mathir_have timeout || _mathir_have gtimeout; then
            local tm
            tm=$(_mathir_have timeout && echo timeout || echo gtimeout)
            printf '%s' "$request" | "$tm" "$MATHIR_TIMEOUT" nc "$MATHIR_HOST" "$MATHIR_PORT" 2>/dev/null
            return 0
        fi
        printf '%s' "$request" | nc "$MATHIR_HOST" "$MATHIR_PORT" 2>/dev/null
        return 0
    fi

    echo "MATHIR: no transport available (need bash /dev/tcp or nc)" >&2
    return 1
}

# /dev/tcp transport. No clean half-close from bash, so we bound the read
# with `read -t`. JSON response is one line (json.dumps adds no newlines).
# IMPORTANT: bash `read -t` returns >128 on timeout BUT still populates $line
# with whatever bytes were buffered. We must NOT clobber $line on timeout.
#
# Strategy: read with a short initial timeout (covers daemon processing
# latency, typically <100ms), then probe with a tiny timeout to confirm EOF.
# Net result: ~timeout_1 + ~timeout_2 for most calls (~150ms total).
# With daemon not closing the connection, we can't know exactly when the
# response is done — but a probe-after-read with empty result is a strong
# signal. The daemon processes & sends in <100ms; for very slow handlers
# (e.g. memory_push with extraction) bump MATHIR_READ_TIMEOUT.
_mathir_devtcp_call() {
    local request="$1"
    local t1="${MATHIR_READ_TIMEOUT:-0.1}"
    local t2="${MATHIR_PROBE_TIMEOUT:-0.05}"
    local line= probe=
    # shellcheck disable=SC2169  # /dev/tcp is bash magic, not a real path
    exec 3<>/dev/tcp/"$MATHIR_HOST"/"$MATHIR_PORT" || return 1
    printf '%s' "$request" >&3
    IFS= read -r -t "$t1" line <&3 2>/dev/null
    IFS= read -r -t "$t2" probe <&3 2>/dev/null
    exec 3>&- 2>/dev/null
    exec 3<&- 2>/dev/null
    # Prefer the larger read (handles split packets).
    local result="$line"
    [ ${#probe} -gt ${#result} ] && result="$probe"
    if [ -n "$result" ]; then
        printf '%s' "$result"
        return 0
    fi
    return 1
}

# JSON-encode a string safely for the JSON-RPC payload.
# Handles: backslash, double-quote, and JSON control escapes (\b \f \n \r \t).
# We don't need full unicode escape (daemon accepts UTF-8).
_mathir_json_escape() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\b'/\\b}
    s=${s//$'\f'/\\f}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '%s' "$s"
}

# Build a JSON-RPC request from key-value pairs.
# Args: method, [key value key value ...]
# Values that look like JSON literals (numbers, booleans, null, {...}, [...])
# are emitted as-is; everything else is string-quoted.
_mathir_build_request() {
    local method="$1"; shift
    local request="{\"method\":\"$method\",\"params\":{"
    local first=1 key val
    while [ $# -ge 2 ]; do
        key=$1; val=$2; shift 2
        [ $first -eq 0 ] && request="${request},"
        first=0
        if [[ "$val" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] \
            || [ "$val" = "true" ] || [ "$val" = "false" ] || [ "$val" = "null" ] \
            || [[ "$val" =~ ^\{ ]] || [[ "$val" =~ ^\[ ]]; then
            request="${request}\"${key}\":${val}"
        else
            request="${request}\"${key}\":\"$(_mathir_json_escape "$val")\""
        fi
    done
    request="${request}}}"
    printf '%s' "$request"
}

# Pretty-print response if jq is available, else raw.
_mathir_print() {
    if _mathir_have jq; then
        printf '%s' "$1" | jq . 2>/dev/null || printf '%s\n' "$1"
    else
        printf '%s\n' "$1"
    fi
}

# ----------------------------------------------------------------------------
# Public API — mirrors the PowerShell module.
# ----------------------------------------------------------------------------

mathir_ping() {
    _mathir_print "$(mathir_call_raw "{\"method\":\"ping\",\"params\":{}}")"
}

mathir_stats() {
    _mathir_print "$(mathir_call_raw "{\"method\":\"memory_stats\",\"params\":{}}")"
}

# mathir_recall <query> [k=5]
mathir_recall() {
    local query="$1" k="${2:-5}"
    local req
    req=$(_mathir_build_request "memory_recall" "query" "$query" "k" "$k")
    _mathir_print "$(mathir_call_raw "$req")"
}

# mathir_recall_hybrid <query> [k=5] [vector_weight=1.0] [bm25_weight=1.0]
mathir_recall_hybrid() {
    local query="$1" k="${2:-5}" vw="${3:-1.0}" bw="${4:-1.0}"
    local req
    req=$(_mathir_build_request "memory_hybrid_search" \
        "query" "$query" "k" "$k" "vector_weight" "$vw" "bm25_weight" "$bw")
    _mathir_print "$(mathir_call_raw "$req")"
}

# mathir_save <content> [label=""] [agent="bash"] [block_type="episodic"] [priority=5]
mathir_save() {
    local content="$1" label="${2:-}" agent="${3:-bash}" block_type="${4:-episodic}" priority="${5:-5}"
    local req
    req=$(_mathir_build_request "memory_save" \
        "content" "$content" "label" "$label" "agent" "$agent" \
        "block_type" "$block_type" "priority" "$priority")
    _mathir_print "$(mathir_call_raw "$req")"
}

# mathir_delete <memory_id>
mathir_delete() {
    local mid="$1"
    local req
    req=$(_mathir_build_request "memory_delete" "memory_id" "$mid")
    _mathir_print "$(mathir_call_raw "$req")"
}

# Generic passthrough for any daemon method.
# Usage: mathir_call <method> [k1 v1 k2 v2 ...]
mathir_call() {
    local method="$1"; shift
    local req
    if [ $# -gt 0 ]; then
        req=$(_mathir_build_request "$method" "$@")
    else
        req="{\"method\":\"$method\",\"params\":{}}"
    fi
    _mathir_print "$(mathir_call_raw "$req")"
}

# ----------------------------------------------------------------------------
# Help / self-introspection
# ----------------------------------------------------------------------------
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'EOF'
MATHIR bash module — direct-socket client for the MATHIR daemon.

Functions:
  mathir_ping
  mathir_stats
  mathir_recall <query> [k=5]
  mathir_recall_hybrid <query> [k=5] [vector_weight=1.0] [bm25_weight=1.0]
  mathir_save <content> [label=""] [agent="bash"] [block_type="episodic"] [priority=5]
  mathir_delete <memory_id>
  mathir_call <method> [k1 v1 k2 v2 ...]    # generic

Env vars:
  MATHIR_HOST  (default 127.0.0.1)
  MATHIR_PORT  (default 7338)
  MATHIR_TIMEOUT  (default 10 seconds)

Source this file in your shell:
  source /path/to/mathir_daemon.sh
EOF
    return 0 2>/dev/null || exit 0
fi

# If sourced, print a short banner. If executed, run a ping.
if [ "${BASH_SOURCE[0]:-}" != "${0}" ]; then
    # Sourced
    echo "MATHIR bash module loaded → $MATHIR_HOST:$MATHIR_PORT  (try: mathir_ping)"
else
    # Executed directly → ping
    mathir_ping
fi
