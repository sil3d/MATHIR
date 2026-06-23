"""MATHIR daemon client — thin wrapper for the MATHIR Playground UI.

Connects to the running MATHIR daemon (mathir_mcp/mathir_lib/mathir_daemon.py)
on port 7338 by default. Used by ui_server.py to expose memory endpoints
in the UI (chat history, recall, stats, etc.).

The daemon exposes 17 MCP tools. We wrap a subset that's useful from the UI:

  - memory_save      → store a chat message + metadata
  - memory_recall    → search past memories by query
  - memory_stats     → counts per tier
  - memory_sessions  → recent sessions
  - memory_promote   → promote a memory up one tier
  - memory_decay     → run Ebbinghaus decay

If the daemon is not running, all calls return {error: "..."} instead of raising.
The UI should handle these gracefully (show "MATHIR daemon not running" banner).
"""
import json
import socket
from typing import Any, Dict, List, Optional

from env_config import get_mathir_daemon_host, get_mathir_daemon_port


class DaemonUnavailable(Exception):
    """Raised when the daemon is not reachable."""
    pass


def _call_daemon(method: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
    """Call a daemon RPC method over TCP. Returns the parsed JSON response.

    Raises DaemonUnavailable if connection fails.
    """
    host = get_mathir_daemon_host()
    port = get_mathir_daemon_port()
    payload = json.dumps({"method": method, "params": params or {}}).encode("utf-8")
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.sendall(payload)
        chunks = []
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks).decode("utf-8", errors="replace")
        if not data:
            return {"error": f"Daemon at {host}:{port} returned empty response"}
        return json.loads(data)
    except (socket.error, ConnectionRefusedError, OSError) as e:
        raise DaemonUnavailable(f"Cannot reach MATHIR daemon at {host}:{port}: {e}")
    finally:
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass


def is_daemon_available() -> bool:
    """Check if the daemon is reachable. Returns True/False, never raises."""
    try:
        _call_daemon("ping", timeout=2.0)
        return True
    except (DaemonUnavailable, Exception):
        return False


def daemon_ping() -> dict:
    """Ping the daemon. Returns {pong, dim, uptime} or {error: ...}."""
    try:
        return _call_daemon("ping", timeout=5.0)
    except DaemonUnavailable as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# High-level wrappers used by ui_server.py
# ---------------------------------------------------------------------------
def memory_save(content: str, agent: str = "vision_ui",
                block_type: str = "episodic", label: str = "chat",
                priority: int = 5, project: str = "vision_testing") -> dict:
    """Save a chat message to MATHIR memory."""
    return _call_daemon("memory_save", {
        "content": content,
        "agent": agent,
        "block_type": block_type,
        "label": label,
        "priority": priority,
        "project": project,
    })


def memory_recall(query: str, k: int = 5, project: str = "vision_testing") -> dict:
    """Recall memories by similarity."""
    return _call_daemon("memory_recall", {
        "query": query,
        "k": k,
        "project": project,
    })


def memory_stats(project: str = "vision_testing") -> dict:
    """Memory statistics."""
    return _call_daemon("memory_stats", {"project": project})


def memory_sessions(limit: int = 10) -> dict:
    """Recent memory sessions."""
    return _call_daemon("memory_sessions", {"limit": limit})


def memory_promote(memory_id: str, force: bool = False) -> dict:
    """Promote a memory to the next tier."""
    return _call_daemon("memory_promote", {"memory_id": memory_id, "force": force})


def memory_decay(threshold_days: int = 30) -> dict:
    """Apply Ebbinghaus decay to old memories."""
    return _call_daemon("memory_decay", {"threshold_days": threshold_days})


def memory_delete(memory_id: str) -> dict:
    """Delete a memory by ID."""
    return _call_daemon("memory_delete", {"memory_id": memory_id})