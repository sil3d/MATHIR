#!/usr/bin/env python3
"""
MATHIR Daemon launcher (HTTP) — shim.

Historically this file was a full TCP JSON-RPC daemon. As of v8.5.0 the
canonical server is ``mathir_lib/mathir_server.py`` (Flask + Waitress, HTTP
on port 7338). The TCP daemon was retired because:

  - Raw TCP sockets are fragile (pipe-buffer crashes, no error framing).
  - All MCP clients (``mathir_mcp_server``, ``mathir_client``) speak HTTP.
  - Two separate processes / protocols on the same port caused split-brain
    outages where a watchdog killed a healthy server of the "wrong" protocol.

This file remains as a thin launcher so existing deployment paths
(``auto_start.bat``, ``auto_start_helpers.ps1``, systemd unit, launchd plist)
keep working unchanged. It tries, in order:

  1. ``mathir_lib/mathir_server.py`` (source-tree layout)
  2. ``./mathir_server.py``        (deployed bin/ layout)
  3. ``python -m mathir_mcp`` sub-process (last-resort: relies on installed
     package)

If you genuinely need the legacy TCP daemon for an experimental reason, find
it in ``mathir_mcp/mathir_lib/mathir_daemon.py`` and run it explicitly.
"""

import sys
import os
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    # Source-tree layout: mathir_mcp/bin/mathir_daemon.py → ../mathir_lib/
    _HERE.parent / "mathir_lib" / "mathir_server.py",
    # Flat deployed layout: ~/.config/{agent}/bin/mathir_server.py
    _HERE / "mathir_server.py",
    # Package deployed by install_smart.py: ~/.config/{agent}/tools/mathir_mcp/mathir_lib/
    _HERE.parent / "tools" / "mathir_mcp" / "mathir_lib" / "mathir_server.py",
    # Other common relative layouts
    _HERE.parent / "mathir_lib" / "mathir_server.py",
]
_server_path = next((p for p in _CANDIDATES if p.is_file()), None)


def _launch_via_module():
    """Last-resort fallback: spawn `python -m mathir_mcp` and wait."""
    cmd = [sys.executable, "-m", "mathir_mcp"]
    sys.stderr.write(f"[mathir_daemon shim] Falling back to: {' '.join(cmd)}\n")
    try:
        return subprocess.call(cmd)
    except OSError as e:
        sys.stderr.write(f"[mathir_daemon shim] python -m mathir_mcp failed: {e}\n")
        return 2


def main():
    if _server_path is None:
        sys.stderr.write(
            "[mathir_daemon shim] mathir_server.py not found in:\n"
            + "\n".join(f"  - {p}" for p in _CANDIDATES) + "\n"
            "Trying `python -m mathir_mcp` instead.\n"
        )
        return _launch_via_module()

    # Make mathir_lib importable and dispatch
    sys.path.insert(0, str(_server_path.parent))
    try:
        import mathir_server  # type: ignore
    except Exception as e:
        sys.stderr.write(
            f"[mathir_daemon shim] Failed to import mathir_server "
            f"from {_server_path.parent}: {e}\n"
            "Trying `python -m mathir_mcp` instead.\n"
        )
        return _launch_via_module()
    return mathir_server.main() or 0


if __name__ == "__main__":
    sys.exit(main())
