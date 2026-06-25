#!/usr/bin/env python3
"""
MATHIR Server Watchdog — v8.5.0
Ensures the unified server is running on port 7338.
Run this at startup or periodically via cron/task scheduler.

Usage:
  python mathir_watchdog.py           # check + start if needed
  python mathir_watchdog.py --status  # just check status
  python mathir_watchdog.py --kill    # kill all MATHIR servers
"""

import sys
import os
import subprocess
import urllib.request
import time
import argparse
from pathlib import Path

PORT = 7338
HEALTH_URL = f"http://127.0.0.1:{PORT}/health"
SERVER_SCRIPT = Path(__file__).parent / "mathir_server.py"
WORKDIR = Path(os.environ.get("USERPROFILE", str(Path.home())))

def check_server() -> bool:
    """Check if server is responding on port 7338."""
    try:
        res = urllib.request.urlopen(HEALTH_URL, timeout=3)
        return res.status == 200
    except Exception:
        return False

def check_port() -> bool:
    """Check if port 7338 is in use."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", PORT)) == 0
    except Exception:
        return False

def start_server() -> int:
    """Start the server in background. Returns PID."""
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        cwd=str(WORKDIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    return proc.pid

def kill_servers():
    """Kill all MATHIR server processes."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/IM", "python.exe", "/FI", f"WINDOWTITLE eq mathir*"],
            capture_output=True,
        )
        # Also kill by port
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if f":{PORT}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    else:
        result = subprocess.run(
            ["lsof", "-ti", f":{PORT}"],
            capture_output=True, text=True,
        )
        for pid in result.stdout.strip().split("\n"):
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True)

def main():
    parser = argparse.ArgumentParser(description="MATHIR Server Watchdog")
    parser.add_argument("--status", action="store_true", help="Check server status")
    parser.add_argument("--kill", action="store_true", help="Kill all MATHIR servers")
    parser.add_argument("--force", action="store_true", help="Force restart")
    args = parser.parse_args()

    if args.kill:
        print("Killing all MATHIR servers...")
        kill_servers()
        print("Done.")
        return

    # Check status
    port_open = check_port()
    server_ok = check_server() if port_open else False

    if args.status:
        print(f"Port {PORT}: {'OPEN' if port_open else 'CLOSED'}")
        print(f"Server: {'RESPONDING' if server_ok else 'NOT RESPONDING'}")
        return

    # Auto-start if needed
    if server_ok and not args.force:
        print(f"Server OK on port {PORT}")
        return

    if not port_open:
        print(f"Port {PORT} not in use. Starting server...")
    else:
        print(f"Port {PORT} open but server not responding. Restarting...")
        kill_servers()
        time.sleep(2)

    pid = start_server()
    print(f"Server starting (PID: {pid}). Waiting 35s for embedder...")

    # Wait for server to be ready
    for i in range(7):
        time.sleep(5)
        if check_server():
            print(f"Server ready after {(i+1)*5}s")
            return

    print("Server still not ready after 35s. Check logs.")
    sys.exit(1)

if __name__ == "__main__":
    main()
