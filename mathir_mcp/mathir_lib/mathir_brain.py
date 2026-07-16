"""
MATHIR Brain Stack — All-in-One Launcher
=========================================
Starts:
1. The daemon (if not running)
2. The watchdog (monitors daemon, restarts on crash)
3. The universal injection proxy (auto-injects memories into LLM calls)
4. (Optional) The pre-cognitive priming file watcher

Usage:
    python mathir_brain.py start    # Start all services
    python mathir_brain.py stop     # Stop all services
    python mathir_brain.py status   # Show status

v8.9.4: switched from mathir_inject_proxy.py to mathir_proxy.py. The old
script spoke to the daemon over raw TCP JSON-RPC (`_fast_call`, a direct
socket connection expecting newline-delimited JSON), a protocol retired
in the v8.5.0 HTTP/Flask rewrite -- every injection attempt through it
was silently failing (falls back to "no memories found" rather than
crashing, so this went unnoticed). mathir_proxy.py speaks HTTP to the
daemon like everything else, supports both Anthropic's native
/v1/messages and OpenAI-compatible /v1/chat/completions, and is the
actively maintained, security-reviewed implementation.
"""
import sys
import os
import time
import subprocess
import argparse
import urllib.request
import urllib.error
from pathlib import Path

HOST = '127.0.0.1'
DAEMON_PORT = int(os.environ.get("MATHIR_PORT", "7338"))
PROXY_PORT = int(os.environ.get("MATHIR_PROXY_PORT", "7339"))

# Portable script resolution: all scripts live next to this file.
_HERE = Path(__file__).parent.resolve()
DAEMON_SCRIPT = _HERE / "mathir_daemon.py"
WATCHDOG_SCRIPT = _HERE / "mathir_watchdog.py"
PROXY_SCRIPT = _HERE / "mathir_proxy.py"

# Anthropic by default (matches Claude Code, the most common client for
# this launcher); override with MATHIR_PROXY_TARGET for an OpenAI-
# compatible provider or a local model server (Ollama/llama.cpp/etc).
LLM_TARGET = os.environ.get("MATHIR_PROXY_TARGET", "https://api.anthropic.com")


def is_port_listening(port: int, timeout: float = 1.0) -> bool:
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((HOST, port))
        s.close()
        return True
    except (socket.error, ConnectionRefusedError):
        return False


def is_server_alive(timeout: float = 2.0) -> bool:
    """Probe the HTTP /health endpoint (returns True on HTTP 200).

    v8.5.0+: the server is HTTP/Flask, not the old TCP JSON-RPC daemon.
    An HTTP GET replaces the raw-socket ``sendall(b'{"method":"ping"...}')``
    probe, which no longer works and had a ``\\n`` framing divergence
    between the two copies.
    """
    health_url = f"http://{HOST}:{DAEMON_PORT}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ConnectionError, TimeoutError):
        return False
    except Exception:
        return False


def start_detached(script: Path, args: list, name: str) -> bool:
    """Start a script as detached process."""
    try:
        kwargs = {
            'stdin': subprocess.DEVNULL,
            'stdout': subprocess.DEVNULL,
            'stderr': subprocess.DEVNULL,
        }
        if sys.platform == 'win32':
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            kwargs['creationflags'] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            kwargs['close_fds'] = True
        else:
            kwargs['start_new_session'] = True
        proc = subprocess.Popen([sys.executable, str(script)] + args, **kwargs)
        print(f"  [OK] {name} started (PID {proc.pid})")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def start_all():
    print(f"=== Starting MATHIR Brain Stack ===")
    print(f"  Daemon port: {DAEMON_PORT}")
    print(f"  Proxy port: {PROXY_PORT}")
    print(f"  LLM target: {LLM_TARGET}")
    print()
    
    # 1. Daemon
    if is_server_alive():
        print("  [OK] Daemon already running")
    else:
        print("  Starting daemon...")
        if not start_detached(DAEMON_SCRIPT, [], "daemon"):
            print("  [ABORT] Daemon failed to start")
            return False
        # Wait for daemon to load model
        print("  Waiting for daemon to load model (30s)...")
        for i in range(30):
            time.sleep(1)
            if is_server_alive():
                print(f"  [OK] Daemon ready after {i+1}s")
                break
        else:
            print("  [WARN] Daemon not responding yet (model still loading)")
    
    # 2. Watchdog
    if is_port_listening(DAEMON_PORT):
        print("  Starting watchdog...")
        start_detached(WATCHDOG_SCRIPT, ['--interval', '15', '--cooldown', '10'], "watchdog")
    
    # 3. Proxy
    print("  Starting inject proxy...")
    start_detached(PROXY_SCRIPT, ['--target', LLM_TARGET, '--port', str(PROXY_PORT)], "proxy")
    
    time.sleep(2)
    
    print()
    print("=== Status ===")
    print(f"  Daemon (port {DAEMON_PORT}): {'ALIVE' if is_server_alive() else 'DOWN'}")
    print(f"  Proxy (port {PROXY_PORT}): {'ALIVE' if is_port_listening(PROXY_PORT) else 'DOWN'}")
    print()
    print("Point your LLM client at the proxy (pick the one matching its wire format):")
    print(f"  ANTHROPIC_BASE_URL=http://127.0.0.1:{PROXY_PORT}       (Claude Code etc. -- no /v1)")
    print(f"  OPENAI_BASE_URL=http://127.0.0.1:{PROXY_PORT}/v1        (OpenAI-compatible tools -- /v1 required)")
    print()


def stop_all():
    print("=== Stopping MATHIR Brain Stack ===")
    
    import psutil
    targets = ['mathir_daemon.py', 'mathir_watchdog.py', 'mathir_proxy.py']
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info.get('cmdline') or [])
            for target in targets:
                if target in cmdline:
                    print(f"  Killing PID {proc.info['pid']} ({target})")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    print("  Done.")


def status():
    print("=== MATHIR Brain Stack Status ===")
    print()
    print(f"  Daemon (port {DAEMON_PORT}):")
    if is_server_alive():
        print(f"    Status: ALIVE")
    else:
        print(f"    Status: DOWN")
    print()
    print(f"  Inject proxy (port {PROXY_PORT}):")
    if is_port_listening(PROXY_PORT):
        print(f"    Status: LISTENING")
        print(f"    Forward to: {LLM_TARGET}")
    else:
        print(f"    Status: DOWN")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MATHIR Brain Stack Manager')
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('start', help='Start all services')
    sub.add_parser('stop', help='Stop all services')
    sub.add_parser('status', help='Show status')
    args = parser.parse_args()
    
    if args.cmd == 'start':
        start_all()
    elif args.cmd == 'stop':
        stop_all()
    elif args.cmd == 'status':
        status()
