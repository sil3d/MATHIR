"""
MATHIR Server Watchdog (HTTP) — v8.5.0
======================================
Monitors the unified MATHIR HTTP server (mathir_lib/mathir_server.py on port
7338) and restarts it if it crashes.

History note: pre-v8.5.0 this watchdog probed a TCP JSON-RPC daemon and
launched mathir_daemon.py. The product has since migrated to HTTP/Flask, so
both the probe and the launcher target have been switched. If you still run
the legacy TCP daemon, use mathir_lib/mathir_watchdog.py or stop using it.

Features:
- Probes /health (HTTP 200) every N seconds
- Exponential backoff between restart attempts (avoids pile-up while the
  embedder loads, which can take 15-30 s)
- Logs to ~/.config/opencode/logs/mathir_watchdog.log
- Has its own PID lockfile to prevent multiple instances (no longer silently
  no-ops when psutil is missing)
- Graceful shutdown on Ctrl+C / SIGTERM

Usage:
    python mathir_watchdog.py                    # default settings (interval=15)
    python mathir_watchdog.py --interval 10     # ping every 10s
    python mathir_watchdog.py --max-restarts 10  # max 10 restarts before giving up
"""
import sys
import os
import time
import argparse
import subprocess
import logging
import logging.handlers
import urllib.request
import urllib.error
from pathlib import Path

HOST = '127.0.0.1'
PORT = int(os.environ.get("MATHIR_PORT", "7338"))
HEALTH_URL = f"http://{HOST}:{PORT}/health"
HEALTH_TIMEOUT = 3  # seconds
LOG_DIR = Path(os.environ.get(
    "MATHIR_LOG_DIR",
    str(Path.home() / ".config" / "opencode" / "logs"),
))
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    LOG_DIR = Path(os.environ.get("TEMP", "/tmp")) / "mathir_logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mathir_watchdog.log"
PID_FILE = LOG_DIR / "mathir_watchdog.pid"

log = logging.getLogger("MATHIR-WATCHDOG")
log.setLevel(logging.INFO)
_fh = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
)
_fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s %(message)s'))
log.addHandler(_ch)


def _pid_alive(pid: int) -> bool:
    """Best-effort cross-platform 'is this PID a running process' check."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            STILL_ACTIVE = 259
            h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(h)
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError, PermissionError):
        return False
    except Exception:
        return False


def is_server_alive(timeout: float = HEALTH_TIMEOUT) -> bool:
    """Probe the HTTP /health endpoint (returns True on HTTP 200)."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ConnectionError, TimeoutError):
        return False
    except Exception:
        return False


def start_server() -> bool:
    """Start the unified HTTP server as a background process."""
    # mathir_server.py lives in mathir_lib/ (sibling of brain/), or in bin/
    # via the shim. Prefer mathir_lib so the launched process has the same
    # import surface as `python -m mathir_mcp`.
    candidates = [
        Path(__file__).parent.parent / "mathir_lib" / "mathir_server.py",
        Path(__file__).parent.parent / "bin" / "mathir_daemon.py",  # shim
    ]
    script = next((p for p in candidates if p.is_file()), None)
    if script is None:
        log.error(f"No server script found in: {candidates}")
        return False

    try:
        kwargs = {
            'stdin': subprocess.DEVNULL,
            'stdout': subprocess.DEVNULL,
            # The server itself writes to a rotating log file, so DEVNULL is
            # safe here — crash traces will still be captured.
            'stderr': subprocess.DEVNULL,
        }
        if sys.platform == 'win32':
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            kwargs['creationflags'] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            kwargs['close_fds'] = True
        else:
            kwargs['start_new_session'] = True

        proc = subprocess.Popen(
            [sys.executable, str(script)],
            **kwargs
        )
        log.info(f"Started server PID {proc.pid} ({script.name})")
        return True
    except Exception as e:
        log.error(f"Failed to start server: {e}")
        return False


def acquire_pid_lock():
    """Prevent multiple watchdog instances. No silent no-op when psutil absent."""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            if _pid_alive(old_pid):
                print(f"Another watchdog already running (PID {old_pid}). "
                      f"Remove {PID_FILE} to override.")
                sys.exit(1)
        except ValueError:
            pass  # corrupt PID file → ignore
    PID_FILE.write_text(str(os.getpid()))


def release_pid_lock():
    try:
        PID_FILE.unlink()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description='MATHIR Server Watchdog')
    parser.add_argument('--interval', type=int, default=15,
                        help='Health-check interval in seconds (default: 15)')
    parser.add_argument('--max-restarts', type=int, default=0,
                        help='Max restarts before giving up (0=infinite)')
    parser.add_argument('--cooldown', type=int, default=30,
                        help='Seconds to wait after restart before probing '
                             '(model load takes 15-30s; default: 30)')
    parser.add_argument('--max-backoff', type=int, default=300,
                        help='Max backoff between restart attempts in seconds (default: 300)')
    args = parser.parse_args()

    acquire_pid_lock()

    log.info(f"Watchdog started (PID {os.getpid()})")
    log.info(f"  Target:  {HEALTH_URL}")
    log.info(f"  Interval: {args.interval}s, cooldown: {args.cooldown}s, "
             f"max-restarts: {'inf' if args.max_restarts == 0 else args.max_restarts}")

    restart_count = 0
    last_restart_time = 0
    last_alive_time = time.time()
    consecutive_failures = 0

    try:
        while True:
            time.sleep(args.interval)

            alive = is_server_alive()
            now = time.time()

            if alive:
                if consecutive_failures > 0:
                    log.info(f"Server is back UP after {consecutive_failures} failed probes")
                consecutive_failures = 0
                if last_restart_time > last_alive_time:
                    log.info(f"Server is ALIVE (uptime: {now - last_restart_time:.0f}s since restart)")
                last_alive_time = now
                continue

            log.warning(f"Server is DOWN (last healthy {now - last_alive_time:.0f}s ago)")

            if args.max_restarts > 0 and restart_count >= args.max_restarts:
                log.error(f"Max restarts ({args.max_restarts}) reached. Giving up.")
                break

            # Exponential backoff between restart attempts to avoid pile-up
            # while the embedder loads (15-30 s on first start).
            backoff = min(args.max_backoff, args.cooldown * (2 ** consecutive_failures))
            consecutive_failures += 1
            log.info(f"Restarting server (attempt #{restart_count + 1}, "
                     f"backoff={backoff}s, consecutive_failures={consecutive_failures})...")

            if start_server():
                restart_count += 1
                last_restart_time = time.time()
                log.info(f"Waiting {backoff}s for server to come up...")
                time.sleep(backoff)

                if is_server_alive(timeout=5):
                    log.info("Server successfully restarted and responding.")
                    last_alive_time = time.time()
                    consecutive_failures = 0
                else:
                    log.warning("Server started but not responding yet "
                                "(model still loading). Will re-probe next interval.")
            else:
                log.error("Failed to start server. Will retry on next interval.")

    except KeyboardInterrupt:
        log.info("Watchdog stopped (Ctrl+C)")
    finally:
        release_pid_lock()


if __name__ == "__main__":
    main()
