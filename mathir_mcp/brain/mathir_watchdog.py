"""
MATHIR Daemon Watchdog
======================
Monitors the MATHIR daemon and restarts it if it crashes.
Designed to run as a background service on Windows.

Features:
- Pings daemon every N seconds
- Restarts if no response
- Logs to ~/.config/opencode/logs/mathir_watchdog.log
- Has its own PID lockfile to prevent multiple instances
- Graceful shutdown on Ctrl+C

Usage:
    python mathir_watchdog.py                    # default settings
    python mathir_watchdog.py --interval 10     # ping every 10s
    python mathir_watchdog.py --max-restarts 10  # max 10 restarts before giving up
"""
import sys
import os
import time
import socket
import argparse
import subprocess
import logging
from pathlib import Path

HOST = '127.0.0.1'
PORT = int(os.environ.get("MATHIR_PORT", "7338"))
PING_TIMEOUT = 2  # seconds
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "mathir_watchdog.log"
PID_FILE = LOG_DIR / "mathir_watchdog.pid"

LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("MATHIR-WATCHDOG")
log.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s %(message)s'))
log.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s %(message)s'))
log.addHandler(ch)


def is_daemon_alive(timeout: float = PING_TIMEOUT) -> bool:
    """Check if daemon responds to ping."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((HOST, PORT))
        # Send ping
        s.sendall(b'{"method":"ping","params":{}}')
        chunks = []
        try:
            s.settimeout(timeout)
            chunk = s.recv(4096)
            if chunk:
                chunks.append(chunk)
        except socket.timeout:
            pass
        s.close()
        if not chunks:
            return False
        body = b''.join(chunks).decode('utf-8', errors='ignore').strip()
        return 'pong' in body.lower() or 'dim' in body
    except (socket.error, ConnectionRefusedError):
        return False
    except Exception:
        return False


def start_daemon() -> bool:
    """Start the daemon as a background process."""
    # Daemon lives in mathir_lib/ (sibling of brain/), not next to this script.
    daemon_script = Path(__file__).parent.parent / "mathir_lib" / "mathir_daemon.py"
    if not daemon_script.exists():
        log.error(f"Daemon script not found: {daemon_script}")
        return False
    
    try:
        # Start as detached process
        kwargs = {
            'stdin': subprocess.DEVNULL,
            'stdout': subprocess.DEVNULL,
            'stderr': subprocess.DEVNULL,
        }
        if sys.platform == 'win32':
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            kwargs['creationflags'] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            kwargs['close_fds'] = True
        else:
            kwargs['start_new_session'] = True
        
        proc = subprocess.Popen(
            [sys.executable, str(daemon_script)],
            **kwargs
        )
        log.info(f"Started daemon PID {proc.pid}")
        return True
    except Exception as e:
        log.error(f"Failed to start daemon: {e}")
        return False


def acquire_pid_lock():
    """Prevent multiple watchdog instances."""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            # Check if process still alive
            import psutil
            if psutil.pid_exists(old_pid):
                print(f"Another watchdog already running (PID {old_pid})")
                sys.exit(1)
        except (ImportError, ValueError):
            pass  # psutil not available or stale PID file
    PID_FILE.write_text(str(os.getpid()))


def release_pid_lock():
    if PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='MATHIR Daemon Watchdog')
    parser.add_argument('--interval', type=int, default=15, help='Ping interval in seconds (default: 15)')
    parser.add_argument('--max-restarts', type=int, default=0, help='Max restarts before giving up (0=infinite)')
    parser.add_argument('--cooldown', type=int, default=5, help='Seconds to wait after restart before pinging')
    args = parser.parse_args()
    
    acquire_pid_lock()
    
    log.info(f"Watchdog started (PID {os.getpid()})")
    log.info(f"  Target: {HOST}:{PORT}")
    log.info(f"  Ping interval: {args.interval}s")
    log.info(f"  Max restarts: {'infinite' if args.max_restarts == 0 else args.max_restarts}")
    log.info(f"  Cooldown: {args.cooldown}s")
    
    restart_count = 0
    last_restart_time = 0
    last_alive_time = time.time()
    
    try:
        while True:
            time.sleep(args.interval)
            
            alive = is_daemon_alive()
            now = time.time()
            
            if alive:
                if not (last_alive_time > last_restart_time):
                    log.info(f"Daemon is ALIVE (uptime: {now - last_restart_time:.0f}s since restart)")
                last_alive_time = now
                continue
            
            log.warning(f"Daemon is DOWN (last seen {now - last_alive_time:.0f}s ago)")
            
            if args.max_restarts > 0 and restart_count >= args.max_restarts:
                log.error(f"Max restarts ({args.max_restarts}) reached. Giving up.")
                break
            
            log.info(f"Restarting daemon (attempt #{restart_count + 1})...")
            if start_daemon():
                restart_count += 1
                last_restart_time = time.time()
                log.info(f"Waiting {args.cooldown}s for daemon to come up...")
                time.sleep(args.cooldown)
                
                # Verify it's up
                if is_daemon_alive(timeout=5):
                    log.info("Daemon successfully restarted and responding.")
                    last_alive_time = time.time()
                else:
                    log.warning("Daemon started but not responding yet (model still loading).")
            else:
                log.error("Failed to start daemon. Will retry on next interval.")
    
    except KeyboardInterrupt:
        log.info("Watchdog stopped (Ctrl+C)")
    finally:
        release_pid_lock()


if __name__ == "__main__":
    main()
