#!/usr/bin/env python3
"""GOD MODE STOP — kill headless workers launched by god_mode_start.py.

Usage:
    python god_mode_stop.py --all
    python god_mode_stop.py --name opencode
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

_STATE_DIR = Path(os.environ.get(
    "MATHIR_HOME", str(Path.home() / ".config" / "MATHIR")
)) / "logs"
STATE_FILE = _STATE_DIR / "god_mode_workers.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"workers": []}
    return {"workers": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def kill_pid(pid: int) -> bool:
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Stop MATHIR god-mode headless workers")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--name", default=None, help="Stop only the worker registered under this god-mode name")
    args = ap.parse_args()

    if not args.all and not args.name:
        ap.error("pass --all or --name <worker-name>")

    state = load_state()
    workers = state.get("workers", [])
    remaining = []
    stopped = []
    for w in workers:
        if args.all or w.get("name") == args.name:
            ok = kill_pid(w["pid"])
            stopped.append({**w, "killed": ok})
        else:
            remaining.append(w)

    state["workers"] = remaining
    save_state(state)
    print(json.dumps({"stopped": stopped, "still_running": remaining}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
