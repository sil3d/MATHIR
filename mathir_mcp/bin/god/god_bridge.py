#!/usr/bin/env python3
"""
GOD BRIDGE — Polling daemon for MATHIR god-mode orchestrator.

Runs in background, polls the MATHIR HTTP API, filters god:* labels,
and notifies the user via console + optional beep + desktop notification.

Usage:
    # Worker mode (polling for tasks dispatched to me)
    python god_bridge.py --mode worker --name mimo-code --interval 5

    # Orchestrator mode (watching for worker results)
    python god_bridge.py --mode orchestrator --interval 5

    # Observer mode (watching everything god:*)
    python god_bridge.py --mode observer --interval 10

Cross-platform: Windows (winsound beep) + Linux/Mac (terminal bell + paplay optional).

Author: opencode-glm52 (god orchestrator, Mycerise V2 Tauri)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DAEMON = os.environ.get("MATHIR_DAEMON_URL", "http://localhost:7338")

_DEFAULT_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
_STATE_DIR = Path(os.environ.get("MYCERISE_STATE_DIR", _DEFAULT_CONFIG_DIR / "mycerise"))
DEFAULT_DB_STATE_FILE = _STATE_DIR / "god_bridge_state.json"
LOG_FILE = Path(os.environ.get("MYCERISE_LOG_FILE", _STATE_DIR / "god_bridge.log"))


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"last_seen_id": ""}
    return {"last_seen_id": ""}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def post_json(url: str, payload: dict, timeout: int = 5) -> dict | None:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        log(f"HTTP ERROR {url}: {e}", level="ERROR")
        return None


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def beep() -> None:
    """Cross-platform notification beep."""
    if sys.platform == "win32":
        try:
            import winsound  # type: ignore
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            print("\a", end="", flush=True)
    else:
        print("\a", flush=True)
        try:
            import subprocess
            subprocess.Popen(
                ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def poll_worker(name: str, daemon: str) -> list[dict]:
    """Poll for tasks assigned to me. Returns list of tasks."""
    r = post_json(f"{daemon}/api/god/poll", {"agent": name, "status": "pending"})
    if r is None:
        return []
    task = r.get("task")
    return [task] if task else []


def poll_memories_by_label(daemon: str, agent: str | None, label_prefix: str, last_seen_ids: set[str], project: str = "Mycerise_V2_Taur") -> list[dict]:
    """Fetch recent memories, return those matching label_prefix that are new since last_seen_ids.

    Uses /api/memories which returns label inside metadata.
    """
    params = f"?project={project}&limit=100"
    if agent:
        params += f"&agent={agent}"
    url = f"{daemon}/api/memories{params}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log(f"polling {url}: {e}", level="ERROR")
        return []
    matches: list[dict] = []
    for mem in data.get("memories", []):
        meta = mem.get("metadata", {}) or {}
        if not isinstance(meta, dict):
            continue
        label = meta.get("label", "")
        mem_id = mem.get("memory_id", "")
        if not label.startswith(label_prefix):
            continue
        if mem_id in last_seen_ids:
            continue
        matches.append({"memory_id": mem_id, "label": label, "agent": mem.get("metadata", {}).get("agent") if isinstance(mem.get("metadata"), dict) else None, "content": meta.get("content", "")[:300]})
    return matches


def poll_audit(daemon: str, agent: str | None, last_id: str, label_prefix: str) -> list[dict]:
    """Legacy wrapper kept for compatibility. Use poll_memories_by_label instead."""
    return poll_memories_by_label(daemon, agent, label_prefix, {last_id} if last_id else set())


def run_worker_mode(args: argparse.Namespace) -> int:
    log(f"WORKER MODE — name={args.name} interval={args.interval}s daemon={args.daemon}")
    while True:
        tasks = poll_worker(args.name, args.daemon)
        if tasks:
            for t in tasks:
                log(f"NEW TASK: {t['label']}", level="TASK")
                try:
                    content = json.loads(t["content"]) if isinstance(t["content"], str) else t["content"]
                    log(f"  content preview: {json.dumps(content)[:200]}", level="TASK")
                except Exception:
                    log(f"  content: {str(t['content'])[:200]}", level="TASK")
            beep()
        time.sleep(args.interval)


def run_orchestrator_mode(args: argparse.Namespace) -> int:
    state_path = Path(args.state_file)
    state = load_state(state_path)
    seen: set[str] = set(state.get("seen_ids", []))
    log(f"ORCHESTRATOR MODE - watching god:result:* interval={args.interval}s daemon={args.daemon} project={args.project}")
    while True:
        new_entries = poll_memories_by_label(
            args.daemon,
            agent=None,
            label_prefix="god:result:",
            last_seen_ids=seen,
            project=args.project,
        )
        if new_entries:
            for entry in new_entries:
                log(f"NEW RESULT: {entry['label']}", level="RESULT")
                log(f"  agent: {entry.get('agent')} preview: {entry.get('content','')[:200]}", level="RESULT")
                seen.add(entry["memory_id"])
                state["seen_ids"] = sorted(seen)[-200:]
                save_state(state_path, state)
            beep()
        time.sleep(args.interval)


def run_observer_mode(args: argparse.Namespace) -> int:
    log(f"OBSERVER MODE - watching god:* interval={args.interval}s daemon={args.daemon} project={args.project}")
    seen: set[str] = set()
    while True:
        for prefix in ("god:task:", "god:result:", "god:reply:", "god:reg:", "god:shutdown:"):
            entries = poll_memories_by_label(args.daemon, None, prefix, seen, project=args.project)
            for entry in entries:
                mem_id = entry.get("memory_id", "")
                kind = prefix.rstrip(":")
                log(f"[{kind}] {entry.get('label')} agent={entry.get('agent')} preview={entry.get('content','')[:150]}", level="OBS")
                seen.add(mem_id)
        time.sleep(args.interval)


def main() -> int:
    p = argparse.ArgumentParser(description="MATHIR god-mode bridge daemon")
    p.add_argument("--mode", choices=("worker", "orchestrator", "observer"), required=True)
    p.add_argument("--name", help="Worker name (for worker mode)")
    p.add_argument("--daemon", default=DEFAULT_DAEMON, help=f"MATHIR daemon URL (default: {DEFAULT_DAEMON})")
    p.add_argument("--interval", type=int, default=5, help="Poll interval in seconds (default: 5)")
    p.add_argument("--state-file", default=str(DEFAULT_DB_STATE_FILE), help="State file for last-seen tracking")
    p.add_argument("--project", default="Mycerise_V2_Taur", help="MATHIR project name (for orchestrator/observer modes)")
    args = p.parse_args()

    if args.mode == "worker" and not args.name:
        log("--name required in worker mode", level="ERROR")
        return 2

    try:
        if args.mode == "worker":
            return run_worker_mode(args)
        if args.mode == "orchestrator":
            return run_orchestrator_mode(args)
        return run_observer_mode(args)
    except KeyboardInterrupt:
        log("shutdown (SIGINT)", level="INFO")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
