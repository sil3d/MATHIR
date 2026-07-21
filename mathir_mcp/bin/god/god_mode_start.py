#!/usr/bin/env python3
"""GOD MODE START — on-demand launcher for headless god-mode workers.

This is deliberately NOT wired into autostart. God mode should only start
when a human explicitly asks the agent they're talking to ("go into god
mode" / "coordinate with my other coding agents"). At that point the
orchestrating agent should:
  1. Run `--detect` to see which coding-agent CLIs actually exist on this
     machine (don't assume -- offer only what's really installed). --detect
     is read-only: it only checks PATH, it never executes anything. Nothing
     runs until the human picks tools and --launch is called explicitly.
  2. Ask the human which of the detected tools to use, and how many worker
     processes to launch (usually one per tool, but nothing stops running
     the same tool twice under different --name values for parallel work
     on the same project).
  3. Tell the human that model selection is per-tool: each CLI already has
     its own default model configured (opencode.json, ~/.openclaude/
     settings.json, codex's config.toml, etc.) -- that's the model
     god_worker_daemon.py will use unless told to override. Don't try to
     unify model flags across tools, each has its own convention.
  4. Run `--launch name1,name2,...` for the tools the human picked.

Each launched worker is a detached background process (survives this
script exiting). It loads its own skills/subagents/tools as normal, but
the DECISION of which tools to launch and when always belongs to the human
+ orchestrator, never to a worker deciding on its own. PIDs + metadata are
recorded in a state file so god_mode_stop.py can find and stop them later.

Usage:
    python god_mode_start.py --detect
    python god_mode_start.py --launch opencode,mimo --cwd D:/path/to/project
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Known coding-agent CLIs this launcher can drive headless, and the
# god-worker-daemon --tool value each maps to (see god_worker_daemon.py's
# TOOL_REGISTRY -- each tool name here IS its own --tool choice already,
# kept in sync manually since the two scripts are launched separately).
# Popularity/coverage researched 2026-07-16 (see god_worker_daemon.py's
# module docstring for sources). Amazon Q Developer CLI is deliberately
# excluded -- its open-source repo states it's no longer maintained,
# superseded by the closed-source Kiro CLI.
KNOWN_TOOLS = [
    "opencode", "mimo", "claude", "openclaude",
    "codex", "gemini", "aider", "cursor-agent", "copilot",
]

_STATE_DIR = Path(os.environ.get(
    "MATHIR_HOME", str(Path.home() / ".config" / "MATHIR")
)) / "logs"
STATE_FILE = _STATE_DIR / "god_mode_workers.json"

THIS_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = THIS_DIR / "god_worker_daemon.py"


def which(name: str) -> str | None:
    from shutil import which as _which
    return _which(name)


def detect() -> list[dict]:
    found = []
    for tool in KNOWN_TOOLS:
        path = which(tool)
        if path:
            found.append({"tool": tool, "path": path})
    return found


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"workers": []}
    return {"workers": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def launch_worker(tool: str, name: str, cwd: str, project: str | None, model: str | None) -> int:
    cmd = [
        sys.executable, str(WORKER_SCRIPT),
        "--tool", tool, "--name", name, "--cwd", cwd,
    ]
    if project:
        cmd += ["--project", project]
    if model:
        cmd += ["--model", model]

    log_path = _STATE_DIR / f"god_worker_{name}.log"
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, **kwargs
    )
    return proc.pid


def main() -> int:
    ap = argparse.ArgumentParser(description="MATHIR god-mode on-demand worker launcher")
    ap.add_argument("--detect", action="store_true", help="List installed coding-agent CLIs as JSON")
    ap.add_argument("--launch", default=None, help="Comma-separated tool names to launch (from --detect)")
    ap.add_argument("--cwd", default=None, help="Working directory (project root) for launched workers")
    ap.add_argument("--project", default=None)
    ap.add_argument("--model", default=None, help="Override model for ALL launched workers (rarely needed -- see module docstring)")
    ap.add_argument("--name-suffix", default="", help="Append to each worker's god-mode name (e.g. for a second parallel instance)")
    args = ap.parse_args()

    if args.detect:
        print(json.dumps({"detected": detect()}, indent=2))
        return 0

    if not args.launch:
        ap.error("pass --detect or --launch")

    if not args.cwd:
        ap.error("--launch requires --cwd")

    detected_names = {d["tool"] for d in detect()}
    requested = [t.strip() for t in args.launch.split(",") if t.strip()]
    unknown = [t for t in requested if t not in KNOWN_TOOLS]
    if unknown:
        print(json.dumps({"error": f"unknown tool(s): {unknown}. Known: {KNOWN_TOOLS}"}))
        return 2
    missing = [t for t in requested if t not in detected_names]
    if missing:
        print(json.dumps({"error": f"not installed on this machine: {missing}"}))
        return 2

    state = load_state()
    launched = []
    for tool in requested:
        name = f"{tool}{args.name_suffix}"
        pid = launch_worker(tool, name, args.cwd, args.project, args.model)
        entry = {
            "tool": tool, "name": name, "pid": pid, "cwd": args.cwd,
            "project": args.project, "started_at": datetime.now(timezone.utc).isoformat(),
        }
        state["workers"].append(entry)
        launched.append(entry)

    save_state(state)
    print(json.dumps({"launched": launched, "state_file": str(STATE_FILE)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
