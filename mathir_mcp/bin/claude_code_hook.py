#!/usr/bin/env python3
"""MATHIR Tier-A auto-injection hook for Claude Code.

Called by Claude Code's UserPromptSubmit hook BEFORE the model sees
the user's message.  Queries the MATHIR daemon at 127.0.0.1:7338
and prints relevant memories to stdout.  Claude Code captures stdout
and injects it as a <user-prompt-submit-hook> block — the model
cannot ignore it.

Usage in .claude/settings.json (project or global):
  {
    "hooks": {
      "UserPromptSubmit": [{
        "type": "command",
        "command": "python <path>/claude_code_hook.py"
      }]
    }
  }

The hook reads the user's message from stdin (JSON with a "message"
field) and uses it as the MATHIR query context.
"""

import json
import sys
import urllib.request
import urllib.error
import os
from pathlib import Path

# Windows consoles/pipes can default to a non-UTF-8 codepage (cp1252),
# which raises UnicodeEncodeError on any non-Latin-1 character (arrows,
# smart quotes, non-ASCII names in recalled memory content, etc.) and
# silently kills the whole hook -- Claude Code just sees no output rather
# than a real error. Force UTF-8 on stdout with a safe fallback instead of
# crashing on the first character it can't map.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DAEMON = "http://127.0.0.1:7338"
TIMEOUT = 3  # seconds — must be fast or Claude Code will skip it

# God Mode cross-agent relay (v8.9.4+): without this, a pending god:task
# message only ever surfaces if the agent actively calls mathir_god_agent()
# -- which nothing prompts it to do, so the human ends up manually copying
# "X said Y" between agent windows. This hook already runs on every single
# turn, so piggybacking a poll here makes every agent check automatically,
# with no separate always-running process required.
GOD_AGENT_NAME = os.environ.get("MATHIR_GOD_AGENT_NAME", "claude-code")
_SEEN_TASKS_PATH = Path(os.environ.get(
    "MATHIR_HOME", str(Path.home() / ".config" / "MATHIR")
)) / "logs" / f"god_seen_{GOD_AGENT_NAME}.json"
_SEEN_TASKS_MAX = 200  # cap so the file can't grow unbounded

try:
    from mathir_sanitize import sanitize_block as _sanitize_god_block
except ImportError:
    def _sanitize_god_block(text, max_bytes=8192):
        # Fallback if mathir_sanitize.py isn't importable from this path --
        # same threat model as the memory-injection block below, so still
        # strip the one breakout token that matters here.
        return (text or "").replace("</mathir-", "")[:max_bytes]


def _load_seen_tasks() -> set:
    try:
        return set(json.loads(_SEEN_TASKS_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _save_seen_tasks(seen: set) -> None:
    try:
        _SEEN_TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Keep only the most recent N (sets are unordered, but this is just
        # a soft cap against unbounded growth, not a strict LRU).
        trimmed = list(seen)[-_SEEN_TASKS_MAX:]
        _SEEN_TASKS_PATH.write_text(json.dumps(trimmed), encoding="utf-8")
    except Exception:
        pass  # never let dedup bookkeeping block the hook


def _check_god_relay(project: str, cwd: str) -> str:
    """Poll for a pending god:task addressed to this agent; return a
    formatted, sanitized block if there's a new one, else "".
    """
    try:
        payload = json.dumps({
            "agent": GOD_AGENT_NAME, "status": "pending",
            "project": project, "cwd": cwd,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{DAEMON}/api/god/poll", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""

    task = data.get("task")
    if not task:
        return ""

    task_id = task.get("memory_id", "")
    seen = _load_seen_tasks()
    if not task_id or task_id in seen:
        return ""  # already surfaced this one -- don't spam every turn

    content = _sanitize_god_block(task.get("content", ""))
    if not content:
        return ""

    # Ack server-side FIRST: /api/god/poll always returns the single oldest
    # still-pending task, so without this the same message blocks every
    # queued task behind it forever (client-side dedup alone isn't enough
    # -- it can only skip re-showing what it already saw, it can't make the
    # server hand back the *next* one). If the ack call fails, fall back to
    # the local seen-set alone so this turn doesn't loop, but the queue may
    # stay stuck until the daemon is reachable again.
    try:
        ack_payload = json.dumps({
            "memory_id": task_id, "status": "delivered",
            "project": project, "cwd": cwd,
        }).encode("utf-8")
        ack_req = urllib.request.Request(
            f"{DAEMON}/api/god/ack", data=ack_payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(ack_req, timeout=TIMEOUT).read()
    except Exception:
        pass

    seen.add(task_id)
    _save_seen_tasks(seen)

    label = task.get("label", "")
    return (
        f'<mathir-god-message to="{GOD_AGENT_NAME}" label="{label}">\n'
        f"A message is waiting from another MATHIR God Mode agent:\n"
        f"{content}\n"
        f"</mathir-god-message>"
    )


def main():
    # Read the hook input from stdin
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        hook_input = {}

    # Claude Code sends the field as "prompt"; keep "message" as a fallback
    # for other harnesses (Codex, OpenClaude) that may wire this same script
    # into their own UserPromptSubmit-equivalent hook with a different key.
    user_message = hook_input.get("prompt") or hook_input.get("message", "")
    if not user_message:
        return

    # Detect project from CWD
    cwd = os.getcwd()
    project = os.path.basename(cwd)

    # Query MATHIR /api/context
    try:
        payload = json.dumps({
            "task": user_message[:500],
            "k": 8,
            "project": project,
            "cwd": cwd,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{DAEMON}/api/context",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError):
        # Daemon not running — silent fail, don't block the user
        return
    except Exception:
        return

    context = data.get("context", "")
    total = data.get("total", 0)

    # God Mode relay check runs independently of memory-context results --
    # a pending cross-agent message should surface even on a turn where
    # nothing matched semantically.
    god_block = _check_god_relay(project, cwd)
    if god_block:
        print(god_block)

    if total == 0 or not context:
        return

    # Print to stdout — Claude Code injects this into the conversation
    print(f"<mathir-auto-injection project=\"{project}\" memories=\"{total}\">")
    print(context)
    print("</mathir-auto-injection>")


if __name__ == "__main__":
    main()
