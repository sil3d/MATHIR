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

DAEMON = "http://127.0.0.1:7338"
TIMEOUT = 3  # seconds — must be fast or Claude Code will skip it


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

    if total == 0 or not context:
        return

    # Print to stdout — Claude Code injects this into the conversation
    print(f"<mathir-auto-injection project=\"{project}\" memories=\"{total}\">")
    print(context)
    print("</mathir-auto-injection>")


if __name__ == "__main__":
    main()
