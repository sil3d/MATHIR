#!/usr/bin/env python3
"""GOD WORKER DAEMON — headless polling + execution for MATHIR god-mode.

god_bridge.py (this directory) only detects and beeps on new tasks; it never
executes anything, so a human still has to sit in the chat window and type
"poll for tasks" after every single task. That defeats the point of
autonomous multi-agent orchestration (see 2026-07-16 god-mode session:
passive relay hooks and interactive chat only ever advance on a human-typed
turn — there is no background timer inside any of these CLIs).

This script closes that gap: it polls /api/god/poll directly (no LLM call
wasted on empty polls) and, once a task is claimed, spawns the target
coding-agent CLI in non-interactive/headless mode to actually run it. The
spawned agent is instructed to report its own result via memory_save and
this script just loops to the next poll — no human needed after startup.

WHO EXECUTES: this script and the human never execute the task themselves.
The claimed worker CLI does, in its own process, with its own tools/skills/
subagents. This script's only job is poll -> claim -> spawn -> verify -> loop.
The orchestrator (a live LLM session, not this script) is what decides WHICH
tools to launch in the first place (see god_mode_start.py --detect, which is
read-only and launches nothing on its own) and what to do if a worker goes
silent (see the "WORKER SILENCE" section of mathir_god_orchestre's built-in
guide in mathir_mcp_server.py).

SAFETY: running unattended requires bypassing each tool's interactive
approval prompts, since there is no human to click "allow" -- see
TOOL_REGISTRY below for the specific flag each CLI needs. Only point this at
a project/cwd you trust the assigned tasks to run in. Start with
--max-tasks 1 to test before leaving it running indefinitely.

KNOWN FOOTGUN (verified against real bug reports, not assumed): OpenCode/
Mimo hang forever on ANY bash call if their config's "permission.bash" is
left at the default "ask" -- the confirmation prompt creates a Promise that
never resolves headlessly (github.com/anomalyco/opencode issues #14473,
#3503). --auto does NOT cover this; you additionally need an explicit
"permission": {"*": "allow"} (or at least "bash": "allow") in that tool's
own opencode.json/mimocode.json. This script cannot fix that from the
outside -- it's the target CLI's own config file. Verified in this session:
after setting permission to the correct object-form schema, a real bash
call resolved via `action.action=allow` immediately, with zero hang.

MODEL: this script does not force a model by default -- each CLI's own
configured default model is used (whatever the user picked in that tool's
own config). Pass --model only to override it; don't try to standardize
model selection across tools here, each has its own flag/config convention.

Sources for the flags below (fetched 2026-07-16, not guessed):
  - opencode/mimo: opencode.ai/docs (opencode-family CLIs, `run --auto`)
  - claude/openclaude: code.claude.com/docs/en/headless (`-p` + skip-permissions)
  - codex: developers.openai.com/codex/{noninteractive,cli/reference}
  - gemini: geminicli.com/docs/cli/{headless,cli-reference}
  - aider: aider.chat/docs/scripting.html
  - cursor-agent: cursor.com/docs/cli/{headless,using} (binary is
    "cursor-agent", NOT "agent" -- docs examples alias it, verified against
    praison.ai's install guide before trusting it)
  - copilot: docs.github.com/en/copilot/reference/copilot-cli-reference
  - Amazon Q Developer CLI deliberately excluded: the open-source repo
    (github.com/aws/amazon-q-developer-cli) states it "is no longer being
    actively maintained" and was superseded by the closed-source Kiro CLI.

Usage:
    python god_worker_daemon.py --tool opencode --name opencode --cwd D:/path/to/project
    python god_worker_daemon.py --tool codex    --name codex    --cwd D:/path/to/project
    python god_worker_daemon.py --tool gemini   --name gemini   --cwd D:/path/to/project
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Windows redirects this process's own stdout to a log file via a raw file
# descriptor (see god_mode_start.py's launch_worker), which makes Python
# fall back to the system codepage (cp1252 here) instead of UTF-8 -- verified
# live, 2026-07-21: streaming a real model's stdout (which contains gear/
# checkmark glyphs, em-dashes, etc.) crashed the _pump reader thread on the
# very first non-ASCII character with UnicodeEncodeError, silently killing
# all further live logging for that stream (the thread just dies; nothing
# else notices). Force UTF-8 with a safe fallback on our own stdout/stderr
# so log() can never crash on the content it's asked to print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DAEMON = os.environ.get("MATHIR_DAEMON_URL", "http://127.0.0.1:7338")
# Default lowered from an earlier 1800s (30 min) to 300s (5 min): live-tested
# 2026-07-16 that openclaude's `-p` headless mode can hang the process alive
# (never closes stdout/stderr, so subprocess.run() blocks) either mid-task or
# after finishing, with no god:result ever saved. Successful *trivial* runs
# (single poll+ack probes) completed in 20-90s.
#
# Raised to 450s on 2026-07-21 after a real, substantial task (explore the
# codebase + write a prioritized analysis) genuinely took ~5m04s on
# opencode/glm-5.2 -- it spawns explore subagents (each 1-3 min) for deep
# code search, which a trivial probe never does. The earlier 240s/300s
# timeouts were killing these runs seconds before they would have finished
# and saved a real result -- this was miscalibration, not a hang or a
# reliability problem with opencode itself (confirmed by re-running the
# exact same task manually with a long timeout: it completed cleanly).
# Pick --timeout deliberately based on task size, don't just accept this
# default for anything beyond a trivial probe.
TASK_TIMEOUT_SECONDS = int(os.environ.get("MATHIR_GOD_TASK_TIMEOUT", "450"))


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [{level}] {msg}", flush=True)


def post_json(url: str, payload: dict, timeout: int = 10) -> dict | None:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        log(f"HTTP error on {url}: {e}", level="ERROR")
        return None


def result_was_saved(daemon: str, project: str, task_id: str) -> bool:
    """Exit code 0 alone doesn't prove the task was actually completed --
    a headless run can exit cleanly after doing nothing useful (observed:
    the model finished its turn without ever calling memory_save). Check
    for the real god:result label the worker was told to save.
    """
    try:
        url = f"{daemon}/api/memories?project={project}&limit=50"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log(f"result check failed: {e}", level="ERROR")
        return False
    needle = f"god:result:{task_id}:"
    for mem in data.get("memories", []):
        meta = mem.get("metadata", {}) or {}
        if isinstance(meta, dict) and str(meta.get("label", "")).startswith(needle):
            return True
    return False


def poll_task(daemon: str, agent: str, project: str, cwd: str) -> dict | None:
    r = post_json(f"{daemon}/api/god/poll", {
        "agent": agent, "status": "pending", "project": project, "cwd": cwd,
    })
    return r.get("task") if r else None


def ack_task(daemon: str, memory_id: str, status: str, project: str, cwd: str) -> None:
    post_json(f"{daemon}/api/god/ack", {
        "memory_id": memory_id, "status": status, "project": project, "cwd": cwd,
    })


def build_prompt(task_content: str, task_id: str, name: str) -> str:
    try:
        info = json.loads(task_content)
        desc = info.get("description", task_content)
    except (json.JSONDecodeError, TypeError):
        desc = task_content
    return (
        f'You are the MATHIR god-mode worker "{name}", running headless — no human is '
        f"watching this terminal, so you must finish the whole task and report the result "
        f"yourself; no one will prompt you again. This means: NEVER ask a clarifying question "
        f"or wait for confirmation before proceeding to the next tool call.\n\n"
        f"YOU ARE ALREADY DISPATCHED -- do NOT call mathir_god_agent() and do NOT save any "
        f"god:reg:* self-registration memory. That flow is only for workers that poll for "
        f"tasks themselves; you were handed this task directly in this prompt. Calling "
        f"mathir_god_agent() here does nothing useful and wastes your time budget -- verified "
        f"live, 2026-07-21: 3 workers given this exact instruction wrong burned their entire "
        f"timeout re-registering in a loop and never reached the actual task. Skip straight to "
        f"the MATHIR memory checks below, then EXECUTE THIS TASK directly.\n\n"
        f"MATHIR is your memory, not an optional tool. Before doing ANY work, check MATHIR "
        f"(memory_smart_search / memory_recall) for whether this or an equivalent task was "
        f"already completed -- never redo work MATHIR already has a record of, and never start "
        f"blind. After finishing, save your result to MATHIR. Follow every MATHIR guardrail "
        f"returned in your context at all times; they are not optional and must never be "
        f"worked around. If something you observe looks unexpected or suspicious mid-task, "
        f"stop and think before continuing -- don't push through blindly. If you hit a "
        f"pre-existing bug unrelated to your task, don't ignore it: verify its real root cause "
        f"first, then fix it. Skipping MATHIR means working blind and producing low-quality "
        f"work -- not acceptable. Write senior-level, production-grade code (Anthropic/OpenAI/"
        f"NASA quality) -- never stubs, never placeholders, never code that merely compiles.\n\n"
        f"EXECUTE THIS TASK: {desc}\n\n"
        f"When done, call memory_save with label='god:result:{task_id}:orchestrator:completed', "
        f"content=a JSON summary of what you did, block_type='episodic', priority=7."
    )


def _kill_tree(pid: int) -> None:
    """subprocess.run(shell=True)'s built-in timeout kill only kills the
    cmd.exe shell it spawned, not the real child (node.exe) that shell
    launched -- verified live (2026-07-16): after a 5-min timeout fired,
    the outer python process was gone but the openclaude node.exe processes
    were still running as orphans, un-killed. taskkill /T kills the whole
    tree rooted at the shell's PID, not just the shell itself.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _run(cmd: list[str], cwd: str, timeout_seconds: int, extra_env: dict[str, str] | None = None, task_id: str = "") -> subprocess.CompletedProcess:
    # On Windows, npm-installed CLIs (opencode, mimo, openclaude) are .cmd
    # shims, not .exe -- subprocess.run(["opencode", ...]) fails with
    # WinError 2 because CreateProcess can't exec a .cmd directly; it needs
    # to go through cmd.exe (shell=True) to resolve PATHEXT. Native .exe
    # binaries (claude.exe, codex, etc.) work fine either way.
    #
    # Popen (not subprocess.run) is used deliberately so timeout handling
    # can explicitly kill the whole process tree (see _kill_tree) instead
    # of relying on subprocess.run's default single-process kill, which is
    # provably insufficient under shell=True (see _kill_tree docstring).
    env = {**os.environ, **extra_env} if extra_env else None
    is_win = sys.platform == "win32"
    # CREATE_NO_WINDOW suppresses the visible console window shell=True would
    # otherwise pop up for every headless worker invocation (reported live,
    # 2026-07-16 -- this daemon is meant to run unattended in the
    # background, a window stealing focus every poll cycle defeats that).
    win_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if is_win else 0
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        # Windows consoles default to the system codepage (cp1252 on this
        # machine), which can't decode arbitrary model output (em-dashes,
        # smart quotes, non-Latin text) -- verified live, 2026-07-21: a
        # MiniMax-M3 run crashed the internal pipe-reader thread with
        # UnicodeDecodeError on byte 0x9d. Force UTF-8 with a safe fallback
        # instead of raising on the first character it can't map.
        encoding="utf-8", errors="replace",
        env=env, shell=is_win,
        creationflags=win_flags,
        start_new_session=not is_win,
    )

    # proc.communicate() blocks until exit and only hands back output at the
    # very end -- verified live, 2026-07-21: the human running `Get-Content
    # -Wait` on this daemon's log saw nothing between "CLAIMED" and the final
    # result/error for several minutes on a real task, because nothing was
    # written until completion. Stream each line to our own logger (which the
    # launcher already redirects to a per-worker log file) as it's produced,
    # so a live `tail -f` shows the model's actual tool calls/output in real
    # time instead of going dark mid-task.
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    prefix = f"[{task_id}] " if task_id else ""

    def _pump(stream, sink: list[str], tag: str) -> None:
        for line in iter(stream.readline, ""):
            if not line:
                break
            sink.append(line)
            log(f"{prefix}{tag}: {line.rstrip()}")
        stream.close()

    import threading
    t_out = threading.Thread(target=_pump, args=(proc.stdout, stdout_lines, "stdout"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, stderr_lines, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        t_out.join(timeout=10)
        t_err.join(timeout=10)
        raise
    t_out.join(timeout=10)
    t_err.join(timeout=10)
    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(stdout_lines), "".join(stderr_lines))


# ── Per-tool headless command builders ──────────────────────────────────
# Each entry returns the full argv (tool name included). Verified against
# each project's own docs (see module docstring for sources) -- no flag
# here is guessed.

def _cmd_opencode_family(tool: str, prompt: str, model: str | None, agent_profile: str | None) -> list[str]:
    cmd = [tool, "run", "--auto"]
    if model:
        cmd += ["-m", model]
    if agent_profile:
        cmd += ["--agent", agent_profile]
    cmd += [prompt]
    return cmd


def _cmd_claude_family(tool: str, prompt: str, model: str | None) -> list[str]:
    cmd = [tool, "-p", prompt, "--dangerously-skip-permissions"]
    if model:
        cmd += ["--model", model]
    return cmd


def _cmd_codex(prompt: str, model: str | None) -> list[str]:
    cmd = ["codex", "exec", "--sandbox", "workspace-write", "--skip-git-repo-check"]
    if model:
        cmd += ["-m", model]
    cmd += [prompt]
    return cmd


def _cmd_gemini(prompt: str, model: str | None) -> list[str]:
    cmd = ["gemini", "-p", prompt, "--approval-mode=yolo"]
    if model:
        cmd += ["-m", model]
    return cmd


def _cmd_aider(prompt: str, model: str | None) -> list[str]:
    cmd = ["aider", "--yes-always", "--message", prompt]
    if model:
        cmd += ["--model", model]
    return cmd


def _cmd_cursor_agent(prompt: str, model: str | None) -> list[str]:
    # cursor-agent has no documented model-override flag (github.com/
    # cursor.com/docs/cli/using, checked 2026-07-16) -- it uses whatever
    # model is configured in the Cursor app/CLI itself. Silently ignoring
    # `model` here (rather than passing an unverified flag) is deliberate.
    return ["cursor-agent", "-p", "--force", prompt]


def _cmd_copilot(prompt: str, model: str | None) -> list[str]:
    cmd = ["copilot", "-p", prompt, "--allow-all-tools", "-s"]
    if model:
        cmd += [f"--model={model}"]
    return cmd


# Per-tool env var overrides for hang mitigation. openclaude is built on
# openclaw, which has two documented hang bugs (github.com/openclaw/openclaw
# #8288, #12904, checked 2026-07-16): tool-call failures can wait up to the
# 600s agent-level default before erroring, and OPENCLAUDE_QUERY_HARD_MAX_MS
# (the overall per-query hard cap) defaults to 1800000ms (30 min) -- which
# matches the multi-minute hangs observed live in this session. Lowering it
# here makes openclaude self-terminate a stuck query well before our own
# subprocess timeout would have to SIGKILL the whole process from outside.
TOOL_ENV_OVERRIDES: dict[str, dict[str, str]] = {
    "openclaude": {"OPENCLAUDE_QUERY_HARD_MAX_MS": "120000"},  # 2 min
}


# tool name -> (builder, needs_agent_profile)
TOOL_REGISTRY: dict[str, callable] = {
    "opencode": lambda prompt, model, agent_profile: _cmd_opencode_family("opencode", prompt, model, agent_profile),
    "mimo": lambda prompt, model, agent_profile: _cmd_opencode_family("mimo", prompt, model, agent_profile),
    "claude": lambda prompt, model, agent_profile: _cmd_claude_family("claude", prompt, model),
    "openclaude": lambda prompt, model, agent_profile: _cmd_claude_family("openclaude", prompt, model),
    "codex": lambda prompt, model, agent_profile: _cmd_codex(prompt, model),
    "gemini": lambda prompt, model, agent_profile: _cmd_gemini(prompt, model),
    "aider": lambda prompt, model, agent_profile: _cmd_aider(prompt, model),
    "cursor-agent": lambda prompt, model, agent_profile: _cmd_cursor_agent(prompt, model),
    "copilot": lambda prompt, model, agent_profile: _cmd_copilot(prompt, model),
}


def _execute_with_retries(
    cmd: list[str], cwd: str, daemon: str, project: str, task_id: str,
    max_retries: int, timeout_seconds: int, extra_env: dict[str, str] | None = None,
) -> bool:
    """Run cmd up to (1 + max_retries) times, stopping at the first attempt
    that actually saves a god:result. Two distinct failure modes observed
    live (2026-07-16, openclaude), both worth retrying rather than giving up
    immediately:
      1. Exits cleanly (code 0) having done nothing useful -- model
         non-determinism on multi-step tool flows, not a deterministic bug.
      2. Never exits at all (hangs, no god:result ever saved) -- caught by
         `timeout_seconds` instead of blocking indefinitely.
    """
    attempts = max(1, max_retries + 1)
    for attempt in range(1, attempts + 1):
        suffix = "" if attempts == 1 else f" (attempt {attempt}/{attempts})"
        try:
            result = _run(cmd, cwd, timeout_seconds, extra_env, task_id=task_id)
            if result.returncode != 0:
                log(f"task {task_id} exited {result.returncode}{suffix}: {result.stderr[-500:]}", level="ERROR")
            elif not result_was_saved(daemon, project, task_id):
                log(f"task {task_id} exited 0 but no god:result was saved{suffix} -- retrying" if attempt < attempts
                    else f"task {task_id} exited 0 but no god:result was saved{suffix} -- giving up", level="ERROR")
            else:
                log(f"task {task_id} finished{suffix}")
                return True
        except subprocess.TimeoutExpired:
            log(f"task {task_id} timed out after {timeout_seconds}s{suffix}", level="ERROR")
        except Exception as e:
            log(f"task {task_id} crashed: {e}{suffix}", level="ERROR")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="MATHIR god-mode headless worker daemon")
    ap.add_argument("--tool", choices=sorted(TOOL_REGISTRY), required=True)
    ap.add_argument("--name", required=True, help="God-mode agent name to poll/claim under")
    ap.add_argument("--cwd", required=True, help="Working directory (project root)")
    ap.add_argument("--project", default=None, help="MATHIR project name (defaults to basename of --cwd)")
    ap.add_argument("--daemon", default=DEFAULT_DAEMON)
    ap.add_argument("--interval", type=int, default=10, help="Seconds between empty polls")
    ap.add_argument("--model", default=None, help="Override this worker's default model (rarely needed -- see module docstring)")
    ap.add_argument("--agent-profile", default=None, help="--agent value for opencode-family tools (opencode/mimo only)")
    ap.add_argument("--max-tasks", type=int, default=0, help="Stop after N tasks (0 = run forever)")
    ap.add_argument("--retries", type=int, default=2, help="Extra attempts on failure/silent-no-op before giving up (default 2, so 3 attempts total)")
    ap.add_argument("--timeout", type=int, default=TASK_TIMEOUT_SECONDS, help=f"Seconds before killing a hung attempt (default {TASK_TIMEOUT_SECONDS})")
    args = ap.parse_args()

    project = args.project or Path(args.cwd).name
    log(
        f"WORKER DAEMON tool={args.tool} name={args.name} cwd={args.cwd} "
        f"project={project} interval={args.interval}s max_tasks={args.max_tasks or 'inf'}"
    )

    build_cmd = TOOL_REGISTRY[args.tool]
    done = 0
    try:
        while True:
            task = poll_task(args.daemon, args.name, project, args.cwd)
            if not task or not task.get("memory_id"):
                time.sleep(args.interval)
                continue

            memory_id = task["memory_id"]
            label = task.get("label", "")
            parts = label.split(":")
            task_id = parts[2] if len(parts) == 5 else "unknown"

            log(f"CLAIMED {label}")
            ack_task(args.daemon, memory_id, "running", project, args.cwd)
            prompt = build_prompt(task.get("content", ""), task_id, args.name)
            cmd = build_cmd(prompt, args.model, args.agent_profile)

            succeeded = _execute_with_retries(
                cmd, args.cwd, args.daemon, project, task_id, args.retries, args.timeout,
                TOOL_ENV_OVERRIDES.get(args.tool),
            )
            ack_task(args.daemon, memory_id, "completed" if succeeded else "failed", project, args.cwd)

            done += 1
            if args.max_tasks and done >= args.max_tasks:
                log(f"max-tasks ({args.max_tasks}) reached, exiting")
                return 0
    except KeyboardInterrupt:
        log("shutdown (SIGINT)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
