# MATHIR God Mode — Multi-Agent Orchestration

## The Problem

You have multiple AI agents (Claude Code, MiMo, Codex, OpenCode, Kilo...) — each in its own terminal, each with its own strengths. But they don't talk to each other. You're the bottleneck: copying context between terminals, deciding who does what, checking results manually.

**What if your agents could coordinate themselves?**

## The Solution

**MATHIR God Mode** turns MATHIR's shared memory into a **cross-process message queue**. One agent becomes the **orchestrator** (the brain), the others become **workers** (the hands). The orchestrator decomposes a directive, assigns tasks based on each worker's strengths, monitors progress, and verifies results.

No new infrastructure. No message broker. No API gateway. Just MATHIR — the memory layer your agents already use.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MATHIR DAEMON (port 7338)                     │
│                    Shared Memory = Message Queue                 │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ god:reg   │  │ god:task  │  │ god:task  │  │ god:result│       │
│  │ mimo:idle │  │ t1:mimo   │  │ t2:codex  │  │ t1:done   │       │
│  │           │  │ :pending  │  │ :pending  │  │           │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────┬───────────────┬───────────────────────────┘
                      │               │
         ┌────────────┤               ├────────────┐
         │            │               │            │
    ┌─────────┐  ┌─────────┐    ┌─────────┐  ┌─────────┐
    │ Worker  │  │ Worker  │    │ Worker  │  │ Orchest.│
    │ (MiMo)  │  │ (Codex) │    │(OpenCode)│  │ (Claude)│
    │ Terminal│  │ Terminal│    │ Terminal│  │ Terminal│
    │   1     │  │   2     │    │   3     │  │   4     │
    └─────────┘  └─────────┘    └─────────┘  └─────────┘
```

---

## How It Works (3 Steps)

### Step 1 — Workers Self-Identify

Open each worker terminal and call `mathir_god_agent()` with **no arguments**.

The tool asks the agent to introduce itself honestly:

```
→ mathir_god_agent()

← "SELF-IDENTIFICATION REQUIRED"
   Tell me: what agent are you? Your strengths? Weaknesses? Tools?
```

The agent self-assesses and calls back:

```
→ mathir_god_agent(
    name="mimo",
    capabilities="code,test,fast-execution",
    introduction="I am MiMo-7B, a lightweight reasoning model. I excel at
    fast code generation and mechanical tasks. I'm weaker on complex
    multi-file architecture. I have file system access and can run tests."
  )

← "Registered. Polling for tasks..."
```

Each agent identifies itself naturally — **you never tell them who they are**.

### Step 2 — Orchestrator Assigns

In the orchestrator terminal (typically your strongest reasoning model):

```
→ mathir_god_orchestre(directive="Refactor auth module, write tests, update docs")
```

The orchestrator receives:
- Every worker's **full profile** (name, capabilities, self-assessment)
- The **directive** to decompose
- **Assignment principles** (match strength to task, don't waste deep reasoners on simple work)

The orchestrator then:
1. Decomposes the directive into concrete tasks
2. Reads each worker's self-assessment
3. Assigns tasks to the **best-suited worker** for each task
4. Dispatches via `memory_save`

Example assignment logic:
```
Task: "Refactor auth.py — extract validate_token()" 
  → mimo (said "I excel at fast code generation")

Task: "Design new auth architecture across 5 files"
  → claude-code (said "I excel at complex multi-file architecture")

Task: "Write 20 unit tests for auth module"
  → codex (said "I'm fast at bulk mechanical work")
```

### Step 3 — Workers Execute & Report

Workers poll via `mathir_god_agent()` in a loop. When a task arrives:

1. Worker receives the task description
2. Executes using its own tools (Edit, Bash, etc.)
3. Reports results via `memory_save`
4. Polls for the next task

The orchestrator monitors results, verifies quality, and dispatches dependent tasks.

---

## Label Protocol

All coordination uses structured labels in MATHIR memories:

```
god:{type}:{id}:{target}:{status}
```

| Label | Meaning |
|---|---|
| `god:reg:mimo:mimo:idle` | Worker "mimo" registered and waiting |
| `god:task:a1b2c3d4:mimo:pending` | Task dispatched to mimo |
| `god:task:a1b2c3d4:mimo:running` | Mimo accepted the task |
| `god:result:a1b2c3d4:orchestrator:completed` | Mimo finished the task |
| `god:task:00000000:mimo:shutdown` | Tell mimo to stop |

---

## MCP Tools

### `mathir_god_agent` — Worker Tool

| Call | Effect |
|---|---|
| `mathir_god_agent()` | Self-identification prompt |
| `mathir_god_agent(name="help")` | Full usage guide |
| `mathir_god_agent(name="mimo", capabilities="code,test", introduction="...")` | Register + poll |

Returns: `identify`, `waiting`, `task_found`, or `shutdown`.

### `mathir_god_orchestre` — Orchestrator Tool

| Call | Effect |
|---|---|
| `mathir_god_orchestre(directive="help")` | Full usage guide |
| `mathir_god_orchestre(directive="Refactor auth + tests")` | Discover workers, get assignment instructions |

Parameters:
- `strategy`: `"auto"` (default), `"parallel"`, `"sequential"`
- `verify`: Review results before marking verified (default `True`)
- `auto_merge`: Merge git branches without asking (default `False`)

---

## Daemon Routes

Two HTTP endpoints added to the MATHIR daemon:

| Route | Method | Purpose |
|---|---|---|
| `/api/god/poll` | POST | Query pending tasks for a specific worker. Atomically claims the task (`BEGIN IMMEDIATE` around the SELECT+UPDATE) so two pollers sharing a name can't double-claim it. |
| `/api/god/ack` | POST | Flip a claimed task's label in place (`pending`→`running`→`completed`/`failed`) instead of creating a duplicate memory per state change |
| `/api/god/agents` | GET/POST | List registered workers with profiles |

---

## Client-side tooling (`bin/god/`)

The MCP tools above are designed for single-turn agent sessions — they return immediately, they can't loop forever. For **long-running polling** (a worker waiting for tasks across many turns, or an orchestrator watching for results), use the standalone bridge daemon shipped in [`mathir_mcp/bin/god/`](../mathir_mcp/bin/god/).

### Why a separate client bridge?

| Need | MCP tool | `god_bridge.py` |
|---|---|---|
| Run inside an agent's tool loop | ✅ | ❌ (external process) |
| Block until a task arrives | ❌ (returns immediately) | ✅ (polls every N seconds) |
| Cross-platform without dependencies | ✅ (uses existing MCP) | ✅ (stdlib only) |
| Notify on new events (beep + log) | ❌ | ✅ |

### Modes

| Mode | What it does | When to use |
|---|---|---|
| `worker` | Polls `/api/god/poll` for tasks dispatched to `--name <me>` | Each worker terminal |
| `orchestrator` | Watches `/api/memories` for new `god:result:*` entries | Orchestrator terminal |
| `observer` | Logs every `god:*` event | Debug / monitoring |

### Quick start

```bash
# Worker terminal
python bin/god/god_bridge.py --mode worker --name mimo-code --interval 5

# Orchestrator terminal (separate)
python bin/god/god_bridge.py --mode orchestrator --interval 5 --project <your-project>

# Observer
python bin/god/god_bridge.py --mode observer --interval 10
```

### Cross-platform

| Shell | Pollers |
|---|---|
| Python (cross-platform) | `god_bridge.py` |
| Windows PowerShell | `god_poll.ps1` |
| POSIX bash | `god_poll.sh` |

Env vars (override per machine, no hardcoded paths):

| Var | Default | Purpose |
|---|---|---|
| `MATHIR_DAEMON_URL` | `http://localhost:7338` | Daemon URL |
| `MATHIR_STATE_DIR` | `$XDG_CONFIG_HOME/mathir` | State + log dir |
| `MATHIR_LOG_FILE` | derived from `MATHIR_STATE_DIR` | Log file path |

Full spec: [`bin/god/PROTOCOL.md`](../mathir_mcp/bin/god/PROTOCOL.md) · Usage: [`bin/god/README.md`](../mathir_mcp/bin/god/README.md).

### Headless, unattended workers (`god_mode_start.py` / `god_worker_daemon.py`)

`god_bridge.py` only notifies — a human still has to see the beep and drive the agent. For fully unattended execution, `god_mode_start.py --launch <tool> --name <n> --cwd <path>` spawns a detached `god_worker_daemon.py` process that does the whole loop itself: poll `/api/god/poll` → `ack` to `running` → spawn the target CLI headlessly with its documented flags (e.g. `opencode run --auto`, `claude -p --dangerously-skip-permissions`, `codex exec --sandbox workspace-write`) → stream output live to a log file → retry on a silent no-op (process exits 0 but never called `memory_save`) or a hang past its timeout → `ack` `completed`/`failed`. Stop it with `god_mode_stop.py --name <n>` or `--all`.

Because relying on the orchestrator LLM's own memory to relay every worker's result across a long dispatch has failed in practice, pull results deterministically instead:

```bash
python bin/god/god_mode_report.py --cwd <project-path>
```

This reads the project's SQLite DB directly and groups records by task, correctly handling multi-target fan-out. **Orchestrators should run this after every dispatch round**, not rely on their own recollection of what workers reported.

Full usage, env vars, and troubleshooting: [`bin/god/README.md`](../mathir_mcp/bin/god/README.md).

---

## Design Principles

1. **No new infrastructure** — MATHIR's existing memory is the message queue
2. **Self-identification** — agents describe themselves, you don't configure them
3. **Smart assignment** — orchestrator matches task complexity to agent capability
4. **Single-poll-and-return** — MCP tools can't loop; they return immediately
5. **Protocol-level coordination** — structured labels, not free-text parsing
6. **Agent-agnostic** — works with any agent that has MCP access to MATHIR

---

## Requirements

- MATHIR daemon running on port 7338
- 2+ terminals with AI agents that have MATHIR MCP access
- v8.8.0+ of `mathir_mcp_server.py` and `mathir_server.py`
