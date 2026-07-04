# MATHIR God Orchestrator — Design Specification

**Date:** 2026-07-04
**Author:** Prince Gildas Mbama Kombila
**Version:** v1.0 (MVP)
**Status:** Approved

---

## 1. Overview

Two new MCP tools (`mathir_god_orchestre` + `mathir_god_agent`) that enable cross-process multi-agent orchestration via MATHIR's shared memory daemon. No existing framework does this — CrewAI, AutoGen, LangGraph all orchestrate agents within a single process. MATHIR God orchestrates agents **across terminals, across tools** (Claude Code, Codex, MiMo, OpenCode, Kilo, etc.) using the existing MATHIR daemon (port 7338) as the communication bus.

### What it is NOT

- NOT a new daemon or service — uses existing MATHIR daemon
- NOT a process spawner — user opens terminals manually
- NOT a replacement for agent-internal subagents — this is cross-process coordination

---

## 2. Architecture

```
┌──────────────────────────────────────────────┐
│           Terminal 1: ORCHESTRATOR            │
│           (Claude Code / any agent)          │
│                                              │
│  mathir_god_orchestre(directive="...")        │
│    │                                         │
│    ├─ Discover idle workers (god:reg:*)      │
│    ├─ Decompose into TaskGraph (DAG)         │
│    ├─ Present plan → user validates          │
│    ├─ Create git worktrees per task          │
│    ├─ Dispatch ready tasks (god:task:*)      │
│    └─ Monitor loop:                          │
│         poll results → verify → dispatch     │
│         dependents → merge when all done     │
└──────────────────┬───────────────────────────┘
                   │
            MATHIR DAEMON (port 7338)
            Shared memory = message queue
            Labels: god:{type}:{id}:{target}:{status}
                   │
     ┌─────────────┼─────────────┐
     │             │             │
┌────▼────┐  ┌────▼────┐  ┌────▼────┐
│ Term 2  │  │ Term 3  │  │ Term 4  │
│ Worker  │  │ Worker  │  │ Worker  │
│ (MiMo)  │  │ (Codex) │  │(OpenCode│
│         │  │         │  │         │
│ god_    │  │ god_    │  │ god_    │
│ agent() │  │ agent() │  │ agent() │
│         │  │         │  │         │
│ poll →  │  │ poll →  │  │ poll →  │
│ execute │  │ execute │  │ execute │
│ → report│  │ → report│  │ → report│
└─────────┘  └─────────┘  └─────────┘
```

---

## 3. Label Protocol

All coordination happens via `memory_save` with structured labels:

```
god:{type}:{id}:{target}:{status}
```

| Field | Values | Description |
|---|---|---|
| `type` | `reg`, `task`, `result`, `plan` | Message type |
| `id` | 8-char hex (e.g. `a1b2c3d4`) | Unique task/worker ID |
| `target` | agent name or `orchestrator` | Who this message is for |
| `status` | `idle`, `busy`, `pending`, `running`, `completed`, `failed`, `verified`, `shutdown` | Current state |

### Message types

**Registration (worker → daemon):**
```python
memory_save(
    content='{"capabilities": ["code", "test", "debug"], "pid": 1234}',
    label="god:reg:mimo:idle",
    block_type="working_memory",
    priority=3
)
```

**Task dispatch (orchestrator → worker):**
```python
memory_save(
    content='{"description": "Refactor auth.py", "worktree_branch": "god/a1b2c3d4", "context": "..."}',
    label="god:task:a1b2c3d4:mimo:pending",
    block_type="working_memory",
    priority=7
)
```

**Task acceptance (worker → daemon):**
```python
memory_save(
    content="accepted",
    label="god:task:a1b2c3d4:mimo:running",
    block_type="working_memory",
    priority=7
)
```

**Result (worker → orchestrator):**
```python
memory_save(
    content='{"summary": "Refactored auth.py: extracted validate_token(), 3 files changed", "branch": "god/a1b2c3d4", "files_changed": ["src/auth.py", "src/utils.py"]}',
    label="god:result:a1b2c3d4:orchestrator:completed",
    block_type="episodic",
    priority=7
)
```

**Verification (orchestrator → daemon):**
```python
memory_save(
    content="verified: code review passed, tests green",
    label="god:task:a1b2c3d4:mimo:verified",
    block_type="episodic",
    priority=6
)
```

**Shutdown (orchestrator → worker):**
```python
memory_save(
    content="shutdown requested by orchestrator",
    label="god:task:00000000:mimo:shutdown",
    block_type="working_memory",
    priority=9
)
```

### Block type strategy

| Message type | block_type | Why |
|---|---|---|
| Registrations | `working_memory` | Transient — decay naturally |
| Active tasks | `working_memory` | Transient — cleaned up after completion |
| Results | `episodic` | Kept for history and learning |
| Plans | `episodic` | Kept for audit trail |

---

## 4. Tool: `mathir_god_agent` (Worker)

### Signature

```python
@mcp.tool()
async def mathir_god_agent(
    name: str,              # "mimo", "codex", "opencode", etc.
    capabilities: str = "", # comma-separated: "code,test,debug,review,docs"
    poll_interval: int = 8, # seconds between polls
    worktree: bool = True   # use git worktree for task isolation
) -> str:
```

### Behavior

1. **Register:** `memory_save(label="god:reg:{name}:idle")`
2. **Poll loop** (runs indefinitely):
   - Call daemon `POST /api/god/poll {"agent": name, "status": "pending"}`
   - If no task → sleep(poll_interval) → continue
   - If shutdown task → cleanup registration → exit loop
   - If task found:
     a. **Accept:** `memory_save(label="god:task:{id}:{name}:running")`
     b. **Update reg:** `memory_save(label="god:reg:{name}:busy")`
     c. **Setup worktree** (if enabled): `git worktree add .worktrees/god-{id} -b god/{id}`
     d. **Execute:** Feed the task description to the agent as a prompt. The agent uses its own tools (Edit, Bash, etc.) to complete the work.
     e. **Report:** `memory_save(label="god:result:{id}:orchestrator:completed")` with summary of changes
     f. **Update reg:** `memory_save(label="god:reg:{name}:idle")`
     g. Continue polling

### Error handling

- If execution fails → `memory_save(label="god:result:{id}:orchestrator:failed")` with error details
- If daemon unreachable → retry 3 times with backoff, then log error and continue polling
- If git worktree fails → execute without worktree, warn in result

---

## 5. Tool: `mathir_god_orchestre` (Orchestrator)

### Signature

```python
@mcp.tool()
async def mathir_god_orchestre(
    directive: str,              # "Refactor auth + tests + docs"
    strategy: str = "auto",      # "parallel", "sequential", "auto"
    verify: bool = True,         # orchestrator reviews each result
    auto_merge: bool = False     # merge branches without confirmation
) -> str:
```

### Behavior

1. **Discover workers:**
   - Call daemon `GET /api/god/agents`
   - Returns list of registered workers with capabilities and status
   - If no workers → error: "No workers registered. Open terminals and run mathir_god_agent() first."

2. **Plan (LLM-driven):**
   - The orchestrator agent (an LLM) decomposes the directive into tasks
   - Each task gets: description, required capabilities, dependencies
   - Tasks are organized as a DAG (TaskGraph)
   - Strategy: `"auto"` = LLM decides parallel vs sequential per task; `"parallel"` = all tasks at once; `"sequential"` = one after another

3. **Present plan to user:**
   - Show task list with assignments and dependencies
   - Wait for user approval (semi-autonomous)
   - User can modify assignments or task descriptions

4. **Setup worktrees:**
   - `git worktree add .worktrees/god-{id} -b god/{id}` for each task
   - Each worker operates on its own branch

5. **Dispatch:**
   - `memory_save(label="god:task:{id}:{agent}:pending")` for each ready task (no unmet dependencies)
   - Save the full plan: `memory_save(label="god:plan:{directive_id}:orchestrator:active")`

6. **Monitor loop:**
   - Poll for results: `memory_smart_search("god:result:*:orchestrator")`
   - On task completed:
     - If `verify=True`: orchestrator reviews the result (reads the diff, checks quality)
     - Mark as verified: `memory_save(label="god:task:{id}:{agent}:verified")`
     - Check if dependents are now unblocked → dispatch them
   - On task failed:
     - Find another idle worker with matching capabilities
     - Reassign, or alert user if no worker available
   - On worker timeout (no result in 5 min):
     - Alert user, offer to reassign

7. **Merge:**
   - When all tasks verified:
     - If `auto_merge=True`: merge all branches automatically
     - Else: present merge plan to user, wait for confirmation
   - `git merge god/{id}` for each task branch
   - On conflict: alert user, do NOT auto-resolve
   - Cleanup: `git worktree remove .worktrees/god-{id}`

### TaskGraph structure

```python
{
    "directive_id": "f1e2d3c4",
    "directive": "Refactor auth + tests + docs",
    "created_at": "2026-07-04T14:30:00",
    "tasks": {
        "t1": {
            "description": "Refactor auth.py: extract validate_token()",
            "agent": "mimo",
            "capabilities_required": ["code"],
            "depends_on": [],
            "status": "completed",
            "worktree_branch": "god/t1-a1b2c3d4"
        },
        "t2": {
            "description": "Write unit tests for refactored auth module",
            "agent": "codex",
            "capabilities_required": ["code", "test"],
            "depends_on": ["t1"],
            "status": "running",
            "worktree_branch": "god/t2-b2c3d4e5"
        }
    }
}
```

---

## 6. Daemon Routes

### `POST /api/god/poll`

Optimized task polling — faster than `memory_smart_search` for high-frequency polling.

```python
@app.route("/api/god/poll", methods=["POST"])
def api_god_poll():
    agent = request.json["agent"]
    status = request.json.get("status", "pending")
    # Direct SQL: SELECT * FROM memories 
    #   WHERE label LIKE 'god:task:%:{agent}:{status}'
    #   ORDER BY priority DESC, created_at ASC
    #   LIMIT 1
```

**Request:** `{"agent": "mimo", "status": "pending"}`
**Response:** `{"task": {...}}` or `{"task": null}`

### `GET /api/god/agents`

List all registered workers.

```python
@app.route("/api/god/agents", methods=["GET"])
def api_god_agents():
    # Direct SQL: SELECT * FROM memories
    #   WHERE label LIKE 'god:reg:%'
    #   ORDER BY created_at DESC
```

**Response:**
```json
{
    "agents": [
        {"name": "mimo", "status": "idle", "capabilities": ["code", "test"]},
        {"name": "codex", "status": "busy", "capabilities": ["code", "fast"]},
        {"name": "opencode", "status": "idle", "capabilities": ["review", "docs"]}
    ]
}
```

---

## 7. Module: `mathir_god.py`

Business logic shared between MCP tools and daemon routes.

### Classes

```python
class GodProtocol:
    """Label encoding/decoding and message helpers."""
    @staticmethod
    def make_label(type, id, target, status) -> str
    @staticmethod
    def parse_label(label) -> dict
    @staticmethod
    def generate_task_id() -> str  # 8-char hex

class WorkerRegistry:
    """Track registered workers."""
    def register(name, capabilities) -> None
    def unregister(name) -> None
    def set_status(name, status) -> None
    def list_idle() -> list[dict]
    def find_by_capability(cap) -> list[str]

class TaskGraph:
    """DAG of tasks with dependencies."""
    def add_task(id, desc, agent, depends_on) -> None
    def set_status(id, status) -> None
    def get_ready_tasks() -> list[dict]  # no unmet deps, status=queued
    def get_blocked_tasks() -> list[dict]
    def is_all_done() -> bool
    def to_json() -> str

class WorktreeManager:
    """Git worktree lifecycle."""
    def create(task_id, branch_name) -> Path
    def merge(task_id) -> tuple[bool, str]  # (success, message)
    def cleanup(task_id) -> None
    def list_active() -> list[dict]
```

---

## 8. Error Handling

| Scenario | Action |
|---|---|
| No workers registered | Orchestrator errors with instructions to open terminals |
| Worker goes offline mid-task | Timeout after 5 min → alert user → offer reassign |
| Task execution fails | Worker reports `failed` → orchestrator reassigns or alerts |
| Git worktree creation fails | Execute without isolation, warn in result |
| Git merge conflict | Alert user, do NOT auto-resolve |
| Daemon unreachable | Retry 3x with exponential backoff → error message |
| Duplicate worker name | Overwrite previous registration (latest wins) |
| All workers busy, tasks pending | Queue tasks, dispatch when a worker becomes idle |

---

## 9. Testing Strategy

### Unit tests (`tests/test_god.py`, ~200 lines)

- `GodProtocol`: label encode/decode round-trip, ID generation uniqueness
- `TaskGraph`: add tasks, set dependencies, get_ready_tasks correctness, cycle detection
- `WorkerRegistry`: register, unregister, find_by_capability filtering
- `WorktreeManager`: mock git commands, verify branch naming

### Integration tests

- Full flow: register 2 mock workers → dispatch 3 tasks (2 parallel, 1 dependent) → verify correct ordering
- Failure recovery: worker fails → verify reassignment
- Shutdown: send shutdown → verify worker exits loop

---

## 10. File Changes

| File | Action | Estimated lines |
|---|---|---|
| `mathir_lib/mathir_god.py` | **NEW** — GodProtocol, WorkerRegistry, TaskGraph, WorktreeManager | ~300 |
| `mathir_lib/mathir_mcp_server.py` | **MODIFY** — add 2 MCP tools | ~150 |
| `mathir_lib/mathir_server.py` | **MODIFY** — add 2 daemon routes | ~60 |
| `tests/test_god.py` | **NEW** — unit + integration tests | ~200 |
| **Total** | | **~710 lines** |

---

## 11. Decision Log

| # | Decision | Alternatives considered | Why this option |
|---|---|---|---|
| 1 | Memory-as-Queue (Approach A) | Dedicated SQL table (B), File sidecar (C) | Zero new dependencies, uses existing infra, simplest MVP |
| 2 | Polling (not push) | Webhooks, daemon push, filesystem watch | Polling works with all agents, no new ports, simple |
| 3 | Auto-register | Manual registration, heartbeat | Low friction — worker calls one function and is ready |
| 4 | Semi-autonomous | Full autonomous, manual assisted | Safe default — user validates plan before dispatch |
| 5 | Auto-execution loop | Prompt injection, user copy-paste | Fully automatic, maximizes parallelism |
| 6 | Git worktrees | File locking, no isolation (MVP) | Standard git isolation, safe merge, well-understood |
| 7 | Structured labels | Separate DB table, JSON in content only | Labels are indexed, fast to query, human-readable |
| 8 | working_memory for tasks | episodic for everything | working_memory decays naturally = free cleanup |
| 9 | 2 daemon routes | Use memory_smart_search only | Direct SQL is much faster for high-frequency polling |
| 10 | Single orchestrator | Multi-orchestrator | Avoids coordination complexity for MVP |

---

## 12. Non-Goals (v1)

- Mode 2: daemon-driven process spawning
- Push notifications / webhooks
- Multi-orchestrator coordination
- Intelligent load balancing (round-robin is enough)
- Cross-machine orchestration (localhost only)
- UI dashboard for God mode

---

## 13. Future Work (v2+)

- **Push notifications:** daemon SSE/WebSocket to eliminate polling latency
- **Worker capability learning:** MATHIR remembers which worker is best at what (from past results)
- **Multi-orchestrator:** multiple orchestrators coordinate via a meta-protocol
- **Remote workers:** workers on different machines connect via network
- **God Dashboard:** web UI showing live task graph, worker status, progress bars
