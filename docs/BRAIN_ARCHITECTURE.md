# MATHIR Brain Architecture

**6-phase system + 3-layer auto-cache that makes MATHIR proactive, never-blocking, and brain-like.**

> **v8.9.4** — 6-tier architecture (guardrail tier: push-based always-active rules, immune to decay). 3-layer auto-cache (L1 Embedding / L2 Recall / L3 Session) and `/api/cache/stats` endpoint. Phase 1 reimplemented in v8.5.0 as **`mathir_proxy.py` on port 7339**, now the recommended universal path for any tool: Anthropic `/v1/messages` **and** OpenAI-compatible `/v1/chat/completions`, with per-request multi-upstream routing (`X-Mathir-Upstream`) across an allowlist of ~30 providers plus any local model server. Self-healing on all 3 OSes as of v8.9.4 (systemd `Restart=on-failure`, launchd `KeepAlive`, Windows Task Scheduler healthcheck every 5 min, no admin required). **Phase 6 added in v8.9.x**: god-mode orchestration + client polling bridge (`bin/god/`). See AGENT.md §"Brain Architecture" for the updated architecture. The legacy `mathir_mcp/brain/` fork (including the old `mathir_inject_proxy.py` on 8182) was removed in v8.9.4 — it was stale and unimported; `mathir_lib/mathir_inject_proxy.py` (the maintained copy) and `mathir_lib/mathir_proxy.py` (the recommended one) are unaffected.

## The Problem

Two failure modes of the original design:
1. **LLMs forget to recall** — even when told MATHIR is their memory, agents forget to call `recall`. They get distracted by other tasks.
2. **Daemon crashes** — when the daemon dies, the entire memory system is offline. No recovery.

## The Solution: 5 Phases

### Phase 1 — Universal Auto-Inject Proxy (`mathir_proxy.py`, port 7339)

An OpenAI-compatible HTTP proxy that sits between your LLM client and the real LLM API. **Works for ANY OpenAI-compatible agent** (Claude Code via `OPENAI_BASE_URL`, Cursor, Cline, Continue, Codex, Gemini via `OPENAI_BASE_URL`, etc.) — not just opencode/mimocode.

**Flow:**
```
User message → Proxy (port 7339) → Inject memories → Real LLM API → Response
                ↓
              daemon /api/context in <300ms
```

**Effect:** The LLM never needs to call `recall`. Memories are pre-injected into the system prompt on every request. Just like a human doesn't "search their brain" — they just know.

**Usage:**
```bash
python mathir_mcp/mathir_lib/mathir_proxy.py --port 7339 --target https://api.anthropic.com
# Then in your agent -- pick the one matching your tool's wire format:
export ANTHROPIC_BASE_URL=http://127.0.0.1:7339        # Claude Code etc. -- no /v1
export OPENAI_BASE_URL=http://127.0.0.1:7339/v1         # OpenAI-compatible tools -- /v1 required
```

For opencode/mimocode (which have their own plugin), you don't need the proxy — they auto-inject via `mathir-auto-inject.ts` (TypeScript plugin that hooks `session.started` + `experimental.chat.system.transform`).

### Phase 2 — Daemon Watchdog (`mathir_watchdog.py`)

Background process that pings the daemon and restarts it if it crashes.

**Verified recovery time:** ~30s (7s detection + 1s restart + 20s model load).

**Usage:**
```bash
python mathir_watchdog.py --interval 15 --cooldown 10
```

### Phase 3 — Spreading Activation (`mathir_spread.py`)

When you recall "Tauri", the link graph automatically activates related memories: "Rust", "IPC", "desktop app", "Cargo", "axum" — even if they don't have the highest cosine similarity to the query.

**Inspired by:** Collins & Loftus (1975) spreading activation theory.

**Schema:** New `memory_links` table with `(source_id, target_id, weight, created_at)`. Built via cosine similarity > 0.7.

**Result:** Recall returns the initial vector hits + their linked memories (1-2 hops, decay 0.5).

**Usage:**
```bash
# Build links for all memories (one-time, ~30s for 300 memories)
python mathir_spread.py build_all
```

### Phase 4 — Consolidation / Sleep (`mathir_consolidate.py`)

Nightly process that mimics what the brain does during slow-wave sleep:
- **Merge** near-duplicates (cosine > 0.95)
- **Decay** unused memories (Ebbinghaus: 5%/month if no access)
- **Boost** frequently-accessed memories (stability += 0.1 per access)
- **Archive** dead memories (stability < 0.05)

**Usage:**
```bash
python mathir_consolidate.py        # Run consolidation
python mathir_consolidate.py dry    # Dry run (no changes)
```

**Schedule:** Run via Windows Task Scheduler nightly, or cron on Linux.

### Phase 5 — Pre-Cognitive Priming (`mathir_prime.py`)

Senses environmental context BEFORE the user even asks:
- Current working directory
- Git branch + last commit
- Recently modified files (last 24h)

This is added to the recall query so the LLM gets project-relevant memories, not just literal text matches.

**Effect:** When working in Mycerise_V2_Taur, the query becomes "fix the bug" + "project:Mycerise_V2_Taur" + "branch:main" → retrieves project-specific memories.

### Phase 6 — Multi-Agent Orchestration Bridge (`bin/god/`)

Turns MATHIR's shared memory into a **cross-process message queue** for coordinating multiple AI agents across terminals:

- **Server-side:** `mathir_lib/mathir_god.py` exposes `/api/god/poll` + `/api/god/agents` HTTP routes. Workers register, orchestrators dispatch tasks via `memory_save(label="god:task:...")`.
- **Client-side:** `bin/god/god_bridge.py` is a standalone polling daemon for terminals (3 modes: `worker` / `orchestrator` / `observer`). Stdlib-only, cross-platform (Windows / Linux / macOS).
- **Protocol:** Structured labels (`god:task:{id}:{worker}:pending`, `god:result:{id}:{worker}:completed`) — see `bin/god/PROTOCOL.md`.

**Why a separate client bridge?** MCP tools return immediately — they can't block waiting for events. `god_bridge.py` runs as an external process, polls every N seconds, beeps + logs on new events. **Effect:** One orchestrator can dispatch to N workers across N terminals, with zero manual relay.

**Usage:**
```bash
# Worker terminal (waits across many agent turns)
python mathir_mcp/bin/god/god_bridge.py --mode worker --name mimo --interval 5

# Orchestrator terminal
python mathir_mcp/bin/god/god_bridge.py --mode orchestrator --interval 5 --project <name>
```

## 3-Layer Auto-Cache (v8.7.0)

Three transparent caching layers sit between MCP tool calls and the daemon, eliminating redundant work:

| Layer | Scope | Strategy | Size | Invalidation |
|---|---|---|---|---|
| **L1 Embedding Cache** | `encode()` output | LRU | 1024 entries | Never (deterministic — same text always produces same vector) |
| **L2 Recall Cache** | Search results (`recall`, `smart_search`, `hybrid_search`) | TTL 60 s | 256 entries | Immediate on any `memory_save` / `memory_delete` write |
| **L3 Session Cache** | `session_start` / `context` (top-20 per project) | TTL 5 min | top-20/project | Immediate on any write |

**Effect:** Repeated recalls for the same query within 60 s are served from L2 in <1 ms instead of hitting the vector index. Session starts within the same 5-minute window reuse L3. L1 removes all redundant embedding computation permanently.

**Monitoring:** `GET /api/cache/stats` returns hit/miss counts and eviction stats for all three layers.

## All-in-One Launcher

```bash
python mathir_brain.py start    # Start daemon + watchdog + proxy
python mathir_brain.py status   # Show status
python mathir_brain.py stop     # Stop all

# Cache statistics (v8.7.0)
curl http://127.0.0.1:7338/api/cache/stats
```

## Pointing your LLM client to the proxy

### OpenCode (`opencode.json`)
```json
{
  "provider": {
    "anthropic": {
      "options": {
        "baseUrl": "http://127.0.0.1:7339"
      }
    }
  }
}
```

### MiMo Code (`mimocode.json`)
```json
{
  "provider": {
    "default": {
      "baseUrl": "http://127.0.0.1:7339"
    }
  }
}
```

After this, every LLM call gets `<mathir-auto-injection>` block prepended to the system prompt automatically.

## Why this is "brain-like"

| Brain | MATHIR |
|---|---|
| Hippocampus indexes episodes | `memory_links` graph |
| Prefrontal cortex holds working memory | Auto-injected top-K in system prompt |
| Long-term memory retrieval | Spreading activation (1-2 hops) |
| Sleep consolidates memories | `mathir_consolidate.py` |
| Reticular activating system filters | Pre-cognitive priming (cwd, git, files) |
| No "explicit search" needed | LLM never calls recall — memories appear |
| Synaptic facilitation (repeated access = faster) | 3-layer auto-cache (L1/L2/L3) |

## Files

All brain-phase scripts live in `mathir_mcp/mathir_lib/` (the old `mathir_mcp/brain/` fork was a stale, unimported duplicate, removed in v8.9.4 -- see CHANGELOG).

- `mathir_mcp/mathir_lib/mathir_proxy.py` — Phase 1 (v8.5.0+, current): universal auto-inject proxy on port 7339, Anthropic + OpenAI-compatible. `mathir-brain` launches this.
- `mathir_mcp/mathir_lib/mathir_inject_proxy.py` — Phase 1 (legacy, unused): the original auto-inject proxy on port 8182. Speaks the raw TCP JSON-RPC protocol the daemon dropped in the v8.5.0 HTTP rewrite, so its injection has been silently non-functional since then. No longer launched by `mathir_brain.py` as of v8.9.4 (was, previously, by mistake). Kept in the tree for now; not recommended.
- `mathir_mcp/mathir_lib/mathir_watchdog.py` — Phase 2: daemon watchdog
- `mathir_mcp/mathir_lib/mathir_spread.py` — Phase 3: spreading activation
- `mathir_mcp/mathir_lib/mathir_consolidate.py` — Phase 4: sleep consolidation
- `mathir_mcp/mathir_lib/mathir_prime.py` — Phase 5: pre-cognitive priming
- `mathir_mcp/mathir_lib/mathir_god.py` — Phase 6 (v8.8.0+): god-mode server logic (`GodProtocol`, `TaskGraph`, `WorkerRegistry`, `WorktreeManager`)
- `mathir_mcp/bin/god/god_bridge.py` — Phase 6 (v8.9.2+): god-mode client polling daemon (3 modes)
- `mathir_mcp/bin/god/god_poll.{ps1,sh}` — Phase 6: lightweight cross-platform pollers
- `mathir_mcp/bin/god/PROTOCOL.md` — Phase 6: label spec + message flow
- `mathir_mcp/brain/mathir_brain.py` — All-in-one launcher
- `mathir_mcp/opencode_templates/plugins/mathir-auto-inject.ts` — Tier-A plugin (opencode/mimocode only)

## Dependencies

- `aiohttp` (for the proxy): `pip install aiohttp`
- `psutil` (for the launcher): `pip install psutil`
