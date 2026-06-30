# MATHIR Brain Architecture

**5-phase system that makes MATHIR proactive, never-blocking, and brain-like.**

> **v8.5.0+ note:** Phase 1 was reimplemented in v8.5.0 as **`mathir_proxy.py` on port 7339** (OpenAI-compatible universal proxy, replaces the legacy `mathir_inject_proxy.py` on 8182). See AGENT.md §"Brain Architecture" for the updated architecture. The 8182 legacy proxy still ships in `mathir_mcp/brain/` for backward compatibility but is no longer the recommended path.

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
python mathir_mcp/mathir_lib/mathir_proxy.py --port 7339
# Then in your agent:
export OPENAI_BASE_URL=http://127.0.0.1:7339/v1
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

## All-in-One Launcher

```bash
python mathir_brain.py start    # Start daemon + watchdog + proxy
python mathir_brain.py status   # Show status
python mathir_brain.py stop     # Stop all
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

## Files

- `mathir_mcp/mathir_lib/mathir_proxy.py` — Phase 1 (v8.5.0+): universal auto-inject proxy on port 7339
- `mathir_mcp/brain/mathir_inject_proxy.py` — Phase 1 (legacy): auto-inject proxy on port 8182 (kept for backward compat)
- `mathir_mcp/brain/mathir_watchdog.py` — Phase 2: daemon watchdog
- `mathir_mcp/brain/mathir_spread.py` — Phase 3: spreading activation
- `mathir_mcp/brain/mathir_consolidate.py` — Phase 4: sleep consolidation
- `mathir_mcp/brain/mathir_prime.py` — Phase 5: pre-cognitive priming
- `mathir_mcp/brain/mathir_brain.py` — All-in-one launcher
- `mathir_mcp/opencode_templates/plugins/mathir-auto-inject.ts` — Tier-A plugin (opencode/mimocode only)

## Dependencies

- `aiohttp` (for the proxy): `pip install aiohttp`
- `psutil` (for the launcher): `pip install psutil`
