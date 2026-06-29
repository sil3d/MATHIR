# MATHIR — Outreach Email Templates

Each email focuses on **what MATHIR does**, not what others don't have.
No assumptions about other providers' internal work.

---

## Claude Memory vs MATHIR — Technical Comparison (internal reference, NOT sent)

| Dimension | Claude Memory | MATHIR |
|---|---|---|
| Architecture | RAG on chat history, summary updated every 24h | 5-tier cognitive memory (working/episodic/semantic/procedural/immunological) |
| Storage | Anthropic cloud servers | SQLite local, zero cloud dependency |
| Forgetting | None — memories persist indefinitely | Ebbinghaus decay (5%/30 days), archive when stability < 0.05 |
| Promotion | None | working→episodic→semantic→procedural based on recall_count + age |
| Deduplication | None | Auto-merge near-duplicates (cosine > 0.95) |
| Link graph | None | Spreading activation between memories (Collins & Loftus 1975) |
| Anomaly detection | None | Mahalanobis distance, AUC = 1.0 on test set |
| Edge deployment | None (cloud only) | Jetson Orin (30ms), Raspberry Pi (CPU), RC car (autonomous) |
| Cross-provider | No — locked to Claude | Works with any LLM via MCP |
| MCP tools | 0 (proprietary) | 23 tools via Model Context Protocol |
| Import/export | Basic | Full API + JSON + MCP |
| Cost | $20/mo+ (Pro plan) | Free (MIT license) |
| Privacy | Data on Anthropic servers | Memory never leaves your machine |

---

## Email Templates

All emails are in English. Each highlights MATHIR's **unique technical capabilities**.

---

### Template 1: Anthropic

**Subject:** MATHIR — open-source cognitive memory with anomaly detection, works with Claude

Hi Anthropic Team,

I'm Prince Gildas, an independent developer. I built [MATHIR](https://github.com/sil3d/MATHIR) — a 5-tier cognitive memory system with 23 MCP tools that's fully open-source and works with Claude via MCP.

**What MATHIR does that's unique:**
- **Anomaly detection on inputs** — Mahalanobis distance-based, AUC = 1.0 on test set. Catches prompt injections, data leakage, and unusual patterns in real time. No other memory system does this.
- **Ebbinghaus decay** — memories fade when unused (5%/30 days), archive when stability < 0.05. This matches how human memory actually works.
- **Tier promotion** — working_memory → episodic → semantic → procedural based on recall frequency and age. The system learns what matters.
- **Consolidation** — auto-merges near-duplicates (cosine > 0.95), keeping the canonical version with an audit trail.
- **Link graph** — spreading activation (Collins & Loftus 1975) connects related memories for contextual recall.

**Deployment:** Runs on Jetson Orin (30ms recall, 500 MB VRAM), Raspberry Pi (CPU-only), or any machine with Python. SQLite local — no cloud dependency. MIT licensed.

I'm not asking for promotion. I wanted to share that MATHIR exists as a reference implementation for what cross-provider cognitive memory looks like — with the anomaly detection layer that's missing from every proprietary memory system I've tested.

GitHub: https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

### Template 2: OpenAI

**Subject:** MATHIR — cross-provider cognitive memory with anomaly detection

Hi OpenAI Team,

I'm Prince Gildas, an independent developer. I built [MATHIR](https://github.com/sil3d/MATHIR) — a 5-tier cognitive memory system with 23 MCP tools that works with any LLM via the Model Context Protocol.

**Core capabilities:**
- **Anomaly detection** — Mahalanobis distance-based, AUC = 1.0. Detects prompt injection, data leakage, and unusual patterns in <1ms. This is the only memory system I've found that has this.
- **Ebbinghaus decay** — memories follow forgetting curves. Unused memories archive automatically after 30 days. Used memories strengthen.
- **Tier promotion** — working → episodic → semantic → procedural. Memory quality improves with use.
- **Consolidation** — auto-merges duplicates (cosine > 0.95), keeping one canonical version.
- **Link graph** — spreading activation connects related memories (Collins & Loftus 1975).

**Performance:** 401 ops/s recall, p50 = 2.29ms. Runs on Jetson Orin (500 MB VRAM), Raspberry Pi, or any machine. SQLite local, zero cloud dependency. MIT licensed.

226 tests passing. 23 MCP tools via Model Context Protocol.

GitHub: https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

### Template 3: Google

**Subject:** MATHIR — open-source cognitive memory for any LLM

Hi Google Team,

I'm Prince Gildas, an independent developer. I built [MATHIR](https://github.com/sil3d/MATHIR) — a 5-tier cognitive memory system that works with any LLM through the Model Context Protocol (MCP).

**What makes MATHIR different:**
- **Anomaly detection** — Mahalanobis distance-based, AUC = 1.0 on test set. Detects prompt injection, data leakage, and unusual patterns. Unique feature in any memory system.
- **Ebbinghaus forgetting curve** — memories decay when unused, strengthen when recalled. This is how human memory actually works.
- **Consolidation** — auto-merges near-duplicates (cosine > 0.95) with audit trail.
- **Link graph** — spreading activation connects memories across sessions.
- **Edge deployment** — Jetson Orin (30ms recall), Raspberry Pi (CPU-only), RC car (autonomous).

23 MCP tools. SQLite local, no cloud dependency. MIT licensed.

GitHub: https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

### Template 4: NVIDIA

**Subject:** MATHIR — edge cognitive memory for Jetson, tested on Orin Nano

Hi NVIDIA Team,

I built [MATHIR](https://github.com/sil3d/MATHIR), a 5-tier cognitive memory system that runs on Jetson Orin at 30ms recall with 500 MB VRAM. It's designed for autonomous systems that need to remember across sessions.

**What it does:**
- **Anomaly detection** — Mahalanobis distance, AUC = 1.0. Catches sensor failures, prompt injections, unusual patterns.
- **Ebbinghaus decay** — memories fade when unused, strengthen when recalled.
- **Tier promotion** — working → episodic → semantic → procedural based on usage.
- **Consolidation** — auto-merges duplicates, keeps canonical version.
- **Link graph** — spreading activation for contextual recall.

**Jetson performance:**
- bge-large-en-v1.5 on CUDA fp16 — tested on Orin Nano
- SQLite + sqlite-vec — no external DB
- <30ms recall, 67 TOPS peak
- 4-stage validation pipeline: laptop → Raspberry Pi → Jetson → RC car

Part of the MATHIR Roadmap: https://github.com/sil3d/MATHIR#-roadmap

Best,
Prince Gildas

---

### Template 5: Cursor / Windsurf

**Subject:** MATHIR — 23 MCP tools for persistent memory across coding sessions

Hi [Cursor/Windsurf] Team,

MATHIR is a 5-tier cognitive memory system with 23 MCP tools. It integrates natively with your MCP configuration — zero custom code needed.

**What MATHIR gives your users:**
- Memory that persists across coding sessions (not just one chat)
- Works with any LLM connected to the IDE
- Anomaly detection (AUC = 1.0) — catches prompt injection and data leakage
- Edge deployment — runs locally, no cloud dependency
- 23 tools: `memory_save`, `memory_recall`, `memory_by_path`, `memory_recall_quality`, `memory_incoming_links`, etc.

**Quick setup:**
```json
{ "mcpServers": { "mathir": { "command": "mathir-mcp" } } }
```

Already verified on Cursor. MIT licensed.

GitHub: https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

### Template 6: xAI / MiniMax / Qwen / Emerging Providers

**Subject:** MATHIR — portable cognitive memory for your LLM, MIT, 23 tools

Hi [Provider] Team,

I built [MATHIR](https://github.com/sil3d/MATHIR), a 5-tier cognitive memory system that works with any LLM via MCP. It's designed to give your models **memory that persists across sessions** — without requiring users to stay on one platform.

**Unique capabilities:**
- **Anomaly detection** — Mahalanobis distance, AUC = 1.0. No other memory system has this.
- **Ebbinghaus decay** — memories fade when unused, strengthen when recalled.
- **Tier promotion** — working → episodic → semantic → procedural.
- **Consolidation** — auto-merges duplicates with audit trail.
- **Link graph** — spreading activation for contextual recall.
- **Edge deployment** — Jetson Orin (30ms), Raspberry Pi, RC car.

23 MCP tools. SQLite local, zero cloud dependency. MIT licensed.

GitHub: https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

### Template 7: Generic (Any Provider)

**Subject:** MATHIR — open-source cognitive memory with anomaly detection

Hi [Team],

I'm Prince Gildas, an independent developer. I built [MATHIR](https://github.com/sil3d/MATHIR) — a 5-tier cognitive memory system with 23 MCP tools.

**What MATHIR does:**
- **Anomaly detection** — Mahalanobis distance-based, AUC = 1.0. Catches prompt injection, data leakage, unusual patterns. No other memory system has this.
- **Ebbinghaus decay** — memories fade when unused, strengthen when recalled.
- **Tier promotion** — working → episodic → semantic → procedural.
- **Consolidation** — auto-merges near-duplicates with audit trail.
- **Link graph** — spreading activation for contextual recall.

**Performance:** 401 ops/s recall, p50 = 2.29ms. Jetson Orin (30ms), Raspberry Pi, any machine. SQLite local. MIT licensed.

226 tests passing. 23 MCP tools.

GitHub: https://github.com/sil3d/MATHIR

Best,
Prince Gildas
