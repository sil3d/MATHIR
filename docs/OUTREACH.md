# MATHIR — Outreach Templates

Comparison of Claude Memory vs MATHIR, plus email templates for outreach to major AI providers.

---

## Claude Memory vs MATHIR — Comparison

| Feature | Claude Memory | MATHIR |
|---|---|---|
| **Type** | Built-in (Anthropic) | Open-source MCP layer |
| **Provider** | Claude only | Any LLM (Claude, GPT, Gemini, Ollama, local) |
| **Architecture** | RAG on chat history | 5-tier cognitive memory (working/episodic/semantic/procedural/immunological) |
| **Storage** | Anthropic cloud servers | SQLite local on your machine |
| **Privacy** | Data on Anthropic servers | Memory never leaves your machine |
| **Decay** | No forgetting curve | Ebbinghaus decay (5%/30 days) |
| **Promotion** | No tier promotion | working→episodic→semantic→procedural |
| **Consolidation** | No deduplication | Auto-merge near-duplicates (cosine > 0.95) |
| **Link graph** | No | Spreading activation links between memories |
| **Anomaly detection** | No | AUC=1.0, immunological tier |
| **Edge deployment** | No (cloud only) | Jetson, Raspberry Pi, RC car, ONNX |
| **Cost** | $20/mo+ (Pro plan) | Free (MIT license) |
| **Cross-provider** | ❌ Vendor lock-in | ✅ Any LLM, any provider |
| **MCP tools** | 0 (proprietary) | 23 tools via Model Context Protocol |
| **Import/export** | Basic (ChatGPT, Gemini) | Full MCP + API + JSON export |
| **Availability** | Pro/Max/Team/Enterprise only | Free for everyone |

**Key insight:** Claude Memory is a closed ecosystem feature. MATHIR is the open-source, cross-provider alternative that gives you the same (and more) capabilities without vendor lock-in.

---

## Email Templates

### Template 1: Generic Outreach (All Providers)

**Subject:** Open-source memory layer that works with [Provider] — MIT, 23 MCP tools, zero vendor lock-in

Hi [Team/Name],

I'm Prince Gildas, an independent developer. I built [MATHIR](https://github.com/sil3d/MATHIR) — a 5-tier cognitive memory layer for LLMs that's fully open-source and works with any provider, including [Provider].

**Why this matters:**
- Claude has memory, GPT has memory, Gemini has memory — but they're all **vendor-locked**
- MATHIR gives your users **the same memory capabilities** without requiring them to stay on one platform
- 23 MCP tools, Ebbinghaus decay, tier promotion, anomaly detection (AUC=1.0)
- MIT licensed, ~500 MB VRAM, runs on Jetson/Raspberry Pi

**Current stats:**
- 226 tests passing
- 5-tier cognitive memory (working/episodic/semantic/procedural/immunological)
- Edge deployment on Jetson Orin (30ms recall) and Raspberry Pi
- Hybrid search: vector + BM25 + RRF fusion

I'd love to discuss how MATHIR could complement [Provider]'s memory offerings. Even if you don't mention it publicly, having a reference implementation shows the ecosystem is maturing.

Best,
Prince Gildas
https://github.com/sil3d/MATHIR

---

### Template 2: Anthropic (Claude Memory Comparison)

**Subject:** MATHIR vs Claude Memory — open-source alternative with 5-tier cognition

Hi Anthropic Team,

Claude Memory is a great feature — but it's locked to your platform. I built [MATHIR](https://github.com/sil3d/MATHIR), an open-source alternative that gives any LLM the same (and more) memory capabilities.

**Claude Memory limitations:**
- Vendor-locked: memories don't transfer to GPT, Gemini, or local models
- No decay, no consolidation, no link graph
- No anomaly detection
- Requires Pro plan ($20/mo+)

**What MATHIR adds:**
- Cross-provider: works with Claude, GPT, Gemini, Ollama — any LLM
- 5-tier cognitive memory with Ebbinghaus decay
- Anomaly detection (AUC=1.0) — unique feature no provider has
- 23 MCP tools, MIT licensed, runs on edge devices
- No subscription required

MATHIR could be a reference implementation for what "memory for all LLMs" looks like. I'm not asking for promotion — just wanted to share that the open-source ecosystem is building what your proprietary features do.

Best,
Prince Gildas
https://github.com/sil3d/MATHIR

---

### Template 3: OpenAI (GPT Memory)

**Subject:** Cross-provider memory layer — works with GPT and every other LLM

Hi OpenAI Team,

GPT Memory is a great feature, but it's locked to your platform. I built [MATHIR](https://github.com/sil3d/MATHIR) — an open-source memory layer that works with GPT, Claude, Gemini, Ollama, and any other LLM via MCP.

**Why this matters:**
- Users want memory that persists across providers, not just within one
- MATHIR has 23 MCP tools, Ebbinghaus decay, anomaly detection (AUC=1.0)
- It runs locally (SQLite), no cloud dependency, no vendor lock-in
- MIT licensed, ~500 MB VRAM, edge-ready (Jetson, Raspberry Pi)

MATHIR doesn't compete with GPT Memory — it complements it by making memory portable across providers.

Best,
Prince Gildas
https://github.com/sil3d/MATHIR

---

### Template 4: Google (Gemini)

**Subject:** MATHIR — open-source memory that works with Gemini and beyond

Hi Google Team,

Gemini has memory capabilities, but like all providers, it's vendor-locked. I built [MATHIR](https://github.com/sil3d/MATHIR) — a 5-tier cognitive memory layer that works with Gemini, Claude, GPT, Ollama, and any MCP-compatible client.

**Key differentiator:**
- MATHIR is the first cross-provider memory system with anomaly detection (AUC=1.0)
- 23 MCP tools, Ebbinghaus decay, tier promotion, link graph
- Edge deployment: Jetson Orin (30ms recall), Raspberry Pi (CPU-only)
- MIT licensed, no cloud dependency

MATHIR could help your users who want memory that follows them across Gemini, Claude, and local models.

Best,
Prince Gildas
https://github.com/sil3d/MATHIR

---

### Template 5: NVIDIA (Edge + GPU)

**Subject:** MATHIR on Jetson — edge memory for autonomous systems

Hi NVIDIA Team,

I built [MATHIR](https://github.com/sil3d/MATHIR), a 5-tier cognitive memory system that runs on Jetson Orin at 30ms recall with 500 MB VRAM. It's the memory layer for autonomous systems that need to remember across sessions.

**What makes it relevant to Jetson:**
- bge-large-en-v1.5 embeddings on CUDA fp16 — tested on Orin Nano
- SQLite + sqlite-vec for vector search — no external DB needed
- 23 MCP tools, full lifecycle management (decay, consolidation, promotion)
- Part of a 4-stage validation pipeline: laptop → Raspberry Pi → Jetson → RC car

MATHIR is designed for the exact use case Jetson enables: **memory that works in the real world, on real hardware, without cloud.**

Best,
Prince Gildas
https://github.com/sil3d/MATHIR

---

### Template 6: Cursor / Windsurf (IDE Integration)

**Subject:** MATHIR memory layer — works with your MCP integration

Hi [Cursor/Windsurf] Team,

MATHIR is a 5-tier cognitive memory system with 23 MCP tools. It integrates natively with your MCP configuration — no custom code needed.

**Why your users would benefit:**
- Memory persists across coding sessions (not just one chat)
- Works with any LLM they connect to your IDE
- 23 tools: memory_save, memory_recall, memory_by_path, memory_recall_quality, etc.
- Edge-ready: runs locally on their machine, no cloud dependency

MATHIR is already verified on Cursor and compatible with your MCP plugin system.

Best,
Prince Gildas
https://github.com/sil3d/MATHIR

---

### Template 7: xAI / MiniMax / Qwen (Emerging Providers)

**Subject:** Open-source memory for your LLM — cross-provider, MIT, 23 tools

Hi [xAI/MiniMax/Qwen] Team,

I built [MATHIR](https://github.com/sil3d/MATHIR) — a 5-tier cognitive memory system that works with any LLM via MCP. It's designed to be the memory layer that makes your models **remember across sessions**.

**Why this matters for emerging providers:**
- Claude has memory, GPT has memory — but they're vendor-locked
- MATHIR gives YOUR users memory that works with YOUR model, and persists even if they switch providers
- 23 MCP tools, Ebbinghaus decay, anomaly detection (AUC=1.0)
- MIT licensed, ~500 MB VRAM, no cloud dependency

MATHIR could help differentiate your offering by providing portable, provider-agnostic memory.

Best,
Prince Gildas
https://github.com/sil3d/MATHIR
