# MATHIR — Outreach Email Templates

Rewritten v3: natural, professional, problem-first approach.

---

## Email 1: Anthropic

**Subject:** Memory for all LLMs — not just one

Hi Anthropic Team,

I'm a solo developer working on open-source tools for the AI agent ecosystem. My current project is MATHIR (https://github.com/sil3d/MATHIR) — a cognitive memory layer that runs locally and connects to any LLM via MCP.

The problem I keep hitting: every memory feature I try is tied to one provider. When I switch from Claude to GPT or Ollama, my context disappears. MATHIR solves this — it's a local SQLite-based memory with 23 MCP tools, and it works with any LLM out of the box.

A few things I think would interest your team:
- Anomaly detection on inputs (Mahalanobis distance, AUC=1.0) — catches prompt injection and data leakage in <1ms
- Ebbinghaus decay — memories fade when unused, strengthen when recalled
- Edge deployment — runs on Jetson Orin at 30ms recall, 500 MB VRAM

I'm not looking for promotion or integration. I just thought it would be interesting to see what an open-source, cross-provider memory looks like — and whether there are things worth borrowing or discussing.

Happy to share the codebase or jump on a call if there's interest.

Best,
Prince Gildas
https://github.com/sil3d/MATHIR

---

## Email 2: OpenAI

**Subject:** Cross-provider memory that stays with the user

Hi OpenAI Team,

I built a tool I want to share — not for integration, just because I think it's relevant to what you're building.

MATHIR (https://github.com/sil3d/MATHIR) is a cognitive memory system that lives locally (SQLite, no cloud) and connects to any LLM via MCP. The core idea: memory should follow the user, not the provider.

Right now, if a user builds context in ChatGPT and switches to a local Llama model, everything is lost. MATHIR fixes that — same memory, any model.

What makes it technically interesting:
- Anomaly detection (Mahalanobis distance, AUC=1.0) — catches prompt injection and unusual patterns
- Ebbinghaus forgetting curve — memories decay when unused, promote when recalled
- 23 MCP tools, runs on Jetson/Raspberry Pi, MIT licensed

I'm not asking for anything — just sharing that the open-source ecosystem is building portable memory, and it might be worth watching.

https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

## Email 3: Google

**Subject:** Memory that works across Gemini, Claude, and local models

Hi Google Team,

I'm Prince Gildas, a developer. Quick pitch: MATHIR (https://github.com/sil3d/MATHIR) is a local memory system for LLMs that doesn't lock you into one provider.

The use case: an enterprise agent that needs to remember context across multiple models — Gemini for some tasks, Claude for others, local Llama for on-premise. Today that means separate memory silos. MATHIR unifies them.

23 MCP tools, SQLite local, anomaly detection (Mahalanobis, AUC=1.0), Ebbinghaus decay, edge-ready for Jetson. MIT licensed.

I'm sharing this because the cross-provider memory problem is real, and the open-source community is starting to solve it. Would be curious if Google has plans in this space too.

https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

## Email 4: NVIDIA

**Subject:** Cognitive memory that runs on Jetson — tested, documented, open-source

Hi NVIDIA Team,

I built MATHIR (https://github.com/sil3d/MATHIR), a cognitive memory system that runs on Jetson Orin with 30ms recall at 500 MB VRAM. It's designed for autonomous systems that need to remember across sessions.

What it does:
- 5-tier memory (working/episodic/semantic/procedural/immunological)
- Anomaly detection via Mahalanobis distance (AUC=1.0)
- bge-large-en-v1.5 embeddings on CUDA fp16
- SQLite + sqlite-vec, no external DB

I have a 4-stage validation pipeline: laptop → Raspberry Pi → Jetson → RC car. The Jetson stage is where the real work happens.

I'm sharing because I think there's value in having a documented, open-source reference for what cognitive memory looks like on edge hardware. Happy to share benchmarks or discuss the architecture.

https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

## Email 5: Cursor / Windsurf

**Subject:** 23 MCP tools for persistent memory across coding sessions

Hi,

I built MATHIR (https://github.com/sil3d/MATHIR) — a memory system with 23 MCP tools that integrates natively with Cursor/Windsurf.

The value for your users: memory that persists across sessions, works with any LLM, and runs locally. No cloud dependency, no vendor lock-in. Quick setup — one line in the MCP config.

Anomaly detection (AUC=1.0) catches prompt injection. Ebbinghaus decay keeps memories fresh. Edge deployment on Jetson/Raspberry Pi.

MIT licensed, 226 tests passing.

https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

## Email 6: xAI / MiniMax / Qwen / Emerging Providers

**Subject:** Open-source memory for your LLM — local, portable, 23 tools

Hi [Team],

I'm building MATHIR (https://github.com/sil3d/MATHIR) — a cognitive memory layer that works with any LLM via MCP. The pitch is simple: memory should belong to the user, not the provider.

What's technically novel:
- Anomaly detection (Mahalanobis distance, AUC=1.0) — unique in any memory system
- Ebbinghaus decay, tier promotion, auto-consolidation
- 23 MCP tools, SQLite local, edge-ready

For emerging providers, the cross-provider angle matters: users who build context in MATHIR can switch between your model and others without losing everything.

https://github.com/sil3d/MATHIR

Best,
Prince Gildas

---

## Email 7: Generic (Any Provider)

**Subject:** Open-source cognitive memory with anomaly detection

Hi,

I built MATHIR (https://github.com/sil3d/MATHIR) — a cognitive memory system that connects to any LLM via MCP.

The key thing: anomaly detection on inputs. Mahalanobis distance, AUC=1.0. Catches prompt injection and data leakage before it reaches the model. I haven't seen this in any other memory system.

23 MCP tools, Ebbinghaus decay, edge deployment (Jetson, Raspberry Pi). SQLite local, MIT licensed.

Happy to share the code or discuss the architecture.

https://github.com/sil3d/MATHIR

Best,
Prince Gildas
