# MATHIR — Outreach Email Templates v4

Problem-first, story-driven, human tone.

---

## Email 1: Anthropic

**Subject:** Your users are losing 3 months of context when they switch tools

Hi,

A developer on Reddit described this yesterday: "I spent 3 months building context in Claude — my coding patterns, my project architecture, my preferences. Then I tried Cursor for a week. Day 1: Claude doesn't remember anything I taught it."

That's not a Claude problem. That's a memory problem. Every provider locks context to their platform. When a user switches tools, they start from zero.

I'm building something that fixes this. It's called MATHIR — a local memory system that works with any LLM via MCP. The memory lives on the user's machine (SQLite), not in your cloud. When they switch from Claude to GPT to Ollama, their context follows.

It also does something I haven't seen anywhere else: anomaly detection on inputs. Mahalanobis distance, AUC=1.0 on test set. Catches prompt injection before it reaches the model. That's a 1ms check that most systems skip entirely.

I'm not asking for integration or promotion. I just think the cross-provider memory problem is real, and it would be valuable for Anthropic to know that open-source tools are starting to solve it.

https://github.com/sil3d/MATHIR

Prince Gildas

---

## Email 2: OpenAI

**Subject:** Your users' memory disappears the moment they leave ChatGPT

Hi,

I keep hearing the same story from developers: "I built 3 months of context in ChatGPT. Tried a local Llama model for privacy. Lost everything."

That's the core problem with provider-locked memory. Users invest time building context, then lose it all when they try a different tool. MATHIR solves this — it's a local memory that works with any LLM, including yours.

The other gap: there's no anomaly layer. When someone sends a prompt injection through ChatGPT, nothing flags it. MATHIR catches it in 1ms — Mahalanobis distance, AUC=1.0.

I built this as open-source (MIT, https://github.com/sil3d/MATHIR). It's not competing with GPT Memory — it's solving a problem that vendor lock-in creates.

Prince Gildas

---

## Email 3: Google

**Subject:** Users are losing context across Gemini, Claude, and local models

Hi,

The scenario that keeps coming up: a user builds context in Gemini for weeks. They need to test something in a local model for latency. Switch to Llama. Context gone. Switch back to Gemini. Start over.

That's the memory portability problem. MATHIR is a local memory system that solves it — same memory across Gemini, Claude, GPT, Ollama, any MCP client.

The technical angle that might interest you: anomaly detection on inputs. Mahalanobis distance, AUC=1.0. Most memory systems don't have this. When someone tries prompt injection through Gemini, MATHIR catches it in 1ms.

Built as open-source (MIT). Edge-deployable on Jetson (30ms recall, 500 MB VRAM). Not asking for anything — just sharing that this problem is being solved.

https://github.com/sil3d/MATHIR

Prince Gildas

---

## Email 4: NVIDIA

**Subject:** Cognitive memory that actually runs on Jetson — not just benchmarks

Hi,

Everyone talks about edge AI. But try running a memory system on a Jetson that forgets nothing — it's brutal. Most memory solutions assume cloud, assume unlimited storage, assume a GPU farm.

MATHIR is different. It runs on Jetson Orin at 30ms recall with 500 MB VRAM. SQLite + sqlite-vec, no external DB, no cloud. The memory stays on the device.

What it actually does: 5-tier cognitive memory (working/episodic/semantic/procedural/immunological), anomaly detection via Mahalanobis distance (AUC=1.0), Ebbinghaus decay, consolidation, link graph. All of it local.

I'm testing this on a 3D-printed RC car — real sensors, real noise, real failures. The Jetson stage is where it gets interesting.

Happy to share benchmarks or architecture details.

https://github.com/sil3d/MATHIR

Prince Gildas

---

## Email 5: Cursor / Windsurf

**Subject:** Your users' context vanishes between sessions — here's a fix

Hi,

Quick one: developers using Cursor/Windsurf keep losing context between coding sessions. They explain their project to the AI, close the app, come back tomorrow — and the AI remembers nothing.

MATHIR (https://github.com/sil3d/MATHIR) fixes this. It's a memory layer with 23 MCP tools that persists across sessions. It connects to any LLM your users connect to — Claude, GPT, local Llama, whatever.

One line in the MCP config:
```json
{ "mcpServers": { "mathir": { "command": "mathir-mcp" } } }
```

Runs locally, no cloud dependency. MIT licensed.

Prince Gildas

---

## Email 6: xAI / MiniMax / Qwen / Emerging Providers

**Subject:** Your users need memory that follows them, not your platform

Hi,

When someone switches from one LLM to another, they lose everything. That's not a bug — it's how provider-locked memory works.

MATHIR (https://github.com/sil3d/MATHIR) is an open-source memory that lives on the user's machine. It connects to any LLM via MCP — including yours. When users switch from your model to Claude or GPT, their context stays.

The technical novelty: anomaly detection on inputs. Mahalanobis distance, AUC=1.0. No other memory system does this. When someone tries prompt injection, MATHIR catches it in 1ms.

For emerging providers, portable memory is a differentiator. Users who build context in MATHIR can try your model without losing everything.

Prince Gildas

---

## Email 7: Generic

**Subject:** Memory that follows the user, not the provider

Hi,

When a user switches from one LLM to another, they lose everything. That's the fundamental problem with provider-locked memory.

MATHIR (https://github.com/sil3d/MATHIR) is a local memory system that works with any LLM via MCP. The memory lives on the user's machine, not in any cloud. It also does anomaly detection on inputs — catches prompt injection in 1ms, something no other memory system does.

23 MCP tools, SQLite, edge-ready (Jetson, Raspberry Pi). MIT licensed.

https://github.com/sil3d/MATHIR

Prince Gildas
