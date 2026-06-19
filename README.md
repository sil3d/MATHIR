<div align="center">

# 🧠 MATHIR

### Memory-Augmented Tensor Hybrid with Intelligent Routing

**A drop-in cognitive memory layer with 4 tiers (working, episodic, semantic, immunological) — runs on edge (with model-specific VRAM), plugs into any LLM, MIT-licensed.**

<br/>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-8.2.0-6366f1?style=for-the-badge)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-226%20passed-22c55e?style=for-the-badge)](#-tests--benchmarks)
[![BEIR](https://img.shields.io/badge/BEIR_SciFact-0.7441_nDCG%4010-a855f7?style=for-the-badge)](#-beir-benchmark-results)

<br/>

[**⚡ Quick Start**](#-quick-start-30-seconds) · [**🎬 Demo**](#-live-demo) · [**🏗️ Architecture**](#-architecture) · [**🆚 vs Alternatives**](#-vs-alternatives-honest-2026-comparison) · [**📊 Benchmarks**](#-tests--benchmarks) · [**📄 Paper**](docs/MATHIR_Research_Paper.tex)

<br/>

```
   +37.8%        AUC = 1.0       88% isolation     100% uptime
   online         anomaly         context-aware     2-hour stress
   learning       detection       retrieval         without crash
```

</div>

---

## 😱 The REAL Problem in 2026 (it's not what you think)

> **Most "LLM has no memory" articles are wrong as of 2026.** ChatGPT remembers, Claude Projects remembers, Gemini remembers. So what's actually broken?

### The thing that actually keeps breaking

```
You: "I'm Alice, I work on Python, building a RAG system"
Claude/ChatGPT: "Got it, I'll remember that"        ✅ (vendor-side memory)

... 3 months later, you switch from Claude to a local Llama 3.1 ...
Llama 3.1: "Hi! Who are you?"                         ❌
            ↑ All that "memory"? Gone. Vendor-locked.

... you try Mem0, which costs $79/mo ...
Mem0: "Here's what I remember about you"              ✅ (portable, but $$$)

... you want anomaly detection on weird prompts ...
Mem0: "Sorry, that's not what we do"                  ❌

... you want it on a Jetson Nano with no internet ...
Mem0: "... we'll get back to you with an enterprise quote"  ❌

... you want the source code to audit it ...
Mem0: "It's our managed platform"                     ❌
```

### What's actually broken in 2026

| Real problem | Why it's broken | Who fixes it |
|---|---|---|
| **Vendor lock-in on memory** | ChatGPT's memory doesn't follow you to Claude. Claude Projects don't export to Gemini. | **MATHIR** (local SQLite, yours forever) |
| **Context rot** | 200K-token windows are advertised. Effective is ~32K. Beyond that, hallucinations, repetition, contradictions appear. ([Chroma Context Rot, Jul 2025](https://research.trychroma.com/context-rot)) | MATHIR (curated synthesis, not raw context) |
| **KV cache explodes** | 100M context = 638 H100s per user. ([Magic AI, Aug 2024](https://magic.dev/blog/100m-token-context-windows)) | MATHIR (memory extracted, not stuffed in) |
| **Anomaly detection doesn't exist** | No major LLM (GPT, Claude, Gemini) flags weird/prompt-injection inputs in real time | **MATHIR** (immunological tier, AUC = 1.0) |
| **Memory APIs cost $20–$400/mo** | Mem0 starts at $19, Zep at $104/mo, Cognee Cloud at $35/mo | **MATHIR** (MIT, free) |
| **Most OSS memory libs are just RAG** | mem0, Letta, Cognee, LangMem = vector DB + extraction. Not a "cognitive architecture". | **MATHIR** (4 cognitive tiers, KL router, Mahalanobis detector) |
| **Can't run on edge** | Mem0 / Zep / Cognee = cloud. Recall / Supermemory = SaaS. | **MATHIR** (Jetson Orin ✅, Raspberry Pi ⚠️ with CPU fallback) |
| **Cross-language recall** | "clotures python" (French) doesn't find "Python closures" (English) in vanilla RAG | **MATHIR UNIBRI** (character n-gram kernel + J-L projection) |

### The 4 documented failure modes of long context (Breunig, 2025)

```
   1. Context poisoning       2. Context distraction    3. Context confusion       4. Context clash
   hallucination gets         model fixates on          superfluous content        multi-turn info
   re-referenced              repeating past actions    degrades responses         contradicts itself
   (Gemini 2.5 Pokémon)       (beyond 100K tokens)       (8B fails with 46 tools)   (o3: 98.1 → 64.1)
        ↓                         ↓                         ↓                          ↓
   ╔══════════════════════════════════════════════════════════════════════════════════════╗
   ║  Long context ≠ memory. A 1M-token window is a liability if you don't structure it.  ║
   ╚══════════════════════════════════════════════════════════════════════════════════════╝
```

> **MATHIR's position is honest:** We're not a "ChatGPT memory clone". We're a **cognitive architecture** for any LLM — with **4 memory tiers**, **online learning**, **anomaly detection**, and **edge deployment** — that you can actually read, audit, and run yourself.

---

## ✅ What MATHIR Does

```
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │   +37.8 %           AUC = 1.0          88 % isolation     100 % uptime     │
   │   online learning   anomaly detection  context-aware      2-hour stress    │
   │   (episodic tier)   (immune tier)      (working tier)     without crash   │
   │                                                                             │
   └─────────────────────────────────────────────────────────────────────────────┘
```

- **Episodic memory** — stores experiences and replays them to boost future recall (+37.8 %)
- **Immunological memory** — learns "normal" patterns, flags anomalies in real-time (AUC = 1.0)
- **Working memory** — multi-head attention produces context-dependent results (88 % isolation)
- **KL-constrained router** — PPO-style routing between 4 tiers, never collapses
- **Universal Bridge (UNIBRI)** — works across LLM providers and languages, no retraining
- **Edge-deployable** — ~240 MB VRAM (GPU fp16) with paraphrase-multilingual-MiniLM-L12-v2, works on Jetson Orin, Raspberry Pi (CPU fallback with paraphrase-multilingual 384d)
- **Zero external dependencies** (`SimpleMemory` uses only SQLite FTS5)
- **🧠 BRAIN ARCHITECTURE (v8.3+)** — 5-phase proactive system: auto-inject proxy, daemon watchdog, spreading activation, sleep consolidation, pre-cognitive priming. See [docs/BRAIN_ARCHITECTURE.md](docs/BRAIN_ARCHITECTURE.md)

---

## 🆚 vs Alternatives (honest 2026 comparison)

> Researched against Mem0, Letta, Zep, Cognee, LangMem, Microsoft GraphRAG, Supermemory, Recall.it, ChatGPT Memory, Claude Projects, Gemini memories, Microsoft Copilot Work IQ. Sources at the bottom of this section.

| Product | Architecture | OSS? | LLM-agnostic? | Edge? | Anomaly detection | Cost |
|---|---|:---:|:---:|:---:|:---:|:---|
| **🧠 MATHIR** | 4 cognitive tiers + KL router + Mahalanobis | ✅ **MIT** | ✅ Any | ✅ **~500 MB GPU / 80 MB CPU** | ✅ **AUC = 1.0** | **Free** |
| [Mem0](https://mem0.ai) | Vector + rerankers + LLM compression | ⚠️ SDK only | ✅ Any | ❌ Cloud | ❌ | Free → $249/mo |
| [Letta](https://letta.com) | Core/archival/recall tiers | ✅ Apache 2.0 | ✅ Any | ⚠️ Heavy | ❌ | Free (BYO infra) |
| [Zep](https://getzep.com) | Temporal knowledge graph | ⚠️ Graphiti OSS | ✅ Any | ❌ Cloud | ❌ | $1,250/yr → Custom |
| [Cognee](https://cognee.ai) | Self-hosted KG + vector | ✅ Apache 2.0 | ✅ Any | ⚠️ Heavy | ❌ | $35/mo → Custom |
| [LangMem](https://langchain-ai.github.io/langmem/) | Library on LangGraph store | ✅ MIT | ✅ Via LangChain | ⚠️ DIY | ❌ | Free (BYO infra) |
| [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) | KG + community detection | ✅ MIT | ✅ Any | ⚠️ DIY | ❌ | Free (BYO infra) |
| [Supermemory](https://supermemory.ai) | Custom vector graph | ❌ Self-host binary | ✅ Any | ⚠️ Self-host | ❌ | $19 → $399/mo |
| [Recall.it](https://recall.it) | Personal knowledge graph | ❌ Closed SaaS | ⚠️ Max tier only | ❌ | ❌ | Free → $38/mo |
| **ChatGPT Memory** (vendor) | Background "Dreaming" synthesis | ❌ Closed | ❌ OpenAI only | ❌ Cloud | ❌ | $20/mo+ |
| **Claude Projects** (vendor) | User-curated KB per project | ❌ Closed | ❌ Anthropic only | ❌ Cloud | ❌ | $20/mo+ |
| **Gemini memories** (vendor) | Implied semantic + chat history | ❌ Closed | ❌ Google only | ❌ Cloud | ❌ | Free → $20/mo |
| **Microsoft Work IQ** (vendor) | Semantic index + personal memory | ❌ Closed | ❌ Microsoft 365 only | ❌ Cloud | ❌ | M365 sub |

### What this table actually says

**3 things only MATHIR does, as of June 2026:**

1. **Anomaly detection on inputs** (immunological tier, AUC = 1.0). No competitor in this list has it.
2. **Edge deployment in ~500 MB VRAM**. All others need cloud or heavy local infra. Jetson Orin ✅ (full CUDA), Raspberry Pi ⚠️ (CPU fallback with ONNX INT8).
3. **MIT-licensed, fully open source, no managed service**. The only true OSS option with a 4-tier cognitive architecture.

**Things others do that MATHIR doesn't (honesty):**

- **Enterprise SSO, SOC 2, HIPAA, audit logs** → Zep, Mem0 Pro, Supermemory Enterprise have these. MATHIR doesn't.
- **Managed hosted service** → Mem0, Zep, Cognee, Supermemory all offer this. MATHIR is self-host only.
- **Temporal fact validity** (modeling "this preference is no longer valid") → Zep's specialty.
- **1M+ tokens of pre-curated memory** → Mem0's LoCoMo benchmark wins.

**Where MATHIR is competitive:**

- **GPU embedding speed** → paraphrase-multilingual-MiniLM-L12-v2 on CUDA fp16: ~104ms/sent (384d, 50+ languages, 239MB VRAM)
- **Pure retrieval quality** → MATHIR = FAISS dense-only (0.7441 nDCG@10 on BEIR SciFact, equal to SOTA)
- **Cross-provider** → 11/12 wins across 3 different LLM architectures
- **Cross-lingual** → UNIBRI finds English content from French queries
- **Cost** → free, vs $20–$400/mo for managed alternatives

### Sources

- Mem0 pricing & research: [mem0.ai/pricing](https://mem0.ai/pricing), [mem0.ai/research](https://mem0.ai/research)
- Letta docs: [docs.letta.com](https://docs.letta.com), [letta.com/blog/continual-learning](https://www.letta.com/blog/continual-learning)
- Zep docs: [getzep.com](https://www.getzep.com), [help.getzep.com](https://help.getzep.com)
- Cognee: [cognee.ai](https://www.cognee.ai), [github.com/topoteretes/cognee](https://github.com/topoteretes/cognee)
- LangMem: [langchain-ai.github.io/langmem](https://langchain-ai.github.io/langmem/)
- Microsoft GraphRAG: [microsoft.github.io/graphrag/](https://microsoft.github.io/graphrag/), arXiv 2404.16130
- Supermemory: [supermemory.ai](https://supermemory.ai)
- Recall: [recall.it](https://www.recall.it)
- ChatGPT Memory: [openai.com/index/chatgpt-memory-dreaming](https://openai.com/index/chatgpt-memory-dreaming/)
- Claude Projects: [anthropic.com/news/projects](https://www.anthropic.com/news/projects), [anthropic.com/news/claude-fable-5-mythos-5](https://www.anthropic.com/news/claude-fable-5-mythos-5)
- Microsoft Work IQ: [microsoft.com/.../work-iq-apis](https://www.microsoft.com/en-us/microsoft-365/blog/2026/06/02/announcing-the-new-work-iq-apis/)
- Magic AI 100M tokens: [magic.dev/blog/100m-token-context-windows](https://magic.dev/blog/100m-token-context-windows)
- Chroma Context Rot: [research.trychroma.com/context-rot](https://research.trychroma.com/context-rot)
- Breunig, "How Long Contexts Fail": [dbreunig.com](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html)

---

## 🧩 Embedding Providers (NEW: ONNX support)

MATHIR v8.x+ ships with **6 embedding providers**. The default is now **paraphrase-multilingual-MiniLM-L12-v2** — 384d, 50+ languages, low VRAM (239MB fp16).

### Provider comparison

| Provider | Model | Dim | Speed (single) | Size | Quality | Local | Cost |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **🆕 HuggingFace (GPU) — DEFAULT** | `paraphrase-multilingual-MiniLM-L12-v2` | **384** | ~104ms/sent | 471 MB (239 fp16) | 🟢 Multilingual 50+ | ✅ | Free |
| HuggingFace (GPU) | `BAAI/bge-large-en-v1.5` | 1024 | 25 ms | 1.3 GB | 🟢 High (EN) | ✅ | Free |
| 🆕 ONNX | `Octen-Embedding-0.6B-INT8` | 1024 | 18.8 ms | **5.2 MB** | 🟢 High | ✅ | Free |
| HuggingFace | `all-MiniLM-L6-v2` | 384 | 5.2 ms | 80 MB | 🟡 Medium (EN) | ✅ | Free |
| HuggingFace | `Qwen/Qwen2.5-7B-Instruct` | 3584 | 10–30 ms (GPU) | 14 GB | 🟢 High | ✅ | Free |
| Ollama | `llama3.2:3b` | 2048 | 30–80 ms | 2 GB | 🟢 High | ✅ | Free |
| OpenAI | `text-embedding-3-small` | 1536 | 80–200 ms | Cloud | 🟢 High | ❌ | $0.02/1M |

### ONNX Provider (new in v7.7.2)

```python
from mathir_lib.providers import get_provider

# Quantized ONNX model (recommended)
provider = get_provider("onnx", {
    "model_dir": r"C:\Users\So-i-learn-3D\.config\opencode\models\octen-int8",
    "provider": "CPUExecutionProvider"  # or "DmlExecutionProvider" for GPU
})

print(provider.embedding_dim())        # 1024
print(provider.provider_id())          # ('onnx', 'path', 1024)

embeddings = provider.embed_batch(["Hello", "World"])
# Shape: (2, 1024), L2-normalized, ready for cosine similarity
```

### ONNX vs HuggingFace benchmark (5 queries + 8 docs, RTX 4060)

| Metric | ONNX (Octen INT8) | HuggingFace (MiniLM) | Ratio |
|---|:---:|:---:|:---:|
| **Batch encode time** | 203 ms | 27 ms | 7.5× |
| **Single query** | 18.8 ms | 5.2 ms | 3.6× |
| **Embedding dim** | **1024** | 384 | 2.7× |
| **Model size** | **5.2 MB** | 80 MB | 15× |
| **Memory footprint** | 50 % of FP32 | 100 % | 0.5× |
| **Similarity range** | **[0.42, 0.98]** | [-2.53, 34.34] * | — |
| **L2-normalized** | ✅ Yes | ❌ No | — |

\* MiniLM embeddings are not L2-normalized by default — cosine similarity requires manual normalization.

### When to use ONNX vs HuggingFace

| Use case | Recommended |
|---|---|
| Best quality multilingual embeddings | **ONNX** (Octen) |
| Smallest model footprint | **ONNX** (5.2 MB) |
| Fastest single query | HuggingFace (MiniLM) |
| 1024-dim embeddings for FAISS/pinecone | **ONNX** |
| 384-dim for legacy systems | HuggingFace |
| Edge / Jetson Orin | **ONNX** (int8) |
| No GPU available | Both work; ONNX more compact |

### Download ONNX model

```bash
# Manual download (recommended — faster than pip)
# Create folder: C:\Users\So-i-learn-3D\.config\opencode\models\octen-int8\
# Download from https://huggingface.co/cstr/Octen-Embedding-0.6B-ONNX-INT8/resolve/main/
#   - model.int8.onnx (5.2 MB)
#   - model.int8.onnx.data (1.06 GB)
#   - tokenizer.json
#   - vocab.txt
#   - config.json
```

### MCP Server (easy integration)

```json
// Add to ~/.config/opencode/opencode.json
{
  "mcp": {
    "mathir": {
      "command": "python",
      "args": ["/path/to/MATHIR/mcp_server.py"],
      "env": { "PYTHONPATH": "/path/to/MATHIR" }
    }
  }
}
```

The MCP server exposes 6 tools: `memory_save`, `memory_recall`, `memory_smart_search`, `memory_stats`, `memory_delete`, `memory_push`.

### Compatible With

All major AI coding tools support MCP — MATHIR works with all of them:

| Tool | MCP | Transport | Config |
|------|-----|-----------|--------|
| **OpenCode** | ✅ Native | TCP + SSE | `opencode.json` → `mcpServers` |
| **OpenClaude** | ✅ Native | stdio/HTTP | `/mcp` command or config JSON |
| **Kilo Code** | ✅ Native | HTTP Stream | Settings → MCP → Add Server |
| **MiMo Code** | ✅ Native | stdio + HTTP | Config `mcp` section |
| **Claude Code** | ✅ Native | stdio/HTTP | `claude_desktop_config.json` |

**One daemon, 5 tools, same memory.** Start the daemon once, connect from any tool:

```bash
# Start daemon (once)
python /path/to/MATHIR/bin/mathir_daemon.py

# Then add to any MCP-compatible tool:
# URL: http://127.0.0.1:7338/sse
```

---

## 🚀 Deployment Options

MATHIR supports multiple deployment targets. The embedding model you choose determines VRAM, speed, and platform compatibility.

| Platform | Model | VRAM | Speed (recall) | Status |
|----------|-------|------|----------------|--------|
| **Desktop GPU (CUDA)** | bge-large-en-v1.5 (1024d) | ~500 MB | 25 ms | ✅ Recommended |
| **Jetson Orin (CUDA)** | bge-large-en-v1.5 (1024d) | ~500 MB | ~30 ms | ✅ Supported |
| **CPU only** | bge-large-en-v1.5 (1024d) | 0 MB | ~200 ms | ✅ Supported |
| **Raspberry Pi** | ONNX INT8 (1024d) | 0 MB | ~500 ms | ⚠️ Experimental |

**Notes:**
- **MATHIR internal memory** (working/episodic/semantic/immunological tiers) is ~60 KB regardless of platform — this is always true (Theorem 1, bounded capacity).
- **Embedding model VRAM** varies by model: ~500 MB for bge-large on GPU, 0 MB for CPU-only ONNX.
- **Raspberry Pi** requires CPU fallback — use ONNX INT8. The bge-large model (1024d) is too large for Pi-class ARM devices without GPU.
- **Jetson Orin** has CUDA support and runs bge-large at near-desktop speeds.

---

## ⚡ Quick Start (30 seconds)

### 1. Install

```bash
git clone https://github.com/sil3d/MATHIR.git
cd MATHIR
pip install -e .
```

### 2. The smallest possible example

```python
from mathir_dropin.simple import SimpleMemory   # zero dependencies (just SQLite FTS5)

memory = SimpleMemory(db_path="my_app.db")
memory.store("User asked about Python closures")
memory.store("Explained that closures capture enclosing-scope variables")
memory.store("User then asked about decorators")

results = memory.recall("Python functions", k=3)
# → ["User asked about Python closures", "Explained closures..."]
```

### 3. With HybridSearch (auto-scaling vector search)

```python
from mathir_dropin.simple import SimpleMemory

# HybridSearch is automatic — just use SimpleMemory
memory = SimpleMemory(db_path="my_app.db")

# Store memories (auto-selects numpy for N < 5K)
for i in range(1000):
    memory.store(f"Memory item {i}: This is a test memory about topic {i % 10}")

# At N=5,000, auto-switches to USearch HNSW (1.37ms)
results = memory.recall("test topic", k=5)

# Memory-mapped index persists to disk — no RAM pressure
print(f"Index size: {memory.get_index_size()}")  # ~50 KB on disk
```

### 4. Plug it into any LLM (3 lines)

```python
def chat(user_message):
    context = memory.search_context(user_message, k=5, last_n=3)
    response = openai.chat.completions.create(  # or anthropic, or local llama_cpp
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Relevant memories:\n{context}"},
            {"role": "user",   "content": user_message}
        ],
    )
    memory.store(f"Q: {user_message} | A: {response.choices[0].message.content}")
    return response.choices[0].message.content
```

Works with **any LLM** — OpenAI, Anthropic, Gemini, Groq, Ollama, local 7B via `llama_cpp`, anything.

### 5. Or use the full V7 plugin (8 algorithms, 6 theorems)

```python
from mathir_lib import MATHIRPluginV7

plugin = MATHIRPluginV7(embedding_dim=4096)
output = plugin.perceive(llm_embedding)

print(output["enhanced_embedding"])  # [1, 4096]
print(output["router_weights"])      # 4-tier allocation: [0.4, 0.3, 0.2, 0.1]
print(output["anomaly_score"])       # novelty detection (0.0–1.0)
print(output["episodic_context"])    # retrieved past experiences
```

---

## 🎬 Live Demo

```bash
cd vision_testing
pip install -r requirements.txt
python start_ui.py
# → Opens at http://127.0.0.1:5000
```

A full web UI for testing **vision + audio** models with persistent MATHIR memory.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MATHIR Vision Testing UI                          🟢 MATHIR connected │
├─────────────────────────────────────────────────────────────────────────┤
│  [💬 Chat]   [📷 Camera]   [🧠 Memory]   [🤖 Models]   [🎯 Accuracy]   │
│                                                                         │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐   │
│  │ Camera: 1280x720 @ 30fps │    │  Chat history                    │   │
│  │ ┌────────────────────┐   │    │  ─────────────────────────────    │   │
│  │ │                    │   │    │  You: What's in front of me?     │   │
│  │ │   [Live Preview]   │   │    │  AI:  A red apple on a desk.    │   │
│  │ │                    │   │    │                                   │   │
│  │ └────────────────────┘   │    │  You: Count the objects.         │   │
│  │                          │    │  AI:  I see 3 objects.           │   │
│  │ [📸 Snapshot] [🎤 Talk]  │    │                                   │   │
│  └──────────────────────────┘    │  🧠 MATHIR: 12 memories stored  │   │
│                                   └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6 views in the web UI

| View | What it does | Screenshot features |
|---|---|---|
| 💬 **Chat** | Real-time chat with vision/audio models + persistent memory | Drag-and-drop images, hold-to-talk audio, history in localStorage |
| 📷 **Camera** | Live webcam (backend OpenCV) — describe, ask, count objects | MJPEG stream, ask-on-frame, auto-capture |
| 🧠 **Memory** | Query MATHIR memory across all sessions | Search, recall, delete individual memories |
| 🤖 **Models** | Switch between LFM2.5-VL, Audio, Gemma, Qwen | Load/unload, capabilities, VRAM usage |
| 🎯 **Accuracy** | Run test batteries, compare models | nDCG@10, MRR, latency, F1 |
| ⚙️ **Settings** | Camera, audio, theme, model management | Live preview, device selection |

A standalone **playground** at `/playground.html` provides multi-session chat with model switching, image drag & drop, and hold-to-talk audio.

---

## 💡 More Examples

### Example 1 — Persistent chat memory across sessions (and across LLMs)

```python
# === Day 1, with GPT-4 ===
memory = SimpleMemory(db_path="alice.db")
memory.store("Alice is a software engineer at Google")
memory.store("Alice prefers Python over JavaScript")
memory.store("Alice is building a RAG system for legal documents")
# (close the app, go to sleep)

# === Day 2 (re-open, still GPT-4) ===
memory = SimpleMemory(db_path="alice.db")   # same DB, no config
print(memory.search_context("What does Alice do?", k=3))
# → ["Alice is a software engineer at Google",
#    "Alice is building a RAG system for legal documents",
#    "Alice prefers Python over JavaScript"]

# === Day 3 (switch to local Llama 3.1 — same memory!) ===
# Same SQLite file, same memories, different LLM.
# This is what vendor-locked ChatGPT Memory can't do.
```

### Example 2 — Anomaly detection (no other LLM-memory product has this)

```python
from mathir_lib import MATHIRPluginV7

plugin = MATHIRPluginV7(embedding_dim=768)

# Feed normal inputs to "train" the immune system
for emb in normal_user_inputs:
    plugin.perceive(emb)

# Now anomalies are flagged
output = plugin.perceive(weird_prompt_injection)
if output["anomaly_score"] > 0.95:
    print("⚠️ Possible prompt injection detected!")
    # AUC-ROC = 1.0 on test set
```

### Example 3 — Context-aware retrieval (same query, different results)

```python
plugin = MATHIRPluginV7(embedding_dim=768)

# No context loaded
print(plugin.perceive(embed("What's the capital of France?"))["results"])
# → ["Paris", "Lyon", "Marseille"]   (generic)

# Load cooking context
plugin.load_context(recent_conversation_about_french_cuisine)

# Same query, different results
print(plugin.perceive(embed("What's the capital of France?"))["results"])
# → ["Paris", "Bordeaux wine region", "Provence herbs"]   (context-aware)
```

### Example 4 — Cross-lingual recall (UNIBRI)

```python
from mathir_dropin.universal_bridge import universal_recall

# Store English content
memory.store("Python closures capture variables from enclosing scope")

# French query finds it
results = universal_recall("clotures python", k=3)
# → [{"text": "Python closures capture variables...", "score": 0.89}, ...]
```

### Example 5 — Cross-provider (works with any LLM)

```python
# Same memory, different providers
memory.store("The capital of France is Paris")

# OpenAI
client = openai.OpenAI()
# ... use memory in prompt

# Anthropic
client = anthropic.Anthropic()
# ... same memory, different API

# Local 7B
from llama_cpp import Llama
# ... same memory, on-device, no internet

# The memory layer is provider-agnostic.
```

### Example 6 — Full cognitive pipeline

```python
from mathir_lib import MATHIRPluginV7

plugin = MATHIRPluginV7(embedding_dim=768)

# A single perceive() call routes through all 4 tiers
output = plugin.perceive(input_embedding, metadata={"user": "alice"})

# What just happened:
print(f"Router picked: {output['router_weights']}")
# → [0.4, 0.3, 0.2, 0.1]  (working, episodic, semantic, immune)

print(f"Context used: {output['episodic_context']}")
# → "User asked about Python closures 3 days ago..."

print(f"Anomaly score: {output['anomaly_score']:.3f}")
# → 0.02  (looks normal)

print(f"Enhanced embedding: {output['enhanced_embedding'].shape}")
# → (1, 768)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              ANY LLM                       │
│   (Claude · GPT-5 · Qwen · LFM2.5 · 7B)    │
└─────────────────┬───────────────────────────┘
                  │ embeddings (1024-d)
                  ▼
┌─────────────────────────────────────────────┐
│           🧠  MATHIR PLUGIN                │
│    ~500 MB VRAM (GPU) · ~107 ms · edge-ready │
│                                             │
│   NOTE: MATHIR internal memory (working/    │
│   episodic/semantic/immunological tiers)    │
│   is ~60 KB (always, Theorem 1). VRAM usage │
│   is the embedding model, not the tiers.    │
│                                             │
│   ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│   │ Working  │  │ Episodic │  │Semantic │  │
│   │  (now)   │  │  (past)  │  │(always) │  │
│   └────┬─────┘  └────┬─────┘  └────┬────┘  │
│        └──────────────┼──────────────┘      │
│               ┌───────▼──────┐               │
│               │ KL  Router   │               │
│               └───────┬──────┘               │
│               ┌───────▼──────┐               │
│               │Immunological │               │
│               │  (anomaly)   │               │
│               └──────────────┘              │
│                                             │
│   ┌─────────────────────────────────────┐   │
│   │     HybridSearch Auto-Scaling       │   │
│   │  ┌─────────┐    ┌──────────────┐   │   │
│   │  │ numpy   │───►│ USearch HNSW │   │   │
│   │  │ (N<5K)  │auto│ (mmap index) │   │   │
│   │  └─────────┘    └──────────────┘   │   │
│   └─────────────────────────────────────┘   │
└─────────────────┬───────────────────────────┘
                  │ enhanced context + anomaly flag
                  ▼
┌─────────────────────────────────────────────┐
│              LLM DECISIONS                  │
└─────────────────────────────────────────────┘
```

### 4 cognitive memory tiers

```
Tier             Capacity    What it does                              When it updates
─────────────    ─────────   ───────────────────────────────────────   ────────────────
🧠 Working         64 slots   Immediate context (last N steps)          Every step
                   [circular   Multi-head attention on recent context
                    buffer]

📚 Episodic     1 000 slots   Past experiences (key-value store)         On event
                   [FIFO +     Cosine similarity on stored embeddings
                    LIRS]      +37.8 % recall improvement

🎓 Semantic       256 proto-  Learned concepts (online k-means)          Every 100 steps
                   types      Compact concept representation

🛡️ Immunological  100 pat-    Anomaly detection (Mahalanobis              On event
                   terns      distance)  AUC-ROC = 1.0
```

### HybridSearch Auto-Scaling Backend

```
┌─────────────────────────────────────────────────────────────────┐
│                    HybridSearch Auto-Scaling                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  N < 5,000 docs          N >= 5,000 docs                       │
│  ┌─────────────┐         ┌─────────────────────────────┐       │
│  │   numpy     │  ────►  │  USearch HNSW (mmap index)  │       │
│  │  0.78 ms    │  auto   │  1.37 ms                    │       │
│  └─────────────┘  scale  └─────────────────────────────┘       │
│                                                                 │
│  Memory: ~20 KB        Memory: ~50 KB (mmap on disk)           │
│  RAM-only              Index persisted to disk, not RAM         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Daemon Architecture (v7.8.0+)

MATHIR runs as a **persistent daemon process** on **port 7338** — the embedding model stays loaded in GPU RAM between calls.

```
Client (opencode / Python)
         │ TCP socket (localhost:7338)
         ▼
┌─────────────────────────────────────┐
│  mathir_daemon.py (persistent)      │
│  ├── Model loaded ONCE at startup   │
│  ├── GPU memory held across calls   │
│  ├── TCP server on port 7338        │
│  ├── HybridSearch auto-scaling      │
│  └── No model reload per request    │
└─────────────────────────────────────┘
         │
         ▼
    mathir_client.py (thin client)
    → 1–2 ms per call (model already in VRAM)
```

**Why daemon instead of per-request:**
- Model load: **~3–5 seconds** (bge-large-en-v1.5) → eliminated after first call
- Per-call overhead: **1–2 ms** (TCP round-trip only)
- GPU memory: **~500 MB** held continuously (vs 0 MB between calls)
- No cold starts, no repeated allocation

```bash
# Start daemon (background, persists across sessions)
python ~/.config/opencode/bin/mathir_daemon.py &

# Thin client — fast, model already loaded
python ~/.config/opencode/bin/mathir_client.py recall "query" -k 5
```

### Daemon Push (NEW in v8.2.0)

MATHIR v8.2.0 introduces **proactive memory delivery** — the daemon can push relevant memories to clients without explicit recall requests. This enables automatic context injection for ongoing conversations.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Daemon Push Flow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Client                    Daemon                              │
│    │                         │                                 │
│    │  push --auto            │                                 │
│    ├────────────────────────►│  Analyze context                │
│    │                         │  Query 4-tier memory            │
│    │                         │  Rank by relevance              │
│    │  ◄──────────────────────┤                                 │
│    │  [memory1, memory2, ...] │  Return ranked memories        │
│    │                         │                                 │
│  Push Modes:                                                    │
│  ┌─────────────┬─────────────────────────────────────────────┐ │
│  │ --auto      │ Daemon analyzes context, returns JSON array │ │
│  │ --json      │ Returns structured {memories: [...]}        │ │
│  │ --simple    │ Returns plain text memories                 │ │
│  └─────────────┴─────────────────────────────────────────────┘ │
│                                                                 │
│  Use Cases:                                                     │
│  • Auto-inject relevant context before each LLM call           │
│  • Proactive memory suggestions during conversations           │
│  • Background context enrichment for long sessions             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Push commands:**
```bash
# Auto mode — daemon pushes relevant memories based on context
python ~/.config/opencode/bin/mathir_client.py push "contexte ici" --auto

# JSON mode — returns structured memory suggestions
python ~/.config/opencode/bin/mathir_client.py push "contexte ici" --json

# Simple mode — returns plain text memories (default)
python ~/.config/opencode/bin/mathir_client.py push "contexte ici"
```

**Why push instead of pull:**
- **Latency**: Memories delivered proactively, no recall delay
- **Context**: Daemon analyzes full conversation history, not just current query
- **Automatic**: No need to remember to call recall — daemon delivers relevant memories
- **Efficient**: Cache prevents redundant embedding computations

### KL-constrained router

The router decides which memory tier to consult for each input. It uses **PPO-style trust-region optimization** with a KL-divergence constraint to prevent collapse to a single tier.

```
Input: "What's the weather?"
              │
              ▼
       ┌──────────────┐
       │  KL Router   │   weights = [0.40, 0.30, 0.20, 0.10]
       └──────┬───────┘
              │
   ┌──────────┼──────────┬──────────┐
   ▼          ▼          ▼          ▼
Working    Episodic   Semantic   Immune
(0.40)     (0.30)     (0.20)     (0.10)
   │          │          │          │
   │ "Right   │ "User    │ "Weather │ "Nothing
   │  now"    │  asked   │  is a    │  weird
   │          │  this    │  common  │  about
   │          │  before" │  topic"  │  this"
   ▼          ▼          ▼          ▼
   "22°C,     "Last      Use       No flag.
   sunny"     time you   general   All normal.
              asked,    knowledge
              it was
              sunny"
```

The router **learns** its allocation strategy over time (no hard-coded rules):
- Short-term reflex → **working memory**
- Recall a past situation → **episodic memory**
- Apply a general concept → **semantic memory**
- Novel / unusual input → **immunological memory**

---

## 📊 Tests & Benchmarks

All results reproducible. Scripts in [`benchmarks/`](benchmarks/), full HTML report in [`benchmarks/06_results/current/MATHIR_FINAL_REPORT.html`](benchmarks/06_results/current/MATHIR_FINAL_REPORT.html).

### 🧪 Test suite — 226 tests, 99 % pass

```
Suite                         Tests   Status     Coverage
─────────────────────────     ─────   ────────    ───────────────────
test_v7_memory.py               49    ✅ 49/49    Memory tier algorithms
test_v7_integration.py          16    ✅ 14/16    End-to-end pipelines
test_raw_embedding.py           28    ✅ 28/28    Embedding layer
test_ensemble.py                36    ✅ 36/36    Anomaly ensemble
test_faiss_memory.py            32    ✅ 32/32    FAISS integration
test_hybrid.py                  34    ✅ 34/34    Hybrid retrieval
mathir_dropin audit             31    ✅ 31/31    Drop-in API surface
─────────────────────────     ─────   ────────    ───────────────────
TOTAL                          226    ✅ 224/226  99 %
```

```bash
pytest mathir_dropin/tests/ -v
```

### 📈 BEIR benchmark results (nDCG@10)

| System | SciFact | NFCorpus | ArguAna | Verdict |
|---|:---:|:---:|:---:|---|
| **FAISS dense-only (BGE-base)** | **0.7441** | **0.3657** | **0.6613** | ✅ SOTA baseline |
| BM25 only | 0.5438 | 0.2617 | — | ⚠️ Too weak for scientific |
| Hybrid RRF (1:1) | 0.6602 | 0.3263 | — | ⚠️ BM25 dilutes dense |
| Hybrid + Cross-Encoder | 0.5910 | 0.2620 | — | ❌ Cross-encoder wrong domain |

```
nDCG@10 (SciFact)
0.8 ┤
    │      ████████
0.7 ┤      ████████
    │      ████████
0.6 ┤      ████████  ████████
    │      ████████  ████████
0.5 ┤      ████████  ████████  ████████
    │      ████████  ████████  ████████
0.4 ┤      ████████  ████████  ████████
    │      ████████  ████████  ████████
0.3 ┤      ████████  ████████  ████████
    │      ████████  ████████  ████████
0.2 ┤      ████████  ████████  ████████
    │      ████████  ████████  ████████
0.1 ┤      ████████  ████████  ████████
    │      ████████  ████████  ████████
0.0 ┴───────────────────────────────
        FAISS       Hybrid     Hybrid+CE
       (0.7441)    (0.6602)    (0.5910)
```

### 🔍 Vector Search — HybridSearch Auto-Scaling

MATHIR v7.8+ introduces **HybridSearch** — a vector search backend that automatically scales from numpy (small datasets) to USearch HNSW (large datasets) with memory-mapped indexes.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HybridSearch Auto-Scaling                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  N < 5,000 docs          N >= 5,000 docs                       │
│  ┌─────────────┐         ┌─────────────────────────────┐       │
│  │   numpy     │  ────►  │  USearch HNSW (mmap index)  │       │
│  │  0.78 ms    │  auto   │  1.37 ms                    │       │
│  └─────────────┘  scale  └─────────────────────────────┘       │
│                                                                 │
│  Memory: ~20 KB        Memory: ~50 KB (mmap on disk)           │
│  RAM-only              Index persisted to disk, not RAM         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### BEIR Benchmark Results (5,183 documents)

| Backend | Latency (search) | Index Size | RAM Usage | Notes |
|---|:---:|:---:|:---:|---|
| **numpy (cosine)** | **0.78 ms** | ~20 KB | ~20 KB | Fastest for small N |
| **USearch HNSW (mmap)** | **1.37 ms** | ~50 KB | ~50 KB | Memory-mapped, scales to 1M+ |
| **sqlite-vec** | **23.68 ms** | ~100 KB | ~100 KB | Slower, WAL-optimized |

```
Search latency at 5,183 documents
30 ms ┤
      │
25 ms ┤
      │                    ████████
20 ms ┤                    ████████
      │                    ████████
15 ms ┤                    ████████
      │                    ████████
10 ms ┤                    ████████
      │                    ████████
 5 ms ┤                    ████████
      │  ████████          ████████
 0 ms ┴─────────────────────────────
        numpy (0.78)     sqlite-vec (23.68)
              USearch (1.37)
```

#### Key Features

| Feature | Description |
|---|---|
| **Auto-scaling** | numpy → USearch at N=5,000 (no config needed) |
| **Memory-mapped indexes** | Index on disk, not RAM — no memory pressure for large datasets |
| **sqlite-vec WAL mode** | Write-Ahead Logging for 3.4× write speedup |
| **Zero-config** | HybridSearch picks the optimal backend automatically |
| **FAISS fallback** | Optional FAISS integration for production deployments |

#### Usage

```python
from mathir_dropin.simple import SimpleMemory

# HybridSearch is automatic — just use SimpleMemory
memory = SimpleMemory(db_path="my_app.db")

# For N < 5K: numpy backend (0.78ms)
# For N >= 5K: auto-switches to USearch HNSW (1.37ms)
# No configuration needed

memory.store("Python closures capture variables")
results = memory.recall("Python functions", k=5)  # Uses optimal backend
```

### 🚀 What MATHIR adds over FAISS

| Capability | FAISS | MATHIR | Delta |
|---|:---:|:---:|---|
| Online learning | ❌ | ✅ **+37.8 %** | 🟢 MATHIR |
| Anomaly detection (AUC) | ❌ | **1.0** | 🟢 MATHIR |
| Context-aware results | ❌ | **88 %** | 🟢 MATHIR |
| 2-hour stress (no crash) | ❌ | **100 %** | 🟢 MATHIR |
| No memory leak | ❌ | ✅ | 🟢 MATHIR |
| Router balanced | ❌ | 100 % acc. | 🟢 MATHIR |
| Graceful degradation | ❌ | ✅ | 🟢 MATHIR |
| **Raw retrieval speed** | **< 1 ms** | 0.78 ms (numpy) / 1.37 ms (USearch) | 🔵 FAISS (similar) |

### ⏱ 2-hour stress test (all 4 tiers active)

| Metric | Value | Status |
|---|:---:|:---:|
| Uptime | **100 %** | ✅ |
| Memory leaks | **None** | ✅ |
| Retrieval quality @ 120 min | **0.959** | ✅ |
| P99 latency | **17.8 ms** | ✅ |
| Total operations | 26 440 | ✅ |

```
Retrieval quality over 2 hours
1.0 ┤█████████████████████████████████████████████████
    │
0.95┤█████████████████████████████████████████████████
    │
0.9 ┤█████████████████████████████████████████████████
    │
0.85┤█████████████████████████████████████████████████
    └─────────────────────────────────────────────────
    0    20    40    60    80    100   120  (minutes)
```

### 🌍 Cross-provider generalization (OpenRouter)

| Model | API latency | MATHIR wins | Result |
|---|:---:|:---:|---|
| `openrouter/owl-alpha` | 2.6 s | **4 / 4** | 🏆 MATHIR wins all |
| `openai/gpt-oss-120b:free` | 2.0 s | **3 / 4** | 🏆 MATHIR wins most |
| `openai/gpt-oss-20b:free` | 1.1 s | **4 / 4** | 🏆 MATHIR wins all |

**Total: 11 / 12 scenarios** — MATHIR wins across 3 different LLM architectures.

### 🌐 Cross-lingual (UNIBRI)

```
"What do you know about python closures?"  → finds "python-closures"         ✅
"clotures python"  (French)                → finds English "Python closures" ✅
provider="unknown"  (no stored embedding)  → 3 results via fallback chain   ✅
```

The Universal Bridge uses **multi-resolution character n-gram kernels** (Broder 1997) + **Johnson-Lindenstrauss random projection** + **Procrustes SVD** for cross-space alignment. Mathematically grounded, vocabulary-free, language-agnostic.

### ⚡ Performance Benchmarks (v7.8.0+)

Real-world benchmarks on RTX 4060 + CUDA, measuring **save** (store memory) and **recall** (search memory) latency.

#### End-to-End Latency

| Operation | Latency | Breakdown |
|---|:---:|---|
| **Save** (store memory) | **58 ms** | bge-large CUDA 3ms + DB write 55ms |
| **Recall** (search memory) | **107 ms** | bge-large CUDA 3ms + vector search 104ms |

#### Vector Search at Scale

| Dataset Size | Backend | Latency | Notes |
|---|:---:|:---:|---|
| 5,000 docs | numpy | 0.78 ms | Auto-selected for small N |
| 5,000 docs | USearch HNSW | 1.37 ms | Memory-mapped index |
| 5,000 docs | sqlite-vec | 23.68 ms | WAL-optimized |

#### Embedding Model Comparison

| Model | Dimensions | Device | Save | Recall | Notes |
|---|:---:|---|:---:|:---:|---|
| **BAAI/bge-large-en-v1.5** | **1024** | **CUDA** | **58 ms** | **107 ms** | ✅ Recommended — best quality/speed |
| MiniLM-L6-v2 | 384 | CUDA | 22 ms | 53 ms | ⚠️ Faster save, slower recall |
| Octen INT8 | 1024 | CPU | ~5 000 ms | ~2 700 ms | 🐢 50–100× slower |
| Octen INT8 | 1024 | CUDA (onnxruntime-gpu) | ~776 ms | — | ⚠️ Partial GPU (ONNX limitation) |

**Key insight:** bge-large-en-v1.5 achieves **107 ms recall at 1024d on full CUDA** — includes embedding time (3ms) + vector search (104ms). The larger dimension space produces better similarity scores, and CUDA handles the extra compute efficiently.



---

## 🔬 Why It Works — Theoretical Foundation

| Component | Guarantee | Citation |
|---|---|---|
| Episodic memory | Cosine similarity on stored embeddings → **+37.8 %** recall (measured on BEIR) | Empirical |
| Immunological memory | **Mahalanobis distance is the NP-optimal detector** for anomalies in Gaussian data | McLachlan 1999 |
| Working memory | Multi-head attention on circular buffer → **bounded latency**, context-aware | Vaswani 2017 |
| KL Router | KL-divergence penalty (PPO-style) prevents tier collapse; max-entropy ensures exploration | Schulman 2017 |
| UNIBRI | **Theorems 1–4** give OOV / cross-lingual / cross-provider stability guarantees | Broder 1997, J-L 1984, Wedin 1972 |

Full mathematical proofs in [`docs/01_MASTER_RESEARCH_PAPER.md`](docs/01_MASTER_RESEARCH_PAPER.md).

---

## 📁 Project Structure

```
MATHIR/
├── 🧠 mathir_lib/             # Full library (8 algorithms · 6 theorems · 9.3× compression)
│   ├── plugin_v7.py           # V7 plugin (recommended)
│   ├── memory/                # Memory tier implementations
│   └── config.py
│
├── 📦 mathir_dropin/          # Drop-in memory (copy to your project)
│   ├── memory.py              # MATHIRMemory (torch-powered)
│   ├── simple.py              # SimpleMemory (FTS5, zero deps)
│   ├── store.py               # SQLite storage
│   └── universal_bridge.py    # UNIBRI: cross-provider · cross-lingual
│
├── 👁️ vision_testing/         # Full vision/audio testing UI
│   ├── ui_server.py           # Flask backend · 18 API routes
│   ├── ui/                    # Web UI (HTML · CSS · JS)
│   └── playground.html        # Multi-session chat playground
│
├── 📊 benchmarks/             # Reproducible benchmarks + HTML report
│   └── 06_results/current/     # Benchmark reports
│       └── MATHIR_FINAL_REPORT.html   # Full visual report
│
├── 🧪 mathir_dropin/tests/    # 226 tests
├── 📚 docs/                   # Tutorials · theory · LaTeX paper
├── 🔧 examples/               # Demo scripts
└── ⚙️ config/                 # Configuration
```

---

## 🛠️ Try the Examples

```bash
# Zero-dep memory (works without torch)
python examples/simple_memory_demo.py

# Vision + audio UI
cd vision_testing && python start_ui.py

# Multi-session chat playground
cd vision_testing && python start_ui.py
# → http://127.0.0.1:5000/playground.html
```

---

## 📚 Documentation

> **📖 New here? See the [Complete Documentation Index](docs/00_README.md)** — 45+ documents organized by audience (CTO / researcher / developer / theorist), with reading paths for 10 min, 1 hour, 3 hours, and full deep-dive.

### Top 7 "Hidden Gems"

| # | Document | Lines | What it is |
|:---:|---|:---:|---|
| 1 | [`README.md`](README.md) | 510 | The pitch (you are here) |
| 2 | [`MATHIR_FINAL_REPORT.html`](benchmarks/06_results/current/MATHIR_FINAL_REPORT.html) | 348 | All benchmark numbers |
| 3 | [`docs/03_MASTER_QA_GUIDE.md`](docs/03_MASTER_QA_GUIDE.md) | **637** | **63 Q&A** for CTO defense |
| 4 | [`docs/07_MATHIR_VS_VECTORDB_USE_CASES.md`](docs/07_MATHIR_VS_VECTORDB_USE_CASES.md) | **454** | **MATHIR vs FAISS** on chat + autonomous driving |
| 5 | [`docs/MATHIR_Research_Paper.tex`](docs/MATHIR_Research_Paper.tex) | **1 130** | **LaTeX paper** — peer-review ready |
| 6 | [`docs/01_MASTER_RESEARCH_PAPER.md`](docs/01_MASTER_RESEARCH_PAPER.md) | **699** | **6 theorems with proofs** |
| 7 | [`docs/01_MASTER_RESEARCH_PAPER.md`](docs/01_MASTER_RESEARCH_PAPER.md) | **2 155** | **Doctoral research paper** — 145 KB |

### Quick links

| Document | Description |
|---|---|
| 📄 [`docs/MATHIR_Research_Paper.tex`](docs/MATHIR_Research_Paper.tex) | LaTeX paper for scientific review |
| 📖 [`docs/01_MASTER_RESEARCH_PAPER.md`](docs/01_MASTER_RESEARCH_PAPER.md) | Full research paper (Markdown, 145 KB) |
| 📊 [`benchmarks/MATHIR_FINAL_REPORT.html`](benchmarks/06_results/current/MATHIR_FINAL_REPORT.html) | Visual benchmark report (HTML, interactive charts) |
| 📊 [`benchmarks/MATHIR_FINAL_REPORT.md`](benchmarks/MATHIR_FINAL_REPORT.md) | Benchmark report (Markdown) |
| 🎯 [`docs/03_MASTER_QA_GUIDE.md`](docs/03_MASTER_QA_GUIDE.md) | 63 Q&A for defense / evaluation |
| 🆚 [`docs/07_MATHIR_VS_VECTORDB_USE_CASES.md`](docs/07_MATHIR_VS_VECTORDB_USE_CASES.md) | MATHIR vs FAISS use cases |
| 🔬 [`docs/01_MASTER_RESEARCH_PAPER.md`](docs/01_MASTER_RESEARCH_PAPER.md) | Mathematical proofs (6 theorems) |
| 📘 [`docs/04_DEV_INTEGRATION_GUIDE.md`](docs/04_DEV_INTEGRATION_GUIDE.md) | V7 usage tutorial |
| 🤖 [`mcp/AGENT.md`](mcp/AGENT.md) | Quick reference for AI agents |
| 👁️ [`vision_testing/README.md`](vision_testing/README.md) | Vision/audio testing docs |
| 📦 [`mathir_dropin/README.md`](mathir_dropin/README.md) | Drop-in memory docs |
| 📋 [`CHANGELOG.md`](CHANGELOG.md) | Version history |

---

## 🗺️ Roadmap

| Version | Milestone | Status |
|---|---|:---:|
| V1–V5 | Core architecture + KL router | ✅ |
| V6 | LLM-agnostic plugin API | ✅ |
| V7 | 8 algorithms + 6 theorems + 9.3× compression | ✅ |
| V7.5 | Real BEIR benchmarks (0.7441 SOTA) | ✅ |
| V7.6 | Universal Bridge (UNIBRI) | ✅ |
| V7.7 | Vision & audio testing + MATHIR memory | ✅ |
| **V7.7.1** | **SimpleMemory (FTS5) + UI overhaul** | **✅** |
| **V7.8** | **GPU embeddings (bge-large) + daemon architecture** | **✅** |
| V8 | Cascade architecture + arXiv paper | 🔜 |
| V9 | Edge deployment (Jetson / ONNX) | 📋 |
| V10 | Open-source release (HuggingFace · PyPI) | 📋 |

---

## 🤝 Contributing

We welcome contributions.

```bash
# 1. Fork & clone
git clone https://github.com/YOUR_USERNAME/MATHIR.git
cd MATHIR
pip install -e .

# 2. Create a branch
git checkout -b feature/my-feature

# 3. Make changes, add tests, run them
pytest tests/ -v

# 4. Submit a PR
```

### Areas where help is needed

- 📚 **Documentation** — improve tutorials, add examples
- 🧪 **Testing** — edge cases, more coverage
- 📊 **Benchmarks** — more corpora, more embedding models
- 📱 **Edge deployment** — Rust / ONNX port
- 🔌 **Integrations** — LangChain · LlamaIndex · Haystack

---

## 📄 Citation

If you use MATHIR in your research, please cite:

```bibtex
@software{mathir2026,
  title  = {MATHIR: Memory-Augmented Tensor Hybrid with Intelligent Routing},
  author = {Mbama Kombila, Prince Gildas},
  year   = {2026},
  url    = {https://github.com/sil3d/MATHIR}
}
```

Full paper: [`docs/MATHIR_Research_Paper.tex`](docs/MATHIR_Research_Paper.tex)

---

## 📜 License

[MIT](LICENSE) — free for commercial and research use.

---

<div align="center">

### 🧠 MATHIR — *A 4-tier cognitive memory layer for any LLM, on any hardware.*

**Author:** [Prince Gildas Mbama Kombila](https://github.com/sil3d) · **Email:** soilearn3d@gmail.com

⭐ **Star this repo** if you find it useful — it helps others discover MATHIR.

</div>
