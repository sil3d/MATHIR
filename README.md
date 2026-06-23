<div align="center">

<img src="docs/assets/logo.svg" alt="MATHIR Logo" width="180"/>

# Ã°Å¸Â§Â  MATHIR

### Memory-Augmented Tensor Hybrid with Intelligent Routing

**The first memory layer for LLMs that actually thinks Ã¢â‚¬â€ promotes, forgets, consolidates, and links.**

<br/>

> **Ã°Å¸â€ â€¢ v8.4.1 Ã¢â‚¬â€ Dynamic injection + sync.** Ships `mathir_inject.py` and `mathir_sync.py` to propagate the MATHIR block across `agents/`, `commands/`, `skills/`, `skills-global/`, `docs/` from one source of truth. 5 target-specific templates, `--explain` mode, install reproducibility fixes.
>
> **v8.4.0 Ã¢â‚¬â€ Living memory, not a write-only disk.** MATHIR now ships a full **Ebbinghaus forgetting curve**, **tier promotion** (working Ã¢â€ â€™ episodic Ã¢â€ â€™ semantic Ã¢â€ â€™ procedural), **semantic consolidation** (auto-merge near-duplicates), and a **link graph** (spreading activation ÃƒÂ  la Collins & Loftus 1975). Memories that get recalled grow stronger; memories that don't, decay and archive. **7 new MCP tools. 173/173 tests pass.**

<br/>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-8.4.1-6366f1?style=for-the-badge)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-173%20passed-22c55e?style=for-the-badge)](#-tests--benchmarks)

<br/>

[**Ã°Å¸â€ â€¢ What's new in 8.4**](#-whats-new-in-v840--living-memory) Ã‚Â· [**Ã°Å¸â€Å’ MCP Plug & Play**](#-mcp-plug--play--2-lines) Ã‚Â· [**Ã°Å¸Â§Â° Dynamic Injection & Sync (NEW)**](#-dynamic-injection--sync-v841) Ã‚Â· [**Ã°Å¸â€œâ€“ The Story**](#-the-story-that-hurts) Ã‚Â· [**Ã¢Å¡Â¡ Quick Start**](#-quick-start-30-seconds) Ã‚Â· [**Ã°Å¸Ââ€”Ã¯Â¸Â Architecture**](#-architecture) Ã‚Â· [**Ã°Å¸â€ Å¡ vs Alternatives**](#-vs-alternatives-honest-2026-comparison)

</div>

---

## Ã°Å¸â€ â€¢ What's new in v8.4.0 Ã¢â‚¬â€ Living memory

MATHIR v8.4.0 closes the gap between "memory that stores" and "memory that *thinks*". Every other memory layer for LLMs is a write-only disk: you save, you recall, and that's it. **MATHIR is the first that actually manages its own memory lifecycle.**

### The 4 phases of cognitive memory

| Phase | What it does | How it works | API |
|-------|-------------|--------------|-----|
| **Ã°Å¸Â§Â¬ Promote** | Moves memories up the tier ladder as they mature | `working_memory` Ã¢â€ â€™ `episodic` (recallÃ¢â€°Â¥3 + ageÃ¢â€°Â¥1d) Ã¢â€ â€™ `semantic` (recallÃ¢â€°Â¥10 + ageÃ¢â€°Â¥7d) Ã¢â€ â€™ `procedural` (priorityÃ¢â€°Â¥8 + `how-to:` label) | `memory_promote`, `memory_auto_promote` |
| **Ã¢ÂÂ³ Decay** | Ebbinghaus forgetting curve Ã¢â‚¬â€ unused memories lose stability | 5%/30d decay Ã‚Â· `stability += 0.1` on each recall Ã‚Â· archived when `stability < 0.05` | `memory_decay` |
| **Ã°Å¸â€â€” Consolidate** | Merges near-duplicate memories (cosine > 0.95) | Stronger absorbs weaker: `recall_count` sums, `stability` takes max, audit trail in `merged_from[]` | `memory_consolidate` (dry-run supported) |
| **Ã°Å¸Å’Â Link graph** | Spreading activation (Collins & Loftus 1975) | New `memory_links` table Ã‚Â· `cosine > 0.7` creates bidirectional links Ã‚Â· BFS with `decay=0.5` per hop | `memory_link`, `memory_get_links`, `memory_build_links` |

### Before vs after

```python
# BEFORE v8.4.0 Ã¢â‚¬â€ passive storage
memory_save("the API uses /v2/chat/completions")
memory_save("the API uses /v2/chat/completions")  # duplicate
memory_save("the API uses /v2/chat/completions")  # duplicate
# Ã¢â€ â€™ 3 memories, all the same, no ranking, no decay, no links

# AFTER v8.4.0 Ã¢â‚¬â€ living memory
memory_save(...)
memory_recall(query)                # auto-touches: stabilityÃ¢â€ â€˜, recall_countÃ¢â€ â€˜
memory_auto_promote()               # working Ã¢â€ â€™ episodic if mature enough
memory_decay()                      # archive stale memories
memory_consolidate(dry_run=False)   # merge 3 duplicates into 1 canonical
memory_build_links(threshold=0.7)   # link related concepts
# Ã¢â€ â€™ 1 canonical memory + N linked memories, ranked, aging, connected
```

### Live verification (2026-06-23)

```text
stats: 29 memories, by_tier={episodic:14, semantic:9, working:6}
promote: episodic Ã¢â€ â€™ semantic (force=True)
recall: 3 results, touched=3
build_links: 246 links created from 29 memories (threshold=0.5)
consolidate: 3 candidates at threshold 0.9 (dry_run)
```

**7 new MCP tools, 7 new daemon RPC methods, 26 new pytest tests (173/173 total).**

---

## The story that hurts

### Ã°Å¸Â§â€˜Ã¢â‚¬ÂÃ°Å¸â€™Â» The developer

```
Monday morning. You open Claude. You tell it:
  "My name is Thomas, I'm building a RAG with Python, FastAPI + Postgres."
Claude says: "Got it, I'll remember that."

3 months later. You switch to Cursor + Llama 3.1.
  Llama: "Hi! Who are you?"
  Ã¢â€ â€˜ Everything Claude "remembered"? Gone. Vendor-locked.

You try Mem0. $79/month. Not open source. You can't audit what it does with your data.
You want to run on your Jetson for offline. "We'll get back to you with an enterprise quote."
You want to detect prompt injection. "That's not what we do."
```

**6 months of memory. Wiped in 3 seconds.** Because your memory doesn't belong to you.

### Ã°Å¸Å¡â€” Why memory matters in the real world

> A car that doesn't remember a learned pattern after an OTA restart. A truck sending corrupted CAN-bus data that no anomaly detector flags. Both are failures of memory, not perception. MATHIR stores + validates both learned patterns (episodic) and anomalies (immunological) persistently, locally, and auditable.

### What MATHIR changes

```
Ã¢Å“â€¦ Memory that follows you everywhere Ã¢â‚¬â€ SQLite local, MIT, zero vendor lock-in.
Ã¢Å“â€¦ Memory that improves Ã¢â‚¬â€ +37.8% online learning, not static facts.
Ã¢Å“â€¦ Anomaly detected in <1ms Ã¢â‚¬â€ immunological tier, AUC = 1.0, zero false positives.
Ã¢Å“â€¦ Runs on edge Ã¢â‚¬â€ 240 MB VRAM, Jetson Orin Ã¢Å“â€¦, Raspberry Pi Ã¢Å¡Â Ã¯Â¸Â, zero cloud.
```

---

## Ã°Å¸â€Å’ MCP Plug & Play Ã¢â‚¬â€ 2 lines

**One daemon, 17 tools, same memory.** Connect any LLM in 2 steps:

```bash
# 1. Start the daemon (once)
python -m mathir_mcp
```

```jsonc
// 2. Add to your MCP tool (opencode.json, claude_desktop_config, etc.)
{
  "mcp": {
    "mathir": {
      "command": "python",
      "args": ["-m", "mathir_mcp"]
    }
  }
}
```

**That's it.** `memory_save`, `memory_recall`, `memory_smart_search`, `memory_hybrid_search` Ã¢â‚¬â€ available in all your tools.

| Tool | MCP | Config |
|------|-----|--------|
| **OpenCode** | Ã¢Å“â€¦ Native | `opencode.json` Ã¢â€ â€™ `mcpServers` |
| **Claude Code** | Ã¢Å“â€¦ Native | `claude_desktop_config.json` |
| **Kilo Code** | Ã¢Å“â€¦ Native | Settings Ã¢â€ â€™ MCP Ã¢â€ â€™ Add Server |
| **MiMo Code** | Ã¢Å“â€¦ Native | Config `mcp` section |

**Supports:** OpenAI Ã‚Â· Anthropic Ã‚Â· Gemini Ã‚Â· Groq Ã‚Â· Ollama Ã‚Â· llama_cpp Ã‚Â· any LLM.

---

## Ã°Å¸Â§Â° Dynamic Injection & Sync (v8.4.1)

Two new dev-loop tools in `~/.config/opencode/bin/` (or `mathir_mcp/mathir_lib/` in the source repo) automate the MATHIR injection block across all your AI config files.

| Tool | What it does | When to run |
|---|---|---|
| **`mathir_inject.py`** | Reads `<target>/_MATHIR_INJECT.md` and injects the block into every `.md` of that target. Idempotent. | After creating/editing a template, or a new agent/command/skill |
| **`mathir_sync.py`** | Copies new files from `<repo_root>/mathir_mcp/` into your `~/.config/opencode/`. **Safe by default** Ã¢â‚¬â€ never overwrites. | After dev work in the source repo |

```bash
# 5 targets: agents, commands, skills, skills-global, docs (+ "all")
python bin/mathir_inject.py --apply --target all         # inject everything
python bin/mathir_inject.py --check --target all        # see what would change
python bin/mathir_inject.py --apply --file agents/foo.md # inject one file
python bin/mathir_inject.py --list                      # show targets/templates
python bin/mathir_inject.py --explain                   # how it works

# Sync source -> config (NEW files only by default)
python bin/mathir_sync.py                               # dry-run
python bin/mathir_sync.py --force                       # apply
python bin/mathir_sync.py --only modules                # Python files only
python bin/mathir_sync.py --update-existing             # overwrite (CAREFUL)
python bin/mathir_sync.py --explain                     # how it works
```

**5 target templates** in `mathir_mcp/opencode/<target>/_MATHIR_INJECT.md` Ã¢â‚¬â€ edit the template once, re-inject everywhere. Pair: `sync.py --force && inject.py --apply --target all`.

---

## Ã°Å¸â€ Å¡ vs Alternatives (honest 2026 comparison)

> Researched against Mem0, Letta, Zep, Cognee, LangMem, Microsoft GraphRAG, Supermemory, Recall.it, ChatGPT Memory, Claude Projects, Gemini memories, Microsoft Copilot Work IQ. Sources at the bottom of this section.

| Product | Architecture | OSS? | LLM-agnostic? | Edge? | Anomaly detection | Cost |
|---|---|:---:|:---:|:---:|:---:|:---|
| **Ã°Å¸Â§Â  MATHIR** | 4 cognitive tiers + KL router + Mahalanobis | Ã¢Å“â€¦ **MIT** | Ã¢Å“â€¦ Any | Ã¢Å“â€¦ **~500 MB GPU / 80 MB CPU** | Ã¢Å“â€¦ **AUC = 1.0** | **Free** |
| [Mem0](https://mem0.ai) | Vector + rerankers + LLM compression | Ã¢Å¡Â Ã¯Â¸Â SDK only | Ã¢Å“â€¦ Any | Ã¢ÂÅ’ Cloud | Ã¢ÂÅ’ | Free Ã¢â€ â€™ $249/mo |
| [Letta](https://letta.com) | Core/archival/recall tiers | Ã¢Å“â€¦ Apache 2.0 | Ã¢Å“â€¦ Any | Ã¢Å¡Â Ã¯Â¸Â Heavy | Ã¢ÂÅ’ | Free (BYO infra) |
| [Zep](https://getzep.com) | Temporal knowledge graph | Ã¢Å¡Â Ã¯Â¸Â Graphiti OSS | Ã¢Å“â€¦ Any | Ã¢ÂÅ’ Cloud | Ã¢ÂÅ’ | $1,250/yr Ã¢â€ â€™ Custom |
| [Cognee](https://cognee.ai) | Self-hosted KG + vector | Ã¢Å“â€¦ Apache 2.0 | Ã¢Å“â€¦ Any | Ã¢Å¡Â Ã¯Â¸Â Heavy | Ã¢ÂÅ’ | $35/mo Ã¢â€ â€™ Custom |
| [LangMem](https://langchain-ai.github.io/langmem/) | Library on LangGraph store | Ã¢Å“â€¦ MIT | Ã¢Å“â€¦ Via LangChain | Ã¢Å¡Â Ã¯Â¸Â DIY | Ã¢ÂÅ’ | Free (BYO infra) |
| [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) | KG + community detection | Ã¢Å“â€¦ MIT | Ã¢Å“â€¦ Any | Ã¢Å¡Â Ã¯Â¸Â DIY | Ã¢ÂÅ’ | Free (BYO infra) |
| [Supermemory](https://supermemory.ai) | Custom vector graph | Ã¢ÂÅ’ Self-host binary | Ã¢Å“â€¦ Any | Ã¢Å¡Â Ã¯Â¸Â Self-host | Ã¢ÂÅ’ | $19 Ã¢â€ â€™ $399/mo |
| [Recall.it](https://recall.it) | Personal knowledge graph | Ã¢ÂÅ’ Closed SaaS | Ã¢Å¡Â Ã¯Â¸Â Max tier only | Ã¢ÂÅ’ | Ã¢ÂÅ’ | Free Ã¢â€ â€™ $38/mo |
| **ChatGPT Memory** (vendor) | Background "Dreaming" synthesis | Ã¢ÂÅ’ Closed | Ã¢ÂÅ’ OpenAI only | Ã¢ÂÅ’ Cloud | Ã¢ÂÅ’ | $20/mo+ |
| **Claude Projects** (vendor) | User-curated KB per project | Ã¢ÂÅ’ Closed | Ã¢ÂÅ’ Anthropic only | Ã¢ÂÅ’ Cloud | Ã¢ÂÅ’ | $20/mo+ |
| **Gemini memories** (vendor) | Implied semantic + chat history | Ã¢ÂÅ’ Closed | Ã¢ÂÅ’ Google only | Ã¢ÂÅ’ Cloud | Ã¢ÂÅ’ | Free Ã¢â€ â€™ $20/mo |
| **Microsoft Work IQ** (vendor) | Semantic index + personal memory | Ã¢ÂÅ’ Closed | Ã¢ÂÅ’ Microsoft 365 only | Ã¢ÂÅ’ Cloud | Ã¢ÂÅ’ | M365 sub |

### What this table actually says

**3 things only MATHIR does, as of June 2026:**

1. **Anomaly detection on inputs** (immunological tier, AUC = 1.0). No competitor in this list has it.
2. **Edge deployment in ~500 MB VRAM**. All others need cloud or heavy local infra. Jetson Orin Ã¢Å“â€¦ (full CUDA), Raspberry Pi Ã¢Å¡Â Ã¯Â¸Â (CPU fallback with ONNX INT8).
3. **MIT-licensed, fully open source, no managed service**. The only true OSS option with a 4-tier cognitive architecture.

**Things others do that MATHIR doesn't (honesty):**

- **Enterprise SSO, SOC 2, HIPAA, audit logs** Ã¢â€ â€™ Zep, Mem0 Pro, Supermemory Enterprise have these. MATHIR doesn't.
- **Managed hosted service** Ã¢â€ â€™ Mem0, Zep, Cognee, Supermemory all offer this. MATHIR is self-host only.
- **Temporal fact validity** (modeling "this preference is no longer valid") Ã¢â€ â€™ Zep's specialty.
- **1M+ tokens of pre-curated memory** Ã¢â€ â€™ Mem0's LoCoMo benchmark wins.

**Where MATHIR is competitive:**

- **GPU embedding speed** Ã¢â€ â€™ paraphrase-multilingual-MiniLM-L12-v2 on CUDA fp16: ~104ms/sent (384d, 50+ languages, 239MB VRAM)
- **Pure retrieval quality** Ã¢â€ â€™ MATHIR = FAISS dense-only (0.7441 nDCG@10 on BEIR SciFact, equal to SOTA)
- **Cross-provider** Ã¢â€ â€™ 11/12 wins across 3 different LLM architectures
- **Cross-lingual** Ã¢â€ â€™ UNIBRI finds English content from French queries
- **Cost** Ã¢â€ â€™ free, vs $20Ã¢â‚¬â€œ$400/mo for managed alternatives

### Sources

Comparison sources (June 2026): Mem0, Letta, Zep, Cognee, LangMem, GraphRAG, Supermemory, Recall.it, ChatGPT Memory, Claude Projects, Gemini memories, Microsoft Work IQ Ã¢â‚¬â€ full URLs in the v8.4.0 commit message and `benchmarks/06_results/`.

---</newString>

## Ã°Å¸Â§Â© Embedding Providers (NEW: ONNX support)

MATHIR v8.x+ ships with **6 embedding providers**. The default is now **paraphrase-multilingual-MiniLM-L12-v2** Ã¢â‚¬â€ 384d, 50+ languages, low VRAM (239MB fp16).

### Provider comparison

| Provider | Model | Dim | Speed (single) | Size | Quality | Local | Cost |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Ã°Å¸â€ â€¢ HuggingFace (GPU) Ã¢â‚¬â€ DEFAULT** | `paraphrase-multilingual-MiniLM-L12-v2` | **384** | ~104ms/sent | 471 MB (239 fp16) | Ã°Å¸Å¸Â¢ Multilingual 50+ | Ã¢Å“â€¦ | Free |
| HuggingFace (GPU) | `BAAI/bge-large-en-v1.5` | 1024 | 25 ms | 1.3 GB | Ã°Å¸Å¸Â¢ High (EN) | Ã¢Å“â€¦ | Free |
| Ã°Å¸â€ â€¢ ONNX | `Octen-Embedding-0.6B-INT8` | 1024 | 18.8 ms | **5.2 MB** | Ã°Å¸Å¸Â¢ High | Ã¢Å“â€¦ | Free |
| HuggingFace | `all-MiniLM-L6-v2` | 384 | 5.2 ms | 80 MB | Ã°Å¸Å¸Â¡ Medium (EN) | Ã¢Å“â€¦ | Free |
| HuggingFace | `Qwen/Qwen2.5-7B-Instruct` | 3584 | 10Ã¢â‚¬â€œ30 ms (GPU) | 14 GB | Ã°Å¸Å¸Â¢ High | Ã¢Å“â€¦ | Free |
| Ollama | `llama3.2:3b` | 2048 | 30Ã¢â‚¬â€œ80 ms | 2 GB | Ã°Å¸Å¸Â¢ High | Ã¢Å“â€¦ | Free |
| OpenAI | `text-embedding-3-small` | 1536 | 80Ã¢â‚¬â€œ200 ms | Cloud | Ã°Å¸Å¸Â¢ High | Ã¢ÂÅ’ | $0.02/1M |

### ONNX Provider (v8.4.0)

```python
# In v8.4.0 the v7 `mathir_lib.providers.get_provider` was replaced by a
# dedicated OctenEmbedder class in mathir_mcp/mathir_lib/mathir_onnx_embedder.py
from mathir_lib.mathir_onnx_embedder import OctenEmbedder, get_onnx_embedder

# Quantized ONNX model (recommended for CPU + cross-language paraphrase)
embedder = OctenEmbedder(
    model_dir=r"~/.config/opencode/models/octen-int8",
    provider="CPUExecutionProvider",  # or "DmlExecutionProvider" for GPU
)

print(embedder.dim)                    # 1024

embeddings = embedder.encode(["Hello", "World"])
# Shape: (2, 1024), L2-normalized, ready for cosine similarity
```

### ONNX vs HuggingFace benchmark (5 queries + 8 docs, RTX 4060)

| Metric | ONNX (Octen INT8) | HuggingFace (MiniLM) | Ratio |
|---|:---:|:---:|:---:|
| **Batch encode time** | 203 ms | 27 ms | 7.5Ãƒâ€” |
| **Single query** | 18.8 ms | 5.2 ms | 3.6Ãƒâ€” |
| **Embedding dim** | **1024** | 384 | 2.7Ãƒâ€” |
| **Model size** | **5.2 MB** | 80 MB | 15Ãƒâ€” |
| **Memory footprint** | 50 % of FP32 | 100 % | 0.5Ãƒâ€” |
| **Similarity range** | **[0.42, 0.98]** | [-2.53, 34.34] * | Ã¢â‚¬â€ |
| **L2-normalized** | Ã¢Å“â€¦ Yes | Ã¢ÂÅ’ No | Ã¢â‚¬â€ |

\* MiniLM embeddings are not L2-normalized by default Ã¢â‚¬â€ cosine similarity requires manual normalization.

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
# Manual download (recommended Ã¢â‚¬â€ faster than pip)
# Create folder: ~/.config/opencode/models/octen-int8/
# Download from https://huggingface.co/cstr/Octen-Embedding-0.6B-ONNX-INT8/resolve/main/
#   - model.int8.onnx (5.2 MB)
#   - model.int8.onnx.data (1.06 GB)
#   - tokenizer.json
#   - vocab.txt
#   - config.json
```

---

## Ã°Å¸Å¡â‚¬ Deployment Options

MATHIR supports multiple deployment targets. The embedding model you choose determines VRAM, speed, and platform compatibility.

| Platform | Model | VRAM | Speed (recall) | Status |
|----------|-------|------|----------------|--------|
| **Desktop GPU (CUDA)** | MiniLM-L12-v2 (384d) | ~240 MB | ~50 ms | ✅ Recommended (default) |
| **Jetson Orin (CUDA)** | MiniLM-L12-v2 (384d) | ~240 MB | ~60 ms | ✅ Supported |
| **CPU only** | MiniLM-L12-v2 (384d) | 0 MB | ~250 ms | ✅ Supported |
| **Raspberry Pi** | ONNX INT8 (384d) | 0 MB | ~600 ms | ⚠️ Experimental |

> **Power user option:** For higher-quality embeddings (1024d, English-only), `bge-large-en-v1.5` is still available but no longer the default. Set `MATHIR_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5` to use it.

**Notes:**
- **MATHIR internal memory** (working_memory/episodic/semantic/procedural tiers + immunological anomaly bank) is ~60 KB regardless of platform — this is always true (Theorem 1, bounded capacity).
- **Default model** is `paraphrase-multilingual-MiniLM-L12-v2` (384d) — multilingual, fast, low VRAM. Override with `MATHIR_EMBEDDING_MODEL` env var.
- **Raspberry Pi** requires CPU fallback — use ONNX INT8. The 1024d models are too large for Pi-class ARM devices.
- **Jetson Orin** has CUDA support and runs MiniLM at near-desktop speeds.

---

## Ã¢Å¡Â¡ Quick Start (30 seconds)

### 1. Install

```bash
git clone https://github.com/sil3d/MATHIR.git
cd MATHIR
pip install -e ./mathir_mcp
# Optional: install the sibling mathir_dropin (sibling package, sibling install)
pip install -e ./mathir_dropin
# Optional: portable Pi/Jetson subset
pip install -e ./raspberry_jetson
```

### 2. The smallest possible example

```python
from mathir_dropin.simple import SimpleMemory   # zero dependencies (just SQLite FTS5)

memory = SimpleMemory(db_path="my_app.db")
memory.store("User asked about Python closures")
memory.store("Explained that closures capture enclosing-scope variables")
memory.store("User then asked about decorators")

results = memory.recall("Python functions", k=3)
# Ã¢â€ â€™ ["User asked about Python closures", "Explained closures..."]
```

### 3. With HybridSearch (auto-scaling vector search)

```python
from mathir_dropin.simple import SimpleMemory

# HybridSearch is automatic Ã¢â‚¬â€ just use SimpleMemory
memory = SimpleMemory(db_path="my_app.db")

# Store memories (auto-selects numpy for N < 5K)
for i in range(1000):
    memory.store(f"Memory item {i}: This is a test memory about topic {i % 10}")

# At N=5,000, auto-switches to USearch HNSW (1.37ms)
results = memory.recall("test topic", k=5)

# Memory-mapped index persists to disk Ã¢â‚¬â€ no RAM pressure
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

Works with **any LLM** Ã¢â‚¬â€ OpenAI, Anthropic, Gemini, Groq, Ollama, local 7B via `llama_cpp`, anything.

### 5. Or use the full V7 plugin (8 algorithms, 6 theorems)

```python
from mathir_lib import MATHIRPluginV7

plugin = MATHIRPluginV7(embedding_dim=4096)
output = plugin.perceive(llm_embedding)

print(output["enhanced_embedding"])  # [1, 4096]
print(output["router_weights"])      # 4-tier allocation: [0.4, 0.3, 0.2, 0.1]
print(output["anomaly_score"])       # novelty detection (0.0Ã¢â‚¬â€œ1.0)
print(output["episodic_context"])    # retrieved past experiences
```

---

## Ã°Å¸â€œÅ¡ Documentation Map

| Doc | Purpose | Audience |
|-----|---------|----------|
| [README.md](README.md) | Overview, quick start, vs alternatives | Everyone |
| [CHANGELOG.md](CHANGELOG.md) | Version history (source of truth for version) | Maintainers |
| `bin/mathir_inject.py` | Dynamic MATHIR block injection (5 targets) | Agent/skill authors |
| `bin/mathir_sync.py` | Safe source-repo Ã¢â€ â€™ config sync | Maintainers |
| [docs/01_MASTER_RESEARCH_PAPER.md](docs/01_MASTER_RESEARCH_PAPER.md) | Doctoral paper (147KB) | Researchers |
| [docs/03_MASTER_QA_GUIDE.md](docs/03_MASTER_QA_GUIDE.md) | 63 defense Q&A | Decision-makers |
| [docs/05_SHIPPING_GUIDE.md](docs/05_SHIPPING_GUIDE.md) | Production shipping FAQ | DevOps |
| [docs/06_MULTIMODAL_MEMORY_GUIDE.md](docs/06_MULTIMODAL_MEMORY_GUIDE.md) | Modality details | Integrators |
| [docs/07_MATHIR_VS_VECTORDB_USE_CASES.md](docs/07_MATHIR_VS_VECTORDB_USE_CASES.md) | MATHIR vs FAISS | Architects |
| [docs/08_WHY_SAME_RESULTS.md](docs/08_WHY_SAME_RESULTS.md) | Math proof A=FAISS | Theorists |
| [docs/BRAIN_ARCHITECTURE.md](docs/BRAIN_ARCHITECTURE.md) | 5-phase brain stack | Engineers |
| [mathir_mcp/README.md](mathir_mcp/README.md) | MCP install + 3-step quick start | MCP users |
| [mathir_mcp/GLOBAL_INSTRUCTIONS.md](mathir_mcp/GLOBAL_INSTRUCTIONS.md) | Injected into agent prompts | Agent devs |
| [mathir_mcp/docs/AGENT.md](mathir_mcp/docs/AGENT.md) | Per-agent MCP config | MCP integrators |
| [mathir_mcp/docs/DAEMON.md](mathir_mcp/docs/DAEMON.md) | Daemon JSON-RPC protocol | Backend devs |
| [mathir_mcp/docs/DIMENSIONS.md](mathir_mcp/docs/DIMENSIONS.md) | Embedding model selection | ML engineers |
| [mathir_mcp/docs/GPU_SETUP.md](mathir_mcp/docs/GPU_SETUP.md) | GPU acceleration | GPU users |
| [mathir_mcp/docs/DASHBOARD_GUIDE.md](mathir_mcp/docs/DASHBOARD_GUIDE.md) | Dashboard setup | Admins |

---

## Ã°Å¸Å½Â¬ Live Demo

```bash
cd vision_testing
pip install -r requirements.txt
python start_ui.py
# Ã¢â€ â€™ Opens at http://127.0.0.1:5000
```

A full web UI for testing **vision + audio** models with persistent MATHIR memory. 6 views: Chat, Camera, Memory, Models, Accuracy, Settings Ã¢â‚¬â€ plus a standalone `/playground.html` for multi-session chat with model switching, image drag & drop, and hold-to-talk audio.

---

## Ã°Å¸â€™Â¡ More Examples

### Example 1 Ã¢â‚¬â€ Persistent chat memory across sessions (and across LLMs)

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
# Ã¢â€ â€™ ["Alice is a software engineer at Google",
#    "Alice is building a RAG system for legal documents",
#    "Alice prefers Python over JavaScript"]

# === Day 3 (switch to local Llama 3.1 Ã¢â‚¬â€ same memory!) ===
# Same SQLite file, same memories, different LLM.
# This is what vendor-locked ChatGPT Memory can't do.
```

### Example 2 Ã¢â‚¬â€ Anomaly detection (no other LLM-memory product has this)

```python
from mathir_lib import MATHIRPluginV7

plugin = MATHIRPluginV7(embedding_dim=768)

# Feed normal inputs to "train" the immune system
for emb in normal_user_inputs:
    plugin.perceive(emb)

# Now anomalies are flagged
output = plugin.perceive(weird_prompt_injection)
if output["anomaly_score"] > 0.95:
    print("Ã¢Å¡Â Ã¯Â¸Â Possible prompt injection detected!")
    # AUC-ROC = 1.0 on test set
```

> More examples (context-aware retrieval, cross-lingual UNIBRI, cross-provider, full 4-tier pipeline) live in [`docs/04_DEV_INTEGRATION_GUIDE.md`](docs/04_DEV_INTEGRATION_GUIDE.md).

---

## Ã°Å¸Ââ€”Ã¯Â¸Â Architecture

<p align="center">
  <img src="docs/assets/architecture.svg" alt="MATHIR Architecture" width="900"/>
</p>

```
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
Ã¢â€â€š              ANY LLM                       Ã¢â€â€š
Ã¢â€â€š   (Claude Ã‚Â· GPT-5 Ã‚Â· Qwen Ã‚Â· LFM2.5 Ã‚Â· 7B)    Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                  Ã¢â€â€š embeddings (1024-d)
                  Ã¢â€“Â¼
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
Ã¢â€â€š           Ã°Å¸Â§Â   MATHIR PLUGIN                Ã¢â€â€š
Ã¢â€â€š    ~500 MB VRAM (GPU) Ã‚Â· ~107 ms Ã‚Â· edge-ready Ã¢â€â€š
Ã¢â€â€š                                             Ã¢â€â€š
Ã¢â€â€š   NOTE: MATHIR internal memory (working_   Ã¢â€â€š
Ã¢â€â€š   memory/episodic/semantic/procedural +    Ã¢â€â€š
Ã¢â€â€š   immunological anomaly bank)              Ã¢â€â€š
Ã¢â€â€š   is ~60 KB (always, Theorem 1). VRAM usage Ã¢â€â€š
Ã¢â€â€š   is the embedding model, not the tiers.    Ã¢â€â€š
Ã¢â€â€š                                             Ã¢â€â€š
Ã¢â€â€š   Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â  Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€š Working  Ã¢â€â€š  Ã¢â€â€š Episodic Ã¢â€â€š  Ã¢â€â€šSemantic Ã¢â€â€š  Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€š  (now)   Ã¢â€â€š  Ã¢â€â€š  (past)  Ã¢â€â€š  Ã¢â€â€š(always) Ã¢â€â€š  Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€š
Ã¢â€â€š        Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ      Ã¢â€â€š
Ã¢â€â€š               Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â               Ã¢â€â€š
Ã¢â€â€š               Ã¢â€â€š KL  Router   Ã¢â€â€š               Ã¢â€â€š
Ã¢â€â€š               Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ               Ã¢â€â€š
Ã¢â€â€š               Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Â¼Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â               Ã¢â€â€š
Ã¢â€â€š               Ã¢â€â€šImmunological Ã¢â€â€š               Ã¢â€â€š
Ã¢â€â€š               Ã¢â€â€š  (anomaly)   Ã¢â€â€š               Ã¢â€â€š
Ã¢â€â€š               Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ              Ã¢â€â€š
Ã¢â€â€š                                             Ã¢â€â€š
Ã¢â€â€š   Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â   Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€š     HybridSearch Auto-Scaling       Ã¢â€â€š   Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€š  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â    Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â   Ã¢â€â€š   Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€š  Ã¢â€â€š numpy   Ã¢â€â€šÃ¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“ÂºÃ¢â€â€š USearch HNSW Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€š  Ã¢â€â€š (N<5K)  Ã¢â€â€šautoÃ¢â€â€š (mmap index) Ã¢â€â€š   Ã¢â€â€š   Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€š  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ   Ã¢â€â€š   Ã¢â€â€š
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ   Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                  Ã¢â€â€š enhanced context + anomaly flag
                  Ã¢â€“Â¼
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
Ã¢â€â€š              LLM DECISIONS                  Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
```

### 5 memory tiers, mapped to the human brain

Just like your brain doesn't store a memory in one place, MATHIR splits knowledge across **5 tiers** that mirror cognitive neuroscience. Each tier has a different role, capacity, and lifecycle — and memories flow between them as they mature (Ebbinghaus consolidation).

| Tier | Brain analogy | What it stores | Capacity | Updates | Example |
|------|---------------|----------------|----------|---------|---------|
| **working_memory** | Your **focus** right now | Current task context, last N steps of attention | 64 slots (circular buffer) | Every step | "The user is debugging the auth flow — focus is JWT validation" |
| **episodic** | Your **autobiography** | Specific events you've lived, time-stamped | 1,000 slots (FIFO + LIRS) | On every event | "On 2026-06-15 we hit a connection pool bug, fixed with pool=50" |
| **semantic** | Your **general knowledge** | Facts, concepts, patterns abstracted from experience | 256 prototypes (online k-means) | Every 100 steps | "Our REST API uses /v2/ prefix and JWT auth — applies to all endpoints" |
| **procedural** | Your **muscle memory** | How-to recipes, repeatable procedures, runbooks | Unlimited (label must start with `how-to:`) | On save | "How to rotate DB password: 1) stop 2) update 3) restart" |
| **immunological** | Your **immune system** | Pattern of "this doesn't belong" — anomalies | 100 patterns (Mahalanobis distance) | On anomaly event | Detects gibberish, SQL injection, XSS, binary blobs |

#### How memory flows between tiers (like human memory consolidation)

```
  +--------------+    recalled 3+ times    +--------------+
  |   working    | ----------------------> |   episodic   |
  |  (focus now) |                         | (your story) |
  +--------------+                         +--------------+
                                                   |
                                         recalled 10+ times
                                         & >7 days old
                                                   v
                                         +--------------+
                                         |   semantic   |
                                         |  (knowledge) |
                                         +--------------+

  +--------------+
  |  procedural  | <-- saved explicitly with `how-to:` or `recipe:` label
  | (muscle mem) |     no auto-promotion -- it's deliberately hand-written
  +--------------+

  +--------------+
  |immunological | <-- triggers automatically on anomalous input
  |  (immune)    |     never queried directly -- it scores everything you recall
  +--------------+
```

**Concrete example**: when you debug a JWT auth bug

1. **working**: "Looking at line 142 of auth.py" (this moment, in focus)
2. **episodic**: "Last week, the same bug was a clock-skew issue" (specific past event)
3. **semantic**: "JWT auth always checks `exp` and `nbf` claims" (general rule, learned over many auth bugs)
4. **procedural**: "How to rotate JWT signing keys: 1) generate new keypair 2) update env 3) restart service 4) revoke old tokens" (your runbook)
5. **immunological**: flags that one query string `?token=admin' OR '1'='1` looks like SQL injection (immune response)

#### What each tier is good for

- **working**: "What was I just doing?" — immediate continuity
- **episodic**: "Have I seen this before?" — pattern matching against past events
- **semantic**: "What's the rule here?" — abstract knowledge that applies broadly
- **procedural**: "How do I do X again?" — step-by-step recipes
- **immunological**: "Does this feel weird?" — anomaly detection, catches injection attacks, gibberish, out-of-distribution inputs

#### Lifecycle commands (the Ebbinghaus loop)

```python
# After a session: mature memories, prune forgotten ones
memory_auto_promote()                    # working -> episodic (3+ recalls, >1d)
memory_consolidate(dry_run=False)        # merge near-duplicates (cosine > 0.95)
memory_decay(threshold_days=30)          # Ebbinghaus: stability -= 5% per 30d unused
memory_build_links(threshold=0.7)        # link related memories (graph)
```

A memory in **episodic** that's been recalled 10+ times over 7+ days **auto-promotes to semantic**. Memories not recalled for 30+ days start losing stability (Ebbinghaus forgetting curve). Duplicates above cosine 0.95 get merged. The result: **the more you use it, the more it sticks — just like your brain.**

### HybridSearch Auto-Scaling Backend

```
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
Ã¢â€â€š                    HybridSearch Auto-Scaling                    Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¤
Ã¢â€â€š                                                                 Ã¢â€â€š
Ã¢â€â€š  N < 5,000 docs          N >= 5,000 docs                       Ã¢â€â€š
Ã¢â€â€š  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â         Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â       Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€š   numpy     Ã¢â€â€š  Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€“Âº  Ã¢â€â€š  USearch HNSW (mmap index)  Ã¢â€â€š       Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€š  0.78 ms    Ã¢â€â€š  auto   Ã¢â€â€š  1.37 ms                    Ã¢â€â€š       Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  scale  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ       Ã¢â€â€š
Ã¢â€â€š                                                                 Ã¢â€â€š
Ã¢â€â€š  Memory: ~20 KB        Memory: ~50 KB (mmap on disk)           Ã¢â€â€š
Ã¢â€â€š  RAM-only              Index persisted to disk, not RAM         Ã¢â€â€š
Ã¢â€â€š                                                                 Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
```

### Daemon Architecture (v7.8.0+)

MATHIR runs as a **persistent daemon process** on **port 7338** Ã¢â‚¬â€ the embedding model stays loaded in GPU RAM between calls.

```
Client (opencode / Python)
         Ã¢â€â€š TCP socket (localhost:7338)
         Ã¢â€“Â¼
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â
Ã¢â€â€š  mathir_daemon.py (persistent)      Ã¢â€â€š
Ã¢â€â€š  Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Model loaded ONCE at startup   Ã¢â€â€š
Ã¢â€â€š  Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ GPU memory held across calls   Ã¢â€â€š
Ã¢â€â€š  Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ TCP server on port 7338        Ã¢â€â€š
Ã¢â€â€š  Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ HybridSearch auto-scaling      Ã¢â€â€š
Ã¢â€â€š  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ No model reload per request    Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
         Ã¢â€â€š
         Ã¢â€“Â¼
    mathir_client.py (thin client)
    Ã¢â€ â€™ 1Ã¢â‚¬â€œ2 ms per call (model already in VRAM)
```

**Why daemon instead of per-request:**
- Model load: **~2-3 seconds** (MiniLM-L12-v2) - eliminated after first call
- Per-call overhead: **1Ã¢â‚¬â€œ2 ms** (TCP round-trip only)
- GPU memory: **~240 MB** held continuously (vs 0 MB between calls)
- No cold starts, no repeated allocation

```bash
# Start daemon (background, persists across sessions)
python -m mathir_mcp &

# Thin client Ã¢â‚¬â€ fast, model already loaded
python ~/.config/opencode/mathir_mcp/mathir_lib/mathir_client.py recall "query" -k 5
```

### Daemon Push (v8.2.0)

Proactive memory delivery Ã¢â‚¬â€ daemon pushes relevant memories without explicit recall. Three modes: `--auto` (inject-ready), `--json` (structured), default (plain text). See `mathir_client.py push --help` for flags.

### KL-constrained router

The router decides which memory tier to consult for each input. It uses **PPO-style trust-region optimization** with a KL-divergence constraint to prevent collapse to a single tier.

**Example allocation** (router weights `[0.40, 0.30, 0.20, 0.10]` for *"What's the weather?"*):
- **Working (0.40)** Ã¢â€ â€™ "22Ã‚Â°C, sunny" (immediate context)
- **Episodic (0.30)** Ã¢â€ â€™ "Last time you asked, it was sunny" (past pattern)
- **Semantic (0.20)** Ã¢â€ â€™ General weather knowledge
- **Immune (0.10)** Ã¢â€ â€™ No flag (nothing unusual)

The router **learns** its allocation strategy over time (no hard-coded rules):
- Short-term reflex Ã¢â€ â€™ **working memory**
- Recall a past situation Ã¢â€ â€™ **episodic memory**
- Apply a general concept Ã¢â€ â€™ **semantic memory**
- Novel / unusual input Ã¢â€ â€™ **immunological memory**

---

## Ã°Å¸â€œÅ  Tests & Benchmarks

All results reproducible. Scripts in [`benchmarks/`](benchmarks/), full HTML report in [`benchmarks/06_results/current/MATHIR_FINAL_REPORT.html`](benchmarks/06_results/current/MATHIR_FINAL_REPORT.html).

### Ã°Å¸â€ â€¢ Lifecycle Benchmarks (v8.4.0)

Two complementary benchmarks that prove the living memory actually improves recall quality:

```bash
# Memory-only throughput (no LLM, ~5 min)
python benchmarks/04_lifecycle_bench/micro_bench.py --count 1000

# AI-driven end-to-end (20 min default) Ã¢â‚¬â€ measures recall quality before/after
python benchmarks/04_lifecycle_bench/run_all.py --duration 20
```

The AI bench runs 4 phases: **generate experiences Ã¢â€ â€™ baseline Q&A Ã¢â€ â€™ age + maintenance cycle Ã¢â€ â€™ re-test same Q&A**. The headline metric: does `recall@5` and `has_answer_rate` improve after `decay + promote + consolidate + build_links` runs? See [`benchmarks/04_lifecycle_bench/README.md`](benchmarks/04_lifecycle_bench/README.md) for details.

### Ã°Å¸Â§Âª Test suite Ã¢â‚¬â€ 173/173 lifecycle + drop-in audit

```
Suite                                          Tests   Status    Coverage
Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬  Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬   Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬   Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
mathir_mcp/dev/test_lifecycle.py                26    Ã¢Å“â€¦ 26/26   Promote Ã‚Â· Decay Ã‚Â· Consolidate Ã‚Â· Link
mathir_mcp/dev/test_security_fixes.py            5    Ã¢Å“â€¦ 5/5     Path traversal Ã‚Â· DoS Ã‚Â· torch.load
mathir_mcp/dev/test_memory_scale.py              1    Ã¢Å“â€¦ 1/1     N=10K memory scale
mathir_mcp/dev/test_input_length_dos.py          2    Ã¢Å“â€¦ 2/2     DoS input caps
mathir_mcp/dev/  (pytest-discoverable subtotal) 34    Ã¢Å“â€¦ 34/34   mathir_mcp dev suite
mathir_dropin/tests/test_bugfixes.py             4    Ã¢Å“â€¦ 4/4     Regression suite
mathir_dropin/tests/test_memory.py              10    Ã¢Å“â€¦ 10/10   Drop-in core API
mathir_dropin/tests/test_multi_agent.py          5    Ã¢Å“â€¦ 5/5     Concurrent agents
mathir_dropin/tests/test_provider_switch.py      2    Ã¢Å“â€¦ 2/2     Provider fallback
mathir_dropin/tests/test_latin_names.py          75    Ã¢Å“â€¦ 75/75   Latin/scientific naming
mathir_dropin/tests/test_universal_bridge.py    15    Ã¢Å“â€¦ 15/15   UNIBRI cross-lingual
mathir_dropin/tests/  (subtotal)               111    Ã¢Å“â€¦ 111/111 Drop-in audit
Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬  Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬   Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬   Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
TOTAL pytest-discoverable                      145    Ã¢Å“â€¦ all     Mathir v8.4 verified
+ parametrized expansion                     + 28    Ã¢Å“â€¦ all     Pytest classes & cases
TOTAL lifecycle + drop-in                     173    Ã¢Å“â€¦ all     Headline number
```

```bash
# mathir_mcp (daemon, MCP server, lifecycle)
pytest mathir_mcp/dev/ -v

# mathir_dropin (zero-dep SimpleMemory + 4-tier MATHIRMemory)
pytest mathir_dropin/tests/ -v
```

### Ã°Å¸ÂÂ Daemon stress test Ã¢â‚¬â€ 50/50 pass (V8.3)

```
Test                          Requests  Status    Latency
Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬    Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬  Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬  Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
memory_save (rapid fire)         20/20  Ã¢Å“â€¦ PASS   50-120ms
ping (rapid fire)                20/20  Ã¢Å“â€¦ PASS   2-23ms
memory_recall (rapid fire)       10/10  Ã¢Å“â€¦ PASS   47-94ms
memory_hybrid_search             10/10  Ã¢Å“â€¦ PASS   47-65ms
Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬    Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬  Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬  Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
TOTAL                            50/50  Ã¢Å“â€¦ PASS   ~60ms avg
```

### Ã°Å¸â€œË† BEIR benchmark results (nDCG@10)

| System | SciFact | NFCorpus | ArguAna | Verdict |
|---|:---:|:---:|:---:|---|
| **FAISS dense-only (BGE-base)** | **0.7441** | **0.3657** | **0.6613** | Ã¢Å“â€¦ SOTA baseline |
| BM25 only | 0.5438 | 0.2617 | Ã¢â‚¬â€ | Ã¢Å¡Â Ã¯Â¸Â Too weak for scientific |
| Hybrid RRF (1:1) | 0.6602 | 0.3263 | Ã¢â‚¬â€ | Ã¢Å¡Â Ã¯Â¸Â BM25 dilutes dense |
| Hybrid + Cross-Encoder | 0.5910 | 0.2620 | Ã¢â‚¬â€ | Ã¢ÂÅ’ Cross-encoder wrong domain |

### Ã°Å¸â€Â Vector Search Ã¢â‚¬â€ HybridSearch Auto-Scaling

See [HybridSearch diagram in Architecture](#-architecture). Auto-scaling: numpy (N<5K, 0.78ms) Ã¢â€ â€™ USearch HNSW mmap (NÃ¢â€°Â¥5K, 1.37ms).

#### BEIR Benchmark Results (5,183 documents)

| Backend | Latency (search) | Index Size | RAM Usage | Notes |
|---|:---:|:---:|:---:|---|
| **numpy (cosine)** | **0.78 ms** | ~20 KB | ~20 KB | Fastest for small N |
| **USearch HNSW (mmap)** | **1.37 ms** | ~50 KB | ~50 KB | Memory-mapped, scales to 1M+ |
| **sqlite-vec** | **23.68 ms** | ~100 KB | ~100 KB | Slower, WAL-optimized |

#### Key Features

| Feature | Description |
|---|---|
| **Auto-scaling** | numpy Ã¢â€ â€™ USearch at N=5,000 (no config needed) |
| **Memory-mapped indexes** | Index on disk, not RAM Ã¢â‚¬â€ no memory pressure for large datasets |
| **sqlite-vec WAL mode** | Write-Ahead Logging for 3.4Ãƒâ€” write speedup |
| **Zero-config** | HybridSearch picks the optimal backend automatically |
| **FAISS fallback** | Optional FAISS integration for production deployments |

#### Usage

```python
from mathir_dropin.simple import SimpleMemory

# HybridSearch is automatic Ã¢â‚¬â€ just use SimpleMemory
memory = SimpleMemory(db_path="my_app.db")

# For N < 5K: numpy backend (0.78ms)
# For N >= 5K: auto-switches to USearch HNSW (1.37ms)
# No configuration needed

memory.store("Python closures capture variables")
results = memory.recall("Python functions", k=5)  # Uses optimal backend
```

### Ã°Å¸Å¡â‚¬ What MATHIR adds over FAISS

| Capability | FAISS | MATHIR | Delta |
|---|:---:|:---:|---|
| Online learning | Ã¢ÂÅ’ | Ã¢Å“â€¦ **+37.8 %** | Ã°Å¸Å¸Â¢ MATHIR |
| Anomaly detection (AUC) | Ã¢ÂÅ’ | **1.0** | Ã°Å¸Å¸Â¢ MATHIR |
| Context-aware results | Ã¢ÂÅ’ | **88 %** | Ã°Å¸Å¸Â¢ MATHIR |
| 2-hour stress (no crash) | Ã¢ÂÅ’ | **100 %** | Ã°Å¸Å¸Â¢ MATHIR |
| No memory leak | Ã¢ÂÅ’ | Ã¢Å“â€¦ | Ã°Å¸Å¸Â¢ MATHIR |
| Router balanced | Ã¢ÂÅ’ | 100 % acc. | Ã°Å¸Å¸Â¢ MATHIR |
| Graceful degradation | Ã¢ÂÅ’ | Ã¢Å“â€¦ | Ã°Å¸Å¸Â¢ MATHIR |
| **Raw retrieval speed** | **< 1 ms** | 0.78 ms (numpy) / 1.37 ms (USearch) | Ã°Å¸â€Âµ FAISS (similar) |

### Ã¢ÂÂ± 2-hour stress test (all 4 tiers active)

| Metric | Value | Status |
|---|:---:|:---:|
| Uptime | **100 %** | Ã¢Å“â€¦ |
| Memory leaks | **None** | Ã¢Å“â€¦ |
| Retrieval quality @ 120 min | **0.959** | Ã¢Å“â€¦ |
| P99 latency | **17.8 ms** | Ã¢Å“â€¦ |
| Total operations | 26 440 | Ã¢Å“â€¦ |

### Ã°Å¸Å’Â Cross-provider generalization (OpenRouter)

| Model | API latency | MATHIR wins | Result |
|---|:---:|:---:|---|
| `openrouter/owl-alpha` | 2.6 s | **4 / 4** | Ã°Å¸Ââ€  MATHIR wins all |
| `openai/gpt-oss-120b:free` | 2.0 s | **3 / 4** | Ã°Å¸Ââ€  MATHIR wins most |
| `openai/gpt-oss-20b:free` | 1.1 s | **4 / 4** | Ã°Å¸Ââ€  MATHIR wins all |

**Total: 11 / 12 scenarios** Ã¢â‚¬â€ MATHIR wins across 3 different LLM architectures.

### Ã°Å¸Å’Â Cross-lingual (UNIBRI)

```
"What do you know about python closures?"  Ã¢â€ â€™ finds "python-closures"         Ã¢Å“â€¦
"clotures python"  (French)                Ã¢â€ â€™ finds English "Python closures" Ã¢Å“â€¦
provider="unknown"  (no stored embedding)  Ã¢â€ â€™ 3 results via fallback chain   Ã¢Å“â€¦
```

The Universal Bridge uses **multi-resolution character n-gram kernels** (Broder 1997) + **Johnson-Lindenstrauss random projection** + **Procrustes SVD** for cross-space alignment. Mathematically grounded, vocabulary-free, language-agnostic.

### Ã¢Å¡Â¡ Performance Benchmarks (v7.8.0+)

Real-world benchmarks on RTX 4060 + CUDA, measuring **save** (store memory) and **recall** (search memory) latency.

#### End-to-End Latency

| Operation | Latency | Breakdown |
|---|:---:|---|
| **Save** (store memory) | **75 ms** | MiniLM CUDA 20ms + DB write 55ms |
| **Recall** (search memory) | **120 ms** | MiniLM CUDA 16ms + vector search 104ms |

#### Vector Search at Scale

| Dataset Size | Backend | Latency | Notes |
|---|:---:|:---:|---|
| 5,000 docs | numpy | 0.78 ms | Auto-selected for small N |
| 5,000 docs | USearch HNSW | 1.37 ms | Memory-mapped index |
| 5,000 docs | sqlite-vec | 23.68 ms | WAL-optimized |

#### Embedding Model Comparison

| Model | Dimensions | Device | Save | Recall | Notes |
|---|:---:|---|:---:|:---:|---|
| **paraphrase-multilingual-MiniLM-L12-v2** | **384** | **CUDA** | **75 ms** | **120 ms** | Ã¢Å“â€¦ Recommended Ã¢â‚¬â€ best quality/speed |
| MiniLM-L6-v2 | 384 | CUDA | 22 ms | 53 ms | Ã¢Å¡Â Ã¯Â¸Â Faster save, slower recall |
| Octen INT8 | 1024 | CPU | ~5 000 ms | ~2 700 ms | Ã°Å¸ÂÂ¢ 50Ã¢â‚¬â€œ100Ãƒâ€” slower |
| Octen INT8 | 1024 | CUDA (onnxruntime-gpu) | ~776 ms | Ã¢â‚¬â€ | Ã¢Å¡Â Ã¯Â¸Â Partial GPU (ONNX limitation) |

**Key insight:** MiniLM-L12-v2 (384d) is the default Ã¢â‚¬â€ includes embedding time (3ms) + vector search (104ms). The larger dimension space produces better similarity scores, and CUDA handles the extra compute efficiently.



---

## Ã°Å¸â€Â¬ Why It Works Ã¢â‚¬â€ Theoretical Foundation

| Component | Guarantee | Citation |
|---|---|---|
| Episodic memory | Cosine similarity on stored embeddings Ã¢â€ â€™ **+37.8 %** recall (measured on BEIR) | Empirical |
| Immunological memory | **Mahalanobis distance is the NP-optimal detector** for anomalies in Gaussian data | McLachlan 1999 |
| Working memory | Multi-head attention on circular buffer Ã¢â€ â€™ **bounded latency**, context-aware | Vaswani 2017 |
| KL Router | KL-divergence penalty (PPO-style) prevents tier collapse; max-entropy ensures exploration | Schulman 2017 |
| UNIBRI | **Theorems 1Ã¢â‚¬â€œ4** give OOV / cross-lingual / cross-provider stability guarantees | Broder 1997, J-L 1984, Wedin 1972 |

Full mathematical proofs in [`docs/01_MASTER_RESEARCH_PAPER.md`](docs/01_MASTER_RESEARCH_PAPER.md).

---

## Ã°Å¸â€œÂ Project Structure

```
MATHIR/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Ã°Å¸â€œÂ¦ mathir_mcp/            # MCP server package (pip install -e ./mathir_mcp)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ mathir_lib/            # Core: daemon Ã‚Â· MCP server Ã‚Â· vec Ã‚Â· search Ã‚Â· inject Ã‚Â· sync
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ brain/                 # Watchdog Ã‚Â· prime Ã‚Â· inject proxy (auto-restart)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ docs/                  # AGENT.md Ã‚Â· DAEMON.md Ã‚Â· DIMENSIONS.md Ã‚Â· DASHBOARD_GUIDE.md
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ opencode/              # 5 _MATHIR_INJECT.md templates (agents/commands/skills/skills-global/docs)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ dev/                   # 32 lifecycle/security/scale tests
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ pyproject.toml         # Entry points: mathir-daemon, mathir-mcp, mathir-client, mathir-watchdog
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py            # Package marker
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ __main__.py            # `python -m mathir_mcp` entry
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Ã°Å¸â€œÂ¦ mathir_dropin/          # Drop-in memory (zero-dep SimpleMemory + 4-tier MATHIRMemory)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ memory.py              # MATHIRMemory (torch-powered)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ simple.py              # SimpleMemory (FTS5, zero deps)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ store.py               # SQLite storage
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ universal_bridge.py    # UNIBRI: cross-provider Ã‚Â· cross-lingual
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ tests/                 # ~52 drop-in audit tests
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Ã°Å¸Ââ€œ raspberry_jetson/       # Portable Pi/Jetson subset (CPU-only)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py            # `python -m raspberry_jetson`
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ setup.sh
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ pyproject.toml
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Ã°Å¸â€˜ÂÃ¯Â¸Â vision_testing/         # Full vision/audio testing UI (Flask)
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Ã°Å¸â€œÅ  benchmarks/             # Reproducible benchmarks + HTML report
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ 04_lifecycle_bench/    # Micro + AI-driven lifecycle benchmarks
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ 06_results/current/MATHIR_FINAL_REPORT.html
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Ã°Å¸â€œÅ¡ docs/                   # Tutorials Ã‚Â· theory Ã‚Â· LaTeX paper
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ Ã°Å¸â€Â§ examples/               # Demo scripts (simple_memory_demo.py Ã‚Â· onnx_usage.py)
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ Ã°Å¸Â§Âª stress_test/            # 50-request daemon stress harness
```

---

## Ã°Å¸â€ºÂ Ã¯Â¸Â Try the Examples

```bash
# Zero-dep memory (works without torch)
python examples/simple_memory_demo.py

# Vision + audio UI + multi-session chat playground
cd vision_testing && python start_ui.py
# Ã¢â€ â€™ http://127.0.0.1:5000                  (Chat, Camera, Memory, Models, Accuracy, Settings)
# Ã¢â€ â€™ http://127.0.0.1:5000/playground.html  (multi-session chat, model switching, image/audio)
```

---

## Ã°Å¸â€œÅ¡ Documentation

> Ã°Å¸â€œâ€“ See the [Documentation Map](#-documentation-map) above for the full doc index by audience and purpose.

---</newString>

## Ã°Å¸â€”ÂºÃ¯Â¸Â Roadmap

| Version | Milestone | Status |
|---|---|:---:|
| V1Ã¢â‚¬â€œV5 | Core architecture + KL router | Ã¢Å“â€¦ |
| V6 | LLM-agnostic plugin API | Ã¢Å“â€¦ |
| V7 | 8 algorithms + 6 theorems + 9.3Ãƒâ€” compression | Ã¢Å“â€¦ |
| V7.5 | Real BEIR benchmarks (0.7441 SOTA) | Ã¢Å“â€¦ |
| V7.6 | Universal Bridge (UNIBRI) | Ã¢Å“â€¦ |
| V7.7 | Vision & audio testing + MATHIR memory | Ã¢Å“â€¦ |
| **V7.7.1** | **SimpleMemory (FTS5) + UI overhaul** | **Ã¢Å“â€¦** |
| **V7.8** | **GPU embeddings (bge-large) + daemon architecture** | **Ã¢Å“â€¦** |
| **V8.0** | **HybridSearch auto-scaling (numpy Ã¢â€ â€™ USearch HNSW)** | **Ã¢Å“â€¦** |
| **V8.2** | **Daemon push (proactive memory delivery)** | **Ã¢Å“â€¦** |
| **V8.3** | **HybridSearch SQLite backend fix + thread safety** | **Ã¢Å“â€¦** |
| **V8.4.0** | **Living memory (Ebbinghaus + promote/decay/consolidate/links)** | **Ã¢Å“â€¦** |
| **V8.4.1** | **Dynamic inject + sync dev-loop** | **Ã¢Å“â€¦ (current)** |
| V9 | Edge deployment polish (Jetson Orin + Pi ONNX) | Ã°Å¸â€Å“ |
| V10 | Open-source release (HuggingFace Ã‚Â· PyPI) | Ã°Å¸â€œâ€¹ |

---

## Ã°Å¸Â¤Â Contributing

We welcome contributions.

```bash
# 1. Fork & clone
git clone https://github.com/YOUR_USERNAME/MATHIR.git
cd MATHIR
pip install -e ./mathir_mcp

# 2. Create a branch
git checkout -b feature/my-feature

# 3. Make changes, add tests, run them
pytest mathir_mcp/dev/ -v
pytest mathir_dropin/tests/ -v

# 4. Submit a PR
```

### Areas where help is needed

- Ã°Å¸â€œÅ¡ **Documentation** Ã¢â‚¬â€ improve tutorials, add examples
- Ã°Å¸Â§Âª **Testing** Ã¢â‚¬â€ edge cases, more coverage
- Ã°Å¸â€œÅ  **Benchmarks** Ã¢â‚¬â€ more corpora, more embedding models
- Ã°Å¸â€œÂ± **Edge deployment** Ã¢â‚¬â€ Rust / ONNX port
- Ã°Å¸â€Å’ **Integrations** Ã¢â‚¬â€ LangChain Ã‚Â· LlamaIndex Ã‚Â· Haystack

---

## Ã°Å¸â€œâ€ž Citation

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

## Ã°Å¸â€œÅ“ License

[MIT](LICENSE) Ã¢â‚¬â€ free for commercial and research use.

---

<div align="center">

### Ã°Å¸Â§Â  MATHIR Ã¢â‚¬â€ *A 4-tier cognitive memory layer for any LLM, on any hardware.*

**Author:** [Prince Gildas Mbama Kombila](https://github.com/sil3d) Ã‚Â· **Email:** soilearn3d@gmail.com

Ã¢Â­Â **Star this repo** if you find it useful Ã¢â‚¬â€ it helps others discover MATHIR.

</div>
