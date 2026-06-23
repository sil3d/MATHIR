# MATHIR — Agent Specification

**The adaptive memory layer for any LLM.**

**Version:** 7.7.0 (V7.7 — Vision & Audio Testing, 2026-06-06)

---

## 🆕 What's New in V7.7

### Vision & Audio Testing Environment

**`vision_testing/`** — Complete setup for testing vision/audio models with MATHIR memory:

**Models Downloaded (Q4_0, fits 8GB VRAM):**
- **LFM2.5-VL-1.6B-GGUF** (vision-language) — 1.2 GB
  - [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF)
- **LFM2.5-Audio-1.5B-GGUF** (audio understanding) — 1.0 GB
  - [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF)

**Binaries Copied from Mycerise_V2_Taur** (41 files): llama-server.exe, llama-cli.exe, all ggml DLLs, libllama.so, libmtmd.so, convert_lfm2_to_gguf.py, convert_to_gguf.py

**Unified Python Interface** (`vision_testing/vision_test.py`):
- `ModelRegistry` — extensible registry for vision/audio models
- `LlamaServer` — manages llama-server.exe subprocess with vision/audio support
- `VirtualRoom` — virtual room with objects for testing
- `VisionTester` — main orchestrator with MATHIR memory integration

**Test scenarios built-in:**
1. Object recognition ("What's in the room?")
2. Change detection ("What did I just do?")
3. Memory recall via `universal_recall()`
4. Cross-model consistency (same room, different models)

**Quick Start:**
```python
from vision_testing.vision_test import VisionTester

tester = VisionTester("LFM2.5-VL-1.6B-Q4_0", port=8080)
tester.setup_memory()  # Initialize MATHIR
tester.start_server()  # Start llama-server.exe

# Create a room and ask about it
tester.room.add_object("lamp", "desk")
tester.room.add_object("book", "shelf")
response = tester.ask_model("What's in the room?")
print(response)

# Move objects and ask what changed
tester.room.move_object("lamp", "floor")
response = tester.ask_model("What did I just do?")

# Recall from MATHIR across models
results = tester.memory.universal_recall(
    query="What was on the desk earlier?",
    k=3
)
```

**Adding New Models:**
1. Download GGUF to `vision_testing/models/YourModel/`
2. Add entry to `ModelRegistry.VL_MODELS` or `AUDIO_MODELS`
3. Specify path + mmproj (+ tokenizer + vocoder for audio)

See `vision_testing/README.md` for full documentation.

---

## What's New in V7.6

### Universal Bridge (UNIBRI) — Cross-Provider, Cross-Lingual, Vocabulary-Free

**`mathir_dropin/universal_bridge.py`** (570 lines) + **`mathir_dropin/latin_names.py`** (~700 lines) — MATHIR now works with **any word, any language, any embedding provider** without retraining.

**New API: `MATHIRMemory.universal_recall()`**

```python
from mathir_dropin import MATHIRMemory
import torch

memory = MATHIRMemory(embedding_dim=384, db_path="memory.db")

# Cross-lingual: French query finds English content
results = memory.universal_recall(query="clotures python", k=3)

# Cross-provider fallback (works even if provider has no stored embeddings)
results = memory.universal_recall(
    query_embedding=torch.randn(1, 384),
    k=3,
    provider="minimax"  # Falls back to primary embeddings + FTS5
)

# Conversational queries (no FTS5 stopword interference)
results = memory.universal_recall(
    query="What do you know about python closures?",
    k=3
)
```

### Mathematical Foundation (11 Theorems)

| ID | Statement | Algorithm |
|---|---|---|
| 1 | Unbiased Kernel Estimator | Broder 1997 n-gram Jaccard |
| 2.1-2.2 | Token-aware + dimension projection | Regex cascade + JL lemma |
| 2.3 | Diacritic invariance $T(s)=T(t) \Rightarrow J(N_k(s),N_k(t))=1$ | NFKD + combining-mark strip |
| 2.4 | Case insensitivity | Lowercase + cache |
| 2.6-2.8 | Compound SPLIT sound + complete, $O(n \cdot L_R)$ | Trie DP |
| 2.9-2.10 | Abbreviation lookup $O(1)$ | Hash table |
| 3.1, 3.3 | RRF monotonicity + calibration-freeness | Cormack 2009 |
| 4 | Wedin perturbation bound (cross-space stability) | $\|\hat R - R^*\|_F \leq C \cdot \varepsilon / \sigma_{\min}(A^T B)$ |

### Latin Name Handler (6 Categories)

| Category | Example | Function |
|---|---|---|
| **Taxonomic names** | "Homo sapiens Linnaeus 1758" | `parse_taxonomic_name()` |
| **Diacritics** | "Schrödinger" / "Müller" / "Łukasiewicz" | `normalize_diacritics()` |
| **Roman numerals** | "Henry VIII" / "World War II" | `parse_roman_numeral()` |
| **Abbreviations** | 78 built-in (DNA, RNA, MRI, PCR, ATP) | `expand_abbreviation()` |
| **Compound terms** | "sternocleidomastoid" → [sterno, cleido, mastoid] | `split_compound()` |
| **Genus abbreviation** | "E. coli" / "H. sapiens" | Auto-expand |

### Verified Across 4 Architectures

| Model | Architecture | Status |
|---|---|---|
| qwen3:0.6b | Qwen | ✓ |
| lfm2.5-thinking:1.2b | Liquid LFM | ✓ |
| granite4:350m | IBM Granite | ✓ |
| qwen3.5:2b | Qwen 3.5 | (timeout, model too large) |

### Test Results: 137/137 PASS

```
test_memory.py            10/10  ✓
test_bugfixes.py           4/4   ✓
test_multi_agent.py        5/5   ✓
test_universal_bridge.py  15/15  ✓ (NEW)
test_latin_names.py      103/103 ✓ (NEW)
```

---

## What's New in V7.3

### Production-Ready Drop-In Package (`mathir_dropin/`)

A single-folder, self-contained package for production use:

```python
from mathir_dropin import MATHIRMemory
import torch

memory = MATHIRMemory(embedding_dim=384, db_path="my_memory.db")
mid = memory.store(torch.randn(1, 384), metadata={"text": "hello"})
results = memory.recall(torch.randn(1, 384), k=5)
memory.reset()  # full reset
memory.delete(mid)  # specific delete
```

**Features**:
- ✅ SQLite persistence (one `.db` file, inspectable with `sqlite3`)
- ✅ FTS5 full-text search
- ✅ Thread-safe (RLock around writes)
- ✅ Multi-agent (20+ concurrent, no data loss)
- ✅ Multi-modal (text, image, audio, video)
- ✅ Cross-model plug-and-play (same dim)
- ✅ 10 critical tests pass
- ✅ 5 files, ~1500 lines, **ONE FOLDER** to ship

**See `mathir_dropin/README.md` for the 5-minute quickstart.**

### Visual Diagrams Suite (`visualizations/`)

8 production-quality PNG diagrams + a self-contained HTML report:

```bash
python visualizations/generate_diagrams.py
python visualizations/build_report.py
open visualizations/visual_report.html
```

---

## What MATHIR Is

MATHIR is a **plug-and-play memory plugin** that gives any LLM the ability to learn, remember, and adapt in real-time on edge hardware.

It is NOT a model. It is NOT a vector database. It is NOT RAG.

It is a **cognitive memory system** — the hippocampus of AI.

```
ANY LLM  →  embeddings  →  MATHIR  →  enhanced context  →  LLM decides
(eyes)                     (memory)                        (executor)
```

---

## Design Principles

### 1. No Hardcoded Values

Every dimension, capacity, threshold, and parameter is **config-driven**.

```yaml
# config/default.yaml
memory:
  embedding_dim: 4096        # Match the LLM
  internal_dim: 272          # MATHIR's working dimension
  working_capacity: 64       # Circular buffer slots
  episodic_capacity: 1000    # Key-value store slots
  semantic_prototypes: 256   # Online k-means clusters
  immunological_capacity: 100 # Anomaly detector bank
  kl_coefficient: 0.01       # Router constraint
  anomaly_threshold: 2.0     # Novelty detection threshold
  decay_rates: [0.9, 0.7, 0.5]  # Temporal retention

compression:
  enabled: true
  method: "turboquant"       # "turboquant", "none"
  bits: 3                    # 2, 3, 4
  episodic_only: false       # Compress all tiers

inference:
  backend: "pytorch"         # "pytorch", "onnx", "rust"
  device: "auto"             # "cpu", "cuda", "auto"
  precision: "float32"       # "float32", "float16", "int8"
```

**No magic numbers in code.** Every constant comes from config.

### 2. Fast by Default

| Target | CPU | GPU | Edge |
|---|---|---|---|
| Inference | <2ms | <1ms | <5ms |
| Memory | <60 KB | <60 KB | <60 KB |
| Episodic recall | <0.1ms | <0.05ms | <0.5ms |

Achieved via:
- TurboQuant 3-bit compression (10× reduction)
- Rust memory core via PyO3 (7× speedup)
- ONNX inference via `ort` crate (3-5× speedup)
- Vectorized PyTorch (no Python loops)

### 3. Easy to Connect with Any LLM

```python
from mathir_lib import MATHIRPlugin

# Works with ANY embedding dimension
plugin = MATHIRPlugin(embedding_dim=4096)  # LLaMA-8B
plugin = MATHIRPlugin(embedding_dim=3584)  # Qwen2.5-7B
plugin = MATHIRPlugin(embedding_dim=1536)  # OpenAI
plugin = MATHIRPlugin(embedding_dim=1024)  # Cohere

# One function call
output = plugin.perceive(llm_embedding)

# Store and recall
plugin.store({'embedding': emb, 'action': act})
memories = plugin.recall(query, k=5)
```

No LLM-specific code. No framework lock-in. Just embeddings in, context out.

### 4. Learns Online

Unlike vector databases (store only), RAG (retrieve only), or long context (pass through):
- **Semantic prototypes** shift with each observation
- **Episodic memory** fills with real experiences
- **Router** adapts its allocation strategy
- **Anomaly detector** evolves its "normal" baseline

MATHIR never stops learning.

### 5. Pluggable Architecture

Every component is replaceable:

```
MATHIRPlugin
├── EmbeddingProvider    ← swap: OpenAI, Ollama, HuggingFace, direct
├── WorkingMemory        ← swap: circular buffer, attention, sliding window
├── EpisodicMemory       ← swap: in-memory, LanceDB, Qdrant
├── SemanticMemory       ← swap: k-means, prototypes, clustering
├── ImmunologicalMemory  ← swap: distance-based, learned, statistical
├── Router               ← swap: KL-constrained, learned, heuristic
├── Compressor           ← swap: TurboQuant, none, custom
└── Projection           ← swap: linear, bottleneck, MLP
```

---

## Architecture

### MATHIRPlugin

```python
class MATHIRPlugin(nn.Module):
    """
    Adaptive memory plugin for any LLM.
    
    Args:
        embedding_dim: The LLM's embedding dimension (any value)
        config: Optional config dict (uses defaults if None)
    
    Input:
        embedding: [B, D] tensor from the LLM
        
    Output:
        enhanced_embedding: [B, D] with memory context
        router_weights: [B, 4] memory tier allocation
        anomaly_score: [B] novelty detection scores
        kl_loss: scalar KL divergence loss
    """
```

### Memory Tiers

| Tier | Capacity | Function | Update | Retrieval |
|---|---|---|---|---|
| **Working** | 64 | Immediate context | Every step | Attention |
| **Episodic** | 1000 | Past experiences | On event | Similarity |
| **Semantic** | 256 | Learned concepts | Every 100 steps | Prototype match |
| **Immunological** | 100 | Anomaly detection | On event | Distance threshold |

### Router

KL-constrained soft allocation across 4 tiers. Prevents collapse to single tier.

```python
router_weights = softmax(router_net(x))  # [B, 4]
kl_loss = kl_div(router_weights, uniform_target)
output = sum(w_i * tier_i(x) for w_i, tier_i in zip(weights, tiers))
```

### Compression (TurboQuant)

Data-oblivious online vector quantization:
1. Hadamard rotation (O(d log d))
2. Beta distribution modeling
3. Scalar quantization (table lookup)

Result: 10.7× compression at 3-bit, <0.1 perplexity loss.

---

## File Structure

```
MATHIR/
├── AGENT.md                    # This file (master specification)
├── CHANGELOG.md                # Version history
├── README.md                   # Project overview
├── FUTURE_VISION.md            # Strategic roadmap
├── IMPLEMENTATION.md           # Build plan
├── config/
│   ├── default.yaml            # Default configuration
│   ├── edge.yaml               # Edge deployment config
│   ├── research.yaml           # Research/benchmark config
│   ├── v7.yaml                 # V7 configuration (enables all 8 advances)
│   └── hardware_info.json      # Hardware detection
├── mathir_lib/
│   ├── __init__.py             # Package entry point
│   ├── plugin.py               # MATHIRPlugin (V6 main API)
│   ├── plugin_v7.py            # MATHIRPluginV7 (V7, 100% backward compatible)
│   ├── config.py               # load_config / get_default_config / merge_config / validate_config
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── working.py          # WorkingMemory (V6)
│   │   ├── episodic.py         # EpisodicMemory (V6)
│   │   ├── semantic.py         # SemanticMemory (V6)
│   │   ├── immunological.py    # ImmunologicalMemory + MahalanobisImmunologicalMemory (V7)
│   │   ├── ebbinghaus.py       # EbbinghausMemory (V7, Theorem 2)
│   │   ├── sparse_coding.py    # SparseCodingMemory (V7, Theorem 5)
│   │   ├── variational.py      # VariationalMemory (V7)
│   │   ├── cross_attention.py  # CrossAttentionMemory (V7)
│   │   ├── hyperbolic.py       # HyperbolicMemory (V7, Poincaré ball)
│   │   ├── infonce.py          # InfoNCELoss (V7)
│   │   └── neural_ode.py       # NeuralODEMemory (V7, RK4)
│   ├── router.py               # KL-constrained router
│   ├── compression.py          # TurboQuant implementation
│   ├── projections.py          # Input/output projections
│   ├── providers/              # Embedding providers
│   │   ├── __init__.py
│   │   ├── base.py             # EmbeddingProvider ABC
│   │   ├── openai.py           # OpenAI embeddings
│   │   ├── ollama.py           # Ollama embeddings
│   │   ├── huggingface.py      # HuggingFace embeddings
│   │   └── direct.py           # Direct tensor input
│   ├── mhc.py                  # Manifold-Constrained Hyper-Connections
│   ├── mathir_v5.py            # V5 architecture (legacy)
│   └── components.py           # Shared components
├── rust/                       # Rust acceleration
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── memory.rs           # Memory core (PyO3)
│   │   ├── compression.rs      # TurboQuant in Rust
│   │   ├── search.rs           # Similarity search (SIMD)
│   │   └── inference.rs        # ONNX inference (ort)
│   └── mathir_core/            # Python extension
│       └── __init__.py
├── examples/
│   ├── basic_usage.py          # Simple plugin usage
│   ├── with_ollama.py          # Ollama integration
│   ├── with_openai.py          # OpenAI integration
│   ├── with_huggingface.py     # HuggingFace integration
│   ├── driving_demo.py         # Driving scenario
│   ├── benchmark_vs_rag.py     # Compare vs RAG
│   ├── edge_deploy.py          # Edge deployment
│   └── v7_advanced_demo.py     # V7 — all 8 advances in one runnable script
├── benchmarks/
│   ├── retention.py            # Long-term retention test
│   ├── generalization.py       # Cross-scenario test
│   ├── latency.py              # Inference speed test
│   ├── compression.py          # Compression quality test
│   ├── mathir_vs_rag.py        # V6 MATHIR vs RAG comparison
│   ├── benchmark_with_mock.py  # Mock-LLM benchmark runner
│   ├── dry_run.py              # Pipeline smoke test
│   └── v6_vs_v7.py             # V6 vs V7 comparison (9.3× compression measured)
├── tests/
│   ├── test_plugin.py          # Plugin API tests
│   ├── test_memory.py          # Memory tier tests
│   ├── test_compression.py     # TurboQuant tests
│   ├── test_router.py          # Router tests
│   ├── test_providers.py       # Embedding provider tests
│   ├── stress_test.py          # V6 stress test (13/13 pass)
│   ├── test_v7_memory.py       # V7 unit tests (49/49 pass)
│   └── test_v7_integration.py  # V7 integration tests (16 tests)
├── tools/
│   ├── export_onnx.py          # ONNX export script
│   ├── deploy_edge.py          # Edge deployment
│   └── benchmark_runner.py     # Benchmark automation
├── docs/
│   ├── ARCHITECTURE.md         # Technical deep-dive
│   ├── API.md                  # API reference
│   ├── DEPLOYMENT.md           # Deployment guide
│   ├── THEORY_V7.md            # V7 doctoral-grade math (58 KB, 6 theorems)
│   ├── PROOFS.md               # Formal proofs of V7 theorems
│   ├── V7_PAPER.md             # V7 NeurIPS-style paper
│   ├── V7_MIGRATION_GUIDE.md   # V6 → V7 migration
│   ├── V7_TUTORIAL.md          # V7 hands-on tutorial
│   ├── BENCHMARK_V6_VS_V7.md   # V6 vs V7 measured results
│   ├── MATHIR_VS_RAG_COMPARISON.md  # MATHIR vs RAG analysis
│   ├── KV_CACHE_RESEARCH_REPORT.md
│   └── RUST_ML_RESEARCH_REPORT.md
├── mathir_model.py             # Legacy: core MATHIR
├── driving_env.py              # Legacy: driving simulator
├── train_evolution.py          # Legacy: evolution training
├── train_mathir_v5.py          # Legacy: V5 training
├── dashboard_live.py           # Legacy: live dashboard
├── benchmark.py                # Legacy: benchmarks
└── setup.py                    # Package setup
```

---

## API Reference

### MATHIRPlugin (V6)

```python
class MATHIRPlugin(nn.Module):
    def __init__(self, embedding_dim: int, config: dict = None): ...
    def perceive(self, embedding: Tensor) -> dict: ...
    def store(self, experience: dict) -> None: ...
    def recall(self, query: Tensor, k: int = 3) -> list: ...
    def forget(self, threshold: float = 0.1) -> None: ...
    def get_stats(self) -> dict: ...
    def compress(self, method: str = "turboquant", bits: int = 3) -> None: ...
    def export_onnx(self, path: str) -> None: ...
```

### MATHIRPluginV7 (Doctoral-Grade, 100% backward compatible)

```python
class MATHIRPluginV7(nn.Module):
    """
    V7 plugin with all 8 theoretical advances over V6.
    Drop-in replacement for MATHIRPlugin — same .perceive() / .store() / .recall() interface.

    Enable V7 features via config (or config/v7.yaml):
        episodic_type: "ebbinghaus"      # Theorem 2
        semantic_type: "hyperbolic"      # Poincaré ball
        immune_type:    "mahalanobis"    # Theorem 4
        use_variational: True
        use_sparse_coding: True          # Theorem 5
        use_cross_attention: True
        use_neural_ode: True
        use_infonce: True
    """

    def __init__(self, embedding_dim: int, config: dict = None): ...
    def perceive(self, embedding: Tensor) -> dict: ...
    def store(self, experience: dict) -> None: ...
    def recall(self, query: Tensor, k: int = 3) -> list: ...
    def forget(self, threshold: float = 0.1) -> None: ...
    def get_stats(self) -> dict: ...
```

### EmbeddingProvider

```python
class EmbeddingProvider(ABC):
    def embed_text(self, text: str) -> Tensor: ...
    def embed_batch(self, texts: List[str]) -> Tensor: ...
    def embedding_dim(self) -> int: ...
```

### Config

```python
def load_config(path: str = None) -> dict: ...
def get_default_config() -> dict: ...
def merge_config(base: dict, override: dict) -> dict: ...
```

---

## Configuration Schema

```yaml
# Full configuration schema
memory:
  embedding_dim: int          # LLM embedding dimension (any value)
  internal_dim: int           # MATHIR internal dimension (default: 272)
  working_capacity: int       # Working memory slots (default: 64)
  episodic_capacity: int      # Episodic memory slots (default: 1000)
  semantic_prototypes: int    # Semantic prototypes (default: 256)
  immunological_capacity: int # Immune bank size (default: 100)
  kl_coefficient: float       # Router KL constraint (default: 0.01)
  anomaly_threshold: float    # Novelty threshold (default: 2.0)
  decay_rates: list[float]    # Temporal retention (default: [0.9, 0.7, 0.5])

compression:
  enabled: bool               # Enable compression (default: true)
  method: str                 # "turboquant" | "none" (default: "turboquant")
  bits: int                   # Quantization bits: 2|3|4 (default: 3)
  episodic_only: bool         # Only compress episodic (default: false)

inference:
  backend: str                # "pytorch" | "onnx" | "rust" (default: "pytorch")
  device: str                 # "cpu" | "cuda" | "auto" (default: "auto")
  precision: str              # "float32" | "float16" | "int8" (default: "float32")

router:
  type: str                   # "kl_constrained" | "learned" | "uniform"
  num_memories: int           # Number of memory tiers (default: 4)
  entropy_coefficient: float  # Entropy bonus (default: 0.01)
  temperature: float          # Softmax temperature (default: 1.0)

providers:
  default: str                # "ollama" | "openai" | "huggingface" | "direct"
  ollama:
    url: str                  # Ollama server URL (default: "http://localhost:11434")
    model: str                # Model name (default: "llama3.2:3b")
  openai:
    api_key: str              # API key (from env: OPENAI_API_KEY)
    model: str                # Model name (default: "text-embedding-3-small")
  huggingface:
    model: str                # Model name (default: "Qwen/Qwen2.5-7B-Instruct")
    device: str               # Device (default: "auto")
```

---

## Embedding Dimensions Reference

| Model | Dim | Provider | Priority |
|---|---|---|---|
| Qwen2.5-0.5B | 896 | HuggingFace | Edge |
| Qwen2.5-7B | 3584 | HuggingFace | **Primary** |
| LLaMA 3.1-8B | 4096 | HuggingFace | **Primary** |
| LLaMA 3.2-1B | 2048 | HuggingFace | Edge |
| Mistral-7B | 4096 | HuggingFace | **Primary** |
| Gemma 2-2B | 2304 | HuggingFace | Edge |
| Gemma 3-12B | 3840 | HuggingFace | Primary |
| DeepSeek-V3 | 7168 | HuggingFace | Advanced |
| OpenAI text-embedding-3-small | 1536 | API | Primary |
| OpenAI text-embedding-3-large | 3072 | API | Primary |
| Cohere embed-v4 | 1024 | API | Primary |
| Ollama (any model) | varies | API | **Easiest** |

---

## Performance Targets

| Metric | Current | Target | Method |
|---|---|---|---|
| Inference (CPU) | 7.3ms | <2ms | ONNX + Rust |
| Inference (GPU) | 1-2ms | <0.5ms | ONNX + TensorRT |
| Inference (edge) | 30-50ms | <5ms | ONNX + Rust |
| Memory size | 1.4 MB | <60 KB | TurboQuant 3-bit |
| Episodic recall | 0.8ms | <0.05ms | Rust SIMD |
| Quality loss | 0% | <0.1% | ONNX float32 |
| Retrieval quality (default) | 31.6% | 45.7% | Approach D for batch |
| Retrieval quality (online) | 31.6% | 31.6% | Approach A (parity with FAISS) |
| Hybrid retrieval (cold) | 494 ms | 494 ms | Approach D baseline (V7.1) |
| Hybrid retrieval (warm + cache, V7.2) | 6 ms median | 3-220 ms | LRU result cache, 80-85% hit rate |
| Cache hit rate (chat, V7.2) | 0% | 80-85% | LRU on `(query, doc)` pairs |
| Cache hit rate (driving, V7.2) | 0% | 90%+ | Per-scene repeat traffic |
| Cache speedup (V7.2) | 1× | **5-12×** | Cache hit short-circuits BM25 + cross-encoder |

---

## Compatibility Matrix

| LLM | Embedding | Provider | MATHIR | Notes |
|---|---|---|---|---|
| Qwen2.5 (local) | HuggingFace | ✅ | ✅ | Best open-source |
| LLaMA 3.x (local) | HuggingFace | ✅ | ✅ | Fully compatible |
| Mistral (local) | HuggingFace | ✅ | ✅ | Fully compatible |
| DeepSeek (local) | HuggingFace | ✅ | ✅ | Use bottleneck |
| OpenAI | API | ✅ | ✅ | Best API |
| Cohere | API | ✅ | ✅ | Best embedding API |
| Ollama | API | ✅ | ✅ | Easiest local |
| Claude | ❌ No API | ⚠️ | ✅ | Use separate encoder |
| GPT-4o | ❌ No API | ⚠️ | ✅ | Use separate encoder |

For Claude/GPT: use a separate embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) to generate embeddings from the same text.

---

## Use Cases — VectorDB vs MATHIR

Two real-world deployments are validated end-to-end in [`docs/MATHIR_VS_VECTORDB_USE_CASES.md`](docs/MATHIR_VS_VECTORDB_USE_CASES.md). The summary is below.

### Use Case 1 — LLM Chat Assistant

**Architecture:**

```
User → LLM (Claude / GPT-5 / Qwen) → turn embedding
                                     ↓
                       MATHIR HybridEpisodicMemory
                       (dense + BM25 + cross-encoder + LRU cache)
                                     ↓
                       top-5 context chunks → LLM answer
```

**Where VectorDB wins:**
- Sub-10 ms SLA (autocomplete, typeahead): VectorDB is the only system under budget.
- First-turn cold start with no history: 0.05 ms vs 494 ms cold.
- Streaming corpus, re-index every minute: VectorDB's `add()` is O(1); MATHIR's BM25+CE rebuild is O(n).

**Where MATHIR wins:**
- **Follow-up turns** — the LRU cache hits 80-85% of the time, serving paraphrases of prior questions in **6 ms median**. VectorDB has no notion of conversation context.
- **Technical / domain queries** — BM25 + cross-encoder catches "Navier-Stokes", "boundary layer", "Reynolds number" (technical terms that pure cosine misses). Quality 45.7% vs 31.6% = **+14.1 pp**.
- **Multi-turn synthesis** — the LLM can read top-5 chunks and synthesize; VectorDB only returns top-1 with no score breakdown.
- **Anomaly flag** — when the LLM has answered a question wrong before, immunological memory flags the repeat and the policy can re-route.

**Measured (chat, `benchmarks/stress_cache_warm.py`):**

| System | Median | P95 | QPS | Quality | Cache hit |
|---|---:|---:|---:|---:|---:|
| FAISS VectorDB | 0.05 ms | 0.18 ms | 20,392 | 31.6% | n/a |
| MATHIR D (no cache) | 2,329 ms | 3,140 ms | 0.4 | 52.5% | 0% |
| **MATHIR D (warm + cache)** | **6 ms** | 2,466 ms | **2.5** | 52.5% | **82.7%** |

The cache's **median** of 6 ms is the user-perceived latency for a follow-up turn. The mean stays high because the cold misses pay 494 ms; as the session progresses, mean collapses toward median.

### Use Case 2 — Autonomous Driving (VLM Plugin)

**Architecture:**

```
VLM (Qwen3-VL / LLaVA-1.6) → per-frame embedding
                              ↓
                MATHIR HybridEpisodicMemory
                (dense + BM25 + cross-encoder + LRU cache + anomaly)
                              ↓
                top-5 past situations + novelty flag → decision head
```

**Where VectorDB wins:**
- Static HD map, fixed routes: 0.05 ms per query.
- Pre-recorded fleet data, batch offline analytics: 20,392 QPS.
- 50 Hz control loop sub-frame budget: VectorDB is the only system under 1 ms.

**Where MATHIR wins (VectorDB literally cannot do this):**

| Capability | VectorDB | MATHIR V7.2 |
|---|:---:|:---:|
| Find nearest neighbour | ✅ (0.05 ms) | ✅ (6 ms warm) |
| Update corpus in real-time | ⚠️ (rebuild) | ✅ (`store()` is O(log n)) |
| **Adapt the index to the route** | ❌ | ✅ (episodic store fills with what the car actually sees) |
| **Detect novel situations** | ❌ | ✅ (immunological / Mahalanobis memory) |
| **Bias policy with retrieved context** | ⚠️ (top-1 only) | ✅ (top-5 + scores + novelty flag) |
| **Cross-correlate symbolic labels with embeddings** | ❌ | ✅ (BM25 stage) |

**Why driving is the killer use case for MATHIR:** VectorDB treats all 4 environments (highway, city, country, tunnel) the same — same cosine, same retrieval. MATHIR's episodic memory **differentiates them within 30 minutes** because `store()` calls fill the bank with the situations the policy actually handles, and the immunological memory flags the tunnels the car has never seen before.

**Safety argument (VectorDB has no novelty signal):** If the embedding of "black blob in the middle of the road" is not in the corpus, VectorDB returns the nearest miss ("road surface", "shadow") with high confidence. MATHIR's immunological memory returns a high anomaly score on the same input, which the policy head can route to an emergency maneuver. This is **Theorem 4 (Anomaly Optimality)** from V7, validated empirically in `benchmarks/v6_vs_v7.py`.

### Side-by-Side (Pareto Frontier)

```
   Quality
   0.50 ┤                                          ● D cold (2 QPS, 0.457)
        │                                          ◆ D warm (5+ QPS, 0.457)
   0.45 ┤
        │
   0.40 ┤
        │
   0.35 ┤
        │  ● A (657, 0.316)  ● C (97, 0.316)  ● FAISS (20,392, 0.316)
   0.30 ┤
        │
   0.25 ┤
        │  ● B (425, 0.291)
   0.20 ┤
        │  ● V7 default (1,338, 0.197)
        └────────────────────────────────────────────────────────────
            10⁰        10¹       10²       10³        10⁴        10⁵
                                   QPS (log)
```

- **FAISS** dominates the right edge (speed).
- **MATHIR D cold** dominates the upper-left (quality).
- **MATHIR D warm** closes most of the speed gap: from 2 QPS to 5+ QPS with the cache, while keeping D's quality.

### Production Deployment Cheat-Sheet

**Chat assistant:**
- Use MATHIR with `use_result_cache=True`, `use_adaptive_rerank=True`.
- Pre-warm the cache with the top-100 most-asked questions for faster cold start.
- For sub-10 ms SLA, wrap VectorDB in front of MATHIR as the L1 retriever (cascade, V8 preview).
- Watch: `cache_hit_rate` (> 70% after warmup), `median_latency_ms` (< 50 ms), `anomaly_score_distribution` (< 30% anomalies).

**Autonomous driving:**
- Use MATHIR with cache on, adaptive rerank on, anomaly detection on.
- Cache hit rate is 90%+ in driving (higher than chat) because driving revisits the same situation frequently.
- Always enable novelty detection. VectorDB has no notion of "out of distribution."
- The 50 Hz control loop runs **without** retrieval; retrieval is at 1-2 Hz for *perception events*. The 5+ QPS warm MATHIR is more than enough headroom.
- Watch: `anomaly_score > 2.0` rate (< 5% on known routes, < 15% on new routes), `episodic_store_size` (cap at 100K frames, LRU-evict).

### Cache Contract (read this if you deploy)

| Property | Value |
|---|---|
| Cache key | `(query_text, query_embedding_fingerprint)` |
| Cache value | top-k `(indices, scores)` |
| Capacity | 10,000 entries (default) |
| Eviction | LRU |
| Score preservation | 100% — cached results return the **same** scores |
| Quality regression | 0% — cache does not modify scores |
| Invalidation | explicit `clear_cache()`; opt-in `ttl_seconds` config |

The cache **does not** change what MATHIR returns. It only short-circuits the BM25 + cross-encoder stages when it has seen the same `(query_text, query_embedding)` pair before. This is why the quality on the warm path is exactly 45.7% — identical to the cold path.

**Read the full deployment guide: [`docs/MATHIR_VS_VECTORDB_USE_CASES.md`](docs/MATHIR_VS_VECTORDB_USE_CASES.md)**

---

## Version Strategy

| Version | Focus | Status | Evidence |
|---|---|---|---|
| V1-V3 | Core architecture | ✅ Done | `mathir_model.py`, `train_evolution.py` |
| V4 | mHC integration | ✅ Done | `mathir_lib/mhc_v5.py` |
| V5 | KL router + immune memory | ✅ Done | `mathir_lib/mathir_v5.py` |
| V5.1 | Bug fixes + clean codebase | ✅ Done | 21 bugs fixed in 18 files |
| **V6** | **MATHIRPlugin API** | ✅ **Done** | `mathir_lib/plugin.py` (12/12 tests pass, 13/13 stress) |
| **V7** | **8 novel algorithms + 6 theorems** | ✅ **Done** | `plugin_v7.py` (49/49 tests, 9.3× compression) |
| **V7.1** | **Retrieval research: 4 new approaches** | ✅ **Done** | `mathir_lib/retrieval/` (130/130 tests, +26pp quality) |
| **V7.2** | **Latency optimization + use case docs** | ✅ **Done** | LRU cache (5-12× speedup, 80-85% hit), adaptive rerank, ONNX CE, `docs/MATHIR_VS_VECTORDB_USE_CASES.md` (62 new tests) |
| **V7.5** | **Real BEIR benchmarks** (SciFact 0.7441, NFCorpus 0.3657, ArguAna 0.6613) + LIRS 100% recovery, Router 100% accuracy, Ensemble 100% cold-start | ✅ **Done** | 4 swarm fixes, security CLEAN |
| **V7.6** | **Universal Bridge (UNIBRI) + Latin Names** — cross-provider, cross-lingual, vocabulary-free, mathematically proven (11 theorems) | ✅ **Done** | 137/137 tests pass, verified with 3 Ollama model architectures |
| V8 | Cascade architecture + arXiv | 🔜 Next | Auto-routes Approach A vs D per query |
| V9 | Rust acceleration (PyO3) | 📋 Planned | Design in `docs/RUST_ML_RESEARCH_REPORT.md` |
| V10 | ONNX export + edge binary | 📋 Planned | Target Jetson/8GB |
| V11 | Open source release | 📋 Planned | arXiv + PyPI |

### V7 Theoretical Advances

V7 is **doctoral-grade**: 8 new memory modules and 6 new theorems (see [`docs/THEORY_V7.md`](docs/THEORY_V7.md), [`docs/PROOFS.md`](docs/PROOFS.md), [`docs/V7_PAPER.md`](docs/V7_PAPER.md)).

**8 new algorithms** (all in `mathir_lib/memory/`):

| Module | Theoretical basis |
|---|---|
| `EbbinghausMemory` | Theorem 2 — retention guarantee via `R(t)=e^(-t/S)` |
| `SparseCodingMemory` | Theorem 5 — reconstruction bound `O(sσ²/K)` |
| `VariationalMemory` | Reparameterization + ELBO (uncertainty per slot) |
| `CrossAttentionMemory` | Learned Q/K/V addressing |
| `HyperbolicMemory` | Poincaré ball — exponential volume for trees |
| `InfoNCELoss` | Mutual-information lower bound (Oord et al. 2018) |
| `NeuralODEMemory` | Continuous-time dynamics (RK4 integrator) |
| `MahalanobisImmunologicalMemory` | Theorem 4 — Neyman-Pearson optimal anomaly |

**6 new theorems** (formal proofs in `docs/PROOFS.md`):

1. **Information Capacity** — `I(X;M) ≤ (N+P+W+I+2V+s)d log(1+SNR)`
2. **Retention Guarantee** — `Acc(K) ≥ 1 - O(KLη/N)` (Ebbinghaus)
3. **Router Convergence** — `O(1/ε)` iterations (Robbins-Monro)
4. **Anomaly Optimality** — Neyman-Pearson for Gaussian normal data
5. **Sparse Coding Bound** — `O(sσ²/K)` error (Olshausen-Field)
6. **mHC Geometry** — contraction mapping guarantee (overrelaxed Sinkhorn-Knopp)

**Measured impact** (`docs/BENCHMARK_V6_VS_V7.md`):

- 9.3× smaller memory footprint (1,088,000 → 116,976 bytes for 1000 × 272-dim embeddings)
- 49/49 unit tests pass (`tests/test_v7_memory.py`)
- 16 integration tests pass (`tests/test_v7_integration.py`)
- 100% backward compatible with V6 (`MATHIRPluginV7` accepts the same `.perceive() / .store() / .recall()` interface and the same default config)

---

## V7.1 Retrieval Research Findings

Doctoral-level research conducted in June 2026 identified a **12-14% retrieval-quality gap** in the V7 default path. Four new approaches were built, unit-tested, and benchmarked on a real textbook corpus (White's *Fluid Mechanics*, 885 pages, 200 chunks, 50 ground-truth queries).

### Diagnosis

The 64-dim semantic projection in V7's default retrieval path was the **root cause of the 12-14% quality loss**. Projecting a 384-dim LLM embedding down to 64 dims for prototype matching throws away information that the raw embedding carries. This shows up as:

- V7 default overlap: **19.7%**
- FAISS on raw 384-dim: **31.6%** (the natural dense baseline)
- → Gap: **~12pp** in favor of "don't project, just use raw"

### The Four Approaches

#### Approach A — Raw Embedding Bypass

- **What:** Skip the 64-dim projection; use the LLM's full embedding dimension directly for similarity search.
- **Quality:** 31.6% overlap — matches raw FAISS.
- **Throughput:** 657 QPS, 1.54 ms median latency.
- **Best for:** Online / interactive workloads — recommended as the new default.
- **Module:** `mathir_lib/retrieval/raw_embedding.py`
- **Tests:** 28 unit tests.

#### Approach B — Multi-Encoder Ensemble

- **What:** Combine 2-3 encoders (e.g., MiniLM + BGE-small) with score-level fusion (RRF or weighted sum).
- **Quality:** 29.1% overlap — surprisingly *below* single-encoder raw.
- **Throughput:** 425 QPS, 2.20 ms median latency.
- **Best for:** Domains where no single encoder dominates; otherwise the complexity tax hurts.
- **Module:** `mathir_lib/retrieval/multi_encoder.py`
- **Tests:** 36 unit tests.

#### Approach C — FAISS-Backed Index

- **What:** Build a FAISS index (HNSW or IVF) over MATHIR's episodic memory for sub-linear search.
- **Quality:** 31.6% overlap — same as raw dense; FAISS doesn't change recall on top of dense vectors.
- **Throughput:** 97 QPS, 8.88 ms median latency — slower than the brute-force numpy implementation.
- **Best for:** Very large corpora (≥50K chunks) where brute-force becomes the bottleneck.
- **Module:** `mathir_lib/retrieval/faiss_index.py`
- **Tests:** 32 unit tests.

#### Approach D — Hybrid BM25 + Dense + Cross-Encoder

- **What:** Three-stage pipeline:
  1. **BM25** (lexical, sparse) — captures exact term matches that dense retrieval misses.
  2. **Dense retrieval** (raw 384-dim) — captures semantic similarity.
  3. **Cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — joint query+passage scoring.
  - Results from stages 1+2 are fused with **Reciprocal Rank Fusion (RRF, k=60)**.
  - The top-50 fused candidates are re-ranked by the cross-encoder.
- **Quality:** **45.7% overlap** — the winner, by 14.1pp over dense-only.
- **Throughput:** 2 QPS, 494 ms median latency.
- **Best for:** Batch, offline, RAG-style workloads where quality dominates latency.
- **Module:** `mathir_lib/retrieval/hybrid_bm25_ce.py`
- **Tests:** 34 unit tests.

### Master Comparison Table

| System | Overlap (quality) | Throughput | Latency (median) | Verdict |
|---|:---:|:---:|:---:|---|
| FAISS VectorDB (raw 384-dim) | 31.6% | 20,392 QPS | 0.05 ms | Edge / ultra-low-latency |
| V7 default (64-dim projection) | 19.7% | 1,338 QPS | 0.66 ms | **Deprecated** — quality regression |
| **Approach A (Raw Embedding)** | 31.6% | 657 QPS | 1.54 ms | ✅ **New default** — best balance |
| Approach B (Multi-Encoder) | 29.1% | 425 QPS | 2.20 ms | Niche (multi-encoder domains) |
| Approach C (FAISS-backed) | 31.6% | 97 QPS | 8.88 ms | Scale-out (≥50K chunks) |
| **Approach D (Hybrid BM25+CE)** | **45.7%** | 2 QPS | 494 ms | ✅ **Quality king** — batch/offline |

### Three Prioritized Recommendations

1. **Default → Approach A.** Drop-in, 100% backward compatible. Restores parity with raw FAISS (31.6%) while keeping MATHIR's 4-tier memory and online learning.
2. **Batch / RAG → Approach D.** When you can afford ~500ms per query (offline, retrieval-augmented generation, RAGAS eval, etc.), the +14.1pp quality is worth it.
3. **Edge / Streaming → FAISS raw.** For ≤50K chunks, brute-force numpy on raw embeddings beats FAISS in latency (0.05ms) at no quality cost.

### Validation

- **130 new unit tests pass** (A: 28, B: 36, C: 32, D: 34).
- **49/49 V7 tests still pass** — no regressions.
- **16/16 V7 integration tests still pass** — no regressions.
- **100% backward compatible** with `MATHIRPluginV7` (same `.perceive() / .store() / .recall()` interface; new behavior is opt-in via `config["retrieval"]["strategy"]`).

### Why This Matters

Pure dense retrieval is **not** the ceiling. The 12-14pp gap closed by Approach A *and* the +14.1pp gained by Approach D together represent a **+26 percentage point** improvement on a real-world retrieval task. The lesson: **dense models miss exact lexical matches, and 64-dim projections miss everything else.** The hybrid BM25 + dense + CE pipeline is what production RAG should have been doing all along.

See `docs/RETRIEVAL_RESEARCH_REPORT.md` for the full doctoral analysis and `docs/BENCHMARK_V6_VS_V7.md` for the master comparison table.

---

## Testing Strategy

### Unit Tests
```bash
pytest tests/test_plugin.py       # Plugin API
pytest tests/test_memory.py       # Memory tiers
pytest tests/test_compression.py  # TurboQuant
pytest tests/test_router.py       # Router
pytest tests/test_providers.py    # Embedding providers
pytest tests/stress_test.py       # V6 stress test (13/13 pass)
pytest tests/test_v7_memory.py    # V7 unit tests (49/49 pass)
pytest tests/test_v7_integration.py  # V7 integration (16 tests)
```

### Integration Tests
```bash
pytest tests/test_integration.py  # Full pipeline
pytest tests/test_ollama.py       # Ollama integration
pytest tests/test_huggingface.py  # HuggingFace integration
```

### Benchmarks
```bash
python benchmarks/retention.py    # Long-term retention
python benchmarks/generalization.py  # Cross-scenario
python benchmarks/latency.py      # Inference speed
python benchmarks/compression.py  # Compression quality
```

---

## Deployment Targets

| Platform | Language | Binary Size | Latency |
|---|---|---|---|
| Linux x86_64 | Rust | ~5 MB | <2ms |
| Windows x86_64 | Rust | ~5 MB | <2ms |
| macOS ARM64 | Rust | ~5 MB | <1ms |
| NVIDIA Jetson | Rust + CUDA | ~10 MB | <1ms |
| Raspberry Pi 5 | Rust (CPU) | ~5 MB | <5ms |
| WASM (browser) | Rust + wasm-pack | ~2 MB | <10ms |
| Python package | Python + PyO3 | ~15 MB | <2ms |

---

## License

MIT

---

## References

- [TurboQuant Paper](https://arxiv.org/abs/2504.19874)
- [DeepSeek mHC Paper](https://arxiv.org/abs/2512.24880)
- [KV Cache Research Report](docs/KV_CACHE_RESEARCH_REPORT.md)
- [Rust ML Research Report](docs/RUST_ML_RESEARCH_REPORT.md)
- [Future Vision](FUTURE_VISION.md)
- [Implementation Plan](IMPLEMENTATION.md)
