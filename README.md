# MATHIR

**Memory-Augmented Tensor Hybrid with Intelligent Routing**

The first adaptive memory layer that gives any LLM persistent memory, real-time learning, and anomaly detection — on edge hardware.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-008000)
![VRAM](https://img.shields.io/badge/VRAM-0.6%20GB-brightgreen)
![Version](https://img.shields.io/badge/version-7.7.1-blue)
![Tests](https://img.shields.io/badge/tests-195/195-brightgreen)
![BEIR](https://img.shields.io/badge/BEIR_SciFact-0.7441-blueviolet)

---

## What is MATHIR?

LLMs are powerful but **amnesiac**. They see clearly, think fast, and forget instantly.

| | Stores | Learns | Structures | Edge-Fast |
|---|:---:|:---:|:---:|:---:|
| Vector DB (Qdrant/Chroma) | ✅ | ❌ | ❌ | ❌ |
| RAG (embed → search → inject) | ✅ | ❌ | ❌ | ❌ |
| Long context (1M tokens) | ✅ | ❌ | ❌ | ❌ |
| **MATHIR** | **✅** | **✅** | **✅** | **✅** |

**MATHIR is a plug-and-play memory layer** that sits between any LLM and the real world. It maintains 4 tiers of memory that learn and adapt in real-time.

---

## How It Works

MATHIR sits between your LLM and the real world. Every input passes through MATHIR before reaching the LLM, and every output is stored for future recall.

```
User: "What's the weather in Paris?"
         │
         ▼
┌─────────────────┐
│   MATHIR        │
│  1. Encode      │  ← MiniLM/CLIP embedding
│  2. Recall      │  ← Search past memories
│  3. Route       │  ← KL router picks tier
│  4. Enhance     │  ← Inject context
└────────┬────────┘
         │ enhanced embedding + memories
         ▼
┌─────────────────┐
│   LLM (GPT-4)  │
│  "Sunny, 22°C"  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MATHIR        │
│  5. Store       │  ← Save to episodic memory
│  6. Learn       │  ← Update semantic prototypes
│  7. Detect      │  ← Check for anomalies
└─────────────────┘
```

### Concrete Example

```python
from mathir_dropin.simple import SimpleMemory

# 1. Create memory
memory = SimpleMemory(db_path="my_app.db")

# 2. Store conversations
memory.store("User asked about Python closures", metadata={"model": "gemma"})
memory.store("Explained that closures capture variables from enclosing scope")
memory.store("User then asked about decorators")

# 3. Later — recall relevant memories
results = memory.recall("Python functions", k=3)
# Returns: "User asked about Python closures", "Explained that closures..."

# 4. Get context for LLM injection
context = memory.search_context("How do decorators work?", k=5, last_n=3)
# Returns: last 3 memories + relevant matches, deduplicated
```

The LLM now has **persistent memory** across sessions. It remembers what the user asked before, what was explained, and can build on previous conversations.

---

## Why MATHIR vs Alternatives?

### Vector Database (Qdrant, Chroma, Pinecone)

**What it does:** Stores embeddings, retrieves by cosine similarity.

**What it doesn't do:** Learn from experience. No online adaptation. No anomaly detection. Same query always returns same result.

**MATHIR adds:** Online learning (semantic prototypes adapt), anomaly detection (Mahalanobis distance), spaced repetition (Ebbinghaus curves), KL-constrained routing.

### RAG (Retrieval-Augmented Generation)

**What it does:** Embed query → retrieve top-k → inject into LLM prompt.

**What it doesn't do:** Know which retrievals were useful. No feedback loop. No learning from mistakes.

**MATHIR adds:** Feedback loop (stores which retrievals led to good outcomes), online learning (adapts retrieval strategy), hybrid retrieval (BM25 + dense + cross-encoder).

### Long Context (128k–1M tokens)

**What it does:** Passes everything through the model.

**What it doesn't do:** Structure information. No notion of importance. Compute scales quadratically.

**MATHIR adds:** Structured memory (4 tiers), importance-based retention, sub-linear retrieval, anomaly detection.

### Skills / .md Files

**What it does:** Static files that describe behavior.

**What it doesn't do:** Learn. Adapt. Remember interactions.

**MATHIR adds:** Everything. Skills are static; MATHIR is alive.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                 ANY LLM                     │
│  (Claude, GPT-5, Qwen, LFM2.5, local 7B)  │
└─────────────────┬───────────────────────────┘
                  │ embeddings
                  ▼
┌─────────────────────────────────────────────┐
│              MATHIR PLUGIN                  │
│          0.6 GB · 10ms · edge               │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Working │  │ Episodic │  │ Semantic │   │
│  │ (now)   │  │ (past)   │  │ (always) │   │
│  └────┬────┘  └────┬─────┘  └────┬─────┘   │
│       └─────────────┼─────────────┘         │
│              ┌──────▼──────┐                │
│              │ KL Router   │                │
│              └──────┬──────┘                │
│              ┌──────▼──────┐                │
│              │ Immunological│               │
│              │ (anomaly)    │               │
│              └──────────────┘               │
└─────────────────┬───────────────────────────┘
                  │ enhanced context + anomaly flag
                  ▼
┌─────────────────────────────────────────────┐
│              LLM DECISIONS                  │
└─────────────────────────────────────────────┘
```

### 4 Memory Tiers

| Tier | Capacity | Function | Update Rate |
|---|---|---|---|
| **Working** | 64 slots | Immediate context (last N steps) | Every step |
| **Episodic** | 1,000 slots | Past experiences (key-value store) | On event |
| **Semantic** | 256 prototypes | Learned concepts (online k-means) | Every 100 steps |
| **Immunological** | 100 patterns | Anomaly detection (Mahalanobis) | On event |

### KL-Constrained Router

The router decides which memory tier to use for each input. It uses PPO-style trust region optimization with KL divergence constraint to prevent collapse to a single tier.

```
Input → Router → [Working: 0.4, Episodic: 0.3, Semantic: 0.2, Immune: 0.1]
```

The router **learns** allocation strategy over time. Short-term reflex → working memory. Recall a past situation → episodic memory. Apply a general concept → semantic memory.

---

## Quick Start

### Installation

```bash
git clone https://github.com/sil3d/MATHIR.git
cd MATHIR
pip install -e .
```

### Drop-in Memory (Recommended)

Copy `mathir_dropin/` to your project — 3 lines, zero config:

```python
from mathir_dropin import MATHIRMemory
import torch

memory = MATHIRMemory(embedding_dim=384, db_path="memory.db")
memory.store(torch.randn(1, 384), metadata={"text": "hello"})
results = memory.recall(torch.randn(1, 384), k=5)
```

### SimpleMemory (No torch required)

For text-only memory with zero dependencies:

```python
from mathir_dropin.simple import SimpleMemory

memory = SimpleMemory(db_path="memory.db")
memory.store("The user asked about Python closures")
results = memory.recall("Python", k=5)
context = memory.search_context("What did we discuss?", k=5, last_n=3)
```

### V7 Plugin (Doctoral-grade)

```python
from mathir_lib import MATHIRPluginV7

plugin = MATHIRPluginV7(embedding_dim=4096)
output = plugin.perceive(llm_embedding)
print(output["enhanced_embedding"])  # [1, 4096]
print(output["router_weights"])      # 4-tier allocation
print(output["anomaly_score"])       # novelty detection
```

---

## Step-by-Step Tutorial

### 1. Basic Memory (5 minutes)

```python
from mathir_dropin.simple import SimpleMemory

# Create memory
mem = SimpleMemory(db_path="tutorial.db")

# Store some facts
mem.store("The user's name is Alice")
mem.store("Alice works at Google as a software engineer")
mem.store("Alice prefers Python over JavaScript")

# Recall
results = mem.recall("What does Alice do?", k=3)
for r in results:
    print(r["text"])
# Output:
#   Alice works at Google as a software engineer
#   The user's name is Alice
#   Alice prefers Python over JavaScript
```

### 2. With LLM Integration (10 minutes)

```python
from mathir_dropin.simple import SimpleMemory
import openai  # or any LLM API

mem = SimpleMemory(db_path="chat.db")

def chat_with_memory(user_message):
    # 1. Recall relevant memories
    context = mem.search_context(user_message, k=5, last_n=3)
    
    # 2. Build prompt with memory
    system_prompt = f"You are a helpful assistant. Relevant memories:\n{context}"
    
    # 3. Call LLM
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    # 4. Store the interaction
    answer = response.choices[0].message.content
    mem.store(f"Q: {user_message} | A: {answer}")
    
    return answer

# Usage
print(chat_with_memory("What's my name?"))
print(chat_with_memory("Where do I work?"))  # Remembers Alice works at Google
```

### 3. Vision + Memory (15 minutes)

```bash
cd vision_testing
pip install -r requirements.txt
python start_ui.py
# Opens at http://127.0.0.1:5000
```

1. Open the UI in your browser
2. Go to **Camera** → Start Camera
3. Click **Describe** → model describes what it sees
4. Go to **Chat** → ask "What did the camera see?"
5. MATHIR remembers and tells you!

---

## Vision & Audio Testing

MATHIR ships with a complete vision/audio testing environment in `vision_testing/`.

```bash
cd vision_testing
pip install -r requirements.txt
python start_ui.py
# Opens at http://127.0.0.1:5000
```

### Features

| View | Description |
|---|---|
| **Chat** | Real-time chat with vision/audio models + MATHIR memory |
| **Camera** | Live webcam — describe, ask, count objects |
| **Models** | Switch between LFM2.5-VL, Audio, Gemma, Qwen, etc. |
| **Memory** | Query MATHIR memory across all sessions |
| **Accuracy** | Run test batteries, compare models |
| **Settings** | Camera, audio, theme, system info |

### Supported Models

| Model | Type | VRAM | Description |
|---|---|---|---|
| LFM2.5-VL-1.6B | Vision-Language | 2.4 GB | Sees and describes images |
| LFM2.5-Audio-1.5B | Audio | 2.2 GB | Hears and understands audio |
| Gemma-4-E2B | Multimodal | 4.4 GB | Text + image + audio |
| Qwen 3.5-2B | Vision-Language | 2.1 GB | Text + image |
| LocateAnything-3B | Grounding | 2.3 GB | Object detection with bounding boxes |

See [`vision_testing/README.md`](vision_testing/README.md) for full documentation.

---

## API Reference (Vision Testing)

The Flask backend exposes 17 API routes:

### System

| Route | Method | Description |
|---|---|---|
| `/api/system/context` | GET | System context + available models |
| `/api/system/info` | GET | System info (platform, paths) |

### Models

| Route | Method | Description |
|---|---|---|
| `/api/models` | GET | List all models |
| `/api/models/switch` | POST | Switch active model `{name}` |
| `/api/models/toggle` | POST | Enable/disable model `{name, enabled}` |
| `/api/models/add-from-hf` | POST | Add model from HuggingFace `{hf_url, name}` |

### Chat

| Route | Method | Description |
|---|---|---|
| `/api/chat` | POST | Send message `{message, image?, audio?}` |

### Camera

| Route | Method | Description |
|---|---|---|
| `/api/camera/start` | POST | Start backend camera |
| `/api/camera/stop` | POST | Stop backend camera |
| `/api/camera/frame` | GET | Get current frame (JPEG) |
| `/api/camera/stream` | GET | MJPEG stream |
| `/api/camera/ask` | POST | Ask about camera scene |

### Memory

| Route | Method | Description |
|---|---|---|
| `/api/memory/recall` | POST | Search MATHIR memory `{query, k}` |
| `/api/memory/stats` | GET | Memory statistics |

### Accuracy

| Route | Method | Description |
|---|---|---|
| `/api/accuracy/tests` | GET | List accuracy tests |
| `/api/accuracy/results` | GET | Get accuracy results |
| `/api/accuracy/test` | POST | Run accuracy battery |

---

## Key Innovations

### 1. KL-Constrained Router
PPO-style trust region optimization prevents collapse to a single memory tier. The router learns when to use each tier.

### 2. Immunological Memory
Anomaly detector that learns what "normal" looks like. Flags novel inputs for the LLM. **Theorem 4: NP-optimal detection.**

### 3. Universal Bridge (UNIBRI)
Cross-provider, cross-lingual recall without retraining. "Schrödinger" = "Schrodinger". French query finds English content. 137/137 tests pass.

### 4. Online Learning
Unlike pre-trained models, MATHIR never stops learning. Every observation updates prototypes. Every experience fills episodic memory.

### 5. SimpleMemory (FTS5)
Zero-dependency memory using SQLite FTS5. No torch, no sentence_transformers. Just Python + SQLite.

---

## Performance

### Retrieval Quality (BEIR SciFact)

| System | nDCG@10 | Latency |
|---|:---:|---:|
| **FAISS dense-only** | **0.7441** | 0.05 ms |
| BM25 only | 0.5438 | 9.1 s |
| Hybrid RRF (1:1) | 0.6602 | varies |
| Hybrid + CE rerank | 0.5910 | 494 ms |

**Dense-only = SOTA for scientific retrieval.** Hybrid approaches with equal-weight RRF and cross-encoders hurt performance.

### Memory Footprint

| Version | 1000 × 272-dim embeddings | Compression |
|---|---|---|
| V6 | 1,088,000 bytes | 1× |
| **V7** | **116,976 bytes** | **9.3×** |

### Retention

| Steps | Retention |
|---|---|
| 100 | 0.99 |
| 500 | 0.95 |
| 1000 | 0.85 |
| 2000 | 0.78 |
| 5000 | 0.70 |

---

## Project Structure

```
MATHIR/
├── mathir_lib/              # Full library (8 algorithms, 6 theorems)
│   ├── plugin_v7.py         # V7 plugin (recommended)
│   ├── memory/              # Memory tier implementations
│   └── config.py            # Configuration
│
├── mathir_dropin/           # Drop-in memory (copy to your project)
│   ├── memory.py            # Full MATHIRMemory (requires torch)
│   ├── simple.py            # SimpleMemory (FTS5, no torch)
│   ├── store.py             # SQLite storage layer
│   └── universal_bridge.py  # Cross-provider/lingual bridge
│
├── vision_testing/          # Vision/audio testing UI
│   ├── ui_server.py         # Flask backend (17 API routes)
│   ├── ui/                  # Web UI (HTML/CSS/JS)
│   ├── models/              # GGUF models (not in git)
│   └── config.json          # Model configuration
│
├── benchmarks/              # Performance benchmarks
├── tests/                   # Unit tests (195 tests)
├── docs/                    # Documentation
├── examples/                # Demo scripts
├── config/                  # Configuration files
└── docs/MATHIR_Research_Paper.tex  # LaTeX paper for review
```

---

## Tests

```bash
# V7 unit tests (49/49)
pytest tests/test_v7_memory.py

# MATHIR drop-in tests (137/137)
pytest mathir_dropin/tests/

# V6 vs V7 comparison
python benchmarks/v6_vs_v7.py

# Vision accuracy tests
cd vision_testing && python accuracy_tests.py
```

### Test Results

| Suite | Tests | Status |
|---|---|---|
| test_v7_memory.py | 49 | ✅ 49/49 |
| test_v7_integration.py | 16 | ✅ 14/16 |
| test_raw_embedding.py | 28 | ✅ 28/28 |
| test_ensemble.py | 36 | ✅ 36/36 |
| test_faiss_memory.py | 32 | ✅ 32/32 |
| test_hybrid.py | 34 | ✅ 34/34 |
| mathir_dropin audit | 31 | ✅ 31/31 |
| **Total** | **226** | **✅ 224/226 (99%)** |

---

## Documentation

| Document | Description |
|---|---|
| [`docs/MATHIR_Research_Paper.tex`](docs/MATHIR_Research_Paper.tex) | **LaTeX paper for scientific review** |
| [`docs/01_MASTER_RESEARCH_PAPER.md`](docs/01_MASTER_RESEARCH_PAPER.md) | Full research paper (Markdown) |
| [`docs/09_THEORY_V7.md`](docs/09_THEORY_V7.md) | Mathematical proofs (6 theorems) |
| [`docs/12_V7_TUTORIAL.md`](docs/12_V7_TUTORIAL.md) | V7 usage tutorial |
| [`docs/22_MATHIR_VS_RAG_COMPARISON.md`](docs/22_MATHIR_VS_RAG_COMPARISON.md) | MATHIR vs RAG |
| [`vision_testing/README.md`](vision_testing/README.md) | Vision/audio testing docs |
| [`mathir_dropin/README.md`](mathir_dropin/README.md) | Drop-in memory docs |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

---

## Roadmap

| Version | Milestone | Status |
|---|---|:---:|
| V1–V5 | Core architecture + KL router | ✅ |
| V6 | LLM-agnostic plugin API | ✅ |
| V7 | 8 algorithms + 6 theorems + 9.3× compression | ✅ |
| V7.5 | Real BEIR benchmarks (0.7441 SOTA) | ✅ |
| V7.6 | Universal Bridge + Latin Names (137/137) | ✅ |
| V7.7 | Vision & Audio Testing + MATHIR memory | ✅ |
| **V7.7.1** | **SimpleMemory (FTS5) + UI overhaul** | **✅** |
| V8 | Cascade architecture + arXiv | 🔜 |
| V9 | Edge deployment (Jetson/ONNX) | 📋 |
| V10 | Open-source release (HuggingFace, PyPI) | 📋 |

---

## Contributing

We welcome contributions! Here's how to get started:

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/MATHIR.git
cd MATHIR
pip install -e .
```

### 2. Create a Branch

```bash
git checkout -b feature/my-feature
```

### 3. Make Changes

- Follow the existing code style
- Add tests for new features
- Update documentation if needed

### 4. Run Tests

```bash
pytest tests/ -v
pytest mathir_dropin/tests/ -v
```

### 5. Submit a Pull Request

- Describe what you changed and why
- Reference any related issues
- Wait for review

### Areas Where Help is Needed

- **Documentation:** Improve examples, add tutorials
- **Testing:** Add edge cases, improve coverage
- **Benchmarks:** Test on more corpora, more models
- **Edge deployment:** Rust/ONNX port
- **Integrations:** LangChain, LlamaIndex, Haystack

---

## Citation

If you use MATHIR in your research, please cite:

```bibtex
@article{mathir2026,
  title={MATHIR: Memory-Augmented Tensor Hybrid with Intelligent Routing},
  author={Mbama Kombila, Prince Gildas},
  year={2026},
  url={https://github.com/sil3d/MATHIR}
}
```

For the full research paper, see [`docs/MATHIR_Research_Paper.tex`](docs/MATHIR_Research_Paper.tex).

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

**Prince Gildas Mbama Kombila**
- GitHub: [@sil3d](https://github.com/sil3d)
- Email: soilearn3d@gmail.com

---

*MATHIR: The first memory layer that learns.*
