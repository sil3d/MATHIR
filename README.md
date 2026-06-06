# MATHIR

**Memory-Augmented Tensor Hybrid with Intelligent Routing**

The first adaptive memory layer that gives any LLM persistent memory, real-time learning, and anomaly detection — on edge hardware.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-008000)
![VRAM](https://img.shields.io/badge/VRAM-0.6%20GB-brightgreen)
![Version](https://img.shields.io/badge/version-7.7.1-blue)

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

## Key Innovations

### 1. KL-Constrained Router
PPO-style trust region optimization prevents collapse to a single memory tier. The router learns when to use each tier.

### 2. Immunological Memory
Anomaly detector that learns what "normal" looks like. Flags novel inputs for the LLM. **Theorem 4: NP-optimal detection.**

### 3. Universal Bridge (UNIBRI)
Cross-provider, cross-lingual recall without retraining. "Schrödinger" = "Schrodinger". French query finds English content. 137/137 tests pass.

### 4. Online Learning
Unlike pre-trained models, MATHIR never stops learning. Every observation updates prototypes. Every experience fills episodic memory.

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
│   ├── models/              # GGUF models
│   └── config.json          # Model configuration
│
├── benchmarks/              # Performance benchmarks
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── examples/                # Demo scripts
└── config/                  # Configuration files
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

---

## Documentation

| Document | Description |
|---|---|
| [`docs/01_MASTER_RESEARCH_PAPER.md`](docs/01_MASTER_RESEARCH_PAPER.md) | Full research paper |
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

## Citation

```bibtex
@article{mathir2026,
  title={MATHIR: Memory-Augmented Tensor Hybrid with Intelligent Routing},
  author={Mbama Kombila, Prince Gildas},
  year={2026},
  url={https://github.com/sil3d/MATHIR}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Author

**Prince Gildas Mbama Kombila**
- GitHub: [@sil3d](https://github.com/sil3d)
- Email: soilearn3d@gmail.com
