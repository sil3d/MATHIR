# 📚 MATHIR — Master Project Index

**The ONE file for navigating the entire MATHIR project.**

> V7.5.1 — Real BEIR benchmarks. All 4 memory tiers stress-tested (2hr each).
> Router accuracy 100%, LIRS recovery 100%, cold-start detection 100%.

---

## 🚀 30-Second Orientation

| I want to... | Go to |
|--------------|-------|
| Understand what MATHIR is | [`README.md`](README.md) |
| See the API / class spec | [`AGENT.md`](AGENT.md) |
| See what's new | [`CHANGELOG.md`](CHANGELOG.md) |
| Defend my thesis | [`docs/00_README.md`](docs/00_README.md) → `01_RESEARCH_PAPER.md` |
| Ship MATHIR in production | [`mathir_dropin/README.md`](mathir_dropin/README.md) |
| Run a benchmark | [`benchmarks/`](benchmarks/) |
| Run a test | `pytest tests/ -q` |
| Read results JSONs | [`results/`](results/) |
| See pretty diagrams | [`docs/visualizations/visual_report.html`](docs/visualizations/visual_report.html) |
| Look at legacy V1–V3 code | [`legacy_v1_v3/`](legacy_v1_v3/) |

---

## 🏗️ Top-Level Project Structure

```
D:/SECRET_PROJECT/MATHIR/
│
├── README.md                       (root - main entry point)
├── CHANGELOG.md                    (root - what changed per version)
├── AGENT.md                        (root - API / class specification)
├── FUTURE_VISION.md                (root - strategic roadmap)
├── IMPLEMENTATION.md               (root - build plan)
├── MASTER_INDEX.md                 (root - this file)
├── LICENSE
├── requirements.txt
├── setup.py
│
├── mathir_lib/                     (V6/V7 research code, 15 memory modules)
│   ├── __init__.py
│   ├── plugin.py                   (V6 plugin)
│   ├── plugin_v7.py                (V7 plugin, 8 novel algorithms)
│   ├── memory/                     (15 memory modules)
│   ├── providers/                  (5 embedding providers)
│   ├── compression.py              (TurboQuant)
│   ├── config.py
│   ├── router.py                   (KL-constrained router)
│   ├── mhc.py                      (V6 shim for V4 mHC)
│   └── legacy/                     (V1-V5 archived)
│
├── mathir_dropin/                  (production drop-in package)
│   ├── __init__.py
│   ├── memory.py                   (MATHIRMemory)
│   ├── store.py                    (SQLiteStore + FTS5)
│   ├── config.py
│   ├── exceptions.py
│   ├── _demo.py                    (11-step end-to-end demo)
│   ├── README.md
│   └── tests/                      (10 critical tests for the drop-in)
│
├── tests/                          (ALL research tests consolidated)
│   ├── __init__.py                 (NEW in V7.4)
│   ├── conftest.py
│   ├── test_v7_memory.py           (49 unit tests)
│   ├── test_v7_integration.py      (16 integration tests)
│   ├── test_hybrid.py              (V7.1 Approach D, 28 tests)
│   ├── test_hybrid_cache.py        (V7.2 cache, 62 tests)
│   ├── test_hybrid_adaptive.py     (V7.2 adaptive rerank, 34 tests)
│   ├── test_raw_embedding.py       (V7.1 Approach A, 28 tests)
│   ├── test_ensemble.py            (V7.1 Approach B)
│   ├── test_faiss_memory.py        (V7.1 Approach C)
│   └── stress_test.py              (V6 deep stress, 13 tests)
│
├── benchmarks/                     (ALL benchmarks consolidated)
│   ├── __init__.py                 (NEW in V7.4)
│   ├── compare_all_approaches.py
│   ├── approach_d_vs_faiss.py
│   ├── book_stress_test.py
│   ├── book_stress_test_real_emb.py
│   ├── real_stress_test.py
│   ├── stress_cache_warm.py
│   ├── optimization_comparison.py
│   ├── v6_vs_v7.py
│   └── streamlit_app.py
│
├── examples/                       (V7 demos)
│   ├── multimodal_demo.py
│   ├── v7_advanced_demo.py
│   └── with_minimax.py             (basic usage with any LLM API)
│
├── docs/                           (CONSOLIDATED documentation)
│   ├── 00_README.md                ← entry point with full TOC
│   ├── 01_RESEARCH_PAPER.md        ← doctoral paper (21K words)
│   ├── 02_REFERENCE.md             ← single-file reference (5.8K)
│   ├── 03_QA_GUIDE.md              ← 100+ defense Q&A
│   ├── 04_INTEGRATION_GUIDE.md     ← dev guide (5.2K)
│   ├── 05_SHIPPING_GUIDE.md        ← production deployment
│   ├── 06_MULTIMODAL_GUIDE.md      ← text/image/audio/video
│   ├── 07_USE_CASES.md             ← chat + driving vs VectorDB
│   ├── 08_THEORY_V7.md             ← 6 theorems
│   ├── 09_V7_PAPER.md              ← NeurIPS-style paper
│   ├── 10_PROOFS.md                ← theorem proofs
│   ├── 11_TUTORIAL.md              ← V7 tutorial
│   ├── 12_MIGRATION_GUIDE.md       ← V6→V7
│   ├── 13_BENCHMARK_RESULTS.md     ← retrieval research
│   ├── 14_WHY_SAME_RESULTS.md      ← mathematical proof
│   ├── 15_THEORY.md                ← early theory
│   ├── 16_DEPLOYMENT.md            ← legacy deployment
│   ├── 17_HOW_TRAINING_WORKS.md    ← legacy
│   ├── 18_OLLAMA_INTEGRATION.md    ← legacy
│   ├── 19_OLLAMA_SETUP.md          ← legacy
│   ├── 20_CUDA_SETUP.md            ← legacy
│   ├── 21_QUICK_START.md           ← legacy
│   ├── 21a_QUICKSTART.md           ← legacy (alt)
│   ├── 22_KV_CACHE_RESEARCH.md     ← research
│   ├── 23_RUST_ML_RESEARCH.md      ← research
│   ├── 25b_BENCHMARK_V6_VS_V7.md   ← V6 vs V7
│   ├── 25c_VS_RAG_COMPARISON.md    ← MATHIR vs RAG
│   ├── 26_MATHIR_JOURNAL.md        ← scientific journal
│   ├── 27_IMPROVEMENTS_V2.md       ← V2 changelog
│   ├── 28_IMPROVEMENTS_V3.md       ← V3 changelog
│   ├── 29_IMPROVEMENTS_V5.md       ← V5 changelog
│   ├── 30_JOURNAL_DE_BORD.md       ← French journal
│   ├── 31_PREUVES_MATH.tex         ← LaTeX proofs
│   ├── 32_MATHIR_VS_RAG.html       ← HTML report
│   ├── 33_MATHIR.md                ← early doc
│   ├── 34_GITHUB_ASSETS.md         ← GitHub copy
│   └── visualizations/             ← 8 PNGs + HTML + scripts
│       ├── README.md
│       ├── generate_diagrams.py
│       ├── build_report.py
│       ├── 01_architecture_main.png
│       ├── 02_4_memory_tiers.png
│       ├── 03_retrieval_comparison.png
│       ├── 04_latency_quality_tradeoff.png
│       ├── 05_multi_agent_stress.png
│       ├── 06_multimodal_fusion.png
│       ├── 07_theorem_network.png
│       ├── 08_version_timeline.png
│       └── visual_report.html      (1.9 MB, self-contained)
│
├── results/                        (ALL benchmark JSONs consolidated)
│   ├── README.md                   (explains each file)
│   ├── compare_all_approaches.json
│   ├── approach_d_vs_faiss.json
│   ├── book_stress_test.json
│   ├── book_stress_test_real_emb.json
│   ├── real_stress_test.json
│   ├── stress_cache_warm.json
│   ├── latency_optimization.json
│   ├── v6_vs_v7.json
│   ├── capacity_log.json
│   ├── capacity_log_V1_V2.json
│   ├── benchmark_results.json
│   └── mathir_best_params.json
│
├── legacy_v1_v3/                   (V1-V3 archived, kept untouched)
│
├── tools/                          (deploy_edge.py, etc.)
├── config/                         (YAML configs: default, edge, research, v7)
├── checkpoints_saved_V1_V2/        (legacy checkpoints)
│
└── reorganize_v74.ps1              (one-shot script to finalize V7.4 cleanup)
```

---

## 🧭 Navigation Map

| Use Case | Entry Point |
|----------|-------------|
| **Defending my thesis** | [`docs/00_README.md`](docs/00_README.md) → `docs/01_RESEARCH_PAPER.md` |
| **Reviewing research** | [`docs/09_V7_PAPER.md`](docs/09_V7_PAPER.md) + [`docs/08_THEORY_V7.md`](docs/08_THEORY_V7.md) |
| **Deploying to production** | [`mathir_dropin/README.md`](mathir_dropin/README.md) + [`docs/05_SHIPPING_GUIDE.md`](docs/05_SHIPPING_GUIDE.md) |
| **Integrating into my project** | [`docs/04_INTEGRATION_GUIDE.md`](docs/04_INTEGRATION_GUIDE.md) |
| **Multimodal (image/audio/video)** | [`docs/06_MULTIMODAL_GUIDE.md`](docs/06_MULTIMODAL_GUIDE.md) + [`examples/multimodal_demo.py`](examples/multimodal_demo.py) |
| **Running benchmarks** | [`benchmarks/`](benchmarks/) — outputs go to [`results/`](results/) |
| **Reading benchmark results** | [`results/README.md`](results/README.md) |
| **Understanding the 6 theorems** | [`docs/10_PROOFS.md`](docs/10_PROOFS.md) |
| **Visual diagrams** | [`docs/visualizations/visual_report.html`](docs/visualizations/visual_report.html) |
| **Project history** | [`CHANGELOG.md`](CHANGELOG.md) |
| **V6 → V7 migration** | [`docs/12_MIGRATION_GUIDE.md`](docs/12_MIGRATION_GUIDE.md) |
| **V1–V3 code** | [`legacy_v1_v3/`](legacy_v1_v3/) |

---

## ✅ Quick-Start Recipes

### 1. Run the full test suite

```bash
# Research tests
pytest tests/ -q

# Drop-in package tests
pytest mathir_dropin/tests/ -v
```

### 2. Try the drop-in package (5 minutes)

```bash
python mathir_dropin/_demo.py
```

### 3. Re-run all benchmarks

```bash
python benchmarks/compare_all_approaches.py
python benchmarks/approach_d_vs_faiss.py
python benchmarks/book_stress_test.py
python benchmarks/book_stress_test_real_emb.py
python benchmarks/real_stress_test.py
python benchmarks/stress_cache_warm.py
python benchmarks/optimization_comparison.py
python benchmarks/v6_vs_v7.py
```

Each script writes its result into [`results/`](results/).

### 4. Regenerate the visualizations

```bash
python docs/visualizations/generate_diagrams.py
python docs/visualizations/build_report.py
# Then open docs/visualizations/visual_report.html in a browser
```

### 5. Run a specific example

```bash
python examples/v7_advanced_demo.py
python examples/multimodal_demo.py
python examples/with_minimax.py
```

### 6. Finalize the V7.4 reorganization (one-time)

```powershell
# From the project root, in PowerShell:
powershell -ExecutionPolicy Bypass -File .\reorganize_v74.ps1
```

This will:
- Rename all `docs/*.md` files with numbered prefixes.
- Move all PNGs and the HTML report from the old `visualizations/` to
  `docs/visualizations/`.
- Delete the original `*_results.json` files at the project root.
- Delete the now-empty `visualizations/` directory.
- Clear `__pycache__` and `.pytest_cache`.

---

## 🛠️ Code Reference

| Component | Path |
|-----------|------|
| V6 plugin | [`mathir_lib/plugin.py`](mathir_lib/plugin.py) |
| V7 plugin (8 algorithms) | [`mathir_lib/plugin_v7.py`](mathir_lib/plugin_v7.py) |
| V7 memory modules | [`mathir_lib/memory/`](mathir_lib/memory/) |
| Embedding providers (5) | [`mathir_lib/providers/`](mathir_lib/providers/) |
| TurboQuant compression | [`mathir_lib/compression.py`](mathir_lib/compression.py) |
| KL-constrained router | [`mathir_lib/router.py`](mathir_lib/router.py) |
| Drop-in `MATHIRMemory` | [`mathir_dropin/memory.py`](mathir_dropin/memory.py) |
| Drop-in `SQLiteStore` (FTS5) | [`mathir_dropin/store.py`](mathir_dropin/store.py) |
| Legacy V1–V3 code | [`legacy_v1_v3/`](legacy_v1_v3/) |

---

## 📈 Status — V7.4 (2026-06-03)

| Area | Status |
|------|--------|
| V7.3 production drop-in | ✅ Complete (10 tests pass) |
| V7.2 LRU result cache + adaptive rerank | ✅ Complete (62+34 tests) |
| V7.1 retrieval research (4 approaches) | ✅ Complete (130+ tests) |
| V7 novel algorithms (8) + 6 theorems | ✅ Complete |
| V6 LLM-agnostic plugin | ✅ Stable |
| Documentation reorganization | ✅ V7.4 (this commit) |
| Tests consolidation | ✅ V7.4 |
| Results consolidation | ✅ V7.4 |
| Visualizations in `docs/` | ✅ V7.4 |

---

*Last updated: V7.4 (2026-06-03).*
