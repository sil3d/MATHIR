# 📚 MATHIR — Documentation Hub

**Welcome to the MATHIR documentation.**

This is the entry point for everything in `docs/`. All files are numbered
for easy reading order. Start at the top, or jump straight to what you need
in the table below.

---

## 🗂️ Index — Pick What You Need

### 🎓 For the Doctoral Defense

| # | File | What's in it |
|---|------|--------------|
| **01** | [`01_RESEARCH_PAPER.md`](01_RESEARCH_PAPER.md) | The 21 000-word doctoral paper. **Read this first.** |
| **02** | [`02_REFERENCE.md`](02_REFERENCE.md) | The single-file complete reference (5 800 words). Quick refresh. |
| **03** | [`03_QA_GUIDE.md`](03_QA_GUIDE.md) | 100+ defense Q&A — every tough question answered. |
| **08** | [`08_THEORY_V7.md`](08_THEORY_V7.md) | The 6 formal theorems (capacity, retention, router, anomaly, sparse, mHC). |
| **09** | [`09_V7_PAPER.md`](09_V7_PAPER.md) | NeurIPS-style V7 paper (12 000+ words). |
| **10** | [`10_PROOFS.md`](10_PROOFS.md) | Step-by-step proof of each theorem. |
| **11** | [`11_TUTORIAL.md`](11_TUTORIAL.md) | Hands-on V7 tutorial (3 500 words). |

### 🚀 For Production Deployment

| # | File | What's in it |
|---|------|--------------|
| **04** | [`04_INTEGRATION_GUIDE.md`](04_INTEGRATION_GUIDE.md) | Developer guide — 7 provider recipes (5 200 words). |
| **05** | [`05_SHIPPING_GUIDE.md`](05_SHIPPING_GUIDE.md) | How to ship: 1 folder, 3 lines, SQLite. |
| **06** | [`06_MULTIMODAL_GUIDE.md`](06_MULTIMODAL_GUIDE.md) | Text, image, audio, video with CLIP, CLAP, ImageBind. |
| **12** | [`12_MIGRATION_GUIDE.md`](12_MIGRATION_GUIDE.md) | Migrate from V6 to V7. |

### 🧠 For the Research Community

| # | File | What's in it |
|---|------|--------------|
| **07** | [`07_USE_CASES.md`](07_USE_CASES.md) | Chat + driving vs VectorDB. |
| **13** | [`13_BENCHMARK_RESULTS.md`](13_BENCHMARK_RESULTS.md) | V7.1 retrieval-quality research results. |
| **14** | [`14_WHY_SAME_RESULTS.md`](14_WHY_SAME_RESULTS.md) | Mathematical proof why A=C=FAISS (JL lemma) and D>FAISS. |
| **15** | [`15_THEORY.md`](15_THEORY.md) | Early-stage theory (precursor to V7). |
| **25b** | [`25b_BENCHMARK_V6_VS_V7.md`](25b_BENCHMARK_V6_VS_V7.md) | V6 vs V7 head-to-head. |
| **25c** | [`25c_VS_RAG_COMPARISON.md`](25c_VS_RAG_COMPARISON.md) | MATHIR vs RAG. |
| **26** | [`26_MATHIR_JOURNAL.md`](26_MATHIR_JOURNAL.md) | Scientific journal. |
| **32** | [`32_MATHIR_VS_RAG.html`](32_MATHIR_VS_RAG.html) | HTML version of the RAG comparison. |

### 🛠️ For Operators / DevOps

| # | File | What's in it |
|---|------|--------------|
| **16** | [`16_DEPLOYMENT.md`](16_DEPLOYMENT.md) | Deployment guide (legacy). |
| **17** | [`17_HOW_TRAINING_WORKS.md`](17_HOW_TRAINING_WORKS.md) | Training pipeline walk-through. |
| **18** | [`18_OLLAMA_INTEGRATION.md`](18_OLLAMA_INTEGRATION.md) | Ollama integration (legacy). |
| **19** | [`19_OLLAMA_SETUP.md`](19_OLLAMA_SETUP.md) | Ollama setup (legacy). |
| **20** | [`20_CUDA_SETUP.md`](20_CUDA_SETUP.md) | CUDA / GPU setup. |
| **21** | [`21_QUICK_START.md`](21_QUICK_START.md) | Quick start (canonical). |
| **21a** | [`21a_QUICKSTART.md`](21a_QUICKSTART.md) | Quick start (alternate). |
| **22** | [`22_KV_CACHE_RESEARCH.md`](22_KV_CACHE_RESEARCH.md) | KV-cache research report. |
| **23** | [`23_RUST_ML_RESEARCH.md`](23_RUST_ML_RESEARCH.md) | Rust + ML research. |

### 🕰️ Historical / Changelogs

| # | File | What's in it |
|---|------|--------------|
| **27** | [`27_IMPROVEMENTS_V2.md`](27_IMPROVEMENTS_V2.md) | V2 changelog. |
| **28** | [`28_IMPROVEMENTS_V3.md`](28_IMPROVEMENTS_V3.md) | V3 changelog. |
| **29** | [`29_IMPROVEMENTS_V5.md`](29_IMPROVEMENTS_V5.md) | V5 changelog. |
| **30** | [`30_JOURNAL_DE_BORD.md`](30_JOURNAL_DE_BORD.md) | French "journal de bord". |
| **31** | [`31_PREUVES_MATH.tex`](31_PREUVES_MATH.tex) | LaTeX mathematical proofs. |
| **33** | [`33_MATHIR.md`](33_MATHIR.md) | Early architecture document. |
| **34** | [`34_GITHUB_ASSETS.md`](34_GITHUB_ASSETS.md) | GitHub repo assets & copy. |

### 🖼️ Visualizations (separate sub-folder)

- [`visualizations/`](visualizations/) — 8 PNG diagrams + `visual_report.html` + scripts.
- Start with [`visualizations/README.md`](visualizations/README.md).
- The self-contained HTML report is in
  [`visualizations/visual_report.html`](visualizations/visual_report.html).

---

## 📊 Benchmark Results

All benchmark JSONs now live in [`../results/`](../results/). See
[`../results/README.md`](../results/README.md) for the full index of what
each file contains and how to re-generate them.

---

## 🆕 What's New in V7.4

The project has been reorganized for clarity. The headline changes:

1. **Numbered doc prefixes** — every doc now starts with `NN_NAME.md` so
   alphabetical directory listing = reading order.
2. **`docs/00_README.md` (this file)** — single entry point with TOC.
3. **`docs/visualizations/`** — all 8 PNGs and the HTML report now live
   under `docs/`, not at the project root.
4. **`results/`** — all benchmark JSONs consolidated (renamed
   `*_results.json` → descriptive names, no suffix cruft).
5. **`MASTER_INDEX.md` at the project root** — one navigation document
   for the entire project.
6. **Tests consolidated in `tests/`** — with a fresh `__init__.py`.
7. **Benchmarks in `benchmarks/`** — with their own `__init__.py` and
   paths updated to write into `results/`.
8. **One-shot `reorganize_v74.ps1` script** at the project root — run it
   to do the binary file moves (PNGs, HTML) and the deletions.

See [`../CHANGELOG.md`](../CHANGELOG.md) for the full changelog entry.

---

## 🧭 How to Navigate From Here

- **Defending?** → 01, 02, 03, 08, 09, 10, 11
- **Shipping?** → 04, 05, 06, 12, then `mathir_dropin/README.md`
- **Researching?** → 07, 13, 14, 15, 24, 25b, 25c, 26
- **Operating?** → 16, 17, 18, 19, 20, 21
- **Curious about history?** → 27, 28, 29, 30, 31, 33, 34
- **Want pretty pictures?** → `visualizations/visual_report.html`

---

*Last reorganized: V7.4 (2026-06-03).*
