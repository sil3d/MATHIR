# MATHIR Benchmarks

Performance benchmarks, stress tests, and accuracy evaluations for the MATHIR cognitive architecture.

## Directory Structure

```
benchmarks/
├── scripts/          # All active benchmark scripts
├── results/          # JSON output from benchmark runs
├── results_final/    # Consolidated final results (verified)
├── reports/          # HTML/MD visual reports
├── _deprecated/      # Old scripts (kept for reference, not maintained)
├── beir_data/        # BEIR dataset cache
├── controlled_emb_cache/  # Pre-computed embeddings
└── README.md         # This file
```

## Quick Start

```bash
# From the project root (D:\SECRET_PROJECT\MATHIR)
cd benchmarks/scripts

# Run a specific benchmark
python test_kl_router_accuracy.py
python test_immunological_anomaly_detection.py
python test_episodic_memory_2hour_stress.py

# Run from project root (scripts add parent dirs to sys.path)
python benchmarks/scripts/test_fts.py
```

## Active Benchmarks

### Vector Search (moved from root)

| Script | What it tests | Output |
|--------|--------------|--------|
| `benchmark_beir.py` | BEIR SciFact dataset — numpy vs USearch vs sqlite-vec | `results/` |
| `benchmark_unified.py` | All backends against FAISS baseline | `results/` |

### Memory Tier Tests

| Script | What it tests | Duration | Output |
|--------|--------------|----------|--------|
| `test_episodic_memory_2hour_stress.py` | Episodic memory under sustained load (5000 ops, 24 checkpoints) | ~5 min | `results/episodic_*.json` |
| `test_working_memory_2hour_stress.py` | Working memory capacity and eviction under load | ~5 min | `results/working_*.json` |
| `test_working_memory_context.py` | Working memory context window behavior | ~3 min | `results/context_*.json` |
| `test_immunological_2hour_stress.py` | Immunological memory under sustained anomaly injection | ~5 min | `results/immunological_*.json` |
| `test_immunological_anomaly_detection.py` | AUC-ROC for anomaly detection (Mahalanobis distance) | ~2 min | `results/immunological_results.json` |
| `test_episodic_memory_online_learning.py` | Episodic online learning behavior | ~3 min | `results/` |

### Router Tests

| Script | What it tests | Duration | Output |
|--------|--------------|----------|--------|
| `test_kl_router_accuracy.py` | KL router routing accuracy (target: >25% random baseline) | ~2 min | `results/router_accuracy_results.json` |
| `test_kl_router_2hour_stress.py` | KL router under sustained load | ~5 min | `results/router_stress_results.json` |
| `test_fifo_vs_lirs.py` | FIFO vs LIRS cache eviction comparison | ~2 min | `results/fifo_vs_lirs_results.json` |

### Cold Start & Warmup

| Script | What it tests | Duration | Output |
|--------|--------------|----------|--------|
| `test_cold_start_ensemble.py` | Ensemble cold-start behavior | ~2 min | stdout |
| `test_cold_start_latency.py` | Cold-start latency measurement | ~1 min | stdout |
| `test_warmup_transition.py` | Warmup phase transition behavior | ~1 min | stdout |

### Cross-Provider & External

| Script | What it tests | Duration | Output |
|--------|--------------|----------|--------|
| `cross_provider_self_correction_test.py` | Self-correction across OpenRouter providers | ~10 min | `results/cross_provider_*.json` |
| `openrouter_multiprovider_benchmark.py` | Multi-provider OpenRouter benchmark | ~10 min | `results/openrouter_*.json` |
| `openrouter_free_model_probe.py` | Probe free OpenRouter models | ~5 min | `results/openrouter_free_model_probe.json` |
| `ollama_one_by_one.py` | Sequential Ollama model testing | ~5 min | `results/ollama_one_by_one.json` |
| `ollama_test.py` | Ollama integration test | ~5 min | stdout |

### Head-to-Head Comparisons

| Script | What it tests | Duration | Output |
|--------|--------------|----------|--------|
| `mathir_vs_faiss_full.py` | MATHIR vs FAISS — all capabilities (664 lines) | ~15 min | `results/` |
| `multi_dataset_efficient.py` | Multi-dataset BEIR with GPU-accelerated CE reranking | ~10 min | `results/` |
| `test_integration_2hour_stress.py` | Full MATHIR stack integration stress test (829 lines) | ~2 hours | `results/` |

### Demos & Utilities

| Script | What it does |
|--------|-------------|
| `universal_recall_demo.py` | Demo universal recall across memory tiers |
| `latin_demo.py` | Demo Latin name matching |
| `latin_e2e.py` | End-to-end Latin name pipeline |
| `test_fts.py` | Test full-text search (FTS5) |
| `sync_report.py` | Sync final report from results_final/ to reports/ |

## Viewing Results

### JSON Results

Results are saved as JSON in `results/`. Each file contains:
- Test parameters and configuration
- Per-checkpoint metrics (latency, accuracy, memory usage)
- Summary statistics

### Final Consolidated Results

`results_final/` contains verified, consolidated results:
- `beir/` — BEIR benchmark results (SciFact, ArguAna, NFCorpus)
- `stress_tests/` — 2-hour stress test results (all tiers)
- `memory_tiers/` — Memory tier performance results
- `MATHIR_FINAL_REPORT.html` — Comprehensive HTML report

### Visual Reports

Open `reports/MATHIR_FINAL_REPORT.html` in a browser for:
- 4 performance charts (latency, accuracy, memory, comparison)
- 7 data tables with all raw numbers
- MATHIR vs FAISS side-by-side comparison

## Deprecated Benchmarks

The `_deprecated/` directory contains older benchmark scripts that are no longer maintained:
- `real_sota_benchmark.py` / `real_sota_benchmark_v2.py` — early SOTA comparisons
- `comprehensive_stress_test.py` — monolithic stress test
- `mathir_vs_rag.py` — early RAG comparison
- `streamlit_app.py` — old Streamlit dashboard
- `openrouter_test_run*.py` — one-off OpenRouter test iterations
- `openrouter_quick_probe.py` — quick probe (superseded)
- `openrouter_stress_test.py` — stress test (superseded by multiprovider_benchmark)

These are kept for historical reference. Do not run them without updating imports.

## Adding New Benchmarks

1. Create your script in `scripts/`
2. Use `os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "output.json")` for output paths
3. Add `sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))` for project imports
4. Save results as JSON in `results/`
5. Update this README

## Dependencies

Benchmarks require the MATHIR project dependencies:
- `torch` (PyTorch)
- `numpy`
- `scikit-learn` (for AUC-ROC in anomaly detection)
- `requests` (for OpenRouter/Ollama benchmarks)

Core MATHIR modules: `mathir_dropin`, `mathir_lib`
