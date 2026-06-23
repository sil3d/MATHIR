# `_deprecated/` — The Journey from v7 → v8.4.0

> *"laisse les deprecated pour qu'ils comprennent que je reviens de loin"*
> — user-mandated decision, 2026-06-23

This directory is **intentionally kept visible** (not gitignored) so that future contributors
and AI agents can SEE the project's evolution. The deprecated code is part of the narrative —
it documents what was tried, what was abandoned, and what replaced it.

## The Story in 7 Acts

| Version | Era | Where the code lives today | Status |
|---|---|---|---|
| **v1–v3** | Pre-history | `_deprecated/legacy_v1_v3/` | Ancient experiments (driving env, optimize_mathir, train_evolution). Kept as historical reference. |
| **v4–v5** | The dark ages | `_deprecated/legacy_v4_v5/` | `mhc.py`, `plugin_v6.py`. The "monolithic plugin" era before the v7 modular split. |
| **v6** | Pre-modular | merged into `legacy_v4_v5/plugin_v6.py` | Single-file plugin, ~800 LOC. The "everything in one place" anti-pattern. |
| **v7** | Modular explosion | `_deprecated/benchmarks_legacy/`, `benchmarks/_deprecated/v7/`, `docs/_deprecated/`, `examples/_deprecated/` | 4-tier memory (`working_memory / episodic / semantic / immunological`), 8 algorithms, 6 theorems. Beautiful in theory, unmaintainable in practice. |
| **v7.7.x** | ONNX providers | `_deprecated/benchmark_onnx.py`, README in `docs/_deprecated/` | Quantized Octen INT8 embedder. Still the recommended CPU path in v8.4.0. |
| **v7 → v8 transition** | The Great Simplification | `_deprecated/mcp_server.py`, `_deprecated/AGENT.md`, `_deprecated/MATHIR_FINAL_REPORT.md` | The single-file `mcp_server.py` was split into the `mathir_mcp/` package. |
| **v8.4.0** | Living Memory | The **canonical** code: `mathir_mcp/mathir_lib/` (15 modules, ~3700 LOC, 173 tests) | **4 tiers, 17 MCP tools, 4-phase lifecycle (promote/decay/consolidate/link graph).** What we ship today. |

## Directory Map

```
_deprecated/
├── AGENT.md                      ← v7 agent guide (referenced mathir_lib.plugin_v7)
├── MATHIR_FINAL_REPORT.md        ← v7.7 final state report
├── mcp_server.py                 ← the OLD single-file MCP server (pre-package split)
├── setup.py                      ← old setup.py (replaced by pyproject.toml)
├── benchmark_vec.py              ← v7 bench: pure VecMemory comparison
├── benchmark_gpu_vec.py          ← v7 bench: GPU vs CPU VecMemory
├── benchmark_onnx.py             ← v7.7 bench: ONNX vs sentence-transformers
│
├── archive/                      ← ~25 v6/v7 helper scripts (training, stress, OpenRouter probes)
│   ├── SWARM_STATE.md            ← historical swarm context files
│   ├── MATHIR_Results_Report.html ← v6 benchmark HTML reports
│   ├── train.bat, setup_cuda_*.bat ← Windows training/CUDA setup scripts
│   └── ... (~25 files)
│
├── legacy_root/                  ← v8.4.0 cleanup: 4 root-level duplicates moved here
│   ├── README.md                 ← explains why each file is here
│   ├── mathir_search.py          ← bit-identical to mathir_mcp/mathir_lib/mathir_search.py
│   ├── mathir_vec.py             ← stale v7 stub (7,199 B vs canonical 77,599 B)
│   ├── mathir_vec_optimized.py   ← unmaintained optimization, no importers
│   └── mathir_gpu_vec.py         ← unmaintained GPU path, no importers
│
├── legacy_v1_v3/                 ← ancient (2023-era): training, driving env, app_streamlit
├── legacy_v4_v5/                 ← v4-v5 era: mhc.py, plugin_v6.py
│
├── benchmarks_legacy/            ← old benchmarks that pre-date the v8.4.0 lifecycle bench
├── results_legacy/               ← JSON reports from old benchmark runs
└── scripts/                      ← one-off scripts (dataset_research, openrouter probes)
```

## What Survived (Not Deprecated)

These were considered but **kept** in v8.4.0:

| Component | Why kept |
|---|---|
| `mathir_dropin/` | Still the canonical memory backend for the MCP server. |
| `vision_testing/` | Standalone vision/audio testing UI. Not part of MATHIR core but still maintained. |
| `stress_test/` | Stress harness for the daemon. Still works. |
| `raspberry_jetson/` | Portable wrapper for Pi/Jetson. Version bumped 8.3.0 → 8.4.0 in this release. |
| `benchmarks/04_lifecycle_bench/` | The current benchmark suite — proves the +52% recall@5 result. |
| `examples/onnx_usage.py` | Migrated to v8.4.0 (uses `OctenEmbedder`). |
| `examples/simple_memory_demo.py` | Already on `mathir_dropin.simple`, no migration needed. |

## What Was Killed (Truly Gone)

| Removed | Why |
|---|---|
| `mathir_lib.providers` (v7) | Replaced by `mathir_lib.mathir_onnx_embedder.OctenEmbedder` (single class, no factory). |
| `mathir_lib.plugin_v7.MATHIRPluginV7` | Replaced by direct use of `VecMemory` + `mathir_daemon`. |
| `mathir_lib.router.KLConstrainedRouter` | Replaced by daemon-internal routing. |
| `mathir_lib.memory.raw_episodic` | Subsumed by `VecMemory.store(...)`. |
| `mathir_lib.memory.immunological` | Now an internal anomaly-detection slot in `memory_risks.py`, NOT a tier. |
| The 5-tier taxonomy (`... | immunological`) | Collapsed to **4 tiers**: `working_memory | episodic | semantic | procedural`. |

## How to Read This If You're New

1. **You are here**: v8.4.0 — `mathir_mcp/mathir_lib/` is the only canonical code.
2. **Read** `mathir_mcp/ONBOARDING.md` first (3-step install + 17 MCP tools).
3. **Then** `mathir_mcp/CHANGELOG.md` (what changed in v8.4.0).
4. **Only if curious** about the journey: skim `_deprecated/AGENT.md` (v7 worldview) and `_deprecated/MATHIR_FINAL_REPORT.md` (v7.7 sign-off).
5. **Never** import from `_deprecated/`. If you find yourself wanting to, the v8.4.0 equivalent is in `mathir_mcp/mathir_lib/`.

## If You Want To Restore Deprecated Code

```bash
# Move it back (preserves git history)
git mv _deprecated/legacy_root/mathir_vec.py mathir_vec.py
# Then run the broken tests, see why we moved it in the first place
python -m pytest mathir_mcp/dev/
```

`git log --follow <file>` traces the move history — you'll see the exact commit that
consolidated it, with the rationale in the commit message.