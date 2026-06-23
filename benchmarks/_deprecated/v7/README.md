# `benchmarks/_deprecated/v7/` — v7-era benchmarks removed in v8.4.0

Seven test files were importing v7 modules that no longer exist in v8.4.0. They were
moved here in v8.4.0 to keep the v8.4.0 benchmark suite (`benchmarks/04_lifecycle_bench/`)
100% v7-free while preserving history.

## Files moved

| File | v7 import that broke |
|---|---|
| `test_episodic_memory_2hour_stress.py` | `from mathir_lib.memory.raw_episodic import RawEmbeddingEpisodicMemory` |
| `test_episodic_memory_online_learning.py` | same as above |
| `test_immunological_2hour_stress.py` | `from mathir_lib.memory.immunological import MahalanobisImmunologicalMemory` |
| `test_immunological_anomaly_detection.py` | same as above |
| `test_kl_router_accuracy.py` | `from mathir_lib.router import KLConstrainedRouter` |
| `test_kl_router_2hour_stress.py` | same as above |
| `test_working_memory_context.py` | `from mathir_lib.plugin_v7 import MATHIRPluginV7` |

## v7 → v8.4.0 module mapping

| v7 module | v8.4.0 replacement |
|---|---|
| `mathir_lib.memory.raw_episodic` | `mathir_lib.mathir_vec.VecMemory` (single class handles all tiers) |
| `mathir_lib.memory.immunological` | **Removed.** Immunological is now an internal anomaly slot in `memory_risks.py`, NOT a tier. |
| `mathir_lib.router` | Subsumed by `mathir_lib.mathir_daemon` (handles all routing internally) |
| `mathir_lib.plugin_v7` | Replaced by direct use of `VecMemory` + daemon RPC |
| `mathir_lib.config` | Replaced by `MATHIR_DATA_DIR`, `MATHIR_CONFIG_DIR`, `MATHIR_EMBEDDING_MODEL` env vars |

## Files that stayed in `benchmarks/03_vector_search_benchmarks/`

These do NOT import v7 modules and remain in the active benchmark suite:

- `benchmark_beir.py` — HybridSearch against BEIR SciFact dataset (uses portable shim)
- `multi_dataset_efficient.py`
- `test_integration_2hour_stress.py`
- `test_working_memory_2hour_stress.py` (different from the v7 one)

## If you want to revive a v7 test

Don't. The architecture changed fundamentally:
- v7 had 4 memory classes (`raw_episodic`, `immunological`, etc.) + a separate router + plugin
- v8.4.0 has **one** `VecMemory` class that handles all tiers via the `block_type` parameter

To reproduce a v7-era test in v8.4.0, rewrite it using `VecMemory.store(..., block_type='working_memory')`
and the daemon RPC methods (`memory_recall`, `memory_promote`, etc.).

## Why kept here (not deleted)

User-mandated decision (2026-06-23): the v7 tests are part of the project's evolution
story. They can be safely deleted in a future release if the maintenance cost outweighs
the storytelling value.