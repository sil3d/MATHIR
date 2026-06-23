# `examples/_deprecated/` — v7-era examples (kept for historical reference)

Three Python example files from the v7 era. They reference APIs that no longer exist in v8.4.0
(`mathir_lib.MATHIRPluginV7`, the 5-tier taxonomy, etc.) and **will not run** without
modification. Kept visible to document what was possible in v7.

## What's here

| File | Topic |
|---|---|
| `multimodal_demo.py` | v7 multimodal memory demo (image + text embeddings) |
| `v7_advanced_demo.py` | v7 advanced features demo (KL router, immunological tier) |
| `with_minimax.py` | v7 demo integrated with MiniMax LLM (predecessor of MiniMax-M3) |

## Current examples

For v8.4.0, use these instead:

- `examples/simple_memory_demo.py` — works out of the box (uses `mathir_dropin.simple`)
- `examples/onnx_usage.py` — migrated to v8.4.0 in this release (uses `OctenEmbedder`)

## Why kept here (not deleted)

User-mandated decision (2026-06-23): the v7 examples are part of the project's evolution
story. They can be safely deleted in a future release if the maintenance cost outweighs
the storytelling value.