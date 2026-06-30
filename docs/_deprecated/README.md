# `docs/_deprecated/` — v7-era documentation (kept for historical reference)

These 10 markdown/HTML files document the v7 MATHIR architecture. They are **not maintained**
and reference APIs that no longer exist (e.g. `mathir_lib.plugin_v7`, `mathir_lib.providers`,
the 5-tier taxonomy with `immunological`).

## What's here

| File | Topic |
|---|---|
| `09_THEORY_V7.md` | v7 theory (8 algorithms, 6 theorems) |
| `11_PROOFS.md` | Mathematical proofs for v7 algorithms |
| `12_V7_TUTORIAL.md` | v7-era hands-on tutorial |
| `13_V7_MIGRATION_GUIDE.md` | Guide for migrating code to v7 (now itself needs migration to v8.4.0) |
| `14_RETRIEVAL_RESEARCH_RESULTS.md` | v7 retrieval research |
| `15_BENCHMARK_V6_VS_V7.md` | v6 vs v7 benchmark comparison |
| `22_MATHIR_VS_RAG_COMPARISON.md` | MATHIR vs RAG (v7-era framing) |
| `28_HOW_TRAINING_WORKS.md` | v7 online learning / training pipeline |
| `38_MATHIR_VS_RAG_COMPARISON.html` | HTML version of the RAG comparison |
| `mathir_v71_failure_analysis.md` | Post-mortem of a v7.1 bug |

## Current docs

For v8.4.0, read these instead:

- `INSTALL_FOR_AGENT/AGENT.md` (updated in v8.4.0) — agent integration guide
- `docs/DAEMON.md` — daemon architecture
- `docs/DASHBOARD_GUIDE.md` (updated in v8.4.0) — dashboard usage
- `docs/DIMENSIONS.md` — embedding model dimensions
- `mathir_mcp/ONBOARDING.md` — 3-step onboarding
- `mathir_mcp/INSTALL_FOR_AGENTS.md` — integration for AI agent hosts
- `mathir_mcp/CHANGELOG.md` — what changed

## Why kept here (not deleted)

User-mandated decision (2026-06-23): the v7 docs are part of the project's evolution story.
They can be safely deleted in a future release if the maintenance cost outweighs the
storytelling value.