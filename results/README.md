# 📊 MATHIR — Results Index

**All benchmark/stress-test/training output JSONs live here.**

> ⚠️ **DEPRECATED — Old results moved.** This directory previously contained old benchmark results. Those have been moved to `_deprecated/results/`. The authoritative benchmark files are now in `benchmarks/06_results/current/`:
> - `benchmarks/06_results/current/multi_dataset_efficient_results.json` — Real BEIR SciFact + NFCorpus
> - `benchmarks/06_results/current/controlled_results.json` — Controlled 5-system comparison on SciFact
> - `benchmarks/06_results/current/MATHIR_FINAL_REPORT.html` — Full HTML report

---

## 📁 Files in this directory

This directory is intentionally kept for backward compatibility. All new benchmark results are written to `benchmarks/06_results/current/`. Historical results have been moved to `_deprecated/results/`.

---

## 🔗 How to read them

- All JSONs are flat (no nested metadata) and are safe to load with
  `json.load(open(...))` — see the schemas in each script that produces
  them.
- `compare_all_approaches.json` is the canonical v8.5 retrieval-quality
  report — start there.
- `book_stress_test_real_emb.json` is the most realistic real-embedding
  benchmark (uses MiniLM).
- `v6_vs_v7.json` is the version-comparison report for the V7 release
  notes.

---

## 🛠️ Re-generating the results

Each benchmark script in `benchmarks/` writes its own JSON to this
directory. To re-run them all:

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

Output paths are hard-coded in each script — see the `os.path.join(...)`
call near the end of each one.

---

*Last reorganized: v8.5 (2026-06-03). BEIR benchmarks added v8.5 (2026-06-05).*
