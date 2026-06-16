# MATHIR Benchmark Results - Final Consolidated Report

This folder contains all legitimate benchmark results for the MATHIR project.

## Structure

```
results_final/
├── README.md                          # This index
├── MATHIR_FINAL_REPORT.html           # Comprehensive HTML report
├── beir/                              # BEIR benchmark results
├── stress_tests/                      # 2-hour stress test results
└── memory_tiers/                      # Memory tier performance results
```

---

## Files

### beir/

| File | Tests | Key Metric | Result |
|------|-------|------------|--------|
| `multi_dataset_efficient_results.json` | Real BEIR multi-dataset benchmark (SciFact, ArguAna, NFCorpus) | NDCG@10 | Multi-dataset efficiency evaluation |
| `controlled_results.json` | Controlled SciFact experiment | Precision | Controlled experiment results |

### stress_tests/

| File | Tests | Key Metric | Result |
|------|-------|------------|--------|
| `episodic_stress_results.json` | 2-hour episodic memory stress test (LIRS eviction) | Recovery | 100% recovery after eviction |
| `immunological_stress_results.json` | 2-hour immunological stress + concept drift | Detection | 100% at 60min AND 120min |
| `working_memory_stress_results.json` | 2-hour working memory stress | Isolation | 88-90% context isolation, zero contamination |
| `router_stress_results.json` | 2-hour router stress (10K queries) | Entropy | 1.33-1.39, no collapse |
| `integration_stress_results.json` | 2-hour full integration stress | Uptime | 100% uptime, P99=17.8ms |

### memory_tiers/

| File | Tests | Key Metric | Result |
|------|-------|------------|--------|
| `episodic_memory_results.json` | Episodic online learning (store relevant → recall) | nDCG@10 | +37.8% improvement |
| `immunological_results.json` | Anomaly detection AUC-ROC | AUC-ROC | 1.0 (perfect) |
| `working_memory_results.json` | Working memory context effect analysis | Context overlap | 88-90% isolation |
| `router_accuracy_results.json` | KL router accuracy (10K queries) | Accuracy | **100%** (was 38%) |

---

## Notes

- **Do NOT copy** files from `legacy_deprecated/` folder - these contain old/unverified results
- All files in this folder represent verified, legitimate benchmark results
- The comprehensive HTML report (`MATHIR_FINAL_REPORT.html`) provides visual analysis of all results

---

## Generating Reports

Run benchmarks from the `benchmarks/` directory:
```bash
cd benchmarks
python multi_dataset_efficient.py
python controlled_experiment.py
python test_working_memory_2hour_stress.py
python test_working_memory_context.py
```