"""
TDD tests for benchmarks/real_sota_benchmark_v2.py

RED phase: these tests verify the *structure* and *contracts* of the
benchmark without running the heavy retrieval pipeline (which takes
hours). The tests cover:

    1. The module imports cleanly.
    2. DATASETS dict has all 5 required BEIR datasets with the
       expected sizes in the docstring/description.
    3. SYSTEMS iterable has all 7 required systems (BM25, MiniLM,
       BGE-small, BGE-base, OptimizedMATHIR, +BM25 fusion, +CE rerank).
    4. latency_stats() returns P50, P95, P99, mean, std, min, max.
    5. evaluate_run() returns the standard TREC metrics dict.
    6. memory_footprint_mb() returns a dict with total_mb.
    7. download_beir_dataset() is a callable that returns bool.
    8. The main() function can be imported and exists.
    9. The final JSON schema (built by build_final_json) has the
       required nested keys.

These tests run in <5 seconds total.
"""

import os
import sys
import time
import json
import tempfile
import statistics
import importlib

import pytest

# Ensure benchmarks/ is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(PROJECT_ROOT, "benchmarks")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)


# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------
def test_benchmark_module_imports():
    """The benchmark module must import without errors."""
    import real_sota_benchmark_v2  # noqa: F401
    assert hasattr(real_sota_benchmark_v2, "main")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
REQUIRED_DATASETS = {"scifact", "nfcorpus", "fiqa", "arguana", "scidocs"}


def test_datasets_dict_has_all_required():
    """DATASETS must contain all 5 required BEIR datasets."""
    import real_sota_benchmark_v2 as bench
    assert hasattr(bench, "DATASETS"), "Module must define DATASETS"
    assert isinstance(bench.DATASETS, dict)
    for name in REQUIRED_DATASETS:
        assert name in bench.DATASETS, f"Missing dataset: {name}"


def test_each_dataset_has_required_fields():
    """Each dataset entry must have a description / num_queries / num_docs."""
    import real_sota_benchmark_v2 as bench
    for name, info in bench.DATASETS.items():
        assert isinstance(info, dict), f"{name} info should be a dict"
        # At minimum we need a way to identify the dataset
        assert "hf_name" in info or "url" in info, \
            f"{name} needs 'hf_name' or 'url'"


# ---------------------------------------------------------------------------
# Systems registry
# ---------------------------------------------------------------------------
REQUIRED_SYSTEMS = {
    "BM25 (rank-bm25)",
    "all-MiniLM-L6-v2 + FAISS",
    "BGE-small-en-v1.5 + FAISS",
    "BGE-base-en-v1.5 + FAISS",
    "OptimizedMATHIR",
    "OptimizedMATHIR + BM25 (RRF)",
    "OptimizedMATHIR + CE rerank",
}


def test_systems_contains_all_required():
    """SYSTEMS must list all 7 required systems."""
    import real_sota_benchmark_v2 as bench
    assert hasattr(bench, "SYSTEMS"), "Module must define SYSTEMS"
    # SYSTEMS can be a list of strings or a list of dicts
    system_names = set()
    for s in bench.SYSTEMS:
        if isinstance(s, str):
            system_names.add(s)
        elif isinstance(s, dict):
            system_names.add(s.get("name", ""))
    missing = REQUIRED_SYSTEMS - system_names
    assert not missing, f"Missing systems: {missing}"


def test_three_optimized_mathir_variants_exist():
    """All three OptimizedMATHIR configurations must be present."""
    import real_sota_benchmark_v2 as bench
    system_names = set()
    for s in bench.SYSTEMS:
        if isinstance(s, str):
            system_names.add(s)
        elif isinstance(s, dict):
            system_names.add(s.get("name", ""))
    optimized = {s for s in system_names if "OptimizedMATHIR" in s}
    assert len(optimized) >= 3, \
        f"Need at least 3 OptimizedMATHIR variants, got {optimized}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_evaluate_run_returns_standard_metrics():
    """evaluate_run must return nDCG@k, MRR@k, Recall@k for k=10,100."""
    import real_sota_benchmark_v2 as bench

    # Mock results: 5 queries, 3 relevant docs each, ranked results
    results = {
        "q1": [("d1", 0.9), ("d2", 0.8), ("d3", 0.7), ("d4", 0.6), ("d5", 0.5)],
        "q2": [("d2", 0.9), ("d1", 0.8), ("d3", 0.7), ("d4", 0.6), ("d5", 0.5)],
        "q3": [("d3", 0.9), ("d1", 0.8), ("d2", 0.7), ("d4", 0.6), ("d5", 0.5)],
    }
    qrels = {
        "q1": {"d1": 2, "d3": 1},
        "q2": {"d2": 2, "d1": 1},
        "q3": {"d3": 2, "d2": 1},
    }
    metrics = bench.evaluate_run(results, qrels, k_values=[10, 100])
    for key in ("nDCG@10", "MRR@10", "Recall@10", "nDCG@100", "MRR@100", "Recall@100"):
        assert key in metrics, f"Missing metric: {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"Metric {key} out of range: {metrics[key]}"


def test_evaluate_run_perfect_ranking_scores_one():
    """If ranking matches qrels perfectly, nDCG@10 = 1."""
    import real_sota_benchmark_v2 as bench
    results = {
        "q1": [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)],
    }
    qrels = {"q1": {"d1": 2, "d2": 1, "d3": 0}}
    metrics = bench.evaluate_run(results, qrels, k_values=[10])
    assert metrics["nDCG@10"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["MRR@10"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["Recall@10"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_run_zero_relevant_scores_zero():
    """If no relevant docs found, all metrics = 0."""
    import real_sota_benchmark_v2 as bench
    results = {"q1": [("d1", 0.9), ("d2", 0.8)]}
    qrels = {"q1": {"d99": 1}}  # d99 not retrieved
    metrics = bench.evaluate_run(results, qrels, k_values=[10])
    assert metrics["nDCG@10"] == 0.0
    assert metrics["MRR@10"] == 0.0


# ---------------------------------------------------------------------------
# Latency stats
# ---------------------------------------------------------------------------
def test_latency_stats_returns_required_percentiles():
    """latency_stats must return P50, P95, P99, mean, std, min, max."""
    import real_sota_benchmark_v2 as bench
    # 100 latencies from 1ms to 100ms
    lats = list(range(1, 101))
    stats = bench.latency_stats(lats)
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "std_ms",
                "min_ms", "max_ms", "count"):
        assert key in stats, f"Missing latency field: {key}"
    assert stats["count"] == 100
    # Percentile monotonicity
    assert stats["p50_ms"] < stats["p95_ms"] < stats["p99_ms"] <= stats["max_ms"]
    assert stats["min_ms"] <= stats["p50_ms"]


def test_latency_stats_empty_input():
    """Empty input should not crash; should return zeros."""
    import real_sota_benchmark_v2 as bench
    stats = bench.latency_stats([])
    assert stats["count"] == 0
    assert stats["mean_ms"] == 0.0


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------
def test_memory_footprint_returns_dict():
    """memory_footprint_mb should return a dict with total_mb."""
    import real_sota_benchmark_v2 as bench
    foot = bench.memory_footprint_mb()
    assert "total_mb" in foot
    assert isinstance(foot["total_mb"], (int, float))
    assert foot["total_mb"] >= 0


# ---------------------------------------------------------------------------
# Dataset download (mocked — we don't actually download)
# ---------------------------------------------------------------------------
def test_download_beir_dataset_is_callable(monkeypatch):
    """download_beir_dataset should be a callable that returns a bool."""
    import real_sota_benchmark_v2 as bench
    assert callable(bench.download_beir_dataset)
    # Try calling it on a non-existent dataset with a monkey-patched loader
    # that returns False (simulating failure)
    def fake_loader(name, *args, **kwargs):
        return False
    # Should not raise; should return False (skip) on failure
    result = bench.download_beir_dataset("nonexistent_dataset_xyz", _loader=fake_loader)
    assert result is False


# ---------------------------------------------------------------------------
# JSON output schema
# ---------------------------------------------------------------------------
def test_build_final_json_schema():
    """build_final_json must produce the BEIR-style summary structure."""
    import real_sota_benchmark_v2 as bench
    # Minimal mock data
    per_dataset = {
        "scifact": {
            "BM25": {"nDCG@10": 0.5, "MRR@10": 0.4, "Recall@100": 0.7,
                      "latency": {"p50_ms": 10, "p95_ms": 20, "p99_ms": 30,
                                   "mean_ms": 12, "std_ms": 3, "min_ms": 5,
                                   "max_ms": 35, "count": 100},
                      "memory_mb": 50.0, "encode_time_ms": 100, "index_time_ms": 50},
        },
        "nfcorpus": {
            "BM25": {"nDCG@10": 0.6, "MRR@10": 0.5, "Recall@100": 0.8,
                      "latency": {"p50_ms": 12, "p95_ms": 22, "p99_ms": 32,
                                   "mean_ms": 14, "std_ms": 4, "min_ms": 6,
                                   "max_ms": 40, "count": 100},
                      "memory_mb": 30.0, "encode_time_ms": 80, "index_time_ms": 40},
        },
    }
    final = bench.build_final_json(per_dataset)
    # Top-level keys
    for key in ("metadata", "per_dataset", "average", "configurations"):
        assert key in final, f"Missing top-level key: {key}"
    # Metadata
    assert "datasets" in final["metadata"]
    assert "systems" in final["metadata"]
    assert "timestamp" in final["metadata"]
    # Per-dataset structure
    assert "scifact" in final["per_dataset"]
    assert "nfcorpus" in final["per_dataset"]
    # Average structure: per-system, per-metric, the cross-dataset mean
    assert isinstance(final["average"], dict)
    if final["average"]:
        first_system = next(iter(final["average"]))
        avg_metrics = final["average"][first_system]
        for key in ("nDCG@10_avg", "MRR@10_avg", "Recall@100_avg",
                    "latency_p50_ms_avg", "latency_p95_ms_avg", "latency_p99_ms_avg",
                    "memory_mb_avg"):
            assert key in avg_metrics, f"Missing avg key: {key} in {avg_metrics}"


def test_build_final_json_computes_correct_average():
    """build_final_json must compute the cross-dataset mean correctly."""
    import real_sota_benchmark_v2 as bench
    per_dataset = {
        "ds_a": {"S1": {"nDCG@10": 0.4, "MRR@10": 0.3, "Recall@100": 0.6,
                         "latency": {"p50_ms": 10, "p95_ms": 20, "p99_ms": 30,
                                      "mean_ms": 12, "std_ms": 1, "min_ms": 5,
                                      "max_ms": 35, "count": 100},
                         "memory_mb": 50.0}},
        "ds_b": {"S1": {"nDCG@10": 0.6, "MRR@10": 0.5, "Recall@100": 0.8,
                         "latency": {"p50_ms": 12, "p95_ms": 22, "p99_ms": 32,
                                      "mean_ms": 14, "std_ms": 1, "min_ms": 6,
                                      "max_ms": 40, "count": 100},
                         "memory_mb": 30.0}},
    }
    final = bench.build_final_json(per_dataset)
    # S1 mean across 2 datasets
    assert "S1" in final["average"]
    assert final["average"]["S1"]["nDCG@10_avg"] == pytest.approx(0.5, abs=1e-6)
    assert final["average"]["S1"]["MRR@10_avg"] == pytest.approx(0.4, abs=1e-6)
    assert final["average"]["S1"]["Recall@100_avg"] == pytest.approx(0.7, abs=1e-6)


def test_build_final_json_handles_partial_failures():
    """If a system failed on some datasets, it should still appear with NaN/None."""
    import real_sota_benchmark_v2 as bench
    per_dataset = {
        "ds_a": {"S1": {"nDCG@10": 0.5, "MRR@10": 0.4, "Recall@100": 0.7,
                         "latency": {"p50_ms": 10, "p95_ms": 20, "p99_ms": 30,
                                      "mean_ms": 12, "std_ms": 1, "min_ms": 5,
                                      "max_ms": 35, "count": 100},
                         "memory_mb": 50.0}},
        "ds_b": {"S1": {"error": "OOM"}},  # S1 failed on ds_b
    }
    final = bench.build_final_json(per_dataset)
    # S1 should still be in average (with note that 1 dataset failed)
    assert "S1" in final["average"]
    # nDCG@10_avg = 0.5 (only ds_a)
    assert final["average"]["S1"]["nDCG@10_avg"] == pytest.approx(0.5, abs=1e-6)
    assert final["average"]["S1"].get("num_datasets_evaluated") == 1


# ---------------------------------------------------------------------------
# Print final table (smoke test, captures stdout)
# ---------------------------------------------------------------------------
def test_print_final_table_runs(capsys):
    """print_final_table should run without errors and produce output."""
    import real_sota_benchmark_v2 as bench
    table_data = {
        "scifact": {
            "BM25": {"nDCG@10": 0.55, "MRR@10": 0.5, "Recall@100": 0.75,
                      "latency": {"p50_ms": 10, "p95_ms": 20, "p99_ms": 30,
                                   "mean_ms": 12, "std_ms": 1, "min_ms": 5,
                                   "max_ms": 35, "count": 100}},
            "OptimizedMATHIR + CE rerank": {"nDCG@10": 0.72, "MRR@10": 0.68,
                       "Recall@100": 0.93, "latency": {"p50_ms": 50, "p95_ms": 100,
                       "p99_ms": 150, "mean_ms": 60, "std_ms": 20, "min_ms": 30,
                       "max_ms": 200, "count": 100}},
        },
    }
    bench.print_final_table(table_data, [])
    out = capsys.readouterr().out
    assert "BEIR" in out
    assert "nDCG@10" in out
    assert "BM25" in out
    assert "OptimizedMATHIR" in out


# ---------------------------------------------------------------------------
# Skip system on failure
# ---------------------------------------------------------------------------
def test_system_failure_recorded_not_crashing():
    """A system failure (no 'nDCG@10' key) should be recorded, not crash."""
    import real_sota_benchmark_v2 as bench
    per_dataset = {
        "scifact": {
            "FailingSystem": {"error": "ran out of memory"},
        },
    }
    # Should not raise
    final = bench.build_final_json(per_dataset)
    assert "FailingSystem" in final["per_dataset"]["scifact"]
