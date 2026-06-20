"""
MATHIR — Benchmark Suite
========================

Standalone, runnable benchmarks and the interactive Streamlit dashboard.
Each script writes its own JSON to ``results/`` (or accepts an
``--output`` flag).

Run them all:

    python benchmarks/compare_all_approaches.py
    python benchmarks/approach_d_vs_faiss.py
    python benchmarks/book_stress_test.py
    python benchmarks/book_stress_test_real_emb.py
    python benchmarks/real_stress_test.py
    python benchmarks/stress_cache_warm.py
    python benchmarks/optimization_comparison.py
    python benchmarks/v6_vs_v7.py

Streamlit dashboard:

    streamlit run benchmarks/streamlit_app.py
"""
