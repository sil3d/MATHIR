"""
MATHIR — Research Test Suite
============================

Consolidated V6 / V7 / V7.1 / V7.2 / V7.3 unit- and integration-tests
for the research code under ``mathir_lib/``.

Run them all:

    pytest tests/ -q

Individual files:

    pytest tests/test_v7_memory.py        # 49 V7 unit tests
    pytest tests/test_v7_integration.py   # 16 V7 integration tests
    pytest tests/test_hybrid.py           # 28 V7.1 Approach D unit tests
    pytest tests/test_hybrid_cache.py     # 62 V7.2 cache tests
    pytest tests/test_hybrid_adaptive.py  # 34 V7.2 adaptive-rerank tests
    pytest tests/test_raw_embedding.py    # 28 V7.1 Approach A tests
    pytest tests/test_ensemble.py         # V7.1 Approach B tests
    pytest tests/test_faiss_memory.py     # V7.1 Approach C tests
    pytest tests/stress_test.py           # 13 V6 deep-stress tests (script)
"""
