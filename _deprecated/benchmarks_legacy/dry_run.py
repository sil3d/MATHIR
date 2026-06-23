"""
Dry-run version of mathir_vs_rag.py that uses random tensors instead of API.
Tests the benchmark LOGIC without needing an API key.
"""

import os
import sys
import time
import json
import random
import statistics
from typing import List, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_lib import MATHIRPlugin


# Stub provider (random embeddings)
class StubProvider:
    def __init__(self, dim=1024):
        self._dim = dim

    def embed_text(self, text):
        # Use text hash to get reproducible embeddings
        h = hash(text) % 1000
        return torch.randn(1, self._dim) + h * 0.001

    def embed_batch(self, texts):
        return torch.stack([self.embed_text(t).squeeze(0) for t in texts])

    def embedding_dim(self):
        return self._dim


# Import the benchmark functions
from mathir_vs_rag import (
    SimpleRAG, SimpleVectorDB, build_scenarios,
    run_mathir_scenario, run_rag_scenario, run_vectordb_scenario, print_report
)


def main():
    print("DRY RUN — using random embeddings (no API needed)")
    print("=" * 60)

    provider = StubProvider(dim=1024)
    scenarios = build_scenarios()
    scenarios = scenarios[:1]  # Just first scenario for speed

    results = []
    for scenario in scenarios:
        print(f"\nScenario: {scenario.name}")
        print(f"  {len(scenario.memories)} memories, {len(scenario.queries)} queries")

        # MATHIR
        t0 = time.time()
        r = run_mathir_scenario(scenario, provider, 1024)
        print(f"  MATHIR:   acc={r.accuracy*100:.0f}%, {r.avg_query_time_ms:.1f}ms/query")
        results.append(r)

        # RAG
        t0 = time.time()
        r = run_rag_scenario(scenario, provider)
        print(f"  RAG:      acc={r.accuracy*100:.0f}%, {r.avg_query_time_ms:.1f}ms/query")
        results.append(r)

        # VectorDB
        t0 = time.time()
        r = run_vectordb_scenario(scenario, provider, 1024)
        print(f"  VectorDB: acc={r.accuracy*100:.0f}%, {r.avg_query_time_ms:.1f}ms/query")
        results.append(r)

    print_report(results)


if __name__ == "__main__":
    main()
