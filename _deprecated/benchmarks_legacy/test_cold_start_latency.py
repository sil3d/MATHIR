"""Test cold-start latency."""
import time
import numpy as np
import torch
from mathir_lib.memory.immunological import EnsembleColdStartImmunologicalMemory

embed_dim = 384
ensemble = EnsembleColdStartImmunologicalMemory(
    capacity=100,
    feature_dim=embed_dim,
    threshold=2.0,
    cold_start_k=3,
    voting_threshold=2,
)

# Train with 9 samples (cold-start)
train = torch.randn(9, embed_dim)
for i in range(9):
    ensemble.store(train[i:i+1])

# Test latency
test = torch.randn(100, embed_dim)

# Warmup
_ = ensemble.get_anomaly_score(test[:10])

# Timed runs
runs = 10
start = time.perf_counter()
for _ in range(runs):
    scores = ensemble.get_anomaly_score(test)
elapsed = time.perf_counter() - start
per_sample_ms = (elapsed / runs / 100) * 1000

print(f"Latency: {per_sample_ms:.4f} ms per sample")
print(f"Requirement: <1ms per sample")
status = "PASS" if per_sample_ms < 1 else "FAIL"
print(f"Status: {status}")
