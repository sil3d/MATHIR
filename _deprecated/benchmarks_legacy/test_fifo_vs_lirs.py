"""
Test LIRS vs FIFO Eviction Policy Comparison
==========================================

Compares recovery rates between FIFO and LIRS eviction policies
to verify that LIRS improves recovery after eviction from ~88% to95%+.
"""

import json
import os
import sys
import traceback
from typing import Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mathir_lib.memory.raw_episodic import RawEmbeddingEpisodicMemory


def compute_recovery_rate(
    mem: RawEmbeddingEpisodicMemory,
    test_embeddings: list,
    tracked_indices: list,
    num_stores_before_ref: int,
) -> float:
    """
    Compute recovery rate after eviction.
    
    1. Store num_stores_before_ref embeddings (fill to capacity)
    2. Record references for tracked_indices (these items are "important")
    3. Store more embeddings (cause evictions)
    4. Search for tracked embeddings and check if they're found
5. Recovery rate = how many tracked items are still retrievable
    """
    mem.reset()
    
    # Store initial embeddings up to capacity
    for i in range(num_stores_before_ref):
        mem.store(test_embeddings[i % len(test_embeddings)])
    
    # Record references for tracked items BEFORE eviction (these are "important")
    # This is what LIRS uses to decide what to keep
    for idx in tracked_indices:
        if idx < mem.get_usage():
            query = test_embeddings[idx]
            mem.search(query, k=1)
    
    # Store more (cause evictions)
    num_extra = 500  # Store more than capacity to cause evictions
    for i in range(num_extra):
        mem.store(test_embeddings[(num_stores_before_ref + i) % len(test_embeddings)])
    
    # Check recovery - can we still find the tracked items?
    # We check by searching for each tracked embedding and seeing if
    # the top result has very high similarity (meaning the same embedding is still there)
    recovered = 0
    for idx in tracked_indices:
        query = test_embeddings[idx]
        _, sims = mem.search(query, k=1)
        if sims.numel() > 0 and sims[0].item() > 0.99:  # Very high similarity = same embedding
            recovered += 1
    
    return recovered / len(tracked_indices) if tracked_indices else 0.0


def run_fifo_vs_lirs_test() -> Dict:
    """Run FIFO vs LIRS comparison test."""
    print("=" * 80)
    print("FIFO vs LIRS EVICTION POLICY COMPARISON")
    print("=" * 80)
    
    # Create test embeddings (simulating document embeddings)
    np.random.seed(42)
    embedding_dim = 384
    num_embeddings = 2000
    capacity = 1000
    
    print(f"\nCreating {num_embeddings} test embeddings (dim={embedding_dim})...")
    test_embeddings = [
        torch.randn(embedding_dim)
        for _ in range(num_embeddings)
    ]
    
    # Indices of items we want to track (will be stored early)
    tracked_indices = list(range(50))  # First 50 items
    
    results = {}
    
    # Test FIFO
    print("\n[TEST 1] FIFO Eviction Policy")
    print("-" * 40)
    mem_fifo = RawEmbeddingEpisodicMemory(
        capacity=capacity,
        embedding_dim=embedding_dim,
        eviction_policy="FIFO",
    )
    
    recovery_fifo = compute_recovery_rate(
        mem_fifo, test_embeddings, tracked_indices, capacity
    )
    print(f"  FIFO Recovery Rate: {recovery_fifo:.4f} ({recovery_fifo * 100:.2f}%)")
    results["fifo"] = {
        "recovery_rate": recovery_fifo,
        "eviction_policy": "FIFO",
    }
    
    # Test LIRS
    print("\n[TEST 2] LIRS Eviction Policy")
    print("-" * 40)
    mem_lirs = RawEmbeddingEpisodicMemory(
        capacity=capacity,
        embedding_dim=embedding_dim,
        eviction_policy="LIRS",
    )
    
    recovery_lirs = compute_recovery_rate(
        mem_lirs, test_embeddings, tracked_indices, capacity
    )
    print(f"  LIRS Recovery Rate: {recovery_lirs:.4f} ({recovery_lirs * 100:.2f}%)")
    results["lirs"] = {
        "recovery_rate": recovery_lirs,
        "eviction_policy": "LIRS",
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"  FIFO Recovery Rate: {recovery_fifo:.4f} ({recovery_fifo * 100:.2f}%)")
    print(f"  LIRS Recovery Rate: {recovery_lirs:.4f} ({recovery_lirs * 100:.2f}%)")
    if recovery_fifo > 0:
        improvement = (recovery_lirs - recovery_fifo) / recovery_fifo
        print(f"  Improvement: {improvement:+.2%}")
    else:
        improvement = float('inf') if recovery_lirs > 0 else 0
        print(f"  Improvement: infinite" if recovery_lirs > 0 else "  Improvement: 0%")
    
    results["summary"] = {
        "fifo_recovery": recovery_fifo,
        "lirs_recovery": recovery_lirs,
        "improvement_pct": improvement * 100 if improvement != float('inf') else float('inf'),
        "target_met": recovery_lirs >= 0.95,
    }
    
    return results


def main():
    """Run the comparison test."""
    try:
        results = run_fifo_vs_lirs_test()
        
        # Save results
        results_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fifo_vs_lirs_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        return results
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
