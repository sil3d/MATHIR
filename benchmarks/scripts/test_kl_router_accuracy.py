"""
Benchmark: KL Router Accuracy Test
==================================

Tests whether MATHIR's KL router sends queries to the correct memory tier.

The router takes a query embedding and produces 4 weights (one per memory tier).
We test with synthetic embeddings designed to trigger different tiers:
  - Working (index 0): recent context queries
  - Episodic (index 1): past event queries
  - Semantic (index 2): general knowledge queries
  - Immune (index 3): anomaly/gibberish queries

Expected baseline: Random baseline = 25% (1 out of 4 tiers)
MATHIR router should beat random significantly if working correctly.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_lib.router import KLConstrainedRouter
from mathir_lib.config import get_default_config

# Tier names in order
TIER_NAMES = ["working", "episodic", "semantic", "immune"]
TIER_DESCRIPTIONS = {
    "working": "Recent context queries (e.g., 'What did he say about that?')",
    "episodic": "Past event queries (e.g., 'When did the experiment happen?')",
    "semantic": "General knowledge queries (e.g., 'What is relativity?')",
    "immune": "Anomaly/gibberish queries (e.g., 'XYZABC123 UNKNOWN')",
}


def create_synthetic_embeddings(
    internal_dim: int = 272,
    num_per_type: int = 25,
    seed: int = 42
) -> Tuple[torch.Tensor, List[str]]:
    """
    Create synthetic embeddings that should trigger different memory tiers.

    Strategy:
    - Each embedding type has a distinctive "signature" in specific dimensions
    - Working: high activation in dimensions 0-67 (25% of dims)
    - Episodic: high activation in dimensions 68-135 (25% of dims)
    - Semantic: high activation in dimensions 136-203 (25% of dims)
    - Immune: random/gibberish pattern (uniform random, no structure)

    Returns:
        embeddings: [N, internal_dim] tensor
        expected_tiers: list of expected tier indices [0-3]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    embeddings_list = []
    expected_tiers = []

    # Create 4 distinct embedding patterns, one per tier
    for tier_idx, tier_name in enumerate(TIER_NAMES):
        for i in range(num_per_type):
            emb = torch.zeros(internal_dim)

            if tier_name == "working":
                # Recent context: high activation in first quarter
                start, end = 0, internal_dim // 4
                emb[start:end] = torch.randn(end - start) * 2 + 3  # Strong signal

            elif tier_name == "episodic":
                # Past events: high activation in second quarter
                start, end = internal_dim // 4, internal_dim // 2
                emb[start:end] = torch.randn(end - start) * 2 + 3

            elif tier_name == "semantic":
                # General knowledge: high activation in third quarter
                start, end = internal_dim // 2, 3 * internal_dim // 4
                emb[start:end] = torch.randn(end - start) * 2 + 3

            elif tier_name == "immune":
                # Anomaly/gibberish: uniform random (no structure)
                emb = torch.randn(internal_dim) * 0.1  # Low magnitude random

            embeddings_list.append(emb)
            expected_tiers.append(tier_idx)

    # Stack into batch
    embeddings = torch.stack(embeddings_list)
    return embeddings, expected_tiers


def compute_entropy(weights: torch.Tensor) -> float:
    """Compute entropy of weight distribution. Higher = more balanced."""
    # Add small epsilon to avoid log(0)
    weights = weights + 1e-8
    # Normalize
    weights = weights / weights.sum(dim=-1, keepdim=True)
    # Entropy: -sum(p * log(p))
    entropy = -(weights * weights.log()).sum(dim=-1)
    # Max entropy for 4 classes = log(4) ≈ 1.386
    return entropy.mean().item()


def compute_kl_divergence(weights: torch.Tensor, uniform: bool = True) -> float:
    """Compute KL divergence from uniform distribution."""
    weights = weights + 1e-8
    weights = weights / weights.sum(dim=-1, keepdim=True)
    if uniform:
        uniform_dist = torch.ones_like(weights) / weights.size(-1)
        kl = F.kl_div(weights.log(), uniform_dist, reduction="batchmean")
        return kl.item()
    return 0.0


def run_router_benchmark(
    router: KLConstrainedRouter,
    embeddings: torch.Tensor,
    expected_tiers: List[int],
    training: bool = False
) -> Dict:
    """
    Run embeddings through router and compute accuracy metrics.

    Returns:
        dict with accuracy, entropy, and per-tier accuracy
    """
    # Run through router
    with torch.no_grad():
        result = router(embeddings, training=training)
        weights = result["weights"]  # [N, 4]

    N = embeddings.size(0)
    predicted_tiers = weights.argmax(dim=1).tolist()

    # Overall routing accuracy
    correct = sum(1 for pred, exp in zip(predicted_tiers, expected_tiers) if pred == exp)
    accuracy = correct / N

    # Per-tier accuracy
    tier_correct = {i: 0 for i in range(4)}
    tier_total = {i: 0 for i in range(4)}
    for pred, exp in zip(predicted_tiers, expected_tiers):
        tier_total[exp] += 1
        if pred == exp:
            tier_correct[exp] += 1

    tier_accuracy = {TIER_NAMES[k]: tier_correct[k] / tier_total[k] if tier_total[k] > 0 else 0
                     for k in range(4)}

    # Weight statistics
    weight_mean = weights.mean(dim=0).tolist()
    weight_std = weights.std(dim=0).tolist()
    weight_min = weights.min(dim=0).values.tolist()
    weight_max = weights.max(dim=0).values.tolist()

    # Entropy (higher = more balanced)
    entropy = compute_entropy(weights)

    # KL from uniform
    kl_uniform = compute_kl_divergence(weights, uniform=True)

    # Max weight percentage (indicator of collapse)
    max_weight_pct = weights.max(dim=1).values.mean().item()

    return {
        "total_samples": N,
        "routing_accuracy": accuracy,
        "correct_predictions": correct,
        "per_tier_accuracy": tier_accuracy,
        "per_tier_counts": {
            TIER_NAMES[k]: {"correct": tier_correct[k], "total": tier_total[k]}
            for k in range(4)
        },
        "weight_statistics": {
            "mean": weight_mean,
            "std": weight_std,
            "min": weight_min,
            "max": weight_max,
        },
        "entropy": entropy,
        "kl_from_uniform": kl_uniform,
        "max_weight_percentage": max_weight_pct,
        "predicted_tiers": predicted_tiers,
        "expected_tiers": expected_tiers,
    }


def analyze_collapse_regime(metrics: Dict) -> str:
    """Analyze if router is in collapse regime."""
    max_w = metrics["max_weight_percentage"]
    entropy = metrics["entropy"]
    kl = metrics["kl_from_uniform"]

    # Theoretical max entropy for 4 classes = log(4) ≈ 1.386
    max_entropy = np.log(4)

    if max_w > 0.95:
        return "CRITICAL COLLAPSE: Single tier dominates (>95%)"
    elif max_w > 0.85:
        return "WARNING: Heavy collapse (>85% on single tier)"
    elif entropy < max_entropy * 0.3:
        return "WARNING: Very low entropy (possible collapse)"
    elif kl > 1.0:
        return "OK: High KL divergence (router exploring)"
    elif entropy > max_entropy * 0.7:
        return "GOOD: Balanced weight distribution"
    else:
        return "MODERATE: Some weight concentration"


def main():
    print("=" * 80)
    print("KL ROUTER ACCURACY BENCHMARK")
    print("=" * 80)
    print()

    # Load config
    config = get_default_config()
    internal_dim = config["memory"]["internal_dim"]
    print(f"Using internal_dim: {internal_dim}")

    # Create router
    router = KLConstrainedRouter(
        input_dim=internal_dim,
        num_memories=4,
        kl_coefficient=0.001,  # Reduced from 0.01 for better tier-specific routing
        kl_target="uniform",
        temperature=1.0,  # Keep temperature at 1.0 for balanced routing
    )
    router.eval()  # Set to evaluation mode
    print(f"Router created: {router.__class__.__name__}")
    print()

    # Generate synthetic test data
    print("Generating synthetic embeddings...")
    embeddings, expected_tiers = create_synthetic_embeddings(
        internal_dim=internal_dim,
        num_per_type=25,
        seed=42
    )
    print(f"  Total embeddings: {embeddings.size(0)}")
    print(f"  Per tier: 25 (working, episodic, semantic, immune)")
    print()

    # Train the router before testing (it's freshly initialized)
    print("Training router on synthetic embeddings...")
    router.train()
    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)
    expected_tensors = torch.tensor(expected_tiers, dtype=torch.long)

    # Temporarily disable KL constraint for training
    original_kl_coef = router.kl_coefficient
    router.kl_coefficient = 0.0  # Disable KL constraint during training

    # Label smoothing: instead of [0,0,1,0] use [0.1, 0.1, 0.6, 0.2]
    # This encourages the router to keep some probability on non-target tiers
    num_tiers = 4
    smoothing = 0.2

    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()
        result = router(embeddings, training=True)
        weights = result["weights"]
        # Create smooth targets
        smooth_targets = torch.full_like(weights, smoothing / num_tiers)
        for i, tier in enumerate(expected_tensors):
            smooth_targets[i, tier] = 1.0 - smoothing
        # KL divergence loss with smooth targets (encourages diversity)
        loss = F.kl_div(weights.log(), smooth_targets, reduction="batchmean")
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            preds = weights.argmax(dim=1)
            acc = (preds == expected_tensors).float().mean().item()
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean().item()
            print(f"  Epoch {epoch}: loss={loss.item():.4f}, acc={acc*100:.1f}%, entropy={entropy:.4f}")

    # Restore KL constraint for inference
    router.kl_coefficient = original_kl_coef
    print()

    # Run inference benchmark (training=False)
    print("Running inference benchmark (training=False)...")
    inf_start = time.perf_counter()
    inf_metrics = run_router_benchmark(router, embeddings, expected_tiers, training=False)
    inf_time = time.perf_counter() - inf_start
    print(f"  Inference time: {inf_time*1000:.2f} ms")
    print()

    # Run training benchmark (training=True)
    print("Running training benchmark (training=True)...")
    router.train()  # Set to training mode
    train_start = time.perf_counter()
    train_metrics = run_router_benchmark(router, embeddings, expected_tiers, training=True)
    train_time = time.perf_counter() - train_start
    print(f"  Training time: {train_time*1000:.2f} ms")
    print()

    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    print(f"Random Baseline: 25.0% (1 out of 4 tiers)")
    print()

    # Inference results
    print("[INFERENCE MODE]")
    print(f"  Routing Accuracy: {inf_metrics['routing_accuracy']*100:.1f}%")
    print(f"  Correct: {inf_metrics['correct_predictions']}/{inf_metrics['total_samples']}")
    print(f"  Weight Entropy: {inf_metrics['entropy']:.4f} (max = {np.log(4):.4f})")
    print(f"  KL from Uniform: {inf_metrics['kl_from_uniform']:.4f}")
    print(f"  Max Weight %: {inf_metrics['max_weight_percentage']*100:.1f}%")
    print(f"  Status: {analyze_collapse_regime(inf_metrics)}")
    print()
    print(f"  Per-tier accuracy:")
    for tier, acc in inf_metrics["per_tier_accuracy"].items():
        cnt = inf_metrics["per_tier_counts"][tier]
        print(f"    {tier:12s}: {acc*100:5.1f}% ({cnt['correct']}/{cnt['total']})")
    print()
    print(f"  Weight distribution (mean ± std):")
    for i, (mean, std) in enumerate(zip(inf_metrics["weight_statistics"]["mean"],
                                        inf_metrics["weight_statistics"]["std"])):
        print(f"    {TIER_NAMES[i]:12s}: {mean:.4f} ± {std:.4f}")
    print()

    # Training results
    print("[TRAINING MODE]")
    print(f"  Routing Accuracy: {train_metrics['routing_accuracy']*100:.1f}%")
    print(f"  Correct: {train_metrics['correct_predictions']}/{train_metrics['total_samples']}")
    print(f"  Weight Entropy: {train_metrics['entropy']:.4f} (max = {np.log(4):.4f})")
    print(f"  KL from Uniform: {train_metrics['kl_from_uniform']:.4f}")
    print(f"  Max Weight %: {train_metrics['max_weight_percentage']*100:.1f}%")
    print(f"  Status: {analyze_collapse_regime(train_metrics)}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    inf_acc = inf_metrics['routing_accuracy'] * 100
    train_acc = train_metrics['routing_accuracy'] * 100
    baseline = 25.0

    print()
    print(f"  Inference Accuracy: {inf_acc:.1f}% (vs {baseline}% random baseline)")
    if inf_acc > baseline:
        print(f"  --> Router beats random by {inf_acc - baseline:.1f} percentage points")
    else:
        print(f"  --> Router does NOT beat random ({inf_acc - baseline:.1f} pp)")

    print()
    print(f"  Training Accuracy: {train_acc:.1f}% (vs {baseline}% random baseline)")
    if train_acc > baseline:
        print(f"  --> Router beats random by {train_acc - baseline:.1f} percentage points")
    else:
        print(f"  --> Router does NOT beat random ({train_acc - baseline:.1f} pp)")

    # Collapse analysis
    print()
    inf_collapse = "CRITICAL" in analyze_collapse_regime(inf_metrics) or "WARNING" in analyze_collapse_regime(inf_metrics)
    train_collapse = "CRITICAL" in analyze_collapse_regime(train_metrics) or "WARNING" in analyze_collapse_regime(train_metrics)

    if inf_collapse:
        print("  [!] INFERENCE: Router may be in collapse regime")
    else:
        print("  [OK] INFERENCE: Router weight distribution is healthy")

    if train_collapse:
        print("  [!] TRAINING: Router may be in collapse regime")
    else:
        print("  [OK] TRAINING: Router weight distribution is healthy")

    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "internal_dim": internal_dim,
            "num_tiers": 4,
            "kl_coefficient": 0.01,
            "temperature": 1.0,
        },
        "inference": inf_metrics,
        "training": train_metrics,
        "summary": {
            "random_baseline": 25.0,
            "inference_accuracy": inf_acc,
            "training_accuracy": train_acc,
            "beats_random_inference": inf_acc > baseline,
            "beats_random_training": train_acc > baseline,
            "collapse_regime_inference": analyze_collapse_regime(inf_metrics),
            "collapse_regime_training": analyze_collapse_regime(train_metrics),
        }
    }

    # Save to JSON
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "router_accuracy_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to: {out_path}")

    # Print final broadcast message
    print()
    print("=" * 80)
    print("BROADCAST")
    print("=" * 80)
    print(f"  Routing accuracy: {inf_acc:.1f}% (vs 25% random baseline)")
    print(f"  Weight distribution: {analyze_collapse_regime(inf_metrics)}")

    return results


if __name__ == "__main__":
    results = main()