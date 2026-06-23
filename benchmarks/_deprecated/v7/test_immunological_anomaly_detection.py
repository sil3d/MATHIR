"""
Immunological Memory Anomaly Detection Benchmark
==================================================

Tests whether MATHIR's immunological memory can detect anomalies /
out-of-distribution (OOD) inputs using Mahalanobis distance.

Protocol:
  Phase 1: TRAIN — Feed 500 "normal" embeddings to build Mahalanobis threshold
  Phase 2: TEST  — Feed 100 normal + 50 anomalous samples
           — Compute AUC-ROC: how well does immunological score separate normal from anomaly?

Baseline: FAISS just returns nearest neighbors — NO novelty signal

Usage:
    python benchmarks/test_immunological_anomaly_detection.py
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from sklearn.metrics import roc_auc_score

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Import MATHIR components
from mathir_lib.memory.immunological import (
    ImmunologicalMemory,
    MahalanobisImmunologicalMemory,
)


# ============================================================================
# EMBEDDING MODEL (for realistic 384-dim embeddings)
# ============================================================================

class Embedder:
    """Lightweight embedder using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"  Loading: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
        print(f"  Embedding dim: {self.dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_data(
    n_normal_train: int = 500,
    n_normal_test: int = 100,
    n_anomaly: int = 50,
    embed_dim: int = 384,
    normal_std: float = 0.5,
    anomaly_offset: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic embeddings with clear in-distribution vs OOD samples.

    Normal: cluster around center with tight std
    Anomaly: embeddings far from center (offset by anomaly_offset stds)

    Returns:
        train_normal: [n_normal_train, embed_dim] — for training
        test_normal: [n_normal_test, embed_dim] — normal test samples
        test_anomaly: [n_anomaly, embed_dim] — anomalous samples
        labels: [n_normal_test + n_anomaly] — 0=normal, 1=anomaly
    """
    print(f"\n  Generating synthetic data:")
    print(f"    Normal train: {n_normal_train}, Normal test: {n_normal_test}, Anomaly: {n_anomaly}")
    print(f"    Embed dim: {embed_dim}, Normal std: {normal_std}, Anomaly offset: {anomaly_offset}")

    # Random center for normal distribution
    center = np.random.randn(embed_dim) * 0.1

    # Training normal samples
    train_normal = np.random.randn(n_normal_train, embed_dim) * normal_std + center

    # Test normal samples (same distribution as training)
    test_normal = np.random.randn(n_normal_test, embed_dim) * normal_std + center

    # Anomaly samples (far from center in random direction)
    direction = np.random.randn(embed_dim)
    direction = direction / np.linalg.norm(direction)  # unit vector
    anomaly_center = center + anomaly_offset * normal_std * direction

    test_anomaly = np.random.randn(n_anomaly, embed_dim) * normal_std + anomaly_center

    # Labels for test set
    labels = np.array([0] * n_normal_test + [1] * n_anomaly)

    print(f"    Train normal mean dist to center: {np.linalg.norm(train_normal.mean(axis=0) - center):.4f}")
    print(f"    Test anomaly mean dist to center: {np.linalg.norm(test_anomaly.mean(axis=0) - center):.4f}")

    return train_normal, test_normal, test_anomaly, labels


# ============================================================================
# REAL TEXT DATA (using sample texts)
# ============================================================================

# Normal texts: everyday sentences
NORMAL_TEXTS = [
    "The weather today is sunny with clear skies.",
    "A person is walking down the street carrying groceries.",
    "The meeting is scheduled for two o'clock in the afternoon.",
    "The book is on the table near the window.",
    "Children are playing in the park during recess.",
    "The restaurant serves delicious food at reasonable prices.",
    "Traffic is moving slowly due to road construction.",
    "The concert starts at eight and runs until ten.",
    "She loves reading mystery novels in her spare time.",
    "The train arrives at platform five on time.",
    "The cat is sleeping on the warm sunny windowsill.",
    "He ordered a large pizza with extra cheese.",
    "The library closes at nine PM on weekdays.",
    "She painted the walls a light blue color.",
    "The baby is crying because she is hungry.",
] * 34  # 510 samples for training + testing

# Anomaly texts: system/computer error messages (clearly OOD)
ANOMALY_TEXTS = [
    "CRITICAL SYSTEM FAILURE DETECTED IN CORE MODULE",
    "UNEXPECTED INTERRUPT OCCURRED AT MEMORY ADDRESS ZERO",
    "WARNING: Unauthorized access attempt logged",
    "ERROR: Stack overflow in recursive function call",
    "ALERT: Temperature spike exceeds safety threshold",
    "CRASH: Null pointer exception in production system",
    "FATAL: Database connection pool exhausted",
    "BREACH: Security protocol violation detected",
    "EMERGENCY: Shutdown signal received from main control",
    "DISASTER: File system corruption detected at sector 0",
    "SEVERE: Kernel panic - not syncing",
    "FATAL: Out of memory in heap allocation",
    "ERROR: Segmentation fault at address 0x00000000",
    "CRITICAL: CPU overheating at 120 degrees Celsius",
    "ALERT: Disk I/O error on primary storage device",
] * 4  # 60 samples


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark(
    use_real_text: bool = False,
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    anomaly_offset: float = 6.0,
    normal_std: float = 0.5,
) -> Dict[str, Any]:
    """
    Run the immunological anomaly detection benchmark.

    Args:
        use_real_text: If True, use real text embeddings (slower but realistic)
        embedder_model: Model name for embedding
        anomaly_offset: How many std devs away anomaly center is from normal center
        normal_std: Standard deviation of normal distribution

    Returns:
        results dictionary with AUC-ROC and other metrics
    """
    print("\n" + "=" * 70)
    print("IMMUNOLOGICAL MEMORY ANOMALY DETECTION BENCHMARK")
    print("=" * 70)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "use_real_text": use_real_text,
        "embedder_model": embedder_model,
        "seed": SEED,
        "anomaly_offset": anomaly_offset,
        "normal_std": normal_std,
    }

    # Initialize embedder
    embedder = Embedder(model_name=embedder_model)
    embed_dim = embedder.dim
    results["embedding_dim"] = embed_dim

    # Generate or load data
    if use_real_text:
        print("\n[Phase 0] Loading real text data...")

        # Use real text samples: 500 train + 100 test normal + 50 anomaly
        train_normal_texts = NORMAL_TEXTS[:500]
        test_normal_texts = NORMAL_TEXTS[500:600]
        test_anomaly_texts = ANOMALY_TEXTS[:50]

        print(f"  Normal texts: {len(train_normal_texts)} train + {len(test_normal_texts)} test")
        print(f"  Anomaly texts: {len(test_anomaly_texts)}")

        # Encode
        print("  Encoding train normal texts...")
        train_normal_embs = embedder.encode(train_normal_texts)
        print("  Encoding test normal texts...")
        test_normal_embs = embedder.encode(test_normal_texts)
        print("  Encoding anomaly texts...")
        test_anomaly_embs = embedder.encode(test_anomaly_texts)

        labels = np.array([0] * len(test_normal_texts) + [1] * len(test_anomaly_texts))
    else:
        print("\n[Phase 0] Generating synthetic embeddings...")
        train_normal_embs, test_normal_embs, test_anomaly_embs, labels = generate_synthetic_data(
            n_normal_train=500,
            n_normal_test=100,
            n_anomaly=50,
            embed_dim=embed_dim,
            normal_std=normal_std,
            anomaly_offset=anomaly_offset,
        )

    results["n_train_normal"] = len(train_normal_embs)
    results["n_test_normal"] = len(test_normal_embs)
    results["n_test_anomaly"] = len(test_anomaly_embs)

    # Convert to tensors
    train_normal_tensor = torch.from_numpy(train_normal_embs).float()
    test_normal_tensor = torch.from_numpy(test_normal_embs).float()
    test_anomaly_tensor = torch.from_numpy(test_anomaly_embs).float()

    # =========================================================================
    # Phase 1: TRAIN — Build immunological memory with normal samples
    # =========================================================================
    print("\n[Phase 1] Training immunological memory on normal samples...")

    # Initialize Mahalanobis immunological memory
    immune = MahalanobisImmunologicalMemory(
        capacity=500,
        feature_dim=embed_dim,
        threshold=2.0,
        ema_decay=0.01,
        regularization=1e-4,
    )

    # Store training normal samples INDIVIDUALLY for richer memory bank
    for i in range(len(train_normal_tensor)):
        immune.store(train_normal_tensor[i:i+1])

    stats = immune.get_stats()
    print(f"  Memory stored: {stats['count']} patterns")
    print(f"  Updates: {stats['n_updates']}")

    # =========================================================================
    # Phase 2: TEST — Compute anomaly scores
    # =========================================================================
    print("\n[Phase 2] Testing on normal + anomalous samples...")

    # Compute anomaly scores for all test samples
    all_test_tensor = torch.cat([test_normal_tensor, test_anomaly_tensor], dim=0)

    with torch.no_grad():
        anomaly_scores = immune.get_anomaly_score(all_test_tensor)
        anomaly_scores_np = anomaly_scores.cpu().numpy()

    # Separate scores
    normal_scores = anomaly_scores_np[:len(test_normal_tensor)]
    anomaly_scores_detected = anomaly_scores_np[len(test_normal_tensor):]

    print(f"  Normal scores — mean: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
    print(f"  Anomaly scores — mean: {anomaly_scores_detected.mean():.4f}, std: {anomaly_scores_detected.std():.4f}")
    print(f"  Separation (diff of means): {anomaly_scores_detected.mean() - normal_scores.mean():.4f}")

    # =========================================================================
    # Compute AUC-ROC
    # =========================================================================
    auc_roc = roc_auc_score(labels, anomaly_scores_np)
    print(f"\n  AUC-ROC: {auc_roc:.4f}")
    print(f"  (0.5 = random, 1.0 = perfect separation)")

    results["auc_roc"] = auc_roc
    results["normal_score_mean"] = float(normal_scores.mean())
    results["normal_score_std"] = float(normal_scores.std())
    results["anomaly_score_mean"] = float(anomaly_scores_detected.mean())
    results["anomaly_score_std"] = float(anomaly_scores_detected.std())
    results["score_separation"] = float(anomaly_scores_detected.mean() - normal_scores.mean())

    # =========================================================================
    # Compare: Basic Immunological Memory (Euclidean only)
    # =========================================================================
    print("\n[Phase 3] Baseline: Basic Immunological Memory (Euclidean)...")

    basic_immune = ImmunologicalMemory(
        capacity=500,
        feature_dim=embed_dim,
        threshold=2.0,
    )

    # Train - store individually like Mahalanobis version
    for i in range(len(train_normal_tensor)):
        basic_immune.store(train_normal_tensor[i:i+1])

    # Test
    with torch.no_grad():
        basic_scores = basic_immune.get_anomaly_score(all_test_tensor)
        basic_scores_np = basic_scores.cpu().numpy()

    normal_basic = basic_scores_np[:len(test_normal_tensor)]
    anomaly_basic = basic_scores_np[len(test_normal_tensor):]

    basic_auc = roc_auc_score(labels, basic_scores_np)
    print(f"  Basic AUC-ROC: {basic_auc:.4f}")

    results["basic_auc_roc"] = basic_auc
    results["basic_normal_score_mean"] = float(normal_basic.mean())
    results["basic_anomaly_score_mean"] = float(anomaly_basic.mean())

    # =========================================================================
    # FAISS Baseline (no novelty detection — just nearest neighbor distance)
    # =========================================================================
    print("\n[Phase 4] FAISS baseline (no novelty signal)...")

    import faiss

    # Build FAISS index with training normal samples
    # Use IndexFlatL2 for actual L2 distance (not cosine similarity)
    index = faiss.IndexFlatL2(embed_dim)
    index.add(train_normal_embs.astype("float32"))

    # Query test samples — FAISS returns L2 distance to nearest neighbor
    # This is NOT an anomaly score — it's just nearest neighbor distance
    test_all = np.vstack([test_normal_embs, test_anomaly_embs]).astype("float32")
    distances, _ = index.search(test_all, k=1)

    # FAISS returns cosine similarity (higher = more similar = more "normal")
    # For AUC, we invert: use -similarity as "anomaly score" (higher = more anomalous)
    faiss_anomaly_scores = -distances.flatten()

    faiss_auc = roc_auc_score(labels, faiss_anomaly_scores)
    print(f"  FAISS AUC-ROC: {faiss_auc:.4f}")
    print(f"  (Note: FAISS has NO novelty detection — it just returns nearest neighbor similarity)")

    results["faiss_auc_roc"] = faiss_auc

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mahalanobis Immunological Memory AUC-ROC: {auc_roc:.4f}")
    print(f"  Basic Immunological Memory AUC-ROC:       {basic_auc:.4f}")
    print(f"  FAISS (no novelty detection) AUC-ROC:     {faiss_auc:.4f}")
    print("=" * 70)

    # Interpretation
    if auc_roc >= 0.9:
        interpretation = "EXCELLENT — immunological memory strongly separates normal from anomaly"
    elif auc_roc >= 0.8:
        interpretation = "GOOD — immunological memory effectively detects anomalies"
    elif auc_roc >= 0.7:
        interpretation = "MODERATE — some anomaly detection capability"
    elif auc_roc >= 0.6:
        interpretation = "WEAK — marginal anomaly detection"
    else:
        interpretation = "POOR — near random, needs tuning"

    results["interpretation"] = interpretation
    print(f"\n  Interpretation: {interpretation}")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Immunological Memory Anomaly Detection Benchmark")
    parser.add_argument("--real-text", action="store_true", help="Use real text embeddings (slower)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model")
    parser.add_argument("--output", default="benchmarks/immunological_results.json",
                        help="Output JSON file")

    args = parser.parse_args()

    print("\nConfiguration:")
    print(f"  Real text: {args.real_text}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")

    # Run benchmark
    results = run_benchmark(
        use_real_text=args.real_text,
        embedder_model=args.model,
    )

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_path}")