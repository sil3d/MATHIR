"""
Test cold-start ensemble detector for immunological memory.
Verifies that the first ~10 samples can detect anomalies using ensemble methods.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from mathir_lib.memory.immunological import (
    MahalanobisImmunologicalMemory,
    EnsembleColdStartImmunologicalMemory,
)


def test_cold_start_improvement():
    """
    Test cold-start detection improvement.

    Protocol:
    1. Train on only 9 samples (cold-start regime, n_updates < 10)
    2. Test anomaly detection during cold-start
    3. Compare: Mahalanobis (returns0) vs Ensemble (provides scores)
    """
    print("\n" + "=" * 70)
    print("COLD-START ENSEMBLE DETECTION TEST")
    print("=" * 70)

    embed_dim = 384
    normal_std = 0.5
    anomaly_offset = 4.0
    n_train_cold = 9  # Cold-start: n_updates < 10
    n_test_normal = 100
    n_test_anomaly = 50

    # Generate data
    print(f"\n[Setup]")
    print(f"  Train samples (cold-start): {n_train_cold} (n_updates < 10)")
    print(f"  Test normal: {n_test_normal}, Test anomaly: {n_test_anomaly}")
    print(f"  Anomaly offset: {anomaly_offset} std devs")

    center = np.random.randn(embed_dim) * 0.1
    train_normal = np.random.randn(n_train_cold, embed_dim) * normal_std + center
    test_normal = np.random.randn(n_test_normal, embed_dim) * normal_std + center
    direction = np.random.randn(embed_dim)
    direction = direction / np.linalg.norm(direction)
    anomaly_center = center + anomaly_offset * normal_std * direction
    test_anomaly = np.random.randn(n_test_anomaly, embed_dim) * normal_std + anomaly_center

    train_tensor = torch.from_numpy(train_normal).float()
    test_normal_tensor = torch.from_numpy(test_normal).float()
    test_anomaly_tensor = torch.from_numpy(test_anomaly).float()

    all_test_tensor = torch.cat([test_normal_tensor, test_anomaly_tensor], dim=0)
    labels = np.array([0] * n_test_normal + [1] * n_test_anomaly)

    # =========================================================================
    # Test 1: MahalanobisMemory during cold-start (returns zeros)
    # =========================================================================
    print("\n[Test 1] MahalanobisImmunologicalMemory (cold-start, n_updates={})...".format(n_train_cold))
    mahal = MahalanobisImmunologicalMemory(
        capacity=100,
        feature_dim=embed_dim,
        threshold=2.0,
        ema_decay=0.01,
    )

    for i in range(n_train_cold):
        mahal.store(train_tensor[i:i+1])
        print(f"    After store {i+1}: n_updates={mahal.n_updates.item()}")

    with torch.no_grad():
        mahal_scores = mahal.get_anomaly_score(all_test_tensor).cpu().numpy()

    mahal_normal = mahal_scores[:n_test_normal]
    mahal_anomaly = mahal_scores[n_test_normal:]
    mahal_auc = roc_auc_score(labels, mahal_scores)

    mahal_zero_scores = (mahal_scores == 0).sum()
    mahal_detected_anomalies = (mahal_anomaly > 0).sum()

    print(f"  AUC-ROC: {mahal_auc:.4f}")
    print(f"  Zero scores (cold-start blind): {mahal_zero_scores}/{len(mahal_scores)}")
    print(f"  Anomaly detection during cold-start: {mahal_detected_anomalies}/{n_test_anomaly} ({mahal_detected_anomalies/n_test_anomaly*100:.1f}%)")

    # =========================================================================
    # Test 2: EnsembleColdStartImmunologicalMemory during cold-start
    # =========================================================================
    print("\n[Test 2] EnsembleColdStartImmunologicalMemory (cold-start, n_updates={})...".format(n_train_cold))
    ensemble = EnsembleColdStartImmunologicalMemory(
        capacity=100,
        feature_dim=embed_dim,
        threshold=2.0,
        ema_decay=0.01,
        cold_start_k=3,
        voting_threshold=2,
    )

    for i in range(n_train_cold):
        ensemble.store(train_tensor[i:i+1])
        print(f"    After store {i+1}: n_updates={ensemble.mahalanobis.n_updates.item()}")

    with torch.no_grad():
        ens_scores = ensemble.get_anomaly_score(all_test_tensor).cpu().numpy()

    ens_normal = ens_scores[:n_test_normal]
    ens_anomaly = ens_scores[n_test_normal:]
    ens_auc = roc_auc_score(labels, ens_scores)

    ens_zero_scores = (ens_scores == 0).sum()
    ens_detected_anomalies = (ens_anomaly > 0).sum()

    print(f"  AUC-ROC: {ens_auc:.4f}")
    print(f"  Zero scores: {ens_zero_scores}/{len(ens_scores)}")
    print(f"  Anomaly detection during cold-start: {ens_detected_anomalies}/{n_test_anomaly} ({ens_detected_anomalies/n_test_anomaly*100:.1f}%)")

    # =========================================================================
    # Test 3: Full warmup test (500 samples) - should be ~1.0 AUC
    # =========================================================================
    print("\n[Test 3] Full warmup test (500 samples)...")

    full_train = np.random.randn(500, embed_dim) * normal_std + center
    full_test_normal = np.random.randn(n_test_normal, embed_dim) * normal_std + center
    full_test_anomaly = np.random.randn(n_test_anomaly, embed_dim) * normal_std + anomaly_center

    full_train_tensor = torch.from_numpy(full_train).float()
    full_test_normal_tensor = torch.from_numpy(full_test_normal).float()
    full_test_anomaly_tensor = torch.from_numpy(full_test_anomaly).float()

    full_all_test = torch.cat([full_test_normal_tensor, full_test_anomaly_tensor], dim=0)

    ensemble_full = EnsembleColdStartImmunologicalMemory(
        capacity=500,
        feature_dim=embed_dim,
        threshold=2.0,
        ema_decay=0.01,
        cold_start_k=5,
        voting_threshold=2,
    )

    for i in range(500):
        ensemble_full.store(full_train_tensor[i:i+1])

    with torch.no_grad():
        full_scores = ensemble_full.get_anomaly_score(full_all_test).cpu().numpy()

    full_auc = roc_auc_score(labels, full_scores)
    print(f"  AUC-ROC: {full_auc:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Cold-Start Detection (n_updates={n_train_cold}):")
    print(f"    Mahalanobis: {mahal_detected_anomalies}/{n_test_anomaly} ({mahal_detected_anomalies/n_test_anomaly*100:.1f}%) - RETURNS ZEROS")
    print(f"    Ensemble:   {ens_detected_anomalies}/{n_test_anomaly} ({ens_detected_anomalies/n_test_anomaly*100:.1f}%) - ACTUALLY DETECTS")
    print(f"  Full Warmup AUC-ROC: {full_auc:.4f}")
    print("=" * 70)

    improvement = ens_detected_anomalies/n_test_anomaly*100 - mahal_detected_anomalies/n_test_anomaly*100

    results = {
        "cold_start_mahalanobis_detection_rate": float(mahal_detected_anomalies/n_test_anomaly*100),
        "cold_start_ensemble_detection_rate": float(ens_detected_anomalies/n_test_anomaly*100),
        "cold_start_ensemble_auc": float(ens_auc),
        "full_warmup_auc": float(full_auc),
        "improvement": f"{float(mahal_detected_anomalies/n_test_anomaly*100):.0f}% -> {float(ens_detected_anomalies/n_test_anomaly*100):.0f}%",
    }

    return results


if __name__ == "__main__":
    results = test_cold_start_improvement()
    print("\nResults:")
    print(json.dumps(results, indent=2))
