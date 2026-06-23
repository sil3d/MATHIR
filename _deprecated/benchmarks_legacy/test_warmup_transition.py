"""Verify warmup transition."""
import numpy as np
import torch
from mathir_lib.memory.immunological import EnsembleColdStartImmunologicalMemory

embed_dim = 384
normal_std = 0.5
anomaly_offset = 4.0

# Generate test data once
center = np.random.randn(embed_dim) * 0.1
test_normal = np.random.randn(100, embed_dim) * normal_std + center
direction = np.random.randn(embed_dim)
direction = direction / np.linalg.norm(direction)
anomaly_center = center + anomaly_offset * normal_std * direction
test_anomaly = np.random.randn(50, embed_dim) * normal_std + anomaly_center

test_normal_tensor = torch.from_numpy(test_normal).float()
test_anomaly_tensor = torch.from_numpy(test_anomaly).float()
all_test = torch.cat([test_normal_tensor, test_anomaly_tensor], dim=0)

# Test at different n_updates
for n_train in [5, 9, 10, 50, 500]:
    np.random.seed(42)
    torch.manual_seed(42)

    ensemble = EnsembleColdStartImmunologicalMemory(
        capacity=500,
        feature_dim=embed_dim,
        threshold=2.0,
        cold_start_k=3,
        voting_threshold=2,
    )

    train = np.random.randn(n_train, embed_dim) * normal_std + center
    train_tensor = torch.from_numpy(train).float()

    for i in range(n_train):
        ensemble.store(train_tensor[i:i+1])

    n_updates = ensemble.mahalanobis.n_updates.item()
    using = "ensemble" if n_updates < 10 else "mahalanobis"

    with torch.no_grad():
        scores = ensemble.get_anomaly_score(all_test).cpu().numpy()

    normal_scores = scores[:100]
    anomaly_scores = scores[100:]
    normal_mean = normal_scores.mean()
    anomaly_mean = anomaly_scores.mean()
    separation = anomaly_mean - normal_mean

    print(f"n_train={n_train:3d}, n_updates={n_updates:3d}, using={using:10s}, "
          f"normal_mean={normal_mean:.2f}, anomaly_mean={anomaly_mean:.2f}, "
          f"separation={separation:.2f}")
