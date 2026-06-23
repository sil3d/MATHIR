# How MATHIR Training Works

## Overview

MATHIR uses **Reinforcement Learning** with a **KL-constrained router** to learn memory allocation strategies.

---

## Training Algorithm

```python
from mathir_lib import MATHIRPluginV7
from mathir_lib.config import load_config

config = load_config("config/v7.yaml")
plugin = MATHIRPluginV7(embedding_dim=4096, config=config)

# Training loop
for step in range(1, 1_000_000):
    # 1. Observe environment
    obs = env.get_state()
    
    # 2. MATHIR perceives and remembers
    output = plugin.perceive(obs)
    action = policy(output["enhanced_embedding"])
    
    # 3. Execute action, get reward
    reward = env.step(action)
    
    # 4. Store experience
    plugin.store({
        "embedding": obs,
        "action": action,
        "outcome": reward,
    })
    
    # 5. Router learns allocation strategy
    # KL constraint prevents collapse to single tier
    router_loss = compute_router_loss(output["router_weights"])
    router_loss.backward()
```

---

## Key Components

### 1. KL-Constrained Router

The router decides which memory tier to use for each input. It uses PPO-style trust region optimization with a KL divergence constraint to prevent collapse.

```python
# Router output
output = plugin.perceive(embedding)
print(output["router_weights"])
# [0.4, 0.3, 0.2, 0.1] = [working, episodic, semantic, immune]
```

### 2. Online Learning

Unlike pre-trained models, MATHIR learns during inference:

- **Working memory**: Updates every step (circular buffer)
- **Episodic memory**: Updates on event (key-value store)
- **Semantic memory**: Updates every 100 steps (online k-means)
- **Immunological memory**: Updates on event (anomaly detector)

### 3. Anomaly Detection

Immunological memory uses Mahalanobis distance (Theorem 4: NP-optimal) to detect novel inputs.

```python
output = plugin.perceive(embedding)
if output["anomaly_score"] > threshold:
    print("Novel situation detected!")
```

---

## Benchmarks

### Retention Test

Measures how well MATHIR retains information over time:

| Steps | Retention |
|---|---|
| 100 | 0.99 |
| 500 | 0.95 |
| 1000 | 0.85 |
| 2000 | 0.78 |
| 5000 | 0.70 |

### BEIR SciFact

| System | nDCG@10 |
|---|:---:|
| FAISS dense-only | **0.7441** |
| BM25 only | 0.5438 |
| Hybrid RRF | 0.6602 |

---

## Running Training

```bash
# V7 demo (all 8 algorithms)
python examples/v7_advanced_demo.py

# Unit tests
pytest tests/test_v7_memory.py

# V6 vs V7 comparison
python benchmarks/v6_vs_v7.py
```

---

## References

- [V7 Paper](docs/10_V7_PAPER.md) — NeurIPS-style draft
- [Theory](docs/09_THEORY_V7.md) — Mathematical proofs
- [Migration Guide](docs/13_V7_MIGRATION_GUIDE.md) — V6 → V7
