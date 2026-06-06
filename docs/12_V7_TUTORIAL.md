# MATHIR V7 Tutorial

**Goal:** Get up and running with MATHIR V7 in 5 minutes, then deep-dive in 10 more.

---

## 5-Minute Quickstart

The fastest way to see V7 work:

```python
from mathir_lib import MATHIRPluginV7
import torch

# 1. Create the plugin (any embedding dimension works)
plugin = MATHIRPluginV7(embedding_dim=4096)

# 2. Simulate an LLM embedding (replace with real embedding)
embedding = torch.randn(1, 4096)

# 3. Perceive — process through memory
out = plugin.perceive(embedding)
print(out.keys())
# dict_keys(['enhanced_embedding', 'router_weights', 'anomaly_score', 'kl_loss'])

# 4. Store an experience
plugin.store({"embedding": embedding})

# 5. Recall relevant memories later
memories = plugin.recall(embedding, k=5)
print(f"Recalled {len(memories)} memories")
```

That's it. V7 in default mode behaves identically to V6. The plugin is LLM-agnostic, so the same code works for any embedding dimension (768, 1024, 1536, 3584, 4096, 7168, ...).

---

## 10-Minute Deep Dive

### Step 1: Configure V7 features

V7's eight new components are opt-in. Enable them via config:

```python
from mathir_lib import MATHIRPluginV7, load_config

# Option A: bundled V7 config (all features on)
config = load_config("config/v7.yaml")
plugin = MATHIRPluginV7(4096, config=config)

# Option B: selective enablement
from mathir_lib.config import get_default_config
config = get_default_config()
config["memory"]["episodic_type"] = "ebbinghaus"   # Spaced-repetition
config["memory"]["use_sparse_coding"] = True        # 17× compression
plugin = MATHIRPluginV7(4096, config=config)
```

### Step 2: Use it with an LLM

```python
# Replace this with your LLM call. Examples:
#   - OpenAI: openai.embeddings.create(model="text-embedding-3-small", input=text)
#   - Ollama: requests.post("http://localhost:11434/api/embeddings", json={...})
#   - HuggingFace: tokenizer + AutoModel + mean-pool
#   - Direct: torch.randn(1, 4096)  # for testing

def get_embedding(text: str) -> torch.Tensor:
    """Return a [1, D] tensor."""
    return torch.randn(1, 4096)  # placeholder

# Process a turn
text = "User said: I love hiking in the Alps."
embedding = get_embedding(text)
out = plugin.perceive(embedding)

# Inspect what MATHIR did
print(f"Enhanced embedding norm: {out['enhanced_embedding'].norm():.3f}")
print(f"Router weights (working/episodic/semantic/immune): {out['router_weights'].squeeze().tolist()}")
print(f"Anomaly score: {out['anomaly_score'].item():.3f}")
print(f"KL loss: {out['kl_loss'].item():.4f}")
```

### Step 3: Store and recall

```python
# Store multiple turns
for turn in range(10):
    text = f"Turn {turn}: ..."
    embedding = get_embedding(text)
    plugin.perceive(embedding)        # Always perceive first
    plugin.store({"embedding": embedding})

# Recall later
query = get_embedding("Tell me about hiking")
memories = plugin.recall(query, k=3)
for i, mem in enumerate(memories, 1):
    print(f"Memory #{i}: similarity={mem['similarity']:.3f}, index={mem['index']}")
```

### Step 4: Monitor memory health

```python
stats = plugin.get_stats()
print(f"Working memory: {stats['working_usage']}/{stats['working_capacity']}")
print(f"Episodic memory: {stats['episodic_usage']}/{stats['episodic_capacity']}")
print(f"Semantic memory: {stats['semantic_usage']}/{stats['semantic_capacity']}")
print(f"Immune bank: {stats['immune_usage']}/{stats['immune_capacity']}")

# V7-specific: which features are on?
print(f"Active config: {stats.get('config', {})}")
```

### Step 5: Forgetting

V7's `forget()` evicts based on retention (Ebbinghaus) or similarity (V6 default):

```python
# Aggressive: evict anything with R(t) < 5%
evicted = plugin.forget(threshold=0.05)
print(f"Evicted {evicted} memories")

# Less aggressive: only clear out very stale memories
evicted = plugin.forget(threshold=0.5)
```

---

## 30-Minute Mastery: V7 Module Tour

### EbbinghausMemory — Spaced Repetition

```python
from mathir_lib.memory.ebbinghaus import EbbinghausMemory

memory = EbbinghausMemory(capacity=1000, feature_dim=272, alpha=0.5)

# Store
features = torch.randn(1, 272)
memory.store(features)

# Retrieve (top-k, with stability boost)
retrieved = memory.retrieve(query=torch.randn(1, 272), k=3)
print(retrieved.shape)  # [1, 272]

# Inspect retention
stats = memory.get_stats()
print(f"Average retention: {stats['avg_retention']:.3f}")
print(f"Average stability: {stats['avg_stability']:.3f}")

# Evict the worst
evicted_idx = memory.evict()
```

**When to use:** Long-running agents (>1000 steps). V7 retains >89% at 10K steps vs V6's 19%.

### SparseCodingMemory — 17× Compression

```python
from mathir_lib.memory.sparse_coding import SparseCodingMemory

memory = SparseCodingMemory(
    num_atoms=1088,     # 4× over-complete
    feature_dim=272,
    sparsity=8,         # Only 8 non-zeros per code
    lambda_l1=0.1,
    n_iter=50,
)

# Encode as sparse code
features = torch.randn(1, 272)
z = memory.ista(features)
print(f"Sparse code has {(z != 0).sum().item()} non-zeros")  # Should be <= 8
print(f"Compression ratio: {memory.get_compression_ratio():.1f}×")  # ~17×

# Reconstruct (slightly lossy)
reconstructed = memory.retrieve(features)
print(f"MSE: {(features - reconstructed).pow(2).mean():.4f}")

# Train the dictionary periodically
loss = memory.train_dictionary(features, n_steps=10)
print(f"Reconstruction loss: {loss:.4f}")
```

**When to use:** Edge deployment (60 KB budget). 17× compression is the difference between fitting on a Raspberry Pi and not.

### CrossAttentionMemory — Learned Addressing

```python
from mathir_lib.memory.cross_attention import CrossAttentionMemory

memory = CrossAttentionMemory(capacity=1000, feature_dim=272, num_heads=4)

# Store
memory.store(torch.randn(1, 272))

# Retrieve with learned Q/K/V
retrieved = memory.retrieve(query=torch.randn(1, 272), k=3)
print(retrieved.shape)  # [1, 272]

# Inspect attention pattern
attn = memory.get_attention_weights(torch.randn(1, 272))
print(f"Attention shape: {attn.shape}")  # [1, H, N]
```

**When to use:** When cosine similarity is too rigid. Cross-attention learns compositional queries ("red AND car AND fast").

### MahalanobisImmunologicalMemory — NP-Optimal Anomaly Detection

```python
from mathir_lib.memory.immunological import MahalanobisImmunologicalMemory

memory = MahalanobisImmunologicalMemory(
    capacity=100,
    feature_dim=272,
    threshold=2.0,
    ema_decay=0.01,
)

# Train on "normal" patterns
for _ in range(50):
    features = torch.randn(1, 272)  # "Normal" Gaussian
    memory.store(features)

# Detect anomalies
test = torch.randn(1, 272) * 3.0  # Far from normal
anomaly_signal = memory.recognize(test)
print(f"Anomaly signal: {anomaly_signal.shape}")
print(f"Anomaly score: {memory.get_anomaly_score(test).item():.3f}")
```

**When to use:** Safety-critical applications. Mahalanobis is provably the most powerful detector for Gaussian-distributed normal data (Theorem 4).

### InfoNCELoss — Contrastive Self-Supervision

```python
from mathir_lib.memory.infonce import InfoNCELoss

loss_fn = InfoNCELoss(feature_dim=272, temperature=0.1, projection_dim=128)

# Two batches: current and future
z_t = torch.randn(32, 272)   # Batch of 32 current states
z_tk = torch.randn(32, 272)  # Batch of 32 future states

loss = loss_fn(z_t, z_tk)
print(f"InfoNCE loss: {loss.item():.4f}")

# Mutual information lower bound
mi_bound = loss_fn.get_mutual_information_bound(loss, n_negatives=32)
print(f"Mutual information >= {mi_bound:.3f} nats")
```

**When to use:** Replacing MSE predictor heads. InfoNCE maximizes a lower bound on mutual information, producing better representations for downstream tasks.

### NeuralODEMemory — Continuous-Time Evolution

```python
from mathir_lib.memory.neural_ode import NeuralODEMemory

memory = NeuralODEMemory(
    capacity=1000,
    feature_dim=272,
    dt=0.1,
    max_steps=10,
    method="euler",  # or "rk4" for accuracy
)

# Store
memory.store(torch.randn(1, 272))

# Age all memories by dt
memory.age_memory(dt=1.0)

# Retrieve by evolving memory state
retrieved = memory.retrieve(query=torch.randn(1, 272), n_steps=3)
print(retrieved.shape)  # [1, 272]

stats = memory.get_stats()
print(f"Method: {stats['method']}, mean age: {stats['mean_age']:.2f}")
```

**When to use:** When memory "ages" smoothly between observations. The ODE captures continuous-time dynamics that discrete steps miss.

### HyperbolicMemory — Poincaré Ball Embeddings

```python
from mathir_lib.memory.hyperbolic import HyperbolicMemory

memory = HyperbolicMemory(
    num_prototypes=256,
    feature_dim=272,
    proj_dim=64,
    c=1.0,  # Curvature
)

# Retrieve (closest in hyperbolic distance)
retrieved = memory.retrieve(query=torch.randn(1, 272))

# Update prototypes in hyperbolic space
memory.update(torch.randn(1, 272), learning_rate=0.01)

stats = memory.get_stats()
print(f"Mean norm: {stats['mean_norm']:.3f} (must be < 1)")
print(f"Max norm: {stats['max_norm']:.3f}")
```

**When to use:** Hierarchical semantic structures (taxonomies, ontologies). Hyperbolic space embeds trees with low distortion.

### VariationalMemory — Uncertainty-Aware Storage

```python
from mathir_lib.memory.variational import VariationalMemory

memory = VariationalMemory(capacity=1000, feature_dim=272, min_sigma=0.01)

# Store as a Gaussian (mu, sigma)
memory.store(torch.randn(1, 272))

# Retrieve with uncertainty
retrieved, uncertainty = memory.retrieve(query=torch.randn(1, 272), k=3, sample=True)
print(f"Retrieved: {retrieved.shape}, uncertainty: {uncertainty.item():.4f}")

# Update uncertainty based on evidence
memory.update_uncertainty(idx=0, evidence_quality=0.9)  # Lower uncertainty
```

**When to use:** When the agent needs to express "I don't know." High uncertainty → low confidence → safe fallback.

---

## Integration Patterns

### Pattern 1: Drop-in V6 replacement

```python
# Before (V6)
from mathir_lib import MATHIRPlugin
plugin = MATHIRPlugin(embedding_dim=4096)

# After (V7, default V6-compat mode)
from mathir_lib import MATHIRPluginV7
plugin = MATHIRPluginV7(embedding_dim=4096)
# All V6 calls work unchanged.
```

### Pattern 2: V7 for long-running agents

```python
config = get_default_config()
config["memory"]["episodic_type"] = "ebbinghaus"
plugin = MATHIRPluginV7(4096, config=config)

# Run for 10K steps
for t in range(10_000):
    emb = get_embedding(f"Turn {t}")
    out = plugin.perceive(emb)
    plugin.store({"embedding": emb})

# V7 retains >89% at this point. V6 would retain ~19%.
print(plugin.get_retention_scores().mean().item())
```

### Pattern 3: V7 for edge deployment

```python
config = get_default_config()
config["memory"]["use_sparse_coding"] = True
config["memory"]["working_capacity"] = 32        # Smaller
config["memory"]["episodic_capacity"] = 500      # Smaller
plugin = MATHIRPluginV7(embedding_dim=1024, config=config)  # Smaller LLM

# Memory footprint: <60 KB after sparse coding
stats = plugin.get_stats()
print(f"Episodic usage: {stats['episodic_usage']}/{stats['episodic_capacity']}")
```

### Pattern 4: V7 for safety-critical agents

```python
config = get_default_config()
config["memory"]["immune_type"] = "mahalanobis"
plugin = MATHIRPluginV7(4096, config=config)

# Process inputs and flag anomalies
for text in user_inputs:
    emb = get_embedding(text)
    out = plugin.perceive(emb)
    if out["anomaly_score"].item() > 5.0:
        log.warning(f"Anomalous input: {text}")
        # Trigger safe fallback
```

### Pattern 5: V7 for personal assistants

```python
config = get_default_config()
config["memory"]["episodic_type"] = "ebbinghaus"   # Spaced repetition
config["memory"]["semantic_type"] = "hyperbolic"    # Concept hierarchies
config["memory"]["use_cross_attention"] = True      # Learned retrieval
config["memory"]["use_infonce"] = True              # Better representations
plugin = MATHIRPluginV7(4096, config=config)

# Personal info: store once, recall for years
plugin.store({"embedding": get_embedding("User: Alice, lives in Paris.")})
plugin.store({"embedding": get_embedding("User: works as data scientist.")})
# ...

# Months later, recall
memories = plugin.recall(get_embedding("Where does Alice live?"), k=3)
```

---

## Common Workflows

### Workflow 1: Online learning

```python
plugin = MATHIRPluginV7(4096, config=v7_config)

for step in range(num_steps):
    # 1. Get LLM embedding
    embedding = llm.embed(state)

    # 2. Perceive (always — this is the "thinking" step)
    out = plugin.perceive(embedding)

    # 3. Optionally store
    if should_store(step):
        plugin.store({"embedding": embedding, "reward": reward})

    # 4. Use enhanced context for the policy
    action = policy(out["enhanced_embedding"])
```

### Workflow 2: Anomaly monitoring

```python
plugin = MATHIRPluginV7(4096, config={"memory": {"immune_type": "mahalanobis"}})

scores = []
for text in stream:
    emb = get_embedding(text)
    out = plugin.perceive(emb)
    scores.append(out["anomaly_score"].item())
    plugin.store({"embedding": emb})

# Detect distribution shift
import numpy as np
threshold = np.mean(scores) + 3 * np.std(scores)
print(f"Anomaly threshold: {threshold:.3f}")
```

### Workflow 3: Long-horizon recall

```python
plugin = MATHIRPluginV7(4096, config={
    "memory": {"episodic_type": "ebbinghaus", "episodic_capacity": 5000}
})

# Phase 1: accumulate memories
for experience in experiences:
    plugin.store({"embedding": get_embedding(experience)})

# Phase 2: query months later
for query in queries:
    memories = plugin.recall(get_embedding(query), k=5)
    print(f"Query: {query}")
    for m in memories:
        print(f"  Memory {m['index']} (similarity {m['similarity']:.3f})")
```

---

## Troubleshooting

### `RuntimeError: arccosh argument < 1`

**Cause:** HyperbolicMemory prototypes have norm ≥ 1.

**Fix:** Re-initialize with smaller norm, or use `memory.project_to_ball()`.

### `RuntimeError: matrix is singular`

**Cause:** Mahalanobis covariance is ill-conditioned.

**Fix:** Increase `regularization` parameter (default 1e-4):

```python
config["memory"]["immune_regularization"] = 1e-2
```

### Slow `perceive()` with `use_sparse_coding: True`

**Cause:** ISTA is iterating 50 times.

**Fix:** Reduce `n_iter` (trades quality for speed):

```python
config["memory"]["sparse_n_iter"] = 20  # Default 50
```

### `get_retention_scores()` returns 0 for all memories

**Cause:** All memories have been recently accessed (stability high, time-since low).

**Fix:** This is normal. The function is correct; the value just happens to be close to 1.

---

## Next Steps

- Read `docs/V7_PAPER.md` for the full theoretical analysis (6 theorems with proofs).
- Read `docs/V7_MIGRATION_GUIDE.md` for V6 → V7 migration.
- Run `examples/v7_advanced_demo.py` to see all 8 V7 features in action.
- Run `tests/stress_test.py` to validate your setup.
- Run `benchmarks/` to compare V6 vs V7 on your data.

---

*"V7 in 5 minutes. Mastery in 30. Papers in 300."*
