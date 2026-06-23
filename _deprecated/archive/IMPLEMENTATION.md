# MATHIR Implementation Plan

**How to get from here to there.**

---

## Current State (V7.3 — June 2026)

MATHIR is now at **V7.3 (Production-Ready Drop-In + Visualizations)**. V6 (LLM-agnostic plugin), V7 (8 novel algorithms, 6 theorems), V7.1 (4 new retrieval approaches A–D, +26pp quality, 130 new tests), V7.2 (LRU result cache with 5-12× speedup + adaptive rerank + ONNX cross-encoder), and **V7.3 (production-ready `mathir_dropin/` package with SQLite persistence + multi-agent support + 10 critical tests + 8 high-quality PNG diagrams + self-contained HTML visual report)** are all complete, tested, and benchmarked.

**V7.3 highlights**:
- `mathir_dropin/` — single-folder production distribution (5 files, 1500 lines, 10 tests)
- SQLite persistence (one `.db` file, inspectable)
- Multi-agent (20+ concurrent, no data loss)
- Cross-model plug-and-play
- Reset / Delete / Forget API
- Multi-modal examples (CLIP, CLAP, ImageBind)
- 8 PNG diagrams + HTML report (`visualizations/`)
- 6 new comprehensive docs (~50K words total)

The next milestones are the **cascade architecture** (V8 — auto-route between Approach A and D, with VectorDB as the L1 retriever), CARLA integration, edge deployment, and open-source release — see [FUTURE_VISION.md](FUTURE_VISION.md) for the strategic roadmap.

For the doctoral-grade mathematical treatment, see [`docs/THEORY_V7.md`](docs/THEORY_V7.md) (58 KB) and the NeurIPS-style paper at [`docs/V7_PAPER.md`](docs/V7_PAPER.md). For the V7.1 retrieval research report, see [`docs/RETRIEVAL_RESEARCH_REPORT.md`](docs/RETRIEVAL_RESEARCH_REPORT.md).

### What Exists

| Component | Status | Files |
|---|---|---|
| Core MATHIR model | ✅ Working | `mathir_model.py` |
| V5 architecture | ✅ Working | `mathir_lib/mathir_v5.py` |
| mHC (Sinkhorn-Knopp) | ✅ Working | `mathir_lib/mhc_v5.py` |
| 3-tier memory | ✅ Working | `mathir_model.py:MATHIRMemory` |
| KL-constrained router | ✅ Working | `mathir_lib/mathir_v5.py:KLConstrainedRouter` |
| Driving environment | ✅ Working | `driving_env.py` |
| Training loop | ✅ Working | `train_evolution.py` |
| Benchmarks | ✅ Working | `benchmark.py` |
| Dashboards | ✅ Working | `dashboard_live.py`, `app_streamlit.py` |
| V5 memory stubs | ✅ Fixed | `mathir_lib/mathir_v5.py` (real implementations) |
| Bug fixes | ✅ Done | 21 bugs fixed across 18 files |

### What's Done (V6 + V7 + V7.1)

| Component | Status | Version | Files |
|---|---|---|---|
| MATHIRPlugin API | ✅ Built (V6) | V6 | `mathir_lib/plugin.py` |
| 4-tier memory | ✅ Built (V6) | V6 | `mathir_lib/memory/{working,episodic,semantic,immunological}.py` |
| Embedding providers | ✅ Built (V6) | V6 | `mathir_lib/providers/{openai,ollama,huggingface,direct}.py` |
| TurboQuant compression | ✅ Built (V6) | V6 | `mathir_lib/compression.py` |
| V6 stress test (13/13 pass) | ✅ Built (V6) | V6 | `tests/stress_test.py` |
| V7 plugin (8 algorithms) | ✅ Built (V7) | V7 | `mathir_lib/plugin_v7.py` |
| EbbinghausMemory | ✅ Built (V7) | V7 | `mathir_lib/memory/ebbinghaus.py` |
| SparseCodingMemory | ✅ Built (V7) | V7 | `mathir_lib/memory/sparse_coding.py` |
| VariationalMemory | ✅ Built (V7) | V7 | `mathir_lib/memory/variational.py` |
| CrossAttentionMemory | ✅ Built (V7) | V7 | `mathir_lib/memory/cross_attention.py` |
| HyperbolicMemory | ✅ Built (V7) | V7 | `mathir_lib/memory/hyperbolic.py` |
| InfoNCELoss | ✅ Built (V7) | V7 | `mathir_lib/memory/infonce.py` |
| NeuralODEMemory | ✅ Built (V7) | V7 | `mathir_lib/memory/neural_ode.py` |
| MahalanobisImmunologicalMemory | ✅ Built (V7) | V7 | `mathir_lib/memory/immunological.py` |
| V7 6-theorem theory | ✅ Built (V7) | V7 | `docs/THEORY_V7.md` (58 KB) |
| V7 paper (NeurIPS-style) | ✅ Built (V7) | V7 | `docs/V7_PAPER.md` |
| V7 unit tests (49/49 pass) | ✅ Built (V7) | V7 | `tests/test_v7_memory.py` |
| V7 integration tests | ✅ Built (V7) | V7 | `tests/test_v7_integration.py` |
| V6 vs V7 benchmark (9.3× compression) | ✅ Built (V7) | V7 | `benchmarks/v6_vs_v7.py` |
| **Approach A — Raw Embedding Bypass** | ✅ Built (V7.1) | V7.1 | `mathir_lib/retrieval/raw_embedding.py` |
| **Approach B — Multi-Encoder Ensemble** | ✅ Built (V7.1) | V7.1 | `mathir_lib/retrieval/multi_encoder.py` |
| **Approach C — FAISS-Backed Index** | ✅ Built (V7.1) | V7.1 | `mathir_lib/retrieval/faiss_index.py` |
| **Approach D — Hybrid BM25 + Dense + Cross-Encoder** | ✅ Built (V7.1) | V7.1 | `mathir_lib/retrieval/hybrid_bm25_ce.py` |
| **Master comparison benchmark (5 systems × 50 queries)** | ✅ Built (V7.1) | V7.1 | `compare_all_approaches_results.json` |
| **Approach D vs FAISS deep benchmark** | ✅ Built (V7.1) | V7.1 | `approach_d_vs_faiss_results.json` |
| **Retrieval approach A unit tests (28/28 pass)** | ✅ Built (V7.1) | V7.1 | `tests/test_approach_a_raw.py` |
| **Retrieval approach B unit tests (36/36 pass)** | ✅ Built (V7.1) | V7.1 | `tests/test_approach_b_multi_encoder.py` |
| **Retrieval approach C unit tests (32/32 pass)** | ✅ Built (V7.1) | V7.1 | `tests/test_approach_c_faiss.py` |
| **Retrieval approach D unit tests (34/34 pass)** | ✅ Built (V7.1) | V7.1 | `tests/test_approach_d_hybrid.py` |
| **Doctoral retrieval research report** | ✅ Built (V7.1) | V7.1 | `docs/RETRIEVAL_RESEARCH_REPORT.md` |
| **LRU result cache on `(query, doc)` pairs** | ✅ Built (V7.2) | V7.2 | `mathir_lib/memory/hybrid_episodic.py` (`ResultCache` class) |
| **Adaptive re-ranking (skip cross-encoder on agreement)** | ✅ Built (V7.2) | V7.2 | `mathir_lib/memory/hybrid_episodic.py` (`adaptive_rerank` config) |
| **ONNX cross-encoder backend (2-3× faster)** | ✅ Built (V7.2) | V7.2 | `mathir_lib/memory/hybrid_episodic.py` (`_OnnxCrossEncoder`) |
| **Cross-encoder tiers (MiniLM-L-6 / TinyBERT-L-2 / Electra-base)** | ✅ Built (V7.2) | V7.2 | `mathir_lib/memory/hybrid_episodic.py` |
| **Cache + adaptive unit tests (62/62 pass)** | ✅ Built (V7.2) | V7.2 | `tests/test_hybrid_cache.py`, `tests/test_hybrid_adaptive.py` |
| **4-scenario cache stress benchmark** | ✅ Built (V7.2) | V7.2 | `benchmarks/stress_cache_warm.py` |
| **Cache + adaptive raw results** | ✅ Built (V7.2) | V7.2 | `stress_cache_warm_results.json` |
| **VectorDB vs MATHIR use case documentation** | ✅ Built (V7.2) | V7.2 | `docs/MATHIR_VS_VECTORDB_USE_CASES.md` |

**V7.1 retrieval results:**

| System | Overlap (quality) | Throughput | Latency |
|---|:---:|:---:|:---:|
| V7 default (64-dim projection) | 19.7% | 1,338 QPS | 0.66 ms |
| FAISS (raw 384-dim) | 31.6% | 20,392 QPS | 0.05 ms |
| **Approach A (Raw Embedding)** | 31.6% | 657 QPS | 1.54 ms |
| Approach B (Multi-Encoder) | 29.1% | 425 QPS | 2.20 ms |
| Approach C (FAISS-backed) | 31.6% | 97 QPS | 8.88 ms |
| **Approach D (Hybrid BM25+CE)** | **45.7%** | 2 QPS | 494 ms |

**Default strategy:** Approach A (online/interactive). **Batch strategy:** Approach D. The 12-14% quality gap is closed by switching from V7's 64-dim projection to raw 384-dim embeddings; the +14.1pp further gain comes from going dense → hybrid (BM25 + dense + cross-encoder).

### What Doesn't Exist Yet

| Component | Status | Needed For |
|---|---|---|
| **Cascade architecture (auto-route A vs D)** | ❌ Not built (V8) | Adaptive per-query cost/quality trade-off |
| **Confidence calibrator for Approach A scores** | ❌ Not built (V8) | Decide when to escalate to D |
| **Batched cross-encoder service** | ❌ Not built (V8) | 3× throughput on Approach D at same quality |
| **Hybrid index at store time (BM25 + dense precompute)** | ❌ Not built (V8) | Read-only cascade at query time |
| InfoNCE wired into training loop | ⚠️ Module exists, not wired | Online self-supervised learning |
| CARLA integration | ❌ Not built | Real driving benchmarks |
| ONNX export | ❌ Not built | Edge deployment |
| Rust core (PyO3) | ❌ Not built | 7× memory ops speedup |
| C++ wrapper | ❌ Not built | Jetson/RB5 deployment |
| ROS2 integration | ❌ Not built | Robotics deployment |
| HuggingFace model card | ❌ Not built | Community adoption |
| arXiv paper submission | ❌ Not built | Academic credibility (draft at `docs/V7_PAPER.md`) |

**Note:** Hybrid retrieval (Approach D) is **done** as of V7.1. The cascade that automatically routes between Approach A and Approach D based on confidence is the next milestone (V8).

### Measured V6 vs V7 Results

From [`docs/BENCHMARK_V6_VS_V7.md`](docs/BENCHMARK_V6_VS_V7.md) (`benchmarks/v6_vs_v7.py`):

| Metric | V6 | V7 | Gain |
|---|---|---|---|
| Compression (1000 × 272-dim embeddings) | 1,088,000 bytes | 116,976 bytes | **9.3× smaller** |
| Inference latency P50 (dim=1024, 100 iters) | 2.25 ms | 2.76 ms | +0.5 ms overhead |
| Router min weight (n=100) | 0.239 | 0.229 | similar (no collapse) |
| Anomaly F1 (synthetic data) | 0.5 | 0.5 | tied on synthetic (Theorem 4 holds for Gaussian data) |

---

## Phase 1: MATHIRPlugin API (Month 1)

### Goal

Refactor MATHIR into a clean, LLM-agnostic memory plugin.

### What Changes

| Current | New |
|---|---|
| `MATHIRAgent` (driving-specific) | `MATHIRPlugin` (generic) |
| Expects `{'camera': ..., 'state': ...}` | Accepts any `[B, D]` tensor |
| CNN vision encoder built-in | LLM provides embeddings |
| Reward-driven learning | Self-supervised objectives |
| Fixed config (84x84, 5-dim state) | Config-driven (any dim) |

### Implementation

#### Step 1: Create `mathir_lib/plugin.py`

```python
"""
MATHIR Plugin — Adaptive memory for any LLM.

Usage:
    from mathir_lib.plugin import MATHIRPlugin
    
    plugin = MATHIRPlugin(embedding_dim=768)
    
    # Perceive and remember
    output = plugin.perceive(llm_embedding)
    
    # Store experience
    plugin.store({'embedding': emb, 'action': act, 'outcome': rew})
    
    # Recall
    memories = plugin.recall(query_embedding, k=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tensor
from .mhc_v5 import ManifoldConstrainedLinearV5


class MATHIRPlugin(nn.Module):
    """
    Adaptive memory plugin for any LLM.
    
    Three-tier memory:
        - Working: immediate context (64 slots, attention retrieval)
        - Episodic: past experiences (1000 slots, similarity retrieval)
        - Semantic: learned concepts (256 prototypes, online k-means)
        
    Features:
        - KL-constrained router (prevents collapse)
        - Immunological memory (anomaly detection)
        - Online learning (never stops adapting)
        - Plug-and-play (works with any LLM)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        working_capacity: int = 64,
        episodic_capacity: int = 1000,
        semantic_prototypes: int = 256,
        kl_coefficient: float = 0.01,
        anomaly_threshold: float = 2.0,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input projection (LLM dim → MATHIR dim)
        self.input_proj = nn.Linear(embedding_dim, 272)
        
        # Working memory
        self.working_buffer = nn.Parameter(
            torch.zeros(working_capacity, 272), requires_grad=False
        )
        self.working_ptr = 0
        self.working_attention = nn.MultiheadAttention(
            272, num_heads=4, batch_first=True, dropout=0.1
        )
        
        # Episodic memory
        self.episodic_keys = nn.Parameter(
            torch.zeros(episodic_capacity, 64), requires_grad=False
        )
        self.episodic_values = nn.Parameter(
            torch.zeros(episodic_capacity, 272), requires_grad=False
        )
        self.episodic_ptr = 0
        self.episodic_count = 0
        self.episodic_encoder = nn.Linear(272, 64)
        
        # Semantic memory
        self.semantic_prototypes = nn.Parameter(
            torch.randn(semantic_prototypes, 64), requires_grad=False
        )
        self.semantic_usage = nn.Parameter(
            torch.zeros(semantic_prototypes), requires_grad=False
        )
        self.semantic_down = nn.Linear(272, 64)
        self.semantic_up = nn.Linear(64, 272)
        
        # Immunological memory
        self.immune_bank = nn.Parameter(
            torch.zeros(100, 272), requires_grad=False
        )
        self.immune_ptr = 0
        self.immune_count = 0
        self.anomaly_threshold = anomaly_threshold
        
        # KL-constrained router
        self.router = nn.Sequential(
            nn.Linear(272, 128),
            nn.GELU(),
            nn.Linear(128, 4),  # [working, episodic, semantic, immune]
        )
        self.register_buffer(
            'prev_probs', torch.ones(4) / 4
        )
        self.kl_coefficient = kl_coefficient
        
        # Output projection (MATHIR dim → LLM dim)
        self.output_proj = nn.Linear(272, embedding_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(272)
    
    def perceive(self, embedding: Tensor) -> Dict[str, Tensor]:
        """
        Process an embedding through the memory system.
        
        Args:
            embedding: [B, D] tensor from the LLM
            
        Returns:
            enhanced_embedding: [B, D] with memory context
            router_weights: which memory tier was used
            anomaly_score: how novel this input is
            kl_loss: KL divergence loss for router training
        """
        # Project to MATHIR dimension
        x = self.input_proj(embedding)
        
        # Store in working memory
        self._store_working(x)
        
        # Retrieve from each tier
        working_ctx = self._retrieve_working(x)
        episodic_ctx = self._retrieve_episodic(x)
        semantic_ctx = self._retrieve_semantic(x)
        immune_ctx = self._retrieve_immune(x)
        
        # Router allocation
        router_logits = self.router(x)
        router_weights = F.softmax(router_logits, dim=-1)
        
        # KL constraint
        kl_loss = self._compute_kl(router_logits)
        
        # Weighted fusion
        w = router_weights.chunk(4, dim=-1)
        output = (
            w[0] * working_ctx +
            w[1] * episodic_ctx +
            w[2] * semantic_ctx +
            w[3] * immune_ctx
        )
        
        # Residual + normalize
        output = self.layer_norm(output + x)
        
        # Project back to LLM dimension
        enhanced = self.output_proj(output)
        
        # Anomaly score
        anomaly = self._compute_anomaly(x)
        
        return {
            'enhanced_embedding': enhanced,
            'router_weights': router_weights,
            'anomaly_score': anomaly,
            'kl_loss': kl_loss,
        }
    
    def store(self, experience: Dict[str, Tensor]):
        """
        Store an experience for later recall.
        
        Args:
            experience: dict with 'embedding', 'action' (optional), 'outcome' (optional)
        """
        emb = self.input_proj(experience['embedding'].detach())
        
        # Store in episodic memory
        key = self.episodic_encoder(emb.mean(0, keepdim=True)).squeeze(0)
        idx = self.episodic_ptr % self.episodic_keys.size(0)
        self.episodic_keys[idx] = key.detach()
        self.episodic_values[idx] = emb.mean(0).detach()
        self.episodic_ptr += 1
        self.episodic_count = min(
            self.episodic_count + 1, self.episodic_keys.size(0)
        )
        
        # Update semantic prototypes
        self._update_semantic(emb)
        
        # Update immune bank
        self._update_immune(emb)
    
    def recall(self, query: Tensor, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant memories.
        
        Args:
            query: [B, D] tensor to search for
            k: number of memories to retrieve
            
        Returns:
            list of memory dicts, ranked by relevance
        """
        x = self.input_proj(query)
        
        if self.episodic_count < k:
            return []
        
        # Search episodic memory
        key = self.episodic_encoder(x)
        sims = F.cosine_similarity(
            key.unsqueeze(1),
            self.episodic_keys[:self.episodic_count].unsqueeze(0),
            dim=-1
        )
        top_k = sims.topk(min(k, self.episodic_count), dim=1)
        
        memories = []
        for i in range(top_k.indices.size(1)):
            idx = top_k.indices[0, i].item()
            memories.append({
                'embedding': self.episodic_values[idx],
                'similarity': top_k.values[0, i].item(),
                'index': idx,
            })
        
        return memories
    
    def forget(self, threshold: float = 0.1):
        """Prune irrelevant memories (controlled forgetting)."""
        if self.episodic_count == 0:
            return
        
        # Compute usage scores
        keys = self.episodic_keys[:self.episodic_count]
        sims = F.cosine_similarity(
            keys.unsqueeze(1), keys.unsqueeze(0), dim=-1
        )
        usage = sims.mean(dim=1)
        
        # Prune low-usage memories
        mask = usage > threshold
        if mask.sum() < self.episodic_count:
            self.episodic_keys[:mask.sum()] = self.episodic_keys[:self.episodic_count][mask]
            self.episodic_values[:mask.sum()] = self.episodic_values[:self.episodic_count][mask]
            self.episodic_count = mask.sum().item()
            self.episodic_ptr = self.episodic_count
    
    def get_stats(self) -> Dict:
        """Get memory utilization statistics."""
        return {
            'working_usage': min(self.working_ptr, self.working_buffer.size(0)),
            'working_capacity': self.working_buffer.size(0),
            'episodic_usage': self.episodic_count,
            'episodic_capacity': self.episodic_keys.size(0),
            'semantic_usage': (self.semantic_usage > 0).sum().item(),
            'semantic_capacity': self.semantic_prototypes.size(0),
            'immune_usage': self.immune_count,
            'immune_capacity': self.immune_bank.size(0),
        }
    
    # --- Internal methods ---
    
    def _store_working(self, x: Tensor):
        """Store in working memory circular buffer."""
        batch_size = x.size(0)
        with torch.no_grad():
            indices = (
                self.working_ptr + torch.arange(batch_size, device=x.device)
            ) % self.working_buffer.size(0)
            self.working_buffer[indices] = x.detach()
            self.working_ptr = (self.working_ptr + batch_size) % self.working_buffer.size(0)
    
    def _retrieve_working(self, x: Tensor) -> Tensor:
        """Retrieve from working memory via attention."""
        stored = min(self.working_ptr, self.working_buffer.size(0))
        if stored == 0:
            return torch.zeros_like(x)
        context = self.working_buffer[:stored].unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.working_attention(x.unsqueeze(1), context, context)
        return out.squeeze(1)
    
    def _retrieve_episodic(self, x: Tensor) -> Tensor:
        """Retrieve from episodic memory via similarity."""
        if self.episodic_count < 10:
            return torch.zeros_like(x)
        key = self.episodic_encoder(x)
        sims = F.cosine_similarity(
            key.unsqueeze(1),
            self.episodic_keys[:self.episodic_count].unsqueeze(0),
            dim=-1
        )
        top_k = sims.topk(min(3, self.episodic_count), dim=1)[1]
        return self.episodic_values[top_k].mean(1)
    
    def _retrieve_semantic(self, x: Tensor) -> Tensor:
        """Retrieve from semantic memory via prototype matching."""
        projected = self.semantic_down(x)
        sims = F.cosine_similarity(
            projected.unsqueeze(1),
            self.semantic_prototypes.unsqueeze(0),
            dim=-1
        )
        idx = sims.argmax(dim=1)
        return self.semantic_up(self.semantic_prototypes[idx])
    
    def _retrieve_immune(self, x: Tensor) -> Tensor:
        """Detect anomalies via immune memory."""
        if self.immune_count < 10:
            return torch.zeros_like(x)
        dists = torch.cdist(x, self.immune_bank[:self.immune_count])
        min_dist = dists.min(dim=1)[0]
        anomaly = (min_dist > self.anomaly_threshold).float().unsqueeze(-1)
        return anomaly * x
    
    def _update_semantic(self, x: Tensor):
        """Update semantic prototypes via online k-means."""
        with torch.no_grad():
            projected = self.semantic_down(x)
            sims = F.cosine_similarity(
                projected.unsqueeze(1),
                self.semantic_prototypes.unsqueeze(0),
                dim=-1
            )
            idx = sims.argmax(dim=1)
            alpha = 0.01
            for i in range(x.size(0)):
                self.semantic_prototypes[idx[i]] = (
                    (1 - alpha) * self.semantic_prototypes[idx[i]] +
                    alpha * projected[i].detach()
                )
                self.semantic_usage[idx[i]] += 1
    
    def _update_immune(self, x: Tensor):
        """Update immune memory bank."""
        with torch.no_grad():
            idx = self.immune_ptr % self.immune_bank.size(0)
            self.immune_bank[idx] = x.mean(0).detach()
            self.immune_ptr += 1
            self.immune_count = min(
                self.immune_count + 1, self.immune_bank.size(0)
            )
    
    def _compute_kl(self, logits: Tensor) -> Tensor:
        """Compute KL divergence loss for router."""
        target = self.prev_probs.unsqueeze(0).expand_as(logits)
        kl = F.kl_div(
            F.log_softmax(logits, dim=-1),
            target,
            reduction='batchmean',
            log_target=False
        )
        # Update previous policy
        with torch.no_grad():
            self.prev_probs = F.softmax(logits, dim=-1).mean(dim=0)
        return self.kl_coefficient * kl
    
    def _compute_anomaly(self, x: Tensor) -> Tensor:
        """Compute anomaly score for input."""
        if self.immune_count < 10:
            return torch.zeros(x.size(0), device=x.device)
        dists = torch.cdist(x, self.immune_bank[:self.immune_count])
        return dists.min(dim=1)[0]
```

#### Step 2: Create `examples/` directory

```
examples/
├── basic_usage.py          # Simple plugin usage
├── with_claude.py          # Integration with Claude API
├── with_openai.py          # Integration with OpenAI API
├── with_local_llm.py       # Integration with local 7B model
├── driving_demo.py         # Driving scenario demo
└── benchmark_vs_rag.py     # Compare MATHIR vs RAG
```

#### Step 3: Update `mathir_lib/__init__.py`

```python
from .plugin import MATHIRPlugin
```

### Verification

```python
from mathir_lib.plugin import MATHIRPlugin
import torch

plugin = MATHIRPlugin(embedding_dim=768)
embedding = torch.randn(4, 768)

output = plugin.perceive(embedding)
assert output['enhanced_embedding'].shape == (4, 768)
assert output['router_weights'].shape == (4, 4)

plugin.store({'embedding': embedding})
memories = plugin.recall(embedding, k=3)

print("MATHIRPlugin: OK")
```

---

## Phase 2: Self-Supervised Learning (Month 2)

### Goal

Enable MATHIR to learn without reward signals — using the LLM's embeddings themselves.

### Objectives

| Objective | Method | What It Learns |
|---|---|---|
| **Reconstruction** | Autoencoder loss | Can memory reconstruct input from prototypes? |
| **Prediction** | Next-embedding loss | Can memory predict what comes next? |
| **Novelty** | Anomaly detection loss | Is this input novel or expected? |
| **Retention** | Long-term recall loss | Can memory retain info over N steps? |

### Implementation

```python
class SelfSupervisedLoss(nn.Module):
    """Self-supervised objectives for MATHIR learning."""
    
    def __init__(self, plugin: MATHIRPlugin):
        super().__init__()
        self.plugin = plugin
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(272, 272),
            nn.GELU(),
            nn.Linear(272, 272)
        )
        
        # Reconstruction head
        self.reconstructor = nn.Sequential(
            nn.Linear(272, 272),
            nn.GELU(),
            nn.Linear(272, 272)
        )
    
    def forward(self, embeddings: List[Tensor]) -> Dict[str, Tensor]:
        """
        Compute self-supervised losses.
        
        Args:
            embeddings: list of [B, D] tensors (sequential observations)
            
        Returns:
            dict with 'prediction_loss', 'reconstruction_loss',
            'novelty_loss', 'retention_loss', 'total_loss'
        """
        losses = {}
        
        # Prediction: predict next embedding from current + memory
        if len(embeddings) >= 2:
            current = self.plugin.input_proj(embeddings[-2])
            target = self.plugin.input_proj(embeddings[-1])
            predicted = self.predictor(current)
            losses['prediction_loss'] = F.mse_loss(predicted, target.detach())
        
        # Reconstruction: reconstruct input from memory
        x = self.plugin.input_proj(embeddings[-1])
        output = self.plugin.perceive(embeddings[-1])
        enhanced = self.plugin.input_proj(output['enhanced_embedding'])
        reconstructed = self.reconstructor(enhanced)
        losses['reconstruction_loss'] = F.mse_loss(reconstructed, x.detach())
        
        # Novelty: anomaly score should be low for familiar inputs
        anomaly = output['anomaly_score']
        losses['novelty_loss'] = anomaly.mean()  # Minimize anomaly
        
        # Retention: recall should match original after N steps
        if len(embeddings) >= 10:
            query = embeddings[-1]
            memories = self.plugin.recall(query, k=1)
            if memories:
                recalled = memories[0]['embedding']
                target = self.plugin.input_proj(query).mean(0)
                losses['retention_loss'] = F.mse_loss(recalled, target.detach())
        
        # Total loss
        losses['total_loss'] = sum(losses.values())
        
        return losses
```

### Verification

```python
plugin = MATHIRPlugin(embedding_dim=768)
loss_fn = SelfSupervisedLoss(plugin)

# Simulate 10 steps
embeddings = [torch.randn(4, 768) for _ in range(10)]
for emb in embeddings:
    plugin.perceive(emb)
    plugin.store({'embedding': emb})

losses = loss_fn(embeddings)
assert losses['total_loss'].requires_grad
print(f"Self-supervised loss: {losses['total_loss'].item():.4f}")
```

---

## Phase 3: Demo with 3 LLMs (Month 3)

### Goal

Prove MATHIR works with different LLM architectures.

### LLMs to Demo

| LLM | Type | Embedding Dim | Why |
|---|---|---|---|
| Claude 3.5 | Commercial API | 1024 | Most capable |
| GPT-4o | Commercial API | 1536 | Most popular |
| Qwen2.5-7B | Local | 3584 | Open source |

### Demo Script

```python
# examples/with_claude.py
import anthropic
from mathir_lib.plugin import MATHIRPlugin

client = anthropic.Anthropic()
plugin = MATHIRPlugin(embedding_dim=1024)

# Get Claude embedding
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "I see a red traffic light ahead."}]
)

# Extract embedding (via API or internal)
embedding = extract_embedding(response)  # [1, 1024]

# Process through MATHIR
output = plugin.perceive(embedding)
enhanced = output['enhanced_embedding']

# Use enhanced embedding for next Claude call
# (inject as system context or prepend to message)
```

### Benchmark vs RAG

```python
# examples/benchmark_vs_rag.py
"""
Compare MATHIR vs RAG on:
1. Retrieval accuracy
2. Learning speed
3. Memory efficiency
4. Inference latency
"""
```

---

## Phase 4: CARLA Integration (Month 4)

### Goal

Prove MATHIR works on real driving scenarios.

### What to Build

1. CARLA environment wrapper
2. MATHIR + VLM integration (Qwen3-VL for perception)
3. Benchmark suite (CARLA scenarios)
4. Comparison: MATHIR vs RAG vs long context

### CARLA Scenarios

| Scenario | What It Tests |
|---|---|
| Town01 highway | Long-term retention (lane keeping) |
| Town03 intersection | Episodic memory (past decisions) |
| Town04 tunnel | Semantic memory (concept transfer) |
| Weather change | Adaptation (differentiable plasticity) |
| Novel obstacle | Anomaly detection (immunological memory) |

---

## Phase 5: Paper Draft (Month 4)

### Title

*"MATHIR: Adaptive Hierarchical Memory for Long-Horizon Embodied AI"*

### Abstract

> We present MATHIR, an adaptive memory layer that gives embodied AI agents the ability to learn, remember, and adapt in real-time. Unlike static memory solutions (vector databases, RAG), MATHIR maintains three tiers of memory — working (immediate context), episodic (past experiences), and semantic (learned concepts) — that learn online via self-supervised objectives. MATHIR uses Manifold-Constrained Hyper-Connections (mHC) with Sinkhorn-Knopp projection for gradient stability, and a KL-constrained router that prevents memory collapse. At 0.6GB VRAM and 10ms inference, MATHIR runs on edge hardware where larger models cannot. We demonstrate plug-and-play integration with three LLM architectures (Claude, GPT-4o, Qwen2.5-7B) and show significant improvements over RAG and long-context baselines on CARLA driving scenarios.

### Key Contributions

1. **First adaptive memory layer** for LLMs that learns online
2. **Three-tier hierarchical memory** with KL-constrained routing
3. **Immunological memory** for anomaly detection in embodied AI
4. **Plug-and-play API** that works with any LLM architecture
5. **Edge deployment** at 0.6GB VRAM, 10ms inference

### Experiments

| Experiment | Baseline | MATHIR | Metric |
|---|---|---|---|
| CARLA retention | RAG | +45% | Long-term recall accuracy |
| CARLA adaptation | Long context | +30% | Recovery from perturbation |
| Edge latency | Vector DB | 5x faster | Inference time |
| Memory efficiency | RAG | 3x smaller | VRAM usage |

---

## Phase 6: Open Source Release (Month 5)

### Checklist

- [ ] Clean codebase (remove dead code, add docstrings)
- [ ] Tests (unit tests, integration tests, benchmark tests)
- [ ] Documentation (README, API docs, tutorials)
- [ ] HuggingFace model card
- [ ] Papers With Code submission
- [ ] GitHub Actions CI/CD
- [ ] PyPI package
- [ ] Docker image
- [ ] Example notebooks

---

## Phase 7: Edge Deployment (Month 6)

### Targets

| Platform | Hardware | VRAM | Status |
|---|---|---|---|
| NVIDIA Jetson Orin | 8GB | 0.6GB | 📋 Planned |
| Qualcomm RB5 | 4GB | 0.6GB | 📋 Planned |
| Intel RealSense | 4GB | 0.6GB | 📋 Planned |
| Raspberry Pi 5 | 8GB | 0.6GB (CPU) | 📋 Planned |

### Export Formats

| Format | Use Case |
|---|---|
| ONNX | Cross-platform inference |
| TensorRT | NVIDIA GPU optimization |
| CoreML | Apple Silicon |
| TFLite | Mobile/edge |

---

## Timeline Summary

| Month | Milestone | Deliverable |
|---|---|---|
| **1** | MATHIRPlugin API | `mathir_lib/plugin.py` + examples |
| **2** | Self-supervised learning | `SelfSupervisedLoss` + benchmarks |
| **3** | Demo with 3 LLMs | Working demos + benchmark vs RAG |
| **4** | CARLA + Paper | CARLA integration + paper draft |
| **5** | Open source | GitHub release + HuggingFace |
| **6** | Edge deployment | Jetson demo + ONNX export |

---

## Dependencies

| Phase | New Dependencies |
|---|---|
| 1 | None (PyTorch only) |
| 2 | None (PyTorch only) |
| 3 | `anthropic`, `openai`, `transformers` |
| 4 | `carla` (CARLA simulator) |
| 5 | `pytest`, `black`, `flake8`, `sphinx` |
| 6 | `onnx`, `onnxruntime`, `tensorrt` |

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| CARLA too complex | Start with simple scenarios, iterate |
| Paper rejected | Target multiple venues (ICLR, ICML, NeurIPS) |
| Low adoption | Build community early (GitHub, Discord) |
| Edge deployment hard | Start with ONNX, optimize later |
| Self-supervised doesn't work | Fall back to reward-driven learning |

---

*"The best time to build a memory layer was yesterday. The second best time is now."*

---

See [FUTURE_VISION.md](FUTURE_VISION.md) for the strategic direction.
See [README.md](README.md) for the current state.
