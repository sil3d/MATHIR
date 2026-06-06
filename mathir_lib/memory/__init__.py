"""
MATHIR Memory Modules
V6: 4-tier (working, episodic, semantic, immunological)
V7: + Variational, SparseCoding, Ebbinghaus, CrossAttention, Hyperbolic, InfoNCE, NeuralODE, Mahalanobis
"""

# V6 modules
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .immunological import ImmunologicalMemory, MahalanobisImmunologicalMemory, EnsembleColdStartImmunologicalMemory

# V7 modules — high impact
from .ebbinghaus import EbbinghausMemory
from .sparse_coding import SparseCodingMemory
from .variational import VariationalMemory

# V7 modules — advanced
from .cross_attention import CrossAttentionMemory
from .hyperbolic import HyperbolicMemory
from .infonce import InfoNCELoss
from .neural_ode import NeuralODEMemory

# Raw-embedding episodic (no projection bottleneck)
from .raw_episodic import RawEmbeddingEpisodicMemory

# V8 (experimental) — external index backends
try:
    from .faiss_episodic import FAISSBackedEpisodicMemory
    _HAS_FAISS_BACKED = True
except ImportError:
    FAISSBackedEpisodicMemory = None  # type: ignore
    _HAS_FAISS_BACKED = False

# Adaptive multi-encoder retrieval (multi-dim cosine ensemble)
from .ensemble_episodic import EnsembleEpisodicMemory

# Hybrid retrieval (BM25 + Dense + Cross-Encoder Re-Rank)
from .hybrid_episodic import HybridEpisodicMemory

__all__ = [
    # V6
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ImmunologicalMemory",
    "MahalanobisImmunologicalMemory",
    "EnsembleColdStartImmunologicalMemory",
    # V7 — high impact
    "EbbinghausMemory",
    "SparseCodingMemory",
    "VariationalMemory",
    # V7 — advanced
    "CrossAttentionMemory",
    "HyperbolicMemory",
    "InfoNCELoss",
    "NeuralODEMemory",
    # Raw-embedding episodic (bypass the projection bottleneck)
    "RawEmbeddingEpisodicMemory",
    # V8 — experimental
    "FAISSBackedEpisodicMemory",
    # Adaptive multi-encoder retrieval
    "EnsembleEpisodicMemory",
    # Hybrid retrieval (BM25 + Dense + Cross-Encoder Re-Rank)
    "HybridEpisodicMemory",
]
