"""
MATHIR ONNX Embedder Usage Examples (v8.5.0)
============================================

Demonstrates the Octen-Embedding-0.6B-ONNX-INT8 embedder used by the
MATHIR MCP server. The old v7 ``mathir_lib.providers.get_provider``
factory was removed in v8.5.0 — use ``OctenEmbedder`` directly.

Run:
    python examples/onnx_usage.py
"""

import os
import sys

# Make ``mathir_lib`` importable when running from a fresh checkout.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "mathir_mcp"))
sys.path.insert(0, r"C:\Users\So-i-learn-3D\.config\opencode\bin")

from mathir_lib.mathir_onnx_embedder import OctenEmbedder, get_onnx_embedder

# Default model dir (matches where the model is actually installed).
DEFAULT_MODEL_DIR = r"C:\Users\So-i-learn-3D\.config\opencode\models\octen-int8"


# ---------------------------------------------------------------------------
# Example 1 — Basic usage
# ---------------------------------------------------------------------------
print("=" * 60)
print("Example 1: Basic OctenEmbedder Usage")
print("=" * 60)

provider = OctenEmbedder(model_dir=DEFAULT_MODEL_DIR)

print(f"Provider: {type(provider).__name__}")
print(f"Embedding dim: {provider.dim}")

texts = [
    "How to configure VPN on Windows?",
    "Installation de Python sur Mac",
    "Debug a Python function",
]

embeddings = provider.encode(texts)
print(f"Input:  {len(texts)} texts")
print(f"Output shape: {embeddings.shape}")
print(f"Output dtype: {embeddings.dtype}")

# Cosine similarities (embeddings are already L2-normalised).
sim = embeddings @ embeddings.T
print("\nSimilarity matrix:")
for i, t in enumerate(texts):
    for j, t2 in enumerate(texts):
        if i < j:
            print(f"  {t[:30]:30s} <-> {t2[:30]:30s}: {sim[i, j]:.3f}")


# ---------------------------------------------------------------------------
# Example 2 — Multilingual quality (the actual strength of Octen)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 2: Multilingual Embedding Quality")
print("=" * 60)

import time

ml_texts = [
    "How to configure VPN on Windows?",          # English
    "Installation de Python sur Mac",             # French
    "Comment configurer un VPN sous Windows",     # French (paraphrase)
    "Cómo instalar Python en Mac",                # Spanish
    "How to debug a Python function",             # English (paraphrase)
]

start = time.perf_counter()
ml_emb = provider.encode(ml_texts)
ml_ms = (time.perf_counter() - start) * 1000

print(f"Encoded {len(ml_texts)} multilingual texts in {ml_ms:.1f} ms")
print(f"Embedding dim: {provider.dim} (vs MiniLM's 384 — ~2.7x more dimensions)")

# Cross-language paraphrase detection
sim_ml = ml_emb @ ml_emb.T
print("\nCross-language similarity (Octen INT8 should detect paraphrases):")
pairs = [(0, 1, "EN-Q1  <-> FR-install"),
         (0, 2, "EN-Q1  <-> FR-VPN   (paraphrase!)"),
         (1, 3, "FR-ins <-> ES-ins   (paraphrase!)"),
         (0, 4, "EN-Q1  <-> EN-debug (unrelated)")]
for i, j, label in pairs:
    print(f"  {label:35s}: {sim_ml[i, j]:.3f}")


# ---------------------------------------------------------------------------
# Example 3 — MCP Server integration (paths updated for v8.5.0)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Example 3: MCP Server Integration")
print("=" * 60)

print("""
To use MATHIR as an MCP server (v8.5.0 layout):

1. Add to your MCP config (e.g., ~/.config/opencode/opencode.json):

{
  "mcp": {
    "mathir": {
      "command": "python",
      "args": ["D:\\\\SECRET_PROJECT\\\\MATHIR\\\\mathir_mcp\\\\mathir_lib\\\\mathir_mcp_server.py"],
      "env": {
        "PYTHONPATH": "D:\\\\SECRET_PROJECT\\\\MATHIR\\\\mathir_mcp"
      }
    }
  }
}

2. Use the embedder in your code:

from mathir_lib.mathir_onnx_embedder import OctenEmbedder

embedder = OctenEmbedder(model_dir="C:/Users/So-i-learn-3D/.config/opencode/models/octen-int8")

# Encode a single insight
embedding = embedder.encode(["Important insight"])[0]

# Encode a search query and rank candidates
query_emb = embedder.encode(["search query"])[0]
similarities = all_embeddings @ query_emb  # cosine sim (vectors are L2-normalised)
""")

print("\n" + "=" * 60)
print("Examples Complete")
print("=" * 60)
