"""
MATHIR ONNX Provider Usage Examples
"""

import sys
sys.path.insert(0, r'C:\Users\So-i-learn-3D\.config\opencode\bin')

# Example 1: Basic usage
print("=" * 60)
print("Example 1: Basic ONNX Provider Usage")
print("=" * 60)

from mathir_lib.providers import get_provider

# Create ONNX provider
provider = get_provider("onnx", {
    "model_dir": r"C:\Users\So-i-learn-3D\.config\opencode\models\octen-int8"
})

print(f"Provider: {provider.provider_id()}")
print(f"Embedding dim: {provider.embedding_dim()}")

# Encode texts
texts = [
    "How to configure VPN on Windows?",
    "Installation de Python sur Mac",
    "Debug a Python function"
]

embeddings = provider.embed_batch(texts)
print(f"Input: {len(texts)} texts")
print(f"Output shape: {embeddings.shape}")
print(f"Output dtype: {embeddings.dtype}")

# Compute similarities
sim = embeddings @ embeddings.T
print(f"\nSimilarity matrix:")
for i, t in enumerate(texts):
    for j, t2 in enumerate(texts):
        if i < j:
            print(f"  {t[:30]}... <-> {t2[:30]}...: {sim[i,j]:.3f}")

# Example 2: Compare with HuggingFace
print("\n" + "=" * 60)
print("Example 2: ONNX vs HuggingFace Comparison")
print("=" * 60)

import time

# ONNX
start = time.perf_counter()
onnx_emb = provider.embed_batch(texts)
onnx_time = (time.perf_counter() - start) * 1000

# HuggingFace
hf_provider = get_provider("huggingface", {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cuda"
})

start = time.perf_counter()
hf_emb = hf_provider.embed_batch(texts)
hf_time = (time.perf_counter() - start) * 1000

print(f"ONNX (Octen INT8): {onnx_time:.1f}ms, dim={provider.embedding_dim()}")
print(f"HuggingFace (MiniLM): {hf_time:.1f}ms, dim={hf_provider.embedding_dim()}")
print(f"Speed ratio: {onnx_time/hf_time:.2f}x")
print(f"Quality: ONNX has {provider.embedding_dim()/hf_provider.embedding_dim():.1f}x more dimensions")

# Example 3: MCP Server integration
print("\n" + "=" * 60)
print("Example 3: MCP Server Integration")
print("=" * 60)

print("""
To use MATHIR as an MCP server:

1. Add to your MCP config (e.g., ~/.config/opencode/opencode.json):

{
  "mcp": {
    "mathir": {
      "command": "python",
      "args": ["D:\\SECRET_PROJECT\\MATHIR\\mcp_server.py"],
      "env": {
        "PYTHONPATH": "D:\\SECRET_PROJECT\\MATHIR"
      }
    }
  }
}

2. Use in your code:

from mathir_lib.providers import get_provider

provider = get_provider("onnx", {
    "model_dir": "C:/Users/So-i-learn-3D/.config/opencode/models/octen-int8"
})

# Save memory
embedding = provider.embed_batch(["Important insight"])[0]

# Recall memories
query_emb = provider.embed_batch(["search query"])[0]
similarities = all_embeddings @ query_emb
""")

print("\n" + "=" * 60)
print("Examples Complete")
print("=" * 60)
