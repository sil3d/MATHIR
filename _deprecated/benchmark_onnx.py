"""
MATHIR ONNX Provider Benchmark
Compares ONNX (Octen INT8) vs HuggingFace (MiniLM) vs OpenAI
"""

import time
import sys
sys.path.insert(0, r'C:\Users\So-i-learn-3D\.config\opencode\bin')

print("=" * 60)
print("MATHIR ONNX Provider Benchmark")
print("=" * 60)

# Test data
queries = [
    "Comment configurer le VPN sur Windows?",
    "How to install Python on Mac?",
    "Installation de sqlite-vec pour la recherche vectorielle",
    "What is the best way to learn programming?",
    "Debug a Python function that returns None"
]

documents = [
    "Guide d'installation de Python sur Windows et Mac",
    "Configuration du VPN pour accès à distance",
    "Installation de sqlite-vec pour la recherche vectorielle rapide",
    "Apprendre à programmer avec Python - tutoriel débutant",
    "Déboguer une fonction Python qui retourne None",
    "Optimisation des performances d'une application web",
    "Installation de Docker sur Ubuntu",
    "Configuration de git pour le travail en équipe"
]

# 1. ONNX Provider (Octen INT8)
print("\n1. ONNX Provider (Octen-Embedding-0.6B-ONNX-INT8)")
from mathir_lib.providers.onnx import ONNXProvider

onnx_provider = ONNXProvider({
    "model_dir": r"C:\Users\So-i-learn-3D\.config\opencode\models\octen-int8"
})

# Warmup
for _ in range(3):
    onnx_provider.embed_batch(["warmup"])

start = time.perf_counter()
query_emb = onnx_provider.embed_batch(queries)
doc_emb = onnx_provider.embed_batch(documents)
sim = query_emb @ doc_emb.T
onnx_time = (time.perf_counter() - start) * 1000

print(f"   Encode {len(queries)} queries + {len(documents)} docs: {onnx_time:.1f}ms")
print(f"   Query shape: {query_emb.shape}")
print(f"   Doc shape: {doc_emb.shape}")
print(f"   Embedding dim: {onnx_provider.embedding_dim()}")
print(f"   Provider ID: {onnx_provider.provider_id()}")

# 2. HuggingFace Provider (MiniLM)
print("\n2. HuggingFace Provider (MiniLM-L6-v2)")
from mathir_lib.providers.huggingface import HuggingFaceProvider

hf_provider = HuggingFaceProvider({
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cuda"
})

# Warmup
for _ in range(3):
    hf_provider.embed_batch(["warmup"])

start = time.perf_counter()
query_emb2 = hf_provider.embed_batch(queries)
doc_emb2 = hf_provider.embed_batch(documents)
sim2 = query_emb2 @ doc_emb2.T
hf_time = (time.perf_counter() - start) * 1000

print(f"   Encode {len(queries)} queries + {len(documents)} docs: {hf_time:.1f}ms")
print(f"   Query shape: {query_emb2.shape}")
print(f"   Doc shape: {doc_emb2.shape}")
print(f"   Embedding dim: {hf_provider.embedding_dim()}")
print(f"   Provider ID: {hf_provider.provider_id()}")

# 3. Speed comparison
print("\n3. Speed Comparison")
print(f"   ONNX (Octen INT8): {onnx_time:.1f}ms")
print(f"   HuggingFace (MiniLM): {hf_time:.1f}ms")
print(f"   Ratio: {onnx_time/hf_time:.2f}x")

# 4. Single query speed (100 runs)
print("\n4. Single Query Speed (100 runs)")
times_onnx = []
for _ in range(100):
    start = time.perf_counter()
    onnx_provider.embed_batch(["test query"])
    times_onnx.append(time.perf_counter() - start)

times_hf = []
for _ in range(100):
    start = time.perf_counter()
    hf_provider.embed_batch(["test query"])
    times_hf.append(time.perf_counter() - start)

avg_onnx = sum(times_onnx) / len(times_onnx) * 1000
avg_hf = sum(times_hf) / len(times_hf) * 1000

print(f"   ONNX (Octen INT8): {avg_onnx:.1f}ms")
print(f"   HuggingFace (MiniLM): {avg_hf:.1f}ms")
print(f"   Ratio: {avg_onnx/avg_hf:.2f}x")

# 5. Quality comparison
print("\n5. Quality Comparison")
print(f"   ONNX similarity range: [{sim.min():.3f}, {sim.max():.3f}], mean={sim.mean():.3f}")
print(f"   HuggingFace similarity range: [{sim2.min():.3f}, {sim2.max():.3f}], mean={sim2.mean():.3f}")

# 6. Memory usage
import os
import psutil
process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"\n6. Memory Usage: {mem_mb:.1f} MB")

print("\n" + "=" * 60)
print("Benchmark Complete")
print("=" * 60)
