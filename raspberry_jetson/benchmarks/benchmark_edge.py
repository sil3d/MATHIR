#!/usr/bin/env python3
"""
benchmark_edge.py — Benchmark MATHIR on Raspberry Pi / Jetson
Measures: cold start, warm embedding, batch, recall, memory usage.
"""

import time
import json
import sys
import os
import argparse
import urllib.request
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback for systems without psutil
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except:
            return 0
    return 0


def get_gpu_memory():
    """Get GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1024 / 1024
    except:
        pass
    return 0


def benchmark_ollama(url="http://localhost:11434", model="nomic-embed-text", num_embeddings=10):
    """Benchmark Ollama embedding."""
    print(f"\n=== Ollama Benchmark ({model}) ===")
    results = {}

    # Test connection
    try:
        req = urllib.request.Request(f"{url}/api/tags")
        urllib.request.urlopen(req, timeout=5)
        print("  Connection: OK")
    except Exception as e:
        print(f"  Connection: FAILED ({e})")
        return None

    # Cold start
    print("  Measuring cold start...")
    payload = json.dumps({"model": model, "prompt": "Hello world"}).encode()
    req = urllib.request.Request(
        f"{url}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    start = time.time()
    urllib.request.urlopen(req, timeout=30)
    cold_start = time.time() - start
    results["cold_start_ms"] = round(cold_start * 1000, 1)
    print(f"  Cold start: {results['cold_start_ms']}ms")

    # Warm embeddings
    print(f"  Measuring {num_embeddings} warm embeddings...")
    times = []
    for i in range(num_embeddings):
        start = time.time()
        urllib.request.urlopen(req, timeout=30)
        times.append((time.time() - start) * 1000)
    avg_warm = sum(times) / len(times)
    results["warm_avg_ms"] = round(avg_warm, 1)
    results["warm_min_ms"] = round(min(times), 1)
    results["warm_max_ms"] = round(max(times), 1)
    print(f"  Warm avg: {results['warm_avg_ms']}ms (min: {results['warm_min_ms']}ms, max: {results['warm_max_ms']}ms)")

    # Batch embedding
    texts = [f"This is test sentence number {i}" for i in range(10)]
    payload = json.dumps({"model": model, "prompt": texts}).encode()
    req = urllib.request.Request(
        f"{url}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    start = time.time()
    urllib.request.urlopen(req, timeout=60)
    batch_time = (time.time() - start) * 1000
    results["batch_10_ms"] = round(batch_time, 1)
    results["batch_per_text_ms"] = round(batch_time / 10, 1)
    print(f"  Batch (10): {results['batch_10_ms']}ms ({results['batch_per_text_ms']}ms/text)")

    # Memory
    results["ram_mb"] = round(get_memory_usage(), 1)
    results["gpu_mb"] = round(get_gpu_memory(), 1)
    print(f"  RAM: {results['ram_mb']}MB | GPU: {results['gpu_mb']}MB")

    return results


def benchmark_onnx(model_dir=None, num_embeddings=10):
    """Benchmark ONNX embedding."""
    print(f"\n=== ONNX Benchmark ===")
    results = {}

    try:
        from onnx_embedder import OctenEmbedder
        embedder = OctenEmbedder(model_dir)
    except Exception as e:
        print(f"  Failed to load ONNX model: {e}")
        return None

    # Cold start (already loaded, but measure first embedding)
    print("  Measuring cold start...")
    start = time.time()
    embedder.embed("Hello world")
    cold_start = (time.time() - start) * 1000
    results["cold_start_ms"] = round(cold_start, 1)
    print(f"  Cold start: {results['cold_start_ms']}ms")

    # Warm embeddings
    print(f"  Measuring {num_embeddings} warm embeddings...")
    times = []
    for i in range(num_embeddings):
        start = time.time()
        embedder.embed(f"This is test sentence number {i}")
        times.append((time.time() - start) * 1000)
    avg_warm = sum(times) / len(times)
    results["warm_avg_ms"] = round(avg_warm, 1)
    results["warm_min_ms"] = round(min(times), 1)
    results["warm_max_ms"] = round(max(times), 1)
    print(f"  Warm avg: {results['warm_avg_ms']}ms (min: {results['warm_min_ms']}ms, max: {results['warm_max_ms']}ms)")

    # Batch
    texts = [f"This is test sentence number {i}" for i in range(10)]
    start = time.time()
    embedder.embed_batch(texts)
    batch_time = (time.time() - start) * 1000
    results["batch_10_ms"] = round(batch_time, 1)
    results["batch_per_text_ms"] = round(batch_time / 10, 1)
    print(f"  Batch (10): {results['batch_10_ms']}ms ({results['batch_per_text_ms']}ms/text)")

    # Dimensions
    results["dimensions"] = embedder.embedding_dim
    print(f"  Dimensions: {results['dimensions']}")

    # Memory
    results["ram_mb"] = round(get_memory_usage(), 1)
    results["gpu_mb"] = round(get_gpu_memory(), 1)
    print(f"  RAM: {results['ram_mb']}MB | GPU: {results['gpu_mb']}MB")

    return results


def benchmark_recall(db_path=".mathir/mathir.db", num_queries=10):
    """Benchmark recall latency."""
    print(f"\n=== Recall Benchmark ===")
    results = {}

    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        # Check if table exists
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if not any("memory" in t[0] for t in tables):
            print("  No memories in database")
            return None

        times = []
        for i in range(num_queries):
            start = time.time()
            conn.execute("SELECT content FROM memory LIMIT 10").fetchall()
            times.append((time.time() - start) * 1000)

        avg_recall = sum(times) / len(times)
        results["recall_avg_ms"] = round(avg_recall, 1)
        results["recall_min_ms"] = round(min(times), 1)
        results["recall_max_ms"] = round(max(times), 1)
        print(f"  Recall avg: {results['recall_avg_ms']}ms")

        conn.close()
    except Exception as e:
        print(f"  Recall benchmark failed: {e}")
        return None

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MATHIR on edge devices")
    parser.add_argument("--provider", choices=["ollama", "onnx", "all"], default="all")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="nomic-embed-text")
    parser.add_argument("--onnx-model-dir", default=None)
    parser.add_argument("--num-embeddings", type=int, default=10)
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         MATHIR Edge Benchmark                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Provider: {args.provider} | Device: {args.device}")

    all_results = {}

    if args.provider in ["ollama", "all"]:
        results = benchmark_ollama(args.ollama_url, args.ollama_model, args.num_embeddings)
        if results:
            all_results["ollama"] = results

    if args.provider in ["onnx", "all"]:
        results = benchmark_onnx(args.onnx_model_dir, args.num_embeddings)
        if results:
            all_results["onnx"] = results

    # Recall benchmark
    recall = benchmark_recall()
    if recall:
        all_results["recall"] = recall

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Recommendation
    print("\n=== Recommendation ===")
    if "ollama" in all_results:
        avg_ms = all_results["ollama"]["warm_avg_ms"]
        if avg_ms < 50:
            print("  GPU mode recommended for production")
        elif avg_ms < 200:
            print("  CPU mode is OK for development")
        else:
            print("  Consider ONNX for better CPU performance")


if __name__ == "__main__":
    main()
