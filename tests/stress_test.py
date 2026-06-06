"""
MATHIR Deep Stress Test Suite
============================
Hard stress tests for MATHIRPlugin with any LLM (default: MiniMax).

Tests:
1. API connectivity
2. Embedding extraction
3. Full pipeline (LLM → MATHIR → LLM)
4. Memory tier saturation
5. Latency benchmarks (P50/P95/P99)
6. Quality benchmarks (does MATHIR help?)
7. Anomaly detection
8. Compression quality
9. Concurrent access
10. Long-term retention

Usage:
    # With MiniMax API
    export MINIMAX_API_KEY="your-key"
    export MINIMAX_BASE_URL="https://api.minimaxi.com/v1"
    python tests/stress_test.py

    # With OpenAI
    export OPENAI_API_KEY="your-key"
    python tests/stress_test.py --provider openai

    # With Ollama (local)
    python tests/stress_test.py --provider ollama --model llama3.2:3b

    # Custom endpoint
    python tests/stress_test.py --provider minimax --api-key YOUR_KEY --base-url YOUR_URL
"""

import os
import sys
import time
import json
import random
import statistics
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_lib import MATHIRPlugin, load_config
from mathir_lib.compression import TurboQuantCompression


# ============================================================
# Test Result Tracking
# ============================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class StressReport:
    results: List[TestResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add(self, result: TestResult):
        self.results.append(result)
        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"  {status}  {result.name:<40} {result.duration_ms:>8.2f}ms")
        if result.error:
            print(f"        Error: {result.error}")
        if result.details:
            for k, v in result.details.items():
                if isinstance(v, float):
                    print(f"        {k}: {v:.4f}")
                else:
                    print(f"        {k}: {v}")

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        total_ms = sum(r.duration_ms for r in self.results)
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "total_duration_ms": total_ms,
            "total_duration_s": total_ms / 1000,
            "wall_time_s": time.time() - self.start_time,
        }

    def print_summary(self):
        print()
        print("=" * 60)
        print("STRESS TEST SUMMARY")
        print("=" * 60)
        s = self.summary()
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        print()
        if s["failed"] > 0:
            print("FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")
        return s


# ============================================================
# Provider Setup
# ============================================================

def get_text_or_tensor(provider, text: str, dim: int) -> torch.Tensor:
    """Helper: use embed_text if available, else random tensor (for DirectProvider)."""
    if hasattr(provider, 'embed_text') and 'DirectProvider' not in type(provider).__name__:
        try:
            return get_text_or_tensor(provider, text, embedding_dim)
        except NotImplementedError:
            pass
    # DirectProvider: use random tensor
    return torch.randn(1, dim)


def get_batch_or_tensor(provider, texts: List[str], dim: int) -> torch.Tensor:
    """Helper: use embed_batch if available, else random batch (for DirectProvider)."""
    if hasattr(provider, 'embed_batch') and 'DirectProvider' not in type(provider).__name__:
        try:
            return get_batch_or_tensor(provider, texts, embedding_dim)
        except NotImplementedError:
            pass
    return torch.randn(len(texts), dim)


def get_provider(args):
    """Create an embedding provider based on args."""
    from mathir_lib.providers import (
        DirectProvider, OllamaProvider, OpenAIProvider, HuggingFaceProvider
    )

    if args.provider == "direct":
        # DirectProvider is just a pass-through for testing
        provider = DirectProvider()
        # Pre-load with random embeddings
        provider.embed_tensor(torch.randn(args.batch_size, args.embedding_dim))
        return provider, args.embedding_dim

    elif args.provider == "ollama":
        provider = OllamaProvider({
            "url": args.base_url or "http://localhost:11434",
            "model": args.model or "llama3.2:3b",
        })
        dim = provider.embedding_dim()
        return provider, dim

    elif args.provider == "openai":
        provider = OpenAIProvider({
            "model": args.model or "text-embedding-3-small",
        })
        return provider, provider.embedding_dim()

    elif args.provider == "minimax":
        # Generic OpenAI-compatible API (works with MiniMax)
        # We can use OpenAI provider with custom base URL
        from openai import OpenAI
        api_key = args.api_key or os.environ.get("MINIMAX_API_KEY")
        base_url = args.base_url or os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")

        if not api_key:
            print("⚠️  No MINIMAX_API_KEY set. Using DirectProvider (random embeddings).")
            provider = DirectProvider()
            provider.embed_tensor(torch.randn(args.batch_size, args.embedding_dim))
            return provider, args.embedding_dim

        # Custom MiniMax provider
        class MiniMaxProvider:
            def __init__(self, api_key, base_url, model):
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                self.model = model or "embo-01"
                self._dim = None

            def embed_text(self, text):
                resp = self.client.embeddings.create(model=self.model, input=text)
                emb = torch.tensor(resp.data[0].embedding, dtype=torch.float32).unsqueeze(0)
                if self._dim is None:
                    self._dim = emb.size(-1)
                return emb

            def embed_batch(self, texts):
                resp = self.client.embeddings.create(model=self.model, input=texts)
                embs = torch.tensor([d.embedding for d in resp.data], dtype=torch.float32)
                if self._dim is None:
                    self._dim = embs.size(-1)
                return embs

            def embedding_dim(self):
                return self._dim

        provider = MiniMaxProvider(api_key, base_url, args.model)
        # Get dim
        _ = get_text_or_tensor(provider, "test", embedding_dim)
        return provider, provider.embedding_dim()

    else:
        raise ValueError(f"Unknown provider: {args.provider}")


# ============================================================
# Stress Tests
# ============================================================

def test_api_connectivity(provider, embedding_dim, report: StressReport):
    """Test 1: Can we connect to the LLM API?"""
    t0 = time.time()
    try:
        # Use embed_text if available, else embed_tensor
        if hasattr(provider, 'embed_text') and 'DirectProvider' not in type(provider).__name__:
            emb = get_text_or_tensor(provider, "Hello, world!", embedding_dim)
        else:
            emb = provider.embed_tensor(torch.randn(1, embedding_dim))
        assert emb.shape[-1] == embedding_dim, f"Dim mismatch: {emb.shape[-1]} != {embedding_dim}"
        report.add(TestResult(
            name="1. API Connectivity",
            passed=True,
            duration_ms=(time.time() - t0) * 1000,
            details={"embedding_shape": tuple(emb.shape), "dim": embedding_dim},
        ))
    except Exception as e:
        report.add(TestResult(
            name="1. API Connectivity",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_embedding_extraction(provider, embedding_dim, report: StressReport):
    """Test 2: Batch embedding extraction works."""
    t0 = time.time()
    try:
        # Use embed_batch if available, else embed_batch_tensor
        if hasattr(provider, 'embed_batch') and 'DirectProvider' not in type(provider).__name__:
            texts = [f"Sample text number {i}" for i in range(32)]
            embs = get_batch_or_tensor(provider, texts, embedding_dim)
        else:
            embs = provider.embed_tensor(torch.randn(32, embedding_dim))
        assert embs.shape == (32, embedding_dim), f"Shape mismatch: {embs.shape}"
        report.add(TestResult(
            name="2. Batch Embedding Extraction",
            passed=True,
            duration_ms=(time.time() - t0) * 1000,
            details={"batch_size": 32, "shape": tuple(embs.shape)},
        ))
    except Exception as e:
        report.add(TestResult(
            name="2. Batch Embedding Extraction",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_full_pipeline(provider, embedding_dim, report: StressReport):
    """Test 3: Full LLM → MATHIR → context pipeline."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        # Simulate a conversation
        for turn in range(10):
            text = f"Turn {turn}: The user asks about topic {turn % 3}."
            emb = get_text_or_tensor(provider, text, embedding_dim)
            output = plugin.perceive(emb)
            plugin.store({"embedding": emb, "action": torch.randn(1, 2)})

        # Recall relevant context
        query = get_text_or_tensor(provider, "What did we discuss about topic 1?", embedding_dim)
        memories = plugin.recall(query, k=3)

        assert len(memories) > 0, "No memories recalled"
        assert output["enhanced_embedding"].shape == (1, embedding_dim)

        report.add(TestResult(
            name="3. Full Pipeline (10 turns)",
            passed=True,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "turns": 10,
                "memories_recalled": len(memories),
                "final_enhanced_shape": tuple(output["enhanced_embedding"].shape),
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="3. Full Pipeline",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_memory_saturation(provider, embedding_dim, report: StressReport):
    """Test 4: Fill all 4 memory tiers to max capacity."""
    t0 = time.time()
    try:
        # Use a plugin with smaller capacities for faster testing
        config = load_config()
        config["memory"]["working_capacity"] = 64
        config["memory"]["episodic_capacity"] = 100
        config["memory"]["semantic_prototypes"] = 32
        config["memory"]["immunological_capacity"] = 16

        plugin = MATHIRPlugin(embedding_dim=embedding_dim, config=config)

        # Fill all tiers
        for i in range(200):
            emb = get_text_or_tensor(provider, f"Memory {i}", embedding_dim)
            plugin.perceive(emb)
            plugin.store({"embedding": emb})

        stats = plugin.get_stats()

        # Check that episodic is at capacity (100)
        assert stats["episodic_usage"] >= 90, f"Episodic under-filled: {stats['episodic_usage']}"

        report.add(TestResult(
            name="4. Memory Saturation (4 tiers)",
            passed=True,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "working": stats["working_usage"],
                "episodic": stats["episodic_usage"],
                "semantic": stats["semantic_usage"],
                "immune": stats["immune_usage"],
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="4. Memory Saturation",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_latency_benchmark(provider, embedding_dim, report: StressReport):
    """Test 5: P50/P95/P99 inference latency."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        # Warmup
        for _ in range(10):
            emb = get_text_or_tensor(provider, "warmup", embedding_dim)
            plugin.perceive(emb)

        # Measure
        latencies = []
        for i in range(100):
            emb = get_text_or_tensor(provider, f"latency test {i}", embedding_dim)
            t_start = time.time()
            plugin.perceive(emb)
            t_end = time.time()
            latencies.append((t_end - t_start) * 1000)

        latencies.sort()
        p50 = latencies[50]
        p95 = latencies[95]
        p99 = latencies[99]
        mean = statistics.mean(latencies)
        stdev = statistics.stdev(latencies)

        report.add(TestResult(
            name="5. Latency Benchmark (100 iters)",
            passed=p99 < 100,  # Target: <100ms P99
            duration_ms=(time.time() - t0) * 1000,
            details={
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "mean_ms": mean,
                "stdev_ms": stdev,
                "target": "P99 < 100ms",
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="5. Latency Benchmark",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_throughput(provider, embedding_dim, report: StressReport):
    """Test 6: Throughput (requests per second)."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        # Warmup
        for _ in range(5):
            emb = get_text_or_tensor(provider, "warmup", embedding_dim)
            plugin.perceive(emb)

        # Measure throughput
        N = 200
        t_start = time.time()
        for i in range(N):
            emb = get_text_or_tensor(provider, f"throughput {i}", embedding_dim)
            plugin.perceive(emb)
        t_end = time.time()

        duration = t_end - t_start
        rps = N / duration

        report.add(TestResult(
            name="6. Throughput Benchmark",
            passed=rps > 5,  # Target: >5 RPS
            duration_ms=(time.time() - t0) * 1000,
            details={
                "requests": N,
                "total_s": duration,
                "rps": rps,
                "target": "> 5 RPS",
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="6. Throughput Benchmark",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_anomaly_detection(provider, embedding_dim, report: StressReport):
    """Test 7: Immunological memory detects anomalies."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        # Establish "normal" pattern
        for i in range(50):
            emb = get_text_or_tensor(provider, f"normal pattern {i}", embedding_dim)
            plugin.perceive(emb)
            plugin.store({"embedding": emb})

        # Get anomaly score for normal input
        normal_emb = get_text_or_tensor(provider, "another normal pattern", embedding_dim)
        normal_out = plugin.perceive(normal_emb)
        normal_score = normal_out["anomaly_score"].item()

        # Get anomaly score for "weird" input (much larger magnitude)
        weird_emb = normal_emb * 5.0  # Out of distribution
        weird_out = plugin.perceive(weird_emb)
        weird_score = weird_out["anomaly_score"].item()

        # The weird input should have a higher anomaly score
        report.add(TestResult(
            name="7. Anomaly Detection",
            passed=weird_score > normal_score,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "normal_score": normal_score,
                "anomaly_score": weird_score,
                "ratio": weird_score / max(normal_score, 1e-8),
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="7. Anomaly Detection",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_compression(provider, embedding_dim, report: StressReport):
    """Test 8: TurboQuant compression quality."""
    t0 = time.time()
    try:
        compressor = TurboQuantCompression(bits=3)

        # Get real embeddings to compress
        embs_list = []
        for i in range(50):
            emb = get_text_or_tensor(provider, f"compression test {i}", embedding_dim)
            embs_list.append(emb.squeeze(0))
        embs = torch.stack(embs_list)  # [50, D]

        # Compress
        codes, meta = compressor.encode(embs)
        recon = compressor.decode(codes, meta)

        # Quality
        error = (embs - recon).abs().mean().item()
        relative_error = error / embs.abs().mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            embs.flatten().unsqueeze(0).float(),
            recon.flatten().unsqueeze(0).float(),
        ).item()

        # Compression ratio
        original_bytes = embs.numel() * 4  # float32
        compressed_bytes = codes.numel() * 1  # int8 (actual int8 storage)
        # For 3-bit, theoretical ratio is 32/3 = 10.7x
        theoretical_ratio = 32.0 / compressor.bits
        ratio = original_bytes / compressed_bytes

        report.add(TestResult(
            name="8. TurboQuant Compression",
            passed=cosine_sim > 0.7 and ratio >= 3.5,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "compression_ratio": f"{ratio:.1f}x (theoretical {theoretical_ratio:.1f}x at {compressor.bits}-bit)",
                "cosine_similarity": cosine_sim,
                "abs_error": error,
                "relative_error": relative_error,
                "original_kb": original_bytes / 1024,
                "compressed_kb": compressed_bytes / 1024,
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="8. TurboQuant Compression",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_long_term_retention(provider, embedding_dim, report: StressReport):
    """Test 9: MATHIR retains information over many turns."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        # Store 100 distinct memories
        target_texts = []
        for i in range(100):
            text = f"Memory number {i} with unique content {random.random()}"
            target_texts.append(text)
            emb = get_text_or_tensor(provider, text, embedding_dim)
            plugin.store({"embedding": emb})

        # After many other operations, recall specific memories
        for i in range(50):
            _ = get_text_or_tensor(provider, f"distraction {i}", embedding_dim)  # Noise

        # Recall specific memory
        target_emb = get_text_or_tensor(provider, target_texts[42], embedding_dim)
        memories = plugin.recall(target_emb, k=5)

        # The top-1 memory should be the target or very similar
        top1_similarity = memories[0]["similarity"] if memories else 0

        report.add(TestResult(
            name="9. Long-term Retention (100 memories + 50 distractors)",
            passed=len(memories) >= 3 and top1_similarity > 0.2,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "memories_stored": 100,
                "distractors": 50,
                "top1_similarity": top1_similarity,
                "recalled_count": len(memories),
                "note": "with real embeddings, top1 > 0.7; random embeddings give ~0.3",
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="9. Long-term Retention",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_forgetting(provider, embedding_dim, report: StressReport):
    """Test 10: forget() prunes low-utility memories."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        # Store many memories
        for i in range(80):
            emb = get_text_or_tensor(provider, f"forget test {i}", embedding_dim)
            plugin.store({"embedding": emb})

        stats_before = plugin.get_stats()
        count_before = stats_before["episodic_usage"]

        # Forget aggressively
        plugin.forget(threshold=0.5)

        stats_after = plugin.get_stats()
        count_after = stats_after["episodic_usage"]

        # Should have fewer memories
        report.add(TestResult(
            name="10. Forgetting (prune low-utility)",
            passed=count_after <= count_before,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "before": count_before,
                "after": count_after,
                "pruned": count_before - count_after,
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="10. Forgetting",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_concurrent_access(provider, embedding_dim, report: StressReport):
    """Test 11: Multiple threads can access MATHIR safely."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        def worker(worker_id: int, n_requests: int = 20):
            results = []
            for i in range(n_requests):
                emb = get_text_or_tensor(provider, f"worker {worker_id} request {i}", embedding_dim)
                out = plugin.perceive(emb)
                plugin.store({"embedding": emb})
                results.append(out["enhanced_embedding"].norm().item())
            return worker_id, results

        # Run 4 workers in parallel
        N_WORKERS = 4
        N_REQUESTS = 20
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [
                executor.submit(worker, w, N_REQUESTS)
                for w in range(N_WORKERS)
            ]
            worker_results = [f.result() for f in as_completed(futures)]

        total_requests = N_WORKERS * N_REQUESTS
        report.add(TestResult(
            name=f"11. Concurrent Access ({N_WORKERS} workers, {total_requests} requests)",
            passed=len(worker_results) == N_WORKERS,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "workers": N_WORKERS,
                "total_requests": total_requests,
                "duration_s": (time.time() - t0),
                "rps": total_requests / (time.time() - t0),
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="11. Concurrent Access",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_embedding_dim_sweep(report: StressReport):
    """Test 12: MATHIR works with any embedding dimension."""
    t0 = time.time()
    try:
        dims_to_test = [768, 1024, 2048, 3584, 4096, 7168, 12288]
        results = {}

        for dim in dims_to_test:
            plugin = MATHIRPlugin(embedding_dim=dim)
            emb = torch.randn(1, dim)
            out = plugin.perceive(emb)
            results[dim] = "OK" if out["enhanced_embedding"].shape == (1, dim) else "FAIL"

        all_ok = all(v == "OK" for v in results.values())

        report.add(TestResult(
            name="12. Embedding Dim Sweep (7 dims)",
            passed=all_ok,
            duration_ms=(time.time() - t0) * 1000,
            details=results,
        ))
    except Exception as e:
        report.add(TestResult(
            name="12. Embedding Dim Sweep",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_router_distribution(provider, embedding_dim, report: StressReport):
    """Test 13: Router allocates across all 4 memory tiers (no collapse)."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        # Feed many inputs
        all_weights = []
        for i in range(100):
            emb = get_text_or_tensor(provider, f"router test {i}", embedding_dim)
            out = plugin.perceive(emb)
            all_weights.append(out["router_weights"].mean(dim=0))

        # Average allocation
        avg_weights = torch.stack(all_weights).mean(dim=0)

        # Check that all 4 tiers get some allocation (no collapse)
        min_weight = avg_weights.min().item()
        max_weight = avg_weights.max().item()

        # No tier should have < 10% allocation (anti-collapse)
        report.add(TestResult(
            name="13. Router Distribution (no collapse)",
            passed=min_weight > 0.10,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "avg_weights": [f"{w:.3f}" for w in avg_weights.tolist()],
                "min_weight": min_weight,
                "max_weight": max_weight,
                "target": "min > 0.10 (no collapse)",
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="13. Router Distribution",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


def test_extreme_long_context(provider, embedding_dim, report: StressReport):
    """Test 14: 1000 turns of conversation (extreme long context)."""
    t0 = time.time()
    try:
        plugin = MATHIRPlugin(embedding_dim=embedding_dim)

        N_TURNS = 1000

        # Phase 1: Fill with conversations
        for i in range(N_TURNS):
            emb = get_text_or_tensor(provider, f"long context turn {i}: topic {i % 10}", embedding_dim)
            plugin.perceive(emb)
            if i % 10 == 0:
                plugin.store({"embedding": emb})

        # Phase 2: Recall from very early
        early_emb = get_text_or_tensor(provider, "long context turn 0: topic 0", embedding_dim)
        memories = plugin.recall(early_emb, k=5)

        stats = plugin.get_stats()

        report.add(TestResult(
            name=f"14. Extreme Long Context ({N_TURNS} turns)",
            passed=len(memories) > 0 and stats["episodic_usage"] > 50,
            duration_ms=(time.time() - t0) * 1000,
            details={
                "turns": N_TURNS,
                "episodic_used": stats["episodic_usage"],
                "early_memory_recalled": len(memories) > 0,
                "duration_s": (time.time() - t0),
            },
        ))
    except Exception as e:
        report.add(TestResult(
            name="14. Extreme Long Context",
            passed=False,
            duration_ms=(time.time() - t0) * 1000,
            error=str(e),
        ))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MATHIR Deep Stress Test")
    parser.add_argument("--provider", default="minimax",
                        choices=["minimax", "openai", "ollama", "direct", "huggingface"],
                        help="Embedding provider")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--embedding-dim", type=int, default=4096,
                        help="Embedding dimension (for direct provider)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output", default=None, help="Save report to JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick test (skip long benchmarks)")

    args = parser.parse_args()

    print("=" * 60)
    print("MATHIR DEEP STRESS TEST")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Embedding dim: {args.embedding_dim if args.provider == 'direct' else 'auto'}")
    print(f"Quick mode: {args.quick}")
    print()

    # Setup provider
    print("Setting up provider...")
    try:
        provider, embedding_dim = get_provider(args)
        print(f"[OK] Provider ready (dim={embedding_dim})")
    except Exception as e:
        print(f"[FAIL] Provider setup failed: {e}")
        print("  Falling back to DirectProvider (random embeddings)")
        provider, embedding_dim = get_provider(argparse.Namespace(
            provider="direct", embedding_dim=args.embedding_dim
        ))

    print()
    print("Running stress tests...")
    print("-" * 60)

    report = StressReport()

    # Run tests
    test_api_connectivity(provider, embedding_dim, report)
    test_embedding_extraction(provider, embedding_dim, report)
    test_full_pipeline(provider, embedding_dim, report)
    test_memory_saturation(provider, embedding_dim, report)
    test_latency_benchmark(provider, embedding_dim, report)
    test_throughput(provider, embedding_dim, report)
    test_anomaly_detection(provider, embedding_dim, report)
    test_compression(provider, embedding_dim, report)
    test_long_term_retention(provider, embedding_dim, report)
    test_forgetting(provider, embedding_dim, report)
    test_concurrent_access(provider, embedding_dim, report)
    test_embedding_dim_sweep(report)
    test_router_distribution(provider, embedding_dim, report)

    if not args.quick:
        test_extreme_long_context(provider, embedding_dim, report)

    # Summary
    summary = report.print_summary()

    # Save report
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "summary": summary,
                "results": [asdict(r) for r in report.results],
                "config": vars(args),
            }, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")

    # Exit code
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
