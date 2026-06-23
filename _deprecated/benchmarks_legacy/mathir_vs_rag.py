"""
MATHIR vs RAG vs Vector DB — Benchmark Comparison
=================================================

Compares three memory approaches on identical tasks:
  1. MATHIR — adaptive memory plugin (learns online)
  2. RAG — retrieval-augmented generation (vector similarity)
  3. Vector DB — flat storage + cosine similarity

Uses your MiniMax API for embeddings (key passed via env var or CLI).

Usage:
    # With MiniMax API key in env var
    export MINIMAX_API_KEY="sk-..."
    export MINIMAX_BASE_URL="https://api.minimaxi.com/v1"
    python benchmarks/mathir_vs_rag.py

    # Or pass at runtime
    python benchmarks/mathir_vs_rag.py --api-key "sk-..." --base-url "https://api.minimaxi.com/v1"

    # Quick test (fewer queries)
    python benchmarks/mathir_vs_rag.py --quick

    # Save results
    python benchmarks/mathir_vs_rag.py --output results.json
"""

import os
import sys
import time
import json
import random
import argparse
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

import torch
import numpy as np
from openai import OpenAI

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_lib import MATHIRPlugin
from mathir_lib.compression import TurboQuantCompression


# ============================================================
# MiniMax Embedding Provider
# ============================================================

class MiniMaxProvider:
    """Minimal MiniMax API client for embeddings."""

    def __init__(self, api_key: str, base_url: str, model: str = "embo-01"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._dim = None

    def embed_text(self, text: str) -> torch.Tensor:
        resp = self.client.embeddings.create(model=self.model, input=text)
        emb = torch.tensor(resp.data[0].embedding, dtype=torch.float32).unsqueeze(0)
        if self._dim is None:
            self._dim = emb.size(-1)
        return emb

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        embs = torch.tensor([d.embedding for d in resp.data], dtype=torch.float32)
        if self._dim is None:
            self._dim = embs.size(-1)
        return embs

    def embedding_dim(self) -> int:
        return self._dim


# ============================================================
# RAG Implementation
# ============================================================

class SimpleRAG:
    """Standard RAG: vector storage + cosine similarity retrieval."""

    def __init__(self, provider: MiniMaxProvider):
        self.provider = provider
        self.embeddings = []  # List[Tensor]
        self.texts = []      # List[str]

    def store(self, text: str) -> None:
        emb = self.provider.embed_text(text)
        self.embeddings.append(emb)
        self.texts.append(text)

    def recall(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self.embeddings:
            return []
        query_emb = self.provider.embed_text(query)
        similarities = []
        for emb in self.embeddings:
            sim = torch.nn.functional.cosine_similarity(
                query_emb.squeeze(0), emb.squeeze(0), dim=-1
            ).item()
            similarities.append(sim)
        # Top-k
        top_k = sorted(enumerate(similarities), key=lambda x: -x[1])[:k]
        return [(self.texts[i], s) for i, s in top_k]

    def store_batch(self, texts: List[str]) -> None:
        embs = self.provider.embed_batch(texts)
        self.embeddings.extend([e.unsqueeze(0) for e in embs])
        self.texts.extend(texts)

    def size(self) -> int:
        return len(self.texts)


# ============================================================
# Vector DB Implementation (numpy-based)
# ============================================================

class SimpleVectorDB:
    """Flat vector storage with numpy for fast similarity."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None  # [N, D] numpy
        self.texts = []

    def add(self, text: str, embedding: torch.Tensor) -> None:
        emb_np = embedding.squeeze(0).numpy()
        if self.vectors is None:
            self.vectors = emb_np.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, emb_np])
        self.texts.append(text)

    def add_batch(self, texts: List[str], embeddings: torch.Tensor) -> None:
        embs_np = embeddings.numpy()
        if self.vectors is None:
            self.vectors = embs_np
        else:
            self.vectors = np.vstack([self.vectors, embs_np])
        self.texts.extend(texts)

    def search(self, query_emb: torch.Tensor, k: int = 3) -> List[Tuple[str, float]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []
        q = query_emb.squeeze(0).numpy()
        # Cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q)
        sims = self.vectors @ q / (norms + 1e-8)
        top_k = np.argsort(-sims)[:k]
        return [(self.texts[i], float(sims[i])) for i in top_k]

    def size(self) -> int:
        return len(self.texts)


# ============================================================
# Test Scenarios
# ============================================================

@dataclass
class TestScenario:
    """A memory test scenario."""
    name: str
    description: str
    memories: List[str]            # Stuff to remember
    queries: List[Dict[str, Any]]  # Questions to test
    # Each query: {"text": "...", "expected_topic": "...", "must_recall_index": int}


def build_scenarios() -> List[TestScenario]:
    """Build the test scenarios for benchmarking."""

    # Scenario 1: Personal info recall
    personal_memories = [
        "My name is Alice and I live in Paris, France.",
        "I work as a data scientist at a startup called TechCorp.",
        "My favorite programming language is Python.",
        "I have a cat named Schrödinger who is 3 years old.",
        "I graduated from MIT in 2020 with a degree in computer science.",
        "My favorite food is pizza, especially margherita.",
        "I play the guitar in my free time.",
        "I run a blog about machine learning and AI.",
        "My partner's name is Bob and they work in finance.",
        "I love hiking in the Alps during summer.",
        "I drive a Tesla Model 3.",
        "I read science fiction books, especially Isaac Asimov.",
        "My favorite season is autumn because of the colors.",
        "I was born on March 15, 1995.",
        "I have a master's degree in statistics from Stanford.",
    ]

    personal_queries = [
        {"text": "What is my name?", "expected_topic": "Alice", "must_recall_index": 0},
        {"text": "Where do I live?", "expected_topic": "Paris", "must_recall_index": 1},
        {"text": "What is my job?", "expected_topic": "data scientist", "must_recall_index": 1},
        {"text": "What is my pet's name?", "expected_topic": "Schrödinger", "must_recall_index": 3},
        {"text": "When did I graduate from MIT?", "expected_topic": "2020", "must_recall_index": 4},
        {"text": "What car do I drive?", "expected_topic": "Tesla", "must_recall_index": 10},
        {"text": "When was I born?", "expected_topic": "1995", "must_recall_index": 13},
    ]

    # Scenario 2: Project knowledge
    project_memories = [
        "MATHIR is a memory-augmented neural network for autonomous driving.",
        "It uses 3-tier hierarchical memory: working, episodic, semantic.",
        "The KL-constrained router prevents memory collapse.",
        "TurboQuant provides 10x compression with minimal quality loss.",
        "The system targets 8GB VRAM consumer GPUs.",
        "MATHIR beats LSTM in long-term retention benchmarks.",
        "The vision encoder is a CNN called QuantumVisionEncoder.",
        "The immunological memory detects anomalies via distance threshold.",
        "The semantic memory uses online k-means with 256 prototypes.",
        "The episodic memory has 1000 slots with cosine similarity retrieval.",
    ]

    project_queries = [
        {"text": "How many memory tiers does MATHIR have?", "expected_topic": "3-tier", "must_recall_index": 1},
        {"text": "What is TurboQuant?", "expected_topic": "compression", "must_recall_index": 3},
        {"text": "What is the KL router for?", "expected_topic": "prevent collapse", "must_recall_index": 2},
        {"text": "What does the semantic memory use?", "expected_topic": "k-means", "must_recall_index": 8},
    ]

    # Scenario 3: Sequential conversation (long-term)
    seq_memories = [f"Day {i}: I went to the {random.choice(['park', 'gym', 'office', 'cafe', 'library'])} and did {random.choice(['reading', 'coding', 'meeting', 'workout', 'lunch'])}." for i in range(1, 51)]

    seq_queries = [
        {"text": "What did I do on Day 1?", "expected_topic": "Day 1", "must_recall_index": 0},
        {"text": "What did I do on Day 25?", "expected_topic": "Day 25", "must_recall_index": 24},
        {"text": "What did I do on Day 50?", "expected_topic": "Day 50", "must_recall_index": 49},
    ]

    return [
        TestScenario(
            name="Personal Info Recall (15 memories, 7 queries)",
            description="Test factual recall of personal information",
            memories=personal_memories,
            queries=personal_queries,
        ),
        TestScenario(
            name="Project Knowledge (10 memories, 4 queries)",
            description="Test technical knowledge recall",
            memories=project_memories,
            queries=project_queries,
        ),
        TestScenario(
            name="Long Sequential (50 memories, 3 queries)",
            description="Test long-term sequential memory",
            memories=seq_memories,
            queries=seq_queries,
        ),
    ]


# ============================================================
# Benchmark Runners
# ============================================================

@dataclass
class ScenarioResult:
    name: str
    system: str
    store_time_ms: float
    avg_query_time_ms: float
    correct_recalls: int
    total_queries: int
    accuracy: float
    memory_mb: float
    details: Dict[str, Any] = field(default_factory=dict)


def run_mathir_scenario(scenario: TestScenario, provider: MiniMaxProvider, dim: int) -> ScenarioResult:
    """Test MATHIR on a scenario."""
    plugin = MATHIRPlugin(embedding_dim=dim)

    # Store memories
    t0 = time.time()
    for text in scenario.memories:
        emb = provider.embed_text(text)
        plugin.perceive(emb)
        plugin.store({"embedding": emb})
    store_time = (time.time() - t0) * 1000

    # Query
    query_times = []
    correct = 0
    details = []
    for q in scenario.queries:
        t0 = time.time()
        query_emb = provider.embed_text(q["text"])
        # Get enhanced context + recall
        out = plugin.perceive(query_emb)
        memories = plugin.recall(query_emb, k=3)
        # Check if correct memory was recalled
        recalled_indices = [m["index"] for m in memories]
        is_correct = q["must_recall_index"] in recalled_indices
        if is_correct:
            correct += 1
        query_time = (time.time() - t0) * 1000
        query_times.append(query_time)
        details.append({
            "query": q["text"],
            "expected_idx": q["must_recall_index"],
            "recalled": recalled_indices,
            "correct": is_correct,
            "time_ms": query_time,
        })

    # Memory size estimate
    param_count = sum(p.numel() for p in plugin.parameters())
    buffer_count = sum(b.numel() for b in plugin.buffers())
    memory_mb = (param_count + buffer_count) * 4 / 1024 / 1024

    return ScenarioResult(
        name=scenario.name,
        system="MATHIR",
        store_time_ms=store_time,
        avg_query_time_ms=statistics.mean(query_times) if query_times else 0,
        correct_recalls=correct,
        total_queries=len(scenario.queries),
        accuracy=correct / len(scenario.queries) if scenario.queries else 0,
        memory_mb=memory_mb,
        details={"queries": details},
    )


def run_rag_scenario(scenario: TestScenario, provider: MiniMaxProvider) -> ScenarioResult:
    """Test simple RAG on a scenario."""
    rag = SimpleRAG(provider)

    # Store memories (batch for speed)
    t0 = time.time()
    rag.store_batch(scenario.memories)
    store_time = (time.time() - t0) * 1000

    # Query
    query_times = []
    correct = 0
    details = []
    for q in scenario.queries:
        t0 = time.time()
        results = rag.recall(q["text"], k=3)
        # Check if the target text is in the top-3 results
        target_text = scenario.memories[q["must_recall_index"]]
        is_correct = any(target_text == r[0] for r in results)
        if is_correct:
            correct += 1
        query_time = (time.time() - t0) * 1000
        query_times.append(query_time)
        details.append({
            "query": q["text"],
            "expected_text": target_text[:50] + "...",
            "top_results": [r[0][:50] + "..." for r in results],
            "correct": is_correct,
            "time_ms": query_time,
        })

    # Memory size: just the stored vectors
    memory_mb = rag.size() * provider.embedding_dim() * 4 / 1024 / 1024

    return ScenarioResult(
        name=scenario.name,
        system="RAG",
        store_time_ms=store_time,
        avg_query_time_ms=statistics.mean(query_times) if query_times else 0,
        correct_recalls=correct,
        total_queries=len(scenario.queries),
        accuracy=correct / len(scenario.queries) if scenario.queries else 0,
        memory_mb=memory_mb,
        details={"queries": details},
    )


def run_vectordb_scenario(scenario: TestScenario, provider: MiniMaxProvider, dim: int) -> ScenarioResult:
    """Test flat vector DB on a scenario."""
    vdb = SimpleVectorDB(dim=dim)

    # Store memories (batch for speed)
    t0 = time.time()
    embs = provider.embed_batch(scenario.memories)
    vdb.add_batch(scenario.memories, embs)
    store_time = (time.time() - t0) * 1000

    # Query
    query_times = []
    correct = 0
    details = []
    for q in scenario.queries:
        t0 = time.time()
        query_emb = provider.embed_text(q["text"])
        results = vdb.search(query_emb, k=3)
        # Check if the target text is in the top-3 results
        target_text = scenario.memories[q["must_recall_index"]]
        is_correct = any(target_text == r[0] for r in results)
        if is_correct:
            correct += 1
        query_time = (time.time() - t0) * 1000
        query_times.append(query_time)
        details.append({
            "query": q["text"],
            "expected_text": target_text[:50] + "...",
            "top_results": [r[0][:50] + "..." for r in results],
            "correct": is_correct,
            "time_ms": query_time,
        })

    # Memory size
    memory_mb = vdb.size() * dim * 4 / 1024 / 1024

    return ScenarioResult(
        name=scenario.name,
        system="VectorDB",
        store_time_ms=store_time,
        avg_query_time_ms=statistics.mean(query_times) if query_times else 0,
        correct_recalls=correct,
        total_queries=len(scenario.queries),
        accuracy=correct / len(scenario.queries) if scenario.queries else 0,
        memory_mb=memory_mb,
        details={"queries": details},
    )


# ============================================================
# Report
# ============================================================

def print_report(results: List[ScenarioResult]):
    """Print a formatted comparison report."""
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS — MATHIR vs RAG vs Vector DB")
    print("=" * 80)

    # Group by scenario
    scenarios = {}
    for r in results:
        scenarios.setdefault(r.name, []).append(r)

    for scenario_name, scenario_results in scenarios.items():
        print()
        print(f"## {scenario_name}")
        print("-" * 80)
        print(f"{'System':<12} {'Accuracy':>10} {'Store(ms)':>12} {'Query(ms)':>12} {'Memory(MB)':>12}")
        print("-" * 80)
        for r in scenario_results:
            print(f"{r.system:<12} {r.accuracy*100:>9.1f}% {r.store_time_ms:>11.1f} {r.avg_query_time_ms:>11.1f} {r.memory_mb:>11.3f}")

    # Overall comparison
    print()
    print("=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)

    systems = ["MATHIR", "RAG", "VectorDB"]
    for sys in systems:
        sys_results = [r for r in results if r.system == sys]
        if not sys_results:
            continue
        avg_accuracy = statistics.mean([r.accuracy for r in sys_results])
        total_correct = sum(r.correct_recalls for r in sys_results)
        total_queries = sum(r.total_queries for r in sys_results)
        avg_query_ms = statistics.mean([r.avg_query_time_ms for r in sys_results])
        max_memory = max(r.memory_mb for r in sys_results)

        print(f"\n{sys}:")
        print(f"  Overall accuracy: {avg_accuracy*100:.1f}% ({total_correct}/{total_queries} correct)")
        print(f"  Avg query time: {avg_query_ms:.1f}ms")
        print(f"  Peak memory: {max_memory:.3f}MB")

    # Winner
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    mathir_results = [r for r in results if r.system == "MATHIR"]
    rag_results = [r for r in results if r.system == "RAG"]
    vdb_results = [r for r in results if r.system == "VectorDB"]

    if mathir_results and rag_results:
        m_acc = statistics.mean([r.accuracy for r in mathir_results])
        r_acc = statistics.mean([r.accuracy for r in rag_results])
        v_acc = statistics.mean([r.accuracy for r in vdb_results]) if vdb_results else 0
        m_time = statistics.mean([r.avg_query_time_ms for r in mathir_results])
        r_time = statistics.mean([r.avg_query_time_ms for r in rag_results])

        print(f"  Accuracy:  MATHIR={m_acc*100:.1f}%  RAG={r_acc*100:.1f}%  VDB={v_acc*100:.1f}%")
        print(f"  Latency:   MATHIR={m_time:.1f}ms  RAG={r_time:.1f}ms")
        print(f"  Winner:    {'MATHIR' if m_acc > max(r_acc, v_acc) else 'RAG' if r_acc > v_acc else 'VectorDB'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MATHIR vs RAG vs VectorDB benchmark")
    parser.add_argument("--api-key", default=None, help="MiniMax API key (or set MINIMAX_API_KEY env)")
    parser.add_argument("--base-url", default=None, help="API base URL (or set MINIMAX_BASE_URL env)")
    parser.add_argument("--model", default="embo-01", help="Embedding model name")
    parser.add_argument("--output", default=None, help="Save report to JSON file")
    parser.add_argument("--quick", action="store_true", help="Run only first scenario")

    args = parser.parse_args()

    # Get API key from args or env
    api_key = args.api_key or os.environ.get("MINIMAX_API_KEY")
    base_url = args.base_url or os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")

    if not api_key:
        print("ERROR: No API key provided.")
        print("Set MINIMAX_API_KEY env var or use --api-key argument")
        sys.exit(1)

    print("=" * 80)
    print("MATHIR vs RAG vs VectorDB — Benchmark")
    print("=" * 80)
    print(f"API: {base_url}")
    print(f"Model: {args.model}")
    print()

    # Setup provider
    print("[1/4] Connecting to MiniMax API...")
    try:
        provider = MiniMaxProvider(api_key=api_key, base_url=base_url, model=args.model)
        # Get dim
        test_emb = provider.embed_text("test")
        dim = provider.embedding_dim()
        print(f"      OK. Embedding dim: {dim}")
    except Exception as e:
        print(f"      FAILED: {e}")
        sys.exit(1)

    # Build scenarios
    scenarios = build_scenarios()
    if args.quick:
        scenarios = scenarios[:1]

    # Run benchmarks
    results = []
    for i, scenario in enumerate(scenarios):
        print()
        print(f"[{i+2}/{len(scenarios)+1}] Running scenario: {scenario.name}")
        print(f"      {len(scenario.memories)} memories, {len(scenario.queries)} queries")

        # MATHIR
        print("      [MATHIR]   ", end="", flush=True)
        t0 = time.time()
        mathir_result = run_mathir_scenario(scenario, provider, dim)
        print(f"acc={mathir_result.accuracy*100:.0f}% ({mathir_result.correct_recalls}/{mathir_result.total_queries}), {time.time()-t0:.1f}s")
        results.append(mathir_result)

        # RAG
        print("      [RAG]      ", end="", flush=True)
        t0 = time.time()
        rag_result = run_rag_scenario(scenario, provider)
        print(f"acc={rag_result.accuracy*100:.0f}% ({rag_result.correct_recalls}/{rag_result.total_queries}), {time.time()-t0:.1f}s")
        results.append(rag_result)

        # VectorDB
        print("      [VectorDB] ", end="", flush=True)
        t0 = time.time()
        vdb_result = run_vectordb_scenario(scenario, provider, dim)
        print(f"acc={vdb_result.accuracy*100:.0f}% ({vdb_result.correct_recalls}/{vdb_result.total_queries}), {time.time()-t0:.1f}s")
        results.append(vdb_result)

    # Report
    print_report(results)

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
