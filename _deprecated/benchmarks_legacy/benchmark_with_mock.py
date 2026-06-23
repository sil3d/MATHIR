"""
Benchmark with semantic mock embeddings.

Uses a simple but realistic embedding strategy:
- TF-IDF-like vectors (word presence/absence)
- Scaled by text length
- This simulates what real embeddings look like:
  - Semantically similar texts have similar vectors
  - Dissimilar texts have different vectors
"""

import os
import sys
import time
import json
import random
import math
import re
import argparse
import statistics
from collections import Counter
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_lib import MATHIRPlugin
from mathir_lib.compression import TurboQuantCompression


# ============================================================
# Semantic Mock Provider
# ============================================================

class SemanticMockProvider:
    """
    Generates deterministic semantic embeddings from text.
    Similar texts → similar vectors.
    """

    def __init__(self, dim: int = 768, vocab_size: int = 10000):
        self.dim = dim
        self.vocab_size = vocab_size
        self._vocab = {}
        self._idf = {}
        self._build_vocab()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1]

    def _build_vocab(self):
        """Build a simple vocabulary and IDF weights."""
        # Use a fixed set of common English words
        common_words = [
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it',
            'name', 'live', 'work', 'favorite', 'programming', 'language', 'python',
            'cat', 'dog', 'pet', 'year', 'graduated', 'university', 'food', 'pizza',
            'play', 'guitar', 'blog', 'machine', 'learning', 'partner', 'finance',
            'hiking', 'alps', 'drive', 'tesla', 'model', 'books', 'fiction',
            'season', 'autumn', 'born', 'degree', 'statistics', 'memory', 'tier',
            'hierarchical', 'router', 'prevent', 'collapse', 'compression', 'gpu',
            'vram', 'consumer', 'benchmark', 'lstm', 'long', 'term', 'retention',
            'encoder', 'cnn', 'anomaly', 'distance', 'threshold', 'prototypes',
            'episodic', 'slots', 'cosine', 'similarity', 'turboquant', 'mhc',
            'sinkhorn', 'stability', 'algorithm', 'autonomous', 'driving',
            'vehicle', 'safety', 'neural', 'network', 'training', 'data',
            'paris', 'france', 'alice', 'bob', 'techcorp', 'mit', 'stanford',
            'day', 'park', 'gym', 'office', 'cafe', 'library', 'reading',
            'coding', 'meeting', 'workout', 'lunch', 'scientist', 'startup',
            'schrodinger', 'margherita', 'autumn', 'asimov', 'isaac',
            'computer', 'science', 'mathematics', 'california', 'newyork',
            'losangeles', 'sanfrancisco', 'london', 'tokyo', 'sydney',
            'mountain', 'beach', 'sunset', 'sunrise', 'morning', 'evening'
        ]
        for i, w in enumerate(common_words[:self.vocab_size]):
            self._vocab[w] = i
        # Random IDF
        for w, i in self._vocab.items():
            self._idf[i] = 1.0 + random.random() * 2.0

    def _text_to_vector(self, text: str) -> torch.Tensor:
        tokens = self._tokenize(text)
        if not tokens:
            return torch.zeros(1, self.dim)
        # TF-IDF
        tf = Counter(tokens)
        # Use hashing trick to project to fixed dim
        vec = torch.zeros(self.dim)
        for word, count in tf.items():
            if word in self._vocab:
                # Hash word to a dimension
                h = hash(word) % self.dim
                idf = self._idf.get(self._vocab[word], 1.0)
                vec[h] += (1.0 + math.log(count)) * idf
        # Normalize
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        # Add small random component for realism
        torch.manual_seed(hash(text) % 10000)
        vec = vec + torch.randn(self.dim) * 0.01
        vec = vec / vec.norm()
        return vec.unsqueeze(0)

    def embed_text(self, text: str) -> torch.Tensor:
        return self._text_to_vector(text)

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        return torch.cat([self._text_to_vector(t) for t in texts], dim=0)

    def embedding_dim(self) -> int:
        return self.dim


# ============================================================
# RAG and VectorDB (same as mathir_vs_rag.py)
# ============================================================

class SimpleRAG:
    def __init__(self, provider):
        self.provider = provider
        self.embeddings = []
        self.texts = []

    def store(self, text):
        emb = self.provider.embed_text(text)
        self.embeddings.append(emb)
        self.texts.append(text)

    def recall(self, query, k=3):
        if not self.embeddings:
            return []
        query_emb = self.provider.embed_text(query)
        sims = []
        for emb in self.embeddings:
            sim = torch.nn.functional.cosine_similarity(
                query_emb.squeeze(0), emb.squeeze(0), dim=-1
            ).item()
            sims.append(sim)
        top_k = sorted(enumerate(sims), key=lambda x: -x[1])[:k]
        return [(self.texts[i], s) for i, s in top_k]

    def store_batch(self, texts):
        embs = self.provider.embed_batch(texts)
        self.embeddings.extend([e.unsqueeze(0) for e in embs])
        self.texts.extend(texts)

    def size(self):
        return len(self.texts)


class SimpleVectorDB:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = None
        self.texts = []

    def add(self, text, embedding):
        emb_np = embedding.squeeze(0).numpy()
        if self.vectors is None:
            self.vectors = emb_np.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, emb_np])
        self.texts.append(text)

    def add_batch(self, texts, embeddings):
        embs_np = embeddings.numpy()
        if self.vectors is None:
            self.vectors = embs_np
        else:
            self.vectors = np.vstack([self.vectors, embs_np])
        self.texts.extend(texts)

    def search(self, query_emb, k=3):
        if self.vectors is None or len(self.vectors) == 0:
            return []
        q = query_emb.squeeze(0).numpy()
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q)
        sims = self.vectors @ q / (norms + 1e-8)
        top_k = np.argsort(-sims)[:k]
        return [(self.texts[i], float(sims[i])) for i in top_k]

    def size(self):
        return len(self.texts)


# ============================================================
# Scenarios
# ============================================================

@dataclass
class TestScenario:
    name: str
    description: str
    memories: List[str]
    queries: List[Dict[str, Any]]


def build_scenarios() -> List[TestScenario]:
    random.seed(42)

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
        {"text": "Where do I live?", "expected_topic": "Paris", "must_recall_index": 0},
        {"text": "What is my job?", "expected_topic": "data scientist", "must_recall_index": 1},
        {"text": "What is my pet's name?", "expected_topic": "Schrödinger", "must_recall_index": 3},
        {"text": "When did I graduate from MIT?", "expected_topic": "2020", "must_recall_index": 4},
        {"text": "What car do I drive?", "expected_topic": "Tesla", "must_recall_index": 10},
        {"text": "When was I born?", "expected_topic": "1995", "must_recall_index": 13},
        {"text": "What programming language do I like?", "expected_topic": "Python", "must_recall_index": 2},
    ]

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
        {"text": "What is MATHIR designed for?", "expected_topic": "autonomous driving", "must_recall_index": 0},
    ]

    activities = ['park', 'gym', 'office', 'cafe', 'library']
    actions = ['reading', 'coding', 'meeting', 'workout', 'lunch']
    seq_memories = [
        f"Day {i}: I went to the {random.choice(activities)} and did {random.choice(actions)}."
        for i in range(1, 31)
    ]

    seq_queries = [
        {"text": "What did I do on Day 1?", "expected_topic": "Day 1", "must_recall_index": 0},
        {"text": "What did I do on Day 15?", "expected_topic": "Day 15", "must_recall_index": 14},
        {"text": "What did I do on Day 30?", "expected_topic": "Day 30", "must_recall_index": 29},
    ]

    return [
        TestScenario("Personal Info Recall", "Factual recall of personal info", personal_memories, personal_queries),
        TestScenario("Project Knowledge", "Technical knowledge recall", project_memories, project_queries),
        TestScenario("Long Sequential", "Long-term sequential memory", seq_memories, seq_queries),
    ]


# ============================================================
# Runners
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
    details: List[Dict] = field(default_factory=list)


def run_mathir(scenario, provider, dim):
    plugin = MATHIRPlugin(embedding_dim=dim)
    t0 = time.time()
    for text in scenario.memories:
        emb = provider.embed_text(text)
        plugin.perceive(emb)
        plugin.store({"embedding": emb})
    store_time = (time.time() - t0) * 1000

    query_times = []
    correct = 0
    details = []
    for q in scenario.queries:
        t0 = time.time()
        query_emb = provider.embed_text(q["text"])
        out = plugin.perceive(query_emb)
        memories = plugin.recall(query_emb, k=3)
        recalled_idx = [m["index"] for m in memories]
        is_correct = q["must_recall_index"] in recalled_idx
        if is_correct:
            correct += 1
        query_times.append((time.time() - t0) * 1000)
        details.append({
            "query": q["text"],
            "expected_idx": q["must_recall_index"],
            "expected_text": scenario.memories[q["must_recall_index"]][:60],
            "recalled": recalled_idx,
            "correct": is_correct,
        })

    param_count = sum(p.numel() for p in plugin.parameters())
    buffer_count = sum(b.numel() for b in plugin.buffers())
    memory_mb = (param_count + buffer_count) * 4 / 1024 / 1024

    return ScenarioResult(
        name=scenario.name, system="MATHIR",
        store_time_ms=store_time,
        avg_query_time_ms=statistics.mean(query_times),
        correct_recalls=correct, total_queries=len(scenario.queries),
        accuracy=correct / len(scenario.queries),
        memory_mb=memory_mb, details=details,
    )


def run_rag(scenario, provider):
    rag = SimpleRAG(provider)
    t0 = time.time()
    rag.store_batch(scenario.memories)
    store_time = (time.time() - t0) * 1000

    query_times = []
    correct = 0
    details = []
    for q in scenario.queries:
        t0 = time.time()
        results = rag.recall(q["text"], k=3)
        target = scenario.memories[q["must_recall_index"]]
        is_correct = any(target == r[0] for r in results)
        if is_correct:
            correct += 1
        query_times.append((time.time() - t0) * 1000)
        details.append({
            "query": q["text"],
            "expected_text": target[:60],
            "top_results": [r[0][:60] for r in results],
            "correct": is_correct,
        })

    memory_mb = rag.size() * provider.embedding_dim() * 4 / 1024 / 1024

    return ScenarioResult(
        name=scenario.name, system="RAG",
        store_time_ms=store_time,
        avg_query_time_ms=statistics.mean(query_times),
        correct_recalls=correct, total_queries=len(scenario.queries),
        accuracy=correct / len(scenario.queries),
        memory_mb=memory_mb, details=details,
    )


def run_vectordb(scenario, provider, dim):
    vdb = SimpleVectorDB(dim)
    t0 = time.time()
    embs = provider.embed_batch(scenario.memories)
    vdb.add_batch(scenario.memories, embs)
    store_time = (time.time() - t0) * 1000

    query_times = []
    correct = 0
    details = []
    for q in scenario.queries:
        t0 = time.time()
        query_emb = provider.embed_text(q["text"])
        results = vdb.search(query_emb, k=3)
        target = scenario.memories[q["must_recall_index"]]
        is_correct = any(target == r[0] for r in results)
        if is_correct:
            correct += 1
        query_times.append((time.time() - t0) * 1000)
        details.append({
            "query": q["text"],
            "expected_text": target[:60],
            "top_results": [r[0][:60] for r in results],
            "correct": is_correct,
        })

    memory_mb = vdb.size() * dim * 4 / 1024 / 1024

    return ScenarioResult(
        name=scenario.name, system="VectorDB",
        store_time_ms=store_time,
        avg_query_time_ms=statistics.mean(query_times),
        correct_recalls=correct, total_queries=len(scenario.queries),
        accuracy=correct / len(scenario.queries),
        memory_mb=memory_mb, details=details,
    )


# ============================================================
# Report
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("MATHIR vs RAG vs VectorDB — Semantic Mock Benchmark")
    print("=" * 70)
    print(f"Using semantic mock embeddings (dim={args.dim})")
    print("This simulates real embeddings: similar texts -> similar vectors")
    print()

    provider = SemanticMockProvider(dim=args.dim)
    scenarios = build_scenarios()
    results = []

    for scenario in scenarios:
        print(f"Scenario: {scenario.name}")
        print(f"  Memories: {len(scenario.memories)}, Queries: {len(scenario.queries)}")

        # MATHIR
        t0 = time.time()
        r = run_mathir(scenario, provider, args.dim)
        print(f"  MATHIR:   acc={r.accuracy*100:.0f}% ({r.correct_recalls}/{r.total_queries}), {r.avg_query_time_ms:.1f}ms/q, {r.memory_mb:.2f}MB  [{time.time()-t0:.1f}s]")
        results.append(r)

        # RAG
        t0 = time.time()
        r = run_rag(scenario, provider)
        print(f"  RAG:      acc={r.accuracy*100:.0f}% ({r.correct_recalls}/{r.total_queries}), {r.avg_query_time_ms:.1f}ms/q, {r.memory_mb:.2f}MB  [{time.time()-t0:.1f}s]")
        results.append(r)

        # VectorDB
        t0 = time.time()
        r = run_vectordb(scenario, provider, args.dim)
        print(f"  VectorDB: acc={r.accuracy*100:.0f}% ({r.correct_recalls}/{r.total_queries}), {r.avg_query_time_ms:.1f}ms/q, {r.memory_mb:.2f}MB  [{time.time()-t0:.1f}s]")
        results.append(r)
        print()

    # Summary
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)
    for sys_name in ["MATHIR", "RAG", "VectorDB"]:
        sys_results = [r for r in results if r.system == sys_name]
        avg_acc = statistics.mean([r.accuracy for r in sys_results])
        total_correct = sum(r.correct_recalls for r in sys_results)
        total_queries = sum(r.total_queries for r in sys_results)
        avg_time = statistics.mean([r.avg_query_time_ms for r in sys_results])
        avg_mem = statistics.mean([r.memory_mb for r in sys_results])
        print(f"  {sys_name:<10}  Accuracy: {avg_acc*100:.1f}% ({total_correct}/{total_queries})  Query: {avg_time:.1f}ms  Memory: {avg_mem:.2f}MB")

    # Verdict
    print()
    m = statistics.mean([r.accuracy for r in results if r.system == "MATHIR"])
    r = statistics.mean([r.accuracy for r in results if r.system == "RAG"])
    v = statistics.mean([r.accuracy for r in results if r.system == "VectorDB"])
    print(f"  WINNER: {'MATHIR' if m > max(r,v) else 'RAG' if r > v else 'VectorDB'}  ({max(m,r,v)*100:.1f}% accuracy)")

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        print(f"\nSaved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
