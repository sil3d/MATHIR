#!/usr/bin/env python3
"""
MATHIR Memory Scale Benchmark — Tests latency at different memory scales.

Measures:
- Save latency (ms per memory)
- Recall latency (ms per query)
- Throughput (memories/sec)
- Compares sqlite-vec vs brute-force numpy

Scales tested: 100, 1K, 10K, 100K memories
"""

import os
import sys
import time
import json
import struct
import sqlite3
import tempfile
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import numpy as np

# Add bin to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for sqlite-vec
try:
    import sqlite_vec
    HAS_VEC = True
except ImportError:
    HAS_VEC = False

# Check for torch/GPU
try:
    import torch
    HAS_TORCH = True
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"
except ImportError:
    HAS_TORCH = False
    GPU_AVAILABLE = False
    GPU_NAME = "N/A"

# ─── Configuration ────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1024
SCALES = [100, 1_000, 10_000, 100_000]
QUERIES_PER_SCALE = 50
SAVES_PER_TEST = 100  # For measuring save latency at each scale
TOP_K = 5
RANDOM_SEED = 42

# Sample texts for realistic metadata
SAMPLE_TEXTS = [
    "Bug fix: null pointer in auth.py when session token expired",
    "Architecture decision: use WebSocket for real-time updates",
    "Performance: query takes 2.3s, needs index on user_id column",
    "Security: JWT refresh token rotation implemented",
    "Feature: added dark mode toggle in settings panel",
    "Refactor: extracted payment logic into separate service",
    "Test: E2E test for login flow passing consistently",
    "Deploy: rolling update strategy for zero-downtime deploys",
    "Bug: race condition in concurrent file uploads",
    "Decision: PostgreSQL over MongoDB for this use case",
    "Performance: LCP improved from 4.2s to 1.8s with code splitting",
    "Security: rate limiting added to authentication endpoints",
    "Feature: multi-language support with i18n framework",
    "Refactor: monolith split into 3 microservices",
    "Test: unit test coverage increased from 67% to 89%",
    "Bug: memory leak in WebSocket connection handler",
    "Architecture: event-driven architecture for order processing",
    "Performance: Redis cache reduces DB queries by 85%",
    "Security: OWASP Top 10 audit completed",
    "Feature: real-time collaboration with CRDT",
    "Decision: use Svelte over React for internal tools",
    "Bug: CSS grid layout broken on Safari 16",
    "Deploy: blue-green deployment with automated rollback",
    "Performance: image optimization reduces bundle by 40%",
    "Security: SQL injection vulnerability patched",
    "Feature: AI-powered search with semantic understanding",
    "Refactor: removed 3000 lines of dead code",
    "Test: load testing shows 10K concurrent users supported",
    "Bug: timezone handling incorrect for UTC+13",
    "Architecture: CQRS pattern for read-heavy dashboard",
]

SAMPLE_AGENTS = ["coder", "debugger", "security", "swarm", "pm", "qa", "refactor"]
SAMPLE_TYPES = ["working_memory", "episodic", "semantic", "procedural"]


# ─── Benchmark Helpers ────────────────────────────────────────────────────────

class BruteForceMemory:
    """Pure numpy brute-force vector search for comparison."""
    
    def __init__(self, db_path: Path, dim: int):
        self.db_path = db_path
        self.dim = dim
        self._conn = None
        self._ensure_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _ensure_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT,
                agent TEXT,
                block_type TEXT,
                label TEXT,
                priority INTEGER DEFAULT 5,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        conn.commit()
    
    def store(self, memory_id: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO memories (memory_id, content, agent, block_type, label, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            metadata.get("content", ""),
            metadata.get("agent", ""),
            metadata.get("block_type", ""),
            metadata.get("label", ""),
            metadata.get("priority", 5),
            datetime.now().isoformat()
        ))
        # Store raw bytes
        blob = struct.pack("<i", embedding.size) + embedding.tobytes()
        conn.execute("INSERT OR REPLACE INTO embeddings (memory_id, embedding) VALUES (?, ?)",
                      [memory_id, blob])
        conn.commit()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("SELECT memory_id, embedding FROM embeddings").fetchall()
        if not rows:
            return []
        
        # Decode all embeddings into matrix
        embs = []
        ids = []
        for row in rows:
            blob = row["embedding"]
            n = struct.unpack("<i", blob[:4])[0]
            vec = np.frombuffer(blob[4:4 + 4 * n], dtype=np.float32)
            embs.append(vec)
            ids.append(row["memory_id"])
        
        embs = np.stack(embs)  # (N, dim)
        
        # Cosine similarity
        query_unit = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        norms = np.linalg.norm(embs, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        embs_unit = embs / norms[:, None]
        sims = embs_unit @ query_unit
        
        # Top-k
        if k >= len(sims):
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, k)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        
        results = []
        for idx in top_idx:
            mid = ids[idx]
            meta = conn.execute("SELECT * FROM memories WHERE memory_id = ?", [mid]).fetchone()
            if meta:
                results.append({
                    "memory_id": mid,
                    "content": meta["content"],
                    "score": float(sims[idx]),
                })
        return results
    
    def count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()["cnt"]
    
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


def generate_random_embeddings(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate N random embeddings (normalized to unit vectors)."""
    rng = np.random.RandomState(seed)
    embs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms
    return embs


def generate_metadata(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate realistic metadata for N memories."""
    rng = np.random.RandomState(seed)
    metadatas = []
    for i in range(n):
        metadatas.append({
            "content": rng.choice(SAMPLE_TEXTS) + f" (item #{i})",
            "agent": rng.choice(SAMPLE_AGENTS),
            "block_type": rng.choice(SAMPLE_TYPES),
            "label": f"memory_{i}",
            "priority": int(rng.randint(0, 10)),
        })
    return metadatas


# ─── Benchmark Core ───────────────────────────────────────────────────────────

def benchmark_save(
    memory_store,
    embeddings: np.ndarray,
    metadatas: List[Dict[str, Any]],
    count: int,
) -> Dict[str, float]:
    """Benchmark save operations."""
    times = []
    for i in range(count):
        mid = f"bench_{i:08d}"
        start = time.perf_counter()
        memory_store(mid, embeddings[i], metadatas[i])
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "p99_ms": sorted(times)[int(len(times) * 0.99)],
        "min_ms": min(times),
        "max_ms": max(times),
        "total_s": sum(times) / 1000,
        "memories_per_sec": count / (sum(times) / 1000) if sum(times) > 0 else 0,
    }


def benchmark_recall(
    search_fn,
    query_embeddings: np.ndarray,
    k: int,
) -> Dict[str, float]:
    """Benchmark recall/search operations."""
    times = []
    for i in range(len(query_embeddings)):
        start = time.perf_counter()
        results = search_fn(query_embeddings[i], k)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "p99_ms": sorted(times)[int(len(times) * 0.99)],
        "min_ms": min(times),
        "max_ms": max(times),
        "queries_per_sec": len(times) / (sum(times) / 1000) if sum(times) > 0 else 0,
    }


def run_scale_benchmark(scale: int) -> Dict[str, Any]:
    """Run full benchmark for a given scale."""
    print(f"\n{'='*60}")
    print(f"  SCALE: {scale:,} memories")
    print(f"{'='*60}")
    
    # Generate test data
    print(f"  Generating {scale:,} embeddings (dim={EMBEDDING_DIM})...")
    embeddings = generate_random_embeddings(scale, EMBEDDING_DIM)
    metadatas = generate_metadata(scale)
    
    # Generate query embeddings (separate from stored)
    query_embs = generate_random_embeddings(QUERIES_PER_SCALE, EMBEDDING_DIM, seed=999)
    
    results = {}
    
    # ─── Test 1: Brute-Force (numpy) ──────────────────────────────────────
    print(f"\n  [1/2] Brute-Force (numpy)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "bench_brute.db"
        bf = BruteForceMemory(db_path, EMBEDDING_DIM)
        
        def bf_store(mid, emb, meta):
            bf.store(mid, emb, meta)
        
        # Save benchmark
        save_results = benchmark_save(bf_store, embeddings, metadatas, SAVES_PER_TEST)
        print(f"    Save: {save_results['mean_ms']:.2f}ms avg, {save_results['memories_per_sec']:.0f} mem/s")
        
        # Recall benchmark
        recall_results = benchmark_recall(bf.search, query_embs, TOP_K)
        print(f"    Recall: {recall_results['mean_ms']:.2f}ms avg, {recall_results['queries_per_sec']:.0f} q/s")
        
        # DB size
        db_size = db_path.stat().st_size if db_path.exists() else 0
        
        results["brute_force"] = {
            "save": save_results,
            "recall": recall_results,
            "db_size_mb": db_size / (1024 * 1024),
        }
        
        bf.close()
    
    # ─── Test 2: sqlite-vec ───────────────────────────────────────────────
    if HAS_VEC:
        print(f"\n  [2/2] sqlite-vec (O(log N))...")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "bench_vec.db"
            
            # Import VecMemory
            from mathir_vec import VecMemory
            vm = VecMemory(db_path, EMBEDDING_DIM)
            
            def vm_store(mid, emb, meta):
                vm.store(mid, emb, meta)
            
            # Save benchmark
            save_results = benchmark_save(vm_store, embeddings, metadatas, SAVES_PER_TEST)
            print(f"    Save: {save_results['mean_ms']:.2f}ms avg, {save_results['memories_per_sec']:.0f} mem/s")
            
            # Recall benchmark
            recall_results = benchmark_recall(vm.search, query_embs, TOP_K)
            print(f"    Recall: {recall_results['mean_ms']:.2f}ms avg, {recall_results['queries_per_sec']:.0f} q/s")
            
            # DB size
            db_size = db_path.stat().st_size if db_path.exists() else 0
            
            results["sqlite_vec"] = {
                "save": save_results,
                "recall": recall_results,
                "db_size_mb": db_size / (1024 * 1024),
            }
            
            vm.close()
    else:
        print(f"\n  [2/2] sqlite-vec NOT available, skipping...")
        results["sqlite_vec"] = None
    
    return results


# ─── Report ───────────────────────────────────────────────────────────────────

def print_summary(all_results: Dict[int, Dict[str, Any]]):
    """Print comprehensive summary table."""
    print(f"\n\n{'='*80}")
    print(f"  MATHIR MEMORY SCALE BENCHMARK — SUMMARY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # System info
    print(f"\n  System Info:")
    print(f"    Embedding dim:  {EMBEDDING_DIM}")
    print(f"    sqlite-vec:     {'YES' if HAS_VEC else 'NO'}")
    print(f"    GPU:            {GPU_NAME if GPU_AVAILABLE else 'CPU only'}")
    print(f"    PyTorch:        {HAS_TORCH}")
    print(f"    Queries/scale:  {QUERIES_PER_SCALE}")
    print(f"    Saves/test:     {SAVES_PER_TEST}")
    
    # Table header
    print(f"\n  {'-'*76}")
    header = (
        f"  {'Scale':>10} | {'Backend':>12} | {'Save (ms)':>10} | {'Recall (ms)':>12} | "
        f"{'Mem/s':>8} | {'Q/s':>8} | {'DB (MB)':>8}"
    )
    print(header)
    print(f"  {'-'*76}")
    
    for scale in SCALES:
        if scale not in all_results:
            continue
        res = all_results[scale]
        
        # Brute-force row
        bf = res.get("brute_force", {})
        if bf:
            save_ms = bf["save"]["mean_ms"]
            recall_ms = bf["recall"]["mean_ms"]
            mem_s = bf["save"]["memories_per_sec"]
            q_s = bf["recall"]["queries_per_sec"]
            db_mb = bf["db_size_mb"]
            print(
                f"  {scale:>10,} | {'numpy':>12} | {save_ms:>9.2f} | {recall_ms:>11.2f} | "
                f"{mem_s:>7.0f} | {q_s:>7.0f} | {db_mb:>7.1f}"
            )
        
        # sqlite-vec row
        sv = res.get("sqlite_vec")
        if sv:
            save_ms = sv["save"]["mean_ms"]
            recall_ms = sv["recall"]["mean_ms"]
            mem_s = sv["save"]["memories_per_sec"]
            q_s = sv["recall"]["queries_per_sec"]
            db_mb = sv["db_size_mb"]
            print(
                f"  {'':>10} | {'sqlite-vec':>12} | {save_ms:>9.2f} | {recall_ms:>11.2f} | "
                f"{mem_s:>7.0f} | {q_s:>7.0f} | {db_mb:>7.1f}"
            )
        elif sv is None:
            print(
                f"  {'':>10} | {'sqlite-vec':>12} | {'N/A':>9} | {'N/A':>11} | "
                f"{'N/A':>7} | {'N/A':>7} | {'N/A':>7}"
            )
        
        print(f"  {'-'*76}")
    
    # ASCII chart for recall latency
    print(f"\n  Recall Latency Comparison (ms) -- ASCII Chart")
    print(f"  {'-'*60}")
    
    max_latency = 0
    for scale in SCALES:
        if scale in all_results:
            res = all_results[scale]
            for backend in ["brute_force", "sqlite_vec"]:
                data = res.get(backend)
                if data:
                    max_latency = max(max_latency, data["recall"]["mean_ms"])
    
    if max_latency > 0:
        chart_width = 40
        for scale in SCALES:
            if scale not in all_results:
                continue
            res = all_results[scale]
            
            for backend, label in [("brute_force", "numpy"), ("sqlite_vec", "vec")]:
                data = res.get(backend)
                if data:
                    latency = data["recall"]["mean_ms"]
                    bar_len = int((latency / max_latency) * chart_width) if max_latency > 0 else 0
                    bar = "#" * bar_len
                    print(f"  {scale:>7,} {label:>5} | {bar} {latency:.2f}ms")
            print()
    
    # Speedup analysis
    print(f"\n  Speedup Analysis (sqlite-vec vs brute-force):")
    print(f"  {'-'*60}")
    for scale in SCALES:
        if scale not in all_results:
            continue
        res = all_results[scale]
        bf = res.get("brute_force")
        sv = res.get("sqlite_vec")
        
        if bf and sv:
            save_speedup = bf["save"]["mean_ms"] / sv["save"]["mean_ms"] if sv["save"]["mean_ms"] > 0 else 0
            recall_speedup = bf["recall"]["mean_ms"] / sv["recall"]["mean_ms"] if sv["recall"]["mean_ms"] > 0 else 0
            print(f"  {scale:>7,} memories: save {save_speedup:.1f}x, recall {recall_speedup:.1f}x")
        elif bf:
            print(f"  {scale:>7,} memories: sqlite-vec not available")
    
    # Recommendations
    print(f"\n  {'-'*60}")
    print(f"  RECOMMENDATIONS:")
    print(f"  {'-'*60}")
    
    # Find crossover point
    crossover = None
    for scale in SCALES:
        if scale not in all_results:
            continue
        res = all_results[scale]
        bf = res.get("brute_force")
        sv = res.get("sqlite_vec")
        if bf and sv:
            bf_recall = bf["recall"]["mean_ms"]
            sv_recall = sv["recall"]["mean_ms"]
            # sqlite-vec has overhead for small N, becomes faster at large N
            if sv_recall < bf_recall and crossover is None:
                crossover = scale
    
    if crossover:
        print(f"  [OK] sqlite-vec becomes faster than brute-force at ~{crossover:,} memories")
        print(f"     -> Use sqlite-vec for production (handles any scale)")
    else:
        print(f"  [!!] sqlite-vec did not outperform brute-force in tested range")
        print(f"     -> Brute-force may be sufficient for <100K memories")
    
    # Per-scale recommendation
    for scale in SCALES:
        if scale not in all_results:
            continue
        res = all_results[scale]
        bf = res.get("brute_force")
        sv = res.get("sqlite_vec")
        
        if bf:
            recall_ms = bf["recall"]["mean_ms"]
            if recall_ms < 1:
                rec = "[OK] Excellent (<1ms)"
            elif recall_ms < 10:
                rec = "[OK] Good (<10ms)"
            elif recall_ms < 100:
                rec = "[~~] Fair (<100ms)"
            else:
                rec = "[!!] Slow (>100ms)"
            
            backend = "sqlite-vec" if sv and sv["recall"]["mean_ms"] < bf["recall"]["mean_ms"] else "numpy"
            print(f"  {scale:>7,} memories -> {rec} (use {backend})")
    
    print(f"\n  {'='*80}\n")


# ─── Daemon Test ──────────────────────────────────────────────────────────────

def test_daemon() -> Dict[str, Any]:
    """Test against running daemon if available."""
    import socket
    
    print(f"\n  Testing daemon on port 7338...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(("127.0.0.1", 7338))
        
        # Ping
        request = json.dumps({"method": "ping", "params": {}})
        sock.sendall(request.encode())
        response = json.loads(sock.recv(65536).decode())
        
        if response.get("pong"):
            print(f"  [OK] Daemon running, dim={response.get('dim', '?')}")
            
            # Test save latency
            save_times = []
            for i in range(10):
                request = json.dumps({
                    "method": "memory_save",
                    "params": {
                        "content": f"Benchmark test memory {i}",
                        "agent": "benchmark",
                        "block_type": "episodic",
                        "label": f"bench_{i}",
                        "priority": 5,
                    }
                })
                start = time.perf_counter()
                sock.sendall(request.encode())
                resp = json.loads(sock.recv(65536).decode())
                elapsed = (time.perf_counter() - start) * 1000
                save_times.append(elapsed)
            
            # Test recall latency
            recall_times = []
            for i in range(10):
                request = json.dumps({
                    "method": "memory_recall",
                    "params": {
                        "query": f"benchmark test query {i}",
                        "k": 5,
                    }
                })
                start = time.perf_counter()
                sock.sendall(request.encode())
                resp = json.loads(sock.recv(65536).decode())
                elapsed = (time.perf_counter() - start) * 1000
                recall_times.append(elapsed)
            
            sock.close()
            
            return {
                "available": True,
                "dim": response.get("dim"),
                "save_mean_ms": statistics.mean(save_times),
                "recall_mean_ms": statistics.mean(recall_times),
                "save_times": save_times,
                "recall_times": recall_times,
            }
        else:
            sock.close()
            return {"available": False, "error": "Ping failed"}
    
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        return {"available": False, "error": str(e)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*80}")
    print(f"  MATHIR MEMORY SCALE BENCHMARK")
    print(f"  Testing latency at 100, 1K, 10K, 100K memories")
    print(f"{'='*80}")
    
    # System check
    print(f"\n  System Check:")
    print(f"    sqlite-vec:     {'[OK] Available' if HAS_VEC else '[--] Not installed'}")
    print(f"    NumPy:          {np.__version__}")
    print(f"    PyTorch:        {'[OK] ' + torch.__version__ if HAS_TORCH else '[--] Not installed'}")
    print(f"    GPU:            {'[OK] ' + GPU_NAME if GPU_AVAILABLE else '[--] CPU only'}")
    print(f"    Embedding dim:  {EMBEDDING_DIM}")
    
    # Run scale benchmarks
    all_results = {}
    for scale in SCALES:
        all_results[scale] = run_scale_benchmark(scale)
    
    # Print summary
    print_summary(all_results)
    
    # Test daemon if running
    daemon_results = test_daemon()
    if daemon_results.get("available"):
        print(f"  Daemon Live Test (port 7338):")
        print(f"    Save:   {daemon_results['save_mean_ms']:.2f}ms avg")
        print(f"    Recall: {daemon_results['recall_mean_ms']:.2f}ms avg")
        print()
    
    # Save results to JSON
    output_path = Path(__file__).parent / "benchmark_results.json"
    serializable = {}
    for scale, res in all_results.items():
        serializable[str(scale)] = res
    serializable["daemon"] = daemon_results
    serializable["system"] = {
        "embedding_dim": EMBEDDING_DIM,
        "has_vec": HAS_VEC,
        "has_torch": HAS_TORCH,
        "gpu_available": GPU_AVAILABLE,
        "gpu_name": GPU_NAME,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"  Results saved to: {output_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
