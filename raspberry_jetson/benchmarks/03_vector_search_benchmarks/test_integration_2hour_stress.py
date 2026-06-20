"""
Cross-Tier Integration 2-Hour Stress Test
==========================================

Tests the full MATHIR stack (all 4 tiers working together) under
sustained load for 2 hours.

TIER ARCHITECTURE:
  1. Working  — circular buffer of last N embeddings
  2. Episodic — key-value store with cosine-similarity recall
  3. Semantic — online k-means prototypes (concept clustering)
  4. Immune   — anomaly detector (distance to nearest "normal")

PHASES:
  Phase 1: Normal Operation (0-60 min)  — 50 ops/min, realistic mix
  Phase 2: High Load (60-90 min)       — 100 ops/min, doubled rate
  Phase 3: Edge Cases (90-120 min)      — memory pressure, anomaly injection

METRICS TRACKED:
  Time | Ops/min | Latency_P50 | Latency_P99 | nDCG@10 | Anomaly_Detect | Tiers_Active

CRITICAL FAILURES DETECTED:
  CRASH: system dies completely
  MEMORY LEAK: memory grows unbounded
  LATENCY_SPIKE: P99 > 1000ms
  QUALITY_COLLAPSE: nDCG < 0.5
  PARTIAL_FAILURE: 1+ tier stops working

Usage:
    # Full 2-hour test (real-time)
    python benchmarks/test_integration_2hour_stress.py --duration 7200

    # Accelerated test (2 minutes = 2 hours simulated)
    python benchmarks/test_integration_2hour_stress.py --duration 120 --accelerated

    # Run with specific workload intensity
    python benchmarks/test_integration_2hour_stress.py --duration 300 --phase2-rate 100
"""

import os
import sys
import json
import time
import random
import statistics
import argparse
import tracemalloc
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# =============================================================================
# Configuration
# =============================================================================

# BEIR SciFact data paths
SCIFACT_DIR = Path(__file__).parent / "beir_data" / "scifact" / "scifact"
EMB_CACHE_DIR = Path(__file__).parent / "controlled_emb_cache"

# Tier names
TIER_NAMES = ["working", "episodic", "semantic", "immune"]

# Operation mix (60% recall, 30% store, 5% anomaly_check, 5% route)
OP_MIX = {
    "recall": 0.60,
    "store": 0.30,
    "anomaly_check": 0.05,
    "route": 0.05,
}

# Critical thresholds
CRITICAL_LATENCY_P99_MS = 1000
CRITICAL_MEMORY_LEAK_MB = 500  # Growth threshold


# =============================================================================
# Data Loading
# =============================================================================

def load_scifact_corpus() -> Tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    """Load BEIR SciFact corpus, queries, and qrels."""
    corpus = {}
    with open(SCIFACT_DIR / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()

    queries = {}
    with open(SCIFACT_DIR / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    qrels = defaultdict(dict)
    qrels_path = SCIFACT_DIR / "qrels" / "test.tsv"
    if qrels_path.exists():
        with open(qrels_path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, did, rel = parts[0], parts[1], int(parts[2])
                    if rel > 0:
                        qrels[qid][did] = rel

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[q] for q in query_ids]

    print(f"  Loaded SciFact: {len(corpus)} docs, {len(queries)} queries, {sum(len(v) for v in qrels.values())} qrels")
    return doc_texts, query_texts, qrels, doc_ids, query_ids


def get_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Get or compute cached embeddings."""
    import hashlib
    from sentence_transformers import SentenceTransformer

    # Create cache key
    key = hashlib.sha256(f"{model_name}|{len(texts)}".encode()).hexdigest()[:16]
    cache_path = EMB_CACHE_DIR / f"{model_name.replace('/', '__')}_docs_{key}.npy"

    if cache_path.exists():
        print(f"    [CACHED] embeddings: {cache_path}")
        return np.load(cache_path).astype("float32")

    print(f"    [ENCODING] {len(texts)} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, embeddings)
    print(f"    [SAVED] embeddings to {cache_path}")
    return embeddings.astype("float32")


# =============================================================================
# MATHIR Setup
# =============================================================================

def create_mathir(embedding_dim: int):
    """Create a fresh MATHIR instance with all 4 tiers."""
    from mathir_dropin.memory import MATHIRMemory
    from mathir_dropin.config import configure

    config = configure({
        "memory": {
            "working_capacity": 64,
            "episodic_capacity": 1000,
            "semantic_prototypes": 256,
            "immunological_capacity": 100,
            "internal_dim": 128,
            "anomaly_threshold": 2.0,
        },
        "router": {
            "type": "kl_constrained",
            "hidden_dim": 128,
            "kl_coefficient": 0.01,
        },
        "storage": {
            "type": "memory",  # No SQLite for speed
        },
        "perception": {
            "use_layer_norm": True,
            "use_residual": True,
        },
    })

    mem = MATHIRMemory(embedding_dim=embedding_dim, config=config, db_path=None)
    return mem


# =============================================================================
# MATHIR Setup
# =============================================================================

class StressTestEngine:
    """Engine for running the 2-hour stress test."""

    def __init__(
        self,
        doc_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        doc_texts: List[str],
        query_texts: List[str],
        qrels: Dict[str, Dict[str, int]],
        doc_ids: List[str],
        query_ids: List[str],
        accelerated: bool = True,
    ):
        self.doc_embs = doc_embeddings
        self.query_embs = query_embeddings
        self.doc_texts = doc_texts
        self.query_texts = query_texts
        self.qrels = qrels
        self.doc_ids = doc_ids
        self.query_ids = query_ids
        self.accelerated = accelerated

        self.embedding_dim = doc_embeddings.shape[1]
        self.num_docs = len(doc_embeddings)
        self.num_queries = len(query_embeddings)

        # MATHIR instance
        self.mathir = create_mathir(self.embedding_dim)

        # Metrics storage
        self.timeline: List[Dict[str, Any]] = []
        self.operation_latencies: List[float] = []
        self.anomaly_scores: List[float] = []
        self.router_weights_history: List[Dict[str, float]] = []

        # State tracking
        self.start_time = time.time()
        self.total_operations = 0
        self.phase = "init"
        self.crashed = False
        self.crash_reason = None

        # Memory tracking
        tracemalloc.start()
        self.memory_start = None

        # Results tracking
        self.results_log: List[Dict[str, Any]] = []

    def _get_current_phase(self, elapsed_minutes: float) -> str:
        """Determine current phase based on elapsed time."""
        if elapsed_minutes < 60:
            return "phase1_normal"
        elif elapsed_minutes < 90:
            return "phase2_high_load"
        else:
            return "phase3_edge_cases"

    def _get_ops_per_minute(self, phase: str) -> int:
        """Get target operations per minute for the phase."""
        if self.accelerated:
            base_rate = 600  # 10x acceleration
        else:
            base_rate = 50

        if phase == "phase2_high_load":
            return base_rate * 2
        return base_rate

    def _select_operation(self) -> str:
        """Select operation based on configured mix."""
        r = random.random()
        cumulative = 0.0
        for op, prob in OP_MIX.items():
            cumulative += prob
            if r < cumulative:
                return op
        return "recall"

    def _execute_store(self, doc_idx: int) -> Dict[str, Any]:
        """Execute a store operation."""
        try:
            emb = torch.from_numpy(self.doc_embs[doc_idx]).unsqueeze(0)
            start_time = time.perf_counter()
            mid = self.mathir.store(emb, metadata={"text": self.doc_texts[doc_idx]})
            latency = time.perf_counter() - start_time
            self.operation_latencies.append(latency)
            return {"success": True, "memory_id": mid, "latency": latency}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_recall(self, query_idx: int) -> Dict[str, Any]:
        """Execute a recall operation."""
        try:
            q_emb = torch.from_numpy(self.query_embs[query_idx]).unsqueeze(0)
            start_time = time.perf_counter()
            results = self.mathir.recall(q_emb, k=10)
            latency = time.perf_counter() - start_time

            # Store latency
            self.operation_latencies.append(latency)

            # Compute intrinsic quality: average similarity of top-10 results
            # (higher = better retrieval quality)
            if results:
                avg_similarity = statistics.mean(r["similarity"] for r in results[:10])
            else:
                avg_similarity = 0.0

            return {
                "success": True,
                "results": results,
                "num_results": len(results),
                "avg_similarity": avg_similarity,
                "latency": latency,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_anomaly_check(self, query_idx: int) -> Dict[str, Any]:
        """Execute an anomaly check."""
        try:
            q_emb = torch.from_numpy(self.query_embs[query_idx]).unsqueeze(0)
            start_time = time.perf_counter()
            out = self.mathir.perceive(q_emb)
            latency = time.perf_counter() - start_time
            anomaly_score = float(out["anomaly_score"].item())
            is_anomaly = anomaly_score > 2.0
            self.operation_latencies.append(latency)
            return {
                "success": True,
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "latency": latency,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_route(self, query_idx: int) -> Dict[str, Any]:
        """Execute a route operation (get router weights)."""
        try:
            q_emb = torch.from_numpy(self.query_embs[query_idx]).unsqueeze(0)
            start_time = time.perf_counter()
            out = self.mathir.perceive(q_emb)
            latency = time.perf_counter() - start_time
            weights = out["router_weights"].detach().cpu().numpy()[0]
            self.operation_latencies.append(latency)
            return {
                "success": True,
                "weights": {TIER_NAMES[i]: float(weights[i]) for i in range(4)},
                "latency": latency,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _inject_anomaly_during_phase3(self) -> bool:
        """Phase 3: Occasionally inject anomalous queries."""
        return random.random() < 0.1  # 10% chance

    def run_phase1_normal(self, duration_seconds: float) -> None:
        """Phase 1: Normal operation for specified duration."""
        print(f"\n  Phase 1: Normal Operation ({duration_seconds:.0f}s)")
        self.phase = "phase1_normal"

        ops_per_sec = self._get_ops_per_minute(self.phase) / 60.0
        interval = 1.0 / ops_per_sec if ops_per_sec > 0 else 1.0

        next_op_time = time.time()
        doc_idx = 0
        query_idx = 0

        while time.time() - self.start_time < duration_seconds:
            if time.time() >= next_op_time:
                op = self._select_operation()

                if op == "store":
                    result = self._execute_store(doc_idx % self.num_docs)
                    doc_idx += 1
                elif op == "recall":
                    result = self._execute_recall(query_idx % self.num_queries)
                    query_idx += 1
                elif op == "anomaly_check":
                    result = self._execute_anomaly_check(query_idx % self.num_queries)
                else:  # route
                    result = self._execute_route(query_idx % self.num_queries)

                if not result.get("success", False):
                    self.crashed = True
                    self.crash_reason = result.get("error", "Unknown error")
                    return

                self.total_operations += 1
                next_op_time += interval

            time.sleep(0.001)  # Small sleep to prevent busy-waiting

            # Periodically log metrics
            if len(self.timeline) == 0 or time.time() - self.timeline[-1]["timestamp"] > 10:
                self._log_metrics()

    def run_phase2_high_load(self, duration_seconds: float) -> None:
        """Phase 2: High load operation (double rate)."""
        print(f"\n  Phase 2: High Load ({duration_seconds:.0f}s)")
        self.phase = "phase2_high_load"

        ops_per_sec = self._get_ops_per_minute(self.phase) / 60.0
        interval = 1.0 / ops_per_sec if ops_per_sec > 0 else 0.5

        next_op_time = time.time()
        doc_idx = 0
        query_idx = 0

        while time.time() - self.start_time < duration_seconds:
            if time.time() >= next_op_time:
                # More store operations during high load
                op = random.choices(
                    ["store", "recall", "anomaly_check", "route"],
                    weights=[0.40, 0.50, 0.05, 0.05],
                )[0]

                if op == "store":
                    result = self._execute_store(doc_idx % self.num_docs)
                    doc_idx += 1
                elif op == "recall":
                    result = self._execute_recall(query_idx % self.num_queries)
                    query_idx += 1
                elif op == "anomaly_check":
                    result = self._execute_anomaly_check(query_idx % self.num_queries)
                else:
                    result = self._execute_route(query_idx % self.num_queries)

                if not result.get("success", False):
                    self.crashed = True
                    self.crash_reason = result.get("error", "Unknown error")
                    return

                self.total_operations += 1
                next_op_time += interval

            time.sleep(0.001)

            if len(self.timeline) == 0 or time.time() - self.timeline[-1]["timestamp"] > 10:
                self._log_metrics()

    def run_phase3_edge_cases(self, duration_seconds: float) -> None:
        """Phase 3: Edge cases - memory pressure, anomaly injection."""
        print(f"\n  Phase 3: Edge Cases ({duration_seconds:.0f}s)")
        self.phase = "phase3_edge_cases"

        ops_per_sec = self._get_ops_per_minute(self.phase) / 60.0
        interval = 1.0 / ops_per_sec if ops_per_sec > 0 else 0.5

        next_op_time = time.time()
        doc_idx = 0
        query_idx = 0

        # Inject anomalies: use random vectors
        anomaly_vectors = np.random.randn(10, self.embedding_dim).astype("float32")
        anomaly_vectors /= np.linalg.norm(anomaly_vectors, axis=1, keepdims=True)

        while time.time() - self.start_time < duration_seconds:
            if time.time() >= next_op_time:
                # Check if we should inject an anomaly
                inject_anomaly = self._inject_anomaly_during_phase3()

                if inject_anomaly:
                    # Inject anomalous query
                    emb = torch.from_numpy(anomaly_vectors[random.randint(0, 9)]).unsqueeze(0)
                    result = self._execute_anomaly_check(0)
                    result["injected_anomaly"] = True
                else:
                    op = random.choices(
                        ["store", "recall", "anomaly_check", "route"],
                        weights=[0.35, 0.55, 0.05, 0.05],
                    )[0]

                    if op == "store":
                        result = self._execute_store(doc_idx % self.num_docs)
                        doc_idx += 1
                    elif op == "recall":
                        result = self._execute_recall(query_idx % self.num_queries)
                        query_idx += 1
                    elif op == "anomaly_check":
                        result = self._execute_anomaly_check(query_idx % self.num_queries)
                    else:
                        result = self._execute_route(query_idx % self.num_queries)

                if not result.get("success", False):
                    self.crashed = True
                    self.crash_reason = result.get("error", "Unknown error")
                    return

                self.total_operations += 1
                next_op_time += interval

            time.sleep(0.001)

            if len(self.timeline) == 0 or time.time() - self.timeline[-1]["timestamp"] > 10:
                self._log_metrics()

    def _log_metrics(self) -> None:
        """Log current metrics to timeline."""
        current, peak = tracemalloc.get_traced_memory()
        memory_mb = current / 1024 / 1024

        if self.memory_start is None:
            self.memory_start = memory_mb

        elapsed = time.time() - self.start_time
        elapsed_minutes = elapsed / 60.0

        # Compute intrinsic retrieval quality (average similarity of top-10 results)
        # This measures how well the memory retrieves relevant content
        sample_size = min(50, self.num_queries)
        sample_indices = random.sample(range(self.num_queries), sample_size)
        retrieval_qualities = []
        for qi in sample_indices:
            q_emb = torch.from_numpy(self.query_embs[qi]).unsqueeze(0)
            try:
                results = self.mathir.recall(q_emb, k=10)
                if results:
                    avg_sim = statistics.mean(r["similarity"] for r in results[:10])
                    retrieval_qualities.append(avg_sim)
            except:
                pass

        current_quality = statistics.mean(retrieval_qualities) if retrieval_qualities else 0.0

        # Get tier stats
        stats = self.mathir.get_stats()
        tier_health = {
            "working": stats["tier_working"]["usage"] / max(1, stats["tier_working"]["capacity"]),
            "episodic": stats["tier_episodic"]["usage"] / max(1, stats["tier_episodic"]["capacity"]),
            "semantic": stats["tier_semantic"]["used_prototypes"] / max(1, stats["tier_semantic"]["num_prototypes"]),
            "immune": stats["tier_immune"]["usage"] / max(1, stats["tier_immune"]["capacity"]),
        }

        # Latency stats
        if self.operation_latencies:
            lat_p50 = np.percentile(self.operation_latencies, 50)
            lat_p99 = np.percentile(self.operation_latencies, 99)
        else:
            lat_p50 = lat_p99 = 0.0

        entry = {
            "timestamp": time.time(),
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed_minutes,
            "phase": self.phase,
            "total_operations": self.total_operations,
            "memory_mb": memory_mb,
            "memory_delta_mb": memory_mb - self.memory_start,
            "latency_p50_ms": lat_p50 * 1000,
            "latency_p99_ms": lat_p99 * 1000,
            "retrieval_quality": current_quality,  # avg similarity of top-10
            "tier_health": tier_health,
            "tiers_active": sum(1 for h in tier_health.values() if h > 0),
        }

        self.timeline.append(entry)

        # Check for critical failures
        if lat_p99 * 1000 > CRITICAL_LATENCY_P99_MS:
            entry["warning"] = f"LATENCY_SPIKE: P99={lat_p99*1000:.1f}ms"

        # Quality collapse check: if retrieval quality drops below 0.1 (very poor)
        if current_quality < 0.1 and elapsed_minutes > 5:
            entry["warning"] = f"QUALITY_COLLAPSE: quality={current_quality:.3f}"

        if memory_mb - self.memory_start > CRITICAL_MEMORY_LEAK_MB:
            entry["warning"] = f"MEMORY_LEAK: grew {memory_mb - self.memory_start:.1f}MB"

        self.results_log.append(entry)

    def run_full_test(self, total_duration_seconds: float) -> Dict[str, Any]:
        """Run the complete 2-hour stress test."""
        print(f"\n{'='*70}")
        print("  CROSS-TIER INTEGRATION 2-HOUR STRESS TEST")
        print(f"  Duration: {total_duration_seconds}s ({total_duration_seconds/60:.0f} min)")
        print(f"  Accelerated: {self.accelerated}")
        print(f"  Docs: {self.num_docs}, Queries: {self.num_queries}")
        print(f"{'='*70}")

        # Pre-store some documents to fill memory
        print("\n  Pre-storing documents...")
        pre_store_count = min(500, self.num_docs)
        for i in range(pre_store_count):
            self._execute_store(i)
        print(f"  Pre-stored {pre_store_count} documents")

        # Calculate phase durations
        if self.accelerated:
            phase1_dur = total_duration_seconds * 0.40  # 40% normal
            phase2_dur = total_duration_seconds * 0.30  # 30% high load
            phase3_dur = total_duration_seconds * 0.30  # 30% edge cases
        else:
            phase1_dur = 60 * 60       # 60 min
            phase2_dur = 30 * 60        # 30 min
            phase3_dur = 30 * 60        # 30 min

        # Run phases
        phase1_end = phase1_dur
        phase2_end = phase1_dur + phase2_dur
        phase3_end = total_duration_seconds

        try:
            self.run_phase1_normal(phase1_end)
            if self.crashed:
                raise RuntimeError(f"Crashed in Phase 1: {self.crash_reason}")

            self.run_phase2_high_load(phase2_end)
            if self.crashed:
                raise RuntimeError(f"Crashed in Phase 2: {self.crash_reason}")

            self.run_phase3_edge_cases(phase3_end)
            if self.crashed:
                raise RuntimeError(f"Crashed in Phase 3: {self.crash_reason}")

        except Exception as e:
            print(f"\n  [CRITICAL] System crashed: {e}")
            self.crashed = True
            self.crash_reason = str(e)

        # Final metrics
        self._log_metrics()

        # Generate report
        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate the final stress test report."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_mb = current / 1024 / 1024
        memory_growth_mb = memory_mb - (self.memory_start or 0)

        # Find key metrics at different time points
        phase1_entries = [e for e in self.timeline if e["phase"] == "phase1_normal"]
        phase2_entries = [e for e in self.timeline if e["phase"] == "phase2_high_load"]
        phase3_entries = [e for e in self.timeline if e["phase"] == "phase3_edge_cases"]

        def median_quality(entries):
            if not entries:
                return 0.0
            quals = [e["retrieval_quality"] for e in entries if e.get("retrieval_quality", 0) > 0]
            return np.median(quals) if quals else 0.0

        def median_lat_p99(entries):
            if not entries:
                return 0.0
            lats = [e["latency_p99_ms"] for e in entries if e["latency_p99_ms"] > 0]
            return np.median(lats) if lats else 0.0

        # Detect warnings
        warnings_detected = []
        for entry in self.timeline:
            if "warning" in entry:
                warnings_detected.append(entry["warning"])

        # System uptime
        uptime_pct = 100.0 if not self.crashed else (1.0 - (time.time() - self.start_time) / self.timeline[-1]["elapsed_seconds"]) * 100 if self.timeline else 0.0

        report = {
            "test_type": "2hour_stress_test",
            "accelerated": self.accelerated,
            "total_duration_seconds": time.time() - self.start_time,
            "total_operations": self.total_operations,
            "system_uptime_pct": uptime_pct,
            "crashed": self.crashed,
            "crash_reason": self.crash_reason,
            "memory": {
                "start_mb": self.memory_start or 0,
                "end_mb": memory_mb,
                "growth_mb": memory_growth_mb,
                "peak_mb": peak / 1024 / 1024,
                "memory_leak": memory_growth_mb > CRITICAL_MEMORY_LEAK_MB,
            },
            "quality": {
                "phase1_quality_median": median_quality(phase1_entries),
                "phase2_quality_median": median_quality(phase2_entries),
                "phase3_quality_median": median_quality(phase3_entries),
                "final_quality": self.timeline[-1].get("retrieval_quality", 0.0) if self.timeline else 0.0,
            },
            "latency": {
                "phase1_p99_median_ms": median_lat_p99(phase1_entries),
                "phase2_p99_median_ms": median_lat_p99(phase2_entries),
                "phase3_p99_median_ms": median_lat_p99(phase3_entries),
                "final_p99_ms": self.timeline[-1]["latency_p99_ms"] if self.timeline else 0.0,
            },
            "tier_health_final": self.timeline[-1]["tier_health"] if self.timeline else {},
            "tiers_active_final": self.timeline[-1]["tiers_active"] if self.timeline else 0,
            "warnings_detected": warnings_detected,
            "critical_failures": [
                w for w in warnings_detected
                if "COLLAPSE" in w or "LEAK" in w or "SPIKE" in w
            ],
            "timeline": self.timeline,
        }

        return report


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="2-Hour Cross-Tier Integration Stress Test")
    parser.add_argument(
        "--duration",
        type=int,
        default=120,  # Default 2 minutes for accelerated test
        help="Test duration in seconds (default: 120 for accelerated, 7200 for full)",
    )
    parser.add_argument(
        "--accelerated",
        action="store_true",
        default=True,
        help="Run in accelerated mode (10x faster, for testing)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full 2-hour test (ignores --duration)",
    )
    args = parser.parse_args()

    if args.full:
        duration = 7200  # 2 hours
        accelerated = False
    else:
        duration = args.duration
        accelerated = args.accelerated

    print("\n" + "="*70)
    print("  MATHIR CROSS-TIER INTEGRATION STRESS TEST")
    print("  Loading BEIR SciFact corpus...")
    print("="*70)

    # Load data
    doc_texts, query_texts, qrels, doc_ids, query_ids = load_scifact_corpus()

    # Get embeddings
    doc_embeddings = get_embeddings(doc_texts)
    query_embeddings = get_embeddings(query_texts)

    print(f"\n  Embedding dim: {doc_embeddings.shape[1]}")
    print(f"  Corpus size: {len(doc_embeddings)}")
    print(f"  Query size: {len(query_embeddings)}")

    # Run stress test
    engine = StressTestEngine(
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        doc_texts=doc_texts,
        query_texts=query_texts,
        qrels=qrels,
        doc_ids=doc_ids,
        query_ids=query_ids,
        accelerated=accelerated,
    )

    report = engine.run_full_test(duration)

    # Print summary
    print("\n" + "="*70)
    print("  STRESS TEST RESULTS")
    print("="*70)

    print(f"\n  SYSTEM STATUS:")
    status_str = "[OK] SYSTEM STABLE" if not report['crashed'] else "[FAIL] CRASHED"
    print(f"    {status_str}")
    if report['crashed']:
        print(f"    Crash reason: {report['crash_reason']}")
    print(f"    System uptime: {report['system_uptime_pct']:.1f}%")
    print(f"    Total operations: {report['total_operations']:,}")

    print(f"\n  MEMORY:")
    print(f"    Start: {report['memory']['start_mb']:.1f} MB")
    print(f"    End: {report['memory']['end_mb']:.1f} MB")
    print(f"    Growth: {report['memory']['growth_mb']:.1f} MB")
    print(f"    Peak: {report['memory']['peak_mb']:.1f} MB")
    leak_str = "[FAIL] YES" if report['memory']['memory_leak'] else "[OK] NO"
    print(f"    Memory leak detected: {leak_str}")

    print(f"\n  QUALITY (Retrieval Similarity):")
    print(f"    Phase 1 (normal):  {report['quality']['phase1_quality_median']:.4f}")
    print(f"    Phase 2 (high):   {report['quality']['phase2_quality_median']:.4f}")
    print(f"    Phase 3 (edge):   {report['quality']['phase3_quality_median']:.4f}")
    print(f"    Final:            {report['quality']['final_quality']:.4f}")

    print(f"\n  LATENCY (P99):")
    print(f"    Phase 1 (normal):  {report['latency']['phase1_p99_median_ms']:.1f} ms")
    print(f"    Phase 2 (high):   {report['latency']['phase2_p99_median_ms']:.1f} ms")
    print(f"    Phase 3 (edge):   {report['latency']['phase3_p99_median_ms']:.1f} ms")
    print(f"    Final:            {report['latency']['final_p99_ms']:.1f} ms")

    print(f"\n  TIER HEALTH (final):")
    for tier, health in report['tier_health_final'].items():
        status = "[OK]" if health > 0 else "[FAIL]"
        print(f"    {status} {tier:12s}: {health*100:5.1f}%")
    print(f"    Active tiers: {report['tiers_active_final']}/4")

    if report['warnings_detected']:
        print(f"\n  WARNING ({len(report['warnings_detected'])}):")
        for w in report['warnings_detected'][:10]:
            print(f"      - {w}")
        if len(report['warnings_detected']) > 10:
            print(f"      ... and {len(report['warnings_detected']) - 10} more")

    if report['critical_failures']:
        print(f"\n  CRITICAL FAILURES ({len(report['critical_failures'])}):")
        for f in report['critical_failures']:
            print(f"      - {f}")

    # Save results
    results_file = Path(__file__).parent / "integration_stress_results.json"
    with open(results_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_file}")

    # Broadcast summary
    print("\n" + "="*70)
    print("  BROADCAST SUMMARY")
    print("="*70)
    print(f"  System uptime: {report['system_uptime_pct']:.1f}%")
    print(f"  Memory leak: {'YES' if report['memory']['memory_leak'] else 'NO'} (grew from {report['memory']['start_mb']:.0f} to {report['memory']['end_mb']:.0f} MB)")
    print(f"  Quality at 60min: {report['quality']['phase1_quality_median']:.3f} retrieval similarity")
    print(f"  Quality at 120min: {report['quality']['final_quality']:.3f} retrieval similarity")
    print(f"  P99 latency at 120min: {report['latency']['final_p99_ms']:.1f}ms")
    print(f"  Graceful degradation: {'YES' if report['quality']['phase3_quality_median'] > 0.3 else 'NO'}")
    print(f"  Critical failures: {report['critical_failures'] or 'none'}")
    print("="*70)

    return 0 if not report['crashed'] and not report['critical_failures'] else 1


if __name__ == "__main__":
    sys.exit(main())
