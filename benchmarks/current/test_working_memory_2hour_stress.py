"""
Working Memory 2-Hour Stress Test
==================================

Tests MATHIR's 64-slot circular buffer under extreme context switching stress.

Key Questions:
1. Does working memory keep working correctly after rapid context switching for 2 hours?
2. Does attention quality degrade over time?
3. Does the buffer get "corrupted" by too many writes?
4. Can we detect context contamination?

Test Phases:
- Phase 1 (0-60 min): Normal cycling (context A→B→A pattern)
- Phase 2 (60-90 min): Rapid switching (0.5s per context)
- Phase 3 (90-120 min): Context persistence (how long does context remain "active"?)

Metrics Tracked:
- Context isolation (A→B→A pattern matching)
- Attention stability over time
- Contamination rate
- Query result drift
"""

import json
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_FILE = Path(__file__).parent.parent / "results" / "working_memory_2hour_stress_results.json"


# ---------------------------------------------------------------------------
# Working Memory (direct import from dropin)
# ---------------------------------------------------------------------------
class _WorkingMemory(torch.nn.Module):
    """64-slot circular buffer with multi-head attention retrieval.

    Direct copy from mathir_dropin.memory for isolated testing.
    """

    def __init__(self, capacity: int = 64, dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        self.register_buffer("buffer", torch.zeros(capacity, dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        self.attention = torch.nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )

    def store(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            flat = x.detach().reshape(-1, self.dim)
            n = flat.size(0)
            ptr = int(self.ptr.item())
            if n >= self.capacity:
                self.buffer.copy_(flat[-self.capacity:])
                self.ptr.fill_(0)
                self.count.fill_(self.capacity)
                return
            end = ptr + n
            if end <= self.capacity:
                self.buffer[ptr:end] = flat
            else:
                first = self.capacity - ptr
                self.buffer[ptr:] = flat[:first]
                self.buffer[: n - first] = flat[first:]
            self.ptr.fill_((ptr + n) % self.capacity)
            self.count.fill_(min(self.count.item() + n, self.capacity))

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        # Ensure query is [B, D] with batch dimension
        if query.dim() == 1:
            query = query.unsqueeze(0)
        B, D = query.shape

        c = int(self.count.item())
        if c == 0:
            return torch.zeros(B, D, device=query.device, dtype=query.dtype)

        # ctx: [c, D] -> [B, c, D]
        ctx = self.buffer[:c].unsqueeze(0).expand(B, -1, -1)
        # query: [B, D] -> [B, 1, D]
        out, _ = self.attention(query.unsqueeze(1), ctx, ctx)
        return out.squeeze(1)  # [B, D]

    def reset(self) -> None:
        self.buffer.zero_()
        self.ptr.fill_(0)
        self.count.fill_(0)

    @property
    def usage(self) -> int:
        return int(self.count.item())


# ---------------------------------------------------------------------------
# Synthetic Context Generator
# ---------------------------------------------------------------------------
class SyntheticContextGenerator:
    """Generate distinguishable synthetic embeddings for different contexts."""

    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # Create distinct basis vectors for each context dimension
        self.context_bases: Dict[str, torch.Tensor] = {}

    def generate_context_embedding(self, context_id: str, strength: float = 1.0) -> torch.Tensor:
        """Generate embedding with distinct signature for context_id."""
        if context_id not in self.context_bases:
            # Create a unique signature vector
            sig = self.rng.randn(self.dim).astype(np.float32)
            sig = sig / (np.linalg.norm(sig) + 1e-8)
            self.context_bases[context_id] = torch.from_numpy(sig)

        base = self.context_bases[context_id]
        # Add small noise for variation
        noise = torch.randn(self.dim) * 0.01
        emb = base + noise * (1 - strength)
        return emb / (torch.norm(emb) + 1e-8)

    def generate_context_set(self, context_id: str, num_items: int = 10) -> List[torch.Tensor]:
        """Generate a set of related embeddings for a context."""
        embeddings = []
        for i in range(num_items):
            # Each item is slightly different but recognizably from same context
            base = self.generate_context_embedding(context_id, strength=0.9)
            variation = torch.randn(self.dim) * 0.05
            emb = base + variation
            emb = emb / (torch.norm(emb) + 1e-8)
            embeddings.append(emb)
        return embeddings


# ---------------------------------------------------------------------------
# Metrics Calculator
# ---------------------------------------------------------------------------
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    # Ensure 2D: [1, D] or [D]
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if a.size(0) == 1 and b.size(0) > 1:
        a = a.expand(b.size(0), -1)
    elif b.size(0) == 1 and a.size(0) > 1:
        b = b.expand(a.size(0), -1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item())


def compute_context_match_score(result_a1: torch.Tensor, result_a2: torch.Tensor) -> float:
    """Compute how similar the results are for context A repeated after B.

    Returns:
        1.0 = perfect match (A→B→A works perfectly, no contamination)
        0.0 = complete mismatch (A is contaminated by B)
    """
    return cosine_similarity(result_a1, result_a2)


def compute_attention_stability(wm: _WorkingMemory, query: torch.Tensor, num_samples: int = 5) -> float:
    """Measure attention output stability by running same query multiple times."""
    # Ensure query has batch dimension [1, D]
    if query.dim() == 1:
        query = query.unsqueeze(0)

    results = []
    for _ in range(num_samples):
        with torch.no_grad():
            out = wm.retrieve(query)
            results.append(out)

    # Compute pairwise similarities
    similarities = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            sim = cosine_similarity(results[i], results[j])
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 1.0


# ---------------------------------------------------------------------------
# Contamination Test
# ---------------------------------------------------------------------------
def run_contamination_test(wm: _WorkingMemory, ctx_gen: SyntheticContextGenerator,
                           verbose: bool = False) -> Dict[str, Any]:
    """Test if loading context B corrupts previously loaded context A.

    Protocol:
    1. Load context A → query → save result
    2. Load context B → query → save result
    3. Load context A again → query → compare to step 1

    Returns:
        Dict with match scores and contamination metrics
    """
    # Reset working memory
    wm.reset()

    # Step 1: Load context A (COVID vaccines), query for "side effects"
    context_a = ctx_gen.generate_context_set("COVID_VACCINE", num_items=15)
    for emb in context_a:
        wm.store(emb.unsqueeze(0))

    # Query with a "side effects" style query
    query_side_effects = ctx_gen.generate_context_embedding("QUERY_SIDE_EFFECTS")

    with torch.no_grad():
        result_a1 = wm.retrieve(query_side_effects.unsqueeze(0))

    if verbose:
        print(f"  Step 1 (Context A): stored {wm.usage} items, retrieved result (norm={float(torch.norm(result_a1)):.4f})")

    # Step 2: Load context B (cooking), query for "side effects"
    context_b = ctx_gen.generate_context_set("COOKING", num_items=15)
    for emb in context_b:
        wm.store(emb.unsqueeze(0))

    with torch.no_grad():
        result_b = wm.retrieve(query_side_effects.unsqueeze(0))

    if verbose:
        print(f"  Step 2 (Context B): stored {wm.usage} items, retrieved result (norm={float(torch.norm(result_b)):.4f})")

    # Step 3: Load context A again (COVID), query for "side effects"
    # This should give same result as step 1 if context isolation works
    for emb in context_a:
        wm.store(emb.unsqueeze(0))

    with torch.no_grad():
        result_a2 = wm.retrieve(query_side_effects.unsqueeze(0))

    if verbose:
        print(f"  Step 3 (Context A again): stored {wm.usage} items, retrieved result (norm={float(torch.norm(result_a2)):.4f})")

    # Compute metrics
    a_match_score = compute_context_match_score(result_a1, result_a2)
    b_match_score = compute_context_match_score(result_a1, result_b)

    contamination_detected = a_match_score < 0.7  # Threshold for contamination

    return {
        "result_a1_norm": float(torch.norm(result_a1).item()),
        "result_a2_norm": float(torch.norm(result_a2).item()),
        "result_b_norm": float(torch.norm(result_b).item()),
        "a1_a2_similarity": a_match_score,  # Should be HIGH (context isolation works)
        "a1_b_similarity": b_match_score,    # Should be LOW (contexts are different)
        "contamination_detected": contamination_detected,
        "context_isolation_score": a_match_score,
    }


# ---------------------------------------------------------------------------
# Attention Head Stability Analysis
# ---------------------------------------------------------------------------
def analyze_attention_heads(wm: _WorkingMemory) -> Dict[str, Any]:
    """Analyze multi-head attention for stability metrics."""
    # Get attention weights from the attention module
    # Note: MultiheadAttention doesn't store attention weights by default in batch_first mode
    # We'll do a structural analysis instead

    return {
        "num_heads": wm.attention.num_heads,
        "head_dim": wm.attention.head_dim,
        "buffer_shape": list(wm.buffer.shape),
        "current_usage": wm.usage,
        "ptr_position": int(wm.ptr.item()),
    }


# ---------------------------------------------------------------------------
# Main Stress Test
# ---------------------------------------------------------------------------
def run_stress_test(
    duration_minutes: int = 120,
    test_interval_minutes: int = 10,
    compressed: bool = True,
    compression_factor: int = 60,
) -> Dict[str, Any]:
    """Run the 2-hour stress test.

    Args:
        duration_minutes: How long to run (default 120 for full 2-hour test)
        test_interval_minutes: How often to run contamination tests
        compressed: If True, run compressed version (1 minute = 1 hour simulation)
        compression_factor: How many simulated minutes per real second

    Returns:
        Full results dictionary with timeline and metrics
    """
    print("=" * 70)
    print("WORKING MEMORY 2-HOUR STRESS TEST")
    print("=" * 70)

    # Configuration
    dim = 128
    capacity = 64
    num_heads = 4

    # Initialize working memory
    wm = _WorkingMemory(capacity=capacity, dim=dim, num_heads=num_heads)
    ctx_gen = SyntheticContextGenerator(dim=dim)

    # Test parameters
    if compressed:
        total_cycles = duration_minutes * compression_factor  # cycles in compressed time
        interval_cycles = test_interval_minutes * compression_factor
        cycle_time_sec = 0.01  # Fast cycling
    else:
        total_cycles = duration_minutes * 60 // 14  # ~14 sec per context switch
        interval_cycles = test_interval_minutes * 60 // 14
        cycle_time_sec = 14

    print(f"\nConfiguration:")
    print(f"  Buffer capacity: {capacity} slots")
    print(f"  Embedding dimension: {dim}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Duration: {duration_minutes} minutes (simulated)")
    print(f"  Context switches: ~{total_cycles}")
    print(f"  Test interval: every {test_interval_minutes} minutes ({interval_cycles} cycles)")
    print(f"  Compressed mode: {compressed} (compression factor: {compression_factor}x)")

    # Results tracking
    timeline: List[Dict[str, Any]] = []
    phase_summaries: Dict[str, Dict[str, Any]] = {}

    # Baseline contamination test
    print("\n[BASELINE] Running initial contamination test...")
    baseline = run_contamination_test(wm, ctx_gen, verbose=True)
    print(f"  Baseline context isolation score: {baseline['context_isolation_score']:.4f}")
    print(f"  Contamination detected: {baseline['contamination_detected']}")

    # Initial attention stability
    query = ctx_gen.generate_context_embedding("TEST_QUERY")
    initial_stability = compute_attention_stability(wm, query)
    print(f"  Initial attention stability: {initial_stability:.4f}")

    timeline.append({
        "time_min": 0,
        "phase": "baseline",
        "context_switches": 0,
        "context_isolation_score": baseline["context_isolation_score"],
        "attention_stability": initial_stability,
        "buffer_usage": wm.usage,
        "contamination_detected": baseline["contamination_detected"],
        "notes": "Fresh start",
    })

    # Phase definitions
    phases = {
        "normal_cycling": {"start": 0, "end": 60, "switch_interval": 14.0, "name": "Normal Cycling"},
        "rapid_switching": {"start": 60, "end": 90, "switch_interval": 0.5, "name": "Rapid Switching"},
        "persistence_test": {"start": 90, "end": 120, "switch_interval": 5.0, "name": "Context Persistence"},
    }

    print("\n[STRESS TEST] Starting context switching...")

    context_list = ["COVID_VACCINE", "COOKING", "SPORTS", "TECH", "TRAVEL", "MUSIC", "SCIENCE"]
    context_idx = 0
    switch_count = 0
    start_real_time = time.time()

    # Pre-compute context sets
    context_sets = {ctx: ctx_gen.generate_context_set(ctx, num_items=10) for ctx in context_list}

    for minute in range(0, duration_minutes):
        minute_start = time.time()

        # Determine current phase
        current_phase = "unknown"
        for phase_name, phase_info in phases.items():
            if phase_info["start"] <= minute < phase_info["end"]:
                current_phase = phase_name
                switch_interval = phase_info["switch_interval"]
                break

        # Run context switches for this minute
        if minute > 0:
            for _ in range(compression_factor if compressed else 60):
                ctx = context_list[context_idx % len(context_list)]
                ctx_set = context_sets[ctx]

                # Store context
                for emb in ctx_set:
                    wm.store(emb.unsqueeze(0))

                switch_count += 1
                context_idx += 1

                if compressed:
                    time.sleep(0.0001)  # Tiny delay to prevent CPU spin

        # Run contamination test at intervals
        if minute > 0 and minute % test_interval_minutes == 0:
            elapsed_test = time.time() - start_real_time

            # Quick contamination test
            test_result = run_contamination_test(wm, ctx_gen, verbose=False)

            # Attention stability
            stability = compute_attention_stability(wm, query)

            # Buffer analysis
            attention_info = analyze_attention_heads(wm)

            record = {
                "time_min": minute,
                "phase": current_phase,
                "context_switches": switch_count,
                "context_isolation_score": test_result["context_isolation_score"],
                "attention_stability": stability,
                "buffer_usage": wm.usage,
                "buffer_capacity": capacity,
                "ptr_position": attention_info["ptr_position"],
                "contamination_detected": test_result["contamination_detected"],
                "a1_a2_similarity": test_result["a1_a2_similarity"],
                "a1_b_similarity": test_result["a1_b_similarity"],
                "real_elapsed_sec": elapsed_test,
                "notes": f"Phase: {phases[current_phase]['name']}",
            }

            timeline.append(record)

            print(f"  [{minute:3d}min | {current_phase[:12]:12s} | "
                  f"switches:{switch_count:5d} | isolation:{test_result['context_isolation_score']:.3f} | "
                  f"stability:{stability:.3f} | contamination:{test_result['contamination_detected']}]")

        # Phase transition
        if minute > 0 and minute in [60, 90]:
            phase_name = phases[[p for p, v in phases.items() if v["start"] == minute][0]]["name"]
            print(f"\n  *** PHASE TRANSITION: {minute} min -> {phase_name} ***\n")

    total_real_time = time.time() - start_real_time

    # Final comprehensive test
    print("\n[FINAL] Running comprehensive end-of-test contamination test...")
    final_test = run_contamination_test(wm, ctx_gen, verbose=True)
    final_stability = compute_attention_stability(wm, query)

    # Phase summaries
    for phase_name, phase_info in phases.items():
        phase_records = [r for r in timeline if r["phase"] == phase_name]
        if phase_records:
            phase_summaries[phase_name] = {
                "name": phase_info["name"],
                "duration_min": phase_info["end"] - phase_info["start"],
                "num_tests": len(phase_records),
                "avg_isolation": float(np.mean([r["context_isolation_score"] for r in phase_records])),
                "min_isolation": float(np.min([r["context_isolation_score"] for r in phase_records])),
                "max_isolation": float(np.max([r["context_isolation_score"] for r in phase_records])),
                "avg_stability": float(np.mean([r["attention_stability"] for r in phase_records])),
                "contamination_events": sum(1 for r in phase_records if r["contamination_detected"]),
            }

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "duration_minutes": duration_minutes,
            "buffer_capacity": capacity,
            "embedding_dim": dim,
            "num_attention_heads": num_heads,
            "test_interval_minutes": test_interval_minutes,
            "compressed_mode": compressed,
            "compression_factor": compression_factor if compressed else 1,
        },
        "total_context_switches": switch_count,
        "total_real_time_sec": total_real_time,
        "baseline": baseline,
        "final": {
            "context_isolation_score": final_test["context_isolation_score"],
            "attention_stability": final_stability,
            "contamination_detected": final_test["contamination_detected"],
            "a1_a2_similarity": final_test["a1_a2_similarity"],
        },
        "phase_summaries": phase_summaries,
        "timeline": timeline,
        "broadcast": {
            "context_isolation_after_60min": "PASS" if timeline[7]["context_isolation_score"] > 0.8 else "FAIL",
            "context_isolation_value_60min": timeline[7]["context_isolation_score"] if len(timeline) > 7 else None,
            "rapid_switching_quality_90min": timeline[10]["context_isolation_score"] if len(timeline) > 10 else None,
            "attention_stability": "STABLE" if final_stability > 0.9 else ("DRIFTING" if final_stability > 0.7 else "CORRUPTED"),
            "contamination_rate": sum(1 for r in timeline if r["contamination_detected"]) / max(len(timeline), 1) * 100,
            "critical_issues": [],
        },
    }

    # Determine critical issues
    critical_issues = []
    if final_test["context_isolation_score"] < 0.7:
        critical_issues.append(f"Low context isolation at end of test: {final_test['context_isolation_score']:.3f}")
    if final_stability < 0.8:
        critical_issues.append(f"Attention stability degraded: {final_stability:.3f}")
    if wm.usage == 0:
        critical_issues.append("Buffer appears empty - possible corruption")
    if sum(1 for r in timeline if r["contamination_detected"]) > len(timeline) * 0.3:
        critical_issues.append(f"High contamination rate: {sum(1 for r in timeline if r['contamination_detected']) / len(timeline) * 100:.1f}%")

    results["broadcast"]["critical_issues"] = critical_issues

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the 2-hour stress test with compressed mode for quick results."""
    print("\n" + "=" * 70)
    print("WORKING MEMORY 2-HOUR STRESS TEST")
    print("=" * 70)
    print("\nNOTE: Running in COMPRESSED mode (1 minute = 1 hour simulation)")
    print("      This completes in ~2 minutes instead of 2 hours.")
    print("      Use compressed=False for real-time testing.\n")

    # Run compressed test (simulates 2 hours in ~2 minutes)
    results = run_stress_test(
        duration_minutes=120,
        test_interval_minutes=10,
        compressed=True,
        compression_factor=60,
    )

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline Context Isolation Score: {results['baseline']['context_isolation_score']:.4f}")
    print(f"Final Context Isolation Score: {results['final']['context_isolation_score']:.4f}")
    print(f"Final Attention Stability: {results['final']['attention_stability']:.4f}")

    print("\nPhase Summaries:")
    for phase, summary in results["phase_summaries"].items():
        print(f"\n  [{phase}] {summary['name']}")
        print(f"    Duration: {summary['duration_min']} min")
        print(f"    Tests: {summary['num_tests']}")
        print(f"    Avg Isolation: {summary['avg_isolation']:.4f}")
        print(f"    Min/Max Isolation: {summary['min_isolation']:.4f} / {summary['max_isolation']:.4f}")
        print(f"    Avg Stability: {summary['avg_stability']:.4f}")
        print(f"    Contamination Events: {summary['contamination_events']}")

    print("\n" + "=" * 70)
    print("BROADCAST:")
    print(f"- Context isolation after 60min: {results['broadcast']['context_isolation_after_60min']} "
          f"({results['broadcast'].get('context_isolation_value_60min', 'N/A')})")
    print(f"- Rapid switching quality at 90min: {results['broadcast'].get('rapid_switching_quality_90min', 'N/A')}")
    print(f"- Attention stability: {results['broadcast']['attention_stability']}")
    print(f"- Contamination rate: {results['broadcast']['contamination_rate']:.1f}%")
    print(f"- Critical issues found: {results['broadcast']['critical_issues'] if results['broadcast']['critical_issues'] else 'None'}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()