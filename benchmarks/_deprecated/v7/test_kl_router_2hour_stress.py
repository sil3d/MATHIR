"""
KL Router 2-Hour Stress Test
============================

Tests whether MATHIR's KL Router stays balanced under sustained load.

Simulation Parameters:
    Total time: 2 hours (compressed to ~minutes for test execution)
    Total queries routed: ~10,000
    Query mix: 25% each tier (uniform target)
    Test interval: every 10 minutes (simulated time)

Phases:
    Phase 1 (0-60 min): Normal routing with diverse queries
    Phase 2 (60-90 min): Heavy sustained load
    Phase 3 (90-120 min): Adversarial queries (stress specific tiers)

Metrics Tracked:
    - Weight entropy (max=1.39 for perfectly balanced, 0 for collapse)
    - Max weight tier
    - KL loss
    - Collapse risk

Success Criteria:
    - Entropy stays > 1.0 (router balanced)
    - No single tier dominates (>85% weight)
    - KL constraint prevents collapse
"""

import os
import sys
import json
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathir_lib.router import KLConstrainedRouter
from mathir_lib.config import get_default_config

# Tier names in order
TIER_NAMES = ["working", "episodic", "semantic", "immune"]
MAX_ENTROPY = math.log(4)  # ≈ 1.386


@dataclass
class StressMetrics:
    """Metrics snapshot at a point in time."""
    time_min: int
    total_routed: int
    weight_entropy: float
    max_entropy: float
    entropy_ratio: float
    max_weight_tier: str
    max_weight_pct: float
    kl_loss: float
    collapse_risk: str
    phase: str
    tier_weights: List[float]
    notes: str


class QueryGenerator:
    """Generates synthetic query embeddings for different memory tiers."""
    
    def __init__(self, internal_dim: int = 272, seed: int = 42):
        self.internal_dim = internal_dim
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
    
    def generate(self, tier: Optional[int] = None, batch_size: int = 1) -> torch.Tensor:
        """
        Generate synthetic embeddings.
        
        Args:
            tier: Specific tier to target (0-3), or None for random
            batch_size: Number of embeddings to generate
            
        Returns:
            [batch_size, internal_dim] tensor
        """
        embeddings = []
        
        for _ in range(batch_size):
            # Select tier
            if tier is None:
                t = self.rng.randint(0, 4)
            else:
                t = tier
            
            emb = torch.zeros(self.internal_dim)
            
            if t == 0:  # Working: recent context
                start, end = 0, self.internal_dim // 4
                emb[start:end] = torch.randn(end - start) * 2 + 3
            
            elif t == 1:  # Episodic: past events
                start, end = self.internal_dim // 4, self.internal_dim // 2
                emb[start:end] = torch.randn(end - start) * 2 + 3
            
            elif t == 2:  # Semantic: general knowledge
                start, end = self.internal_dim // 2, 3 * self.internal_dim // 4
                emb[start:end] = torch.randn(end - start) * 2 + 3
            
            else:  # Immune: anomaly/gibberish
                emb = torch.randn(self.internal_dim) * 0.1
            
            embeddings.append(emb)
        
        return torch.stack(embeddings)
    
    def generate_adversarial_immune(self, batch_size: int = 1) -> torch.Tensor:
        """Generate only immune/anomaly queries (stress immune tier)."""
        return self.generate(tier=3, batch_size=batch_size)
    
    def generate_adversarial_working(self, batch_size: int = 1) -> torch.Tensor:
        """Generate only working/recent-context queries (stress working tier)."""
        return self.generate(tier=0, batch_size=batch_size)
    
    def generate_diverse(self, batch_size: int = 4) -> Tuple[torch.Tensor, List[int]]:
        """Generate diverse batch with 25% each tier."""
        embeddings = []
        expected_tiers = []
        per_tier = batch_size // 4
        
        for tier in range(4):
            emb = self.generate(tier=tier, batch_size=per_tier)
            embeddings.append(emb)
            expected_tiers.extend([tier] * per_tier)
        
        # Handle remainder with random tiers
        remainder = batch_size % 4
        if remainder > 0:
            for _ in range(remainder):
                tier = self.rng.randint(0, 4)
                emb = self.generate(tier=tier, batch_size=1)
                embeddings.append(emb)
                expected_tiers.append(tier)
        
        return torch.cat(embeddings, dim=0), expected_tiers


class KLRouterStressTest:
    """2-hour stress test for KL Router."""
    
    def __init__(
        self,
        internal_dim: int = 272,
        total_queries: int = 10000,
        test_interval_min: int = 10,
        time_scale: float = 1.0,  # 1.0 = real-time, >1 = faster simulation
    ):
        self.internal_dim = internal_dim
        self.total_queries = total_queries
        self.test_interval_min = test_interval_min
        self.time_scale = time_scale
        
        # Initialize router
        self.router = KLConstrainedRouter(
            input_dim=internal_dim,
            num_memories=4,
            kl_coefficient=0.01,
            kl_target="uniform",
            temperature=1.0,
        )
        self.router.train()
        
        # Initialize query generator
        self.query_gen = QueryGenerator(internal_dim=internal_dim)
        
        # Metrics storage
        self.metrics_history: List[StressMetrics] = []
        self.total_routed = 0
        
        # Phase configuration
        self.phases = {
            "phase1": {"name": "Normal Routing", "duration_min": 60, "start_min": 0},
            "phase2": {"name": "Sustained Load", "duration_min": 30, "start_min": 60},
            "phase3": {"name": "Adversarial", "duration_min": 30, "start_min": 90},
        }
    
    def compute_entropy(self, weights: torch.Tensor) -> float:
        """Compute entropy of weight distribution."""
        weights = weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        entropy = -(weights * weights.log()).sum(dim=-1)
        return entropy.mean().item()
    
    def get_current_phase(self, elapsed_min: float) -> Tuple[str, str]:
        """Get current phase name and description."""
        for phase_id, phase in self.phases.items():
            start = phase["start_min"]
            end = start + phase["duration_min"]
            if start <= elapsed_min < end:
                return phase_id, phase["name"]
        if elapsed_min >= 120:
            return "complete", "Test Complete"
        return "warmup", "Warmup"
    
    def assess_collapse_risk(self, entropy: float, max_weight_pct: float) -> str:
        """Assess collapse risk based on entropy and max weight."""
        max_entropy = MAX_ENTROPY
        entropy_ratio = entropy / max_entropy
        
        if max_weight_pct > 0.95 or entropy < 0.3:
            return "CRITICAL"
        elif max_weight_pct > 0.85 or entropy < 0.5:
            return "HIGH"
        elif max_weight_pct > 0.70 or entropy_ratio < 0.6:
            return "MEDIUM"
        elif entropy_ratio < 0.8:
            return "LOW"
        else:
            return "MINIMAL"
    
    def route_batch(self, embeddings: torch.Tensor) -> Dict:
        """Route a batch of embeddings through the router."""
        result = self.router(embeddings, training=True)
        weights = result["weights"]
        
        # Get stats
        mean_weights = weights.mean(dim=0)
        max_weight_pct = mean_weights.max().item()
        max_tier_idx = mean_weights.argmax().item()
        
        return {
            "weights": weights,
            "mean_weights": mean_weights,
            "entropy": self.compute_entropy(weights),
            "max_weight_pct": max_weight_pct,
            "max_tier": TIER_NAMES[max_tier_idx],
            "kl_loss": result["kl_loss"].item(),
        }
    
    def run_phase1_normal(self, queries_per_minute: int = 83) -> None:
        """Phase 1: Normal diverse routing for 60 minutes."""
        print("\n" + "=" * 60)
        print("PHASE 1: NORMAL ROUTING (0-60 min)")
        print("=" * 60)
        print("Testing balanced routing with diverse queries...")
        print()
        
        phase_end = 60
        current_min = len(self.metrics_history) * self.test_interval_min
        
        while current_min < phase_end and self.total_routed < self.total_queries:
            # Generate diverse batch
            embeddings, _ = self.query_gen.generate_diverse(batch_size=queries_per_minute)
            
            # Route
            result = self.route_batch(embeddings)
            self.total_routed += embeddings.size(0)
            
            # Record metrics at intervals
            if self.total_routed % (queries_per_minute * self.test_interval_min) < queries_per_minute:
                metrics = StressMetrics(
                    time_min=current_min,
                    total_routed=self.total_routed,
                    weight_entropy=result["entropy"],
                    max_entropy=MAX_ENTROPY,
                    entropy_ratio=result["entropy"] / MAX_ENTROPY,
                    max_weight_tier=result["max_tier"],
                    max_weight_pct=result["max_weight_pct"],
                    kl_loss=result["kl_loss"],
                    collapse_risk=self.assess_collapse_risk(result["entropy"], result["max_weight_pct"]),
                    phase="Phase 1: Normal",
                    tier_weights=result["mean_weights"].tolist(),
                    notes="Balanced diverse routing",
                )
                self.metrics_history.append(metrics)
                self._print_metrics(metrics)
            
            current_min = self.total_routed / queries_per_minute
    
    def run_phase2_sustained(self, queries_per_minute: int = 83) -> None:
        """Phase 2: Heavy sustained load for 30 minutes."""
        print("\n" + "=" * 60)
        print("PHASE 2: SUSTAINED LOAD (60-90 min)")
        print("=" * 60)
        print("Heavy sustained load - checking for tier dominance...")
        print()
        
        phase_start = 60
        phase_end = 90
        
        while self.total_routed < phase_end * queries_per_minute and self.total_routed < self.total_queries:
            # Generate diverse batch
            embeddings, _ = self.query_gen.generate_diverse(batch_size=queries_per_minute)
            
            # Route
            result = self.route_batch(embeddings)
            self.total_routed += embeddings.size(0)
            
            # Record metrics at intervals
            if self.total_routed % (queries_per_minute * self.test_interval_min) < queries_per_minute:
                current_min = self.total_routed / queries_per_minute
                metrics = StressMetrics(
                    time_min=int(current_min),
                    total_routed=self.total_routed,
                    weight_entropy=result["entropy"],
                    max_entropy=MAX_ENTROPY,
                    entropy_ratio=result["entropy"] / MAX_ENTROPY,
                    max_weight_tier=result["max_tier"],
                    max_weight_pct=result["max_weight_pct"],
                    kl_loss=result["kl_loss"],
                    collapse_risk=self.assess_collapse_risk(result["entropy"], result["max_weight_pct"]),
                    phase="Phase 2: Sustained Load",
                    tier_weights=result["mean_weights"].tolist(),
                    notes="Heavy sustained load",
                )
                self.metrics_history.append(metrics)
                self._print_metrics(metrics)
    
    def run_phase3_adversarial(self, queries_per_minute: int = 83) -> None:
        """Phase 3: Adversarial queries to stress specific tiers."""
        print("\n" + "=" * 60)
        print("PHASE 3: ADVERSARIAL QUERIES (90-120 min)")
        print("=" * 60)
        print("Stress testing with adversarial query patterns...")
        print()
        
        phase_start = 90
        phase_end = 120
        adversarial_cycle = [
            ("immune_only", self.query_gen.generate_adversarial_immune),
            ("working_only", self.query_gen.generate_adversarial_working),
        ]
        cycle_idx = 0
        
        while self.total_routed < phase_end * queries_per_minute and self.total_routed < self.total_queries:
            # Alternate between adversarial patterns
            name, gen_fn = adversarial_cycle[cycle_idx % 2]
            embeddings = gen_fn(batch_size=queries_per_minute)
            cycle_idx += 1
            
            # Route
            result = self.route_batch(embeddings)
            self.total_routed += embeddings.size(0)
            
            # Record metrics at intervals
            if self.total_routed % (queries_per_minute * self.test_interval_min) < queries_per_minute:
                current_min = self.total_routed / queries_per_minute
                metrics = StressMetrics(
                    time_min=int(current_min),
                    total_routed=self.total_routed,
                    weight_entropy=result["entropy"],
                    max_entropy=MAX_ENTROPY,
                    entropy_ratio=result["entropy"] / MAX_ENTROPY,
                    max_weight_tier=result["max_tier"],
                    max_weight_pct=result["max_weight_pct"],
                    kl_loss=result["kl_loss"],
                    collapse_risk=self.assess_collapse_risk(result["entropy"], result["max_weight_pct"]),
                    phase=f"Phase 3: Adversarial ({name})",
                    tier_weights=result["mean_weights"].tolist(),
                    notes=f"Adversarial: {name}",
                )
                self.metrics_history.append(metrics)
                self._print_metrics(metrics)
    
    def _print_metrics(self, m: StressMetrics) -> None:
        """Print a metrics snapshot."""
        print(
            f"  [{m.time_min:3.0f} min] "
            f"Entropy: {m.weight_entropy:.3f}/{m.max_entropy:.3f} ({m.entropy_ratio:.1%}) | "
            f"Max: {m.max_weight_tier} ({m.max_weight_pct:.1%}) | "
            f"KL: {m.kl_loss:.4f} | "
            f"Risk: {m.collapse_risk}"
        )
    
    def run(self) -> Dict:
        """Run the full 2-hour stress test."""
        print("=" * 60)
        print("KL ROUTER 2-HOUR STRESS TEST")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total queries to route: {self.total_queries}")
        print(f"Test intervals: every {self.test_interval_min} minutes")
        print(f"Max entropy (balanced): {MAX_ENTROPY:.3f}")
        print()
        
        start_time = time.time()
        
        # Run phases
        self.run_phase1_normal()
        self.run_phase2_sustained()
        self.run_phase3_adversarial()
        
        elapsed = time.time() - start_time
        
        # Final assessment
        return self.generate_report(elapsed)
    
    def generate_report(self, elapsed: float) -> Dict:
        """Generate final report and broadcast."""
        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        
        # Find key metrics
        initial_metrics = self.metrics_history[0] if self.metrics_history else None
        mid_metrics = self.metrics_history[len(self.metrics_history)//2] if self.metrics_history else None
        final_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Check for collapse
        collapse_detected = False
        for m in self.metrics_history:
            if m.collapse_risk in ("CRITICAL", "HIGH"):
                collapse_detected = True
                break
        
        # Entropy at key points
        entropy_60 = None
        entropy_120 = None
        for m in self.metrics_history:
            if m.time_min >= 55 and entropy_60 is None:
                entropy_60 = m.weight_entropy
            if m.time_min >= 115 and entropy_120 is None:
                entropy_120 = m.weight_entropy
        
        # Build report
        report = {
            "test_info": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_queries_routed": self.total_routed,
                "test_duration_seconds": elapsed,
                "max_entropy_possible": MAX_ENTROPY,
                "internal_dim": self.internal_dim,
            },
            "initial_state": asdict(initial_metrics) if initial_metrics else None,
            "mid_state": asdict(mid_metrics) if mid_metrics else None,
            "final_state": asdict(final_metrics) if final_metrics else None,
            "metrics_timeline": [asdict(m) for m in self.metrics_history],
            "summary": {
                "entropy_at_60min": entropy_60,
                "entropy_at_120min": entropy_120,
                "collapse_detected": collapse_detected,
                "max_tier_dominance": max(m.max_weight_pct for m in self.metrics_history) if self.metrics_history else 0,
                "min_entropy": min(m.weight_entropy for m in self.metrics_history) if self.metrics_history else 0,
                "kl_constraint_worked": not collapse_detected,
                "router_stable": final_metrics.weight_entropy > 1.0 if final_metrics else False,
            },
            "broadcast": {
                "weight_entropy_at_60min": f"{entropy_60:.3f}" if entropy_60 else "N/A",
                "weight_entropy_at_120min": f"{entropy_120:.3f}" if entropy_120 else "N/A",
                "collapsed_to_one_tier": "YES" if collapse_detected else "NO",
                "kl_constraint_prevented_collapse": "YES" if not collapse_detected else "NO",
                "router_stable": "YES" if (final_metrics and final_metrics.weight_entropy > 1.0) else "NO",
                "final_entropy_ratio": f"{final_metrics.entropy_ratio:.1%}" if final_metrics else "N/A",
                "critical_issues": [],
            },
        }
        
        # Add critical issues
        issues = []
        if collapse_detected:
            issues.append("Router experienced collapse or near-collapse during stress test")
        if final_metrics and final_metrics.weight_entropy < 1.0:
            issues.append(f"Final entropy ({final_metrics.weight_entropy:.3f}) below stability threshold (1.0)")
        if final_metrics and final_metrics.max_weight_pct > 0.85:
            issues.append(f"Final max tier weight ({final_metrics.max_weight_pct:.1%}) indicates tier dominance")
        
        report["broadcast"]["critical_issues"] = issues
        
        # Print summary
        print()
        print("KEY FINDINGS:")
        print(f"  Entropy at 60min:  {entropy_60:.3f} (max={MAX_ENTROPY:.3f})" if entropy_60 else "  Entropy at 60min:  N/A")
        print(f"  Entropy at 120min: {entropy_120:.3f} (max={MAX_ENTROPY:.3f})" if entropy_120 else "  Entropy at 120min: N/A")
        print(f"  Collapse detected: {'YES - WARNING!' if collapse_detected else 'NO'}")
        print(f"  KL constraint worked: {'YES' if not collapse_detected else 'NO'}")
        print(f"  Router stable: {'YES' if (final_metrics and final_metrics.weight_entropy > 1.0) else 'NO'}")
        
        if issues:
            print()
            print("CRITICAL ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Broadcast
        print()
        print("=" * 60)
        print("BROADCAST")
        print("=" * 60)
        b = report["broadcast"]
        print(f"  Weight entropy at 60min: {b['weight_entropy_at_60min']} (max=1.386)")
        print(f"  Weight entropy at 120min: {b['weight_entropy_at_120min']}")
        print(f"  Collapsed to one tier: {b['collapsed_to_one_tier']}")
        print(f"  KL constraint prevented collapse: {b['kl_constraint_prevented_collapse']}")
        print(f"  Router stable: {b['router_stable']}")
        
        # Save results
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "router_stress_results.json"
        )
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print()
        print(f"Full results saved to: {out_path}")
        
        return report


def main():
    """Run the stress test."""
    print("\n" + "#" * 60)
    print("# KL Router 2-Hour Stress Test")
    print("#" * 60)
    
    # For faster testing, we'll use fewer queries but simulate the full timeline
    # In production, set total_queries=10000 for full 2-hour test
    test = KLRouterStressTest(
        internal_dim=272,
        total_queries=10000,  # Full 10k queries
        test_interval_min=10,
    )
    
    results = test.run()
    
    return results


if __name__ == "__main__":
    results = main()
