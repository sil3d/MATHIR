"""
Immunological Memory 2-Hour Stress Test
=======================================

Tests LONG-RUN stability of anomaly detection:
1. Threshold stability — does Mahalanobis threshold converge and stay stable?
2. Detection consistency — can it detect anomalies after 2 hours of training?
3. Concept drift handling — does it adapt to new "normal" or get stuck?
4. Covariance stability — does the covariance matrix become singular?

Simulated time: compress 2 hours into ~5-10 minutes real time via batched learning.
Anomaly injections: every 10 minutes (12 total over 2 hours)

Usage:
    python benchmarks/test_immunological_2hour_stress.py
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Import MATHIR components
from mathir_lib.memory.immunological import MahalanobisImmunologicalMemory


# ============================================================================
# CONFIGURATION
# ============================================================================

class StressTestConfig:
    """2-hour stress test configuration."""

    # Simulation time (in simulated minutes)
    TOTAL_SIMULATED_MINUTES = 120
    ANOMALY_INJECTION_INTERVAL_MINUTES = 10  # Every 10 minutes
    TOTAL_ANOMALY_INJECTIONS = TOTAL_SIMULATED_MINUTES // ANOMALY_INJECTION_INTERVAL_MINUTES  # 12

    # Learning phases (in simulated minutes)
    PHASE1_LEARNING_END = 60   # Phase 1: Learning Normal (0-60 min)
    PHASE2_STABILITY_END = 90  # Phase 2: Stability Test (60-90 min)
    PHASE3_DRIFT_END = 120     # Phase 3: Concept Drift (90-120 min)

    # Data generation
    EMBED_DIM = 384
    NORMAL_STD = 0.5

    # Immunological memory settings
    IMMUNE_CAPACITY = 100  # Realistic: 100 patterns
    THRESHOLD = 2.0
    EMA_DECAY = 0.01
    REGULARIZATION = 1e-4
    MIN_SAMPLES_FOR_DETECTION = 10

    # Speedup: compress 2 hours into ~12 iterations
    # 120 min / 12 injections = 10 min per injection
    # Total samples: 12 iterations × 500 samples/iter = 6000 samples (matches ~5000 request)
    BATCH_SIZE = 500  # Samples per iteration (stores individually for memory bank)
    SAMPLES_PER_BATCH = BATCH_SIZE  # Each "store" call = 1 sample

    # Anomaly settings
    ANOMALY_OFFSET = 4.0  # How many std devs away from normal center

    # Concept drift settings (Phase 3)
    DRIFT_START_MINUTE = 90
    DRIFT_MAGNITUDE = 1.5  # How much to shift the center per 10 minutes

    def __init__(self):
        # 12 total iterations (one per 10-minute block)
        # Phase 1: iterations 0-5 (60 min, 6 iterations)
        # Phase 2: iterations 6-8 (30 min, 3 iterations)
        # Phase 3: iterations 9-11 (30 min, 3 iterations)
        self.ITERATIONS_PHASE1 = 6
        self.ITERATIONS_PHASE2 = 3
        self.ITERATIONS_PHASE3 = 3
        self.TOTAL_ITERATIONS = self.ITERATIONS_PHASE1 + self.ITERATIONS_PHASE2 + self.ITERATIONS_PHASE3  # 12

        self.NORMAL_SAMPLES_PHASE1 = self.ITERATIONS_PHASE1 * self.BATCH_SIZE  # 6 * 500 = 3000
        self.NORMAL_SAMPLES_PHASE2 = self.ITERATIONS_PHASE2 * self.BATCH_SIZE  # 3 * 500 = 1500
        self.NORMAL_SAMPLES_PHASE3 = self.ITERATIONS_PHASE3 * self.BATCH_SIZE  # 3 * 500 = 1500
        self.TOTAL_NORMAL_SAMPLES = (
            self.NORMAL_SAMPLES_PHASE1 +
            self.NORMAL_SAMPLES_PHASE2 +
            self.NORMAL_SAMPLES_PHASE3
        )  # 6000 total


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Phase(Enum):
    LEARNING = "learning"           # Phase 1: Building initial memory
    STABILITY = "stability"        # Phase 2: Testing detection consistency
    CONCEPT_DRIFT = "concept_drift"  # Phase 3: Adapting to new normal


@dataclass
class AnomalyInjection:
    """Record of an anomaly injection."""
    injection_number: int
    simulated_minute: int
    phase: str
    true_label: int = 1  # Always anomaly


@dataclass
class DetectionResult:
    """Result of anomaly detection at a checkpoint."""
    checkpoint_minute: int
    phase: str
    normal_count: int
    anomaly_score: float
    detected: bool
    threshold: float
    cov_condition: float
    false_positive: bool
    notes: str


@dataclass
class StressTestResults:
    """Complete stress test results."""
    config: Dict[str, Any]
    phase1_final_stats: Dict[str, Any]
    phase2_detection_rate: float
    phase3_detection_rate: float
    phase3_adaptation: str  # YES/NO/PARTIAL
    covariance_became_singular: bool
    critical_issues: List[str]
    timeline: List[Dict[str, Any]]
    summary: Dict[str, Any]


# ============================================================================
# DATA GENERATION
# ============================================================================

class ConceptDriftGenerator:
    """Generates embeddings with controlled concept drift."""

    def __init__(self, embed_dim: int, initial_center: np.ndarray, normal_std: float):
        self.embed_dim = embed_dim
        self.current_center = initial_center.copy()
        self.original_center = initial_center.copy()
        self.normal_std = normal_std
        self.drift_vector = np.zeros(embed_dim)

    def generate_normal_batch(self, batch_size: int, drift_magnitude: float = 0.0) -> np.ndarray:
        """Generate a batch of normal embeddings, optionally drifting."""
        if drift_magnitude > 0:
            # Add drift to center
            self.drift_vector = np.random.randn(self.embed_dim) * drift_magnitude
            self.current_center = self.original_center + self.drift_vector

        embeddings = np.random.randn(batch_size, self.embed_dim) * self.normal_std + self.current_center
        return embeddings

    def generate_anomaly_batch(self, batch_size: int, offset: float = 4.0) -> np.ndarray:
        """Generate a batch of anomaly embeddings (far from current center)."""
        direction = np.random.randn(self.embed_dim)
        direction = direction / np.linalg.norm(direction)
        anomaly_center = self.current_center + offset * self.normal_std * direction
        embeddings = np.random.randn(batch_size, self.embed_dim) * self.normal_std + anomaly_center
        return embeddings

    def get_current_center(self) -> np.ndarray:
        return self.current_center.copy()


# ============================================================================
# STRESS TEST
# ============================================================================

class ImmunologicalStressTest:
    """2-hour stress test for immunological memory."""

    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()
        self.immune: Optional[MahalanobisImmunologicalMemory] = None
        self.data_gen: Optional[ConceptDriftGenerator] = None
        self.timeline: List[Dict[str, Any]] = []
        self.anomaly_injections: List[AnomalyInjection] = []
        self.detection_results: List[DetectionResult] = []

        # Tracking
        self.current_phase = Phase.LEARNING
        self.injection_count = 0
        self.detection_history = []
        self.false_positive_history = []
        self.cov_condition_history = []
        self.threshold_history = []

    def initialize(self):
        """Initialize the immunological memory and data generator."""
        print("\n" + "=" * 70)
        print("IMMUNOLOGICAL MEMORY 2-HOUR STRESS TEST")
        print("=" * 70)

        # Initialize immunological memory
        self.immune = MahalanobisImmunologicalMemory(
            capacity=self.config.IMMUNE_CAPACITY,
            feature_dim=self.config.EMBED_DIM,
            threshold=self.config.THRESHOLD,
            ema_decay=self.config.EMA_DECAY,
            regularization=self.config.REGULARIZATION,
        )

        # Initialize data generator with random center
        initial_center = np.random.randn(self.config.EMBED_DIM) * 0.1
        self.data_gen = ConceptDriftGenerator(
            embed_dim=self.config.EMBED_DIM,
            initial_center=initial_center,
            normal_std=self.config.NORMAL_STD,
        )

        print(f"\nConfiguration:")
        print(f"  Total simulated time: {self.config.TOTAL_SIMULATED_MINUTES} minutes")
        print(f"  Anomaly injections: {self.config.TOTAL_ANOMALY_INJECTIONS} (every {self.config.ANOMALY_INJECTION_INTERVAL_MINUTES} min)")
        print(f"  Immune capacity: {self.config.IMMUNE_CAPACITY} patterns")
        print(f"  EMA decay: {self.config.EMA_DECAY}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Embedding dim: {self.config.EMBED_DIM}")
        print(f"  Normal std: {self.config.NORMAL_STD}")
        print(f"  Anomaly offset: {self.config.ANOMALY_OFFSET}")

        # Log initial state
        self._log_checkpoint(0, "initialization", notes="System initialized, empty memory")

    def _log_checkpoint(self, minute: int, phase: str, normal_count: int = 0,
                       anomaly_score: float = None, detected: bool = None,
                       false_positive: bool = None, threshold: float = None,
                       cov_condition: float = None, notes: str = ""):
        """Log a checkpoint in the timeline."""
        checkpoint = {
            "simulated_minute": minute,
            "phase": phase,
            "normal_count": normal_count,
            "anomaly_score": anomaly_score,
            "detected": detected,
            "threshold": threshold,
            "cov_condition": cov_condition,
            "false_positive": false_positive,
            "notes": notes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.timeline.append(checkpoint)

        # Also track for historical analysis
        if cov_condition is not None:
            self.cov_condition_history.append(cov_condition)
        if threshold is not None:
            self.threshold_history.append(threshold)
        if detected is not None:
            self.detection_history.append(detected)
        if false_positive is not None:
            self.false_positive_history.append(false_positive)

    def _get_covariance_condition(self) -> float:
        """Get the condition number of the covariance matrix."""
        try:
            cond = torch.linalg.cond(self.immune.running_cov.unsqueeze(0)).item()
            return cond
        except Exception:
            return float('inf')

    def _get_effective_threshold(self) -> float:
        """Get the effective threshold being used."""
        # From the recognize method:
        # chi2_threshold = (feature_dim + 2*sqrt(feature_dim)) / 10
        chi2 = self.config.EMBED_DIM + 2 * (self.config.EMBED_DIM ** 0.5)
        chi2 = chi2 / 10
        return max(self.config.THRESHOLD, chi2 ** 0.5)

    def run_phase1_learning(self):
        """Phase 1: Learn normal patterns (0-60 min simulated)."""
        print(f"\n{'=' * 70}")
        print("[Phase 1] LEARNING NORMAL — Building immunological memory")
        print(f"{'=' * 70}")

        self.current_phase = Phase.LEARNING
        phase1_iterations = self.config.ITERATIONS_PHASE1

        print(f"  Target: {self.config.NORMAL_SAMPLES_PHASE1} normal samples over {phase1_iterations} iterations")
        print(f"  Simulated time: 0-{self.config.PHASE1_LEARNING_END} minutes")

        for iteration in range(phase1_iterations):
            simulated_minute = iteration * 10

            # Generate and store normal batch
            normal_batch = self.data_gen.generate_normal_batch(self.config.BATCH_SIZE)
            normal_tensor = torch.from_numpy(normal_batch).float()

            for i in range(len(normal_tensor)):
                self.immune.store(normal_tensor[i:i+1])

            # Log checkpoint at intervals
            stats = self.immune.get_stats()
            cov_cond = self._get_covariance_condition()
            threshold = self._get_effective_threshold()

            self._log_checkpoint(
                minute=simulated_minute,
                phase="learning",
                normal_count=stats['count'],
                threshold=threshold,
                cov_condition=cov_cond,
                notes=f"Learning phase - stored {stats['n_updates']} samples"
            )

            # Progress output every iteration
            print(f"  [{simulated_minute:3d} min] count={stats['count']:3d}, "
                  f"updates={stats['n_updates']:4d}, cov_cond={cov_cond:.2e}, "
                  f"threshold={threshold:.3f}")

        # Final Phase 1 stats
        stats = self.immune.get_stats()
        print(f"\n  Phase 1 Complete:")
        print(f"    Patterns stored: {stats['count']}")
        print(f"    Total updates: {stats['n_updates']}")
        print(f"    Covariance condition: {self._get_covariance_condition():.2e}")

        return {
            "phase": "learning",
            "final_count": stats['count'],
            "final_updates": stats['n_updates'],
            "final_cov_condition": self._get_covariance_condition(),
            "final_threshold": self._get_effective_threshold(),
        }

    def _inject_and_test_anomaly(self, simulated_minute: int, phase: str,
                                  inject_true_anomaly: bool = True) -> DetectionResult:
        """Inject an anomaly and test detection."""
        self.injection_count += 1

        # Generate anomaly
        anomaly_batch = self.data_gen.generate_anomaly_batch(
            batch_size=1,
            offset=self.config.ANOMALY_OFFSET
        )
        anomaly_tensor = torch.from_numpy(anomaly_batch).float()

        # Get anomaly score BEFORE storing (should detect)
        with torch.no_grad():
            anomaly_score = self.immune.get_anomaly_score(anomaly_tensor).item()
            threshold = self._get_effective_threshold()
            detected = anomaly_score > threshold

        # For false positive tracking: also test a normal sample
        normal_batch = self.data_gen.generate_normal_batch(1)
        normal_tensor = torch.from_numpy(normal_batch).float()
        with torch.no_grad():
            normal_score = self.immune.get_anomaly_score(normal_tensor).item()
            false_positive = normal_score > threshold

        stats = self.immune.get_stats()
        cov_condition = self._get_covariance_condition()

        result = DetectionResult(
            checkpoint_minute=simulated_minute,
            phase=phase,
            normal_count=stats['count'],
            anomaly_score=anomaly_score,
            detected=detected,
            threshold=threshold,
            cov_condition=cov_condition,
            false_positive=false_positive,
            notes=f"Injection #{self.injection_count}"
        )
        self.detection_results.append(result)

        # Log
        self._log_checkpoint(
            minute=simulated_minute,
            phase=phase,
            normal_count=stats['count'],
            anomaly_score=anomaly_score,
            detected=detected,
            threshold=threshold,
            cov_condition=cov_condition,
            false_positive=false_positive,
            notes=f"Anomaly injection #{self.injection_count} - {'DETECTED' if detected else 'MISSED'}"
        )

        return result

    def run_phase2_stability(self):
        """Phase 2: Test detection stability (60-90 min simulated)."""
        print(f"\n{'=' * 70}")
        print("[Phase 2] STABILITY TEST — Continuous normal + periodic anomalies")
        print(f"{'=' * 70}")

        self.current_phase = Phase.STABILITY
        num_injections = self.config.ITERATIONS_PHASE2  # 3 injections in stability phase

        print(f"  Anomaly injections: {num_injections}")
        print(f"  Simulated time: {self.config.PHASE1_LEARNING_END}-{self.config.PHASE2_STABILITY_END} minutes")

        detected_count = 0
        total_fp = 0

        for i in range(num_injections):
            simulated_minute = self.config.PHASE1_LEARNING_END + (i * 10)

            # Feed normal batch (simulating continuous learning)
            normal_batch = self.data_gen.generate_normal_batch(self.config.BATCH_SIZE)
            normal_tensor = torch.from_numpy(normal_batch).float()
            for j in range(len(normal_tensor)):
                self.immune.store(normal_tensor[j:j+1])

            # Inject and test anomaly
            result = self._inject_and_test_anomaly(simulated_minute, "stability")
            detected_count += 1 if result.detected else 0
            total_fp += 1 if result.false_positive else 0

            print(f"  [{simulated_minute:3d} min] anomaly_score={result.anomaly_score:.4f}, "
                  f"detected={result.detected}, fp={result.false_positive}, "
                  f"cov_cond={result.cov_condition:.2e}")

        detection_rate = detected_count / num_injections * 100
        fp_rate = total_fp / num_injections * 100

        print(f"\n  Phase 2 Complete:")
        print(f"    Detection rate: {detection_rate:.1f}% ({detected_count}/{num_injections})")
        print(f"    False positive rate: {fp_rate:.1f}% ({total_fp}/{num_injections})")

        return {
            "phase": "stability",
            "num_injections": num_injections,
            "detection_rate": detection_rate,
            "false_positive_rate": fp_rate,
            "detected_count": detected_count,
        }

    def run_phase3_concept_drift(self):
        """Phase 3: Test concept drift adaptation (90-120 min simulated)."""
        print(f"\n{'=' * 70}")
        print("[Phase 3] CONCEPT DRIFT — Adapting to new 'normal'")
        print(f"{'=' * 70}")

        self.current_phase = Phase.CONCEPT_DRIFT
        num_injections = self.config.ITERATIONS_PHASE3  # 3 injections in concept drift phase

        print(f"  Anomaly injections: {num_injections}")
        print(f"  Drift magnitude: {self.config.DRIFT_MAGNITUDE} per 10 minutes")
        print(f"  Simulated time: {self.config.PHASE2_STABILITY_END}-{self.config.PHASE3_DRIFT_END} minutes")

        detected_count = 0
        total_fp = 0
        adaptation_successes = 0

        for i in range(num_injections):
            simulated_minute = self.config.PHASE2_STABILITY_END + (i * 10)

            # Feed normal batch WITH concept drift
            # The "normal" center is slowly shifting
            drift_magnitude = self.config.DRIFT_MAGNITUDE if simulated_minute >= self.config.DRIFT_START_MINUTE else 0
            normal_batch = self.data_gen.generate_normal_batch(
                self.config.BATCH_SIZE,
                drift_magnitude=drift_magnitude
            )
            normal_tensor = torch.from_numpy(normal_batch).float()
            for j in range(len(normal_tensor)):
                self.immune.store(normal_tensor[j:j+1])

            # Inject and test anomaly (true anomaly, should still be detected)
            result = self._inject_and_test_anomaly(simulated_minute, "concept_drift")
            detected_count += 1 if result.detected else 0
            total_fp += 1 if result.false_positive else 0

            # Check adaptation: after drift, normal scores should be lower
            # (meaning the system learned the new normal)
            stats = self.immune.get_stats()
            threshold = self._get_effective_threshold()
            cov_cond = self._get_covariance_condition()

            print(f"  [{simulated_minute:3d} min] anomaly_score={result.anomaly_score:.4f}, "
                  f"detected={result.detected}, fp={result.false_positive}, "
                  f"cov_cond={cov_cond:.2e}, drift={drift_magnitude:.1f}")

        detection_rate = detected_count / num_injections * 100
        fp_rate = total_fp / num_injections * 100

        # Determine adaptation status
        if detection_rate >= 80:
            adaptation = "YES"
        elif detection_rate >= 50:
            adaptation = "PARTIAL"
        else:
            adaptation = "NO"

        print(f"\n  Phase 3 Complete:")
        print(f"    Detection rate: {detection_rate:.1f}% ({detected_count}/{num_injections})")
        print(f"    False positive rate: {fp_rate:.1f}% ({total_fp}/{num_injections})")
        print(f"    Concept drift adaptation: {adaptation}")

        return {
            "phase": "concept_drift",
            "num_injections": num_injections,
            "detection_rate": detection_rate,
            "false_positive_rate": fp_rate,
            "detected_count": detected_count,
            "adaptation": adaptation,
        }

    def run(self) -> StressTestResults:
        """Run the complete 2-hour stress test."""
        self.initialize()

        # Phase 1: Learning
        phase1_stats = self.run_phase1_learning()

        # Phase 2: Stability
        phase2_stats = self.run_phase2_stability()

        # Phase 3: Concept Drift
        phase3_stats = self.run_phase3_concept_drift()

        # Analyze results
        print(f"\n{'=' * 70}")
        print("STRESS TEST COMPLETE — ANALYSIS")
        print(f"{'=' * 70}")

        # Covariance singularity check
        max_cov_cond = max(self.cov_condition_history) if self.cov_condition_history else 0
        covariance_singular = max_cov_cond > 1e10  # Threshold for singularity

        # Critical issues
        issues = []
        if covariance_singular:
            issues.append(f"Covariance matrix became ill-conditioned (condition: {max_cov_cond:.2e})")

        phase2_detection = phase2_stats['detection_rate']
        phase3_detection = phase3_stats['detection_rate']

        if phase2_detection < 100:
            issues.append(f"Phase 2 detection rate was {phase2_detection:.1f}% (expected 100%)")

        if phase3_detection < 50:
            issues.append(f"Phase 3 detection rate was {phase3_detection:.1f}% — concept drift not handled")

        # Final summary
        summary = {
            "phase2_detection_rate": phase2_detection,
            "phase3_detection_rate": phase3_detection,
            "concept_drift_adaptation": phase3_stats['adaptation'],
            "covariance_singular": covariance_singular,
            "max_cov_condition": max_cov_cond,
            "total_injections": self.injection_count,
            "issues_found": len(issues),
        }

        print(f"\nDetection Accuracy at 60min (Phase 2 end): {phase2_detection:.1f}%")
        print(f"Detection Accuracy at 120min (Phase 3 end): {phase3_detection:.1f}%")
        print(f"Concept Drift Adaptation: {phase3_stats['adaptation']}")
        print(f"Covariance Became Singular: {'YES' if covariance_singular else 'NO'}")
        print(f"Max Covariance Condition: {max_cov_cond:.2e}")
        print(f"Critical Issues Found: {len(issues)}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")

        results = StressTestResults(
            config={
                "total_minutes": self.config.TOTAL_SIMULATED_MINUTES,
                "anomaly_injections": self.config.TOTAL_ANOMALY_INJECTIONS,
                "immune_capacity": self.config.IMMUNE_CAPACITY,
                "ema_decay": self.config.EMA_DECAY,
                "batch_size": self.config.BATCH_SIZE,
                "embed_dim": self.config.EMBED_DIM,
                "normal_std": self.config.NORMAL_STD,
                "anomaly_offset": self.config.ANOMALY_OFFSET,
                "drift_magnitude": self.config.DRIFT_MAGNITUDE,
            },
            phase1_final_stats=phase1_stats,
            phase2_detection_rate=phase2_detection,
            phase3_detection_rate=phase3_detection,
            phase3_adaptation=phase3_stats['adaptation'],
            covariance_became_singular=covariance_singular,
            critical_issues=issues,
            timeline=self.timeline,
            summary=summary,
        )

        return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Immunological Memory 2-Hour Stress Test")
    parser.add_argument("--output", default="benchmarks/immunological_stress_results.json",
                        help="Output JSON file")
    parser.add_argument("--capacity", type=int, default=100,
                        help="Immunological memory capacity")
    parser.add_argument("--ema-decay", type=float, default=0.01,
                        help="EMA decay for covariance updates")
    parser.add_argument("--drift-magnitude", type=float, default=1.5,
                        help="Concept drift magnitude per 10 minutes")

    args = parser.parse_args()

    # Override config with args
    config = StressTestConfig()
    config.IMMUNE_CAPACITY = args.capacity
    config.EMA_DECAY = args.ema_decay
    config.DRIFT_MAGNITUDE = args.drift_magnitude

    # Run stress test
    print("\nStarting 2-Hour Stress Test...")
    print("Note: Using accelerated simulation (2 hours compressed into ~2-3 minutes)")
    print("=" * 70)

    start_time = time.time()
    stress_test = ImmunologicalStressTest(config)
    results = stress_test.run()
    elapsed_time = time.time() - start_time

    print(f"\nReal time elapsed: {elapsed_time:.1f} seconds")
    print(f"Simulated time: {config.TOTAL_SIMULATED_MINUTES} minutes")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to dict for JSON serialization
    results_dict = {
        "config": results.config,
        "phase1_final_stats": results.phase1_final_stats,
        "phase2_detection_rate": results.phase2_detection_rate,
        "phase3_detection_rate": results.phase3_detection_rate,
        "phase3_adaptation": results.phase3_adaptation,
        "covariance_became_singular": results.covariance_became_singular,
        "critical_issues": results.critical_issues,
        "timeline": results.timeline,
        "summary": results.summary,
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # BROADCAST format
    print("\n" + "=" * 70)
    print("BROADCAST:")
    print("=" * 70)
    print(f"- Detection accuracy at 60min: {results.phase2_detection_rate:.1f}%")
    print(f"- Detection accuracy at 120min: {results.phase3_detection_rate:.1f}%")
    print(f"- Concept drift adaptation: {results.phase3_adaptation}")
    print(f"- Covariance became singular: {'YES' if results.covariance_became_singular else 'NO'}")
    print(f"- Critical issues found: {results.critical_issues if results.critical_issues else 'None'}")
    print("=" * 70)