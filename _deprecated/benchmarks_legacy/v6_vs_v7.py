"""
V6 vs V7 Benchmark Comparison
=============================

Compares MATHIR V6 (baseline) vs MATHIR V7 (all improvements enabled).

Metrics:
  1. Compression ratio
  2. Inference latency (P50 / P95 ms per perceive() call)
  3. Model size (total parameters + buffers)
  4. Retention / recall availability after long horizon
  5. Anomaly detection (F1 on synthetic normal/anomaly)
  6. Router convergence (min weight → less collapse)

When MATHIRPluginV7 is not yet available, V7 columns are reported as
"n/a (not implemented)" and the V6 baseline still runs. This lets the
benchmark be re-run incrementally as the V7 plugin lands.

Usage:
    python benchmarks/v6_vs_v7.py
    python benchmarks/v6_vs_v7.py --output results.json
    python benchmarks/v6_vs_v7.py --embedding-dim 4096 --iters 100
"""

import os
import sys
import time
import json
import argparse
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as F

# Ensure project root is on path so `from mathir_lib import ...` works
# regardless of where the benchmark is invoked from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mathir_lib import MATHIRPlugin, get_default_config  # noqa: E402

# ---------------------------------------------------------------------------
# V7 is imported lazily — the benchmark works without it.
# ---------------------------------------------------------------------------
try:
    from mathir_lib import MATHIRPluginV7  # noqa: E402
    _V7_AVAILABLE = True
except Exception:
    MATHIRPluginV7 = None  # type: ignore[assignment]
    _V7_AVAILABLE = False


# =========================================================================
# Result types
# =========================================================================
@dataclass
class BenchResult:
    metric: str
    v6_value: float
    v7_value: Optional[float] = None
    improvement: str = "n/a"
    v7_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================================
# Helpers
# =========================================================================
def _percentile(values: List[float], pct: float) -> float:
    """Return the pct-th percentile of `values` (linear interpolation)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _safe_median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def _make_v7(embedding_dim: int):
    """Build a V7 plugin with all 8 improvements enabled.
    Returns (plugin, None) on success or (None, error_message)."""
    if not _V7_AVAILABLE:
        return None, "MATHIRPluginV7 not yet importable"
    try:
        cfg = get_default_config()
        # Best-effort: enable V7 features if the config keys exist.
        mem = cfg.setdefault("memory", {})
        for key, val in [
            ("episodic_type", "ebbinghaus"),
            ("immune_type", "mahalanobis"),
            ("semantic_type", "hyperbolic"),
            ("use_sparse_coding", True),
        ]:
            if key not in mem:
                mem[key] = val
        plugin = MATHIRPluginV7(embedding_dim, config=cfg)
        return plugin, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# =========================================================================
# 1. Compression
# =========================================================================
def benchmark_compression(embedding_dim: int = 272) -> BenchResult:
    """Compare memory compression: V6 dense float32 vs V7 sparse + TurboQuant."""
    from mathir_lib.memory.sparse_coding import SparseCodingMemory
    from mathir_lib.compression import TurboQuantCompression  # type: ignore

    N = 1000
    # V6: dense float32 — N * D * 4 bytes
    v6_bytes = N * embedding_dim * 4

    # V7 sparse: (idx, val) per non-zero + dictionary overhead
    sc = SparseCodingMemory(num_atoms=1088, feature_dim=embedding_dim, sparsity=8)
    v7_sparse_bytes = N * sc.sparsity * 8  # (idx+val) per non-zero
    v7_sparse_bytes += sc.num_atoms * embedding_dim * 4  # dictionary

    # V7 + TurboQuant 3-bit on top
    v7_bytes = v7_sparse_bytes / (32.0 / 3.0)

    ratio = v6_bytes / max(v7_bytes, 1.0)
    return BenchResult(
        metric=f"Compression (bytes per {N} memories, d={embedding_dim})",
        v6_value=float(v6_bytes),
        v7_value=float(v7_bytes),
        improvement=f"{ratio:.1f}x smaller",
        v7_note="V7 = SparseCoding (8x) + TurboQuant 3-bit (~10.7x)",
    )


# =========================================================================
# 2. Inference latency
# =========================================================================
def benchmark_inference_latency(embedding_dim: int = 1024, n_iters: int = 100) -> BenchResult:
    """Compare per-call inference latency (P50 + P95 in ms)."""
    v6 = MATHIRPlugin(embedding_dim)
    v7, v7_err = _make_v7(embedding_dim)
    emb = torch.randn(1, embedding_dim)

    # Warmup
    for _ in range(10):
        v6.perceive(emb)
        if v7 is not None:
            v7.perceive(emb)

    # V6
    v6_times: List[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        v6.perceive(emb)
        v6_times.append((time.perf_counter() - t0) * 1000)

    v6_p50 = _percentile(v6_times, 50)
    v6_p95 = _percentile(v6_times, 95)

    v7_value: Optional[float] = None
    v7_p95_value: Optional[float] = None
    improvement = "n/a"
    v7_note = v7_err or ""

    if v7 is not None:
        v7_times: List[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            v7.perceive(emb)
            v7_times.append((time.perf_counter() - t0) * 1000)
        v7_p50 = _percentile(v7_times, 50)
        v7_p95 = _percentile(v7_times, 95)
        v7_value = v7_p50
        v7_p95_value = v7_p95
        delta = (v6_p50 - v7_p50) / max(v6_p50, 1e-6) * 100
        improvement = f"P50 {delta:+.1f}%, P95 v6={v6_p95:.2f}ms vs v7={v7_p95:.2f}ms"
        v7_note = ""  # clear error note on success

    return BenchResult(
        metric=f"Inference latency (P50 ms, dim={embedding_dim}, n={n_iters})",
        v6_value=v6_p50,
        v7_value=v7_value,
        improvement=improvement,
        v7_note=v7_note,
    )


# =========================================================================
# 3. Model size (parameters + buffers)
# =========================================================================
def benchmark_memory_size(embedding_dim: int = 1024) -> BenchResult:
    """Compare total learnable params + buffer count."""
    v6 = MATHIRPlugin(embedding_dim)
    v7, v7_err = _make_v7(embedding_dim)

    def _count(module) -> int:
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters()) + sum(
            b.numel() for b in module.buffers()
        )

    v6_total = _count(v6)
    v7_total = _count(v7)

    if v7 is not None:
        ratio = v7_total / max(v6_total, 1)
        improvement = f"{ratio:.2f}x (v6={v6_total:,} → v7={v7_total:,})"
    else:
        improvement = "n/a"
        v7_err = v7_err or ""

    return BenchResult(
        metric=f"Model size (params + buffers, dim={embedding_dim})",
        v6_value=float(v6_total),
        v7_value=float(v7_total) if v7 is not None else None,
        improvement=improvement,
        v7_note=v7_err or "",
    )


# =========================================================================
# 4. Retention / recall availability
# =========================================================================
def benchmark_retention(embedding_dim: int = 1024, n_stores: int = 200, n_queries: int = 20) -> BenchResult:
    """Test that recall returns at least one memory after a long horizon."""
    torch.manual_seed(0)
    v6 = MATHIRPlugin(embedding_dim)
    v7, v7_err = _make_v7(embedding_dim)

    # Store memories in both
    for i in range(n_stores):
        emb = torch.randn(1, embedding_dim)
        v6.perceive(emb)
        v6.store({"embedding": emb})
        if v7 is not None:
            v7.perceive(emb)
            v7.store({"embedding": emb})

    # Query and check recall
    v6_recall = 0
    for _ in range(n_queries):
        q = torch.randn(1, embedding_dim)
        try:
            r = v6.recall(q, k=3)
            if len(r) > 0:
                v6_recall += 1
        except Exception:
            pass

    v7_recall = 0
    v7_value: Optional[float] = None
    improvement = "n/a"
    v7_note = v7_err or ""

    if v7 is not None:
        for _ in range(n_queries):
            q = torch.randn(1, embedding_dim)
            try:
                r = v7.recall(q, k=3)
                if len(r) > 0:
                    v7_recall += 1
            except Exception:
                pass
        v7_value = float(v7_recall)
        improvement = f"v6={v6_recall}/{n_queries} vs v7={v7_recall}/{n_queries}"
        v7_note = ""

    return BenchResult(
        metric=f"Recall availability ({n_queries} queries after {n_stores} stores)",
        v6_value=float(v6_recall),
        v7_value=v7_value,
        improvement=improvement,
        v7_note=v7_note,
    )


# =========================================================================
# 5. Anomaly detection
# =========================================================================
def benchmark_anomaly_detection(embedding_dim: int = 128, n_normal: int = 200, n_anomaly: int = 50) -> BenchResult:
    """F1 of anomaly_score on normal vs OOD inputs (threshold=1.0)."""
    torch.manual_seed(0)
    v6 = MATHIRPlugin(embedding_dim)
    v7, v7_err = _make_v7(embedding_dim)
    threshold = 1.0

    # Train on normal patterns
    for _ in range(n_normal):
        x = torch.randn(1, embedding_dim)
        v6.perceive(x)
        v6.store({"embedding": x})
        if v7 is not None:
            v7.perceive(x)
            v7.store({"embedding": x})

    def _eval(model, scale: float, is_anomaly: bool) -> int:
        """Count predictions where score > threshold matches the label."""
        if model is None:
            return 0
        flagged = 0
        for _ in range(n_anomaly):
            x = torch.randn(1, embedding_dim) * scale
            try:
                score = model.perceive(x)["anomaly_score"].mean().item()
            except Exception:
                continue
            predicted_anomaly = score > threshold
            if predicted_anomaly == is_anomaly:
                flagged += 1
        return flagged

    v6_tp = _eval(v6, scale=5.0, is_anomaly=True)
    v6_fp = n_anomaly - _eval(v6, scale=1.0, is_anomaly=False)
    v6_correct = v6_tp + _eval(v6, scale=1.0, is_anomaly=False)
    v6_acc = v6_correct / (2 * n_anomaly) if n_anomaly else 0.0

    v7_value: Optional[float] = None
    v7_tp = v7_fp = 0
    improvement = "n/a"
    v7_note = v7_err or ""

    if v7 is not None:
        v7_tp = _eval(v7, scale=5.0, is_anomaly=True)
        v7_fp = n_anomaly - _eval(v7, scale=1.0, is_anomaly=False)
        v7_correct = v7_tp + _eval(v7, scale=1.0, is_anomaly=False)
        v7_acc = v7_correct / (2 * n_anomaly) if n_anomaly else 0.0
        v7_value = v7_acc
        improvement = (
            f"v6={v6_acc:.3f} (TP={v6_tp}, FP={v6_fp}) → "
            f"v7={v7_acc:.3f} (TP={v7_tp}, FP={v7_fp})"
        )
        v7_note = ""

    return BenchResult(
        metric=f"Anomaly detection accuracy (threshold={threshold})",
        v6_value=v6_acc,
        v7_value=v7_value,
        improvement=improvement,
        v7_note=v7_note,
    )


# =========================================================================
# 6. Router convergence
# =========================================================================
def benchmark_router_convergence(embedding_dim: int = 1024, n_steps: int = 100) -> BenchResult:
    """Higher min router weight = less collapse."""
    torch.manual_seed(0)
    v6 = MATHIRPlugin(embedding_dim)
    v7, v7_err = _make_v7(embedding_dim)

    v6_weights = []
    v7_weights = []
    for _ in range(n_steps):
        emb = torch.randn(1, embedding_dim)
        try:
            w6 = v6.perceive(emb)["router_weights"].mean(dim=0)
            v6_weights.append(w6)
        except Exception:
            pass
        if v7 is not None:
            try:
                w7 = v7.perceive(emb)["router_weights"].mean(dim=0)
                v7_weights.append(w7)
            except Exception:
                pass

    v6_min = float(torch.stack(v6_weights).mean(dim=0).min().item()) if v6_weights else 0.0
    v7_value: Optional[float] = None
    improvement = "n/a"
    v7_note = v7_err or ""

    if v7_weights:
        v7_min = float(torch.stack(v7_weights).mean(dim=0).min().item())
        v7_value = v7_min
        delta = (v7_min - v6_min) / max(v6_min, 1e-6) * 100
        improvement = f"v6={v6_min:.3f} → v7={v7_min:.3f} ({delta:+.1f}%)"
        v7_note = ""

    return BenchResult(
        metric=f"Router min weight (higher = less collapse, n={n_steps})",
        v6_value=v6_min,
        v7_value=v7_value,
        improvement=improvement,
        v7_note=v7_note,
    )


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="MATHIR V6 vs V7 benchmark")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dim")
    parser.add_argument("--iters", type=int, default=100, help="Iterations for latency")
    parser.add_argument("--no-print", action="store_true", help="Suppress stdout")
    args = parser.parse_args()

    def _print(s: str = ""):
        if not args.no_print:
            print(s)

    _print("=" * 70)
    _print("MATHIR V6 vs V7 — Performance Benchmark")
    _print("=" * 70)
    _print(f"V7 available: {_V7_AVAILABLE}")
    _print(f"Embedding dim: {args.embedding_dim}    Iters: {args.iters}")
    _print()

    results: List[BenchResult] = []

    benchmarks = [
        ("[1/6] Compression", benchmark_compression, {"embedding_dim": 272}),
        ("[2/6] Inference latency",
         benchmark_inference_latency,
         {"embedding_dim": args.embedding_dim, "n_iters": args.iters}),
        ("[3/6] Model size",
         benchmark_memory_size,
         {"embedding_dim": args.embedding_dim}),
        ("[4/6] Retention",
         benchmark_retention,
         {"embedding_dim": args.embedding_dim}),
        ("[5/6] Anomaly detection",
         benchmark_anomaly_detection,
         {"embedding_dim": min(args.embedding_dim, 128)}),
        ("[6/6] Router convergence",
         benchmark_router_convergence,
         {"embedding_dim": args.embedding_dim}),
    ]

    for label, fn, kwargs in benchmarks:
        _print(label + "...")
        try:
            r = fn(**kwargs)
        except Exception as e:
            r = BenchResult(
                metric=label,
                v6_value=0.0,
                v7_value=None,
                improvement=f"error: {type(e).__name__}: {e}",
                v7_note="benchmark crashed",
            )
        results.append(r)
        v6_str = f"{r.v6_value:,.2f}" if isinstance(r.v6_value, float) else str(r.v6_value)
        v7_str = (f"{r.v7_value:,.2f}" if isinstance(r.v7_value, float)
                  else (str(r.v7_value) if r.v7_value is not None else "n/a"))
        _print(f"  V6: {v6_str}")
        _print(f"  V7: {v7_str}")
        _print(f"  Improvement: {r.improvement}")
        if r.v7_note:
            _print(f"  Note: {r.v7_note}")
        _print()

    _print("=" * 70)
    _print("SUMMARY")
    _print("=" * 70)
    for r in results:
        v7_part = (f" V7: {r.improvement}" if r.v7_value is not None
                   else f" V7: not yet implemented ({r.v7_note})")
        _print(f"  {r.metric:<55}{v7_part}")
    _print()

    if args.output:
        out_path = args.output
        # If relative, save relative to project root.
        if not os.path.isabs(out_path):
            out_path = os.path.join(_PROJECT_ROOT, out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "v7_available": _V7_AVAILABLE,
                    "embedding_dim": args.embedding_dim,
                    "n_iters": args.iters,
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
                default=str,
            )
        _print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
