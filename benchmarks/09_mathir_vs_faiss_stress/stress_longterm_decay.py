#!/usr/bin/env python3
"""
Stress test #2: "long-term memory" resilience -- does MATHIR's lifecycle
management (decay / auto-promote / consolidate) hurt or help retrieval
quality over time, compared to a static FAISS/BM25 index which never
changes?

This is the key differentiator MATHIR claims over a plain vector index:
FAISS is a static, append-only structure -- it has no concept of "this
memory hasn't been touched in months, deprioritize it" or "these two
memories are near-duplicates, merge them." MATHIR's decay/consolidation is
supposed to keep the *useful* memories accessible while letting cruft fade,
without hurting recall of what still matters. This test measures that
empirically instead of assuming it:

  1. BASELINE: run all test queries against the freshly-inserted corpus,
     record nDCG@10/MRR@10 (this is what run_mathir_comparison.py already
     measured as "5_MATHIR_recall").
  2. AGE: call memory_decay(threshold_days=0) to force every memory through
     one decay cycle immediately (simulating months of inactivity in a
     single call -- an accelerated "long-term" simulation, since we can't
     literally wait months in a benchmark run). Then memory_auto_promote()
     to let anything that's still being recalled bubble back up, mimicking
     what would happen if a real user kept asking about certain topics.
  3. RE-MEASURE: re-run the SAME queries post-decay, record nDCG@10/MRR@10.
  4. CONSOLIDATE: call memory_consolidate(dry_run=False) to merge
     near-duplicate memories (real textbooks have plenty of near-duplicate
     boilerplate/repeated definitions across chapters), then re-measure a
     third time to see if consolidation hurts recall while it reduces
     stored memory count (an efficiency claim, tested for a quality cost).

Honest framing: this is a single accelerated decay/consolidate pass, not a
literal multi-month field trial. It's the best proxy achievable in a
benchmark run, and is reported as such -- not oversold as "N months of real
usage".

Requires: run_mathir_comparison.py already run for --dataset (so the
isolated MATHIR project is populated).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mathir_adapter  # noqa: E402

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

BEIR_DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data"
OUT_DIR = Path(__file__).resolve().parent / "results"


def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    return GenericDataLoader(str(data_path)).load(split="test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    args = parser.parse_args()

    if not mathir_adapter.ping():
        print("ERROR: MATHIR daemon not reachable on port 7338.")
        sys.exit(1)

    corpus, queries, qrels = load_dataset(args.dataset)
    project = f"beir_bench_{args.dataset}"
    adapter = mathir_adapter.MathirBEIR(project=project)

    if not adapter.already_populated(len(corpus)):
        print(f"Project '{project}' not populated -- run run_mathir_comparison.py --dataset {args.dataset} first.")
        sys.exit(1)
    for doc_id, doc in corpus.items():
        adapter.text_to_docid[doc.get("text", "")] = doc_id

    evaluator = EvaluateRetrieval()

    def measure(label):
        t0 = time.time()
        scores = {qid: adapter.search_recall(qtext, top_k=100) for qid, qtext in queries.items()}
        elapsed = time.time() - t0
        ndcg, mrr, recall_cap, _ = evaluator.evaluate(qrels, scores, k_values=[10, 100])
        result = {
            "nDCG@10": ndcg["NDCG@10"], "MRR@10": mrr["MAP@10"],
            "Recall@100": recall_cap["Recall@100"], "time_s": elapsed,
        }
        print(f"  [{label}] {result}")
        return result

    results = {"dataset": args.dataset, "phases": {}}

    print(f"\n=== Phase 1: BASELINE (fresh corpus) — {args.dataset} ===")
    results["phases"]["1_baseline"] = measure("baseline")

    print(f"\n=== Phase 2: decay + auto_promote (accelerated aging simulation) ===")
    decay_resp = adapter.decay(threshold_days=0)
    print(f"  memory_decay response: {json.dumps(decay_resp)[:300]}")
    promote_resp = adapter.auto_promote()
    print(f"  memory_auto_promote response: {json.dumps(promote_resp)[:300]}")
    results["phases"]["2_post_decay"] = measure("post_decay")

    print(f"\n=== Phase 3: consolidate (merge near-duplicates) ===")
    stats_before = mathir_adapter.call("memory_stats", {"project": project})
    consolidate_resp = adapter.consolidate(threshold=0.95, dry_run=False)
    print(f"  memory_consolidate response: {json.dumps(consolidate_resp)[:300]}")
    stats_after = mathir_adapter.call("memory_stats", {"project": project})
    results["memory_count_before_consolidate"] = stats_before.get("total")
    results["memory_count_after_consolidate"] = stats_after.get("total")
    results["phases"]["3_post_consolidate"] = measure("post_consolidate")

    print(f"\n=== Summary for {args.dataset} ===")
    b = results["phases"]["1_baseline"]["nDCG@10"]
    d = results["phases"]["2_post_decay"]["nDCG@10"]
    c = results["phases"]["3_post_consolidate"]["nDCG@10"]
    print(f"  nDCG@10: baseline={b:.4f}  post_decay={d:.4f} (delta={d-b:+.4f})  "
          f"post_consolidate={c:.4f} (delta={c-b:+.4f})")
    print(f"  memory count: {results['memory_count_before_consolidate']} -> {results['memory_count_after_consolidate']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"stress_longterm_decay_{args.dataset}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")
    return results


if __name__ == "__main__":
    main()
