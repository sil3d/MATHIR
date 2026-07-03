#!/usr/bin/env python3
"""
Real MATHIR-vs-FAISS/BM25/hybrid BEIR comparison -- the proper redo of the
old MATHIR test the user flagged as using MATHIR without its full
capabilities and being badly coded (see benchmarks/_deprecated/v7/
test_episodic_memory_online_learning.py: custom hand-rolled nDCG, not the
real beir evaluator).

This script:
  1. Loads each BEIR dataset (scifact, nfcorpus, arguana already have real,
     previously-computed FAISS-only/BM25-only/hybrid-RRF/hybrid+CE numbers
     in benchmarks/06_results/archive/multi_dataset_efficient_results.json
     -- fluid_mechanics will join this list once its queries.jsonl/qrels
     exist, see benchmarks/07_utilities/generate_fluid_mechanics_queries.py).
  2. Inserts the corpus into an ISOLATED MATHIR project via the real HTTP
     daemon (see mathir_adapter.py docstring for why this can never pollute
     live agent memory) -- using MATHIR's real embedder, real tiers, real
     anomaly/risk checks, i.e. its actual full capability, not a
     reimplementation.
  3. Runs every test-split query through MATHIR's own memory_recall (pure
     semantic recall, what a live agent calls) AND memory_hybrid_search
     (MATHIR's own vector+BM25 RRF fusion), scoring both with the SAME real
     `beir.retrieval.evaluation.EvaluateRetrieval` used by
     multi_dataset_efficient.py -- so numbers are directly comparable,
     apples to apples, not a custom metric.
  4. Merges MATHIR's numbers into the existing per-dataset results structure
     and writes the combined comparison to
     benchmarks/09_mathir_vs_faiss_stress/results/mathir_vs_faiss_results.json

Idempotent: re-running skips re-inserting a dataset's corpus into MATHIR if
the isolated project already reports >= the expected document count.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mathir_adapter  # noqa: E402

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

BEIR_DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data"
EXISTING_RESULTS = Path(__file__).resolve().parent.parent / "06_results" / "archive" / "multi_dataset_efficient_results.json"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_FILE = OUT_DIR / "mathir_vs_faiss_results.json"

DATASETS = ["scifact", "nfcorpus", "arguana", "fluid_mechanics"]
TOP_K = 100


def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    if not data_path.exists():
        return None
    return GenericDataLoader(str(data_path)).load(split="test")


def evaluate(evaluator, qrels, scores, label, dataset_results, time_s):
    ndcg, mrr, recall_cap, recall = evaluator.evaluate(qrels, scores, k_values=[10, 100])
    ndcg10 = ndcg["NDCG@10"]
    mrr10 = mrr["MAP@10"]
    rec100 = recall_cap["Recall@100"]
    print(f"  [{label}] nDCG@10: {ndcg10:.4f}, MRR@10: {mrr10:.4f}, Recall@100: {rec100:.4f}, Time: {time_s:.2f}s")
    dataset_results[label] = {"nDCG@10": ndcg10, "MRR@10": mrr10, "Recall@100": rec100, "time_s": time_s}


def main():
    if not mathir_adapter.ping():
        print("ERROR: MATHIR daemon not reachable on port 7338. Start it before running this benchmark.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if EXISTING_RESULTS.exists():
        with open(EXISTING_RESULTS, "r", encoding="utf-8") as f:
            combined = json.load(f)
    else:
        combined = {"metadata": {"model": "BAAI/bge-base-en-v1.5", "datasets": [], "device": "unknown"}, "results": {}}

    combined.setdefault("results", {})
    combined["metadata"]["mathir_added"] = True

    evaluator = EvaluateRetrieval()

    for dataset in DATASETS:
        print(f"\n{'='*60}\nDATASET: {dataset}\n{'='*60}")
        loaded = load_dataset(dataset)
        if loaded is None:
            print(f"  SKIP: {dataset} not found on disk.")
            continue
        corpus, queries, qrels = loaded
        if not queries or not qrels:
            print(f"  SKIP: {dataset} has no queries/qrels yet (queries.jsonl/qrels/test.tsv missing or empty). "
                  f"This is expected for fluid_mechanics until an LLM backend with credits generates them "
                  f"(see benchmarks/07_utilities/generate_fluid_mechanics_queries.py).")
            continue
        print(f"  Corpus: {len(corpus)}, Queries: {len(queries)}, Qrels: {len(qrels)}")

        project = f"beir_bench_{dataset}"
        adapter = mathir_adapter.MathirBEIR(project=project)

        if adapter.already_populated(len(corpus)):
            print(f"  Project '{project}' already has >= {len(corpus)} memories -- skipping insert, rebuilding text->id map...")
            for doc_id, doc in corpus.items():
                adapter.text_to_docid[doc.get("text", "")] = doc_id
        else:
            print(f"  Inserting {len(corpus)} docs into isolated MATHIR project '{project}'...")
            adapter.insert_corpus(corpus)

        dataset_results = combined["results"].setdefault(dataset, {})

        print("  MATHIR memory_recall (pure semantic recall)...")
        start = time.time()
        recall_scores = {qid: adapter.search_recall(qtext, top_k=TOP_K) for qid, qtext in queries.items()}
        recall_time = time.time() - start
        evaluate(evaluator, qrels, recall_scores, "5_MATHIR_recall", dataset_results, recall_time)

        print("  MATHIR memory_hybrid_search (vector+BM25 RRF, MATHIR's own fusion)...")
        start = time.time()
        hybrid_scores = {qid: adapter.search_hybrid(qtext, top_k=TOP_K) for qid, qtext in queries.items()}
        hybrid_time = time.time() - start
        evaluate(evaluator, qrels, hybrid_scores, "6_MATHIR_hybrid", dataset_results, hybrid_time)

        if dataset not in combined["metadata"]["datasets"]:
            combined["metadata"]["datasets"].append(dataset)

        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"  Saved running results to {OUT_FILE}")

    print(f"\n\n{'='*60}\nFINAL COMBINED RESULTS\n{'='*60}")
    print(json.dumps(combined, indent=2))
    return combined


if __name__ == "__main__":
    main()
