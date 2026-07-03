#!/usr/bin/env python3
"""
Stress test #1: query-noise / "auto-correct" robustness.

Simulates the kind of imperfect input a real user types (typos, dropped
letters, transposed characters) and measures how much retrieval quality
degrades for each approach, compared to the clean-query baseline nDCG@10
already computed by benchmarks/03_vector_search_benchmarks/multi_dataset_efficient.py
(BM25, FAISS-dense) and benchmarks/09_mathir_vs_faiss_stress/run_mathir_comparison.py
(MATHIR recall / MATHIR hybrid).

Hypothesis under test: BM25 is a literal token-overlap method, so it should
degrade sharply on typo'd queries (a misspelled keyword just doesn't match).
Dense/semantic methods (FAISS embeddings, MATHIR's embedder) encode meaning,
not exact tokens, so they should be comparatively robust -- this is the
"auto-correct" claim being tested empirically, not asserted.

No LLM calls needed -- typo injection is a deterministic character-level
perturbation (seeded, reproducible), not an LLM rewrite.

Usage:
    python stress_typo_robustness.py --dataset scifact --n-queries 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mathir_adapter  # noqa: E402

import numpy as np
import torch
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BEIR_DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data"
OUT_DIR = Path(__file__).resolve().parent / "results"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Auto-detect GPU, same convention as multi_dataset_efficient.py's DEVICE --
# this was hardcoded to "cpu" before, which is slow for encoding a full BEIR
# corpus; fixed to use CUDA whenever it's available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KEYBOARD_NEIGHBORS = {
    "a": "qs", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh",
    "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm",
    "o": "ip", "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry", "u": "yi",
    "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}


def inject_typos(text: str, rng: random.Random, n_edits: int = 2) -> str:
    """Character-level noise: swap adjacent chars, drop a char, or hit a
    keyboard-neighbor key, applied at a few random positions."""
    chars = list(text)
    for _ in range(n_edits):
        if len(chars) < 4:
            break
        i = rng.randint(0, len(chars) - 2)
        op = rng.choice(["swap", "drop", "neighbor"])
        if op == "swap":
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop":
            del chars[i]
        elif op == "neighbor":
            c = chars[i].lower()
            if c in KEYBOARD_NEIGHBORS:
                chars[i] = rng.choice(KEYBOARD_NEIGHBORS[c])
    return "".join(chars)


def load_dataset(name):
    data_path = BEIR_DATA_DIR / name
    nested = data_path / name
    if nested.exists():
        data_path = nested
    return GenericDataLoader(str(data_path)).load(split="test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not mathir_adapter.ping():
        print("ERROR: MATHIR daemon not reachable on port 7338.")
        sys.exit(1)

    corpus, queries, qrels = load_dataset(args.dataset)
    rng = random.Random(args.seed)

    q_ids = list(queries.keys())
    rng.shuffle(q_ids)
    q_ids = q_ids[: min(args.n_queries, len(q_ids))]
    qrels_sub = {qid: qrels[qid] for qid in q_ids if qid in qrels}

    clean_queries = {qid: queries[qid] for qid in q_ids}
    noisy_queries = {qid: inject_typos(queries[qid], rng, n_edits=2) for qid in q_ids}

    print(f"Dataset: {args.dataset}, {len(q_ids)} queries perturbed with 2 char-level edits each.")
    print("Example clean -> noisy:")
    for qid in q_ids[:5]:
        print(f"  '{clean_queries[qid]}'  ->  '{noisy_queries[qid]}'")

    evaluator = EvaluateRetrieval()
    results = {"dataset": args.dataset, "n_queries": len(q_ids), "approaches": {}}

    def run_eval(label, scores):
        ndcg, mrr, recall_cap, _ = evaluator.evaluate(qrels_sub, scores, k_values=[10])
        return {"nDCG@10": ndcg["NDCG@10"], "MRR@10": mrr["MAP@10"]}

    # --- BM25 ---
    print("\n--- BM25 ---")
    doc_ids = list(corpus.keys())
    tokenized_corpus = [corpus[d].get("text", "").lower().split() for d in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    def bm25_search(qtexts):
        out = {}
        for qid, qtext in qtexts.items():
            scores = bm25.get_scores(qtext.lower().split())
            top = np.argsort(scores)[::-1][:100]
            out[qid] = {doc_ids[i]: float(scores[i]) for i in top}
        return out

    bm25_clean = run_eval("bm25_clean", bm25_search(clean_queries))
    bm25_noisy = run_eval("bm25_noisy", bm25_search(noisy_queries))
    results["approaches"]["BM25"] = {"clean": bm25_clean, "noisy": bm25_noisy}
    print(f"  clean: {bm25_clean}, noisy: {bm25_noisy}")

    # --- FAISS dense ---
    print(f"\n--- FAISS dense (BAAI/bge-base-en-v1.5, device={DEVICE}) ---")
    import faiss
    encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)
    doc_texts = [corpus[d].get("text", "") for d in doc_ids]
    corpus_embs = encoder.encode(doc_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(corpus_embs.shape[1])
    if DEVICE == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(corpus_embs.astype(np.float32))

    def faiss_search(qtexts):
        qids = list(qtexts.keys())
        qembs = encoder.encode([qtexts[q] for q in qids], convert_to_numpy=True)
        scores, indices = index.search(qembs.astype(np.float32), 100)
        out = {}
        for i, qid in enumerate(qids):
            out[qid] = {doc_ids[indices[i][j]]: float(scores[i][j]) for j in range(100) if indices[i][j] < len(doc_ids)}
        return out

    faiss_clean = run_eval("faiss_clean", faiss_search(clean_queries))
    faiss_noisy = run_eval("faiss_noisy", faiss_search(noisy_queries))
    results["approaches"]["FAISS_dense"] = {"clean": faiss_clean, "noisy": faiss_noisy}
    print(f"  clean: {faiss_clean}, noisy: {faiss_noisy}")

    # --- MATHIR recall (assumes corpus already inserted by run_mathir_comparison.py) ---
    print("\n--- MATHIR memory_recall ---")
    project = f"beir_bench_{args.dataset}"
    adapter = mathir_adapter.MathirBEIR(project=project)
    if not adapter.already_populated(len(corpus)):
        print(f"  Project '{project}' not populated yet -- run run_mathir_comparison.py first. Skipping MATHIR.")
    else:
        for doc_id, doc in corpus.items():
            adapter.text_to_docid[doc.get("text", "")] = doc_id

        def mathir_search(qtexts):
            return {qid: adapter.search_recall(qtext, top_k=100) for qid, qtext in qtexts.items()}

        mathir_clean = run_eval("mathir_clean", mathir_search(clean_queries))
        mathir_noisy = run_eval("mathir_noisy", mathir_search(noisy_queries))
        results["approaches"]["MATHIR_recall"] = {"clean": mathir_clean, "noisy": mathir_noisy}
        print(f"  clean: {mathir_clean}, noisy: {mathir_noisy}")

    # --- Degradation summary ---
    print(f"\n=== Robustness summary (nDCG@10 drop from clean -> noisy query) for {args.dataset} ===")
    for name, r in results["approaches"].items():
        drop = r["clean"]["nDCG@10"] - r["noisy"]["nDCG@10"]
        pct = (drop / r["clean"]["nDCG@10"] * 100) if r["clean"]["nDCG@10"] > 0 else 0.0
        print(f"  {name:16s}: clean={r['clean']['nDCG@10']:.4f}  noisy={r['noisy']['nDCG@10']:.4f}  drop={drop:.4f} ({pct:.1f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"stress_typo_robustness_{args.dataset}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")
    return results


if __name__ == "__main__":
    main()
