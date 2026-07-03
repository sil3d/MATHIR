#!/usr/bin/env python3
"""
Benchmark cross-encoder reranking on top of e5-small retrieval.

Measures: does reranking the top-30 candidates down to top-10 improve
hit@10 and nDCG@10, especially for formula-type queries?

Uses the same fluid mechanics dataset as compare_e5_large.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _env  # noqa
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mathir_mcp" / "mathir_lib"))

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "05_test_data" / "beir_data" / "fluid_mechanics" / "fluid_mechanics"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "06_results" / "current"
FORMULA_KEYWORDS = [
    "equation", "coefficient", "formula", "number", "ratio", "factor", "definition",
]


def load_corpus():
    corpus = {}
    with (DATA_DIR / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            title = d.get("title", "")
            text = d.get("text", "")
            corpus[d["_id"]] = f"[{title}] {text}" if title else text
    return corpus


def load_queries_and_qrels():
    queries = {}
    with (DATA_DIR / "queries.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    qrels = {}
    with (DATA_DIR / "qrels" / "test.tsv").open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, doc_id = parts[0], parts[1]
                qrels.setdefault(qid, []).append(doc_id)
    return queries, qrels


def has_formula_kw(q):
    return any(k in q.lower() for k in FORMULA_KEYWORDS)


def ndcg_at_k(ranked_ids, gold_ids, k=10):
    gold = set(gold_ids)
    dcg = sum(1.0 / np.log2(i + 2) for i, rid in enumerate(ranked_ids[:k]) if rid in gold)
    ideal = sorted([1.0] * min(len(gold), k), reverse=True)
    idcg = sum(1.0 / np.log2(i + 2) for i, _ in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    from sentence_transformers import SentenceTransformer, CrossEncoder

    corpus = load_corpus()
    queries, qrels = load_queries_and_qrels()
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    formula_qids = [qid for qid in queries if has_formula_kw(queries[qid])]
    other_qids = [qid for qid in queries if not has_formula_kw(queries[qid])]
    all_qids = list(queries.keys())
    print(f"Corpus: {len(corpus_texts)} chunks | Queries: {len(all_qids)} total "
          f"({len(formula_qids)} formula, {len(other_qids)} other)\n")

    # --- Step 1: encode corpus with e5-small ---
    print("Loading e5-small embedder...")
    embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu",
                                   model_kwargs={"low_cpu_mem_usage": False})
    print("Encoding corpus...")
    prefixed_corpus = ["passage: " + t for t in corpus_texts]
    corpus_embs = embedder.encode(prefixed_corpus, convert_to_numpy=True,
                                  show_progress_bar=True, batch_size=64)
    corpus_norms = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)

    # --- Step 2: load cross-encoder ---
    print("\nLoading cross-encoder (ms-marco-MiniLM-L-6-v2)...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Cross-encoder loaded.\n")

    TOP_K_RETRIEVE = 30  # first-pass retrieval depth
    TOP_K_FINAL = 10     # final output

    results_baseline = {"formula": [], "other": [], "all": []}
    results_reranked = {"formula": [], "other": [], "all": []}
    rerank_latencies = []

    for i, qid in enumerate(all_qids):
        q_text = queries[qid]
        q_emb = embedder.encode("query: " + q_text, convert_to_numpy=True)
        q_norm = q_emb / np.linalg.norm(q_emb)

        # First-pass: cosine similarity top-30
        sims = corpus_norms @ q_norm
        top_idx = np.argsort(-sims)[:TOP_K_RETRIEVE]
        top_ids = [corpus_ids[j] for j in top_idx]
        top_texts = [corpus_texts[j] for j in top_idx]

        gold = set(qrels.get(qid, []))

        # Baseline: top-10 from cosine alone
        baseline_ids = top_ids[:TOP_K_FINAL]
        baseline_hit = 1 if gold & set(baseline_ids) else 0
        baseline_ndcg = ndcg_at_k(baseline_ids, gold, TOP_K_FINAL)

        # Reranked: cross-encoder on top-30, take top-10
        t0 = time.perf_counter()
        pairs = [[q_text, t] for t in top_texts]
        ce_scores = cross_encoder.predict(pairs)
        rerank_ms = (time.perf_counter() - t0) * 1000
        rerank_latencies.append(rerank_ms)

        reranked_order = np.argsort(-ce_scores)
        reranked_ids = [top_ids[j] for j in reranked_order[:TOP_K_FINAL]]
        reranked_hit = 1 if gold & set(reranked_ids) else 0
        reranked_ndcg = ndcg_at_k(reranked_ids, gold, TOP_K_FINAL)

        cat = "formula" if has_formula_kw(q_text) else "other"
        results_baseline[cat].append({"hit": baseline_hit, "ndcg": baseline_ndcg})
        results_baseline["all"].append({"hit": baseline_hit, "ndcg": baseline_ndcg})
        results_reranked[cat].append({"hit": reranked_hit, "ndcg": reranked_ndcg})
        results_reranked["all"].append({"hit": reranked_hit, "ndcg": reranked_ndcg})

        status = "IMPROVED" if reranked_hit > baseline_hit else ("SAME" if reranked_hit == baseline_hit else "WORSE")
        if (i + 1) % 10 == 0 or status == "IMPROVED":
            print(f"  [{i+1}/{len(all_qids)}] {qid} ({cat}) "
                  f"baseline={baseline_hit} rerank={reranked_hit} "
                  f"ndcg {baseline_ndcg:.3f}->{reranked_ndcg:.3f} "
                  f"[{rerank_ms:.0f}ms] {status}")

    print("\n" + "=" * 70)
    print("RESULTS: e5-small cosine top-10 vs e5-small + cross-encoder rerank")
    print("=" * 70)

    output = {}
    for cat in ["formula", "other", "all"]:
        b = results_baseline[cat]
        r = results_reranked[cat]
        n = len(b)
        b_hit = sum(x["hit"] for x in b) / n if n else 0
        r_hit = sum(x["hit"] for x in r) / n if n else 0
        b_ndcg = np.mean([x["ndcg"] for x in b]) if n else 0
        r_ndcg = np.mean([x["ndcg"] for x in r]) if n else 0
        delta_hit = (r_hit - b_hit) * 100
        delta_ndcg = r_ndcg - b_ndcg

        print(f"\n  {cat.upper()} ({n} queries):")
        print(f"    Baseline  hit@10={b_hit*100:.1f}%  nDCG@10={b_ndcg:.4f}")
        print(f"    Reranked  hit@10={r_hit*100:.1f}%  nDCG@10={r_ndcg:.4f}")
        print(f"    Delta     hit@10={delta_hit:+.1f}pp  nDCG@10={delta_ndcg:+.4f}")

        output[cat] = {
            "n": n,
            "baseline_hit10": round(b_hit, 4),
            "reranked_hit10": round(r_hit, 4),
            "baseline_ndcg10": round(float(b_ndcg), 4),
            "reranked_ndcg10": round(float(r_ndcg), 4),
            "delta_hit_pp": round(delta_hit, 2),
            "delta_ndcg": round(float(delta_ndcg), 4),
        }

    avg_rerank_ms = np.mean(rerank_latencies) if rerank_latencies else 0
    print(f"\n  Avg rerank latency: {avg_rerank_ms:.1f}ms per query (top-{TOP_K_RETRIEVE} candidates)")

    output["rerank_latency_ms"] = round(float(avg_rerank_ms), 1)
    output["model"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    output["retrieval_depth"] = TOP_K_RETRIEVE
    output["final_k"] = TOP_K_FINAL

    out_path = OUT_DIR / "rerank_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
