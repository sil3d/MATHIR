#!/usr/bin/env python3
"""
e5-small (384d) vs e5-large-v2 (1024d) on the fluid mechanics benchmark.

Goal: measure whether 2.7x more dimensions actually closes the formula gap
(41.9% formula vs 70.0% procedural seen with e5-small).

Also measures: latency per query, total encoding time, nDCG@10.
"""
import json
import time
from pathlib import Path

import numpy as np

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


def evaluate(encode_fn, corpus_ids, corpus_embs, queries, qrels, subset_qids, k=10):
    hits = 0
    ndcgs = []
    latencies = []
    for qid in subset_qids:
        t0 = time.perf_counter()
        q_emb = encode_fn(queries[qid])
        latencies.append((time.perf_counter() - t0) * 1000)

        q_norm = q_emb / np.linalg.norm(q_emb)
        c_norm = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)
        sims = c_norm @ q_norm
        top_idx = np.argsort(-sims)[:k]
        top_ids = [corpus_ids[i] for i in top_idx]
        gold = set(qrels.get(qid, []))
        if gold & set(top_ids):
            hits += 1
        ndcgs.append(ndcg_at_k(top_ids, gold, k))

    return {
        "hit_rate": hits / len(subset_qids) if subset_qids else 0.0,
        "ndcg10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "avg_query_ms": float(np.mean(latencies)) if latencies else 0.0,
        "n": len(subset_qids),
    }


def main():
    from sentence_transformers import SentenceTransformer

    corpus = load_corpus()
    queries, qrels = load_queries_and_qrels()
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    formula_qids = [qid for qid in queries if has_formula_kw(queries[qid])]
    other_qids = [qid for qid in queries if not has_formula_kw(queries[qid])]
    all_qids = list(queries.keys())
    print(f"Corpus: {len(corpus_texts)} chunks | Queries: {len(all_qids)} total "
          f"({len(formula_qids)} formula, {len(other_qids)} other)\n")

    models_config = {
        "e5-small (384d, current)": {
            "model_id": "intfloat/multilingual-e5-small",
            "passage_prefix": "passage: ",
            "query_prefix": "query: ",
        },
        "e5-large-v2 (1024d)": {
            "model_id": "intfloat/e5-large-v2",
            "passage_prefix": "passage: ",
            "query_prefix": "query: ",
        },
    }

    all_results = {}

    for name, cfg in models_config.items():
        model_id = cfg["model_id"]
        pp, qp = cfg["passage_prefix"], cfg["query_prefix"]

        print(f"{'='*60}")
        print(f"[{name}] Loading {model_id} ...")
        t_load = time.perf_counter()
        model = SentenceTransformer(model_id, device="cpu",
                                    model_kwargs={"low_cpu_mem_usage": False})
        load_s = time.perf_counter() - t_load
        dim = model.get_sentence_embedding_dimension()
        print(f"[{name}] dim={dim}, loaded in {load_s:.1f}s")

        print(f"[{name}] Encoding {len(corpus_texts)} corpus chunks ...")
        t_enc = time.perf_counter()
        prefixed = [pp + t for t in corpus_texts]
        corpus_embs = model.encode(prefixed, convert_to_numpy=True,
                                   show_progress_bar=True, batch_size=64)
        enc_s = time.perf_counter() - t_enc
        print(f"[{name}] Corpus encoded in {enc_s:.1f}s")

        def make_encode_fn(m, prefix):
            def fn(text):
                return m.encode(prefix + text, convert_to_numpy=True, show_progress_bar=False)
            return fn

        encode_fn = make_encode_fn(model, qp)

        r_formula = evaluate(encode_fn, corpus_ids, corpus_embs, queries, qrels, formula_qids)
        r_other = evaluate(encode_fn, corpus_ids, corpus_embs, queries, qrels, other_qids)
        r_all = evaluate(encode_fn, corpus_ids, corpus_embs, queries, qrels, all_qids)

        all_results[name] = {
            "dim": dim,
            "model_id": model_id,
            "load_time_s": round(load_s, 2),
            "corpus_encode_s": round(enc_s, 2),
            "formula": r_formula,
            "other": r_other,
            "all": r_all,
        }

        print(f"\n[{name}] RESULTS:")
        print(f"  Formula  hit@10={r_formula['hit_rate']*100:.1f}%  nDCG@10={r_formula['ndcg10']:.4f}  ({r_formula['n']} q)")
        print(f"  Other    hit@10={r_other['hit_rate']*100:.1f}%  nDCG@10={r_other['ndcg10']:.4f}  ({r_other['n']} q)")
        print(f"  ALL      hit@10={r_all['hit_rate']*100:.1f}%  nDCG@10={r_all['ndcg10']:.4f}  ({r_all['n']} q)")
        print(f"  Avg query latency: {r_all['avg_query_ms']:.1f}ms")
        print()

        del model, corpus_embs

    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':35s} {'dim':>5s} {'Formula%':>9s} {'Other%':>8s} {'All%':>7s} {'nDCG@10':>8s} {'Lat(ms)':>8s}")
    for name, r in all_results.items():
        print(f"{name:35s} {r['dim']:5d} "
              f"{r['formula']['hit_rate']*100:8.1f}% "
              f"{r['other']['hit_rate']*100:7.1f}% "
              f"{r['all']['hit_rate']*100:6.1f}% "
              f"{r['all']['ndcg10']:8.4f} "
              f"{r['all']['avg_query_ms']:7.1f}")

    gap_small = all_results["e5-small (384d, current)"]["other"]["hit_rate"] - all_results["e5-small (384d, current)"]["formula"]["hit_rate"]
    gap_large = all_results["e5-large-v2 (1024d)"]["other"]["hit_rate"] - all_results["e5-large-v2 (1024d)"]["formula"]["hit_rate"]
    print(f"\nFormula gap (other - formula): e5-small={gap_small*100:.1f}pp  e5-large={gap_large*100:.1f}pp")

    out = OUT_DIR / "e5_small_vs_large_fluid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
