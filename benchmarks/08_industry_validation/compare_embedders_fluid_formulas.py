#!/usr/bin/env python3
"""
Targeted follow-up to the fluid mechanics benchmark finding: formula/
definition questions hit at 41.9% vs 70.0% for procedural questions with
the current default (e5-small). Tests whether a DIFFERENT embedder
(bge-base-en-v1.5, which scored higher in absolute BEIR nDCG@10 earlier
this session -- 0.7376 on scifact vs e5-small's 0.6770) specifically
helps on the formula-heavy subset where e5-small underperforms.

Pure local test (sentence-transformers + numpy brute-force cosine, no
daemon), mirroring the earlier embedder-comparison methodology.
"""
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "05_test_data" / "beir_data" / "fluid_mechanics" / "fluid_mechanics"
FORMULA_KEYWORDS = ['equation', 'coefficient', 'formula', 'number', 'ratio', 'factor', 'definition']


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
        next(f)  # header: query-id, corpus-id, score
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, doc_id, _score = parts[0], parts[1], parts[2]
                qrels.setdefault(qid, []).append(doc_id)
    return queries, qrels


def has_formula_kw(q):
    return any(k in q.lower() for k in FORMULA_KEYWORDS)


def evaluate(model, model_name, corpus_ids, corpus_embs, queries, qrels, subset_qids, k=10):
    hits = 0
    for qid in subset_qids:
        q_emb = model.encode(queries[qid], convert_to_numpy=True, show_progress_bar=False)
        q_norm = q_emb / np.linalg.norm(q_emb)
        corpus_norm = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)
        sims = corpus_norm @ q_norm
        top_idx = np.argsort(-sims)[:k]
        top_ids = {corpus_ids[i] for i in top_idx}
        gold = set(qrels.get(qid, []))
        if gold & top_ids:
            hits += 1
    return hits / len(subset_qids) if subset_qids else 0.0


def main():
    from sentence_transformers import SentenceTransformer

    corpus = load_corpus()
    queries, qrels = load_queries_and_qrels()
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    formula_qids = [qid for qid in queries if has_formula_kw(queries[qid])]
    other_qids = [qid for qid in queries if not has_formula_kw(queries[qid])]
    print(f"{len(formula_qids)} formula-related, {len(other_qids)} other questions")

    models = {
        "e5_small (current default)": ("intfloat/multilingual-e5-small", "passage: ", "query: "),
        "bge_base": ("BAAI/bge-base-en-v1.5", "", ""),
    }

    results = {}
    for name, (model_id, passage_prefix, query_prefix) in models.items():
        print(f"\n[{name}] loading {model_id} ...", flush=True)
        model = SentenceTransformer(model_id, device="cpu")
        print(f"[{name}] encoding {len(corpus_texts)} corpus chunks ...", flush=True)
        prefixed_texts = [passage_prefix + t for t in corpus_texts]
        corpus_embs = model.encode(prefixed_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)

        # Apply query prefix by wrapping queries dict access at eval time
        class PrefixedModel:
            def __init__(self, m, prefix):
                self.m = m
                self.prefix = prefix

            def encode(self, text, **kw):
                return self.m.encode(self.prefix + text, **kw)

        pm = PrefixedModel(model, query_prefix)

        formula_acc = evaluate(pm, name, corpus_ids, corpus_embs, queries, qrels, formula_qids)
        other_acc = evaluate(pm, name, corpus_ids, corpus_embs, queries, qrels, other_qids)
        results[name] = {"formula_hit_rate": formula_acc, "other_hit_rate": other_acc}
        print(f"[{name}] formula hit@10={formula_acc*100:.1f}%  other hit@10={other_acc*100:.1f}%")
        del model

    print("\n" + "=" * 60)
    print("Embedder comparison on fluid mechanics formula subset")
    print("=" * 60)
    for name, r in results.items():
        print(f"{name:30s} formula={r['formula_hit_rate']*100:5.1f}%  other={r['other_hit_rate']*100:5.1f}%")

    out = Path(__file__).resolve().parent.parent / "06_results" / "current" / "fluid_formula_embedder_comparison.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
