#!/usr/bin/env python3
"""
Compare embedding models on the HotpotQA multi-hop retrieval task, purely
locally (no MATHIR daemon involved, no shared state touched) -- mirrors the
methodology of benchmarks/07_utilities/compare_embedding_models.py from the
earlier BEIR investigation.

WHY LOCAL, NOT VIA THE DAEMON: the live MATHIR daemon's embedder is a shared
resource another agent session is actively using for the fluid-mechanics
benchmark track. Swapping its config to test a different embedder would risk
disrupting that work and re-triggering the project-state confounds already
documented this session. This script encodes the HotpotQA corpus/queries
directly with sentence-transformers and does brute-force cosine ranking in
numpy -- mathematically identical to what MATHIR's sqlite-vec exact search
does (already proven equivalent earlier this session), so the comparison is
valid without touching the daemon at all.

Compares the current MATHIR default (paraphrase-multilingual-MiniLM-L12-v2,
384d, paraphrase/STS-trained) against intfloat/multilingual-e5-small (384d,
retrieval-trained, needs "query: "/"passage: " prefixes) -- the one
embedder swap shown to help on some BEIR corpora earlier this session,
never tested on genuine multi-hop retrieval until now.

Usage:
    python compare_embedders_hotpot.py --n 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DATA_FILE = _HERE / "data" / "hotpotqa_bridge_hard.json"

K_VALUES = [2, 4, 8]

MODELS = {
    "default_minilm": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5_small": {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


def evaluate_model(model, model_cfg: dict, items: list) -> dict:
    """Encode all paragraphs+questions for this model, rank by cosine, score both_gold@k."""
    per_k_hits = {k: 0 for k in K_VALUES}
    per_k_gold_sum = {k: 0 for k in K_VALUES}
    n = len(items)

    for item in items:
        paragraphs = item["paragraphs"]
        gold_titles = set(item["gold_titles"])
        texts = [model_cfg["passage_prefix"] + p["text"] for p in paragraphs]
        gold_flags = [p["title"] in gold_titles for p in paragraphs]

        para_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        q_emb = model.encode(model_cfg["query_prefix"] + item["question"],
                             convert_to_numpy=True, show_progress_bar=False)

        # Cosine similarity via normalized dot product (mathematically
        # identical to MATHIR's sqlite-vec exact search).
        para_norm = para_embs / np.linalg.norm(para_embs, axis=1, keepdims=True)
        q_norm = q_emb / np.linalg.norm(q_emb)
        sims = para_norm @ q_norm
        ranked_idx = np.argsort(-sims)

        for k in K_VALUES:
            top_idx = ranked_idx[:k]
            n_gold_in_topk = sum(1 for i in top_idx if gold_flags[i])
            if n_gold_in_topk == 2:
                per_k_hits[k] += 1
            per_k_gold_sum[k] += n_gold_in_topk

    return {
        f"both_gold@{k}": per_k_hits[k] / n for k in K_VALUES
    } | {
        f"mean_gold_recall@{k}": per_k_gold_sum[k] / (2 * n) for k in K_VALUES
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    with DATA_FILE.open("r", encoding="utf-8") as f:
        items = json.load(f)[:args.n]

    print(f"Evaluating {len(items)} HotpotQA bridge+hard questions, local brute-force cosine, no daemon.")

    from sentence_transformers import SentenceTransformer

    results = {}
    for key, cfg in MODELS.items():
        print(f"\n[{key}] loading {cfg['name']} ...", flush=True)
        model = SentenceTransformer(cfg["name"], device="cpu")
        print(f"[{key}] encoding + scoring {len(items)} questions ...", flush=True)
        results[key] = evaluate_model(model, cfg, items)
        del model

    print("\n" + "=" * 70)
    print(f"Embedder comparison on HotpotQA multi-hop (n={len(items)}, local, no daemon)")
    print("=" * 70)
    header = f"{'model':<16}" + "".join(f"both@{k:<7}" for k in K_VALUES)
    print(header)
    print("-" * 70)
    for key, r in results.items():
        row = f"{key:<16}"
        for k in K_VALUES:
            row += f"{r[f'both_gold@{k}']*100:>7.1f}%"
        print(row)
    print("=" * 70)

    out_path = _HERE / "results" / "embedder_comparison_hotpot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"n_questions": len(items), "results": results}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
