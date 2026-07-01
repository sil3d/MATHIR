#!/usr/bin/env python3
"""
Download HotpotQA (distractor setting) for MATHIR's industrial multi-hop
retrieval benchmark.

HotpotQA is THE standard academic multi-hop QA benchmark (Yang et al., EMNLP
2018), used by every GraphRAG / multi-hop retrieval paper. In the distractor
setting each question comes with 10 paragraphs: exactly 2 "gold" supporting
paragraphs that BOTH must be found to answer, plus 8 distractors. This is the
ideal test for whether a graph-based retriever (PPR-LTE) beats flat retrieval
(hybrid_search): flat search reliably finds the paragraph most similar to the
question, but often misses the SECOND gold paragraph that's linked only via a
bridge entity, not by direct similarity to the question.

We filter to `type == "bridge"` AND `level == "hard"` questions -- the subset
where multi-hop reasoning genuinely matters most (bridge questions require
chaining through an entity; "comparison" questions can sometimes be answered
from one paragraph, so they're less discriminating for a graph retriever).

Output: benchmarks/10_multihop/data/hotpotqa_bridge_hard.json
(a list of {id, question, answer, gold_titles: [t1, t2], paragraphs: [{title, text}, ...]})

Usage:
    python download_hotpotqa.py --n 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_FILE = OUT_DIR / "hotpotqa_bridge_hard.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200,
                        help="Number of bridge+hard questions to keep (deterministic, first-N after filtering).")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading HotpotQA distractor validation split (streaming)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)

    kept = []
    scanned = 0
    for ex in ds:
        scanned += 1
        if ex.get("type") != "bridge" or ex.get("level") != "hard":
            continue

        ctx = ex["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]  # list of list-of-sentences, parallel to titles
        paragraphs = []
        for t, sents in zip(titles, sentences):
            paragraphs.append({"title": t, "text": " ".join(sents).strip()})

        gold_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))  # unique, order-preserving
        # Only keep well-formed items: exactly the 2 gold titles must be present
        # among the 10 paragraph titles (HotpotQA guarantees this, but verify).
        para_titles = {p["title"] for p in paragraphs}
        if not all(g in para_titles for g in gold_titles):
            continue
        if len(gold_titles) != 2:
            continue  # multi-hop-2 is the canonical HotpotQA structure

        kept.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_titles": gold_titles,
            "paragraphs": paragraphs,
        })
        if len(kept) >= args.n:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"Scanned {scanned} questions, kept {len(kept)} bridge+hard multi-hop questions.")
    print(f"Wrote {OUT_FILE}")
    if kept:
        ex = kept[0]
        print("\nExample:")
        print("  Q:", ex["question"])
        print("  A:", ex["answer"])
        print("  gold titles:", ex["gold_titles"])
        print("  #paragraphs:", len(ex["paragraphs"]))


if __name__ == "__main__":
    main()
