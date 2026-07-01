#!/usr/bin/env python3
"""
Industrial multi-hop retrieval benchmark for MATHIR on HotpotQA (distractor,
bridge+hard subset). Purely retrieval-scored -- NO LLM API calls, NO judge,
so it's free and infinitely reproducible.

The question this benchmark answers: does MATHIR's graph-based retriever
(PPR-LTE, which treats the link graph as the search substrate) beat flat
retrieval (hybrid_search / plain vector recall) on genuine multi-hop
questions, where the answer requires finding TWO gold paragraphs and the
second is linked to the first only via a bridge entity?

Protocol per question:
  1. Ingest all 10 paragraphs as isolated MATHIR memories (project per question).
  2. build_links -> MATHIR builds its cosine-similarity link graph over the 10.
  3. Run each retriever; for each, check whether the top-k retrieved memories
     include BOTH gold paragraphs.

Headline metric: both_gold@k -- the fraction of questions where BOTH gold
paragraphs are in the top-k. This is the metric that actually matters for
multi-hop: retrieving only one gold paragraph means the question is
unanswerable, so partial credit is misleading. We also report mean
gold-recall (0/1/2 out of 2) as a finer-grained secondary signal.

k is reported at 2 (strictest -- exactly the gold count), 4, and 8.

Usage:
    python run_hotpot_multihop.py --n 50
    python run_hotpot_multihop.py --all --output results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ADAPTER_DIR = _HERE.parent / "08_industry_validation"
sys.path.insert(0, str(_ADAPTER_DIR))
from mathir_adapter import MathirAdapter  # noqa: E402

DATA_FILE = _HERE / "data" / "hotpotqa_bridge_hard.json"
DEFAULT_OUTPUT = _HERE / "results" / "hotpot_multihop_results.json"

# Retrievers to compare. Each is (label, callable(adapter, project, query, k)->list_of_memory_ids).
K_VALUES = [2, 4, 8]
GRAPH_LINK_THRESHOLD = 0.5  # lower than the 0.7 default: with only 10 short
# paragraphs per project, 0.7 often yields an empty graph, starving PPR-LTE of
# any substrate. 0.5 gives the graph edges to actually work with. This is a
# deliberate, disclosed choice -- and it's applied identically for every
# retriever, so it doesn't bias the comparison (flat retrievers ignore the graph).


def _ids_from_results(resp: dict) -> list:
    """Extract ordered memory_ids from any retriever's response dict."""
    results = resp.get("results", []) if isinstance(resp, dict) else []
    out = []
    for r in results:
        mid = r.get("memory_id")
        if mid:
            out.append(mid)
    return out


def evaluate_question(adapter: MathirAdapter, item: dict, max_k: int) -> dict:
    qid = item["id"]
    project = f"hotpot_{qid}"
    question = item["question"]
    gold_titles = set(item["gold_titles"])

    # 1. Ingest 10 paragraphs; map memory_id -> title.
    id_to_title = {}
    gold_mem_ids = set()
    for para in item["paragraphs"]:
        resp = adapter.add(project=project, content=para["text"],
                           agent="hotpot", label=para["title"][:200])
        mid = resp.get("memory_id") if isinstance(resp, dict) else None
        if not mid:
            continue
        id_to_title[mid] = para["title"]
        if para["title"] in gold_titles:
            gold_mem_ids.add(mid)

    # 2. Build the link graph over this project's 10 memories.
    try:
        adapter.build_links(project=project, threshold=GRAPH_LINK_THRESHOLD)
    except Exception:
        pass  # PPR-LTE will just have a sparse/empty graph; still measurable.

    # 3. Run each retriever, record ordered memory_ids.
    retriever_runs = {}
    timings = {}

    t0 = time.time()
    retriever_runs["recall_vector"] = _ids_from_results(
        adapter.recall(project=project, query=question, k=max_k))
    timings["recall_vector"] = (time.time() - t0) * 1000

    t0 = time.time()
    retriever_runs["hybrid_search"] = _ids_from_results(
        adapter.hybrid_search(project=project, query=question, k=max_k))
    timings["hybrid_search"] = (time.time() - t0) * 1000

    t0 = time.time()
    try:
        retriever_runs["ppr_lte_graph"] = _ids_from_results(
            adapter.ppr_lte_search(project=project, query=question, k=max_k))
    except Exception as e:
        retriever_runs["ppr_lte_graph"] = []
        retriever_runs["_ppr_error"] = str(e)[:200]
    timings["ppr_lte_graph"] = (time.time() - t0) * 1000

    # 4. Per-retriever, per-k: how many gold memories in top-k, and both-gold flag.
    per_retriever = {}
    for name, ordered_ids in retriever_runs.items():
        if name.startswith("_"):
            continue
        by_k = {}
        for k in K_VALUES:
            topk = set(ordered_ids[:k])
            n_gold = len(topk & gold_mem_ids)
            by_k[k] = {"gold_found": n_gold, "both_gold": n_gold == 2}
        per_retriever[name] = by_k

    return {
        "id": qid,
        "question": question,
        "gold_titles": item["gold_titles"],
        "n_gold_memories": len(gold_mem_ids),
        "per_retriever": per_retriever,
        "timings_ms": timings,
        "ppr_error": retriever_runs.get("_ppr_error"),
    }


def aggregate(results: list) -> dict:
    retriever_names = ["recall_vector", "hybrid_search", "ppr_lte_graph"]
    n = len(results)
    summary = {}
    for name in retriever_names:
        summary[name] = {}
        for k in K_VALUES:
            both = sum(1 for r in results
                       if r["per_retriever"].get(name, {}).get(k, {}).get("both_gold"))
            gold_sum = sum(r["per_retriever"].get(name, {}).get(k, {}).get("gold_found", 0)
                           for r in results)
            summary[name][f"both_gold@{k}"] = both / n if n else 0.0
            summary[name][f"mean_gold_recall@{k}"] = (gold_sum / (2 * n)) if n else 0.0
    return {"n_questions": n, "by_retriever": summary}


def print_summary(agg: dict):
    print("\n" + "=" * 78)
    print(f"HotpotQA Multi-Hop Retrieval Results (bridge+hard, n={agg['n_questions']})")
    print("Metric: both_gold@k = fraction of questions where BOTH gold paragraphs")
    print("are in the top-k (the metric that actually matters for multi-hop).")
    print("=" * 78)
    header = f"{'retriever':<18}" + "".join(f"both@{k:<7}" for k in K_VALUES) + "".join(f"recall@{k:<5}" for k in K_VALUES)
    print(header)
    print("-" * 78)
    for name, stats in agg["by_retriever"].items():
        row = f"{name:<18}"
        for k in K_VALUES:
            row += f"{stats[f'both_gold@{k}']*100:>7.1f}%"
        for k in K_VALUES:
            row += f"{stats[f'mean_gold_recall@{k}']*100:>8.1f}%"
        print(row)
    print("=" * 78)
    print("If PPR-LTE (graph) beats hybrid_search on both_gold@k, that's real")
    print("evidence MATHIR's link graph adds genuine multi-hop retrieval value")
    print("that flat vector/hybrid search cannot -- the core MATHIR differentiator.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of questions (first N).")
    parser.add_argument("--all", action="store_true", help="Run all questions in the dataset.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--daemon-url", type=str, default="http://127.0.0.1:7338")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run download_hotpotqa.py first.")
        sys.exit(1)

    with DATA_FILE.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not args.all:
        items = items[:args.n]

    adapter = MathirAdapter(daemon_url=args.daemon_url)
    max_k = max(K_VALUES)

    results = []
    for i, item in enumerate(items, 1):
        try:
            r = evaluate_question(adapter, item, max_k)
            results.append(r)
            h = r["per_retriever"].get("hybrid_search", {}).get(2, {}).get("both_gold")
            p = r["per_retriever"].get("ppr_lte_graph", {}).get(2, {}).get("both_gold")
            print(f"[{i}/{len(items)}] {item['id']}  hybrid@2={'Y' if h else 'n'}  ppr@2={'Y' if p else 'n'}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(items)}] {item['id']} FAILED: {e}", flush=True)

    agg = aggregate(results)
    print_summary(agg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"summary": agg, "results": results,
                   "config": {"graph_link_threshold": GRAPH_LINK_THRESHOLD, "k_values": K_VALUES}},
                  f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
