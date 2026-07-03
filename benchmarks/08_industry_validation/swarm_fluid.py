#!/usr/bin/env python3
"""
Swarm benchmark on the fluid_mechanics BEIR-format corpus.

Unlike LongMemEval/LoCoMo (which test end-to-end QA with LLM answer + LLM judge),
this is a pure RETRIEVAL benchmark: for each question, did the top-k retrieved
memories include the ground-truth chunk?

This is much faster (no LLM calls in the loop) and tells us the raw retrieval
quality of each MATHIR mode on a dense, technical corpus.

Each test row:
  {qid, mode, top_k_ids, hit_at_k, ndcg_at_10, mrr, latency_ms}

Modes: hybrid_search, ppr_lte, smfm, ad, confrank, confrank_fast
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mathir_adapter  # type: ignore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"D:\SECRET_PROJECT\MATHIR\benchmarks\05_test_data\beir_data\fluid_mechanics\fluid_mechanics")
CORPUS_FILE = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"
QRELS_FILE = DATA_DIR / "qrels" / "test.tsv"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_corpus() -> list[dict]:
    docs = []
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def load_queries() -> list[dict]:
    qs = []
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def load_qrels() -> dict[str, str]:
    """Returns {query_id: relevant_corpus_id} (single gold per query)."""
    qrels = {}
    with open(QRELS_FILE, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                qid, cid, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:
                    qrels[qid] = cid
    return qrels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> int:
    return 1 if gold_id in retrieved_ids[:k] else 0


def mrr(retrieved_ids: list[str], gold_id: str) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid == gold_id:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> float:
    """Binary relevance: 1 if gold is in top-k, 0 otherwise."""
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == gold_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------
PROJECT = "fluid_mechanics_bench_v2"


def run_mode(adapter: mathir_adapter.MathirAdapter,
             mode: str,
             query: str,
             k: int) -> tuple[list[str], float, str | None]:
    """Returns (list_of_doc_ids, latency_ms, error_msg_or_None)."""
    t0 = time.perf_counter()
    error = None
    retrieved = []

    try:
        if mode == "hybrid_search":
            r = adapter.hybrid_search(PROJECT, query, k=k)
            retrieved = [_extract_doc_id(item) for item in r.get("results", [])]

        elif mode == "ppr_lte":
            r = adapter.ppr_lte_search(PROJECT, query, k=k)
            retrieved = [_extract_doc_id(item) for item in r.get("results", [])]

        elif mode == "smfm":
            r = adapter.smfm_search(PROJECT, query, k=k)
            retrieved = [_extract_doc_id(item) for item in r.get("results", [])]

        elif mode == "ad":
            r = adapter.ad_score_search(PROJECT, query, k=k)
            retrieved = [_extract_doc_id(item) for item in r.get("results", [])]

        elif mode == "confrank":
            r = adapter.confrank_search(PROJECT, query, k=k)
            retrieved = [_extract_doc_id(item) for item in r.get("results", [])]

        elif mode == "confrank_fast":
            r = adapter.confrank_fast(PROJECT, query, k=k)
            retrieved = [_extract_doc_id(item) for item in r.get("results", [])]

        else:
            error = f"unknown mode: {mode}"

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return retrieved, elapsed_ms, error


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def ingest_corpus(adapter: mathir_adapter.MathirAdapter, docs: list[dict]) -> int:
    count = 0
    for d in docs:
        chunk_id = d["_id"]
        text = d.get("text", "")
        title = d.get("title", "")
        # Build rich content
        content = f"[{title}] {text}" if title else text
        try:
            # Encode chunk_id into the content with a sentinel so we can
            # extract it from retrieval results later.
            tagged_content = f"{chunk_id}|||{content}"
            adapter.add(
                project=PROJECT,
                content=tagged_content,
                agent="swarm_fluid",
                block_type="semantic",
                label=chunk_id,
                priority=5,
            )
            count += 1
        except Exception as e:
            print(f"  Ingest failed for {chunk_id}: {e}")
    return count


def _extract_doc_id(result_item: dict) -> str:
    """Extract the document identifier from a retrieval result.

    The server assigns its own memory_id, so we encode the chunk_id into
    the content with a sentinel prefix during ingest (chunk_id|||content).
    The content field is what the search returns, so we parse the prefix
    back out here.
    """
    content = result_item.get("content", "") or result_item.get("text", "")
    if "|||" in content:
        # New format: chunk_id is the first segment before the sentinel
        return content.split("|||", 1)[0].strip()
    # Legacy format: just the auto-generated memory_id
    return result_item.get("memory_id", result_item.get("id", ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str, default="hybrid_search,ppr_lte,smfm,ad,confrank,confrank_fast",
                        help="Comma-separated list of retrieval modes to test")
    parser.add_argument("--k", type=int, default=10, help="Top-k to retrieve per question")
    parser.add_argument("--output", type=str, required=True, help="JSON summary output path")
    parser.add_argument("--checkpoint", type=str, required=True, help="JSONL checkpoint path")
    parser.add_argument("--run-label", type=str, default="swarm_fluid")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip question_ids already in checkpoint")
    parser.add_argument("--no-ingest", action="store_true", help="Skip corpus ingestion (assume already ingested)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)

    print(f"[swarm_fluid] modes: {modes}")
    print(f"[swarm_fluid] k={args.k}")

    adapter = mathir_adapter.MathirAdapter()

    # --- Ingest ---
    if not args.no_ingest:
        print(f"[swarm_fluid] Ingesting corpus...")
        docs = load_corpus()
        n_ingested = ingest_corpus(adapter, docs)
        print(f"[swarm_fluid] Ingested {n_ingested}/{len(docs)} chunks into project '{PROJECT}'")
    else:
        print(f"[swarm_fluid] Skipping ingest (--no-ingest)")
        docs = load_corpus()

    # --- Load queries + qrels ---
    queries = load_queries()
    qrels = load_qrels()
    print(f"[swarm_fluid] {len(queries)} queries, {len(qrels)} qrels")

    if args.limit > 0:
        queries = queries[:args.limit]
    print(f"[swarm_fluid] Running on {len(queries)} queries × {len(modes)} modes = {len(queries)*len(modes)} trials")

    # --- Resume support ---
    done_keys = set()
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("kind") == "trial":
                    done_keys.add((obj.get("qid"), obj.get("mode")))
        if done_keys:
            print(f"[swarm_fluid] --resume: skipping {len(done_keys)} already-done trials")

    # --- Write header ---
    mode = "w" if not (args.resume and checkpoint_path.exists()) else "a"
    if not (args.resume and checkpoint_path.exists()):
        with checkpoint_path.open(mode, encoding="utf-8") as cf:
            cf.write(json.dumps({
                "kind": "header",
                "benchmark": "Fluid Mechanics BEIR",
                "corpus_name": "Fluid Mechanics (White 7ed + Cengel/Cimbala)",
                "corpus_source": ("White 2011 7ed + Cengel/Cimbala PDF textbooks, "
                                  "chunked to 150-300 words via build_fluid_mechanics_corpus.py"),
                "run_label": args.run_label,
                "modes": modes,
                "k": args.k,
                "n_chunks": len(docs),
                "n_queries": len(queries),
                "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }) + "\n")

    # --- Run trials ---
    results = []  # (qid, mode, hit_at_k, mrr, ndcg, latency_ms, error)
    for qi, q in enumerate(queries, start=1):
        qid = q["_id"]
        qtext = q["text"]
        gold = qrels.get(qid)
        if not gold:
            print(f"  [Q {qi}/{len(queries)}] {qid}: no qrel, skipping")
            continue

        print(f"  [Q {qi:>3}/{len(queries)}] {qid}: {qtext[:60]!r} gold={gold}")
        for m in modes:
            if (qid, m) in done_keys:
                continue
            retrieved, latency_ms, error = run_mode(adapter, m, qtext, args.k)
            hit = recall_at_k(retrieved, gold, args.k)
            mrr_v = mrr(retrieved, gold)
            ndcg = ndcg_at_k(retrieved, gold, args.k)

            results.append({
                "kind": "trial",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "qid": qid,
                "query_text": qtext,
                "mode": m,
                "gold_id": gold,
                "n_retrieved": len(retrieved),
                "top5_ids": retrieved[:5],
                "hit_at_k": hit,
                "mrr": mrr_v,
                "ndcg_at_10": ndcg,
                "elapsed_ms": round(latency_ms, 1),
                "error": error,
            })

            with checkpoint_path.open("a", encoding="utf-8") as cf:
                cf.write(json.dumps(results[-1]) + "\n")

            tag = "HIT" if hit else "MISS"
            err_tag = f" ERR={error[:40]}" if error else ""
            print(f"      {m:>14}  {tag}  ndcg10={ndcg:.3f}  mrr={mrr_v:.3f}  {latency_ms:>6.0f}ms{err_tag}")

    # --- Summary ---
    summary = {"per_mode": {}}
    for m in modes:
        mode_results = [r for r in results if r["mode"] == m]
        if not mode_results:
            continue
        n = len(mode_results)
        hits = sum(r["hit_at_k"] for r in mode_results)
        avg_mrr = sum(r["mrr"] for r in mode_results) / n
        avg_ndcg = sum(r["ndcg_at_10"] for r in mode_results) / n
        avg_latency = sum(r["elapsed_ms"] for r in mode_results) / n
        summary["per_mode"][m] = {
            "n_questions": n,
            "recall_at_k": hits / n,
            "mrr": round(avg_mrr, 4),
            "ndcg_at_10": round(avg_ndcg, 4),
            "avg_latency_ms": round(avg_latency, 1),
        }

    summary["modes"] = modes
    summary["k"] = args.k
    summary["n_questions"] = len(queries)
    summary["n_chunks"] = len(docs)
    summary["run_label"] = args.run_label
    summary["corpus_name"] = "Fluid Mechanics (White 7ed + Cengel/Cimbala)"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 76)
    print(f"SWARM FLUID RESULTS  ({len(queries)} questions, {len(modes)} modes, k={args.k})")
    print("=" * 76)
    print(f"{'mode':>16}  {'R@K':>6}  {'MRR':>6}  {'NDCG':>6}  {'avg_ms':>7}")
    for m, s in summary["per_mode"].items():
        print(f"{m:>16}  {s['recall_at_k']:.3f}  {s['mrr']:.3f}  {s['ndcg_at_10']:.3f}  {s['avg_latency_ms']:>6.0f}ms")
    print()
    print(f"Wrote: {output_path}")
    print(f"JSONL: {checkpoint_path}")


if __name__ == "__main__":
    main()
