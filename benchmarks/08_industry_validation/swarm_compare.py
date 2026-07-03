#!/usr/bin/env python3
"""
swarm_compare.py — compare N search modes head-to-head on the same questions.

Each question uses ONLY ONE mode as the answer source (no hybrid fallback).
Reports per-mode accuracy, latency, and per-mode diagnostics.

This is the proper A/B harness: it directly measures each mode's contribution
rather than hybrid-with-side-effects.

Outputs:
- benchmarks/06_results/current/swarm_<run_label>.json  (full per-question, per-mode)
- benchmarks/06_results/current/swarm_<run_label>.log   (stdout stream)
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Late import -- the mathir_adapter file uses mathir_advanced/mathir_confrank
sys.path.insert(0, str(ROOT.parent.parent / "mathir_mcp" / "mathir_lib"))

import mathir_adapter  # type: ignore
import run_longmemeval as rl  # type: ignore
import llm_client  # type: ignore

_MODES = ["hybrid_search", "confrank", "confrank_fast", "antipode", "ppr_lte", "smfm", "ad"]


def _call_mode(adapter, mode: str, project: str, question: str, k: int) -> dict:
    """Returns a dict with mode + results list + diagnostics + elapsed_ms."""
    t0 = time.monotonic()
    try:
        if mode == "hybrid_search":
            r = adapter.hybrid_search(project=project, query=question, k=k)
            results = r.get("results", []) if isinstance(r, dict) else []
        elif mode == "confrank":
            r = adapter.confrank_search(project=project, query=question, k=k)
            results = r.get("results", []) if isinstance(r, dict) else []
        elif mode == "confrank_fast":
            r = adapter.confrank_fast(project=project, query=question, k=k)
            results = r.get("results", []) if isinstance(r, dict) else []
        elif mode == "antipode":
            r = adapter.antipode_search(project=project, query=question, k=k)
            results = r.get("results", []) if isinstance(r, dict) else []
        elif mode == "ppr_lte":
            r = adapter.ppr_lte_search(project=project, query=question, k=k)
            results = r.get("results", []) if isinstance(r, dict) else []
        elif mode == "smfm":
            r = adapter.smfm_search(project=project, query=question, k=k)
            results = r.get("results", []) if isinstance(r, dict) else []
        elif mode == "ad":
            r = adapter.ad_score_search(project=project, query=question, k=k)
            results = r.get("results", []) if isinstance(r, dict) else []
        else:
            results = []
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {"mode": mode, "n_results": len(results), "results": results,
                "diagnostics": r.get("diagnostics", {}) if isinstance(r, dict) else {},
                "elapsed_ms": elapsed_ms, "error": None}
    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {"mode": mode, "n_results": 0, "results": [],
                "diagnostics": {}, "elapsed_ms": elapsed_ms,
                "error": f"{type(e).__name__}: {str(e)[:200]}"}


def generate_answer_with_results(question: str, retrieved_contents: list, mode: str) -> str:
    """Single-mode answer generation. Skip LLM call if no retrieved contents."""
    if not retrieved_contents:
        return "I cannot answer this question based on the available memories."
    messages = rl.build_generation_prompt(question, retrieved_contents)
    answer_model = os.environ.get("MATHIR_BENCHMARK_ANSWER_MODEL") or None
    answer_max = int(os.environ.get("MATHIR_BENCHMARK_ANSWER_MAX_TOKENS", "16000"))
    try:
        return llm_client.chat(messages, temperature=0.0,
                               max_tokens=answer_max, model=answer_model)
    except Exception as e:
        return f"[LLM_ERROR: {e}]"


def judge(question: str, gold: str, generated: str, qtype: str) -> tuple[bool | None, str]:
    judge_model = os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL") or None
    judge_max = int(os.environ.get("MATHIR_BENCHMARK_JUDGE_MAX_TOKENS", "8000"))
    messages = rl.build_judge_prompt(qtype, question, gold, generated)
    try:
        resp = llm_client.chat(messages, temperature=0.0,
                                max_tokens=judge_max, model=judge_model)
        return rl._parse_judge_verdict(resp), resp
    except Exception as e:
        return None, f"[JUDGE_ERROR: {e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=1)
    parser.add_argument("--question-types", type=str,
                        default="knowledge-update,multi-session,"
                                "single-session-user,temporal-reasoning")
    parser.add_argument("--modes", type=str,
                        default="hybrid_search,confrank,antipode,ppr_lte,smfm,ad")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional hard cap on number of questions.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run-label", type=str, default="swarm")
    parser.add_argument("--dataset", type=str,
                        default=str(rl.DEFAULT_DATASET_PATH))
    args = parser.parse_args()

    # Header
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    question_types_filter = [t.strip() for t in args.question_types.split(",") if t.strip()]

    dataset_path = Path(args.dataset)
    all_questions = rl.load_dataset(dataset_path)
    questions = rl.sample_questions(all_questions, args.per_type, False, question_types_filter)
    if args.limit:
        questions = questions[: args.limit]

    print(f"[swarm] {len(questions)} questions x {len(modes)} modes = {len(questions)*len(modes)} trials")
    print(f"[swarm] modes: {modes}")

    adapter = mathir_adapter.MathirAdapter()

    # Write a header line
    with open(checkpoint_path, "w", encoding="utf-8") as cf:
        cf.write(json.dumps({"kind": "header", "benchmark": "LongMemEval",
                            "corpus_name": "LongMemEval-S (cleaned)",
                            "corpus_source": "xiaowu0162/longmemeval-cleaned (MIT)",
                            "run_label": args.run_label,
                            "modes": modes,
                            "k": args.k,
                            "n_questions": len(questions),
                            "started_at_utc": rl._utcnow_iso(),
                            "mathir_lib_version": rl._mathir_lib_version(),
                            "script_version": "swarm_compare 1.0"
                            }) + "\n")

    # Output container
    output: dict = {
        "summary": {},
        "results": [],
    }

    score_by_mode: dict[str, dict] = {m: {"CORRECT": 0, "INCORRECT": 0, "UNCLEAR": 0, "errors": 0, "total_ms": 0.0} for m in modes}

    for qi, q in enumerate(questions, start=1):
        qid = q["question_id"]
        qtype = q.get("question_type", "unknown")
        qtext = q["question"]
        gold = q.get("answer", "")
        project = f"longmemeval_{qid}"

        print(f"\n[Q {qi}/{len(questions)}] {qid} ({qtype}) -- gold: {str(gold)[:60]}")

        # First, ensure the project has the data ingested (we can use
        # any mode that needs it). For speed we just check via hybrid_search.
        sanity = adapter.hybrid_search(project=project, query=qtext, k=1)
        if not sanity.get("results"):
            print(f"  [WARN] no memories for {qid}; skipping")
            continue

        per_mode_records: list[dict] = []
        for mode in modes:
            call = _call_mode(adapter, mode, project, qtext, args.k)

            # Generate + judge if the call produced results.
            verdict = None
            judge_raw = None
            generated = ""
            if call["n_results"] > 0 and not call["error"]:
                retrieved_contents = [r.get("content", "") for r in call["results"]]
                generated = generate_answer_with_results(qtext, retrieved_contents, mode)
                verdict, judge_raw = judge(qtext, gold, generated, qtype)

            # Per-mode record
            rec = {
                "mode": mode,
                "qid": qid,
                "qtype": qtype,
                "n_results": call["n_results"],
                "elapsed_ms": round(call["elapsed_ms"], 1),
                "verdict": verdict,
                "judge_raw": judge_raw,
                "generated_answer": generated,
                "error": call["error"],
                "diagnostics": call["diagnostics"],
                "top_contents": [r.get("content", "")[:160] for r in call["results"][:3]],
            }
            per_mode_records.append(rec)

            # Update score
            if call["error"]:
                score_by_mode[mode]["errors"] += 1
            if verdict is True:
                score_by_mode[mode]["CORRECT"] += 1
            elif verdict is False:
                score_by_mode[mode]["INCORRECT"] += 1
            else:
                score_by_mode[mode]["UNCLEAR"] += 1
            score_by_mode[mode]["total_ms"] += call["elapsed_ms"]

            # Per-mode JSONL line
            with open(checkpoint_path, "a", encoding="utf-8") as cf:
                cf.write(json.dumps({"kind": "trial",
                                     "timestamp_utc": rl._utcnow_iso(),
                                     "question_id": qid,
                                     "question_type": qtype,
                                     **rec}) + "\n")

            v = ("CORRECT" if verdict is True else
                 "INCORRECT" if verdict is False else
                 "UNCLEAR" if verdict is None else "ERROR")
            print(f"  {mode:>14}  {v:>10}  {call['n_results']:>2}res  {call['elapsed_ms']:>6.0f}ms"
                  + ("  ERR=" + call["error"][:60] if call["error"] else ""))

        output["results"].append({
            "question_id": qid,
            "question_type": qtype,
            "question_text": qtext,
            "ground_truth_answer": gold,
            "modes": per_mode_records,
        })

    # Summary
    summary = {
        "by_mode": {
            m: {
                **score_by_mode[m],
                "n_questions": len(questions),
                "accuracy": (
                    score_by_mode[m]["CORRECT"]
                    / max(1, len(questions) - score_by_mode[m]["errors"])
                ),
            }
            for m in modes
        }
    }
    output["summary"] = summary
    output["config"] = {
        "modes": modes,
        "k": args.k,
        "n_questions": len(questions),
        "run_label": args.run_label,
        "mathir_lib_version": rl._mathir_lib_version(),
        "started_at_utc": rl._utcnow_iso(),
        "finished_at_utc": rl._utcnow_iso(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 64)
    print(f"SWARM RESULTS  ({len(questions)} questions, {len(modes)} modes)")
    print("=" * 64)
    print(f"{'mode':<16}{'CORR':>5}{'INC':>5}{'UNC':>5}{'err':>5}{'acc':>8}{'avg_ms':>10}")
    for m in modes:
        s = score_by_mode[m]
        denom = max(1, len(questions) - s["errors"])
        acc = s["CORRECT"] / denom if denom else 0
        avg_ms = s["total_ms"] / max(1, len(questions))
        print(f"{m:<16}{s['CORRECT']:>5}{s['INCORRECT']:>5}{s['UNCLEAR']:>5}{s['errors']:>5}{acc*100:>7.1f}%{avg_ms:>9.0f}ms")
    print("=" * 64)
    print(f"Wrote: {output_path}")
    print(f"JSONL: {checkpoint_path}")


if __name__ == "__main__":
    main()