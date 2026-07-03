"""
swarm_locomo.py — head-to-head mode comparison on LoCoMo (cross-corpus validation).

LoCoMo is a different corpus from LongMemEval:
- 10 conversations x ~22 sessions x ~22 turns = ~4840 turns
- 5 categories: 1 (multi-hop), 2 (temporal), 3 (open-domain),
  4 (single-hop), 5 (adversarial)
- 199 questions per conversation -> 1990 total

This harness picks N conversations x N questions and runs the same
modes A/B as swarm_compare.py. The difference is the corpus: LoCoMo
has longer conversations, more categories, and more nuanced
abstention questions. This is the cross-corpus validation needed to
break the 58.3% saturation we observed on LongMemEval.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent / "mathir_mcp" / "mathir_lib"))

import mathir_adapter  # type: ignore
import run_locomo as locomo_mod  # type: ignore
import llm_client  # type: ignore

# Same mode list as swarm_compare.py
_ALL_MODES = ["hybrid_search", "confrank", "confrank_fast", "antipode",
              "ppr_lte", "smfm", "ad"]

# Category names from LoCoMo (matches Mem0/Zep reference harness)
LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
# Default scored categories (matches published methodology)
DEFAULT_SCORED_CATEGORIES = {1, 2, 3, 4}


def _call_mode(adapter, mode: str, project: str, question: str, k: int) -> dict:
    """Returns a dict with mode + results + diagnostics + elapsed_ms."""
    t0 = time.monotonic()
    try:
        fn_map = {
            "hybrid_search": lambda: adapter.hybrid_search(project=project, query=question, k=k),
            "confrank":       lambda: adapter.confrank_search(project=project, query=question, k=k),
            "confrank_fast":  lambda: adapter.confrank_fast(project=project, query=question, k=k),
            "antipode":       lambda: adapter.antipode_search(project=project, query=question, k=k),
            "ppr_lte":        lambda: adapter.ppr_lte_search(project=project, query=question, k=k),
            "smfm":           lambda: adapter.smfm_search(project=project, query=question, k=k),
            "ad":             lambda: adapter.ad_score_search(project=project, query=question, k=k),
        }
        r = fn_map[mode]()
        results = (r.get("results", []) if isinstance(r, dict) else []) or []
        # Re-key based on mode name (different modes tag scores differently)
        # We keep raw "results" as-is -- the runner uses .get("content", "")
        # which is the same across all modes.
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {"mode": mode, "n_results": len(results), "results": results,
                "diagnostics": r.get("diagnostics", {}) if isinstance(r, dict) else {},
                "elapsed_ms": elapsed_ms, "error": None}
    except Exception as e:
        return {"mode": mode, "n_results": 0, "results": [],
                "diagnostics": {}, "elapsed_ms": (time.monotonic() - t0) * 1000.0,
                "error": f"{type(e).__name__}: {str(e)[:200]}"}


def _generate_answer(question: str, retrieved_contents: list, mode: str) -> str:
    if not retrieved_contents:
        return "I cannot answer this question based on the available memories."
    # LoCoMo uses an inline prompt template (see ANSWER_GENERATION_PROMPT
    # in run_locomo.py). Format with memories + question.
    memories_block = "\n".join(f"- {c}" for c in retrieved_contents)
    user_prompt = locomo_mod.ANSWER_GENERATION_PROMPT.format(
        reference_date="2023",  # LoCoMo QA questions are 2023 era
        memories=memories_block,
        question=question,
    )
    answer_model = os.environ.get("MATHIR_BENCHMARK_ANSWER_MODEL") or None
    answer_max = int(os.environ.get("MATHIR_BENCHMARK_ANSWER_MAX_TOKENS", "16000"))
    try:
        return llm_client.chat(
            [{"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=answer_max, model=answer_model,
        )
    except Exception as e:
        return f"[LLM_ERROR: {e}]"


def _judge(question: str, gold: str, generated: str) -> tuple[bool | None, str]:
    """LoCoMo judge uses category-3 specific prompt (open-domain) since
    most QA items are short answers. Falls back to a generic yes/no
    judge otherwise."""
    judge_model = os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL") or None
    judge_max = int(os.environ.get("MATHIR_BENCHMARK_JUDGE_MAX_TOKENS", "8000"))
    system_msg = "You are a strict evaluator for a memory QA benchmark."
    user_msg = (
        f"Question: {question}\n"
        f"Expected answer: {gold}\n"
        f"Generated answer: {generated}\n\n"
        "Does the generated answer contain or match the expected answer "
        "(semantically equivalent is acceptable)? Reply strictly with "
        "CORRECT or INCORRECT."
    )
    try:
        resp = llm_client.chat(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_msg}],
            temperature=0.0, max_tokens=judge_max, model=judge_model,
        )
        text = resp.strip().upper()
        if "CORRECT" in text and "INCORRECT" not in text:
            return True, resp
        if "INCORRECT" in text:
            return False, resp
        return None, resp
    except Exception as e:
        return None, f"[JUDGE_ERROR: {e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=2,
                        help="Number of conversations to ingest and test on")
    parser.add_argument("--start-conv-idx", type=int, default=0,
                        help="Index of the first conversation (default 0). "
                             "Set to a higher value to avoid reusing a project that "
                             "already has data from a previous run.")
    parser.add_argument("--questions-per-cat", type=int, default=1,
                        help="Questions to sample per LoCoMo category (1-5)")
    parser.add_argument("--modes", type=str, default="hybrid_search,confrank,confrank_fast,ppr_lte,ad",
                        help="Comma-separated list of search modes to compare")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run-label", type=str, default="swarm_locomo")
    parser.add_argument("--include-cat-5", action="store_true",
                        help="Include adversarial category 5 (excluded by default)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing JSONL checkpoint: skip question_ids "
                             "that are already present in the checkpoint. Header is preserved; "
                             "new trials are appended.")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if any(m not in _ALL_MODES for m in modes):
        raise SystemExit(f"Unknown modes. Valid: {_ALL_MODES}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)

    scored_categories = set(DEFAULT_SCORED_CATEGORIES)
    if args.include_cat_5:
        scored_categories.add(5)

    # Load LoCoMo dataset
    dataset = locomo_mod.load_dataset()
    n_available = len(dataset)
    n_conv = min(args.conversations, n_available)
    conv_indices = list(range(args.start_conv_idx, args.start_conv_idx + n_conv))

    # Resume support: load already-done question_ids to skip them
    resume_skip_ids: set = set()
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except:
                    continue
                if obj.get("kind") == "trial":
                    resume_skip_ids.add(obj.get("question_id"))
        if resume_skip_ids:
            print(f"[swarm_locomo] --resume: skipping {len(resume_skip_ids)} already-done question_ids")

    # Write (or resume-append) header
    if args.resume and checkpoint_path.exists():
        # Append mode: keep existing header, just open for appending trials
        checkpoint_mode = "a"
        print(f"[swarm_locomo] checkpoint appending to {checkpoint_path} (resume)")
    else:
        # Fresh write
        checkpoint_mode = "w"
        with open(checkpoint_path, "w", encoding="utf-8") as cf:
            cf.write(json.dumps({
                "kind": "header",
                "benchmark": "LoCoMo",
                "corpus_name": "LoCoMo v1 (10 conversations)",
                "corpus_source": ("snap-research/locomo @ github.com/snap-research/locomo, "
                                  "file locomo10.json (CC BY-NC 4.0)"),
                "run_label": args.run_label,
                "modes": modes,
                "k": args.k,
                "n_conversations": n_conv,
                "questions_per_cat": args.questions_per_cat,
                "scored_categories": sorted(scored_categories),
                "started_at_utc": locomo_mod._utcnow_iso(),
            }) + "\n")
        print(f"[swarm_locomo] checkpoint writing to {checkpoint_path}")

    print(f"[swarm_locomo] {n_conv} conversations x {args.questions_per_cat}q/cat x "
          f"{len(scored_categories)} cats x {len(modes)} modes = "
          f"{n_conv * args.questions_per_cat * len(scored_categories) * len(modes)} trials")

    adapter = mathir_adapter.MathirAdapter()

    # Ingest all conversations
    print("[swarm_locomo] Ingesting conversations into MATHIR...")
    for conv_idx in conv_indices:
        conv = dataset[conv_idx]
        project = f"locomo_conv_{conv_idx}"
        n_turns, last_dt = locomo_mod.ingest_conversation(adapter, project, conv)
        print(f"  conv {conv_idx} -> {project}: {n_turns} turns ingested, last_date={last_dt[:20]}")
    print("[swarm_locomo] Ingest done.")

    # Run query phase
    output: dict = {"summary": {}, "results": []}
    score_by_mode: dict[str, dict] = {
        m: {"CORRECT": 0, "INCORRECT": 0, "UNCLEAR": 0, "errors": 0,
            "total_ms": 0.0, "by_category": {}}
        for m in modes
    }

    for conv_idx in conv_indices:
        conv = dataset[conv_idx]
        project = f"locomo_conv_{conv_idx}"
        reference_date = ""
        # Get reference date from last session
        for _sk, dt, _t in locomo_mod.iter_sessions(conv):
            if dt:
                reference_date = dt

        # Filter QAs to scored categories
        qa_list = conv.get("qa", [])
        sampled_per_cat = {}
        for qa in qa_list:
            cat = qa.get("category")
            if cat not in scored_categories:
                continue
            sampled_per_cat.setdefault(cat, []).append(qa)
        # Sample N per cat
        selected = []
        for cat in sorted(sampled_per_cat.keys()):
            selected.extend(sampled_per_cat[cat][: args.questions_per_cat])

        print(f"\n[conv {conv_idx}] {len(selected)} questions selected")
        for qi, qa in enumerate(selected, start=1):
            qtext = qa.get("question", "")
            gold = str(qa.get("answer", ""))
            cat = qa.get("category", "?")
            qid = locomo_mod._qa_id(conv_idx, qa)
            if resume_skip_ids and qid in resume_skip_ids:
                print(f"  [Q {qi}/{len(selected)}] {qid} cat={cat} -- skipped (already done)")
                continue
            print(f"  [Q {qi}/{len(selected)}] {qid} cat={cat} q={qtext[:50]!r} gold={gold[:30]!r}")

            per_mode_records = []
            for mode in modes:
                call = _call_mode(adapter, mode, project, qtext, args.k)
                verdict = None
                judge_raw = None
                generated = ""
                if call["n_results"] > 0 and not call["error"]:
                    retrieved = [r.get("content", "") for r in call["results"]]
                    generated = _generate_answer(qtext, retrieved, mode)
                    verdict, judge_raw = _judge(qtext, gold, generated)

                rec = {
                    "mode": mode,
                    "qid": qid,
                    "category": cat,
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

                if call["error"]:
                    score_by_mode[mode]["errors"] += 1
                if verdict is True:
                    score_by_mode[mode]["CORRECT"] += 1
                elif verdict is False:
                    score_by_mode[mode]["INCORRECT"] += 1
                else:
                    score_by_mode[mode]["UNCLEAR"] += 1
                score_by_mode[mode]["total_ms"] += call["elapsed_ms"]
                score_by_mode[mode]["by_category"].setdefault(str(cat), {})
                score_by_mode[mode]["by_category"][str(cat)]["n"] = (
                    score_by_mode[mode]["by_category"][str(cat)].get("n", 0) + 1
                )
                if verdict is True:
                    score_by_mode[mode]["by_category"][str(cat)]["CORRECT"] = (
                        score_by_mode[mode]["by_category"][str(cat)].get("CORRECT", 0) + 1
                    )

                with open(checkpoint_path, "a", encoding="utf-8") as cf:
                    cf.write(json.dumps({"kind": "trial",
                                          "timestamp_utc": locomo_mod._utcnow_iso(),
                                          "conversation_idx": conv_idx,
                                          "question_id": qid,
                                          "category": cat,
                                          **rec}) + "\n")

                v = ("CORRECT" if verdict is True else
                     "INCORRECT" if verdict is False else
                     "UNCLEAR" if verdict is None else "ERROR")
                print(f"    {mode:>14}  {v:>10}  {call['n_results']:>2}res  "
                      f"{call['elapsed_ms']:>6.0f}ms"
                      + ("  ERR=" + call["error"][:60] if call["error"] else ""))

            output["results"].append({
                "conversation_idx": conv_idx,
                "question_id": qid,
                "category": cat,
                "question_text": qtext,
                "ground_truth_answer": gold,
                "modes": per_mode_records,
            })

    n_questions = sum(len(r.get("modes", [])) // max(1, len(modes))
                    for r in output["results"])
    summary = {
        "by_mode": {
            m: {
                **score_by_mode[m],
                "n_questions": n_questions,
                "accuracy": (score_by_mode[m]["CORRECT"]
                             / max(1, n_questions - score_by_mode[m]["errors"])),
            }
            for m in modes
        }
    }
    output["summary"] = summary
    output["config"] = {
        "modes": modes,
        "k": args.k,
        "n_conversations": n_conv,
        "questions_per_cat": args.questions_per_cat,
        "scored_categories": sorted(scored_categories),
        "run_label": args.run_label,
        "mathir_lib_version": locomo_mod._mathir_lib_version(),
        "started_at_utc": locomo_mod._utcnow_iso(),
        "finished_at_utc": locomo_mod._utcnow_iso(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 76)
    print(f"SWARM LoCoMo RESULTS  ({n_conv} conv x {args.questions_per_cat}/cat = {n_questions} questions, "
          f"{len(modes)} modes)")
    print("=" * 76)
    print(f"{'mode':<16}{'CORR':>5}{'INC':>5}{'UNC':>5}{'err':>5}{'acc':>8}{'avg_ms':>10}")
    for m in modes:
        s = score_by_mode[m]
        denom = max(1, n_questions - s["errors"])
        acc = s["CORRECT"] / denom if denom else 0
        avg_ms = s["total_ms"] / max(1, n_questions)
        print(f"{m:<16}{s['CORRECT']:>5}{s['INCORRECT']:>5}{s['UNCLEAR']:>5}{s['errors']:>5}"
              f"{acc*100:>7.1f}%{avg_ms:>9.0f}ms")
    print("=" * 76)
    print(f"Wrote: {output_path}")
    print(f"JSONL: {checkpoint_path}")


if __name__ == "__main__":
    main()