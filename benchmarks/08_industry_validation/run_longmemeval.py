#!/usr/bin/env python3
"""
LongMemEval benchmark runner for MATHIR.

Evaluates MATHIR (memory MCP server, via MathirAdapter) against the
LongMemEval-S academic long-term-memory benchmark, using the same
task/methodology that produced the published Mem0 (66.9%) and Zep (75.1%)
numbers, so results are broadly comparable.

Pipeline per question:
    1. Ingest  -- every turn of every haystack session is saved as a MATHIR
                  memory in a project namespace unique to the question
                  (`longmemeval_{question_id}`), so there is zero
                  cross-contamination between questions and no cleanup is
                  needed.
    2. Search  -- MATHIR is queried with the question text (hybrid search,
                  top-k configurable via --k).
    3. Generate -- an LLM answers the question using ONLY the retrieved
                  memory contents as context, and is told to say so
                  explicitly if it cannot answer from the given context
                  (needed to score abstention ("_abs") questions correctly).
    4. Judge   -- an LLM-as-judge (category-aware prompt, reconstructed from
                  the official LongMemEval evaluation script -- see
                  `JUDGE_PROMPT_TEMPLATES` below) compares the generated
                  answer to the ground-truth answer and returns a strict
                  yes/no verdict.

Usage:
    python run_longmemeval.py --per-type 3
    python run_longmemeval.py --full --k 5 --output results.json
    python run_longmemeval.py --per-type 5 --question-types single-session-user,temporal-reasoning

NOTE on the judge prompts: the official LongMemEval repo
(github.com/xiaowu0162/LongMemEval, src/evaluation/evaluate_qa.py) uses
category-specific prompt templates. This script reconstructs those templates
in good faith from a fetch of that file (binary yes/no judge, with
temporal-reasoning tolerating off-by-one-day errors, knowledge-update
accepting mixed old/new info as long as the final answer reflects the latest
update, and abstention ("_abs") categories judged on whether the model
correctly declines to answer). This is NOT a byte-for-byte copy of the
original source (network access to fetch and diff the literal file was not
fully verified against a local checkout at the time this script was
written) -- treat it as a faithful reconstruction, not the verbatim
original.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

BENCHMARKS_08_DIR = Path(__file__).resolve().parent
BENCHMARKS_ROOT = BENCHMARKS_08_DIR.parent

sys.path.insert(0, str(BENCHMARKS_08_DIR))

from mathir_adapter import MathirAdapter  # noqa: E402
import llm_client  # noqa: E402

DEFAULT_DATASET_PATH = (
    BENCHMARKS_ROOT / "05_test_data" / "longmemeval" / "longmemeval_s_cleaned.json"
)
DEFAULT_OUTPUT_PATH = (
    BENCHMARKS_ROOT / "06_results" / "current" / "longmemeval_results.json"
)


# ---------------------------------------------------------------------------
# Judge prompt templates
# ---------------------------------------------------------------------------
# Reconstructed in good faith from the official LongMemEval evaluation script
# (github.com/xiaowu0162/LongMemEval, src/evaluation/evaluate_qa.py). All
# templates ask for a strict verdict and are parsed the same way the official
# script does: `label = 'yes' in response.lower()`.

_JUDGE_HEADER = (
    "I will give you a question, a correct/expected answer, and a response "
    "from a model. Please answer 'yes' if the response contains/matches the "
    "correct answer, is equivalent in meaning to the correct answer, or "
    "correctly shows the required intermediate steps/reasoning. Answer 'no' "
    "if the response is missing the correct answer, contradicts it, or only "
    "gives partial/incomplete information.\n\n"
)

JUDGE_PROMPT_TEMPLATES = {
    "default": (
        _JUDGE_HEADER
        + "Question: {question}\n"
        "Correct answer: {gold_answer}\n"
        "Model response: {generated_answer}\n\n"
        "Does the model response contain/match the correct answer? "
        "Answer strictly with 'CORRECT' or 'INCORRECT', nothing else."
    ),
    "temporal-reasoning": (
        _JUDGE_HEADER
        + "This is a temporal-reasoning question, so do NOT penalize "
        "off-by-one errors for the number of days (e.g. an answer that is "
        "one day earlier/later than the correct answer due to inclusive/"
        "exclusive date counting should still be judged correct).\n\n"
        "Question: {question}\n"
        "Correct answer: {gold_answer}\n"
        "Model response: {generated_answer}\n\n"
        "Does the model response contain/match the correct answer (allowing "
        "for the off-by-one-day tolerance described above)? "
        "Answer strictly with 'CORRECT' or 'INCORRECT', nothing else."
    ),
    "knowledge-update": (
        _JUDGE_HEADER
        + "This is a knowledge-update question: the correct answer reflects "
        "the LATEST update to a fact that changed over time. If the "
        "response contains some outdated/previous information along with "
        "the updated (correct) answer, it should still be considered "
        "CORRECT, as long as the final/updated answer given is right and "
        "not contradicted. If the response only gives an outdated answer "
        "and does not reflect the latest update, it is INCORRECT.\n\n"
        "Question: {question}\n"
        "Correct (latest/updated) answer: {gold_answer}\n"
        "Model response: {generated_answer}\n\n"
        "Answer strictly with 'CORRECT' or 'INCORRECT', nothing else."
    ),
    "single-session-preference": (
        "I will give you a question about a user's stated preferences, a "
        "rubric describing the desired response, and a model response. "
        "Please answer 'CORRECT' if the response satisfies the desired "
        "response / correctly uses the user's stated personal preference "
        "information. The model does NOT need to reflect every single "
        "point in the rubric -- judge whether the core preference-driven "
        "recommendation/answer is satisfied.\n\n"
        "Question: {question}\n"
        "Desired response / rubric: {gold_answer}\n"
        "Model response: {generated_answer}\n\n"
        "Answer strictly with 'CORRECT' or 'INCORRECT', nothing else."
    ),
    "abstention": (
        "I will give you a question that has NO real answer available in "
        "the given context/history (an abstention/unanswerable question), "
        "and a model response. Please answer 'CORRECT' if the model "
        "response correctly identifies the question as unanswerable / "
        "correctly declines to answer / states it does not have enough "
        "information, rather than fabricating an answer. Answer "
        "'INCORRECT' if the model instead confidently produces an answer "
        "(a hallucination), since there is no real answer to be found.\n\n"
        "Question: {question}\n"
        "Model response: {generated_answer}\n\n"
        "Answer strictly with 'CORRECT' or 'INCORRECT', nothing else."
    ),
}


def _judge_template_for(question_type: str) -> str:
    if question_type.endswith("_abs"):
        return JUDGE_PROMPT_TEMPLATES["abstention"]
    base_type = question_type
    if base_type in JUDGE_PROMPT_TEMPLATES:
        return JUDGE_PROMPT_TEMPLATES[base_type]
    return JUDGE_PROMPT_TEMPLATES["default"]


def _parse_judge_verdict(judge_response: str) -> bool:
    """Parses the judge's free-text verdict into a bool, adapting the
    official script's `label = 'yes' in response.lower()` approach for our
    CORRECT/INCORRECT wording.

    Strategy: strip the word "incorrect" out of the text before checking for
    "correct", so "INCORRECT" doesn't get misread as a positive verdict via
    substring matching. If what remains still contains "correct", the
    verdict is positive; otherwise (including when only "incorrect" was
    present, or neither word appears) it's negative."""
    text = judge_response.strip().lower()
    text_without_incorrect = text.replace("incorrect", "")
    return "correct" in text_without_incorrect


# ---------------------------------------------------------------------------
# Answer-generation prompt
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = (
    "You are answering a question using ONLY the memory snippets provided "
    "below as context. These snippets come from a long history of prior "
    "conversation sessions and may be incomplete or only partially "
    "relevant. Answer the question as precisely and concisely as possible "
    "based solely on the given context.\n\n"
    "If the context does not contain enough information to answer the "
    "question, you MUST explicitly say so -- respond with a sentence that "
    "clearly states you cannot answer the question based on the given "
    "context/information (do not guess or fabricate an answer)."
)


def build_generation_prompt(question: str, retrieved_contents: list) -> list:
    context_block = "\n".join(f"- {c}" for c in retrieved_contents) if retrieved_contents else "(no relevant memories retrieved)"
    user_msg = (
        f"Context (retrieved memory snippets):\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context above:"
    )
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_judge_prompt(question_type: str, question: str, gold_answer: str, generated_answer: str) -> list:
    template = _judge_template_for(question_type)
    text = template.format(question=question, gold_answer=gold_answer, generated_answer=generated_answer)
    return [
        {"role": "system", "content": "You are a strict evaluator/judge for a question-answering benchmark."},
        {"role": "user", "content": text},
    ]


# ---------------------------------------------------------------------------
# Dataset loading / sampling
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: Path) -> list:
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"LongMemEval dataset not found at {dataset_path}. "
            f"Run `python download_datasets.py --dataset longmemeval` first."
        )
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", list(data.values()))
    return data


def sample_questions(all_questions: list, per_type: int, full: bool, question_types_filter: list = None) -> list:
    if question_types_filter:
        all_questions = [q for q in all_questions if q.get("question_type") in question_types_filter]

    if full:
        return sorted(all_questions, key=lambda q: q["question_id"])

    by_type = defaultdict(list)
    for q in all_questions:
        by_type[q.get("question_type", "unknown")].append(q)

    sampled = []
    for qtype in sorted(by_type.keys()):
        questions = sorted(by_type[qtype], key=lambda q: q["question_id"])
        sampled.extend(questions[:per_type])
    return sampled


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------

def run_one_question(adapter: MathirAdapter, question: dict, k: int) -> dict:
    question_id = question["question_id"]
    question_type = question.get("question_type", "unknown")
    question_text = question["question"]
    gold_answer = question.get("answer", "")

    project = f"longmemeval_{question_id}"

    haystack_sessions = question.get("haystack_sessions", [])
    haystack_dates = question.get("haystack_dates", [])

    # 1. Ingest
    num_ingested = 0
    for i, session in enumerate(haystack_sessions):
        date = haystack_dates[i] if i < len(haystack_dates) else "unknown-date"
        for turn in session:
            role = turn.get("role", "unknown")
            text = turn.get("content", "")
            if not text:
                continue
            content = f"[{date}] {role}: {text}"
            adapter.add(project=project, content=content, agent="longmemeval")
            num_ingested += 1

    # 2. Search
    search_start = time.monotonic()
    results = adapter.search(project=project, query=question_text, k=k)
    search_latency_ms = (time.monotonic() - search_start) * 1000.0
    retrieved_contents = [r.get("content", "") for r in results]

    # 3. Generate
    gen_messages = build_generation_prompt(question_text, retrieved_contents)
    generated_answer = llm_client.chat(gen_messages, temperature=0.0, max_tokens=512)

    # 4. Judge
    import os
    judge_model = os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL") or None
    judge_messages = build_judge_prompt(question_type, question_text, gold_answer, generated_answer)
    judge_response = llm_client.chat(judge_messages, temperature=0.0, max_tokens=32, model=judge_model)
    judge_verdict = _parse_judge_verdict(judge_response)

    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question_text,
        "ground_truth_answer": gold_answer,
        "generated_answer": generated_answer,
        "judge_verdict": judge_verdict,
        "judge_raw_response": judge_response,
        "num_ingested": num_ingested,
        "num_retrieved": len(results),
        "search_latency_ms": search_latency_ms,
    }


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: list, failed: list) -> dict:
    by_type = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        t = r["question_type"]
        by_type[t]["n"] += 1
        if r["judge_verdict"]:
            by_type[t]["correct"] += 1

    total_n = len(results)
    total_correct = sum(1 for r in results if r["judge_verdict"])
    overall_accuracy = (total_correct / total_n) if total_n else 0.0

    by_type_summary = {}
    for t, counts in sorted(by_type.items()):
        n = counts["n"]
        acc = (counts["correct"] / n) if n else 0.0
        by_type_summary[t] = {"n": n, "correct": counts["correct"], "accuracy": acc}

    return {
        "overall_accuracy": overall_accuracy,
        "overall_n": total_n,
        "overall_correct": total_correct,
        "by_question_type": by_type_summary,
        "num_failed": len(failed),
        "failed_question_ids": [f["question_id"] for f in failed],
    }


def print_summary_table(summary: dict) -> None:
    print()
    print("=" * 64)
    print("LongMemEval Results Summary")
    print("=" * 64)
    print(f"{'question_type':<32} {'n':>5} {'accuracy':>10}")
    print("-" * 64)
    for qtype, stats in summary["by_question_type"].items():
        print(f"{qtype:<32} {stats['n']:>5} {stats['accuracy']*100:>9.1f}%")
    print("-" * 64)
    print(f"{'OVERALL':<32} {summary['overall_n']:>5} {summary['overall_accuracy']*100:>9.1f}%")
    print("=" * 64)
    if summary["num_failed"]:
        print(f"WARNING: {summary['num_failed']} question(s) failed and were excluded from accuracy: "
              f"{summary['failed_question_ids']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval benchmark runner for MATHIR")
    parser.add_argument("--per-type", type=int, default=3,
                         help="Number of questions to sample per question_type (deterministic, sorted by question_id). Ignored if --full is passed.")
    parser.add_argument("--full", action="store_true",
                         help="Use every question in the dataset instead of sampling.")
    parser.add_argument("--question-types", type=str, default=None,
                         help="Comma-separated list of question_type values to restrict to.")
    parser.add_argument("--k", type=int, default=10, help="Top-k memories to retrieve per question.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
                         help="Path to write full JSON results.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH),
                         help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--daemon-url", type=str, default="http://127.0.0.1:7338")
    args = parser.parse_args()

    question_types_filter = None
    if args.question_types:
        question_types_filter = [t.strip() for t in args.question_types.split(",") if t.strip()]

    dataset_path = Path(args.dataset)
    all_questions = load_dataset(dataset_path)
    print(f"[run_longmemeval] loaded {len(all_questions)} question instances from {dataset_path}")

    questions = sample_questions(all_questions, args.per_type, args.full, question_types_filter)
    print(f"[run_longmemeval] selected {len(questions)} questions to run "
          f"({'full dataset' if args.full else f'--per-type {args.per_type}'})")

    adapter = MathirAdapter(daemon_url=args.daemon_url)
    print(f"[run_longmemeval] MATHIR daemon reachable at {args.daemon_url}")

    results = []
    failed = []

    for idx, question in enumerate(questions, start=1):
        qid = question.get("question_id", f"unknown_{idx}")
        qtype = question.get("question_type", "unknown")
        print(f"[{idx}/{len(questions)}] {qid} ({qtype}) ...", end=" ", flush=True)
        try:
            result = run_one_question(adapter, question, args.k)
            results.append(result)
            verdict_str = "CORRECT" if result["judge_verdict"] else "INCORRECT"
            print(f"{verdict_str} (retrieved={result['num_retrieved']}, "
                  f"search_latency_ms={result['search_latency_ms']:.0f})")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append({
                "question_id": qid,
                "question_type": qtype,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    summary = aggregate_results(results, failed)
    print_summary_table(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "summary": summary,
        "results": results,
        "failed": failed,
        "config": {
            "per_type": args.per_type,
            "full": args.full,
            "question_types_filter": question_types_filter,
            "k": args.k,
            "dataset": str(dataset_path),
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"[run_longmemeval] wrote full results to {output_path}")


if __name__ == "__main__":
    main()
