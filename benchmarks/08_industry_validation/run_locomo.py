#!/usr/bin/env python3
"""
LoCoMo benchmark runner for MATHIR.

Ingests each LoCoMo conversation's turns into an isolated MATHIR project,
then for every QA pair: hybrid-searches MATHIR for relevant memories,
generates an answer grounded in those memories via an LLM, and judges the
generated answer against the ground-truth answer with a binary
CORRECT/WRONG LLM judge (the "J-score" methodology used by Mem0/Zep's
published LoCoMo numbers).

Judge prompt provenance: the CORRECT/WRONG judge prompt and rubric below
are adapted, close to verbatim, from mem0ai/memory-benchmarks
(benchmarks/locomo/prompts.py, JUDGE_PROMPT / _JUDGE_TEMPLATE, no-evidence
variant), fetched directly from GitHub while building this script. The
answer-generation prompt is a condensed adaptation of that same file's
ANSWER_GENERATION_PROMPT (shortened for this first pass; the reasoning
structure and temporal-grounding / no-hedging rules are preserved).

Category mapping (matches the original LoCoMo paper and Mem0/Zep's
reference harness):
    1 = multi-hop
    2 = temporal reasoning
    3 = open-domain
    4 = single-hop
    5 = adversarial -- EXCLUDED from headline accuracy by default (both the
        original paper and Mem0/Zep's harnesses exclude it). Still run if
        explicitly requested via --categories including 5, but always
        reported separately.

Usage:
    python run_locomo.py --conversations 2 [--all] [--categories 1,2,3,4]
                          [--k 10] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mathir_adapter import MathirAdapter  # noqa: E402
import llm_client  # noqa: E402

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = BENCHMARKS_ROOT / "05_test_data" / "locomo" / "locomo10.json"
DEFAULT_OUTPUT = BENCHMARKS_ROOT / "06_results" / "current" / "locomo_results.json"

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
DEFAULT_SCORED_CATEGORIES = [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Prompts (adapted from mem0ai/memory-benchmarks benchmarks/locomo/prompts.py)
# ---------------------------------------------------------------------------

ANSWER_GENERATION_PROMPT = """You are answering a question using retrieved memories from past conversations.

## Rules
1. Read every memory below before answering -- relevant details may be anywhere in the list, not just at the top.
2. Combine facts across multiple memories about the same topic; do not rely on a single memory in isolation.
3. Prefer the most SPECIFIC detail available (a name, date, or number beats a vague description).
4. These conversations took place around {reference_date}. Reason about time relative to that date, not today's date.
5. Give a direct, specific answer. Do NOT say "not specified" or "the memories don't say" -- if any memory is relevant, give your best answer from the available evidence.
6. Never invent a name, date, or place that does not appear in the memories below.

## Memories (most relevant first)
{memories}

## Question
{question}

Answer concisely in one or two sentences, grounded only in the memories above.
"""

JUDGE_SYSTEM_PROMPT = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. Judge semantic meaning, not exact wording. Emotions/sentiments in the same positive/negative family count as paraphrases.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g. "5 months" matches "six months"). Relative dates that are consistent with a specific date in the same window are CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches.

6. **SAME REFERENT**: If the generated answer references the same named entity, person, or concept as the gold answer, mark CORRECT -- even with different phrasing or extra detail.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: Assess whether the system recalled the right fact. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels.
"""


def load_dataset() -> list:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"LoCoMo dataset not found at {DATASET_PATH}. Run "
            f"`python download_datasets.py --dataset locomo` first."
        )
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_sessions(conversation: dict):
    """Yield (session_key, date_time_str, turns) for session_1, session_2, ... in order,
    stopping when the next numbered session key is missing."""
    idx = 1
    while True:
        key = f"session_{idx}"
        if key not in conversation:
            break
        date_key = f"{key}_date_time"
        date_time = conversation.get(date_key, "")
        yield key, date_time, conversation[key]
        idx += 1


def ingest_conversation(adapter: MathirAdapter, project: str, conversation: dict) -> tuple[int, str]:
    """Save every turn of every session as a MATHIR memory. Returns (count, last_date_time)."""
    count = 0
    last_date_time = ""
    for _session_key, date_time, turns in iter_sessions(conversation):
        if date_time:
            last_date_time = date_time
        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            content = f"[{date_time}] {speaker}: {text}"
            caption = turn.get("blip_captions")
            if caption:
                content += f" [image: {caption}]"
            adapter.add(project=project, content=content, agent="locomo")
            count += 1
    return count, last_date_time


def build_memories_block(results: list) -> str:
    if not results:
        return "(No relevant memories found)"
    lines = []
    for r in results:
        lines.append(str(r.get("content", "")))
    return "\n".join(lines)


def parse_judge_verdict(raw: str) -> tuple[bool | None, str]:
    """Parse the judge's JSON response into (verdict, reasoning). verdict is None on parse failure."""
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        label = str(data.get("label", "")).strip().upper()
        reasoning = str(data.get("reasoning", ""))
        if label == "CORRECT":
            return True, reasoning
        if label == "WRONG":
            return False, reasoning
        return None, reasoning
    except (json.JSONDecodeError, AttributeError, TypeError):
        upper = text.upper()
        if "CORRECT" in upper and "WRONG" not in upper:
            return True, text
        if "WRONG" in upper:
            return False, text
        return None, text


def run(args: argparse.Namespace) -> dict:
    dataset = load_dataset()
    n_available = len(dataset)

    if args.all:
        conv_indices = list(range(n_available))
    else:
        conv_indices = list(range(min(args.conversations, n_available)))

    scored_categories = set(args.categories)

    adapter = MathirAdapter()

    answer_model = os.environ.get("MATHIR_BENCHMARK_ANSWER_MODEL") or None
    judge_model = os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL") or None
    # Reasoning/"thinking" models spend tokens on internal reasoning before
    # the visible answer -- those tokens count against max_tokens on most
    # providers, so a low ceiling truncates the response before any answer
    # appears. Defaults raised well above a plain non-reasoning model's needs.
    answer_max_tokens = int(os.environ.get("MATHIR_BENCHMARK_ANSWER_MAX_TOKENS", "16000"))
    judge_max_tokens = int(os.environ.get("MATHIR_BENCHMARK_JUDGE_MAX_TOKENS", "8000"))

    per_question_results = []
    failures = []
    ingest_stats = []

    for conv_idx in conv_indices:
        conv = dataset[conv_idx]
        project = f"locomo_conv_{conv_idx}"

        print(f"[ingest] conversation {conv_idx} -> project '{project}' ...")
        t0 = time.time()
        try:
            n_turns, last_date_time = ingest_conversation(adapter, project, conv["conversation"])
        except Exception as e:
            print(f"[ingest] FAILED for conversation {conv_idx}: {e}", file=sys.stderr)
            failures.append({"conversation_idx": conv_idx, "stage": "ingest", "error": str(e)})
            continue
        ingest_elapsed = time.time() - t0
        ingest_stats.append({"conversation_idx": conv_idx, "turns_ingested": n_turns, "elapsed_s": ingest_elapsed})
        print(f"[ingest] conversation {conv_idx}: {n_turns} turns ingested in {ingest_elapsed:.1f}s")

        reference_date = last_date_time or "2023"

        qa_list = conv.get("qa", [])
        for qa in qa_list:
            category = qa.get("category")
            if category not in scored_categories:
                continue

            question = qa.get("question", "")
            ground_truth = qa.get("answer", qa.get("adversarial_answer", ""))
            ground_truth_str = str(ground_truth)

            record = {
                "conversation_idx": conv_idx,
                "category": category,
                "question": question,
                "ground_truth_answer": ground_truth_str,
                "generated_answer": None,
                "judge_verdict": None,
                "judge_reasoning": None,
                "num_retrieved": None,
                "search_latency_ms": None,
                "error": None,
            }

            try:
                t0 = time.time()
                search_results = adapter.search(project=project, query=question, k=args.k)
                search_latency_ms = (time.time() - t0) * 1000.0
                record["num_retrieved"] = len(search_results)
                record["search_latency_ms"] = round(search_latency_ms, 1)
            except Exception as e:
                record["error"] = f"search failed: {e}"
                per_question_results.append(record)
                failures.append({"conversation_idx": conv_idx, "stage": "search", "question": question, "error": str(e)})
                continue

            try:
                memories_block = build_memories_block(search_results)
                gen_prompt = ANSWER_GENERATION_PROMPT.format(
                    reference_date=reference_date,
                    memories=memories_block,
                    question=question,
                )
                generated_answer = llm_client.chat(
                    messages=[{"role": "user", "content": gen_prompt}],
                    temperature=0.0,
                    max_tokens=answer_max_tokens,
                    model=answer_model,
                )
                record["generated_answer"] = generated_answer.strip()
            except Exception as e:
                record["error"] = f"generation failed: {e}"
                per_question_results.append(record)
                failures.append({"conversation_idx": conv_idx, "stage": "generate", "question": question, "error": str(e)})
                continue

            try:
                judge_prompt = JUDGE_PROMPT.format(
                    question=question,
                    answer=ground_truth_str,
                    response=record["generated_answer"],
                )
                judge_raw = llm_client.chat(
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=judge_max_tokens,
                    model=judge_model,
                )
                verdict, reasoning = parse_judge_verdict(judge_raw)
                record["judge_verdict"] = verdict
                record["judge_reasoning"] = reasoning
                if verdict is None:
                    record["error"] = f"judge output unparseable: {judge_raw!r}"
                    failures.append({"conversation_idx": conv_idx, "stage": "judge_parse", "question": question, "error": judge_raw})
            except Exception as e:
                record["error"] = f"judge failed: {e}"
                failures.append({"conversation_idx": conv_idx, "stage": "judge", "question": question, "error": str(e)})

            per_question_results.append(record)

    summary = summarize(per_question_results, scored_categories)
    summary["failures_count"] = len(failures)
    summary["failures"] = failures
    summary["ingest_stats"] = ingest_stats
    summary["conversations_run"] = conv_indices
    summary["categories_scored"] = sorted(c for c in scored_categories if c != 5)
    summary["adversarial_included"] = 5 in scored_categories

    output = {
        "config": {
            "conversations": conv_indices,
            "categories": sorted(scored_categories),
            "k": args.k,
            "judge_model_override": judge_model,
        },
        "summary": summary,
        "results": per_question_results,
    }
    return output


def summarize(records: list, scored_categories: set) -> dict:
    by_category = {}
    for cat in sorted(scored_categories):
        cat_records = [r for r in records if r["category"] == cat]
        n = len(cat_records)
        judged = [r for r in cat_records if r["judge_verdict"] is not None]
        n_correct = sum(1 for r in judged if r["judge_verdict"] is True)
        n_error = n - len(judged)
        accuracy = (n_correct / len(judged)) if judged else None
        by_category[cat] = {
            "category_name": CATEGORY_NAMES.get(cat, str(cat)),
            "n": n,
            "n_judged": len(judged),
            "n_correct": n_correct,
            "n_error": n_error,
            "accuracy": accuracy,
        }

    headline_cats = [c for c in scored_categories if c != 5]
    headline_records = [r for r in records if r["category"] in headline_cats]
    headline_judged = [r for r in headline_records if r["judge_verdict"] is not None]
    headline_correct = sum(1 for r in headline_judged if r["judge_verdict"] is True)
    overall_accuracy = (headline_correct / len(headline_judged)) if headline_judged else None

    result = {
        "overall_accuracy_categories_1_4": overall_accuracy,
        "overall_n_categories_1_4": len(headline_records),
        "overall_n_judged_categories_1_4": len(headline_judged),
        "by_category": by_category,
    }
    return result


def print_summary_table(summary: dict) -> None:
    print()
    print("=" * 60)
    print("LoCoMo Benchmark Results (categories 1-4 = headline; 5 = adversarial, excluded)")
    print("=" * 60)
    print(f"{'category':<14}{'n':>6}{'judged':>8}{'accuracy':>12}")
    for cat, stats in sorted(summary["by_category"].items()):
        acc_str = f"{stats['accuracy']*100:.1f}%" if stats["accuracy"] is not None else "n/a"
        label = f"{cat}={stats['category_name']}"
        marker = " (excluded)" if cat == 5 else ""
        print(f"{label:<14}{stats['n']:>6}{stats['n_judged']:>8}{acc_str:>12}{marker}")
    print("-" * 60)
    overall = summary["overall_accuracy_categories_1_4"]
    overall_str = f"{overall*100:.1f}%" if overall is not None else "n/a"
    print(
        f"OVERALL (cats 1-4): {overall_str}  "
        f"({summary['overall_n_judged_categories_1_4']}/{summary['overall_n_categories_1_4']} judged)"
    )
    if summary["failures_count"]:
        print(f"Failures (not counted in accuracy): {summary['failures_count']}")
    print("=" * 60)


def parse_categories(raw: str) -> list:
    cats = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        cats.append(int(part))
    return cats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LoCoMo benchmark against MATHIR")
    parser.add_argument("--conversations", type=int, default=2, help="Number of conversations to run (first N by index)")
    parser.add_argument("--all", action="store_true", help="Run all 10 conversations")
    parser.add_argument(
        "--categories",
        type=str,
        default="1,2,3,4",
        help="Comma-separated QA categories to run (default: 1,2,3,4; category 5=adversarial excluded unless listed)",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of memories to retrieve per question (capped at 100 server-side)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to write full JSON results")
    args = parser.parse_args()
    args.categories = parse_categories(args.categories)

    output = run(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print_summary_table(output["summary"])
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
