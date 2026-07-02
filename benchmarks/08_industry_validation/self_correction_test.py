#!/usr/bin/env python3
"""
Tests whether MATHIR-retrieved context helps MiniMax models SELF-CORRECT
their own errors, rather than just measuring first-pass accuracy.

Real question from the session goal: "il faut voir si MiniMax 2.7/3
arrive a corriger leurs erreurs grace a MATHIR ameliore la recherche des
resultats, rendre le tout plus intelligent."

Uses the 5 real INCORRECT trials from the earlier e5-small LongMemEval
run (benchmarks/06_results/current/longmemeval_e5small_30q.json.jsonl) --
each already has the question, the model's wrong answer, the ground
truth, the judge's critique (WHY it was wrong), and the actual MATHIR-
retrieved context (already backed by e5-small, the improved embedder).

Protocol per failed question:
  1. Show the ANSWER model (MiniMax-M2.7) its own prior wrong answer +
     the JUDGE's critique (MiniMax-M3's reasoning for why it was wrong)
     + the SAME MATHIR-retrieved context it had originally.
  2. Ask it to produce a corrected answer.
  3. Re-judge the corrected answer with MiniMax-M3.
  4. Measure: did self-correction succeed on cases that were originally wrong?

This isolates "can the model use feedback + already-good retrieval to
fix itself" from "did retrieval get the right answer on the first try" --
a genuinely different, real question about MATHIR's downstream effect on
model intelligence, not just retrieval quality.
"""
from __future__ import annotations

import json
from pathlib import Path

from llm_client import chat

CHECKPOINT = Path(__file__).resolve().parent.parent / "06_results" / "current" / "longmemeval_e5small_30q.json.jsonl"
OUTPUT = Path(__file__).resolve().parent.parent / "06_results" / "current" / "self_correction_results.json"

ANSWER_MODEL = "MiniMax-M2.7"
JUDGE_MODEL = "MiniMax-M3"


def load_wrong_trials():
    trials = []
    with CHECKPOINT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") not in ("header", "summary") and obj.get("judge_verdict") is False:
                trials.append(obj)
    return trials


def build_self_correction_prompt(trial: dict) -> str:
    context = trial.get("retrieved_top_k_contents", "")
    if isinstance(context, list):
        context = "\n".join(str(c) for c in context)
    return f"""You previously answered a question incorrectly. You are given the same
retrieved memory context, your prior wrong answer, and a critique explaining
why it was wrong. Use this feedback to produce a corrected answer.

RETRIEVED CONTEXT:
{context}

QUESTION: {trial['question_text']}

YOUR PRIOR (WRONG) ANSWER: {trial['generated_answer']}

CRITIQUE OF WHY IT WAS WRONG: {trial.get('judge_raw_response', '')}

Given the context, the question, your prior mistake, and the critique, provide
a corrected, accurate answer. Be direct -- do not just restate the critique."""


def judge_prompt(question: str, ground_truth: str, corrected_answer: str) -> str:
    return f"""Question: {question}
Ground truth answer: {ground_truth}
Model's answer: {corrected_answer}

Does the model's answer correctly and fully address the question, matching
the ground truth in substance (not necessarily exact wording)? Respond with
exactly one word: CORRECT or INCORRECT. Then on a new line, briefly explain why."""


def main():
    wrong_trials = load_wrong_trials()
    print(f"[self_correction] {len(wrong_trials)} originally-incorrect trials to attempt self-correction on")

    results = []
    for i, trial in enumerate(wrong_trials, 1):
        qid = trial["question_id"]
        print(f"[{i}/{len(wrong_trials)}] {qid} ({trial['question_type']}) ...", end=" ", flush=True)

        correction_prompt = build_self_correction_prompt(trial)
        try:
            corrected_answer = chat(
                [{"role": "user", "content": correction_prompt}],
                temperature=0.0, max_tokens=1024, model=ANSWER_MODEL,
            )
        except Exception as e:
            print(f"FAILED (correction call): {e}")
            results.append({"question_id": qid, "error": str(e)})
            continue

        jp = judge_prompt(trial["question_text"], trial["ground_truth_answer"], corrected_answer)
        try:
            judge_raw = chat(
                [{"role": "user", "content": jp}],
                temperature=0.0, max_tokens=512, model=JUDGE_MODEL,
            )
        except Exception as e:
            print(f"FAILED (judge call): {e}")
            results.append({"question_id": qid, "error": str(e)})
            continue

        self_corrected = "INCORRECT" not in judge_raw.upper().split("\n")[0] and "CORRECT" in judge_raw.upper()
        verdict_str = "SELF-CORRECTED" if self_corrected else "STILL WRONG"
        print(verdict_str)

        results.append({
            "question_id": qid,
            "question_type": trial["question_type"],
            "question_text": trial["question_text"],
            "ground_truth": trial["ground_truth_answer"],
            "original_wrong_answer": trial["generated_answer"],
            "original_critique": trial.get("judge_raw_response", ""),
            "corrected_answer": corrected_answer,
            "judge_raw": judge_raw,
            "self_corrected": self_corrected,
        })

    n_corrected = sum(1 for r in results if r.get("self_corrected"))
    n_total = len([r for r in results if "error" not in r])
    print(f"\n=== SELF-CORRECTION RESULTS ===")
    print(f"{n_corrected}/{n_total} originally-wrong answers were self-corrected using MATHIR context + judge feedback")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump({"n_corrected": n_corrected, "n_total": n_total, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
