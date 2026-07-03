#!/usr/bin/env python3
"""
Multi-LLM benchmark via Groq (ultra-fast LPU inference).

Tests multiple models on the 5 originally-failed LongMemEval questions
with the same MATHIR-retrieved context. Isolates LLM reasoning quality
from retrieval quality.

Then runs the best-performing model on all 30 LongMemEval questions
to get a full accuracy comparison vs MiniMax-M2.7 (83.3%).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _env  # noqa
except ImportError:
    pass

CHECKPOINT = Path(__file__).resolve().parent.parent / "06_results" / "current" / "longmemeval_e5small_30q.json.jsonl"
OUTPUT = Path(__file__).resolve().parent.parent / "06_results" / "current" / "groq_multi_llm_results.json"

GROQ_BASE = "https://api.groq.com/openai/v1"

MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
]

JUDGE_MODEL = "llama-3.3-70b-versatile"

FAILED_QIDS = ["031748ae_abs", "09ba9854", "1903aded", "09d032c9", "0a34ad58"]


def groq_chat(messages, model, temperature=0.0, max_tokens=2048, retries=3):
    api_key = os.environ.get("GROQ_API_KEY", "")
    url = f"{GROQ_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "MATHIR-Benchmark/1.0",
    }
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"]
            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
            return cleaned if cleaned else raw
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"[429, wait {wait}s] ", end="", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq HTTP {e.code}: {err_body[:200]}") from e


def load_all_trials():
    trials = []
    with CHECKPOINT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("kind") not in ("header", "summary"):
                trials.append(obj)
    return trials


def build_answer_prompt(trial):
    ctx = trial.get("retrieved_top_k_contents", "")
    if isinstance(ctx, list):
        ctx = "\n".join(str(c) for c in ctx)
    return f"""You are answering a question using ONLY the retrieved memory context below.
Be precise and direct. If the context doesn't contain enough information, say so clearly.

RETRIEVED CONTEXT:
{ctx}

QUESTION: {trial['question_text']}

Answer concisely:"""


def build_judge_prompt(question, ground_truth, answer):
    return f"""You are a strict judge. Evaluate whether the answer is correct.

Question: {question}
Ground truth: {ground_truth}
Model answer: {answer}

Does the answer correctly match the ground truth in substance?
First line: exactly CORRECT or INCORRECT.
Second line: brief reason."""


def run_on_subset(trials, models, label=""):
    all_results = {}
    for model_name in models:
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model_results = []
        correct = 0

        for i, trial in enumerate(trials, 1):
            qid = trial["question_id"]
            qtype = trial.get("question_type", "?")
            print(f"  [{i}/{len(trials)}] {qid} ({qtype}) ... ", end="", flush=True)

            prompt = build_answer_prompt(trial)
            try:
                answer = groq_chat(
                    [{"role": "user", "content": prompt}],
                    model=model_name, temperature=0.0, max_tokens=2048,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                model_results.append({"question_id": qid, "error": str(e)})
                time.sleep(3)
                continue

            time.sleep(1)

            jp = build_judge_prompt(trial["question_text"], trial["ground_truth_answer"], answer)
            try:
                judge_raw = groq_chat(
                    [{"role": "user", "content": jp}],
                    model=JUDGE_MODEL, temperature=0.0, max_tokens=512,
                )
            except Exception as e:
                print(f"JUDGE ERR: {e}")
                model_results.append({"question_id": qid, "error": f"judge: {e}"})
                time.sleep(3)
                continue

            first_line = judge_raw.strip().split("\n")[0].upper()
            is_correct = "CORRECT" in first_line and "INCORRECT" not in first_line
            if is_correct:
                correct += 1
            print("CORRECT" if is_correct else "WRONG")

            model_results.append({
                "question_id": qid,
                "question_type": qtype,
                "ground_truth": trial["ground_truth_answer"],
                "answer": answer[:500],
                "judge_verdict": is_correct,
                "judge_raw": judge_raw[:300],
            })
            time.sleep(1)

        n_valid = len([r for r in model_results if "error" not in r])
        acc = correct / n_valid if n_valid else 0
        all_results[model_name] = {
            "correct": correct,
            "total": n_valid,
            "accuracy": acc,
            "results": model_results,
        }
        print(f"\n  >> {short}: {correct}/{n_valid} ({acc*100:.1f}%)")

    return all_results


def main():
    all_trials = load_all_trials()
    failed_trials = [t for t in all_trials if t["question_id"] in FAILED_QIDS]

    print(f"Phase 1: {len(failed_trials)} failed questions x {len(MODELS)} models")
    print(f"Phase 2: best model on all {len(all_trials)} questions\n")

    # Phase 1: test all models on the 5 failed questions
    phase1 = run_on_subset(failed_trials, MODELS, "failed-5")

    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS: 5 originally-failed questions")
    print("=" * 70)
    print(f"{'Model':45s} {'Score':>10s}")
    print("-" * 57)
    print(f"{'MiniMax-M2.7 (baseline)':45s} {'0/5 (0%)':>10s}")
    for model_name, r in sorted(phase1.items(), key=lambda x: -x[1]["accuracy"]):
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"{short:45s} {r['correct']}/{r['total']} ({r['accuracy']*100:.0f}%)")

    # Phase 2: best model on all 30 questions
    best_model = max(phase1, key=lambda k: phase1[k]["accuracy"])
    best_acc = phase1[best_model]["accuracy"]
    print(f"\nBest model: {best_model} ({best_acc*100:.0f}% on failed subset)")
    print(f"\nPhase 2: running {best_model} on ALL {len(all_trials)} questions...")

    phase2 = run_on_subset(all_trials, [best_model], "all-30")

    print("\n" + "=" * 70)
    print("PHASE 2: Full 30-question comparison")
    print("=" * 70)
    orig_correct = sum(1 for t in all_trials if t.get("judge_verdict") is True)
    print(f"{'MiniMax-M2.7 (e5-small)':45s} {orig_correct}/{len(all_trials)} ({orig_correct/len(all_trials)*100:.1f}%)")
    for model_name, r in phase2.items():
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        print(f"{short + ' (e5-small, same ctx)':45s} {r['correct']}/{r['total']} ({r['accuracy']*100:.1f}%)")

    output_data = {
        "phase1_failed_5": phase1,
        "phase2_all_30": phase2,
        "best_model": best_model,
        "baseline": {"model": "MiniMax-M2.7", "correct": orig_correct, "total": len(all_trials)},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
