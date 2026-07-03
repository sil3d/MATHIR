#!/usr/bin/env python3
"""
Multi-LLM benchmark: test whether stronger FREE models (OpenRouter) succeed
on the 5 questions that MiniMax-M2.7 failed on with the same MATHIR context.

Isolates "LLM reasoning quality" from "retrieval quality" — same MATHIR
retrieved context, different answer models.

Uses the existing llm_client.py with OpenRouter backend.
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
OUTPUT = Path(__file__).resolve().parent.parent / "06_results" / "current" / "multi_llm_comparison.json"

MODELS = [
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

JUDGE_MODEL = "openai/gpt-oss-120b:free"

FAILED_QIDS = ["031748ae_abs", "09ba9854", "1903aded", "09d032c9", "0a34ad58"]


def openrouter_chat(messages, model, temperature=0.0, max_tokens=2048, retries=3):
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MATHIR_API_KEY", "")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    referer = os.environ.get("MATHIR_OPENROUTER_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"]
            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
            return cleaned if cleaned else raw
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"[429 rate-limit, waiting {wait}s] ", end="", flush=True)
                time.sleep(wait)
                continue
            raise


def load_failed_trials():
    trials = []
    with CHECKPOINT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("question_id") in FAILED_QIDS:
                trials.append(obj)
    return trials


def build_answer_prompt(trial):
    ctx = trial.get("retrieved_top_k_contents", "")
    if isinstance(ctx, list):
        ctx = "\n".join(str(c) for c in ctx)
    return f"""You are answering a question using ONLY the retrieved memory context below.
Be precise and direct. If the context doesn't contain enough information, say so.

RETRIEVED CONTEXT:
{ctx}

QUESTION: {trial['question_text']}

Answer:"""


def build_judge_prompt(question, ground_truth, answer):
    return f"""You are a strict judge evaluating whether an answer is correct.

Question: {question}
Ground truth answer: {ground_truth}
Model's answer: {answer}

Does the model's answer correctly address the question, matching the ground truth
in substance (not necessarily exact wording)? Respond with exactly one word on the
first line: CORRECT or INCORRECT. Then briefly explain why."""


def main():
    trials = load_failed_trials()
    print(f"Testing {len(trials)} failed questions across {len(MODELS)} models\n")

    all_results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1].replace(":free", "")
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model_results = []
        correct = 0

        for trial in trials:
            qid = trial["question_id"]
            qtype = trial.get("question_type", "?")
            print(f"  [{qid}] ({qtype}) ... ", end="", flush=True)

            prompt = build_answer_prompt(trial)
            try:
                answer = openrouter_chat(
                    [{"role": "user", "content": prompt}],
                    model=model_name, temperature=0.0, max_tokens=2048,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                model_results.append({"question_id": qid, "error": str(e)})
                time.sleep(10)
                continue

            time.sleep(8)

            jp = build_judge_prompt(trial["question_text"], trial["ground_truth_answer"], answer)
            try:
                judge_raw = openrouter_chat(
                    [{"role": "user", "content": jp}],
                    model=JUDGE_MODEL, temperature=0.0, max_tokens=512,
                )
            except Exception as e:
                print(f"JUDGE ERROR: {e}")
                model_results.append({"question_id": qid, "error": f"judge: {e}"})
                time.sleep(10)
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

            time.sleep(8)

        n_valid = len([r for r in model_results if "error" not in r])
        all_results[model_name] = {
            "correct": correct,
            "total": n_valid,
            "accuracy": correct / n_valid if n_valid else 0,
            "results": model_results,
        }
        print(f"\n  >> {short}: {correct}/{n_valid} correct ({correct/n_valid*100:.0f}%)" if n_valid else "  >> no valid results")

    print("\n" + "=" * 60)
    print("SUMMARY: Same 5 failed questions, same MATHIR context, different LLMs")
    print("=" * 60)
    print(f"{'Model':50s} {'Score':>8s}")
    print("-" * 60)
    ref = {"MiniMax-M2.7 (original)": "0/5 (0%)"}
    for k, v in ref.items():
        print(f"{k:50s} {v:>8s}")
    for model_name, r in all_results.items():
        short = model_name.split("/")[-1].replace(":free", "")
        score = f"{r['correct']}/{r['total']} ({r['accuracy']*100:.0f}%)"
        print(f"{short:50s} {score:>8s}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
