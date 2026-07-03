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

Crash-safety:
    Each completed QA record is APPENDED to a JSONL checkpoint file
    (default: alongside --output as <output>.jsonl). If the run is killed
    mid-question, every finished QA up to that point is preserved on disk.
    Pass --no-checkpoint to disable. Pass --resume to skip QA ids whose
    latest 'result' line has a hard CORRECT/WRONG verdict.

Evidence trail (industrial-grade):
    Each JSONL 'result' line captures the full question text, ground-truth
    answer, generated answer, judge raw response, retrieved top-k memory
    contents, and per-QA timing. Each JSONL file starts with a 'header'
    line containing corpus name + source + LLM backend env, so a third
    party can audit exactly what was tested by reading the JSONL alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
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


# ---------------------------------------------------------------------------
# Evidence capture + corpus metadata helpers (industrial-grade)
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with seconds. For evidence trail."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mathir_lib_version() -> str:
    """Best-effort MATHIR library version string. Returns 'unknown' on failure."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return f"mathir_mcp=={version('mathir_mcp')}"
        except PackageNotFoundError:
            pass
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:7338/version", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("version", "unknown")
    except Exception:
        return "unknown"


def _qa_id(conv_idx: int, qa: dict) -> str:
    """Stable per-QA identifier for --resume dedup. Falls back to a hash
    of question text + category if no explicit id is in the dataset."""
    explicit = qa.get("id") or qa.get("qa_id")
    if explicit:
        return f"locomo_c{conv_idx}_{explicit}"
    return f"locomo_c{conv_idx}_cat{qa.get('category')}_{abs(hash(qa.get('question',''))) % 10**8}"


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
    """Save every turn of every session as a MATHIR memory. Returns (count, last_date_time).

    FIX (2026-07-01): the LoCoMo JSON has the sessions nested under
    conversation["conversation"] (alongside speaker_a / speaker_b keys),
    not at the top level. Pass conversation["conversation"] to
    iter_sessions, or strip the wrapping. We accept BOTH for backward
    compatibility: if conversation has a "conversation" sub-dict with
    session_1 in it, use that; otherwise treat conversation as the
    session dict directly.
    """
    count = 0
    last_date_time = ""
    sessions_obj = conversation
    if "conversation" in conversation and isinstance(conversation.get("conversation"), dict):
        # LoCoMo schema: { conversation: {session_1, session_1_date_time, ...}, qa: [...]}
        if any(k.startswith("session_") for k in conversation["conversation"]):
            sessions_obj = conversation["conversation"]
    for _session_key, date_time, turns in iter_sessions(sessions_obj):
        if date_time:
            last_date_time = date_time
        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            content = f"[{date_time}] {speaker}: {text}"
            caption = turn.get("blip_captions")
            if caption:
                content += f" [image: {caption}]"
            # Tier routing: LoCoMo conversations have two named speakers
            # (the human / the assistant) and rarely use system/tool roles,
            # but we keep the same heuristic as the LongMemEval pipeline
            # so the ingested memory exercises all tiers, not just
            # 'episodic'. LoCoMo QA questions span categories 1-4 which
            # benefit from semantic-tier storage of recurring facts.
            speaker_lower = speaker.lower()
            if speaker_lower in ("system", "operator"):
                block_type = "semantic"
                label = "locomo-system"
                priority = 7
            elif speaker_lower in ("tool", "function"):
                block_type = "procedural"
                label = "locomo-tool"
                priority = 6
            else:
                text_lower = text.strip().lower()
                if (text_lower.startswith(("instruction:", "[inst]", "step 1:", "step 1."))
                        or "how to " in text_lower[:80]):
                    block_type = "procedural"
                    label = "locomo-instruction"
                    priority = 6
                else:
                    block_type = "episodic"
                    label = ""
                    priority = 5
            adapter.add(
                project=project, content=content, agent="locomo",
                block_type=block_type, label=label, priority=priority,
            )
            count += 1
    return count, last_date_time


def build_memories_block(results: list, max_chars: int = 0) -> str:
    if not results:
        return "(No relevant memories found)"
    cap = max_chars or int(os.environ.get("MATHIR_BENCHMARK_CONTEXT_MAX_CHARS", "0"))
    lines = []
    total = 0
    for r in results:
        text = str(r.get("content", ""))
        if cap and total + len(text) > cap:
            remaining = cap - total
            if remaining > 100:
                lines.append(text[:remaining] + "...")
            break
        lines.append(text)
        total += len(text)
    return "\n".join(lines)


def parse_judge_verdict(raw: str) -> tuple[bool | None, str]:
    """Parse the judge's JSON response into (verdict, reasoning).

    verdict is True for CORRECT, False for WRONG, None on parse failure.

    Strategy:
    1. Try strict JSON parse. If it yields an explicit "label": "CORRECT"
       or "WRONG", trust it.
    2. Otherwise (free-text fallback), look at the FINAL CORRECT/WRONG
       word in the response (after </think> if present), the same
       "final-verdict" rule run_longmemeval uses -- this avoids the
       substring trap where reasoning text containing both words
       ("the response is partially correct but wrong overall") would
       otherwise get misclassified by naive substring matching.
    """
    text = raw.strip()
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
        pass
    # Fallback: final-word rule on the raw text, after reasoning tags.
    lower = text.lower()
    if "</think>" in lower:
        lower = lower.rsplit("</think>", 1)[1]
    words = re.findall(r"\b(correct|wrong)\b", lower)
    if not words:
        return None, text
    return words[-1] == "correct", text


def _open_checkpoint(checkpoint_path: Path):
    """Append-only handle. Caller closes via .close(). Line-buffered."""
    return open(checkpoint_path, "a", encoding="utf-8", buffering=1)


def _append_checkpoint(fh, record: dict) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


def _default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".jsonl")


def _load_resume_state(checkpoint_path: Path) -> tuple[set, dict | None]:
    """Scan an existing JSONL checkpoint. Returns:
        (set of qa_ids whose latest 'result' line has a hard verdict,
         header dict if first line was a header, else None)
    UNCLEAR (None) verdicts are excluded from the skip set so --resume
    will retry them.
    """
    if not checkpoint_path.is_file():
        return set(), None
    latest_verdict: dict[str, bool] = {}
    header = None
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("kind")
            if kind == "header":
                header = rec
                continue
            if kind != "result":
                continue
            qid = rec.get("qa_id")
            if not qid:
                continue
            v = rec.get("judge_verdict")
            if v is True or v is False:
                latest_verdict[qid] = v
    return set(latest_verdict.keys()), header


def run(args: argparse.Namespace, checkpoint_fh=None, resume_skip_ids: set | None = None) -> dict:
    if resume_skip_ids is None:
        resume_skip_ids = set()
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

            qa_id = _qa_id(conv_idx, qa)
            if qa_id in resume_skip_ids:
                print(f"[resume] skip qa_id={qa_id} cat={category} (already judged)", flush=True)
                continue

            question = qa.get("question", "")
            ground_truth = qa.get("answer", qa.get("adversarial_answer", ""))
            ground_truth_str = str(ground_truth)

            record = {
                "qa_id": qa_id,
                "conversation_idx": conv_idx,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, str(category)),
                "question": question,
                "ground_truth_answer": ground_truth_str,
                "generated_answer": None,
                "judge_verdict": None,
                "judge_reasoning": None,
                "judge_raw_response": None,
                "num_retrieved": None,
                "search_latency_ms": None,
                "error": None,
                "reference_date": reference_date,
                "retrieved_top_k_contents": None,
            }

            try:
                t0 = time.time()
                # Full MATHIR surface: try the primary hybrid_search first,
                # fall back to plain vector recall if hybrid isn't available
                # (e.g. mid-migration between versions).
                search_resp = adapter.hybrid_search(project=project, query=question, k=args.k)
                search_results = search_resp.get("results", []) if isinstance(search_resp, dict) else []
                search_latency_ms = (time.time() - t0) * 1000.0
                record["num_retrieved"] = len(search_results)
                record["search_latency_ms"] = round(search_latency_ms, 1)
                record["retrieved_top_k_contents"] = [r.get("content", "") for r in search_results]
                # Mirror the new full-capacity evidence trail on the record.
                record["primary_search_mode"] = "hybrid_search"
                # Capture raw response metadata (e.g. rrf_score) for downstream
                # analysis without re-running the search.
                for r in search_results:
                    pass  # the JSON dump below already covers it via the adapter
            except Exception as e:
                record["error"] = f"search failed: {e}"
                per_question_results.append(record)
                if checkpoint_fh is not None:
                    _append_checkpoint(checkpoint_fh, {"kind": "result", **record})
                failures.append({"conversation_idx": conv_idx, "stage": "search", "question": question, "error": str(e)})
                continue

            if getattr(args, "cross_model", None):
                # Cross-model mode: same retrieved context, EACH listed
                # model generates + gets judged (judge model held fixed).
                # See run_longmemeval.py's run_one_question_cross_model()
                # for the full rationale.
                record.pop("generated_answer", None)
                record.pop("judge_verdict", None)
                record.pop("judge_reasoning", None)
                record["per_model"] = {}
                try:
                    memories_block = build_memories_block(search_results)
                    gen_prompt = ANSWER_GENERATION_PROMPT.format(
                        reference_date=reference_date,
                        memories=memories_block,
                        question=question,
                    )
                    for model_id in args.cross_model:
                        generated_answer = llm_client.chat(
                            messages=[{"role": "user", "content": gen_prompt}],
                            temperature=0.0,
                            max_tokens=answer_max_tokens,
                            model=model_id,
                        ).strip()
                        judge_prompt = JUDGE_PROMPT.format(
                            question=question,
                            answer=ground_truth_str,
                            response=generated_answer,
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
                        record["per_model"][model_id] = {
                            "generated_answer": generated_answer,
                            "judge_verdict": verdict,
                            "judge_reasoning": reasoning,
                            "judge_raw_response": judge_raw,
                        }
                except Exception as e:
                    record["error"] = f"cross-model generate/judge failed: {e}"
                    per_question_results.append(record)
                    if checkpoint_fh is not None:
                        _append_checkpoint(checkpoint_fh, {"kind": "result", **record})
                    failures.append({"conversation_idx": conv_idx, "stage": "cross_model", "question": question, "error": str(e)})
                    continue

                verdicts = [v["judge_verdict"] for v in record["per_model"].values()]
                judged_verdicts = [v for v in verdicts if v is not None]
                record["all_correct"] = bool(judged_verdicts) and all(judged_verdicts)
                record["any_correct"] = any(v is True for v in judged_verdicts)
                record["all_agree"] = len(set(judged_verdicts)) <= 1
                per_question_results.append(record)
                if checkpoint_fh is not None:
                    _append_checkpoint(checkpoint_fh, {"kind": "result", **record})
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
                if checkpoint_fh is not None:
                    _append_checkpoint(checkpoint_fh, {"kind": "result", **record})
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
                record["judge_raw_response"] = judge_raw
                if verdict is None:
                    record["error"] = f"judge output unparseable: {judge_raw!r}"
                    failures.append({"conversation_idx": conv_idx, "stage": "judge_parse", "question": question, "error": judge_raw})
            except Exception as e:
                record["error"] = f"judge failed: {e}"
                failures.append({"conversation_idx": conv_idx, "stage": "judge", "question": question, "error": str(e)})

            per_question_results.append(record)
            if checkpoint_fh is not None:
                _append_checkpoint(checkpoint_fh, {"kind": "result", **record})

    if getattr(args, "cross_model", None):
        summary = summarize_cross_model(per_question_results, args.cross_model)
    else:
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


def summarize_cross_model(records: list, models: list) -> dict:
    """Per-model accuracy (same retrieved context for every model, categories
    1-4 only) plus a cross-model consistency summary."""
    headline_records = [r for r in records if r["category"] != 5 and "per_model" in r]
    n = len(headline_records)

    per_model_correct = {m: 0 for m in models}
    per_model_judged = {m: 0 for m in models}
    for r in headline_records:
        for m, v in r.get("per_model", {}).items():
            if v["judge_verdict"] is not None:
                per_model_judged[m] += 1
                if v["judge_verdict"] is True:
                    per_model_correct[m] += 1

    per_model_summary = {
        m: {
            "n": n,
            "n_judged": per_model_judged[m],
            "correct": per_model_correct[m],
            "accuracy": (per_model_correct[m] / per_model_judged[m]) if per_model_judged[m] else None,
        }
        for m in models
    }

    all_correct_n = sum(1 for r in headline_records if r.get("all_correct"))
    any_correct_n = sum(1 for r in headline_records if r.get("any_correct"))
    all_agree_n = sum(1 for r in headline_records if r.get("all_agree"))

    return {
        "models": models,
        "overall_n_categories_1_4": n,
        "per_model": per_model_summary,
        "all_models_correct_rate": (all_correct_n / n) if n else 0.0,
        "any_model_correct_rate": (any_correct_n / n) if n else 0.0,
        "all_models_agree_rate": (all_agree_n / n) if n else 0.0,
    }


def print_cross_model_summary_table(summary: dict) -> None:
    print()
    print("=" * 72)
    print("LoCoMo Cross-Model Consistency Results (categories 1-4)")
    print("(same MATHIR-retrieved context for every model below)")
    print("=" * 72)
    print(f"{'model':<40} {'n_judged':>10} {'accuracy':>10}")
    print("-" * 72)
    for model, stats in summary["per_model"].items():
        acc_str = f"{stats['accuracy']*100:.1f}%" if stats["accuracy"] is not None else "n/a"
        print(f"{model:<40} {stats['n_judged']:>10} {acc_str:>10}")
    print("-" * 72)
    print(f"All models correct on the same question: {summary['all_models_correct_rate']*100:.1f}%")
    print(f"At least one model correct:               {summary['any_model_correct_rate']*100:.1f}%")
    print(f"All models agree (right OR wrong):         {summary['all_models_agree_rate']*100:.1f}%")
    print("=" * 72)
    print("A high 'all models correct' / 'all models agree' rate is evidence that")
    print("MATHIR's retrieved context carries the answer, not the choice of model --")
    print("i.e. empirical support for cross-provider memory portability.")
    print()


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
    parser.add_argument("--cross-model", type=str, default=None,
                        help="Comma-separated list of 2+ real model ids (e.g. "
                             "'MiniMax-M2.7,MiniMax-M3'). When set, search runs ONCE "
                             "per question and EACH listed model generates + gets "
                             "judged from that same retrieved context, to measure "
                             "whether accuracy depends on the model or on MATHIR's "
                             "retrieval. Replaces the normal single-model run.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to JSONL checkpoint file. Each completed QA is "
                             "appended after judge verdict so a kill mid-run preserves "
                             "everything up to that point. Default: <output>.jsonl.")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable JSONL checkpointing (NOT recommended for long "
                             "runs -- a kill loses all completed work).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the existing JSONL checkpoint: skip "
                             "any qa_id that already has a 'result' line with a "
                             "hard CORRECT/WRONG verdict. UNCLEAR (None) verdicts "
                             "are retried.")
    parser.add_argument("--dataset-version", type=str, default="LoCoMo v1 (10 conversations, snap-research/locomo)",
                        help="Human-readable corpus identifier written into the "
                             "checkpoint header.")
    parser.add_argument("--dataset-source", type=str,
                        default="snap-research/locomo @ github.com/snap-research/locomo, file locomo10.json (CC BY-NC 4.0).",
                        help="Provenance string for the corpus (URL, paper, license).")
    parser.add_argument("--run-label", type=str, default=None,
                        help="Optional human label for this run.")
    args = parser.parse_args()
    args.categories = parse_categories(args.categories)
    if args.cross_model:
        cross_model = [m.strip() for m in args.cross_model.split(",") if m.strip()]
        if len(cross_model) < 2:
            raise SystemExit("--cross-model needs at least 2 comma-separated model ids to compare")
        args.cross_model = cross_model
        print(f"[run_locomo] cross-model mode: {cross_model}")
    else:
        args.cross_model = None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_fh = None
    checkpoint_path = None
    resume_skip_ids: set = set()
    if not args.no_checkpoint:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else _default_checkpoint_path(output_path)
        if args.resume:
            resume_skip_ids, _existing_header = _load_resume_state(checkpoint_path)
            if resume_skip_ids:
                print(f"[run_locomo] --resume: found {len(resume_skip_ids)} already-judged qa_id(s) in {checkpoint_path} -- will skip")
        checkpoint_fh = _open_checkpoint(checkpoint_path)
        print(f"[run_locomo] checkpointing after each QA to {checkpoint_path}")
        _append_checkpoint(checkpoint_fh, {
            "kind": "header",
            "benchmark": "LoCoMo",
            "corpus_name": args.dataset_version,
            "corpus_source": args.dataset_source,
            "dataset_file": str(DATASET_PATH),
            "run_label": args.run_label,
            "k": args.k,
            "categories": sorted(args.categories),
            "conversations": None if args.all else args.conversations,
            "all_conversations": args.all,
            "cross_model": getattr(args, "cross_model", None),
            "answer_model_env": os.environ.get("MATHIR_BENCHMARK_ANSWER_MODEL"),
            "judge_model_env": os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL"),
            "answer_max_tokens_env": os.environ.get("MATHIR_BENCHMARK_ANSWER_MAX_TOKENS"),
            "judge_max_tokens_env": os.environ.get("MATHIR_BENCHMARK_JUDGE_MAX_TOKENS"),
            "llm_backend_env": os.environ.get("MATHIR_LLM_BACKEND"),
            "llm_api_base_env": os.environ.get("MATHIR_API_BASE"),
            "llm_api_model_env": os.environ.get("MATHIR_API_MODEL"),
            "mathir_lib_version": _mathir_lib_version(),
            "script_version": "1.1.0 (industrial-grade: full evidence + resume)",
            "started_at_utc": _utcnow_iso(),
            "host": os.environ.get("COMPUTERNAME", "unknown"),
            "n_resume_skipped": len(resume_skip_ids),
        })

    try:
        output = run(args, checkpoint_fh=checkpoint_fh, resume_skip_ids=resume_skip_ids)
    finally:
        if checkpoint_fh is not None:
            checkpoint_fh.close()
            print(f"[run_locomo] checkpoint file closed: {checkpoint_path}")

    # Enrich the final JSON with corpus metadata + run config so it's
    # self-describing for downstream readers / auditors.
    output["benchmark"] = "LoCoMo"
    output["corpus"] = {
        "name": args.dataset_version,
        "source": args.dataset_source,
        "file": str(DATASET_PATH),
    }
    output["run_label"] = args.run_label
    output["config"].update({
        "answer_model_env": os.environ.get("MATHIR_BENCHMARK_ANSWER_MODEL"),
        "judge_model_env": os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL"),
        "answer_max_tokens_env": os.environ.get("MATHIR_BENCHMARK_ANSWER_MAX_TOKENS"),
        "judge_max_tokens_env": os.environ.get("MATHIR_BENCHMARK_JUDGE_MAX_TOKENS"),
        "llm_backend_env": os.environ.get("MATHIR_LLM_BACKEND"),
        "llm_api_base_env": os.environ.get("MATHIR_API_BASE"),
        "llm_api_model_env": os.environ.get("MATHIR_API_MODEL"),
        "mathir_lib_version": _mathir_lib_version(),
        "script_version": "1.1.0 (industrial-grade: full evidence + resume)",
        "started_at_utc": _utcnow_iso(),
        "finished_at_utc": _utcnow_iso(),
        "host": os.environ.get("COMPUTERNAME", "unknown"),
        "resume_used": args.resume,
        "n_resume_skipped": len(resume_skip_ids),
    })
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if checkpoint_path is not None:
        with open(checkpoint_path, "a", encoding="utf-8", buffering=1) as fh:
            fh.write(json.dumps({"kind": "summary", "summary": output["summary"]}, ensure_ascii=False) + "\n")
        print(f"[run_locomo] wrote final summary to {checkpoint_path}")

    if args.cross_model:
        print_cross_model_summary_table(output["summary"])
    else:
        print_summary_table(output["summary"])
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
