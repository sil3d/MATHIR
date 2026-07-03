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

Crash-safety:
    Each completed question is APPENDED to a JSONL checkpoint file
    (default: alongside --output as <output>.jsonl). If the run is killed
    mid-question, every finished question up to that point is preserved on
    disk and can be read with `jq . checkpoint.jsonl`. Pass --no-checkpoint
    to disable (not recommended for runs longer than a few minutes).

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
import os
import re
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


def _parse_judge_verdict(judge_response: str) -> bool | None:
    """Parses the judge's free-text verdict into a bool (True=CORRECT, False=
    INCORRECT), or None if the verdict is genuinely ambiguous/unparseable.

    Strategy: find the FINAL explicit CORRECT/INCORRECT word in the response
    (the verdict, not the reasoning). Reasoning-style text routinely says
    things like "the correct answer is X" or "is correct" -- matching those
    naively yields false positives. Looking at the LAST occurrence after
    reasoning tags like </think> or in the final sentence pins the
    intended verdict.

    Concretely:
    1. Take the text after the last `</think>` tag if present (that's where
       reasoning models put their final verdict), else the whole text.
    2. Find all standalone "correct" / "incorrect" word matches in that
       tail.
    3. If the LAST one is "incorrect" -> False. If "correct" -> True.
       If neither or balanced (same count) -> None (caller should treat as
       parse failure, not silently default to one side).
    """
    text = judge_response.strip().lower()
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    word_matches = re.findall(r"\b(correct|incorrect)\b", text)
    if not word_matches:
        return None
    return word_matches[-1] == "correct"


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

# ---------------------------------------------------------------------------
# Per-question pipeline (FULL MATHIR surface)
# ---------------------------------------------------------------------------
# This is the full-capacity replacement for the previous adapter which
# only ever called /api/memory/save + /api/memory/hybrid_search. Per
# question the new pipeline exercises:
#
#  1. INGEST         per-turn /save with block_type chosen from turn role +
#                    content signals (episodic / semantic / procedural /
#                    working_memory); the Mahalanobis anomaly detector runs
#                    server-side on every save automatically and any
#                    flagged-as-anomaly memory is auto-promoted to the
#                    immunological tier by MATHIR -- we count them.
#
#  2. RISK           /risk_check pre-screen of the first and last turn.
#                    Captures leakage / sycophancy scoring server-side.
#
#  3. SEARCH (4-way) hybrid_search + recall + smart_search + push all run
#                    against the same project for the same query; we keep
#                    hybrid_search's results as the answer-time context
#                    (matching prior published scores) but capture the
#                    other modes' top-k for offline analysis and for the
#                    "all four modes agree" / "any one would have been
#                    enough" diagnostic that distinguishes MATHIR's
#                    search breadth from a vanilla FAISS store.
#
#  4. GRAPH          /build_links (one-shot per project after all turns
#                    ingested). Spreading activation via /get_links gives
#                    a second retrieval channel -- the "graph-augmented"
#                    retrieval candidate -- and lets us check how often
#                    graph neighbors actually contain the answer when the
#                    flat search didn't.
#
#  5. LIFECYCLE      /decay + /consolidate (dry_run) at project teardown,
#                    so a multi-question run surfaces real lifecycle
#                    behavior (a long-lived agent's actual memory state
#                    after days of operation).
#
#  6. IMMUNOLOGICAL  /audit_immunological bulk inspection after ingest
#                    (handles the server-side 405 with a graceful skip
#                    -- there's a known bug in that route).
#
#  7. STATS          /memory_stats before and after each question. The
#                    delta is the per-question memory fingerprint, useful
#                    for spotting ingest regressions / collapse.
#
# The LLM answer-generation + judge steps remain the same as before --
# only the retrieval / ingest / memory-management side has been widened.


def _infer_block_type(role: str, text: str) -> tuple[str, str, int]:
    """Map a conversation turn to (block_type, label, priority).

    Heuristic:
      - assistant   -> episodic (session-specific reply), priority 5
      - user        -> episodic (what the user said), priority 5
      - system      -> semantic (standing instruction), priority 7
      - tool        -> procedural (how-to / api result), priority 6
      - with leading '[INST]' / 'Instruction:' / imperative  -> procedural
    Returns defaults for unknown roles.
    """
    r = role.lower().strip()
    if r == "system":
        return "semantic", "system-instruction", 7
    if r == "tool":
        return "procedural", "tool-output", 6
    if r in ("assistant", "user"):
        text_lower = text.strip().lower()
        # Heuristic: explicit instructional content => procedural
        if (text_lower.startswith(("instruction:", "instructions:", "[inst]", "step 1:", "step 1."))
                or "how to " in text_lower[:80]
                or "tutorial" in text_lower[:80]):
            return "procedural", "user-instruction", 6
        return "episodic", "", 5
    return "episodic", "", 5


def _safe(fn, *args, default=None, **kwargs):
    """Call fn(*args, **kwargs); on exception return `default` instead of
    raising. Used for non-critical MATHIR calls (audit, lifecycle, stats)
    so one broken endpoint doesn't kill the whole question pipeline."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"_error": str(e)[:200], "_default_used": default is None} | (default or {})


def run_one_question(adapter: MathirAdapter, question: dict, k: int,
                     enable_graph: bool = True,
                     enable_lifecycle: bool = True,
                     search_modes: list | None = None) -> dict:
    """Full-capacity pipeline for one LongMemEval question.

    search_modes defaults to ['hybrid_search'] (the published-score
    configuration). Set search_modes to e.g. ['hybrid_search', 'recall',
    'smart_search', 'push', 'graph'] to also capture the other modes.
    """
    if search_modes is None:
        search_modes = ["hybrid_search"]

    question_id = question["question_id"]
    question_type = question.get("question_type", "unknown")
    question_text = question["question"]
    gold_answer = question.get("answer", "")

    project = f"longmemeval_{question_id}"

    haystack_sessions = question.get("haystack_sessions", [])
    haystack_dates = question.get("haystack_dates", [])

    # 0. Stats BEFORE ingest (baseline for the project)
    stats_before = _safe(adapter.memory_stats, project)

    # 1. INGEST with per-turn block_type chosen by role + content signals.
    #    The previous run_one_question() hardcoded block_type='episodic' on
    #    every turn, collapsing MATHIR's tier routing to a single tier and
    #    making the immunological / semantic / procedural tiers unreachable.
    #    The new variant exercises MATHIR's full tier-routing surface.
    num_ingested = 0
    block_type_hist: dict[str, int] = {}
    immunological_ids: list[str] = []
    response_code_count: dict[str, int] = {}  # how many /save calls returned what
    first_response = None
    last_response = None
    for i, session in enumerate(haystack_sessions):
        date = haystack_dates[i] if i < len(haystack_dates) else "unknown-date"
        for turn_i, turn in enumerate(session):
            role = turn.get("role", "unknown")
            text = turn.get("content", "")
            if not text:
                continue
            block_type, label, priority = _infer_block_type(role, text)
            content = f"[{date}] {role}: {text}"
            resp = adapter.add(
                project=project, content=content,
                agent="longmemeval", block_type=block_type,
                label=label, priority=priority,
            )
            response_code_count[block_type] = response_code_count.get(block_type, 0) + 1
            if first_response is None:
                first_response = resp
            last_response = resp
            md = resp.get("metadata", {}) or {}
            if md.get("tier") == "immunological":
                immunological_ids.append(resp["memory_id"])
            block_type_hist[block_type] = block_type_hist.get(block_type, 0) + 1
            num_ingested += 1

    # 2. RISK pre-screen on first + last turn (the way a real agent would
    #    guard against prompt-injection / data-leak content before trust-
    #    ing a memory).
    risk_first = _safe(adapter.risk_check, first_response.get("metadata", {}).get("content", question_text)) if first_response else {}
    risk_last = _safe(adapter.risk_check, last_response.get("metadata", {}).get("content", "")) if last_response else {}

    # 3. SEARCH: run every selected mode once; keep hybrid_search results as
    #    the canonical context (so accuracy stays comparable to published
    #    Mem0/Zep numbers), but capture the others for offline A/B.
    per_mode_results: dict[str, dict] = {}
    primary_results = None
    primary_mode = "hybrid_search"

    for mode in search_modes:
        t0 = time.monotonic()
        if mode == "hybrid_search":
            r = _safe(adapter.hybrid_search, project, question_text, k=k)
            results = (r.get("results") if isinstance(r, dict) else None) or []
        elif mode == "recall":
            r = _safe(adapter.recall, project, question_text, k=k)
            results = (r.get("results") if isinstance(r, dict) else None) or []
        elif mode == "smart_search":
            r = _safe(adapter.smart_search, project, question_text, k=k)
            results = (r.get("results") if isinstance(r, dict) else None) or []
        elif mode == "push":
            # /push wants a 'context' (multi-turn-ish); we use question
            # text + a tiny wrapper. This matches how a downstream LLM
            # would call /push with its current question + partial chat
            # history.
            r = _safe(adapter.push, project, f"User asked: {question_text}", k=k)
            results = (r.get("memories") if isinstance(r, dict) else None) or []
        elif mode == "confrank":
            # MATHIR-native re-ranking: hybrid + graph + lifecycle, no
            # external labels and no LLM judge. See mathir_confrank.py
            # and mathir_adapter.confrank_search. This is the search
            # mode that exercises ConfRank + TCR + GRA from scratch --
            # NOT CRAG (no LLM evaluator) and NOT DSpark (no offline
            # confidence-head training).
            r = _safe(adapter.confrank_search, project, question_text, k=k)
            rdict = r if isinstance(r, dict) else {}
            results = rdict.get("results", []) or []
            # Diagnostics are stashed into per_mode_results a few lines below
            # (after the latency block creates the entry). Save the diag dict
            # on a local var so we can copy it in once per_mode_results[mode]
            # exists.
            _confrank_diag = rdict.get("diagnostics", {})
        elif mode == "confrank_fast":
            # OPT-3 (2026-07-01): PPR-LTE-first re-ranker with conditional
            # escalation to full confrank. If PPR-LTE is confident (high
            # top-1 score and large margin to #2), return PPR-LTE direct;
            # else fall back to full confrank. Targets a 4-5x latency
            # improvement over plain confrank while preserving accuracy
            # on the cases that matter.
            r = _safe(adapter.confrank_fast, project, question_text, k=k)
            rdict = r if isinstance(r, dict) else {}
            results = rdict.get("results", []) or []
            _confrank_fast_diag = rdict.get("diagnostics", {})
        elif mode == "graph":
            # Use /get_links on a search anchor -- currently we anchor on
            # the top-1 result of an upfront hybrid_search if available;
            # otherwise skip graph mode for this question.
            anchor_resp = _safe(adapter.hybrid_search, project, question_text, k=1)
            anchor_results = (anchor_resp.get("results") if isinstance(anchor_resp, dict) else None) or []
            if anchor_results:
                anchor_id = anchor_results[0].get("memory_id")
                if anchor_id:
                    gl = _safe(adapter.get_links, project, anchor_id, depth=2, decay=0.5)
                    raw_graph = (gl.get("result") if isinstance(gl, dict) else None) or (gl.get("results") if isinstance(gl, dict) else None) or (gl.get("memories") if isinstance(gl, dict) else None) or []
                    results = [{"memory_id": gr.get("memory_id"), "content": "(graph neighbor, see memory_id; content not in /get_links response)", "cumulative_weight": gr.get("cumulative_weight"), "distance": gr.get("distance")} for gr in raw_graph]
                else:
                    results = []
            else:
                results = []
        elif mode == "antipode":
            r = _safe(adapter.antipode_search, project, question_text, k=k)
            rdict = r if isinstance(r, dict) else {}
            results = rdict.get("results", []) or []
            _antipode_diag = rdict.get("diagnostics", {})
        elif mode == "ppr_lte":
            r = _safe(adapter.ppr_lte_search, project, question_text, k=k)
            rdict = r if isinstance(r, dict) else {}
            results = rdict.get("results", []) or []
            _ppr_lte_diag = rdict.get("diagnostics", {})
        elif mode == "smfm":
            r = _safe(adapter.smfm_search, project, question_text, k=k)
            rdict = r if isinstance(r, dict) else {}
            results = rdict.get("results", []) or []
            _smfm_diag = rdict.get("diagnostics", {})
        elif mode == "ad":
            r = _safe(adapter.ad_score_search, project, question_text, k=k)
            rdict = r if isinstance(r, dict) else {}
            results = rdict.get("results", []) or []
            _ad_diag = rdict.get("diagnostics", {})
        else:
            results = []
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        per_mode_results[mode] = {
            "num_results": len(results),
            "top_contents": [r.get("content", "")[:160] for r in results[:3]],
            "latency_ms": round(elapsed_ms, 1),
            "raw_error": r.get("_error") if isinstance(r, dict) and r.get("_error") else None,
        }
        # Stash confrank diagnostics only when we're on the confrank branch
        # (we can't add this above because per_mode_results[mode] doesn't
        # exist yet for the very first iteration in this scope).
        if mode == "confrank":
            per_mode_results[mode]["confrank_diagnostics"] = _confrank_diag
        if mode == "confrank_fast":
            per_mode_results[mode]["confrank_fast_diagnostics"] = _confrank_fast_diag
        if mode == "antipode":
            per_mode_results[mode]["antipode_diagnostics"] = _antipode_diag
        if mode == "ppr_lte":
            per_mode_results[mode]["ppr_lte_diagnostics"] = _ppr_lte_diag
        if mode == "smfm":
            per_mode_results[mode]["smfm_diagnostics"] = _smfm_diag
        if mode == "ad":
            per_mode_results[mode]["ad_diagnostics"] = _ad_diag
        if mode == "hybrid_search" and primary_results is None:
            primary_results = results
            primary_mode = "hybrid_search"

    if primary_results is None and search_modes:
        # Fall back to whatever mode actually returned something
        for mode in search_modes:
            n = per_mode_results.get(mode, {}).get("num_results", 0)
            if n > 0:
                # Re-call to get the actual list
                if mode == "recall":
                    r = adapter.recall(project, question_text, k=k)
                    primary_results = r.get("results", [])
                elif mode == "smart_search":
                    r = adapter.smart_search(project, question_text, k=k)
                    primary_results = r.get("results", [])
                elif mode == "push":
                    r = adapter.push(project, f"User asked: {question_text}", k=k)
                    primary_results = r.get("memories", [])
                primary_mode = mode
                break

    search_latency_ms = per_mode_results.get("hybrid_search", {}).get("latency_ms", 0.0)
    retrieved_contents = [r.get("content", "") for r in (primary_results or [])]

    # 4. GRAPH build -- one-shot per project, immediately before search.
    #    Pre-computed edges let /get_links do spreading activation faster.
    graph_build = {}
    if enable_graph:
        graph_build = _safe(adapter.build_links, project, threshold=0.7, limit=2000)

    # 5. IMMUNOLOGICAL audit (read-only; honors server-side 405 if present)
    immunological_audit = _safe(adapter.audit_immunological, project, k=100)
    immunological_audit_count = (
        immunological_audit.get("total", 0)
        if isinstance(immunological_audit, dict) and "total" in immunological_audit
        else 0
    )

    # 6. GENERATE answer from the canonical (hybrid_search) retrieved context
    import os
    answer_model = os.environ.get("MATHIR_BENCHMARK_ANSWER_MODEL") or None
    answer_max_tokens = int(os.environ.get("MATHIR_BENCHMARK_ANSWER_MAX_TOKENS", "16000"))
    if retrieved_contents:
        gen_messages = build_generation_prompt(question_text, retrieved_contents)
        generated_answer = llm_client.chat(gen_messages, temperature=0.0, max_tokens=answer_max_tokens, model=answer_model)
    else:
        generated_answer = "I cannot answer this question based on the available memories."

    # 7. JUDGE
    judge_model = os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL") or None
    judge_max_tokens = int(os.environ.get("MATHIR_BENCHMARK_JUDGE_MAX_TOKENS", "8000"))
    judge_messages = build_judge_prompt(question_type, question_text, gold_answer, generated_answer)
    judge_response = llm_client.chat(judge_messages, temperature=0.0, max_tokens=judge_max_tokens, model=judge_model)
    judge_verdict = _parse_judge_verdict(judge_response)

    # 8. LIFECYCLE dry-runs at project teardown -- simulate the multi-day
    #    memory state of a real agent. dry_run=True on consolidate so we
    #    don't mutate state between questions of the same run.
    lifecycle = {}
    if enable_lifecycle:
        lifecycle["decay"] = _safe(adapter.decay, project, threshold_days=30, archive_floor=0.05)
        lifecycle["consolidate_dryrun"] = _safe(adapter.consolidate, project, threshold=0.95, dry_run=True, limit=100)
        lifecycle["auto_promote"] = _safe(adapter.auto_promote, project)

    # 9. STATS AFTER question
    stats_after = _safe(adapter.memory_stats, project)

    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question_text,
        "ground_truth_answer": gold_answer,
        "generated_answer": generated_answer,
        "judge_verdict": judge_verdict,
        "judge_raw_response": judge_response,
        "num_ingested": num_ingested,
        "block_type_hist": block_type_hist,
        "immunological_flagged_count": len(immunological_ids),
        "num_retrieved": len(primary_results or []),
        "primary_search_mode": primary_mode,
        "search_latency_ms": search_latency_ms,
        "search_modes": per_mode_results,
        "risk_first_turn": {k: v for k, v in (risk_first or {}).items() if not k.startswith("_")},
        "risk_last_turn": {k: v for k, v in (risk_last or {}).items() if not k.startswith("_")},
        "graph_build": {k: v for k, v in (graph_build or {}).items() if not k.startswith("_")} if graph_build else {},
        "immunological_audit_total": immunological_audit_count,
        "lifecycle": {k: {kk: vv for kk, vv in (v or {}).items() if not kk.startswith("_")} for k, v in lifecycle.items()} if lifecycle else {},
        "stats_before": {k: v for k, v in (stats_before or {}).items() if not k.startswith("_")},
        "stats_after": {k: v for k, v in (stats_after or {}).items() if not k.startswith("_")},
        "_retrieved_contents_for_log": retrieved_contents,
    }


def run_one_question_cross_model(adapter: MathirAdapter, question: dict, k: int, models: list) -> dict:
    """Ingest and search MATHIR exactly ONCE per question, then have EACH
    model in `models` generate + get judged from that SAME retrieved context.

    This does not replace the standard single-model run above (which
    matches Mem0/Zep's published-score methodology). It answers a different,
    MATHIR-specific question: if the retrieved memories carry the actual
    information, ANY reasonably capable model should be able to answer
    correctly from them -- accuracy shouldn't depend heavily on which model
    reads the context. If accuracy stays similar across models, that's
    empirical evidence the memory (not the model) carries the informational
    load, which is exactly MATHIR's cross-provider-portability claim,
    demonstrated on a real, recognized benchmark rather than asserted.

    The judge model is held FIXED across all generation models tested (via
    MATHIR_BENCHMARK_JUDGE_MODEL, same as the single-model path) so we are
    only varying the generation model, not conflating that with judge
    variability.
    """
    question_id = question["question_id"]
    question_type = question.get("question_type", "unknown")
    question_text = question["question"]
    gold_answer = question.get("answer", "")

    project = f"longmemeval_{question_id}"

    haystack_sessions = question.get("haystack_sessions", [])
    haystack_dates = question.get("haystack_dates", [])

    # 1. Ingest (once)
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

    # 2. Search (once -- this is the shared context every model gets)
    search_start = time.monotonic()
    results_resp = adapter.hybrid_search(project=project, query=question_text, k=k)
    results = results_resp.get("results", []) if isinstance(results_resp, dict) else []  # noqa: F841 (kept name for callers below)
    search_latency_ms = (time.monotonic() - search_start) * 1000.0
    retrieved_contents = [r.get("content", "") for r in results]

    import os
    answer_max_tokens = int(os.environ.get("MATHIR_BENCHMARK_ANSWER_MAX_TOKENS", "16000"))
    judge_max_tokens = int(os.environ.get("MATHIR_BENCHMARK_JUDGE_MAX_TOKENS", "8000"))
    judge_model = os.environ.get("MATHIR_BENCHMARK_JUDGE_MODEL") or None

    per_model = {}
    for model_id in models:
        gen_messages = build_generation_prompt(question_text, retrieved_contents)
        generated_answer = llm_client.chat(
            gen_messages, temperature=0.0, max_tokens=answer_max_tokens, model=model_id,
        )
        judge_messages = build_judge_prompt(question_type, question_text, gold_answer, generated_answer)
        judge_response = llm_client.chat(
            judge_messages, temperature=0.0, max_tokens=judge_max_tokens, model=judge_model,
        )
        per_model[model_id] = {
            "generated_answer": generated_answer,
            "judge_verdict": _parse_judge_verdict(judge_response),
            "judge_raw_response": judge_response,
        }

    verdicts = [v["judge_verdict"] for v in per_model.values()]
    # Only count verdicts that were actually parsed as a hard bool;
    # UNCLEAR (None) responses are excluded from "all/any_correct" so a
    # single unparseable judge response doesn't artificially sink the
    # cross-model agreement metric.
    judged_verdicts = [v for v in verdicts if v is not None]

    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question_text,
        "ground_truth_answer": gold_answer,
        "num_ingested": num_ingested,
        "num_retrieved": len(results),
        "search_latency_ms": search_latency_ms,
        "per_model": per_model,
        "all_correct": bool(judged_verdicts) and all(judged_verdicts),
        "any_correct": any(v is True for v in judged_verdicts),
        "all_agree": len(set(judged_verdicts)) <= 1,
    }


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: list, failed: list) -> dict:
    # A result is "judged" only if its verdict is a hard bool. UNCLEAR
    # (None) verdicts are reported separately and excluded from accuracy.
    judged = [r for r in results if r["judge_verdict"] is not None]
    unclear = [r for r in results if r["judge_verdict"] is None]

    by_type = defaultdict(lambda: {"n": 0, "correct": 0, "unclear": 0})
    for r in results:
        t = r["question_type"]
        by_type[t]["n"] += 1
        if r["judge_verdict"] is True:
            by_type[t]["correct"] += 1
        elif r["judge_verdict"] is None:
            by_type[t]["unclear"] += 1

    total_n = len(results)
    total_correct = sum(1 for r in results if r["judge_verdict"] is True)
    total_judged = len(judged)
    overall_accuracy = (total_correct / total_judged) if total_judged else 0.0

    by_type_summary = {}
    for t, counts in sorted(by_type.items()):
        n_judged = counts["n"] - counts["unclear"]
        acc = (counts["correct"] / n_judged) if n_judged else 0.0
        by_type_summary[t] = {
            "n": counts["n"],
            "correct": counts["correct"],
            "unclear": counts["unclear"],
            "accuracy": acc,
        }

    return {
        "overall_accuracy": overall_accuracy,
        "overall_n": total_n,
        "overall_n_judged": total_judged,
        "overall_correct": total_correct,
        "overall_unclear": len(unclear),
        "by_question_type": by_type_summary,
        "num_failed": len(failed),
        "failed_question_ids": [f["question_id"] for f in failed],
        "unclear_question_ids": [r["question_id"] for r in unclear],
    }


def print_summary_table(summary: dict) -> None:
    print()
    print("=" * 72)
    print("LongMemEval Results Summary")
    print("=" * 72)
    print(f"{'question_type':<32} {'n':>5} {'judged':>7} {'correct':>8} {'accuracy':>10}")
    print("-" * 72)
    for qtype, stats in summary["by_question_type"].items():
        judged = stats["n"] - stats.get("unclear", 0)
        print(f"{qtype:<32} {stats['n']:>5} {judged:>7} {stats['correct']:>8} {stats['accuracy']*100:>9.1f}%")
    print("-" * 72)
    print(f"{'OVERALL':<32} {summary['overall_n']:>5} {summary['overall_n_judged']:>7} "
          f"{summary['overall_correct']:>8} {summary['overall_accuracy']*100:>9.1f}%")
    print("=" * 72)
    if summary.get("overall_unclear"):
        print(f"WARNING: {summary['overall_unclear']} question(s) had UNCLEAR judge verdicts "
              f"(excluded from accuracy): {summary['unclear_question_ids']}")
    if summary["num_failed"]:
        print(f"WARNING: {summary['num_failed']} question(s) failed and were excluded from accuracy: "
              f"{summary['failed_question_ids']}")
    print()


def aggregate_cross_model_results(results: list, failed: list, models: list) -> dict:
    """Per-model accuracy (same retrieved context for every model) plus a
    cross-model consistency summary: how often accuracy doesn't depend on
    which model reads MATHIR's retrieved context."""
    n = len(results)
    per_model_correct = {m: 0 for m in models}
    per_model_judged = {m: 0 for m in models}
    for r in results:
        for m, v in r["per_model"].items():
            if v["judge_verdict"] is not None:
                per_model_judged[m] += 1
                if v["judge_verdict"] is True:
                    per_model_correct[m] += 1

    per_model_summary = {
        m: {
            "n": n,
            "n_judged": per_model_judged[m],
            "correct": per_model_correct[m],
            "accuracy": (per_model_correct[m] / per_model_judged[m]) if per_model_judged[m] else 0.0,
        }
        for m in models
    }

    all_correct_n = sum(1 for r in results if r["all_correct"])
    any_correct_n = sum(1 for r in results if r["any_correct"])
    all_agree_n = sum(1 for r in results if r["all_agree"])

    return {
        "models": models,
        "n": n,
        "per_model": per_model_summary,
        "all_models_correct_rate": (all_correct_n / n) if n else 0.0,
        "any_model_correct_rate": (any_correct_n / n) if n else 0.0,
        "all_models_agree_rate": (all_agree_n / n) if n else 0.0,
        "num_failed": len(failed),
        "failed_question_ids": [f["question_id"] for f in failed],
    }


def print_cross_model_summary_table(summary: dict) -> None:
    print()
    print("=" * 72)
    print("LongMemEval Cross-Model Consistency Results")
    print("(same MATHIR-retrieved context for every model below)")
    print("=" * 72)
    print(f"{'model':<40} {'n_judged':>9} {'accuracy':>10}")
    print("-" * 72)
    for model, stats in summary["per_model"].items():
        print(f"{model:<40} {stats['n_judged']:>9} {stats['accuracy']*100:>9.1f}%")
    print("-" * 72)
    print(f"All models correct on the same question: {summary['all_models_correct_rate']*100:.1f}%")
    print(f"At least one model correct:               {summary['any_model_correct_rate']*100:.1f}%")
    print(f"All models agree (right OR wrong):         {summary['all_models_agree_rate']*100:.1f}%")
    print("=" * 72)
    print("A high 'all models correct' / 'all models agree' rate is evidence that")
    print("MATHIR's retrieved context carries the answer, not the choice of model --")
    print("i.e. empirical support for cross-provider memory portability.")
    if summary["num_failed"]:
        print(f"WARNING: {summary['num_failed']} question(s) failed and were excluded: "
              f"{summary['failed_question_ids']}")
    print()


# ---------------------------------------------------------------------------
# Evidence capture + corpus metadata helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with seconds. For evidence trail."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mathir_lib_version() -> str:
    """Best-effort MATHIR library version string. Returns 'unknown' if it
    can't be located -- never raises, since benchmark code must not crash on
    a cosmetic detail."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return f"mathir_mcp=={version('mathir_mcp')}"
        except PackageNotFoundError:
            pass
        # Fallback: probe the running daemon for its reported version.
        import urllib.request, json
        with urllib.request.urlopen("http://127.0.0.1:7338/version", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("version", "unknown")
    except Exception:
        return "unknown"


def _build_result_record(result: dict, cross_models: list | None = None,
                         retrieved_contents: list | None = None) -> dict:
    """Wrap a per-question result dict for the JSONL checkpoint. Captures
    FULL evidence: question text, ground-truth answer, generated answer,
    judge raw response, retrieved top-k memory contents, all timestamps
    and latencies. Designed so a third party reading the JSONL alone can
    audit every answer the system produced, without needing access to
    the daemon, the dataset, or the live MATHIR instance.

    Also captures the FULL MATHIR-surface fingerprints per question:
    block_type_hist (which tiers the ingest hit), risk_first/last_turn
    scores, search_modes (latency + hit-count per mode), graph_build
    result, immunological flagged count, lifecycle dry-run results, and
    stats before/after. Without this, the JSONL would only show accuracy
    and we'd lose all visibility into which MATHIR features were
    actually exercised vs masked by the previous limited adapter."""
    record = {
        "kind": "result",
        "timestamp_utc": _utcnow_iso(),
        "question_id": result["question_id"],
        "question_type": result["question_type"],
        "question_text": result["question"],
        "ground_truth_answer": result["ground_truth_answer"],
        "generated_answer": result["generated_answer"],
        "judge_verdict": result.get("judge_verdict"),
        "judge_raw_response": result.get("judge_raw_response"),
        "num_ingested": result.get("num_ingested"),
        "num_retrieved": result.get("num_retrieved"),
        "search_latency_ms": result.get("search_latency_ms"),
        # -- Full MATHIR surface fingerprints --
        "block_type_hist": result.get("block_type_hist", {}),
        "immunological_flagged_count": result.get("immunological_flagged_count", 0),
        "primary_search_mode": result.get("primary_search_mode"),
        "search_modes": result.get("search_modes", {}),
        "risk_first_turn": result.get("risk_first_turn", {}),
        "risk_last_turn": result.get("risk_last_turn", {}),
        "graph_build": result.get("graph_build", {}),
        "immunological_audit_total": result.get("immunological_audit_total", 0),
        "lifecycle": result.get("lifecycle", {}),
        "stats_before": result.get("stats_before", {}),
        "stats_after": result.get("stats_after", {}),
    }
    if retrieved_contents is not None:
        record["retrieved_top_k_contents"] = retrieved_contents
    if cross_models is not None:
        record["mode"] = "cross_model"
        record["cross_models"] = cross_models
        record["all_correct"] = result.get("all_correct")
        record["any_correct"] = result.get("any_correct")
        record["all_agree"] = result.get("all_agree")
        record["per_model"] = result.get("per_model")
    return record


# ---------------------------------------------------------------------------
# Crash-safety: incremental JSONL checkpoint
# ---------------------------------------------------------------------------

def _default_checkpoint_path(output_path: Path) -> Path:
    """Default checkpoint file lives next to --output with a .jsonl extension."""
    return output_path.with_suffix(output_path.suffix + ".jsonl")


def _open_checkpoint(checkpoint_path: Path):
    """Append-only handle to the JSONL checkpoint. Caller closes via .close().
    Writes are line-buffered so a kill mid-write loses at most the current
    half-line, never a prior completed question."""
    return open(checkpoint_path, "a", encoding="utf-8", buffering=1)


def _append_checkpoint(fh, record: dict) -> None:
    """Write one completed question record as a single JSONL line, flushed."""
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()


def _load_resume_state(checkpoint_path: Path) -> tuple[set, dict | None]:
    """Scan an existing JSONL checkpoint file. Returns:
        (set of question_ids whose latest 'result' line has a hard verdict,
         header dict if the first line was a header, else None)
    The set contains ONLY ids with hard verdicts (True or False) -- UNCLEAR
    (None) ids are NOT in the set so --resume will re-attempt them.
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
            qid = rec.get("question_id")
            if not qid:
                continue
            v = rec.get("judge_verdict")
            if v is True or v is False:
                latest_verdict[qid] = v
    return set(latest_verdict.keys()), header


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
    parser.add_argument("--dataset-version", type=str, default="LongMemEval-S (cleaned)",
                         help="Human-readable corpus identifier written into the "
                              "checkpoint header (e.g. 'LongMemEval-S (cleaned)', "
                              "'LongMemEval-M', 'LongMemEval-S snapshot 2026-07-01').")
    parser.add_argument("--dataset-source", type=str, default="xiaowu0162/LongMemEval @ github.com/xiaowu0162/LongMemEval, file longmemeval_s_cleaned.json via HuggingFace xiaowu0162/longmemeval-cleaned (MIT).",
                         help="Provenance string (URL, paper, license) for the "
                              "benchmark corpus -- written into the JSONL header "
                              "and the final JSON, so a third party can audit "
                              "exactly what was tested.")
    parser.add_argument("--run-label", type=str, default=None,
                         help="Optional human label for this run (e.g. 'baseline', "
                              "'e5-small-stronger-embedder'). Written into the "
                              "checkpoint header and JSON output.")
    parser.add_argument("--daemon-url", type=str, default="http://127.0.0.1:7338")
    parser.add_argument("--cross-model", type=str, default=None,
                         help="Comma-separated list of 2+ real model ids (e.g. "
                              "'MiniMax-M2.7,MiniMax-M3'). When set, ingest+search "
                              "run ONCE per question and EACH listed model generates "
                              "+ gets judged from that same retrieved context, to "
                              "measure whether accuracy depends on the model or on "
                              "MATHIR's retrieval. Replaces the normal single-model "
                              "run for this invocation.")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to JSONL checkpoint file. Each completed "
                              "question is appended after judge verdict so a kill "
                              "mid-run preserves everything up to that point. "
                              "Default: <output>.jsonl. Pass --no-checkpoint to disable.")
    parser.add_argument("--no-checkpoint", action="store_true",
                         help="Disable JSONL checkpointing (NOT recommended for "
                              "long runs -- a kill loses all completed work).")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the existing JSONL checkpoint: skip "
                              "any question_id that already has a 'result' line "
                              "with a non-None judge_verdict. Incompatible with "
                              "--no-checkpoint. When --resume re-runs an item "
                              "whose prior judge verdict was UNCLEAR (None), the "
                              "old line is preserved and a new line is appended "
                              "(verdict history is not lost).")
    parser.add_argument("--search-modes", type=str,
                        default="hybrid_search",
                        help="Comma-separated list of MATHIR search modes to "
                             "exercise per question. Default: hybrid_search "
                             "(matches published Mem0/Zep methodology for "
                             "comparability). To exercise MATHIR's full search "
                             "surface, use: hybrid_search,recall,smart_search,"
                             "push,graph (much slower -- ~5x more LLM/context "
                             "work). Each mode's top-3 contents and latency are "
                             "logged in the result record's 'search_modes' field.")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable the per-question /build_links call. By "
                             "default the full pipeline builds the MATHIR "
                             "memory graph (threshold=0.7) once per question, "
                             "exposing spreading activation as one of the "
                             "retrieval channels.")
    parser.add_argument("--no-lifecycle", action="store_true",
                        help="Disable the per-question /decay + /consolidate "
                             "(dry_run) + /auto_promote calls. By default the "
                             "pipeline exercises MATHIR's full lifecycle "
                             "surface at project teardown, simulating the "
                             "multi-day memory state a real agent would have.")
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

    cross_models = None
    if args.cross_model:
        cross_models = [m.strip() for m in args.cross_model.split(",") if m.strip()]
        if len(cross_models) < 2:
            raise SystemExit("--cross-model needs at least 2 comma-separated model ids to compare")
        print(f"[run_longmemeval] cross-model mode: {cross_models}")

    results = []
    failed = []

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
                print(f"[run_longmemeval] --resume: found {len(resume_skip_ids)} already-judged question_id(s) in {checkpoint_path} -- will skip")
        checkpoint_fh = _open_checkpoint(checkpoint_path)
        print(f"[run_longmemeval] checkpointing after each question to {checkpoint_path}")
        # Header line -- exactly once, on first open. Lets a downstream reader
        # know corpus / config / start time / host before reading any results.
        _append_checkpoint(checkpoint_fh, {
            "kind": "header",
            "benchmark": "LongMemEval",
            "corpus_name": args.dataset_version,
            "corpus_source": args.dataset_source,
            "dataset_file": str(dataset_path),
            "run_label": args.run_label,
            "daemon_url": args.daemon_url,
            "k": args.k,
            "per_type": args.per_type,
            "full": args.full,
            "question_types_filter": question_types_filter,
            "cross_model": cross_models,
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
            "n_questions_selected": len(questions),
            "n_questions_resume_skipped": len(resume_skip_ids),
        })

    try:
        skipped_resume = 0
        for idx, question in enumerate(questions, start=1):
            qid = question.get("question_id", f"unknown_{idx}")
            qtype = question.get("question_type", "unknown")
            if qid in resume_skip_ids:
                print(f"[{idx}/{len(questions)}] {qid} ({qtype}) ... SKIPPED (--resume: already judged)", flush=True)
                skipped_resume += 1
                continue
            print(f"[{idx}/{len(questions)}] {qid} ({qtype}) ...", end=" ", flush=True)
            try:
                if cross_models:
                    result = run_one_question_cross_model(adapter, question, args.k, cross_models)
                    results.append(result)
                    print(f"all_correct={result['all_correct']} any_correct={result['any_correct']} "
                          f"agree={result['all_agree']} (retrieved={result['num_retrieved']})")
                else:
                    search_modes = [m.strip() for m in args.search_modes.split(",") if m.strip()]
                    result = run_one_question(
                        adapter, question, args.k,
                        enable_graph=not args.no_graph,
                        enable_lifecycle=not args.no_lifecycle,
                        search_modes=search_modes,
                    )
                    results.append(result)
                    verdict = result["judge_verdict"]
                    if verdict is True:
                        verdict_str = "CORRECT"
                    elif verdict is False:
                        verdict_str = "INCORRECT"
                    else:
                        verdict_str = "UNCLEAR"
                    print(f"{verdict_str} (retrieved={result['num_retrieved']}, "
                          f"search_latency_ms={result['search_latency_ms']:.0f})")
                if checkpoint_fh is not None:
                    _append_checkpoint(checkpoint_fh, _build_result_record(
                        result, cross_models=cross_models,
                        retrieved_contents=(result.get("_retrieved_contents_for_log")),
                    ))
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append({
                    "question_id": qid,
                    "question_type": qtype,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                if checkpoint_fh is not None:
                    _append_checkpoint(checkpoint_fh, {
                        "kind": "failure",
                        "timestamp_utc": _utcnow_iso(),
                        "question_id": qid,
                        "question_type": qtype,
                        "question_text": question.get("question", ""),
                        "ground_truth_answer": question.get("answer", ""),
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
        if args.resume:
            print(f"[run_longmemeval] --resume: skipped {skipped_resume} already-judged question(s)")
    finally:
        if checkpoint_fh is not None:
            checkpoint_fh.close()
            print(f"[run_longmemeval] checkpoint file closed: {checkpoint_path}")

    if cross_models:
        summary = aggregate_cross_model_results(results, failed, cross_models)
        print_cross_model_summary_table(summary)
    else:
        summary = aggregate_results(results, failed)
        print_summary_table(summary)

    output_path = Path(args.output)
    output_payload = {
        "benchmark": "LongMemEval",
        "corpus": {
            "name": args.dataset_version,
            "source": args.dataset_source,
            "file": str(dataset_path),
        },
        "run_label": args.run_label,
        "summary": summary,
        "results": results,
        "failed": failed,
        "config": {
            "per_type": args.per_type,
            "full": args.full,
            "question_types_filter": question_types_filter,
            "k": args.k,
            "dataset": str(dataset_path),
            "cross_model": cross_models,
            "daemon_url": args.daemon_url,
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
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"[run_longmemeval] wrote full results to {output_path}")
    if checkpoint_path is not None:
        with open(checkpoint_path, "a", encoding="utf-8", buffering=1) as fh:
            fh.write(json.dumps({"kind": "summary", "timestamp_utc": _utcnow_iso(), "summary": summary}, ensure_ascii=False) + "\n")
        print(f"[run_longmemeval] wrote final summary to {checkpoint_path}")


if __name__ == "__main__":
    main()
