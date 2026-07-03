#!/usr/bin/env python3
"""
Multi-Agent Shared Memory Benchmark for MATHIR.

Tests whether MATHIR can make "dumb" (free) models smart by sharing memory
between multiple agents, and whether agents can effectively communicate
through shared MATHIR memory.

Architecture:
  Phase 1 — Ingest: One agent ingests LoCoMo conversation into MATHIR
  Phase 2 — Solo baseline: Each agent answers questions WITHOUT memory (baseline)
  Phase 3 — MATHIR-assisted: Each agent answers questions WITH MATHIR memory
  Phase 4 — Multi-agent collaboration: Orchestrator dispatches sub-questions
            to 4 agents, each writes findings to MATHIR, then a synthesizer
            reads all findings and produces the final answer

Models (all free except orchestrator):
  - mimo-v2.5-free (worker A)
  - deepseek-v4-flash-free (worker B)
  - nemotron-3-ultra-free (worker C)
  - north-mini-code-free (worker D)
  - minimax-m3 (orchestrator — paid, minimal calls)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import _env  # noqa
except ImportError:
    pass

from mathir_adapter import MathirAdapter
import llm_client

BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = BENCHMARKS_ROOT / "05_test_data" / "locomo" / "locomo10.json"
OUTPUT_DIR = BENCHMARKS_ROOT / "06_results" / "current"

# ─── Models ───────────────────────────────────────────────────────────
_ALL_FREE = {
    "mimo":     "mimo-v2.5-free",
    "deepseek": "deepseek-v4-flash-free",
    "nemotron": "nemotron-3-ultra-free",
    "north":    "north-mini-code-free",
}
_skip = set(os.environ.get("SKIP_AGENTS", "").split(",")) - {""}
FREE_AGENTS = {k: v for k, v in _ALL_FREE.items() if k not in _skip}

ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "minimax-m3")
USE_PAID_ORCHESTRATOR = os.environ.get("USE_PAID_ORCHESTRATOR", "0") == "1"

# ─── API config ───────────────────────────────────────────────────────
ZEN_KEY = os.environ.get("OPENCODE_ZEN_KEY", "")
ZEN_BASE = "https://opencode.ai/zen/v1"

MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE = "https://api.minimax.io/v1"

MATHIR_PROJECT = "multi_agent_bench"

# ─── Prompts ──────────────────────────────────────────────────────────

ANSWER_PROMPT = """You are answering a question about a past conversation.

{context}

Question: {question}

Answer concisely in one or two sentences. Be specific — use names, dates, details."""

ANSWER_WITH_MEMORY_PROMPT = """You are answering a question using retrieved memories from past conversations.

## Memories (most relevant first)
{memories}

## Question
{question}

Answer concisely in one or two sentences, grounded only in the memories above.
Give a direct, specific answer. Do NOT say "not specified" or "I don't know"."""

DECOMPOSE_PROMPT = """You are an orchestrator. Break this question into {n_agents} independent sub-questions
that different research agents can investigate in parallel using a shared memory bank.

Question: {question}

Return a JSON array of exactly {n_agents} strings, each a focused sub-question.
Example: ["What is X's job?", "When did Y happen?", "Where was Z located?", "Who was involved in W?"]

Return ONLY the JSON array, no other text."""

SYNTHESIZE_PROMPT = """You are synthesizing answers from {n_agents} research agents who investigated
sub-questions in parallel using a shared memory bank.

Original question: {question}

Agent findings:
{findings}

Synthesize a single, concise answer (1-2 sentences) that combines the best evidence
from all agents. Be specific — use names, dates, details from the findings."""

JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

Rules:
1. Partial credit: if the answer includes at least one correct item, mark CORRECT.
2. Paraphrases count as correct. Judge meaning, not exact wording.
3. Extra detail is fine — never penalize for being more specific.
4. Dates within 14 days are CORRECT.
5. Only mark WRONG if the answer contains ZERO correct information.

Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON: {{"reasoning": "one sentence", "label": "CORRECT or WRONG"}}"""


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def zen_chat(messages, model, max_tokens=1024, temperature=0.0):
    """Call OpenCode Zen or MiniMax native API depending on model."""
    import urllib.request
    is_minimax = model.startswith("MiniMax")
    api_base = MINIMAX_BASE if is_minimax else ZEN_BASE
    api_key = MINIMAX_KEY if is_minimax else ZEN_KEY
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "MATHIR-MultiAgent/1.0",
    }
    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=payload, headers=headers, method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"]
            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
            return cleaned if cleaned else raw
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"zen_chat failed after 3 attempts: {e}")


def judge_answer(question, gold, generated, judge_model="deepseek-v4-flash-free"):
    """Judge a generated answer against gold. Returns (verdict: bool, reasoning: str)."""
    prompt = JUDGE_PROMPT.format(question=question, answer=gold, response=generated)
    raw = zen_chat([{"role": "user", "content": prompt}], model=judge_model, max_tokens=256)
    try:
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            label = obj.get("label", "").upper().strip()
            return "CORRECT" in label, obj.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        pass
    return "CORRECT" in raw.upper(), raw[:200]


def ingest_conversation(adapter: MathirAdapter, conv, conv_idx: int):
    """Ingest a LoCoMo conversation into MATHIR."""
    project = MATHIR_PROJECT
    conversation = conv.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "A")
    speaker_b = conversation.get("speaker_b", "B")
    count = 0
    sessions = sorted(k for k in conversation if k.startswith("session_") and not k.endswith("date_time"))
    for sess_key in sessions:
        date_key = f"{sess_key}_date_time"
        date_str = conversation.get(date_key, "")
        turns = conversation.get(sess_key, [])
        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            if not text:
                continue
            content = f"[{date_str}] [{speaker}]: {text}" if date_str else f"[{speaker}]: {text}"
            adapter.add(
                project=project,
                content=content,
                agent="ingestor",
                block_type="episodic",
                label=f"conv{conv_idx}_{sess_key}_turn{count}",
                priority=5,
            )
            count += 1
    return count


def get_memories(adapter: MathirAdapter, question: str, k: int = 10):
    """Retrieve relevant memories for a question."""
    results = adapter.hybrid_search(project=MATHIR_PROJECT, query=question, k=k)
    memories = results.get("results", results.get("memories", []))
    if not memories:
        results = adapter.recall(project=MATHIR_PROJECT, query=question, k=k)
        memories = results.get("results", results.get("memories", []))
    return memories


def build_memories_block(memories, max_chars=8000):
    """Format memories into a text block for the prompt."""
    if not memories:
        return "(no memories found)"
    lines = []
    total = 0
    for i, m in enumerate(memories, 1):
        text = m.get("content", m.get("text", ""))
        if total + len(text) > max_chars:
            break
        lines.append(f"[Memory {i}] {text}")
        total += len(text)
    return "\n".join(lines)


# ─── Phase 2: Solo baseline (no memory) ──────────────────────────────

def solo_baseline(question, model):
    """Answer a question without any memory — pure model capability."""
    prompt = ANSWER_PROMPT.format(
        context="You have no prior context about this conversation.",
        question=question,
    )
    return zen_chat([{"role": "user", "content": prompt}], model=model, max_tokens=512)


# ─── Phase 3: MATHIR-assisted single agent ───────────────────────────

def mathir_assisted(question, model, adapter):
    """Answer using MATHIR memory retrieval."""
    memories = get_memories(adapter, question)
    mem_block = build_memories_block(memories)
    prompt = ANSWER_WITH_MEMORY_PROMPT.format(memories=mem_block, question=question)
    return zen_chat([{"role": "user", "content": prompt}], model=model, max_tokens=512)


# ─── Phase 4: Multi-agent collaboration ──────────────────────────────

def multi_agent_collab(question, adapter, agent_models, orchestrator_model=None):
    """
    Orchestrator decomposes question → N agents investigate via MATHIR →
    each agent writes findings → synthesizer reads all findings.

    If USE_PAID_ORCHESTRATOR=0, uses a free model for decomposition too.
    """
    n_agents = len(agent_models)

    # Step 1: Decompose question into sub-questions
    decomp_model = orchestrator_model if USE_PAID_ORCHESTRATOR else "deepseek-v4-flash-free"
    decompose_prompt = DECOMPOSE_PROMPT.format(question=question, n_agents=n_agents)
    raw = zen_chat([{"role": "user", "content": decompose_prompt}], model=decomp_model, max_tokens=512)

    try:
        m = re.search(r'\[[\s\S]*\]', raw)
        sub_questions = json.loads(m.group(0)) if m else [question] * n_agents
    except (json.JSONDecodeError, AttributeError):
        sub_questions = [question] * n_agents

    if len(sub_questions) < n_agents:
        sub_questions += [question] * (n_agents - len(sub_questions))
    sub_questions = sub_questions[:n_agents]

    # Step 2: Each agent investigates its sub-question via MATHIR
    findings = []
    agent_names = list(agent_models.keys())
    for i, (agent_name, model) in enumerate(agent_models.items()):
        sub_q = sub_questions[i]
        memories = get_memories(adapter, sub_q)
        mem_block = build_memories_block(memories, max_chars=4000)

        prompt = ANSWER_WITH_MEMORY_PROMPT.format(memories=mem_block, question=sub_q)
        answer = zen_chat([{"role": "user", "content": prompt}], model=model, max_tokens=512)

        # Agent writes its findings to shared MATHIR memory
        adapter.add(
            content=f"Agent {agent_name} investigating '{sub_q}': {answer}",
            agent=f"agent_{agent_name}",
            block_type="working_memory",
            label=f"finding_{agent_name}",
            priority=7,
            project=MATHIR_PROJECT,
        )

        findings.append(f"Agent {agent_name} ({model}): {answer}")

    # Step 3: Synthesize all findings
    synth_model = orchestrator_model if USE_PAID_ORCHESTRATOR else "deepseek-v4-flash-free"
    findings_text = "\n\n".join(findings)
    synth_prompt = SYNTHESIZE_PROMPT.format(
        question=question, findings=findings_text, n_agents=n_agents,
    )
    final = zen_chat([{"role": "user", "content": synth_prompt}], model=synth_model, max_tokens=512)

    return final, sub_questions, findings


def main():
    if not ZEN_KEY:
        print("ERROR: OPENCODE_ZEN_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("MATHIR Multi-Agent Shared Memory Benchmark")
    print("=" * 60)
    print(f"Free agents:     {list(FREE_AGENTS.values())}")
    print(f"Orchestrator:    {ORCHESTRATOR_MODEL} (paid={USE_PAID_ORCHESTRATOR})")
    print(f"MATHIR project:  {MATHIR_PROJECT}")
    print()

    adapter = MathirAdapter()
    dataset = load_dataset()
    conv = dataset[0]  # Use first conversation

    # Phase 1: Ingest
    print("[Phase 1] Ingesting conversation into MATHIR...")
    t0 = time.time()
    n_turns = ingest_conversation(adapter, conv, 0)
    t_ingest = time.time() - t0
    print(f"  Ingested {n_turns} turns in {t_ingest:.1f}s")

    # Select questions (cats 1-4, skip adversarial)
    qa_pairs = [qa for qa in conv.get("qa", []) if qa.get("category", 0) in [1, 2, 3, 4]]
    max_q = int(os.environ.get("MAX_QUESTIONS", "20"))
    qa_pairs = qa_pairs[:max_q]
    print(f"  {len(qa_pairs)} questions selected (max {max_q})")
    print()

    results = []
    judge_model = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash-free")

    for qi, qa in enumerate(qa_pairs):
        question = qa["question"]
        gold = qa["answer"]
        cat = qa.get("category", "?")
        cat_name = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}.get(cat, "?")
        print(f"[Q{qi+1}/{len(qa_pairs)}] cat={cat_name}: {question[:80]}...")

        result = {
            "question": question,
            "gold_answer": gold,
            "category": cat,
            "category_name": cat_name,
        }

        # Phase 2: Solo baseline (each free agent, no memory)
        baselines = {}
        for agent_name, model in FREE_AGENTS.items():
            try:
                ans = solo_baseline(question, model)
                verdict, reasoning = judge_answer(question, gold, ans, judge_model)
                baselines[agent_name] = {
                    "answer": ans, "correct": verdict, "reasoning": reasoning,
                }
                mark = "OK" if verdict else "X"
                print(f"  baseline {agent_name}: [{mark}]")
            except Exception as e:
                baselines[agent_name] = {"answer": "", "correct": False, "error": str(e)}
                print(f"  baseline {agent_name}: [ERR] {e}")
        result["baselines"] = baselines

        # Phase 3: MATHIR-assisted (each free agent, with memory)
        assisted = {}
        for agent_name, model in FREE_AGENTS.items():
            try:
                ans = mathir_assisted(question, model, adapter)
                verdict, reasoning = judge_answer(question, gold, ans, judge_model)
                assisted[agent_name] = {
                    "answer": ans, "correct": verdict, "reasoning": reasoning,
                }
                mark = "OK" if verdict else "X"
                print(f"  +MATHIR  {agent_name}: [{mark}]")
            except Exception as e:
                assisted[agent_name] = {"answer": "", "correct": False, "error": str(e)}
                print(f"  +MATHIR  {agent_name}: [ERR] {e}")
        result["mathir_assisted"] = assisted

        # Phase 4: Multi-agent collaboration
        try:
            final, sub_qs, findings = multi_agent_collab(
                question, adapter, FREE_AGENTS,
                orchestrator_model=ORCHESTRATOR_MODEL,
            )
            verdict, reasoning = judge_answer(question, gold, final, judge_model)
            result["multi_agent"] = {
                "answer": final,
                "correct": verdict,
                "reasoning": reasoning,
                "sub_questions": sub_qs,
                "agent_findings": findings,
            }
            mark = "OK" if verdict else "X"
            print(f"  COLLAB:  [{mark}]")
        except Exception as e:
            result["multi_agent"] = {"answer": "", "correct": False, "error": str(e)}
            print(f"  COLLAB:  [ERR] {e}")

        results.append(result)
        print()

    # ─── Summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    n = len(results)

    # Baseline per agent
    print(f"\n{'Agent':<12} {'Baseline':>10} {'+ MATHIR':>10} {'Delta':>8}")
    print("-" * 42)
    for agent_name in FREE_AGENTS:
        base_correct = sum(1 for r in results if r["baselines"].get(agent_name, {}).get("correct"))
        base_judged = sum(1 for r in results if "correct" in r["baselines"].get(agent_name, {}))
        math_correct = sum(1 for r in results if r["mathir_assisted"].get(agent_name, {}).get("correct"))
        math_judged = sum(1 for r in results if "correct" in r["mathir_assisted"].get(agent_name, {}))

        base_pct = f"{base_correct}/{base_judged}" if base_judged else "n/a"
        math_pct = f"{math_correct}/{math_judged}" if math_judged else "n/a"

        if base_judged and math_judged:
            delta = (math_correct/math_judged - base_correct/base_judged) * 100
            delta_str = f"{delta:+.0f}pp"
        else:
            delta_str = ""

        print(f"{agent_name:<12} {base_pct:>10} {math_pct:>10} {delta_str:>8}")

    # Multi-agent
    collab_correct = sum(1 for r in results if r.get("multi_agent", {}).get("correct"))
    collab_judged = sum(1 for r in results if "correct" in r.get("multi_agent", {}))
    print(f"\n{'Multi-agent collab':<20}: {collab_correct}/{collab_judged}")

    # Per category
    print(f"\n{'Category':<15} {'Baseline avg':>12} {'+ MATHIR avg':>12} {'Collab':>8}")
    print("-" * 50)
    for cat in [1, 2, 3, 4]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        cat_name = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}[cat]

        base_scores = []
        math_scores = []
        for r in cat_results:
            for agent_name in FREE_AGENTS:
                b = r["baselines"].get(agent_name, {})
                if "correct" in b:
                    base_scores.append(1 if b["correct"] else 0)
                m = r["mathir_assisted"].get(agent_name, {})
                if "correct" in m:
                    math_scores.append(1 if m["correct"] else 0)

        base_pct = f"{sum(base_scores)/len(base_scores)*100:.0f}%" if base_scores else "n/a"
        math_pct = f"{sum(math_scores)/len(math_scores)*100:.0f}%" if math_scores else "n/a"
        collab_n = sum(1 for r in cat_results if r.get("multi_agent", {}).get("correct"))
        collab_t = sum(1 for r in cat_results if "correct" in r.get("multi_agent", {}))
        collab_str = f"{collab_n}/{collab_t}" if collab_t else "n/a"

        print(f"{cat_name:<15} {base_pct:>12} {math_pct:>12} {collab_str:>8}")

    # Save results
    output_path = OUTPUT_DIR / "multi_agent_bench.json"
    summary = {
        "benchmark": "MATHIR Multi-Agent Shared Memory",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "free_agents": FREE_AGENTS,
            "orchestrator": ORCHESTRATOR_MODEL,
            "judge": judge_model,
            "use_paid_orchestrator": USE_PAID_ORCHESTRATOR,
        },
        "n_questions": n,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
