"""
AI-driven cognitive benchmark for the v8.5.0 living memory.

Simulates a real AI agent working on a long-term project:
  Phase A — Generation:   LLM creates N "experiences" (bug fixes, decisions, notes)
                         and saves each to MATHIR memory.
  Phase B — Baseline:    LLM answers K questions using memory_recall().
                         Measure recall@5, precision@5, MRR, answer quality.
  Phase C — Aging:       Simulate 30 days of inactivity (last_recalled_at = -30d).
                         Run maintenance cycle:
                           - decay_all()
                           - auto_promote_all()
                           - consolidate_all(dry_run=False)
                           - build_links_all()
  Phase D — Re-test:     LLM re-answers the SAME K questions.
                         Measure recall@5, precision@5, MRR, answer quality.

Key metric: does recall quality STAY THE SAME or IMPROVE despite archiving?

Usage:
  python ai_cognitive_bench.py --experiences 50 --questions 20 --duration 20
  python ai_cognitive_bench.py --experiences 100 --questions 30 --out results_ai.json

Environment (read at runtime, never in this file):
  MATHIR_LLM_BACKEND  : "api" | "ollama" | "auto" (default: auto)
  MATHIR_API_KEY      : required if backend=api
  MATHIR_API_BASE     : default https://api.minimax.chat/v1
  MATHIR_API_MODEL    : default MiniMax-M2.7
  MATHIR_OLLAMA_URL   : default http://127.0.0.1:11434
  MATHIR_OLLAMA_MODEL : default qwen3.5:2b
"""
import os
import sys
import json
import time
import shutil
import argparse
import random
import re
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np

# Auto-load .env from benchmarks/ root via shared helper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _env  # noqa: F401 — populates os.environ
except ImportError:
    pass

# Bootstrap
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent / "mathir_mcp"
_LIB = _PKG_ROOT / "mathir_lib"
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_LIB))

from mathir_vec import VecMemory
from llm_client import LLMClient, LLMUnavailable


# ---------------------------------------------------------------------------
# Project simulation — deterministic topic pool so the test is reproducible
# ---------------------------------------------------------------------------

PROJECT_TOPICS = [
    "authentication bug in login flow",
    "rate limiting issue in API gateway",
    "database connection pool exhausted",
    "memory leak in WebSocket handler",
    "race condition in concurrent writes",
    "timeout error on third-party API call",
    "JSON serialization crash on null fields",
    "CORS preflight failure for POST requests",
    "JWT token expiration edge case",
    "file upload size limit exceeded",
    "SQL injection vector in search query",
    "N+1 query problem in user dashboard",
    "session cookie not being set on iOS",
    "Docker image build fails on ARM64",
    "Redis cache invalidation race",
    "memory store schema migration",
    "embedding dimension mismatch in recall",
    "link graph traversal slow on 10K memories",
    "consolidate merging wrong memories",
    "decay archive threshold too aggressive",
    "promote rules not catching mature memories",
    "touch recall missing on hybrid search",
    "MCP tool registration order issue",
    "daemon port conflict on restart",
    "GPU embedder not loading in WSL",
    "cross-lingual recall failing for French queries",
    "priority field ignored in save handler",
    "audit log missing for delete operations",
    "session memory not shared across agents",
    "vector index rebuild on every save",
]

QUESTION_TEMPLATES = [
    "What is the most recent engineering issue we ran into and how did we resolve it?",
    "Which file or function had a bug recently and what was the fix?",
    "What was a recent root cause analysis we did, and what did we learn?",
    "What is one concrete technical detail from a recent debugging session?",
    "What mistake or pattern do we want to avoid in future work, based on past experience?",
    "What error message or symptom was most memorable from recent work?",
    "Which subsystem had a performance or stability issue recently?",
    "What architectural decision was made recently and why?",
    "What was the most subtle or tricky bug we found?",
    "What is one thing we know now that we did not know two weeks ago?",
]


# ---------------------------------------------------------------------------
# Generation phase — LLM creates experiences
# ---------------------------------------------------------------------------

GEN_SYSTEM = (
    "You are a senior engineer working on MATHIR (a memory system). "
    "Generate a realistic, detailed engineering note about the given topic. "
    "Write 4-6 sentences (200-400 words). Be specific and technical: include "
    "file names, function names, error messages, line numbers, and a clear "
    "description of root cause, fix, and lessons learned. Output ONLY the note, "
    "no preamble or commentary."
)


def generate_experience(llm: LLMClient, topic: str, seed_n: int) -> Dict:
    """One LLM call produces one experience."""
    prompt = (
        f"Engineering note #{seed_n}: about '{topic}'.\n"
        f"Write a detailed 4-6 sentence memory (200-400 words) describing:\n"
        f"  1. What was the symptom or observation?\n"
        f"  2. What was the root cause or decision?\n"
        f"  3. What was the specific fix (file/function/error)?\n"
        f"  4. What did we learn that applies to future work?\n"
        f"Be technical and specific. Include concrete details."
    )
    try:
        text = llm.chat(prompt, system=GEN_SYSTEM, max_tokens=400, temperature=0.7)
    except Exception as e:
        text = f"Note about {topic}: (LLM error: {e})"
    return {
        "topic": topic,
        "content": text.strip(),
        "label": f"bench-{topic[:30].replace(' ', '-')}",
    }


# ---------------------------------------------------------------------------
# Recall phase — LLM answers questions using memory
# ---------------------------------------------------------------------------

QA_SYSTEM = (
    "You are answering questions using ONLY the memory snippets provided. "
    "Read each snippet carefully. If the answer is in the snippets, give a "
    "concrete technical answer citing the snippet number [N]. If the answer "
    "is NOT in the snippets, reply exactly: 'I don't know based on these "
    "memories.' Be concise (1-3 sentences). Output ONLY the answer."
)


def _format_snippets(snippets: List[Dict], k: int = 5) -> str:
    out = []
    for i, s in enumerate(snippets[:k], 1):
        content = s.get("content") or s.get("modality_text") or ""
        label = s.get("label", "?")
        out.append(f"[{i}] (label={label}) {content}")
    return "\n\n".join(out) if out else "(no memories found)"


def answer_question(llm: LLMClient, question: str, snippets: List[Dict]) -> Tuple[str, int]:
    """Returns (answer_text, k_used). The LLM uses ONLY the snippets — no external knowledge."""
    formatted = _format_snippets(snippets, k=5)
    prompt = (
        f"Question: {question}\n\n"
        f"Memory snippets (top {len(snippets[:5])}):\n{formatted}\n\n"
        f"Answer using only the snippets above. Cite [N] if useful:"
    )
    try:
        ans = llm.chat(prompt, system=QA_SYSTEM, max_tokens=200, temperature=0.2)
    except Exception as e:
        ans = f"(LLM error: {e})"
    return ans.strip(), len(snippets[:5])


# ---------------------------------------------------------------------------
# Recall-quality scoring
# ---------------------------------------------------------------------------

def _tokenize(s: str) -> set:
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", s.lower()))


def recall_metrics(snippets: List[Dict], expected_topic: str, ground_truth_content: str) -> Dict:
    """
    Compute retrieval-quality metrics against ground truth.

    Because the question is now BLIND (no topic in it), the only way to find
    the right snippet is via the embedding/similarity. We score:
      - recall_at_k     : fraction of GT tokens found in top-K snippets
      - precision_at_k  : fraction of top-K snippets that contain GT tokens
      - mrr             : 1/rank of first useful snippet
      - has_answer      : recall_at_5 >= 0.15 (meaningful overlap)
      - top1_topic_match: whether the top-1 snippet matches the expected topic
    """
    k = 5
    retrieved = snippets[:k]
    gt_tokens = _tokenize(ground_truth_content)
    # Strip common stopwords to avoid trivial matches
    STOP = {"the", "and", "for", "with", "from", "this", "that", "have", "has",
            "was", "were", "are", "been", "but", "not", "you", "your", "our",
            "they", "their", "what", "which", "when", "where", "how", "why",
            "into", "than", "then", "also", "such", "some", "any", "all",
            "can", "could", "should", "would", "may", "might", "will", "shall"}
    gt_tokens -= STOP
    if not gt_tokens:
        return {"recall_at_5": 0, "precision_at_5": 0, "mrr": 0, "has_answer": False,
                "top1_topic_match": False}

    # Per-snippet tokens
    snippet_tokens = []
    for s in retrieved:
        toks = _tokenize(s.get("content") or s.get("modality_text") or "") - STOP
        snippet_tokens.append(toks)

    # Recall@K: union of GT tokens found across all retrieved snippets
    retrieved_tokens = set()
    for toks in snippet_tokens:
        retrieved_tokens |= toks
    recall = len(gt_tokens & retrieved_tokens) / len(gt_tokens)

    # Precision@K: fraction of retrieved snippets with any GT overlap
    useful = sum(1 for toks in snippet_tokens if toks & gt_tokens)
    precision = useful / k

    # MRR
    mrr = 0.0
    for i, toks in enumerate(snippet_tokens, 1):
        if toks & gt_tokens:
            mrr = 1.0 / i
            break

    has_answer = recall >= 0.15

    # Top-1 topic match: do the GT tokens appear in top-1?
    top1_match = bool(snippet_tokens[0] & gt_tokens) if snippet_tokens else False

    return {
        "recall_at_5": round(recall, 4),
        "precision_at_5": round(precision, 4),
        "mrr": round(mrr, 4),
        "has_answer": has_answer,
        "top1_topic_match": top1_match,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def _force_age(memory: VecMemory, days: int):
    """Set last_recalled_at = 0 and timestamp = now - days for all memories."""
    conn = sqlite3.connect(str(memory.db_path))
    try:
        sk = memory._schema_kind()
        ts = time.time() - days * 86400
        if sk == "legacy":
            conn.execute("UPDATE memories SET last_recalled_at = 0, timestamp = ?", (ts,))
        else:
            # new schema: metadata JSON
            import json
            for mid, meta_str in conn.execute("SELECT memory_id, metadata FROM memories").fetchall():
                md = json.loads(meta_str) if meta_str else {}
                md["last_recalled_at"] = 0
                md["timestamp"] = ts
                conn.execute("UPDATE memories SET metadata = ? WHERE memory_id = ?",
                             (json.dumps(md), mid))
        conn.commit()
    finally:
        conn.close()


def _get_embedder():
    """Use the project's embedder for vector search."""
    from mathir_mcp_server import get_embedder
    return get_embedder()


def _embed(memory: VecMemory, embedder, text: str) -> np.ndarray:
    from mathir_daemon import _embedding_to_numpy
    return _embedding_to_numpy(embedder.encode(text))


def _save(memory: VecMemory, embedder, content: str, label: str, priority: int = 5):
    emb = _embed(memory, embedder, content)
    import uuid
    mid = f"bench_{uuid.uuid4().hex[:8]}"
    memory.store(mid, emb, {
        "agent": "ai-bench",
        "block_type": "episodic",
        "label": label,
        "priority": priority,
        "content": content,
    })
    return mid


def _recall(memory: VecMemory, embedder, query: str, k: int = 5) -> List[Dict]:
    emb = _embed(memory, embedder, query)
    return memory.search(query_embedding=emb, k=k)


def run(experiences: int, questions: int, dim: int, seed: int, out: Path,
        duration_min: int):
    random.seed(seed)
    np.random.seed(seed)
    print(f"\n=== AI Cognitive Benchmark ===")
    print(f"    experiences={experiences}, questions={questions}, "
          f"duration_target={duration_min}min\n")

    # Health check
    llm = LLMClient()
    h = llm.health()
    # Print the actual model being used
    model_in_use = (os.environ.get("MATHIR_API_MODEL") or
                    (os.environ.get("MATHIR_OLLAMA_MODEL", "qwen3.5:2b")
                     if llm.backend == "ollama" else "default"))
    print(f"LLM backend={h['backend']} model={model_in_use} ok={h['ok']} error={h['error']}")
    if not h["ok"]:
        raise LLMUnavailable(f"LLM not reachable: {h['error']}")

    # Setup
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="mathir_aibench_"))
    db = tmp / "ai_bench.db"
    memory = VecMemory(db, dim)

    # Defer embedder loading until inside try/except (it may fail)
    try:
        embedder = _get_embedder()
    except Exception as e:
        print(f"  ! Could not load project embedder: {e}")
        print(f"  ! Falling back to a deterministic hash-based embedder.")
        embedder = _HashEmbedder(dim)

    topics = random.sample(PROJECT_TOPICS, min(experiences, len(PROJECT_TOPICS)))
    if experiences > len(PROJECT_TOPICS):
        topics += random.choices(PROJECT_TOPICS, k=experiences - len(PROJECT_TOPICS))

    deadline = time.time() + duration_min * 60
    results = {
        "config": {
            "experiences": experiences, "questions": questions,
            "dim": dim, "seed": seed, "duration_min": duration_min,
            "llm_backend": h["backend"],
        },
        "phase_A": {},
        "phase_B_baseline": {},
        "phase_C_maintenance": {},
        "phase_D_after": {},
        "comparison": {},
    }

    # =================================================================
    # PHASE A — Generation: LLM creates experiences, save each to memory
    # =================================================================
    print(f"\n[A] Generating {experiences} experiences...")
    gen_t0 = time.perf_counter()
    experiences_data = []
    for i, topic in enumerate(topics):
        if time.time() > deadline:
            print(f"  ! Time budget hit at {i}/{experiences}")
            break
        exp = generate_experience(llm, topic, seed_n=i)
        # Also store a duplicate for consolidation testing
        _save(memory, embedder, exp["content"], exp["label"] + "-v1")
        _save(memory, embedder, exp["content"], exp["label"] + "-v2")  # intentional duplicate
        experiences_data.append(exp)
        if (i + 1) % 5 == 0:
            print(f"  ... {i+1}/{experiences} done")
    gen_wall = time.perf_counter() - gen_t0
    results["phase_A"] = {
        "experiences_generated": len(experiences_data),
        "memories_stored": len(experiences_data) * 2,  # v1 + v2
        "wall_s": round(gen_wall, 2),
    }
    print(f"  -> {results['phase_A']['memories_stored']} memories stored in {gen_wall:.1f}s")

    # =================================================================
    # PHASE B — Baseline: ask questions, measure recall quality
    # =================================================================
    print(f"\n[B] Baseline: answering {questions} questions...")
    q_topics = random.sample(topics, min(questions, len(topics)))
    if questions > len(topics):
        q_topics += random.choices(topics, k=questions - len(topics))
    base_metrics = []
    base_answers = []
    for i, qt in enumerate(q_topics):
        if time.time() > deadline:
            print(f"  ! Time budget hit at {i}/{questions}")
            break
        # Blind question: no topic in it — only the right snippet can answer
        question = random.choice(QUESTION_TEMPLATES)  # no .format — templates are blind
        gt_content = next((e["content"] for e in experiences_data if e["topic"] == qt), "")
        snippets = _recall(memory, embedder, question, k=5)
        ans, n_used = answer_question(llm, question, snippets)
        m = recall_metrics(snippets, qt, gt_content)
        base_metrics.append(m)
        base_answers.append({"q": question, "a": ans, "metrics": m, "topic": qt})

    results["phase_B_baseline"] = {
        "questions_answered": len(base_metrics),
        "wall_s": round(time.perf_counter() - gen_t0 - 0, 2),
        "metrics_summary": _summarize(base_metrics),
        "answers": base_answers,
    }
    print(f"  -> recall@5={results['phase_B_baseline']['metrics_summary']['recall_at_5_mean']:.3f}, "
          f"has_answer={results['phase_B_baseline']['metrics_summary']['has_answer_rate']:.0%}")

    # =================================================================
    # PHASE C — Aging + maintenance
    # =================================================================
    print(f"\n[C] Maintenance: aging 30d, then decay/promote/consolidate/build_links...")
    _force_age(memory, days=30)
    print("  aged 30 days")

    t0 = time.perf_counter()
    decay_res = memory.decay_all(threshold_days=30, archive_floor=0.05)
    decay_wall = time.perf_counter() - t0
    print(f"  decay:     {decay_res.get('decayed')} decayed, "
          f"{decay_res.get('archived')} archived in {decay_wall:.2f}s")

    t0 = time.perf_counter()
    promoted = memory.auto_promote_all()
    promote_wall = time.perf_counter() - t0
    print(f"  promote:   {len(promoted)} promoted in {promote_wall:.2f}s")

    t0 = time.perf_counter()
    cons_res = memory.consolidate_all(threshold=0.95, dry_run=False)
    cons_wall = time.perf_counter() - t0
    print(f"  consolidate: {cons_res.get('merged')} merged in {cons_wall:.2f}s")

    t0 = time.perf_counter()
    link_res = memory.build_links_all(threshold=0.7, limit=10000)
    link_wall = time.perf_counter() - t0
    print(f"  build_links: {link_res.get('links_created')} links in {link_wall:.2f}s")

    results["phase_C_maintenance"] = {
        "decay": {"result": decay_res, "wall_s": round(decay_wall, 2)},
        "promote": {"promoted": len(promoted), "wall_s": round(promote_wall, 2)},
        "consolidate": {"result": cons_res, "wall_s": round(cons_wall, 2)},
        "build_links": {"result": link_res, "wall_s": round(link_wall, 2)},
    }

    # =================================================================
    # PHASE D — Re-test with same questions
    # =================================================================
    print(f"\n[D] Re-test: same {questions} questions after maintenance...")
    after_metrics = []
    after_answers = []
    for i, qt in enumerate(q_topics):
        if time.time() > deadline:
            print(f"  ! Time budget hit at {i}/{questions}")
            break
        question = random.choice(QUESTION_TEMPLATES)  # blind — no topic
        gt_content = next((e["content"] for e in experiences_data if e["topic"] == qt), "")
        snippets = _recall(memory, embedder, question, k=5)
        ans, n_used = answer_question(llm, question, snippets)
        m = recall_metrics(snippets, qt, gt_content)
        after_metrics.append(m)
        after_answers.append({"q": question, "a": ans, "metrics": m, "topic": qt})

    results["phase_D_after"] = {
        "questions_answered": len(after_metrics),
        "wall_s": round(time.perf_counter() - gen_t0 - 0, 2),
        "metrics_summary": _summarize(after_metrics),
        "answers": after_answers,
    }
    print(f"  -> recall@5={results['phase_D_after']['metrics_summary']['recall_at_5_mean']:.3f}, "
          f"has_answer={results['phase_D_after']['metrics_summary']['has_answer_rate']:.0%}")

    # =================================================================
    # COMPARISON
    # =================================================================
    b = results["phase_B_baseline"]["metrics_summary"]
    a = results["phase_D_after"]["metrics_summary"]
    results["comparison"] = {
        "recall_at_5": {"before": b["recall_at_5_mean"], "after": a["recall_at_5_mean"],
                        "delta": round(a["recall_at_5_mean"] - b["recall_at_5_mean"], 4)},
        "precision_at_5": {"before": b["precision_at_5_mean"], "after": a["precision_at_5_mean"],
                           "delta": round(a["precision_at_5_mean"] - b["precision_at_5_mean"], 4)},
        "mrr": {"before": b["mrr_mean"], "after": a["mrr_mean"],
                "delta": round(a["mrr_mean"] - b["mrr_mean"], 4)},
        "has_answer_rate": {"before": b["has_answer_rate"], "after": a["has_answer_rate"],
                            "delta": round(a["has_answer_rate"] - b["has_answer_rate"], 4)},
    }
    print(f"\n=== COMPARISON ===")
    print(f"  recall@5:    {b['recall_at_5_mean']:.3f} -> {a['recall_at_5_mean']:.3f} "
          f"({results['comparison']['recall_at_5']['delta']:+.3f})")
    print(f"  precision@5: {b['precision_at_5_mean']:.3f} -> {a['precision_at_5_mean']:.3f} "
          f"({results['comparison']['precision_at_5']['delta']:+.3f})")
    print(f"  has_answer:  {b['has_answer_rate']:.0%} -> {a['has_answer_rate']:.0%} "
          f"({results['comparison']['has_answer_rate']['delta']:+.0%})")

    # Save
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out}")

    memory.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return results


def _summarize(metrics: List[Dict]) -> Dict:
    if not metrics:
        return {}
    return {
        "recall_at_5_mean": round(np.mean([m["recall_at_5"] for m in metrics]), 4),
        "precision_at_5_mean": round(np.mean([m["precision_at_5"] for m in metrics]), 4),
        "mrr_mean": round(np.mean([m["mrr"] for m in metrics]), 4),
        "has_answer_rate": round(np.mean([m["has_answer"] for m in metrics]), 4),
        "top1_topic_match_rate": round(np.mean([m.get("top1_topic_match", False) for m in metrics]), 4),
    }


class _HashEmbedder:
    """Deterministic fallback embedder if the project embedder can't load."""
    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, text: str):
        v = np.zeros(self.dim, dtype=np.float32)
        for i, tok in enumerate(_tokenize(text)):
            h = hash(tok) % self.dim
            v[h] += 1.0 / (1 + i * 0.01)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def main():
    p = argparse.ArgumentParser(description="MATHIR AI cognitive benchmark")
    p.add_argument("--experiences", type=int, default=50)
    p.add_argument("--questions", type=int, default=20)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=int, default=20,
                   help="Duration budget in minutes (default 20)")
    p.add_argument("--out", type=Path, default=Path("results_ai.json"))
    p.add_argument("--provider", choices=["auto", "ollama", "openrouter", "api"],
                   default="auto",
                   help="LLM provider (default: auto — picks from env)")
    p.add_argument("--model", help="Override the model name (e.g. openai/gpt-oss-120b:free)")
    args = p.parse_args()
    # If --provider or --model given, push them to env BEFORE LLMClient init
    if args.provider != "auto":
        os.environ["MATHIR_LLM_BACKEND"] = args.provider
    if args.model:
        os.environ["MATHIR_API_MODEL"] = args.model
    run(args.experiences, args.questions, args.dim, args.seed, args.out, args.duration)


if __name__ == "__main__":
    main()
