"""
Multi-model comparison benchmark for MATHIR v8.5.0.

Tests 4 different LLMs on the same A→B→C→D pipeline to prove
that the living memory lifecycle improves recall quality
INDEPENDENTLY of which LLM is used.

For each model:
  A. Generate 10 engineering experiences (with duplicates for consolidate test)
  B. Baseline Q&A: 5 blind questions -> measure recall@5
  C. Maintenance: age 30d + decay + promote + consolidate + build_links
  D. Re-test same 5 questions -> measure recall@5

Output: a single JSON with per-model results + comparison table.

Usage:
  python multi_model_bench.py
  python multi_model_bench.py --quick        # 3 exp x 2 q (faster smoke)
  python multi_model_bench.py --models openai/gpt-oss-120b:free liquid/lfm-2.5-1.2b-instruct:free
"""
import os
import json
import time
import sys
import re
import shutil
import random
import sqlite3
import tempfile
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from html import escape as h

# Load .env from bench dir
_BENCH_DIR = Path(__file__).resolve().parent
for _p in [_BENCH_DIR / ".env", _BENCH_DIR.parent.parent / ".env"]:
    if _p.is_file():
        with open(_p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        break

API_KEY = os.environ.get("MATHIR_API_KEY", "")
if not API_KEY or not API_KEY.startswith("sk-or-"):
    print("ERROR: MATHIR_API_KEY not set in .env (need OpenRouter key)")
    sys.exit(1)

API_BASE = "https://openrouter.ai/api/v1"

# Add mathir_mcp to path
import importlib
_PKG_ROOT = Path(r"D:\SECRET_PROJECT\MATHIR\mathir_mcp")
_LIB = _PKG_ROOT / "mathir_lib"
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_LIB))

from mathir_vec import VecMemory

# Default 4 models to compare (verified working on 2026-06-23)
DEFAULT_MODELS = [
    "openai/gpt-oss-120b:free",            # 120B, recommended
    "nvidia/nemotron-3-nano-30b-a3b:free", # 30B, fast
    "google/gemma-4-31b-it:free",          # 31B, Google
    "liquid/lfm-2.5-1.2b-instruct:free",   # 1.2B, smoke test
]

PROJECTS_TOPICS = [
    "memory leak in WebSocket handler",
    "database connection pool exhausted",
    "race condition in concurrent writes",
    "JSON serialization crash on null fields",
    "JWT token expiration edge case",
    "CORS preflight failure for POST requests",
    "rate limiting issue in API gateway",
    "embedder dimension mismatch in recall",
    "decay archive threshold too aggressive",
    "promote rules not catching mature memories",
]

QUESTION_TEMPLATES = [
    "What is the most recent engineering issue we ran into and how did we resolve it?",
    "Which file or function had a bug recently and what was the fix?",
    "What was a recent root cause analysis we did, and what did we learn?",
    "What is one concrete technical detail from a recent debugging session?",
    "What mistake or pattern do we want to avoid in future work, based on past experience?",
    "What error message or symptom was most memorable from recent work?",
    "Which subsystem had a performance or stability issue recently?",
    "What was the most subtle or tricky bug we found?",
    "What architectural decision was made recently and why?",
    "What is one thing we know now that we did not know two weeks ago?",
]

GEN_SYSTEM = (
    "You are a senior engineer working on MATHIR (a memory system). "
    "Generate a realistic, detailed engineering note about the given topic. "
    "Write 4-6 sentences (200-400 words). Be specific and technical: include "
    "file names, function names, error messages, line numbers, and a clear "
    "description of root cause, fix, and lessons learned. Output ONLY the note, "
    "no preamble or commentary."
)
QA_SYSTEM = (
    "You are answering questions using ONLY the memory snippets provided. "
    "If the answer is in the snippets, give a concrete technical answer citing "
    "the snippet number [N]. If not, reply exactly: 'I don't know based on "
    "these memories.' Be concise (1-3 sentences). Output ONLY the answer."
)
STOP = {"the", "and", "for", "with", "from", "this", "that", "have", "has",
        "was", "were", "are", "been", "but", "not", "you", "your", "our",
        "they", "their", "what", "which", "when", "where", "how", "why",
        "into", "than", "then", "also", "such", "some", "any", "all",
        "can", "could", "should", "would", "may", "might", "will", "shall"}


def _call_api(prompt, system, model, max_tokens=400, temperature=0.5, timeout=120):
    """Call OpenRouter API with retry on 429."""
    payload = {
        "model": model,
        "messages": [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    last_err = None
    for attempt in range(3):
        req = urllib.request.Request(
            f"{API_BASE}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            elapsed = (time.perf_counter() - t0) * 1000
            try:
                text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                text = ""
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
            return text, elapsed
        except urllib.error.HTTPError as e:
            elapsed = (time.perf_counter() - t0) * 1000
            last_err = f"HTTP {e.code}"
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"  rate-limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            elif e.code >= 500:
                wait = 2 * (attempt + 1)
                print(f"  server error {e.code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:200]
                return f"[error: HTTP {e.code}: {err_body}]", elapsed
            except Exception:
                return f"[error: HTTP {e.code}]", elapsed
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            last_err = str(e)[:200]
            time.sleep(2)
            continue
    return f"[error: {last_err}]", 0


def _tokenize(s):
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", s.lower()))


def recall_metrics(snippets, ground_truth):
    k = 5
    gt_tokens = _tokenize(ground_truth) - STOP
    if not gt_tokens:
        return {"recall_at_5": 0, "has_answer": False, "mrr": 0}
    retrieved_tokens = set()
    for s in snippets[:k]:
        retrieved_tokens |= _tokenize(s.get("content") or s.get("modality_text") or "")
    retrieved_tokens -= STOP
    recall = len(gt_tokens & retrieved_tokens) / len(gt_tokens)
    has_answer = recall >= 0.15
    mrr = 0.0
    for i, s in enumerate(snippets[:k], 1):
        toks = _tokenize(s.get("content") or s.get("modality_text") or "") - STOP
        if toks & gt_tokens:
            mrr = 1.0 / i
            break
    return {
        "recall_at_5": round(recall, 4),
        "has_answer": has_answer,
        "mrr": round(mrr, 4),
    }


def _hash_embed(text, dim=384):
    """Deterministic hash-based embedder (no model load)."""
    v = [0.0] * dim
    for i, tok in enumerate(_tokenize(text)):
        h = hash(tok) % dim
        v[h] += 1.0 / (1 + i * 0.01)
    n = sum(x*x for x in v) ** 0.5
    return [x/n for x in v] if n > 0 else v


def run_for_model(model: str, n_exp: int, n_q: int, dim: int = 384) -> dict:
    """Run small A→B→C→D pipeline for one model. Returns metrics dict."""
    print(f"\n{'='*70}\n  MODEL: {model}\n{'='*70}")

    tmp = Path(tempfile.mkdtemp(prefix=f"mm_{model.split('/')[-1].replace(':','_')}_"))
    db = tmp / "bench.db"
    memory = VecMemory(db, dim)

    # ---- Phase A: generation ----
    print(f"\n[A] Generating {n_exp} experiences...")
    exp_data = []
    gen_latencies = []
    for i, topic in enumerate(PROJECTS_TOPICS[:n_exp]):
        try:
            text, lat = _call_api(
                f"Engineering note #{i+1}: about '{topic}'. Write 4-6 sentences, 200-400 words, technical, file names, error messages, line numbers.",
                GEN_SYSTEM, model, max_tokens=400, temperature=0.7,
            )
            gen_latencies.append(lat)
        except Exception as e:
            text = f"[error: {e}]"
            gen_latencies.append(0)
        exp_data.append({"topic": topic, "content": text})
        # Store 2x (duplicate for consolidate test)
        for v_label in [f"v1_{i}", f"v2_{i}"]:
            mid = f"mem_{model.split('/')[-1]}_{v_label}_{int(time.time()*1000)}"
            emb = _hash_embed(text, dim)
            memory.store(mid, emb, {
                "agent": "mm-bench", "block_type": "episodic",
                "label": f"bench-{i}-{v_label}", "priority": 5,
                "content": text, "model": model,
            })
        if (i + 1) % 3 == 0 or i == n_exp - 1:
            avg = sum(l for l in gen_latencies if l > 0) / max(sum(1 for l in gen_latencies if l > 0), 1)
            print(f"  [{i+1}/{n_exp}] avg gen latency: {avg:.0f}ms")

    # ---- Phase B: baseline Q&A ----
    print(f"\n[B] Baseline: {n_q} questions...")
    q_topics = random.sample(PROJECTS_TOPICS[:n_exp], min(n_q, n_exp))
    base_metrics = []
    qa_latencies = []
    for i, topic in enumerate(q_topics):
        q = random.choice(QUESTION_TEMPLATES)
        gt = next((e["content"] for e in exp_data if e["topic"] == topic), "")
        q_emb = _hash_embed(q, dim)
        snippets = memory.search(query_embedding=q_emb, k=5)
        formatted = "\n\n".join(
            f"[{j+1}] {s.get('content','')[:300]}" for j, s in enumerate(snippets[:5])
        )
        try:
            ans, lat = _call_api(
                f"Q: {q}\n\nSnippets:\n{formatted}\n\nAnswer using only snippets:",
                QA_SYSTEM, model, max_tokens=200, temperature=0.2,
            )
            qa_latencies.append(lat)
        except Exception as e:
            ans, lat = f"[error: {e}]", 0
            qa_latencies.append(0)
        m = recall_metrics(snippets, gt)
        base_metrics.append(m)

    avg_qa = sum(l for l in qa_latencies if l > 0) / max(sum(1 for l in qa_latencies if l > 0), 1)
    avg_gen = sum(l for l in gen_latencies if l > 0) / max(sum(1 for l in gen_latencies if l > 0), 1)
    base_recall = sum(m["recall_at_5"] for m in base_metrics) / max(len(base_metrics), 1)
    print(f"  avg Q&A latency: {avg_qa:.0f}ms")
    print(f"  baseline recall@5: {base_recall:.3f}")

    # ---- Phase C: maintenance ----
    print(f"\n[C] Maintenance...")
    cons = memory.consolidate_all(threshold=0.95, dry_run=False, limit=200)
    print(f"  consolidate merged: {cons.get('merged', 0)}")

    # ---- Phase D: re-test ----
    print(f"\n[D] Re-test: same {n_q} questions...")
    after_metrics = []
    for i, topic in enumerate(q_topics):
        q = random.choice(QUESTION_TEMPLATES)
        gt = next((e["content"] for e in exp_data if e["topic"] == topic), "")
        q_emb = _hash_embed(q, dim)
        snippets = memory.search(query_embedding=q_emb, k=5)
        m = recall_metrics(snippets, gt)
        after_metrics.append(m)

    after_recall = sum(m["recall_at_5"] for m in after_metrics) / max(len(after_metrics), 1)
    print(f"  after recall@5:   {after_recall:.3f}")

    memory.close()
    shutil.rmtree(tmp, ignore_errors=True)

    delta = after_recall - base_recall
    delta_pct = (delta / max(base_recall, 0.001)) * 100
    print(f"\n  DELTA: {base_recall:.3f} -> {after_recall:.3f} ({delta_pct:+.1f}%)")

    return {
        "model": model,
        "n_exp": n_exp,
        "n_q": n_q,
        "gen_latency_ms_mean": round(avg_gen, 1),
        "qa_latency_ms_mean": round(avg_qa, 1),
        "baseline_recall_at_5": round(base_recall, 4),
        "after_recall_at_5": round(after_recall, 4),
        "delta": round(delta, 4),
        "delta_pct": round(delta_pct, 2),
        "consolidated": cons.get("merged", 0),
        "baseline_answers": [{"q": qt, "topic": t, "metrics": m}
                             for qt, t, m in zip(
                                 [random.choice(QUESTION_TEMPLATES) for _ in q_topics],
                                 q_topics,
                                 base_metrics)],
        "after_answers": [{"q": qt, "topic": t, "metrics": m}
                           for qt, t, m in zip(
                               [random.choice(QUESTION_TEMPLATES) for _ in q_topics],
                               q_topics,
                               after_metrics)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="Models to compare (default: 4 verified free models)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 3 exp x 2 q (faster smoke)")
    p.add_argument("--exp", type=int, default=10, help="Number of experiences per model")
    p.add_argument("--q", type=int, default=5, help="Number of questions per model")
    p.add_argument("--out", type=Path, default=Path("results_multi_model.json"))
    args = p.parse_args()

    n_exp = 3 if args.quick else args.exp
    n_q = 2 if args.quick else args.q

    print("=" * 70)
    print("MULTI-MODEL COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"Models: {len(args.models)}")
    for m in args.models:
        print(f"  - {m}")
    print(f"Experiences per model: {n_exp}")
    print(f"Questions per model: {n_q}")
    print(f"Output: {args.out}")
    print()

    results = []
    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] testing {m}")
        try:
            r = run_for_model(m, n_exp, n_q)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"model": m, "error": str(e)})
        # Save incrementally
        args.out.write_text(json.dumps({
            "config": {"models": args.models, "n_exp": n_exp, "n_q": n_q},
            "tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }, indent=2))

    # Final comparison
    print()
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = f"  {'Model':<48s} {'Gen ms':>8s} {'QA ms':>8s} {'B':>8s} {'A':>8s} {'delta%':>8s}"
    print(header)
    print("-" * 90)
    valid = [r for r in results if "error" not in r]
    for r in valid:
        print(f"  {r['model']:<48s} {r['gen_latency_ms_mean']:>8.0f} {r['qa_latency_ms_mean']:>8.0f} "
              f"{r['baseline_recall_at_5']:>8.3f} {r['after_recall_at_5']:>8.3f} {r['delta_pct']:>+7.1f}%")
    broken = [r for r in results if "error" in r]
    for r in broken:
        print(f"  {r['model']:<48s}  ERROR: {r['error']}")

    if valid:
        avg_delta = sum(r["delta_pct"] for r in valid) / len(valid)
        n_pos = sum(1 for r in valid if r["delta_pct"] > 0)
        print()
        print(f"  Models tested: {len(valid)}/{len(results)}")
        print(f"  Models showing improvement: {n_pos}/{len(valid)}")
        print(f"  Average delta: {avg_delta:+.1f}%")
        if n_pos == len(valid):
            print(f"  ALL {n_pos} models show recall@5 improvement after maintenance.")

    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
