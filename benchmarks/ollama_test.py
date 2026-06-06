#!/usr/bin/env python3
"""
MATHIR + Ollama Test - Using user's local models
Uses /api/generate (not /api/chat) since chat endpoint timeouts.
Handles Unicode in responses.
"""
import json, time, urllib.request, urllib.error
from datetime import datetime
import sys, os
sys.path.insert(0, "D:/SECRET_PROJECT/MATHIR")
import warnings; warnings.filterwarnings('ignore')

import torch

OLLAMA = "http://localhost:11434"
USER_MODELS = ["qwen3:0.6b", "lfm2.5-thinking:1.2b", "granite4:350m", "qwen3.5:2b"]


def ollama_generate(model, prompt, timeout=90):
    """Call Ollama generate API (works better than chat for these models)."""
    data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            duration = time.time() - start
            result = json.loads(resp.read())
            return {
                "success": True,
                "duration_s": round(duration, 2),
                "content": result.get("response", "").encode("ascii", "replace").decode("ascii"),
                "raw": result.get("response", ""),
            }
    except urllib.error.HTTPError as e:
        return {"success": False, "duration_s": time.time() - start, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"success": False, "duration_s": time.time() - start, "error": str(e)[:80]}


def get_embedding_st(text):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return torch.from_numpy(model.encode([text])).float()


print("=" * 70)
print("MATHIR + OLLAMA TEST (using your local models)")
print("=" * 70)

# Test connectivity
print("\n[1] Testing Ollama models...")
working_models = []
for model in USER_MODELS:
    r = ollama_generate(model, "Say OK", timeout=60)
    if r["success"]:
        working_models.append(model)
        print(f"  [OK]   {model:<28} {r['duration_s']}s")
    else:
        print(f"  [FAIL] {model:<28} {r.get('error', '?')[:30]}")

print(f"\nWorking: {len(working_models)}/{len(USER_MODELS)}")
if not working_models:
    print("Aborting.")
    sys.exit(0)

# Setup MATHIR
db = "mathir_ollama_test.db"
if os.path.exists(db):
    os.remove(db)
from mathir_dropin import MATHIRMemory
mathir = MATHIRMemory(embedding_dim=384, db_path=db)

# Store test memories
print("\n[2] Storing 5 programming concept memories...")
memories = [
    {"text": "Python closures are functions that capture their lexical scope", "concept": "python-closures"},
    {"text": "Rust ownership system prevents data races at compile time", "concept": "rust-ownership"},
    {"text": "TypeScript adds optional static typing to JavaScript", "concept": "typescript"},
    {"text": "React uses a virtual DOM to optimize rendering performance", "concept": "react"},
    {"text": "Docker containers package applications with dependencies", "concept": "docker"},
]
for mem in memories:
    emb = get_embedding_st(mem["text"])
    mid = mathir.store(embedding=emb, metadata=mem, provider="sentence-transformers")
    print(f"  [{mid[:8]}] {mem['concept']}")

# ========================================================================
# TEST 1: Each Ollama model asks a question
# ========================================================================
print("\n[3] Each Ollama model asks a different question...")
questions = {
    "qwen3:0.6b": "What is a Python closure?",
    "lfm2.5-thinking:1.2b": "Tell me about Rust memory management",
    "granite4:350m": "What is TypeScript used for?",
    "qwen3.5:2b": "Explain React's virtual DOM",
}

retrieval_results = []
for model in working_models:
    if model not in questions:
        continue
    q = questions[model]
    print(f"\n--- {model} ---")
    print(f"  Question: '{q}'")

    r = ollama_generate(model, q, timeout=60)
    if r["success"]:
        ans = r["content"][:120].replace("\n", " ")
        print(f"  Answer: {ans}...")

    # Use the question to retrieve from MATHIR
    results = mathir.universal_recall(query=q, k=2)
    if results:
        for res in results[:1]:
            concept = res.get("metadata", {}).get("concept", "?")
            text = res.get("metadata", {}).get("text", "")[:60]
            print(f"  MATHIR: {concept} - '{text}...'")
            retrieval_results.append({"model": model, "question": q, "found": concept})
    else:
        print("  MATHIR: (no results)")
        retrieval_results.append({"model": model, "question": q, "found": None})
    time.sleep(1)

# ========================================================================
# TEST 2: Cross-language via Ollama
# ========================================================================
print("\n[4] Cross-language: Model responds in French, MATHIR finds English memory")
if "qwen3:0.6b" in working_models:
    r = ollama_generate("qwen3:0.6b",
        "En francais: Que savez-vous sur les fermetures Python (closures)?",
        timeout=60)
    if r["success"]:
        fr = r["content"][:150].replace("\n", " ")
        print(f"  qwen3 FR: {fr}...")

    # Use French query directly
    fr_q = "clotures python"
    results = mathir.universal_recall(query=fr_q, k=2)
    print(f"  Query '{fr_q}' (FR->EN bridge):")
    for res in results[:1]:
        concept = res.get("metadata", {}).get("concept", "?")
        print(f"    -> {concept}")

# ========================================================================
# TEST 3: Model A stores, Model B retrieves
# ========================================================================
print("\n[5] Cross-model: Model A explains, Model B queries")
if len(working_models) >= 2:
    model_a, model_b = working_models[0], working_models[1]

    r_a = ollama_generate(model_a, "Explain Rust ownership in one short sentence.", timeout=60)
    if r_a["success"]:
        a_text = r_a["content"][:200]
        print(f"  {model_a} explains: {a_text[:100]}...")

        emb = get_embedding_st(a_text)
        mid = mathir.store(embedding=emb, metadata={
            "text": a_text, "concept": "rust-explained", "generated_by": model_a
        }, provider=f"ollama-{model_a}")
        print(f"  Stored [{mid[:8]}]")

        # Now Model B asks
        r_b = ollama_generate(model_b, "What is Rust ownership?", timeout=60)
        if r_b["success"]:
            print(f"  {model_b} asks: {r_b['content'][:80]}...")

        results = mathir.universal_recall(query="Rust ownership", k=3)
        print(f"  MATHIR retrieved:")
        for res in results:
            gen_by = res.get("metadata", {}).get("generated_by", "seed")
            concept = res.get("metadata", {}).get("concept", "?")
            text = res.get("metadata", {}).get("text", "")[:60]
            print(f"    - {concept} (by {gen_by}) - '{text}...'")

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Working Ollama models: {len(working_models)}/{len(USER_MODELS)}")
print(f"  {', '.join(working_models)}")
print(f"\nRetrieval success: {sum(1 for r in retrieval_results if r.get('found'))}/{len(retrieval_results)}")

print("\nResults saved to: D:/SECRET_PROJECT/MATHIR/benchmarks/ollama_test_results.json")

# Save
results = {
    "timestamp": datetime.now().isoformat(),
    "user_models": USER_MODELS,
    "working_models": working_models,
    "retrieval_results": retrieval_results,
    "retrieval_success_rate": f"{sum(1 for r in retrieval_results if r.get('found'))}/{len(retrieval_results)}",
}
with open("D:/SECRET_PROJECT/MATHIR/benchmarks/ollama_test_results.json", "w") as f:
    json.dump(results, f, indent=2)