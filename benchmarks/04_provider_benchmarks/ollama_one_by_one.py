#!/usr/bin/env python3
"""
Test Ollama models ONE BY ONE - showing each model's result clearly.
"""
import json, time, urllib.request, urllib.error
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import warnings; warnings.filterwarnings('ignore')

# Auto-load .env from benchmarks/ root via shared helper
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import _env  # noqa: F401
except ImportError:
    pass

import torch

OLLAMA = os.environ.get("MATHIR_OLLAMA_URL", "http://localhost:11434")


def ollama_generate(model, prompt, timeout=60):
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
            return {
                "success": True,
                "duration_s": round(time.time() - start, 2),
                "content": result.get("response", "") if False else json.loads(resp.read()).get("response", "")
            }
    except urllib.error.HTTPError as e:
        return {"success": False, "duration_s": round(time.time() - start, 2), "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"success": False, "duration_s": round(time.time() - start, 2), "error": str(e)[:80]}


def get_embedding(text):
    from sentence_transformers import SentenceTransformer
    return torch.from_numpy(SentenceTransformer("all-MiniLM-L6-v2").encode([text])).float()


def test_model(model_name, question, db_name):
    """Test a single model: ask question, retrieve from MATHIR."""
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    # Fresh DB for each test
    if os.path.exists(db_name):
        os.remove(db_name)
    from mathir_dropin import MATHIRMemory
    mathir = MATHIRMemory(embedding_dim=384, db_path=db_name)

    # Store 5 memories
    memories = [
        {"text": "Python closures are functions that capture their lexical scope", "concept": "python-closures"},
        {"text": "Rust ownership system prevents data races at compile time", "concept": "rust-ownership"},
        {"text": "TypeScript adds optional static typing to JavaScript", "concept": "typescript"},
        {"text": "React uses a virtual DOM to optimize rendering performance", "concept": "react"},
        {"text": "Docker containers package applications with dependencies", "concept": "docker"},
    ]
    for mem in memories:
        emb = get_embedding(mem["text"])
        mathir.store(embedding=emb, metadata=mem, provider="sentence-transformers")
    print(f"  Stored 5 memories")

    # Ask the question
    print(f"  Question: '{question}'")
    r = ollama_generate(model_name, question)
    if r["success"]:
        ans = r["content"][:200].encode("ascii", "replace").decode("ascii")
        print(f"  Answer ({r['duration_s']}s): {ans}...")
    else:
        print(f"  Answer: FAIL ({r.get('error', '?')})")
        return

    # Retrieve from MATHIR
    results = mathir.universal_recall(query=question, k=2)
    if results:
        print(f"  MATHIR retrieved:")
        for res in results:
            concept = res.get("metadata", {}).get("concept", "?")
            text = res.get("metadata", {}).get("text", "")[:70]
            print(f"    -> {concept}: '{text}...'")
    else:
        print(f"  MATHIR retrieved: (nothing)")

    return {"model": model_name, "answer_duration": r["duration_s"], "answer": r["content"][:200], "retrieved": [res.get("metadata", {}).get("concept", "?") for res in results] if results else []}


# Test each model individually
models_to_test = [
    ("qwen3:0.6b", "What is a Python closure? Explain briefly.", "test_qwen3.db"),
    ("lfm2.5-thinking:1.2b", "Tell me about Rust memory management", "test_lfm.db"),
    ("granite4:350m", "What is TypeScript used for?", "test_granite.db"),
    ("qwen3.5:2b", "Explain React's virtual DOM", "test_qwen35.db"),
]

all_results = []
for model, question, db in models_to_test:
    result = test_model(model, question, db)
    if result:
        all_results.append(result)
    time.sleep(2)  # Cooldown between models

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY - 1 model at a time")
print("=" * 70)
for r in all_results:
    found = r["retrieved"][0] if r.get("retrieved") else "NOTHING"
    print(f"  {r['model']:<28} -> {found}")
print("=" * 70)

# Save
with open("os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ollama_one_by_one.json"), "w") as f:
    json.dump(all_results, f, indent=2)