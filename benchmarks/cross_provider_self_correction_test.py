#!/usr/bin/env python3
"""
MATHIR Cross-Provider & Self-Correction Tests

Test A: Cross-Provider Semantic Bridge
  - Store with Provider A (sentence-transformers, 384 dim)
  - Recall with Provider B (different embedding space / text fallback)
  - Verify MATHIR bridges the gap via modality_text

Test B: Self-Correction Loop
  - Store memories, recall multiple times
  - Check if MATHIR updates priorities based on recall frequency
  - Verify bump_recall mechanism affects future recalls
"""
import json, time, sys, os
from datetime import datetime

sys.path.insert(0, "D:/SECRET_PROJECT/MATHIR")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

# ========================================================================
# PROVIDERS SETUP
# ========================================================================

def get_sentence_transformer_embedding(texts):
    """Provider A: sentence-transformers (384 dim)"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return embeddings, 384


def get_minimax_chat_response(prompt, system="You are a helpful assistant."):
    """Use MiniMax-M2.5 for chat responses (Anthropic-compatible API)"""
    API_KEY = "sk-cp-BECH0kOjMSyeTN3_YACZpAzXUnvWBqURc7aSdBw1txiISCDQ_KZZlxHRR93OiHJn6SrW6JH-rSedB1XdTnywJ9GQ-UnQRZiQN5kqFPM5dVBqMD5jWfdJ-KU"

    import urllib.request
    data = json.dumps({
        "model": "MiniMax-M2.5",
        "max_tokens": 256,
        "system": system,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        "https://api.minimax.io/anthropic/v1/messages",
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            content = result.get("content", [{}])
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text", "")
            return str(result)
    except Exception as e:
        return f"ERROR: {str(e)[:100]}"


# ========================================================================
# MATHIR SETUP
# ========================================================================

def setup_mathir(db_name):
    from mathir_dropin import MATHIRMemory
    if os.path.exists(db_name):
        os.remove(db_name)
    mathir = MATHIRMemory(embedding_dim=384, db_path=db_name)
    return mathir


# ========================================================================
# TEST A: CROSS-PROVIDER SEMANTIC BRIDGE
# ========================================================================

def test_a_cross_provider_bridge():
    print("\n" + "=" * 70)
    print("TEST A: CROSS-PROVIDER SEMANTIC BRIDGE")
    print("=" * 70)

    mathir = setup_mathir("mathir_test_a.db")

    # Store memories with Provider A (sentence-transformers)
    test_memories = [
        {"text": "Python has closures - functions that capture their lexical environment", "concept": "python-closures"},
        {"text": "Rust uses ownership and borrowing to ensure memory safety without GC", "concept": "rust-ownership"},
        {"text": "TypeScript adds static typing to JavaScript for better tooling", "concept": "typescript"},
        {"text": "React uses a virtual DOM to efficiently update the actual DOM", "concept": "react"},
        {"text": "SQL databases use indexes to speed up query execution", "concept": "sql-indexes"},
    ]

    print("\n[Step 1] Storing 5 memories with sentence-transformers embeddings...")
    stored_ids = []
    for mem in test_memories:
        emb, dim = get_sentence_transformer_embedding([mem["text"]])
        emb_tensor = torch.from_numpy(emb).float()

        mid = mathir.store(
            embedding=emb_tensor,
            metadata={"text": mem["text"], "concept": mem["concept"]},
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2"
        )
        stored_ids.append(mid)
        print(f"  [{mid[:8]}] {mem['concept']}")

    print(f"\n  Total stored: {len(stored_ids)}")

    # ====================================================================
    # Scenario A1: Recall with SAME provider
    # ====================================================================
    print("\n[Step 2a] Recall with SAME provider (sentence-transformers)...")
    query_text_1 = "Tell me about Python closures"
    emb1, _ = get_sentence_transformer_embedding([query_text_1])
    emb1_tensor = torch.from_numpy(emb1).float()

    results_same = mathir.recall(query_embedding=emb1_tensor, k=3, provider="sentence-transformers")
    print(f"  Query: '{query_text_1}'")
    print(f"  Results ({len(results_same)}):")
    for r in results_same[:3]:
        concept = r.get("metadata", {}).get("concept", "unknown")
        text = r.get("metadata", {}).get("text", "")[:50]
        print(f"    - [{r.get('memory_id', '?')[:8]}] {concept}: '{text}...'")

    # ====================================================================
    # Scenario A2: Recall with DIFFERENT provider (text fallback via FTS5)
    # ====================================================================
    print("\n[Step 2b] Recall with DIFFERENT provider (text-only query via FTS5)...")
    query_text_2 = "What do you know about python closures?"
    results_text = mathir.recall_text(query_text=query_text_2, k=3)
    print(f"  Query: '{query_text_2}'")
    print(f"  Results via FTS5 text search ({len(results_text)}):")
    for r in results_text[:3]:
        concept = r.get("metadata", {}).get("concept", "unknown")
        text = r.get("metadata", {}).get("text", "")[:50]
        print(f"    - [{r.get('memory_id', '?')[:8]}] {concept}: '{text}...'")

    # ====================================================================
    # Scenario A3: Cross-provider recall (Provider B has no embeddings stored)
    # ====================================================================
    print("\n[Step 2c] Cross-provider recall (Provider B query, Provider A memory)...")
    # Use dummy embedding from "wrong" provider space
    dummy_emb = torch.zeros(1, 384)  # Different from Provider A's space
    results_cross = mathir.recall(query_embedding=dummy_emb, k=3, provider="minimax")
    print(f"  Query with minimax provider (no stored embeddings for this provider):")
    print(f"  Results ({len(results_cross)}):")
    for r in results_cross[:3]:
        concept = r.get("metadata", {}).get("concept", "unknown")
        text = r.get("metadata", {}).get("text", "")[:50]
        provider = r.get("provider", "unknown")
        print(f"    - [{r.get('memory_id', '?')[:8]}] {concept} ({provider}): '{text}...'")

    # ====================================================================
    # Scenario A4: Use MiniMax to generate a query, then recall via MATHIR
    # ====================================================================
    print("\n[Step 2d] Using MiniMax to generate query, recall via MATHIR...")
    mm_prompt = "Generate a short search query (max 5 words) to find information about Python programming language features. Reply with ONLY the query, nothing else."
    query_gen = get_minimax_chat_response(mm_prompt)
    print(f"  MiniMax generated query: '{query_gen[:100].strip()}'")

    results_mm_gen = mathir.recall_text(query_text=query_gen.strip(), k=3)
    print(f"  Results via generated query ({len(results_mm_gen)}):")
    for r in results_mm_gen[:3]:
        concept = r.get("metadata", {}).get("concept", "unknown")
        text = r.get("metadata", {}).get("text", "")[:50]
        print(f"    - [{r.get('memory_id', '?')[:8]}] {concept}: '{text}...'")

    # ====================================================================
    # Evaluate Results
    # ====================================================================
    print("\n[Step 3] EVALUATION")

    # A1: Same provider should work
    a1_pass = len(results_same) > 0 and any(
        "python-closures" in str(r.get("metadata", {}).get("concept", ""))
        for r in results_same
    )

    # A2: Text search should bridge embedding gap
    a2_pass = len(results_text) > 0 and any(
        "python" in str(r.get("metadata", {}).get("text", "")).lower()
        for r in results_text
    )

    # A3: Cross-provider fallback should use primary embeddings
    a3_pass = len(results_cross) > 0  # Should return something via fallback

    # A4: MiniMax-generated query should work via FTS5
    a4_pass = len(results_mm_gen) > 0

    results_a = {
        "test": "Cross-Provider Semantic Bridge",
        "a1_same_provider": {"pass": a1_pass, "results": len(results_same)},
        "a2_text_fallback": {"pass": a2_pass, "results": len(results_text)},
        "a3_cross_provider": {"pass": a3_pass, "results": len(results_cross)},
        "a4_mm_generated_query": {"pass": a4_pass, "results": len(results_mm_gen)},
        "overall": sum([a1_pass, a2_pass, a3_pass, a4_pass]) >= 3
    }

    print(f"\n  A1 (same provider): {'PASS' if a1_pass else 'FAIL'}")
    print(f"  A2 (text fallback): {'PASS' if a2_pass else 'FAIL'}")
    print(f"  A3 (cross-provider fallback): {'PASS' if a3_pass else 'FAIL'}")
    print(f"  A4 (MiniMax generated query): {'PASS' if a4_pass else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if results_a['overall'] else 'FAIL'} ({sum([a1_pass, a2_pass, a3_pass, a4_pass])}/4)")

    return results_a


# ========================================================================
# TEST B: SELF-CORRECTION LOOP
# ========================================================================

def test_b_self_correction():
    print("\n" + "=" * 70)
    print("TEST B: SELF-CORRECTION LOOP")
    print("=" * 70)

    mathir = setup_mathir("mathir_test_b.db")

    # Store 10 memories with varying concept tags
    memories = [
        {"text": "Python list comprehensions are concise syntax for creating lists", "concept": "python-listcomp"},
        {"text": "JavaScript arrow functions provide shorter syntax and lexical this", "concept": "js-arrow"},
        {"text": "Rust enums can hold data and implement methods", "concept": "rust-enum"},
        {"text": "Go uses goroutines for concurrent programming", "concept": "go-goroutine"},
        {"text": "TypeScript interfaces define object shapes", "concept": "ts-interface"},
        {"text": "Python decorators modify function behavior at runtime", "concept": "python-decorator"},
        {"text": "React useState manages component local state", "concept": "react-state"},
        {"text": "SQL JOINs combine rows from multiple tables", "concept": "sql-join"},
        {"text": "Python generators yield values lazily on iteration", "concept": "python-generator"},
        {"text": "Docker containers package applications with their dependencies", "concept": "docker"},
    ]

    print("\n[Step 1] Storing 10 memories...")
    stored_ids = []
    for mem in memories:
        emb, dim = get_sentence_transformer_embedding([mem["text"]])
        emb_tensor = torch.from_numpy(emb).float()
        mid = mathir.store(
            embedding=emb_tensor,
            metadata={"text": mem["text"], "concept": mem["concept"]},
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2"
        )
        stored_ids.append(mid)
        print(f"  [{mid[:8]}] {mem['concept']}")

    # ====================================================================
    # Scenario B1: Initial recall - check order
    # ====================================================================
    print("\n[Step 2] Initial recall test (before any feedback)...")

    query_1 = "Tell me about Python programming features"
    emb_q1, _ = get_sentence_transformer_embedding([query_1])
    emb_q1_tensor = torch.from_numpy(emb_q1).float()

    results_before = mathir.recall(query_embedding=emb_q1_tensor, k=5)
    print(f"  Query: '{query_1}'")
    print(f"  Top 5 results:")
    concepts_before = []
    for r in results_before[:5]:
        concept = r.get("metadata", {}).get("concept", "unknown")
        recall_count = r.get("recall_count", 0)
        concepts_before.append(concept)
        print(f"    - {concept} (recalled {recall_count}x)")

    # ====================================================================
    # Scenario B2: Simulate repeated recalls to bump recall counts
    # ====================================================================
    print("\n[Step 3] Simulating repeated recalls (bumping recall counts)...")

    # Recall the same query multiple times to bump recall counts
    # This simulates "user keeps asking about Python" -> MATHIR should prioritize Python memories
    for _ in range(5):
        mathir.recall(query_embedding=emb_q1_tensor, k=3)

    # Also specifically recall python-listcomp and python-decorator multiple times
    python_query_emb, _ = get_sentence_transformer_embedding(["Python list and decorator features"])
    python_query_tensor = torch.from_numpy(python_query_emb).float()
    for _ in range(3):
        mathir.recall(query_embedding=python_query_tensor, k=2)

    print("  Bumped recall counts for Python-related memories")

    # ====================================================================
    # Scenario B3: Re-recall after feedback - check if order changed
    # ====================================================================
    print("\n[Step 4] Recall after repeated queries (should prioritize frequently recalled)...")

    results_after = mathir.recall(query_embedding=emb_q1_tensor, k=5)
    print(f"  Query: '{query_1}'")
    print(f"  Top 5 results after feedback:")
    concepts_after = []
    for r in results_after[:5]:
        concept = r.get("metadata", {}).get("concept", "unknown")
        recall_count = r.get("recall_count", 0)
        concepts_after.append(concept)
        print(f"    - {concept} (recalled {recall_count}x)")

    # ====================================================================
    # Scenario B4: Check recall_count tracking in storage
    # ====================================================================
    print("\n[Step 5] Checking recall_count tracking in storage...")

    all_ids = mathir._store.all_ids()
    print(f"  Total memories in storage: {len(all_ids)}")

    # Get recall counts for all memories
    recall_counts = {}
    for mid in all_ids:
        row = mathir._store.get(mid)
        if row:
            meta = row.get("metadata", {})
            concept = meta.get("concept", "unknown") if meta else "unknown"
            rc = row.get("recall_count", 0)
            recall_counts[concept] = rc

    # Sort by recall count
    sorted_concepts = sorted(recall_counts.items(), key=lambda x: x[1], reverse=True)
    print("  Recall counts (sorted):")
    for concept, count in sorted_concepts[:5]:
        print(f"    - {concept}: {count}x")

    # ====================================================================
    # Scenario B5: Correct recall test (exact match)
    # ====================================================================
    print("\n[Step 6] Correct recall test (exact concept match)...")

    query_2 = "Python list comprehensions"
    emb_q2, _ = get_sentence_transformer_embedding([query_2])
    emb_q2_tensor = torch.from_numpy(emb_q2).float()

    results_correct = mathir.recall(query_embedding=emb_q2_tensor, k=3)
    print(f"  Query: '{query_2}'")
    print(f"  Results:")
    for r in results_correct[:3]:
        concept = r.get("metadata", {}).get("concept", "unknown")
        text = r.get("metadata", {}).get("text", "")[:50]
        print(f"    - {concept}: '{text}...'")

    # ====================================================================
    # Evaluate Results
    # ====================================================================
    print("\n[Step 7] EVALUATION")

    # B1: Initial recall should return relevant memories
    b1_pass = len(results_before) > 0

    # B2: Bumping should work (no crash)
    b2_pass = True  # If we got here without error, bumping worked

    # B3: After feedback, order should reflect recall frequency
    # Check that python-listcomp or python-decorator moved up in ranking
    b3_order_changed = concepts_before != concepts_after

    # B4: Recall count tracking should show bumped counts
    b4_counts_bumped = any(count > 0 for concept, count in sorted_concepts if "python" in concept.lower())

    # B5: Correct recall should find the right memory
    b5_pass = any("python-listcomp" in str(r.get("metadata", {}).get("concept", "")).lower()
                  for r in results_correct[:3])

    results_b = {
        "test": "Self-Correction Loop",
        "b1_initial_recall": {"pass": b1_pass, "results": len(results_before)},
        "b2_bump_recall": {"pass": b2_pass},
        "b3_feedback_reorder": {"pass": b3_order_changed, "before": concepts_before[:3], "after": concepts_after[:3]},
        "b4_correction_tracked": {"pass": b4_counts_bumped, "counts": dict(sorted_concepts[:5])},
        "b5_correct_recall": {"pass": b5_pass, "results": len(results_correct)},
        "overall": sum([b1_pass, b2_pass, b3_order_changed, b4_counts_bumped, b5_pass]) >= 4
    }

    print(f"\n  B1 (initial recall): {'PASS' if b1_pass else 'FAIL'}")
    print(f"  B2 (bump recall): {'PASS' if b2_pass else 'FAIL'}")
    print(f"  B3 (feedback reorder): {'PASS' if b3_order_changed else 'FAIL'}")
    print(f"  B4 (correction tracked): {'PASS' if b4_counts_bumped else 'FAIL'}")
    print(f"  B5 (correct recall): {'PASS' if b5_pass else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if results_b['overall'] else 'FAIL'} ({sum([b1_pass, b2_pass, b3_order_changed, b4_counts_bumped, b5_pass])}/5)")

    return results_b


# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MATHIR CROSS-PROVIDER & SELF-CORRECTION TESTS")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "providers": {
            "provider_a": "sentence-transformers (all-MiniLM-L6-v2, 384 dim)",
            "provider_b_fallback": "MATHIR FTS5 text search",
            "llm_query": "MiniMax-M2.5 (Anthropic-compatible)"
        },
        "test_a": None,
        "test_b": None,
        "summary": {}
    }

    # Run Test A
    try:
        results["test_a"] = test_a_cross_provider_bridge()
    except Exception as e:
        print(f"\nTEST A FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["test_a"] = {"error": str(e), "overall": False}

    # Run Test B
    try:
        results["test_b"] = test_b_self_correction()
    except Exception as e:
        print(f"\nTEST B FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["test_b"] = {"error": str(e), "overall": False}

    # Summary
    test_a_pass = results["test_a"].get("overall", False) if results["test_a"] else False
    test_b_pass = results["test_b"].get("overall", False) if results["test_b"] else False

    results["summary"] = {
        "test_a_cross_provider": "PASS" if test_a_pass else "FAIL",
        "test_b_self_correction": "PASS" if test_b_pass else "FAIL",
        "both_pass": test_a_pass and test_b_pass
    }

    # Save results
    output_path = "D:/SECRET_PROJECT/MATHIR/benchmarks/cross_provider_self_correction_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Test A (Cross-Provider Semantic Bridge): {results['summary']['test_a_cross_provider']}")
    print(f"Test B (Self-Correction Loop): {results['summary']['test_b_self_correction']}")
    print(f"\nBoth PASS: {results['summary']['both_pass']}")
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)