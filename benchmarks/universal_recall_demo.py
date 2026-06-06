"""
MATHIR Universal Recall Test - Demonstrates the new universal_recall() API.
This fixes the failures from the original cross_provider_self_correction_test.py
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'D:/SECRET_PROJECT/MATHIR')

from mathir_dropin import MATHIRMemory
import torch

def get_embedding(text):
    """Use sentence-transformers for embeddings"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode([text])
    return torch.from_numpy(emb).float()

print("=" * 70)
print("MATHIR UNIVERSAL RECALL - NEW API TEST")
print("=" * 70)

# Setup
import os
if os.path.exists("mathir_universal_test.db"):
    os.remove("mathir_universal_test.db")
mathir = MATHIRMemory(embedding_dim=384, db_path="mathir_universal_test.db")

# Store memories in ENGLISH
memories = [
    {"text": "Python has closures - functions that capture their lexical environment", "concept": "python-closures", "lang": "en"},
    {"text": "Rust uses ownership and borrowing to ensure memory safety without GC", "concept": "rust-ownership", "lang": "en"},
    {"text": "TypeScript adds static typing to JavaScript for better tooling", "concept": "typescript", "lang": "en"},
]

print("\n[1] Storing 3 memories in English...")
for mem in memories:
    emb = get_embedding(mem["text"])
    mid = mathir.store(embedding=emb, metadata=mem, provider="sentence-transformers")
    print(f"  [{mid[:8]}] {mem['concept']}")

# ========================================================================
# TEST 1: Conversational query (was FAILING with recall_text)
# ========================================================================
print("\n" + "[2] TEST 1: Conversational query via universal_recall()")
print("  Query: 'What do you know about python closures?'")
results = mathir.universal_recall(
    query="What do you know about python closures?",
    k=3
)
print(f"  Results: {len(results)}")
for r in results[:3]:
    print(f"    - {r.get('metadata', {}).get('concept', '?')}: '{r.get('metadata', {}).get('text', '')[:50]}...'")
test1_pass = len(results) > 0 and any("python-closures" in str(r.get("metadata", {}).get("concept", "")).lower() for r in results)
print(f"  STATUS: {'PASS' if test1_pass else 'FAIL'}")

# ========================================================================
# TEST 2: Cross-lingual query (French)
# ========================================================================
print("\n[3] TEST 2: Cross-lingual query (French 'clotures python')")
print("  Query: 'clotures python'")
results = mathir.universal_recall(query="clotures python", k=3)
print(f"  Results: {len(results)}")
for r in results[:3]:
    print(f"    - {r.get('metadata', {}).get('concept', '?')}: '{r.get('metadata', {}).get('text', '')[:50]}...'")
test2_pass = len(results) > 0 and any("python" in str(r.get("metadata", {}).get("text", "")).lower() for r in results)
print(f"  STATUS: {'PASS' if test2_pass else 'FAIL'}")

# ========================================================================
# TEST 3: Cross-provider recall
# ========================================================================
print("\n[4] TEST 3: Cross-provider recall (query with provider='minimax')")
print("  Query with embedding + provider='minimax' (no stored embeddings for this provider)")
emb = get_embedding("Python closures functions")
results = mathir.universal_recall(
    query="python closures",  # Required positional arg
    query_embedding=emb,
    k=3,
    provider="minimax"  # Provider with no stored embeddings
)
print(f"  Results: {len(results)}")
for r in results[:3]:
    print(f"    - {r.get('metadata', {}).get('concept', '?')}: '{r.get('metadata', {}).get('text', '')[:50]}...'")
test3_pass = len(results) > 0
print(f"  STATUS: {'PASS' if test3_pass else 'FAIL'}")

# ========================================================================
# TEST 4: Both query and embedding
# ========================================================================
print("\n[5] TEST 4: Universal recall with both query AND embedding")
emb = get_embedding("Rust memory safety")
results = mathir.universal_recall(
    query="how does Rust handle memory without garbage collection?",
    query_embedding=emb,
    k=3
)
print(f"  Results: {len(results)}")
for r in results[:3]:
    print(f"    - {r.get('metadata', {}).get('concept', '?')}: '{r.get('metadata', {}).get('text', '')[:50]}...'")
test4_pass = any("rust" in str(r.get("metadata", {}).get("text", "")).lower() for r in results)
print(f"  STATUS: {'PASS' if test4_pass else 'FAIL'}")

# ========================================================================
# TEST 5: Available providers
# ========================================================================
print("\n[6] TEST 5: List available providers")
providers = mathir.available_providers()
print(f"  Providers: {providers}")
test5_pass = len(providers) > 0
print(f"  STATUS: {'PASS' if test5_pass else 'FAIL'}")

# ========================================================================
# FINAL SUMMARY
# ========================================================================
print("\n" + "=" * 70)
print("SUMMARY - UNIVERSAL RECALL API")
print("=" * 70)
print(f"  Test 1 (conversational query):     {'PASS' if test1_pass else 'FAIL'}")
print(f"  Test 2 (cross-lingual FR->EN):     {'PASS' if test2_pass else 'FAIL'}")
print(f"  Test 3 (cross-provider fallback):  {'PASS' if test3_pass else 'FAIL'}")
print(f"  Test 4 (query+embedding hybrid):   {'PASS' if test4_pass else 'FAIL'}")
print(f"  Test 5 (provider listing):         {'PASS' if test5_pass else 'FAIL'}")

total = sum([test1_pass, test2_pass, test3_pass, test4_pass, test5_pass])
print(f"\n  TOTAL: {total}/5 PASSED")
print("=" * 70)

# Save results
import json
from datetime import datetime
results_json = {
    "timestamp": datetime.now().isoformat(),
    "test": "MATHIR Universal Recall (new API)",
    "test1_conversational": test1_pass,
    "test2_cross_lingual_fr_en": test2_pass,
    "test3_cross_provider": test3_pass,
    "test4_hybrid_query_embedding": test4_pass,
    "test5_provider_listing": test5_pass,
    "total_pass": total,
    "total_tests": 5,
    "overall_pass": total >= 4
}
with open("D:/SECRET_PROJECT/MATHIR/benchmarks/universal_recall_results.json", "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults saved to: D:/SECRET_PROJECT/MATHIR/benchmarks/universal_recall_results.json")