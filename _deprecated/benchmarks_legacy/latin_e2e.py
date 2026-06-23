"""
End-to-end test: Latin name matching through MATHIR universal_recall
Uses the same embedding model for store and query (no random vectors).
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
from sentence_transformers import SentenceTransformer
from mathir_dropin import MATHIRMemory
from mathir_dropin.latin_names import latin_match

db = "mathir_latin_e2e.db"
if os.path.exists(db): os.remove(db)
mathir = MATHIRMemory(embedding_dim=384, db_path=db)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store memories with REAL text
memories = [
    {"text": "Homo sapiens is the scientific name for modern humans who evolved 300,000 years ago", "concept": "homo-sapiens"},
    {"text": "Escherichia coli K-12 is a model bacterium used extensively in molecular biology labs", "concept": "e-coli"},
    {"text": "Acetylsalicylic acid is the IUPAC chemical name for the drug commonly known as aspirin", "concept": "aspirin"},
    {"text": "The sternocleidomastoid muscle in the neck rotates and flexes the head", "concept": "scm-muscle"},
    {"text": "Erwin Schrödinger proposed the famous thought experiment about quantum mechanics", "concept": "schrodinger"},
]
print("Storing 5 memories...")
for mem in memories:
    emb = torch.from_numpy(model.encode([mem["text"]])).float()
    mathir.store(embedding=emb, metadata=mem, provider="sentence-transformers")

# Query with various forms - SAME query gets expanded
print("\nTest: Latin name variants found via universal_recall")
print("=" * 70)
test_cases = [
    ("Homo sapiens", "homo-sapiens", "canonical Latin name"),
    ("H. sapiens", "homo-sapiens", "genus abbreviated"),
    ("Escherichia coli", "e-coli", "canonical bacteria"),
    ("E coli", "e-coli", "abbreviated bacteria"),
    ("Schrödinger", "schrodinger", "with German diacritics"),
    ("Schrodinger", "schrodinger", "without diacritics"),
    ("acetylsalicylic acid", "aspirin", "IUPAC name"),
    ("aspirin", "aspirin", "common name"),
    ("sternocleidomastoid", "scm-muscle", "compound medical term"),
    ("Henry VIII", "homo-sapiens", "Roman numeral (won't match expected)"),
]

correct = 0
total = 0
for query, expected_concept, description in test_cases:
    results = mathir.universal_recall(query=query, k=3)
    if results:
        top = results[0].get("metadata", {}).get("concept", "?")
        match = "[OK]" if top == expected_concept else "[??]"
        if top == expected_concept:
            correct += 1
        total += 1
        print(f"  {match} '{query}' ({description})")
        print(f"     -> top: {top} (expected: {expected_concept})")
    else:
        print(f"  [FAIL] '{query}' -> no results")

print(f"\n{correct}/{total} direct matches via universal_recall")

# Now test latin_match directly
print("\n\nDirect latin_match scores:")
print("=" * 70)
pairs = [
    ("Homo sapiens", "Homo sapiens", 1.0),
    ("E. coli", "Escherichia coli", 1.0),
    ("Schrodinger", "Schrödinger", 1.0),
    ("Henry VIII", "Henry 8", 1.0),
    ("DNA", "deoxyribonucleic acid", 1.0),
    ("acetaminophen", "paracetamol", 0.5),  # similar drugs
    ("Homo sapiens", "Python closures", 0.0),  # unrelated
]
for q, c, expected in pairs:
    score = latin_match(q, c)
    pass_test = "OK" if abs(score - expected) < 0.1 else "??"
    print(f"  {pass_test} '{q}' <-> '{c}' = {score:.2f} (expected ~{expected})")