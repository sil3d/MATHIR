"""
MATHIR Latin Name Demo - shows new capabilities
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from mathir_dropin.latin_names import (
    parse_taxonomic_name,
    normalize_diacritics,
    parse_roman_numeral,
    expand_abbreviation,
    split_compound,
    latin_match,
)

def safe_print(s):
    """Print string safely on Windows console (cp1252 encoding)."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


print("=" * 70)
print("MATHIR Latin Name Handler - DEMO")
print("=" * 70)

# 1. Taxonomic parsing
print("\n[1] Taxonomic name parsing")
test_names = [
    "Homo sapiens",
    "Homo sapiens sapiens",
    "Escherichia coli K-12",
    "E. coli",
    "Panthera leo",
    "Homo sapiens Linnaeus 1758",
    "Homo sapiens L.",
]
for name in test_names:
    parsed = parse_taxonomic_name(name)
    print(f"  {name:<35} -> genus={parsed.get('genus')}, species={parsed.get('species')}")

# 2. Diacritics
print("\n[2] Diacritic normalization")
diacritic_tests = [
    "Schrödinger", "Müller", "François", "Łukasiewicz", "Straße", "naïve"
]
for name in diacritic_tests:
    norm = normalize_diacritics(name)
    print(f"  {name:<20} -> {norm}")

# 3. Roman numerals
print("\n[3] Roman numeral parsing")
romans = ["VIII", "XII", "XIX", "IV", "M", "III"]
for r in romans:
    val = parse_roman_numeral(r)
    print(f"  {r:<6} -> {val}")

# 4. Abbreviations
print("\n[4] Abbreviation expansion")
abbrs = ["DNA", "RNA", "ATP", "MRI", "CT", "PCR", "CNS", "E. coli", "H. sapiens"]
for a in abbrs:
    expanded = expand_abbreviation(a)
    print(f"  {a:<12} -> {expanded}")

# 5. Compound terms
print("\n[5] Compound term splitting")
compounds = [
    "sternocleidomastoid",
    "gastrocnemius",
    "temporomandibular",
    "tracheobronchial",
    "carpometacarpal"
]
for c in compounds:
    parts = split_compound(c)
    print(f"  {c:<22} -> {parts}")

# 6. Latin name matching
print("\n[6] Latin name matching (similarity scores)")
matches = [
    ("Homo sapiens", "Homo sapiens"),
    ("E. coli", "Escherichia coli"),
    ("Schrodinger", "Schrödinger"),
    ("Henry VIII", "Henry 8"),
    ("DNA", "deoxyribonucleic acid"),
    ("Homo sapiens", "Python closures"),  # unrelated
    ("acetaminophen", "paracetamol"),  # same drug
]
for q, c in matches:
    score = latin_match(q, c)
    print(f"  {q:<25} <-> {c:<25} = {score:.2f}")

# 7. End-to-end with MATHIR
print("\n[7] End-to-end with MATHIR universal_recall")
from mathir_dropin import MATHIRMemory
import torch
from sentence_transformers import SentenceTransformer

db = "mathir_latin_demo.db"
if os.path.exists(db): os.remove(db)
mathir = MATHIRMemory(embedding_dim=384, db_path=db)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store memories with various Latin names
memories = [
    {"text": "Homo sapiens is the scientific name for modern humans", "concept": "homo-sapiens"},
    {"text": "Escherichia coli K-12 is a model organism in biology", "concept": "e-coli"},
    {"text": "Acetylsalicylic acid is the chemical name for aspirin", "concept": "aspirin"},
    {"text": "Sternocleidomastoid is a muscle in the neck", "concept": "scm-muscle"},
    {"text": "Erwin Schrödinger proposed the famous thought experiment", "concept": "schrodinger"},
]
print("  Storing 5 memories with Latin names...")
for mem in memories:
    emb = torch.from_numpy(model.encode([mem["text"]])).float()
    mathir.store(embedding=emb, metadata=mem, provider="sentence-transformers")

# Query with various forms
print("\n  Querying with different forms of the same name:")
queries = [
    ("H. sapiens", "genus abbreviation"),
    ("E coli", "no period in abbreviation"),
    ("Schrödinger", "with diacritics"),
    ("Schrodinger", "without diacritics"),
    ("acetyl salicylic acid", "split into 3 words"),
    ("sternocleidomastoid", "compound medical term"),
    ("Henry VIII", "Roman numeral in name"),
]
for query, description in queries:
    results = mathir.universal_recall(query=query, k=1)
    if results:
        concept = results[0].get("metadata", {}).get("concept", "?")
        print(f"    '{query}' ({description}) -> {concept}")
    else:
        print(f"    '{query}' ({description}) -> (no match)")

print("\n" + "=" * 70)
print("ALL DONE - Latin name handler works")
print("=" * 70)