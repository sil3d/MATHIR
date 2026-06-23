# SHARED CONTEXT — Latin Names & Technical Nomenclature Handling

Started: 2026-06-06

## PROBLEM
MATHIR's universal_bridge currently handles:
- Common words (English, French) via n-grams
- Different embedding spaces via Procrustes
- But NOT specifically Latin/scientific/proper nouns

Need to handle:
- **Taxonomic names**: "Homo sapiens", "Escherichia coli", "Panthera leo"
- **Anatomical Latin**: "sternocleidomastoid", "gastrocnemius", "latissimus dorsi"
- **Pharmaceutical**: "acetaminophen", "ibuprofen", "acetylsalicylic acid"
- **Legal Latin**: "habeas corpus", "pro bono", "ex parte"
- **Astronomical**: "Alpha Centauri", "Sirius", "Betelgeuse"
- **Proper names with diacritics**: "Schrödinger", "Müller", "François"
- **Roman numerals**: "Henry VIII", "World War II", "Type III"
- **Multi-word technical**: "major histocompatibility complex"

## CONSTRAINTS
- Algorithm must work WITHOUT retraining
- Must work with existing UNIBRI infrastructure
- Must handle case-insensitive matching
- Must preserve diacritics and special characters
- Must handle abbreviations (acronyms vs full names)

## EXISTING INTELLIGENCE
- UNIBRI uses multi-resolution character n-grams (Broder 1997)
- Has normalize_unicode (NFKC/NFD)
- Has transliteration function
- FTS5 with porter unicode61 tokenizer
- Cross-lingual matching works for French/English

## ACTIVE AGENTS
- @math: Design Latin name algorithm (taxonomic structure, diacritic handling, case rules)
- @background-researcher: Research Latin nomenclature rules, scientific name conventions
- @internet_search: Find existing Latin name libraries (GlobalNames, GBIF, ITIS)
- @coder: Implement the algorithm in universal_bridge.py
- @debugger: Find edge cases (empty parts, abbreviations, mixed scripts)
- @test: Write comprehensive test suite

## DELIVERABLES
1. New `latin_names.py` module with:
   - Taxonomic name parser (genus + species + author)
   - Anatomical term normalizer
   - Diacritic-aware matching
   - Abbreviation expansion (DNA, RNA, etc.)
2. Integration with universal_recall
3. Test suite with real Latin names
4. Benchmark results