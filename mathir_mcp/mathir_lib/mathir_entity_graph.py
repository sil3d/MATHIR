#!/usr/bin/env python3
"""
Entity extraction for MATHIR's entity-linked graph layer.

WHY THIS EXISTS: MATHIR's original link graph (build_links_all in
mathir_vec.py) connects memories by EMBEDDING COSINE SIMILARITY. A HotpotQA
multi-hop benchmark (benchmarks/10_multihop/, 2026-07-01) proved that a
similarity graph structurally cannot encode multi-hop "bridge" relations:
two paragraphs connected only by a shared entity (e.g. "the actress who
played X" -> "that actress became Chief of Protocol") are often NOT
embedding-similar, so no similarity edge is ever created, and graph-based
retrieval (PPR-LTE) has no substrate to chain across. This is the concrete
architectural gap between MATHIR and entity-graph systems like Zep/Graphiti.

This module extracts named entities so memories that MENTION THE SAME ENTITY
can be linked, regardless of embedding similarity -- giving the graph the
bridge edges it was missing.

Extraction strategy (both local, no LLM, no network -- honoring MATHIR's
edge-first design):
  1. spaCy NER (en_core_web_sm) when available -- proper named-entity
     recognition (PERSON, ORG, GPE, WORK_OF_ART, EVENT, etc.).
  2. Regex fallback when spaCy isn't installed -- capitalized multi-word
     sequences (proper-noun phrases), which for encyclopedic/Wikipedia-style
     text captures most bridge entities. Clearly weaker than real NER, but
     keeps the feature working with zero heavy dependencies.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Set

# spaCy entity labels worth linking on. Excludes purely numeric/quantity
# labels (CARDINAL, ORDINAL, PERCENT, MONEY, QUANTITY) which create noisy,
# non-discriminating edges (every doc mentioning "one" or "1990" would link).
_USEFUL_SPACY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "WORK_OF_ART",
    "EVENT", "PRODUCT", "NORP", "LAW", "LANGUAGE",
}

# Stopword-ish leading tokens that shouldn't start an entity in the regex
# fallback (sentence-initial capitalization is not an entity signal).
_REGEX_STOP_STARTS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She",
    "They", "We", "I", "In", "On", "At", "For", "And", "But", "Or", "As",
    "When", "While", "After", "Before", "During", "However", "Although",
}

_CAP_SEQ = re.compile(r"\b([A-Z][a-zA-Z0-9.&'-]*(?:\s+[A-Z][a-zA-Z0-9.&'-]*)*)\b")


@lru_cache(maxsize=1)
def _load_spacy():
    """Load spaCy NER once, cached. Returns None if unavailable."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
        except Exception:
            return None
    except ImportError:
        return None


def _normalize(ent: str) -> str:
    """Normalize an entity string for matching: collapse whitespace, strip
    trailing punctuation, lowercase. Two memories mentioning 'Shirley Temple'
    and 'shirley temple.' should match."""
    e = re.sub(r"\s+", " ", ent).strip().strip(".,;:!?\"'()[]").lower()
    return e


def extract_entities(text: str, min_len: int = 3) -> Set[str]:
    """Extract normalized named entities from text.

    Uses spaCy NER if available, else a capitalized-sequence regex fallback.
    Returns a set of normalized entity strings (lowercased). Entities shorter
    than min_len characters are dropped (too generic to be discriminating).
    """
    if not text or not text.strip():
        return set()

    nlp = _load_spacy()
    ents: Set[str] = set()

    if nlp is not None:
        doc = nlp(text[:100000])  # cap length to bound per-call cost
        for e in doc.ents:
            if e.label_ in _USEFUL_SPACY_LABELS:
                norm = _normalize(e.text)
                if len(norm) >= min_len:
                    ents.add(norm)
    else:
        # Regex fallback: capitalized multi-word sequences.
        for m in _CAP_SEQ.finditer(text):
            phrase = m.group(1)
            first_tok = phrase.split()[0]
            # Skip single sentence-initial stopwords, but keep multi-word
            # phrases even if they start with a common word (rare).
            if phrase in _REGEX_STOP_STARTS:
                continue
            if len(phrase.split()) == 1 and first_tok in _REGEX_STOP_STARTS:
                continue
            norm = _normalize(phrase)
            if len(norm) >= min_len:
                ents.add(norm)

    return ents
