#!/usr/bin/env python3
"""
Build a BEIR-format corpus.jsonl from two real fluid-mechanics textbooks:
  - White, "Fluid Mechanics", 7th ed. (885 pages)
  - Cengel & Cimbala, "Fluid Mechanics: Fundamentals and Applications" (2036 pages)

This replaces the project's old disavowed toy benchmark (50 hand-curated
queries over 200 chunks of a single textbook, per docs/SOTA_RESEARCH_2024_2026.md)
with a real, larger-scale BEIR-format dataset built the honest way: real PDF
text extraction (pypdf), heuristic front/back-matter trimming calibrated by
actually inspecting sample pages (not guessed blindly), paragraph-aware
chunking, and an auditable noise filter.

Front/back-matter page ranges (0-indexed, inclusive), calibrated by manually
inspecting extracted text at multiple points in each PDF:

  White_2011_7ed_Fluid-Mechanics.pdf (885 pages):
    - pages 0-21: title page, McGraw-Hill series list, author bio, table of
      contents, preface, acknowledgments -- no chapter prose.
    - pages 22-843: real chapter content, including end-of-chapter Problems
      (real descriptive fluid-mechanics word problems -- legitimate passages).
    - pages 844-884: Appendix A/B (property tables, compressible flow
      tables), conversion factors, answers to selected problems, index --
      tables/references, not prose.

  Yunus.pdf (Cengel & Cimbala, 2036 pages):
    - pages 0-25: title page, author bios, table of contents, preface.
    - pages 26-954: real chapter content (Chapters 1-15 + property-table
      appendices A-1..A-10, which get mostly filtered out downstream by the
      noise heuristic anyway since they're numeric tables).
    - pages 955-2036 (1081 pages!): NOT normal back matter -- inspection
      showed this PDF bundles the full McGraw-Hill INSTRUCTOR'S SOLUTIONS
      MANUAL (per-problem "Solution / Analysis ..." answers, one problem at
      a time, "PROPRIETARY MATERIAL... Limited distribution..." boilerplate
      repeated on every page) plus a glossary and an errata list, appended
      after the textbook proper. This is a different content type/register
      from textbook prose (terse, numeric, repetitive boilerplate) and is
      deliberately excluded here so the corpus stays "textbook passages",
      not "worked homework answers". This is why Yunus's back-matter skip
      (1081 pages) is far larger than White's (41 pages) -- it's not a
      guess, it was determined by reading actual extracted text at several
      points in the file (see benchmarks/.superpowers/sdd report for the
      page-by-page inspection that led to this cutoff).

Chunking: paragraph-aware where pypdf's per-page text has blank-line-ish
paragraph breaks; otherwise a simple word-count sliding window. Target
150-300 words/chunk. This is intentionally simple (not over-engineered).

Noise filter (applied per chunk, logged for auditability):
  - drop chunks with < 50 words (likely page-break/figure-caption artifacts)
  - drop chunks where alphabetic characters are < 60% of all non-whitespace
    characters (equation-heavy pages, table dumps, figure-axis-label soup)

Output: BEIR-format corpus.jsonl (`{"_id", "title", "text"}` per line) at
benchmarks/05_test_data/beir_data/fluid_mechanics/fluid_mechanics/corpus.jsonl
(double-nested directory, matching the scifact/nfcorpus/arguana convention
used by benchmarks/03_vector_search_benchmarks/multi_dataset_efficient.py).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# Switched from pypdf to PyMuPDF (fitz): pypdf's text extraction on these two
# PDFs produced systematic character corruption -- Unicode replacement
# characters and mangled ligature/symbol codes (e.g. "Navier�Stokes",
# "H11005" in place of subscripts/special glyphs) in roughly a third of
# sampled chunks, likely from embedded math fonts pypdf doesn't decode
# correctly. PyMuPDF was verified on the exact same pages that were garbled
# under pypdf and produced clean prose with correct ligatures (e.g. "flow",
# "definition") and no replacement characters.

# Windows consoles default to cp1252 which chokes on some PDF-extracted
# ligature/unicode chars (e.g. U+FB02). Reconfigure stdout to be tolerant
# so sanity-check printing never crashes an otherwise-successful run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BOOKS = {
    "white": {
        "path": Path(__file__).resolve().parent.parent / "Fluid_mecanique_book" / "White_2011_7ed_Fluid-Mechanics.pdf",
        "title": "White, Fluid Mechanics 7th ed.",
        "start_page": 22,   # 0-indexed, inclusive
        "end_page": 843,    # 0-indexed, inclusive
    },
    "yunus": {
        "path": Path(__file__).resolve().parent.parent / "Fluid_mecanique_book" / "Yunus.pdf",
        "title": "Cengel & Cimbala, Fluid Mechanics: Fundamentals and Applications",
        "start_page": 26,   # 0-indexed, inclusive
        "end_page": 954,    # 0-indexed, inclusive (excludes bundled solutions manual, pages 955-2036)
    },
}

MIN_WORDS = 50
TARGET_MIN_WORDS = 150
TARGET_MAX_WORDS = 300
MIN_ALPHA_RATIO = 0.60

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "05_test_data"
    / "beir_data"
    / "fluid_mechanics"
    / "fluid_mechanics"
)


def alpha_ratio(text: str) -> float:
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    alpha = sum(1 for c in non_ws if c.isalpha())
    return alpha / len(non_ws)


def split_paragraphs(page_text: str) -> list[str]:
    """Paragraph-aware split: prefer blank-line-ish breaks, else return whole page as one block."""
    # pypdf often emits single '\n' between lines within a paragraph and no
    # reliable double-newline for paragraph breaks -- so also split on
    # sentence-ish boundaries followed by a capital start as a fallback isn't
    # needed; a simple word-window over the whole page is fine (per prompt:
    # "don't over-engineer this").
    parts = re.split(r"\n\s*\n", page_text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else ([page_text.strip()] if page_text.strip() else [])


def chunk_words(words: list[str], target_min: int, target_max: int) -> list[list[str]]:
    """Simple sliding window over a word list, no overlap."""
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = min(i + target_max, n)
        chunks.append(words[i:j])
        i = j
    return chunks


def extract_book_chunks(book_key: str, cfg: dict) -> tuple[list[dict], dict]:
    doc = fitz.open(str(cfg["path"]))
    n_pages = len(doc)
    start = cfg["start_page"]
    end = min(cfg["end_page"], n_pages - 1)

    stats = {"pages_read": 0, "raw_chunks": 0, "filtered_short": 0, "filtered_noise": 0, "kept": 0}
    chunks = []
    chunk_idx_per_page = {}

    for page_num in range(start, end + 1):
        try:
            text = doc[page_num].get_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        stats["pages_read"] += 1

        paragraphs = split_paragraphs(text)
        # Flatten all words on the page (paragraph-aware chunking within a
        # page, sliding window as fallback for long paragraphs).
        for para in paragraphs:
            words = para.split()
            if not words:
                continue
            for word_chunk in chunk_words(words, TARGET_MIN_WORDS, TARGET_MAX_WORDS):
                stats["raw_chunks"] += 1
                chunk_text = " ".join(word_chunk)
                n_words = len(word_chunk)

                if n_words < MIN_WORDS:
                    stats["filtered_short"] += 1
                    continue
                if alpha_ratio(chunk_text) < MIN_ALPHA_RATIO:
                    stats["filtered_noise"] += 1
                    continue

                c_idx = chunk_idx_per_page.get(page_num, 0)
                chunk_idx_per_page[page_num] = c_idx + 1
                _id = f"{book_key}_p{page_num}_c{c_idx}"
                chunks.append(
                    {
                        "_id": _id,
                        "title": f"{cfg['title']} — page {page_num}",
                        "text": chunk_text,
                    }
                )
                stats["kept"] += 1

    return chunks, stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    all_stats = {}

    for book_key, cfg in BOOKS.items():
        print(f"\n=== {book_key}: {cfg['path'].name} ===")
        chunks, stats = extract_book_chunks(book_key, cfg)
        all_stats[book_key] = stats
        all_chunks.extend(chunks)
        print(f"  Pages with text read: {stats['pages_read']}")
        print(f"  Raw chunks before filtering: {stats['raw_chunks']}")
        print(f"  Filtered (< {MIN_WORDS} words): {stats['filtered_short']}")
        print(f"  Filtered (< {MIN_ALPHA_RATIO*100:.0f}% alphabetic): {stats['filtered_noise']}")
        print(f"  KEPT: {stats['kept']}")

    print(f"\n=== TOTAL corpus size: {len(all_chunks)} chunks ===")
    for book_key, stats in all_stats.items():
        print(f"  {book_key}: {stats['kept']} chunks survived "
              f"({stats['filtered_short']} short + {stats['filtered_noise']} noise filtered out "
              f"of {stats['raw_chunks']} raw)")

    out_path = OUT_DIR / "corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks to {out_path}")

    print("\n=== Sample chunks (sanity check) ===")
    import random
    random.seed(42)
    sample = random.sample(all_chunks, min(8, len(all_chunks)))
    for c in sample:
        print(f"\n--- {c['_id']} | {c['title']} ---")
        print(c["text"][:400])

    return all_chunks, all_stats


if __name__ == "__main__":
    main()
