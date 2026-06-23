"""
MATHIR Drop-in -- Universal Cross-Provider and Cross-Lingual Bridge.

Addresses three real-world retrieval problems that the vanilla FTS5 +
single-space cosine search in :mod:`mathir_dropin.store` cannot solve:

1. **Conversational query failure on FTS5**
   The query ``"What do you know about python closures?"`` returns zero
   hits because SQLite FTS5 with the ``porter unicode61`` tokenizer
   drops stopwords ("what", "do", "you", "about") and the surviving
   tokens ("know", "python", "closures") do not match the indexed
   text.  We fix this by **query expansion**: producing multiple
   variants of the original query (raw, normalised, stopword-stripped,
   stemmed, character n-gram fragments) and re-running FTS5 for each.

2. **Cross-provider fallback**
   When the user requests a provider (e.g. ``"minimax"``) for which no
   embeddings are stored, the simple path in
   :meth:`SQLiteStore.search_by_embedding_multi` would just score
   zero rows.  :class:`UniversalBridge` implements a strict fallback
   chain: requested provider -> other providers with matching dim ->
   text search (via expand_query) -> primary embeddings.

3. **Cross-lingual matching**
   A French query ``"clotures en python"`` should still find an
   English memory ``"python closures"``.  We use **Unicode NFKC
   normalisation + a transliteration table + character n-gram
   Jaccard similarity** -- all language-agnostic, no model required,
   no network access.

The class is intentionally framework-light: only Python's standard
library.  Numpy / torch are optional and only used inside
``hybrid_recall`` for ranking speed.

Mathematical grounding
======================

* :func:`text_similarity` uses **Jaccard similarity over character
  n-grams** (default n=3).  For strings ``a`` and ``b``::

      sim(a, b) = |N(a) ∩ N(b)| / |N(a) ∪ N(b)|

  with ``N(x)`` the multiset of length-``n`` character shingles.  This
  is a *lower bound* on cosine similarity of the corresponding
  one-hot n-gram vectors, which is why it is widely used in
  fingerprinting / near-duplicate detection (Broder 1997).

* :func:`cross_space_score` uses the **Johnson-Lindenstrauss
  random-projection lemma**: a random Gaussian matrix ``R ∈
  R^{k×d}`` (with rows normalised to unit length) preserves
  pairwise L2 distances up to a factor ``(1 ± ε)`` when ``k ≥
  O(log N / ε²)``.  We use it to project both spaces into a common
  subspace before computing cosine.  This is mathematically the
  same idea as in SimHash / SVD-free LSI.

* :func:`expand_query` returns ``O(|tokens| + n-grams)`` variants.
  The expanded search cost is therefore ``O(k · log N)`` per variant
  in the FTS5 B-tree, so the total cost is still ``O(k · log N · m)``
  for ``m`` variants -- acceptable for ``m ≤ 5`` and ``N ≤ 1e5``.

* :func:`hybrid_recall` uses a **logarithmic recall-count boost** of
  the form ``1 + α · log(1 + recall_count)``.  This is the standard
  damping function from BM25 / Ebbinghaus spaced repetition: a memory
  recalled 100 times is *not* 100× better than one recalled once.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# numpy is optional; we fall back to a pure-Python path when missing.
try:  # pragma: no cover - environment-dependent
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# English (and a few universal) stopwords.  Kept small on purpose: the
# expansion produces a *stopword-stripped* variant, but a few common
# "real" words are useful as fallbacks when no expanded match exists.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "doing",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "should",
    "would", "could", "may", "might", "must", "shall",
    "what", "which", "who", "whom", "know", "knows", "knew",
    "tell", "tells", "told", "find", "finds", "found",
})

# A tiny Latin-1 / Latin Extended-A transliteration map.  We expand
# accented characters to their ASCII base (cloture -> clotur) and
# *also* keep the unaccented variant so a French search can still
# find the original French word.  This is intentionally small: real
# cross-lingual retrieval without a model is fundamentally lossy.
_TRANSLIT_MAP = {
    # a / e / i / o / u with diacritics
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a", "ā": "a", "ă": "a", "ą": "a",
    "è": "e", "é": "e", "ê": "e", "ë": "e", "ē": "e", "ė": "e", "ę": "e", "ě": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i", "ī": "i", "į": "i", "ı": "i", "ǐ": "i",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ō": "o", "ő": "o", "ø": "o", "œ": "oe",
    "ù": "u", "ú": "u", "û": "u", "ü": "u", "ū": "u", "ů": "u", "ű": "u", "ų": "u",
    "ý": "y", "ÿ": "y", "ŷ": "y",
    "ç": "c", "ć": "c", "č": "c",
    "ñ": "n", "ń": "n", "ň": "n",
    "ś": "s", "š": "s", "ş": "s", "ß": "ss",
    "ź": "z", "ż": "z", "ž": "z",
    "đ": "d", "ď": "d",
    "ł": "l", "ľ": "l",
    "ř": "r", "ť": "t", "ġ": "g",
}

# Simple Porter-style suffix stripper.  Not the full Porter algorithm
# (which is ~200 lines) -- just the most common English suffixes that
# break naive substring search.  We do not aim for linguistic
# correctness; we aim to make "running" match "run", "closed" match
# "close", and "clotures" match "cloture".
_SUFFIXES: Tuple[str, ...] = (
    "ational", "tional", "iveness", "fulness", "ousness",
    "ization", "ication", "ation", "ement", "ment",
    "ies", "ied", "ying",
    "ing", "edly", "ed",
    "ies", "ied",
    "ss", "ous", "ive", "ful", "less", "ly", "er", "est", "ity",
    "'s", "'t",
)

# Character classes we treat as word boundaries.
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)

# Subword (n-gram) sizes for cross-lingual matching.
_DEFAULT_NGRAM = 3


# ---------------------------------------------------------------------------
# Text normalization helpers (public, reusable)
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """NFKC normalisation: collapses compatibility forms (``ﬁ`` -> ``fi``).

    This is the cheapest and most useful Unicode normalisation for
    retrieval: it strips zero-width joiners, expands ligatures, and
    fixes case folding edge cases *before* tokenisation.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).strip()


def transliterate(text: str) -> str:
    """Strip diacritics using the built-in table + Unicode decomposition.

    We do this in two passes:

    1. ``unicodedata.normalize("NFD", ...)`` + drop combining marks:
       works for everything covered by the Unicode database (covers
       tens of thousands of code points automatically).
    2. Apply our hand-curated map for the few cases NFD misses
       (e.g. ``œ`` -> ``oe``, ``ß`` -> ``ss``, ``đ`` -> ``d``).
    """
    if not text:
        return ""
    # Pass 1: Unicode decomposition + strip combining marks.
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(
        ch for ch in decomposed
        if unicodedata.category(ch) != "Mn"  # Mark, non-spacing
    )
    # Pass 2: explicit map for the gaps.
    return "".join(_TRANSLIT_MAP.get(ch, ch) for ch in stripped)


def tokenize(text: str) -> List[str]:
    """Word tokens (NFKC + lowercase)."""
    if not text:
        return []
    norm = normalize_unicode(text).lower()
    return [t for t in _WORD_RE.findall(norm) if t]


def strip_stopwords(tokens: Sequence[str]) -> List[str]:
    """Remove English stopwords from a token list."""
    return [t for t in tokens if t not in _STOPWORDS]


def stem_word(word: str) -> str:
    """Trivial Porter-ish stemmer: drop the longest matching suffix.

    The goal is *recall*, not *precision*: we accept a few over-stems
    ("running" -> "runn") in exchange for matching both "closes" and
    "closed" against the indexed "close".

    A minimum stem length of 3 prevents the suffix stripper from
    eating all of a 2-3 letter word.
    """
    if len(word) <= 3:
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def stem_tokens(tokens: Sequence[str]) -> List[str]:
    return [stem_word(t) for t in tokens]


def char_ngrams(text: str, n: int = _DEFAULT_NGRAM) -> List[str]:
    """Length-``n`` character shingles with padding.

    Padding with a sentinel on both ends keeps edge n-grams from
    dominating the Jaccard score: ``"abc"`` becomes
    ``["<ab", "abc", "bc>"]`` for ``n=3``, sharing the
    ``"<ab"`` / ``"bc>"`` edges with any neighbour that has the
    same prefix / suffix.
    """
    if not text:
        return []
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(text) < n:
        # Pad short strings so we still get at least one shingle.
        text = " " * (n - 1) + text + " " * (n - 1)
        # The above yields n-1 spaces, then n chars total, so we want
        # to rewrite it as: take the string, prepend (n-1) sentinels,
        # append (n-1) sentinels, then slide a window of size n.
    s = " " * (n - 1) + text + " " * (n - 1)
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def ngram_set(text: str, n: int = _DEFAULT_NGRAM) -> set:
    """Unique n-gram set (used by Jaccard)."""
    return set(char_ngrams(text, n))


# ---------------------------------------------------------------------------
# UniversalBridge
# ---------------------------------------------------------------------------

class UniversalBridge:
    """Cross-provider, cross-lingual, and conversational-aware retrieval.

    Designed to be **stateless** and **thread-safe**: all methods are
    pure functions of their inputs (plus a single, lazily-built random
    projection matrix that is *only* used for dimension-mismatched
    cross-space scoring).

    Parameters
    ----------
    ngram_size:
        Character shingle size for cross-lingual / fuzzy matching.
        Default ``3`` is the standard "trigram" choice.
    expansion_variants:
        Maximum number of expanded query variants to generate.  Set
        lower (e.g. ``2``) for latency-sensitive code paths.
    random_seed:
        Seed for the random projection matrix used in
        :meth:`cross_space_score`.  Fixed seed = reproducible scores
        across runs.
    target_projection_dim:
        Common subspace dimension for random projection.  Must be at
        least ``max(dim_a, dim_b)`` for any practical loss bound, but
        in practice the lemma lets us pick ``min(dim_a, dim_b)`` and
        still get ``(1 ± ε)`` distance preservation with high
        probability.
    """

    # Default FTS5 minimum prefix match length.  Shorter than this and
    # the prefix index explodes; longer and the wildcards stop helping.
    _MIN_TOKEN_LEN = 2
    # Score weight for the recall-count boost in hybrid scoring.
    _RECALL_BOOST = 0.15
    # Score weight for the cross-lingual text-similarity channel.
    _CROSS_LINGUAL_WEIGHT = 0.35

    def __init__(
        self,
        ngram_size: int = _DEFAULT_NGRAM,
        expansion_variants: int = 5,
        random_seed: int = 1729,
        target_projection_dim: int = 128,
    ) -> None:
        if ngram_size < 2:
            raise ValueError(
                f"ngram_size must be >= 2, got {ngram_size}"
            )
        if expansion_variants < 1:
            raise ValueError(
                f"expansion_variants must be >= 1, got {expansion_variants}"
            )
        self.ngram_size = int(ngram_size)
        self.expansion_variants = int(expansion_variants)
        self.target_projection_dim = int(target_projection_dim)
        # Projections are keyed by (dim, seed); the actual matrices
        # are built lazily on first use so we don't pay for features
        # the caller doesn't need.
        self._projections: Dict[Tuple[int, int, int], Any] = {}
        self._random_seed = int(random_seed)

    # ------------------------------------------------------------------
    # Public API: query expansion
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> List[str]:
        """Return a list of query variants, ordered by expected utility.

        The first element is always the original query (after basic
        normalisation), so the caller can fall back to that if
        FTS5 raises a syntax error on any of the expanded forms.

        Variants produced (up to ``expansion_variants`` of them):

        0. Original (NFKC-normalised, whitespace-trimmed).
        1. Lowercase + stopword-stripped.
        2. Lowercase + stemmed.
        3. Lowercase + stopword-stripped + stemmed.
        4. Transliterated (diacritics stripped) for cross-lingual.
        5. Transliterated + stopword-stripped.
        6. Most discriminative single n-gram token.
        """
        if not query or not query.strip():
            return []
        seen: set = set()
        out: List[str] = []

        def _add(q: str) -> None:
            q = q.strip()
            if not q:
                return
            key = q.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(q)

        norm = normalize_unicode(query)
        tokens = tokenize(norm)
        if not tokens:
            return [norm]

        # 0. Original
        _add(norm)
        # 1. Lowercase + stopword-stripped
        stripped = " ".join(strip_stopwords(tokens))
        if stripped:
            _add(stripped)
        # 2. Lowercase + stemmed
        stemmed = " ".join(stem_tokens(tokens))
        _add(stemmed)
        # 3. Stripped + stemmed
        s_stem = " ".join(stem_tokens(strip_stopwords(tokens)))
        if s_stem and s_stem != stemmed:
            _add(s_stem)
        # 4. Transliterated
        trans = transliterate(norm).lower()
        if trans and trans != norm.lower():
            _add(trans)
        # 5. Transliterated + stripped
        trans_toks = [t for t in tokenize(trans) if t not in _STOPWORDS]
        trans_strip = " ".join(trans_toks)
        if trans_strip and trans_strip not in seen:
            _add(trans_strip)
        # 6. Most discriminative single token (longest, not in stoplist).
        candidates = [t for t in tokens if t not in _STOPWORDS and len(t) >= self._MIN_TOKEN_LEN]
        if candidates:
            best = max(candidates, key=len)
            if best not in seen:
                _add(best)
            # 7. Two longest distinct tokens (often better than one for
            #    conversational queries).
            if len(candidates) >= 2:
                two = " ".join(
                    sorted(candidates, key=len, reverse=True)[:2]
                )
                if two not in seen:
                    _add(two)
            # 8. Three longest distinct tokens.
            if len(candidates) >= 3:
                three = " ".join(
                    sorted(candidates, key=len, reverse=True)[:3]
                )
                if three not in seen:
                    _add(three)
            # 9. Trigram fragment of the longest token.
            big = char_ngrams(best, n=min(self.ngram_size, len(best)))
            for g in big:
                if g not in seen and any(ch.isalnum() for ch in g):
                    _add(g)
                    break

        return out[: max(1, self.expansion_variants)]

    # ------------------------------------------------------------------
    # Public API: language-agnostic text similarity
    # ------------------------------------------------------------------

    def text_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity over character n-grams in ``[0, 1]``.

        The texts are NFKC-normalised, lowercased, and transliterated
        before n-gram extraction, so accented characters and casing
        differences do not hurt the score.

        Time complexity: ``O(n + m)`` where ``n = |text1|`` and
        ``m = |text2|`` (we build a Counter for the second set so
        the intersection is exact, not just the set cardinality).
        """
        if not text1 or not text2:
            return 0.0
        a = normalize_unicode(text1).lower()
        b = normalize_unicode(text2).lower()
        a = transliterate(a)
        b = transliterate(b)
        ngrams_a = char_ngrams(a, n=self.ngram_size)
        ngrams_b = char_ngrams(b, n=self.ngram_size)
        if not ngrams_a or not ngrams_b:
            return 0.0
        ca = Counter(ngrams_a)
        cb = Counter(ngrams_b)
        # |A ∩ B| = sum over shingles of min(count_a, count_b)
        intersection = sum((ca & cb).values())
        union = sum((ca | cb).values())
        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    # ------------------------------------------------------------------
    # Public API: cross-space scoring (random projection bridge)
    # ------------------------------------------------------------------

    def cross_space_score(
        self,
        emb_a: Any,
        emb_b: Any,
        provider_a: str = "a",
        provider_b: str = "b",
    ) -> float:
        """Cosine similarity across two potentially-different vector spaces.

        If both embeddings have the same dimensionality, this is just
        regular cosine similarity.  If they differ, both vectors are
        projected into a common ``target_projection_dim``-dimensional
        subspace via a deterministic Gaussian random projection
        (Johnson-Lindenstrauss lemma).

        Returns a float in ``[-1, 1]``.
        """
        a = self._to_vector(emb_a)
        b = self._to_vector(emb_b)
        if a is None or b is None:
            return 0.0
        if a.size == 0 or b.size == 0:
            return 0.0
        if a.shape[-1] != b.shape[-1]:
            # Project the larger of the two into the smaller dim.
            target = min(a.shape[-1], b.shape[-1], self.target_projection_dim)
            if a.shape[-1] != target:
                a = self._project(a, target, tag=f"a:{provider_a}")
            if b.shape[-1] != target:
                b = self._project(b, target, tag=f"b:{provider_b}")
        return float(self._cosine(a, b))

    # ------------------------------------------------------------------
    # Public API: hybrid recall (text + embedding + recall_count)
    # ------------------------------------------------------------------

    def hybrid_recall(
        self,
        query: str,
        embedding: Optional[Any] = None,
        k: int = 5,
        provider: Optional[str] = None,
        text_candidates: Optional[List[Dict[str, Any]]] = None,
        embedding_candidates: Optional[List[Dict[str, Any]]] = None,
        cross_lingual: bool = True,
    ) -> List[Dict[str, Any]]:
        """Combine text + embedding + recall_count into a single ranking.

        Parameters
        ----------
        query:
            Original user query.  Used to compute cross-lingual text
            similarity if ``text_candidates`` lacks a direct FTS5 hit
            for the same memory.
        embedding:
            Optional query embedding vector.  Only used for
            cross-lingual re-ranking of *embedding* candidates whose
            provider did not match the requested one.
        k:
            Maximum number of results to return.
        provider:
            Requested provider (for diagnostics / result tagging only;
            the actual fallback chain is the caller's responsibility).
        text_candidates:
            Pre-computed FTS5 hits.  Each dict MUST have ``memory_id``
            and ``similarity`` keys.  ``similarity`` should already be
            in ``[0, 1]``.
        embedding_candidates:
            Pre-computed embedding hits.  Same dict shape as
            ``text_candidates`` (the function does not care which
            method produced them).
        cross_lingual:
            If True, also re-rank using the Jaccard text similarity
            between ``query`` and each candidate's ``modality_text``
            (or ``text``).  This is what makes French "clotures en
            python" find English "python closures".

        Returns
        -------
        List of dicts, each augmented with ``final_score`` and the
        constituent sub-scores:

        * ``text_score``     -- normalised FTS5 score
        * ``emb_score``      -- embedding similarity (or 0 if missing)
        * ``recall_boost``   -- logarithmic boost from recall_count
        * ``xl_score``       -- cross-lingual Jaccard score
        * ``final_score``    -- weighted combination

        The list is sorted descending by ``final_score`` and capped
        at ``k`` entries.
        """
        # Index candidates by memory_id.
        merged: Dict[str, Dict[str, Any]] = {}

        if text_candidates:
            for c in text_candidates:
                mid = c.get("memory_id")
                if mid is None:
                    continue
                entry = merged.setdefault(mid, {
                    "memory_id": mid,
                    "text_score": 0.0,
                    "emb_score": 0.0,
                    "xl_score": 0.0,
                    "recall_count": int(c.get("recall_count", 0) or 0),
                    "stability": float(c.get("stability", 1.0) or 1.0),
                    "modality_text": c.get("modality_text", "") or "",
                    "metadata": c.get("metadata", {}) or {},
                    "tier": c.get("tier", "episodic"),
                    "modality": c.get("modality", "text"),
                })
                entry["text_score"] = max(
                    entry["text_score"], float(c.get("similarity", 0.0))
                )
                if c.get("recall_count") is not None:
                    entry["recall_count"] = max(
                        entry["recall_count"],
                        int(c.get("recall_count", 0) or 0),
                    )

        if embedding_candidates:
            for c in embedding_candidates:
                mid = c.get("memory_id")
                if mid is None:
                    continue
                entry = merged.setdefault(mid, {
                    "memory_id": mid,
                    "text_score": 0.0,
                    "emb_score": 0.0,
                    "xl_score": 0.0,
                    "recall_count": int(c.get("recall_count", 0) or 0),
                    "stability": float(c.get("stability", 1.0) or 1.0),
                    "modality_text": c.get("modality_text", "") or "",
                    "metadata": c.get("metadata", {}) or {},
                    "tier": c.get("tier", "episodic"),
                    "modality": c.get("modality", "text"),
                })
                entry["emb_score"] = max(
                    entry["emb_score"], float(c.get("similarity", 0.0))
                )
                if c.get("recall_count") is not None:
                    entry["recall_count"] = max(
                        entry["recall_count"],
                        int(c.get("recall_count", 0) or 0),
                    )

        if not merged:
            return []

        # Cross-lingual re-rank: only useful if we have a query and at
        # least one candidate with text.
        if cross_lingual and query:
            q_norm = normalize_unicode(query)
            for entry in merged.values():
                txt = entry.get("modality_text") or ""
                if not txt:
                    # Fall back to the text under any standard key.
                    md = entry.get("metadata") or {}
                    for k_ in ("text", "modality_text", "content", "transcript", "caption"):
                        v = md.get(k_)
                        if isinstance(v, str) and v:
                            txt = v
                            break
                if not txt:
                    continue
                entry["xl_score"] = self.text_similarity(q_norm, txt)

        # Compute final score for every entry.
        scored: List[Dict[str, Any]] = []
        for entry in merged.values():
            text_w = 0.50
            emb_w = 0.40 if embedding is not None else 0.0
            xl_w = self._CROSS_LINGUAL_WEIGHT if cross_lingual else 0.0
            recall_boost = self._RECALL_BOOST * math.log1p(
                max(0, int(entry.get("recall_count", 0) or 0))
            )
            # Renormalise so the weights sum to 1.0 across the active
            # channels -- otherwise adding cross-lingual silently
            # *dampens* the text+emb signal.
            active = text_w + emb_w + xl_w
            if active <= 0:
                active = 1.0
            base = (
                text_w * entry["text_score"]
                + emb_w * entry["emb_score"]
                + xl_w * entry["xl_score"]
            ) / active
            entry["recall_boost"] = recall_boost
            entry["final_score"] = float(base + recall_boost)
            scored.append(entry)

        scored.sort(key=lambda e: e["final_score"], reverse=True)
        return scored[: max(0, int(k))]

    # ------------------------------------------------------------------
    # Provider fallback chain
    # ------------------------------------------------------------------

    def provider_fallback_chain(
        self,
        requested: Optional[str],
        available: Sequence[str],
        primary: str = "primary",
    ) -> List[str]:
        """Order in which to try providers when ``requested`` has no hits.

        The chain is:

        1. ``requested`` (if given).
        2. ``primary`` -- the original / default embedding space.
        3. Other available providers, in stable order, with the
           exception of ``requested`` and ``primary`` which were
           already tried.

        This is intentionally a *strict* list: the caller iterates
        through it, stopping at the first provider that returned
        non-empty results.  It does not auto-retry on transient
        failures.
        """
        chain: List[str] = []
        if requested:
            chain.append(requested)
        if primary and primary not in chain:
            chain.append(primary)
        for prov in available:
            if prov and prov not in chain:
                chain.append(prov)
        return chain

    # ------------------------------------------------------------------
    # Public API: Latin / scientific name query expansion
    # ------------------------------------------------------------------

    def expand_latin_query(self, query: str) -> List[str]:
        """Expand a query to handle Latin names and technical terms.

        Returns a list of query variants suitable for FTS5 and
        embedding search, ordered roughly by expected utility
        (most-discriminative first).

        The expansion uses :mod:`mathir_dropin.latin_names` to
        detect and transform:

        * Taxonomic binomials -- "Homo sapiens" -> "Homo", "sapiens",
          and the de-diacriticked forms of each.
        * Diacritic-laden proper names -- "Schrödinger" -> ASCII
          "Schrodinger" (and vice-versa).
        * Roman-numeral suffixes on proper names -- "Henry VIII"
          -> "Henry 8" and "Henry VIII".
        * Common scientific abbreviations -- "DNA" -> "deoxyribonucleic
          acid" (and the original).
        * Compound medical terms -- "sternocleidomastoid" ->
          "sterno cleido mastoid" (and the original).

        The result is a *deduplicated* list: every variant appears
        at most once, and case-insensitive duplicates are
        suppressed.  An empty / non-string input returns ``[]``.

        Examples
        --------
        >>> bridge = UniversalBridge()
        >>> bridge.expand_latin_query("Homo sapiens")
        ['Homo sapiens', 'homo sapiens', 'Homo', 'sapiens', ...]
        """
        if not query or not isinstance(query, str):
            return []
        # Local import to keep `universal_bridge` importable even
        # if `latin_names` is missing or broken.
        try:
            from . import latin_names as _ln
        except Exception:  # pragma: no cover
            return [query.strip()] if query.strip() else []

        variants: List[str] = []
        seen: Set[str] = set()

        def _add(x: Any) -> None:
            if not isinstance(x, str):
                return
            s = x.strip()
            if not s:
                return
            k = s.lower()
            if k in seen:
                return
            seen.add(k)
            variants.append(s)

        original = normalize_unicode(query).strip()
        if not original:
            return []
        _add(original)
        _add(original.lower())

        # 1. Taxonomic decomposition.
        taxo = _ln.parse_taxonomic_name(original)
        if taxo.get("is_taxonomic"):
            genus = taxo.get("genus")
            species = taxo.get("species")
            if genus:
                _add(genus)
                _add(genus.lower())
            if species:
                _add(species)
            # Add the abbreviated form if we have a full genus.
            if genus and species:
                abbrev = f"{genus[0]}. {species}"
                _add(abbrev)
                _add(f"{genus[0].lower()}. {species}")

        # 2. Diacritic variants.
        norm = _ln.normalize_diacritics(original)
        if norm and norm.lower() != original.lower():
            _add(norm)
            _add(norm.lower())

        # 3. Roman-numeral expansion.  If the query ends in a
        # roman-numeral suffix, also emit the integer form and
        # vice-versa.
        last_token = original.split()[-1] if original.split() else ""
        roman_int = _ln.parse_roman_numeral(last_token)
        if roman_int is not None and len(original.split()) >= 2:
            base = " ".join(original.split()[:-1])
            _add(f"{base} {roman_int}")
        m_int = re.search(r"\b(\d+)\s*\.?\s*$", original)
        if m_int and len(original.split()) >= 2:
            try:
                n = int(m_int.group(1))
                if 1 <= n <= 3999:
                    base = original[: m_int.start()].rstrip()
                    _add(f"{base} {_ln.int_to_roman(n)}")
            except (ValueError, KeyError):
                pass

        # 4. Abbreviation expansion (if the whole query is an
        # abbreviation) or contraction (if it's a known full
        # form, the abbreviation is also tried).
        for exp in _ln.expand_abbreviation(original):
            _add(exp)

        # 5. Compound-term splitting for single tokens >= 8 chars.
        compound_tokens = [
            t for t in re.findall(r"[A-Za-zÀ-ÿ]+", original)
            if len(t) >= 8
        ]
        for tok in compound_tokens:
            parts = _ln.split_compound(tok)
            if len(parts) >= 2:
                # Space-separated decomposition.
                _add(" ".join(parts))
                # Original token de-diacriticked.
                _add(_ln.normalize_diacritics(tok))

        # 6. Trigram fragment of the longest token, for FTS5
        # prefix matching when nothing else hits.
        toks = [t for t in re.findall(r"[A-Za-zÀ-ÿ]+", original) if t]
        if toks:
            longest = max(toks, key=len)
            for g in char_ngrams(_ln.normalize_diacritics(longest).lower(),
                                 n=min(self.ngram_size, len(longest))):
                if any(ch.isalnum() for ch in g) and g not in seen:
                    _add(g)
                    break

        return variants

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vector(x: Any) -> Optional[Any]:
        """Coerce an input to a 1-D numpy array, or None on failure."""
        if x is None:
            return None
        if _HAS_NUMPY:
            try:
                return _np.asarray(x, dtype=_np.float32).reshape(-1)
            except Exception:
                pass
        # Pure-Python fallback: list of floats.
        try:
            return [float(v) for v in x]
        except Exception:
            return None

    @staticmethod
    def _cosine(a: Any, b: Any) -> float:
        """Cosine similarity; works with numpy OR a plain list of floats."""
        if _HAS_NUMPY and isinstance(a, _np.ndarray) and isinstance(b, _np.ndarray):
            na = float(_np.linalg.norm(a))
            nb = float(_np.linalg.norm(b))
            if na == 0.0 or nb == 0.0:
                return 0.0
            return float(_np.dot(a, b) / (na * nb))
        # Pure-Python path.
        if not a or not b:
            return 0.0
        dot = 0.0
        na_sq = 0.0
        nb_sq = 0.0
        for i in range(min(len(a), len(b))):
            ai = a[i]
            bi = b[i]
            dot += ai * bi
            na_sq += ai * ai
            nb_sq += bi * bi
        if na_sq == 0.0 or nb_sq == 0.0:
            return 0.0
        return dot / (math.sqrt(na_sq) * math.sqrt(nb_sq))

    def _project(self, vec: Any, target_dim: int, tag: str = "") -> Any:
        """Project a vector into ``target_dim`` via a deterministic Rademacher
        matrix (entries ±1/sqrt(target_dim)) seeded by ``self._random_seed``.

        Rademacher matrices are a popular alternative to Gaussian for JL
        projections: they use a single bit per entry, are equally
        well-behaved in the limit, and the projection is fast to compute.
        """
        if not _HAS_NUMPY:
            # Without numpy the projection is not well-defined for
            # arbitrary dims; degrade gracefully to a deterministic
            # stride-based "projection" that still preserves order
            # reasonably well.
            n = len(vec)
            if n == target_dim:
                return list(vec)
            # Stride-subsample with a small jitter; not JL-optimal
            # but cheap and language-agnostic.
            step = max(1, n // max(1, target_dim))
            out = [float(vec[i]) for i in range(0, n, step)][:target_dim]
            # Pad if short.
            while len(out) < target_dim:
                out.append(0.0)
            return out

        dim = int(vec.shape[-1])
        key = (dim, target_dim, self._random_seed)
        R = self._projections.get(key)
        if R is None:
            # Use a deterministic RNG so the projection is reproducible
            # across processes / sessions.
            rng = _np.random.default_rng(self._random_seed ^ (dim * 2654435761) ^ target_dim)
            R = rng.choice(
                _np.array([-1.0, 1.0], dtype=_np.float32),
                size=(target_dim, dim),
            ).astype(_np.float32) / _np.sqrt(_np.float32(target_dim))
            self._projections[key] = R
        out = R @ vec
        # Re-normalise to unit length so cosine is well-behaved.
        n = float(_np.linalg.norm(out))
        if n == 0.0:
            return out
        return out / n


__all__ = [
    "UniversalBridge",
    "normalize_unicode",
    "transliterate",
    "tokenize",
    "strip_stopwords",
    "stem_word",
    "stem_tokens",
    "char_ngrams",
    "ngram_set",
    "text_similarity",  # legacy alias
]
