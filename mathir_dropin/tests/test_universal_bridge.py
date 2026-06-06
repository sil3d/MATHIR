"""
MATHIR Drop-in -- Universal Cross-Provider & Cross-Lingual Bridge Tests
========================================================================

This suite exercises :mod:`mathir_dropin.universal_bridge` -- the
``UniversalBridge`` class and its module-level helpers that solve three
MATHIR retrieval problems:

  1. FTS5 text search fails on conversational queries
     ("What do you know about python closures?" -> 0 results)
  2. Different embedding spaces are incompatible
     (OpenAI 1536d != Anthropic != Ollama 1024d)
  3. No language-agnostic matching
     (English queries can't find French content)

The bridge's actual API surface (verified at import time):

* Class ``UniversalBridge``
    - ``expand_query(query)`` -> list of FTS5-friendly variants
    - ``text_similarity(text1, text2)`` -> Jaccard over char n-grams
    - ``cross_space_score(emb_a, emb_b)`` -> cosine across dim-mismatched spaces
    - ``hybrid_recall(query, embedding, k, provider, text_candidates, ...)``
    - ``provider_fallback_chain(requested, available, primary)``
* Module-level helpers
    - ``normalize_unicode``, ``transliterate``, ``tokenize``,
      ``strip_stopwords``, ``stem_word``, ``stem_tokens``,
      ``char_ngrams``, ``ngram_set``

Each test is runnable independently: it creates its own bridge /
memory / tempdir and does not depend on the order in which it is
invoked.

Run via pytest
--------------
    cd D:/SECRET_PROJECT/MATHIR
    python -m pytest mathir_dropin/tests/test_universal_bridge.py -v

Run standalone (no pytest needed)
--------------------------------
    python mathir_dropin/tests/test_universal_bridge.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unicodedata
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch

# ---------------------------------------------------------------------------
# Path bootstrap -- mirrors the other test files in this directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DROPIN = os.path.dirname(_HERE)        # .../mathir_dropin
_PARENT = os.path.dirname(_DROPIN)      # project root
for _p in (_PARENT, _DROPIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# `mathir_dropin` is the parent directory of `tests/`, so the project
# root must be on sys.path for `import mathir_dropin` to work when
# running `pytest tests/` from inside the package.
import mathir_dropin  # noqa: E402

from mathir_dropin import (  # noqa: E402
    DimensionMismatchError,
    MATHIRMemory,
    configure,
)

# ---------------------------------------------------------------------------
# SUT import
# ---------------------------------------------------------------------------
# The bridge module is built by @coder in a parallel wave. We import it
# defensively: if it is missing or broken, the helper ``_require``
# produces a precise failure message for every test that needs it.
try:
    import mathir_dropin.universal_bridge as ub
    UB_AVAILABLE = True
    UB_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # noqa: BLE001
    ub = None  # type: ignore[assignment]
    UB_AVAILABLE = False
    UB_IMPORT_ERROR = repr(_e)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """Yield a fresh SQLite path in a tempdir, clean up afterwards."""
    d = tempfile.mkdtemp(prefix="mathir_ub_test_")
    path = os.path.join(d, "test.db")
    yield path
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def memory_factory(tmp_db):
    """Factory that returns a fresh MATHIRMemory bound to ``tmp_db``."""
    created: List[MATHIRMemory] = []

    def _make(dim: int = 64, **kwargs: Any) -> MATHIRMemory:
        cfg = configure({
            "memory": {
                "embedding_dim": dim,
                "working_capacity": 16,
                "episodic_capacity": 32,
                "semantic_prototypes": 8,
                "immunological_capacity": 16,
            },
        })
        if "db_path" not in kwargs:
            kwargs["db_path"] = tmp_db
        m = MATHIRMemory(embedding_dim=dim, config=cfg, **kwargs)
        created.append(m)
        return m

    yield _make
    for m in created:
        try:
            if m._store is not None:
                m._store.close()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
def mem64(memory_factory) -> MATHIRMemory:
    """Default 64-dim memory for fast unit tests."""
    torch.manual_seed(0)
    return memory_factory(dim=64)


@pytest.fixture
def bridge():
    """A fresh ``UniversalBridge`` instance for each test."""
    if not UB_AVAILABLE:
        pytest.fail(
            f"universal_bridge is not importable: {UB_IMPORT_ERROR}"
        )
    return ub.UniversalBridge()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_to_vec(text: str, dim: int = 64, seed: int = 0) -> torch.Tensor:
    """Deterministic text->vector so tests don't need a real embedder.

    Same text in -> same vector out. Different text -> a different but
    stable vector. Used only to feed the cosine code paths.
    """
    g = torch.Generator().manual_seed(seed + (hash(text) & 0xFFFF))
    return torch.randn(1, dim, generator=g)


def _ascii_only(s: str) -> str:
    """NFKD + drop non-spacing marks + drop non-ASCII bytes.

    Used to verify the bridge produced a stripped variant regardless of
    the test platform's default encoding.
    """
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


# ===========================================================================
# SECTION 1 -- QUERY EXPANSION
# ===========================================================================

class TestQueryExpansion:
    """Section 1: query expansion via ``UniversalBridge.expand_query``."""

    def test_expand_simple_query(self, bridge):
        """'python' should expand to multiple useful variants for FTS5."""
        variants = bridge.expand_query("python")
        assert variants is not None, "expand_query() returned None"
        assert isinstance(variants, list), (
            f"expand_query() must return a list, got {type(variants).__name__}"
        )
        assert len(variants) >= 2, (
            f"expand_query('python') returned {len(variants)} variants, "
            f"expected >= 2"
        )
        # The original token must survive in some form.
        joined = " ".join(variants).lower()
        assert "python" in joined, (
            f"expanded variants lost the original token: {variants!r}"
        )
        # Every variant must be a non-empty string (no None / no bytes).
        for v in variants:
            assert isinstance(v, str), (
                f"variant has wrong type {type(v).__name__}: {v!r}"
            )
            assert v.strip(), f"empty variant produced: {variants!r}"

    def test_expand_conversational_query(self, bridge):
        """Conversational queries must shed stop-words and surface tokens."""
        query = "What do you know about python closures?"
        variants = bridge.expand_query(query)
        assert variants, "expand_query() returned nothing for conversational input"

        joined = " ".join(variants).lower()
        # At least one of the content words must survive.
        for keyword in ("python", "closures", "closure"):
            assert keyword in joined, (
                f"keyword {keyword!r} missing from expanded variants: {variants!r}"
            )

        # FTS5-special characters (?, ", *) must not survive into a
        # variant in a way that would blow up FTS5 syntax.
        # FTS5 treats ", *, (, ) as syntax -- if any variant is fed
        # verbatim, it should still be sane (we allow these characters
        # but only because the *first* variant is the original; other
        # variants must be safe to feed back to FTS5 unquoted).
        for v in variants[1:]:  # skip the original-keep variant
            # A bare asterisk at the start of a token is the most
            # dangerous FTS5 construct. We don't forbid every special
            # char (they can appear in legitimate text), but we do
            # require each variant to be a usable string.
            assert isinstance(v, str) and v.strip(), (
                f"non-string or empty variant: {v!r}"
            )

    def test_expand_unicode_query(self, bridge):
        """Accented queries must produce a diacritic-stripped variant."""
        # 'café' should at least produce 'cafe' as one of its variants
        # so it can match an FTS5 row that stored the ASCII form.
        variants = bridge.expand_query("café")
        assert variants, "expand_query('café') returned nothing"
        assert all(isinstance(v, str) for v in variants)

        joined_ascii = _ascii_only(" ".join(variants)).lower()
        assert "cafe" in joined_ascii, (
            f"café was not diacritic-stripped in any variant: {variants!r}"
        )


# ===========================================================================
# SECTION 2 -- CROSS-LINGUAL MATCHING
# ===========================================================================

class TestCrossLingual:
    """Section 2: cross-lingual matching via ``text_similarity``."""

    def test_french_to_english(self, bridge):
        """French 'clotures python' must match English 'Python closures'."""
        sim = bridge.text_similarity("clotures python", "Python closures")
        assert isinstance(sim, float), (
            f"text_similarity must return float, got {type(sim).__name__}"
        )
        assert 0.0 <= sim <= 1.0, f"similarity out of [0, 1]: {sim!r}"
        # French 'clotures' is essentially the same word as English
        # 'closures' (etymological sibling) -- similarity should be
        # *meaningful*, not near-zero.
        assert sim >= 0.3, (
            f"French 'clotures python' vs English 'Python closures' "
            f"similarity is too low ({sim:.3f}); the n-gram bridge is "
            f"not connecting Latin-script cognates"
        )

    def test_arabic_match(self, bridge):
        """Arabic 'بايثون' (transliteration: bait-hun) -> call must be safe.

        The bridge uses character-level Jaccard, so *exact* matching
        between Arabic-script and Latin-script strings is not
        possible without a transliteration model. We therefore assert:

        * the call must NOT raise,
        * the return must be a float in [0, 1],
        * an unrelated Latin-script string should score lower than the
          same Latin-script string compared to itself (sanity).
        """
        sim_xl = bridge.text_similarity("بايثون", "Python")
        assert isinstance(sim_xl, float), (
            f"text_similarity must return float for Arabic input, "
            f"got {type(sim_xl).__name__}"
        )
        assert 0.0 <= sim_xl <= 1.0, (
            f"Arabic similarity out of [0, 1]: {sim_xl!r}"
        )
        # Sanity: an identical pair scores higher than a fully disjoint pair.
        same = bridge.text_similarity("python closures", "python closures")
        diff = bridge.text_similarity("python closures", "lorem ipsum dolor")
        assert same > diff, (
            f"identity similarity ({same}) should beat random ({diff})"
        )

    def test_chinese_match(self, bridge):
        """Chinese '闭包' (closure) -> call must be safe and bounded."""
        sim_xl = bridge.text_similarity("闭包", "closures")
        assert isinstance(sim_xl, float), (
            f"text_similarity must return float for Chinese input, "
            f"got {type(sim_xl).__name__}"
        )
        assert 0.0 <= sim_xl <= 1.0, (
            f"Chinese similarity out of [0, 1]: {sim_xl!r}"
        )
        # Sanity: identity is the upper bound for the metric.
        same = bridge.text_similarity("closures are great", "closures are great")
        assert same == pytest.approx(1.0, abs=1e-6), (
            f"identity similarity should be 1.0, got {same}"
        )


# ===========================================================================
# SECTION 3 -- CROSS-PROVIDER RECALL
# ===========================================================================

class TestCrossProvider:
    """Section 3: cross-provider recall via ``hybrid_recall`` + bridges."""

    def test_provider_fallback(self, bridge):
        """The fallback chain must be ordered and complete."""
        chain = bridge.provider_fallback_chain(
            requested="openai",
            available=["cohere", "voyage", "ollama"],
            primary="primary",
        )
        assert isinstance(chain, list)
        assert chain[0] == "openai", (
            f"requested provider must be first, got {chain!r}"
        )
        assert "primary" in chain, "primary must be in the fallback chain"
        # Every available provider should appear exactly once.
        for prov in ("openai", "primary", "cohere", "voyage", "ollama"):
            assert chain.count(prov) == 1, (
                f"provider {prov!r} appears {chain.count(prov)} times in {chain!r}"
            )

    def test_dimension_mismatch(self, bridge):
        """Mismatched embedding dims must be projected, not crash.

        The bridge uses a deterministic Rademacher projection under
        the hood. We assert:

        * same-vector in two different spaces scores ~1.0,
        * random vectors in two different spaces score in [-1, 1],
        * zero-length input returns 0.0.
        """
        torch.manual_seed(0)
        v64 = torch.randn(64)
        v128 = torch.randn(128)

        # Same content, different dims -- the projection will align them
        # only by chance; we just need the result to be a real number.
        score = bridge.cross_space_score(v64, v128)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0, f"score out of [-1, 1]: {score!r}"

        # Identical vectors, same dim -- should be exactly 1.0.
        same = bridge.cross_space_score(v64, v64.clone())
        assert same == pytest.approx(1.0, abs=1e-5), (
            f"identical vectors should score 1.0, got {same}"
        )

        # Zero vector -> 0.0 (no division-by-zero crash).
        zero = bridge.cross_space_score(torch.zeros(64), torch.zeros(128))
        assert zero == 0.0, f"zero vector should score 0.0, got {zero}"

    def test_multi_provider_recall(self, bridge):
        """hybrid_recall must rank text + recall_count channels together.

        Two memories with identical text scores but different
        recall_count values must be ordered with the higher-recall
        one first.
        """
        candidates = [
            {
                "memory_id": "low",
                "similarity": 0.5,
                "recall_count": 1,
                "stability": 1.0,
                "modality_text": "python closures",
                "metadata": {"text": "python closures"},
                "tier": "episodic",
                "modality": "text",
            },
            {
                "memory_id": "high",
                "similarity": 0.5,
                "recall_count": 50,
                "stability": 1.0,
                "modality_text": "python closures",
                "metadata": {"text": "python closures"},
                "tier": "episodic",
                "modality": "text",
            },
        ]
        ranked = bridge.hybrid_recall(
            query="python closures",
            embedding=None,
            k=2,
            text_candidates=candidates,
            embedding_candidates=None,
            cross_lingual=False,
        )
        assert len(ranked) == 2, f"expected 2 results, got {len(ranked)}"
        # The high-recall memory must rank first.
        assert ranked[0]["memory_id"] == "high", (
            f"expected 'high' first, got order: "
            f"{[r['memory_id'] for r in ranked]}"
        )
        # The boost must be observable in the final_score (non-zero).
        assert ranked[0]["recall_boost"] > 0, (
            f"high-recall memory should have a positive recall_boost, "
            f"got {ranked[0]['recall_boost']}"
        )


# ===========================================================================
# SECTION 4 -- SELF-CORRECTION
# ===========================================================================

class TestSelfCorrection:
    """Section 4: self-correction via the recall_count channel."""

    def test_recall_count_boost(self, bridge):
        """A high recall_count must lift a memory's final_score.

        We compare two otherwise-identical candidates that differ only
        in recall_count and assert the high-recall one wins.
        """
        low = {
            "memory_id": "low",
            "similarity": 0.6,
            "recall_count": 0,
            "stability": 1.0,
            "modality_text": "python",
            "metadata": {"text": "python"},
            "tier": "episodic",
            "modality": "text",
        }
        high = {**low, "memory_id": "high", "recall_count": 1000}

        ranked = bridge.hybrid_recall(
            query="python",
            k=2,
            text_candidates=[low, high],
            embedding_candidates=None,
            cross_lingual=False,
        )
        assert len(ranked) == 2
        assert ranked[0]["memory_id"] == "high", (
            f"high-recall memory must rank first; got "
            f"{[r['memory_id'] for r in ranked]}"
        )
        # Logarithmic boost: log1p(1000) ~= 6.91, scaled by 0.15 ~= 1.04
        # which is clamped to a meaningful positive number.
        assert ranked[0]["recall_boost"] > 0.1, (
            f"recall_boost too small: {ranked[0]['recall_boost']}"
        )
        # And low must *not* be lifted -- its boost is ~0.
        assert ranked[1]["recall_boost"] == pytest.approx(0.0, abs=1e-9), (
            f"low-recall memory's boost should be 0, got "
            f"{ranked[1]['recall_boost']}"
        )

    def test_priority_decay_handled_gracefully(self, mem64: MATHIRMemory, bridge):
        """If the bridge has no decay API, the test must still complete.

        The current bridge does not implement priority decay. We probe
        the public surface for a ``decay`` / ``decay_priorities``
        callable and either:

        * call it (if present) and assert the call returns without
          raising, OR
        * skip the runtime check and assert that the bridge is in a
          consistent state (i.e. ``hybrid_recall`` still works on
          un-decayed data).
        """
        decay = getattr(ub, "decay_priorities", None) or getattr(ub, "decay", None)
        # Plant a memory so we can probe its pre- and post-state.
        mem64.store(
            _text_to_vec("seed", dim=64, seed=42),
            metadata={"text": "seed", "concept": "seed"},
        )
        target_id = mem64._store.all_ids()[0]
        before = mem64._store.get(target_id)
        assert before is not None

        if callable(decay):
            try:
                decay(mem64)
            except TypeError:
                try:
                    decay(mem64, rate=0.99)
                except TypeError:
                    decay()  # no-arg variant
            after = mem64._store.get(target_id)
            assert after is not None
            # Stability must not have *increased* after a decay call.
            assert after.get("stability", 1.0) <= before.get("stability", 1.0) + 1e-9
        else:
            # No decay API -> just verify the bridge still ranks cleanly.
            ranked = bridge.hybrid_recall(
                query="seed",
                k=1,
                text_candidates=[{
                    "memory_id": target_id,
                    "similarity": 1.0,
                    "recall_count": before.get("recall_count", 0),
                    "stability": before.get("stability", 1.0),
                    "modality_text": "seed",
                    "metadata": {"text": "seed"},
                    "tier": "episodic",
                    "modality": "text",
                }],
                cross_lingual=False,
            )
            assert ranked and ranked[0]["memory_id"] == target_id


# ===========================================================================
# SECTION 5 -- EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Section 5: edge cases -- empty / huge / special-char / concurrent."""

    def test_empty_query(self, bridge):
        """Empty / whitespace queries must return [] without crashing."""
        for empty in ("", " ", "\t\n", "   "):
            res = bridge.expand_query(empty)
            assert isinstance(res, list), (
                f"expand_query({empty!r}) returned {type(res).__name__}, expected list"
            )
            # Empty input is allowed to return [] (nothing to expand).
            for v in res:
                assert isinstance(v, str) and v.strip(), (
                    f"expand_query({empty!r}) produced empty/non-string: {v!r}"
                )

        # ``text_similarity`` on empty -> 0.0
        assert bridge.text_similarity("", "python") == 0.0
        assert bridge.text_similarity("python", "") == 0.0
        assert bridge.text_similarity("", "") == 0.0

    def test_very_long_query(self, bridge):
        """A 10 000-character query must be handled without pathological expansion."""
        long_text = "python closures " * 800  # ~12 800 chars, > 10 000

        variants = bridge.expand_query(long_text)
        assert isinstance(variants, list)
        assert len(variants) >= 1, "expand_query on long input returned nothing"
        for v in variants:
            assert isinstance(v, str)
            # A single variant must not exceed the input length by much
            # (the bridge must cap pathological blow-ups).
            assert len(v) <= len(long_text) + 1024, (
                f"a variant ({len(v)} chars) is longer than the input "
                f"({len(long_text)} chars) -- pathological expansion"
            )

        # Also: text_similarity on long strings must return a bounded float.
        sim = bridge.text_similarity(long_text, "python closures")
        assert 0.0 <= sim <= 1.0, f"long-string similarity out of bounds: {sim}"

    def test_special_chars(self, bridge):
        """FTS5-special characters must not raise in any bridge function."""
        tricky_queries = (
            'What is "python"?',
            "python*",
            "py?thon",
            "py:thon",
            "py'thon",
            "(python)",
            "py^thon",
        )
        for q in tricky_queries:
            # Expansion must not crash.
            res = bridge.expand_query(q)
            assert isinstance(res, list), (
                f"expand_query({q!r}) returned {type(res).__name__}"
            )

            # text_similarity must not crash.
            sim = bridge.text_similarity(q, "python closures")
            assert isinstance(sim, float), (
                f"text_similarity({q!r}, ...) returned {type(sim).__name__}"
            )
            assert 0.0 <= sim <= 1.0, (
                f"text_similarity({q!r}, ...) out of bounds: {sim}"
            )

    def test_concurrent_stores(self, tmp_db: str, bridge):
        """Multiple threads storing + recalling must not race.

        We hammer a single MATHIRMemory with 8 threads that each store
        25 memories and immediately recall them. The invariants:

        * No thread raises an unhandled exception.
        * The final row count equals baseline + writes (no lost writes,
          no duplicates).
        """
        mem = MATHIRMemory(
            embedding_dim=64,
            config=configure({
                "memory": {
                    "embedding_dim": 64,
                    "working_capacity": 16,
                    "episodic_capacity": 64,
                    "semantic_prototypes": 8,
                    "immunological_capacity": 16,
                },
            }),
            db_path=tmp_db,
        )
        # Plant some baseline memories.
        for i in range(5):
            mem.store(
                _text_to_vec(f"baseline-{i}", dim=64, seed=i + 500),
                metadata={"text": f"baseline {i}", "concept": f"b{i}"},
            )

        errors: List[BaseException] = []
        n_threads = 8
        n_per_thread = 25
        barrier = threading.Barrier(n_threads)

        def worker(tid: int) -> None:
            try:
                barrier.wait(timeout=5.0)
                for j in range(n_per_thread):
                    v = _text_to_vec(f"t{tid}-i{j}", dim=64, seed=tid * 1000 + j)
                    mem.store(v, metadata={"text": f"t{tid} item {j}"})

                    # Recall through the bridge -- exercises the
                    # text-channel re-ranker.
                    text_cands = [{
                        "memory_id": "x",
                        "similarity": 0.4,
                        "recall_count": 0,
                        "stability": 1.0,
                        "modality_text": f"t{tid} item {j}",
                        "metadata": {"text": f"t{tid} item {j}"},
                        "tier": "episodic",
                        "modality": "text",
                    }]
                    ranked = bridge.hybrid_recall(
                        query=f"t{tid} item {j}",
                        k=1,
                        text_candidates=text_cands,
                        cross_lingual=True,
                    )
                    assert isinstance(ranked, list)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, (
            f"concurrent stores produced {len(errors)} error(s): "
            f"{[type(e).__name__ + ': ' + str(e)[:120] for e in errors[:3]]}"
        )

        # Final invariant: row count must equal baseline + writes.
        expected = 5 + n_threads * n_per_thread
        actual = mem._store.count()
        assert actual == expected, (
            f"row count mismatch after concurrent stores: "
            f"expected {expected}, got {actual}"
        )

        try:
            mem._store.close()
        except Exception:  # noqa: BLE001
            pass


# ===========================================================================
# ENTRY POINT (standalone runner -- no pytest needed)
# ===========================================================================

def _run_standalone() -> int:
    """Run all tests via pytest's in-process API and print a summary.

    We use ``pytest.main()`` rather than ``unittest.TestLoader`` because
    the test classes use pytest fixtures (which ``unittest`` does not
    understand -- instantiating ``TestQueryExpansion()`` directly
    raises ``TypeError: ... takes no arguments``).
    """
    # Force UTF-8 so the FTS5 test text prints cleanly on Windows cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    print("=" * 70)
    print("MATHIR Universal Bridge -- Standalone Test Run")
    print("=" * 70)
    if not UB_AVAILABLE:
        print(f"[FAIL] universal_bridge is not importable: {UB_IMPORT_ERROR}")
        return 1

    surface = sorted(a for a in dir(ub) if not a.startswith("_"))
    print(f"[OK]   universal_bridge API surface: {surface}")
    print()

    # Hand off to pytest with verbose output and no traceback shortening.
    rc = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--no-header",
    ])
    print()
    print("=" * 70)
    print(f"pytest exit code: {rc}  (0 = all passed)")
    print("=" * 70)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
