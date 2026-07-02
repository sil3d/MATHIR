"""Empirical precision/recall test for VecMemory.find_duplicates()'s cosine
threshold, using the REAL current default embedder (e5-small) -- the last
untested lifecycle mechanism this session (promotion and decay were fixed
and tested; consolidation's duplicate-detection quality was never measured
with real embeddings, only exercised with random synthetic vectors in any
prior test coverage).

Constructs known TRUE duplicate pairs (the same fact paraphrased) and known
FALSE duplicate pairs (same topic, different fact -- the classic near-miss
that a naive similarity threshold conflates) and measures whether
threshold=0.95 (MATHIR's default) actually separates them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

try:
    from mathir_lib.mathir_vec import VecMemory
except ImportError:
    from mathir_vec import VecMemory  # type: ignore

# Pairs that ARE genuine near-duplicates: same fact, paraphrased.
TRUE_DUPLICATE_PAIRS = [
    ("The MATHIR daemon runs on port 7338 and uses a Flask+Waitress HTTP server.",
     "MATHIR's daemon listens on port 7338, serving HTTP via Flask and Waitress."),
    ("touch_recall increments recall_count and boosts memory stability by 0.1.",
     "Calling touch_recall bumps up recall_count and adds 0.1 to the memory's stability score."),
    ("The default embedding model is intfloat/multilingual-e5-small, 384 dimensions.",
     "MATHIR now defaults to the intfloat/multilingual-e5-small embedder with a 384d vector size."),
]

# Pairs that are NOT duplicates: same topic/domain, but a DIFFERENT fact --
# the case a naive similarity threshold is most likely to wrongly conflate.
FALSE_DUPLICATE_PAIRS = [
    ("The MATHIR daemon runs on port 7338 and uses a Flask+Waitress HTTP server.",
     "The MATHIR daemon's embedder falls back to CPU if the CUDA device fails to load."),
    ("touch_recall increments recall_count and boosts memory stability by 0.1.",
     "decay_all reduces memory stability by 5% per 30 days of no recall, archiving it below 0.05."),
    ("The default embedding model is intfloat/multilingual-e5-small, 384 dimensions.",
     "The previous default embedding model was paraphrase-multilingual-MiniLM-L12-v2, also 384 dimensions."),
]


def _get_real_embedder():
    """Load the actual current default embedder the same way the daemon does."""
    sys.path.insert(0, str(_MCP_ROOT / "mathir_lib"))
    from mathir_mcp_server import get_embedder  # type: ignore
    return get_embedder()


@pytest.fixture(scope="module")
def embedder():
    return _get_real_embedder()


def test_find_duplicates_default_threshold_precision_recall_on_real_pairs(tmp_path, embedder):
    """The core empirical question: at MATHIR's default threshold=0.95, does
    find_duplicates() correctly flag TRUE duplicate pairs as candidates
    while correctly NOT flagging FALSE duplicate (same-topic, different-fact)
    pairs? Reports the real precision/recall rather than assuming either.
    """
    db = tmp_path / "consolidate_quality.db"
    vm = VecMemory(db, embedding_dim=384)

    id_to_pair_type = {}
    all_texts = []
    for i, (a, b) in enumerate(TRUE_DUPLICATE_PAIRS):
        all_texts.append((f"true_{i}_a", a, "true"))
        all_texts.append((f"true_{i}_b", b, "true"))
    for i, (a, b) in enumerate(FALSE_DUPLICATE_PAIRS):
        all_texts.append((f"false_{i}_a", a, "false"))
        all_texts.append((f"false_{i}_b", b, "false"))

    for mem_id, text, kind in all_texts:
        emb = embedder.encode(text, convert_to_numpy=True)
        vm.store(mem_id, np.asarray(emb, dtype=np.float32),
                 {"content": text, "agent": "t", "block_type": "episodic",
                  "label": "", "priority": 5})
        id_to_pair_type[mem_id] = kind

    candidates = vm.find_duplicates(threshold=0.95, limit=1000)
    flagged_pairs = {frozenset((c["memory_id_a"], c["memory_id_b"])) for c in candidates}

    true_pairs_expected = [frozenset((f"true_{i}_a", f"true_{i}_b")) for i in range(len(TRUE_DUPLICATE_PAIRS))]
    false_pairs_expected = [frozenset((f"false_{i}_a", f"false_{i}_b")) for i in range(len(FALSE_DUPLICATE_PAIRS))]

    true_positives = sum(1 for p in true_pairs_expected if p in flagged_pairs)
    false_positives = sum(1 for p in false_pairs_expected if p in flagged_pairs)

    recall = true_positives / len(true_pairs_expected)
    false_positive_rate = false_positives / len(false_pairs_expected)

    print(f"\n[consolidate quality] threshold=0.95: "
          f"recall={recall:.2f} ({true_positives}/{len(true_pairs_expected)}), "
          f"false_positive_rate={false_positive_rate:.2f} ({false_positives}/{len(false_pairs_expected)})")

    # REAL MEASURED RESULT (2026-07-02, real e5-small embeddings, n=3+3 --
    # small sample, but a concrete, reproducible finding, not a guess):
    # threshold=0.95 gets 2/3 true duplicates right (0.9762, 0.9756) but
    # MISSES one real paraphrase pair (0.9116 -- below threshold) AND
    # WOULD WRONGLY MERGE one genuinely-different-fact pair (0.9554 --
    # "the default embedding model IS e5-small" vs "the PREVIOUS default
    # WAS the old model": structurally near-identical sentences describing
    # opposite facts about past vs. present state). This demonstrates
    # threshold=0.95 is not perfectly calibrated for this embedder: it can
    # both miss real duplicates and merge genuinely distinct facts when
    # they share a similar sentence structure. This assertion locks in the
    # measured behavior so any future embedder/threshold change is a
    # deliberate, visible decision, not a silent regression.
    assert recall == pytest.approx(2 / 3), (
        f"expected 2/3 recall at threshold=0.95 with e5-small on this fixed "
        f"pair set, got {recall} -- if a future embedder change moves this, "
        f"that's a real finding worth investigating, not a stale assertion"
    )
    assert false_positive_rate == pytest.approx(1 / 3), (
        f"expected 1/3 false-positive rate at threshold=0.95 on this fixed "
        f"pair set, got {false_positive_rate}"
    )
