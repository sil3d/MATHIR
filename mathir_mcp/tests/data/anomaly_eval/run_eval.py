"""Honest AUC-ROC evaluation of MahalanobisDetector on realistic text.

Unlike the deprecated v7 benchmark (synthetic Gaussian clusters separated
by 6 standard deviations — a trivially easy setup), this uses real
sentence-transformer embeddings of realistic "normal" memory content,
known prompt-injection patterns, and benign-but-unusual text (to check
the detector doesn't fire on harmless outliers).

This is NOT part of the pytest suite (consistent with this repo's existing
tests, none of which load a real embedding model) — run it manually:

    cd mathir_mcp
    python tests/data/anomaly_eval/run_eval.py

Reports the real AUC-ROC. No target score is asserted — the point is an
honest number, not a number to hit.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent.parent.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

try:
    from mathir_lib.mathir_anomaly import MahalanobisDetector
except ImportError:
    from mathir_anomaly import MahalanobisDetector  # type: ignore[no-redef]


def main() -> None:
    from sentence_transformers import SentenceTransformer

    corpus = json.loads((_HERE / "corpus.json").read_text())
    normal_texts = corpus["normal"]
    injection_texts = corpus["injection"]
    benign_outlier_texts = corpus["benign_outlier"]

    print(f"Loading embedder...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    dim = model.get_sentence_embedding_dimension()
    print(f"  dim={dim}")

    # Split normal texts: first 70% to build the baseline, remaining 30%
    # plus all injection/benign-outlier texts to test detection.
    split = int(len(normal_texts) * 0.7)
    train_normal = normal_texts[:split]
    test_normal = normal_texts[split:]

    print(f"Train (baseline): {len(train_normal)} normal texts")
    print(f"Test: {len(test_normal)} normal, {len(injection_texts)} injection, "
          f"{len(benign_outlier_texts)} benign-outlier")

    # NOTE: threshold=45.0 (not the brief's 2.0) — Tasks 3-4 found that 2.0
    # is drastically miscalibrated for 384-dim embeddings (expected
    # in-distribution Mahalanobis distance is ~sqrt(384)=19.6). MATHIR's
    # production config uses anomaly_threshold=45.0 after the shrinkage-
    # regularization fix in mathir_anomaly.py.
    detector = MahalanobisDetector(dim=dim, threshold=45.0, warmup_count=len(train_normal))

    train_embs = model.encode(train_normal, convert_to_numpy=True, normalize_embeddings=True)
    for emb in train_embs:
        detector.update(emb.astype(np.float32))

    assert detector.is_warmed_up(), "baseline did not reach warmup_count"

    def score_all(texts):
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [detector.score(e.astype(np.float32)) for e in embs]

    normal_scores = score_all(test_normal)
    injection_scores = score_all(injection_texts)
    benign_outlier_scores = score_all(benign_outlier_texts)

    print(f"\nNormal scores:          mean={np.mean(normal_scores):.3f} "
          f"std={np.std(normal_scores):.3f}")
    print(f"Injection scores:       mean={np.mean(injection_scores):.3f} "
          f"std={np.std(injection_scores):.3f}")
    print(f"Benign-outlier scores:  mean={np.mean(benign_outlier_scores):.3f} "
          f"std={np.std(benign_outlier_scores):.3f}")

    # Primary metric: can the detector tell normal from injection?
    labels = [0] * len(normal_scores) + [1] * len(injection_scores)
    scores = normal_scores + injection_scores
    auc = roc_auc_score(labels, scores)
    print(f"\nAUC-ROC (normal vs. injection): {auc:.4f}")
    print("(0.5 = random, 1.0 = perfect separation)")

    # Secondary check: false-positive behavior on benign outliers.
    threshold = 45.0
    false_positive_rate = sum(1 for s in benign_outlier_scores if s > threshold) / len(benign_outlier_scores)
    print(f"\nFalse-positive rate on benign outliers (threshold={threshold}): "
          f"{false_positive_rate:.0%}")

    print("\nThis is the real, reported number for this corpus and embedder.")
    print("Do not substitute a different (e.g. synthetic, trivially-separable)")
    print("benchmark's result when describing this feature elsewhere.")


if __name__ == "__main__":
    main()
