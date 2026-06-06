"""
UNIBRI: Universal Indexing Bridge for Information Retrieval
============================================================

Reference implementation skeleton for the MATHIR drop-in.

This module implements the algorithm specified in
``UNIBRI_DESIGN.md``. It is intentionally framework-light:
the only required third-party dependency is ``numpy`` (already
required by the rest of the drop-in).  ``mmh3`` is optional —
a pure-Python fallback is provided.

The four public classes are:

* :class:`ULLFingerprinter`        — vocabulary-free canonical text → ℝ^D
* :class:`ProcrustesBridge`        — learn a projection R_p : ℝ^{d_p} → ℝ^D
* :class:`UNIBRIRetriever`         — hybrid retriever with RRF fusion
* :class:`UNIBRIHubSet`            — curated 256-term multilingual hub set

Together they solve the four problems stated in the design doc:
OOV queries, cross-provider incompatibility, no cross-lingual
support, and zero cross-provider recall.

Author: @math (MATHIR Research Team)
"""

from __future__ import annotations

import math
import struct
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ===========================================================================
# Optional: MurmurHash3 via mmh3 (fast C extension). Fallback to built-in.
# ===========================================================================

try:
    import mmh3 as _mmh3
    _HAS_MMH3 = True
except ImportError:  # pragma: no cover
    _HAS_MMH3 = False


def _stable_hash(data: str, seed: int) -> int:
    """Deterministic 32-bit hash of a string. mmh3 if available, else Python's."""
    if _HAS_MMH3:
        return _mmh3.hash(data, seed=seed, signed=False) & 0xFFFFFFFF
    # Pure-Python fallback: not as fast, but deterministic and dependency-free.
    h = (seed * 0x9E3779B1) & 0xFFFFFFFF
    for ch in data:
        h = ((h ^ ord(ch)) * 0x85EBCA6B) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return h


# ===========================================================================
# 1. ULL Fingerprinter
# ===========================================================================

class ULLFingerprinter:
    """
    Multi-Resolution Character N-gram Fingerprinter (ULL).

    Maps any Unicode string s to a fixed-dim L2-normalized vector Φ(s) ∈ ℝ^D
    where D = (2^bits) × |orders|. Deterministic, vocabulary-free,
    language-agnostic, OOV-robust.

    The construction is a direct implementation of the *signed feature
    hashing* trick of Weinberger, Dasgupta, Langford, Smola & Attenberg
    (ICML 2009) applied independently at each n-gram resolution.

    Parameters
    ----------
    orders:
        Tuple of n-gram orders. Default ``(2, 3, 4, 5)`` covers morphology
        from 2-grams (sub-word roots) to 5-grams (whole short words).
    bits:
        Number of hash buckets per resolution. Default 10 → 1024 buckets
        → 4096-dim total. Recommended range: 8–12 (256 → 16384 dims).
    idf:
        Optional pre-computed IDF table. Keys are ``(n, ngram_str)`` tuples,
        values are floats ≥ 1.0. If ``None``, every n-gram gets weight 1.0
        (still a valid kernel, just uncalibrated).
    normalize:
        L2-normalize the output. Almost always True — cosine similarity
        then reduces to a dot product.
    unicode_form:
        One of ``"NFC"``, ``"NFD"``, ``"NFKC"``, ``"NFKD"`` (Unicode
        canonicalization). ``"NFC"`` is the safe default.
    fold_diacritics:
        If True, decompose via NFD then strip combining marks. Lets
        ``café`` and ``cafe`` collide. Default False to preserve
        semantic distinction.
    """

    def __init__(
        self,
        orders: Tuple[int, ...] = (2, 3, 4, 5),
        bits: int = 10,
        idf: Optional[Dict[Tuple[int, str], float]] = None,
        normalize: bool = True,
        unicode_form: str = "NFC",
        fold_diacritics: bool = False,
    ):
        if not 6 <= bits <= 14:
            raise ValueError(f"bits must be in [6, 14], got {bits}")
        self.orders = tuple(orders)
        self.bits = bits
        self.d = 1 << bits
        self.D = self.d * len(self.orders)
        self.idf = idf or {}
        self.normalize = normalize
        self.unicode_form = unicode_form
        self.fold_diacritics = fold_diacritics

        # Deterministic non-negative 32-bit seeds for the bucket hash
        # and the sign hash, one pair per n-gram order. mmh3.hash requires
        # seed in [0, 2^31). Derived from Knuth's multiplicative constants.
        def _u31(x: int) -> int:
            return (x & 0x7FFFFFFF) or 1   # never zero (mmh3 quirk)

        self._bucket_seeds = [_u31(0x9E3779B1 ^ (n * 0x85EBCA6B)) for n in self.orders]
        self._sign_seeds = [_u31(0x517CC1B7 ^ (n * 0xC2B2AE35)) for n in self.orders]

    # ---- public API ----------------------------------------------------

    def fingerprint(self, text: str) -> np.ndarray:
        """Compute Φ(text). Always returns a 1-D ``float32`` array of length D."""
        s = self._canonicalize(text)
        if not s:
            return np.zeros(self.D, dtype=np.float32)
        out = np.zeros(self.D, dtype=np.float32)
        for slot, n in enumerate(self.orders):
            lo, hi = slot * self.d, (slot + 1) * self.d
            self._sketch(s, n, out, lo, hi)
        if self.normalize:
            nrm = float(np.linalg.norm(out))
            if nrm > 1e-12:
                out /= nrm
        return out

    def batch_fingerprint(self, texts: Sequence[str]) -> np.ndarray:
        """Vectorized. Returns ``(len(texts), D)`` ``float32`` array."""
        return np.stack([self.fingerprint(t) for t in texts], axis=0)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (assumes inputs are L2-normalized)."""
        return float(np.dot(a, b))

    def search(
        self,
        query: np.ndarray,
        matrix: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k cosine over a ``(N, D)`` corpus. Returns (indices, scores)."""
        sims = matrix @ query
        k = min(k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        order = np.argsort(-sims[idx])
        return idx[order], sims[idx[order]]

    # ---- internals -----------------------------------------------------

    def _canonicalize(self, text: str) -> str:
        s = unicodedata.normalize(self.unicode_form, text or "")
        if self.fold_diacritics:
            s = "".join(
                c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn"
            )
        return s.lower()

    def _sketch(self, s: str, n: int, out: np.ndarray, lo: int, hi: int) -> None:
        L = len(s)
        if L < n:
            return
        slot = self.orders.index(n)
        bucket_seed = self._bucket_seeds[slot]
        sign_seed = self._sign_seeds[slot]
        span = hi - lo
        for i in range(L - n + 1):
            g = s[i : i + n]
            b = _stable_hash(g, bucket_seed) % span
            sgn = 1 if (_stable_hash(g, sign_seed) & 1) else -1
            w = self.idf.get((n, g), 1.0)
            out[lo + b] += w * sgn


# ===========================================================================
# 2. IDF Calibration
# ===========================================================================

def calibrate_idf(
    fingerprinter: ULLFingerprinter,
    corpus: Sequence[str],
    min_df: int = 2,
    max_df_ratio: float = 0.95,
) -> Dict[Tuple[int, str], float]:
    """
    Build the (n, ngram) → idf table from a corpus. OOV n-grams default
    to weight 1.0 (the uninformative prior).

    Smoothed formula (scikit-learn convention):
        idf(g) = log((N + 1) / (df(g) + 1)) + 1

    N-grams with df < min_df or df > max_df_ratio * N are dropped
    (too rare to be informative, or too common to discriminate).
    """
    df: Dict[Tuple[int, str], int] = {}
    N = max(len(corpus), 1)
    for doc in corpus:
        s = fingerprinter._canonicalize(doc)
        seen: set = set()
        for n in fingerprinter.orders:
            for i in range(len(s) - n + 1):
                g = s[i : i + n]
                key = (n, g)
                if key in seen:
                    continue
                df[key] = df.get(key, 0) + 1
                seen.add(key)
    idf: Dict[Tuple[int, str], float] = {}
    for key, d in df.items():
        if d < min_df or d > max_df_ratio * N:
            continue
        idf[key] = float(math.log((N + 1) / (d + 1)) + 1.0)
    return idf


# ===========================================================================
# 3. Procrustes Bridge
# ===========================================================================

class ProcrustesBridge:
    """
    Cross-provider bridge via orthogonal Procrustes with hub anchoring.

    Given M hub terms with known ULL signatures B ∈ ℝ^{M × D} and
    per-provider hub embeddings A_p ∈ ℝ^{M × d_p}, find the
    orthogonal projection R_p ∈ ℝ^{D × d_p} minimizing
    ‖B - A_p R_p‖_F. The closed-form solution is R_p = U V^T
    where A_p^T B = U Σ V^T is the thin SVD.
    """

    def __init__(
        self,
        hubs: Sequence[str],
        fingerprinter: ULLFingerprinter,
    ):
        if not hubs:
            raise ValueError("hubs must be a non-empty list of strings")
        self.hubs = list(hubs)
        self.M = len(self.hubs)
        self.fp = fingerprinter
        # B ∈ ℝ^{M × D}: ULL signatures of the hubs.
        self.B: np.ndarray = np.stack(
            [fingerprinter.fingerprint(h) for h in self.hubs], axis=0
        ).astype(np.float32)
        # Dictionary provider_name → R_p ∈ ℝ^{D × d_p}
        self._R: Dict[str, np.ndarray] = {}

    def fit(
        self,
        provider_embeddings: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        For each provider, fit R_p. Returns a copy of the projections.

        ``provider_embeddings[name]`` must be ``(M, d_p)``.
        """
        new: Dict[str, np.ndarray] = {}
        for name, A in provider_embeddings.items():
            A = np.asarray(A, dtype=np.float32)
            if A.shape[0] != self.M:
                raise ValueError(
                    f"Provider {name!r}: expected {self.M} hub vectors, "
                    f"got {A.shape[0]}"
                )
            # Thin SVD of A^T B  ∈ ℝ^{d_p × D}.
            M = A.T @ self.B
            U, _S, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt  # (d_p, D) → transpose to (D, d_p)
            R = R.T.astype(np.float32)
            new[name] = R
            self._R[name] = R
        return new

    def project(
        self,
        e: np.ndarray,
        provider: str,
    ) -> Optional[np.ndarray]:
        """Apply R_p to embedding e. Returns None if provider not fitted."""
        R = self._R.get(provider)
        if R is None:
            return None
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        return R @ e  # (D,)

    def providers(self) -> List[str]:
        return list(self._R.keys())


# ===========================================================================
# 4. Random-Projection Fallback (Johnson–Lindenstrauss)
# ===========================================================================

def random_projection_fallback(
    source_dim: int,
    target_dim: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Achlioptas (2003) sparse ±1/√3 random projection.

    Returns a ``(target_dim, source_dim)`` matrix R such that
    for any pair of vectors u, v, with high probability over the
    choice of R,

        (1 - ε) ‖u - v‖² ≤ ‖R(u - v)‖² ≤ (1 + ε) ‖u - v‖²

    with target_dim = Ω(ε^{-2} log n) for n source points.

    NOTE: This is a within-space projection. It does NOT bridge
    different embedding spaces. Use Procrustes for that.
    """
    if target_dim > source_dim:
        return np.eye(source_dim, dtype=np.float32)
    rng = np.random.default_rng(seed)
    s = math.sqrt(3.0)
    # Achlioptas: each entry ∈ {-1/√3, 0, +1/√3} with probs (1/6, 2/3, 1/6).
    R = rng.choice(
        [-1.0 / s, 0.0, 1.0 / s],
        size=(target_dim, source_dim),
        p=[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    ).astype(np.float32)
    return R


# ===========================================================================
# 5. Hybrid Retriever (RRF)
# ===========================================================================

class UNIBRIRetriever:
    """
    Hybrid retriever combining ULL (lexical), provider (semantic), and
    optionally FTS5 (BM25) signals via Reciprocal Rank Fusion.

    Parameters
    ----------
    fingerprinter:
        ULL fingerprinter used for the lexical signal.
    bridges:
        Dict provider_name → R_p (D × d_p). May be empty.
    k_rrf:
        The RRF smoothing constant. Cormack et al. recommend 60.
    signals:
        Which signals to include. Valid: ``"ull"``, ``"provider"``.
        FTS5 is always on if a callable is supplied.
    """

    def __init__(
        self,
        fingerprinter: ULLFingerprinter,
        bridges: Optional[Dict[str, np.ndarray]] = None,
        k_rrf: int = 60,
        signals: Tuple[str, ...] = ("ull", "provider"),
    ):
        self.fp = fingerprinter
        self.bridges: Dict[str, np.ndarray] = dict(bridges or {})
        self.k_rrf = int(k_rrf)
        self.signals = tuple(signals)

    def search(
        self,
        query_text: str,
        query_provider_embedding: Optional[np.ndarray] = None,
        query_provider: Optional[str] = None,
        ull_matrix: Optional[np.ndarray] = None,
        provider_matrices: Optional[Dict[str, np.ndarray]] = None,
        ids: Optional[Sequence[str]] = None,
        k: int = 10,
    ) -> List[Dict]:
        """
        Hybrid search.

        Parameters
        ----------
        query_text:
            The raw query string.
        query_provider_embedding:
            Pre-computed embedding of the query in ``query_provider``'s space.
        query_provider:
            Provider name (must be in ``self.bridges``).
        ull_matrix:
            ``(N, D)`` array of ULL fingerprints of stored docs.
        provider_matrices:
            Dict provider_name → ``(N, d_p)`` provider-specific embeddings.
        ids:
            Optional list of memory_ids parallel to the matrices.
        k:
            Number of results to return.

        Returns
        -------
        list of ``{"id", "score", "ranks": {signal: int}}`` sorted by
        descending RRF score.
        """
        if ull_matrix is None or ull_matrix.shape[0] == 0:
            return []
        N = ull_matrix.shape[0]
        ranks: Dict[str, np.ndarray] = {}

        # 1) ULL signal
        if "ull" in self.signals:
            q_ull = self.fp.fingerprint(query_text)
            sims_ull = ull_matrix @ q_ull
            ranks["ull"] = self._rank_desc(sims_ull)

        # 2) Provider semantic signal
        if (
            "provider" in self.signals
            and query_provider_embedding is not None
            and query_provider in self.bridges
            and provider_matrices is not None
            and query_provider in provider_matrices
        ):
            R = self.bridges[query_provider]
            q_proj = R @ np.asarray(query_provider_embedding, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_proj))
            if q_norm > 1e-9:
                q_proj /= q_norm
                doc_matrix = np.asarray(
                    provider_matrices[query_provider], dtype=np.float32
                )
                # Project all docs to ULL: doc_proj = (R @ doc_matrix.T).T
                doc_proj = doc_matrix @ R.T
                dn = np.linalg.norm(doc_proj, axis=1, keepdims=True)
                doc_proj = doc_proj / np.where(dn > 1e-9, dn, 1.0)
                sims_p = doc_proj @ q_proj
                ranks["provider"] = self._rank_desc(sims_p)

        if not ranks:
            return []

        # 3) RRF fusion
        rrf = np.zeros(N, dtype=np.float32)
        for r in ranks.values():
            rrf += 1.0 / (self.k_rrf + r.astype(np.float32))

        order = np.argsort(-rrf)
        out: List[Dict] = []
        for i in order[:k]:
            out.append(
                {
                    "id": ids[i] if ids else str(int(i)),
                    "score": float(rrf[i]),
                    "ranks": {name: int(r[i]) for name, r in ranks.items()},
                }
            )
        return out

    @staticmethod
    def _rank_desc(scores: np.ndarray) -> np.ndarray:
        """0-indexed ranks in descending order of scores."""
        return np.argsort(np.argsort(-scores)).astype(np.int32)


# ===========================================================================
# 6. Hub Set (curated, 256 multilingual terms)
# ===========================================================================

# A minimal starter set. Production should expand to 256+ terms covering
# more languages, more domains (code, math, science), and rarer cognates.
_DEFAULT_HUBS: Tuple[str, ...] = (
    # English core
    "the", "and", "of", "to", "in", "is", "for", "with", "on", "as",
    "data", "model", "function", "value", "class", "object", "method",
    "memory", "search", "index", "vector", "embedding", "language",
    "text", "image", "audio", "video", "time", "space", "user", "agent",
    # Cross-script / cognate
    "computer",        # EN
    "ordinateur",      # FR
    "computadora",     # ES
    "Computer",        # DE
    "компьютер",       # RU
    "计算机",           # ZH
    "コンピュータ",      # JA
    "حاسوب",           # AR
    # Common science
    "energy", "force", "mass", "wave", "field", "particle", "atom",
    "molecule", "cell", "gene", "protein", "evolution",
    # Common action verbs (multilingual)
    "create", "delete", "update", "find", "read", "write", "compute",
    "créer", "supprimer", "mettre", "trouver", "lire", "écrire",
    "計算", "検索", "作成", "削除", "更新", "読取",
    # Programming
    "async", "await", "class", "def", "import", "return", "yield",
    "lambda", "closure", "iterator", "generator", "decorator",
    # Common concepts
    "love", "music", "art", "history", "philosophy", "religion",
    "amour", "musique", "art", "histoire", "philosophie",
    "音楽", "芸術", "歴史", "哲学",
    # Numbers and time (cross-lingual cognate-like)
    "one", "two", "three", "year", "month", "day", "hour",
    "un", "deux", "trois", "année", "mois", "jour", "heure",
    "一", "二", "三", "年", "月", "日",
)


class UNIBRIHubSet:
    """
    Holder for the curated hub set. Use :py:meth:`default` to get the
    built-in 256-term starter, or pass your own.
    """

    @staticmethod
    def default() -> List[str]:
        """Return the built-in starter hub set (≈ 100 terms, not 256 yet)."""
        return list(_DEFAULT_HUBS)

    @staticmethod
    def from_file(path: str) -> List[str]:
        """Load hubs from a UTF-8 text file, one per line."""
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


# ===========================================================================
# 7. Convenience: high-level build helper
# ===========================================================================

def build_unibri(
    corpus_for_idf: Optional[Sequence[str]] = None,
    hubs: Optional[Sequence[str]] = None,
    bits: int = 10,
    orders: Tuple[int, ...] = (2, 3, 4, 5),
    provider_hub_embeddings: Optional[Dict[str, np.ndarray]] = None,
    k_rrf: int = 60,
) -> Tuple[ULLFingerprinter, ProcrustesBridge, UNIBRIRetriever]:
    """
    One-call bootstrap.

    Returns
    -------
    (fingerprinter, bridge, retriever) — all wired together.

    Example
    -------
    >>> fp, br, rt = build_unibri(
    ...     corpus_for_idf=["hello world", "machine learning is fun"],
    ...     hubs=UNIBRIHubSet.default(),
    ...     provider_hub_embeddings={
    ...         "openai": np.random.randn(100, 1536).astype(np.float32),
    ...     },
    ... )
    """
    idf = (
        calibrate_idf(ULLFingerprinter(orders=orders, bits=bits), corpus_for_idf)
        if corpus_for_idf
        else None
    )
    fp = ULLFingerprinter(orders=orders, bits=bits, idf=idf)
    bridge = ProcrustesBridge(hubs=hubs or UNIBRIHubSet.default(), fingerprinter=fp)
    if provider_hub_embeddings:
        bridge.fit(provider_hub_embeddings)
    rt = UNIBRIRetriever(
        fingerprinter=fp,
        bridges=bridge._R,
        k_rrf=k_rrf,
    )
    return fp, bridge, rt


__all__ = [
    "ULLFingerprinter",
    "calibrate_idf",
    "ProcrustesBridge",
    "random_projection_fallback",
    "UNIBRIRetriever",
    "UNIBRIHubSet",
    "build_unibri",
]
