#!/usr/bin/env python3
"""
mathir_advanced.py — MATHIR's third-generation retrieval algorithms.

Four novel algorithms, each derived from first-principles math + MATHIR's
unique 5-tier + link-graph + Ebbinghaus lifecycle + Ledoit-Wolf anomaly stack.

Picked from a 4-agent brainstorm session on 2026-07-01:
- ANTIPODE — phantom-mass (Mahalanobis local density) + hub-trap penalty
- PPR-LTE — Personalized PageRank with Lifecycle-Tier Edges (graph is substrate)
- SMFM    — Spectral Mixing for Forgotten Memories (derive-on-read embedding drift)
- AD      — Anomaly Diffusion (detector state machine, self-referential)

NOT in any prior literature. NOT copied from CRAG or DSpark. Pure math.
"""

from __future__ import annotations

import math
from typing import Sequence

# =============================================================================
# ANTIPODE — re-ranker that no static vector DB can replicate
# =============================================================================
# Combines:
#   - Phantom mass: local density of high-Mahalanobis neighbors in embedding
#     space (uses the *living* Ledoit-Wolf shrunk covariance MATHIR maintains).
#   - Hub-trap penalty: high weighted degree but low local clustering
#     (uses decay-pruned link graph, not a static FAISS index topology).
#   - Tier prior: e^{alpha * w_{t(c)}(q)} boost for tier-intent(q) = tier(c).

W_ANTIPODE_ETA   = 0.5   # phantom-mass damping weight
W_ANTIPODE_LAMBDA = 0.3  # hub-trap penalty weight
W_ANTIPODE_ALPHA = 0.2   # tier-prior boost weight
PHANTOM_RADIUS = 1.0   # epsilon-ball radius (multiples of avg ||v_j - v_c||)
ANOMALY_PCTL = 0.95   # Mahalanobis percentile for phantom membership


def local_phantom_mass(
    candidate_emb,
    neighbor_embs: list,
    neighbor_anomalies: list,
    median_pair_dist: float,
    cov_inv_diag_max: float,
) -> float:
    """Count high-anomaly neighbors within a Euclidean ball, weighted by
    Gaussian proximity. Pure embedding-space function.

    candidate_emb: numpy array, shape (d,)
    neighbor_embs:  list of numpy arrays, shape (d,)
    neighbor_anomalies: list of floats (Mahalanobis^2 per neighbor)
    median_pair_dist: median ||v_j - v_c|| across candidates (to scale ball)
    cov_inv_diag_max: max diagonal of inverse covariance (so 1 / sqrt of
                      Mahalanobis^2 threshold ~= dimensionless)
    """
    if not neighbor_embs or median_pair_dist <= 0:
        return 0.0
    radius = max(0.1, PHANTOM_RADIUS * median_pair_dist)
    sigma = radius / 2.0
    sigma2 = 2.0 * sigma * sigma
    threshold = ANOMALY_PCTL * cov_inv_diag_max * cov_inv_diag_max * 4.0  # scaled chi2
    mass = 0.0
    for v, a in zip(neighbor_embs, neighbor_anomalies):
        diff = v - candidate_emb
        d2 = float(diff @ diff)
        if d2 > radius * radius:
            continue
        if a < threshold:
            continue
        mass += math.exp(-d2 / sigma2)
    return mass


def hub_trap_signal(
    weighted_degree: float,
    mean_weighted_degree: float,
    std_weighted_degree: float,
    clustering_coefficient: float,
) -> float:
    """h^+ (1 - L) = z-scored positive degree * (1 - clustering).
    High value = hub with low neighbor-clique-density = hub-trap.
    """
    if std_weighted_degree <= 0:
        return 0.0
    z = (weighted_degree - mean_weighted_degree) / std_weighted_degree
    h_pos = max(0.0, z)
    return h_pos * (1.0 - clustering_coefficient)


def tier_intent_weights(query: str) -> dict:
    """Naive tier-intent classifier from query keywords.
    Returns normalized weights for {working_memory, episodic, semantic, procedural, immunological}.
    Real impl could embed the query + per-tier centroid + softmax; this heuristic
    is intentionally cheap and deterministic.
    """
    q = (query or "").lower()
    weights = {
        "working_memory": 0.1,
        "episodic": 0.3,
        "semantic": 0.3,
        "procedural": 0.1,
        "immunological": 0.2,
    }
    if any(k in q for k in ("how to", "step 1", "tutorial", "instructions")):
        weights["procedural"] = 0.7
        weights["semantic"] = 0.15
    elif any(k in q for k in ("what is", "define", "meaning of")):
        weights["semantic"] = 0.7
        weights["procedural"] = 0.1
    elif any(k in q for k in ("just now", "earlier today", "this session")):
        weights["working_memory"] = 0.5
        weights["episodic"] = 0.4
    elif any(k in q for k in ("suspicious", "weird", "anomaly", "prompt")):
        weights["immunological"] = 0.6
        weights["semantic"] = 0.2
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}
    return weights


def antipode_score(
    conf_score: float,
    phantom_mass: float,
    hub_signal: float,
    tier_weight: float,
) -> float:
    """ANTIPODE(c|q) = ConfRank(c|q) * exp(-eta*Phi - lambda*HubTrap + alpha*tier_weight)"""
    exponent = (
        -W_ANTIPODE_ETA * phantom_mass
        - W_ANTIPODE_LAMBDA * hub_signal
        + W_ANTIPODE_ALPHA * tier_weight
    )
    # Clamp to avoid overflow on extreme negatives
    exponent = max(-30.0, min(30.0, exponent))
    return conf_score * math.exp(exponent)


# =============================================================================
# PPR-LTE — Personalized PageRank over the link graph, with stability damping
# =============================================================================
# PPR-LTE makes the cosine-similarity link graph the SUBSTRATE. Hybrid_search
# only seeds the teleport vector.

PPR_LTE_ALPHA = 0.85   # teleport probability (FIX 1c 2026-07-01: was 0.15, but the link graph is too sparse in MATHIR for graph smoothing to add signal -- teleport dominant works better)
PPR_LTE_KAPPA = 2.0    # stability edge-damping exponent (edge weight *= s_j^kappa)
PPR_LTE_GAMMA = 2.0    # Mahalanobis anomaly discount in final score
PPR_LTE_MAX_ITER = 30
PPR_LTE_EPS = 1e-6


def ppr_lte_transition(weights) -> list[list[float]]:
    """Row-stochastic transition matrix from a square weight matrix.
    Fixes the non-reciprocal cumulative-weight bug of mathir_spread.py by
    ensuring each row sums to 1.0.
    """
    n = len(weights)
    T = [[0.0] * n for _ in range(n)]
    for i in range(n):
        s = sum(weights[i])
        if s <= 0:
            # Self-loop fallback (will be dampened by teleport)
            T[i][i] = 1.0
        else:
            for j in range(n):
                T[i][j] = weights[i][j] / s
    return T


def ppr_lte_iterate(T, teleport, alpha=PPR_LTE_ALPHA,
                    max_iter=PPR_LTE_MAX_ITER, eps=PPR_LTE_EPS):
    """Power iteration: pi_{t+1} = (1-alpha) * pi_t * T + alpha * teleport
    Convergence: |pi_{t+1} - pi_t|_inf < eps.
    """
    n = len(T)
    pi = [1.0 / n] * n
    for _ in range(max_iter):
        pi_new = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += pi[j] * T[j][i]
            pi_new[i] = (1 - alpha) * s + alpha * teleport[i]
        diff = max(abs(pi_new[i] - pi[i]) for i in range(n))
        pi = pi_new
        if diff < eps:
            break
    return pi


def ppr_lte_score(pi, anomaly_scores, gamma=PPR_LTE_GAMMA):
    """PPR-LTE score for memory i = pi*[i] * exp(-gamma * m(i))
    Anomalous memories get exponentially discounted.
    """
    return [p * math.exp(-gamma * a) for p, a in zip(pi, anomaly_scores)]


def ppr_lte_damp_edge_weight(cos_w: float, s_target: float,
                             kappa=PPR_LTE_KAPPA) -> float:
    """Edge weight w_ij *= s_j^kappa: old memories' inbound edges decay
    structurally at query time, without rebuilding the link graph.
    """
    return cos_w * (s_target ** kappa)


# =============================================================================
# SMFM — Spectral Mixing for Forgotten Memories
# =============================================================================
# At recall time, derive a drift-adjusted embedding:
#   e(t) = normalize(s(t) * e_0 + (1 - s(t)) * b)
# where b is a self-calibrating background centroid. The drift is invisible
# at storage but alters the effective retrieval geometry.

SMFM_DEFAULT_KAPPA = 1.0  # unused for now; reserved for tier-weighted variant


def smfm_drift_embedding(e0, b, stability):
    """Derive a stability-weighted drift embedding.

    e0: original embedding (iterable of floats, length d)
    b:  background centroid (iterable of floats, length d)
    stability: scalar in [0, 1]

    Returns the drifted embedding as a list of floats.
    """
    s = max(0.0, min(1.0, float(stability)))
    e0_list = list(e0)
    b_list = list(b)
    drifted = [s * a + (1.0 - s) * c for a, c in zip(e0_list, b_list)]
    n = math.sqrt(sum(x * x for x in drifted))
    if n < 1e-12:
        return e0_list
    return [x / n for x in drifted]


def smfm_score(emb_drifted, query_emb):
    """Cosine similarity of a drifted embedding to a query.
    Pure python implementation for benchmark/test paths.
    """
    if not emb_drifted or not query_emb:
        return 0.0
    n1 = math.sqrt(sum(x * x for x in emb_drifted))
    n2 = math.sqrt(sum(x * x for x in query_emb))
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return sum(a * b for a, b in zip(emb_drifted, query_emb)) / (n1 * n2)


# =============================================================================
# AD — Anomaly Diffusion (self-referential detector state machine)
# =============================================================================
# The detector's threshold sensitivity becomes a function of its own history.
# State: (mu, Sigma, A, P) with P = sigmoid(alpha*A - beta).
# A_t+1 = (1 - lambda_A) A_t + lambda_A * max(0, s^2_raw - tau^2)
# Detector state IS itself a memory retrievable via recall.

AD_LAMBDA_A = 0.05   # EMA decay for anomaly intensity
AD_ALPHA_P = 4.0     # paranoid-mode aggressiveness on top of Mahalanobis
AD_BETA_P = 0.5      # offset before sigmoid turns on
AD_GAMMA = 0.5       # boost coefficient in s*
AD_DEFAULT_TAU = 4.0  # initial chi^2 threshold (4 = dim=384 normal tail)


def ad_update_state(a_t, s2_raw, tau):
    """One-step AD update. Returns new A_t."""
    a_next = (1 - AD_LAMBDA_A) * a_t + AD_LAMBDA_A * max(0.0, s2_raw - tau * tau)
    return a_next


def ad_paranoid_score(s2_raw, p_t, projection_term, gamma=AD_GAMMA):
    """Combined score = sqrt( s^2_raw + gamma * sqrt(P) * Pi(x) )
    where Pi(x) is the projection onto the top anomalous direction.
    """
    boost = gamma * math.sqrt(max(0.0, p_t)) * max(0.0, projection_term)
    return math.sqrt(max(0.0, s2_raw + boost))


def ad_para_probability(a_t, alpha=AD_ALPHA_P, beta=AD_BETA_P):
    """sigmoid(alpha*A - beta) — paranoid mode probability."""
    z = alpha * a_t - beta
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    e = math.exp(z)
    return e / (1.0 + e)
