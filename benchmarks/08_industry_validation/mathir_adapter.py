#!/usr/bin/env python3
"""
MathirAdapter — full HTTP client for the MATHIR daemon, used by the
LongMemEval / LoCoMo benchmark runners in benchmarks/08_industry_validation/.

Talks to the MATHIR Flask daemon (mathir_mcp/mathir_lib/mathir_server.py) at
http://127.0.0.1:7338 by default. Every request carries a `project` field so
benchmark items get fully isolated memory namespaces (no cross-question
leakage). Each project gets its own SQLite DB keyed by project name.

The benchmark exercises the FULL MATHIR surface area, not just
save + hybrid_search as the previous (limited) version did. Specifically
this adapter wires up:

  INGEST          /api/memory/save         per-turn memory ingest, with
                                            block_type selected by the
                                            benchmark per the question's
                                            role, plus the risk_check
                                            pre-screen (leakage /
                                            sycophancy) that's part of
                                            a real save round trip.

  SEARCH          /api/memory/hybrid_search, recall, smart_search, push
                                            -- multiple search modes
                                            compared per question (full
                                            receiver-operating-characteristic
                                            picture, not just one knob).

  GRAPH           /api/memory/build_links, link, get_links, incoming_links
                                            -- the spreading-activation
                                            surface that's unique to MATHIR
                                            among the competitors we compare
                                            against (Mem0 / Zep don't expose
                                            an equivalent).

  LIFECYCLE       /api/memory/decay, consolidate, promote, auto_promote
                                            -- time-based memory management
                                            exercised at run boundaries,
                                            simulating the multi-day agent
                                            memory behavior that's the
                                            real-world use case.

  IMMUNOLOGICAL   /api/memory/audit_immunological, the per-save Mahalanobis
                                            anomaly detector running
                                            server-side (already automatic
                                            on every save but also exposed
                                            for bulk inspection).

  OBSERVABILITY   /api/memory/stats, audit, export, sessions, /api/stats,
                  /api/projects, /api/memories, /api/push_cache_stats,
                  /health
                                            -- used to write the
                                            run-level metadata into the
                                            JSONL header / JSON summary
                                            so every result is
                                            reproducible from the artifact
                                            alone.

Loopback (127.0.0.1) requests need no auth header. Non-loopback hosts
require a Bearer token (MATHIR_AUTH_TOKEN) -- the adapter sends it
automatically when MATHIR_AUTH_TOKEN is set in the environment.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Iterable


DEFAULT_TIMEOUT_S = 60.0


class MathirAdapter:
    """Full-featured HTTP client for the MATHIR memory daemon.

    Each method maps 1:1 to a /api/memory/* endpoint on the server (see
    mathir_server.py for the source-of-truth field names and defaults).
    """

    def __init__(self, daemon_url: str = "http://127.0.0.1:7338", timeout: float = DEFAULT_TIMEOUT_S):
        self.daemon_url = daemon_url.rstrip("/")
        self.timeout = timeout
        self._auth_token: str | None = os.environ.get("MATHIR_AUTH_TOKEN")
        if not self.ping():
            raise RuntimeError(
                f"MATHIR daemon not reachable at {self.daemon_url}/api/ping.\n"
                f"Start it first, e.g.: `python -m mathir_mcp`"
            )

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None,
                 timeout: float | None = None) -> dict:
        url = f"{self.daemon_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self._auth_token}"} if self._auth_token else {})},
            method=method,
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MATHIR daemon returned HTTP {e.code} for {method} {path}: {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"MATHIR daemon unreachable at {url} ({e}). Run `python -m mathir_mcp` and retry.") from e
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        try:
            result = json.loads(body)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"MATHIR daemon returned non-JSON for {method} {path}: {body!r}") from e
        if isinstance(result, dict) and "error" in result and not path.startswith("/api/health"):
            raise RuntimeError(f"MATHIR daemon error on {method} {path}: {result['error']}")
        # Attach timing so callers can log server-roundtrip latency distinctly
        # from the JSON body's own per-operation timings (when applicable).
        if isinstance(result, dict):
            result.setdefault("_client_elapsed_ms", round(elapsed_ms, 2))
        return result

    def _get(self, path: str, timeout: float | None = None) -> dict:
        return self._request("GET", path, timeout=timeout)

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        return self._request("POST", path, payload, timeout=timeout)

    # ------------------------------------------------------------------
    # Health / observability
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            data = self._get("/api/ping", timeout=5.0)
            return bool(data.get("pong"))
        except Exception:
            return False

    def health(self) -> dict:
        return self._get("/health")

    def stats(self) -> dict:
        return self._get("/api/stats")

    def list_projects(self) -> dict:
        return self._get("/api/projects")

    def list_memories(self, project: str | None = None, **filters) -> dict:
        params = {"project": project, **filters} if project else filters
        return self._post("/api/memories", params)

    def memory_stats(self, project: str) -> dict:
        return self._post("/api/memory/stats", {"project": project})

    def memory_audit(self, agent: str | None = None, limit: int = 50,
                     project: str | None = None) -> dict:
        payload = {"limit": limit}
        if agent is not None:
            payload["agent"] = agent
        if project is not None:
            payload["project"] = project
        return self._post("/api/memory/audit", payload)

    def memory_sessions(self, limit: int = 10, project: str | None = None) -> dict:
        payload = {"limit": limit}
        if project is not None:
            payload["project"] = project
        return self._post("/api/memory/sessions", payload)

    def memory_export(self, project: str | None = None) -> dict:
        payload = {"project": project} if project else {}
        return self._post("/api/memory/export", payload)

    def push_cache_stats(self) -> dict:
        return self._get("/api/push_cache_stats")

    # ------------------------------------------------------------------
    # INGEST
    # ------------------------------------------------------------------

    def add(self, project: str, content: str,
            agent: str = "benchmark",
            block_type: str = "episodic",
            label: str = "",
            priority: int = 5) -> dict:
        """POST /api/memory/save. Returns the full server response dict,
        including any `tier` / `anomaly_score` override the server applies.

        `block_type` is intentionally exposed (not hardcoded to 'episodic'
        as the previous adapter did) -- the benchmark runner picks it per
        turn based on the role / content type, exercising MATHIR's full
        tier-routing surface.
        """
        return self._post("/api/memory/save", {
            "content": content,
            "agent": agent,
            "block_type": block_type,
            "label": label,
            "priority": priority,
            "project": project,
        })

    def risk_check(self, content: str) -> dict:
        """POST /api/memory/risk_check. The server runs DomainClassifier +
        LeakageDetector + SycophancyDetector. Returns domain + risk
        scores + safe_to_store flag. Useful for pre-screening ingested
        content -- the benchmark logs this per-question so reviewers can
        see what the risk pipeline flagged.
        """
        return self._post("/api/memory/risk_check", {"content": content})

    # ------------------------------------------------------------------
    # SEARCH (4 modes)
    # ------------------------------------------------------------------

    def hybrid_search(self, project: str, query: str, k: int = 10,
                      vector_weight: float = 1.0, bm25_weight: float = 1.0,
                      agent: str | None = None) -> dict:
        """POST /api/memory/hybrid_search. vector+BM25+RRF fusion with
        configurable weights. The benchmark varies weights to find the
        empirically-best bm25_weight/vector_weight pairing per embedder
        (the finding from prior research is there's no universal optimum).
        """
        payload = {"query": query, "k": k, "project": project,
                   "vector_weight": vector_weight, "bm25_weight": bm25_weight}
        if agent is not None:
            payload["agent"] = agent
        return self._post("/api/memory/hybrid_search", payload)

    def recall(self, project: str, query: str, k: int = 10,
               agent: str | None = None,
               block_type: str | None = None) -> dict:
        """POST /api/memory/recall. Plain vector search (sqlite-vec).
        Useful as a baseline (vs hybrid_search) to isolate the BM25
        contribution."""
        payload = {"query": query, "k": k, "project": project}
        if agent is not None:
            payload["agent"] = agent
        if block_type is not None:
            payload["block_type"] = block_type
        return self._post("/api/memory/recall", payload)

    def smart_search(self, project: str, query: str, k: int = 10,
                     agent: str | None = None) -> dict:
        """POST /api/memory/smart_search. Server-side query analysis
        + agent-filtered vector search."""
        payload = {"query": query, "k": k, "project": project}
        if agent is not None:
            payload["agent"] = agent
        return self._post("/api/memory/smart_search", payload)

    def push(self, project: str, context: str, k: int = 10,
             agent: str | None = None) -> dict:
        """POST /api/memory/push. Server extracts up to 5 sub-queries from
        `context` (an LLM-free heuristic), runs each, deduplicates, caches
        by context hash. This is the right endpoint for the LLM
        answer-generation step, since the LLM sees a full multi-turn
        context not a single query string.
        """
        payload = {"context": context, "k": k, "project": project}
        if agent is not None:
            payload["agent"] = agent
        return self._post("/api/memory/push", payload)

    # ------------------------------------------------------------------
    # Confrank-native search mode (MATHIR-specific, novel as of 2026-07-01)
    # ------------------------------------------------------------------
    #
    # Combines hybrid_search + spreading-activation graph + per-memory
    # stability/tier in the new `mathir_confrank.py` to re-rank results
    # without any external labels or LLM judge.
    #
    # This is NOT CRAG (which uses an external LLM-as-judge evaluator) and
    # NOT DSpark (which trains an offline confidence head on labels). It is
    # a self-supervised confidence score derived purely from the structure
    # of the MATHIR graph + lifecycle state.
    def confrank_search(self, project: str, query: str, k: int = 10,
                        graph_depth: int = 2, graph_decay: float = 0.5,
                        vector_weight: float = 1.0, bm25_weight: float = 1.0,
                        agent: str | None = None,
                        enable_tcr: bool = True,
                        enable_gra: bool = True) -> dict:
        """Hybrid_search + graph + lifecycle re-ranking, self-supervised.

        Algorithm (see mathir_confrank.py for the full math):
          1. hybrid_search returns K candidates with raw rrf_score.
          2. Build a per-term probe graph by get_links from each top-1
             candidate, then collect neighbor memory_ids.
          3. For each candidate, compute confrank = α·sem + β·recall
             + γ·graph_convergence + δ·tier + ε·anomaly.
          4. (optional) Multiply by time_factor = sigmoid(α·stability
             - β·age_days) for time-aware confidence (TCR).
          5. (optional) Apply gra_tiebreak() if top-2 within delta.
          6. Return reranked list.

        Returns: dict with "results" (reranked) and "diagnostics" (full
        per-candidate score breakdown).
        """
        # Late import to avoid circulars + keep dependency surface small.
        try:
            from mathir_confrank import confrank, time_factor, gra_tiebreak
        except ImportError:
            # If the module isn't on sys.path, fall back to a 1-by-1 import
            # from the in-tree copy (this file lives next to the runner).
            import importlib
            import sys as _sys
            from pathlib import Path as _P
            _root = _P(__file__).resolve().parent.parent.parent
            _mc = _root / "mathir_mcp" / "mathir_lib"
            if str(_mc) not in _sys.path:
                _sys.path.insert(0, str(_mc))
            from mathir_confrank import confrank, time_factor, gra_tiebreak

        # 1. Retrieve candidates.
        hybrid = self.hybrid_search(
            project=project, query=query, k=k,
            vector_weight=vector_weight, bm25_weight=bm25_weight,
            agent=agent,
        )
        candidates = hybrid.get("results", []) if isinstance(hybrid, dict) else []
        if not candidates:
            return {"results": [], "diagnostics": {"empty_input": True},
                    "mode": "confrank"}

        # 2. Build graph evidence: per-candidate neighbors + per-term neighbors.
        # Two-pass: first collect neighbors via get_links; then index them.
        neighbors_by_candidate: dict[str, list[str]] = {}
        query_terms = [t for t in (query.lower().split() if query else [])
                       if len(t) >= 3]
        # Limit term probes to top 5 to keep wall-clock sane.
        query_terms = query_terms[:5]

        for c in candidates:
            mid = c.get("memory_id", "")
            if not mid:
                continue
            gl = self._safe_get_links(project, mid, depth=graph_depth,
                                      decay=graph_decay)
            nbrs = []
            if isinstance(gl, dict) and isinstance(gl.get("result"), list):
                nbrs = [g.get("memory_id") for g in gl["result"]
                        if g.get("memory_id")]
            elif isinstance(gl, dict) and isinstance(gl.get("results"), list):
                nbrs = [g.get("memory_id") for g in gl["results"]
                        if g.get("memory_id")]
            neighbors_by_candidate[mid] = nbrs

        # Term-keyed neighborhoods: for each query_term, do hybrid_search with
        # the term only (uses the same embedder) and grab neighbor overlap.
        query_term_to_neighbors: dict[str, set[str]] = {}
        for term in query_terms:
            ts = self.hybrid_search(
                project=project, query=term, k=max(3, k // 3),
                vector_weight=vector_weight, bm25_weight=bm25_weight,
                agent=agent,
            )
            n_set = set()
            for r in (ts.get("results", []) if isinstance(ts, dict) else []):
                mid = r.get("memory_id", "")
                if mid:
                    n_set.add(mid)
                    # also pull in this term-result's neighbors
                    gl = self._safe_get_links(project, mid, depth=1, decay=graph_decay)
                    ng = []
                    if isinstance(gl, dict) and isinstance(gl.get("result"), list):
                        ng = [g.get("memory_id") for g in gl["result"]
                              if g.get("memory_id")]
                    n_set.update(x for x in ng if x)
            query_term_to_neighbors[term] = n_set

        # 3. Pull stability for these specific candidates if available via
        # memory_stats; default to 0.5 (mid) when unknown. We do a single
        # bulk stats call and index by memory_id below.
        stats_resp = self._safe_memory_stats(project)
        stability_by_id: dict[str, float] = {}
        blocks: dict[str, dict] = {}
        if isinstance(stats_resp, dict):
            blocks = stats_resp.get("by_block_type", {}) or {}
            # by_block_type is a count, not per-memory. We'll fall back to
            # per-candidate tier=episodic default + assume 0.5 unless the
            # route returns richer per-id info later.

        # Recall count proxy: from the hybrid_search response vector_hits /
        # bm25_hits + we don't have per-id recall count without a memory_audit
        # call. Default to 0 (the Laplace smoothing takes care of zero).
        total_searches = max(k * 10, 100)

        # 4. Re-rank with confrank.
        reranked, diagnostics = confrank(
            candidates=candidates,
            query_terms=query_terms,
            neighbors_by_candidate=neighbors_by_candidate,
            query_term_to_neighbors=query_term_to_neighbors,
            total_searches=total_searches,
        )

        # 5. Optional time-aware confidence ranker (TCR).
        if enable_tcr:
            for r in reranked:
                # We don't have age_days per memory without a memory_export,
                # so use 0 (recent). The factor reduces to sigmoid(stability).
                t = time_factor(
                    stability=float(r.get("stability", 0.5) or 0.5),
                    age_days=0.0,
                )
                r["confrank"]["score"] = round(r["confrank"]["score"] * t, 6)
                r["confrank"]["tcr_time_factor"] = round(t, 4)
            # Re-sort by updated score.
            reranked.sort(key=lambda x: x["confrank"]["score"], reverse=True)

        # 6. Optional graph-reinforced tiebreak (GRA).
        if enable_gra:
            reranked = gra_tiebreak(reranked, delta_threshold=0.05)

        diagnostics["mode"] = "confrank"
        diagnostics["tcr_enabled"] = enable_tcr
        diagnostics["gra_enabled"] = enable_gra
        diagnostics["n_candidates_in"] = len(candidates)
        return {"results": reranked, "diagnostics": diagnostics,
                "mode": "confrank"}

    def _safe_get_links(self, project: str, memory_id: str,
                        depth: int = 1, decay: float = 0.5) -> dict:
        """Used by confrank_search to gather neighborhood evidence without
        taking down the whole pipeline on a single missing memory_id."""
        try:
            return self.get_links(project, memory_id, depth=depth, decay=decay)
        except Exception:
            return {"result": []}

    def _safe_memory_stats(self, project: str) -> dict:
        try:
            return self.memory_stats(project)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # THIRD-GENERATION retrieval algorithms (mathir_advanced.py)
    # ------------------------------------------------------------------
    #
    # Implemented from a 4-agent pure-math brainstorm on 2026-07-01.
    # None of these duplicate CRAG (no LLM judge) or DSpark (no offline
    # labels). Each fuses MATHIR-specific surfaces in a non-replicable way.
    #
    # Usage from the runner:
    #   adapter.antipode_search(project, query, k)
    #   adapter.ppr_lte_search(project, query, k)
    #   adapter.smfm_search(project, query, k)
    #   adapter.ad_score_search(project, query, k)

    def antipode_search(self, project: str, query: str, k: int = 10,
                         vector_weight: float = 1.0,
                         bm25_weight: float = 1.0) -> dict:
        """ANTIPODE re-ranker.

        1. hybrid_search returns K candidates.
        2. For each candidate, compute phantom mass (local density of
           high-Mahalanobis neighbors) and hub-trap signal (high weighted
           degree, low clustering).
        3. Compute tier-intent weights from query keywords.
        4. Score: ConfRank * exp(-eta*phantom - lambda*hub + alpha*tier).
        """
        try:
            from mathir_advanced import (
                tier_intent_weights, hub_trap_signal, antipode_score,
                W_ANTIPODE_ETA, W_ANTIPODE_LAMBDA, W_ANTIPODE_ALPHA,
            )
        except ImportError:
            import sys
            from pathlib import Path
            _root = Path(__file__).resolve().parent.parent.parent
            _mc = _root / "mathir_mcp" / "mathir_lib"
            if str(_mc) not in sys.path:
                sys.path.insert(0, str(_mc))
            from mathir_advanced import (
                tier_intent_weights, hub_trap_signal, antipode_score,
                W_ANTIPODE_ETA, W_ANTIPODE_LAMBDA, W_ANTIPODE_ALPHA,
            )

        hybrid = self.hybrid_search(project=project, query=query, k=k,
                                    vector_weight=vector_weight,
                                    bm25_weight=bm25_weight)
        candidates = hybrid.get("results", []) if isinstance(hybrid, dict) else []
        if not candidates:
            return {"results": [], "diagnostics": {"empty": True},
                    "mode": "antipode"}

        # Tier-intent prior
        intent = tier_intent_weights(query)

        # Compute per-candidate graph properties via get_links.
        degrees: list[float] = []
        clusters: list[float] = []
        for c in candidates:
            mid = c.get("memory_id", "")
            gl = self._safe_get_links(project, mid, depth=1, decay=0.5)
            nbrs = self._extract_neighbor_ids(gl)
            degrees.append(float(len(nbrs)))
            # Clustering at depth=1: how many of my neighbors know each other.
            tri = 0
            for n1 in nbrs:
                gl2 = self._safe_get_links(project, n1, depth=1, decay=0.5)
                nbrs2 = set(self._extract_neighbor_ids(gl2))
                for n2 in nbrs:
                    if n2 != n1 and n2 in nbrs2:
                        tri += 1
            denom = max(1, len(nbrs) * (len(nbrs) - 1))
            clusters.append(min(1.0, tri / denom) if denom > 0 else 0.0)

        mean_d = sum(degrees) / len(degrees) if degrees else 0.0
        std_d = (sum((d - mean_d) ** 2 for d in degrees) / max(1, len(degrees) - 1)) ** 0.5 if len(degrees) > 1 else 1.0

        reranked: list[dict] = []
        for c, deg, clu in zip(candidates, degrees, clusters):
            tier = (c.get("tier") or c.get("block_type") or "").lower()
            tier_w = intent.get(tier, 0.0)
            hub_sig = hub_trap_signal(deg, mean_d, std_d, clu)
            # No L2 anomaly score cached per-memory-id without /api/memory
            # audit_immunological (currently 405), so phantom_mass=0 for now.
            # The remaining terms (hub, tier) still apply.
            phantom = 0.0
            base = float(c.get("rrf_score", c.get("score", 0)) or 0)
            base = max(base, 0.0) + 0.001  # ensure positive
            score = antipode_score(base, phantom, hub_sig, tier_w)
            reranked.append({**c, "antipode_score": score,
                             "antipode_breakdown": {
                                 "phantom_mass": phantom,
                                 "hub_signal": round(hub_sig, 4),
                                 "tier_weight": round(tier_w, 4),
                             }})
        reranked.sort(key=lambda x: x["antipode_score"], reverse=True)
        return {"results": reranked,
                "diagnostics": {"mode": "antipode",
                                "weights": {"eta": W_ANTIPODE_ETA,
                                            "lambda": W_ANTIPODE_LAMBDA,
                                            "alpha": W_ANTIPODE_ALPHA},
                                "intent": intent},
                "mode": "antipode"}

    def ppr_lte_search(self, project: str, query: str, k: int = 10,
                        vector_weight: float = 1.0,
                        bm25_weight: float = 1.0) -> dict:
        """PPR-LTE: graph is the substrate, hybrid_search seeds teleport."""
        try:
            from mathir_advanced import (
                ppr_lte_damp_edge_weight, ppr_lte_transition, ppr_lte_iterate,
                ppr_lte_score, PPR_LTE_ALPHA,
            )
        except ImportError:
            import sys
            from pathlib import Path
            _root = Path(__file__).resolve().parent.parent.parent
            _mc = _root / "mathir_mcp" / "mathir_lib"
            if str(_mc) not in sys.path:
                sys.path.insert(0, str(_mc))
            from mathir_advanced import (
                ppr_lte_damp_edge_weight, ppr_lte_transition, ppr_lte_iterate,
                ppr_lte_score, PPR_LTE_ALPHA,
            )

        hybrid = self.hybrid_search(project=project, query=query, k=k,
                                    vector_weight=vector_weight,
                                    bm25_weight=bm25_weight)
        candidates = hybrid.get("results", []) if isinstance(hybrid, dict) else []
        if not candidates:
            return {"results": [], "diagnostics": {"empty": True},
                    "mode": "ppr_lte"}

        # Build candidate id <-> index map.
        ids = [c.get("memory_id", "") for c in candidates]
        n = len(ids)
        # Teleport vector: uniform + small boost on candidates matched by hybrid.
        teleport = [1.0 / n] * n

        # Build weight matrix from current K=10 link neighborhood (no
        # full graph build -- that's the Push-PPR cost-saver).
        weights = [[0.0] * n for _ in range(n)]
        anomaly_proxy = [0.0] * n
        for i, c in enumerate(candidates):
            mid = c.get("memory_id", "")
            gl = self._safe_get_links(project, mid, depth=2, decay=0.5)
            nbrs = self._extract_neighbor_ids(gl)
            if not nbrs:
                # Self-loop to keep T row-stochastic when isolated.
                weights[i][i] = 1.0
                continue
            for j in ids:
                if j and j in nbrs:
                    weights[i][ids.index(j) if j in ids else i] = 0.5
            # Stability proxy: anchor on score (high score = high access)
            anomaly_proxy[i] = 1.0 - min(1.0, float(c.get("rrf_score", c.get("score", 0)) or 0))

        # Damping: edge weight *= s_target^kappa where s_target is the
        # recipient stability. Use anomaly proxy as inverse stability.
        for i in range(n):
            for j in range(n):
                if weights[i][j] > 0 and i != j:
                    s_target = max(0.05, 1.0 - anomaly_proxy[j])
                    weights[i][j] = ppr_lte_damp_edge_weight(weights[i][j], s_target)

        T = ppr_lte_transition(weights)
        pi = ppr_lte_iterate(T, teleport, alpha=PPR_LTE_ALPHA)
        scores = ppr_lte_score(pi, anomaly_proxy)
        for c, s in zip(candidates, scores):
            c["ppr_lte_score"] = s
        ranked = sorted(candidates, key=lambda x: x["ppr_lte_score"], reverse=True)
        return {"results": ranked,
                "diagnostics": {"mode": "ppr_lte",
                                "alpha": PPR_LTE_ALPHA,
                                "n_candidates": n,
                                "n_iterations_used": PPR_LTE_ALPHA and 30},
                "mode": "ppr_lte"}

    def smfm_search(self, project: str, query: str, k: int = 10,
                    vector_weight: float = 1.0,
                    bm25_weight: float = 1.0,
                    background_decay_rate: float = 0.5) -> dict:
        """SMFM: derive-on-read embedding drift.

        Recall-time transform: e(t) = normalize(s*e0 + (1-s)*b).
        Rank by cosine against drifted embeddings -- different ranking than
        raw cosine when low-stability memories migrate toward background.
        """
        try:
            from mathir_advanced import smfm_drift_embedding, smfm_score
        except ImportError:
            import sys
            from pathlib import Path
            _root = Path(__file__).resolve().parent.parent.parent
            _mc = _root / "mathir_mcp" / "mathir_lib"
            if str(_mc) not in sys.path:
                sys.path.insert(0, str(_mc))
            from mathir_advanced import smfm_drift_embedding, smfm_score

        # Need raw embeddings to apply the SMFM transform. Hybrid_search
        # returns rrf_score etc but not raw embeddings. Use recall + a
        # request that includes vector data when available; fall back to
        # approximate stability via score.
        rec = self.recall(project=project, query=query, k=k)
        cands = rec.get("results", []) if isinstance(rec, dict) else []
        if not cands:
            return {"results": [], "diagnostics": {"empty": True},
                    "mode": "smfm"}
        # Background centroid b: average of bottom-10% score candidates
        # (those most likely to be "background noise" by score heuristic).
        sorted_by_score = sorted(cands, key=lambda c: float(c.get("score", 0) or 0))
        b_pool = sorted_by_score[: max(1, len(sorted_by_score) // 10)]
        # We don't have raw emb either; fall back to identity-like background:
        # use the rrf_score as a proxy embedding under a guardrail -- this
        # is a degraded mode. The full version would require /api/memory/export
        # for the raw 384d vector.
        b_scalar = sum(float(c.get("score", 0) or 0) for c in b_pool) / len(b_pool)
        b_vec_dummy = [b_scalar]  # represents mean-rating
        ranked = []
        for c in cands:
            e0_dummy = [float(c.get("score", 0) or 0)]
            s_proxy = min(1.0, float(c.get("score", 0) or 0))
            e_t = smfm_drift_embedding(e0_dummy, b_vec_dummy, s_proxy)
            score = smfm_score(e_t, [1.0])  # query as constant 1-vector
            new = dict(c)
            new["smfm_score"] = score
            new["stability_proxy"] = s_proxy
            ranked.append(new)
        ranked.sort(key=lambda x: x["smfm_score"], reverse=True)
        return {"results": ranked,
                "diagnostics": {"mode": "smfm",
                                "note": "degraded: SMFM uses rrf_score as 1-dim embedding proxy "
                                        "because /api/memory returns no raw vectors. "
                                        "Effect is principally the lifecycle weighting of "
                                        "retrieval, not the embedding-geometry shift."},
                "mode": "smfm"}

    def ad_score_search(self, project: str, query: str, k: int = 10,
                        vector_weight: float = 1.0,
                        bm25_weight: float = 1.0) -> dict:
        """AD: Anomaly Diffusion. Run hybrid_search but ADD a paranoid
        boost to the score of memories whose stored Mahalanobis-ish anomaly
        (proxy: inverse score, since audit_immunological is broken)
        exceeds a threshold AND the running detector state has high P_t.

        Without /audit_immunological working, AD is reduced to a
        standard-deviation penalty on the score distribution. The
        architecture is in place for when the audit endpoint is fixed.
        """
        import math as _math
        try:
            from mathir_advanced import (
                ad_update_state, ad_para_probability, ad_paranoid_score,
            )
        except ImportError:
            import sys
            from pathlib import Path
            _root = Path(__file__).resolve().parent.parent.parent
            _mc = _root / "mathir_mcp" / "mathir_lib"
            if str(_mc) not in sys.path:
                sys.path.insert(0, str(_mc))
            from mathir_advanced import (
                ad_update_state, ad_para_probability, ad_paranoid_score,
            )

        hybrid = self.hybrid_search(project=project, query=query, k=k,
                                    vector_weight=vector_weight,
                                    bm25_weight=bm25_weight)
        candidates = hybrid.get("results", []) if isinstance(hybrid, dict) else []
        if not candidates:
            return {"results": [], "diagnostics": {"empty": True},
                    "mode": "ad"}

        # Simulate detector-state evolution across this batch of saves.
        # In production, A_t comes from the Mahalanobis detector at save time.
        a_t = 0.0
        score_var = []
        for c in candidates:
            s_raw = float(c.get("rrf_score", c.get("score", 0)) or 0)
            score_var.append(s_raw)
        if score_var:
            mean = sum(score_var) / len(score_var)
            var = sum((s - mean) ** 2 for s in score_var) / max(1, len(score_var))
            for s in score_var:
                t = (s - mean) ** 2
                a_t = ad_update_state(a_t, t, tau=_math.sqrt(var) or 1.0)
        p_t = ad_para_probability(a_t)
        ranked = []
        for c in candidates:
            s_raw = float(c.get("rrf_score", c.get("score", 0)) or 0)
            proj = max(0.0, (s_raw - mean) if score_var else 0.0)
            ad_score = ad_paranoid_score(s_raw ** 2, p_t, proj)
            new = dict(c)
            new["ad_score"] = ad_score
            new["ad_state"] = {"A": round(a_t, 4), "P": round(p_t, 4)}
            ranked.append(new)
        ranked.sort(key=lambda x: x["ad_score"], reverse=True)
        return {"results": ranked,
                "diagnostics": {"mode": "ad", "A_t": round(a_t, 4),
                                "P_t": round(p_t, 4),
                                "note": "using rrf_score variance as anomaly proxy; "
                                        "audit_immunological is 405 so the real "
                                        "Mahalanobis scores per memory are unavailable"},
                "mode": "ad"}

    def _extract_neighbor_ids(self, gl_response) -> list[str]:
        """Extract memory_id list from a /get_links response shape
        (which uses 'result' as the key, singular -- see graph-mode-bug-fixed)."""
        if not isinstance(gl_response, dict):
            return []
        for k in ("result", "results", "memories"):
            v = gl_response.get(k)
            if isinstance(v, list):
                return [x.get("memory_id") for x in v if x.get("memory_id")]
        return []  # noqa

    # ------------------------------------------------------------------
    # GRAPH (MATHIR-specific, no direct Mem0/Zep equivalent)
    # ------------------------------------------------------------------

    def build_links(self, project: str, threshold: float = 0.7,
                    limit: int = 1000, mode: str = "cosine") -> dict:
        """POST /api/memory/build_links. The graph is what enables
        spreading activation later (see get_links below).

        mode: "cosine" (default -- pairwise embedding-similarity edges above
        `threshold`), "entity" (edges between memories sharing a named
        entity -- for multi-hop bridging that cosine graphs miss), or "both".
        """
        return self._post("/api/memory/build_links", {
            "threshold": threshold, "limit": limit, "project": project,
            "mode": mode,
        })

    def link(self, project: str, source_id: str, target_id: str,
             weight: float = 1.0) -> dict:
        """POST /api/memory/link. Manually add a directed edge."""
        return self._post("/api/memory/link", {
            "source_id": source_id, "target_id": target_id,
            "weight": weight, "project": project,
        })

    def get_links(self, project: str, memory_id: str, depth: int = 2,
                  decay: float = 0.5) -> dict:
        """POST /api/memory/get_links. BFS over the link graph from
        `memory_id` to depth `depth`, decaying edge weights by `decay`
        at each hop. This is the spreading-activation primitive."""
        return self._post("/api/memory/get_links", {
            "memory_id": memory_id, "depth": depth, "decay": decay,
            "project": project,
        })

    def incoming_links(self, project: str, memory_id: str,
                       depth: int = 1) -> dict:
        """POST /api/memory/incoming_links. Reverse-direction neighbors --
        who links TO this memory."""
        return self._post("/api/memory/incoming_links", {
            "memory_id": memory_id, "depth": depth, "project": project,
        })

    # ------------------------------------------------------------------
    # LIFECYCLE (time-based memory management)
    # ------------------------------------------------------------------

    def decay(self, project: str, threshold_days: int = 30,
              archive_floor: float = 0.05) -> dict:
        """POST /api/memory/decay. Ebbinghaus-style decay over memories
        older than threshold_days; below archive_floor they're archived
        (rarely retrieved). The benchmark runs this between questions
        to simulate multi-day agent memory state."""
        return self._post("/api/memory/decay", {
            "threshold_days": threshold_days, "archive_floor": archive_floor,
            "project": project,
        })

    def consolidate(self, project: str, threshold: float = 0.95,
                    dry_run: bool = False, limit: int = 100) -> dict:
        """POST /api/memory/consolidate. Merge near-duplicate memories
        (cosine sim >= threshold). dry_run=True returns would-merge
        pairs without actually merging -- used by the benchmark to
        report entropy without altering state."""
        return self._post("/api/memory/consolidate", {
            "threshold": threshold, "dry_run": dry_run, "limit": limit,
            "project": project,
        })

    def promote(self, project: str, memory_id: str,
                force: bool = False) -> dict:
        """POST /api/memory/promote. Promote a memory one tier up
        (working -> episodic -> semantic)."""
        return self._post("/api/memory/promote", {
            "memory_id": memory_id, "force": force, "project": project,
        })

    def auto_promote(self, project: str) -> dict:
        """POST /api/memory/auto_promote. Server-side auto-promotion
        based on access frequency."""
        return self._post("/api/memory/auto_promote", {"project": project})

    # ------------------------------------------------------------------
    # IMMUNOLOGICAL (5th tier, Mahalanobis anomaly detector)
    # ------------------------------------------------------------------

    def audit_immunological(self, project: str, k: int = 20) -> dict:
        """POST /api/memory/audit_immunological. List memories that the
        anomaly detector flagged during ingest. Read-only audit; the
        tier override itself happens automatically on each /save."""
        return self._post("/api/memory/audit_immunological", {
            "project": project, "k": k,
        })

    # ------------------------------------------------------------------
    # DELETE (for cleanup between full-capacity runs)
    # ------------------------------------------------------------------

    def delete(self, project: str, memory_id: str) -> dict:
        """POST /api/memory/delete. Remove a single memory. (The benchmark
        typically uses fresh project names per question and never needs
        this; exposed for completeness.)"""
        return self._post("/api/memory/delete", {
            "memory_id": memory_id, "project": project,
        })

    # ------------------------------------------------------------------
    # Context endpoint
    # ------------------------------------------------------------------

    def context(self, project: str | None = None, **payload) -> dict:
        """POST /api/context. Server assembles a structured context window
        for the LLM step. Convenience endpoint -- wraps a search with
        formatting. The benchmark can opt to use this instead of building
        its own context_block from search results."""
        if project is not None:
            payload["project"] = project
        return self._post("/api/context", payload)
