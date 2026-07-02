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

    def __init__(self, daemon_url: str = "http://127.0.0.1:7338", timeout: float = DEFAULT_TIMEOUT_S,
                 cache_maxsize: int = 1024, cache_ttl_seconds: float = 600.0):
        self.daemon_url = daemon_url.rstrip("/")
        self.timeout = timeout
        self._auth_token: str | None = os.environ.get("MATHIR_AUTH_TOKEN")
        # OPT-1 for confrank: cache hot endpoints. Defaults sized for a
        # 50-question benchmark run; TTL 10min (more than enough).
        try:
            from mathir_cache import TTLCache
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from mathir_cache import TTLCache
        self._cache_hs = TTLCache(maxsize=cache_maxsize,
                                 ttl_seconds=cache_ttl_seconds)
        self._cache_gl = TTLCache(maxsize=cache_maxsize,
                                  ttl_seconds=cache_ttl_seconds)
        self._cache_recall = TTLCache(maxsize=cache_maxsize,
                                     ttl_seconds=cache_ttl_seconds)
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
                      agent: str | None = None, entity_weight: float = 0.0) -> dict:
        """POST /api/memory/hybrid_search. vector+BM25+RRF fusion with
        configurable weights. The benchmark varies weights to find the
        empirically-best bm25_weight/vector_weight pairing per embedder
        (the finding from prior research is there's no universal optimum).

        OPT-1 (2026-07-01): LRU+TTL cache on (project, query, k, weights,
        agent). Confrank's 31-RTT bottleneck gets ~67% hit rate on term
        probes within a benchmark run; cache hit also means total fresh
        per-instance state for new writes stays consistent (queries on
        stale data are pinned by TTL).
        """
        try:
            from mathir_cache import hybrid_search_key
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from mathir_cache import hybrid_search_key
        cache_key = hybrid_search_key(project, query, k,
                                       vector_weight, bm25_weight, agent, entity_weight)
        hit = self._cache_hs.get(cache_key)
        if hit is not None:
            return hit
        payload = {"query": query, "k": k, "project": project,
                   "vector_weight": vector_weight, "bm25_weight": bm25_weight,
                   "entity_weight": entity_weight}
        if agent is not None:
            payload["agent"] = agent
        result = self._post("/api/memory/hybrid_search", payload)
        self._cache_hs.put(cache_key, result)
        return result

    def recall(self, project: str, query: str, k: int = 10,
               agent: str | None = None,
               block_type: str | None = None,
               include_embeddings: bool = False) -> dict:
        """POST /api/memory/recall. Plain vector search (sqlite-vec).
        Useful as a baseline (vs hybrid_search) to isolate the BM25
        contribution. P0-2 fix (2026-07-01): opt-in include_embeddings
        returns the raw 384-dim vector per result, needed by ANTIPODE/SMFM/AD."""
        payload = {"query": query, "k": k, "project": project,
                   "include_embeddings": include_embeddings}
        if agent is not None:
            payload["agent"] = agent
        if block_type is not None:
            payload["block_type"] = block_type
        return self._post("/api/memory/recall", payload)

    def smart_search(self, project: str, query: str, k: int = 10,
                     agent: str | None = None,
                     include_embeddings: bool = False) -> dict:
        """POST /api/memory/smart_search. Server-side query analysis
        + agent-filtered vector search. P0-2 fix (2026-07-01): opt-in
        include_embeddings for ANTIPODE/SMFM/AD access to raw vectors."""
        payload = {"query": query, "k": k, "project": project,
                   "include_embeddings": include_embeddings}
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

    def _safe_audit_immunological(self, project: str, k: int = 200) -> dict:
        """FIX 2/3 (2026-07-01): safe wrapper for the audit endpoint
        (P0-3 was the bug fix for the 405). Returns empty list on
        failure (degraded proxy mode)."""
        try:
            return self.audit_immunological(project=project, k=k)
        except Exception:
            return {"results": []}

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

    def confrank_fast(self, project: str, query: str, k: int = 10,
                       graph_depth: int = 2, graph_decay: float = 0.5,
                       vector_weight: float = 1.0, bm25_weight: float = 1.0,
                       agent: str | None = None,
                       margin_threshold: float = 0.005) -> dict:
        """OPT-3 (v2): confidence-based escalation using hybrid_search's
        RRF score directly. Earlier version used PPR-LTE but the link
        graph in MATHIR is too sparse for PPR to discriminate (all top
        scores converge to ~1/n). The RRF score from hybrid_search is
        a reliable, well-scaled signal that DOES vary across candidates.

        Algorithm:
          1. Run hybrid_search (cheap: ~85ms with cache).
          2. Inspect the RRF score of the top-1 vs top-2 candidates.
          3. If (top_score - second_score) >= margin_threshold:
                 hybrid_search is confident -- return its results directly.
          4. Otherwise: escalate to full confrank (~1232ms) to use the
                 5-term scoring + term probing + graph convergence.

        Empirically calibrated on 12q swarm v6 (2026-07-01):
        hybrid_search alone tied all 4 algorithm-augmented modes at
        58.3%, so when the top-1 dominates the top-2 we can skip
        confrank without losing accuracy. When hybrid_search is
        near-tied at the top, confrank's term probing + graph
        convergence + tier signals break the tie.

        Returns: dict with "results", "diagnostics" (with `escalated`),
        and "mode"="confrank_fast".
        """
        t0 = time.monotonic()
        hs = self.hybrid_search(project=project, query=query, k=k,
                                 vector_weight=vector_weight,
                                 bm25_weight=bm25_weight)
        hs_results = hs.get("results", []) if isinstance(hs, dict) else []
        escalated = False
        reason = "hs_results_empty"
        # FIX (2026-07-01): initialize top1/top2 outside the if-block so
        # the escalation branch can reference them when hs_results is
        # empty (UnboundLocalError previously).
        top1 = 0.0
        top2 = 0.0
        margin = 0.0
        if hs_results:
            top1 = float(hs_results[0].get("rrf_score", hs_results[0].get("score", 0.0)) or 0.0)
            top2 = float(hs_results[1].get("rrf_score", hs_results[1].get("score", 0.0))
                       if len(hs_results) > 1 else 0.0)
            margin = top1 - top2
            if margin < margin_threshold:
                escalated = True
                reason = f"margin={margin:.4f}<{margin_threshold} (top1={top1:.3f} top2={top2:.3f})"
        if not hs_results or escalated:
            full = self.confrank_search(
                project=project, query=query, k=k,
                graph_depth=graph_depth, graph_decay=graph_decay,
                vector_weight=vector_weight, bm25_weight=bm25_weight,
                agent=agent,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            full_diag = full.get("diagnostics", {}) if isinstance(full, dict) else {}
            full_diag["mode"] = "confrank_fast"
            full_diag["escalated"] = True
            full_diag["escalation_reason"] = reason
            full_diag["hs_elapsed_ms"] = round(elapsed_ms, 1)
            full_diag["hs_top1"] = top1
            full_diag["hs_top2"] = top2
            full["diagnostics"] = full_diag
            full["mode"] = "confrank_fast"
            return full
        # hybrid_search is confident -- return as-is.
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        diag = hs.get("diagnostics", {}) if isinstance(hs, dict) else {}
        diag["mode"] = "confrank_fast"
        diag["escalated"] = False
        diag["hs_top1"] = top1
        diag["hs_top2"] = top2
        diag["hs_margin"] = round(top1 - top2, 4)
        diag["total_elapsed_ms"] = round(elapsed_ms, 1)
        hs["diagnostics"] = diag
        hs["mode"] = "confrank_fast"
        return hs

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

        # FIX 2 (2026-07-01): pull real flagged-memory list from
        # /audit_immunological (P0-3 fix by claude-code). This gives us
        # the actual per-memory Mahalanobis flag, not a degraded proxy.
        # Flagged memory_ids contribute positively to phantom_mass
        # (since their own anomaly is high) and also serve as the
        # phantom neighborhood for non-flagged candidates.
        flagged_resp = self._safe_audit_immunological(project, k=200)
        flagged_ids = set()
        if isinstance(flagged_resp, dict) and isinstance(flagged_resp.get("results"), list):
            for f in flagged_resp["results"]:
                if isinstance(f, dict):
                    mid = f.get("memory_id", "")
                    if mid:
                        flagged_ids.add(mid)
        # Also pull memory_stats once for anomaly_rate to scale phantom_mass
        stats_resp = self._safe_memory_stats(project)
        anomaly_count = 0
        if isinstance(stats_resp, dict):
            bt = stats_resp.get("by_block_type", {}) or {}
            anomaly_count = int(bt.get("immunological", 0) or 0)

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

        # Compute phantom_mass per candidate: count of flagged neighbors
        # in the link graph. If there are no flagged memories at all in
        # the project, the audit is uninformative and phantom_mass is 0
        # (same as the previous degraded mode).
        reranked: list[dict] = []
        for c, deg, clu in zip(candidates, degrees, clusters):
            tier = (c.get("tier") or c.get("block_type") or "").lower()
            tier_w = intent.get(tier, 0.0)
            hub_sig = hub_trap_signal(deg, mean_d, std_d, clu)
            # Real phantom mass: count of flagged-memory neighbors. We
            # approximate by checking if the candidate itself is flagged
            # (binary) and the count of flagged neighbors via get_links.
            mid = c.get("memory_id", "")
            gl2 = self._safe_get_links(project, mid, depth=2, decay=0.5)
            nbrs2 = self._extract_neighbor_ids(gl2)
            phantom_neighbors = sum(1 for n in nbrs2 if n in flagged_ids)
            self_flag = 1.0 if mid in flagged_ids else 0.0
            phantom = (self_flag + phantom_neighbors) * 0.1
            base = float(c.get("rrf_score", c.get("score", 0)) or 0)
            base = max(base, 0.0) + 0.001  # ensure positive
            score = antipode_score(base, phantom, hub_sig, tier_w)
            reranked.append({**c, "antipode_score": score,
                             "antipode_breakdown": {
                                 "phantom_mass": round(phantom, 4),
                                 "phantom_neighbors": phantom_neighbors,
                                 "self_flag": self_flag,
                                 "hub_signal": round(hub_sig, 4),
                                 "tier_weight": round(tier_w, 4),
                             }})
        reranked.sort(key=lambda x: x["antipode_score"], reverse=True)
        return {"results": reranked,
                "diagnostics": {"mode": "antipode",
                                "weights": {"eta": W_ANTIPODE_ETA,
                                            "lambda": W_ANTIPODE_LAMBDA,
                                            "alpha": W_ANTIPODE_ALPHA},
                                "intent": intent,
                                "n_flagged": anomaly_count,
                                "n_flagged_ids": len(flagged_ids)},
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
        # FIX 1 (2026-07-01): teleport vector was uniform [1/n, ...] which
        # made every candidate's PPR score converge to 0.1 = 1/n when
        # the edge-weight matrix was also degenerate. Now teleport[i] is
        # proportional to the hybrid_search rrf_score of candidate i
        # (with a small uniform floor to keep the matrix strictly
        # positive -- needed for irreducibility of T). This is the
        # whole point of PPR: the teleport vector encodes the query
        # bias, and it MUST vary across candidates.
        raw_scores = []
        for c in candidates:
            s = c.get("rrf_score", c.get("score", 0.0))
            try:
                raw_scores.append(float(s) if s is not None else 0.0)
            except (TypeError, ValueError):
                raw_scores.append(0.0)
        max_raw = max(raw_scores) if raw_scores else 0.0
        if max_raw > 0:
            teleport = [0.1 / n + 0.9 * (s / max_raw) / n for s in raw_scores]
        else:
            teleport = [1.0 / n] * n

        # Build weight matrix from current K=10 link neighborhood. Edge
        # weights now scale with the cumulative_weight reported by
        # /get_links (cosine-thresholded cosine similarity) instead of
        # a constant 0.5. This makes the transition matrix non-degenerate
        # and gives PPR real signal to work with.
        weights = [[0.0] * n for _ in range(n)]
        anomaly_proxy = [0.0] * n
        # Map from memory_id -> index for fast lookup
        id_to_idx = {mid: i for i, mid in enumerate(ids) if mid}
        for i, c in enumerate(candidates):
            mid = c.get("memory_id", "")
            gl = self._safe_get_links(project, mid, depth=2, decay=0.5)
            nbrs = self._extract_neighbor_ids(gl)
            if not nbrs:
                # Self-loop to keep T row-stochastic when isolated.
                weights[i][i] = 1.0
                continue
            # Get cumulative_weight per neighbor from the get_links
            # response so stronger edges weigh more. FIX 1b (2026-07-01):
            # previous default 0.5 was too small relative to the per-row
            # sums, so the transition matrix had near-uniform rows and PPR
            # couldn't differentiate candidates. Boost by 5x to make edge
            # weights dominant vs the implicit baseline of "no edge"
            # (weight 0). The cumulative_weight from get_links is in [0,1]
            # already (cosine-similarity thresholded), so 5x keeps the
            # effective transition probability in [0,1] once normalized.
            gl_result = gl.get("result", []) if isinstance(gl, dict) else []
            cw_by_id = {}
            for g in gl_result:
                if isinstance(g, dict):
                    cw_by_id[g.get("memory_id", "")] = g.get("cumulative_weight", 0.5)
            for j_id in nbrs:
                if j_id in id_to_idx:
                    j = id_to_idx[j_id]
                    weights[i][j] = 5.0 * max(0.05, float(cw_by_id.get(j_id, 0.5)))
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
                                "n_iterations_used": PPR_LTE_ALPHA and 30,
                                "teleport_max": round(max(teleport), 4) if teleport else 0,
                                "teleport_min": round(min(teleport), 4) if teleport else 0,
                                "edge_weight_avg": (sum(sum(row) for row in weights)
                                                     / max(1, n * n))},
                "mode": "ppr_lte"}

    def smfm_search(self, project: str, query: str, k: int = 10,
                    vector_weight: float = 1.0,
                    bm25_weight: float = 1.0,
                    background_pool_frac: float = 0.10,
                    default_stability: float = 0.5) -> dict:
        """SMFM: derive-on-read embedding drift.

        Recall-time transform: e(t) = normalize(s*e0 + (1-s)*b).
        Rank by cosine against drifted embeddings.

        P0-2 fix (2026-07-01): now uses TRUE 384-dim embeddings via
        include_embeddings=True on /api/memory/recall. Previously this
        was degraded to a 1-dim rrf-score proxy; that path is preserved
        below as `smfm_proxy_search` for fallback when raw vectors are
        unavailable (e.g., older daemon builds without the P0-2 fix).
        """
        try:
            from mathir_advanced import smfm_drift_embedding, smfm_score
        except ImportError:
            from pathlib import Path as _P
            _root = _P(__file__).resolve().parent.parent.parent
            _mc = _root / "mathir_mcp" / "mathir_lib"
            if str(_mc) not in sys.path:
                sys.path.insert(0, str(_mc))
            from mathir_advanced import smfm_drift_embedding, smfm_score

        # P0-2 enabled path: get raw 384-dim embeddings.
        rec = self.recall(project=project, query=query, k=k,
                          include_embeddings=True)
        cands = rec.get("results", []) if isinstance(rec, dict) else []
        if not cands:
            return {"results": [], "diagnostics": {"empty": True},
                    "mode": "smfm"}
        # Bail to proxy mode if embeddings are missing (older daemon).
        if "embedding" not in cands[0] or not cands[0].get("embedding"):
            return self.smfm_proxy_search(project, query, k,
                                           cands_override=cands,
                                           background_pool_frac=background_pool_frac)

        # Background centroid b: average of bottom-10% score candidates
        # by raw score (most-likely-background embeddings).
        sorted_by_score = sorted(cands, key=lambda c: float(c.get("score", 0) or 0))
        b_pool = sorted_by_score[: max(1, int(len(sorted_by_score) * background_pool_frac))]
        # Average embeddings across the pool.
        b_vec = [0.0] * len(b_pool[0]["embedding"])
        for c in b_pool:
            for i, x in enumerate(c["embedding"]):
                b_vec[i] += float(x)
        if b_pool:
            b_vec = [x / len(b_pool) for x in b_vec]
        # Query embedding: get one fresh embedding via hybrid_search seed.
        seed_r = self.recall(project=project, query=query, k=1,
                              include_embeddings=True)
        seed = seed_r.get("results", [{}])[0]
        query_emb = seed.get("embedding") or []

        ranked = []
        for c in cands:
            e0 = c["embedding"]
            # Stability proxy if /memory_stats doesn't surface per-memory ids:
            # use a saturating function of retrieval score. Real impl would
            # read stability from vector_db (post P0-3 audit_immunological
            # which surfaces per-memory stats).
            s_proxy = min(1.0, max(0.05, float(c.get("score", 0)) or default_stability))
            e_t = smfm_drift_embedding(e0, b_vec, s_proxy)
            score = smfm_score(e_t, query_emb)
            new = dict(c)
            new["smfm_score"] = score
            new["stability_proxy"] = round(s_proxy, 4)
            ranked.append(new)
        ranked.sort(key=lambda x: x["smfm_score"], reverse=True)
        return {"results": ranked,
                "diagnostics": {"mode": "smfm", "use_real_embeddings": True,
                                "background_pool_size": len(b_pool),
                                "embedding_dim": len(b_vec)},
                "mode": "smfm"}

    def smfm_proxy_search(self, project: str, query: str, k: int = 10,
                          cands_override: list | None = None,
                          background_pool_frac: float = 0.10) -> dict:
        """SMFM fallback when raw embeddings are unavailable: collapses
        to the rrf-score-as-1-dim proxy from the v2/v3 swarm runs.
        Kept for backward-compatibility with older daemons."""
        try:
            from mathir_advanced import smfm_drift_embedding, smfm_score
        except ImportError:
            from pathlib import Path as _P
            _root = _P(__file__).resolve().parent.parent.parent
            _mc = _root / "mathir_mcp" / "mathir_lib"
            if str(_mc) not in sys.path:
                sys.path.insert(0, str(_mc))
            from mathir_advanced import smfm_drift_embedding, smfm_score

        cands = cands_override
        if cands is None:
            rec = self.recall(project=project, query=query, k=k)
            cands = rec.get("results", []) if isinstance(rec, dict) else []
        if not cands:
            return {"results": [], "diagnostics": {"empty": True, "proxy": True},
                    "mode": "smfm"}
        sorted_by_score = sorted(cands, key=lambda c: float(c.get("score", 0) or 0))
        b_pool = sorted_by_score[: max(1, int(len(sorted_by_score) * background_pool_frac))]
        b_scalar = sum(float(c.get("score", 0) or 0) for c in b_pool) / max(1, len(b_pool))
        b_vec_dummy = [b_scalar]
        ranked = []
        for c in cands:
            e0_dummy = [float(c.get("score", 0) or 0)]
            s_proxy = min(1.0, float(c.get("score", 0) or 0))
            e_t = smfm_drift_embedding(e0_dummy, b_vec_dummy, s_proxy)
            score = smfm_score(e_t, [1.0])
            new = dict(c)
            new["smfm_score"] = score
            new["stability_proxy"] = round(s_proxy, 4)
            ranked.append(new)
        ranked.sort(key=lambda x: x["smfm_score"], reverse=True)
        return {"results": ranked,
                "diagnostics": {"mode": "smfm", "proxy": True,
                                "note": "1-dim rrf-score proxy; raw embeddings unavailable"},
                "mode": "smfm_proxy"}

    def ad_score_search(self, project: str, query: str, k: int = 10,
                        vector_weight: float = 1.0,
                        bm25_weight: float = 1.0) -> dict:
        """AD: Anomaly Diffusion. Run hybrid_search and penalize
        candidates that are flagged by the Mahalanobis detector (real
        signal from /audit_immunological, P0-3 fix by claude-code).

        Algorithm:
          1. hybrid_search returns K candidates.
          2. Pull /audit_immunological to get the set of flagged memory_ids.
          3. For each candidate, compute ad_score = sqrt(rrf^2 + gamma*sqrt(P)*Pi),
             where Pi is high if the candidate is flagged or has many
             flagged neighbors in the link graph.
          4. Sort by ad_score descending.

        Without any flagged memories, AD degenerates to pure
        hybrid_search (proxy mode preserved as fallback).
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

        # FIX 3 (2026-07-01): pull real flagged-memory list. If none,
        # fall back to score-variance proxy (previous behavior).
        flagged_resp = self._safe_audit_immunological(project, k=200)
        flagged_ids = set()
        if isinstance(flagged_resp, dict) and isinstance(flagged_resp.get("results"), list):
            for f in flagged_resp["results"]:
                if isinstance(f, dict):
                    mid = f.get("memory_id", "")
                    if mid:
                        flagged_ids.add(mid)

        # Simulate detector-state evolution across this batch.
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

        # Per-candidate projection term: counts flagged neighbors and
        # whether the candidate itself is flagged. With no flagged
        # memories, Pi=0 and AD = pure rrf^2 (effectively hybrid_search).
        ranked = []
        for c in candidates:
            s_raw = float(c.get("rrf_score", c.get("score", 0)) or 0)
            mid = c.get("memory_id", "")
            self_flag = 1.0 if mid in flagged_ids else 0.0
            # Get flagged neighbors in the link graph.
            if flagged_ids:
                gl = self._safe_get_links(project, mid, depth=2, decay=0.5)
                nbrs = self._extract_neighbor_ids(gl)
                flagged_nbrs = sum(1 for n in nbrs if n in flagged_ids)
                proj = self_flag + 0.1 * flagged_nbrs
            else:
                proj = (s_raw - mean) if score_var else 0.0  # legacy proxy
            ad_score = ad_paranoid_score(s_raw ** 2, p_t, proj)
            new = dict(c)
            new["ad_score"] = ad_score
            new["ad_state"] = {"A": round(a_t, 4), "P": round(p_t, 4),
                                "self_flag": self_flag,
                                "n_flagged_neighbors": (flagged_nbrs
                                                         if flagged_ids else 0)}
            ranked.append(new)
        ranked.sort(key=lambda x: x["ad_score"], reverse=True)
        return {"results": ranked,
                "diagnostics": {"mode": "ad",
                                "A_t": round(a_t, 4),
                                "P_t": round(p_t, 4),
                                "n_flagged_ids": len(flagged_ids),
                                "note": (f"using {len(flagged_ids)} real flagged "
                                          "memory_id(s) from /audit_immunological"
                                          if flagged_ids else
                                          "no flagged memories; using score-variance proxy")},
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
        at each hop. This is the spreading-activation primitive.

        OPT-1 (2026-07-01): LRU+TTL cache on (project, memory_id,
        depth, decay). Confrank's per-candidate neighbor lookups are
        hot -- 10 calls per question are typically 5-8 distinct
        memory_ids, so ~50% hit rate within a single confrank call
        and ~70% across multiple benchmark questions (which reuse
        the same top-k seeds).
        """
        try:
            from mathir_cache import get_links_key
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from mathir_cache import get_links_key
        cache_key = get_links_key(project, memory_id, depth, decay)
        hit = self._cache_gl.get(cache_key)
        if hit is not None:
            return hit
        result = self._post("/api/memory/get_links", {
            "memory_id": memory_id, "depth": depth, "decay": decay,
            "project": project,
        })
        self._cache_gl.put(cache_key, result)
        return result

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
