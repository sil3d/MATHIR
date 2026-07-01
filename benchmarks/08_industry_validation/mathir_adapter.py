#!/usr/bin/env python3
"""
MathirAdapter — thin HTTP client for the MATHIR daemon, used by the
LongMemEval / LoCoMo benchmark runners in benchmarks/08_industry_validation/.

Talks to the MATHIR Flask daemon (mathir_mcp/mathir_lib/mathir_server.py) at
http://127.0.0.1:7338 by default. Every request carries a `project` field —
MATHIR's `_resolve_db(project=..., cwd=...)` picks/creates a SQLite DB keyed
by project name, so giving each benchmark item (a LongMemEval question's
session history, or a LoCoMo conversation) its own unique project string
gives it a fully isolated memory namespace. No cleanup/deletion is needed
between test items — just use a fresh project name per item.

Exact route field names (confirmed by reading mathir_server.py directly,
not guessed):

POST /api/memory/save
    request:  {content, agent, block_type, label, priority, project, cwd}
    response: {memory_id, saved, metadata}
    NOTE: the save route does NOT accept a free-form `metadata` dict — it
    only recognizes agent/block_type/label/priority/content/project/cwd as
    top-level fields. See MathirAdapter.add() below for how this class
    reconciles that with its documented `metadata` parameter.

POST /api/memory/hybrid_search  (vector + BM25 + RRF fusion)
    request:  {query, k, vector_weight, bm25_weight, agent, project, cwd}
    response: {results: [{memory_id, rrf_score, content, agent, score,
               created_at, tier}], query, total, mode, vector_hits,
               bm25_hits}

POST /api/memory/recall  (plain vector search)
    request:  {query, k, agent, block_type, project, cwd}
    response: {results: [...], query, total, touched}

GET /api/ping
    response: {pong: true, uptime, dim}
    Loopback (127.0.0.1) requests need no auth header. Non-loopback hosts
    require a Bearer token (MATHIR_AUTH_TOKEN) — not relevant for local
    benchmarking against 127.0.0.1.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error


class MathirAdapter:
    """Minimal client for the MATHIR memory daemon, scoped for benchmark use."""

    def __init__(self, daemon_url: str = "http://127.0.0.1:7338"):
        self.daemon_url = daemon_url.rstrip("/")
        if not self.ping():
            raise RuntimeError(
                f"MATHIR daemon not reachable at {self.daemon_url}/api/ping.\n"
                f"Start it first, e.g.:\n"
                f"    python -m mathir_mcp\n"
                f"(or whatever entry point runs mathir_lib/mathir_server.py "
                f"in this repo), then retry."
            )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout: float = 60.0) -> dict:
        url = f"{self.daemon_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"MATHIR daemon returned HTTP {e.code} for POST {path}: {body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"MATHIR daemon unreachable at {url} ({e}). "
                f"Run `python -m mathir_mcp` and retry."
            ) from e

        try:
            result = json.loads(body)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"MATHIR daemon returned non-JSON response for POST {path}: {body!r}"
            ) from e

        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"MATHIR daemon error on POST {path}: {result['error']}")
        return result

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """True if the daemon responds to /api/ping, False otherwise (no exception)."""
        try:
            req = urllib.request.Request(f"{self.daemon_url}/api/ping", method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("pong"))
        except Exception:
            return False

    def add(
        self,
        project: str,
        content: str,
        agent: str = "benchmark",
        metadata: dict | None = None,
    ) -> str:
        """POST /api/memory/save with the given project namespace. Returns memory_id.

        The real /api/memory/save route only recognizes these top-level
        fields: content, agent, block_type, label, priority, project, cwd.
        There is no free-form metadata passthrough field on the server.
        So `metadata` here is reconciled as follows:
          - if metadata contains any of "block_type", "label", "priority",
            they override the defaults below (matching the route's own
            field names exactly).
          - any other keys in `metadata` are silently dropped, since the
            route has nowhere to put them. (Documented here per the task
            spec rather than guessing a new server-side field.)
        """
        payload = {
            "content": content,
            "agent": agent,
            "block_type": "episodic",
            "label": "",
            "priority": 5,
            "project": project,
        }
        if metadata:
            for key in ("block_type", "label", "priority"):
                if key in metadata:
                    payload[key] = metadata[key]

        result = self._post("/api/memory/save", payload)
        return result["memory_id"]

    def search(self, project: str, query: str, k: int = 10) -> list:
        """POST /api/memory/hybrid_search scoped to `project`.

        Returns a list of dicts (len <= k), each with at least a 'content'
        key, ranked by relevance (rrf_score / score descending). Other
        fields returned by the route (memory_id, agent, created_at, tier,
        rrf_score, score) are passed through unchanged.
        """
        payload = {"query": query, "k": k, "project": project}
        result = self._post("/api/memory/hybrid_search", payload)
        return result.get("results", [])
