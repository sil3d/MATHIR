#!/usr/bin/env python3
"""
Thin HTTP adapter for benchmarking the REAL MATHIR daemon (port 7338) as a
BEIR-style retriever, isolated from the live agent-memory database.

Why isolation matters: the daemon resolves its SQLite DB per-request from
the `project` param you pass (see mathir_mcp/mathir_lib/mathir_server.py
`_resolve_db()`), which -- when given an explicit `project` name -- writes to
`~/.config/mathir/data/projects/<project>/mathir.db`, completely separate
from this repo's own `.mathir/mathir.db` used for the assistant's real
episodic/semantic memory. Every call in this module passes an explicit
`project` (e.g. "beir_bench_scifact") so benchmark inserts (thousands of
textbook/BEIR chunks) can NEVER pollute real agent memory. Verified via
`mathir_mcp/mathir_lib/mathir_server.py::_resolve_db()`.

This uses MATHIR's real, full-capability HTTP API directly (not the
MCP-tool-call path, which would be far too slow for thousands of inserts
in one conversation) -- the same underlying server, same embedder, same
tiers/decay/consolidation/anomaly-detection code path a live agent uses.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

HOST = "127.0.0.1"
PORT = 7338
BASE_URL = f"http://{HOST}:{PORT}"
TIMEOUT = 60

_ROUTES = {
    "memory_save": ("POST", "/api/memory/save"),
    "memory_recall": ("POST", "/api/memory/recall"),
    "memory_hybrid_search": ("POST", "/api/memory/hybrid_search"),
    "memory_smart_search": ("POST", "/api/memory/smart_search"),
    "memory_stats": ("GET", "/api/memory/stats"),
    "memory_decay": ("POST", "/api/memory/decay"),
    "memory_auto_promote": ("POST", "/api/memory/auto_promote"),
    "memory_consolidate": ("POST", "/api/memory/consolidate"),
    "memory_delete": ("POST", "/api/memory/delete"),
    "ping": ("GET", "/api/ping"),
}


def call(method: str, params: dict | None = None, retries: int = 2):
    params = params or {}
    http_method, route = _ROUTES[method]
    url = f"{BASE_URL}{route}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            if http_method == "GET":
                req = urllib.request.Request(url, method="GET")
            else:
                body = json.dumps(params).encode("utf-8")
                req = urllib.request.Request(
                    url, data=body, method="POST",
                    headers={"Content-Type": "application/json"},
                )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    return {"error": str(last_err)}


def ping() -> bool:
    r = call("ping")
    return bool(r.get("pong"))


class MathirBEIR:
    """Insert a BEIR corpus into an isolated MATHIR project and query it."""

    def __init__(self, project: str, agent: str = "beir_bench", block_type: str = "semantic"):
        self.project = project
        self.agent = agent
        self.block_type = block_type
        # doc text -> BEIR corpus _id, built at insert time so we can map
        # MATHIR's own memory_id back to the original corpus id regardless
        # of which retrieval endpoint we call (hybrid_search doesn't echo
        # back the `label` metadata field, memory_recall does -- content
        # match is 100% reliable across all endpoints since content is
        # always stored+echoed verbatim).
        self.text_to_docid: dict[str, str] = {}

    def already_populated(self, expected_count: int) -> bool:
        stats = call("memory_stats", {"project": self.project})
        n = stats.get("total_memories") or stats.get("total") or stats.get("count")
        return isinstance(n, int) and n >= expected_count

    def insert_corpus(self, corpus: dict, max_workers: int = 6, progress_every: int = 200):
        """corpus: BEIR corpus dict {doc_id: {"title":..., "text":...}}"""
        items = list(corpus.items())
        for doc_id, doc in items:
            self.text_to_docid[doc.get("text", "")] = doc_id

        n_done = 0
        n_err = 0
        t0 = time.time()

        def _save(doc_id, doc):
            content = doc.get("text", "")
            r = call(
                "memory_save",
                {
                    "content": content,
                    "agent": self.agent,
                    "block_type": self.block_type,
                    "label": doc_id,
                    "priority": 5,
                    "project": self.project,
                },
            )
            return doc_id, r

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_save, doc_id, doc) for doc_id, doc in items]
            for fut in as_completed(futures):
                doc_id, r = fut.result()
                n_done += 1
                if "error" in r:
                    n_err += 1
                if n_done % progress_every == 0:
                    elapsed = time.time() - t0
                    print(f"    inserted {n_done}/{len(items)} ({n_err} errors) [{elapsed:.1f}s]")

        elapsed = time.time() - t0
        print(f"    DONE inserting {len(items)} docs into project={self.project}: "
              f"{n_done - n_err} ok, {n_err} errors, {elapsed:.1f}s total")
        return n_done - n_err, n_err

    def search_recall(self, query: str, top_k: int = 100) -> dict:
        """Pure MATHIR semantic recall (memory_recall) -- what a live agent uses."""
        r = call("memory_recall", {"query": query, "k": top_k, "project": self.project})
        return self._to_scores(r)

    def search_hybrid(self, query: str, top_k: int = 100) -> dict:
        """MATHIR's own hybrid (vector + BM25, RRF-fused) search."""
        r = call(
            "memory_hybrid_search",
            {"query": query, "k": min(top_k, 100), "project": self.project},
        )
        return self._to_scores(r)

    def _to_scores(self, response: dict) -> dict:
        out = {}
        for item in response.get("results", []):
            content = item.get("content", "")
            doc_id = self.text_to_docid.get(content)
            if doc_id is None:
                continue
            score = item.get("score", item.get("rrf_score", 0.0))
            out[doc_id] = float(score)
        return out

    def decay(self, threshold_days: int = 0):
        return call("memory_decay", {"project": self.project, "threshold_days": threshold_days})

    def auto_promote(self):
        return call("memory_auto_promote", {"project": self.project})

    def consolidate(self, threshold: float = 0.95, dry_run: bool = False):
        return call("memory_consolidate", {"project": self.project, "threshold": threshold, "dry_run": dry_run})
