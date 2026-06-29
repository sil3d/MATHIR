#!/usr/bin/env python3
"""
MATHIR MCP Server v3 — Thin proxy to daemon (port 7338).
NO embedder loading — daemon handles all embedding.
Safe for multiple concurrent OpenCode sessions.
Keeps get_embedder/get_project_db_path/get_project_name for daemon compatibility.
"""

import hashlib
import json
import os
import sys
import logging
import urllib.request
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [MATHIR-MCP] %(levelname)s %(message)s",
)
log = logging.getLogger("mathir-mcp")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAEMON_URL = os.environ.get("MATHIR_DAEMON_URL", "http://127.0.0.1:7338")
MAX_QUERY_LENGTH = 5000
MAX_CONTENT_LENGTH = 100000
MAX_LABEL_LENGTH = 200
MAX_AGENT_LENGTH = 100

# Block types a client may write. "immunological" is reserved for the internal
# anomaly detector and is rejected on the save path.
_CLIENT_BLOCK_TYPES = {"working_memory", "episodic", "semantic", "procedural"}

try:
    from .mathir_paths import CONFIG_PATH as _P_CONFIG, PROJECTS_DIR as _P_PROJECTS
    from .mathir_paths import LEGACY_DB_PATH as _P_DB, REGISTRY_PATH as _P_REGISTRY
except ImportError:
    from mathir_paths import CONFIG_PATH as _P_CONFIG, PROJECTS_DIR as _P_PROJECTS
    from mathir_paths import LEGACY_DB_PATH as _P_DB, REGISTRY_PATH as _P_REGISTRY

CONFIG_PATH = Path(os.environ.get("MATHIR_CONFIG", str(_P_CONFIG)))
EMBEDDING_DIM = int(os.environ.get("MATHIR_EMBEDDING_DIM", "384"))
PROJECTS_DIR = Path(os.environ.get("MATHIR_PROJECTS_DIR", str(_P_PROJECTS)))
LEGACY_DB_PATH = Path(os.environ.get("MATHIR_DB", str(_P_DB)))
REGISTRY_PATH = Path(os.environ.get("MATHIR_REGISTRY", str(_P_REGISTRY)))


# ---------------------------------------------------------------------------
# Compatibility functions (used by mathir_server.py daemon)
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def get_project_name() -> str:
    """Auto-detect project from CWD."""
    config = load_config()
    if "project" in config:
        return config["project"]
    cwd = Path.cwd()
    # Check if CWD is under a known project
    for proj_dir in PROJECTS_DIR.iterdir() if PROJECTS_DIR.exists() else []:
        if cwd.is_relative_to(proj_dir):
            return proj_dir.name
    return cwd.name


def get_project_db_path(project: str = None) -> Optional[Path]:
    """Resolve DB path for project — matches original v2 logic."""
    # 1. CWD — prefer the project's own DB if it exists
    cwd_db = Path.cwd() / ".mathir" / "mathir.db"
    if cwd_db.exists():
        return cwd_db

    # 2. Registry — match CWD against known project roots, then fall through
    #    to the most-recently-used DB so legacy calls don't crash.
    if REGISTRY_PATH.exists():
        try:
            reg = json.loads(REGISTRY_PATH.read_text())
            projects = reg.get("projects", reg)  # support both {"projects":{}} and flat {}
            cwd = Path.cwd()
            # 2a. Project whose cwd is an ancestor of (or equal to) our CWD
            best_match = None
            best_match_len = -1
            for proj_name, info in projects.items():
                reg_cwd = info.get("cwd", "")
                if not reg_cwd:
                    continue
                reg_cwd_path = Path(reg_cwd)
                # Match: CWD is exactly the project cwd, OR CWD is inside it
                try:
                    cwd.relative_to(reg_cwd_path)
                    match_len = len(reg_cwd_path.parts)
                except ValueError:
                    continue
                if match_len > best_match_len:
                    best_match = info
                    best_match_len = match_len
            if best_match is not None:
                db = Path(best_match.get("db_path", ""))
                if db.exists():
                    return db
                # CWD matches a known project but its DB doesn't exist yet —
                # fall through so we create one for the project root (not cwd).
                return cwd_db
            # 2b. Fallback: most-recently-used DB that exists
            candidates = [
                (Path(info.get("db_path", "")), info.get("last_used", ""))
                for info in projects.values()
                if Path(info.get("db_path", "")).exists()
            ]
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
        except Exception:
            pass

    # 3. Projects dir — most recently modified
    if PROJECTS_DIR.exists():
        candidates = []
        for proj_dir in PROJECTS_DIR.iterdir():
            db = proj_dir / ".mathir" / "mathir.db"
            if db.exists():
                candidates.append((db.stat().st_mtime, db))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

    # 4. Legacy
    if LEGACY_DB_PATH.exists():
        return LEGACY_DB_PATH

    # 5. No DB exists — return the CWD path so the caller creates it
    #    (VecMemory store() handles .mathir/ directory creation)
    return cwd_db


def get_embedder():
    """Load embedder on demand (for daemon compatibility). CACHED."""
    global _cached_embedder
    if _cached_embedder is not None:
        return _cached_embedder
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = load_config().get("embedding", {}).get(
        "model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    _cached_embedder = SentenceTransformer(model_name, device=device)
    return _cached_embedder


_cached_embedder = None


def get_embedder_dim() -> int:
    embedder = get_embedder()
    if hasattr(embedder, 'dim'):
        return embedder.dim
    if hasattr(embedder, 'get_embedding_dimension'):
        return embedder.get_embedding_dimension()
    return EMBEDDING_DIM

# ---------------------------------------------------------------------------
# FastMCP
# ---------------------------------------------------------------------------
mcp = FastMCP("mathir-mcp")


# ---------------------------------------------------------------------------
# Helpers — forward to daemon via HTTP
# ---------------------------------------------------------------------------
def _call_daemon_raw(method: str, params: dict = None) -> dict:
    """Forward call to daemon HTTP API (no augmentation)."""
    # Remove None values and send as flat JSON
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    payload = json.dumps(clean).encode()

    # Map method names to daemon HTTP endpoints
    endpoint_map = {
        "memory_save": "/api/memory/save",
        "memory_recall": "/api/memory/recall",
        "memory_smart_search": "/api/memory/smart_search",
        "memory_hybrid_search": "/api/memory/hybrid_search",
        "memory_delete": "/api/memory/delete",
        "memory_stats": "/api/memory/stats",
        "memory_audit": "/api/memory/audit",
        "memory_export": "/api/memory/export",
        "memory_sessions": "/api/memory/sessions",
        "memory_promote": "/api/memory/promote",
        "memory_auto_promote": "/api/memory/auto_promote",
        "memory_decay": "/api/memory/decay",
        "memory_consolidate": "/api/memory/consolidate",
        "memory_link": "/api/memory/link",
        "memory_get_links": "/api/memory/get_links",
        "memory_build_links": "/api/memory/build_links",
        "memory_incoming_links": "/api/memory/incoming_links",
        "memory_context": "/api/context",
        "memory_session_start": "/api/context",
    }

    endpoint = endpoint_map.get(method, f"/api/memory/{method.replace('memory_', '')}")

    try:
        req = urllib.request.Request(
            f"{DAEMON_URL}{endpoint}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, 'read') else ''
        log.error(f"Daemon HTTP {e.code} on {endpoint}: {body}")
        return {"error": f"Daemon HTTP {e.code}: {body}"}
    except urllib.error.URLError as e:
        log.error(f"Daemon unreachable at {DAEMON_URL}: {e}")
        return {"error": f"Daemon unreachable: {e}"}
    except Exception as e:
        log.error(f"Daemon call failed: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Auto-recall — augment every tool response with related memories
# ---------------------------------------------------------------------------
# Methods where auto-recall is REDUNDANT (already returns matches) or NONSENSICAL
# (no textual content to query with). For everything else, we attach a
# `related_memories` top-3 to the response so the agent sees prior context
# every time it touches MATHIR — without having to call memory_recall itself.
_AUTO_RECALL_SKIP = {
    "memory_recall", "memory_smart_search", "memory_hybrid_search",
    "memory_context", "memory_session_start",
    "memory_stats", "memory_audit", "memory_export", "memory_sessions",
    "memory_auto_promote", "memory_decay", "memory_consolidate",
    "memory_get_links", "memory_build_links", "memory_delete",
    "memory_promote",   # memory_id is a key, not a query
}


def _extract_query(method: str, params: dict) -> Optional[str]:
    """Pick the best textual signal to use as recall query for this call."""
    if method == "memory_save":
        return params.get("content") or params.get("label")
    if method == "memory_link":
        # No content; use label-ish fields if any
        return params.get("source_id") or params.get("target_id")
    # Generic fallbacks
    for field in ("content", "query", "task", "session_title", "label"):
        v = params.get(field)
        if v and isinstance(v, str) and len(v) >= 10:
            return v
    return None


def _augment_response(method: str, params: dict, response: dict) -> dict:
    """Attach related_memories (top-3) to a successful tool response.

    Best-effort: any failure is swallowed so the main call never breaks.
    Threshold for "near-duplicate" is 0.92 — surfaced separately so the agent
    can decide whether the new save is redundant.
    """
    if not isinstance(response, dict) or "error" in response:
        return response
    if method in _AUTO_RECALL_SKIP:
        return response

    query = _extract_query(method, params or {})
    if not query or len(query) < 10:
        return response

    try:
        recall_resp = _call_daemon_raw("memory_recall", {
            "query": query[:MAX_QUERY_LENGTH],
            "k": 3,
            "agent": params.get("agent"),
        })
        results = recall_resp.get("results") if isinstance(recall_resp, dict) else None
        if not results:
            return response

        # Exclude the just-saved memory (self-match) when method == memory_save
        self_id = response.get("memory_id") if method == "memory_save" else None

        related = []
        near_duplicates = []
        for r in results[:5]:
            if self_id and r.get("memory_id") == self_id:
                continue  # skip self-match
            score = float(r.get("score", 0.0))
            meta = r.get("metadata") or {}
            item = {
                "memory_id": r.get("memory_id"),
                "label": meta.get("label", r.get("label", "")),
                "content": (meta.get("content") or r.get("content") or "")[:300],
                "score": round(score, 3),
                "agent": meta.get("agent", r.get("agent", "")),
                "block_type": meta.get("block_type", r.get("block_type", "")),
            }
            related.append(item)
            if score >= 0.92 and method == "memory_save":
                near_duplicates.append(item)
            if len(related) >= 3:
                break

        if related:
            response["related_memories"] = related
        if near_duplicates:
            response["near_duplicates"] = near_duplicates
            log.info(f"auto-recall {method}: {len(related)} related, "
                     f"{len(near_duplicates)} near-duplicate(s)")
        else:
            log.info(f"auto-recall {method}: {len(related)} related attached")
    except Exception as e:
        log.warning(f"auto-recall failed for {method} (non-fatal): {e}")

    return response


def _call_daemon(method: str, params: dict = None) -> dict:
    """Forward call to daemon HTTP API + attach auto-recall context."""
    response = _call_daemon_raw(method, params)
    return _augment_response(method, params, response)


def _check_lengths(**kwargs) -> Optional[dict]:
    """Validate field lengths."""
    limits = {
        "query": MAX_QUERY_LENGTH,
        "content": MAX_CONTENT_LENGTH,
        "label": MAX_LABEL_LENGTH,
        "agent": MAX_AGENT_LENGTH,
    }
    for field, limit in limits.items():
        val = kwargs.get(field)
        if val and len(str(val)) > limit:
            return {"error": f"{field} exceeds {limit} chars"}
    return None


def _sanitize_for_prompt(text: str) -> str:
    """Defang memory-sourced text before it is concatenated into an LLM prompt.

    Defends against stored prompt-injection from recalled memory content:
      - neutralize literal ``</mathir-...>`` closing tags so injected text
        cannot prematurely close any wrapping structure tag;
      - strip markdown heading markers (``### ``) that could impersonate
        prompt-section headers;
      - drop tokenizer special-token markers (``<|``) that some hosts may
        interpret as control tokens.
    Returned text is still readable but no longer tag/heading-shaped.
    """
    if not text:
        return ""
    s = text
    s = s.replace("</mathir-", "&lt;/mathir-")
    s = s.replace("### ", "")
    s = s.replace("<|", "")
    return s


def _content_hash(text: str) -> str:
    """Short SHA-8 fingerprint for log redaction (no cleartext leak)."""
    if not text:
        return "0"
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Tools — thin wrappers over daemon HTTP
# ---------------------------------------------------------------------------

def _auto_classify_block_type(content: str, label: str = "") -> str:
    """Heuristic auto-classification when caller passes block_type='auto'.

    Priority rules (first match wins):
    - procedural: starts with command syntax ($, #, --flag), mentions how-to/to/, step,
      recipe, run, install, configure, deploy
    - working_memory: very short (<200 chars) and contains 'TODO' or 'WIP' or 'draft'
    - semantic: looks like a fact (contains 'is/always/never', or has a definition pattern)
    - episodic (fallback): everything else — events, observations, decisions
    """
    text = (content or "").strip()
    label_lower = (label or "").lower()
    content_lower = text.lower()

    # procedural signals
    proc_signals = ("how-to:", "recipe:", "$ ", "# ", "pip install", "npm install",
                    "python -m", "cd ", "mkdir ", "git ", "docker ", "kubectl ",
                    " to ", " step ", " steps:", "install ", "deploy ", "configure ")
    if any(s in content_lower for s in proc_signals) or label_lower.startswith(("how-to:", "recipe:")):
        return "procedural"

    # working_memory signals (short + TODO-ish)
    if len(text) < 200 and any(s in content_lower for s in ("todo", "wip", "draft", "fixme", "xxx")):
        return "working_memory"

    # semantic signals (definitions, always/never, is a)
    semantic_signals = (" is ", " are ", " always ", " never ", "uses ", "uses:",
                        "based on ", "this is a ", "specifies ", "spec: ")
    if any(s in content_lower for s in semantic_signals) and len(text) < 800:
        return "semantic"

    # episodic fallback
    return "episodic"


@mcp.tool()
def memory_save(
    content: str,
    agent: str = "unknown",
    block_type: str = "episodic",
    label: str = "",
    priority: int = 5,
    project: str = None,
) -> str:
    """Save a memory. Block types: working_memory, episodic, semantic, procedural.

    Pass block_type="auto" to let MATHIR classify the content based on simple
    heuristics (commands/how-tos → procedural, bugs/decisions → episodic,
    facts/general knowledge → semantic, scratchpad → working_memory).
    """
    log.info(
        f"memory_save called: content_len={len(content)} "
        f"content_sha8={_content_hash(content)} label_len={len(label or '')} "
        f"label_sha8={_content_hash(label)} agent={agent} "
        f"block_type={block_type} priority={priority} project={project}"
    )
    _err = _check_lengths(content=content, label=label, agent=agent)
    if _err:
        return json.dumps(_err)
    if block_type == "immunological":
        return json.dumps({"error": "block_type 'immunological' is reserved for the internal anomaly detector and cannot be written by clients"})
    if block_type not in _CLIENT_BLOCK_TYPES and block_type != "auto":
        return json.dumps({"error": f"invalid block_type '{block_type}'. Valid: {sorted(_CLIENT_BLOCK_TYPES)} or 'auto'"})

    if block_type == "auto":
        block_type = _auto_classify_block_type(content, label)
        log.info(f"memory_save auto-classified → block_type={block_type}")

    params = {
        "content": content,
        "agent": agent,
        "block_type": block_type,
        "label": label,
        "priority": priority,
    }
    if project:
        params["project"] = project
    
    log.info(f"memory_save forwarding to daemon: {params.keys()}")
    result = _call_daemon("memory_save", params)
    # Log only structural keys + counts; never raw content/label (may be echoed
    # back via related_memories auto-recall).
    if isinstance(result, dict):
        safe_summary = {
            k: result.get(k)
            for k in ("memory_id", "status", "error")
            if k in result
        }
        if "related_memories" in result:
            safe_summary["related_count"] = len(result["related_memories"])
        if "near_duplicates" in result:
            safe_summary["near_dup_count"] = len(result["near_duplicates"])
        log.info(f"memory_save result: {safe_summary}")
    else:
        log.info("memory_save result: <non-dict>")
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_recall(
    query: str,
    k: int = 5,
    agent: str = None,
    block_type: str = None,
    project: str = None,
) -> str:
    """Search past memories by similarity."""
    _err = _check_lengths(query=query, agent=agent)
    if _err:
        return json.dumps(_err)

    result = _call_daemon("memory_recall", {
        "query": query,
        "k": k,
        "agent": agent,
        "block_type": block_type,
        "project": project,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_smart_search(
    query: str,
    k: int = 10,
    agent: str = None,
    project: str = None,
) -> str:
    """Hybrid semantic + keyword search with cross-lingual support."""
    _err = _check_lengths(query=query, agent=agent)
    if _err:
        return json.dumps(_err)

    result = _call_daemon("memory_smart_search", {
        "query": query,
        "k": k,
        "agent": agent,
        "project": project,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_hybrid_search(
    query: str,
    k: int = 5,
    agent: str = None,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    project: str = None,
) -> str:
    """Hybrid search: vector + BM25 + RRF fusion."""
    _err = _check_lengths(query=query, agent=agent)
    if _err:
        return json.dumps(_err)

    result = _call_daemon("memory_hybrid_search", {
        "query": query,
        "k": k,
        "agent": agent,
        "vector_weight": vector_weight,
        "bm25_weight": bm25_weight,
        "project": project,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_delete(
    memory_id: str,
    reason: str = "user requested",
) -> str:
    """Soft-delete a memory (sets tier to archived)."""
    result = _call_daemon("memory_delete", {
        "memory_id": memory_id,
        "reason": reason,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_stats(project: str = None) -> str:
    """Return memory counts by tier, agent, and project."""
    result = _call_daemon("memory_stats", {"project": project})
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_audit(agent: str = None, limit: int = 50) -> str:
    """Audit log of recent memory operations."""
    result = _call_daemon("memory_audit", {"agent": agent, "limit": limit})
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_export(project: str = None) -> str:
    """Export all memories as JSON."""
    result = _call_daemon("memory_export", {"project": project})
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_sessions(limit: int = 10) -> str:
    """List recent memory sessions."""
    result = _call_daemon("memory_sessions", {"limit": limit})
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_dashboard(action: str = "status") -> str:
    """Dashboard info."""
    result = _call_daemon("memory_stats", {})
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_context(task: str, project: str = None) -> str:
    """Get relevant memories for current task context."""
    result = _call_daemon("memory_context", {"task": task, "project": project})
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_session_start(session_title: str = "", project: str = None) -> str:
    """Start a memory session with relevant context."""
    result = _call_daemon("memory_session_start", {
        "task": session_title or "session start",
        "project": project,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


# --- Lifecycle tools ---

@mcp.tool()
def memory_promote(memory_id: str = None, force: bool = False) -> str:
    """Promote a memory to the next tier."""
    result = _call_daemon("memory_promote", {"memory_id": memory_id, "force": force})
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_auto_promote() -> str:
    """Auto-promote all eligible memories."""
    result = _call_daemon("memory_auto_promote", {})
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_decay(threshold_days: int = 30, archive_floor: float = 0.05) -> str:
    """Apply Ebbinghaus decay to stale memories."""
    result = _call_daemon("memory_decay", {
        "threshold_days": threshold_days,
        "archive_floor": archive_floor,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_consolidate(
    threshold: float = 0.95,
    dry_run: bool = False,
    limit: int = 1000,
) -> str:
    """Merge near-duplicate memories."""
    result = _call_daemon("memory_consolidate", {
        "threshold": threshold,
        "dry_run": dry_run,
        "limit": limit,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_link(
    source_id: str,
    target_id: str,
    weight: float = 1.0,
) -> str:
    """Add a link between two memories."""
    result = _call_daemon("memory_link", {
        "source_id": source_id,
        "target_id": target_id,
        "weight": weight,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_get_links(
    memory_id: str,
    depth: int = 2,
    decay: float = 0.5,
) -> str:
    """BFS traversal of memory link graph."""
    result = _call_daemon("memory_get_links", {
        "memory_id": memory_id,
        "depth": depth,
        "decay": decay,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_build_links(threshold: float = 0.7, limit: int = 1000) -> str:
    """Build link graph from cosine similarities."""
    result = _call_daemon("memory_build_links", {
        "threshold": threshold,
        "limit": limit,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


# ---------------------------------------------------------------------------
# Health check tool (no daemon needed)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Advanced tools — v8.5.1 enhancements
# file_path filter, recall quality signal, backlink graph
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_by_path(file_path: str, k: int = 10) -> str:
    """Search memories that reference a specific file path.

    Filters on metadata.file_path OR content matches against the path string.
    Use case: "show me what I know about mathir_vec.py:142" → returns all memories
    whose content or metadata mentions that file or location.
    """
    try:
        # Recall with the path as query, then post-filter
        recall = _call_daemon("memory_recall", {"query": file_path, "k": max(k * 3, 30)})
        if not isinstance(recall, dict) or "error" in recall:
            return json.dumps(recall if isinstance(recall, dict) else {"error": "recall failed"})

        results = recall.get("results", []) or []
        # Filter: keep only memories whose content/metadata contains the path
        # (either exact match, fuzzy substring, or .ext key)
        path_norm = file_path.replace("\\", "/").lower()
        bare_name = path_norm.rsplit("/", 1)[-1] if "/" in path_norm else path_norm
        out = []
        for r in results:
            meta = r.get("metadata") or {}
            content = str(meta.get("content", "") or r.get("content", ""))
            meta_path = str(meta.get("file_path", "") or meta.get("path", ""))
            cands = (content.lower(), meta_path.lower())
            if any(path_norm in c or bare_name in c for c in cands):
                out.append({
                    "memory_id": r.get("memory_id"),
                    "score": round(float(r.get("score", 0.0)), 3),
                    "label": meta.get("label", r.get("label", "")),
                    "block_type": meta.get("block_type", r.get("block_type", "")),
                    "file_path": meta_path,
                    "content_snippet": content[:200],
                    "agent": meta.get("agent", r.get("agent", "")),
                    "project": meta.get("project", r.get("project", "")),
                    "created_at": meta.get("created_at", ""),
                })
                if len(out) >= k:
                    break
        return json.dumps({"file_path": file_path, "total": len(out), "results": out})
    except Exception as e:
        return json.dumps({"error": _sanitize_error(e, "memory_by_path")})


@mcp.tool()
def memory_recall_quality(query: str, k: int = 5, min_score: float = 0.4) -> str:
    """Recall with explicit quality signal — tells you if your query is too vague.

    Returns top-k memories PLUS a `quality` field:
    - "high":   top-1 score ≥ 0.7
    - "medium": top-1 score ≥ min_score
    - "low":    top-1 score < min_score → DB doesn't have what you're looking for

    Use case: avoid rabbit holes when the DB can't answer your question.
    """
    try:
        recall = _call_daemon("memory_recall", {"query": query, "k": k})
        if not isinstance(recall, dict) or "error" in recall:
            return json.dumps(recall if isinstance(recall, dict) else {"error": "recall failed"})

        results = recall.get("results", []) or []
        if not results:
            return json.dumps({
                "query": query, "quality": "none", "total": 0,
                "suggestion": "No memories matched. Try rephrasing or saving knowledge first.",
                "results": [],
            })

        top1 = float(results[0].get("score", 0.0))
        if top1 >= 0.7:
            quality = "high"
            suggestion = "Strong match — top result is highly relevant."
        elif top1 >= min_score:
            quality = "medium"
            suggestion = "Partial match — review top results for relevance."
        else:
            quality = "low"
            suggestion = (
                f"Top-1 score {top1:.2f} < {min_score:.2f}. "
                "DB likely lacks what you need. Save new knowledge or broaden query."
            )

        # Re-shape results for clarity
        out = []
        for r in results:
            meta = r.get("metadata") or {}
            out.append({
                "memory_id": r.get("memory_id"),
                "score": round(float(r.get("score", 0.0)), 3),
                "label": meta.get("label", r.get("label", "")),
                "content_snippet": (str(meta.get("content", "") or r.get("content", "")))[:200],
                "agent": meta.get("agent", r.get("agent", "")),
                "block_type": meta.get("block_type", r.get("block_type", "")),
            })

        return json.dumps({
            "query": query,
            "quality": quality,
            "top1_score": round(top1, 3),
            "min_score": min_score,
            "total": len(out),
            "suggestion": suggestion,
            "results": out,
        })
    except Exception as e:
        return json.dumps({"error": _sanitize_error(e, "memory_recall_quality")})


@mcp.tool()
def memory_incoming_links(memory_id: str, depth: int = 1) -> str:
    """Get memories that point TO this memory_id (reverse link graph).

    Companion to memory_get_links (which is forward BFS). Useful for:
    - "what memories reference this fact?"
    - "is this memory a leaf or a hub in the link graph?"
    """
    result = _call_daemon("memory_incoming_links", {"memory_id": memory_id, "depth": depth})
    return json.dumps(result) if isinstance(result, dict) else str(result)


# ---------------------------------------------------------------------------
# Health check tool (no daemon needed)
# ---------------------------------------------------------------------------
@mcp.tool()
def mathir_health() -> str:
    """Check if MATHIR daemon is reachable."""
    try:
        req = urllib.request.Request(f"{DAEMON_URL}/ping")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        return json.dumps({"status": "ok", "daemon": data})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# Prompts capability — auto-fetched by MCP-prompt-aware hosts (Claude Desktop,
# Cursor, Cline, Roo, Continue, …) at session start. Universal MCP-native
# alternative to per-host plugin runtime injection.
# ---------------------------------------------------------------------------

@mcp.prompt()
def mathir_session_start(session_title: str = "") -> str:
    """MATHIR auto-context for this session — fetched at session start.

    Returns up to 8 memories relevant to the session title (or the most
    recent episodic memories if no title). Agents that support the MCP
    `prompts` capability will auto-invoke this; others can call it manually.
    """
    task = (session_title or "").strip() or "current session context"
    resp = _call_daemon_raw("memory_context", {"task": task, "k": 8})
    if not isinstance(resp, dict):
        return f"MATHIR: context unavailable ({resp})"
    context = resp.get("context")
    if not context:
        # Fallback: surface most recent episodic memories
        recent = _call_daemon_raw("memory_recall", {"query": task, "k": 5})
        if isinstance(recent, dict) and recent.get("results"):
            lines = [f"## MATHIR — {len(recent['results'])} recent memories"]
            for r in recent["results"]:
                meta = r.get("metadata") or {}
                agent = _sanitize_for_prompt(str(meta.get("agent", "?")))[:40]
                label = _sanitize_for_prompt(meta.get("label", ""))[:120]
                content = _sanitize_for_prompt((meta.get("content") or "")[:200])
                lines.append(f"- [{agent}] {label}:")
                # Quote each content line so injected memory text cannot
                # impersonate prompt structure or host instructions.
                for cline in content.splitlines() or [""]:
                    lines.append(f"> {cline}")
            context = "\n".join(lines)
        else:
            context = "MATHIR: no relevant memories found."
    # Also surface quick stats so the agent knows MATHIR is alive
    stats = _call_daemon_raw("memory_stats", {})
    stats_line = ""
    if isinstance(stats, dict) and not stats.get("error"):
        stats_line = f"\n\n_MATHIR stats: {stats}_"
    return f"{context}{stats_line}"


@mcp.prompt()
def mathir_recall(query: str, k: str | int = 5) -> str:
    """Pull specific memories matching a query — usable as a prompt template.

    `k` is intentionally typed as `str | int` because some MCP clients (notably
    Claude Desktop and Claude Code at time of writing) pass prompt arguments
    as raw strings even when the schema declares an integer. FastMCP then
    raises `ValidationError: int_parsing` and the prompt render fails. We
    coerce here so the prompt survives shell-variable interpolation.
    """
    try:
        k_int = int(k)
    except (TypeError, ValueError):
        k_int = 5
    if k_int < 1:
        k_int = 5
    resp = _call_daemon_raw("memory_recall", {"query": query, "k": k_int})
    if not isinstance(resp, dict) or not resp.get("results"):
        return f"MATHIR: no memories match '{query}'."
    safe_query = _sanitize_for_prompt(query)[:120]
    lines = [f"## MATHIR — {len(resp['results'])} memories for '{safe_query}'"]
    for r in resp["results"]:
        meta = r.get("metadata") or {}
        agent = _sanitize_for_prompt(str(meta.get("agent", "?")))[:40]
        label = _sanitize_for_prompt(meta.get("label", ""))[:120]
        content = _sanitize_for_prompt((meta.get("content") or "")[:200])
        try:
            score = float(r.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        lines.append(f"- [{agent}] (score {score:.2f}) {label}:")
        # Quote each content line so injected memory text cannot impersonate
        # prompt structure or host instructions.
        for cline in content.splitlines() or [""]:
            lines.append(f"> {cline}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Console-script entry point (`mathir-mcp`). Runs the FastMCP stdio server."""
    log.info(f"MATHIR MCP Server v3.1.0 (thin proxy to daemon at {DAEMON_URL})")
    log.info("No embedder loaded — daemon handles all embedding.")
    log.info("Prompts capability enabled: mathir_session_start, mathir_recall")
    mcp.run()


if __name__ == "__main__":
    main()
