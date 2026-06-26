#!/usr/bin/env python3
"""
MATHIR MCP Server v3 — Thin proxy to daemon (port 7338).
NO embedder loading — daemon handles all embedding.
Safe for multiple concurrent OpenCode sessions.
Keeps get_embedder/get_project_db_path/get_project_name for daemon compatibility.
"""

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

CONFIG_PATH = Path(os.environ.get(
    "MATHIR_CONFIG",
    os.path.expanduser("~/.config/opencode/config/mathir.json"),
))
EMBEDDING_DIM = int(os.environ.get("MATHIR_EMBEDDING_DIM", "384"))
PROJECTS_DIR = Path(os.environ.get(
    "MATHIR_PROJECTS_DIR",
    os.path.expanduser("~/.config/opencode/data/projects"),
))
LEGACY_DB_PATH = Path(os.environ.get(
    "MATHIR_DB",
    os.path.expanduser("~/.config/opencode/data/mathir.db"),
))
REGISTRY_PATH = Path(os.environ.get(
    "MATHIR_REGISTRY",
    os.path.expanduser("~/.config/opencode/data/mathir_registry.json"),
))


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
    # 1. Registry
    if REGISTRY_PATH.exists():
        try:
            reg = json.loads(REGISTRY_PATH.read_text())
            projects = reg.get("projects", reg)  # support both {"projects":{}} and flat {}
            for proj_name, info in projects.items():
                db = Path(info.get("db_path", ""))
                if db.exists():
                    return db
        except Exception:
            pass

    # 2. Projects dir — most recently modified
    if PROJECTS_DIR.exists():
        candidates = []
        for proj_dir in PROJECTS_DIR.iterdir():
            db = proj_dir / ".mathir" / "mathir.db"
            if db.exists():
                candidates.append((db.stat().st_mtime, db))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

    # 3. CWD
    cwd_db = Path.cwd() / ".mathir" / "mathir.db"
    if cwd_db.exists():
        return cwd_db

    # 4. Legacy
    if LEGACY_DB_PATH.exists():
        return LEGACY_DB_PATH

    return None


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
def _call_daemon(method: str, params: dict = None) -> dict:
    """Forward call to daemon HTTP API."""
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


# ---------------------------------------------------------------------------
# Tools — thin wrappers over daemon HTTP
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_save(
    content: str,
    agent: str = "unknown",
    block_type: str = "episodic",
    label: str = "",
    priority: int = 5,
    project: str = None,
) -> str:
    """Save a memory. Block types: working_memory, episodic, semantic, procedural."""
    log.info(f"memory_save called: content={content[:50]}... agent={agent} block_type={block_type} label={label} priority={priority} project={project}")
    _err = _check_lengths(content=content, label=label, agent=agent)
    if _err:
        return json.dumps(_err)

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
    log.info(f"memory_save result: {str(result)[:200]}")
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
        "session_title": session_title,
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
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info(f"MATHIR MCP Server v3.0.0 (thin proxy to daemon at {DAEMON_URL})")
    log.info("No embedder loaded — daemon handles all embedding.")
    mcp.run()
