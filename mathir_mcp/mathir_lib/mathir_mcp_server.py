#!/usr/bin/env python3
"""
MATHIR MCP Server v2 — FastMCP-based, single-process, no daemon bridge.
Lazy-loads embedder on first memory operation (fast startup).
Direct DB access via mathir_vec (sqlite-vec accelerated).
"""

import json
import os
import sys
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
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
CONFIG_PATH = Path(os.environ.get(
    "MATHIR_CONFIG",
    os.path.expanduser("~/.config/opencode/config/mathir.json"),
))
EMBEDDING_DIM = int(os.environ.get("MATHIR_EMBEDDING_DIM", "384"))

# Per-project database paths
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
# Security: input length caps
# ---------------------------------------------------------------------------
_input_scale = float(os.environ.get("MCP_INPUT_MAX", "1.0") or "1.0")
if _input_scale <= 0 or _input_scale > 100:
    _input_scale = 1.0
MAX_CONTENT_LENGTH = int(100_000 * _input_scale)
MAX_QUERY_LENGTH = int(5_000 * _input_scale)
MAX_LABEL_LENGTH = int(200 * _input_scale)
MAX_AGENT_LENGTH = int(100 * _input_scale)
MAX_MEMORY_ID_LENGTH = int(64 * _input_scale)
MAX_PROJECT_LENGTH = int(200 * _input_scale)
MAX_REASON_LENGTH = int(500 * _input_scale)
_MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9_:-]{1,64}$")
_PROJECT_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


def _check_lengths(content=None, query=None, label=None, agent=None,
                   memory_id=None, project=None, reason=None):
    for name, val, cap in (
        ("content", content, MAX_CONTENT_LENGTH),
        ("query", query, MAX_QUERY_LENGTH),
        ("label", label, MAX_LABEL_LENGTH),
        ("agent", agent, MAX_AGENT_LENGTH),
        ("memory_id", memory_id, MAX_MEMORY_ID_LENGTH),
        ("project", project, MAX_PROJECT_LENGTH),
        ("reason", reason, MAX_REASON_LENGTH),
    ):
        if val is not None and len(str(val)) > cap:
            return {"error": f"{name} exceeds {cap} chars (got {len(str(val))})"}
    return None


def _validate_memory_id(memory_id: str):
    if memory_id is None:
        return {"error": "memory_id is required"}
    if not _MEMORY_ID_RE.match(str(memory_id)):
        return {"error": "memory_id must match [A-Za-z0-9_:-]{1,64}"}
    return None


def _validate_project(project: str):
    if project is None:
        return None
    if not _PROJECT_SAFE_RE.match(str(project)):
        return {"error": "project must match [A-Za-z0-9._-]{1,200}"}
    return None


# ---------------------------------------------------------------------------
# Project registry (same logic as v1)
# ---------------------------------------------------------------------------
def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                return json.load(f)
        except Exception:
            return {"projects": {}}
    return {"projects": {}}


def save_registry(registry: dict):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def register_project(project_name: str, db_path: str, cwd: str = None):
    actual_cwd = cwd or os.getcwd()
    if Path(actual_cwd) == Path.home():
        return
    registry = load_registry()
    registry["projects"][project_name] = {
        "db_path": db_path,
        "cwd": actual_cwd,
        "last_used": datetime.now().isoformat(),
        "name": project_name,
    }
    save_registry(registry)
    log.info(f"Registered project '{project_name}' in registry")


def get_project_name() -> str:
    project = os.environ.get("MATHIR_PROJECT")
    if project:
        return project
    return Path(os.getcwd()).name


def get_project_db_path(project_name: str = None) -> Optional[Path]:
    if project_name is None:
        project_name = get_project_name()

    cwd = Path.cwd()
    home = Path.home()

    # 1. Check CWD/.mathir/mathir.db
    cwd_db = cwd / ".mathir" / "mathir.db"
    if cwd_db.exists() and cwd != home:
        register_project(project_name, str(cwd_db), str(cwd))
        return cwd_db

    # 2. Check registry
    registry = load_registry()
    if project_name in registry.get("projects", {}):
        info = registry["projects"][project_name]
        db_path = Path(info.get("db_path", ""))
        if db_path.exists():
            return db_path

    # 3. Scan common dirs
    scan_dirs_env = os.environ.get("MATHIR_SCAN_DIRS", "")
    if scan_dirs_env:
        common_dirs = [Path(d) for d in scan_dirs_env.split(os.pathsep) if d.strip()]
    else:
        common_dirs = [home / d for d in ("Documents", "Projects", "dev", "Code")]

    for parent_dir in common_dirs:
        if not parent_dir.exists():
            continue
        try:
            for item in parent_dir.iterdir():
                if item.is_dir() and item.name == project_name:
                    mathir_dir = item / ".mathir"
                    mathir_dir.mkdir(exist_ok=True)
                    db_path = mathir_dir / "mathir.db"
                    register_project(project_name, str(db_path), str(item))
                    return db_path
        except PermissionError:
            continue

    # 4. Fallback: if CWD is home, use first registry DB
    if cwd == home:
        for pname, info in registry.get("projects", {}).items():
            db_path = Path(info.get("db_path", ""))
            if db_path.exists():
                return db_path
        return None

    # 5. Create in CWD
    mathir_dir = cwd / ".mathir"
    try:
        mathir_dir.mkdir(exist_ok=True)
        db_path = mathir_dir / "mathir.db"
    except (PermissionError, OSError):
        fallback_dir = home / ".mathir" / "mathir_global"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        db_path = fallback_dir / "mathir.db"

    register_project(project_name, str(db_path), str(cwd))
    return db_path


# ---------------------------------------------------------------------------
# Lazy embedder (loaded on first memory operation, not at startup)
# ---------------------------------------------------------------------------
_embedder = None
_embedder_loaded_at = None


def get_embedder():
    global _embedder, _embedder_loaded_at
    if _embedder is not None:
        return _embedder

    log.info("Loading embedder (first use)...")
    config = load_config()
    model_name = config.get("embedding", {}).get(
        "model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ONNX path for Pi/Jetson
    use_onnx = os.environ.get("MATHIR_USE_ONNX", "").strip().lower() in ("1", "true", "yes")
    prefer_octen = config.get("embedding", {}).get("prefer_octen", False)
    if use_onnx or prefer_octen:
        try:
            from mathir_onnx_embedder import OctenEmbedder
            model_dir = os.environ.get("MATHIR_ONNX_MODEL_DIR")
            _embedder = OctenEmbedder(model_dir=model_dir)
            log.info(f"Embedder loaded: ONNX/Octen INT8 (dim={_embedder.dim})")
            if os.environ.get("MATHIR_EMBEDDING_DIM") is None:
                os.environ["MATHIR_EMBEDDING_DIM"] = "1024"
            _embedder_loaded_at = datetime.now().isoformat()
            return _embedder
        except Exception as e:
            log.warning(f"ONNX path failed: {e} — falling back to sentence-transformers")

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    _embedder = SentenceTransformer(model_name, device=device)
    model_dim = _embedder.get_embedding_dimension()
    expected_dim = int(os.environ.get("MATHIR_EMBEDDING_DIM", "384"))
    if model_dim != expected_dim:
        raise RuntimeError(
            f"Model outputs {model_dim}d but MATHIR_EMBEDDING_DIM={expected_dim}"
        )

    log.info(f"Embedder loaded: {model_name} on {device} (dim={model_dim})")
    _embedder_loaded_at = datetime.now().isoformat()
    return _embedder


def _encode_to_np(text: str) -> np.ndarray:
    """Encode text to float32 numpy vector."""
    embedder = get_embedder()
    emb = embedder.encode(text)
    if hasattr(emb, "cpu"):
        return emb.cpu().numpy().astype(np.float32).reshape(-1)
    return np.array(emb, dtype=np.float32).reshape(-1)


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# VecMemory cache
# ---------------------------------------------------------------------------
_memory_cache = {}


def _get_vec(project: str = None):
    """Get VecMemory for project (cached)."""
    if project in _memory_cache:
        return _memory_cache[project]

    # Lazy import: mathir_vec is in the same directory
    sys.path.insert(0, str(Path(__file__).parent))
    import mathir_vec

    db_path = get_project_db_path(project)
    if db_path is None:
        raise RuntimeError(f"No database found for project '{project or get_project_name()}'")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    vec = mathir_vec.get_vec_memory(project, embedding_dim=EMBEDDING_DIM)
    _memory_cache[project] = vec
    return vec


def _get_memory(project: str = None):
    """Get MATHIRMemory for lifecycle operations (promote, decay, etc.)."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from mathir_mcp.mathir_dropin.memory import MATHIRMemory
    except ImportError:
        try:
            from mathir_dropin.memory import MATHIRMemory
        except ImportError:
            # Fallback: use mathir_vec directly for basic ops
            return None

    config = load_config()
    db_path = get_project_db_path(project)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return MATHIRMemory(
        embedding_dim=EMBEDDING_DIM,
        config=config,
        db_path=str(db_path),
        provider="mathir-mcp",
        model=config.get("embedding", {}).get("model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    )


# ===========================================================================
# FastMCP server
# ===========================================================================
mcp = FastMCP(
    name="mathir-mcp",
    instructions=(
        "MATHIR — Memory Architecture for Tiered Heuristic Intelligent Retrieval. "
        "5-tier cognitive memory system (working_memory, episodic, semantic, procedural, immunological) "
        "with Ebbinghaus forgetting, promotion, consolidation, and link graph."
    ),
)


# ---------------------------------------------------------------------------
# memory_save
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_save(
    content: str,
    agent: str,
    block_type: str,
    label: str,
    priority: int = 5,
    project: str = None,
) -> str:
    """Save an insight, decision, fact, or observation to long-term memory.

    Args:
        content: The insight or decision to remember
        agent: Agent name (e.g. 'coder', 'swarm')
        block_type: Memory tier: working_memory | episodic | semantic | procedural | immunological
        label: Short label for this memory
        priority: Priority 0-10 (default 5)
        project: Project name (optional, auto-detected from cwd)
    """
    _err = _check_lengths(content=content, label=label, agent=agent, project=project)
    if _err:
        return json.dumps(_err)
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)
    if block_type not in ("working_memory", "episodic", "semantic", "procedural", "immunological"):
        return json.dumps({"error": f"Invalid block_type: {block_type}"})

    embedding_np = _encode_to_np(content)

    import uuid
    memory_id = f"mem_{uuid.uuid4().hex[:8]}"

    vec = _get_vec(project)
    metadata = {
        "agent": agent,
        "block_type": block_type,
        "label": label,
        "priority": priority,
        "content": content,
        "project": project or get_project_name(),
    }
    vec.store(memory_id, embedding_np, metadata)

    proj = project or get_project_name()
    log.info(f"Saved {memory_id}: [{agent}/{block_type}] {label} (project: {proj})")
    return json.dumps({
        "memory_id": memory_id, "agent": agent, "block_type": block_type,
        "label": label, "content": content, "priority": priority,
        "project": proj, "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# memory_recall
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_recall(
    query: str,
    k: int = 5,
    agent: str = None,
    block_type: str = None,
    project: str = None,
) -> str:
    """Search past memories by similarity.

    Args:
        query: Search query
        k: Max results (default 5)
        agent: Filter by agent (optional)
        block_type: Filter by type (optional)
        project: Project name (optional, auto-detected from cwd)
    """
    _err = _check_lengths(query=query, agent=agent, project=project)
    if _err:
        return json.dumps(_err)
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)

    query_np = _encode_to_np(query)
    vec = _get_vec(project)
    results = vec.search(
        query_embedding=query_np, k=k,
        agent_filter=agent, block_type_filter=block_type,
    )
    return json.dumps({
        "results": results, "query": query,
        "total": len(results), "project": project or get_project_name(),
    })


# ---------------------------------------------------------------------------
# memory_smart_search
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_smart_search(
    query: str,
    k: int = 10,
    agent: str = None,
    project: str = None,
) -> str:
    """Hybrid semantic + keyword search with cross-lingual support.

    Args:
        query: Search query (any language)
        k: Max results (default 10)
        agent: Filter by agent (optional)
        project: Project name (optional, auto-detected from cwd)
    """
    _err = _check_lengths(query=query, agent=agent, project=project)
    if _err:
        return json.dumps(_err)
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)

    memory = _get_memory(project)
    if memory is None:
        # Fallback to vector search
        return memory_recall(query=query, k=k, agent=agent, project=project)

    results = memory.universal_recall(query=query, k=k * 2)
    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        if agent and meta.get("agent") != agent:
            continue
        filtered.append({
            "memory_id": r.get("memory_id", ""),
            "content": meta.get("content", r.get("text", "")),
            "agent": meta.get("agent", ""),
            "block_type": meta.get("block_type", ""),
            "label": meta.get("label", ""),
            "score": r.get("similarity", r.get("score", 0.0)),
            "project": meta.get("project", project or get_project_name()),
        })
        if len(filtered) >= k:
            break

    return json.dumps({
        "results": filtered, "query": query,
        "total": len(filtered), "project": project or get_project_name(),
    })


# ---------------------------------------------------------------------------
# memory_hybrid_search
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_hybrid_search(
    query: str,
    k: int = 5,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    project: str = None,
) -> str:
    """Hybrid search: vector similarity + BM25 keyword + RRF fusion.

    Args:
        query: Search query
        k: Max results (1-50, default 5)
        vector_weight: Weight for vector similarity (0-1, default 0.6)
        bm25_weight: Weight for BM25 keyword (0-1, default 0.4)
        project: Optional project filter
    """
    _err = _check_lengths(query=query, project=project)
    if _err:
        return json.dumps(_err)
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)

    k = max(1, min(int(k), 100))

    # Try daemon first (it has BM25), fallback to pure vector
    try:
        import mathir_client
        params = {"query": query, "k": k, "vector_weight": float(vector_weight), "bm25_weight": float(bm25_weight)}
        if project:
            params["project"] = project
        result = mathir_client.call("memory_hybrid_search", params)
        if isinstance(result, dict) and "error" not in result:
            return json.dumps(result)
    except Exception:
        pass

    # Fallback: pure vector search
    return memory_recall(query=query, k=k, project=project)


# ---------------------------------------------------------------------------
# memory_delete
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_delete(
    memory_id: str,
    reason: str = "user requested",
    project: str = None,
) -> str:
    """Delete a memory by ID.

    Args:
        memory_id: ID of the memory to delete
        reason: Reason for deletion (default: 'user requested')
        project: Project name (optional, auto-detected from cwd)
    """
    _err = _validate_memory_id(memory_id)
    if _err:
        return json.dumps(_err)
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)
    _err = _check_lengths(reason=reason, project=project)
    if _err:
        return json.dumps(_err)

    memory = _get_memory(project)
    if memory:
        deleted = memory.delete(memory_id)
    else:
        # Fallback: direct SQL delete
        import sqlite3
        db_path = get_project_db_path(project)
        if db_path and db_path.exists():
            conn = sqlite3.connect(str(db_path))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            deleted = conn.total_changes > 0
            conn.close()
        else:
            deleted = False

    log.info(f"Deleted {memory_id}: {deleted} (reason: {reason})")
    return json.dumps({
        "memory_id": memory_id, "deleted": deleted, "reason": reason,
        "project": project or get_project_name(), "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# memory_stats
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_stats(project: str = None) -> str:
    """Get memory system statistics.

    Args:
        project: Project name (optional, auto-detected from cwd)
    """
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)

    memory = _get_memory(project)
    if memory:
        stats = memory.get_stats()
    else:
        stats = {"error": "memory module not available"}

    db_path = get_project_db_path(project)
    db_size = db_path.stat().st_size if db_path and db_path.exists() else 0
    return json.dumps({
        "stats": stats, "db_path": str(db_path),
        "db_size_bytes": db_size, "embedding_dim": EMBEDDING_DIM,
        "project": project or get_project_name(), "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# memory_audit
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_audit(
    agent: str = None,
    limit: int = 50,
    project: str = None,
) -> str:
    """View memory audit trail and statistics.

    Args:
        agent: Filter by agent (optional)
        limit: Max entries (default 50)
        project: Project name (optional, auto-detected from cwd)
    """
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)

    memory = _get_memory(project)
    stats = memory.get_stats() if memory else {"error": "memory module not available"}
    db_path = get_project_db_path(project)
    return json.dumps({
        "stats": stats, "db_path": str(db_path),
        "project": project or get_project_name(), "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# memory_export
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_export(project: str = None) -> str:
    """Export all memory data as JSON.

    Args:
        project: Project name (optional, auto-detected from cwd)
    """
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)

    import sqlite3
    db_path = get_project_db_path(project)
    if not db_path or not db_path.exists():
        return json.dumps({"error": "No database found", "project": project or get_project_name()})

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    export = {}
    ALLOWED_TABLES = frozenset({"memory_blocks", "session_log", "memory_embeddings", "memory_embeddings_meta", "memories", "memory_links"})
    for table in ALLOWED_TABLES:
        try:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            export[table] = [dict(r) for r in rows]
        except Exception:
            export[table] = []
    conn.close()
    return json.dumps({"data": export, "timestamp": datetime.now().isoformat()})


# ---------------------------------------------------------------------------
# memory_sessions
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_sessions(limit: int = 10, project: str = None) -> str:
    """List recent memory sessions.

    Args:
        limit: Max sessions (default 10)
        project: Project name (optional, auto-detected from cwd)
    """
    _err = _validate_project(project)
    if _err:
        return json.dumps(_err)

    import sqlite3
    db_path = get_project_db_path(project)
    if not db_path or not db_path.exists():
        return json.dumps({"sessions": [], "total": 0})

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM session_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        sessions = [dict(r) for r in rows]
    except Exception:
        sessions = []
    conn.close()
    return json.dumps({"sessions": sessions, "total": len(sessions), "project": project or get_project_name()})


# ---------------------------------------------------------------------------
# memory_dashboard
# ---------------------------------------------------------------------------
@mcp.tool()
def memory_dashboard(action: str = "status") -> str:
    """Launch or check the MATHIR Neural Memory Dashboard.

    Args:
        action: 'status' = check if running, 'start' = launch server, 'open' = open in browser
    """
    import socket
    import subprocess
    import webbrowser

    port = 7420
    stats_server = Path(__file__).parent / "mathir_stats_server.py"

    def is_port_open(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    if action == "status":
        running = is_port_open(port)
        return json.dumps({"running": running, "url": f"http://127.0.0.1:{port}" if running else None})
    elif action == "start":
        if is_port_open(port):
            return json.dumps({"running": True, "url": f"http://127.0.0.1:{port}", "message": "Already running"})
        subprocess.Popen(
            [sys.executable, str(stats_server)],
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return json.dumps({"started": True, "url": f"http://127.0.0.1:{port}"})
    elif action == "open":
        if not is_port_open(port):
            subprocess.Popen(
                [sys.executable, str(stats_server)],
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        webbrowser.open(f"http://127.0.0.1:{port}")
        return json.dumps({"opened": True, "url": f"http://127.0.0.1:{port}"})
    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# ===========================================================================
# Memory lifecycle tools (promote, decay, consolidate, links)
# ===========================================================================

@mcp.tool()
def memory_promote(memory_id: str, force: bool = False) -> str:
    """Promote a memory to the next tier (working_memory → episodic → semantic → procedural).

    Ebbinghaus rules: recall>=3 + age>=1d for working→episodic, recall>=10 + age>=7d for
    episodic→semantic, priority>=8 + label prefix 'how-to:'/'recipe:' for semantic→procedural.
    Set force=true to bypass rules.

    Args:
        memory_id: ID of the memory to promote
        force: Skip rule checks, promote unconditionally (default false)
    """
    _err = _validate_memory_id(memory_id)
    if _err:
        return json.dumps(_err)

    memory = _get_memory()
    if memory is None:
        return json.dumps({"error": "memory module not available — cannot promote"})
    result = memory.promote(memory_id, force=force)
    return json.dumps({"result": result})


@mcp.tool()
def memory_auto_promote() -> str:
    """Scan all memories and auto-promote those that meet tier-transition rules.

    Returns list of promoted memories with old_tier → new_tier.
    """
    memory = _get_memory()
    if memory is None:
        return json.dumps({"error": "memory module not available"})
    promoted = memory.auto_promote_all()
    return json.dumps({"promoted": promoted, "count": len(promoted)})


@mcp.tool()
def memory_decay(
    threshold_days: int = 30,
    archive_floor: float = 0.05,
    dry_run: bool = True,
) -> str:
    """Apply Ebbinghaus decay: reduce stability for memories not recalled recently (5%/30 days).

    Archives memories with stability < archive_floor. Use threshold_days to control aggressiveness.
    dry_run=true returns the plan without modifying anything.

    Args:
        threshold_days: Days of inactivity before decay applies (default 30)
        archive_floor: Stability below this → archived (default 0.05)
        dry_run: If true, return plan without modifying DB (default true)
    """
    memory = _get_memory()
    if memory is None:
        return json.dumps({"error": "memory module not available"})
    result = memory.decay_all(threshold_days=threshold_days, archive_floor=archive_floor)
    return json.dumps({"result": result})


@mcp.tool()
def memory_consolidate(
    threshold: float = 0.95,
    limit: int = 100,
    dry_run: bool = True,
) -> str:
    """Merge near-duplicate memories (cosine > threshold).

    Returns merged pairs and tier distribution. dry_run=true shows the plan.

    Args:
        threshold: Cosine similarity threshold for merging (default 0.95)
        limit: Max pairs to process (default 100)
        dry_run: If true, return plan without modifying DB (default true)
    """
    memory = _get_memory()
    if memory is None:
        return json.dumps({"error": "memory module not available"})
    result = memory.consolidate_all(threshold=threshold, limit=limit, dry_run=dry_run)
    return json.dumps({"result": result})


@mcp.tool()
def memory_link(source_id: str, target_id: str, weight: float = 1.0) -> str:
    """Add a link between two memories in the link graph.

    Links enable spreading activation during recall (1-2 hops, decay 0.5 per hop).

    Args:
        source_id: Source memory ID
        target_id: Target memory ID
        weight: Link weight 0.0-1.0 (default 1.0)
    """
    _err = _validate_memory_id(source_id)
    if _err:
        return json.dumps({"error": f"source_id: {_err['error']}"})
    _err = _validate_memory_id(target_id)
    if _err:
        return json.dumps({"error": f"target_id: {_err['error']}"})

    memory = _get_memory()
    if memory is None:
        return json.dumps({"error": "memory module not available"})
    result = memory.add_link(source_id=source_id, target_id=target_id, weight=weight)
    return json.dumps({"result": result})


@mcp.tool()
def memory_get_links(memory_id: str, depth: int = 1, decay: float = 0.5) -> str:
    """BFS traversal of the link graph from a memory.

    Returns linked memories with distance and cumulative weight (decay**hops).

    Args:
        memory_id: Starting memory ID
        depth: Max hops (1-2, default 1)
        decay: Per-hop weight decay (default 0.5)
    """
    _err = _validate_memory_id(memory_id)
    if _err:
        return json.dumps(_err)

    memory = _get_memory()
    if memory is None:
        return json.dumps({"error": "memory module not available"})
    result = memory.get_links(memory_id=memory_id, depth=depth, decay=decay)
    return json.dumps({"result": result, "count": len(result)})


@mcp.tool()
def memory_build_links(threshold: float = 0.7, limit: int = 1000) -> str:
    """Build the link graph by scanning all memories and adding links between pairs with cosine > threshold.

    Idempotent — safe to run multiple times.

    Args:
        threshold: Cosine similarity threshold for linking (default 0.7)
        limit: Max memories to scan (default 1000)
    """
    memory = _get_memory()
    if memory is None:
        return json.dumps({"error": "memory module not available"})
    result = memory.build_links_all(threshold=threshold, limit=limit)
    return json.dumps({"result": result})


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    log.info("MATHIR MCP Server v2 (FastMCP) starting...")
    log.info(f"Embedding dim: {EMBEDDING_DIM}")
    log.info(f"Config: {CONFIG_PATH}")
    # Pre-warm embedder (25-30s on first load, cached after)
    try:
        get_embedder()
        log.info("Embedder pre-warmed successfully")
    except Exception as e:
        log.error(f"Embedder pre-warm failed: {e}")

    mcp.run()
