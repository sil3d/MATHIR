#!/usr/bin/env python3
"""
MATHIR MCP Server — 4-tier cognitive memory for OpenCode agents.
Exposes MATHIRMemory via MCP protocol (JSON-RPC over stdio).
Supports per-project databases with sqlite-vec acceleration.
"""

import json
import os
import sys
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any
import numpy as np

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [MATHIR-MCP] %(levelname)s %(message)s",
)
log = logging.getLogger("mathir-mcp")

CONFIG_PATH = Path(os.environ.get(
    "MATHIR_CONFIG",
    os.path.expanduser("~/.config/opencode/config/mathir.json")
))
EMBEDDING_DIM = int(os.environ.get("MATHIR_EMBEDDING_DIM", "384"))

# Per-project database support
PROJECTS_DIR = Path(os.environ.get(
    "MATHIR_PROJECTS_DIR",
    os.path.expanduser("~/.config/opencode/data/projects")
))
LEGACY_DB_PATH = Path(os.environ.get(
    "MATHIR_DB",
    os.path.expanduser("~/.config/opencode/data/mathir.db")
))
# Central registry - tracks all projects that have ever used MATHIR
REGISTRY_PATH = Path(os.environ.get(
    "MATHIR_REGISTRY",
    os.path.expanduser("~/.config/opencode/data/mathir_registry.json")
))

_memory_cache = {}
_embedder = None
_embedder_loaded_at = None


def get_project_name() -> str:
    """Get project name from current working directory or environment."""
    # Check environment variable first
    project = os.environ.get("MATHIR_PROJECT")
    if project:
        return project
    
    # Use current working directory name
    cwd = os.getcwd()
    project_name = Path(cwd).name
    
    # Create a short hash for uniqueness if needed
    return project_name


def get_project_db_path(project_name: str = None) -> Path:
    """Get database path for a specific project - stored IN the project directory.
    
    Priority:
    1. Check .mathir/mathir.db in current working directory (the project the agent is in)
    2. Check registry for known project name
    3. Scan common directories for project name
    4. Fall back to current working directory
    """
    if project_name is None:
        project_name = get_project_name()
    
    # FIRST: Check if there's a .mathir/mathir.db in the current working directory
    # But NOT if CWD is the home directory (that's not a real project)
    cwd = Path.cwd()
    home = Path.home()
    cwd_db = cwd / ".mathir" / "mathir.db"
    if cwd_db.exists() and cwd != home:
        log.info(f"Found .mathir/mathir.db in CWD: {cwd_db}")
        # Register for future lookups
        register_project(project_name, str(cwd_db), str(cwd))
        return cwd_db
    
    # SECOND: Check central registry
    registry = load_registry()
    if project_name in registry.get("projects", {}):
        info = registry["projects"][project_name]
        db_path = Path(info.get("db_path", ""))
        if db_path.exists():
            log.info(f"Found project '{project_name}' in registry: {db_path}")
            return db_path
    
    # THIRD: Scan MATHIR_SCAN_DIRS env var or common directories
    scan_dirs_env = os.environ.get("MATHIR_SCAN_DIRS", "")
    if scan_dirs_env:
        common_dirs = [Path(d) for d in scan_dirs_env.split(os.pathsep) if d.strip()]
    else:
        home = Path.home()
        common_dirs = [
            home / "Documents",
            home / "Projects",
            home / "dev",
            home / "Code",
        ]
    
    for parent_dir in common_dirs:
        if not parent_dir.exists():
            continue
        try:
            for item in parent_dir.iterdir():
                if item.is_dir() and item.name == project_name:
                    mathir_dir = item / ".mathir"
                    mathir_dir.mkdir(exist_ok=True)
                    db_path = mathir_dir / "mathir.db"
                    # Register for future lookups
                    register_project(project_name, str(db_path), str(item))
                    log.info(f"Found project '{project_name}' by scanning: {db_path}")
                    return db_path
        except PermissionError:
            continue
    
    # FALLBACK: Create in current working directory
    # But NOT in home directory — use first available project DB instead
    if cwd == home:
        registry = load_registry()
        for pname, info in registry.get("projects", {}).items():
            db_path = Path(info.get("db_path", ""))
            if db_path.exists():
                log.info(f"CWD is home, using first available project DB: {pname} -> {db_path}")
                return db_path
        log.warning(f"Project '{project_name}' not found and no registry DBs available.")
        return None
    
    mathir_dir = cwd / ".mathir"
    mathir_dir.mkdir(exist_ok=True)
    db_path = mathir_dir / "mathir.db"
    
    # Register for future lookups
    register_project(project_name, str(db_path), str(cwd))
    
    return db_path


def load_registry() -> dict:
    """Load the central project registry."""
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                return json.load(f)
        except Exception:
            return {"projects": {}}
    return {"projects": {}}


def save_registry(registry: dict):
    """Save the central project registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def register_project(project_name: str, db_path: str, cwd: str = None):
    """Register a project in the central registry."""
    # Don't register the home directory as a project
    actual_cwd = cwd or os.getcwd()
    if Path(actual_cwd) == Path.home():
        log.debug(f"Skipping registry: '{project_name}' is home directory")
        return
    registry = load_registry()
    registry["projects"][project_name] = {
        "db_path": db_path,
        "cwd": actual_cwd,
        "last_used": datetime.now().isoformat(),
        "name": project_name
    }
    save_registry(registry)
    log.info(f"Registered project '{project_name}' in registry")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def get_memory(project_name: str = None):
    """Get or create MATHIRMemory instance for a project."""
    if project_name is None:
        project_name = get_project_name()
    
    # Check cache
    if project_name in _memory_cache:
        return _memory_cache[project_name]
    
    from mathir_dropin.memory import MATHIRMemory
    config = load_config()
    db_path = get_project_db_path(project_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    memory = MATHIRMemory(
        embedding_dim=EMBEDDING_DIM,
        config=config,
        db_path=str(db_path),
        provider="mathir-mcp",
        model=config.get("embedding", {}).get("model", "BAAI/bge-large-en-v1.5"),
    )
    log.info(f"MATHIRMemory initialized for project '{project_name}': db={db_path}")
    _memory_cache[project_name] = memory
    
    # Register project in central registry
    register_project(project_name, str(db_path), os.getcwd())
    
    return memory


def get_embedder():
    global _embedder, _embedder_loaded_at
    if _embedder is not None:
        log.debug(f"Using cached embedder (loaded at {_embedder_loaded_at})")
        return _embedder
    
    config = load_config()
    prefer_octen = config.get("embedding", {}).get("prefer_octen", False)
    model_name = config.get("embedding", {}).get("model", "BAAI/bge-large-en-v1.5")
    
    import torch
    from sentence_transformers import SentenceTransformer
    
    if torch.cuda.is_available():
        device = "cuda"
        log.info(f"GPU detected: {torch.cuda.get_device_name(0)} — using CUDA acceleration")
    else:
        device = "cpu"
        log.info("No GPU — using CPU")
    
    _embedder = SentenceTransformer(model_name, device=device)
    log.info(f"Embedder loaded: {model_name} on {device}")
    
    _embedder_loaded_at = datetime.now().isoformat()
    return _embedder


TOOLS = [
    {
        "name": "memory_save",
        "description": "Save an insight, decision, fact, or observation to long-term memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The insight or decision to remember"},
                "agent": {"type": "string", "description": "Agent name (e.g. 'coder', 'swarm')"},
                "block_type": {
                    "type": "string",
                    "enum": ["working_memory", "episodic", "semantic", "procedural"],
                    "description": "Memory type"
                },
                "label": {"type": "string", "description": "Short label for this memory"},
                "priority": {"type": "integer", "description": "Priority 0-10", "default": 5},
                "project": {"type": "string", "description": "Project name (optional, auto-detected from cwd)"}
            },
            "required": ["content", "agent", "block_type", "label"]
        }
    },
    {
        "name": "memory_recall",
        "description": "Search past memories by similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "agent": {"type": "string", "description": "Filter by agent (optional)"},
                "k": {"type": "integer", "description": "Max results", "default": 5},
                "block_type": {"type": "string", "description": "Filter by type (optional)"},
                "project": {"type": "string", "description": "Project name (optional, auto-detected from cwd)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_smart_search",
        "description": "Hybrid semantic + keyword search with cross-lingual support.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (any language)"},
                "agent": {"type": "string", "description": "Filter by agent (optional)"},
                "k": {"type": "integer", "description": "Max results", "default": 10},
                "project": {"type": "string", "description": "Project name (optional, auto-detected from cwd)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_audit",
        "description": "View memory audit trail and statistics.",
        "inputSchema": {"type": "object", "properties": {
            "agent": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
            "project": {"type": "string", "description": "Project name (optional, auto-detected from cwd)"}
        }}
    },
    {
        "name": "memory_export",
        "description": "Export all memory data as JSON.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": "Project name (optional, auto-detected from cwd)"}
        }}
    },
    {
        "name": "memory_delete",
        "description": "Delete a memory by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "reason": {"type": "string", "default": "user requested"},
                "project": {"type": "string", "description": "Project name (optional, auto-detected from cwd)"}
            },
            "required": ["memory_id"]
        }
    },
    {
        "name": "memory_sessions",
        "description": "List recent memory sessions.",
        "inputSchema": {"type": "object", "properties": {
            "limit": {"type": "integer", "default": 10},
            "project": {"type": "string", "description": "Project name (optional, auto-detected from cwd)"}
        }}
    },
    {
        "name": "memory_stats",
        "description": "Get memory system statistics.",
        "inputSchema": {"type": "object", "properties": {
            "project": {"type": "string", "description": "Project name (optional, auto-detected from cwd)"}
        }}
    },
    {
        "name": "memory_dashboard",
        "description": "Launch or check the MATHIR Neural Memory Dashboard.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "start", "open"],
                    "description": "Action: status=check if running, start=launch server, open=open in browser",
                    "default": "status"
                }
            }
        }
    }
]


def handle_memory_save(args: dict) -> dict:
    project = args.get("project")
    embedder = get_embedder()
    content = args["content"]
    agent = args["agent"]
    block_type = args["block_type"]
    label = args["label"]
    priority = args.get("priority", 5)

    # Generate embedding
    embedding = embedder.encode(content)
    
    # Convert to numpy array (handle both torch tensors and numpy arrays)
    if hasattr(embedding, 'cpu'):
        embedding_np = embedding.cpu().numpy().astype(np.float32).reshape(-1)
    else:
        embedding_np = np.array(embedding, dtype=np.float32).reshape(-1)
    
    # Generate memory ID
    import uuid
    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    
    # Store in sqlite-vec accelerated memory
    from mathir_vec import get_vec_memory
    vec_mem = get_vec_memory(project, embedding_dim=EMBEDDING_DIM)
    
    metadata = {
        "agent": agent,
        "block_type": block_type,
        "label": label,
        "priority": priority,
        "content": content,
        "project": project or get_project_name(),
    }
    
    vec_mem.store(memory_id, embedding_np, metadata)
    
    log.info(f"Saved memory {memory_id}: [{agent}/{block_type}] {label} (project: {project or get_project_name()})")
    return {
        "memory_id": memory_id,
        "agent": agent,
        "block_type": block_type,
        "label": label,
        "content": content,
        "priority": priority,
        "project": project or get_project_name(),
        "timestamp": datetime.now().isoformat()
    }


def handle_memory_recall(args: dict) -> dict:
    project = args.get("project")
    embedder = get_embedder()
    query = args["query"]
    agent_filter = args.get("agent")
    k = args.get("k", 5)
    block_type_filter = args.get("block_type")

    # Generate query embedding
    query_embedding = embedder.encode(query)
    
    # Convert to numpy array (handle both torch tensors and numpy arrays)
    if hasattr(query_embedding, 'cpu'):
        query_np = query_embedding.cpu().numpy().astype(np.float32).reshape(-1)
    else:
        query_np = np.array(query_embedding, dtype=np.float32).reshape(-1)
    
    # Search using sqlite-vec accelerated memory
    from mathir_vec import get_vec_memory
    vec_mem = get_vec_memory(project, embedding_dim=EMBEDDING_DIM)
    
    results = vec_mem.search(
        query_embedding=query_np,
        k=k,
        agent_filter=agent_filter,
        block_type_filter=block_type_filter
    )
    
    return {"results": results, "query": query, "total": len(results), "project": project or get_project_name()}


def handle_memory_smart_search(args: dict) -> dict:
    project = args.get("project")
    memory = get_memory(project)
    query = args["query"]
    agent_filter = args.get("agent")
    k = args.get("k", 10)

    results = memory.universal_recall(query=query, k=k * 2)

    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        if agent_filter and meta.get("agent") != agent_filter:
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

    return {"results": filtered, "query": query, "total": len(filtered), "project": project or get_project_name()}


def handle_memory_audit(args: dict) -> dict:
    project = args.get("project")
    memory = get_memory(project)
    stats = memory.get_stats()
    db_path = get_project_db_path(project)
    return {
        "stats": stats,
        "db_path": str(db_path),
        "project": project or get_project_name(),
        "timestamp": datetime.now().isoformat()
    }


def handle_memory_export(args: dict) -> dict:
    import sqlite3
    project = args.get("project")
    db_path = get_project_db_path(project)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    export = {}
    # SECURITY: whitelist of allowed table names — never interpolate user input into SQL
    ALLOWED_TABLES = frozenset({"memory_blocks", "session_log", "memory_embeddings", "memory_embeddings_meta"})
    for table in ALLOWED_TABLES:
        try:
            # Table name is from a frozen whitelist, not user input — safe to interpolate
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            export[table] = [dict(r) for r in rows]
        except Exception:
            export[table] = []
    conn.close()
    return {"data": export, "timestamp": datetime.now().isoformat()}


def handle_memory_delete(args: dict) -> dict:
    project = args.get("project")
    memory = get_memory(project)
    memory_id = args["memory_id"]
    reason = args.get("reason", "user requested")
    deleted = memory.delete(memory_id)
    log.info(f"Deleted memory {memory_id}: {deleted} (reason: {reason}, project: {project or get_project_name()})")
    return {
        "memory_id": memory_id,
        "deleted": deleted,
        "reason": reason,
        "project": project or get_project_name(),
        "timestamp": datetime.now().isoformat()
    }


def handle_memory_sessions(args: dict) -> dict:
    import sqlite3
    project = args.get("project")
    limit = args.get("limit", 10)
    db_path = get_project_db_path(project)
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
    return {"sessions": sessions, "total": len(sessions), "project": project or get_project_name()}


def handle_memory_stats(args: dict) -> dict:
    project = args.get("project")
    memory = get_memory(project)
    stats = memory.get_stats()
    db_path = get_project_db_path(project)
    db_size = db_path.stat().st_size if db_path.exists() else 0
    return {
        "stats": stats,
        "db_path": str(db_path),
        "db_size_bytes": db_size,
        "embedding_dim": EMBEDDING_DIM,
        "project": project or get_project_name(),
        "timestamp": datetime.now().isoformat()
    }


def handle_memory_dashboard(args: dict) -> dict:
    import socket
    import subprocess
    import webbrowser
    from pathlib import Path
    
    action = args.get("action", "status")
    port = 7420
    stats_server = Path(__file__).parent / "mathir_stats_server.py"
    
    def is_port_open(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0
    
    def start_server():
        if not stats_server.exists():
            return {"error": f"Dashboard server not found at {stats_server}"}
        subprocess.Popen(
            [sys.executable, str(stats_server)],
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        import time
        time.sleep(2)
        return {"started": True, "url": f"http://127.0.0.1:{port}"}
    
    if action == "status":
        running = is_port_open(port)
        return {
            "running": running,
            "url": f"http://127.0.0.1:{port}" if running else None,
            "pid": None
        }
    elif action == "start":
        if is_port_open(port):
            return {"running": True, "url": f"http://127.0.0.1:{port}", "message": "Already running"}
        result = start_server()
        return result
    elif action == "open":
        if not is_port_open(port):
            start_server()
        webbrowser.open(f"http://127.0.0.1:{port}")
        return {"opened": True, "url": f"http://127.0.0.1:{port}"}
    else:
        return {"error": f"Unknown action: {action}"}


TOOL_HANDLERS = {
    "memory_save": handle_memory_save,
    "memory_recall": handle_memory_recall,
    "memory_smart_search": handle_memory_smart_search,
    "memory_audit": handle_memory_audit,
    "memory_export": handle_memory_export,
    "memory_delete": handle_memory_delete,
    "memory_sessions": handle_memory_sessions,
    "memory_stats": handle_memory_stats,
    "memory_dashboard": handle_memory_dashboard,
}


def send_response(response: dict):
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def send_error(request_id: Any, code: int, message: str):
    send_response({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message}
    })


def handle_request(request: dict):
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mathir-mcp", "version": "1.0.0"}
            }
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS}
        })
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        if tool_name not in TOOL_HANDLERS:
            send_error(req_id, -32602, f"Unknown tool: {tool_name}")
            return
        try:
            result = TOOL_HANDLERS[tool_name](tool_args)
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
                }
            })
        except Exception as e:
            log.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            send_error(req_id, -32000, f"Tool error: {str(e)}")
    elif method == "ping":
        send_response({"jsonrpc": "2.0", "id": req_id, "result": {}})
    else:
        send_error(req_id, -32601, f"Method not found: {method}")


def main():
    log.info("MATHIR MCP Server starting...")
    log.info(f"DB: {LEGACY_DB_PATH}")
    log.info(f"Config: {CONFIG_PATH}")
    log.info(f"Embedding dim: {EMBEDDING_DIM}")
    try:
        get_memory()
        log.info("MATHIRMemory initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize MATHIRMemory: {e}", exc_info=True)
        sys.exit(1)
    
    # Pre-warm embedder cache
    try:
        get_embedder()
        log.info("Embedder pre-warmed successfully")
    except Exception as e:
        log.error(f"Failed to pre-warm embedder: {e}", exc_info=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON: {e}")
            send_error(None, -32700, "Parse error")
            continue
        handle_request(request)


if __name__ == "__main__":
    main()
