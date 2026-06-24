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

# --- Import helpers: try bundled first, then external ---
def _import_mathir_vec():
    """Import mathir_vec from bundled or external location."""
    try:
        from mathir_mcp.mathir_lib import mathir_vec
        return mathir_vec
    except ImportError:
        from mathir_lib import mathir_vec
        return mathir_vec

# --- SECURITY: DoS protection via input length caps ---
# MCP stdin loop has no MAX_CONTENT_LENGTH guard. Daemon caps inputs at lines 27-34,
# but MCP bypasses daemon. These caps prevent OOM / CPU exhaustion via giant payloads.
# Tunable via MCP_INPUT_MAX env var (multiplier applied to all caps; 1.0 = default).
_input_scale = float(os.environ.get("MCP_INPUT_MAX", "1.0") or "1.0")
if _input_scale <= 0 or _input_scale > 100:
    _input_scale = 1.0  # out-of-range values fall back to default
MAX_CONTENT_LENGTH = int(100_000 * _input_scale)   # 100KB default
MAX_QUERY_LENGTH = int(5_000 * _input_scale)       # 5KB default
MAX_LABEL_LENGTH = int(200 * _input_scale)         # 200B default
MAX_AGENT_LENGTH = int(100 * _input_scale)         # 100B default
MAX_MEMORY_ID_LENGTH = int(64 * _input_scale)      # 64B default (e.g. "mem_xxxxxxxx")
MAX_PROJECT_LENGTH = int(200 * _input_scale)       # 200B default (project name)
MAX_REASON_LENGTH = int(500 * _input_scale)        # 500B default (delete reason)
# SECURITY: tight regex for memory_id. Generated IDs look like "mem_xxxxxxxx" (8 hex).
# Reject anything else so memory_id can never be interpolated unsafely into paths/commands.
import re as _re
_MEMORY_ID_RE = _re.compile(r"^[A-Za-z0-9_:-]{1,64}$")
# SECURITY: project name must be a safe filesystem identifier. Reject path traversal chars
# ("..", "/", "\\", NUL) and shell metacharacters even though project_name never reaches a shell.
_PROJECT_SAFE_RE = _re.compile(r"^[A-Za-z0-9._-]{1,200}$")


def _check_lengths(content, query, label, agent, memory_id=None, project=None, reason=None):
    """Enforce per-field length caps. Returns error dict on violation, None on OK."""
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


def _validate_memory_id(memory_id: str) -> dict | None:
    """Validate memory_id format. Returns error dict on violation, None on OK."""
    if memory_id is None:
        return {"error": "memory_id is required"}
    if not _MEMORY_ID_RE.match(str(memory_id)):
        return {"error": "memory_id must match [A-Za-z0-9_:-]{1,64}"}
    return None


def _validate_project(project: str) -> dict | None:
    """Validate project name. Rejects path-traversal and shell-meta chars."""
    if project is None:
        return None  # None = auto-detect from cwd (handled elsewhere)
    if not _PROJECT_SAFE_RE.match(str(project)):
        return {"error": "project must match [A-Za-z0-9._-]{1,200}"}
    return None

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
    
    # Try bundled mathir_dropin first, then external
    try:
        from mathir_mcp.mathir_dropin.memory import MATHIRMemory
    except ImportError:
        from mathir_dropin.memory import MATHIRMemory
    config = load_config()
    db_path = get_project_db_path(project_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    memory = MATHIRMemory(
        embedding_dim=EMBEDDING_DIM,
        config=config,
        db_path=str(db_path),
        provider="mathir-mcp",
        model=config.get("embedding", {}).get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
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
    model_name = config.get("embedding", {}).get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # PI/JETSON PATH: if MATHIR_USE_ONNX=1, try the lightweight ONNX runtime path.
    # Falls back to sentence-transformers if onnxruntime is unavailable so we
    # never block a desktop install that doesn't ship onnxruntime.
    use_onnx = os.environ.get("MATHIR_USE_ONNX", "").strip().lower() in ("1", "true", "yes")
    if use_onnx or prefer_octen:
        try:
            from mathir_onnx_embedder import OctenEmbedder
            model_dir = os.environ.get("MATHIR_ONNX_MODEL_DIR")
            _embedder = OctenEmbedder(model_dir=model_dir)
            log.info(f"Embedder loaded: ONNX/Octen INT8 (dim={_embedder.dim})")
            # NOTE: Octen outputs 1024d. If MATHIR_EMBEDDING_DIM is unset/384,
            # we MUST update it to 1024 or vector-shape errors will occur.
            if os.environ.get("MATHIR_EMBEDDING_DIM") is None:
                os.environ["MATHIR_EMBEDDING_DIM"] = "1024"
            _embedder_loaded_at = datetime.now().isoformat()
            return _embedder
        except Exception as e:
            log.warning(f"MATHIR_USE_ONNX=1 set but ONNX path failed: {e} — falling back to sentence-transformers")

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

    # CRITICAL: assert embedder dim matches EMBEDDING_DIM — mismatch causes runtime vector-shape errors
    model_dim = _embedder.get_embedding_dimension()
    expected_dim = int(os.environ.get("MATHIR_EMBEDDING_DIM", "384"))
    if model_dim != expected_dim:
        raise RuntimeError(
            f"Model {model_name} outputs {model_dim}d but MATHIR_EMBEDDING_DIM={expected_dim}. "
            f"Set MATHIR_EMBEDDING_DIM={model_dim} to match the loaded model."
        )

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
                    "enum": ["working_memory", "episodic", "semantic", "procedural", "immunological"],
                    "description": "Memory tier. immunological stores detected anomalies (prompt injections, suspicious patterns, threat signatures). It is both queryable and writable — save detected threats to it for pattern matching over time."
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
        "name": "memory_hybrid_search",
        "description": "Hybrid search combining vector similarity + BM25 keyword + RRF fusion (k=60). Best for: mixed semantic + exact term queries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                "vector_weight": {"type": "number", "default": 0.6, "minimum": 0, "maximum": 1},
                "bm25_weight": {"type": "number", "default": 0.4, "minimum": 0, "maximum": 1},
                "project": {"type": "string", "description": "Optional project filter"}
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
    },
    {
        "name": "memory_promote",
        "description": "Promote a memory to the next tier (working_memory → episodic → semantic → procedural). Uses Ebbinghaus rules: recall_count >= 3 and age >= 1d for working→episodic, recall_count >= 10 and age >= 7d for episodic→semantic, priority >= 8 + label prefix 'how-to:'/'recipe:' for semantic→procedural. Set force=true to bypass rules.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID of the memory to promote"},
                "force": {"type": "boolean", "default": False, "description": "Skip rule checks, promote unconditionally"}
            },
            "required": ["memory_id"]
        }
    },
    {
        "name": "memory_auto_promote",
        "description": "Scan all memories and auto-promote those that meet tier-transition rules. Returns list of promoted memories with old_tier → new_tier.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "memory_decay",
        "description": "Apply Ebbinghaus decay: reduce stability for memories not recalled recently (5%/30 days), archive those with stability < 0.05. Use threshold_days to control how aggressive the decay is. dry_run=true returns the plan without modifying anything.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "threshold_days": {"type": "integer", "default": 30, "description": "Days of inactivity before decay applies"},
                "archive_floor": {"type": "number", "default": 0.05, "description": "Stability below this → archived"},
                "dry_run": {"type": "boolean", "default": True, "description": "If true, return plan without modifying DB"}
            }
        }
    },
    {
        "name": "memory_consolidate",
        "description": "Merge near-duplicate memories (cosine > threshold). Returns merged pairs and tier distribution. dry_run=true shows the plan.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.95, "description": "Cosine similarity threshold for merging"},
                "limit": {"type": "integer", "default": 100, "description": "Max pairs to process"},
                "dry_run": {"type": "boolean", "default": True, "description": "If true, return plan without modifying DB"}
            }
        }
    },
    {
        "name": "memory_link",
        "description": "Add a link between two memories in the link graph. Links enable spreading activation during recall (1-2 hops, decay 0.5 per hop).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source memory ID"},
                "target_id": {"type": "string", "description": "Target memory ID"},
                "weight": {"type": "number", "default": 1.0, "description": "Link weight (0.0-1.0)"}
            },
            "required": ["source_id", "target_id"]
        }
    },
    {
        "name": "memory_get_links",
        "description": "BFS traversal of the link graph from a memory. Returns linked memories with distance and cumulative weight (decay**hops).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Starting memory ID"},
                "depth": {"type": "integer", "default": 1, "description": "Max hops (1-2)"},
                "decay": {"type": "number", "default": 0.5, "description": "Per-hop weight decay"}
            },
            "required": ["memory_id"]
        }
    },
    {
        "name": "memory_build_links",
        "description": "Build the link graph by scanning all memories and adding links between pairs with cosine > threshold. Use threshold=0.7 for broad associations. Idempotent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": 0.7, "description": "Cosine similarity threshold for linking"},
                "limit": {"type": "integer", "default": 1000, "description": "Max memories to scan"}
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

    # SECURITY: enforce input length caps to prevent DoS via unbounded payloads
    _len_err = _check_lengths(content=content, query=None, label=label, agent=agent, project=project)
    if _len_err is not None:
        return _len_err
    # SECURITY: validate project name (rejects path traversal / shell meta)
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err

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
    mathir_vec = _import_mathir_vec()
    vec_mem = mathir_vec.get_vec_memory(project, embedding_dim=EMBEDDING_DIM)
    
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

    # SECURITY: enforce input length caps to prevent DoS via unbounded queries
    _len_err = _check_lengths(content=None, query=query, label=None, agent=agent_filter, project=project)
    if _len_err is not None:
        return _len_err
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err

    # Generate query embedding
    query_embedding = embedder.encode(query)
    
    # Convert to numpy array (handle both torch tensors and numpy arrays)
    if hasattr(query_embedding, 'cpu'):
        query_np = query_embedding.cpu().numpy().astype(np.float32).reshape(-1)
    else:
        query_np = np.array(query_embedding, dtype=np.float32).reshape(-1)
    
    # Search using sqlite-vec accelerated memory
    mathir_vec = _import_mathir_vec()
    vec_mem = mathir_vec.get_vec_memory(project, embedding_dim=EMBEDDING_DIM)
    
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

    # SECURITY: validate project name and enforce length caps
    _len_err = _check_lengths(content=None, query=query, label=None, agent=agent_filter, project=project)
    if _len_err is not None:
        return _len_err
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err

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


def handle_memory_hybrid_search(args: dict) -> dict:
    """Hybrid search: vector similarity + BM25 keyword + RRF fusion.

    Delegates to the MATHIR daemon's `memory_hybrid_search` RPC (same path the
    CLI uses via `mathir_client.hybrid`). Falls back to a clear error if the
    daemon is not running so the caller can start it.
    """
    project = args.get("project")
    query = args["query"]
    k = args.get("k", 5)
    vector_weight = args.get("vector_weight", 0.6)
    bm25_weight = args.get("bm25_weight", 0.4)

    # SECURITY: validate project name and enforce length caps
    _len_err = _check_lengths(content=None, query=query, label=None, agent=None, project=project)
    if _len_err is not None:
        return _len_err
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err

    # Clamp k to the daemon's hard cap (100) to avoid silent truncation
    k = max(1, min(int(k), 100))

    # mathir_client is in the same dir (mathir_mcp/mathir_lib/) — delegate to daemon
    import mathir_client

    params = {
        "query": query,
        "k": k,
        "vector_weight": float(vector_weight),
        "bm25_weight": float(bm25_weight),
    }
    if project:
        params["project"] = project

    result = mathir_client.call("memory_hybrid_search", params)
    if isinstance(result, dict) and "error" in result:
        return {
            "error": result["error"],
            "hint": "Start the daemon with: python -m mathir_mcp",
            "query": query,
            "project": project or get_project_name(),
        }

    return result


def handle_memory_audit(args: dict) -> dict:
    project = args.get("project")
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
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
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
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
    memory_id = args["memory_id"]
    reason = args.get("reason", "user requested")
    # SECURITY: validate memory_id format and length, project name, and reason length
    _mid_err = _validate_memory_id(memory_id)
    if _mid_err is not None:
        return _mid_err
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
    _len_err = _check_lengths(content=None, query=None, label=None, agent=None, reason=reason)
    if _len_err is not None:
        return _len_err
    memory = get_memory(project)
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
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
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
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
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


# ============================================================================
# Memory lifecycle handlers (Phase 1-4: promote, decay, consolidate, link)
# ============================================================================

def handle_memory_promote(args: dict) -> dict:
    """Promote a memory to the next tier via Ebbinghaus rules."""
    project = args.get("project")
    memory_id = args["memory_id"]
    # SECURITY: validate memory_id format and project name
    _mid_err = _validate_memory_id(memory_id)
    if _mid_err is not None:
        return _mid_err
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
    memory = get_memory(project)
    force = args.get("force", False)
    result = memory.promote(memory_id, force=force)
    return {"result": result}


def handle_memory_auto_promote(args: dict) -> dict:
    """Scan all memories and auto-promote those meeting rules."""
    project = args.get("project")
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
    memory = get_memory(project)
    promoted = memory.auto_promote_all()
    return {"promoted": promoted, "count": len(promoted)}


def handle_memory_decay(args: dict) -> dict:
    """Apply Ebbinghaus decay (5%/30d), archive stability < 0.05."""
    project = args.get("project")
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
    memory = get_memory(project)
    result = memory.decay_all(
        threshold_days=args.get("threshold_days", 30),
        archive_floor=args.get("archive_floor", 0.05),
    )
    return {"result": result}


def handle_memory_consolidate(args: dict) -> dict:
    """Merge near-duplicate memories (cosine > threshold)."""
    project = args.get("project")
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
    memory = get_memory(project)
    result = memory.consolidate_all(
        threshold=args.get("threshold", 0.95),
        limit=args.get("limit", 100),
        dry_run=args.get("dry_run", True),
    )
    return {"result": result}


def handle_memory_link(args: dict) -> dict:
    """Add a link in the link graph."""
    project = args.get("project")
    source_id = args["source_id"]
    target_id = args["target_id"]
    # SECURITY: validate link IDs and project name
    _src_err = _validate_memory_id(source_id)
    if _src_err is not None:
        return {"error": f"source_id: {_src_err['error']}"}
    _tgt_err = _validate_memory_id(target_id)
    if _tgt_err is not None:
        return {"error": f"target_id: {_tgt_err['error']}"}
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
    memory = get_memory(project)
    result = memory.add_link(
        source_id=source_id,
        target_id=target_id,
        weight=args.get("weight", 1.0),
    )
    return {"result": result}


def handle_memory_get_links(args: dict) -> dict:
    """BFS traversal of the link graph."""
    project = args.get("project")
    memory_id = args["memory_id"]
    # SECURITY: validate memory_id and project name
    _mid_err = _validate_memory_id(memory_id)
    if _mid_err is not None:
        return _mid_err
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
    memory = get_memory(project)
    result = memory.get_links(
        memory_id=memory_id,
        depth=args.get("depth", 1),
        decay=args.get("decay", 0.5),
    )
    return {"result": result, "count": len(result)}


def handle_memory_build_links(args: dict) -> dict:
    """Build link graph from cosine > threshold across all memories."""
    project = args.get("project")
    # SECURITY: validate project name
    _proj_err = _validate_project(project)
    if _proj_err is not None:
        return _proj_err
    memory = get_memory(project)
    result = memory.build_links_all(
        threshold=args.get("threshold", 0.7),
        limit=args.get("limit", 1000),
    )
    return {"result": result}


TOOL_HANDLERS = {
    "memory_save": handle_memory_save,
    "memory_recall": handle_memory_recall,
    "memory_smart_search": handle_memory_smart_search,
    "memory_hybrid_search": handle_memory_hybrid_search,
    "memory_audit": handle_memory_audit,
    "memory_export": handle_memory_export,
    "memory_delete": handle_memory_delete,
    "memory_sessions": handle_memory_sessions,
    "memory_stats": handle_memory_stats,
    "memory_dashboard": handle_memory_dashboard,
    "memory_promote": handle_memory_promote,
    "memory_auto_promote": handle_memory_auto_promote,
    "memory_decay": handle_memory_decay,
    "memory_consolidate": handle_memory_consolidate,
    "memory_link": handle_memory_link,
    "memory_get_links": handle_memory_get_links,
    "memory_build_links": handle_memory_build_links,
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


def _check_install_health():
    """Verify the server is being run from a valid MATHIR installation.

    Checks:
      1. mathir_mcp_server.py is inside mathir_lib/ (correct nesting)
      2. mathir_mcp/ is the package root (pyproject.toml exists)
      3. mathir_lib/ has expected modules (mathir_vec.py, mathir_daemon.py)
    Returns (ok: bool, message: str).
    """
    here = Path(__file__).resolve().parent          # mathir_lib/
    pkg_root = here.parent                          # mathir_mcp/

    checks = []

    # 1. Correct nesting: mathir_lib/ inside mathir_mcp/
    if pkg_root.name == "mathir_mcp" and here.name == "mathir_lib":
        checks.append(("OK", "mathir_mcp/mathir_lib/ structure is correct"))
    else:
        checks.append(("WARN", f"Unexpected nesting: {here}"))

    # 2. pyproject.toml exists (package is valid)
    pyproject = pkg_root / "pyproject.toml"
    if pyproject.exists():
        checks.append(("OK", "pyproject.toml found"))
    else:
        checks.append(("WARN", f"pyproject.toml not found at {pkg_root}"))

    # 3. Key modules exist
    for mod in ["mathir_vec.py", "mathir_daemon.py", "mathir_client.py"]:
        if (here / mod).exists():
            checks.append(("OK", f"{mod} found"))
        else:
            checks.append(("WARN", f"{mod} not found in {here}"))

    # 4. Dashboard files exist
    html = here / "mathir_dashboard.html"
    server = here / "mathir_stats_server.py"
    if html.exists() and server.exists():
        checks.append(("OK", "Dashboard files found"))
    else:
        missing = []
        if not html.exists():
            missing.append("mathir_dashboard.html")
        if not server.exists():
            missing.append("mathir_stats_server.py")
        checks.append(("WARN", f"Dashboard files missing: {', '.join(missing)}"))

    # 5. Legacy DB exists
    home = Path.home()
    legacy_db = home / "MATHIR" / "database" / "mathir.cerdb"
    if legacy_db.exists():
        checks.append(("OK", f"Legacy DB: {legacy_db}"))
    else:
        checks.append(("INFO", f"No legacy DB at {legacy_db} (will create if needed)"))

    # Build report
    warnings = [msg for status, msg in checks if status == "WARN"]
    infos = [msg for status, msg in checks if status == "INFO"]

    if warnings:
        return False, (
            "MATHIR install health check FAILED:\n"
            + "\n".join(f"  ⚠ {w}" for w in warnings)
            + "\n\n"
            "SOLUTION: Relancez install.bat (Windows) ou ./install.sh (Linux/Mac) "
            "depuis le dossier mathir_mcp/ pour ré-injecter les configs."
        )

    if infos:
        log.info("Install health check: " + "; ".join(infos))

    return True, "OK"


def main():
    # ── Startup health check ──
    ok, msg = _check_install_health()
    if not ok:
        log.error(msg)
        # Still start — the server may work even if health check warns
        # But the warning is visible in stderr for the agent to surface

    log.info("MATHIR MCP Server starting...")
    log.info(f"DB: {LEGACY_DB_PATH}")
    log.info(f"Config: {CONFIG_PATH}")
    log.info(f"Embedding dim: {EMBEDDING_DIM}")
    try:
        get_memory()
        log.info("MATHIRMemory initialized successfully")
    except ModuleNotFoundError as e:
        if "mathir_dropin" in str(e):
            log.error(
                "mathir_dropin package not found on sys.path. The MATHIR MCP server "
                "depends on the mathir_dropin memory backend (sibling project).\n"
                "Fix one of these:\n"
                "  1. Install mathir_dropin as a package: `pip install -e ./mathir_dropin`\n"
                "  2. Add it to PYTHONPATH: `set PYTHONPATH=<repo_root>;%PYTHONPATH%` (where <repo_root> is the parent of mathir_mcp/ and mathir_dropin/)\n"
                "  3. Run mathir-mcp from a directory where mathir_dropin/ is reachable\n"
                f"Original error: {e}"
            )
        else:
            log.error(f"Failed to initialize MATHIRMemory: {e}", exc_info=True)
        sys.exit(1)
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
        # SECURITY: cap stdin line length to prevent OOM via a 10GB newline-less payload.
        # 1 MB is well above any legitimate JSON-RPC request (MCP tool args are field-capped).
        if len(line) > 1_048_576:
            log.error(f"Stdin line exceeds 1MB ({len(line)} bytes) — dropping")
            send_error(None, -32600, "Request too large")
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
