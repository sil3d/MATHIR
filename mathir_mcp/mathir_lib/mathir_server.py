#!/usr/bin/env python3
"""
MATHIR Unified Server â€” v8.5.0
Flask + Waitress: single process, single port (7338).
Combines: daemon (memory operations) + dashboard (stats) + health probe.

Why this replaces mathir_daemon.py + mathir_stats_server.py:
  - Raw TCP sockets are fragile (pipe buffer crashes, no error framing)
  - Two separate processes = coordination nightmare
  - Flask + Waitress is battle-tested, handles errors gracefully
  - Single port = MCP clients just work

Usage:
  python mathir_server.py                    # default port 7338
  python mathir_server.py --port 8080        # custom port
  python mathir_server.py --host 0.0.0.0     # bind all interfaces (with caution)
"""

import sys
import re
import os
import json
import time
import signal
import threading
import logging
import logging.handlers
import argparse
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# Disable tqdm/HF progress bars BEFORE transformers/sentence-transformers are
# ever imported. When this process is launched detached (auto_start.bat's
# `start "" /B`, or any launcher with no console), sys.stderr can be a handle
# that accepts writes but not the flush() tqdm issues on every refresh --
# that raises OSError: [Errno 22] Invalid argument from deep inside model
# loading (transformers.core_model_loading -> tqdm.std.status_printer) and
# was surfacing as a generic 500 on every memory endpoint that touches the
# embedder. Progress bars are useless on a headless daemon anyway.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

# ---------------------------------------------------------------------------
# Bootstrap path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# Also expose the parent dir so `mathir_lib` (the package, e.g. its
# __init__.py constants like GUARDRAIL_MAX_PER_PROJECT) is importable by
# name even in script mode -- `from . import X` fails outright when this
# file is run directly (no parent package), and a same-directory sibling
# import can't reach __init__.py either since __init__ isn't importable
# by that name. This line is what makes a `from mathir_lib import X`
# fallback actually work, in both script and package launch modes.
sys.path.insert(0, str(_HERE.parent))

# ---------------------------------------------------------------------------
# Logging â€” stderr + rotating file (independent of launcher pipe redirection
# so crash traces survive even when stdout/stderr are DEVNULL'd by a watchdog)
# ---------------------------------------------------------------------------
try:
    from .mathir_paths import LOG_DIR as _P_LOG, PROJECTS_DIR as _P_PROJECTS
    from .mathir_paths import LEGACY_DB_PATH as _P_DB, CONFIG_PATH as _P_CONFIG
    from .mathir_paths import REGISTRY_PATH as _P_REGISTRY, DATA_DIR as _P_DATA
except ImportError:
    from mathir_paths import LOG_DIR as _P_LOG, PROJECTS_DIR as _P_PROJECTS
    from mathir_paths import LEGACY_DB_PATH as _P_DB, CONFIG_PATH as _P_CONFIG
    from mathir_paths import REGISTRY_PATH as _P_REGISTRY, DATA_DIR as _P_DATA

try:
    from .mathir_sanitize import sanitize_line as _sanitize_line
except ImportError:
    from mathir_sanitize import sanitize_line as _sanitize_line

_LOG_DIR = Path(os.environ.get("MATHIR_LOG_DIR", str(_P_LOG)))
try:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    _LOG_DIR = Path(os.environ.get("TEMP", "/tmp")) / "mathir_logs"
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / "mathir_server.log"
_PID_PATH = _LOG_DIR / "mathir_server.pid"

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [MATHIR-SERVER] %(levelname)s %(message)s")
_sh = logging.StreamHandler(sys.stderr)
_sh.setFormatter(_fmt)
_root.addHandler(_sh)
try:
    _fh = logging.handlers.RotatingFileHandler(
        _LOG_PATH, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    _fh.setFormatter(_fmt)
    _root.addHandler(_fh)
except OSError:
    pass  # File logging unavailable â€” stderr still works
log = logging.getLogger("mathir-server")

# ---------------------------------------------------------------------------
# Security limits (carried over from daemon)
# ---------------------------------------------------------------------------
MAX_REQUEST_SIZE = 65536
MAX_CONTEXT_LENGTH = 50000
MAX_CONTENT_LENGTH = 100000
MAX_QUERY_LENGTH = 5000
MAX_LABEL_LENGTH = 500

# ---------------------------------------------------------------------------
# Imports â€” mathir_lib
# ---------------------------------------------------------------------------
from mathir_mcp_server import (
    get_embedder,
    get_model_prefixes,
    get_project_db_path,
    get_project_name,
)
from mathir_push import ContextAnalyzer, PushCache, context_hash, deduplicate_memories
from mathir_cache import embedding_cache, recall_cache, session_cache, cache_stats, invalidate_on_write

# Risk mitigation (optional)
try:
    from memory_risks import DomainClassifier, LeakageDetector, SycophancyDetector
    _risk_enabled = True
except ImportError:
    _risk_enabled = False

try:
    from .mathir_stats_server import load_config
except ImportError:
    from mathir_stats_server import load_config  # type: ignore[no-redef]

_anomaly_config = load_config().get("memory", {})
_ANOMALY_THRESHOLD = _anomaly_config.get("anomaly_threshold", 2.0)
_ANOMALY_WARMUP = _anomaly_config.get("anomaly_warmup_count", 30)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_push_cache = PushCache(ttl_seconds=300, max_size=200)
_push_analyzer = ContextAnalyzer()
_push_lock = threading.Lock()

_vec_cache = {}
_vec_cache_lock = threading.Lock()

# BM25 index cache for /api/memory/hybrid_search, keyed by db_path. Without
# this, every single hybrid_search call rebuilt a fresh BM25Okapi index from
# every row in `memories` -- O(corpus size) tokenization + corpus-statistics
# computation per query. Measured cost on real benchmark data: 566s for
# scifact's query set (5183 docs) vs 17s for an equivalent reference
# hybrid-RRF pass that caches its BM25 index -- a ~30x, purely
# architectural, avoidable cost.
#
# Invalidation strategy: cache is keyed by db_path and stamped with the
# `memories` table's row count at build time; a cheap `SELECT COUNT(*)`
# on each call detects whether the corpus has changed and triggers a
# rebuild only then. This is a row-count heuristic, not a full content
# hash (which would cost as much as the rebuild it's meant to avoid) --
# it correctly detects inserts/deletes (the common case: new memories
# being saved between searches), but will NOT detect the edge case of an
# UPDATE that changes existing row content without changing the row
# count. That's an accepted, documented limitation, not an oversight.
_bm25_cache = {}
_bm25_cache_lock = threading.Lock()

# Same row-count-invalidated caching strategy as _bm25_cache, for the
# optional entity-overlap fusion signal in hybrid_search (opt-in via
# entity_weight, default 0.0 -- no cost when unused).
_entity_cache = {}
_entity_cache_lock = threading.Lock()

_start_time = time.time()

# ---------------------------------------------------------------------------
# Auth gate â€” non-loopback binds REQUIRE MATHIR_AUTH_TOKEN (opt-in, non-breaking)
# ---------------------------------------------------------------------------
_AUTH_TOKEN = os.environ.get('MATHIR_AUTH_TOKEN', '') or ''


def _is_loopback(host: str) -> bool:
    """Return True for loopback / localhost binds that need no auth."""
    return host in ('', '127.0.0.1', '::1', 'localhost')


def _get_vec_mem(db_path, dim):
    key = (str(db_path), dim)
    with _vec_cache_lock:
        if key not in _vec_cache:
            from mathir_vec import VecMemory
            _vec_cache[key] = VecMemory(db_path, dim)
            log.info(f"VecMemory cached for {db_path.name} (dim={dim})")
        return _vec_cache[key]


def _resolve_db(project: str = None, cwd: str = None):
    """Resolve VecMemory + embedder. Returns (vec_mem, db_path, embedder) or raises.

    Routing priority (v8.6.1 â€” local-first, backward-compatible):
      1. cwd/.mathir/mathir.db if it already exists (per-project)
      2. Global ~/.config/MATHIR/data/projects/<project>/mathir.db if it exists (legacy)
      3. Create cwd/.mathir/mathir.db for NEW projects (prefer local going forward)
      4. Fallback -> get_project_db_path() (registry, legacy)
    """
    dim = get_embedder_dim()
    db_path = None
    if cwd:
        cwd_path = Path(cwd)
        local_db = cwd_path / ".mathir" / "mathir.db"
        if local_db.exists():
            db_path = local_db
        elif project:
            global_db = Path(os.environ.get("MATHIR_HOME", str(Path.home() / ".config" / "MATHIR"))) / "data" / "projects" / project / "mathir.db"
            if global_db.exists():
                db_path = global_db
            else:
                local_db.parent.mkdir(parents=True, exist_ok=True)
                db_path = local_db
        else:
            local_db.parent.mkdir(parents=True, exist_ok=True)
            db_path = local_db
    elif project:
        db_path = get_project_db_path(project=project)
    if db_path is None:
        db_path = get_project_db_path(project=project)
    if db_path is None:
        raise ValueError("No project database found. Set MATHIR_PROJECT env var or pass project + cwd.")
    vec_mem = _get_vec_mem(db_path, dim)
    # Pin this DB to whichever model it was first embedded with (or, for a
    # brand-new DB, to the current configured default). This is what lets
    # MATHIR's default embedder change (e.g. to e5-small) apply to NEW
    # projects only, without silently mixing embedding spaces in DBs that
    # already have vectors from the old model. See VecMemory.
    # ensure_embedding_model and MATHIR memory
    # embedder-swap-strongest-positive-result-hotpotqa.
    default_model = load_config().get("embedding", {}).get(
        "model", "intfloat/multilingual-e5-small"
    )
    resolved_model = vec_mem.ensure_embedding_model(default_model)
    embedder = get_embedder(resolved_model)
    query_prefix, passage_prefix = get_model_prefixes(resolved_model)
    embedder.mathir_query_prefix = query_prefix
    embedder.mathir_passage_prefix = passage_prefix
    return vec_mem, db_path, embedder


def _encode_query(embedder, query: str):
    import numpy as np
    prefix = getattr(embedder, 'mathir_query_prefix', '')
    full_text = prefix + query
    cached = embedding_cache.get(full_text)
    if cached is not None:
        return cached
    emb = embedder.encode(full_text)
    if hasattr(emb, 'cpu'):
        result = emb.cpu().numpy().astype('float32').reshape(-1)
    else:
        result = np.array(emb, dtype=np.float32).reshape(-1)
    embedding_cache.put(full_text, result)
    return result


def _encode_passage(embedder, text: str):
    import numpy as np
    prefix = getattr(embedder, 'mathir_passage_prefix', '')
    full_text = prefix + text
    cached = embedding_cache.get(full_text)
    if cached is not None:
        return cached
    emb = embedder.encode(full_text)
    if hasattr(emb, 'cpu'):
        result = emb.cpu().numpy().astype('float32').reshape(-1)
    else:
        result = np.array(emb, dtype=np.float32).reshape(-1)
    embedding_cache.put(full_text, result)
    return result


def get_embedder_dim():
    embedder = get_embedder()
    if hasattr(embedder, 'dim'):
        return embedder.dim
    if hasattr(embedder, 'get_embedding_dimension'):
        return embedder.get_embedding_dimension()
    if hasattr(embedder, 'get_sentence_embedding_dimension'):
        return embedder.get_sentence_embedding_dimension()
    return int(os.environ.get('MATHIR_EMBEDDING_DIM', '384'))


def _sanitize_error(exc, method):
    safe_types = (ValueError, KeyError, TypeError, OSError, PermissionError, FileNotFoundError)
    import sqlite3
    if isinstance(exc, (*safe_types, sqlite3.IntegrityError, sqlite3.OperationalError)):
        return f"{type(exc).__name__}: {str(exc)[:200]}"
    log.error(f"Error in {method}: {exc}", exc_info=True)
    return f"Internal error in {method}: {type(exc).__name__}"


def _validate_input(params: dict) -> Optional[str]:
    for field, cap in (
        ("context", MAX_CONTEXT_LENGTH),
        ("content", MAX_CONTENT_LENGTH),
        ("query", MAX_QUERY_LENGTH),
        ("label", MAX_LABEL_LENGTH),
    ):
        val = params.get(field, "")
        if isinstance(val, str) and len(val) > cap:
            return f"{field} exceeds max length ({cap} chars)"
    for mid_field in ('memory_id', 'source_id', 'target_id'):
        mid = params.get(mid_field, '')
        if isinstance(mid, str) and len(mid) > 64:
            return f"{mid_field} exceeds max length (64 chars)"
    k = params.get('k', 5)
    if not isinstance(k, int) or k < 0 or k > 1000:
        return "k must be an integer between 0 and 1000"
    return None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
from flask import Flask, request, jsonify, send_file, Response

app = Flask(__name__)


# ---------------------------------------------------------------------------
# CORS â€” allow browsers (Tauri webview, Vite dev :3000) to call MATHIR directly
# ---------------------------------------------------------------------------
@app.after_request
def _add_cors_headers(resp):
    origin = request.headers.get("Origin", "*")
    # Restrict to localhost variants â€” MATHIR is a local-only service
    allowed = ("http://localhost", "http://127.0.0.1", "tauri://", "https://tauri.localhost")
    if origin == "*" or any(origin.startswith(a) for a in allowed):
        resp.headers["Access-Control-Allow-Origin"] = origin if origin != "*" else "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        resp.headers["Access-Control-Max-Age"] = "3600"
    return resp


@app.route("/<path:any_path>", methods=["OPTIONS"])
def _cors_preflight(any_path):
    """Handle CORS preflight for all routes."""
    return ("", 204)

# ---------------------------------------------------------------------------
# Dashboard routes (from mathir_stats_server.py)
# ---------------------------------------------------------------------------

_HTML_PATH = _HERE / "mathir_dashboard.html"
_PROJECTS_DIR = Path(os.environ.get("MATHIR_PROJECTS_DIR", str(_P_PROJECTS)))
_LEGACY_DB = Path(os.environ.get("MATHIR_DB", str(_P_DB)))
_CONFIG_PATH = Path(os.environ.get("MATHIR_CONFIG", str(_P_CONFIG)))
_REGISTRY_PATH = Path(os.environ.get("MATHIR_REGISTRY", str(_P_REGISTRY)))


def _get_project_db(project_name=None):
    import sqlite3 as _sql
    if project_name is None or project_name == "legacy":
        if _LEGACY_DB.exists():
            return _sql.connect(str(_LEGACY_DB))
        return None
    if _REGISTRY_PATH.exists():
        try:
            with open(_REGISTRY_PATH) as f:
                reg = json.load(f)
            if project_name in reg.get("projects", {}):
                db_path = Path(reg["projects"][project_name].get("db_path", ""))
                if db_path.exists():
                    return _sql.connect(str(db_path))
        except Exception:
            pass
    return None


def _list_projects():
    projects = []
    seen = set()
    if _LEGACY_DB.exists():
        projects.append({
            "name": "legacy",
            "path": str(_LEGACY_DB),
            "size_bytes": _LEGACY_DB.stat().st_size,
        })
        seen.add(str(_LEGACY_DB))
    if _REGISTRY_PATH.exists():
        try:
            with open(_REGISTRY_PATH) as f:
                reg = json.load(f)
            for name, info in reg.get("projects", {}).items():
                db_path = Path(info.get("db_path", ""))
                if db_path.exists() and str(db_path) not in seen:
                    seen.add(str(db_path))
                    projects.append({
                        "name": name,
                        "path": str(db_path),
                        "size_bytes": db_path.stat().st_size,
                        "last_used": info.get("last_used", ""),
                    })
        except Exception:
            pass
    return projects


# --- Dashboard HTML ---
@app.route("/")
def dashboard():
    if _HTML_PATH.exists():
        return send_file(str(_HTML_PATH), mimetype="text/html")
    return Response("<h1>MATHIR Dashboard not found</h1>", status=404, mimetype="text/html")


# --- Stats API ---
@app.route("/api/stats")
def api_stats():
    project = request.args.get("project")
    conn = _get_project_db(project)
    if conn is None:
        return jsonify({"error": "No database found", "project": project})
    try:
        rows = conn.execute("SELECT metadata FROM memories WHERE metadata IS NOT NULL").fetchall()
    except Exception:
        rows = []
    # Also get block_type from top-level column if available
    try:
        columns = {col[1] for col in conn.execute("PRAGMA table_info(memories)").fetchall()}
        has_block_type_col = "block_type" in columns
        has_agent_col = "agent" in columns
    except Exception:
        has_block_type_col = False
        has_agent_col = False
    tiers = {"working": 0, "episodic": 0, "semantic": 0, "procedural": 0, "immunological": 0, "guardrail": 0, "unknown": 0}
    agents = {}
    total = 0
    for row in rows:
        try:
            meta = json.loads(row["metadata"])
            bt = meta.get("block_type", "unknown")
            agent = meta.get("agent", "unknown")
            tier = "working" if bt == "working_memory" else bt
            if tier not in tiers:
                tier = "unknown"
            tiers[tier] += 1
            agents[agent] = agents.get(agent, 0) + 1
            total += 1
        except Exception:
            tiers["unknown"] += 1
            total += 1
    # If metadata-based counts are 0 but memories exist, count from columns directly
    if total == 0:
        try:
            count_row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
            if count_row and count_row["cnt"] > 0:
                total = count_row["cnt"]
                if has_block_type_col:
                    bt_rows = conn.execute("SELECT block_type, COUNT(*) as cnt FROM memories GROUP BY block_type").fetchall()
                    for r in bt_rows:
                        bt = r["block_type"] or "unknown"
                        tier = "working" if bt == "working_memory" else bt
                        if tier not in tiers:
                            tier = "unknown"
                        tiers[tier] = r["cnt"]
                if has_agent_col:
                    ag_rows = conn.execute("SELECT agent, COUNT(*) as cnt FROM memories GROUP BY agent").fetchall()
                    for r in ag_rows:
                        agents[r["agent"] or "unknown"] = r["cnt"]
        except Exception:
            pass
    db_path = _get_project_db(project)
    db_size = 0
    if project:
        p = Path(project) if Path(project).exists() else None
        if p:
            db_size = p.stat().st_size
    conn.close()
    return jsonify({
        "total_memories": total,
        "tiers": tiers,
        "agents": agents,
        "db_size_bytes": db_size,
        "project": project,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


@app.route("/api/memories")
def api_memories():
    project = request.args.get("project")
    limit = min(int(request.args.get("limit", 100)), 500)
    offset = int(request.args.get("offset", 0))
    agent_filter = request.args.get("agent")
    tier_filter = request.args.get("tier")
    conn = _get_project_db(project)
    if conn is None:
        return jsonify({"error": "No database found"})
    # Detect schema: new (content) vs legacy (modality_text)
    columns = {col[1] for col in conn.execute("PRAGMA table_info(memories)").fetchall()}
    text_col = "content" if "content" in columns else "modality_text"
    ts_col = "created_at" if "created_at" in columns else "timestamp"
    agent_col = "agent" if "agent" in columns else None
    query = f"SELECT memory_id, {text_col}, metadata, tier, {ts_col} FROM memories WHERE 1=1"
    params = []
    if agent_filter:
        if agent_col:
            query += f" AND {agent_col} = ?"
        else:
            query += " AND json_extract(metadata, '$.agent') = ?"
        params.append(agent_filter)
    if tier_filter:
        if "block_type" in columns:
            query += " AND block_type = ?"
        else:
            query += " AND json_extract(metadata, '$.block_type') = ?"
        params.append(tier_filter)
    query += " ORDER BY rowid DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    try:
        rows = conn.execute(query, params).fetchall()
    except Exception:
        rows = []
    memories = []
    for row in rows:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
        except Exception:
            d["metadata"] = {}
        memories.append(d)
    conn.close()
    return jsonify({"memories": memories, "total": len(memories), "project": project})


def _sanitize_for_prompt(text: str) -> str:
    """Make memory text safe to embed in an LLM system prompt.

    Thin wrapper around mathir_sanitize.sanitize_line -- the single shared
    implementation also used by mathir_proxy.py. This function used to have
    its own independent (and buggy: `s.replace(tok, tok.strip())` is a
    no-op) copy of the same logic; see mathir_sanitize.py's module
    docstring for the incident. Do not reimplement this here again.
    """
    return _sanitize_line(text)


@app.route("/api/context", methods=["GET", "POST"])
def api_context():
    """Auto-injection endpoint for OpenCode plugins.
    Returns relevant memories formatted for system prompt injection.
    GET /api/context?task=description&k=8&project=name
    POST /api/context with JSON body: {"task": "...", "k": 8, "project": "..."}
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        task = data.get("task") or data.get("session_title") or ""
        k = min(int(data.get("k", 8)), 20)
        project = data.get("project")
    else:
        task = request.args.get("task") or request.args.get("session_title") or ""
        k = min(int(request.args.get("k", 8)), 20)
        project = request.args.get("project")
    if not task:
        return jsonify({"error": "task parameter required"}), 400
    _cwd = (request.args.get("cwd") if request.method == "GET"
            else (request.get_json(silent=True) or {}).get("cwd"))
    vec_mem = None
    try:
        cached_results = session_cache.get(project or "", task)
        if cached_results is not None:
            results = cached_results
        else:
            vec_mem, db_path, embedder = _resolve_db(project=project, cwd=_cwd)
            query_vec = _encode_query(embedder, task)
            results = vec_mem.search(query_vec, k=k)
            session_cache.put(project or "", results)
        # Normalize results to dicts with metadata
        normalized = []
        for r in results:
            d = {
                "memory_id": r.get("memory_id", ""),
                "score": r.get("score", 0.0),
                "block_type": r.get("metadata", {}).get("block_type", "unknown"),
                "label": r.get("metadata", {}).get("label", ""),
                "content": r.get("metadata", {}).get("content", ""),
                "agent": r.get("metadata", {}).get("agent", ""),
                "priority": r.get("metadata", {}).get("priority", 5),
            }
            # Also check top-level metadata
            if d["block_type"] == "unknown" and "block_type" in r:
                d["block_type"] = r["block_type"]
            if d["label"] == "" and "label" in r:
                d["label"] = r["label"]
            if d["content"] == "" and "content" in r:
                d["content"] = r["content"]
            if d["agent"] == "" and "agent" in r:
                d["agent"] = r["agent"]
            normalized.append(d)
    except Exception as e:
        log.error(f"api_context failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    # â”€â”€ Load guardrails (ALWAYS, regardless of search results) â”€â”€
    guardrails = []
    try:
        if vec_mem is None:
            vec_mem, _, _ = _resolve_db(project=project, cwd=_cwd)
        guardrails = vec_mem.list_guardrails(project=project, k=50)
    except Exception:
        pass

    # Group search results by tier (exclude guardrails from search results
    # since they're shown in their own section)
    tiers: dict[str, list] = {}
    for r in normalized:
        tier = r.get("block_type", "unknown")
        if tier == "guardrail":
            continue
        if tier not in tiers:
            tiers[tier] = []
        tiers[tier].append({
            "label": r.get("label", ""),
            "content": r.get("content", "")[:400],
            "agent": r.get("agent", ""),
            "score": r.get("score", 0.0),
        })
    # Format as injection text (sanitize every field so a stored memory cannot
    # break out of the block or smuggle prompt-instruction tokens).
    lines = []

    # â”€â”€ Guardrails section: ALWAYS FIRST, always visible â”€â”€
    if guardrails:
        lines.append(f"## GUARDRAILS ({len(guardrails)} rules â€” always active)")
        lines.append("These rules MUST be followed at ALL times. They override defaults.\n")
        for g in guardrails:
            ct = _sanitize_for_prompt(g.get("content", ""))[:300]
            lb = _sanitize_for_prompt(g.get("label", ""))
            lines.append(f"  * [{lb}] {ct}")
        lines.append("")

    # â”€â”€ Context memories section â”€â”€
    lines.append(f"## MATHIR Auto-Context â€” {len(normalized)} memories for: {_sanitize_for_prompt(task)[:100]}")
    for tier, items in tiers.items():
        lines.append(f"\n### {_sanitize_for_prompt(tier).upper()} ({len(items)})")
        for item in items:
            ag = _sanitize_for_prompt(item.get('agent', ''))
            lb = _sanitize_for_prompt(item.get('label', ''))
            ct = _sanitize_for_prompt(item.get('content', ''))[:200]
            lines.append(f"> [{ag}] {lb}: {ct}")
    return jsonify({
        "context": "\n".join(lines),
        "tiers": {t: len(v) for t, v in tiers.items()},
        "guardrails_count": len(guardrails),
        "guardrails": [{"label": g.get("label", ""), "content": g.get("content", "")[:300]} for g in guardrails],
        "total": len(normalized),
        "task": task[:200],
    })


@app.route("/api/projects")
def api_projects():
    return jsonify({"projects": _list_projects()})


@app.route("/api/cache/stats", methods=["GET"])
def api_cache_stats():
    return jsonify(cache_stats())


# ---------------------------------------------------------------------------
# Health + startup
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    # Read runtime config so /health reflects the actual installed model
    # + version (user can change them in mathir.json â€” see AGENT.md Â§Changing
    # Models). Fall back to pyproject.toml version if config is missing.
    cfg = {}
    try:
        cfg = load_config()
    except Exception:
        pass
    model_name = (
        cfg.get("embedding", {}).get("model")
        or os.environ.get("MATHIR_EMBEDDING_MODEL")
        or "intfloat/multilingual-e5-small"
    )
    version = (
        cfg.get("version")
        or _detect_version_from_pyproject()
        or "unknown"
    )
    embedding_dim = (
        cfg.get("memory", {}).get("embedding_dim")
        or int(os.environ.get("MATHIR_EMBEDDING_DIM", "384"))
    )

    resp = {
        "status": "ok",
        "uptime": round(time.time() - _start_time, 1),
        "model": model_name.split("/")[-1],  # short name
        "model_full": model_name,
        "version": version,
        "embedding_dim": embedding_dim,
    }

    # Check for newer version on GitHub Releases (cached, never blocks /health)
    try:
        from .mathir_update_check import check_for_update
        update = check_for_update(version)
        resp["latest_version"] = update.get("latest_version", version)
        resp["update_available"] = update.get("update_available", False)
        if update.get("update_available"):
            resp["update_command"] = "python -m mathir_mcp update"
            if update.get("release_url"):
                resp["release_url"] = update["release_url"]
        if update.get("source"):
            resp["update_source"] = update["source"]
        if update.get("error"):
            resp["update_check_error"] = update["error"]
    except Exception as e:
        # Graceful: never break /health on update-check failure
        log.debug(f"update check failed: {e}")
        resp["update_check_error"] = "update checker not loaded"

    # Surface the legacy-schema warning on /health so the agent plugin sees it
    # immediately at session start (the plugin polls /health on session.started).
    # Uses cached schema kind from warmup â€” no re-init of embedder.
    try:
        if _DB_LEGACY_WARNING:
            # Report the most recent legacy warning (any DB path)
            any_warning = next(iter(_DB_LEGACY_WARNING.values()))
            resp["schema"] = "legacy"
            resp["migration_hint"] = any_warning
        elif _DB_SCHEMA_KIND:
            resp["schema"] = next(iter(_DB_SCHEMA_KIND.values()))
    except Exception:
        pass
    return jsonify(resp)


def _detect_version_from_pyproject() -> str:
    """Fallback: read version from mathir_lib/pyproject.toml."""
    try:
        import re as _re
        # pyproject.toml lives in the mathir_mcp package root
        for candidate in [
            Path(__file__).resolve().parent.parent / "pyproject.toml",
            Path(__file__).resolve().parent / "pyproject.toml",
        ]:
            if candidate.is_file():
                m = _re.search(r'version\s*=\s*"([^"]+)"', candidate.read_text())
                if m:
                    return m.group(1)
    except Exception:
        pass
    return "unknown"


@app.route("/api/ping")
def api_ping():
    return jsonify({"pong": True, "uptime": round(time.time() - _start_time, 1), "dim": get_embedder_dim()})


# ---------------------------------------------------------------------------
# Memory API routes â€” all POST, JSON body
# ---------------------------------------------------------------------------

def _get_params():
    """Extract JSON params from request body."""
    try:
        return request.get_json(force=True)
    except Exception:
        return {}


def _validate(params):
    err = _validate_input(params)
    if err:
        return jsonify({"error": err}), 400
    return None


@app.route("/api/memory/save", methods=["POST"])
def memory_save():
    params = _get_params()
    err = _validate(params)
    if err:
        return err
    try:
        vec_mem, _db_path, embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        content = params['content']

        # Risk mitigation
        risk_warnings = []
        if _risk_enabled:
            try:
                classifier = DomainClassifier()
                leakage = LeakageDetector()
                sycophancy = SycophancyDetector()
                domain = classifier.classify(content)
                leak_risk = leakage.check_leakage(domain, domain, content)
                syco_risk = sycophancy.check_sycophancy(content)
                if leak_risk.leakage_risk > 0.5:
                    risk_warnings.append(f"leakage_risk={leak_risk.leakage_risk:.1f}")
                if syco_risk.sycophancy_risk > 0.5:
                    risk_warnings.append(f"sycophancy_risk={syco_risk.sycophancy_risk:.1f}")
            except Exception:
                pass

        emb_np = _encode_passage(embedder, content)
        import uuid
        memory_id = f"mem_{uuid.uuid4().hex}"

        block_type = params.get('block_type', 'episodic')
        # Guardrail tier: enforce min priority and per-project limit
        if block_type == 'guardrail':
            params['priority'] = max(int(params.get('priority', 8)), 8)
            try:
                from . import GUARDRAIL_MAX_PER_PROJECT
            except ImportError:
                from mathir_lib import GUARDRAIL_MAX_PER_PROJECT
            count = vec_mem.count_guardrails(project=params.get('project'))
            if count >= GUARDRAIL_MAX_PER_PROJECT:
                return jsonify({
                    'error': f'guardrail limit reached ({GUARDRAIL_MAX_PER_PROJECT} per project). '
                             f'Delete old guardrails before adding new ones.',
                    'current_count': count,
                    'max': GUARDRAIL_MAX_PER_PROJECT,
                }), 400
        tier_override = None
        # Guardrails are explicit, permanent, user-intentional rules -- they
        # must never be silently reclassified by the anomaly detector. A new
        # guardrail almost always describes a genuinely novel problem (that's
        # why it's being added), which is exactly the content the anomaly
        # detector is tuned to flag -- so without this guard, EVERY newly
        # saved guardrail would get bumped to tier=immunological instead of
        # tier=guardrail and silently drop out of the always-injected
        # guardrail list. Verified live, 2026-07-21: this exact bug fired on
        # a guardrail save (anomaly_score=37.05 -> saved as immunological,
        # never appeared in the always-on GUARDRAILS block).
        if block_type != 'guardrail':
            try:
                anomaly_result = vec_mem.check_and_update_anomaly(
                    emb_np, threshold=_ANOMALY_THRESHOLD, warmup_count=_ANOMALY_WARMUP,
                )
                if anomaly_result["is_anomaly"]:
                    tier_override = "immunological"
                    block_type = "immunological"
                    risk_warnings.append(f"anomaly_score={anomaly_result['score']:.2f}")
            except Exception:
                # Anomaly detection is best-effort â€” never block a save because
                # of it (e.g. corrupt persisted state, dimension mismatch on an
                # old DB). Falls through with tier_override=None.
                pass

        metadata = {
            'agent': params.get('agent', 'unknown'),
            'block_type': block_type,
            'label': params.get('label', ''),
            'priority': params.get('priority', 5),
            'content': content,
            'project': params.get('project') or get_project_name(),
            'risk_warnings': risk_warnings if risk_warnings else None,
            # memory_by_path documents filtering on this field, but nothing
            # ever populated it -- verified live, 2026-07-21: every saved
            # memory had file_path="" so the tool silently fell back to raw
            # content text search only. Populate it from an explicit caller
            # param when given.
            'file_path': params.get('file_path', ''),
        }
        if block_type == 'guardrail':
            metadata['tier'] = 'guardrail'
        elif tier_override:
            metadata['tier'] = tier_override
            metadata['anomaly_score'] = float(anomaly_result['score'])
        vec_mem.store(memory_id, emb_np, metadata)
        invalidate_on_write(project=params.get('project'))
        resp = {'memory_id': memory_id, 'saved': True, 'metadata': metadata}
        _attach_legacy_warning(vec_mem, resp)
        return jsonify(resp)
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_save')}), 500


@app.route("/api/memory/audit_immunological", methods=["POST"])
def memory_audit_immunological():
    params = _get_params()
    try:
        vec_mem, _db_path, _embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        k = min(params.get('k', 20), 200)
        results = vec_mem.list_immunological(project=params.get('project'), k=k)
        return jsonify({"results": results, "total": len(results)})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_audit_immunological')}), 500


@app.route("/api/memory/guardrails", methods=["GET", "POST"])
def memory_guardrails():
    """List all guardrail memories for a project."""
    params = _get_params()
    try:
        vec_mem, _db_path, _embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        k = min(params.get('k', 50), 50)
        results = vec_mem.list_guardrails(project=params.get('project'), k=k)
        return jsonify({"guardrails": results, "total": len(results)})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_guardrails')}), 500


@app.route("/api/memory/reset_anomaly_state", methods=["POST"])
def memory_reset_anomaly_state():
    """Admin/ops route: reset a project's anomaly detection baseline (both
    the persisted DB state and the daemon's in-memory cached detector).
    Use after a config change to anomaly_threshold/warmup_count, or if the
    baseline has drifted from heavy test/probe activity (real gotcha found
    2026-07-02 -- clearing the DB row alone doesn't reset an already-
    running daemon's cached detector)."""
    params = _get_params()
    try:
        vec_mem, _db_path, _embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        vec_mem.reset_anomaly_state()
        return jsonify({"reset": True, "project": params.get("project")})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_reset_anomaly_state')}), 500


@app.route("/api/memory/recall", methods=["POST"])
def memory_recall():
    params = _get_params()
    err = _validate(params)
    if err:
        return err
    try:
        query = params.get('query', '')
        k = min(params.get('k', 5), 1000)
        project = params.get("project")
        agent = params.get('agent')
        block_type = params.get('block_type')
        cached = recall_cache.get(query, k, project=project, agent=agent, block_type=block_type)
        if cached is not None:
            cached['cache'] = 'hit'
            return jsonify(cached)
        vec_mem, _db_path, embedder = _resolve_db(project=project, cwd=params.get("cwd"))
        q_np = _encode_query(embedder, query)
        results = vec_mem.search(
            query_embedding=q_np, k=k,
            agent_filter=agent,
            block_type_filter=block_type,
            include_embeddings=bool(params.get('include_embeddings', False)),
        )
        touched = 0
        try:
            for r in results:
                mid = r.get('memory_id')
                if mid and hasattr(vec_mem, 'touch_recall'):
                    vec_mem.touch_recall(mid)
                    touched += 1
        except Exception:
            pass
        response = {'results': results, 'query': query, 'total': len(results), 'touched': touched, 'cache': 'miss'}
        recall_cache.put(query, k, response, project=project, agent=agent, block_type=block_type)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_recall')}), 500


@app.route("/api/memory/stats", methods=["POST", "GET"])
def memory_stats():
    try:
        params = _get_params() if request.method == "POST" else {}
        vec_mem, _db_path, _embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        return jsonify(vec_mem.stats())
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_stats')}), 500


@app.route("/api/memory/by_path", methods=["POST"])
def memory_by_path():
    """Search memories by real file_path SQL filter (not embedding recall).

    The MCP-layer memory_by_path previously ran a semantic memory_recall for
    the path string, then post-filtered that candidate pool for path/content
    matches -- so a memory with metadata.file_path correctly populated but
    whose CONTENT doesn't embed close to the path string (e.g. a short "test
    save" note) never entered the recall pool in the first place, and never
    surfaced. Verified live, 2026-07-21: a memory saved with an explicit
    file_path still didn't appear in the top results for that exact path.
    Query the file_path column directly instead -- it's a structured filter,
    it should use SQL, not embedding similarity as a proxy for it.
    """
    params = _get_params()
    try:
        vec_mem, _db_path, _embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        file_path = params.get('file_path', '')
        if not file_path:
            return jsonify({'error': 'file_path is required'}), 400
        k = min(int(params.get('k', 10)), 200)

        conn = vec_mem._get_conn()
        path_norm = file_path.replace("\\", "/")
        bare_name = path_norm.rsplit("/", 1)[-1] if "/" in path_norm else path_norm
        like_path = f"%{path_norm}%"
        like_name = f"%{bare_name}%"

        rows = conn.execute(
            "SELECT memory_id, content, metadata, tier, agent, created_at FROM memories "
            "WHERE tier != 'archived' AND ("
            "   json_extract(metadata, '$.file_path') LIKE ? "
            "   OR json_extract(metadata, '$.file_path') LIKE ? "
            "   OR content LIKE ? OR content LIKE ?) "
            "ORDER BY "
            "  (json_extract(metadata, '$.file_path') LIKE ? OR json_extract(metadata, '$.file_path') LIKE ?) DESC, "
            "  created_at DESC "
            "LIMIT ?",
            (like_path, like_name, like_path, like_name, like_path, like_name, k),
        ).fetchall()

        out = []
        for r in rows:
            try:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            out.append({
                "memory_id": r["memory_id"],
                "label": meta.get("label", ""),
                "file_path": meta.get("file_path", ""),
                "content_snippet": (r["content"] or "")[:200],
                "agent": r["agent"],
                "tier": r["tier"],
                "created_at": r["created_at"],
                "matched_structured_field": bool(meta.get("file_path")) and (
                    path_norm.lower() in str(meta.get("file_path", "")).lower()
                    or bare_name.lower() in str(meta.get("file_path", "")).lower()
                ),
            })
        return jsonify({"file_path": file_path, "total": len(out), "results": out})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_by_path')}), 500


@app.route("/api/memory/dashboard", methods=["POST", "GET"])
def memory_dashboard():
    """Dashboard-level view: recent activity, guardrail roster, save trend.

    Distinct from /api/memory/stats (compact tier/agent counts only) --
    memory_dashboard's MCP tool used to just proxy memory_stats verbatim
    (`_call_daemon("memory_stats", {})`, ignoring its own `action` param),
    so the two tools returned byte-for-byte identical JSON with no added
    value from having both. This route adds the genuinely distinct,
    slightly heavier info a dashboard view is for -- not repeated on every
    stats call, since /api/context calls memory_stats-shaped data far more
    often than a human opens a dashboard.
    """
    try:
        params = _get_params() if request.method == "POST" else {}
        vec_mem, _db_path, _embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        stats = vec_mem.stats()
        conn = vec_mem._get_conn()

        recent_rows = conn.execute(
            "SELECT memory_id, label, agent, tier, created_at FROM memories "
            "ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        recent_activity = [dict(r) for r in recent_rows]

        guardrail_rows = conn.execute(
            "SELECT memory_id, label, priority FROM memories WHERE block_type = 'guardrail' "
            "ORDER BY priority DESC"
        ).fetchall()
        guardrails = [dict(r) for r in guardrail_rows]

        now = datetime.now(timezone.utc)
        today_start = now.date().isoformat()
        week_ago = (now - timedelta(days=7)).isoformat()
        created_today = conn.execute(
            "SELECT COUNT(*) AS cnt FROM memories WHERE created_at >= ?", (today_start,)
        ).fetchone()["cnt"]
        created_this_week = conn.execute(
            "SELECT COUNT(*) AS cnt FROM memories WHERE created_at >= ?", (week_ago,)
        ).fetchone()["cnt"]

        return jsonify({
            "total": stats.get("total"),
            "by_block_type": stats.get("by_block_type"),
            "recent_activity": recent_activity,
            "guardrails": {"count": len(guardrails), "items": guardrails},
            "trend": {"created_today": created_today, "created_this_week": created_this_week},
        })
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_dashboard')}), 500


@app.route("/api/memory/delete", methods=["POST"])
def memory_delete():
    params = _get_params()
    try:
        vec_mem, _db_path, _embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        memory_id = params.get('memory_id')
        if not memory_id:
            return jsonify({'error': 'memory_id required'}), 400
        deleted = vec_mem.delete(memory_id)
        invalidate_on_write(project=params.get('project'))
        resp = {'memory_id': memory_id, 'deleted': deleted}
        _attach_legacy_warning(vec_mem, resp)
        return jsonify(resp)
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_delete')}), 500


@app.route("/api/memory/smart_search", methods=["POST"])
def memory_smart_search():
    params = _get_params()
    err = _validate(params)
    if err:
        return err
    try:
        vec_mem, _db_path, embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        query = params.get('query', '')
        k = min(params.get('k', 10), 1000)
        q_np = _encode_query(embedder, query)
        results = vec_mem.search(
            query_embedding=q_np, k=k, agent_filter=params.get('agent'),
            include_embeddings=bool(params.get('include_embeddings', False)),
        )
        return jsonify({'results': results, 'query': query, 'total': len(results)})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_smart_search')}), 500


@app.route("/api/memory/push", methods=["POST"])
def memory_push():
    params = _get_params()
    context_text = params.get('context', '')
    k = min(params.get('k', 10), 1000)
    agent = params.get('agent')
    if not context_text:
        return jsonify({'memories': [], 'total': 0, 'error': 'empty context'})
    try:
        c_hash = context_hash(context_text)
        with _push_lock:
            cached = _push_cache.get(c_hash)
        if cached is not None:
            return jsonify({'memories': cached, 'total': len(cached), 'cached': True})
        queries = _push_analyzer.extract_queries(context_text, max_queries=5)
        vec_mem, _db_path, embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        all_memories = []
        for q in queries:
            q_np = _encode_query(embedder, q)
            results = vec_mem.search(query_embedding=q_np, k=max(3, k // max(len(queries), 1) + 1), agent_filter=agent)
            all_memories.extend(results)
        deduped = deduplicate_memories(all_memories)
        deduped.sort(key=lambda m: m.get('score', 0), reverse=True)
        top = deduped[:k]
        with _push_lock:
            _push_cache.set(c_hash, top)
        return jsonify({'memories': top, 'queries_used': queries, 'total': len(top), 'cached': False})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_push')}), 500


@app.route("/api/memory/hybrid_search", methods=["POST"])
def memory_hybrid_search():
    params = _get_params()
    err = _validate(params)
    if err:
        return err
    try:
        _vec_mem, db_path, embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        query_text = params.get('query', '')
        k = min(params.get('k', 5), 100)
        vector_weight = params.get('vector_weight', 1.0)
        bm25_weight = params.get('bm25_weight', 1.0)
        entity_weight = params.get('entity_weight', 0.0)
        agent_filter = params.get('agent')
        q_np = _encode_query(embedder, query_text)

        import sqlite3 as _sqlite3
        dconn = _sqlite3.connect(str(db_path), check_same_thread=False)
        dconn.row_factory = _sqlite3.Row
        try:
            import sqlite_vec as _sqlite_vec
            dconn.enable_load_extension(True)
            _sqlite_vec.load(dconn)
            dconn.enable_load_extension(False)
            _has_vec = True
        except Exception:
            _has_vec = False

        columns = {col[1] for col in dconn.execute("PRAGMA table_info(memories)").fetchall()}
        text_col = 'content' if 'content' in columns else 'modality_text'

        vector_results = []
        if _has_vec:
            from mathir_vec import _serialize_embedding
            q_hex = _serialize_embedding(q_np)
            sql = f"""
                SELECT m.memory_id, v.distance
                FROM vec_memories v
                JOIN memories m ON v.memory_id = m.memory_id
                WHERE v.embedding MATCH vec_int8(X'{q_hex}') AND k = ?
            """
            params_list = [k * 3]
            if agent_filter:
                sql += " AND m.agent = ?" if 'agent' in columns else " AND json_extract(m.metadata, '$.agent') = ?"
                params_list.append(agent_filter)
            for row in dconn.execute(sql, params_list).fetchall():
                vector_results.append((row['memory_id'], 1.0 - row['distance']))

        bm25_results = []
        try:
            row_count = dconn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        except Exception:
            row_count = 0
        if row_count:
            from mathir_search import _tokenize
            from rank_bm25 import BM25Okapi

            cache_key = str(db_path)
            with _bm25_cache_lock:
                cached = _bm25_cache.get(cache_key)

            if cached is not None and cached["row_count"] == row_count:
                bm25 = cached["bm25"]
                corpus_ids = cached["corpus_ids"]
            else:
                try:
                    rows = dconn.execute(f"SELECT memory_id, {text_col} FROM memories").fetchall()
                except Exception:
                    rows = []
                corpus_ids = [r['memory_id'] for r in rows]
                corpus_texts = [r[1] or '' for r in rows]
                tokenized = [_tokenize(t) for t in corpus_texts]
                bm25 = BM25Okapi(tokenized) if tokenized else None
                with _bm25_cache_lock:
                    _bm25_cache[cache_key] = {
                        "bm25": bm25, "corpus_ids": corpus_ids, "row_count": row_count,
                    }

            if bm25 is not None:
                scores = bm25.get_scores(_tokenize(query_text))
                for mid, sc in sorted(zip(corpus_ids, scores), key=lambda x: x[1], reverse=True):
                    if sc > 0:
                        bm25_results.append((mid, float(sc)))
                    if len(bm25_results) >= k * 3:
                        break

        entity_results = []
        if entity_weight and row_count:
            try:
                from mathir_entity_graph import extract_entities

                cache_key = str(db_path)
                with _entity_cache_lock:
                    cached_ent = _entity_cache.get(cache_key)

                if cached_ent is not None and cached_ent["row_count"] == row_count:
                    entity_index = cached_ent["index"]
                else:
                    try:
                        rows = dconn.execute(f"SELECT memory_id, {text_col} FROM memories").fetchall()
                    except Exception:
                        rows = []
                    entity_index = [(r['memory_id'], extract_entities(r[1] or '')) for r in rows]
                    with _entity_cache_lock:
                        _entity_cache[cache_key] = {"index": entity_index, "row_count": row_count}

                query_entities = extract_entities(query_text)
                if query_entities:
                    scored = []
                    for mid, ents in entity_index:
                        overlap = len(query_entities & ents)
                        if overlap > 0:
                            scored.append((mid, float(overlap)))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    entity_results = scored[:k * 3]
            except Exception:
                entity_results = []  # entity signal is best-effort, never fails the search

        from mathir_search import rrf_fusion
        fused = rrf_fusion(vector_results, bm25_results, vector_weight=vector_weight, bm25_weight=bm25_weight,
                           entity_results=entity_results, entity_weight=entity_weight)

        # Detect schema for hybrid search result building
        columns = {col[1] for col in dconn.execute("PRAGMA table_info(memories)").fetchall()}
        text_col = 'content' if 'content' in columns else 'modality_text'
        ts_col = 'created_at' if 'created_at' in columns else 'timestamp'

        do_rerank = params.get('rerank', False)
        fetch_limit = k * 3 if do_rerank else k

        results = []
        for mid, rrf_score in fused[:fetch_limit]:
            meta = dconn.execute(f"SELECT {text_col}, tier, {ts_col} FROM memories WHERE memory_id = ?", [mid]).fetchone()
            if not meta:
                continue
            agent_val = ''
            if 'agent' in columns:
                agent_val = dconn.execute("SELECT agent FROM memories WHERE memory_id = ?", [mid]).fetchone()[0] or ''
            results.append({
                'memory_id': mid, 'rrf_score': rrf_score, 'content': meta[0] or '',
                'agent': agent_val, 'score': rrf_score, 'created_at': meta[2] or '', 'tier': meta[1] or 'episodic',
                'text': meta[0] or '',
            })

        reranked = False
        if do_rerank and results:
            try:
                from mathir_search import CrossEncoderReranker
                reranker = CrossEncoderReranker()
                results = reranker.rerank(query_text, results, top_k=k)
                reranked = True
            except Exception as e:
                import logging
                logging.getLogger("mathir").warning("Reranking failed, returning RRF results: %s", e)

        dconn.close()
        return jsonify({
            'results': results[:k], 'query': query_text, 'total': len(results[:k]),
            'mode': 'hybrid+rerank' if reranked else 'hybrid',
            'vector_hits': len(vector_results), 'bm25_hits': len(bm25_results),
            'reranked': reranked,
        })
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_hybrid_search')}), 500


@app.route("/api/memory/risk_check", methods=["POST"])
def memory_risk_check():
    if not _risk_enabled:
        return jsonify({'error': 'risk mitigation not available'})
    params = _get_params()
    content = params.get('content', '')
    if not content:
        return jsonify({'error': 'content required'}), 400
    try:
        classifier = DomainClassifier()
        leakage = LeakageDetector()
        sycophancy = SycophancyDetector()
        domain = classifier.classify(content)
        leak_risk = leakage.check_leakage(domain, domain, content)
        syco_risk = sycophancy.check_sycophancy(content)
        return jsonify({
            'domain': domain.value, 'leakage_risk': leak_risk.leakage_risk,
            'sycophancy_risk': syco_risk.sycophancy_risk,
            'safe_to_store': leak_risk.leakage_risk < 0.7 and syco_risk.sycophancy_risk < 0.7,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Lifecycle routes ---

@app.route("/api/memory/promote", methods=["POST"])
def memory_promote():
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        result = vec_mem.promote(params.get('memory_id', ''), force=params.get('force', False))
        invalidate_on_write(project=params.get('project'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_promote')}), 500


@app.route("/api/memory/auto_promote", methods=["POST"])
def memory_auto_promote():
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        promoted = vec_mem.auto_promote_all()
        if promoted:
            invalidate_on_write(project=params.get('project'))
        return jsonify({'promoted': promoted, 'count': len(promoted)})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_auto_promote')}), 500


@app.route("/api/memory/decay", methods=["POST"])
def memory_decay():
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        return jsonify(vec_mem.decay_all(
            threshold_days=params.get('threshold_days', 30),
            archive_floor=params.get('archive_floor', 0.05),
        ))
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_decay')}), 500


@app.route("/api/memory/consolidate", methods=["POST"])
def memory_consolidate():
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        result = vec_mem.consolidate_all(
            threshold=params.get('threshold', 0.95),
            limit=params.get('limit', 100),
            max_results=params.get('max_results', 50),
            dry_run=params.get('dry_run', True),
        )
        if not params.get('dry_run', True):
            invalidate_on_write(project=params.get('project'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_consolidate')}), 500


@app.route("/api/memory/link", methods=["POST"])
def memory_link():
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        return jsonify(vec_mem.add_link(
            source_id=params.get('source_id', ''),
            target_id=params.get('target_id', ''),
            weight=params.get('weight', 1.0),
        ))
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_link')}), 500


@app.route("/api/memory/get_links", methods=["POST"])
def memory_get_links():
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        return jsonify({'result': vec_mem.get_links(
            params.get('memory_id', ''),
            depth=params.get('depth', 1),
            decay=params.get('decay', 0.5),
        )})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_get_links')}), 500


@app.route("/api/memory/incoming_links", methods=["POST"])
def memory_incoming_links():
    """Return all links whose target_id == memory_id (reverse direction)."""
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        memory_id = params.get('memory_id', '')
        if not memory_id:
            return jsonify({'error': 'memory_id is required'}), 400
        with vec_mem._db_lock:
            conn = vec_mem._get_conn()
            rows = conn.execute(
                """
                SELECT source_id, target_id, weight, created_at
                FROM memory_links
                WHERE target_id = ?
                ORDER BY weight DESC
                """,
                (memory_id,),
            ).fetchall()
            result = [dict(row) for row in rows]
        return jsonify({'memory_id': memory_id, 'incoming': result, 'count': len(result)})
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_incoming_links')}), 500


@app.route("/api/memory/build_links", methods=["POST"])
def memory_build_links():
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        # mode: "cosine" (default, embedding-similarity graph -- original
        # behavior), "entity" (entity-shared graph -- links memories that
        # mention the same named entity, for multi-hop bridging), or "both".
        mode = str(params.get('mode', 'cosine')).lower()
        limit = params.get('limit', 1000)
        top_k = params.get('top_k', 8)
        # clean=True wipes the whole link table first so the rebuild is a true
        # rebuild (stale edges for pairs that no longer qualify disappear).
        # Default False keeps existing callers' semantics unchanged.
        if params.get('clean', False):
            vec_mem._get_conn().execute("DELETE FROM memory_links")
            vec_mem._get_conn().commit()
        out = {}
        if mode in ("cosine", "both"):
            out["cosine"] = vec_mem.build_links_all(
                threshold=params.get('threshold', 0.88), limit=limit, top_k=top_k,
            )
        if mode in ("entity", "both"):
            out["entity"] = vec_mem.build_entity_links_all(limit=limit)
        # Back-compat: when a single mode is requested, also flatten its
        # link count to the top level so existing callers that read
        # {"links_created": N} keep working.
        single = out.get("cosine") or out.get("entity") or {}
        if mode != "both" and isinstance(single, dict):
            out.update(single)
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_build_links')}), 500


@app.route("/api/memory/audit", methods=["POST", "GET"])
def memory_audit():
    """Audit log of recent operations.

    FIX (2026-07-02): previously did its own raw SQL against a
    `memory_audit` table that NOTHING in the codebase ever created or
    wrote to -- found via a systematic 23-tool smoke test, this route had
    silently returned empty results for every user, forever. Now delegates
    to VecMemory.get_audit_log(), which is backed by a real table
    populated by save/delete/promote/decay (see _log_audit()).
    """
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        agent = params.get("agent")
        limit = params.get("limit", 50)
        entries = vec_mem.get_audit_log(agent=agent, limit=limit)
        return jsonify({"entries": entries, "total": len(entries)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/export", methods=["POST", "GET"])
def memory_export():
    """Export all memories as JSON, written to a file rather than inlined.

    A full export is meant to be complete, not truncated -- silently
    dropping rows to fit a response size limit would make it a lie (a
    partial export presented as if it were a real backup). Verified live,
    2026-07-21: with 755 memories this route's inline JSON response hit
    137,102 characters, blowing past the caller's context/token limit even
    though each row only carries 5 skinny fields (no content). The fix is
    the same shape used elsewhere for oversized responses -- write the full
    data to a file server-side and return its path + a count, instead of
    inlining a payload that structurally cannot stay small as the DB grows.
    """
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        conn = vec_mem._get_conn()
        # Schema-aware export: new schema uses created_at + metadata JSON, legacy uses timestamp/stability/recall_count
        columns = {col[1] for col in conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "content" in columns:
            # New schema
            rows = conn.execute(
                "SELECT memory_id, tier, created_at, "
                "json_extract(metadata, '$.recall_count') as recall_count, "
                "json_extract(metadata, '$.stability') as stability "
                "FROM memories ORDER BY rowid"
            ).fetchall()
        else:
            # Legacy schema
            rows = conn.execute(
                "SELECT memory_id, tier, timestamp, recall_count, stability "
                "FROM memories ORDER BY rowid"
            ).fetchall()
        memories = [dict(r) for r in rows] if rows else []

        export_dir = _P_DATA / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        project_slug = str(params.get("project") or get_project_name() or "default")
        project_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_slug)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_path = export_dir / f"export_{project_slug}_{ts}.json"
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({"memories": memories, "total": len(memories)}, f, ensure_ascii=False)

        return jsonify({
            "file_path": str(export_path),
            "total": len(memories),
            "note": "Full export written to disk (too large to inline safely). Read the file directly.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/sessions", methods=["POST", "GET"])
def memory_sessions():
    """List recent memory sessions."""
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        limit = params.get("limit", 10)
        conn = vec_mem._get_conn()
        # Schema-aware: new schema uses created_at, legacy uses timestamp
        columns = {col[1] for col in conn.execute("PRAGMA table_info(memories)").fetchall()}
        ts_col = "created_at" if "created_at" in columns else "timestamp"
        rows = conn.execute(
            f"SELECT * FROM memories ORDER BY {ts_col} DESC LIMIT ?",
            (limit,)
        ).fetchall()
        sessions = []
        for r in rows:
            try:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
                sessions.append({
                    "memory_id": r["memory_id"],
                    "agent": meta.get("agent", "unknown"),
                    "label": meta.get("label", ""),
                    "timestamp": r[ts_col] if ts_col in r.keys() else "",
                })
            except:
                pass
        return jsonify({"sessions": sessions, "total": len(sessions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/push_cache_stats", methods=["GET"])
def push_cache_stats():
    with _push_lock:
        return jsonify(_push_cache.stats())


# ---------------------------------------------------------------------------
# God Orchestrator routes (v8.8.0)
# ---------------------------------------------------------------------------

@app.route("/api/memory/recall_quality", methods=["POST"])
def memory_recall_quality():
    """Recall with explicit quality signal (quality: high | medium | low).

    HTTP twin of the MCP memory_recall_quality tool. Added 2026-08-18:
    the MCP endpoint_map and the stdio wrapper (Codex) both routed
    "memory_recall_quality" to this path, which did not exist, so every
    call failed with a daemon 404. Quality is based on the top-1 score AND
    lexical grounding: a deliberately nonsensical out-of-domain query still
    scored 0.839 ("high") purely from embedding-space coincidence (verified
    live 2026-07-21), so a real match must also share >=4-char tokens.
    """
    try:
        params = _get_params()
        query = params.get("query", "")
        k = int(params.get("k", 5))
        min_score = float(params.get("min_score", 0.4))
        if not query:
            return jsonify({"error": "query is required"}), 400

        with app.test_client() as client:
            resp = client.post("/api/memory/recall", json={"query": query, "k": k})
            recall = resp.get_json()
        results = (recall or {}).get("results", []) or []

        if not results:
            return jsonify({
                "query": query, "quality": "none", "total": 0,
                "suggestion": "No memories matched. Try rephrasing or saving knowledge first.",
                "results": [],
            })

        top1 = float(results[0].get("score", 0.0))
        top1_meta = results[0].get("metadata") or {}
        top1_text = " ".join(str(x) for x in (
            top1_meta.get("content", ""), results[0].get("content", ""),
            top1_meta.get("label", ""), results[0].get("label", ""),
        )).lower()
        query_tokens = set(re.findall(r"[a-z0-9]{4,}", query.lower()))
        lexically_grounded = any(tok in top1_text for tok in query_tokens)

        if top1 >= 0.7 and lexically_grounded:
            quality = "high"
            suggestion = "Strong match â€” top result is highly relevant."
        elif top1 >= 0.7:
            quality = "medium"
            suggestion = (
                f"Top-1 score {top1:.2f} looks strong but shares no vocabulary with the "
                "query â€” likely an embedding-space coincidence, not a real match. Review "
                "before trusting."
            )
        elif top1 >= min_score:
            quality = "medium"
            suggestion = "Partial match â€” review top results for relevance."
        else:
            quality = "low"
            suggestion = (
                f"Top-1 score {top1:.2f} < {min_score:.2f}. "
                "DB likely lacks what you need. Save new knowledge or broaden query."
            )

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

        return jsonify({
            "query": query,
            "quality": quality,
            "top1_score": round(top1, 3),
            "min_score": min_score,
            "lexically_grounded": lexically_grounded,
            "total": len(out),
            "suggestion": suggestion,
            "results": out,
        })
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_recall_quality')}), 500


@app.route("/api/god/poll", methods=["POST"])
def api_god_poll():
    """Optimized task polling for god workers.

    When status == "pending" (the normal consume case), the matching row is
    claimed atomically in the same transaction as the SELECT, flipping its
    label to "...:claimed" before returning it. This matters once the same
    agent NAME is shared by many parallel instances of the same tool (e.g.
    100 OpenCode terminals all polling as "opencode" -- see the god-mode
    scaling discussion: a fixed per-tool name only works as a shared *pool*,
    not a unique identity, if two instances can race to read the same still-
    "pending" row before either writes back). Without the atomic claim here,
    two pollers could both read the same pending task and both execute it.
    Polls for any other status (monitoring/inspection) remain a plain read.
    """
    try:
        params = _get_params()
        agent = params.get("agent", "")
        status = params.get("status", "pending")
        if not agent:
            return jsonify({"error": "agent is required"}), 400

        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        conn = vec_mem._get_conn()
        safe_agent = agent.replace("%", r"\%").replace("_", r"\_")
        safe_status = status.replace("%", r"\%").replace("_", r"\_")
        suffix = f":{safe_agent}:{safe_status}"
        select_sql = (
            """SELECT memory_id, metadata, label
               FROM memories
               WHERE label LIKE 'god:task:%'
                 AND label LIKE ? ESCAPE '\\'
                 AND tier != 'archived'
               ORDER BY priority DESC, created_at ASC
               LIMIT 1"""
        )

        if status != "pending":
            row = conn.execute(select_sql, (f"%{suffix}",)).fetchone()
            if not row:
                return jsonify({"task": None})
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            return jsonify({
                "task": {
                    "memory_id": row["memory_id"],
                    "label": row["label"],
                    "content": meta.get("content", ""),
                    "priority": meta.get("priority", 5),
                }
            })

        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(select_sql, (f"%{suffix}",)).fetchone()
            if not row:
                conn.rollback()
                return jsonify({"task": None})

            parts = row["label"].split(":")
            claimed_label = f"god:{parts[1]}:{parts[2]}:{safe_agent}:claimed"
            res = conn.execute(
                "UPDATE memories SET label = ? WHERE memory_id = ? AND label = ?",
                (claimed_label, row["memory_id"], row["label"]),
            )
            if res.rowcount == 0:
                conn.rollback()
                return jsonify({"task": None})  # lost the race -- another poller claimed it first
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return jsonify({
            "task": {
                "memory_id": row["memory_id"],
                "label": claimed_label,
                "content": meta.get("content", ""),
                "priority": meta.get("priority", 5),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/god/ack", methods=["POST"])
def api_god_ack():
    """Mark a god:task as delivered so /api/god/poll stops returning it and
    surfaces the next queued task instead.

    Without this, /api/god/poll always returns the single oldest still-
    pending task for an agent -- a caller that reads it but never changes
    its status (e.g. a passive relay that only wants to *notify*, not run
    the full worker execute-then-report flow) blocks every task behind it
    in the queue forever. This route exists specifically for that passive-
    relay case (see claude_code_hook.py's God Mode relay). Workers running
    the full task protocol should still report real results via the normal
    mathir_god_agent flow, not this endpoint.
    """
    try:
        params = _get_params()
        memory_id = params.get("memory_id", "")
        new_status = params.get("status", "delivered")
        if not memory_id:
            return jsonify({"error": "memory_id is required"}), 400

        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        conn = vec_mem._get_conn()
        cursor = conn.execute(
            "SELECT label FROM memories WHERE memory_id = ?", (memory_id,)
        )
        row = cursor.fetchone()
        if not row or not row["label"]:
            return jsonify({"error": "memory not found"}), 404

        parts = row["label"].split(":")
        if len(parts) != 5 or parts[0] != "god":
            return jsonify({"error": "not a god:task label"}), 400
        parts[4] = new_status
        new_label = ":".join(parts)

        conn.execute(
            "UPDATE memories SET label = ? WHERE memory_id = ?",
            (new_label, memory_id),
        )
        conn.commit()
        return jsonify({"memory_id": memory_id, "label": new_label, "acked": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/god/agents", methods=["GET", "POST"])
def api_god_agents():
    """List all registered god workers."""
    try:
        params = _get_params() if request.method == "POST" else {}
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        conn = vec_mem._get_conn()
        cursor = conn.execute(
            """SELECT memory_id, metadata, label
               FROM memories
               WHERE label LIKE 'god:reg:%' AND tier != 'archived'
               ORDER BY created_at DESC"""
        )
        seen = {}
        for row in cursor.fetchall():
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            label = row["label"] or ""
            parts = label.split(":")
            if len(parts) >= 5:
                name = parts[2]
                if name not in seen:
                    content = meta.get("content", "{}")
                    try:
                        info = json.loads(content) if isinstance(content, str) else content
                    except (json.JSONDecodeError, TypeError):
                        info = {}
                    seen[name] = {
                        "name": name,
                        "status": parts[4],
                        "capabilities": info.get("capabilities", []),
                        "introduction": info.get("introduction", ""),
                    }
        return jsonify({"agents": list(seen.values())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/god/reg", methods=["POST"])
def api_god_reg():
    """Upsert a god worker registration row (status: idle | busy | offline).

    FIX (2026-08-18): mathir_god_agent previously called memory_save for
    every god:reg:* transition, inserting a NEW memory row per poll â€”
    observed live creating 419 near-duplicate god:reg rows for a single
    worker name (all similarity 1.0, same label family, created within a
    47s window, all since archived by consolidate). This route reuses the
    existing registration row (same memory_id, INSERT OR REPLACE) so
    repeated polls never accumulate junk; only the first call for a name
    inserts a row, every later call updates it in place.
    """
    try:
        params = _get_params()
        memory_id, action, label = _god_reg_upsert(
            params.get("name", ""),
            params.get("status", "idle"),
            params.get("content", {}),
            params.get("project"),
            params.get("cwd"),
        )
        return jsonify({
            "memory_id": memory_id, "label": label,
            "action": action, "saved": True,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": _sanitize_error(e, "god_reg")}), 500


def _god_internal_call(path, payload):
    """Call one of our own routes in-process (no network, no port).

    Used by the HTTP god-agent entry points so the register/poll/ack flow
    stays implemented in exactly one place (the existing routes).
    """
    with app.test_client() as client:
        resp = client.post(path, json=payload)
        data = resp.get_json() if resp.is_json else None
        if resp.status_code != 200:
            raise RuntimeError(f"{path} -> HTTP {resp.status_code}: {data}")
        return data or {}


@app.route("/api/god/agent", methods=["POST"])
def api_god_agent():
    """Register a god worker (idle) and poll for one pending task.

    HTTP twin of the MCP mathir_god_agent tool, as a single call: register
    idle -> poll (atomic claim) -> ack running + register busy when a task
    is found, so a plain HTTP client just executes and reports.
    """
    try:
        params = _get_params()
        name = params.get("name", "")
        if not name:
            return jsonify({"error": "name is required"}), 400
        reg_payload = {
            "capabilities": params.get("capabilities", []),
            "introduction": params.get("introduction", ""),
        }
        project = params.get("project")
        cwd = params.get("cwd")
        poll_interval = params.get("poll_interval", 8)

        _god_internal_call("/api/god/reg", {
            "name": name, "status": "idle", "content": reg_payload,
            "project": project, "cwd": cwd,
        })
        polled = _god_internal_call("/api/god/poll", {
            "agent": name, "status": "pending",
            "project": project, "cwd": cwd,
        })
        task = polled.get("task")
        if not task:
            return jsonify({
                "status": "waiting", "reason": "no_pending_task",
                "instruction": f"Call /api/god/agent again after {poll_interval}s",
            })

        task_label = task.get("label", "")
        parts = task_label.split(":")
        task_id = parts[2] if len(parts) >= 3 else "unknown"
        if len(parts) == 5 and parts[4] == "shutdown":
            _god_internal_call("/api/god/reg", {
                "name": name, "status": "offline", "content": reg_payload,
                "project": project, "cwd": cwd,
            })
            return jsonify({"status": "shutdown", "task": task})

        _god_internal_call("/api/god/ack", {
            "memory_id": task["memory_id"], "status": "running",
            "project": project, "cwd": cwd,
        })
        _god_internal_call("/api/god/reg", {
            "name": name, "status": "busy", "content": reg_payload,
            "project": project, "cwd": cwd,
        })
        content = task.get("content", "")
        try:
            task_info = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            task_info = {"description": content}
        return jsonify({
            "status": "task_found",
            "task_id": task_id,
            "description": task_info.get("description", content),
            "task": task,
            "report_instruction": (
                "After completing, report via /api/memory/save with "
                f"label='god:result:{task_id}:orchestrator:completed', "
                "block_type='episodic', priority=7, then call /api/god/agent again."
            ),
        })
    except Exception as e:
        return jsonify({"error": _sanitize_error(e, "god_agent")}), 500


@app.route("/api/god/orchestre", methods=["POST"])
def api_god_orchestre():
    """HTTP twin of the MCP mathir_god_orchestre tool: list registered
    workers and pending results, hand back the dispatch instruction."""
    try:
        params = _get_params()
        directive = params.get("directive", "")
        project = params.get("project")
        cwd = params.get("cwd")

        agents_resp = _god_internal_call("/api/god/agents", {
            "project": project, "cwd": cwd,
        })
        agents = agents_resp.get("agents", [])
        idle_agents = [a for a in agents if a.get("status") == "idle"]

        search = _god_internal_call("/api/memory/smart_search", {
            "query": "god:result orchestrator", "k": 20,
            "project": project, "cwd": cwd,
        })
        pending = []
        for mem in search.get("results", []):
            label = mem.get("label", "")
            if label.startswith("god:result:") and ":completed" in label:
                pending.append(mem)

        return jsonify({
            "status": "ready",
            "directive": directive,
            "registered_workers": agents,
            "idle_workers": idle_agents,
            "pending_results": pending,
            "instruction": (
                f"DIRECTIVE: {directive}\n"
                "Break it into tasks and dispatch each via /api/memory/save with "
                "label='god:task:{8-char-hex}:{agent}:pending', "
                "content='{\"description\": ...}', block_type='working_memory', priority=7. "
                "Monitor with /api/memory/smart_search (query='god:result orchestrator'). "
                "Shutdown workers with label='god:task:00000000:{name}:shutdown'."
            ),
        })
    except Exception as e:
        return jsonify({"error": _sanitize_error(e, "god_orchestre")}), 500


def _god_reg_upsert(name, status, content, project=None, cwd=None):
    """Upsert a god:reg row, reusing the existing memory_id when present.

    Shared by /api/god/reg and /api/god/agent. Returns (memory_id, action,
    label). Raises ValueError on bad input.
    """
    import uuid
    if not name:
        raise ValueError("name is required")
    if status not in ("idle", "busy", "offline"):
        raise ValueError("status must be idle|busy|offline")

    vec_mem, _db_path, embedder = _resolve_db(project=project, cwd=cwd)
    conn = vec_mem._get_conn()

    # Normalize content: accept a dict OR a pre-serialized JSON string
    # (mathir_god_agent historically passed json.dumps({...}) as content).
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            content = {"raw": content}
    content_str = json.dumps(content, ensure_ascii=False)

    label = f"god:reg:{name}:{name}:{status}"
    safe_name = name.replace("%", r"\%").replace("_", r"\_")
    existing = conn.execute(
        """SELECT memory_id FROM memories
           WHERE label LIKE ? ESCAPE '\\'
             AND tier != 'archived'
           ORDER BY created_at DESC LIMIT 1""",
        (f"god:reg:{safe_name}:{safe_name}:%",),
    ).fetchone()
    memory_id = existing["memory_id"] if existing else f"mem_{uuid.uuid4().hex}"
    action = "updated" if existing else "created"

    emb_np = _encode_passage(embedder, content_str)
    metadata = {
        "agent": name,
        "block_type": "working_memory",
        "label": label,
        "priority": 3,
        "content": content_str,
        "project": project or get_project_name(),
        "risk_warnings": None,
        "file_path": "",
        "tier": "working_memory",
    }
    vec_mem.store(memory_id, emb_np, metadata)
    invalidate_on_write(project=project)
    return memory_id, action, label


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

# Schema of the resolved DB at last warmup â€” cached so /health and other
# request handlers don't re-init the embedder just to check the schema.
_DB_SCHEMA_KIND: dict = {}      # str(db_path) -> "new" | "legacy"
_DB_LEGACY_WARNING: dict = {}   # str(db_path) -> warning text


def _attach_legacy_warning(vec_mem, response: dict) -> None:
    """If the resolved DB is on legacy schema, surface a migration hint.

    The agent / MCP client will see this in every tool response so the
    user knows about the migration step without having to read daemon logs.
    """
    db_path = str(getattr(vec_mem, "db_path", "") or "")
    warning = _DB_LEGACY_WARNING.get(db_path)
    if not warning:
        # Fallback: re-check if the warmup dict wasn't populated
        try:
            kind = vec_mem._schema_kind()
            if kind == "legacy":
                warning = (
                    f"LEGACY SCHEMA detected at {db_path}. "
                    "Run: python -m mathir_mcp.mathir_lib.mathir_migrate --apply "
                    "(auto-creates .legacy.bak)"
                )
        except Exception:
            pass
    if warning:
        response["legacy_schema_warning"] = warning


def _warmup():
    """Pre-load embedder + DB in background thread."""
    log.info("Pre-loading embedder...")
    get_embedder()
    log.info("Embedder ready")
    try:
        vec_mem, db_path, embedder = _resolve_db()
        log.info("DB resolved")
        # Detect schema kind ONCE at warmup, cache globally so /health and
        # other request handlers don't re-init the embedder.
        global _DB_SCHEMA_KIND, _DB_LEGACY_WARNING
        try:
            kind = vec_mem._schema_kind()
            _DB_SCHEMA_KIND[str(db_path)] = kind
            if kind == "legacy":
                msg = (
                    f"LEGACY SCHEMA detected at {db_path}. "
                    "Run: python -m mathir_mcp.mathir_lib.mathir_migrate --dry-run "
                    "to preview, then --apply (auto-creates .legacy.bak). "
                    "Without migration, recall/save still work via fallback but "
                    "old columns (modality/modality_text) are read-only."
                )
                log.warning(msg)
                _DB_LEGACY_WARNING[str(db_path)] = msg
        except Exception as e:
            log.debug(f"schema kind detection skipped: {e}")
        # Auto-rebuild vec_memories if empty but memories exist (post-migration)
        try:
            pending = vec_mem.count()
            if pending > 0:
                from mathir_vec import HAS_VEC
                conn = vec_mem._get_conn()
                if HAS_VEC:
                    vec_count = conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0]
                else:
                    vec_count = conn.execute("SELECT COUNT(*) FROM embeddings_brute").fetchone()[0]
                if vec_count == 0 and pending > 0:
                    log.info(f"vec_memories empty but {pending} memories exist â€” rebuilding embeddings...")
                    result = vec_mem.rebuild_vec_index(embedder=embedder)
                    log.info(f"Vec rebuild: {result}")
        except Exception as e:
            log.warning(f"vec index rebuild check failed: {e}")
    except Exception as e:
        log.warning(f"DB warmup failed: {e}")


# ---------------------------------------------------------------------------
# Autonomous maintenance ("sleep") â€” periodic decay/promote/dedupe/link-build
# ---------------------------------------------------------------------------
# Without this, run_maintenance() in mathir_vec.py existed but nothing ever
# called it -- confirmed live, 2026-07-21: no scheduler/cron/background
# thread anywhere in the codebase, so decay/promotion/link-building only
# happened if an agent remembered to call the MCP tools by hand. MATHIR was
# documented as "a brain that dreams" but was purely reactive. This thread
# makes it actually autonomous: it periodically runs full maintenance on
# every DB currently open in this daemon (i.e. every project actively in
# use), so memory naturally decays/consolidates/promotes without any agent
# intervention -- the daemon does not go scanning the filesystem for
# unrelated project DBs it hasn't touched.
#
# Config source of truth is config['maintenance'] in the real runtime config
# (~/.config/MATHIR/config/mathir.json, per guardrail-real-runtime-config-path)
# -- NOT hardcoded here -- so the interval survives daemon restarts and is
# visible/editable in one place instead of an invisible env var default.
# MATHIR_MAINTENANCE_* env vars, if set, override the config file (same
# precedence as every other env-var override in this file).
def _maintenance_config():
    cfg = load_config().get("maintenance", {})
    enabled = os.environ.get("MATHIR_MAINTENANCE_ENABLED")
    enabled = (enabled not in ("0", "false", "False")) if enabled is not None else cfg.get("enabled", True)
    interval_hours = float(os.environ.get(
        "MATHIR_MAINTENANCE_INTERVAL_HOURS", cfg.get("interval_hours", 6)
    ))
    return {
        "enabled": enabled,
        "interval_hours": interval_hours,
        "do_decay": cfg.get("do_decay", True),
        "do_promote": cfg.get("do_promote", True),
        "do_dedupe": cfg.get("do_dedupe", True),
        "do_links": cfg.get("do_links", True),
    }


def _maintenance_loop():
    """Background thread: run_maintenance() on every cached DB, every N hours."""
    cfg = _maintenance_config()
    if not cfg["enabled"]:
        log.info("Autonomous maintenance disabled (config['maintenance']['enabled']=false)")
        return
    interval_s = max(cfg["interval_hours"], 0.1) * 3600
    log.info(f"Autonomous maintenance thread started (every {cfg['interval_hours']}h, cfg={cfg})")
    # Let warmup finish and give the daemon a moment to settle before the
    # first sweep, rather than racing warmup for the DB lock at t=0.
    _SHUTTING_DOWN.wait(timeout=120)
    while not _SHUTTING_DOWN.is_set():
        # Re-read each cycle so editing mathir.json takes effect on the next
        # sweep without requiring a daemon restart.
        cfg = _maintenance_config()
        if not cfg["enabled"]:
            log.info("Autonomous maintenance disabled mid-run, stopping loop")
            return
        with _vec_cache_lock:
            targets = list(_vec_cache.items())
        for (db_path, _dim), vec_mem in targets:
            if _SHUTTING_DOWN.is_set():
                break
            try:
                result = vec_mem.run_maintenance(
                    do_decay=cfg["do_decay"],
                    do_promote=cfg["do_promote"],
                    do_dedupe=cfg["do_dedupe"],
                    do_links=cfg["do_links"],
                )
                log.info(f"Autonomous maintenance for {Path(db_path).name}: {result}")
            except Exception as e:
                log.warning(f"Autonomous maintenance failed for {db_path}: {e}")
        interval_s = max(cfg["interval_hours"], 0.1) * 3600
        _SHUTTING_DOWN.wait(timeout=interval_s)


# ---------------------------------------------------------------------------
# Graceful shutdown + single-instance lock + crash logging
# ---------------------------------------------------------------------------
_SHUTTING_DOWN = threading.Event()


def _shutdown(reason: str = "unknown") -> None:
    """Close all cached VecMemory connections so SQLite WAL is checkpointed.

    Idempotent â€” safe to call from signal handler and atexit. Without this,
    a hard taskkill leaves the -wal sidecar un-checkpointed and the next
    startup can fail to bind or read stale data.
    """
    if _SHUTTING_DOWN.is_set():
        return
    _SHUTTING_DOWN.set()
    log.info(f"Shutting down (reason: {reason})")
    with _vec_cache_lock:
        for (db_path, _dim), vec_mem in list(_vec_cache.items()):
            try:
                close = getattr(vec_mem, "close", None)
                if close:
                    close()
                    log.info(f"Closed VecMemory for {Path(db_path).name}")
            except Exception as e:
                log.warning(f"Error closing VecMemory {db_path}: {e}")
        _vec_cache.clear()
    # Remove PID file
    try:
        _PID_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def _install_signal_handlers() -> None:
    """Catch SIGINT/SIGTERM/SIGBREAK for graceful shutdown + flush log buffer."""
    def _handler(signum, _frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        log.info(f"Received signal {sig_name}")
        _shutdown(sig_name)
        # Give the logger a beat to flush, then exit non-zero so the watchdog
        # knows the slot is free for a fresh start.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if signum == signal.SIGINT else 0)

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass  # Not a main thread or signal unsupported on this platform


def _install_excepthooks() -> None:
    """Capture uncaught exceptions into the rotating log file."""
    def _sys_hook(exc_type, exc, tb):
        is_kb = issubclass(exc_type, KeyboardInterrupt)
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        if is_kb:
            log.info("Interrupted by user (KeyboardInterrupt)")
        else:
            log.error(f"Uncaught exception:\n{msg}")
        if not is_kb:
            sys.__excepthook__(exc_type, exc, tb)

    def _thread_hook(args):
        msg = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        log.error(
            f"Uncaught exception in thread {args.thread.name!r} "
            f"({args.exc_type.__name__}): {msg}"
        )

    sys.excepthook = _sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook


def _pid_alive(pid: int) -> bool:
    """Best-effort cross-platform 'is this PID a running process' check."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(h)
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError, PermissionError):
        return False
    except Exception:
        return False


def _acquire_pid_lock() -> bool:
    """Return True if we may start; False if another live server owns the lock."""
    if _PID_PATH.exists():
        try:
            old_pid = int(_PID_PATH.read_text().strip())
            if _pid_alive(old_pid):
                log.warning(
                    f"Another mathir_server appears to be running (PID {old_pid}, "
                    f"pidfile {_PID_PATH}). Refusing to start to avoid a port/DB race."
                )
                return False
        except (ValueError, OSError):
            pass
        try:
            _PID_PATH.unlink()
        except OSError:
            pass
    try:
        _PID_PATH.write_text(str(os.getpid()))
    except OSError as e:
        log.warning(f"Could not write PID file {_PID_PATH}: {e}")
    return True


def main():
    parser = argparse.ArgumentParser(description="MATHIR Unified Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MATHIR_PORT", "7338")))
    parser.add_argument("--host", default=os.environ.get("MATHIR_HOST", "127.0.0.1"))
    parser.add_argument("--workers", type=int, default=4, help="Waitress threads")
    parser.add_argument("--force", action="store_true",
                        help="Ignore stale PID lock and start anyway")
    args = parser.parse_args()

    _install_excepthooks()

    if not args.force and not _acquire_pid_lock():
        sys.exit(2)

    _install_signal_handlers()
    import atexit
    atexit.register(_shutdown, "atexit")

    log.info(f"MATHIR server starting on {args.host}:{args.port} "
             f"(PID {os.getpid()}, log {_LOG_PATH})")

    # --- Auth gate: non-loopback requires MATHIR_AUTH_TOKEN ---
    if not _is_loopback(args.host):
        if not _AUTH_TOKEN:
            print(
                f"Refusing to bind non-loopback (host={args.host}) without "
                f"MATHIR_AUTH_TOKEN. Set the env var or bind 127.0.0.1.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Install bearer-token auth hook for /api/ paths (skip /health, /ping)
        @app.before_request
        def _require_bearer():
            if request.path.startswith('/api/') and request.path not in ('/api/health', '/api/ping'):
                auth = request.headers.get('Authorization', '')
                if auth != f'Bearer {_AUTH_TOKEN}':
                    return jsonify({'error': 'unauthorized'}), 401

    # Warm up in background
    t = threading.Thread(target=_warmup, daemon=True)
    t.start()

    # Autonomous maintenance ("sleep") â€” decay/promote/dedupe/link-build
    mt = threading.Thread(target=_maintenance_loop, daemon=True, name="mathir-maintenance")
    mt.start()

    try:
        from waitress import serve
        log.info("Using waitress (production WSGI)")
        serve(app, host=args.host, port=args.port, threads=args.workers)
    except ImportError:
        log.warning("waitress not installed, using Flask dev server (not for production)")
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        _shutdown("serve returned")


if __name__ == "__main__":
    main()
