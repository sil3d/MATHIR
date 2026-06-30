#!/usr/bin/env python3
"""
MATHIR Unified Server — v8.5.0
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
import os
import json
import time
import signal
import threading
import logging
import logging.handlers
import argparse
import traceback
from typing import Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Logging — stderr + rotating file (independent of launcher pipe redirection
# so crash traces survive even when stdout/stderr are DEVNULL'd by a watchdog)
# ---------------------------------------------------------------------------
try:
    from .mathir_paths import LOG_DIR as _P_LOG, PROJECTS_DIR as _P_PROJECTS
    from .mathir_paths import LEGACY_DB_PATH as _P_DB, CONFIG_PATH as _P_CONFIG
    from .mathir_paths import REGISTRY_PATH as _P_REGISTRY
except ImportError:
    from mathir_paths import LOG_DIR as _P_LOG, PROJECTS_DIR as _P_PROJECTS
    from mathir_paths import LEGACY_DB_PATH as _P_DB, CONFIG_PATH as _P_CONFIG
    from mathir_paths import REGISTRY_PATH as _P_REGISTRY

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
    pass  # File logging unavailable — stderr still works
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
# Imports — mathir_lib
# ---------------------------------------------------------------------------
from mathir_mcp_server import (
    get_embedder,
    get_project_db_path,
    get_project_name,
)
from mathir_push import ContextAnalyzer, PushCache, context_hash, deduplicate_memories

# Risk mitigation (optional)
try:
    from memory_risks import DomainClassifier, LeakageDetector, SycophancyDetector
    _risk_enabled = True
except ImportError:
    _risk_enabled = False

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_push_cache = PushCache(ttl_seconds=300, max_size=200)
_push_analyzer = ContextAnalyzer()
_push_lock = threading.Lock()

_vec_cache = {}
_vec_cache_lock = threading.Lock()

_start_time = time.time()

# ---------------------------------------------------------------------------
# Auth gate — non-loopback binds REQUIRE MATHIR_AUTH_TOKEN (opt-in, non-breaking)
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

    Routing priority (v8.5.1 fix):
      1. Explicit project + cwd from MCP request -> use cwd/.mathir/mathir.db
      2. Just project -> ~/.config/MATHIR/data/projects/<project>/mathir.db
      3. No project/cwd -> fall back to get_project_db_path() (CWD-based)
    """
    dim = get_embedder_dim()
    db_path = None
    if cwd and project:
        cwd_path = Path(cwd)
        candidate = cwd_path / ".mathir" / "mathir.db"
        if candidate.parent.exists():
            db_path = candidate
        else:
            # Create .mathir/ alongside the project (use project name for DB)
            db_path = Path(os.environ.get("MATHIR_HOME", str(Path.home() / ".config" / "mathir"))) / "data" / "projects" / project / "mathir.db"
    elif project:
        db_path = Path(os.environ.get("MATHIR_HOME", str(Path.home() / ".config" / "mathir"))) / "data" / "projects" / project / "mathir.db"
    if db_path is None:
        db_path = get_project_db_path(project=project)
    if db_path is None:
        raise ValueError("No project database found. Set MATHIR_PROJECT env var or pass project + cwd.")
    vec_mem = _get_vec_mem(db_path, dim)
    embedder = get_embedder()
    return vec_mem, db_path, embedder


def _encode_query(embedder, query: str):
    emb = embedder.encode(query)
    import numpy as np
    if hasattr(emb, 'cpu'):
        return emb.cpu().numpy().astype('float32').reshape(-1)
    return np.array(emb, dtype=np.float32).reshape(-1)


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
# CORS — allow browsers (Tauri webview, Vite dev :3000) to call MATHIR directly
# ---------------------------------------------------------------------------
@app.after_request
def _add_cors_headers(resp):
    origin = request.headers.get("Origin", "*")
    # Restrict to localhost variants — MATHIR is a local-only service
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
    tiers = {"working": 0, "episodic": 0, "semantic": 0, "procedural": 0, "immunological": 0, "unknown": 0}
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
    query = "SELECT memory_id, modality_text, metadata, tier, timestamp FROM memories WHERE 1=1"
    params = []
    if agent_filter:
        query += " AND json_extract(metadata, '$.agent') = ?"
        params.append(agent_filter)
    if tier_filter:
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
    """Make memory text safe to embed in an LLM system prompt: strip
    block-delim / template tokens and CR/LF so a stored memory can't break out
    of the injection region or pose as new instructions."""
    if not text:
        return ""
    s = str(text).replace("\r", " ").replace("\n", " ")
    for tok in ("</mathir-", "{{MATHIR_CONTEXT}}", "<|", "### "):
        s = s.replace(tok, tok.strip())
    return s


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
    try:
        vec_mem, db_path, embedder = _resolve_db(project=project, cwd=request.args.get("cwd") if request.method == "GET" else (request.get_json(silent=True) or {}).get("cwd"))
        query_vec = _encode_query(embedder, task)
        results = vec_mem.search(query_vec, k=k)
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
        return jsonify({"error": str(e)}), 500
    # Group by tier
    tiers: dict[str, list] = {}
    for r in normalized:
        tier = r.get("block_type", "unknown")
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
    lines = [f"## MATHIR Auto-Context — {len(normalized)} memories for: {_sanitize_for_prompt(task)[:100]}"]
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
        "total": len(normalized),
        "task": task[:200],
    })


@app.route("/api/projects")
def api_projects():
    return jsonify({"projects": _list_projects()})


# ---------------------------------------------------------------------------
# Health + startup
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    resp = {
        "status": "ok",
        "uptime": round(time.time() - _start_time, 1),
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "version": "8.5.0",
    }
    # Surface the legacy-schema warning on /health so the agent plugin sees it
    # immediately at session start (the plugin polls /health on session.started).
    # Uses cached schema kind from warmup — no re-init of embedder.
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


@app.route("/api/ping")
def api_ping():
    return jsonify({"pong": True, "uptime": round(time.time() - _start_time, 1), "dim": get_embedder_dim()})


# ---------------------------------------------------------------------------
# Memory API routes — all POST, JSON body
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

        emb_np = _encode_query(embedder, content)
        import uuid
        memory_id = f"mem_{uuid.uuid4().hex}"
        metadata = {
            'agent': params.get('agent', 'unknown'),
            'block_type': params.get('block_type', 'episodic'),
            'label': params.get('label', ''),
            'priority': params.get('priority', 5),
            'content': content,
            'project': params.get('project') or get_project_name(),
            'risk_warnings': risk_warnings if risk_warnings else None,
        }
        vec_mem.store(memory_id, emb_np, metadata)
        resp = {'memory_id': memory_id, 'saved': True, 'metadata': metadata}
        _attach_legacy_warning(vec_mem, resp)
        return jsonify(resp)
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_save')}), 500


@app.route("/api/memory/recall", methods=["POST"])
def memory_recall():
    params = _get_params()
    err = _validate(params)
    if err:
        return err
    try:
        vec_mem, _db_path, embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        query = params.get('query', '')
        k = min(params.get('k', 5), 1000)
        q_np = _encode_query(embedder, query)
        results = vec_mem.search(
            query_embedding=q_np, k=k,
            agent_filter=params.get('agent'),
            block_type_filter=params.get('block_type'),
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
        return jsonify({'results': results, 'query': query, 'total': len(results), 'touched': touched})
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


@app.route("/api/memory/delete", methods=["POST"])
def memory_delete():
    params = _get_params()
    try:
        vec_mem, _db_path, _embedder = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        memory_id = params.get('memory_id')
        if not memory_id:
            return jsonify({'error': 'memory_id required'}), 400
        deleted = vec_mem.delete(memory_id)
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
        results = vec_mem.search(query_embedding=q_np, k=k, agent_filter=params.get('agent'))
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
            q_blob = _serialize_embedding(q_np)
            sql = """
                SELECT m.memory_id, v.distance
                FROM vec_memories v
                JOIN memories m ON v.memory_id = m.memory_id
                WHERE v.embedding MATCH ? AND k = ?
            """
            params_list = [q_blob, k * 3]
            if agent_filter:
                sql += " AND m.agent = ?" if 'agent' in columns else " AND json_extract(m.metadata, '$.agent') = ?"
                params_list.append(agent_filter)
            for row in dconn.execute(sql, params_list).fetchall():
                vector_results.append((row['memory_id'], 1.0 - row['distance']))

        bm25_results = []
        try:
            rows = dconn.execute(f"SELECT memory_id, {text_col} FROM memories").fetchall()
        except Exception:
            rows = []
        if rows:
            from mathir_search import _tokenize
            from rank_bm25 import BM25Okapi
            corpus_ids = [r['memory_id'] for r in rows]
            corpus_texts = [r[1] or '' for r in rows]
            tokenized = [_tokenize(t) for t in corpus_texts]
            if tokenized:
                bm25 = BM25Okapi(tokenized)
                scores = bm25.get_scores(_tokenize(query_text))
                for mid, sc in sorted(zip(corpus_ids, scores), key=lambda x: x[1], reverse=True):
                    if sc > 0:
                        bm25_results.append((mid, float(sc)))
                    if len(bm25_results) >= k * 3:
                        break

        from mathir_search import rrf_fusion
        fused = rrf_fusion(vector_results, bm25_results, vector_weight=vector_weight, bm25_weight=bm25_weight)

        results = []
        for mid, rrf_score in fused[:k]:
            meta = dconn.execute(f"SELECT {text_col}, tier, timestamp FROM memories WHERE memory_id = ?", [mid]).fetchone()
            if not meta:
                continue
            agent_val = ''
            if 'agent' in columns:
                agent_val = dconn.execute("SELECT agent FROM memories WHERE memory_id = ?", [mid]).fetchone()[0] or ''
            results.append({
                'memory_id': mid, 'rrf_score': rrf_score, 'content': meta[0] or '',
                'agent': agent_val, 'score': rrf_score, 'created_at': meta[2] or '', 'tier': meta[1] or 'episodic',
            })
        dconn.close()
        return jsonify({
            'results': results, 'query': query_text, 'total': len(results),
            'mode': 'hybrid', 'vector_hits': len(vector_results), 'bm25_hits': len(bm25_results),
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
        return jsonify(vec_mem.promote(params.get('memory_id', ''), force=params.get('force', False)))
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_promote')}), 500


@app.route("/api/memory/auto_promote", methods=["POST"])
def memory_auto_promote():
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        promoted = vec_mem.auto_promote_all()
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
        return jsonify(vec_mem.consolidate_all(
            threshold=params.get('threshold', 0.95),
            limit=params.get('limit', 100),
            dry_run=params.get('dry_run', True),
        ))
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
        return jsonify(vec_mem.build_links_all(
            threshold=params.get('threshold', 0.7),
            limit=params.get('limit', 1000),
        ))
    except Exception as e:
        return jsonify({'error': _sanitize_error(e, 'memory_build_links')}), 500


@app.route("/api/memory/audit", methods=["POST", "GET"])
def memory_audit():
    """Audit log of recent operations."""
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        agent = params.get("agent")
        limit = params.get("limit", 50)
        conn = vec_mem._get_conn()
        # Check if audit table exists
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "memory_audit" not in tables:
            return jsonify({"entries": [], "total": 0, "note": "audit table not yet created"})
        if agent:
            rows = conn.execute(
                "SELECT * FROM memory_audit WHERE agent=? ORDER BY rowid DESC LIMIT ?",
                (agent, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memory_audit ORDER BY rowid DESC LIMIT ?",
                (limit,)
            ).fetchall()
        entries = [dict(r) for r in rows] if rows else []
        return jsonify({"entries": entries, "total": len(entries)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/export", methods=["POST", "GET"])
def memory_export():
    """Export all memories as JSON."""
    params = _get_params()
    try:
        vec_mem, _, _ = _resolve_db(project=params.get("project"), cwd=params.get("cwd"))
        conn = vec_mem._get_conn()
        rows = conn.execute("SELECT memory_id, tier, stability, recall_count, timestamp FROM memories ORDER BY rowid").fetchall()
        memories = [dict(r) for r in rows] if rows else []
        return jsonify({"memories": memories, "total": len(memories)})
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
        # Sessions are derived from metadata JSON field
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?",
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
                    "timestamp": r["timestamp"],
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
# Startup
# ---------------------------------------------------------------------------

# Schema of the resolved DB at last warmup — cached so /health and other
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
        vec_mem, db_path, _embedder = _resolve_db()
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
    except Exception as e:
        log.warning(f"DB warmup failed: {e}")


# ---------------------------------------------------------------------------
# Graceful shutdown + single-instance lock + crash logging
# ---------------------------------------------------------------------------
_SHUTTING_DOWN = threading.Event()


def _shutdown(reason: str = "unknown") -> None:
    """Close all cached VecMemory connections so SQLite WAL is checkpointed.

    Idempotent — safe to call from signal handler and atexit. Without this,
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
