#!/usr/bin/env python3
"""
MATHIR MCP Server v3 — Thin proxy to daemon (port 7338).
NO embedder loading — daemon handles all embedding.
Safe for multiple concurrent OpenCode sessions.
Keeps get_embedder/get_project_db_path/get_project_name for daemon compatibility.
"""

import hashlib
import json
import re
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
_CLIENT_BLOCK_TYPES = {"working_memory", "episodic", "semantic", "procedural", "guardrail"}

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


# Asymmetric-retrieval embedders need different text prefixes for queries
# vs. stored passages (they were trained with these prefixes; using them
# is what gives the retrieval-quality gain -- see MATHIR memory
# embedder-swap-strongest-positive-result-hotpotqa). Models not listed here
# (e.g. the default paraphrase-trained MiniLM) use no prefix at all.
MODEL_PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    "intfloat/e5-large-v2": ("query: ", "passage: "),
}


def get_model_prefixes(model_name: str) -> tuple:
    """Return (query_prefix, passage_prefix) for a given embedding model.
    Unknown/unregistered models default to no prefix (safe no-op)."""
    return MODEL_PREFIXES.get(model_name, ("", ""))


def get_embedder(model_name: str = None):
    """Load embedder on demand (for daemon compatibility). CACHED PER MODEL.

    ROOT-CAUSE FIX (2026-07-02): this repeatedly hit "Cannot copy out of
    meta tensor; no data! Please use torch.nn.Module.to_empty() instead of
    torch.nn.Module.to()" -- THREE times this session, in different guises
    (at construction, on first .encode(), and even on the CPU device).
    Earlier fixes assumed it was CUDA-specific and fell back to CPU, but
    the CPU path hit the identical error -- proving it was never a device
    problem. Root cause: recent transformers/accelerate versions can
    lazily initialize weights on a "meta" device (for low-memory loading)
    and only materialize them via `.to_empty()`; SentenceTransformer's
    internal `.to(device)` call is incompatible with that path regardless
    of target device. Fix: pass model_kwargs={"low_cpu_mem_usage": False}
    to force eager weight materialization, which sidesteps the meta-device
    path entirely. Verified live: identical model/device that failed with
    NotImplementedError on both cuda and cpu now loads and encodes
    successfully with this kwarg.

    Tries CUDA first when available, falls back to CPU on any load failure
    (now a real device-capability fallback, not a workaround for this bug).

    model_name: when omitted, uses the configured default (embedding.model
    in mathir.json). When given explicitly, loads/caches THAT model instead
    -- this is what lets different project DBs use different embedding
    models within the same daemon process (see VecMemory.ensure_embedding_
    model): an existing DB pinned to the old default keeps working even
    after the configured default changes for new DBs.
    """
    global _cached_embedders
    if model_name is None:
        model_name = load_config().get("embedding", {}).get(
            "model", "intfloat/multilingual-e5-small"
        )
    if model_name in _cached_embedders:
        return _cached_embedders[model_name]
    from sentence_transformers import SentenceTransformer
    import torch
    embedder = None
    if torch.cuda.is_available():
        try:
            embedder = SentenceTransformer(model_name, device="cuda",
                                           model_kwargs={"low_cpu_mem_usage": False})
            embedder.encode("warmup", show_progress_bar=False)
        except Exception as e:
            log.warning(f"CUDA embedder load/encode failed ({e}); falling back to CPU")
            embedder = None
    if embedder is None:
        embedder = SentenceTransformer(model_name, device="cpu",
                                       model_kwargs={"low_cpu_mem_usage": False})
        embedder.encode("warmup", show_progress_bar=False)
    _cached_embedders[model_name] = embedder
    return embedder


_cached_embedders = {}


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
    # Always inject the calling project's context so the daemon can route
    # writes/reads to the right per-project DB instead of falling back to its
    # own CWD. Critical: the daemon was launched from somewhere (Startup
    # folder or shell) — its CWD is NOT the agent's project CWD.
    if "project" not in clean or not clean.get("project"):
        clean["project"] = get_project_name()
    clean["cwd"] = str(Path.cwd())
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
        "memory_audit_immunological": "/api/memory/audit_immunological",
        "memory_guardrails": "/api/memory/guardrails",
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
        "god_poll": "/api/god/poll",
        "god_agents": "/api/god/agents",
        "god_ack": "/api/god/ack",
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
    "memory_stats", "memory_audit", "memory_audit_immunological", "memory_export", "memory_sessions",
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
    file_path: str = "",
) -> str:
    """Save a memory. Block types: working_memory, episodic, semantic, procedural, guardrail.

    Pass block_type="auto" to let MATHIR classify the content based on simple
    heuristics (commands/how-tos → procedural, bugs/decisions → episodic,
    facts/general knowledge → semantic, scratchpad → working_memory).

    block_type="guardrail" saves a critical rule that is ALWAYS auto-injected
    into every /api/context, memory_session_start, and memory_context response.
    Guardrails are immune to decay, cannot be promoted, and have a minimum
    priority of 8. Max 50 guardrails per project.

    file_path: optional source file this memory is about (e.g.
    "mathir_lib/mathir_vec.py"). Enables memory_by_path to actually filter on
    a real structured field instead of falling back to content text search.
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

    if block_type == "guardrail":
        try:
            from . import GUARDRAIL_MIN_PRIORITY
        except ImportError:
            GUARDRAIL_MIN_PRIORITY = 8
        priority = max(priority, GUARDRAIL_MIN_PRIORITY)

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
    if file_path:
        params["file_path"] = file_path
    
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
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    rerank: bool = False,
    project: str = None,
) -> str:
    """Hybrid search: vector + BM25 + RRF fusion + optional cross-encoder reranking.

    Set rerank=True to apply a cross-encoder second pass on the top candidates.
    This is slower (~100ms) but significantly more accurate for ranking precision.
    """
    _err = _check_lengths(query=query, agent=agent)
    if _err:
        return json.dumps(_err)

    result = _call_daemon("memory_hybrid_search", {
        "query": query,
        "k": k,
        "agent": agent,
        "vector_weight": vector_weight,
        "bm25_weight": bm25_weight,
        "rerank": rerank,
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
    """Dashboard view: recent activity, guardrail roster, save trend.

    Distinct from memory_stats (compact tier/agent counts) -- this used to
    silently proxy memory_stats verbatim, returning identical output.
    """
    result = _call_daemon("memory_dashboard", {})
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
    limit: int = 100,
    max_results: int = 50,
) -> str:
    """Merge near-duplicate memories.

    dry_run=True returns a compact preview (capped at max_results pair
    previews with ~100-char snippets) instead of the full per-pair report.
    """
    result = _call_daemon("memory_consolidate", {
        "threshold": threshold,
        "dry_run": dry_run,
        "limit": limit,
        "max_results": max_results,
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
def memory_build_links(threshold: float = 0.88, limit: int = 1000) -> str:
    """Build link graph from cosine similarities.

    threshold default raised from 0.7 to 0.88 -- 0.7 produced an almost-
    complete graph (442,890 links from 666 memories) against this project's
    real embedding model, useless as a "related memories" signal.
    """
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

    Filters on the real metadata.file_path SQL field OR content text match
    against the path string.

    Use case: "show me what I know about mathir_vec.py:142" → returns all memories
    whose content or metadata mentions that file or location.
    """
    try:
        # Direct SQL filter on file_path -- fixed 2026-07-21 from an earlier
        # version that ran a semantic memory_recall for the path string and
        # post-filtered that pool, which silently missed memories whose
        # metadata.file_path was correctly set but whose content didn't
        # embed close to the path string (a structured field deserves a SQL
        # filter, not embedding similarity as a proxy for it).
        result = _call_daemon("memory_by_path", {"file_path": file_path, "k": k})
        return json.dumps(result) if isinstance(result, dict) else str(result)
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
        top1_meta = results[0].get("metadata") or {}
        top1_text = " ".join(str(x) for x in (
            top1_meta.get("content", ""), results[0].get("content", ""),
            top1_meta.get("label", ""), results[0].get("label", ""),
        )).lower()
        # Cosine similarity alone is unreliable at the top end -- verified
        # live, 2026-07-21: a deliberately nonsensical out-of-domain query
        # ("completely nonsense gibberish query xyz123") still scored 0.839
        # ("high") purely from embedding-space coincidence, with zero actual
        # word overlap with the matched memory. Require at least one >=4-char
        # query token to literally appear in the top result before trusting
        # "high" -- a real match should share vocabulary, not just land in a
        # similar region of embedding space.
        query_tokens = set(re.findall(r"[a-z0-9]{4,}", query.lower()))
        lexically_grounded = any(tok in top1_text for tok in query_tokens)

        if top1 >= 0.7 and lexically_grounded:
            quality = "high"
            suggestion = "Strong match — top result is highly relevant."
        elif top1 >= 0.7:
            quality = "medium"
            suggestion = (
                f"Top-1 score {top1:.2f} looks strong but shares no vocabulary with the "
                "query -- likely an embedding-space coincidence, not a real match. Review "
                "before trusting."
            )
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


@mcp.tool()
def memory_audit_immunological(project: str = None, k: int = 20) -> str:
    """List memories flagged in the immunological (anomaly) tier. Read-only —
    this tier can only be populated by the internal anomaly detector."""
    result = _call_daemon("memory_audit_immunological", {
        "project": project,
        "k": k,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


@mcp.tool()
def memory_list_guardrails(project: str = None) -> str:
    """List all guardrail memories for the current project.

    Guardrails are critical rules that are ALWAYS auto-injected into every
    context response (session_start, memory_context, /api/context). They are
    immune to decay and cannot be promoted. Use memory_save with
    block_type="guardrail" to create new guardrails.
    """
    result = _call_daemon("memory_guardrails", {
        "project": project,
        "k": 50,
    })
    return json.dumps(result) if isinstance(result, dict) else str(result)


# ---------------------------------------------------------------------------
# God Orchestrator tools (v8.8.0)
# ---------------------------------------------------------------------------

@mcp.tool()
def mathir_god_agent(
    name: str = "",
    capabilities: str = "",
    introduction: str = "",
    poll_interval: int = 8,
    worktree: bool = True,
) -> str:
    """Register as a God worker agent and poll once for a pending task.

    Call with NO arguments to start the self-identification flow:
    - Step 1: Call mathir_god_agent() → you get an "identify" instruction
    - Step 2: Assess yourself honestly, then call again with your identity:
      mathir_god_agent(name="...", capabilities="...", introduction="...")
    - Step 3: You're registered. The orchestrator will assign tasks based on your profile.

    Call with name="help" for a full usage guide.

    Args:
        name: Your unique worker ID. Leave empty for self-identification.
        capabilities: Comma-separated skills (e.g. "code,test,debug,review,docs")
        introduction: Your self-assessment — who you are, your strengths, weaknesses, what you excel at.
        poll_interval: Seconds between polls suggested to caller (default 8)
        worktree: Use git worktree for task isolation (default True)
    """
    if name == "help":
        return json.dumps({
            "status": "help",
            "guide": (
                "=== MATHIR GOD MODE — WORKER GUIDE ===\n\n"
                "God Mode lets you orchestrate multiple AI agents across terminals.\n"
                "This tool makes your agent a WORKER that receives and executes tasks.\n\n"
                "--- QUICK START (3 steps) ---\n\n"
                "1. Call mathir_god_agent() with NO arguments\n"
                "   → You'll get an 'identify' instruction\n\n"
                "2. Assess yourself honestly, then call again:\n"
                "   mathir_god_agent(\n"
                "       name='claude-code',\n"
                "       capabilities='code,refactor,debug,architecture',\n"
                "       introduction='I am Claude Code by Anthropic. I excel at complex refactoring,\n"
                "                     multi-file architecture, and debugging. I am a thinking model\n"
                "                     with deep reasoning. Weakness: I am slower than lightweight agents.'\n"
                "   )\n\n"
                "3. You're registered! Keep calling mathir_god_agent() to poll for tasks.\n"
                "   When a task arrives, execute it and report back as instructed.\n\n"
                "--- WHAT HAPPENS ---\n\n"
                "• The orchestrator sees your profile (name, capabilities, introduction)\n"
                "• It assigns tasks based on your strengths vs other workers' strengths\n"
                "• You get a task → execute it → report results → poll for next task\n\n"
                "--- SELF-IDENTIFICATION TIPS ---\n\n"
                "Be honest about:\n"
                "  • What model/agent you are (Claude, GPT, MiMo, Codex, etc.)\n"
                "  • Your strengths (reasoning, speed, code quality, testing, etc.)\n"
                "  • Your weaknesses (slow, limited context, no web access, etc.)\n"
                "  • What tasks you're BEST at vs merely capable of\n"
                "  • Your tool access (can you edit files? run tests? search web?)\n\n"
                "The orchestrator uses this to make optimal assignments.\n"
                "Lying about your abilities means you'll get tasks you can't handle well.\n\n"
                "--- RETURN VALUES ---\n\n"
                "status='identify'   → First call, no name. Tell the tool who you are\n"
                "status='waiting'    → No task yet. Call again after poll_interval seconds\n"
                "status='task_found' → Task assigned! Read 'instruction' and 'report_instruction'\n"
                "status='shutdown'   → Orchestrator says stop. Exit the loop\n\n"
                "--- TIPS ---\n\n"
                "• Use Claude Code /loop to auto-poll repeatedly\n"
                "• Call mathir_god_orchestre(directive='help') for orchestrator guide"
            ),
        })

    if not name:
        return json.dumps({
            "status": "identify",
            "instruction": (
                "SELF-IDENTIFICATION REQUIRED\n\n"
                "You are joining a multi-agent team. Before you can receive tasks, "
                "the orchestrator needs to know who you are.\n\n"
                "Assess yourself honestly and call mathir_god_agent again with:\n\n"
                "  name:          A unique ID for yourself (e.g. 'claude-code', 'mimo', 'codex-1')\n"
                "  capabilities:  Comma-separated list of what you can do.\n"
                "                 Common: code, test, debug, review, docs, refactor, architecture,\n"
                "                 search, analysis, fast-execution\n"
                "  introduction:  A honest self-assessment (2-4 sentences):\n"
                "                 - What agent/model are you?\n"
                "                 - What are you BEST at? (not just capable — where do you SHINE?)\n"
                "                 - What are your weaknesses or limitations?\n"
                "                 - What tools do you have access to?\n\n"
                "Example:\n"
                "  mathir_god_agent(\n"
                "      name='claude-code',\n"
                "      capabilities='code,refactor,debug,architecture,review',\n"
                "      introduction='I am Claude Code (Opus). I excel at complex multi-file refactoring, "
                "architecture decisions, and deep debugging. I have full file system access, can run "
                "tests, and use git. Weakness: I am slower and more expensive than lightweight models. "
                "Best used for tasks requiring deep reasoning, not bulk mechanical work.'\n"
                "  )\n\n"
                "BE HONEST. The orchestrator assigns tasks based on your profile. "
                "Overestimating yourself means getting tasks you'll do poorly. "
                "Underestimating means missing tasks you'd excel at."
            ),
        })

    caps = [c.strip() for c in capabilities.split(",") if c.strip()]
    reg_content = json.dumps({
        "capabilities": caps,
        "introduction": introduction,
    })

    _call_daemon_raw("memory_save", {
        "content": reg_content,
        "agent": name,
        "block_type": "working_memory",
        "label": f"god:reg:{name}:{name}:idle",
        "priority": 3,
    })

    log.info(f"[God Worker] {name} registered with capabilities: {caps}")
    result_lines = [f"[God Worker] {name} registered. Capabilities: {caps}. poll_interval={poll_interval}s."]
    if introduction:
        result_lines.append(f"[God Worker] Introduction saved: {introduction[:100]}...")

    try:
        resp = _call_daemon_raw("god_poll", {"agent": name, "status": "pending"})
    except Exception as e:
        log.warning(f"[God Worker] Poll error: {e}")
        return json.dumps({
            "status": "waiting",
            "reason": f"poll_error: {e}",
            "poll_interval": poll_interval,
            "instruction": f"Call mathir_god_agent again after {poll_interval}s",
            "log": result_lines,
        })

    task = resp.get("task") if isinstance(resp, dict) else None
    if not task:
        return json.dumps({
            "status": "waiting",
            "reason": "no_pending_task",
            "poll_interval": poll_interval,
            "instruction": f"Call mathir_god_agent again after {poll_interval}s",
            "log": result_lines,
        })

    task_label = task.get("label", "")
    from mathir_god import GodProtocol
    parsed = GodProtocol.parse_label(task_label)
    if parsed and parsed["status"] == "shutdown":
        _call_daemon_raw("memory_save", {
            "content": "shutdown",
            "agent": name,
            "block_type": "working_memory",
            "label": f"god:reg:{name}:{name}:offline",
            "priority": 3,
        })
        result_lines.append(f"[God Worker] {name} received shutdown.")
        return json.dumps({"status": "shutdown", "log": result_lines})

    task_id = parsed["id"] if parsed else "unknown"
    task_content = task.get("content", "")
    try:
        task_info = json.loads(task_content)
    except (json.JSONDecodeError, TypeError):
        task_info = {"description": task_content}

    description = task_info.get("description", task_content)

    # Accept task — flip the ORIGINAL pending memory's label in place via
    # god_ack. god_poll always returns the oldest still-"pending" row for
    # this agent; merely memory_save-ing a *new* "...:running" label (the
    # old behavior) left the original "...:pending" row untouched, so every
    # subsequent poll kept re-matching and re-serving the same stale task
    # forever instead of advancing to the next one.
    task_memory_id = task.get("memory_id", "")
    if task_memory_id:
        _call_daemon_raw("god_ack", {
            "memory_id": task_memory_id,
            "status": "running",
        })
    else:
        _call_daemon_raw("memory_save", {
            "content": "accepted",
            "agent": name,
            "block_type": "working_memory",
            "label": f"god:task:{task_id}:{name}:running",
            "priority": 7,
        })
    _call_daemon_raw("memory_save", {
        "content": reg_content,
        "agent": name,
        "block_type": "working_memory",
        "label": f"god:reg:{name}:{name}:busy",
        "priority": 3,
    })

    result_lines.append(f"[God Worker] Accepted task {task_id}: {description[:80]}")

    return json.dumps({
        "status": "task_found",
        "task_id": task_id,
        "description": description,
        "instruction": f"EXECUTE THIS TASK: {description}",
        "report_instruction": (
            f"After completing the task, report results by calling memory_save with: "
            f"label='god:result:{task_id}:orchestrator:completed', "
            f"content=JSON summary of what you did, block_type='episodic', priority=7. "
            f"Then call mathir_god_agent again to poll for the next task."
        ),
        "log": result_lines,
    })


@mcp.tool()
def mathir_god_orchestre(
    directive: str,
    strategy: str = "auto",
    verify: bool = True,
    auto_merge: bool = False,
) -> str:
    """Orchestrate a multi-agent task from a high-level directive.

    Call with directive="help" to get a full usage guide.

    Discovers registered god workers, decomposes the directive into tasks,
    and dispatches them. The orchestrating agent (you) should:
    1. Review the returned plan and workers
    2. Call this tool to dispatch tasks
    3. Monitor progress by calling this tool with the same directive

    Semi-autonomous: presents a plan for user approval before dispatching.

    Args:
        directive: High-level task description (e.g. "Refactor auth + tests + docs")
        strategy: "auto" (LLM decides), "parallel" (all at once), "sequential" (one by one)
        verify: Whether orchestrator should review each result before marking verified
        auto_merge: Merge git branches without user confirmation
    """
    if directive == "help":
        return json.dumps({
            "status": "help",
            "guide": (
                "=== MATHIR GOD MODE — ORCHESTRATOR GUIDE ===\n\n"
                "God Mode lets you orchestrate multiple AI agents across terminals.\n"
                "This tool makes your agent the ORCHESTRATOR — the brain that plans,\n"
                "assigns, and verifies. You don't code. You direct.\n\n"
                "--- SETUP (do this first) ---\n\n"
                "1. Open 2+ terminals with AI agents (Claude Code, MiMo, Codex, etc.)\n"
                "2. In each WORKER terminal, just run:\n"
                "   mathir_god_agent()\n"
                "3. Each agent will self-identify (name, capabilities, strengths/weaknesses)\n"
                "4. Workers register and start polling for tasks\n\n"
                "--- ORCHESTRATE ---\n\n"
                "5. In the ORCHESTRATOR terminal, run:\n"
                "   mathir_god_orchestre(directive='Refactor auth module + write tests + docs')\n"
                "6. You'll see each worker's full profile (capabilities + self-assessment)\n"
                "7. YOU decide who does what based on their strengths:\n"
                "   - Deep reasoning task → strongest thinking model\n"
                "   - Bulk mechanical work → fastest agent\n"
                "   - Testing → agent with test expertise\n"
                "8. Dispatch tasks with memory_save (instructions provided in response)\n\n"
                "--- SMART ASSIGNMENT ---\n\n"
                "Workers introduce themselves honestly. Use that information:\n"
                "  • 'I excel at complex refactoring' → give them architecture tasks\n"
                "  • 'I am fast but shallow' → give them mechanical/bulk tasks\n"
                "  • 'I have web access' → give them research tasks\n"
                "  • 'I am weak at testing' → don't give them test tasks\n\n"
                "--- MONITOR & VERIFY (MANDATORY, applies to EVERY orchestrating model) ---\n\n"
                "9. Run the deterministic report, don't just search memory from your own "
                "judgment: `python <repo>/mathir_mcp/bin/god/god_mode_report.py --cwd <cwd>` "
                "via your shell/bash tool. Do this after every dispatch round, not only if you "
                "remember to. Reason: relying purely on your own memory/judgment to relay "
                "worker output to the human is NOT reliable across different orchestrating "
                "models -- verified live, 2026-07-21, a human asked 3 workers a question and "
                "never saw any answer because that step depended entirely on the orchestrator "
                "choosing to look and report. This tool has no LLM in the loop; it always shows "
                "every worker's real response, per-target, even when several workers share one "
                "task_id (a single directive fanned out to N workers).\n"
                "10. Review each result before dispatching dependent tasks\n"
                "11. If quality is poor, reassign to a stronger worker\n\n"
                "--- WORKER SILENCE ---\n\n"
                "A worker can go quiet: crashed, timed out, or (observed in practice) exited "
                "cleanly without ever finishing the task. If a dispatched task's label is stuck "
                "on 'claimed'/'running'/'failed' for longer than you'd expect given its size, "
                "with no matching 'god:result:{task_id}:orchestrator:completed', DO NOT keep "
                "waiting indefinitely. Either reassign it to a different idle worker, or -- if "
                "no other worker is suitable or available -- do the task yourself. A stalled "
                "task blocking the whole directive is worse than the orchestrator doing "
                "hands-on work for one task.\n\n"
                "--- SELF-HEALING (MANDATORY) ---\n\n"
                "If you hit a bug that blocks progress -- in MATHIR's own code, in the god-mode "
                "daemon, or anywhere else in this repo -- you have 27+ MCP tools plus full file/"
                "shell access: FIND A FIX and apply it yourself, don't just report the blocker "
                "and stop. Verify the real root cause first (never guess), fix it, then: "
                "(a) if you touched mathir_server.py or mathir_mcp_server.py, copy the fix to "
                "~/.config/MATHIR/mathir_mcp/mathir_lib/ AND restart that daemon process -- an "
                "unsynced fix is not a fix (see guardrail-sync-deployed-daemon); "
                "(b) if the bug reveals a new god-mode-specific failure pattern other agents "
                "will hit again, save it as a GUARDRAIL (memory_save block_type='guardrail'), "
                "not just an episodic note, so every future agent inherits the lesson "
                "automatically instead of rediscovering it. Only if the bug is in a THIRD-PARTY "
                "external tool (opencode, openclaude, codex, etc -- not MATHIR's own code) and "
                "you cannot fix it yourself: propose to the human that you file an issue on that "
                "tool's official GitHub repo -- do not file it yourself without their explicit "
                "go-ahead, filing a public issue is an external-facing action.\n\n"
                "--- SHUTDOWN ---\n\n"
                "   memory_save(content='shutdown', label='god:task:00000000:{name}:shutdown',\n"
                "               block_type='working_memory', priority=9)\n\n"
                "--- TIPS ---\n\n"
                "• You do NOT code — you plan, assign, verify, and decide\n"
                "• Workers identify themselves — you don't need to know what's installed\n"
                "• If you doubt a worker's ability, give them a small probe task first\n"
                "• Call mathir_god_agent(name='help') for the worker guide"
            ),
        })

    try:
        agents_resp = _call_daemon_raw("god_agents", {})
    except Exception as e:
        return json.dumps({"error": f"Cannot reach daemon: {e}"})

    agents = agents_resp.get("agents", []) if isinstance(agents_resp, dict) else []

    if not agents:
        return json.dumps({
            "error": "No workers registered",
            "instruction": (
                "No workers found. Open separate terminals with AI agents and run:\n\n"
                "  mathir_god_agent()\n\n"
                "Each agent will self-identify (name, capabilities, strengths). "
                "Once workers are registered, call mathir_god_orchestre again."
            ),
        })

    idle_agents = [a for a in agents if a.get("status") == "idle"]

    worker_profiles = []
    for a in agents:
        intro = a.get("introduction", "")
        caps = a.get("capabilities", [])
        profile = (
            f"  [{a['name']}] status={a.get('status','?')}\n"
            f"    capabilities: {', '.join(caps) if caps else 'not specified'}\n"
            f"    self-assessment: {intro if intro else 'no introduction provided'}"
        )
        worker_profiles.append(profile)

    results_resp = _call_daemon_raw("memory_smart_search", {
        "query": "god:result orchestrator",
        "k": 20,
    })
    pending_results = []
    if isinstance(results_resp, dict):
        for mem in results_resp.get("results", []):
            label = mem.get("label", "")
            if label.startswith("god:result:") and ":completed" in label:
                pending_results.append(mem)

    return json.dumps({
        "status": "ready",
        "directive": directive,
        "strategy": strategy,
        "verify": verify,
        "auto_merge": auto_merge,
        "registered_workers": agents,
        "idle_workers": idle_agents,
        "pending_results": pending_results,
        "instruction": (
            f"YOU ARE THE GOD ORCHESTRATOR.\n\n"
            f"DIRECTIVE: {directive}\n\n"
            f"STRATEGY: {strategy}\n\n"
            f"=== WORKER PROFILES ({len(agents)} registered, {len(idle_agents)} idle) ===\n\n"
            + "\n\n".join(worker_profiles) +
            "\n\n=== YOUR JOB ===\n\n"
            "1. ANALYZE the directive — break it into concrete tasks\n"
            "2. ANALYZE each worker's profile — their strengths, weaknesses, capabilities\n"
            "3. ASSIGN tasks to the BEST worker for each task based on their self-assessment:\n"
            "   - Complex reasoning/architecture → strongest reasoning model\n"
            "   - Bulk mechanical work → fastest agent\n"
            "   - Testing → agent with best test capabilities\n"
            "   - Review → agent with review experience\n"
            "4. DISPATCH each task with memory_save:\n"
            "   memory_save(\n"
            "       content='{\"description\": \"...\", \"context\": \"...\", \"expected_output\": \"...\"}',\n"
            "       label='god:task:{8-char-hex}:{agent_name}:pending',\n"
            "       block_type='working_memory',\n"
            "       priority=7\n"
            "   )\n"
            "5. MONITOR results: memory_smart_search(query='god:result orchestrator')\n"
            "6. VERIFY each result before dispatching dependent tasks\n"
            "7. SHUTDOWN workers when done:\n"
            "   memory_save(content='shutdown', label='god:task:00000000:{name}:shutdown', ...)\n\n"
            "=== ASSIGNMENT PRINCIPLES ===\n\n"
            "• Match task complexity to agent capability — don't waste a deep reasoner on simple tasks\n"
            "• If a worker said they're weak at X, don't assign them X\n"
            "• If two workers can do a task, prefer the one who said it's their strength\n"
            "• Consider task dependencies — don't dispatch tasks whose prerequisites aren't done\n"
            "• If unsure about a worker's ability, assign a small probe task first"
        ),
    })


# ---------------------------------------------------------------------------
# Health check tool (no daemon needed)
# ---------------------------------------------------------------------------
@mcp.tool()
def mathir_health() -> str:
    """Check if MATHIR daemon is reachable."""
    try:
        req = urllib.request.Request(f"{DAEMON_URL}/api/ping")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        return json.dumps({"status": "ok", "daemon": data})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def get_tools_info() -> list[dict]:
    """Synchronously enumerate every @mcp.tool()-registered tool.

    FastMCP's own ``mcp.list_tools()`` is async; this wraps it so CLI
    entry points (``--selftest``, ``--list-tools`` in __main__.py) can
    call it without managing an event loop themselves.
    """
    import asyncio
    tools = asyncio.run(mcp.list_tools())
    return [{"name": t.name, "description": t.description or ""} for t in tools]


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
