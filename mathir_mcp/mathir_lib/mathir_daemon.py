#!/usr/bin/env python3
"""
MATHIR Daemon — Persistent background process.
Keeps the embedding model loaded in memory.
Agents connect via a simple JSON-RPC over TCP socket.
"""

import sys
import os
import json
import time
import socket
import threading
import logging
from typing import Optional

# Add bin to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Risk mitigation (PersistBench findings: 53% leakage, >90% sycophancy)
try:
    from memory_risks import DomainClassifier, LeakageDetector, SycophancyDetector
    _risk_enabled = True
except ImportError:
    _risk_enabled = False

# Security limits
MAX_CONNECTIONS = 50        # Max concurrent client connections (prevents thread exhaustion DoS)
CLIENT_TIMEOUT = 30         # Seconds before idle client is disconnected
MAX_REQUEST_SIZE = 65536    # Max bytes per JSON request
MAX_CONTEXT_LENGTH = 50000  # Max chars for context text (prevents embedding CPU DoS)
MAX_CONTENT_LENGTH = 100000 # Max chars for memory content
MAX_QUERY_LENGTH = 5000     # Max chars for search queries
MAX_LABEL_LENGTH = 500      # Max chars for memory labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MATHIR-DAEMON] %(levelname)s %(message)s",
    stream=sys.stderr
)
log = logging.getLogger("mathir-daemon")

HOST = os.environ.get("MATHIR_HOST", "127.0.0.1")
PORT = int(os.environ.get("MATHIR_PORT", "7338"))

# Pre-load everything
log.info("Starting MATHIR daemon...")

from mathir_mcp_server import (
    get_embedder, get_project_db_path, get_project_name,
)
from mathir_push import ContextAnalyzer, PushCache, context_hash, deduplicate_memories

# Global push cache (shared across threads)
_push_cache = PushCache(ttl_seconds=300, max_size=200)
_push_analyzer = ContextAnalyzer()

# VecMemory cache — ONE instance per (db_path, dim), reused across all requests.
# Without this, every handler opens a new SQLite connection → Windows file lock deadlock.
_vec_cache = {}  # key: (str(db_path), dim) → VecMemory instance
_vec_cache_lock = threading.Lock()

def _get_vec_mem(db_path, dim):
    """Get or create a cached VecMemory. Reuses the same SQLite connection."""
    key = (str(db_path), dim)
    with _vec_cache_lock:
        if key not in _vec_cache:
            from mathir_vec import VecMemory
            _vec_cache[key] = VecMemory(db_path, dim)
            log.info(f"VecMemory cached for {db_path.name} (dim={dim})")
        return _vec_cache[key]

# Thread safety: lock for shared mutable state
_push_lock = threading.Lock()

# Connection limiter
_connection_count = 0
_connection_lock = threading.Lock()


def _embedding_to_numpy(emb) -> "numpy.ndarray":
    """Convert an embedding to a flat float32 numpy array.

    Handles both torch tensors (sentence-transformers) and plain numpy arrays.
    This is the single source of truth for embedding conversion — all callers
    in this daemon must use this function.

    Args:
        emb: Embedding output from the model — either a torch.Tensor or numpy.ndarray.

    Returns:
        1-D float32 numpy array suitable for vector store operations.
    """
    import numpy as np

    if hasattr(emb, 'cpu'):
        return emb.cpu().numpy().astype('float32').reshape(-1)
    return np.array(emb, dtype=np.float32).reshape(-1)


def _sanitize_error(exc: Exception, method: str) -> str:
    """Return a safe error message that doesn't leak internals.

    Only safe exception types (ValueError, KeyError, TypeError) pass through
    with their original message. All others get a generic internal error string.

    Args:
        exc: The exception that was raised.
        method: The RPC method name where the error occurred.

    Returns:
        Sanitized error message string safe for client consumption.
    """
    safe_types = (ValueError, KeyError, TypeError)
    if isinstance(exc, safe_types):
        msg = str(exc)[:200]
        return f"{type(exc).__name__}: {msg}"
    log.error(f"Error handling {method}: {exc}", exc_info=True)
    return f"Internal error in {method}"


def _validate_input(params: dict) -> Optional[str]:
    """Validate input parameters against size limits.

    Checks context, content, query, and label lengths against configured
    maximums to prevent CPU/memory DoS attacks.

    Args:
        params: The RPC method parameters dict.

    Returns:
        Error message string if validation fails, None if OK.
    """
    ctx = params.get('context', '')
    if isinstance(ctx, str) and len(ctx) > MAX_CONTEXT_LENGTH:
        return f"context exceeds max length ({MAX_CONTEXT_LENGTH} chars)"
    content = params.get('content', '')
    if isinstance(content, str) and len(content) > MAX_CONTENT_LENGTH:
        return f"content exceeds max length ({MAX_CONTENT_LENGTH} chars)"
    query = params.get('query', '')
    if isinstance(query, str) and len(query) > MAX_QUERY_LENGTH:
        return f"query exceeds max length ({MAX_QUERY_LENGTH} chars)"
    label = params.get('label', '')
    if isinstance(label, str) and len(label) > MAX_LABEL_LENGTH:
        return f"label exceeds max length ({MAX_LABEL_LENGTH} chars)"
    k = params.get('k', 5)
    if not isinstance(k, int) or k < 0 or k > 1000:
        return "k must be an integer between 0 and 1000"
    return None


# Pre-warm the embedder
log.info("Pre-loading embedder...")
get_embedder()
log.info("Embedder ready")


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------

def _resolve_db():
    """Resolve VecMemory + embedder. Returns (vec_mem, db_path, embedder) or raises."""
    dim = get_embedder_dim()
    db_path = get_project_db_path()
    if db_path is None:
        raise ValueError("No project database found. Set MATHIR_PROJECT env var or run from a project directory.")
    vec_mem = _get_vec_mem(db_path, dim)
    embedder = get_embedder()
    return vec_mem, db_path, embedder


def _encode_query(embedder, query: str):
    """Encode a query string to a flat numpy vector."""
    return _embedding_to_numpy(embedder.encode(query))


# ---------------------------------------------------------------------------
# RPC method handlers  (each returns a result dict)
# ---------------------------------------------------------------------------

def _handle_ping(_params):
    return {'pong': True, 'uptime': time.time(), 'dim': get_embedder_dim()}


def _handle_memory_save(params):
    vec_mem, _db_path, embedder = _resolve_db()
    content = params['content']

    # Risk mitigation: check before storing (best-effort, never block save)
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
                risk_warnings.append(f"leakage_risk={leak_risk.leakage_risk:.1f}: {'; '.join(leak_risk.reasons)}")
            if syco_risk.sycophancy_risk > 0.5:
                risk_warnings.append(f"sycophancy_risk={syco_risk.sycophancy_risk:.1f}: {'; '.join(syco_risk.reasons)}")
        except Exception:
            pass

    emb_np = _encode_query(embedder, content)

    import uuid
    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    metadata = {
        'agent': params.get('agent', 'unknown'),
        'block_type': params.get('block_type', 'episodic'),
        'label': params.get('label', ''),
        'priority': params.get('priority', 5),
        'content': content,
        'project': get_project_name(),
        'risk_warnings': risk_warnings if risk_warnings else None,
    }
    vec_mem.store(memory_id, emb_np, metadata)
    return {'memory_id': memory_id, 'saved': True, 'metadata': metadata, 'risk_warnings': risk_warnings}


def _handle_memory_recall(params):
    vec_mem, _db_path, embedder = _resolve_db()
    query = params.get('query', '')
    k = min(params.get('k', 5), 1000)
    q_np = _encode_query(embedder, query)
    results = vec_mem.search(
        query_embedding=q_np, k=k,
        agent_filter=params.get('agent'),
        block_type_filter=params.get('block_type'),
    )
    return {'results': results, 'query': query, 'total': len(results)}


def _handle_memory_stats(_params):
    vec_mem, _db_path, _embedder = _resolve_db()
    return vec_mem.stats()


def _handle_memory_delete(params):
    vec_mem, _db_path, _embedder = _resolve_db()
    memory_id = params.get('memory_id')
    if not memory_id:
        return {'error': 'memory_id required'}
    deleted = vec_mem.delete(memory_id)
    return {'memory_id': memory_id, 'deleted': deleted, 'project': get_project_name()}


def _handle_memory_smart_search(params):
    vec_mem, _db_path, embedder = _resolve_db()
    query = params.get('query', '')
    k = min(params.get('k', 10), 1000)
    q_np = _encode_query(embedder, query)
    results = vec_mem.search(
        query_embedding=q_np, k=k,
        agent_filter=params.get('agent'),
    )
    return {'results': results, 'query': query, 'total': len(results), 'project': get_project_name()}


def _handle_memory_push(params):
    context_text = params.get('context', '')
    k = min(params.get('k', 10), 1000)
    agent = params.get('agent')

    if not context_text:
        return {'memories': [], 'queries_used': [], 'total': 0, 'error': 'empty context'}

    c_hash = context_hash(context_text)
    with _push_lock:
        cached = _push_cache.get(c_hash)

    if cached is not None:
        log.info(f"Push cache hit for hash {c_hash}")
        return {'memories': cached, 'queries_used': ['(cached)'], 'total': len(cached), 'cached': True}

    queries = _push_analyzer.extract_queries(context_text, max_queries=5)
    log.info(f"Push: extracted {len(queries)} queries: {queries}")

    vec_mem, _db_path, embedder = _resolve_db()

    all_memories = []
    for q in queries:
        q_np = _encode_query(embedder, q)
        results = vec_mem.search(
            query_embedding=q_np,
            k=max(3, k // max(len(queries), 1) + 1),
            agent_filter=agent,
        )
        all_memories.extend(results)

    deduped = deduplicate_memories(all_memories)
    deduped.sort(key=lambda m: m.get('score', 0), reverse=True)
    top_memories = deduped[:k]

    with _push_lock:
        _push_cache.set(c_hash, top_memories)

    return {'memories': top_memories, 'queries_used': queries, 'total': len(top_memories), 'cached': False}


def _handle_memory_hybrid_search(params):
    _vec_mem, db_path, embedder = _resolve_db()

    query_text = params.get('query', '')
    k = min(params.get('k', 5), 100)
    vector_weight = params.get('vector_weight', 1.0)
    bm25_weight = params.get('bm25_weight', 1.0)
    agent_filter = params.get('agent')

    q_np = _encode_query(embedder, query_text)

    # Direct SQLite connection (thread-safe with check_same_thread=False)
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

    # Detect schema
    columns = {col[1] for col in dconn.execute("PRAGMA table_info(memories)").fetchall()}
    text_col = 'content' if 'content' in columns else 'modality_text'

    # --- Vector search via sqlite-vec ---
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

    # --- BM25 lexical search ---
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

    # --- RRF fusion ---
    from mathir_search import rrf_fusion
    fused = rrf_fusion(vector_results, bm25_results, vector_weight=vector_weight, bm25_weight=bm25_weight)

    # --- Build final results ---
    results = []
    for mid, rrf_score in fused[:k]:
        meta = dconn.execute(
            f"SELECT {text_col}, tier, timestamp FROM memories WHERE memory_id = ?", [mid]
        ).fetchone()
        if not meta:
            continue
        agent_val = ''
        if 'agent' in columns:
            agent_val = dconn.execute(
                "SELECT agent FROM memories WHERE memory_id = ?", [mid]
            ).fetchone()[0] or ''
        else:
            try:
                meta_row = dconn.execute(
                    "SELECT metadata FROM memories WHERE memory_id = ?", [mid]
                ).fetchone()
                if meta_row and meta_row[0]:
                    agent_val = json.loads(meta_row[0]).get('agent', '')
            except Exception:
                pass
        results.append({
            'memory_id': mid,
            'rrf_score': rrf_score,
            'content': meta[0] or '',
            'agent': agent_val,
            'score': rrf_score,
            'created_at': meta[2] or '',
            'tier': meta[1] or 'episodic',
        })
    dconn.close()
    return {
        'results': results, 'query': query_text, 'total': len(results),
        'project': get_project_name(), 'mode': 'hybrid',
        'vector_hits': len(vector_results), 'bm25_hits': len(bm25_results),
    }


def _handle_memory_risk_check(params):
    if not _risk_enabled:
        return {'error': 'risk mitigation not available (memory_risks module not found)'}
    content = params.get('content', '')
    if not content:
        return {'error': 'content required'}
    try:
        classifier = DomainClassifier()
        leakage = LeakageDetector()
        sycophancy = SycophancyDetector()
        domain = classifier.classify(content)
        leak_risk = leakage.check_leakage(domain, domain, content)
        syco_risk = sycophancy.check_sycophancy(content)
        return {
            'domain': domain.value,
            'leakage_risk': leak_risk.leakage_risk,
            'sycophancy_risk': syco_risk.sycophancy_risk,
            'sensitivity': leak_risk.sensitivity,
            'reasons': leak_risk.reasons + syco_risk.reasons,
            'safe_to_store': leak_risk.leakage_risk < 0.7 and syco_risk.sycophancy_risk < 0.7,
        }
    except Exception as e:
        return {'error': f'risk check failed: {e}'}


def _handle_push_cache_stats(_params):
    with _push_lock:
        return _push_cache.stats()


# Dispatch table  (method name → handler function)
_METHOD_HANDLERS = {
    'ping':                  _handle_ping,
    'memory_save':           _handle_memory_save,
    'memory_recall':         _handle_memory_recall,
    'memory_stats':          _handle_memory_stats,
    'memory_delete':         _handle_memory_delete,
    'memory_smart_search':   _handle_memory_smart_search,
    'memory_push':           _handle_memory_push,
    'memory_hybrid_search':  _handle_memory_hybrid_search,
    'memory_risk_check':     _handle_memory_risk_check,
    'push_cache_stats':      _handle_push_cache_stats,
}


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------

def handle_client(conn, addr):
    """Handle a client connection with security hardening."""
    global _connection_count

    # Increment connection counter
    with _connection_lock:
        _connection_count += 1
        if _connection_count > MAX_CONNECTIONS:
            _connection_count -= 1
            log.warning(f"Connection limit reached ({MAX_CONNECTIONS}), rejecting {addr}")
            try:
                conn.sendall(json.dumps({"error": "server busy, try again"}).encode())
            except Exception:
                pass
            conn.close()
            return

    log.info(f"Client connected: {addr} (active: {_connection_count})")
    try:
        conn.settimeout(CLIENT_TIMEOUT)

        while True:
            try:
                data = conn.recv(MAX_REQUEST_SIZE)
            except socket.timeout:
                log.info(f"Client timed out after {CLIENT_TIMEOUT}s: {addr}")
                break
            except (ConnectionResetError, OSError):
                break

            if not data:
                break

            try:
                request = json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                conn.sendall(json.dumps({"error": "invalid json"}).encode())
                continue

            method = request.get('method')
            params = request.get('params', {})

            validation_error = _validate_input(params)
            if validation_error:
                conn.sendall(json.dumps({'error': validation_error}).encode('utf-8'))
                continue

            start = time.perf_counter()
            try:
                handler = _METHOD_HANDLERS.get(method)
                if handler is None:
                    result = {'error': f'unknown method: {method}'}
                else:
                    result = handler(params)
            except Exception as e:
                result = {'error': _sanitize_error(e, method)}

            elapsed = (time.perf_counter() - start) * 1000
            log.info(f"{method} in {elapsed:.1f}ms")
            result['elapsed_ms'] = elapsed

            try:
                conn.sendall(json.dumps(result).encode('utf-8'))
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
    except Exception as e:
        log.error(f"Client error: {e}")
    finally:
        with _connection_lock:
            _connection_count -= 1
        conn.close()
        log.info(f"Client disconnected: {addr} (active: {_connection_count})")


def get_embedder_dim():
    """Get actual embedding dim from the loaded embedder."""
    embedder = get_embedder()
    if hasattr(embedder, 'dim'):
        return embedder.dim
    if hasattr(embedder, 'get_embedding_dimension'):
        return embedder.get_embedding_dimension()
    if hasattr(embedder, 'get_sentence_embedding_dimension'):
        return embedder.get_sentence_embedding_dimension()
    return int(os.environ.get('MATHIR_EMBEDDING_DIM', '384'))


def main():
    """Run daemon server."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)

    dim = get_embedder_dim()
    log.info(f"Using embedding dim: {dim}")
    log.info(f"MATHIR daemon listening on {HOST}:{PORT}")
    print(f"DAEMON_READY:{PORT}:{dim}", flush=True)

    try:
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        server.close()


if __name__ == '__main__':
    main()
