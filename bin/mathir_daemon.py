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

HOST = '127.0.0.1'
PORT = 7338

# Pre-load everything
log.info("Starting MATHIR daemon...")
sys.path.insert(0, os.path.expanduser('~/.config/opencode/bin'))

from mathir_mcp_server import (
    get_embedder, handle_memory_recall, handle_memory_save,
    handle_memory_stats, handle_memory_smart_search
)
from mathir_push import ContextAnalyzer, PushCache, context_hash, deduplicate_memories

# Global push cache (shared across threads)
_push_cache = PushCache(ttl_seconds=300, max_size=200)
_push_analyzer = ContextAnalyzer()

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
        # Set socket timeout to prevent slow-loris DoS
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

            # Validate input lengths
            validation_error = _validate_input(params)
            if validation_error:
                result = {'error': validation_error}
                conn.sendall(json.dumps(result).encode('utf-8'))
                continue

            start = time.perf_counter()
            try:
                if method == 'memory_recall':
                    from mathir_vec import VecMemory
                    from mathir_mcp_server import get_project_db_path
                    dim = get_embedder_dim()
                    vec_mem = VecMemory(get_project_db_path(), dim)

                    embedder = get_embedder()
                    query = params.get('query', '')
                    k = min(params.get('k', 5), 1000)
                    query_emb = embedder.encode(query)
                    query_np = _embedding_to_numpy(query_emb)
                    results = vec_mem.search(
                        query_embedding=query_np,
                        k=k,
                        agent_filter=params.get('agent'),
                        block_type_filter=params.get('block_type')
                    )
                    result = {'results': results, 'query': query, 'total': len(results)}
                elif method == 'memory_save':
                    from mathir_mcp_server import get_project_db_path, get_project_name
                    from mathir_vec import VecMemory
                    dim = get_embedder_dim()
                    vec_mem = VecMemory(get_project_db_path(), dim)

                    content = params['content']
                    embedder = get_embedder()
                    emb = embedder.encode(content)
                    emb_np = _embedding_to_numpy(emb)

                    import uuid
                    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
                    metadata = {
                        'agent': params.get('agent', 'unknown'),
                        'block_type': params.get('block_type', 'episodic'),
                        'label': params.get('label', ''),
                        'priority': params.get('priority', 5),
                        'content': content,
                        'project': get_project_name()
                    }
                    vec_mem.store(memory_id, emb_np, metadata)
                    result = {'memory_id': memory_id, 'saved': True, 'metadata': metadata}
                elif method == 'memory_stats':
                    from mathir_mcp_server import get_project_db_path
                    from mathir_vec import VecMemory
                    dim = get_embedder_dim()
                    vec_mem = VecMemory(get_project_db_path(), dim)
                    result = vec_mem.stats()
                elif method == 'memory_push':
                    from mathir_vec import VecMemory
                    from mathir_mcp_server import get_project_db_path

                    context_text = params.get('context', '')
                    k = min(params.get('k', 10), 1000)
                    agent = params.get('agent')

                    if not context_text:
                        result = {'memories': [], 'queries_used': [], 'total': 0, 'error': 'empty context'}
                    else:
                        c_hash = context_hash(context_text)
                        with _push_lock:
                            cached = _push_cache.get(c_hash)

                        if cached is not None:
                            log.info(f"Push cache hit for hash {c_hash}")
                            result = {
                                'memories': cached,
                                'queries_used': ['(cached)'],
                                'total': len(cached),
                                'cached': True,
                            }
                        else:
                            queries = _push_analyzer.extract_queries(context_text, max_queries=5)
                            log.info(f"Push: extracted {len(queries)} queries: {queries}")

                            dim = get_embedder_dim()
                            vec_mem = VecMemory(get_project_db_path(), dim)
                            embedder = get_embedder()

                            all_memories = []
                            for q in queries:
                                q_emb = embedder.encode(q)
                                q_np = _embedding_to_numpy(q_emb)
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

                            result = {
                                'memories': top_memories,
                                'queries_used': queries,
                                'total': len(top_memories),
                                'cached': False,
                            }

                elif method == 'memory_smart_search':
                    from mathir_vec import VecMemory
                    from mathir_mcp_server import get_project_db_path, get_project_name
                    dim = get_embedder_dim()
                    vec_mem = VecMemory(get_project_db_path(), dim)
                    embedder = get_embedder()

                    query = params.get('query', '')
                    k = min(params.get('k', 10), 1000)
                    agent_filter = params.get('agent')

                    q_emb = embedder.encode(query)
                    q_np = _embedding_to_numpy(q_emb)

                    results = vec_mem.search(
                        query_embedding=q_np,
                        k=k,
                        agent_filter=agent_filter,
                    )
                    result = {'results': results, 'query': query, 'total': len(results), 'project': get_project_name()}

                elif method == 'memory_delete':
                    from mathir_vec import VecMemory
                    from mathir_mcp_server import get_project_db_path, get_project_name
                    dim = get_embedder_dim()
                    vec_mem = VecMemory(get_project_db_path(), dim)

                    memory_id = params.get('memory_id')
                    if not memory_id:
                        result = {'error': 'memory_id required'}
                    else:
                        deleted = vec_mem.delete(memory_id)
                        result = {'memory_id': memory_id, 'deleted': deleted, 'project': get_project_name()}

                elif method == 'push_cache_stats':
                    with _push_lock:
                        result = _push_cache.stats()

                elif method == 'ping':
                    result = {'pong': True, 'uptime': time.time(), 'dim': get_embedder_dim()}
                else:
                    result = {'error': f'unknown method: {method}'}
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
    if hasattr(embedder, 'get_sentence_embedding_dimension'):
        return embedder.get_sentence_embedding_dimension()
    return int(os.environ.get('MATHIR_EMBEDDING_DIM', '1024'))


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
