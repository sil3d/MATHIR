#!/usr/bin/env python3
"""
MATHIR Batch Recall Test — Benchmark sequential vs parallel vs batch.

Tests whether batching multiple recall queries into a single daemon call
(or parallelizing them) provides meaningful latency improvements.

Usage:
    python ~/.config/opencode/bin/test_batch_recall.py
    python ~/.config/opencode/bin/test_batch_recall.py --queries 5 --repeat 3
"""

import sys
import os
import json
import socket
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# ── Config ──────────────────────────────────────────────────────────────────
HOST = '127.0.0.1'
PORT = 7338
TIMEOUT = 30

# ── Sample queries (diverse to simulate real agent usage) ───────────────────
DEFAULT_QUERIES = [
    "project context and architecture",
    "auth token refresh bug fix",
    "React component patterns",
    "database migration strategy",
    "API endpoint design",
    "deployment configuration",
    "test coverage gaps",
    "performance optimization",
    "security audit findings",
    "documentation status",
]


# ── Socket helpers ──────────────────────────────────────────────────────────
def send_request(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Open a new socket, send one request, return the response."""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(TIMEOUT)
    try:
        client.connect((HOST, PORT))
        request = json.dumps({'method': method, 'params': params})
        client.sendall(request.encode('utf-8'))
        data = client.recv(65536)
        return json.loads(data.decode('utf-8'))
    finally:
        client.close()


def send_batch_request(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Send a batch recall request in a single socket connection.
    This is what the daemon WOULD need to support for true batching.
    
    Since we can't modify the daemon, this is a simulation:
    it opens one connection and sends queries sequentially inside it.
    The real benefit comes from avoiding connection overhead.
    """
    # Simulate batch: single connection, single JSON-RPC call
    # The daemon would need a 'memory_batch_recall' method for this to work.
    # For now, we use parallel threads to simulate the benefit.
    raise NotImplementedError("Daemon does not support batch recall yet")


# ── Benchmark: Sequential ──────────────────────────────────────────────────
def benchmark_sequential(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute all queries one by one, each in a new socket."""
    results = []
    timings = []

    for q in queries:
        params = {
            'query': q['query'],
            'k': q.get('k', 5),
        }
        if q.get('agent'):
            params['agent'] = q['agent']

        start = time.perf_counter()
        result = send_request('memory_recall', params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        timings.append(elapsed_ms)
        results.append(result)

    total_ms = sum(timings)
    return {
        'results': results,
        'timings': timings,
        'total_ms': total_ms,
        'avg_ms': total_ms / len(queries) if queries else 0,
    }


# ── Benchmark: Parallel (threaded) ─────────────────────────────────────────
def benchmark_parallel(queries: List[Dict[str, Any]], max_workers: int = 5) -> Dict[str, Any]:
    """Execute all queries in parallel using threads, each in its own socket."""
    results = [None] * len(queries)
    timings = [0.0] * len(queries)

    def _exec_one(idx: int, q: Dict[str, Any]):
        params = {
            'query': q['query'],
            'k': q.get('k', 5),
        }
        if q.get('agent'):
            params['agent'] = q['agent']

        start = time.perf_counter()
        result = send_request('memory_recall', params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        timings[idx] = elapsed_ms
        results[idx] = result

    overall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_exec_one, i, q) for i, q in enumerate(queries)]
        for f in as_completed(futures):
            f.result()  # propagate exceptions
    overall_ms = (time.perf_counter() - overall_start) * 1000

    return {
        'results': results,
        'timings': timings,
        'total_ms': overall_ms,
        'avg_ms': overall_ms / len(queries) if queries else 0,
    }


# ── Benchmark: True batch (single socket, single encode) ────────────────────
def benchmark_true_batch(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simulate what a real batch recall would do on the daemon side:
    encode ALL queries in one embedder.encode() call (batch encoding),
    then do all searches.

    This is the idealized case — single socket, single round-trip.
    We simulate it locally to show the theoretical upper bound.
    """
    try:
        import numpy as np
    except ImportError:
        return {'error': 'numpy not installed', 'total_ms': 0, 'timings': []}

    sys.path.insert(0, os.path.expanduser('~/.config/opencode/bin'))
    from mathir_mcp_server import get_embedder, get_project_db_path
    from mathir_vec import VecMemory
    from mathir_daemon import get_embedder_dim

    embedder = get_embedder()
    dim = get_embedder_dim()
    vec_mem = VecMemory(get_project_db_path(), dim)

    start = time.perf_counter()

    # Batch encode all queries at once
    query_texts = [q['query'] for q in queries]
    batch_embs = embedder.encode(query_texts)
    if hasattr(batch_embs, 'cpu'):
        batch_np = batch_embs.cpu().numpy().astype('float32')
    else:
        batch_np = np.array(batch_embs, dtype='float32')

    encode_ms = (time.perf_counter() - start) * 1000

    # Search each embedding
    results = []
    search_start = time.perf_counter()
    for i, q in enumerate(queries):
        query_np = batch_np[i].reshape(-1)
        k = q.get('k', 5)
        agent_filter = q.get('agent')

        hits = vec_mem.search(query_embedding=query_np, k=k, agent_filter=agent_filter)
        results.append({
            'query': q['query'],
            'results': hits,
            'total': len(hits),
        })
    search_ms = (time.perf_counter() - search_start) * 1000
    total_ms = (time.perf_counter() - start) * 1000

    return {
        'results': results,
        'total_ms': total_ms,
        'encode_ms': encode_ms,
        'search_ms': search_ms,
        'timings': [search_ms / len(queries)] * len(queries),  # avg per query
    }


# ── Formatting ──────────────────────────────────────────────────────────────
def format_ms(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f}µs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


def print_results(seqs: Dict, pars: Dict, batch: Dict, queries: List):
    """Pretty-print benchmark results."""
    n = len(queries)
    # Use simple ASCII chars for Windows console compatibility
    SEP = '-' * 70
    EQ = '=' * 70

    print()
    print(EQ)
    print(f"  MATHIR Batch Recall Benchmark - {n} queries")
    print(EQ)

    # Sequential
    print(f"\n{SEP}")
    print(f"  1. SEQUENTIAL (1 socket per query)")
    print(SEP)
    print(f"  Total:   {format_ms(seqs['total_ms'])}")
    print(f"  Average: {format_ms(seqs['avg_ms'])} per query")
    for i, t in enumerate(seqs['timings']):
        q = queries[i]['query'][:40]
        print(f"    [{i+1:2d}] {format_ms(t):>8s}  {q}")

    # Parallel
    print(f"\n{SEP}")
    print(f"  2. PARALLEL (5 threads, 5 sockets)")
    print(SEP)
    print(f"  Total:   {format_ms(pars['total_ms'])}")
    print(f"  Average: {format_ms(pars['avg_ms'])} per query")
    for i, t in enumerate(pars['timings']):
        q = queries[i]['query'][:40]
        print(f"    [{i+1:2d}] {format_ms(t):>8s}  {q}")

    # True batch
    if 'error' not in batch:
        print(f"\n{SEP}")
        print(f"  3. TRUE BATCH (1 socket, batch encode)")
        print(SEP)
        print(f"  Total:   {format_ms(batch['total_ms'])}")
        print(f"  Encode:  {format_ms(batch['encode_ms'])} (all {n} queries at once)")
        print(f"  Search:  {format_ms(batch['search_ms'])} (all {n} searches)")
        print(f"  Average: {format_ms(batch['total_ms']/n)} per query")

    # Comparison
    print(f"\n{EQ}")
    print(f"  COMPARISON")
    print(EQ)

    seq_total = seqs['total_ms']
    par_total = pars['total_ms']
    par_speedup = seq_total / par_total if par_total > 0 else 0

    print(f"  Sequential:  {format_ms(seq_total):>10s}  (baseline)")
    print(f"  Parallel:    {format_ms(par_total):>10s}  ({par_speedup:.1f}x faster)")

    if 'error' not in batch:
        bat_total = batch['total_ms']
        bat_speedup = seq_total / bat_total if bat_total > 0 else 0
        print(f"  True batch:  {format_ms(bat_total):>10s}  ({bat_speedup:.1f}x faster)")

    print()

    # Verdict
    print(SEP)
    print(f"  VERDICT")
    print(SEP)

    if par_speedup >= 3.0:
        verdict = (
            f"  [YES] Parallel recall gives {par_speedup:.1f}x speedup.\n"
            f"  -> Implementing batch recall is WORTH IT.\n"
            f"  -> With 10 tool calls, agent saves {format_ms(seq_total - par_total)}."
        )
    elif par_speedup >= 1.5:
        verdict = (
            f"  [WARN] Parallel gives {par_speedup:.1f}x speedup (modest).\n"
            f"  -> Batch recall helps, but gains are modest.\n"
            f"  -> Worth implementing if you frequently do 10+ recalls."
        )
    else:
        verdict = (
            f"  [NO] Parallel gives only {par_speedup:.1f}x speedup.\n"
            f"  -> The bottleneck is embedding computation, not connection overhead.\n"
            f"  -> Batch encoding would help more than parallel sockets."
        )

    print(verdict)

    if 'error' not in batch:
        bat_total = batch['total_ms']
        bat_speedup = seq_total / bat_total if bat_total > 0 else 0
        if bat_speedup >= par_speedup * 1.2:
            print(f"\n  [TARGET] TRUE BATCH (batch encode) gives {bat_speedup:.1f}x speedup.")
            print(f"  -> This is {bat_speedup/par_speedup:.1f}x better than parallel sockets alone.")
            print(f"  -> Recommendation: implement memory_batch_recall in daemon.")

    print(f"\n{EQ}")
    print()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MATHIR Batch Recall Benchmark')
    parser.add_argument('--queries', '-n', type=int, default=10,
                        help='Number of queries to test (default: 10)')
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='Repeat each benchmark N times and average (default: 1)')
    parser.add_argument('--parallel-workers', '-w', type=int, default=5,
                        help='Max threads for parallel test (default: 5)')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()

    # Check daemon
    print("Checking daemon...")
    try:
        ping = send_request('ping', {})
        if 'error' in ping:
            print(f"ERROR: {ping['error']}", file=sys.stderr)
            sys.exit(1)
        dim = ping.get('dim', '?')
        print(f"Daemon OK (dim={dim})")
    except Exception as e:
        print(f"Daemon not running: {e}", file=sys.stderr)
        print(f"Start with: python -m mathir_mcp", file=sys.stderr)
        sys.exit(1)

    # Prepare queries
    queries = []
    for i in range(args.queries):
        idx = i % len(DEFAULT_QUERIES)
        queries.append({'query': DEFAULT_QUERIES[idx], 'k': 5})

    print(f"Running benchmark with {args.queries} queries, {args.repeat} repeat(s)...\n")

    # Run benchmarks
    seq_totals = []
    par_totals = []
    bat_totals = []

    for rep in range(args.repeat):
        if args.repeat > 1:
            print(f"  Repeat {rep+1}/{args.repeat}...")

        seq = benchmark_sequential(queries)
        par = benchmark_parallel(queries, max_workers=args.parallel_workers)

        seq_totals.append(seq['total_ms'])
        par_totals.append(par['total_ms'])

        try:
            bat = benchmark_true_batch(queries)
            if 'error' not in bat:
                bat_totals.append(bat['total_ms'])
        except Exception as e:
            bat = {'error': str(e), 'total_ms': 0, 'timings': []}

    # Average results
    seq_avg = sum(seq_totals) / len(seq_totals)
    par_avg = sum(par_totals) / len(par_totals)

    seqs = {
        'total_ms': seq_avg,
        'avg_ms': seq_avg / args.queries,
        'timings': seq['timings'],  # keep last run's per-query timings
    }
    pars = {
        'total_ms': par_avg,
        'avg_ms': par_avg / args.queries,
        'timings': par['timings'],
    }

    if bat_totals:
        bat_avg = sum(bat_totals) / len(bat_totals)
        batch = {
            'total_ms': bat_avg,
            'encode_ms': bat.get('encode_ms', 0),
            'search_ms': bat.get('search_ms', 0),
            'timings': bat.get('timings', []),
        }
    else:
        batch = {'error': 'batch not available', 'total_ms': 0, 'timings': []}

    # Output
    if args.json:
        output = {
            'queries': args.queries,
            'repeat': args.repeat,
            'sequential_ms': round(seqs['total_ms'], 1),
            'parallel_ms': round(pars['total_ms'], 1),
            'batch_ms': round(batch['total_ms'], 1),
            'speedup_parallel': round(seqs['total_ms'] / pars['total_ms'], 2) if pars['total_ms'] > 0 else 0,
            'speedup_batch': round(seqs['total_ms'] / batch['total_ms'], 2) if batch.get('total_ms', 0) > 0 else 0,
        }
        print(json.dumps(output, indent=2))
    else:
        print_results(seqs, pars, batch, queries)


if __name__ == '__main__':
    main()
