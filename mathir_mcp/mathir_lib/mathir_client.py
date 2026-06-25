#!/usr/bin/env python3
"""
MATHIR Client — HTTP client for the unified MATHIR server (Flask + Waitress).
No model loading on each call — just HTTP requests.
"""

import sys
import os
import json
import argparse
import time
import urllib.request
import urllib.error

# Fix Windows console encoding for Unicode output (→, emojis, accented chars)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

HOST = os.environ.get('MATHIR_HOST', '127.0.0.1')
PORT = int(os.environ.get('MATHIR_PORT', '7338'))
TIMEOUT = 30
# Use IP, not 'localhost' — Windows urllib resolves localhost to IPv6 first
# which times out after 2s. 127.0.0.1 is always IPv4 and instant.
BASE_URL = f'http://{HOST}:{PORT}'


# RPC method → (HTTP method, route)
_METHOD_MAP = {
    'ping':                  ('GET',  '/api/ping'),
    'memory_save':           ('POST', '/api/memory/save'),
    'memory_recall':         ('POST', '/api/memory/recall'),
    'memory_stats':          ('GET',  '/api/memory/stats'),
    'memory_delete':         ('POST', '/api/memory/delete'),
    'memory_smart_search':   ('POST', '/api/memory/smart_search'),
    'memory_push':           ('POST', '/api/memory/push'),
    'memory_hybrid_search':  ('POST', '/api/memory/hybrid_search'),
    'memory_risk_check':     ('POST', '/api/memory/risk_check'),
    'memory_promote':        ('POST', '/api/memory/promote'),
    'memory_auto_promote':   ('POST', '/api/memory/auto_promote'),
    'memory_decay':          ('POST', '/api/memory/decay'),
    'memory_consolidate':    ('POST', '/api/memory/consolidate'),
    'memory_link':           ('POST', '/api/memory/link'),
    'memory_get_links':      ('POST', '/api/memory/get_links'),
    'memory_build_links':    ('POST', '/api/memory/build_links'),
    'push_cache_stats':      ('GET',  '/api/push_cache_stats'),
}


def call(method, params=None):
    """Call server method via HTTP.

    Args:
        method: The RPC method name (e.g. 'memory_recall', 'memory_save').
        params: Dict of parameters for the method (default empty).

    Returns:
        Dict with the server's response, or an error dict if the call fails.
    """
    if params is None:
        params = {}

    mapping = _METHOD_MAP.get(method)
    if mapping is None:
        return {'error': f'unknown method: {method}'}

    http_method, route = mapping
    url = f'{BASE_URL}{route}'

    try:
        if http_method == 'GET':
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode('utf-8'))
        else:
            body = json.dumps(params).encode('utf-8')
            req = urllib.request.Request(
                url, data=body, method='POST',
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
            return json.loads(body)
        except Exception:
            return {'error': f'HTTP {e.code}: {e.reason}'}
    except (urllib.error.URLError, OSError, ConnectionError) as e:
        return {'error': f'Server not running. Start with: python mathir_server.py ({e})'}


def main():
    parser = argparse.ArgumentParser(description='MATHIR fast client (connects to daemon)')
    sub = parser.add_subparsers(dest='cmd', required=True)
    
    # recall
    p = sub.add_parser('recall', help='Recall memories by semantic query')
    p.add_argument('query', help='Search query')
    p.add_argument('-k', type=int, default=5, help='Number of results')
    p.add_argument('--agent', help='Filter by agent')
    p.add_argument('--block-type', help='Filter by memory type')
    p.add_argument('--json', action='store_true', help='Output as JSON')
    p.add_argument('--max-chars', type=int, default=150, help='Max chars per content')
    
    # save
    p = sub.add_parser('save', help='Save a memory')
    p.add_argument('content', help='Content to remember')
    p.add_argument('-a', '--agent', required=True, help='Agent name')
    p.add_argument('-t', '--block-type', default='episodic',
                   choices=['working_memory', 'episodic', 'semantic', 'procedural', 'immunological'],
                   help='Memory type (immunological = anomaly/threat-signature storage)')
    p.add_argument('-l', '--label', required=True, help='Short label')
    p.add_argument('-p', '--priority', type=int, default=5, help='Priority 0-10')
    
    # stats
    sub.add_parser('stats', help='Memory statistics')
    
    # search
    p = sub.add_parser('search', help='Fast text search (no embedding)')
    p.add_argument('query', help='Search query')
    p.add_argument('-k', type=int, default=5, help='Number of results')
    p.add_argument('--max-chars', type=int, default=150, help='Max chars per content')
    
    # push — send context, get relevant memories back
    p = sub.add_parser('push', help='Push context and receive relevant memories')
    p.add_argument('context', help='Context text (conversation, task, etc.)')
    p.add_argument('--auto', action='store_true', help='Output text ready for system prompt injection')
    p.add_argument('--json', action='store_true', help='Output as JSON')
    p.add_argument('-k', type=int, default=6, help='Number of memories to return')
    p.add_argument('--agent', help='Filter by agent')
    p.add_argument('--block-type', help='Filter by memory type')
    p.add_argument('--max-chars', type=int, default=150, help='Max chars per content')

    # hybrid — vector + BM25 + RRF fusion search
    p = sub.add_parser('hybrid', help='Hybrid search (vector + BM25 + RRF fusion)')
    p.add_argument('query', help='Search query')
    p.add_argument('-k', type=int, default=5, help='Number of results')
    p.add_argument('--vector-weight', type=float, default=0.6, help='Vector weight (0-1)')
    p.add_argument('--bm25-weight', type=float, default=0.4, help='BM25 weight (0-1)')
    p.add_argument('--agent', help='Filter by agent')
    p.add_argument('--json', action='store_true', help='Output as JSON')
    p.add_argument('--max-chars', type=int, default=200, help='Max chars per content')

    # ping
    sub.add_parser('ping', help='Check if daemon is running')
    
    args = parser.parse_args()
    
    if args.cmd == 'ping':
        result = call('ping')
        if 'error' in result:
            print(f"Daemon: NOT RUNNING ({result['error']})", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Daemon: OK (uptime: {result.get('uptime', '?')})")
            return
    
    if args.cmd == 'recall':
        params = {'query': args.query, 'k': args.k}
        if args.agent:
            params['agent'] = args.agent
        if args.block_type:
            params['block_type'] = args.block_type
        result = call('memory_recall', params)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)
        
        results = result.get('results', [])
        elapsed = result.get('elapsed_ms', 0)
        print(f"# Recall: {len(results)} results in {elapsed:.0f}ms", file=sys.stderr)
        
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            for r in results:
                label = r.get('label', '?')
                content = r.get('content', '')[:args.max_chars]
                block = r.get('block_type', '?')
                agent = r.get('agent', '?')
                print(f"[{block}/{agent}] {label}: {content}")
    
    elif args.cmd == 'save':
        params = {
            'content': args.content,
            'agent': args.agent,
            'block_type': args.block_type,
            'label': args.label,
            'priority': args.priority
        }
        result = call('memory_save', params)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)
        
        elapsed = result.get('elapsed_ms', 0)
        print(f"# Saved in {elapsed:.0f}ms", file=sys.stderr)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.cmd == 'stats':
        result = call('memory_stats')
        if 'error' in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.cmd == 'search':
        result = call('memory_smart_search', {'query': args.query, 'k': args.k})
        if 'error' in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)
        
        results = result.get('results', [])
        elapsed = result.get('elapsed_ms', 0)
        print(f"# Smart search: {len(results)} results in {elapsed:.0f}ms", file=sys.stderr)
        for r in results:
            label = r.get('label', '?')
            content = r.get('content', '')[:args.max_chars]
            print(f"[{r.get('block_type', '?')}] {label}: {content}")

    elif args.cmd == 'push':
        params = {
            'context': args.context,
            'k': args.k,
        }
        if args.agent:
            params['agent'] = args.agent
        if args.block_type:
            params['block_type'] = args.block_type

        # Try memory_push first, fallback to memory_recall
        result = call('memory_push', params)
        fallback_used = False
        if 'error' in result:
            # Daemon doesn't support memory_push — fallback to recall
            fallback_used = True
            recall_params = {'query': args.context, 'k': args.k}
            if args.agent:
                recall_params['agent'] = args.agent
            if args.block_type:
                recall_params['block_type'] = args.block_type
            result = call('memory_recall', recall_params)

        if 'error' in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)

        # memory_push returns 'memories', memory_recall returns 'results'
        results_list = result.get('memories', result.get('results', []))
        elapsed = result.get('elapsed_ms', 0)
        count = len(results_list)

        if args.json:
            output = {
                'count': count,
                'elapsed_ms': elapsed,
                'fallback_used': fallback_used,
                'results': results_list,
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        elif args.auto:
            # Mode auto: output ready-to-inject text
            if count == 0:
                pass  # No memories — print nothing
            else:
                lines = ["## MATHIR Memory (auto-loaded)"]
                for r in results_list:
                    label = r.get('label', '?')
                    content = r.get('content', '')[:args.max_chars]
                    lines.append(f"- [{label}] {content}")
                print('\n'.join(lines))
        else:
            # Mode simple: human-readable
            tag = " (fallback→recall)" if fallback_used else ""
            print(f"# Push: {count} memories in {elapsed:.0f}ms{tag}", file=sys.stderr)
            for r in results_list:
                label = r.get('label', '?')
                content = r.get('content', '')[:args.max_chars]
                block = r.get('block_type', '?')
                agent = r.get('agent', '?')
                print(f"[{block}/{agent}] {label}: {content}")

    elif args.cmd == 'hybrid':
        params = {
            'query': args.query,
            'k': args.k,
            'vector_weight': args.vector_weight,
            'bm25_weight': args.bm25_weight,
        }
        if args.agent:
            params['agent'] = args.agent
        result = call('memory_hybrid_search', params)

        if 'error' in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)

        results = result.get('results', [])
        elapsed = result.get('elapsed_ms', 0)
        print(f"# Hybrid search: {len(results)} results in {elapsed:.0f}ms (vector={args.vector_weight}, bm25={args.bm25_weight})", file=sys.stderr)

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            for r in results:
                label = r.get('label', '?')
                content = r.get('content', '')[:args.max_chars]
                block = r.get('block_type', '?')
                agent = r.get('agent', '?')
                score = r.get('score', '?')
                print(f"[{block}/{agent}] {label} (score={score}): {content}")


if __name__ == '__main__':
    main()
