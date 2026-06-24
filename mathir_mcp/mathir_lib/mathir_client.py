#!/usr/bin/env python3
"""
MATHIR Client — Fast client that connects to the persistent daemon.
No model loading on each call — just socket communication.
"""

import sys
import os
import json
import socket
import argparse
import time

HOST = '127.0.0.1'
PORT = 7338
TIMEOUT = 30
MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB max response from daemon


def call(method, params=None):
    """Call daemon method via socket.

    Args:
        method: The RPC method name (e.g. 'memory_recall', 'memory_save').
        params: Dict of parameters for the method (default empty).

    Returns:
        Dict with the daemon's response, or an error dict if the call fails.
    """
    if params is None:
        params = {}

    # SECURITY: wrap socket lifecycle in try/finally to avoid FD leaks on exception
    # (previous version leaked the socket on socket.error mid-recv).
    client = None
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(TIMEOUT)
        client.connect((HOST, PORT))

        request = json.dumps({'method': method, 'params': params})
        client.sendall(request.encode('utf-8'))

        # Read full response — TCP is a stream, may need multiple recv calls
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_RESPONSE_SIZE:
                return {'error': f'Response too large ({total} bytes > {MAX_RESPONSE_SIZE} limit)'}
            # Try to parse once we have enough data
            try:
                data = b''.join(chunks)
                result = json.loads(data.decode('utf-8'))
                return result
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Incomplete — keep reading
                continue

        # Connection closed before complete JSON
        return {'error': 'Daemon closed connection before sending complete response'}
    except socket.error as e:
        return {'error': 'Daemon not running. Start with: python mathir_daemon.py'}
    finally:
        # SECURITY: always close the socket to avoid FD leak on any error path
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass


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
