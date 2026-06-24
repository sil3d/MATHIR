#!/usr/bin/env python3
"""
MATHIR CLI — Fast command-line interface to MATHIR memory.
Pre-loads the model once, keeps it in memory.
Use this instead of `python -c "..."` for faster access.
"""

import sys
import os
import json
import argparse
import time

# Add this directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-load on first import
from mathir_mcp_server import get_embedder, handle_memory_recall, handle_memory_save, handle_memory_stats

# Pre-warm the embedder (one-time cost)
_EMBEDDER_READY = False

def ensure_embedder():
    global _EMBEDDER_READY
    if not _EMBEDDER_READY:
        get_embedder()
        _EMBEDDER_READY = True


def cmd_recall(args):
    """Recall memories by query."""
    ensure_embedder()
    start = time.perf_counter()
    result = handle_memory_recall({
        'query': args.query,
        'k': args.k,
        'agent': args.agent,
        'block_type': args.block_type
    })
    elapsed = (time.perf_counter() - start) * 1000
    
    results = result.get('results', [])
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


def cmd_save(args):
    """Save a memory."""
    ensure_embedder()
    start = time.perf_counter()
    result = handle_memory_save({
        'content': args.content,
        'agent': args.agent,
        'block_type': args.block_type,
        'label': args.label,
        'priority': args.priority
    })
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"# Saved in {elapsed:.0f}ms", file=sys.stderr)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_stats(args):
    """Get memory statistics."""
    result = handle_memory_stats({})
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_search(args):
    """Fast text search (no embedding needed)."""
    from mathir_mcp_server import handle_memory_smart_search
    result = handle_memory_smart_search({
        'query': args.query,
        'k': args.k
    })
    results = result.get('results', [])
    print(f"# Smart search: {len(results)} results")
    for r in results:
        label = r.get('label', '?')
        content = r.get('content', '')[:args.max_chars]
        print(f"[{r.get('block_type', '?')}] {label}: {content}")


def main():
    parser = argparse.ArgumentParser(description='MATHIR CLI - fast memory access')
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
    
    args = parser.parse_args()
    
    if args.cmd == 'recall':
        cmd_recall(args)
    elif args.cmd == 'save':
        cmd_save(args)
    elif args.cmd == 'stats':
        cmd_stats(args)
    elif args.cmd == 'search':
        cmd_search(args)


if __name__ == '__main__':
    main()
