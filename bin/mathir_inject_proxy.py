"""
MATHIR Auto-Inject Proxy
========================
A transparent HTTP proxy that sits between OpenCode/MiMo and the LLM API.
For every LLM call, it:
1. Takes the last user message
2. Calls MATHIR daemon for a quick recall (k=3)
3. Injects the memories into the system prompt (replacing {{MATHIR_CONTEXT}})
4. Forwards the modified request to the real LLM API
5. Streams the response back

This makes memory PROACTIVE — the LLM never has to "remember to recall".

Usage:
    # Start proxy on port 8182 (or any port)
    python mathir_inject_proxy.py --target http://localhost:8181 --port 8182
    
    # Point OpenCode/MiMo to use http://localhost:8182 instead of http://localhost:8181
    # In opencode.json: "baseUrl": "http://localhost:8182"
"""
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional

# Ensure daemon client is importable
sys.path.insert(0, str(Path(__file__).parent))
from mathir_client import call as _daemon_call
from mathir_prime import build_priming_context, format_for_injection

log = logging.getLogger("MATHIR-INJECT")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s')

INJECT_MARKER = "{{MATHIR_CONTEXT}}"
INJECT_TIMEOUT_MS = 300  # Max time to wait for recall (must be FAST)
MAX_MEMORIES = 3  # Top-K memories to inject
MAX_TOKENS_PER_MEMORY = 150  # Truncate long memories


def _fast_call(method, params, timeout_s=0.2):
    """Daemon call with SHORT timeout — proxy must never block the LLM."""
    import socket
    HOST, PORT = '127.0.0.1', 7338
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(timeout_s)
        client.connect((HOST, PORT))
        request = json.dumps({'method': method, 'params': params})
        client.sendall(request.encode('utf-8'))
        chunks = []
        while True:
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if b'}' in chunk and len(chunks) > 0:
                # Try to parse; if valid JSON, done
                try:
                    json.loads(b''.join(chunks).decode('utf-8', errors='ignore'))
                    break
                except json.JSONDecodeError:
                    continue
        client.close()
        body = b''.join(chunks).decode('utf-8', errors='ignore').strip()
        if not body:
            return None
        # Last line (daemon sends JSON per line)
        last_line = body.split('\n')[-1]
        return json.loads(last_line)
    except Exception:
        return None


def recall_for_injection(user_message: str) -> str:
    """
    Quick recall from MATHIR daemon.
    Returns formatted markdown block, or empty string on failure.
    MUST be fast (<300ms) or it blocks the LLM call.
    """
    if not user_message or len(user_message.strip()) < 5:
        return ""
    
    try:
        # Add pre-cognitive priming context to query
        priming = build_priming_context()
        priming_str = format_for_injection(priming)
        
        # Build enhanced query: user msg + project context
        query_parts = [user_message[:200].strip()]
        if priming.get('project'):
            query_parts.append(f"project:{priming['project']}")
        if priming.get('git', {}).get('branch'):
            query_parts.append(f"branch:{priming['git']['branch']}")
        query = " ".join(query_parts)
        
        t0 = time.perf_counter()
        result = _fast_call('memory_recall', {
            'query': query,
            'k': MAX_MEMORIES
        }, timeout_s=INJECT_TIMEOUT_MS / 1000)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if not result or 'error' in result:
            log.debug(f"No recall results ({elapsed_ms:.0f}ms): {result.get('error', 'empty') if result else 'no result'}")
            return ""
        
        memories = result.get('results', [])
        if not memories:
            return ""
        
        # Format as compact markdown
        lines = []
        if priming_str:
            lines.append(priming_str + "\n")
        lines.append("**Recent relevant knowledge (from MATHIR, auto-injected):**\n")
        for i, mem in enumerate(memories, 1):
            content = mem.get('content', '').strip()
            if not content:
                continue
            
            # Truncate to max tokens (rough char estimate)
            if len(content) > MAX_TOKENS_PER_MEMORY * 4:
                content = content[:MAX_TOKENS_PER_MEMORY * 4] + "..."
            
            label = mem.get('label', mem.get('memory_id', 'unknown'))
            score = mem.get('score', 0)
            block_type = mem.get('block_type', 'memory')
            
            lines.append(f"{i}. **[{block_type}/{label}]** (relevance: {score:.2f})\n   {content}\n")
        
        log.info(f"Injected {len(memories)} memories in {elapsed_ms:.0f}ms")
        return "\n".join(lines)
    
    except Exception as e:
        log.debug(f"Recall failed: {e}")
        return ""


def inject_into_system_prompt(body: dict) -> dict:
    """
    Inject MATHIR context into the system prompt of an LLM call.
    Handles both 'system' field (single string) and 'system_prompt' (legacy).
    """
    # Find the system prompt
    system = body.get('system') or body.get('system_prompt') or ''
    
    # If no marker in system prompt, append it
    if INJECT_MARKER not in system:
        system = system + "\n\n" + INJECT_MARKER
    
    # Get the last user message for context
    messages = body.get('messages', [])
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            if isinstance(content, str):
                last_user_msg = content
            elif isinstance(content, list):
                # Multimodal: find text parts
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        last_user_msg = part.get('text', '')
                        break
            break
    
    if not last_user_msg:
        return body  # No user message to inject for
    
    # Quick recall
    injection = recall_for_injection(last_user_msg)
    if not injection:
        injection = "_No relevant memories found yet. As you learn, save with `mathir_client.py save`._"
    
    # Replace marker
    system = system.replace(INJECT_MARKER, injection)
    
    # Update body
    body['system'] = system
    return body


# ─────────────────────────────────────────────────────────────
# HTTP Proxy implementation
# ─────────────────────────────────────────────────────────────

async def proxy_handler(client_request, target_url):
    """
    Forward an aiohttp request to target_url with MATHIR injection.
    """
    try:
        import aiohttp
    except ImportError:
        log.error("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)
    
    # Read body
    body_bytes = await client_request.read()
    
    # Parse and inject
    headers = dict(client_request.headers)
    injected = False
    if body_bytes:
        try:
            body = json.loads(body_bytes)
            body = inject_into_system_prompt(body)
            body_bytes = json.dumps(body).encode('utf-8')
            headers['Content-Length'] = str(len(body_bytes))
            injected = True
        except json.JSONDecodeError:
            pass  # Not JSON, pass through
    
    # Forward to target
    async with aiohttp.ClientSession() as session:
        url = target_url.rstrip('/') + str(client_request.path)
        async with session.request(
            method=client_request.method,
            url=url,
            headers=headers,
            data=body_bytes,
            params=client_request.query,
            allow_redirects=False,
        ) as resp:
            response_body = await resp.read()
            return resp.status, dict(resp.headers), response_body


def main():
    parser = argparse.ArgumentParser(description='MATHIR Auto-Inject Proxy')
    parser.add_argument('--target', default='http://localhost:8181', help='Target LLM API URL')
    parser.add_argument('--port', type=int, default=8182, help='Proxy port (default: 8182)')
    parser.add_argument('--host', default='127.0.0.1', help='Proxy host')
    args = parser.parse_args()
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        from aiohttp import web
    except ImportError:
        log.error("aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)
    
    async def handle(request):
        status, headers, body = await proxy_handler(request, args.target)
        return web.Response(status=status, headers=headers, body=body)
    
    app = web.Application()
    app.router.add_route('*', '/{path:.*}', handle)
    
    log.info(f"MATHIR Inject Proxy listening on http://{args.host}:{args.port}")
    log.info(f"Forwarding to {args.target}")
    log.info(f"Timeout: {INJECT_TIMEOUT_MS}ms | Max memories: {MAX_MEMORIES}")
    
    web.run_app(app, host=args.host, port=args.port, handle_signals=True)


if __name__ == "__main__":
    main()
