#!/usr/bin/env python3
"""
MATHIR LLM Proxy — Universal memory-injecting middleware (option D).

Sits between the agent and the real LLM API. Any agent that points its
LLM endpoint at this proxy automatically gets MATHIR memory injected into
the system prompt — universal coverage for OpenAI-compatible clients
(Claude Code via OPENAI_BASE_URL, Codex, Cursor, Cline, Continue, anything
that speaks the OpenAI Chat Completions API).

Why this exists:
  Plugin runtime injection only works for opencode + mimocode. The MCP
  prompts capability only works for hosts that auto-fetch prompts. This
  proxy is the catch-all: it works for EVERY agent that hits an
  OpenAI-compatible /v1/chat/completions endpoint, regardless of MCP
  support, plugin support, or instruction-file compliance.

Architecture:
  Agent -> [MATHIR proxy :7339] -> augment system prompt -> [real LLM API]
  The proxy reads the last user message, queries the MATHIR daemon
  /api/context, and prepends a <mathir-auto-injection> block to the
  system prompt. Non-streaming and SSE streaming both pass through.

Usage:
  # 1. Start the proxy (daemon must already be running on 7338)
  python -m mathir_mcp.mathir_lib.mathir_proxy
  python mathir_mcp/mathir_lib/mathir_proxy.py --port 7339

  # 2. Point your agent at it instead of api.openai.com
  # Claude Code:
  export OPENAI_BASE_URL=http://127.0.0.1:7339/v1
  # Cursor / Cline / Continue: set Base URL to http://127.0.0.1:7339/v1
  # Codex: set base_url in config.toml

Config (env vars):
  MATHIR_DAEMON_URL    default http://127.0.0.1:7338
  MATHIR_PROXY_PORT    default 7339
  MATHIR_PROXY_HOST    default 127.0.0.1
  MATHIR_PROXY_TARGET  default https://api.openai.com/v1   (upstream LLM)
  MATHIR_PROXY_API_KEY forwarded if set, else passthrough from Authorization header
  MATHIR_PROXY_INJECT_K default 8 (memories per request)
  MATHIR_PROXY_DEBUG   default 0 (set 1 to log every augmentation)
  MATHIR_LOG_DIR       default ~/.config/mathir/logs (or $MATHIR_HOME/logs)
"""

import sys
import os
import json
import time
import socket
import ipaddress
import logging
import logging.handlers
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

from flask import Flask, request, Response, stream_with_context

# ---------------------------------------------------------------------------
# Bootstrap — make sibling mathir_lib importable when run as a script
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAEMON_URL = os.environ.get("MATHIR_DAEMON_URL", "http://127.0.0.1:7338").rstrip("/")
TARGET_URL = os.environ.get("MATHIR_PROXY_TARGET", "https://api.openai.com/v1").rstrip("/")
INJECT_K = int(os.environ.get("MATHIR_PROXY_INJECT_K", "8"))
DEBUG = os.environ.get("MATHIR_PROXY_DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# Logging — rotating file independent of launcher redirection
# ---------------------------------------------------------------------------
try:
    from .mathir_paths import LOG_DIR as _P_LOG
except ImportError:
    from mathir_paths import LOG_DIR as _P_LOG
_log_dir = Path(os.environ.get("MATHIR_LOG_DIR", str(_P_LOG)))
try:
    _log_dir.mkdir(parents=True, exist_ok=True)
except OSError:
    _log_dir = Path(os.environ.get("TEMP", "/tmp")) / "mathir_logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _log_dir / "mathir_proxy.log"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [MATHIR-PROXY] %(levelname)s %(message)s",
                    stream=sys.stderr)
log = logging.getLogger("mathir-proxy")
try:
    _fh = logging.handlers.RotatingFileHandler(
        _LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    _fh.setFormatter(logging.Formatter("%(asctime)s [MATHIR-PROXY] %(levelname)s %(message)s"))
    log.addHandler(_fh)
except OSError:
    pass

# ---------------------------------------------------------------------------
# Security helpers — shared sanitizer + SSRF redirect guard + auth allowlist
# ---------------------------------------------------------------------------
# Max bytes of sanitized memory text we'll inject into a system prompt.
INJECT_MAX_BYTES = 8 * 1024

# Substrings that must never appear verbatim in injected memory text —
# they'd let a malicious memory break out of the injection block, recurse
# into the placeholder, or inject chat-template control tokens.
_FORBIDDEN_SUBSTRINGS = (
    "</mathir-",          # break out of <mathir-...> injection block
    "{{MATHIR_CONTEXT}}", # recurse into the placeholder in inject_proxy
    "<|",                 # chat-template tokens: <|im_start|>, <|endoftext|>, ...
)


def sanitize_memory_for_injection(text, max_bytes: int = INJECT_MAX_BYTES) -> str:
    """Sanitize recalled memory text before it enters an LLM system prompt.

    Defense against stored prompt-injection from recalled memory content:
      1. Strip substrings that could break out of the injection block or
         masquerade as chat-template control tokens.
      2. Prefix every line with ``> `` so the model treats recalled memory
         as quoted data, not as fresh instructions from the operator.
      3. Cap total size so a runaway recall cannot drown the prompt.
    """
    if not text:
        return ""
    cleaned = text
    for bad in _FORBIDDEN_SUBSTRINGS:
        cleaned = cleaned.replace(bad, "")
    lines = cleaned.splitlines() or [""]
    quoted = "\n".join(f"> {ln}" if ln else ">" for ln in lines)
    encoded = quoted.encode("utf-8", errors="ignore")
    if len(encoded) > max_bytes:
        quoted = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return quoted


def _is_loopback_url(url: str) -> bool:
    """True iff every address the URL's host resolves to is loopback.

    Fails closed: malformed URLs and unresolvable hosts return False.
    """
    try:
        host = urlparse(url).hostname
    except Exception:
        return False
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for _, _, _, _, sockaddr in infos:
        try:
            if not ipaddress.ip_address(sockaddr[0]).is_loopback:
                return False
        except (ValueError, IndexError):
            return False
    return bool(infos)


class _LoopbackOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects whose target is not loopback (SSRF defense)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_loopback_url(newurl):
            log.warning(f"Refusing non-loopback redirect target: {newurl}")
            raise urllib.error.URLError(
                f"Refusing non-loopback redirect target: {newurl}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_LOOPBACK_OPENER = urllib.request.build_opener(_LoopbackOnlyRedirectHandler())


# Snapshot the configured upstream host at import time. If the operator
# repoints the proxy at runtime to a different host, that runtime host is
# NOT in this allowlist and Authorization will be stripped — preventing
# accidental credential leakage to an unexpected upstream.
_CONFIGURED_UPSTREAM_HOST = ""
try:
    _CONFIGURED_UPSTREAM_HOST = (urlparse(TARGET_URL).hostname or "").lower()
except Exception:
    pass

_AUTH_FORWARD_ALLOWLIST = {
    "127.0.0.1", "localhost", "::1", _CONFIGURED_UPSTREAM_HOST,
}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


# ---------------------------------------------------------------------------
# MATHIR context fetch
# ---------------------------------------------------------------------------
def _extract_last_user_message(messages: list) -> str:
    """Return the last user message text from an OpenAI-style messages array."""
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # OpenAI vision format: [{"type": "text", "text": "..."}, ...]
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            if parts:
                return " ".join(parts)
    return ""


def _fetch_context(task: str, k: int = INJECT_K) -> Optional[str]:
    """Call the MATHIR daemon /api/context and return formatted context, or None."""
    if not task or len(task) < 5:
        return None
    payload = json.dumps({"task": task[:5000], "k": k}).encode()
    try:
        req = urllib.request.Request(
            f"{DAEMON_URL}/api/context",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with _LOOPBACK_OPENER.open(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
        if isinstance(data, dict):
            if data.get("error"):
                log.warning(f"/api/context error: {data['error']}")
                return None
            return data.get("context")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        log.warning(f"MATHIR daemon unreachable at {DAEMON_URL}: {e}")
    except Exception as e:
        log.warning(f"unexpected error fetching context: {e}")
    return None


def _augment_messages(messages: list, context: str) -> list:
    """Inject the MATHIR context block into the messages array.

    Strategy: if there's already a system message, append to it; otherwise
    prepend a new system message. We avoid mutating the user's existing
    system prompt semantics — we just add a clearly-delimited block.
    """
    safe_context = sanitize_memory_for_injection(context)
    block = f"<mathir-auto-injection>\n{safe_context}\n</mathir-auto-injection>"
    new_messages = []
    injected = False
    for msg in messages:
        if not injected and isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                new_messages.append({**msg, "content": content + "\n\n" + block})
                injected = True
                continue
        new_messages.append(msg)
    if not injected:
        new_messages.insert(0, {"role": "system", "content": block})
    return new_messages


# ---------------------------------------------------------------------------
# Routes — transparent OpenAI-compatible proxy
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return {
        "status": "ok",
        "target": TARGET_URL,
        "daemon": DAEMON_URL,
        "inject_k": INJECT_K,
        "version": "1.0.0",
    }


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Augment system prompt with MATHIR context, then forward to upstream."""
    body = request.get_json(force=True, silent=True) or {}
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        # Don't try to augment — just forward as-is
        return _forward(request.path, stream=body.get("stream", False) if isinstance(body, dict) else False)

    task = _extract_last_user_message(messages)
    context = _fetch_context(task)
    if context:
        body["messages"] = _augment_messages(messages, context)
        if DEBUG:
            log.info(f"augmented request (task='{task[:60]}...', +{len(context)} chars context)")
    elif DEBUG:
        log.info(f"no context for task='{(task[:60] + '...') if task else '(empty)'}'")

    stream = bool(body.get("stream"))
    return _forward_with_body(request.path, body, stream=stream)


@app.route("/v1/completions", methods=["POST"])
def completions():
    """Legacy completions endpoint — forward without augmentation (rarely used)."""
    body = request.get_json(force=True, silent=True) or {}
    return _forward_with_body(request.path, body, stream=bool(isinstance(body, dict) and body.get("stream")))


@app.route("/v1/embeddings", methods=["POST"])
@app.route("/v1/models", methods=["GET"])
@app.route("/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
def passthrough(subpath: str = ""):
    """Transparent passthrough for all other OpenAI endpoints."""
    return _forward(request.path, stream=False)


# ---------------------------------------------------------------------------
# Forwarding helpers
# ---------------------------------------------------------------------------
def _build_upstream_url(path: str) -> str:
    # path starts with /v1/... — append to TARGET_URL
    return f"{TARGET_URL}{path}"


def _forward_headers() -> dict:
    """Copy through headers; drop hop-by-hop.

    Authorization is forwarded ONLY when the current upstream hostname is in
    the allowlist (loopback + the configured-openai-style host captured at
    import time). Otherwise it is stripped and we warn loudly — defense
    against credential leakage if the proxy is repointed at runtime.
    """
    try:
        upstream_host = (urlparse(TARGET_URL).hostname or "").lower()
    except Exception:
        upstream_host = ""
    allow_auth = upstream_host in _AUTH_FORWARD_ALLOWLIST
    h = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "connection", "transfer-encoding"):
            continue
        if kl == "authorization" and not allow_auth:
            log.warning(
                "STRIPPING Authorization header: upstream host '%s' not in "
                "allowlist %s (target=%s). Point proxy at loopback or the "
                "configured openai-style host to forward credentials.",
                upstream_host, sorted(_AUTH_FORWARD_ALLOWLIST), TARGET_URL,
            )
            continue
        h[k] = v
    return h


def _forward(path: str, stream: bool):
    """Forward the original request body verbatim."""
    body = request.get_data()
    return _forward_raw(path, body, stream=stream)


def _forward_with_body(path: str, body: dict, stream: bool):
    """Forward with a modified JSON body."""
    return _forward_raw(path, json.dumps(body).encode("utf-8"), stream=stream,
                        extra_headers={"Content-Type": "application/json"})


def _forward_raw(path: str, body: bytes, stream: bool, extra_headers: Optional[dict] = None):
    """Send the request to the upstream LLM API and stream the response back."""
    import requests  # local import — only needed when proxying
    url = _build_upstream_url(path)
    headers = _forward_headers()
    if extra_headers:
        headers.update(extra_headers)
    method = request.method

    try:
        upstream = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            stream=stream,
            timeout=300,  # LLM responses can be slow
        )
    except requests.exceptions.ConnectionError as e:
        log.error(f"upstream connection error: {e}")
        return Response(
            json.dumps({"error": {"message": f"MATHIR proxy: upstream unreachable: {e}",
                                   "type": "proxy_error"}}),
            status=502, content_type="application/json",
        )
    except requests.exceptions.RequestException as e:
        log.error(f"upstream request error: {e}")
        return Response(
            json.dumps({"error": {"message": f"MATHIR proxy: {e}", "type": "proxy_error"}}),
            status=502, content_type="application/json",
        )

    # Pass through status + headers (drop hop-by-hop)
    excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    resp_headers = [(k, v) for k, v in upstream.headers.items() if k.lower() not in excluded]

    if stream:
        def generate():
            try:
                for chunk in upstream.iter_content(chunk_size=4096):
                    if chunk:
                        yield chunk
            except requests.exceptions.RequestException as e:
                log.warning(f"stream interrupted: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
        return Response(stream_with_context(generate()),
                        status=upstream.status_code,
                        headers=resp_headers,
                        content_type=upstream.headers.get("Content-Type", "text/event-stream"))
    else:
        return Response(upstream.content,
                        status=upstream.status_code,
                        headers=resp_headers,
                        content_type=upstream.headers.get("Content-Type", "application/json"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global TARGET_URL
    parser = argparse.ArgumentParser(description="MATHIR LLM Proxy — universal memory injection")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MATHIR_PROXY_PORT", "7339")))
    parser.add_argument("--host", default=os.environ.get("MATHIR_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--target", default=TARGET_URL,
                        help=f"Upstream LLM base URL (default: {TARGET_URL})")
    parser.add_argument("--workers", type=int, default=8, help="Waitress threads")
    args = parser.parse_args()

    TARGET_URL = args.target.rstrip("/")

    log.info(f"MATHIR proxy starting on {args.host}:{args.port}")
    log.info(f"  daemon: {DAEMON_URL}")
    log.info(f"  target: {TARGET_URL}")
    log.info(f"  inject_k: {INJECT_K}, log: {_LOG_PATH}")

    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=args.workers)
    except ImportError:
        log.warning("waitress not installed — using Flask dev server (not for production)")
        app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
