#!/usr/bin/env python3
"""Launch MATHIR unified server with waitress (production WSGI)."""
import sys
import os
import threading
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MATHIR] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("mathir-launcher")

# Pre-warm embedder in background
def warmup():
    log.info("Pre-loading embedder (this takes ~15-30s on first run)...")
    try:
        from mathir_mcp_server import get_embedder
        get_embedder()
        log.info("Embedder ready")
    except Exception as e:
        log.error(f"Embedder warmup failed: {e}")

t = threading.Thread(target=warmup, daemon=True)
t.start()

from mathir_server import app
from waitress import serve

log.info("Starting waitress on 127.0.0.1:7338 (8 threads)")
serve(app, host='127.0.0.1', port=7338, threads=8)
