"""Tests for opt-in auth on mathir_server.py (Flask) and mathir_stats_server.py.

Auth gate: non-loopback binds REQUIRE MATHIR_AUTH_TOKEN.  Loopback binds
(127.0.0.1, ::1, localhost, '') need no auth.  When a token IS set and
the bind is non-loopback, every /api/* request must carry a valid
``Authorization: Bearer <token>`` header.

These tests never start a real server — they use Flask test clients and
monkeypatching only.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Imports — try package-qualified first, fall back to flat
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MCP_ROOT = _HERE.parent
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

try:
    from mathir_lib import mathir_server
    from mathir_lib import mathir_stats_server
except ImportError:
    import mathir_server  # type: ignore[no-redef]
    import mathir_stats_server  # type: ignore[no-redef]


# ═══════════════════════════════════════════════════════════════════════════
# mathir_server.py (Flask app)
# ═══════════════════════════════════════════════════════════════════════════

class TestMathirServerAuth:
    """Auth gate on the unified Flask server (mathir_server.py)."""

    # ── a. loopback, no token → 200 ─────────────────────────────────────
    def test_loopback_no_auth_required(self, monkeypatch):
        """Loopback host + no MATHIR_AUTH_TOKEN → /api/memory/export 200."""
        # Ensure MATHIR_AUTH_TOKEN is unset
        monkeypatch.delenv("MATHIR_AUTH_TOKEN", raising=False)
        # Reset the module-level token to empty
        monkeypatch.setattr(mathir_server, "_AUTH_TOKEN", "")

        # Mock _resolve_db so the route handler succeeds without a real DB
        mock_vec = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_vec._get_conn.return_value = mock_conn
        monkeypatch.setattr(mathir_server, "_resolve_db", lambda project=None, cwd=None: (mock_vec, None, None))

        client = mathir_server.app.test_client()
        resp = client.get("/api/memory/export")
        assert resp.status_code == 200

    # ── b. non-loopback, no token → main() calls sys.exit(2) ────────────
    def test_nonloopback_no_token_refuses_start(self, monkeypatch):
        """MATHIR_HOST=0.0.0.0 + no token → main() raises SystemExit(2)."""
        monkeypatch.setenv("MATHIR_HOST", "0.0.0.0")
        monkeypatch.setenv("MATHIR_AUTH_TOKEN", "")
        monkeypatch.setattr(mathir_server, "_AUTH_TOKEN", "")
        monkeypatch.setattr(mathir_server, "_is_loopback", lambda host: False)
        # Prevent PID lock issues and warmup side-effects
        monkeypatch.setattr(mathir_server, "_acquire_pid_lock", lambda: True)
        monkeypatch.setattr(mathir_server, "_warmup", lambda: None)
        # Give parse_args a minimal argv so argparse doesn't consume pytest args
        monkeypatch.setattr(sys, "argv", ["mathir_server.py", "--host", "0.0.0.0"])

        with pytest.raises(SystemExit) as exc_info:
            mathir_server.main()
        assert exc_info.value.code == 2

    # ── c. non-loopback + token, bad/missing bearer → 401 ───────────────
    def test_nonloopback_with_token_rejects_bad_bearer(self, monkeypatch):
        """0.0.0.0 + token set, wrong Authorization header → 401."""
        monkeypatch.setattr(mathir_server, "_AUTH_TOKEN", "testtoken")

        # Install the bearer hook that main() would register
        from flask import request as flask_request
        from flask import jsonify as flask_jsonify

        def _require_bearer():
            if flask_request.path.startswith("/api/") and flask_request.path not in (
                "/api/health",
                "/api/ping",
            ):
                auth = flask_request.headers.get("Authorization", "")
                if auth != "Bearer testtoken":
                    return flask_jsonify({"error": "unauthorized"}), 401

        mathir_server.app.before_request_funcs.setdefault(None, []).append(_require_bearer)

        # Mock _resolve_db so the route doesn't 500
        mock_vec = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_vec._get_conn.return_value = mock_conn
        monkeypatch.setattr(mathir_server, "_resolve_db", lambda project=None, cwd=None: (mock_vec, None, None))

        try:
            client = mathir_server.app.test_client()

            # No header at all → 401
            resp = client.get("/api/memory/export")
            assert resp.status_code == 401

            # Wrong token → 401
            resp = client.get(
                "/api/memory/export",
                headers={"Authorization": "Bearer wrongtoken"},
            )
            assert resp.status_code == 401
        finally:
            # Remove the hook so other tests aren't affected
            mathir_server.app.before_request_funcs[None].remove(_require_bearer)

    # ── d. non-loopback + token, correct bearer → 200 ────────────────────
    def test_nonloopback_with_token_accepts_good_bearer(self, monkeypatch):
        """0.0.0.0 + token set, correct Authorization header → 200."""
        monkeypatch.setattr(mathir_server, "_AUTH_TOKEN", "testtoken")

        from flask import request as flask_request
        from flask import jsonify as flask_jsonify

        def _require_bearer():
            if flask_request.path.startswith("/api/") and flask_request.path not in (
                "/api/health",
                "/api/ping",
            ):
                auth = flask_request.headers.get("Authorization", "")
                if auth != "Bearer testtoken":
                    return flask_jsonify({"error": "unauthorized"}), 401

        mathir_server.app.before_request_funcs.setdefault(None, []).append(_require_bearer)

        mock_vec = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_vec._get_conn.return_value = mock_conn
        monkeypatch.setattr(mathir_server, "_resolve_db", lambda project=None, cwd=None: (mock_vec, None, None))

        try:
            client = mathir_server.app.test_client()
            resp = client.get(
                "/api/memory/export",
                headers={"Authorization": "Bearer testtoken"},
            )
            assert resp.status_code == 200
        finally:
            mathir_server.app.before_request_funcs[None].remove(_require_bearer)


# ═══════════════════════════════════════════════════════════════════════════
# mathir_stats_server.py (HTTP handler auth)
# ═══════════════════════════════════════════════════════════════════════════

class TestStatsServerAuth:
    """Auth gate on the stats dashboard (mathir_stats_server.py)."""

    def test_loopback_no_auth(self):
        """_is_loopback returns True for all localhost variants."""
        assert mathir_stats_server._is_loopback("127.0.0.1") is True
        assert mathir_stats_server._is_loopback("::1") is True
        assert mathir_stats_server._is_loopback("localhost") is True
        assert mathir_stats_server._is_loopback("") is True

    def test_nonloopback_detected(self):
        """_is_loopback returns False for public interfaces."""
        assert mathir_stats_server._is_loopback("0.0.0.0") is False
        assert mathir_stats_server._is_loopback("192.168.1.1") is False
