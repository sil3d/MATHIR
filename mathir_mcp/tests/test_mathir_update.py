"""Tests for the update checker and updater state machine.

Strategy:
- Unit-test parse_version + cache helpers (no network)
- Mock the GitHub API for check_for_update (test all branches)
- For the state machine: exercise check_only + update --dry-run which
  doesn't touch real files. The full update flow is tested in CI
  (test.yml) against a fresh install.
"""

import json
import time
from pathlib import Path
from unittest import mock

import pytest

from mathir_mcp.mathir_lib import mathir_update_check
from mathir_mcp.mathir_lib.mathir_update_check import (
    parse_version,
    check_for_update,
    _read_cache,
    _write_cache,
    _cache_path,
    _fetch_releases,
    CACHE_FILENAME,
)


# ---------------------------------------------------------------------------
# parse_version
# ---------------------------------------------------------------------------

class TestParseVersion:
    def test_basic(self):
        assert parse_version("8.5.1") == (8, 5, 1)
        assert parse_version("v8.5.1") == (8, 5, 1)

    def test_with_prerelease_suffix(self):
        # '8.5.1-dev.abc1234' -> '8.5.1' (suffix ignored)
        assert parse_version("8.5.1-dev.abc1234") == (8, 5, 1)
        assert parse_version("v8.5.1-rc1") == (8, 5, 1)

    def test_invalid(self):
        assert parse_version("") == (0, 0, 0)
        assert parse_version("garbage") == (0, 0, 0)
        assert parse_version(None) == (0, 0, 0)

    def test_comparison(self):
        assert parse_version("8.5.2") > parse_version("8.5.1")
        assert parse_version("9.0.0") > parse_version("8.99.99")
        assert parse_version("8.5.1") == parse_version("8.5.1-dev.abc")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_write_and_read(self, tmp_path, monkeypatch):
        # Redirect cache to tmp
        cache = tmp_path / CACHE_FILENAME
        monkeypatch.setattr(mathir_update_check, "_cache_path", lambda: cache)

        payload = {"latest_version": "8.5.2", "checked_at": time.time()}
        _write_cache(payload)

        result = _read_cache()
        assert result is not None
        assert result["latest_version"] == "8.5.2"

    def test_stale_cache_rejected(self, tmp_path, monkeypatch):
        cache = tmp_path / CACHE_FILENAME
        monkeypatch.setattr(mathir_update_check, "_cache_path", lambda: cache)
        # 2 hours old, beyond TTL of 1h
        _write_cache({"latest_version": "8.0.0", "checked_at": time.time() - 7200})
        assert _read_cache() is None

    def test_corrupted_cache_rejected(self, tmp_path, monkeypatch):
        cache = tmp_path / CACHE_FILENAME
        monkeypatch.setattr(mathir_update_check, "_cache_path", lambda: cache)
        cache.write_text("not json")
        assert _read_cache() is None


# ---------------------------------------------------------------------------
# check_for_update
# ---------------------------------------------------------------------------

def _fake_release(tag, prerelease=False, draft=False):
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": draft,
        "html_url": f"https://github.com/sil3d/MATHIR/releases/tag/{tag}",
        "assets": [],
    }


class TestCheckForUpdate:
    def test_no_new_release(self, monkeypatch):
        monkeypatch.setattr(mathir_update_check, '_fetch_releases', lambda: [
            _fake_release('v8.5.1'),
            _fake_release('v8.5.0'),
        ])
        result = check_for_update('8.5.1', force=True)
        assert result["update_available"] is False
        assert result["latest_version"] == "8.5.1"
        assert result["source"] == "live"

    def test_new_release_available(self, monkeypatch):
        monkeypatch.setattr(mathir_update_check, '_fetch_releases', lambda: [
            _fake_release('v8.5.2'),
            _fake_release('v8.5.1'),
        ])
        result = check_for_update('8.5.1', force=True)
        assert result["update_available"] is True
        assert result["latest_version"] == "8.5.2"
        assert "releases/tag" in result["release_url"]

    def test_prerelease_excluded_by_default(self, monkeypatch):
        monkeypatch.setattr(mathir_update_check, '_fetch_releases', lambda: [
            _fake_release('v8.6.0-rc1', prerelease=True),
            _fake_release('v8.5.1'),
        ])
        result = check_for_update('8.5.1', include_prerelease=False, force=True)
        # Stable latest is 8.5.1, so no update
        assert result["update_available"] is False
        assert result["latest_version"] == "8.5.1"

    def test_prerelease_included_when_requested(self, monkeypatch):
        monkeypatch.setattr(mathir_update_check, '_fetch_releases', lambda: [
            _fake_release('v8.6.0-rc1', prerelease=True),
            _fake_release('v8.5.1'),
        ])
        result = check_for_update('8.5.1', include_prerelease=True, force=True)
        assert result["update_available"] is True
        # With include_prerelease, latest_version picks the prerelease
        # (8.6.0-rc1 > 8.5.1 by semver)
        assert result["latest_version"] == "8.6.0-rc1"
        assert result["latest_prerelease"] == "8.6.0-rc1"

    def test_drafts_skipped(self, monkeypatch):
        monkeypatch.setattr(mathir_update_check, '_fetch_releases', lambda: [
            _fake_release('v8.5.2-draft', draft=True),
            _fake_release('v8.5.1'),
        ])
        result = check_for_update('8.5.1', force=True)
        # Drafts ignored — latest stable is 8.5.1
        assert result["update_available"] is False

    def test_network_failure_returns_no_info(self, monkeypatch, tmp_path):
        # Point cache to tmp so we don't poison the real one
        monkeypatch.setattr(mathir_update_check, "_cache_path", lambda: tmp_path / "c.json")
        monkeypatch.setattr(mathir_update_check, "_fetch_releases",
                            mock.Mock(side_effect=OSError("no network")))
        result = check_for_update("8.5.1", force=True)
        assert result["update_available"] is False
        assert result["error"] is not None
        assert "no network" in result["error"]

    def test_network_failure_with_stale_cache(self, monkeypatch, tmp_path):
        cache = tmp_path / "c.json"
        # Pre-populate with a stale-but-existing cache
        stale_payload = {
            "latest_version": "8.5.0", "checked_at": time.time() - 100000,
            "update_available": True, "release_url": "x",
        }
        cache.write_text(json.dumps(stale_payload))
        monkeypatch.setattr(mathir_update_check, "_cache_path", lambda: cache)
        monkeypatch.setattr(mathir_update_check, "_fetch_releases",
                            mock.Mock(side_effect=OSError("offline")))
        result = check_for_update("8.5.1", force=True)
        # Stale cache returned with source='cache-stale'
        assert result["source"] == "cache-stale"
        assert result["latest_version"] == "8.5.0"
        assert "offline" in result["error"]

    def test_cached_returns_fast(self, monkeypatch, tmp_path):
        cache = tmp_path / CACHE_FILENAME
        cache.write_text(json.dumps({
            "checked_at": time.time(),
            "latest_version": "8.5.5",
            "update_available": True,
            "release_url": "x",
        }))
        monkeypatch.setattr(mathir_update_check, "_cache_path", lambda: cache)
        # Force flag bypasses cache — but we're testing WITHOUT force
        result = check_for_update("8.5.1")
        assert result["source"] == "cache"
        assert result["latest_version"] == "8.5.5"
        assert result["update_available"] is True

    def test_force_bypasses_cache(self, monkeypatch, tmp_path):
        cache = tmp_path / CACHE_FILENAME
        cache.write_text(json.dumps({
            "checked_at": time.time(),
            "latest_version": "8.5.0",  # stale cache
            "update_available": False,
        }))
        monkeypatch.setattr(mathir_update_check, "_cache_path", lambda: cache)
        monkeypatch.setattr(mathir_update_check, "_fetch_releases", lambda: [
            _fake_release("v8.5.2"),
        ])
        result = check_for_update("8.5.1", force=True)
        assert result["source"] == "live"
        assert result["latest_version"] == "8.5.2"
        assert result["update_available"] is True


# ---------------------------------------------------------------------------
# Updater state machine (check_only + dry-run)
# ---------------------------------------------------------------------------

class TestUpdaterCheckOnly:
    def test_check_only_returns_expected_fields(self, monkeypatch):
        from mathir_mcp.mathir_lib import mathir_updater
        # Force "bundle" mode (no .git)
        monkeypatch.setattr(mathir_updater, "_is_git_install", lambda *_: False)
        result = mathir_updater.check_only(force=True)
        assert "current" in result
        assert "install_mode" in result
        assert "installed_path" in result
        assert result["install_mode"] == "bundle"

    def test_check_only_git_mode(self, monkeypatch):
        from mathir_mcp.mathir_lib import mathir_updater
        monkeypatch.setattr(mathir_updater, "_is_git_install", lambda *_: True)
        result = mathir_updater.check_only(force=True)
        assert result["install_mode"] == "git"

    def test_dry_run_shows_plan(self, monkeypatch):
        from mathir_mcp.mathir_lib import mathir_updater
        # Mock check_for_update to return a fake newer version
        monkeypatch.setattr(mathir_updater, "check_for_update", lambda *a, **kw: {
            "latest_version": "8.5.2", "update_available": True,
            "release_url": "x", "error": None, "source": "live",
        })
        report = mathir_updater.update(dry_run=True)
        assert report["target"] == "8.5.2"
        assert any("DRY-RUN" in s for s in report["steps"])

    def test_no_op_when_already_current(self, monkeypatch, tmp_path):
        from mathir_mcp.mathir_lib import mathir_updater
        # Same version -> target <= current -> return early
        monkeypatch.setattr(mathir_updater, "_mcp_dir", lambda: tmp_path)
        monkeypatch.setattr(mathir_updater, "_read_pyproject_version", lambda *a, **kw: "8.5.1")
        monkeypatch.setattr(mathir_updater, "_is_git_install", lambda *a, **kw: False)
        monkeypatch.setattr(mathir_updater, "check_for_update", lambda *a, **kw: {
            "latest_version": "8.5.1", "update_available": False,
            "release_url": None, "error": None, "source": "live",
        })
        report = mathir_updater.update()
        assert any("already at" in s for s in report["steps"])


class TestUpdaterLocking:
    def test_concurrent_update_blocked(self, monkeypatch, tmp_path):
        from mathir_mcp.mathir_lib import mathir_updater
        # Redirect lock file to tmp
        lock = tmp_path / ".updater.lock"
        monkeypatch.setattr(mathir_updater, "_lock_file", lambda: lock)
        # First acquire should succeed
        assert mathir_updater._acquire_lock() is not None
        # Second acquire should fail (lock held)
        assert mathir_updater._acquire_lock() is None
        # Cleanup
        mathir_updater._release_lock()


class TestUpdaterRollback:
    def test_rollback_without_state_fails_gracefully(self, monkeypatch, tmp_path):
        from mathir_mcp.mathir_lib import mathir_updater
        # No state file -> RuntimeError
        state = tmp_path / ".updater_state.json"
        monkeypatch.setattr(mathir_updater, "_state_file", lambda: state)
        with pytest.raises(RuntimeError, match="no rollback state"):
            mathir_updater.rollback()