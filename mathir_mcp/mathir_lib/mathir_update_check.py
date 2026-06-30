"""MATHIR update checker — query GitHub Releases API for newer versions.

Used by:
1. Daemon `/health` endpoint — exposes `update_available` to MCP clients
2. `python -m mathir_mcp update --check-only` — pre-flight before update
3. The auto-injection plugin in opencode/mimocode — surfaces the warning
   to the agent at session.started

Design choices:
- Never raises — always returns a dict. Network/parse failures populate
  `error` but the caller still gets useful info (current version, cache).
- Caches result to ~/.config/MATHIR/.update_check.json (or HOME)
  with a TTL of 1h. Subsequent calls within the window are free.
- `prerelease` releases are excluded by default — they're for devs.
  Pass include_prerelease=True to opt in.
- Works fully offline: on first failure after install, the cache file
  doesn't exist yet; on subsequent failures, last-known is returned.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger("mathir-update-check")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITHUB_REPO = "sil3d/MATHIR"
GITHUB_API_RELEASES = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
CACHE_FILENAME = ".update_check.json"
CACHE_TTL_SECONDS = 3600          # 1 hour — daemon checks at most once per hour
HTTP_TIMEOUT_SECONDS = 5          # don't block daemon boot if GitHub is slow
USER_AGENT = "mathir-daemon/1.0 (+https://github.com/sil3d/MATHIR)"

# Where to cache the check result. Lives next to other MATHIR state so a
# single `~/.config/MATHIR/` wipe nuke it. Falls back to $HOME if the
# canonical dir doesn't exist yet (fresh install before the daemon
# ever ran).
def _cache_path() -> Path:
    env = os.environ.get("MATHIR_HOME")
    if env:
        return Path(env) / CACHE_FILENAME
    canonical = Path.home() / ".config" / "mathir" / CACHE_FILENAME
    if canonical.parent.exists():
        return canonical
    return Path.home() / CACHE_FILENAME


# ---------------------------------------------------------------------------
# Version parsing — accepts "8.5.1", "v8.5.1", "8.5.1-dev.abc1234"
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_version(tag: str) -> tuple:
    """Return a comparable version tuple from a tag like 'v8.5.1' or '8.5.1-dev.abc'.

    Suffix after the first '-' is ignored (pre-release identifiers).
    Falls back to (0, 0, 0) for unparseable input so caller never crashes.
    """
    if not tag:
        return (0, 0, 0)
    m = _VERSION_RE.match(tag.strip().lstrip("v"))
    if not m:
        return (0, 0, 0)
    return tuple(int(x) for x in m.groups())


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _read_cache() -> Optional[dict]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.debug(f"cache read failed: {e}")
        return None
    checked_at = data.get("checked_at", 0)
    if (time.time() - checked_at) > CACHE_TTL_SECONDS:
        return None  # stale
    return data


def _write_cache(payload: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as e:
        log.debug(f"cache write failed: {e}")


# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------


def _fetch_releases() -> list:
    """GET the GitHub Releases API. Returns a list of release dicts.
    Raises on any failure — caller handles."""
    req = urllib.request.Request(
        GITHUB_API_RELEASES,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _select_latest(releases: list, *, include_prerelease: bool) -> Optional[dict]:
    """Pick the best release from a list of dicts. Skips drafts always."""
    candidates = [r for r in releases if not r.get("draft")]
    if not include_prerelease:
        candidates = [r for r in candidates if not r.get("prerelease")]
    if not candidates:
        return None
    return max(candidates, key=lambda r: parse_version(r.get("tag_name", "")))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_for_update(
    current_version: str,
    *,
    include_prerelease: bool = False,
    force: bool = False,
) -> dict:
    """Return a dict describing whether a newer version is available.

    Never raises. On network failure, returns last cached payload (or
    a 'no info' payload) with `error` populated.

    Returns:
        {
            "checked_at": float (epoch),
            "current_version": "8.5.1",
            "latest_version": "8.5.2",         # or current if no newer
            "latest_prerelease": "8.6.0-rc1",   # only if include_prerelease
            "update_available": bool,
            "release_url": "https://github.com/...",
            "source": "live" | "cache",
            "error": str | None
        }
    """
    now = time.time()

    # 1. Try cache unless caller forces a fresh fetch
    if not force:
        cached = _read_cache()
        if cached:
            cached["source"] = "cache"
            # Honor the include_prerelease toggle even from cache
            if not include_prerelease:
                cached.pop("latest_prerelease", None)
            # Update check result based on cache against current_version
            cached["update_available"] = (
                parse_version(cached.get("latest_version", ""))
                > parse_version(current_version)
            )
            return cached

    # 2. Fresh fetch
    payload = {
        "checked_at": now,
        "current_version": current_version,
        "latest_version": current_version,
        "latest_prerelease": current_version,
        "update_available": False,
        "release_url": None,
        "source": "live",
        "error": None,
    }

    try:
        releases = _fetch_releases()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError) as e:
        # Try stale cache as fallback
        path = _cache_path()
        stale = None
        if path.exists():
            try:
                stale = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if stale:
            stale["source"] = "cache-stale"
            stale["error"] = f"refresh failed: {type(e).__name__}: {str(e)[:120]}"
            return stale
        payload["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        _write_cache(payload)  # cache the 'no info' so we don't hammer GitHub
        return payload

    # 3. Parse + select
    latest = _select_latest(releases, include_prerelease=include_prerelease)
    if latest:
        tag = latest.get("tag_name", "")
        payload["latest_version"] = tag.lstrip("v")
        payload["release_url"] = latest.get("html_url")
        payload["update_available"] = parse_version(payload["latest_version"]) > parse_version(current_version)

    if include_prerelease:
        pre = _select_latest(releases, include_prerelease=True)
        if pre:
            payload["latest_prerelease"] = pre.get("tag_name", "").lstrip("v")

    _write_cache(payload)
    return payload


def is_update_available(current_version: str, **kwargs) -> bool:
    """Convenience: just the boolean."""
    return bool(check_for_update(current_version, **kwargs).get("update_available"))


# ---------------------------------------------------------------------------
# CLI for manual use
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Check for MATHIR updates")
    parser.add_argument("--current", default=os.environ.get("MATHIR_VERSION", "0.0.0"),
                        help="Current installed version (default: $MATHIR_VERSION or 0.0.0)")
    parser.add_argument("--include-prerelease", action="store_true")
    parser.add_argument("--force", action="store_true", help="Bypass cache")
    args = parser.parse_args()

    result = check_for_update(args.current,
                              include_prerelease=args.include_prerelease,
                              force=args.force)
    if result.get("update_available"):
        print(f"UPDATE AVAILABLE: {result['latest_version']}")
        print(f"  current:  {result['current_version']}")
        print(f"  latest:   {result['latest_version']}")
        if result.get("latest_prerelease") and result["latest_prerelease"] != result["latest_version"]:
            print(f"  prerelease: {result['latest_prerelease']}")
        if result.get("release_url"):
            print(f"  url:      {result['release_url']}")
        print(f"  source:   {result.get('source')}")
        print(f"\nRun: python -m mathir_mcp update")
        sys.exit(0)
    else:
        print(f"UP-TO-DATE: {result['current_version']}")
        if result.get("error"):
            print(f"  warning: {result['error']}")
        sys.exit(0)