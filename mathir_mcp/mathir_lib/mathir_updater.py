"""MATHIR updater — state machine for safe version upgrades.

States:
  DETECT       -> git or bundle mode (depending on installed layout)
  DISCOVER     -> current version + target version + new release notes
  BACKUP       -> atomic snapshot of current install
  DOWNLOAD     -> fetch bundle (git fetch + checkout OR zip download)
  APPLY        -> replace mathir_mcp/ contents with new version
  MIGRATE      -> run schema migration if needed (auto-backs up .legacy.bak)
  RESTART      -> stop old daemon, start new one
  REPORT       -> print summary, exit 0
  ROLLBACK     -> if any step fails, restore from backup + restart

Entry points:
  - python -m mathir_mcp update              (CLI: __main__.py)
  - python -m mathir_mcp.mathir_lib.mathir_updater update --check-only
  - python -m mathir_mcp.mathir_lib.mathir_updater rollback

Design principles:
  - Atomic: any failure restores from backup. No half-upgraded state.
  - Idempotent: --check-only is safe to run repeatedly.
  - Transparent: every step logs verbosely.
  - Concurrency-safe: PID file lock prevents concurrent updates.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from .mathir_update_check import (
    parse_version,
    check_for_update,
    GITHUB_API_RELEASES,
)

log = logging.getLogger("mathir-updater")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _mcp_dir() -> Path:
    """Path to the installed mathir_mcp/ directory."""
    env = os.environ.get("MATHIR_HOME")
    if env:
        return Path(env) / "mathir_mcp"
    return Path.home() / ".config" / "MATHIR" / "mathir_mcp"


def _backups_dir() -> Path:
    """Where backups are kept before each update."""
    return _mcp_dir().parent / ".backups"


def _state_file() -> Path:
    """Last-update state (used by rollback)."""
    return _mcp_dir().parent / ".updater_state.json"


def _lock_file() -> Path:
    return _mcp_dir().parent / ".updater.lock"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_pyproject_version(mcp_dir: Path) -> str:
    pp = mcp_dir / "pyproject.toml"
    if not pp.is_file():
        return "unknown"
    try:
        import re
        m = re.search(r'version\s*=\s*"([^"]+)"', pp.read_text(encoding="utf-8"))
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def _is_git_install(mcp_dir: Path) -> bool:
    return (mcp_dir / ".git").exists() or (mcp_dir.parent / ".git").exists()


def _acquire_lock() -> Optional[int]:
    """Atomic PID lock — returns None if another updater is running."""
    lf = _lock_file()
    try:
        lf.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return os.getpid()
    except FileExistsError:
        try:
            existing_pid = int(lf.read_text().strip() or "0")
        except Exception:
            existing_pid = 0
        # Stale lock check
        if existing_pid > 0:
            try:
                import psutil  # type: ignore
                if not psutil.pid_exists(existing_pid):
                    log.warning(f"removing stale lock from PID {existing_pid}")
                    lf.unlink()
                    return _acquire_lock()
            except ImportError:
                pass
        return None


def _release_lock() -> None:
    try:
        _lock_file().unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def check_only(include_prerelease: bool = False, force: bool = False) -> dict:
    """Just report status, don't apply anything."""
    current = _read_pyproject_version(_mcp_dir())
    result = check_for_update(current, include_prerelease=include_prerelease, force=force)
    result["current"] = current
    result["install_mode"] = "git" if _is_git_install(_mcp_dir()) else "bundle"
    result["installed_path"] = str(_mcp_dir())
    return result


def _download_bundle_zip(version: str, dest_zip: Path) -> None:
    """Fetch mathir-bundle-<version>.zip from GitHub Releases."""
    # Find the asset URL by listing releases
    req = urllib.request.Request(
        GITHUB_API_RELEASES,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "mathir-updater/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        releases = json.loads(resp.read())
    asset_url = None
    for r in releases:
        if r.get("tag_name", "").lstrip("v") == version.lstrip("v"):
            for a in r.get("assets", []):
                if a.get("name", "").endswith(".zip"):
                    asset_url = a["browser_download_url"]
                    break
            break
    if not asset_url:
        raise RuntimeError(f"No bundle asset found for v{version} on GitHub Releases")
    log.info(f"downloading {asset_url}")
    with urllib.request.urlopen(asset_url, timeout=120) as r, open(dest_zip, "wb") as f:
        shutil.copyfileobj(r, f)
    log.info(f"downloaded {dest_zip.stat().st_size // 1024} KB")


def _backup_current(installed: Path) -> Path:
    """Snapshot the current install into .backups/<v>-<ts>/."""
    current = _read_pyproject_version(installed)
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = _backups_dir() / f"{current}-{ts}"
    if dest.exists():
        raise RuntimeError(f"backup path already exists: {dest}")
    log.info(f"backing up {installed} -> {dest}")
    shutil.copytree(installed, dest, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", "node_modules"
    ))
    return dest


def _extract_bundle(zip_path: Path, dest_dir: Path) -> None:
    log.info(f"extracting {zip_path} -> {dest_dir}")
    # zip contains mathir_mcp/ at the root
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if not any(m.startswith("mathir_mcp/") for m in members):
            raise RuntimeError(f"zip layout unexpected: first member = {members[0]}")
        zf.extractall(dest_dir.parent)


def _git_checkout(target_ref: str) -> None:
    mcp = _mcp_dir()
    log.info(f"git fetch + checkout {target_ref} in {mcp}")
    subprocess.run(["git", "fetch", "--tags", "--all"], cwd=mcp, check=True)
    # Check for local modifications (refuse update if dirty)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=mcp, capture_output=True, text=True, check=True
    )
    if status.stdout.strip():
        raise RuntimeError(
            f"local modifications detected in {mcp}:\n{status.stdout}\n"
            "Commit or stash them first, then re-run the update."
        )
    subprocess.run(["git", "checkout", target_ref], cwd=mcp, check=True)


def _pip_install_editable() -> None:
    log.info("pip install -e . (rebuild module metadata)")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
        cwd=_mcp_dir(), check=True,
    )


def _sync_agent_copies() -> list:
    """Sync mathir_mcp_server.py + plugins + GLOBAL_INSTRUCTIONS.md to agent copies.

    Returns list of sync actions taken (for the report).
    """
    actions = []
    src_mcp_server = _mcp_dir() / "mathir_lib" / "mathir_mcp_server.py"
    src_plugin = _mcp_dir() / "opencode_templates" / "plugins" / "mathir-auto-inject.ts"
    src_instructions = _mcp_dir() / "GLOBAL_INSTRUCTIONS.md"
    src_agents_md = _mcp_dir() / "opencode_templates" / "AGENTS.md"
    src_config = _mcp_dir() / "config_template.json"

    pairs = [
        ("opencode", Path.home() / ".config" / "opencode" / "tools" / "mathir_mcp" / "mathir_lib" / "mathir_mcp_server.py"),
        ("mimocode", Path.home() / ".config" / "mimocode" / "tools" / "mathir_mcp" / "mathir_lib" / "mathir_mcp_server.py"),
    ]
    for agent, dst in pairs:
        if dst.exists() and src_mcp_server.exists():
            shutil.copy2(src_mcp_server, dst)
            actions.append(f"synced {agent} MCP bridge")

    plugin_pairs = [
        ("opencode", Path.home() / ".config" / "opencode" / "plugins" / "mathir-auto-inject.ts"),
        ("mimocode", Path.home() / ".config" / "mimocode" / "plugins" / "mathir-auto-inject.ts"),
    ]
    for agent, dst in plugin_pairs:
        if dst.exists() and src_plugin.exists():
            shutil.copy2(src_plugin, dst)
            actions.append(f"synced {agent} plugin")

    inst_pairs = [
        ("opencode", Path.home() / ".config" / "opencode" / "GLOBAL_INSTRUCTIONS.md"),
        ("mimocode", Path.home() / ".config" / "mimocode" / "GLOBAL_INSTRUCTIONS.md"),
    ]
    for agent, dst in inst_pairs:
        if dst.exists() and src_instructions.exists():
            shutil.copy2(src_instructions, dst)
            actions.append(f"synced {agent} GLOBAL_INSTRUCTIONS.md")

    return actions


def _restart_daemon() -> None:
    """Restart the daemon by triggering the user's startup script.

    On Windows this is the .bat in the Startup folder. On Linux/macOS
    it's a systemd/launchd unit (user-configurable via $MATHIR_DAEMON_RESTART).
    """
    env_cmd = os.environ.get("MATHIR_DAEMON_RESTART")
    if env_cmd:
        log.info(f"restarting daemon via MATHIR_DAEMON_RESTART: {env_cmd}")
        subprocess.run(env_cmd, shell=True, check=False)
        return

    if sys.platform == "win32":
        bat = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "mathir_daemon_startup.bat"
        if bat.exists():
            log.info(f"restarting daemon via {bat}")
            subprocess.Popen(["cmd", "/c", str(bat)], shell=False)
            return
    log.warning("no daemon restart mechanism configured; please restart manually")


def _verify_health(timeout: int = 30) -> bool:
    """Wait for /health to return ok after restart."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request("http://127.0.0.1:7338/health")
            with urllib.request.urlopen(req, timeout=2) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _migrate_databases(force_apply: bool = False) -> list:
    """Run schema migration on all known DBs. Returns list of actions."""
    log.info("checking all known databases for schema migrations")
    try:
        from .mathir_migrate import discover_dbs, migrate_db
    except ImportError as e:
        log.warning(f"mathir_migrate not importable: {e}")
        return []
    paths = discover_dbs()
    actions = []
    for db in paths:
        result = migrate_db(db, dry_run=True)
        actions.append(f"dry-run {db}: {result.get('action', 'unknown')}")
        if result.get("action") in ("migrate", "apply"):
            if force_apply:
                actions.append(f"applying migration to {db}")
                migrate_db(db, dry_run=False)
            else:
                log.info(f"migration needed for {db}, but --force-apply not set")
    return actions


def _save_state(previous_version: str, new_version: str, backup_path: Path, method: str) -> None:
    state = {
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "previous_version": previous_version,
        "new_version": new_version,
        "backup_path": str(backup_path),
        "method": method,
    }
    _state_file().write_text(json.dumps(state, indent=2), encoding="utf-8")


def update(
    to_version: Optional[str] = None,
    *,
    from_bundle: Optional[Path] = None,
    force_apply: bool = False,
    include_prerelease: bool = False,
    dry_run: bool = False,
) -> dict:
    """Main state machine. Returns a report dict."""
    report = {"steps": [], "warnings": [], "errors": []}
    pid = _acquire_lock()
    if pid is None:
        raise RuntimeError("another update is in progress (lock held)")

    try:
        # DETECT
        is_git = _is_git_install(_mcp_dir())
        mode = "git" if is_git else "bundle"
        report["install_mode"] = mode
        report["installed_path"] = str(_mcp_dir())
        report["steps"].append(f"DETECT: install mode = {mode}")
        log.info(f"install mode = {mode}")

        # DISCOVER
        current = _read_pyproject_version(_mcp_dir())
        report["current"] = current
        if from_bundle:
            # Bundle mode with explicit file: derive version from filename
            target = to_version or from_bundle.stem.split("-")[-1]
            if not target.startswith("v"):
                target = "v" + target if "-" in from_bundle.stem else target
            method = "bundle-local"
        elif to_version:
            target = to_version.lstrip("v")
            method = "git-tag" if is_git else "bundle-version"
        else:
            result = check_for_update(current, include_prerelease=include_prerelease)
            target = result.get("latest_version", current)
            method = "git-tag" if is_git else "bundle-version"
            report["steps"].append(f"DISCOVER: GitHub latest = {target}")

        report["target"] = target
        if parse_version(target) <= parse_version(current):
            report["steps"].append(f"already at v{current} or newer, nothing to do")
            return report

        if dry_run:
            report["steps"].append(f"DRY-RUN: would update {current} -> {target} via {method}")
            return report

        # BACKUP
        backup = _backup_current(_mcp_dir())
        report["steps"].append(f"BACKUP: {backup}")
        _save_state(current, target, backup, method)

        try:
            # DOWNLOAD
            if from_bundle:
                # Just expand into a sibling temp dir then replace
                tmp_extract = _mcp_dir().parent / f".bundle_extract_{int(time.time())}"
                _extract_bundle(from_bundle, tmp_extract)
                # Remove the old install + replace with extracted content
                shutil.rmtree(_mcp_dir())
                shutil.copytree(tmp_extract / "mathir_mcp", _mcp_dir())
                shutil.rmtree(tmp_extract)
            elif method == "git-tag":
                _git_checkout(f"v{target}" if not target.startswith("v") else target)
            else:
                # bundle-version: download bundle
                tmp_zip = _mcp_dir().parent / f".bundle-{target}.zip"
                _download_bundle_zip(target, tmp_zip)
                tmp_extract = _mcp_dir().parent / f".bundle_extract_{int(time.time())}"
                _extract_bundle(tmp_zip, tmp_extract)
                shutil.rmtree(_mcp_dir())
                shutil.copytree(tmp_extract / "mathir_mcp", _mcp_dir())
                shutil.rmtree(tmp_extract)
                tmp_zip.unlink()
            report["steps"].append(f"DOWNLOAD: applied v{target}")

            # APPLY (pip)
            _pip_install_editable()
            report["steps"].append("APPLY: pip install -e . OK")

            # SYNC AGENT COPIES
            sync_actions = _sync_agent_copies()
            report["steps"].extend(sync_actions)

            # MIGRATE
            mig_actions = _migrate_databases(force_apply=force_apply)
            report["steps"].extend(mig_actions)

            # RESTART + VERIFY
            _restart_daemon()
            time.sleep(5)  # give the new daemon a moment
            if _verify_health(timeout=30):
                report["steps"].append("RESTART: /health OK after restart")
            else:
                raise RuntimeError("post-restart health check failed")
        except Exception as e:
            log.error(f"update failed: {e}, rolling back from {backup}")
            report["errors"].append(str(e))
            # ROLLBACK
            try:
                shutil.rmtree(_mcp_dir())
                shutil.copytree(backup, _mcp_dir())
                _pip_install_editable()
                _restart_daemon()
                time.sleep(5)
                if _verify_health(timeout=20):
                    report["steps"].append(f"ROLLBACK: restored from {backup}")
                else:
                    report["errors"].append("rollback health check failed")
            except Exception as rb:
                report["errors"].append(f"rollback also failed: {rb}")
            raise

        report["new_version"] = _read_pyproject_version(_mcp_dir())
        return report
    finally:
        _release_lock()


def rollback() -> dict:
    """Restore from the most recent backup."""
    sf = _state_file()
    if not sf.exists():
        raise RuntimeError("no rollback state found")
    state = json.loads(sf.read_text())
    backup = Path(state["backup_path"])
    if not backup.exists():
        raise RuntimeError(f"backup missing: {backup}")

    pid = _acquire_lock()
    if pid is None:
        raise RuntimeError("another update is in progress (lock held)")
    try:
        log.info(f"rolling back to {state['previous_version']} from {backup}")
        shutil.rmtree(_mcp_dir())
        shutil.copytree(backup, _mcp_dir())
        _pip_install_editable()
        _restart_daemon()
        time.sleep(5)
        ok = _verify_health(timeout=20)
        if not ok:
            raise RuntimeError("rollback health check failed")
        return {
            "rolled_back_to": state["previous_version"],
            "from_backup": str(backup),
            "health_ok": ok,
        }
    finally:
        _release_lock()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_check_only(result: dict) -> str:
    """Format check-style report. Tolerates either a check_only result dict
    or an update --dry-run result dict (which has 'target' instead of
    'latest_version')."""
    lines = []
    lines.append(f"MATHIR Update Check")
    current = result.get("current", "?")
    latest = result.get("latest_version") or result.get("target") or current
    lines.append(f"  current:      v{current}")
    lines.append(f"  latest:       v{latest}")
    lines.append(f"  install mode: {result.get('install_mode', '?')}")
    lines.append(f"  install path: {result.get('installed_path', '?')}")
    update_available = result.get("update_available")
    if update_available is None:
        # update --dry-run: compare versions ourselves
        update_available = parse_version(latest) > parse_version(current)
    if update_available:
        lines.append(f"  [!] UPDATE AVAILABLE")
        lines.append(f"  release:      {result.get('release_url', '(no url)')}")
        lines.append(f"  run:          python -m mathir_mcp update")
    else:
        lines.append(f"  [OK] up-to-date")
    if result.get("error"):
        lines.append(f"  warning: {result['error']}")
    return "\n".join(lines)


def _format_report(report: dict) -> str:
    lines = [f"MATHIR Update Complete: v{report.get('current', '?')} -> v{report.get('new_version', report.get('target', '?'))}"]
    lines.append("")
    lines.append("Steps:")
    for s in report.get("steps", []):
        lines.append(f"  [OK] {s}")
    if report.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  [!] {w}")
    if report.get("errors"):
        lines.append("")
        lines.append("Errors:")
        for e in report["errors"]:
            lines.append(f"  [ERR] {e}")
    return "\n".join(lines)


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mathir_mcp.mathir_lib.mathir_updater",
        description="MATHIR self-updater",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="check for updates, don't apply")
    p_check.add_argument("--include-prerelease", action="store_true")
    p_check.add_argument("--force", action="store_true", help="bypass cache")

    p_update = sub.add_parser("update", help="apply an update")
    p_update.add_argument("--to", dest="to_version", help="specific version (e.g. 8.5.2)")
    p_update.add_argument("--from-bundle", type=Path, help="install from a local bundle zip (offline)")
    p_update.add_argument("--force-apply", action="store_true", help="auto-apply schema migrations without asking")
    p_update.add_argument("--include-prerelease", action="store_true")
    p_update.add_argument("--dry-run", action="store_true", help="show plan only")

    p_rollback = sub.add_parser("rollback", help="restore from last backup")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    try:
        if args.cmd == "check":
            print(_format_check_only(check_only(args.include_prerelease, args.force)))
            return 0
        elif args.cmd == "update":
            report = update(
                to_version=args.to_version,
                from_bundle=args.from_bundle,
                force_apply=args.force_apply,
                include_prerelease=args.include_prerelease,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print(_format_check_only(report))
            else:
                print(_format_report(report))
            return 0 if not report.get("errors") else 1
        elif args.cmd == "rollback":
            result = rollback()
            print(f"Rolled back to v{result['rolled_back_to']}")
            print(f"From backup: {result['from_backup']}")
            print(f"Health: {'OK' if result['health_ok'] else 'FAILED'}")
            return 0 if result["health_ok"] else 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())